"""Unit tests for grounded generation helpers."""

from __future__ import annotations

import uuid

import pytest

from app.core.llm_client import GroundedGenerationError, generate_grounded_draft
from app.core.gemini_generator import (
    GeminiGenerationError,
    REFUSAL_TEXT,
    _build_prompt,
    _coerce_json_text,
    _generation_config_for_mode,
    _parse_result,
    _repair_citation_snippets,
    _validate_result_against_evidence,
)
from app.core.openai_generator import OpenAICompatibleGenerationError
from app.pipeline.contracts import EvidenceItem, EvidencePackage, GroundedAnswerDraft
from app.services.generation import GenerationBackend, generate_answer_from_evidence


def _evidence_item(*, citation_id: str, chunk_id: str, text: str) -> EvidenceItem:
    return EvidenceItem(
        citation_id=citation_id,
        chunk_id=chunk_id,
        tenant_id=uuid.uuid4(),
        namespace_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        chunk_index=0,
        text=text,
        score=0.9,
        sources=("dense", "sparse"),
    )


def test_generate_grounded_draft_uses_evidence_only() -> None:
    """Grounded generation should compose its answer from packaged evidence."""

    evidence_package = EvidencePackage(
        retrieved_chunk_ids=["chunk-1", "chunk-2"],
        selected_evidence_ids=["chunk-1", "chunk-2"],
        items=[
            _evidence_item(
                citation_id="E001",
                chunk_id="chunk-1",
                text="Grounded handles tenant-safe uploads. It stores files in object storage.",
            ),
            _evidence_item(
                citation_id="E002",
                chunk_id="chunk-2",
                text="Ingestion jobs track processing status for each document.",
            ),
        ],
    )

    draft = generate_grounded_draft(
        query_text="How does grounded handle uploads and processing?",
        evidence_package=evidence_package,
    )

    assert "Grounded handles tenant-safe uploads." in draft.answer_text
    assert "Ingestion jobs track processing status for each document." in draft.answer_text
    assert "[E001]" in draft.answer_text
    assert "[E002]" in draft.answer_text
    assert draft.cited_evidence_ids == ["chunk-1", "chunk-2"]
    assert draft.citation_snippets["chunk-1"] == "Grounded handles tenant-safe uploads."
    assert draft.generator_provider == "local-grounded-v1"
    assert draft.support_coverage == 1.0
    assert draft.source_diversity == 2


def test_generate_grounded_draft_prefers_query_aligned_sentence() -> None:
    """Grounded generation should prefer the sentence most aligned with the query."""

    evidence_package = EvidencePackage(
        retrieved_chunk_ids=["chunk-1"],
        selected_evidence_ids=["chunk-1"],
        items=[
            _evidence_item(
                citation_id="E001",
                chunk_id="chunk-1",
                text=(
                    "This system has many components. "
                    "Hybrid retrieval merges sparse and dense search results. "
                    "The project also supports tenant isolation."
                ),
            ),
        ],
    )

    draft = generate_grounded_draft(
        query_text="Explain hybrid retrieval",
        evidence_package=evidence_package,
    )

    assert draft.answer_text.startswith(
        "Hybrid retrieval merges sparse and dense search results. [E001]"
    )
    assert draft.citation_snippets["chunk-1"] == "Hybrid retrieval merges sparse and dense search results."


def test_generate_grounded_draft_answers_concept_definition_question() -> None:
    """Definition-style concept questions should render explanatory evidence, not titles."""

    evidence_package = EvidencePackage(
        retrieved_chunk_ids=["chunk-title", "chunk-definition"],
        selected_evidence_ids=["chunk-title", "chunk-definition"],
        items=[
            _evidence_item(
                citation_id="E001",
                chunk_id="chunk-title",
                text="A continual learning survey\nDefying forgetting in classification tasks.",
            ),
            _evidence_item(
                citation_id="E002",
                chunk_id="chunk-definition",
                text=(
                    "Masana is with Computer Vision Center, UAB. "
                    "Continual learning is the ability of a model to learn from a stream "
                    "of tasks while retaining useful knowledge from previous tasks. "
                    "It works by updating the learner incrementally as new data arrives "
                    "while using strategies that reduce catastrophic forgetting."
                ),
            ),
        ],
    )

    draft = generate_grounded_draft(
        query_text="What is continual learning? How does it work?",
        evidence_package=evidence_package,
    )

    assert "Continual learning is the ability" in draft.answer_text
    assert "It works by updating" in draft.answer_text
    assert "Masana is with Computer Vision Center" not in draft.answer_text
    assert "A continual learning survey [E001]" not in draft.answer_text
    assert "chunk-definition" in draft.cited_evidence_ids
    assert "chunk-title" not in draft.cited_evidence_ids


def test_generate_grounded_draft_rejects_what_is_concept_without_definition_support() -> None:
    """Concept questions should not accept evidence that only mentions the term."""

    evidence_package = EvidencePackage(
        retrieved_chunk_ids=["chunk-title", "chunk-mention"],
        selected_evidence_ids=["chunk-title", "chunk-mention"],
        items=[
            _evidence_item(
                citation_id="E001",
                chunk_id="chunk-title",
                text="A continual learning survey",
            ),
            _evidence_item(
                citation_id="E002",
                chunk_id="chunk-mention",
                text="The paper compares continual learning benchmarks and datasets.",
            ),
        ],
    )

    with pytest.raises(GroundedGenerationError, match="query-aligned support"):
        generate_grounded_draft(
            query_text="What is continual learning?",
            evidence_package=evidence_package,
        )


def test_generate_grounded_draft_synthesizes_dataset_summary_from_structure() -> None:
    """Dataset summaries should synthesize a high-level overview instead of echoing a fragment."""

    evidence_package = EvidencePackage(
        retrieved_chunk_ids=["chunk-header", "chunk-sections"],
        selected_evidence_ids=["chunk-header", "chunk-sections"],
        items=[
            _evidence_item(
                citation_id="E001",
                chunk_id="chunk-header",
                text=(
                    "Samrawit Gebremaryam Bahta\n"
                    "samrawitgebremaryam121@gmail.com\n"
                    "EDUCATION\nBSc in Software Engineering"
                ),
            ),
            _evidence_item(
                citation_id="E002",
                chunk_id="chunk-sections",
                text=(
                    "PROFESSIONAL EXPERIENCE\nAI Engineer at iCog Labs\n"
                    "TECHNICAL SKILLS\nPython, Go, TypeScript\n"
                    "AWARDS\nHuawei Seeds for the Future"
                ),
            ),
        ],
    )

    draft = generate_grounded_draft(
        query_text="What is this dataset about?",
        evidence_package=evidence_package,
    )

    assert draft.answer_text.startswith("The dataset contains")
    assert "Samrawit Gebremaryam Bahta" in draft.answer_text
    assert "professional experience" in draft.answer_text.lower()
    assert "[E001]" in draft.answer_text


def test_generate_grounded_draft_rejects_empty_evidence() -> None:
    """Grounded generation should fail cleanly without evidence."""

    with pytest.raises(GroundedGenerationError):
        generate_grounded_draft(
            query_text="What happened?",
            evidence_package=EvidencePackage(
                retrieved_chunk_ids=[],
                selected_evidence_ids=[],
                items=[],
            ),
        )


def test_generate_grounded_draft_rejects_non_meaningful_greeting_queries() -> None:
    """Short greetings should degrade instead of reusing unrelated evidence."""

    evidence_package = EvidencePackage(
        retrieved_chunk_ids=["chunk-1"],
        selected_evidence_ids=["chunk-1"],
        items=[
            _evidence_item(
                citation_id="E001",
                chunk_id="chunk-1",
                text="Samrawit studies software engineering at Addis Ababa Science and Technology University.",
            ),
        ],
    )

    with pytest.raises(GroundedGenerationError, match="meaningful query terms"):
        generate_grounded_draft(
            query_text="hi",
            evidence_package=evidence_package,
        )


def test_generate_grounded_draft_omits_irrelevant_evidence_items() -> None:
    """Only evidence that actually supports the query should be cited."""

    evidence_package = EvidencePackage(
        retrieved_chunk_ids=["chunk-1", "chunk-2"],
        selected_evidence_ids=["chunk-1", "chunk-2"],
        items=[
            _evidence_item(
                citation_id="E001",
                chunk_id="chunk-1",
                text="Education: BSc in Software Engineering at Addis Ababa Science and Technology University.",
            ),
            _evidence_item(
                citation_id="E002",
                chunk_id="chunk-2",
                text="Awarded for a Huawei Seeds for the Future project in China.",
            ),
        ],
    )

    draft = generate_grounded_draft(
        query_text="What is her education status?",
        evidence_package=evidence_package,
    )

    assert draft.cited_evidence_ids == ["chunk-1"]
    assert "[E002]" not in draft.answer_text


def test_generate_grounded_draft_can_extract_name_like_answer() -> None:
    """Field-style questions should prefer exact answer lines over unrelated descriptive bullets."""

    evidence_package = EvidencePackage(
        retrieved_chunk_ids=["chunk-name", "chunk-experience"],
        selected_evidence_ids=["chunk-name", "chunk-experience"],
        items=[
            _evidence_item(
                citation_id="E001",
                chunk_id="chunk-name",
                text="Samrawit Gebremaryam Bahta\nsamrawitgebremaryam121@gmail.com\n+251-989-985-456",
            ),
            _evidence_item(
                citation_id="E002",
                chunk_id="chunk-experience",
                text="Professional experience: AI Engineer at iCog Labs building grounded retrieval systems.",
            ),
        ],
    )

    draft = generate_grounded_draft(
        query_text="What is the name of the resume owner?",
        evidence_package=evidence_package,
    )

    assert draft.cited_evidence_ids[0] == "chunk-name"
    assert draft.answer_text == "Samrawit Gebremaryam Bahta [E001]"
    assert "[E001]" in draft.answer_text
    assert "[E002]" not in draft.answer_text


def test_generate_grounded_draft_prefers_skills_section_over_incidental_skill_wording() -> None:
    """Skills questions should prefer actual skill sections over unrelated mentions of skills."""

    evidence_package = EvidencePackage(
        retrieved_chunk_ids=["chunk-awards", "chunk-skills"],
        selected_evidence_ids=["chunk-awards", "chunk-skills"],
        items=[
            _evidence_item(
                citation_id="E001",
                chunk_id="chunk-awards",
                text="Enhanced skills in advanced ICT and Artificial intelligence through targeted workshops.",
            ),
            _evidence_item(
                citation_id="E002",
                chunk_id="chunk-skills",
                text=(
                    "TECHNICAL SKILLS\n"
                    "AI & Machine Learning: PyTorch, Scikit-learn, Hugging Face, RAG\n"
                    "Programming Languages: Python, Go, TypeScript"
                ),
            ),
        ],
    )

    draft = generate_grounded_draft(
        query_text="What are her technical skills?",
        evidence_package=evidence_package,
    )

    assert draft.cited_evidence_ids[0] == "chunk-skills"
    assert "AI & Machine Learning: PyTorch" in draft.answer_text
    assert "PyTorch" in draft.answer_text
    assert "[E002]" in draft.answer_text
    assert "[E001]" not in draft.answer_text


def test_generate_grounded_draft_rejects_definition_queries_without_definition_support() -> None:
    """Mentioning a term should not count as defining it."""

    evidence_package = EvidencePackage(
        retrieved_chunk_ids=["chunk-course", "chunk-skills"],
        selected_evidence_ids=["chunk-course", "chunk-skills"],
        items=[
            _evidence_item(
                citation_id="E001",
                chunk_id="chunk-course",
                text="Relevant courses include Introduction to Machine Learning and Advanced Programming.",
            ),
            _evidence_item(
                citation_id="E002",
                chunk_id="chunk-skills",
                text="AI & Machine Learning: PyTorch, Scikit-learn, Hugging Face.",
            ),
        ],
    )

    with pytest.raises(GroundedGenerationError, match="query-aligned support"):
        generate_grounded_draft(
            query_text="What does machine learning mean?",
            evidence_package=evidence_package,
        )


def test_generate_grounded_draft_does_not_misclassify_experience_query_as_name() -> None:
    """Queries mentioning a person should still answer the requested field, not default to name."""

    evidence_package = EvidencePackage(
        retrieved_chunk_ids=["chunk-name", "chunk-experience"],
        selected_evidence_ids=["chunk-name", "chunk-experience"],
        items=[
            _evidence_item(
                citation_id="E001",
                chunk_id="chunk-name",
                text="Samrawit Gebremaryam Bahta\nsamrawit@example.com",
            ),
            _evidence_item(
                citation_id="E002",
                chunk_id="chunk-experience",
                text="PROFESSIONAL EXPERIENCE\nAI Engineer at iCog Labs working on grounded retrieval systems.",
            ),
        ],
    )

    draft = generate_grounded_draft(
        query_text="What is the work experience of the person?",
        evidence_package=evidence_package,
    )

    assert draft.answer_text.startswith("AI Engineer at iCog Labs")
    assert "AI Engineer at iCog Labs" in draft.answer_text
    assert "The person's name is" not in draft.answer_text
    assert draft.cited_evidence_ids == ["chunk-experience"]


def test_generate_grounded_draft_renders_action_questions_from_responsibility_lines() -> None:
    """Action questions should summarize concrete responsibilities, not echo a raw section label."""

    evidence_package = EvidencePackage(
        retrieved_chunk_ids=["chunk-experience"],
        selected_evidence_ids=["chunk-experience"],
        items=[
            _evidence_item(
                citation_id="E001",
                chunk_id="chunk-experience",
                text=(
                    "PROFESSIONAL EXPERIENCE\n"
                    "AI Engineer\n"
                    "iCog Labs\n"
                    "Architected and developed emotionally intelligent AI characters.\n"
                    "Integrated SPMiner-inspired neural mining techniques with improved visualization modules."
                ),
            ),
        ],
    )

    draft = generate_grounded_draft(
        query_text="What did she do at iCog Labs?",
        evidence_package=evidence_package,
    )

    assert draft.answer_text.startswith("In icog labs, the evidence shows work including")
    assert "Architected and developed emotionally intelligent AI characters" in draft.answer_text
    assert "Integrated SPMiner-inspired neural mining techniques" in draft.answer_text


def test_generate_grounded_draft_renders_comparison_questions_as_boolean_with_evidence() -> None:
    """Comparison questions should answer yes/no using the relevant context rather than dumping a snippet."""

    evidence_package = EvidencePackage(
        retrieved_chunk_ids=["chunk-role-1", "chunk-role-2"],
        selected_evidence_ids=["chunk-role-1", "chunk-role-2"],
        items=[
            _evidence_item(
                citation_id="E001",
                chunk_id="chunk-role-1",
                text="AI Engineer\niCog Labs\nArchitected grounded retrieval systems.",
            ),
            _evidence_item(
                citation_id="E002",
                chunk_id="chunk-role-2",
                text="Backend | AI Developer Intern\niCog Labs\nImplemented an Elasticsearch autocomplete system.",
            ),
        ],
    )

    draft = generate_grounded_draft(
        query_text="Is her experience only in iCog Labs?",
        evidence_package=evidence_package,
    )

    assert draft.answer_text.startswith("Yes.")
    assert "icog labs" in draft.answer_text.casefold()


def test_generate_grounded_draft_renders_entity_context_questions_cleanly() -> None:
    """Entity-in-context questions should return a grounded yes/no answer from the matching snippet."""

    evidence_package = EvidencePackage(
        retrieved_chunk_ids=["chunk-experience"],
        selected_evidence_ids=["chunk-experience"],
        items=[
            _evidence_item(
                citation_id="E001",
                chunk_id="chunk-experience",
                text="Integrated SPMiner-inspired neural mining techniques with improved visualization modules.",
            ),
        ],
    )

    draft = generate_grounded_draft(
        query_text="Did she work on pattern miner?",
        evidence_package=evidence_package,
    )

    assert draft.answer_text.startswith("Yes.")
    assert "neural mining techniques" in draft.answer_text


def test_generate_grounded_draft_summarizes_paper_like_dataset_using_title_and_topic() -> None:
    """Paper-like datasets should summarize from titles/topic sentences, not generic section labels."""

    evidence_package = EvidencePackage(
        retrieved_chunk_ids=["chunk-paper"],
        selected_evidence_ids=["chunk-paper"],
        items=[
            _evidence_item(
                citation_id="E001",
                chunk_id="chunk-paper",
                text=(
                    "Explainable AI in Software Engineering\n"
                    "Introduction\n"
                    "This paper examines how explainable AI methods help software teams debug, validate, and govern AI-enabled systems."
                ),
            ),
        ],
    )

    draft = generate_grounded_draft(
        query_text="What is the dataset about?",
        evidence_package=evidence_package,
    )

    assert "Explainable AI in Software Engineering" in draft.answer_text
    assert "The dataset is about" in draft.answer_text
    assert "This paper examines how explainable AI methods help software teams debug" in draft.answer_text
    assert "The dataset contains Introduction" not in draft.answer_text


def test_generate_grounded_draft_renders_collection_answers_from_prose_sentences() -> None:
    """Collection queries on prose should extract relevant sentences rather than raw paragraphs."""

    evidence_package = EvidencePackage(
        retrieved_chunk_ids=["chunk-xai"],
        selected_evidence_ids=["chunk-xai"],
        items=[
            _evidence_item(
                citation_id="E001",
                chunk_id="chunk-xai",
                text=(
                    "One major challenge is that some explanations create a false sense of trust in incorrect outputs. "
                    "Another emerging challenge concerns generative AI and large language models, where open-ended outputs are harder to explain safely. "
                    "Finally, software teams must integrate explanations into documentation, monitoring, testing, and governance workflows."
                ),
            ),
        ],
    )

    draft = generate_grounded_draft(
        query_text="What are the challenges for xAI?",
        evidence_package=evidence_package,
    )

    assert "false sense of trust" in draft.answer_text
    assert "generative AI and large language models" in draft.answer_text


def test_generate_grounded_draft_ignores_outline_headings_for_challenge_lists() -> None:
    """Collection answers should ignore numbered section headings and use challenge prose."""

    evidence_package = EvidencePackage(
        retrieved_chunk_ids=["chunk-outline", "chunk-challenges", "chunk-summary"],
        selected_evidence_ids=["chunk-outline", "chunk-challenges", "chunk-summary"],
        items=[
            _evidence_item(
                citation_id="E001",
                chunk_id="chunk-outline",
                text=(
                    "2. New Opportunities in Software Engineering\n"
                    "Overview text.\n"
                    "3. New Challenges in Software Engineering"
                ),
            ),
            _evidence_item(
                citation_id="E002",
                chunk_id="chunk-challenges",
                text=(
                    "One major challenge is the trade-off between model performance and interpretability. "
                    "Another critical challenge is the evaluation of explanations. "
                    "Scalability is also a concern as models grow larger and more complex."
                ),
            ),
            _evidence_item(
                citation_id="E003",
                chunk_id="chunk-summary",
                text=(
                    "However, challenges such as evaluation, scalability, and balancing accuracy with "
                    "interpretability remain significant."
                ),
            ),
        ],
    )

    draft = generate_grounded_draft(
        query_text="What are the challenges?",
        evidence_package=evidence_package,
    )

    assert "New Opportunities in Software Engineering" not in draft.answer_text
    assert "New Challenges in Software Engineering" not in draft.answer_text
    assert "trade-off between model performance and interpretability" in draft.answer_text
    assert "evaluation of explanations" in draft.answer_text


def test_generate_grounded_draft_ignores_title_case_section_headings_for_collection_answers() -> None:
    """Title-case section headers should not become the final list items."""

    evidence_package = EvidencePackage(
        retrieved_chunk_ids=["chunk-challenges"],
        selected_evidence_ids=["chunk-challenges"],
        items=[
            _evidence_item(
                citation_id="E001",
                chunk_id="chunk-challenges",
                text=(
                    "New Challenges in Software Engineering\n"
                    "One challenge is that inaccurate explanations can create false trust in high-stakes systems.\n"
                    "Another challenge is that generative AI outputs are open-ended, so engineers need ways to surface uncertainty, hallucination risks, and retrieval gaps.\n"
                    "A further challenge is that teams must integrate explanations into testing, monitoring, documentation, and CI/CD workflows."
                ),
            ),
        ],
    )

    draft = generate_grounded_draft(
        query_text="What are the challenges?",
        evidence_package=evidence_package,
    )

    assert "New Challenges in Software Engineering" not in draft.answer_text
    assert "false trust in high-stakes systems" in draft.answer_text
    assert "retrieval gaps" in draft.answer_text


def test_generate_grounded_draft_renders_multiple_collection_families_cleanly() -> None:
    """Questions like methods-and-tools should include both item families, not just the header."""

    evidence_package = EvidencePackage(
        retrieved_chunk_ids=["chunk-methods-tools"],
        selected_evidence_ids=["chunk-methods-tools"],
        items=[
            _evidence_item(
                citation_id="E001",
                chunk_id="chunk-methods-tools",
                text=(
                    "Methods, Technologies, and Tools\n"
                    "XAI methods include intrinsic approaches such as decision trees, linear models, and rule-based systems.\n"
                    "Post-hoc methods include LIME, SHAP, and counterfactual explanations.\n"
                    "Common tools include Captum for PyTorch models, InterpretML for glassbox models and visualizations, and Google's What-If Tool for interactive analysis."
                ),
            ),
        ],
    )

    draft = generate_grounded_draft(
        query_text="What are the methods, the tools?",
        evidence_package=evidence_package,
    )

    assert "decision trees" in draft.answer_text
    assert "decision trees" in draft.answer_text
    assert "LIME, SHAP, and counterfactual explanations" in draft.answer_text
    assert "Captum" in draft.answer_text


def test_generate_grounded_draft_answers_mixed_boolean_action_role_question() -> None:
    """Role + where + what-did questions should return a grounded action answer."""

    evidence_package = EvidencePackage(
        retrieved_chunk_ids=["chunk-experience"],
        selected_evidence_ids=["chunk-experience"],
        items=[
            _evidence_item(
                citation_id="E001",
                chunk_id="chunk-experience",
                text=(
                    "PROFESSIONAL EXPERIENCE\n"
                    "AI Engineer\n"
                    "iCog Labs\n"
                    "Architected and developed emotionally intelligent AI characters.\n"
                    "Integrated SPMiner-inspired neural mining techniques with improved visualization modules."
                ),
            ),
        ],
    )

    draft = generate_grounded_draft(
        query_text="does she has an experiance as a AI enginner if so where and what did she do there",
        evidence_package=evidence_package,
    )

    assert draft.answer_text.startswith("Yes.")
    assert "AI Engineer" in draft.answer_text
    assert "iCog Labs" in draft.answer_text
    assert "Architected and developed emotionally intelligent AI characters" in draft.answer_text


def test_generate_grounded_draft_renders_count_questions_from_structured_items() -> None:
    """Count questions should return a grounded numeric answer when structured item titles are present."""

    evidence_package = EvidencePackage(
        retrieved_chunk_ids=["chunk-projects"],
        selected_evidence_ids=["chunk-projects"],
        items=[
            _evidence_item(
                citation_id="E001",
                chunk_id="chunk-projects",
                text=(
                    "PROJECTS\n"
                    "The Traveler's Pocket Pal - Travel Companion\n"
                    "Developed an AI-powered travel assistant.\n"
                    "StyleCraft - Personalized AI Writing Assistant\n"
                    "Built a RAG-based writing system."
                ),
            ),
        ],
    )

    draft = generate_grounded_draft(
        query_text="How many projects does she have?",
        evidence_package=evidence_package,
    )

    assert draft.answer_text == (
        "The evidence shows 2 projects: The Traveler's Pocket Pal - Travel Companion; "
        "StyleCraft - Personalized AI Writing Assistant [E001]"
    )


def test_generate_grounded_draft_extracts_explicit_count_from_prose_sentence() -> None:
    """Count questions should return a direct numeric phrase when prose states the answer explicitly."""

    evidence_package = EvidencePackage(
        retrieved_chunk_ids=["chunk-fleet"],
        selected_evidence_ids=["chunk-fleet"],
        items=[
            _evidence_item(
                citation_id="E001",
                chunk_id="chunk-fleet",
                text="Fleet Details\nThe city purchased 12 electric vans from Voltara Mobility for the pilot.",
            ),
        ],
    )

    draft = generate_grounded_draft(
        query_text="How many electric vans were purchased?",
        evidence_package=evidence_package,
    )

    assert draft.answer_text == "12 electric vans [E001]"


def test_generate_grounded_draft_extracts_exact_supplier_name_from_prose_sentence() -> None:
    """Supplier/company lookups should return the organization, not the whole paragraph."""

    evidence_package = EvidencePackage(
        retrieved_chunk_ids=["chunk-fleet"],
        selected_evidence_ids=["chunk-fleet"],
        items=[
            _evidence_item(
                citation_id="E001",
                chunk_id="chunk-fleet",
                text="Fleet Details\nThe city purchased 12 electric vans from Voltara Mobility for the pilot.",
            ),
        ],
    )

    draft = generate_grounded_draft(
        query_text="Which company supplied the vans?",
        evidence_package=evidence_package,
    )

    assert draft.answer_text == "Voltara Mobility [E001]"


def test_generate_grounded_draft_computes_numeric_difference_for_comparison_question() -> None:
    """Difference questions should compute the grounded arithmetic result instead of paraphrasing both sentences."""

    evidence_package = EvidencePackage(
        retrieved_chunk_ids=["chunk-old-cost", "chunk-new-cost"],
        selected_evidence_ids=["chunk-old-cost", "chunk-new-cost"],
        items=[
            _evidence_item(
                citation_id="E001",
                chunk_id="chunk-old-cost",
                text="Costs and Savings\nBefore the pilot, the city spent about $18,400 on fuel over six months.",
            ),
            _evidence_item(
                citation_id="E002",
                chunk_id="chunk-new-cost",
                text="Costs and Savings\nDuring the pilot, electricity costs totaled $6,900 over six months.",
            ),
        ],
    )

    draft = generate_grounded_draft(
        query_text="What was the difference between old fuel cost and new electricity cost over six months?",
        evidence_package=evidence_package,
    )

    assert draft.answer_text == "$11,500 [E001] [E002]"


def test_generate_grounded_draft_answers_start_and_end_date_range_cleanly() -> None:
    """Start/end date questions should synthesize a grounded range instead of dropping one endpoint."""

    evidence_package = EvidencePackage(
        retrieved_chunk_ids=["chunk-start", "chunk-end"],
        selected_evidence_ids=["chunk-start", "chunk-end"],
        items=[
            _evidence_item(
                citation_id="E001",
                chunk_id="chunk-start",
                text="Pilot Overview\nIn January 2025, the city of Lydon launched the electric van pilot.",
            ),
            _evidence_item(
                citation_id="E002",
                chunk_id="chunk-end",
                text="Pilot Overview\nThe pilot lasted for six months and ended in June 2025.",
            ),
        ],
    )

    draft = generate_grounded_draft(
        query_text="When did the pilot start and end?",
        evidence_package=evidence_package,
    )

    assert draft.answer_text == "It started in January 2025 [E001] and ended in June 2025 [E002]."


def test_generate_grounded_draft_extracts_time_range_without_location_prefix() -> None:
    """Time-range lookups should return the exact range without malformed wrappers."""

    evidence_package = EvidencePackage(
        retrieved_chunk_ids=["chunk-hours"],
        selected_evidence_ids=["chunk-hours"],
        items=[
            _evidence_item(
                citation_id="E001",
                chunk_id="chunk-hours",
                text="Charging and Operations\nThe vans were usually charged from 10:00 PM to 4:30 AM at the depot.",
            ),
        ],
    )

    draft = generate_grounded_draft(
        query_text="During what hours were the vans usually charged?",
        evidence_package=evidence_package,
    )

    assert draft.answer_text == "10:00 PM to 4:30 AM [E001]"


def test_generate_grounded_draft_renders_recommendations_as_short_items() -> None:
    """Recommendation questions should return the actual recommendations, not a date token."""

    evidence_package = EvidencePackage(
        retrieved_chunk_ids=["chunk-recommend"],
        selected_evidence_ids=["chunk-recommend"],
        items=[
            _evidence_item(
                citation_id="E001",
                chunk_id="chunk-recommend",
                text=(
                    "Committee Recommendation\n"
                    "In August 2025, the committee recommended:\n"
                    "- purchase 6 additional electric vans\n"
                    "- add 3 more charging ports\n"
                    "- create a driver guide focused on energy-efficient route planning"
                ),
            ),
        ],
    )

    draft = generate_grounded_draft(
        query_text="What did the committee recommend in August 2025?",
        evidence_package=evidence_package,
    )

    assert "purchase 6 additional electric vans" in draft.answer_text
    assert "add 3 more charging ports" in draft.answer_text
    assert "August 2025" not in draft.answer_text


def test_generate_grounded_draft_refuses_unsupported_exact_lookup() -> None:
    """Unsupported exact lookups should refuse instead of reusing a nearby sentence."""

    evidence_package = EvidencePackage(
        retrieved_chunk_ids=["chunk-fleet"],
        selected_evidence_ids=["chunk-fleet"],
        items=[
            _evidence_item(
                citation_id="E001",
                chunk_id="chunk-fleet",
                text="The city purchased 9 electric vans from Heliox Transit for the pilot.",
            ),
        ],
    )

    draft = generate_grounded_draft(
        query_text="What color were the vans?",
        evidence_package=evidence_package,
    )

    assert draft.answer_text == "I could not find the answer in the provided context."
    assert draft.cited_evidence_ids == []
    assert draft.citation_snippets == {}


def test_generate_grounded_draft_computes_net_savings_from_grounded_values() -> None:
    """Net savings questions should use deterministic arithmetic over cited evidence."""

    evidence_package = EvidencePackage(
        retrieved_chunk_ids=["chunk-fuel", "chunk-electricity", "chunk-maintenance", "chunk-installation"],
        selected_evidence_ids=["chunk-fuel", "chunk-electricity", "chunk-maintenance", "chunk-installation"],
        items=[
            _evidence_item(
                citation_id="E001",
                chunk_id="chunk-fuel",
                text="Costs and Savings\nBefore the pilot, the city spent about $16,100 on fuel over the period.",
            ),
            _evidence_item(
                citation_id="E002",
                chunk_id="chunk-electricity",
                text="Costs and Savings\nDuring the pilot, electricity costs totaled $6,900.",
            ),
            _evidence_item(
                citation_id="E003",
                chunk_id="chunk-maintenance",
                text="Costs and Savings\nMaintenance savings during the pilot were $3,900.",
            ),
            _evidence_item(
                citation_id="E004",
                chunk_id="chunk-installation",
                text="Costs and Savings\nCharging installation costs were $11,200.",
            ),
        ],
    )

    draft = generate_grounded_draft(
        query_text="What was the net savings after including installation costs?",
        evidence_package=evidence_package,
    )

    assert draft.answer_text == "$1,900 [E001] [E002] [E003] [E004]"


def test_parse_gemini_result_accepts_array_citation_snippets() -> None:
    """Gemini JSON parsing should accept schema-friendly citation snippet arrays."""

    parsed = _parse_result(
        {
            "is_refusal": False,
            "answer_mode": "exact_lookup",
            "answer_text": "Grounded uses hybrid retrieval.",
            "cited_chunk_ids": ["chunk-1"],
            "citation_snippets": [
                {"chunk_id": "chunk-1", "snippet": "Grounded uses hybrid retrieval."}
            ],
        }
    )

    assert parsed.answer_text == "Grounded uses hybrid retrieval."
    assert parsed.is_refusal is False
    assert parsed.answer_mode == "exact_lookup"
    assert parsed.cited_chunk_ids == ["chunk-1"]
    assert parsed.citation_snippets == {
        "chunk-1": "Grounded uses hybrid retrieval."
    }


def test_parse_gemini_result_rejects_empty_array_citation_snippets() -> None:
    """Gemini parsing should fail cleanly when snippet arrays are empty."""

    with pytest.raises(GeminiGenerationError, match="must align exactly"):
        _parse_result(
            {
                "is_refusal": False,
                "answer_mode": "exact_lookup",
                "answer_text": "Grounded uses hybrid retrieval.",
                "cited_chunk_ids": ["chunk-1"],
                "citation_snippets": [],
            }
        )


def test_parse_gemini_result_allows_refusal_without_citations() -> None:
    """Provider refusals should parse cleanly without fake citations."""

    parsed = _parse_result(
        {
            "is_refusal": True,
            "answer_mode": "exact_lookup",
            "answer_text": REFUSAL_TEXT,
            "cited_chunk_ids": [],
            "citation_snippets": [],
        }
    )

    assert parsed.answer_text == REFUSAL_TEXT
    assert parsed.is_refusal is True
    assert parsed.cited_chunk_ids == []
    assert parsed.citation_snippets == {}


def test_validate_result_against_evidence_rejects_unknown_chunk_id() -> None:
    """Gemini validation should reject cited chunk IDs that are not in the evidence package."""

    evidence_package = EvidencePackage(
        retrieved_chunk_ids=["chunk-1"],
        selected_evidence_ids=["chunk-1"],
        items=[
            _evidence_item(
                citation_id="E001",
                chunk_id="chunk-1",
                text="The city purchased 12 electric vans from Voltara Mobility.",
            ),
        ],
    )

    parsed = _parse_result(
        {
            "is_refusal": False,
            "answer_mode": "exact_lookup",
            "answer_text": "Voltara Mobility",
            "cited_chunk_ids": ["chunk-x"],
            "citation_snippets": [
                {"chunk_id": "chunk-x", "snippet": "Voltara Mobility"}
            ],
        }
    )

    with pytest.raises(GeminiGenerationError, match="unknown cited chunk_id"):
        _validate_result_against_evidence(
            result=parsed,
            evidence_package=evidence_package,
            expected_answer_mode="exact_lookup",
        )


def test_validate_result_against_evidence_rejects_ungrounded_snippet() -> None:
    """Gemini validation should reject snippets that are not substrings of the cited chunk."""

    evidence_package = EvidencePackage(
        retrieved_chunk_ids=["chunk-1"],
        selected_evidence_ids=["chunk-1"],
        items=[
            _evidence_item(
                citation_id="E001",
                chunk_id="chunk-1",
                text="The city purchased 12 electric vans from Voltara Mobility.",
            ),
        ],
    )

    parsed = _parse_result(
        {
            "is_refusal": False,
            "answer_mode": "exact_lookup",
            "answer_text": "Voltara Mobility",
            "cited_chunk_ids": ["chunk-1"],
            "citation_snippets": [
                {"chunk_id": "chunk-1", "snippet": "The supplier was Voltara Mobility"}
            ],
        }
    )

    with pytest.raises(GeminiGenerationError, match="not grounded in the chunk"):
        _validate_result_against_evidence(
            result=parsed,
            evidence_package=evidence_package,
            expected_answer_mode="exact_lookup",
        )


def test_repair_citation_snippets_replaces_paraphrased_gemini_quote() -> None:
    """Gemini answers should not be discarded when only the quote is paraphrased."""

    evidence_package = EvidencePackage(
        retrieved_chunk_ids=["chunk-1"],
        selected_evidence_ids=["chunk-1"],
        items=[
            _evidence_item(
                citation_id="E001",
                chunk_id="chunk-1",
                text=(
                    "Continual learning, also known as lifelong learning, studies "
                    "how to learn from an infinite stream of data."
                ),
            ),
        ],
    )
    parsed = _parse_result(
        {
            "is_refusal": False,
            "answer_mode": "open",
            "answer_text": "Continual learning studies an infinite stream of data [E001].",
            "cited_chunk_ids": ["chunk-1"],
            "citation_snippets": [
                {"chunk_id": "chunk-1", "snippet": "continual learning studies streams of data"}
            ],
        }
    )

    repaired = _repair_citation_snippets(
        result=parsed,
        evidence_package=evidence_package,
        expected_answer_mode="open",
    )

    assert repaired.citation_snippets["chunk-1"] in evidence_package.items[0].text
    _validate_result_against_evidence(
        result=repaired,
        evidence_package=evidence_package,
        expected_answer_mode="open",
    )


def test_coerce_json_text_recovers_fenced_json_object() -> None:
    """Gemini JSON coercion should recover fenced JSON cleanly."""

    rendered = """```json
{"is_refusal":false,"answer_mode":"exact_lookup","answer_text":"Heliox Transit","cited_chunk_ids":["chunk-1"],"citation_snippets":[{"chunk_id":"chunk-1","snippet":"Heliox Transit"}]}
```"""

    assert _coerce_json_text(rendered) == (
        '{"is_refusal":false,"answer_mode":"exact_lookup","answer_text":"Heliox Transit","cited_chunk_ids":["chunk-1"],'
        '"citation_snippets":[{"chunk_id":"chunk-1","snippet":"Heliox Transit"}]}'
    )


def test_coerce_json_text_recovers_first_balanced_object_from_preface() -> None:
    """Gemini JSON coercion should recover the first balanced object from noisy text."""

    rendered = (
        'Here is the JSON:\n'
        '{"is_refusal":false,"answer_mode":"exact_lookup","answer_text":"9 electric vans","cited_chunk_ids":["chunk-1"],'
        '"citation_snippets":[{"chunk_id":"chunk-1","snippet":"9 electric vans"}]}\n'
        'Thanks.'
    )

    assert _coerce_json_text(rendered).startswith('{"is_refusal":false,"answer_mode":"exact_lookup","answer_text":"9 electric vans"')


def test_build_gemini_prompt_enforces_strict_refusal_and_minimal_evidence_payload() -> None:
    """The Gemini prompt should carry mode rules and a reduced evidence payload."""

    evidence_package = EvidencePackage(
        retrieved_chunk_ids=["chunk-1"],
        selected_evidence_ids=["chunk-1"],
        items=[
            _evidence_item(
                citation_id="E001",
                chunk_id="chunk-1",
                text="The city purchased 12 electric vans from Voltara Mobility.",
            ),
        ],
    )

    prompt = _build_prompt(
        query_text="Which company supplied the vans?",
        evidence_package=evidence_package,
    )

    assert "MODE: exact_lookup" in prompt
    assert "answer_text MUST contain only the exact supported answer." in prompt
    assert "document_id=" not in prompt
    assert "chunk_role=" not in prompt
    assert "text=The city purchased 12 electric vans from Voltara Mobility." in prompt
    assert '"answer_mode": "exact_lookup", "answer_text"' in prompt  # the example carries the mode asked for
    assert '["chunk_id1"]' not in prompt  # and a real chunk id, never a placeholder a model could copy
    assert "citation_id=" not in prompt


def test_build_gemini_prompt_uses_broader_mode_for_summary_queries() -> None:
    """Broader questions should use the summary/list prompt mode instead of exact extraction mode."""

    evidence_package = EvidencePackage(
        retrieved_chunk_ids=["chunk-1"],
        selected_evidence_ids=["chunk-1"],
        items=[
            _evidence_item(
                citation_id="E001",
                chunk_id="chunk-1",
                text="Explainable AI in Software Engineering\nThis paper examines trustworthy XAI methods for engineering teams.",
            ),
        ],
    )

    prompt = _build_prompt(
        query_text="What is the dataset about?",
        evidence_package=evidence_package,
    )

    assert "MODE: summary" in prompt
    assert "[section:" not in prompt


def test_generation_config_for_exact_lookup_is_deterministic() -> None:
    """Exact QA should use deterministic provider settings for more stable extraction."""

    assert _generation_config_for_mode("exact_lookup") == {
        "temperature": 0,
        "topP": 0.05,
        "topK": 1,
        "maxOutputTokens": 2048,
    }


@pytest.mark.asyncio()
async def test_generate_answer_from_evidence_delegates_to_generation_backend(monkeypatch) -> None:
    """The generation service should delegate to the grounded generator backend."""

    evidence_package = EvidencePackage(
        retrieved_chunk_ids=["chunk-1"],
        selected_evidence_ids=["chunk-1"],
        items=[
            _evidence_item(
                citation_id="E001",
                chunk_id="chunk-1",
                text="Grounded returns answers anchored in retrieved evidence.",
            ),
        ],
    )

    monkeypatch.setattr(
        "app.services.generation.resolve_generation_backend",
        lambda: GenerationBackend(
            provider_name="local_grounded_v1",
            implementation="local",
        ),
    )

    draft = await generate_answer_from_evidence(
        query_text="What does grounded return?",
        evidence_package=evidence_package,
    )

    assert draft.answer_text == (
        "Grounded returns answers anchored in retrieved evidence. [E001]"
    )
    assert draft.generator_provider == "local-grounded-v1"
    assert draft.support_coverage == 1.0


@pytest.mark.asyncio()
async def test_generate_answer_from_evidence_reports_openai_backend_failure(monkeypatch) -> None:
    """Provider-backed generation should fail closed instead of using local fallback."""

    evidence_package = EvidencePackage(
        retrieved_chunk_ids=["chunk-1"],
        selected_evidence_ids=["chunk-1"],
        items=[
            _evidence_item(
                citation_id="E001",
                chunk_id="chunk-1",
                text="Grounded returns answers anchored in retrieved evidence.",
            ),
        ],
    )

    monkeypatch.setattr(
        "app.services.generation.resolve_generation_backend",
        lambda: GenerationBackend(
            provider_name="openai_compatible_v1",
            implementation="openai_compatible",
        ),
    )

    async def fake_generate_openai_compatible_draft(**kwargs):
        del kwargs
        raise OpenAICompatibleGenerationError("provider unavailable")

    monkeypatch.setattr(
        "app.services.generation.generate_openai_compatible_draft",
        fake_generate_openai_compatible_draft,
    )

    with pytest.raises(GroundedGenerationError, match="configured generation provider"):
        await generate_answer_from_evidence(
            query_text="What does grounded return?",
            evidence_package=evidence_package,
        )


@pytest.mark.asyncio()
async def test_generate_answer_from_evidence_reports_gemini_backend_failure(monkeypatch) -> None:
    """Gemini-backed generation should fail closed when unavailable."""

    evidence_package = EvidencePackage(
        retrieved_chunk_ids=["chunk-1"],
        selected_evidence_ids=["chunk-1"],
        items=[
            _evidence_item(
                citation_id="E001",
                chunk_id="chunk-1",
                text="Grounded returns answers anchored in retrieved evidence.",
            ),
        ],
    )

    monkeypatch.setattr(
        "app.services.generation.resolve_generation_backend",
        lambda: GenerationBackend(
            provider_name="gemini_v1",
            implementation="gemini",
        ),
    )

    async def fake_generate_gemini_draft(**kwargs):
        del kwargs
        raise GeminiGenerationError("provider unavailable")

    monkeypatch.setattr(
        "app.services.generation.generate_gemini_draft",
        fake_generate_gemini_draft,
    )

    with pytest.raises(GroundedGenerationError, match="configured generation provider"):
        await generate_answer_from_evidence(
            query_text="What does grounded return?",
            evidence_package=evidence_package,
        )


@pytest.mark.asyncio()
async def test_generate_answer_from_evidence_rejects_provider_banned_phrase(monkeypatch) -> None:
    """Weak provider phrasing should be rejected before it reaches users."""

    evidence_package = EvidencePackage(
        retrieved_chunk_ids=["chunk-1"],
        selected_evidence_ids=["chunk-1"],
        items=[
            _evidence_item(
                citation_id="E001",
                chunk_id="chunk-1",
                text="The city purchased 12 electric vans from Voltara Mobility.",
            ),
        ],
    )

    monkeypatch.setattr(
        "app.services.generation.resolve_generation_backend",
        lambda: GenerationBackend(
            provider_name="gemini_v1",
            implementation="gemini",
        ),
    )

    async def fake_generate_gemini_draft(**kwargs):
        del kwargs
        return GroundedAnswerDraft(
            answer_text="I found relevant information about Voltara Mobility.",
            cited_evidence_ids=["chunk-1"],
            citation_snippets={"chunk-1": "The city purchased 12 electric vans from Voltara Mobility."},
            generator_provider="gemini:gemini-2.5-flash",
            support_coverage=1.0,
            source_diversity=1,
        )

    monkeypatch.setattr(
        "app.services.generation.generate_gemini_draft",
        fake_generate_gemini_draft,
    )

    with pytest.raises(GroundedGenerationError, match="configured generation provider"):
        await generate_answer_from_evidence(
            query_text="Which company supplied the vans?",
            evidence_package=evidence_package,
        )


@pytest.mark.asyncio()
async def test_generate_answer_from_evidence_rejects_weak_provider_field_answer(monkeypatch) -> None:
    """Provider answers that cite irrelevant field evidence should fail closed."""

    evidence_package = EvidencePackage(
        retrieved_chunk_ids=["chunk-name", "chunk-experience"],
        selected_evidence_ids=["chunk-name", "chunk-experience"],
        items=[
            _evidence_item(
                citation_id="E001",
                chunk_id="chunk-name",
                text="Samrawit Gebremaryam Bahta\nsamrawit@example.com",
            ),
            _evidence_item(
                citation_id="E002",
                chunk_id="chunk-experience",
                text="PROFESSIONAL EXPERIENCE\nAI Engineer at iCog Labs working on grounded retrieval systems.",
            ),
        ],
    )

    monkeypatch.setattr(
        "app.services.generation.resolve_generation_backend",
        lambda: GenerationBackend(
            provider_name="gemini_v1",
            implementation="gemini",
        ),
    )

    async def fake_generate_gemini_draft(**kwargs):
        del kwargs
        from app.pipeline.contracts import GroundedAnswerDraft

        return GroundedAnswerDraft(
            answer_text="Samrawit Gebremaryam Bahta [E001] PROFESSIONAL EXPERIENCE [E002]",
            cited_evidence_ids=["chunk-name", "chunk-experience"],
            citation_snippets={
                "chunk-name": "Samrawit Gebremaryam Bahta",
                "chunk-experience": "PROFESSIONAL EXPERIENCE",
            },
            generator_provider="gemini:gemini-2.5-flash",
            support_coverage=1.0,
            source_diversity=2,
        )

    monkeypatch.setattr(
        "app.services.generation.generate_gemini_draft",
        fake_generate_gemini_draft,
    )

    with pytest.raises(GroundedGenerationError, match="configured generation provider"):
        await generate_answer_from_evidence(
            query_text="What is the name of the person?",
            evidence_package=evidence_package,
        )


@pytest.mark.asyncio()
async def test_generate_answer_from_evidence_rejects_weak_provider_summary_answer(monkeypatch) -> None:
    """Fragmentary provider summaries should fail closed."""

    evidence_package = EvidencePackage(
        retrieved_chunk_ids=["chunk-header", "chunk-sections"],
        selected_evidence_ids=["chunk-header", "chunk-sections"],
        items=[
            _evidence_item(
                citation_id="E001",
                chunk_id="chunk-header",
                text=(
                    "Samrawit Gebremaryam Bahta\n"
                    "EDUCATION\nBSc in Software Engineering"
                ),
            ),
            _evidence_item(
                citation_id="E002",
                chunk_id="chunk-sections",
                text=(
                    "PROFESSIONAL EXPERIENCE\nAI Engineer at iCog Labs\n"
                    "TECHNICAL SKILLS\nPython, Go, TypeScript"
                ),
            ),
        ],
    )

    monkeypatch.setattr(
        "app.services.generation.resolve_generation_backend",
        lambda: GenerationBackend(
            provider_name="gemini_v1",
            implementation="gemini",
        ),
    )

    async def fake_generate_gemini_draft(**kwargs):
        del kwargs
        from app.pipeline.contracts import GroundedAnswerDraft

        return GroundedAnswerDraft(
            answer_text="across heterogeneous datasets. [E001]",
            cited_evidence_ids=["chunk-header"],
            citation_snippets={
                "chunk-header": "across heterogeneous datasets.",
            },
            generator_provider="gemini:gemini-2.5-flash",
            support_coverage=0.5,
            source_diversity=1,
        )

    monkeypatch.setattr(
        "app.services.generation.generate_gemini_draft",
        fake_generate_gemini_draft,
    )

    with pytest.raises(GroundedGenerationError, match="configured generation provider"):
        await generate_answer_from_evidence(
            query_text="What is this dataset about?",
            evidence_package=evidence_package,
        )


@pytest.mark.asyncio()
async def test_generate_answer_from_evidence_allows_two_chunk_provider_answer_for_start_end_lookup(
    monkeypatch,
) -> None:
    """Exact multi-part lookups should allow two cited chunks when both are needed."""

    evidence_package = EvidencePackage(
        retrieved_chunk_ids=["chunk-start", "chunk-end"],
        selected_evidence_ids=["chunk-start", "chunk-end"],
        items=[
            _evidence_item(
                citation_id="E001",
                chunk_id="chunk-start",
                text="Pilot Overview\nIn January 2025, the city of Lydon launched the electric van pilot.",
            ),
            _evidence_item(
                citation_id="E002",
                chunk_id="chunk-end",
                text="Pilot Overview\nThe pilot lasted for six months and ended in June 2025.",
            ),
        ],
    )

    monkeypatch.setattr(
        "app.services.generation.resolve_generation_backend",
        lambda: GenerationBackend(
            provider_name="gemini_v1",
            implementation="gemini",
        ),
    )

    async def fake_generate_gemini_draft(**kwargs):
        del kwargs
        return GroundedAnswerDraft(
            answer_text="It started in January 2025 and ended in June 2025.",
            cited_evidence_ids=["chunk-start", "chunk-end"],
            citation_snippets={
                "chunk-start": "In January 2025, the city of Lydon launched the electric van pilot.",
                "chunk-end": "The pilot lasted for six months and ended in June 2025.",
            },
            generator_provider="gemini:gemini-2.5-flash",
            support_coverage=1.0,
            source_diversity=1,
        )

    monkeypatch.setattr(
        "app.services.generation.generate_gemini_draft",
        fake_generate_gemini_draft,
    )

    draft = await generate_answer_from_evidence(
        query_text="When did the pilot start and end?",
        evidence_package=evidence_package,
    )

    assert draft.answer_text == "It started in January 2025 and ended in June 2025."
    assert draft.generator_provider == "gemini:gemini-2.5-flash"


@pytest.mark.asyncio()
async def test_generate_answer_from_evidence_rejects_heading_only_provider_collection_answer(monkeypatch) -> None:
    """Provider list answers should fail closed when they echo headings."""

    evidence_package = EvidencePackage(
        retrieved_chunk_ids=["chunk-paper-tools"],
        selected_evidence_ids=["chunk-paper-tools"],
        items=[
            _evidence_item(
                citation_id="E001",
                chunk_id="chunk-paper-tools",
                text=(
                    "The What-If Tool demonstrates how visual analytics can support model debugging, "
                    "feature sensitivity analysis, and fairness inspection with low coding overhead. "
                    "Together with SHAP and LIME, it shows how XAI tooling is entering practical engineering workflows."
                ),
            ),
        ],
    )

    monkeypatch.setattr(
        "app.services.generation.resolve_generation_backend",
        lambda: GenerationBackend(
            provider_name="gemini_v1",
            implementation="gemini",
        ),
    )

    async def fake_generate_gemini_draft(**kwargs):
        del kwargs
        from app.pipeline.contracts import GroundedAnswerDraft

        return GroundedAnswerDraft(
            answer_text="2. Methods, Technologies, and Tools [E001]",
            cited_evidence_ids=["chunk-paper-tools"],
            citation_snippets={
                "chunk-paper-tools": "2. Methods, Technologies, and Tools",
            },
            generator_provider="gemini:gemini-2.5-flash",
            support_coverage=1.0,
            source_diversity=1,
        )

    monkeypatch.setattr(
        "app.services.generation.generate_gemini_draft",
        fake_generate_gemini_draft,
    )

    with pytest.raises(GroundedGenerationError, match="configured generation provider"):
        await generate_answer_from_evidence(
            query_text="What are the tools?",
            evidence_package=evidence_package,
        )
