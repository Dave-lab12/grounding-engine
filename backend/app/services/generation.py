"""Grounded generation helpers built on top of packaged evidence."""

from __future__ import annotations

from dataclasses import dataclass
import re

from app.config import get_settings
from app.core.bedrock_generator import BedrockGenerationError, generate_bedrock_draft
from app.core.gemini_generator import GeminiGenerationError, generate_gemini_draft
from app.core.llm_client import GroundedGenerationError, generate_grounded_draft
from app.core.openai_generator import (
    OpenAICompatibleGenerationError,
    generate_openai_compatible_draft,
)
from app.core.query_analysis import (
    build_query_profile,
    final_answer_mode,
    is_dataset_summary_query,
    needs_multi_chunk_exact_support,
    tokenize_meaningful_terms,
)
from app.core.telemetry import get_logger
from app.pipeline.contracts import EvidenceItem, EvidencePackage, GroundedAnswerDraft


logger = get_logger("app.generation")

_SUMMARY_ANSWER_NOISE = {
    "about",
    "attached",
    "contain",
    "contains",
    "cover",
    "covers",
    "data",
    "dataset",
    "datasets",
    "document",
    "documents",
    "file",
    "files",
    "include",
    "includes",
    "information",
    "item",
    "items",
    "section",
    "sections",
    "summary",
}

_FIELD_ANSWER_NOISE = {
    "answer",
    "contact",
    "details",
    "field",
    "information",
    "is",
    "listed",
    "name",
    "person",
    "the",
}

_OUTLINE_HEADING_PATTERN = re.compile(r"^\d+(?:\.\d+)*[.)]?\s+[A-Z][A-Za-z0-9/&,\- ]{2,}$")
_HEADING_ONLY_PATTERN = re.compile(r"^[A-Z][A-Z0-9/&,\- ]{2,}$")
_BANNED_PROVIDER_PHRASES = {
    "based on the context",
    "i found relevant",
    "not enough structured evidence",
    "the date is",
}
_UNSUPPORTED_REFUSAL_TEXT = "I could not find the answer in the provided context."


@dataclass(frozen=True)
class GenerationBackend:
    """One concrete grounded generation backend."""

    provider_name: str
    implementation: str = "local"

    async def generate(
        self,
        *,
        query_text: str,
        evidence_package: EvidencePackage,
        agent_instructions: str = "",
    ) -> GroundedAnswerDraft:
        if self.implementation == "gemini":
            return await generate_gemini_draft(
                query_text=query_text,
                evidence_package=evidence_package,
                agent_instructions=agent_instructions,
            )
        if self.implementation == "openai_compatible":
            return await generate_openai_compatible_draft(
                query_text=query_text,
                evidence_package=evidence_package,
                agent_instructions=agent_instructions,
            )
        if self.implementation == "bedrock":
            return await generate_bedrock_draft(
                query_text=query_text,
                evidence_package=evidence_package,
                agent_instructions=agent_instructions,
            )
        return generate_grounded_draft(
            query_text=query_text,
            evidence_package=evidence_package,
            agent_instructions=agent_instructions,
        )


_GENERATION_BACKENDS = {
    "local_grounded_v1": GenerationBackend(
        provider_name="local_grounded_v1",
        implementation="local",
    ),
    "gemini_v1": GenerationBackend(
        provider_name="gemini_v1",
        implementation="gemini",
    ),
    "openai_compatible_v1": GenerationBackend(
        provider_name="openai_compatible_v1",
        implementation="openai_compatible",
    ),
    "bedrock_v1": GenerationBackend(
        provider_name="bedrock_v1",
        implementation="bedrock",
    ),
}


def resolve_generation_backend() -> GenerationBackend:
    """Return the configured grounded generation backend."""

    backend_name = get_settings().generator_backend
    try:
        return _GENERATION_BACKENDS[backend_name]
    except KeyError as exc:
        raise RuntimeError(
            f"Unsupported generator backend configured: {backend_name}."
        ) from exc


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _strip_answer_citations(text: str) -> str:
    return re.sub(r"\[[^\]]+\]", "", text).strip()


def _looks_like_heading_only(text: str) -> bool:
    stripped = re.sub(r"\s+", " ", text).strip().rstrip(":")
    if not stripped:
        return False
    return bool(
        _HEADING_ONLY_PATTERN.fullmatch(stripped)
        or _OUTLINE_HEADING_PATTERN.fullmatch(stripped)
    )


def _significant_terms(text: str, *, noise_terms: set[str]) -> set[str]:
    return {
        term
        for term in tokenize_meaningful_terms(text)
        if term not in noise_terms
    }


def _snippet_matches_evidence(*, snippet: str, evidence_text: str) -> bool:
    normalized_snippet = _normalize_text(snippet)
    normalized_evidence = _normalize_text(evidence_text)
    if not normalized_snippet or not normalized_evidence:
        return False
    return normalized_snippet in normalized_evidence


def _answer_supported_by_snippets(
    *,
    answer_text: str,
    snippets: list[str],
    multi_chunk: bool = False,
    noise_terms: set[str] | None = None,
) -> bool:
    """Return whether the answer is directly supported by the cited snippets."""

    answer_core = _normalize_text(_strip_answer_citations(answer_text))
    if not answer_core:
        return False
    normalized_snippets = [_normalize_text(snippet) for snippet in snippets if snippet.strip()]
    if any(answer_core in snippet for snippet in normalized_snippets):
        return True

    terms = _significant_terms(answer_core, noise_terms=noise_terms or set())
    if not terms:
        return False
    combined_terms: set[str] = set()
    for snippet in normalized_snippets:
        combined_terms.update(tokenize_meaningful_terms(snippet))
    if multi_chunk:
        date_number_tokens = [
            token.casefold()
            for token in re.findall(
                r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|\d{4}|\d[\d,]*(?:\.\d+)?)\b",
                answer_core,
                flags=re.IGNORECASE,
            )
        ]
        if date_number_tokens and all(token in " ".join(normalized_snippets) for token in date_number_tokens):
            return True
        return terms <= combined_terms
    return any(terms <= tokenize_meaningful_terms(snippet) for snippet in normalized_snippets)


def _resolve_cited_items(
    *,
    evidence_package: EvidencePackage,
    draft: GroundedAnswerDraft,
) -> tuple[list[EvidenceItem], str | None]:
    """Resolve cited chunk IDs into evidence items with strict chunk-ID semantics."""

    if len(set(draft.cited_evidence_ids)) != len(draft.cited_evidence_ids):
        return [], "provider returned duplicate cited chunk IDs"

    item_by_chunk_id = {item.chunk_id: item for item in evidence_package.items}
    cited_items: list[EvidenceItem] = []
    for chunk_id in draft.cited_evidence_ids:
        item = item_by_chunk_id.get(chunk_id)
        if item is None:
            return [], "provider returned unknown cited chunk IDs"
        cited_items.append(item)
    return cited_items, None


def _validate_citation_contract(
    *,
    evidence_package: EvidencePackage,
    draft: GroundedAnswerDraft,
) -> tuple[list[EvidenceItem], list[str], str | None]:
    """Validate cited chunk IDs and snippet grounding."""

    cited_items, error = _resolve_cited_items(
        evidence_package=evidence_package,
        draft=draft,
    )
    if error is not None:
        return [], [], error

    snippets: list[str] = []
    for item in cited_items:
        snippet = draft.citation_snippets.get(item.chunk_id)
        if not snippet or not snippet.strip():
            return [], [], "provider omitted citation snippets for cited chunks"
        if not _snippet_matches_evidence(snippet=snippet, evidence_text=item.text):
            return [], [], "provider citation snippet was not grounded in cited evidence"
        snippets.append(snippet.strip())

    if set(draft.citation_snippets) != {item.chunk_id for item in cited_items}:
        return [], [], "provider citation snippets did not align exactly with cited chunk IDs"
    return cited_items, snippets, None


def _validate_refusal_draft(draft: GroundedAnswerDraft) -> str | None:
    """Validate explicit refusal behavior."""

    answer_core = _normalize_text(_strip_answer_citations(draft.answer_text))
    if answer_core != _UNSUPPORTED_REFUSAL_TEXT.casefold():
        return "provider refusal text was not exact"
    if draft.cited_evidence_ids or draft.citation_snippets:
        return "provider refusal cited evidence"
    return None


def _validate_exact_lookup_draft(
    *,
    profile,
    cited_items: list[EvidenceItem],
    snippets: list[str],
    draft: GroundedAnswerDraft,
) -> str | None:
    """Validate exact-lookup provider answers."""

    multi_chunk = needs_multi_chunk_exact_support(profile)
    if not multi_chunk and len(cited_items) != 1:
        return "exact lookup cited too many chunks"
    if multi_chunk and len(cited_items) > 2:
        return "multi-part exact lookup cited too many chunks"
    if len(_strip_answer_citations(draft.answer_text).split()) > (24 if multi_chunk else 18):
        return "exact lookup answer was too long"
    if any(_looks_like_heading_only(snippet) for snippet in snippets):
        return "exact lookup cited heading-only support"
    if not _answer_supported_by_snippets(
        answer_text=draft.answer_text,
        snippets=snippets,
        multi_chunk=multi_chunk,
        noise_terms=_FIELD_ANSWER_NOISE,
    ):
        return "exact lookup answer was not directly supported by snippets"
    return None


def _validate_event_lookup_draft(
    *,
    snippets: list[str],
    draft: GroundedAnswerDraft,
) -> str | None:
    """Validate event lookup answers."""

    answer_core = _strip_answer_citations(draft.answer_text)
    if len(answer_core.split()) < 5 or len(answer_core.split()) > 32:
        return "event lookup answer length was unreasonable"
    if len(re.findall(r"[.!?]", answer_core)) > 1:
        return "event lookup answer contained multiple sentences"
    if any(_looks_like_heading_only(snippet) for snippet in snippets):
        return "event lookup cited heading-only support"
    if not _answer_supported_by_snippets(
        answer_text=draft.answer_text,
        snippets=snippets,
        noise_terms=_FIELD_ANSWER_NOISE,
    ):
        return "event lookup answer was not supported by snippets"
    return None


def _validate_list_or_recommendation_draft(
    *,
    snippets: list[str],
    draft: GroundedAnswerDraft,
) -> str | None:
    """Validate list/recommendation answers."""

    answer_core = _strip_answer_citations(draft.answer_text)
    if len(answer_core.split()) < 3 or len(answer_core.split()) > 90:
        return "list or recommendation answer length was unreasonable"
    items = [
        part.strip(" -*\t")
        for part in re.split(r"[;\n]+", answer_core)
        if part.strip()
    ]
    if not items:
        return "list or recommendation answer contained no items"
    if any(_looks_like_heading_only(item) for item in items):
        return "list or recommendation answer echoed headings"

    combined_terms: set[str] = set()
    for snippet in snippets:
        combined_terms.update(tokenize_meaningful_terms(snippet))

    for item in items:
        item_terms = _significant_terms(item, noise_terms=_FIELD_ANSWER_NOISE)
        if item_terms and not (item_terms & combined_terms):
            return "list or recommendation item lacked cited support"
    return None


def _validate_summary_draft(
    *,
    evidence_package: EvidencePackage,
    cited_items: list[EvidenceItem],
    snippets: list[str],
    draft: GroundedAnswerDraft,
) -> str | None:
    """Validate concise synthesis answers."""

    answer_core = _strip_answer_citations(draft.answer_text)
    if len(answer_core.split()) < 8 or len(answer_core.split()) > 80:
        return "summary answer length was unreasonable"
    if len(evidence_package.items) > 1 and len(cited_items) < 2:
        return "summary answer cited too little evidence"
    answer_terms = _significant_terms(answer_core, noise_terms=_SUMMARY_ANSWER_NOISE)
    snippet_terms: set[str] = set()
    for snippet in snippets:
        snippet_terms.update(_significant_terms(snippet, noise_terms=_SUMMARY_ANSWER_NOISE))
    if len(answer_terms & snippet_terms) < 2:
        return "summary answer lacked enough evidence overlap"
    return None


def _validate_boolean_like_draft(
    *,
    snippets: list[str],
    draft: GroundedAnswerDraft,
) -> str | None:
    """Validate boolean/comparison/entity-context answers."""

    answer_core = _strip_answer_citations(draft.answer_text).strip()
    if not answer_core.casefold().startswith(("yes", "no")):
        return "boolean-like answer did not start with yes or no"
    if not _answer_supported_by_snippets(
        answer_text=draft.answer_text,
        snippets=snippets,
        noise_terms=_FIELD_ANSWER_NOISE,
    ):
        return "boolean-like answer was not supported by snippets"
    return None


def _validate_arithmetic_draft(
    *,
    snippets: list[str],
    draft: GroundedAnswerDraft,
) -> str | None:
    """Validate arithmetic answers conservatively.

    Provider arithmetic is only accepted when the final numeric answer is already
    explicitly present in cited support. Otherwise we prefer local deterministic math.
    """

    answer_core = _strip_answer_citations(draft.answer_text)
    if not re.search(r"\d", answer_core):
        return "arithmetic answer did not contain a numeric result"
    if not _answer_supported_by_snippets(
        answer_text=draft.answer_text,
        snippets=snippets,
        noise_terms=_FIELD_ANSWER_NOISE,
    ):
        return "arithmetic answer was not explicitly present in cited support"
    return None


def _validate_open_draft(
    *,
    snippets: list[str],
    draft: GroundedAnswerDraft,
) -> str | None:
    """Validate general grounded answers conservatively."""

    answer_core = _strip_answer_citations(draft.answer_text)
    if len(answer_core.split()) > 350:
        return "open answer was too long"
    answer_terms = _significant_terms(answer_core, noise_terms=_FIELD_ANSWER_NOISE)
    snippet_terms: set[str] = set()
    for snippet in snippets:
        snippet_terms.update(_significant_terms(snippet, noise_terms=_FIELD_ANSWER_NOISE))
    if len(answer_terms & snippet_terms) < min(3, len(answer_terms)):
        return "open answer was not sufficiently supported by snippets"
    return None


def _provider_validation_error(
    *,
    query_text: str,
    evidence_package: EvidencePackage,
    draft: GroundedAnswerDraft,
) -> str | None:
    """Return a provider validation error string, or None when the draft is acceptable."""

    normalized_answer = _normalize_text(_strip_answer_citations(draft.answer_text))
    if any(phrase in normalized_answer for phrase in _BANNED_PROVIDER_PHRASES):
        return "provider answer contained banned weak phrasing"

    refusal_error = None
    if normalized_answer == _UNSUPPORTED_REFUSAL_TEXT.casefold():
        refusal_error = _validate_refusal_draft(draft)
        return refusal_error

    if not draft.cited_evidence_ids:
        return "provider non-refusal omitted cited chunk IDs"

    cited_items, snippets, citation_error = _validate_citation_contract(
        evidence_package=evidence_package,
        draft=draft,
    )
    if citation_error is not None:
        return citation_error

    profile = build_query_profile(query_text)
    answer_mode = final_answer_mode(profile)

    if answer_mode == "exact_lookup":
        return _validate_exact_lookup_draft(
            profile=profile,
            cited_items=cited_items,
            snippets=snippets,
            draft=draft,
        )
    if answer_mode == "event_lookup":
        return _validate_event_lookup_draft(
            snippets=snippets,
            draft=draft,
        )
    if answer_mode == "list_or_recommendation":
        return _validate_list_or_recommendation_draft(
            snippets=snippets,
            draft=draft,
        )
    if answer_mode == "summary" or is_dataset_summary_query(profile):
        return _validate_summary_draft(
            evidence_package=evidence_package,
            cited_items=cited_items,
            snippets=snippets,
            draft=draft,
        )
    if answer_mode == "boolean_like":
        return _validate_boolean_like_draft(
            snippets=snippets,
            draft=draft,
        )
    if answer_mode == "arithmetic_qa":
        return _validate_arithmetic_draft(
            snippets=snippets,
            draft=draft,
        )
    return _validate_open_draft(
        snippets=snippets,
        draft=draft,
    )


async def generate_answer_from_evidence(
    *,
    query_text: str,
    evidence_package: EvidencePackage,
    agent_instructions: str = "",
) -> GroundedAnswerDraft:
    """Generate a grounded answer draft using the current generation backend."""

    backend = resolve_generation_backend()
    _last_exc: Exception | None = None

    for _attempt in range(2):
        try:
            draft = await backend.generate(
                query_text=query_text,
                evidence_package=evidence_package,
                agent_instructions=agent_instructions,
            )
            if backend.implementation != "local":
                validation_error = _provider_validation_error(
                    query_text=query_text,
                    evidence_package=evidence_package,
                    draft=draft,
                )
                if validation_error is not None:
                    _last_exc = GroundedGenerationError(validation_error)
                    logger.warning(
                        "generation_validation_failed_will_retry",
                        attempt=_attempt,
                        reason=validation_error,
                    )
                    continue
            return draft
        except GroundedGenerationError as exc:
            if backend.implementation == "local":
                raise
            _last_exc = exc
            logger.warning(
                "generation_provider_failed",
                attempt=_attempt,
                configured_backend=backend.provider_name,
                reason=str(exc),
            )
        except (GeminiGenerationError, OpenAICompatibleGenerationError, BedrockGenerationError) as exc:
            _last_exc = exc
            logger.warning(
                "generation_provider_failed",
                attempt=_attempt,
                configured_backend=backend.provider_name,
                reason=str(exc),
            )

    raise GroundedGenerationError(
        "The configured generation provider could not produce a grounded answer."
    ) from _last_exc
