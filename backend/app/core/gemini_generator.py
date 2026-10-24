"""Gemini grounded generation helpers."""

from __future__ import annotations

import asyncio
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from app.config import get_settings
from app.core.telemetry import get_logger
from app.core.query_analysis import (
    build_query_profile,
    final_answer_mode,
    query_focus_hints,
)
from app.core.provider_retry import run_with_retries
from app.pipeline.contracts import EvidencePackage, GroundedAnswerDraft


REFUSAL_TEXT = "I could not find the answer in the provided context."
logger = get_logger("app.gemini_generator")


class GeminiGenerationError(RuntimeError):
    """Raised when the Gemini generation backend cannot return a draft."""


@dataclass(frozen=True)
class GeminiResult:
    """Parsed grounded result returned by Gemini."""

    is_refusal: bool
    answer_mode: str
    answer_text: str
    cited_chunk_ids: list[str]
    citation_snippets: dict[str, str]


def _build_schema() -> dict[str, object]:
    """Return the structured JSON schema expected from Gemini."""

    return {
        "type": "OBJECT",
        "properties": {
            "is_refusal": {"type": "BOOLEAN"},
            "answer_mode": {"type": "STRING"},
            "answer_text": {"type": "STRING"},
            "cited_chunk_ids": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
            },
            "citation_snippets": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "chunk_id": {"type": "STRING"},
                        "snippet": {"type": "STRING"},
                    },
                    "required": ["chunk_id", "snippet"],
                },
            },
        },
        "required": [
            "is_refusal",
            "answer_mode",
            "answer_text",
            "cited_chunk_ids",
            "citation_snippets",
        ],
    }


def _answer_mode(profile) -> str:
    """Return the high-level answer mode for provider prompting."""

    return final_answer_mode(profile)


def _render_mode_instructions(answer_mode: str, query_kind: str) -> list[str]:
    """Return mode-specific prompt rules."""

    if answer_mode == "exact_lookup":
        rules = [
            "MODE: exact_lookup",
            "answer_text MUST contain only the exact supported answer.",
            "Do not add prefixes, explanations, or extra wording.",
            "When a direct span exists in evidence, copy that span as closely as possible.",
            "If the answer is unsupported, set is_refusal=true.",
        ]
        if query_kind == "count":
            rules.append("For count questions, return only the count or count plus explicit unit.")
        return rules

    if answer_mode == "event_lookup":
        return [
            "MODE: event_lookup",
            "Return one concise grounded sentence describing the event.",
            "Do not include unrelated background details.",
            "If the event is unsupported, set is_refusal=true.",
        ]

    if answer_mode == "list_or_recommendation":
        return [
            "MODE: list_or_recommendation",
            "Return only the directly supported items.",
            "Use short semicolon-separated items when there are multiple items.",
            "Do not include section headings or nearby unrelated prose.",
            "If the list or recommendation is unsupported, set is_refusal=true.",
        ]

    if answer_mode == "summary":
        return [
            "MODE: summary",
            "Write a concise grounded answer.",
            "Prefer the main subject and most relevant sections over generic headings.",
            "Do not copy long paragraphs.",
            "If the summary is unsupported, set is_refusal=true.",
        ]

    if answer_mode == "arithmetic_qa":
        return [
            "MODE: arithmetic_qa",
            "Do NOT perform final arithmetic unless a final computed result already appears explicitly in evidence.",
            "If evidence only supplies component values that require calculation, set is_refusal=true.",
            "If a final computed result already appears in evidence, you may extract that exact result.",
        ]

    return [
        "MODE: open",
        "Write a substantial grounded answer using only the supplied evidence.",
        "Use Markdown formatting with short sections or bullets when it improves readability.",
        "For definition or explanation questions, explain what the concept is, how it works, and key mechanisms or criteria found in evidence.",
        "Prefer 3-6 well-developed paragraphs or bullets when the evidence supports that much detail.",
        "Keep citations inline near the claims they support, using citation labels like [E001].",
        "Do not add unsupported details.",
        "If the answer is unsupported, set is_refusal=true.",
    ]


def _build_prompt(*, query_text: str, evidence_package: EvidencePackage) -> str:
    """Render a grounded instruction block for Gemini."""

    profile = build_query_profile(query_text)
    answer_mode = _answer_mode(profile)
    mode_instructions = _render_mode_instructions(answer_mode, profile.query_kind)
    focus_hints = query_focus_hints(profile)
    evidence_sections: list[str] = []
    for item in evidence_package.items:
        rendered_text = item.text.strip()
        if answer_mode in {"summary", "list_or_recommendation", "open"} and item.section_title:
            rendered_text = f"[section: {item.section_title}]\n{rendered_text}"
        evidence_sections.append(
            "\n".join(
                [
                    f"chunk_id={item.chunk_id}",
                    f"text={rendered_text}",
                ]
            )
        )

    # The example carries the mode asked for and a real chunk id: a model that copies the example then copies
    # the right values, where a placeholder ("summary", "chunk_id1") was copied verbatim and rejected.
    example_id = evidence_package.items[0].chunk_id if evidence_package.items else "chunk_id1"
    example = json.dumps(
        {
            "is_refusal": False,
            "answer_mode": answer_mode,
            "answer_text": "your answer here",
            "cited_chunk_ids": [example_id],
            "citation_snippets": [{"chunk_id": example_id, "snippet": "relevant text"}],
        }
    )
    return "\n\n".join(
        [
            "You are a helpful assistant that answers questions using the provided evidence.",
            "Answer the user's question based ONLY on the evidence below.",
            "Extract, synthesize, and explain relevant information from the evidence.",
            "If evidence contains related information, use it to form your answer.",
            "Instructions:",
            "\n".join(f"- {instruction}" for instruction in mode_instructions),
            "Focus hints:",
            "\n".join(f"- {hint}" for hint in focus_hints) if focus_hints else "- Use the query wording to choose the answer shape.",
            "Return this JSON:",
            example,
            f"answer_mode must be exactly {answer_mode!r}, the MODE named above.",
            "cited_chunk_ids holds chunk_id values copied exactly from the evidence below; never invent one.",
            f'When the evidence does not answer the question, set is_refusal to true, answer_text to exactly "{REFUSAL_TEXT}", and cite nothing.',
            "answer_text may contain Markdown, but the overall response must still be valid JSON.",
            "Never refuse unless evidence is completely unrelated.",
            f"Query: {query_text}",
            "Evidence:",
            "\n\n".join(evidence_sections),
        ]
    )


def _generation_config_for_mode(answer_mode: str) -> dict[str, object]:
    """Return deterministic generation settings for the requested answer mode."""

    if answer_mode in {"exact_lookup", "event_lookup", "arithmetic_qa"}:
        return {
            "temperature": 0,
            "topP": 0.05,
            "topK": 1,
            "maxOutputTokens": 2048,  # Increased significantly
        }
    if answer_mode == "list_or_recommendation":
        return {
            "temperature": 0.1,
            "topP": 0.2,
            "topK": 5,
            "maxOutputTokens": 3072,  # Increased
        }
    return {
        "temperature": 0.2,
        "topP": 0.8,
        "topK": 20,
        "maxOutputTokens": 4096,  # Maximum for longer answers
    }


def _extract_text_from_candidate(payload: dict[str, object]) -> str:
    """Extract plain text content from the first Gemini candidate."""

    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise GeminiGenerationError("Gemini returned no candidates.")
    first = candidates[0]
    if not isinstance(first, dict):
        raise GeminiGenerationError("Gemini returned a malformed candidate.")
    content = first.get("content")
    if not isinstance(content, dict):
        raise GeminiGenerationError("Gemini returned no candidate content.")
    parts = content.get("parts")
    if not isinstance(parts, list) or not parts:
        raise GeminiGenerationError("Gemini returned no content parts.")

    texts: list[str] = []
    for part in parts:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            texts.append(part["text"].strip())
    rendered = "\n".join(text for text in texts if text)
    if not rendered:
        raise GeminiGenerationError("Gemini returned empty text content.")
    return rendered


def _extract_balanced_json_object(text: str) -> str | None:
    """Try to recover the first balanced JSON object from model output."""

    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _coerce_json_text(rendered: str) -> str:
    """Normalize Gemini text output into a best-effort JSON string."""

    stripped = rendered.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
        stripped = stripped.strip()
    balanced = _extract_balanced_json_object(stripped)
    return balanced or stripped


def _parse_result(payload: dict[str, object]) -> GeminiResult:
    """Validate Gemini JSON output into the internal grounded result format."""

    is_refusal = payload.get("is_refusal")
    if not isinstance(is_refusal, bool):
        raise GeminiGenerationError("Gemini returned invalid is_refusal.")

    answer_mode = payload.get("answer_mode")
    if not isinstance(answer_mode, str) or not answer_mode.strip():
        raise GeminiGenerationError("Gemini returned invalid answer_mode.")
    answer_mode = answer_mode.strip()

    answer_text = payload.get("answer_text")
    if not isinstance(answer_text, str) or not answer_text.strip():
        raise GeminiGenerationError("Gemini returned an empty answer_text.")
    answer_text = answer_text.strip()

    cited_raw = payload.get("cited_chunk_ids", [])
    if not isinstance(cited_raw, list):
        raise GeminiGenerationError("Gemini returned invalid cited_chunk_ids.")
    cited_chunk_ids: list[str] = []
    for item in cited_raw:
        if not isinstance(item, str) or not item.strip():
            raise GeminiGenerationError("Gemini returned invalid cited_chunk_ids.")
        cited_chunk_ids.append(item.strip())
    if len(set(cited_chunk_ids)) != len(cited_chunk_ids):
        raise GeminiGenerationError("Gemini returned duplicate cited_chunk_ids.")

    snippets_raw = payload.get("citation_snippets", [])
    if not isinstance(snippets_raw, list):
        raise GeminiGenerationError("Gemini returned invalid citation_snippets.")

    citation_snippets: dict[str, str] = {}
    for item in snippets_raw:
        if not isinstance(item, dict):
            raise GeminiGenerationError("Gemini returned invalid citation_snippets.")
        chunk_id = item.get("chunk_id")
        snippet = item.get("snippet")
        if not (
            isinstance(chunk_id, str)
            and chunk_id.strip()
            and isinstance(snippet, str)
            and snippet.strip()
        ):
            raise GeminiGenerationError("Gemini returned invalid citation_snippets.")
        normalized_chunk_id = chunk_id.strip()
        if normalized_chunk_id in citation_snippets:
            raise GeminiGenerationError("Gemini returned duplicate citation snippets for a chunk.")
        citation_snippets[normalized_chunk_id] = snippet.strip()

    if is_refusal:
        # A refusal is a refusal however the model words it; the caller sees the one refusal text and no
        # citations, instead of a provider error over the wording.
        return GeminiResult(
            is_refusal=True,
            answer_mode=answer_mode,
            answer_text=REFUSAL_TEXT,
            cited_chunk_ids=[],
            citation_snippets={},
        )

    if answer_text == REFUSAL_TEXT:
        raise GeminiGenerationError("Gemini returned a refusal text without is_refusal=true.")
    if not cited_chunk_ids:
        raise GeminiGenerationError("Gemini non-refusal must cite at least one chunk.")
    if set(cited_chunk_ids) != set(citation_snippets):
        raise GeminiGenerationError("Gemini citation_snippets must align exactly with cited_chunk_ids.")

    return GeminiResult(
        is_refusal=False,
        answer_mode=answer_mode,
        answer_text=answer_text,
        cited_chunk_ids=cited_chunk_ids,
        citation_snippets=citation_snippets,
    )


def _normalize_for_match(text: str) -> str:
    """Normalize text lightly for substring checks."""

    return re.sub(r"\s+", " ", text.strip())


def _snippet_matches_evidence(*, snippet: str, evidence_text: str) -> bool:
    """Return whether the snippet is grounded in the evidence chunk."""

    if snippet in evidence_text:
        return True
    normalized_snippet = _normalize_for_match(snippet)
    normalized_evidence = _normalize_for_match(evidence_text)
    return bool(normalized_snippet) and normalized_snippet in normalized_evidence


def _mode_family(answer_mode: str) -> str:
    """Group answer modes by compatible validation and repair behavior."""

    if answer_mode in {"open", "summary"}:
        return "synthesis"
    if answer_mode == "list_or_recommendation":
        return "structured_list"
    if answer_mode in {"exact_lookup", "event_lookup", "arithmetic_qa"}:
        return "strict_extraction"
    return answer_mode


def _answer_mode_is_compatible(*, actual: str, expected: str) -> bool:
    return actual == expected or _mode_family(actual) == _mode_family(expected)


def _tokens_for_overlap(text: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[A-Za-z0-9]+", text)
        if len(token) >= 4
    }


def _candidate_evidence_sentences(text: str) -> list[str]:
    candidates = [
        sentence.strip()
        for sentence in re.split(r"(?<!\d\.)(?<=[.!?])\s+|\n+|[\u2022\u00B7]+", text)
        if sentence.strip()
    ]
    return candidates or [text.strip()]


def _best_grounded_snippet(*, answer_text: str, evidence_text: str) -> str:
    answer_terms = _tokens_for_overlap(answer_text)
    best_sentence = ""
    best_score = -1
    for sentence in _candidate_evidence_sentences(evidence_text):
        sentence_terms = _tokens_for_overlap(sentence)
        score = len(answer_terms & sentence_terms)
        if score > best_score:
            best_sentence = sentence
            best_score = score
    return best_sentence or evidence_text.strip()


def _repair_citation_snippets(
    *,
    result: GeminiResult,
    evidence_package: EvidencePackage,
    expected_answer_mode: str,
) -> GeminiResult:
    """Replace provider paraphrased snippets with exact grounded evidence spans."""

    if result.is_refusal or _mode_family(expected_answer_mode) == "strict_extraction":
        return result
    item_by_chunk_id = {item.chunk_id: item for item in evidence_package.items}
    repaired_snippets: dict[str, str] = {}
    for chunk_id in result.cited_chunk_ids:
        item = item_by_chunk_id.get(chunk_id)
        if item is None:
            repaired_snippets[chunk_id] = result.citation_snippets.get(chunk_id, "")
            continue
        snippet = result.citation_snippets.get(chunk_id, "")
        if snippet and _snippet_matches_evidence(snippet=snippet, evidence_text=item.text):
            repaired_snippets[chunk_id] = snippet
            continue
        repaired_snippet = _best_grounded_snippet(
            answer_text=result.answer_text,
            evidence_text=item.text,
        )
        repaired_snippets[chunk_id] = repaired_snippet
        logger.info(
            "gemini_citation_snippet_repaired",
            chunk_id=chunk_id,
            original_snippet_preview=snippet[:180] if snippet else "",
            repaired_snippet_preview=repaired_snippet[:180],
        )
    return GeminiResult(
        is_refusal=result.is_refusal,
        answer_mode=result.answer_mode,
        answer_text=result.answer_text,
        cited_chunk_ids=result.cited_chunk_ids,
        citation_snippets=repaired_snippets,
    )


def _validate_result_against_evidence(
    *,
    result: GeminiResult,
    evidence_package: EvidencePackage,
    expected_answer_mode: str,
) -> None:
    """Validate cited chunk IDs and snippets against the evidence package."""

    allowed_items = {item.chunk_id: item for item in evidence_package.items}
    if not _answer_mode_is_compatible(
        actual=result.answer_mode,
        expected=expected_answer_mode,
    ):
        raise GeminiGenerationError("Gemini returned an unexpected answer_mode.")
    if result.is_refusal:
        return

    for chunk_id in result.cited_chunk_ids:
        item = allowed_items.get(chunk_id)
        if item is None:
            raise GeminiGenerationError("Gemini returned an unknown cited chunk_id.")
        snippet = result.citation_snippets.get(chunk_id)
        if not snippet:
            raise GeminiGenerationError("Gemini omitted a required citation snippet.")
        if not _snippet_matches_evidence(snippet=snippet, evidence_text=item.text):
            raise GeminiGenerationError("Gemini returned a citation snippet not grounded in the chunk.")


async def generate_gemini_draft(
    *,
    query_text: str,
    evidence_package: EvidencePackage,
    agent_instructions: str = "",
) -> GroundedAnswerDraft:
    """Generate a grounded draft using the Gemini API."""

    settings = get_settings()
    if not settings.gemini_api_key:
        raise GeminiGenerationError("GEMINI_API_KEY is not configured.")
    profile = build_query_profile(query_text)
    answer_mode = _answer_mode(profile)

    # Build prompt for logging
    prompt_text = _build_prompt(
        query_text=query_text,
        evidence_package=evidence_package,
    )
    logger.debug(
        "gemini_prompt",
        query_text=query_text,
        evidence_count=len(evidence_package.items),
        answer_mode=answer_mode,
        prompt_length=len(prompt_text),
    )

    model_name = settings.gemini_model
    if not model_name.startswith("models/"):
        model_name = f"models/{model_name}"
    encoded_model = urllib.parse.quote(model_name, safe="/")
    base_url = str(settings.gemini_base_url).rstrip("/")
    url = f"{base_url}/{encoded_model}:generateContent?key={urllib.parse.quote(settings.gemini_api_key, safe='')}"

    request_payload: dict[str, object] = {
        "contents": [
            {
                "parts": [
                    {
                        "text": _build_prompt(
                            query_text=query_text,
                            evidence_package=evidence_package,
                        )
                    }
                ]
            }
        ],
        "generationConfig": {
            **_generation_config_for_mode(answer_mode),
            "responseMimeType": "application/json",
            "responseSchema": _build_schema(),
        },
    }
    if agent_instructions.strip():
        request_payload["systemInstruction"] = {
            "parts": [{"text": agent_instructions.strip()}]
        }

    request = urllib.request.Request(
        url=url,
        data=json.dumps(request_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    def _perform_request() -> str:
        with urllib.request.urlopen(
            request,
            timeout=settings.openai_timeout_seconds,
        ) as response:
            return response.read().decode("utf-8")

    try:
        raw_body = await run_with_retries(
            lambda: asyncio.to_thread(_perform_request),
            max_retries=settings.provider_max_retries,
            backoff_ms=settings.provider_retry_backoff_ms,
        )
        logger.debug(
            "gemini_raw_response",
            response_length=len(raw_body),
            response_preview=raw_body[:500],
        )
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        if exc.code == 429:
            raise GeminiGenerationError(
                f"Gemini generation hit rate limits (HTTP 429): {detail}"
            ) from exc
        raise GeminiGenerationError(
            f"Gemini generation failed with HTTP {exc.code}: {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise GeminiGenerationError("Gemini generation request failed.") from exc

    try:
        response_payload = json.loads(raw_body)
        content_text = _extract_text_from_candidate(response_payload)
        parsed = _parse_result(json.loads(_coerce_json_text(content_text)))
        parsed = _repair_citation_snippets(
            result=parsed,
            evidence_package=evidence_package,
            expected_answer_mode=answer_mode,
        )
        _validate_result_against_evidence(
            result=parsed,
            evidence_package=evidence_package,
            expected_answer_mode=answer_mode,
        )
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        # Check if MAX_TOKENS - provide better error
        if "candidates" in response_payload:
            usage = response_payload.get("usageMetadata", {})
            finish_reason = None
            if "candidates" in response_payload and response_payload["candidates"]:
                finish_reason = response_payload["candidates"][0].get("finishReason")
            if finish_reason == "MAX_TOKENS":
                raise GeminiGenerationError(
                    f"Gemini ran out of tokens (maxOutputTokens limit). "
                    f"Prompt was {usage.get('promptTokenCount', '?')} tokens, "
                    f"generated {usage.get('candidatesTokenCount', '?')} tokens."
                ) from exc
        raise GeminiGenerationError("Gemini generation returned invalid JSON.") from exc

    used_items = [
        item for item in evidence_package.items if item.chunk_id in set(parsed.cited_chunk_ids)
    ]
    support_coverage = min(len(used_items) / max(len(evidence_package.items), 1), 1.0)
    source_diversity = len({source for item in used_items for source in item.sources})

    usage_meta = response_payload.get("usageMetadata", {})
    prompt_tokens = int(usage_meta.get("promptTokenCount", 0))
    completion_tokens = int(usage_meta.get("candidatesTokenCount", 0))

    return GroundedAnswerDraft(
        answer_text=parsed.answer_text,
        cited_evidence_ids=parsed.cited_chunk_ids,
        citation_snippets=parsed.citation_snippets,
        generator_provider=f"gemini:{settings.gemini_model}",
        support_coverage=round(support_coverage, 4),
        source_diversity=source_diversity,
        token_usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )
