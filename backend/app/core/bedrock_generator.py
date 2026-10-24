"""AWS Bedrock grounded generation over the Converse API.

The grounded JSON contract (is_refusal, answer_mode, answer_text, cited_chunk_ids, citation_snippets), its
prompt, snippet repair and evidence validation live in gemini_generator and are shared here rather than
copied, so both provider backends are held to the same rules; only the transport differs. Converse has no
response schema, so the prompt's JSON example carries the shape and the output is coerced and validated.
"""

from __future__ import annotations

import asyncio
import json

from botocore.exceptions import BotoCoreError, ClientError

from app.config import get_settings
from app.core import bedrock_client, gemini_generator
from app.core.gemini_generator import GeminiGenerationError
from app.core.provider_retry import run_with_retries
from app.core.query_analysis import build_query_profile
from app.core.telemetry import get_logger
from app.pipeline.contracts import EvidencePackage, GroundedAnswerDraft


logger = get_logger("app.bedrock_generator")


class BedrockGenerationError(RuntimeError):
    """Raised when the Bedrock generation backend cannot return a draft."""


def _inference_config(answer_mode: str) -> dict[str, object]:
    """Converse inference settings from the shared per-mode profile (topK is not a Converse field)."""

    config = gemini_generator._generation_config_for_mode(answer_mode)
    return {
        "temperature": config["temperature"],
        "topP": config["topP"],
        "maxTokens": config["maxOutputTokens"],
    }


def _message_text(response: dict[str, object]) -> str:
    """Extract the assistant text from a Converse response."""

    output = response.get("output")
    message = output.get("message") if isinstance(output, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, list):
        raise BedrockGenerationError("Bedrock returned no message content.")
    texts = [
        block["text"].strip()
        for block in content
        if isinstance(block, dict) and isinstance(block.get("text"), str) and block["text"].strip()
    ]
    if not texts:
        raise BedrockGenerationError("Bedrock returned empty text content.")
    return "\n".join(texts)


async def generate_bedrock_draft(
    *,
    query_text: str,
    evidence_package: EvidencePackage,
    agent_instructions: str = "",
) -> GroundedAnswerDraft:
    """Generate a grounded draft with a Bedrock model through the Converse API."""

    settings = get_settings()
    if not settings.bedrock_model:
        raise BedrockGenerationError("BEDROCK_MODEL is not configured.")
    answer_mode = gemini_generator._answer_mode(build_query_profile(query_text))
    prompt = gemini_generator._build_prompt(query_text=query_text, evidence_package=evidence_package)
    logger.debug(
        "bedrock_prompt",
        query_text=query_text,
        evidence_count=len(evidence_package.items),
        answer_mode=answer_mode,
        prompt_length=len(prompt),
    )

    request: dict[str, object] = {
        "modelId": settings.bedrock_model,
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": _inference_config(answer_mode),
    }
    if agent_instructions.strip():
        request["system"] = [{"text": agent_instructions.strip()}]

    client = bedrock_client.get_bedrock_runtime_client()
    try:
        response = await run_with_retries(
            lambda: asyncio.to_thread(client.converse, **request),
            max_retries=settings.provider_max_retries,
            backoff_ms=settings.provider_retry_backoff_ms,
            should_retry=bedrock_client.is_retryable_bedrock_error,
        )
    except ClientError as exc:
        raise BedrockGenerationError(
            f"Bedrock generation failed: {bedrock_client.error_detail(exc)}"
        ) from exc
    except BotoCoreError as exc:
        raise BedrockGenerationError("Bedrock generation request failed.") from exc

    stop_reason = response.get("stopReason")
    if stop_reason != "end_turn":
        raise BedrockGenerationError(
            f"Bedrock stopped before the answer was complete (stopReason={stop_reason})."
        )

    try:
        content_text = _message_text(response)
        parsed = gemini_generator._parse_result(json.loads(gemini_generator._coerce_json_text(content_text)))
        parsed = gemini_generator._repair_citation_snippets(
            result=parsed,
            evidence_package=evidence_package,
            expected_answer_mode=answer_mode,
        )
        gemini_generator._validate_result_against_evidence(
            result=parsed,
            evidence_package=evidence_package,
            expected_answer_mode=answer_mode,
        )
    except (json.JSONDecodeError, TypeError) as exc:
        raise BedrockGenerationError("Bedrock generation returned invalid JSON.") from exc
    except GeminiGenerationError as exc:  # the shared contract's rejection, worded for its home module
        raise BedrockGenerationError(f"Bedrock generation output rejected: {exc}") from exc

    cited = set(parsed.cited_chunk_ids)
    used_items = [item for item in evidence_package.items if item.chunk_id in cited]
    support_coverage = min(len(used_items) / max(len(evidence_package.items), 1), 1.0)
    source_diversity = len({source for item in used_items for source in item.sources})

    usage = response.get("usage") or {}
    prompt_tokens = int(usage.get("inputTokens", 0))
    completion_tokens = int(usage.get("outputTokens", 0))

    return GroundedAnswerDraft(
        answer_text=parsed.answer_text,
        cited_evidence_ids=parsed.cited_chunk_ids,
        citation_snippets=parsed.citation_snippets,
        generator_provider=f"bedrock:{settings.bedrock_model}",
        support_coverage=round(support_coverage, 4),
        source_diversity=source_diversity,
        token_usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": int(usage.get("totalTokens", prompt_tokens + completion_tokens)),
        },
    )
