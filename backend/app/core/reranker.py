"""Enterprise reranker interfaces and disabled-safe backends."""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol

from botocore.exceptions import BotoCoreError, ClientError

from app.config import Settings, get_settings
from app.core import bedrock_client
from app.core.provider_retry import run_with_retries
from app.models import ExecutionTier
from app.pipeline.contracts import FusedRetrievedChunk


class RerankerError(RuntimeError):
    """Raised when a provider-backed reranker cannot return a valid result."""


@dataclass(frozen=True)
class RerankerScoredHit:
    """One reranked hit with a stable score and optional debug payload."""

    chunk_id: str
    score: float
    rank: int
    debug: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RerankerResult:
    """Result returned by one reranker invocation."""

    backend_name: str
    applied: bool
    hits: list[RerankerScoredHit]
    debug: dict[str, object] = field(default_factory=dict)


class RetrievalReranker(Protocol):
    """Contract for enterprise rerankers over fused retrieval candidates."""

    backend_name: str

    async def rerank(
        self,
        *,
        query_text: str,
        hits: list[FusedRetrievedChunk],
        limit: int,
    ) -> RerankerResult:
        """Return one ranked view of the provided candidates."""


@dataclass(frozen=True)
class DisabledReranker:
    """Pass-through reranker used when Enterprise reranking is disabled."""

    backend_name: str = "disabled"
    reason: str = "enterprise_reranker_disabled"

    async def rerank(
        self,
        *,
        query_text: str,
        hits: list[FusedRetrievedChunk],
        limit: int,
    ) -> RerankerResult:
        del query_text
        ordered_hits = [
            RerankerScoredHit(
                chunk_id=hit.chunk_id,
                score=hit.fused_score,
                rank=index,
                debug={"reason": self.reason},
            )
            for index, hit in enumerate(hits[:limit], start=1)
        ]
        return RerankerResult(
            backend_name=self.backend_name,
            applied=False,
            hits=ordered_hits,
            debug={"reason": self.reason},
        )


@dataclass(frozen=True)
class StubEnterpriseReranker:
    """Stable no-op backend used to exercise the Enterprise reranker path."""

    backend_name: str = "stub"

    async def rerank(
        self,
        *,
        query_text: str,
        hits: list[FusedRetrievedChunk],
        limit: int,
    ) -> RerankerResult:
        del query_text
        scored_hits = [
            RerankerScoredHit(
                chunk_id=hit.chunk_id,
                score=hit.fused_score,
                rank=index,
                debug={"strategy": "pass_through"},
            )
            for index, hit in enumerate(hits[:limit], start=1)
        ]
        return RerankerResult(
            backend_name=self.backend_name,
            applied=False,
            hits=scored_hits,
            debug={"strategy": "pass_through"},
        )


@dataclass(frozen=True)
class GeminiEnterpriseReranker:
    """Gemini-backed Enterprise reranker for hard retrieval cases."""

    api_key: str
    base_url: str
    model: str
    max_retries: int
    backoff_ms: int
    timeout_seconds: int
    backend_name: str = "gemini_v1"

    @staticmethod
    def _build_prompt(*, query_text: str, hits: list[FusedRetrievedChunk], limit: int) -> str:
        """Render a strict ranking prompt for the model (shared with the Bedrock reranker)."""

        hit_blocks = [
            "\n".join(
                [
                    f"chunk_id={hit.chunk_id}",
                    f"text={hit.text}",
                ]
            )
            for hit in hits
        ]
        return "\n\n".join(
            [
                "You are a strict grounded retrieval reranker.",
                "Rank the supplied chunks by how likely they are to answer the user's question directly and safely.",
                "Prefer chunks with direct answer spans or strong grounded support.",
                "Down-rank chunks that are only loosely related or mostly background context.",
                "Return strict JSON only.",
                "Schema:",
                '{ "hits": [ { "chunk_id": string, "score": number, "reason": string } ] }',
                f"Return at most {limit} hits.",
                f"query={query_text}",
                "candidates:",
                "\n\n".join(hit_blocks),
            ]
        )

    def _build_schema(self) -> dict[str, object]:
        """Return the Gemini JSON schema for reranker results."""

        return {
            "type": "OBJECT",
            "properties": {
                "hits": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "chunk_id": {"type": "STRING"},
                            "score": {"type": "NUMBER"},
                            "reason": {"type": "STRING"},
                        },
                        "required": ["chunk_id", "score", "reason"],
                    },
                }
            },
            "required": ["hits"],
        }

    @staticmethod
    def _extract_candidate_text(payload: dict[str, object]) -> str:
        """Extract the first text candidate from a Gemini response."""

        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise RerankerError("Gemini reranker returned no candidates.")
        first = candidates[0]
        if not isinstance(first, dict):
            raise RerankerError("Gemini reranker returned a malformed candidate.")
        content = first.get("content")
        if not isinstance(content, dict):
            raise RerankerError("Gemini reranker returned no candidate content.")
        parts = content.get("parts")
        if not isinstance(parts, list) or not parts:
            raise RerankerError("Gemini reranker returned no content parts.")

        text_parts = [
            part["text"].strip()
            for part in parts
            if isinstance(part, dict) and isinstance(part.get("text"), str) and part["text"].strip()
        ]
        if not text_parts:
            raise RerankerError("Gemini reranker returned empty text content.")
        return "\n".join(text_parts)

    @staticmethod
    def _coerce_json_text(raw_text: str) -> dict[str, object]:
        """Coerce Gemini response text into one JSON object."""

        normalized = raw_text.strip()
        if normalized.startswith("```"):
            normalized = normalized.strip("`")
            if normalized.startswith("json"):
                normalized = normalized[4:].strip()

        try:
            return json.loads(normalized)
        except json.JSONDecodeError:
            match = None
            depth = 0
            start_index = -1
            for index, char in enumerate(normalized):
                if char == "{":
                    if depth == 0:
                        start_index = index
                    depth += 1
                elif char == "}":
                    if depth == 0:
                        continue
                    depth -= 1
                    if depth == 0 and start_index >= 0:
                        match = normalized[start_index : index + 1]
                        break
            if match is None:
                raise RerankerError("Gemini reranker returned invalid JSON.")
            try:
                return json.loads(match)
            except json.JSONDecodeError as exc:
                raise RerankerError("Gemini reranker returned invalid JSON.") from exc

    @staticmethod
    def _parse_hits(
        payload: dict[str, object],
        *,
        candidate_ids: set[str],
        limit: int,
    ) -> list[RerankerScoredHit]:
        """Validate one reranker payload against the candidate pool."""

        raw_hits = payload.get("hits")
        if not isinstance(raw_hits, list) or not raw_hits:
            raise RerankerError("Gemini reranker returned no hits.")

        parsed_hits: list[RerankerScoredHit] = []
        seen_chunk_ids: set[str] = set()
        for index, raw_hit in enumerate(raw_hits, start=1):
            if not isinstance(raw_hit, dict):
                raise RerankerError("Gemini reranker returned a malformed hit.")
            chunk_id = str(raw_hit.get("chunk_id", "")).strip()
            if not chunk_id or chunk_id not in candidate_ids or chunk_id in seen_chunk_ids:
                continue
            try:
                score = float(raw_hit.get("score"))
            except (TypeError, ValueError) as exc:
                raise RerankerError("Gemini reranker returned an invalid score.") from exc
            reason = str(raw_hit.get("reason", "")).strip() or "model_ranked"
            parsed_hits.append(
                RerankerScoredHit(
                    chunk_id=chunk_id,
                    score=score,
                    rank=index,
                    debug={"reason": reason},
                )
            )
            seen_chunk_ids.add(chunk_id)
            if len(parsed_hits) >= limit:
                break

        if not parsed_hits:
            raise RerankerError("Gemini reranker returned no valid candidate ids.")
        return parsed_hits

    async def rerank(
        self,
        *,
        query_text: str,
        hits: list[FusedRetrievedChunk],
        limit: int,
    ) -> RerankerResult:
        """Call Gemini to rerank fused hits by answerability."""

        if not hits:
            return RerankerResult(backend_name=self.backend_name, applied=True, hits=[])

        request_payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": self._build_prompt(
                                query_text=query_text,
                                hits=hits,
                                limit=limit,
                            )
                        }
                    ]
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": self._build_schema(),
                "temperature": 0,
                "topP": 0.1,
                "topK": 1,
                "maxOutputTokens": 256,
            },
        }
        model_path = urllib.parse.quote(self.model, safe="")
        request = urllib.request.Request(
            url=f"{self.base_url.rstrip('/')}/models/{model_path}:generateContent?key={self.api_key}",
            data=json.dumps(request_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        def _perform_request() -> str:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return response.read().decode("utf-8")

        try:
            raw_body = await run_with_retries(
                lambda: asyncio.to_thread(_perform_request),
                max_retries=self.max_retries,
                backoff_ms=self.backoff_ms,
            )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RerankerError(
                f"Gemini reranker failed with HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RerankerError("Gemini reranker request failed.") from exc

        try:
            payload = self._coerce_json_text(
                self._extract_candidate_text(json.loads(raw_body))
            )
        except json.JSONDecodeError as exc:
            raise RerankerError("Gemini reranker returned invalid JSON.") from exc
        parsed_hits = self._parse_hits(
            payload,
            candidate_ids={hit.chunk_id for hit in hits},
            limit=limit,
        )
        return RerankerResult(
            backend_name=self.backend_name,
            applied=True,
            hits=parsed_hits,
            debug={"candidate_count": len(hits)},
        )


@dataclass(frozen=True)
class BedrockEnterpriseReranker:
    """Bedrock-backed Enterprise reranker: the Gemini reranker's prompt and JSON contract over Converse.

    The contract helpers are shared, not copied, so a rejected payload is reported in their words.
    """

    model: str
    max_retries: int
    backoff_ms: int
    backend_name: str = "bedrock_v1"

    async def rerank(
        self,
        *,
        query_text: str,
        hits: list[FusedRetrievedChunk],
        limit: int,
    ) -> RerankerResult:
        """Call a Bedrock model to rerank fused hits by answerability."""

        if not hits:
            return RerankerResult(backend_name=self.backend_name, applied=True, hits=[])

        prompt = GeminiEnterpriseReranker._build_prompt(query_text=query_text, hits=hits, limit=limit)
        request: dict[str, object] = {
            "modelId": self.model,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            # enough room for `limit` hits with a short reason each; the Gemini backend's 256 is tight
            "inferenceConfig": {"temperature": 0, "topP": 0.1, "maxTokens": 2048},
        }
        client = bedrock_client.get_bedrock_runtime_client()
        try:
            response = await run_with_retries(
                lambda: asyncio.to_thread(client.converse, **request),
                max_retries=self.max_retries,
                backoff_ms=self.backoff_ms,
                should_retry=bedrock_client.is_retryable_bedrock_error,
            )
        except ClientError as exc:
            raise RerankerError(f"Bedrock reranker failed: {bedrock_client.error_detail(exc)}") from exc
        except BotoCoreError as exc:
            raise RerankerError("Bedrock reranker request failed.") from exc

        output = response.get("output") if isinstance(response, dict) else None
        message = output.get("message") if isinstance(output, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        text_parts = [
            block["text"].strip()
            for block in (content or [])
            if isinstance(block, dict) and isinstance(block.get("text"), str) and block["text"].strip()
        ]
        if not text_parts:
            raise RerankerError("Bedrock reranker returned empty text content.")

        payload = GeminiEnterpriseReranker._coerce_json_text("\n".join(text_parts))
        parsed_hits = GeminiEnterpriseReranker._parse_hits(
            payload,
            candidate_ids={hit.chunk_id for hit in hits},
            limit=limit,
        )
        return RerankerResult(
            backend_name=self.backend_name,
            applied=True,
            hits=parsed_hits,
            debug={"candidate_count": len(hits)},
        )


def resolve_retrieval_reranker(
    *,
    execution_tier: ExecutionTier,
    settings: Settings | None = None,
) -> RetrievalReranker:
    """Resolve one reranker backend for the active execution tier."""

    resolved_settings = settings or get_settings()

    if execution_tier is not ExecutionTier.ENTERPRISE:
        return DisabledReranker(reason="non_enterprise_tier")
    if not resolved_settings.enterprise_enabled:
        return DisabledReranker(reason="enterprise_tier_disabled")
    if not resolved_settings.enterprise_reranker_enabled:
        return DisabledReranker(reason="enterprise_reranker_flag_disabled")
    if resolved_settings.enterprise_reranker_backend == "stub":
        return StubEnterpriseReranker()
    if resolved_settings.enterprise_reranker_backend == "gemini_v1":
        if not resolved_settings.gemini_api_key:
            return DisabledReranker(reason="gemini_api_key_missing")
        return GeminiEnterpriseReranker(
            api_key=resolved_settings.gemini_api_key,
            base_url=str(resolved_settings.gemini_base_url),
            model=resolved_settings.gemini_model,
            max_retries=resolved_settings.provider_max_retries,
            backoff_ms=resolved_settings.provider_retry_backoff_ms,
            timeout_seconds=resolved_settings.openai_timeout_seconds,
        )
    if resolved_settings.enterprise_reranker_backend == "bedrock_v1":
        if not resolved_settings.bedrock_model:
            return DisabledReranker(reason="bedrock_model_missing")
        return BedrockEnterpriseReranker(
            model=resolved_settings.bedrock_model,
            max_retries=resolved_settings.provider_max_retries,
            backoff_ms=resolved_settings.provider_retry_backoff_ms,
        )
    return DisabledReranker(reason="enterprise_reranker_backend_disabled")
