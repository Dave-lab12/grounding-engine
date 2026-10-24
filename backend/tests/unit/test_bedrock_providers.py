"""Unit tests for the AWS Bedrock provider backends: generation, embeddings and Enterprise reranking.

Bedrock is reached through boto3's bedrock-runtime client. These tests replace the cached client with a fake
that records every call and answers from canned responses, so they need no AWS credentials and no network.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import pytest

from app.config import Settings, get_settings
from app.core import bedrock_client
from app.core.bedrock_embeddings import BedrockEmbeddingError, embed_texts_bedrock
from app.core.bedrock_generator import BedrockGenerationError, generate_bedrock_draft
from app.core.embeddings import embed_texts
from app.core.query_analysis import build_query_profile, final_answer_mode
from app.core.reranker import (
    BedrockEnterpriseReranker,
    DisabledReranker,
    resolve_retrieval_reranker,
)
from app.models import ExecutionTier
from app.pipeline.contracts import EvidenceItem, EvidencePackage, FusedRetrievedChunk
from app.services.generation import resolve_generation_backend


class _Body:
    """boto3's StreamingBody: read() returns the response bytes."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw


class FakeBedrockRuntime:
    """Records every call; answers converse() and invoke_model() from the canned responses it holds."""

    def __init__(self, *, converse: Any = None, invoke: Any = None) -> None:
        self.converse_calls: list[dict[str, Any]] = []
        self.invoke_calls: list[dict[str, Any]] = []
        self._converse = converse
        self._invoke = invoke

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.converse_calls.append(kwargs)
        if isinstance(self._converse, Exception):
            raise self._converse
        return self._converse

    def invoke_model(self, **kwargs: Any) -> dict[str, Any]:
        self.invoke_calls.append(kwargs)
        if isinstance(self._invoke, Exception):
            raise self._invoke
        return {"body": _Body(self._invoke)}


def _converse_response(text: str, *, stop_reason: str = "end_turn") -> dict[str, Any]:
    """A Converse response as Bedrock returns it: one assistant message, its stop reason, token usage."""

    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": stop_reason,
        "usage": {"inputTokens": 120, "outputTokens": 40, "totalTokens": 160},
    }


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


def _fused_hit(chunk_id: str, score: float) -> FusedRetrievedChunk:
    return FusedRetrievedChunk(
        chunk_id=chunk_id,
        tenant_id=uuid.uuid4(),
        namespace_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        chunk_index=0,
        text=f"Chunk {chunk_id}",
        fused_score=score,
        sources=("dense", "sparse"),
    )


_QUERY = "Which company supplied the vans?"
_EVIDENCE_TEXT = "The vans were supplied by Northwind Logistics in March."


def _package() -> EvidencePackage:
    return EvidencePackage(
        retrieved_chunk_ids=["chunk-1"],
        selected_evidence_ids=["chunk-1"],
        items=[
            _evidence_item(citation_id="E001", chunk_id="chunk-1", text=_EVIDENCE_TEXT)
        ],
    )


def _grounded_answer() -> str:
    """A model answer in the generator's JSON contract, in the mode the query profile expects."""

    return json.dumps(
        {
            "is_refusal": False,
            "answer_mode": final_answer_mode(build_query_profile(_QUERY)),
            "answer_text": "Northwind Logistics",
            "cited_chunk_ids": ["chunk-1"],
            "citation_snippets": [
                {"chunk_id": "chunk-1", "snippet": "supplied by Northwind Logistics"}
            ],
        }
    )


@pytest.fixture()
def bedrock_env(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Every Bedrock backend configured through the environment, with a fresh settings cache."""

    monkeypatch.setenv("GENERATOR_BACKEND", "bedrock_v1")
    monkeypatch.setenv("EMBEDDING_BACKEND", "bedrock_v1")
    monkeypatch.setenv("BEDROCK_REGION", "us-east-1")
    monkeypatch.setenv("BEDROCK_MODEL", "us.anthropic.test-model")
    monkeypatch.setenv("BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
    monkeypatch.setenv("DENSE_EMBEDDING_DIMENSIONS", "4")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _use(monkeypatch: pytest.MonkeyPatch, fake: FakeBedrockRuntime) -> None:
    monkeypatch.setattr(bedrock_client, "get_bedrock_runtime_client", lambda: fake)


# ---- generation -------------------------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_generator_converses_once_and_returns_a_grounded_draft(
    bedrock_env: Any, monkeypatch
) -> None:
    """One Converse call: the grounded prompt as the user message, agent instructions as the system prompt,
    deterministic inference settings; the JSON answer becomes a draft with citations and token usage."""

    fake = FakeBedrockRuntime(converse=_converse_response(_grounded_answer()))
    _use(monkeypatch, fake)

    draft = await generate_bedrock_draft(
        query_text=_QUERY,
        evidence_package=_package(),
        agent_instructions="Answer tersely.",
    )

    assert draft.answer_text == "Northwind Logistics"
    assert draft.cited_evidence_ids == ["chunk-1"]
    assert draft.citation_snippets == {"chunk-1": "supplied by Northwind Logistics"}
    assert draft.generator_provider == "bedrock:us.anthropic.test-model"
    assert draft.token_usage == {
        "prompt_tokens": 120,
        "completion_tokens": 40,
        "total_tokens": 160,
    }
    [call] = fake.converse_calls
    assert call["modelId"] == "us.anthropic.test-model"
    assert call["system"] == [{"text": "Answer tersely."}]
    assert call["inferenceConfig"]["temperature"] == 0
    [message] = call["messages"]
    assert (
        message["role"] == "user"
        and "chunk_id=chunk-1" in message["content"][0]["text"]
    )


@pytest.mark.asyncio()
async def test_generator_rejects_a_truncated_answer(
    bedrock_env: Any, monkeypatch
) -> None:
    """A stop reason other than end_turn (max_tokens, a guardrail) is a provider failure, not an answer."""

    _use(
        monkeypatch,
        FakeBedrockRuntime(converse=_converse_response("{", stop_reason="max_tokens")),
    )

    with pytest.raises(BedrockGenerationError, match="max_tokens"):
        await generate_bedrock_draft(query_text=_QUERY, evidence_package=_package())


@pytest.mark.asyncio()
async def test_generator_rejects_output_outside_the_contract(
    bedrock_env: Any, monkeypatch
) -> None:
    """Text that is not the JSON contract is a BedrockGenerationError, so the generation loop can retry."""

    _use(
        monkeypatch,
        FakeBedrockRuntime(
            converse=_converse_response("Northwind Logistics supplied them.")
        ),
    )

    with pytest.raises(BedrockGenerationError):
        await generate_bedrock_draft(query_text=_QUERY, evidence_package=_package())


@pytest.mark.asyncio()
async def test_generator_requires_a_model(monkeypatch) -> None:
    """Settings built without the .env file, so a model configured there cannot make this test pass."""

    from app.core import bedrock_generator

    settings = Settings(_env_file=None, generator_backend="bedrock_v1", bedrock_model=None)
    monkeypatch.setattr(bedrock_generator, "get_settings", lambda: settings)
    with pytest.raises(BedrockGenerationError, match="BEDROCK_MODEL"):
        await generate_bedrock_draft(query_text=_QUERY, evidence_package=_package())


def test_generation_registry_resolves_the_bedrock_backend(bedrock_env: Any) -> None:
    backend = resolve_generation_backend()
    assert (backend.provider_name, backend.implementation) == ("bedrock_v1", "bedrock")


# ---- embeddings -------------------------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_embeddings_invoke_titan_once_per_text_with_the_configured_dimensions(
    bedrock_env: Any, monkeypatch
) -> None:
    fake = FakeBedrockRuntime(
        invoke={"embedding": [0.1, 0.2, 0.3, 0.4], "inputTextTokenCount": 3}
    )
    _use(monkeypatch, fake)

    embeddings = await embed_texts_bedrock(["alpha", "beta"], purpose="document")

    assert [embedding.text for embedding in embeddings] == ["alpha", "beta"]
    assert embeddings[0].vector == [0.1, 0.2, 0.3, 0.4]
    assert [call["modelId"] for call in fake.invoke_calls] == [
        "amazon.titan-embed-text-v2:0"
    ] * 2
    assert json.loads(fake.invoke_calls[0]["body"]) == {
        "inputText": "alpha",
        "dimensions": 4,
        "normalize": True,
    }


@pytest.mark.asyncio()
async def test_embeddings_reject_a_dimension_mismatch(
    bedrock_env: Any, monkeypatch
) -> None:
    _use(monkeypatch, FakeBedrockRuntime(invoke={"embedding": [0.1, 0.2, 0.3]}))

    with pytest.raises(BedrockEmbeddingError, match="dimension"):
        await embed_texts_bedrock(["alpha"])


@pytest.mark.asyncio()
async def test_embed_texts_dispatches_to_bedrock(bedrock_env: Any, monkeypatch) -> None:
    """EMBEDDING_BACKEND=bedrock_v1 routes the shared embed_texts entry point to Bedrock."""

    _use(monkeypatch, FakeBedrockRuntime(invoke={"embedding": [1.0, 0.0, 0.0, 0.0]}))

    [embedding] = await embed_texts(["alpha"], purpose="query")

    assert embedding.vector == [1.0, 0.0, 0.0, 0.0]


# ---- reranking --------------------------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_reranker_parses_ranked_hits(monkeypatch) -> None:
    """The Bedrock reranker asks the model for strict JSON hits and keeps only known chunk ids, in order."""

    hits_json = json.dumps(
        {
            "hits": [
                {"chunk_id": "chunk-2", "score": 0.98, "reason": "direct answer"},
                {"chunk_id": "chunk-9", "score": 0.9, "reason": "not a candidate"},
                {"chunk_id": "chunk-1", "score": 0.75, "reason": "supporting detail"},
            ]
        }
    )
    fake = FakeBedrockRuntime(converse=_converse_response(hits_json))
    _use(monkeypatch, fake)

    reranker = BedrockEnterpriseReranker(
        model="us.anthropic.test-model", max_retries=1, backoff_ms=1
    )
    result = await reranker.rerank(
        query_text=_QUERY,
        hits=[_fused_hit("chunk-1", 0.8), _fused_hit("chunk-2", 0.7)],
        limit=2,
    )

    assert result.applied is True
    assert [hit.chunk_id for hit in result.hits] == ["chunk-2", "chunk-1"]
    assert fake.converse_calls[0]["modelId"] == "us.anthropic.test-model"


def test_resolve_reranker_returns_the_bedrock_backend_when_configured() -> None:
    settings = Settings(
        enterprise_reranker_backend="bedrock_v1",
        bedrock_model="us.anthropic.test-model",
    )
    reranker = resolve_retrieval_reranker(
        execution_tier=ExecutionTier.ENTERPRISE, settings=settings
    )
    assert isinstance(reranker, BedrockEnterpriseReranker)
    assert reranker.model == "us.anthropic.test-model"


def test_resolve_reranker_disables_cleanly_without_a_bedrock_model() -> None:
    settings = Settings(enterprise_reranker_backend="bedrock_v1", bedrock_model=None)
    reranker = resolve_retrieval_reranker(
        execution_tier=ExecutionTier.ENTERPRISE, settings=settings
    )
    assert isinstance(reranker, DisabledReranker)
    assert reranker.reason == "bedrock_model_missing"


# ---- settings ---------------------------------------------------------------------------------------------


def test_settings_warn_when_bedrock_backends_lack_a_model() -> None:
    settings = Settings(
        generator_backend="bedrock_v1",
        embedding_backend="local_hash_v1",
        enterprise_reranker_backend="bedrock_v1",
        bedrock_model=None,
    )

    warnings = settings.startup_warnings()

    assert sum("BEDROCK_MODEL" in warning for warning in warnings) == 2


# ---- client ------------------------------------------------------------------------------------------------


def test_client_passes_configured_credentials_to_boto3_and_omits_unset_ones(
    monkeypatch,
) -> None:
    """Keys in .env reach boto3 only as arguments; an unset key is left to boto3's default chain."""

    monkeypatch.setenv("BEDROCK_REGION", "eu-west-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-placeholder")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "placeholder-secret")
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        bedrock_client.boto3,
        "client",
        lambda service, **kwargs: seen.update(kwargs, service=service),
    )
    get_settings.cache_clear()
    bedrock_client.reset_bedrock_runtime_client()
    try:
        bedrock_client.get_bedrock_runtime_client()
    finally:
        bedrock_client.reset_bedrock_runtime_client()
        get_settings.cache_clear()

    assert seen["service"] == "bedrock-runtime" and seen["region_name"] == "eu-west-1"
    assert (seen["aws_access_key_id"], seen["aws_secret_access_key"]) == (
        "AKIA-placeholder",
        "placeholder-secret",
    )
    assert "aws_session_token" not in seen
