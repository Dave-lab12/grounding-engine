"""Unit tests for Standard query orchestration and trace persistence."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from app.api.deps import TenantContext
from app.core.query_analysis import QueryPlan, QueryProfile
from app.models import ExecutionTier, FreshnessProfile, UserFacingMode
from app.pipeline.contracts import EvidencePackage, FusedRetrievedChunk, RetrievedChunk
from app.schemas.query import QueryRequest
from app.services.generation import GenerationBackend
from app.services.query import QueryServiceError, execute_standard_query
from app.services.retrieval import RetrievalBundle


@dataclass
class FakeScalarResult:
    """Minimal scalar result wrapper for namespace lookups."""

    value: object | None

    def scalar_one_or_none(self):
        return self.value

    def scalars(self):
        return self

    def all(self):
        return []


class FakeAsyncSession:
    """Minimal async session stub for query service tests."""

    def __init__(self, namespace: object | None) -> None:
        self.namespace = namespace
        self.added = []
        self.committed = False
        self.refreshed = []

    async def execute(self, statement, params=None):
        del statement, params
        return FakeScalarResult(self.namespace)

    def add(self, value) -> None:
        self.added.append(value)

    async def commit(self) -> None:
        self.committed = True

    async def rollback(self) -> None:
        self.committed = False

    async def refresh(self, value) -> None:
        self.refreshed.append(value)


def _tenant_context() -> TenantContext:
    return TenantContext(
        tenant_id=uuid.uuid4(),
        tenant_name="tenant",
        subscription_plan="pro",  # type: ignore[arg-type]
        max_execution_tier=ExecutionTier.ENTERPRISE,
        api_key_id=uuid.uuid4(),
        api_key_label="test-key",
    )


def _settings(**overrides) -> SimpleNamespace:
    values = dict(
        enterprise_enabled=False,
        enterprise_trace_metadata_enabled=True,
        enterprise_auto_routing_enabled=True,
        critical_enabled=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _query_plan(
    *,
    raw_query_text: str,
    query_kind: str = "open",
    attribute_terms: tuple[str, ...] = (),
    retrieval_queries: tuple[str, ...] | None = None,
    used_conversation_context: bool = False,
    document_reference_rank: int | None = None,
) -> QueryPlan:
    return QueryPlan(
        raw_query_text=raw_query_text,
        resolved_query_text=raw_query_text,
        profile=QueryProfile(
            raw_text=raw_query_text,
            normalized_text=raw_query_text.casefold(),
            terms=frozenset({"pilot"}),
            expanded_terms=frozenset({"pilot"}),
            attribute_terms=frozenset(attribute_terms),
            context_terms=frozenset(),
            semantic_tags=frozenset(),
            query_kind=query_kind,  # type: ignore[arg-type]
            document_reference_rank=document_reference_rank,
        ),
        retrieval_query_text=raw_query_text,
        retrieval_queries=retrieval_queries or (raw_query_text,),
        explanation="test-plan",
        used_conversation_context=used_conversation_context,
    )


def _retrieval_bundle(
    tenant_id: uuid.UUID,
    namespace_id: uuid.UUID,
    *,
    debug: dict[str, object] | None = None,
) -> RetrievalBundle:
    document_id = uuid.uuid4()
    sparse_hit = RetrievedChunk(
        chunk_id="chunk-1",
        tenant_id=tenant_id,
        namespace_id=namespace_id,
        document_id=document_id,
        chunk_index=0,
        text="Grounded supports tenant-safe uploads.",
        score=0.8,
        rank=1,
        source="sparse",
    )
    dense_hit = RetrievedChunk(
        chunk_id="chunk-1",
        tenant_id=tenant_id,
        namespace_id=namespace_id,
        document_id=document_id,
        chunk_index=0,
        text="Grounded supports tenant-safe uploads.",
        score=0.9,
        rank=1,
        source="dense",
    )
    fused_hit = FusedRetrievedChunk(
        chunk_id="chunk-1",
        tenant_id=tenant_id,
        namespace_id=namespace_id,
        document_id=document_id,
        chunk_index=0,
        text="Grounded supports tenant-safe uploads.",
        fused_score=0.95,
        sources=("dense", "sparse"),
    )
    return RetrievalBundle(
        sparse_hits=[sparse_hit],
        dense_hits=[dense_hit],
        fused_hits=[fused_hit],
        debug=debug,
    )


def _conflicting_retrieval_bundle(
    tenant_id: uuid.UUID,
    namespace_id: uuid.UUID,
) -> RetrievalBundle:
    document_id = uuid.uuid4()
    chunk_one = FusedRetrievedChunk(
        chunk_id="chunk-1",
        tenant_id=tenant_id,
        namespace_id=namespace_id,
        document_id=document_id,
        chunk_index=0,
        text="Grounded supports tenant-safe uploads.",
        fused_score=0.95,
        sources=("dense",),
    )
    chunk_two = FusedRetrievedChunk(
        chunk_id="chunk-2",
        tenant_id=tenant_id,
        namespace_id=namespace_id,
        document_id=document_id,
        chunk_index=1,
        text="Grounded does not support tenant-safe uploads.",
        fused_score=0.92,
        sources=("sparse",),
    )
    return RetrievalBundle(
        sparse_hits=[],
        dense_hits=[],
        fused_hits=[chunk_one, chunk_two],
        debug=None,
    )


@pytest.mark.asyncio()
async def test_execute_standard_query_persists_trace_for_grounded_answer(monkeypatch) -> None:
    """Standard query execution should persist a trace for grounded answers."""

    tenant_context = _tenant_context()
    namespace_id = uuid.uuid4()
    query_request = QueryRequest(namespace_id=namespace_id, query="How does grounded work?")
    namespace = type(
        "NamespaceStub",
        (),
        {
            "min_execution_tier": ExecutionTier.STANDARD,
            "freshness_profile": FreshnessProfile.BALANCED,
        },
    )()
    session = FakeAsyncSession(namespace=namespace)

    async def fake_retrieve_hybrid_candidates(**kwargs):
        assert kwargs["tenant_id"] == tenant_context.tenant_id
        assert kwargs["namespace_id"] == namespace_id
        return _retrieval_bundle(tenant_context.tenant_id, namespace_id)

    monkeypatch.setattr(
        "app.services.query.retrieve_hybrid_candidates",
        fake_retrieve_hybrid_candidates,
    )
    monkeypatch.setattr(
        "app.services.query.get_settings",
        lambda: _settings(),
    )
    monkeypatch.setattr(
        "app.services.generation.resolve_generation_backend",
        lambda: GenerationBackend(
            provider_name="local_grounded_v1",
            implementation="local",
        ),
    )

    result = await execute_standard_query(
        session=session,
        tenant_context=tenant_context,
        query_request=query_request,
    )

    assert result.response.verification_status == "passed"
    assert result.response.confidence_label == "high"
    assert result.response.support_summary == "grounded"
    assert result.response.degraded_reasons == []
    assert session.committed is True
    assert len(session.added) == 1
    trace = session.added[0]
    assert trace.tenant_id == tenant_context.tenant_id
    assert trace.namespace_id == namespace_id
    assert trace.requested_tier is ExecutionTier.STANDARD
    assert trace.agent_id is None
    assert trace.conversation_id is None
    assert trace.selected_mode is None
    assert trace.router_recommendation is ExecutionTier.STANDARD
    assert trace.effective_tier is ExecutionTier.STANDARD
    assert trace.routing_reason == "standard_default"
    assert trace.generator_provider == "local-grounded-v1"
    assert trace.retrieved_chunk_ids == ["chunk-1"]
    assert trace.selected_evidence_ids == ["chunk-1"]
    assert trace.verifier_result["query_plan"]["query_kind"] == "open"
    assert trace.verifier_result["verification_applied"] is False
    assert trace.verifier_result["verification_outcome"] == "accepted"
    assert trace.verifier_result["verification_mode"] == "standard_response_shaping"
    assert trace.verifier_result["bounded_correction_attempted"] is False
    assert trace.verifier_result["contradiction_detected"] is False
    assert trace.verifier_result["unsupported_claims_detected"] is False
    assert trace.verifier_result["execution_routing"]["request_source"] == "default"


@pytest.mark.asyncio()
async def test_execute_standard_query_uses_raw_user_question_for_generation(monkeypatch) -> None:
    """Generation should answer the raw user question even when retrieval needed follow-up resolution."""

    tenant_context = _tenant_context()
    namespace_id = uuid.uuid4()
    conversation_id = uuid.uuid4()
    query_request = QueryRequest(namespace_id=namespace_id, query="what about that")
    namespace = type(
        "NamespaceStub",
        (),
        {"min_execution_tier": ExecutionTier.STANDARD},
    )()
    session = FakeAsyncSession(namespace=namespace)

    from app.models import MessageRole

    message = type(
        "MessageStub",
        (),
        {"role": MessageRole.USER, "content": "What are the supported programming languages?"},
    )()

    captured: dict[str, str] = {}

    async def fake_messages(**kwargs):
        del kwargs
        return [message]

    async def fake_retrieve_hybrid_candidates(**kwargs):
        return _retrieval_bundle(tenant_context.tenant_id, namespace_id)

    def fake_package_evidence(retrieval_bundle, *, query_text=None, limit=None, execution_tier=None):
        del retrieval_bundle, limit
        assert query_text is not None
        assert "language" in query_text.casefold()
        assert execution_tier is ExecutionTier.STANDARD
        return EvidencePackage(
            retrieved_chunk_ids=["chunk-1"],
            selected_evidence_ids=["chunk-1"],
            items=[
                type(
                    "EvidenceItemStub",
                    (),
                    {
                        "citation_id": "E001",
                        "chunk_id": "chunk-1",
                        "tenant_id": tenant_context.tenant_id,
                        "namespace_id": namespace_id,
                        "document_id": uuid.uuid4(),
                        "chunk_index": 0,
                        "text": "Programming Languages: Python, Go, TypeScript",
                        "score": 0.95,
                        "sources": ("dense", "sparse"),
                        "section_title": "TECHNICAL SKILLS",
                        "section_slug": "technical-skills",
                        "chunk_role": "section_header",
                        "starts_with_heading": True,
                        "is_list_block": True,
                    },
                )()
            ],
        )

    async def fake_generate_answer_from_evidence(*, query_text, evidence_package, **kwargs):
        del evidence_package
        captured["query_text"] = query_text
        return type(
            "GroundedDraftStub",
            (),
            {
                "answer_text": "The listed programming languages are Python, Go, and TypeScript [E001]",
                "cited_evidence_ids": ["chunk-1"],
                "citation_snippets": {"chunk-1": "Programming Languages: Python, Go, TypeScript"},
                "generator_provider": "local-grounded-v1",
                "support_coverage": 1.0,
                "source_diversity": 2,
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
        )()

    monkeypatch.setattr(
        "app.services.query.list_conversation_messages",
        fake_messages,
    )
    monkeypatch.setattr(
        "app.services.query.get_settings",
        lambda: _settings(),
    )
    monkeypatch.setattr(
        "app.services.query.retrieve_hybrid_candidates",
        fake_retrieve_hybrid_candidates,
    )
    monkeypatch.setattr(
        "app.services.query.package_evidence",
        fake_package_evidence,
    )
    monkeypatch.setattr(
        "app.services.query.generate_answer_from_evidence",
        fake_generate_answer_from_evidence,
    )

    await execute_standard_query(
        session=session,
        tenant_context=tenant_context,
        query_request=query_request,
        conversation_id=conversation_id,
    )

    assert captured["query_text"] == "what about that"


@pytest.mark.asyncio()
async def test_execute_standard_query_persists_agent_chat_context(monkeypatch) -> None:
    """Agent chat should persist agent, conversation, and selected mode on the trace."""

    tenant_context = _tenant_context()
    namespace_id = uuid.uuid4()
    agent_id = uuid.uuid4()
    conversation_id = uuid.uuid4()
    query_request = QueryRequest(namespace_id=namespace_id, query="How does grounded work?")
    namespace = type(
        "NamespaceStub",
        (),
        {"min_execution_tier": ExecutionTier.STANDARD},
    )()
    session = FakeAsyncSession(namespace=namespace)

    async def fake_retrieve_hybrid_candidates(**kwargs):
        del kwargs
        return _retrieval_bundle(tenant_context.tenant_id, namespace_id)

    monkeypatch.setattr(
        "app.services.query.retrieve_hybrid_candidates",
        fake_retrieve_hybrid_candidates,
    )
    monkeypatch.setattr(
        "app.services.query.get_settings",
        lambda: _settings(),
    )

    await execute_standard_query(
        session=session,
        tenant_context=tenant_context,
        query_request=query_request,
        agent_id=agent_id,
        conversation_id=conversation_id,
        selected_mode=UserFacingMode.INSTANT,
    )

    trace = session.added[0]
    assert trace.agent_id == agent_id
    assert trace.conversation_id == conversation_id
    assert trace.selected_mode is UserFacingMode.INSTANT


@pytest.mark.asyncio()
async def test_execute_standard_query_degrades_when_no_evidence(monkeypatch) -> None:
    """Standard query execution should return a degraded response when nothing is found."""

    tenant_context = _tenant_context()
    namespace_id = uuid.uuid4()
    query_request = QueryRequest(namespace_id=namespace_id, query="Unknown topic")
    namespace = type(
        "NamespaceStub",
        (),
        {"min_execution_tier": ExecutionTier.STANDARD},
    )()
    session = FakeAsyncSession(namespace=namespace)

    async def fake_retrieve_hybrid_candidates(**kwargs):
        del kwargs
        return RetrievalBundle(sparse_hits=[], dense_hits=[], fused_hits=[])

    monkeypatch.setattr(
        "app.services.query.retrieve_hybrid_candidates",
        fake_retrieve_hybrid_candidates,
    )
    monkeypatch.setattr(
        "app.services.query.get_settings",
        lambda: _settings(),
    )

    result = await execute_standard_query(
        session=session,
        tenant_context=tenant_context,
        query_request=query_request,
    )

    assert result.response.verification_status == "degraded"
    assert result.response.degraded_reasons == ["NO_GROUNDED_EVIDENCE"]
    assert result.response.citations == []
    assert result.response.support_summary == "insufficient"


@pytest.mark.asyncio()
async def test_execute_standard_query_requests_clarification_for_vague_query(monkeypatch) -> None:
    """Very vague chat prompts should degrade as clarification requests, not false answers."""

    tenant_context = _tenant_context()
    namespace_id = uuid.uuid4()
    query_request = QueryRequest(namespace_id=namespace_id, query="hi")
    namespace = type(
        "NamespaceStub",
        (),
        {"min_execution_tier": ExecutionTier.STANDARD},
    )()
    session = FakeAsyncSession(namespace=namespace)

    async def fail_retrieve_hybrid_candidates(**kwargs):
        del kwargs
        raise AssertionError("Small-talk clarification should bypass retrieval.")

    monkeypatch.setattr(
        "app.services.query.retrieve_hybrid_candidates",
        fail_retrieve_hybrid_candidates,
    )
    monkeypatch.setattr(
        "app.services.query.get_settings",
        lambda: _settings(),
    )

    result = await execute_standard_query(
        session=session,
        tenant_context=tenant_context,
        query_request=query_request,
    )

    assert result.response.verification_status == "degraded"
    assert result.response.degraded_reasons == ["QUERY_REQUIRES_CLARIFICATION"]
    assert "Ask me a question about the uploaded documents" in result.response.answer


@pytest.mark.asyncio()
async def test_execute_standard_query_requests_clarification_for_combined_greeting(monkeypatch) -> None:
    """Combined greetings like 'hi how are you' should also bypass retrieval."""

    tenant_context = _tenant_context()
    namespace_id = uuid.uuid4()
    query_request = QueryRequest(namespace_id=namespace_id, query="hi how are you")
    namespace = type(
        "NamespaceStub",
        (),
        {"min_execution_tier": ExecutionTier.STANDARD},
    )()
    session = FakeAsyncSession(namespace=namespace)

    async def fail_retrieve_hybrid_candidates(**kwargs):
        del kwargs
        raise AssertionError("Combined greeting clarification should bypass retrieval.")

    monkeypatch.setattr(
        "app.services.query.retrieve_hybrid_candidates",
        fail_retrieve_hybrid_candidates,
    )
    monkeypatch.setattr(
        "app.services.query.get_settings",
        lambda: _settings(),
    )

    result = await execute_standard_query(
        session=session,
        tenant_context=tenant_context,
        query_request=query_request,
    )

    assert result.response.verification_status == "degraded"
    assert result.response.degraded_reasons == ["QUERY_REQUIRES_CLARIFICATION"]
    assert "attached dataset" in result.response.answer.lower()


@pytest.mark.asyncio()
async def test_execute_standard_query_requests_clarification_for_stretched_greeting(monkeypatch) -> None:
    """Elongated greetings like 'heyyyyyyyyy' should also bypass retrieval."""

    tenant_context = _tenant_context()
    namespace_id = uuid.uuid4()
    query_request = QueryRequest(namespace_id=namespace_id, query="heyyyyyyyyy")
    namespace = type(
        "NamespaceStub",
        (),
        {"min_execution_tier": ExecutionTier.STANDARD},
    )()
    session = FakeAsyncSession(namespace=namespace)

    async def fail_retrieve_hybrid_candidates(**kwargs):
        del kwargs
        raise AssertionError("Stretched greeting clarification should bypass retrieval.")

    monkeypatch.setattr(
        "app.services.query.retrieve_hybrid_candidates",
        fail_retrieve_hybrid_candidates,
    )
    monkeypatch.setattr(
        "app.services.query.get_settings",
        lambda: _settings(),
    )

    result = await execute_standard_query(
        session=session,
        tenant_context=tenant_context,
        query_request=query_request,
    )

    assert result.response.verification_status == "degraded"
    assert result.response.degraded_reasons == ["QUERY_REQUIRES_CLARIFICATION"]


@pytest.mark.asyncio()
async def test_execute_standard_query_requests_clarification_for_how_are_u(monkeypatch) -> None:
    """Short chatty variants like 'how are u' should also bypass retrieval."""

    tenant_context = _tenant_context()
    namespace_id = uuid.uuid4()
    query_request = QueryRequest(namespace_id=namespace_id, query="how are u")
    namespace = type(
        "NamespaceStub",
        (),
        {"min_execution_tier": ExecutionTier.STANDARD},
    )()
    session = FakeAsyncSession(namespace=namespace)

    async def fail_retrieve_hybrid_candidates(**kwargs):
        del kwargs
        raise AssertionError("Small-talk clarification should bypass retrieval.")

    monkeypatch.setattr(
        "app.services.query.retrieve_hybrid_candidates",
        fail_retrieve_hybrid_candidates,
    )
    monkeypatch.setattr(
        "app.services.query.get_settings",
        lambda: _settings(),
    )

    result = await execute_standard_query(
        session=session,
        tenant_context=tenant_context,
        query_request=query_request,
    )

    assert result.response.verification_status == "degraded"
    assert result.response.degraded_reasons == ["QUERY_REQUIRES_CLARIFICATION"]


@pytest.mark.asyncio()
async def test_execute_standard_query_rejects_higher_tier_namespace() -> None:
    """Standard query execution should reject namespaces that require a higher tier."""

    tenant_context = _tenant_context()
    query_request = QueryRequest(namespace_id=uuid.uuid4(), query="Question")
    namespace = type(
        "NamespaceStub",
        (),
        {"min_execution_tier": ExecutionTier.ENTERPRISE},
    )()
    session = FakeAsyncSession(namespace=namespace)

    with pytest.raises(QueryServiceError, match="resolved query path"):
        await execute_standard_query(
            session=session,
            tenant_context=tenant_context,
            query_request=query_request,
        )


@pytest.mark.asyncio()
async def test_execute_standard_query_uses_follow_up_context_for_second_document(monkeypatch) -> None:
    """Follow-up summary queries should preserve document-order references in the query plan."""

    tenant_context = _tenant_context()
    namespace_id = uuid.uuid4()
    conversation_id = uuid.uuid4()
    query_request = QueryRequest(
        namespace_id=namespace_id,
        query="what about the second document",
    )
    namespace = type(
        "NamespaceStub",
        (),
        {"min_execution_tier": ExecutionTier.STANDARD},
    )()
    session = FakeAsyncSession(namespace=namespace)

    captured_query_plan = None

    from app.models import MessageRole

    message = type("MessageStub", (), {"role": MessageRole.USER, "content": "What is the dataset about?"})()

    async def fake_messages(**kwargs):
        del kwargs
        return [message]

    async def fake_retrieve_hybrid_candidates(**kwargs):
        nonlocal captured_query_plan
        captured_query_plan = kwargs["query_plan"]
        return RetrievalBundle(sparse_hits=[], dense_hits=[], fused_hits=[])

    monkeypatch.setattr(
        "app.services.query.list_conversation_messages",
        fake_messages,
    )
    monkeypatch.setattr(
        "app.services.query.get_settings",
        lambda: _settings(),
    )
    monkeypatch.setattr(
        "app.services.query.retrieve_hybrid_candidates",
        fake_retrieve_hybrid_candidates,
    )

    await execute_standard_query(
        session=session,
        tenant_context=tenant_context,
        query_request=query_request,
        conversation_id=conversation_id,
    )

    assert captured_query_plan is not None
    assert captured_query_plan.profile.document_reference_rank == 2
    assert captured_query_plan.profile.query_kind == "summary"


@pytest.mark.asyncio()
async def test_execute_standard_query_uses_multi_turn_conversation_context(monkeypatch) -> None:
    """Multi-turn chat history should inform reference-heavy follow-ups beyond one prior turn."""

    tenant_context = _tenant_context()
    namespace_id = uuid.uuid4()
    conversation_id = uuid.uuid4()
    query_request = QueryRequest(
        namespace_id=namespace_id,
        query="did it mention pattern miner there",
    )
    namespace = type(
        "NamespaceStub",
        (),
        {"min_execution_tier": ExecutionTier.STANDARD},
    )()
    session = FakeAsyncSession(namespace=namespace)

    captured_query_plan = None

    from app.models import MessageRole

    messages = [
        type("MessageStub", (), {"role": MessageRole.USER, "content": "What is her work experience?"})(),
        type(
            "MessageStub",
            (),
            {
                "role": MessageRole.ASSISTANT,
                "content": "She has experience as an AI Engineer at iCog Labs and as a Backend | AI Developer Intern at iCog Labs.",
            },
        )(),
        type("MessageStub", (), {"role": MessageRole.USER, "content": "What did she do at iCog Labs?"})(),
    ]

    async def fake_messages(**kwargs):
        del kwargs
        return messages

    async def fake_retrieve_hybrid_candidates(**kwargs):
        nonlocal captured_query_plan
        captured_query_plan = kwargs["query_plan"]
        return RetrievalBundle(sparse_hits=[], dense_hits=[], fused_hits=[])

    monkeypatch.setattr(
        "app.services.query.list_conversation_messages",
        fake_messages,
    )
    monkeypatch.setattr(
        "app.services.query.get_settings",
        lambda: _settings(),
    )
    monkeypatch.setattr(
        "app.services.query.retrieve_hybrid_candidates",
        fake_retrieve_hybrid_candidates,
    )

    await execute_standard_query(
        session=session,
        tenant_context=tenant_context,
        query_request=query_request,
        conversation_id=conversation_id,
    )

    assert captured_query_plan is not None
    assert captured_query_plan.used_conversation_context is True
    assert "icog labs" in captured_query_plan.resolved_query_text.casefold()


@pytest.mark.asyncio()
async def test_execute_standard_query_persists_enterprise_request_without_enabling_it(monkeypatch) -> None:
    """Enterprise requests should be traceable even when the execution stays on Standard."""

    tenant_context = _tenant_context()
    namespace_id = uuid.uuid4()
    query_request = QueryRequest(
        namespace_id=namespace_id,
        query="What changed in the document?",
        requested_tier=ExecutionTier.ENTERPRISE,
    )
    namespace = type(
        "NamespaceStub",
        (),
        {"min_execution_tier": ExecutionTier.STANDARD},
    )()
    session = FakeAsyncSession(namespace=namespace)

    async def fake_retrieve_hybrid_candidates(**kwargs):
        del kwargs
        return _retrieval_bundle(tenant_context.tenant_id, namespace_id)

    monkeypatch.setattr(
        "app.services.query.retrieve_hybrid_candidates",
        fake_retrieve_hybrid_candidates,
    )
    monkeypatch.setattr(
        "app.services.query.get_settings",
        lambda: _settings(),
    )

    await execute_standard_query(
        session=session,
        tenant_context=tenant_context,
        query_request=query_request,
    )

    trace = session.added[0]
    assert trace.requested_tier is ExecutionTier.ENTERPRISE
    assert trace.router_recommendation is ExecutionTier.ENTERPRISE
    assert trace.effective_tier is ExecutionTier.STANDARD
    assert trace.routing_reason == "enterprise_requested_fallback_standard"
    assert trace.verifier_result["execution_routing"]["request_source"] == "query_request"


@pytest.mark.asyncio()
async def test_execute_standard_query_auto_routes_hard_query_to_enterprise(monkeypatch) -> None:
    """Auto mode should route explainable hard queries to Enterprise when enabled."""

    tenant_context = _tenant_context()
    namespace_id = uuid.uuid4()
    query_request = QueryRequest(namespace_id=namespace_id, query="Compare the pilot costs and savings")
    namespace = type(
        "NamespaceStub",
        (),
        {"min_execution_tier": ExecutionTier.STANDARD},
    )()
    session = FakeAsyncSession(namespace=namespace)

    async def fake_retrieve_hybrid_candidates(**kwargs):
        assert kwargs["execution_tier"] is ExecutionTier.ENTERPRISE
        assert len(kwargs["query_plan"].retrieval_queries) > 3
        assert any("difference" in query.casefold() for query in kwargs["query_plan"].retrieval_queries)
        return _retrieval_bundle(
            tenant_context.tenant_id,
            namespace_id,
            debug={
                "execution_tier": "enterprise",
                "reranker": {"attempted": True, "applied": False, "backend": "disabled"},
                "freshness": {"applied": False, "reason": "not_requested"},
                "top_fused_hits": [{"chunk_id": "chunk-1", "score": 0.95}],
            },
        )

    monkeypatch.setattr(
        "app.services.query.build_query_plan",
        lambda *args, **kwargs: _query_plan(
            raw_query_text=query_request.query,
            query_kind="comparison",
            attribute_terms=("cost", "savings"),
            retrieval_queries=(
                "pilot costs",
                "pilot savings",
                "compare pilot costs and savings",
            ),
        ),
    )
    monkeypatch.setattr(
        "app.services.query.retrieve_hybrid_candidates",
        fake_retrieve_hybrid_candidates,
    )
    monkeypatch.setattr(
        "app.services.query.get_settings",
        lambda: _settings(enterprise_enabled=True),
    )

    await execute_standard_query(
        session=session,
        tenant_context=tenant_context,
        query_request=query_request,
        selected_mode=UserFacingMode.AUTO,
    )

    trace = session.added[0]
    routing = trace.verifier_result["execution_routing"]
    assert trace.requested_tier is ExecutionTier.ENTERPRISE
    assert trace.router_recommendation is ExecutionTier.ENTERPRISE
    assert trace.effective_tier is ExecutionTier.ENTERPRISE
    assert trace.routing_reason == "enterprise_auto_hard_query"
    assert routing["request_source"] == "auto_router"
    assert routing["route_triggers"] == ["comparison_query", "multi_attribute_query", "multi_intent_retrieval"]
    assert trace.verifier_result["retrieval_debug"]["execution_tier"] == "enterprise"
    assert trace.verifier_result["evidence_debug"]["selected_evidence_ids"] == ["chunk-1"]


@pytest.mark.asyncio()
async def test_execute_standard_query_keeps_instant_mode_on_standard(monkeypatch) -> None:
    """Instant mode should not auto-route even when a query looks Enterprise-worthy."""

    tenant_context = _tenant_context()
    namespace_id = uuid.uuid4()
    query_request = QueryRequest(namespace_id=namespace_id, query="Compare the pilot costs and savings")
    namespace = type(
        "NamespaceStub",
        (),
        {"min_execution_tier": ExecutionTier.STANDARD},
    )()
    session = FakeAsyncSession(namespace=namespace)

    async def fake_retrieve_hybrid_candidates(**kwargs):
        assert kwargs["execution_tier"] is ExecutionTier.STANDARD
        return _retrieval_bundle(tenant_context.tenant_id, namespace_id)

    monkeypatch.setattr(
        "app.services.query.build_query_plan",
        lambda *args, **kwargs: _query_plan(
            raw_query_text=query_request.query,
            query_kind="comparison",
            attribute_terms=("cost", "savings"),
            retrieval_queries=(
                "pilot costs",
                "pilot savings",
                "compare pilot costs and savings",
            ),
        ),
    )
    monkeypatch.setattr(
        "app.services.query.retrieve_hybrid_candidates",
        fake_retrieve_hybrid_candidates,
    )
    monkeypatch.setattr(
        "app.services.query.get_settings",
        lambda: _settings(enterprise_enabled=True),
    )

    await execute_standard_query(
        session=session,
        tenant_context=tenant_context,
        query_request=query_request,
        selected_mode=UserFacingMode.INSTANT,
    )

    trace = session.added[0]
    routing = trace.verifier_result["execution_routing"]
    assert trace.requested_tier is ExecutionTier.STANDARD
    assert trace.effective_tier is ExecutionTier.STANDARD
    assert trace.routing_reason == "standard_default"
    assert routing["request_source"] == "default"
    assert routing["route_triggers"] == ["comparison_query", "multi_attribute_query", "multi_intent_retrieval"]


@pytest.mark.asyncio()
async def test_execute_standard_query_keeps_hard_query_on_standard_when_enterprise_disabled(monkeypatch) -> None:
    """Auto routing should remain a no-op when Enterprise is disabled."""

    tenant_context = _tenant_context()
    namespace_id = uuid.uuid4()
    query_request = QueryRequest(namespace_id=namespace_id, query="Compare the pilot costs and savings")
    namespace = type(
        "NamespaceStub",
        (),
        {"min_execution_tier": ExecutionTier.STANDARD},
    )()
    session = FakeAsyncSession(namespace=namespace)

    async def fake_retrieve_hybrid_candidates(**kwargs):
        assert kwargs["execution_tier"] is ExecutionTier.STANDARD
        return _retrieval_bundle(tenant_context.tenant_id, namespace_id)

    monkeypatch.setattr(
        "app.services.query.build_query_plan",
        lambda *args, **kwargs: _query_plan(
            raw_query_text=query_request.query,
            query_kind="comparison",
            attribute_terms=("cost", "savings"),
            retrieval_queries=(
                "pilot costs",
                "pilot savings",
                "compare pilot costs and savings",
            ),
        ),
    )
    monkeypatch.setattr(
        "app.services.query.retrieve_hybrid_candidates",
        fake_retrieve_hybrid_candidates,
    )
    monkeypatch.setattr(
        "app.services.query.get_settings",
        lambda: _settings(enterprise_enabled=False),
    )

    await execute_standard_query(
        session=session,
        tenant_context=tenant_context,
        query_request=query_request,
        selected_mode=UserFacingMode.AUTO,
    )

    trace = session.added[0]
    routing = trace.verifier_result["execution_routing"]
    assert trace.requested_tier is ExecutionTier.STANDARD
    assert trace.effective_tier is ExecutionTier.STANDARD
    assert trace.routing_reason == "standard_default"
    assert routing["request_source"] == "default"


@pytest.mark.asyncio()
async def test_execute_standard_query_uses_enterprise_for_thinking_mode(monkeypatch) -> None:
    """Thinking mode should explicitly route through Enterprise when it is enabled."""

    tenant_context = _tenant_context()
    namespace_id = uuid.uuid4()
    query_request = QueryRequest(namespace_id=namespace_id, query="How does grounded work?")
    namespace = type(
        "NamespaceStub",
        (),
        {"min_execution_tier": ExecutionTier.STANDARD},
    )()
    session = FakeAsyncSession(namespace=namespace)

    async def fake_retrieve_hybrid_candidates(**kwargs):
        assert kwargs["execution_tier"] is ExecutionTier.ENTERPRISE
        return _retrieval_bundle(tenant_context.tenant_id, namespace_id)

    monkeypatch.setattr(
        "app.services.query.retrieve_hybrid_candidates",
        fake_retrieve_hybrid_candidates,
    )
    monkeypatch.setattr(
        "app.services.query.get_settings",
        lambda: _settings(enterprise_enabled=True),
    )

    await execute_standard_query(
        session=session,
        tenant_context=tenant_context,
        query_request=query_request,
        selected_mode=UserFacingMode.THINKING,
    )

    trace = session.added[0]
    assert trace.requested_tier is ExecutionTier.ENTERPRISE
    assert trace.effective_tier is ExecutionTier.ENTERPRISE
    assert trace.routing_reason == "thinking_mode_enterprise"
    assert trace.verifier_result["execution_routing"]["request_source"] == "selected_mode"


@pytest.mark.asyncio()
async def test_execute_standard_query_falls_back_cleanly_for_thinking_mode_when_disabled(monkeypatch) -> None:
    """Thinking mode should stay explainable when Enterprise is disabled."""

    tenant_context = _tenant_context()
    namespace_id = uuid.uuid4()
    query_request = QueryRequest(namespace_id=namespace_id, query="How does grounded work?")
    namespace = type(
        "NamespaceStub",
        (),
        {"min_execution_tier": ExecutionTier.STANDARD},
    )()
    session = FakeAsyncSession(namespace=namespace)

    async def fake_retrieve_hybrid_candidates(**kwargs):
        assert kwargs["execution_tier"] is ExecutionTier.STANDARD
        return _retrieval_bundle(tenant_context.tenant_id, namespace_id)

    monkeypatch.setattr(
        "app.services.query.retrieve_hybrid_candidates",
        fake_retrieve_hybrid_candidates,
    )
    monkeypatch.setattr(
        "app.services.query.get_settings",
        lambda: _settings(enterprise_enabled=False),
    )

    await execute_standard_query(
        session=session,
        tenant_context=tenant_context,
        query_request=query_request,
        selected_mode=UserFacingMode.THINKING,
    )

    trace = session.added[0]
    assert trace.requested_tier is ExecutionTier.ENTERPRISE
    assert trace.effective_tier is ExecutionTier.STANDARD
    assert trace.routing_reason == "thinking_mode_fallback_standard"
    assert trace.verifier_result["execution_routing"]["request_source"] == "selected_mode"


@pytest.mark.asyncio()
async def test_execute_standard_query_auto_routes_namespace_floor_to_critical(monkeypatch) -> None:
    """Auto mode should honor a namespace that requires the Critical tier."""

    tenant_context = _tenant_context()
    namespace_id = uuid.uuid4()
    query_request = QueryRequest(namespace_id=namespace_id, query="Summarize the compliance policy")
    namespace = type(
        "NamespaceStub",
        (),
        {"min_execution_tier": ExecutionTier.CRITICAL},
    )()
    session = FakeAsyncSession(namespace=namespace)

    async def fake_retrieve_hybrid_candidates(**kwargs):
        assert kwargs["execution_tier"] is ExecutionTier.CRITICAL
        return _retrieval_bundle(
            tenant_context.tenant_id,
            namespace_id,
            debug={"execution_tier": "critical"},
        )

    monkeypatch.setattr(
        "app.services.query.retrieve_hybrid_candidates",
        fake_retrieve_hybrid_candidates,
    )
    monkeypatch.setattr(
        "app.services.query.get_settings",
        lambda: _settings(enterprise_enabled=True, critical_enabled=True),
    )

    await execute_standard_query(
        session=session,
        tenant_context=tenant_context,
        query_request=query_request,
        selected_mode=UserFacingMode.AUTO,
    )

    trace = session.added[0]
    routing = trace.verifier_result["execution_routing"]
    assert trace.requested_tier is ExecutionTier.CRITICAL
    assert trace.router_recommendation is ExecutionTier.CRITICAL
    assert trace.effective_tier is ExecutionTier.CRITICAL
    assert trace.routing_reason == "critical_auto_required_tier"
    assert routing["request_source"] == "auto_router"
    assert trace.verifier_result["retrieval_debug"]["execution_tier"] == "critical"


@pytest.mark.asyncio()
async def test_execute_standard_query_persists_critical_request_trace_metadata(monkeypatch) -> None:
    """Critical requests should persist explicit fallback verification metadata."""

    tenant_context = _tenant_context()
    namespace_id = uuid.uuid4()
    query_request = QueryRequest(namespace_id=namespace_id, query="Verify the policy exception")
    namespace = type(
        "NamespaceStub",
        (),
        {"min_execution_tier": ExecutionTier.STANDARD},
    )()
    session = FakeAsyncSession(namespace=namespace)

    async def fake_retrieve_hybrid_candidates(**kwargs):
        assert kwargs["execution_tier"] is ExecutionTier.STANDARD
        return _retrieval_bundle(tenant_context.tenant_id, namespace_id)

    async def fake_generate_answer_from_evidence(*, query_text, evidence_package, **kwargs):
        del query_text, evidence_package
        return type(
            "GroundedDraftStub",
            (),
            {
                "answer_text": "Grounded supports tenant-safe uploads [E001].",
                "cited_evidence_ids": ["chunk-1"],
                "citation_snippets": {"chunk-1": "Grounded supports tenant-safe uploads."},
                "generator_provider": "local-grounded-v1",
                "support_coverage": 0.98,
                "source_diversity": 1,
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
        )()

    monkeypatch.setattr(
        "app.services.query.retrieve_hybrid_candidates",
        fake_retrieve_hybrid_candidates,
    )
    monkeypatch.setattr(
        "app.services.query.generate_answer_from_evidence",
        fake_generate_answer_from_evidence,
    )
    monkeypatch.setattr(
        "app.services.query.get_settings",
        lambda: _settings(enterprise_enabled=True),
    )

    await execute_standard_query(
        session=session,
        tenant_context=tenant_context,
        query_request=query_request,
        selected_mode=UserFacingMode.VERIFIED,
    )

    trace = session.added[0]
    assert trace.requested_tier is ExecutionTier.CRITICAL
    assert trace.effective_tier is ExecutionTier.STANDARD
    assert trace.routing_reason == "critical_requested_fallback_standard"
    assert trace.verifier_result["verification_applied"] is True
    assert trace.verifier_result["verification_outcome"] == "accepted"
    assert trace.verifier_result["verification_mode"] == "critical_requested_fallback_standard"
    assert trace.verifier_result["bounded_correction_attempted"] is False
    assert trace.verifier_result["contradiction_detected"] is False
    assert trace.verifier_result["unsupported_claims_detected"] is False
    assert trace.verifier_result["execution_routing"]["request_source"] == "selected_mode"


@pytest.mark.asyncio()
async def test_execute_standard_query_runs_critical_verifier_when_enabled(monkeypatch) -> None:
    """Verified mode should use the first strict Critical verification pass when enabled."""

    tenant_context = _tenant_context()
    namespace_id = uuid.uuid4()
    query_request = QueryRequest(namespace_id=namespace_id, query="Verify the export workflow")
    namespace = type(
        "NamespaceStub",
        (),
        {"min_execution_tier": ExecutionTier.STANDARD},
    )()
    session = FakeAsyncSession(namespace=namespace)

    async def fake_retrieve_hybrid_candidates(**kwargs):
        assert kwargs["execution_tier"] is ExecutionTier.CRITICAL
        return _retrieval_bundle(tenant_context.tenant_id, namespace_id)

    async def fake_generate_answer_from_evidence(*, query_text, evidence_package, **kwargs):
        del query_text, evidence_package
        return type(
            "GroundedDraftStub",
            (),
            {
                "answer_text": "Grounded supports offline exports [E001].",
                "cited_evidence_ids": ["chunk-1"],
                "citation_snippets": {"chunk-1": "Grounded supports tenant-safe uploads."},
                "generator_provider": "local-grounded-v1",
                "support_coverage": 0.98,
                "source_diversity": 1,
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
        )()

    monkeypatch.setattr(
        "app.services.query.retrieve_hybrid_candidates",
        fake_retrieve_hybrid_candidates,
    )
    monkeypatch.setattr(
        "app.services.query.generate_answer_from_evidence",
        fake_generate_answer_from_evidence,
    )
    monkeypatch.setattr(
        "app.services.query.get_settings",
        lambda: _settings(enterprise_enabled=True, critical_enabled=True),
    )

    await execute_standard_query(
        session=session,
        tenant_context=tenant_context,
        query_request=query_request,
        selected_mode=UserFacingMode.VERIFIED,
    )

    trace = session.added[0]
    assert trace.requested_tier is ExecutionTier.CRITICAL
    assert trace.effective_tier is ExecutionTier.CRITICAL
    assert trace.routing_reason == "critical_requested_enabled"
    assert trace.verifier_result["verification_applied"] is True
    assert trace.verifier_result["verification_outcome"] == "degraded"
    assert trace.verifier_result["verification_reason"] == "UNSUPPORTED_CLAIMS"
    assert trace.verifier_result["critical_verifier"]["decision"] == "refuse"
    assert trace.verifier_result["critical_verifier"]["unsupported_claim_count"] == 1
    assert trace.verifier_result["critical_verifier"]["unsupported_claims_detected"] is True


@pytest.mark.asyncio()
async def test_execute_standard_query_degrades_partial_critical_claims(monkeypatch) -> None:
    """Critical mode should degrade partially supported claims without fully accepting them."""

    tenant_context = _tenant_context()
    namespace_id = uuid.uuid4()
    query_request = QueryRequest(namespace_id=namespace_id, query="Verify tenant-safe exports")
    namespace = type(
        "NamespaceStub",
        (),
        {"min_execution_tier": ExecutionTier.STANDARD},
    )()
    session = FakeAsyncSession(namespace=namespace)

    async def fake_retrieve_hybrid_candidates(**kwargs):
        assert kwargs["execution_tier"] is ExecutionTier.CRITICAL
        return _retrieval_bundle(tenant_context.tenant_id, namespace_id)

    async def fake_generate_answer_from_evidence(*, query_text, evidence_package, **kwargs):
        del query_text, evidence_package
        return type(
            "GroundedDraftStub",
            (),
            {
                "answer_text": "Grounded supports tenant-safe exports [E001].",
                "cited_evidence_ids": ["chunk-1"],
                "citation_snippets": {"chunk-1": "Grounded supports tenant-safe uploads."},
                "generator_provider": "local-grounded-v1",
                "support_coverage": 0.98,
                "source_diversity": 1,
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
        )()

    monkeypatch.setattr(
        "app.services.query.retrieve_hybrid_candidates",
        fake_retrieve_hybrid_candidates,
    )
    monkeypatch.setattr(
        "app.services.query.generate_answer_from_evidence",
        fake_generate_answer_from_evidence,
    )
    monkeypatch.setattr(
        "app.services.query.get_settings",
        lambda: _settings(enterprise_enabled=True, critical_enabled=True),
    )

    result = await execute_standard_query(
        session=session,
        tenant_context=tenant_context,
        query_request=query_request,
        selected_mode=UserFacingMode.VERIFIED,
    )

    trace = session.added[0]
    assert result.response.verification_status == "degraded"
    assert result.response.support_summary == "partial"
    assert result.response.confidence_score < 0.5
    assert trace.verifier_result["verification_reason"] == "PARTIAL_SUPPORT"
    assert trace.verifier_result["critical_verifier"]["decision"] == "degrade"
    assert trace.verifier_result["critical_verifier"]["partially_supported_claim_count"] == 1


@pytest.mark.asyncio()
async def test_execute_standard_query_degrades_critical_contradictions(monkeypatch) -> None:
    """Critical mode should degrade when selected evidence conflicts materially."""

    tenant_context = _tenant_context()
    namespace_id = uuid.uuid4()
    query_request = QueryRequest(namespace_id=namespace_id, query="Verify tenant-safe uploads")
    namespace = type(
        "NamespaceStub",
        (),
        {"min_execution_tier": ExecutionTier.STANDARD},
    )()
    session = FakeAsyncSession(namespace=namespace)

    async def fake_retrieve_hybrid_candidates(**kwargs):
        assert kwargs["execution_tier"] is ExecutionTier.CRITICAL
        return _conflicting_retrieval_bundle(tenant_context.tenant_id, namespace_id)

    async def fake_generate_answer_from_evidence(*, query_text, evidence_package, **kwargs):
        del query_text, evidence_package
        return type(
            "GroundedDraftStub",
            (),
            {
                "answer_text": "Grounded supports tenant-safe uploads [E001].",
                "cited_evidence_ids": ["chunk-1", "chunk-2"],
                "citation_snippets": {
                    "chunk-1": "Grounded supports tenant-safe uploads.",
                    "chunk-2": "Grounded does not support tenant-safe uploads.",
                },
                "generator_provider": "local-grounded-v1",
                "support_coverage": 0.98,
                "source_diversity": 2,
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
        )()

    monkeypatch.setattr(
        "app.services.query.retrieve_hybrid_candidates",
        fake_retrieve_hybrid_candidates,
    )
    monkeypatch.setattr(
        "app.services.query.generate_answer_from_evidence",
        fake_generate_answer_from_evidence,
    )
    monkeypatch.setattr(
        "app.services.query.get_settings",
        lambda: _settings(enterprise_enabled=True, critical_enabled=True),
    )

    result = await execute_standard_query(
        session=session,
        tenant_context=tenant_context,
        query_request=query_request,
        selected_mode=UserFacingMode.VERIFIED,
    )

    trace = session.added[0]
    assert result.response.verification_status == "degraded"
    assert "conflicts on this point" in result.response.answer
    assert result.response.support_summary == "insufficient"
    assert result.response.confidence_score == 0.0
    assert trace.verifier_result["verification_reason"] == "CONTRADICTORY_EVIDENCE"
    assert trace.verifier_result["critical_verifier"]["decision"] == "degrade"
    assert trace.verifier_result["critical_verifier"]["contradiction_detected"] is True
    assert trace.verifier_result["critical_verifier"]["source_conflict_detected"] is True
    assert trace.verifier_result["critical_verifier"]["conflict_policy"]["resolution_mode"] == "surface_conflict"
    assert trace.verifier_result["evidence_debug"]["critical_policy"]["conflict_resolution"]["freshness_profile"] == "balanced"


@pytest.mark.asyncio()
async def test_execute_standard_query_retries_critical_support_once(monkeypatch) -> None:
    """Critical mode should run one bounded corrective retry when support is weak."""

    tenant_context = _tenant_context()
    namespace_id = uuid.uuid4()
    query_request = QueryRequest(namespace_id=namespace_id, query="Verify offline exports")
    namespace = type(
        "NamespaceStub",
        (),
        {"min_execution_tier": ExecutionTier.STANDARD},
    )()
    session = FakeAsyncSession(namespace=namespace)
    retrieval_calls: list[tuple[str, ...]] = []

    async def fake_retrieve_hybrid_candidates(**kwargs):
        retrieval_calls.append(tuple(kwargs["query_plan"].retrieval_queries))
        if len(retrieval_calls) == 1:
            return _retrieval_bundle(tenant_context.tenant_id, namespace_id)
        return RetrievalBundle(
            sparse_hits=[],
            dense_hits=[],
            fused_hits=[
                FusedRetrievedChunk(
                    chunk_id="chunk-2",
                    tenant_id=tenant_context.tenant_id,
                    namespace_id=namespace_id,
                    document_id=uuid.uuid4(),
                    chunk_index=0,
                    text="Grounded supports offline exports.",
                    fused_score=0.97,
                    sources=("dense",),
                )
            ],
            debug=None,
        )

    async def fake_generate_answer_from_evidence(*, query_text, evidence_package, **kwargs):
        del query_text
        text = evidence_package.items[0].text
        if "offline exports" in text:
            answer_text = "Grounded supports offline exports [E001]."
            chunk_id = "chunk-2"
        else:
            answer_text = "Grounded supports offline exports [E001]."
            chunk_id = "chunk-1"
        return type(
            "GroundedDraftStub",
            (),
            {
                "answer_text": answer_text,
                "cited_evidence_ids": [chunk_id],
                "citation_snippets": {chunk_id: text},
                "generator_provider": "local-grounded-v1",
                "support_coverage": 0.98,
                "source_diversity": 1,
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
        )()

    monkeypatch.setattr(
        "app.services.query.retrieve_hybrid_candidates",
        fake_retrieve_hybrid_candidates,
    )
    monkeypatch.setattr(
        "app.services.query.generate_answer_from_evidence",
        fake_generate_answer_from_evidence,
    )
    monkeypatch.setattr(
        "app.services.query.get_settings",
        lambda: _settings(enterprise_enabled=True, critical_enabled=True),
    )

    result = await execute_standard_query(
        session=session,
        tenant_context=tenant_context,
        query_request=query_request,
        selected_mode=UserFacingMode.VERIFIED,
    )

    trace = session.added[0]
    assert len(retrieval_calls) == 2
    assert retrieval_calls[1][0] == "offline exports"
    assert result.response.verification_status == "passed"
    assert result.response.support_summary == "grounded"
    assert result.response.degraded_reasons == []
    assert result.response.confidence_score >= 0.5
    assert trace.verifier_result["verification_outcome"] == "accepted"
    assert trace.verifier_result["critical_verifier"]["bounded_correction_attempted"] is True


@pytest.mark.asyncio()
async def test_execute_standard_query_persists_critical_policy_metadata(monkeypatch) -> None:
    """Critical traces should persist policy visibility for external and internal recovery."""

    tenant_context = _tenant_context()
    namespace_id = uuid.uuid4()
    query_request = QueryRequest(namespace_id=namespace_id, query="Verify tenant-safe uploads")
    namespace = type(
        "NamespaceStub",
        (),
        {
            "min_execution_tier": ExecutionTier.STANDARD,
            "allow_web_fallback": True,
            "allow_internal_model_retrieval": False,
        },
    )()
    session = FakeAsyncSession(namespace=namespace)

    async def fake_retrieve_hybrid_candidates(**kwargs):
        return _retrieval_bundle(tenant_context.tenant_id, namespace_id)

    monkeypatch.setattr(
        "app.services.query.retrieve_hybrid_candidates",
        fake_retrieve_hybrid_candidates,
    )
    monkeypatch.setattr(
        "app.services.query.get_settings",
        lambda: _settings(enterprise_enabled=True, critical_enabled=True),
    )

    await execute_standard_query(
        session=session,
        tenant_context=tenant_context,
        query_request=query_request,
        selected_mode=UserFacingMode.VERIFIED,
    )

    trace = session.added[0]
    assert "critical_verifier" in trace.verifier_result
    assert trace.verifier_result["evidence_debug"]["critical_policy"]["external_fallback"]["allowed"] is True
    assert trace.verifier_result["evidence_debug"]["critical_policy"]["external_fallback"]["reason"] == "allowlisted_policy_enabled"
    assert trace.verifier_result["evidence_debug"]["critical_policy"]["internal_model_retrieval"]["allowed"] is False


@pytest.mark.asyncio()
async def test_execute_standard_query_persists_internal_model_retrieval_policy(monkeypatch) -> None:
    """Critical traces should show when internal model retrieval is policy-enabled."""

    tenant_context = _tenant_context()
    namespace_id = uuid.uuid4()
    query_request = QueryRequest(namespace_id=namespace_id, query="Verify tenant-safe uploads")
    namespace = type(
        "NamespaceStub",
        (),
        {
            "min_execution_tier": ExecutionTier.STANDARD,
            "allow_web_fallback": False,
            "allow_internal_model_retrieval": True,
        },
    )()
    session = FakeAsyncSession(namespace=namespace)

    async def fake_retrieve_hybrid_candidates(**kwargs):
        return _retrieval_bundle(tenant_context.tenant_id, namespace_id)

    monkeypatch.setattr(
        "app.services.query.retrieve_hybrid_candidates",
        fake_retrieve_hybrid_candidates,
    )
    monkeypatch.setattr(
        "app.services.query.get_settings",
        lambda: _settings(enterprise_enabled=True, critical_enabled=True),
    )

    await execute_standard_query(
        session=session,
        tenant_context=tenant_context,
        query_request=query_request,
        selected_mode=UserFacingMode.VERIFIED,
    )

    trace = session.added[0]
    assert trace.verifier_result["evidence_debug"]["critical_policy"]["internal_model_retrieval"]["allowed"] is True
    assert trace.verifier_result["evidence_debug"]["critical_policy"]["internal_model_retrieval"]["reason"] == "policy_enabled"


@pytest.mark.asyncio()
async def test_execute_standard_query_records_internal_retrieval_recovery_when_degraded(monkeypatch) -> None:
    """Critical degraded recovery should re-check the answer against expanded grounded evidence."""

    tenant_context = _tenant_context()
    namespace_id = uuid.uuid4()
    query_request = QueryRequest(namespace_id=namespace_id, query="Verify offline exports")
    namespace = type(
        "NamespaceStub",
        (),
        {
            "min_execution_tier": ExecutionTier.STANDARD,
            "allow_web_fallback": False,
            "allow_internal_model_retrieval": True,
        },
    )()
    session = FakeAsyncSession(namespace=namespace)

    async def fake_retrieve_hybrid_candidates(**kwargs):
        return RetrievalBundle(
            sparse_hits=[],
            dense_hits=[],
            fused_hits=[
                FusedRetrievedChunk(
                    chunk_id="chunk-1",
                    tenant_id=tenant_context.tenant_id,
                    namespace_id=namespace_id,
                    document_id=uuid.uuid4(),
                    chunk_index=0,
                    text="Grounded supports tenant-safe uploads.",
                    fused_score=0.98,
                    sources=("dense",),
                ),
                FusedRetrievedChunk(
                    chunk_id="chunk-2",
                    tenant_id=tenant_context.tenant_id,
                    namespace_id=namespace_id,
                    document_id=uuid.uuid4(),
                    chunk_index=1,
                    text="Grounded supports offline exports.",
                    fused_score=0.93,
                    sources=("sparse",),
                ),
            ],
            debug=None,
        )

    async def fake_generate_answer_from_evidence(*, query_text, evidence_package, **kwargs):
        del query_text, evidence_package
        return type(
            "GroundedDraftStub",
            (),
            {
                "answer_text": "Grounded supports tenant-safe exports [E001].",
                "cited_evidence_ids": ["chunk-1"],
                "citation_snippets": {"chunk-1": "Grounded supports tenant-safe uploads."},
                "generator_provider": "local-grounded-v1",
                "support_coverage": 0.98,
                "source_diversity": 1,
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
        )()

    def fake_package_evidence(retrieval_bundle, *, query_text=None, limit=None, execution_tier=None):
        del query_text, limit, execution_tier
        first_hit = retrieval_bundle.fused_hits[0]
        return EvidencePackage(
            retrieved_chunk_ids=[hit.chunk_id for hit in retrieval_bundle.fused_hits],
            selected_evidence_ids=[first_hit.chunk_id],
            items=[
                type(
                    "EvidenceItemStub",
                    (),
                    {
                        "citation_id": "E001",
                        "chunk_id": first_hit.chunk_id,
                        "tenant_id": first_hit.tenant_id,
                        "namespace_id": first_hit.namespace_id,
                        "document_id": first_hit.document_id,
                        "chunk_index": first_hit.chunk_index,
                        "text": first_hit.text,
                        "score": first_hit.fused_score,
                        "sources": first_hit.sources,
                        "section_title": first_hit.section_title,
                        "section_slug": first_hit.section_slug,
                        "chunk_role": first_hit.chunk_role,
                        "starts_with_heading": first_hit.starts_with_heading,
                        "is_list_block": first_hit.is_list_block,
                    },
                )()
            ],
        )

    monkeypatch.setattr(
        "app.services.query.retrieve_hybrid_candidates",
        fake_retrieve_hybrid_candidates,
    )
    monkeypatch.setattr(
        "app.services.query.generate_answer_from_evidence",
        fake_generate_answer_from_evidence,
    )
    monkeypatch.setattr(
        "app.services.query.package_evidence",
        fake_package_evidence,
    )
    monkeypatch.setattr(
        "app.services.query.get_settings",
        lambda: _settings(enterprise_enabled=True, critical_enabled=True),
    )

    result = await execute_standard_query(
        session=session,
        tenant_context=tenant_context,
        query_request=query_request,
        selected_mode=UserFacingMode.VERIFIED,
    )

    trace = session.added[0]
    assert result.response.verification_status == "degraded"
    assert trace.verifier_result["verification_outcome"] == "degraded"
    assert result.response.confidence_label == "low"
    assert result.response.degraded_reasons
    assert trace.verifier_result["critical_verifier"]["recovery_paths"]["internal_model_retrieval"]["used"] is True
