"""Standard query orchestration and trace persistence."""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from dataclasses import replace

from fastapi import status
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import TenantContext
from app.config import get_settings
from app.core.database import get_session_factory
from app.core.llm_client import GroundedGenerationError
from app.core.query_analysis import (
    ConversationContext,
    build_conversation_context,
    build_query_plan,
    final_answer_mode,
    refine_query_plan_for_execution_tier,
    query_plan_metadata,
    QueryPlan,
)
from app.core.telemetry import bind_execution_context, get_logger
from app.models import ExecutionTier, FreshnessProfile, MessageRole, Namespace, QueryTrace, UserFacingMode
from app.schemas.query import GroundedAnswerResponse, QueryRequest
from app.services.messages import MessageServiceError, list_conversation_messages
from app.services.evidence import package_evidence
from app.services.external_fallback import (
    resolve_external_fallback_policy,
    run_allowlisted_external_fallback,
)
from app.services.generation import generate_answer_from_evidence
from app.services.internal_model_retrieval import (
    resolve_internal_model_retrieval_policy,
    run_internal_model_retrieval,
)
from app.services.response_shaping import (
    ResponseShapingError,
    shape_degraded_response,
    shape_grounded_response,
)
from app.services.retrieval import RetrievalBundle, RetrievalError, retrieve_hybrid_candidates
from app.core.flags import (
    is_freshness_conflict_resolution_enabled,
    is_multi_attempt_crag_enabled,
    max_crag_attempts,
    min_evidence_quality_improvement,
)
from app.services.verification import verify_critical_response


logger = get_logger("app.query")


class QueryServiceError(RuntimeError):
    """Raised when a Standard query cannot be completed."""

    def __init__(self, detail: str, *, status_code: int) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


_SMALLTALK_QUERIES = {
    "hi",
    "hello",
    "hey",
    "hiya",
    "yo",
    "good day",
    "good morning",
    "good afternoon",
    "good evening",
    "how are you",
    "how are u",
    "how r u",
    "how are you doing",
    "can you help me",
    "help me",
    "who are you",
    "what can you do",
    "nice to meet you",
    "ok",
    "okay",
    "thanks",
    "thank you",
    "thank you so much",
}

_GREETING_PREFIXES = ("hi", "hello", "hey", "hiya", "yo")

_TIER_RANK: dict[ExecutionTier, int] = {
    ExecutionTier.STANDARD: 1,
    ExecutionTier.ENTERPRISE: 2,
    ExecutionTier.CRITICAL: 3,
}


def _degraded_reason_for_generation_exception(exc: Exception) -> tuple[str, str]:
    """Map generator/shaping failures to clearer Standard degraded outcomes."""

    message = str(exc).lower()
    if "meaningful query terms" in message:
        return (
            "QUERY_REQUIRES_CLARIFICATION",
            "Please ask a more specific grounded question so I can search the attached evidence.",
        )
    if "query-aligned support" in message:
        return (
            "INSUFFICIENT_QUERY_ALIGNMENT",
            "I found related material, but not enough evidence that directly answers this question.",
        )
    if "configured generation provider" in message:
        return (
            "GENERATION_PROVIDER_FAILED",
            "I do not have an answer for this request because the configured model could not produce a grounded response.",
        )
    return (
        "INSUFFICIENT_SUPPORT",
        "I found some related material, but not enough grounded evidence to answer confidently.",
    )


def _evidence_debug_summary(*, selected_evidence_ids: list[str], evidence_package) -> dict[str, object]:
    """Build a compact evidence summary for persisted trace inspection."""

    return {
        "selected_evidence_ids": list(selected_evidence_ids),
        "item_count": len(evidence_package.items),
        "document_count": len({item.document_id for item in evidence_package.items}),
        "section_slugs": sorted(
            {
                item.section_slug
                for item in evidence_package.items
                if item.section_slug
            }
        )[:5],
    }


def _log_query_identification(*, query_plan: QueryPlan) -> None:
    """Log query classification details used for routing and answer shaping."""

    profile = query_plan.profile
    logger.info(
        "query_identified",
        query_text=query_plan.raw_query_text,
        resolved_query_text=query_plan.resolved_query_text,
        query_kind=profile.query_kind,
        answer_mode=final_answer_mode(profile),
        terms=sorted(profile.terms)[:12],
        attribute_terms=sorted(profile.attribute_terms),
        semantic_tags=sorted(profile.semantic_tags),
        context_terms=sorted(profile.context_terms),
        retrieval_queries=list(query_plan.retrieval_queries),
        used_conversation_context=query_plan.used_conversation_context,
    )


def _log_selected_evidence(*, query_text: str, evidence_package) -> None:
    """Log the final evidence package handed to generation."""

    logger.debug(
        "evidence_package_selected",
        query_text=query_text,
        selected_count=len(evidence_package.items),
        selected_evidence=[
            {
                "citation_id": item.citation_id,
                "chunk_id": item.chunk_id,
                "document_id": str(item.document_id),
                "chunk_index": item.chunk_index,
                "score": item.score,
                "sources": list(item.sources),
                "section_title": item.section_title,
                "section_slug": item.section_slug,
                "chunk_role": item.chunk_role,
                "text_preview": item.text[:220],
            }
            for item in evidence_package.items
        ],
    )


def _build_verifier_trace_result(
    *,
    response: GroundedAnswerResponse,
    query_plan: QueryPlan | None,
    routing_decision: ExecutionRoutingDecision,
    retrieval_bundle: RetrievalBundle,
    evidence_debug: dict[str, object] | None,
    enterprise_trace_metadata_enabled: bool,
    enterprise_enabled: bool,
    run_status: str = "completed",
) -> dict[str, object]:
    """Build persisted verification metadata for current and future Critical flows."""

    degraded_reasons = list(response.degraded_reasons)
    verification_outcome = "accepted" if response.verification_status == "passed" else "degraded"

    return {
        "status": response.verification_status,
        "verification_applied": routing_decision.requested_tier is ExecutionTier.CRITICAL,
        "verification_outcome": verification_outcome,
        "verification_reason": degraded_reasons[0] if degraded_reasons else None,
        "run_status": run_status,
        "verification_mode": (
            "critical_requested_fallback_standard"
            if routing_decision.requested_tier is ExecutionTier.CRITICAL
            else "standard_response_shaping"
        ),
        "bounded_correction_attempted": False,
        "contradiction_detected": False,
        "unsupported_claims_detected": bool(degraded_reasons),
        "critical_verifier": (
            evidence_debug.get("critical_verifier")
            if isinstance(evidence_debug, dict)
            else None
        ),
        "query_plan": query_plan_metadata(query_plan) if query_plan is not None else None,
        "execution_routing": (
            routing_decision.trace_metadata(enterprise_enabled=enterprise_enabled)
            if enterprise_trace_metadata_enabled
            else None
        ),
        "retrieval_debug": retrieval_bundle.debug if enterprise_trace_metadata_enabled else None,
        "evidence_debug": evidence_debug if enterprise_trace_metadata_enabled else None,
    }


def _supports_execution_tier(*, available_tier: ExecutionTier, required_tier: ExecutionTier) -> bool:
    """Return whether the resolved execution tier satisfies one namespace requirement."""

    return _TIER_RANK[available_tier] >= _TIER_RANK[required_tier]


def _is_smalltalk_query(query_text: str) -> bool:
    """Detect greetings and conversational filler that should clarify scope."""

    normalized = re.sub(r"[^a-z0-9\s]+", " ", query_text.strip().casefold())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return False
    if re.fullmatch(r"(?:h+i+|h+e+y+|hello+|hiya+|yo+)", normalized):
        return True
    if normalized in _SMALLTALK_QUERIES:
        return True

    for prefix in _GREETING_PREFIXES:
        if normalized.startswith(f"{prefix} "):
            remainder = normalized[len(prefix) :].strip()
            if (
                remainder in _SMALLTALK_QUERIES
                or remainder in {"there", "there there"}
                or remainder.startswith("how are you")
                or remainder.startswith("can you help me")
                or remainder.startswith("help me")
                or remainder.startswith("what can you do")
                or remainder.startswith("who are you")
                or remainder.startswith("nice to meet you")
            ):
                return True

    tokens = normalized.split()
    return len(tokens) <= 4 and all(
        token in {"hi", "hello", "hey", "hiya", "yo", "thanks", "thank", "okay", "ok"}
        for token in tokens
    )


def _smalltalk_response() -> GroundedAnswerResponse:
    """Return a friendly scoped reply for greetings and conversational filler."""

    return shape_degraded_response(
        reason="QUERY_REQUIRES_CLARIFICATION",
        answer_text=(
            "Hi. I am ready to help with the attached dataset. "
            "Ask me a question about the uploaded documents and I will answer with citations when grounded evidence is available."
        ),
    )


def _requested_tier_from_mode(selected_mode: UserFacingMode | None) -> ExecutionTier | None:
    """Map user-facing modes into their backing execution tiers."""

    if selected_mode is UserFacingMode.THINKING:
        return ExecutionTier.ENTERPRISE
    if selected_mode is UserFacingMode.VERIFIED:
        return ExecutionTier.CRITICAL
    return None


def _enterprise_route_reasons(query_plan: QueryPlan) -> tuple[str, ...]:
    """Return explainable triggers for minimal hard-query Enterprise routing."""

    profile = query_plan.profile
    reasons: list[str] = []

    if profile.query_kind == "comparison":
        reasons.append("comparison_query")
    if profile.query_kind == "action":
        reasons.append("action_query")
    if len(profile.attribute_terms) > 1 and profile.query_kind in {"comparison", "list", "lookup"}:
        reasons.append("multi_attribute_query")
    if profile.document_reference_rank is not None:
        reasons.append("document_reference_query")
    if query_plan.used_conversation_context and profile.query_kind in {"comparison", "entity", "open", "summary"}:
        reasons.append("follow_up_context_query")
    if len(query_plan.retrieval_queries) >= 3 and profile.query_kind in {"comparison", "action", "entity"}:
        reasons.append("multi_intent_retrieval")

    deduped_reasons: list[str] = []
    for reason in reasons:
        if reason not in deduped_reasons:
            deduped_reasons.append(reason)
    return tuple(deduped_reasons)


def _resolve_execution_routing(
    *,
    query_request: QueryRequest,
    selected_mode: UserFacingMode | None,
    namespace_min_execution_tier: ExecutionTier = ExecutionTier.STANDARD,
    query_plan: QueryPlan | None = None,
) -> ExecutionRoutingDecision:
    """Resolve which execution tier is being requested and which is currently active."""

    settings = get_settings()
    requested_from_mode = _requested_tier_from_mode(selected_mode)

    if query_request.requested_tier is not None:
        requested_tier = query_request.requested_tier
        request_source = "query_request"
    elif requested_from_mode is not None:
        requested_tier = requested_from_mode
        request_source = "selected_mode"
    elif (
        selected_mode in {None, UserFacingMode.AUTO}
        and namespace_min_execution_tier is ExecutionTier.CRITICAL
    ):
        requested_tier = ExecutionTier.CRITICAL
        request_source = "auto_router"
    elif (
        query_plan is not None
        and selected_mode in {None, UserFacingMode.AUTO}
        and settings.enterprise_enabled
        and settings.enterprise_auto_routing_enabled
        and _enterprise_route_reasons(query_plan)
    ):
        requested_tier = ExecutionTier.ENTERPRISE
        request_source = "auto_router"
    else:
        requested_tier = ExecutionTier.STANDARD
        request_source = "default"

    route_triggers = _enterprise_route_reasons(query_plan) if query_plan is not None else ()
    router_recommendation = requested_tier

    if requested_tier is ExecutionTier.ENTERPRISE:
        if settings.enterprise_enabled:
            effective_tier = ExecutionTier.ENTERPRISE
            if request_source == "auto_router":
                routing_reason = "enterprise_auto_hard_query"
            elif selected_mode is UserFacingMode.THINKING:
                routing_reason = "thinking_mode_enterprise"
            else:
                routing_reason = "enterprise_requested_enabled"
        else:
            effective_tier = ExecutionTier.STANDARD
            routing_reason = (
                "thinking_mode_fallback_standard"
                if selected_mode is UserFacingMode.THINKING
                else "enterprise_requested_fallback_standard"
            )
    elif requested_tier is ExecutionTier.CRITICAL:
        if settings.critical_enabled:
            effective_tier = ExecutionTier.CRITICAL
            routing_reason = (
                "critical_auto_required_tier"
                if request_source == "auto_router"
                else "critical_requested_enabled"
            )
        else:
            effective_tier = ExecutionTier.STANDARD
            routing_reason = "critical_requested_fallback_standard"
    else:
        effective_tier = ExecutionTier.STANDARD
        routing_reason = "standard_default"

    return ExecutionRoutingDecision(
        requested_tier=requested_tier,
        router_recommendation=router_recommendation,
        effective_tier=effective_tier,
        routing_reason=routing_reason,
        request_source=request_source,
        route_triggers=route_triggers,
    )


@dataclass(frozen=True)
class QueryExecutionResult:
    """Completed Standard query result including persisted trace metadata."""

    response: GroundedAnswerResponse
    trace_id: uuid.UUID


@dataclass(frozen=True)
class ExecutionRoutingDecision:
    """Resolved execution-tier decision for one query request."""

    requested_tier: ExecutionTier
    router_recommendation: ExecutionTier
    effective_tier: ExecutionTier
    routing_reason: str
    request_source: str
    route_triggers: tuple[str, ...] = ()

    def trace_metadata(self, *, enterprise_enabled: bool) -> dict[str, object]:
        """Return compact routing metadata for persisted traces and debugging."""

        return {
            "requested_tier": self.requested_tier.value,
            "router_recommendation": self.router_recommendation.value,
            "effective_tier": self.effective_tier.value,
            "routing_reason": self.routing_reason,
            "request_source": self.request_source,
            "route_triggers": list(self.route_triggers),
            "enterprise_enabled": enterprise_enabled,
        }


async def _get_namespace_for_tenant(
    *,
    session: AsyncSession,
    tenant_id: uuid.UUID,
    namespace_id: uuid.UUID,
) -> Namespace:
    """Resolve a namespace only if it belongs to the authenticated tenant."""

    statement = select(Namespace).where(
        Namespace.tenant_id == tenant_id,
        Namespace.namespace_id == namespace_id,
    )
    result = await session.execute(statement)
    namespace = result.scalar_one_or_none()
    if namespace is None:
        raise QueryServiceError(
            "Namespace not found for tenant.",
            status_code=status.HTTP_404_NOT_FOUND,
        )
    return namespace


async def _persist_query_trace(
    *,
    session: AsyncSession,
    tenant_context: TenantContext,
    query_request: QueryRequest,
    response: GroundedAnswerResponse,
    retrieval_bundle: RetrievalBundle,
    stage_latencies_ms: dict[str, int],
    total_latency_ms: int,
    generator_provider: str,
    query_plan: QueryPlan | None = None,
    evidence_debug: dict[str, object] | None = None,
    routing_decision: ExecutionRoutingDecision | None = None,
    agent_id: uuid.UUID | None = None,
    conversation_id: uuid.UUID | None = None,
    selected_mode: UserFacingMode | None = None,
    run_status: str = "completed",
    token_usage: dict[str, int] | None = None,
) -> QueryTrace:
    """Persist one Standard query trace for later debugging and evaluation."""

    settings = get_settings()
    resolved_routing = routing_decision or ExecutionRoutingDecision(
        requested_tier=ExecutionTier.STANDARD,
        router_recommendation=ExecutionTier.STANDARD,
        effective_tier=ExecutionTier.STANDARD,
        routing_reason="standard_default",
        request_source="default",
    )

    trace = QueryTrace(
        tenant_id=tenant_context.tenant_id,
        namespace_id=query_request.namespace_id,
        agent_id=agent_id,
        conversation_id=conversation_id,
        selected_mode=selected_mode,
        requested_tier=resolved_routing.requested_tier,
        router_recommendation=resolved_routing.router_recommendation,
        effective_tier=resolved_routing.effective_tier,
        routing_reason=resolved_routing.routing_reason,
        query_redacted=query_request.query,
        query_ciphertext=hashlib.sha256(query_request.query.encode("utf-8")).digest(),
        retrieved_chunk_ids=[hit.chunk_id for hit in retrieval_bundle.fused_hits],
        selected_evidence_ids=[citation.chunk_id for citation in response.citations],
        generator_provider=generator_provider,
        verifier_result=_build_verifier_trace_result(
            response=response,
            query_plan=query_plan,
            routing_decision=resolved_routing,
            retrieval_bundle=retrieval_bundle,
            evidence_debug=evidence_debug,
            enterprise_trace_metadata_enabled=settings.enterprise_trace_metadata_enabled,
            enterprise_enabled=settings.enterprise_enabled,
            run_status=run_status,
        ),
        final_answer_redacted=response.answer,
        citations=[citation.model_dump(mode="json") for citation in response.citations],
        overall_confidence=response.confidence_score,
        degraded_reasons=list(response.degraded_reasons),
        stage_latencies_ms=stage_latencies_ms,
        total_latency_ms=total_latency_ms,
        token_usage=token_usage or {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    )
    session.add(trace)
    try:
        await session.commit()
    except SQLAlchemyError as exc:
        await session.rollback()
        raise QueryServiceError(
            "Failed to persist query trace.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from exc

    await session.refresh(trace)
    return trace


async def _resolve_conversation_context(
    *,
    session: AsyncSession,
    tenant_id: uuid.UUID,
    conversation_id: uuid.UUID | None,
    current_query: str,
) -> ConversationContext | None:
    """Return a lightweight rolling conversation context for follow-up expansion."""

    if conversation_id is None:
        return None
    try:
        messages = await list_conversation_messages(
            session=session,
            tenant_id=tenant_id,
            conversation_id=conversation_id,
        )
    except MessageServiceError:
        return None

    history = list(messages)
    if (
        history
        and history[-1].role is MessageRole.USER
        and history[-1].content.strip() == current_query.strip()
    ):
        history = history[:-1]
    if not history:
        return None
    recent_user_queries = [
        message.content
        for message in history
        if message.role is MessageRole.USER
    ][-4:]
    recent_assistant_messages = [
        message.content
        for message in history
        if message.role is MessageRole.ASSISTANT
    ][-2:]
    return build_conversation_context(
        recent_user_queries=recent_user_queries,
        recent_assistant_messages=recent_assistant_messages,
    )


def _apply_critical_verification(
    *,
    response: GroundedAnswerResponse,
    evidence_package,
) -> tuple[GroundedAnswerResponse, dict[str, object]]:
    """Apply the first strict verification pass for Critical responses."""

    verifier_result = verify_critical_response(
        response=response,
        evidence_package=evidence_package,
    )

    metadata = {
        "decision": verifier_result.decision,
        "reason": verifier_result.reason,
        "supported_claim_count": verifier_result.supported_claim_count,
        "partially_supported_claim_count": verifier_result.partially_supported_claim_count,
        "unsupported_claim_count": verifier_result.unsupported_claim_count,
        "contradicted_claim_count": verifier_result.contradicted_claim_count,
        "retry_query_text": verifier_result.retry_query_text,
        "claims": [
            {
                "text": claim.text,
                "status": claim.status,
                "matched_chunk_ids": list(claim.matched_chunk_ids),
                "missing_terms": list(claim.missing_terms),
                "support_score": claim.support_score,
                "contradiction_chunk_ids": list(claim.contradiction_chunk_ids),
            }
            for claim in verifier_result.claims
        ],
        "unsupported_claims_detected": verifier_result.unsupported_claims_detected,
        "contradiction_detected": verifier_result.contradiction_detected,
        "bounded_correction_attempted": verifier_result.bounded_correction_attempted,
    }

    if verifier_result.decision == "accept":
        return response, metadata

    degraded_reasons = list(response.degraded_reasons)
    if verifier_result.reason and verifier_result.reason not in degraded_reasons:
        degraded_reasons.append(verifier_result.reason)

    degraded_response = response.model_copy(
        update={
            "verification_status": "degraded",
            "degraded_reasons": degraded_reasons,
            "support_summary": (
                "insufficient"
                if verifier_result.decision == "refuse" or verifier_result.contradiction_detected
                else "partial"
            ),
            "confidence_label": (
                "low"
                if verifier_result.decision == "refuse" or verifier_result.contradiction_detected
                else response.confidence_label
            ),
            "confidence_score": (
                0.0
                if verifier_result.decision == "refuse" or verifier_result.contradiction_detected
                else min(response.confidence_score, 0.49)
            ),
        }
    )
    return degraded_response, metadata


def _normalize_critical_response(*, response: GroundedAnswerResponse) -> GroundedAnswerResponse:
    """Enforce one consistent shape for final Critical responses."""

    if response.verification_status == "passed":
        return response.model_copy(
            update={
                "support_summary": "grounded",
                "degraded_reasons": [],
                "confidence_score": min(max(response.confidence_score, 0.5), 1.0),
            }
        )

    degraded_reasons = list(response.degraded_reasons)
    if not degraded_reasons:
        degraded_reasons = ["INSUFFICIENT_SUPPORT"]
    return response.model_copy(
        update={
            "support_summary": (
                "insufficient" if response.confidence_score <= 0.0 else response.support_summary
            ),
            "confidence_label": "low",
            "confidence_score": min(response.confidence_score, 0.49),
            "degraded_reasons": degraded_reasons,
        }
    )


_CHUNK_ROLE_AUTHORITY: dict[str, int] = {
    "section_header": 5,
    "section_body": 4,
    "document_header": 3,
    "body": 2,
    "section_list": 1,
    "list": 0,
}


def _chunk_authority(chunk_role: str, score: float) -> tuple[int, float]:
    """Return a sortable (authority_rank, score) pair for one evidence item."""
    return (_CHUNK_ROLE_AUTHORITY.get(chunk_role, 2), score)


def _critical_conflict_policy(*, namespace: Namespace) -> dict[str, object]:
    """Return the namespace-aware conflict handling policy for Critical."""
    freshness_profile = getattr(namespace, "freshness_profile", FreshnessProfile.BALANCED)
    resolution_mode = (
        "prefer_fresher_evidence_when_available"
        if freshness_profile is FreshnessProfile.AGGRESSIVE
        else "surface_conflict"
    )
    return {
        "freshness_profile": freshness_profile.value,
        "resolution_mode": resolution_mode,
        "source_authority_ranking_enabled": True,
    }


def _apply_critical_conflict_disclosure(
    *,
    namespace: Namespace,
    response: GroundedAnswerResponse,
    evidence_package,
    critical_verifier_metadata: dict[str, object],
) -> tuple[GroundedAnswerResponse, dict[str, object]]:
    """Resolve or surface contradictory evidence for Critical responses.

    When resolution_mode is 'prefer_fresher_evidence_when_available' and
    is_freshness_conflict_resolution_enabled() is True, this function actively
    selects the highest-authority (by chunk_role) and highest-score conflicting
    chunk as the authoritative source and rebuilds the answer from its content,
    rather than simply surfacing the conflict.

    In all other cases it surfaces the conflict explicitly.
    """
    if not critical_verifier_metadata.get("contradiction_detected"):
        return response, critical_verifier_metadata

    conflicting_chunk_ids: list[str] = []
    for claim in critical_verifier_metadata.get("claims", []):
        if not isinstance(claim, dict):
            continue
        for chunk_id in claim.get("contradiction_chunk_ids", []):
            if isinstance(chunk_id, str) and chunk_id not in conflicting_chunk_ids:
                conflicting_chunk_ids.append(chunk_id)

    conflicting_items = [
        item for item in evidence_package.items if item.chunk_id in conflicting_chunk_ids
    ]
    conflicting_citation_ids = [item.citation_id for item in conflicting_items]

    conflict_policy = _critical_conflict_policy(namespace=namespace)
    resolution_mode = conflict_policy.get("resolution_mode", "surface_conflict")

    # Active resolution: pick the highest-authority conflicting chunk and use it.
    if (
        resolution_mode == "prefer_fresher_evidence_when_available"
        and is_freshness_conflict_resolution_enabled()
        and conflicting_items
    ):
        winning_item = max(
            conflicting_items,
            key=lambda item: _chunk_authority(item.chunk_role, item.score),
        )
        resolved_response = response.model_copy(
            update={
                "answer": (
                    f"Evidence conflict detected and resolved using the highest-authority source "
                    f"({winning_item.citation_id}, role={winning_item.chunk_role}). "
                    f"Based on that source: {winning_item.text[:400].strip()}"
                ),
                "verification_status": "degraded",
                "degraded_reasons": ["CONFLICT_RESOLVED_BY_SOURCE_AUTHORITY"],
            }
        )
        return resolved_response, {
            **critical_verifier_metadata,
            "source_conflict_detected": True,
            "conflict_policy": conflict_policy,
            "conflicting_chunk_ids": conflicting_chunk_ids,
            "conflicting_citation_ids": conflicting_citation_ids,
            "conflict_resolution_applied": "source_authority",
            "winning_chunk_id": winning_item.chunk_id,
            "winning_citation_id": winning_item.citation_id,
            "winning_chunk_role": winning_item.chunk_role,
        }

    # Default: surface the conflict explicitly.
    citation_summary = (
        ", ".join(conflicting_citation_ids) if conflicting_citation_ids else "the selected evidence"
    )
    disclosed_response = response.model_copy(
        update={
            "answer": (
                "The selected evidence conflicts on this point, so I cannot verify a single grounded answer. "
                f"Review {citation_summary}."
            ),
        }
    )
    return disclosed_response, {
        **critical_verifier_metadata,
        "source_conflict_detected": True,
        "conflict_policy": conflict_policy,
        "conflicting_chunk_ids": conflicting_chunk_ids,
        "conflicting_citation_ids": conflicting_citation_ids,
        "conflict_resolution_applied": "surface_conflict",
    }


def _critical_policy_metadata(*, namespace: Namespace) -> dict[str, object]:
    """Build Critical policy metadata for external and internal recovery paths."""

    external_fallback = resolve_external_fallback_policy(namespace=namespace)
    internal_model_retrieval = resolve_internal_model_retrieval_policy(namespace=namespace)
    return {
        "external_fallback": {
            "allowed": external_fallback.allowed,
            "reason": external_fallback.reason,
            "attempted": external_fallback.attempted,
            "sources_consulted": list(external_fallback.sources_consulted),
        },
        "internal_model_retrieval": {
            "allowed": internal_model_retrieval.allowed,
            "attempted": internal_model_retrieval.attempted,
            "reason": internal_model_retrieval.reason,
        },
        "conflict_resolution": _critical_conflict_policy(namespace=namespace),
    }


def _evidence_quality_score(verifier_metadata: dict[str, object]) -> float:
    """Compute average claim support score from verifier metadata.

    Returns a value in [0, 1].  Used as the quality gate for corrective retrieval:
    a corrective attempt is only accepted when its quality score exceeds the
    first-pass score by at least min_evidence_quality_improvement().
    """
    claims = verifier_metadata.get("claims", [])
    if not claims or not isinstance(claims, list):
        return 0.0
    scores = [
        float(c.get("support_score", 0.0))
        for c in claims
        if isinstance(c, dict)
    ]
    return sum(scores) / len(scores) if scores else 0.0


async def _run_one_corrective_attempt(
    *,
    session: AsyncSession,
    tenant_context: TenantContext,
    namespace: Namespace,
    query_request: QueryRequest,
    query_plan: QueryPlan,
    retry_query_text: str,
    attempt_number: int,
    agent_instructions: str = "",
) -> tuple[RetrievalBundle, object, GroundedAnswerResponse, str, dict[str, object]] | None:
    """Run one corrective retrieval + re-verification pass."""
    retry_plan = replace(
        query_plan,
        resolved_query_text=f"{query_plan.resolved_query_text} {retry_query_text}".strip(),
        retrieval_query_text=retry_query_text,
        retrieval_queries=(retry_query_text, *query_plan.retrieval_queries),
    )

    retrieval_bundle = await retrieve_hybrid_candidates(
        session=session,
        tenant_id=tenant_context.tenant_id,
        namespace_id=query_request.namespace_id,
        query_text=query_request.query,
        query_plan=retry_plan,
        execution_tier=ExecutionTier.CRITICAL,
        freshness_profile=getattr(namespace, "freshness_profile", FreshnessProfile.BALANCED),
    )
    evidence_package = package_evidence(
        retrieval_bundle,
        query_text=retry_plan.resolved_query_text,
        execution_tier=ExecutionTier.CRITICAL,
    )
    if not evidence_package.items:
        return None

    draft = await generate_answer_from_evidence(
        query_text=query_request.query,
        evidence_package=evidence_package,
        agent_instructions=agent_instructions,
    )
    response = shape_grounded_response(draft=draft, evidence_package=evidence_package)
    response = response.model_copy(update={"verification_status": "passed", "degraded_reasons": []})
    response, verifier_metadata = _apply_critical_verification(
        response=response,
        evidence_package=evidence_package,
    )
    if verifier_metadata.get("decision") == "accept":
        response = response.model_copy(
            update={"verification_status": "passed", "degraded_reasons": [], "support_summary": "grounded"}
        )
    verifier_metadata = {
        **verifier_metadata,
        "bounded_correction_attempted": True,
        "corrective_attempt": {
            "attempt_number": attempt_number,
            "retry_query_text": retry_query_text,
            "retrieval_query_count": len(retry_plan.retrieval_queries),
            "selected_evidence_ids": list(evidence_package.selected_evidence_ids),
            "resulting_decision": verifier_metadata.get("decision"),
            "resulting_reason": verifier_metadata.get("reason"),
        },
    }
    return retrieval_bundle, evidence_package, response, draft.generator_provider, verifier_metadata


async def _run_critical_corrective_loop(
    *,
    session: AsyncSession,
    tenant_context: TenantContext,
    namespace: Namespace,
    query_request: QueryRequest,
    query_plan: QueryPlan,
    retry_query_text: str,
    first_pass_verifier_metadata: dict[str, object],
    agent_instructions: str = "",
) -> tuple[RetrievalBundle, object, GroundedAnswerResponse, str, dict[str, object]] | None:
    """Run up to max_crag_attempts() corrective retrieval passes with a quality gate.

    A corrective pass is only accepted when:
    1. It returns at least one evidence item.
    2. The average claim support score improves by at least
       min_evidence_quality_improvement() over the first-pass score.
    3. Or the verifier decision improves to 'accept'.

    All attempts are recorded in trace metadata regardless of outcome.
    """
    multi_attempt = is_multi_attempt_crag_enabled()
    max_attempts = max_crag_attempts() if multi_attempt else 1
    quality_threshold = min_evidence_quality_improvement()
    first_pass_quality = _evidence_quality_score(first_pass_verifier_metadata)

    all_attempts: list[dict[str, object]] = []
    current_retry_text = retry_query_text

    for attempt_number in range(1, max_attempts + 1):
        result = await _run_one_corrective_attempt(
            session=session,
            tenant_context=tenant_context,
            namespace=namespace,
            query_request=query_request,
            query_plan=query_plan,
            retry_query_text=current_retry_text,
            attempt_number=attempt_number,
            agent_instructions=agent_instructions,
        )
        if result is None:
            all_attempts.append({
                "attempt_number": attempt_number,
                "retry_query_text": current_retry_text,
                "outcome": "no_evidence_returned",
            })
            break

        retrieval_bundle, evidence_package, response, generator_provider, verifier_metadata = result
        attempt_quality = _evidence_quality_score(verifier_metadata)
        decision = verifier_metadata.get("decision")

        attempt_record = {
            "attempt_number": attempt_number,
            "retry_query_text": current_retry_text,
            "quality_score": attempt_quality,
            "first_pass_quality": first_pass_quality,
            "quality_improvement": attempt_quality - first_pass_quality,
            "resulting_decision": decision,
            "resulting_reason": verifier_metadata.get("reason"),
            "selected_evidence_ids": list(evidence_package.selected_evidence_ids),
        }
        all_attempts.append(attempt_record)

        # Accept if decision improved to 'accept' or quality improved enough.
        quality_improved = (attempt_quality - first_pass_quality) >= quality_threshold
        if decision == "accept" or quality_improved:
            final_metadata = {
                **verifier_metadata,
                "bounded_correction_attempted": True,
                "corrective_attempts": all_attempts,
                "corrective_attempt": verifier_metadata.get("corrective_attempt"),
            }
            return retrieval_bundle, evidence_package, response, generator_provider, final_metadata

        # Prepare next attempt: focus on remaining missing terms from this attempt.
        next_retry_text = verifier_metadata.get("retry_query_text")
        if isinstance(next_retry_text, str) and next_retry_text.strip():
            current_retry_text = next_retry_text.strip()
        else:
            break

    # No attempt met the quality gate.
    return None


# Keep old name as a thin shim so existing call sites keep working.
async def _run_critical_corrective_retry(
    *,
    session: AsyncSession,
    tenant_context: TenantContext,
    namespace: Namespace,
    query_request: QueryRequest,
    query_plan: QueryPlan,
    retry_query_text: str,
    first_pass_verifier_metadata: dict[str, object] | None = None,
    agent_instructions: str = "",
) -> tuple[RetrievalBundle, object, GroundedAnswerResponse, str, dict[str, object]] | None:
    """Delegate to the multi-attempt corrective loop."""
    return await _run_critical_corrective_loop(
        session=session,
        tenant_context=tenant_context,
        namespace=namespace,
        query_request=query_request,
        query_plan=query_plan,
        retry_query_text=retry_query_text,
        first_pass_verifier_metadata=first_pass_verifier_metadata or {},
        agent_instructions=agent_instructions,
    )


async def _apply_critical_recovery_paths(
    *,
    namespace: Namespace,
    response: GroundedAnswerResponse,
    evidence_package,
    critical_verifier_metadata: dict[str, object],
    retrieval_bundle: RetrievalBundle,
    query_text: str = "",
) -> tuple[GroundedAnswerResponse, object, dict[str, object], dict[str, object]]:
    """Apply bounded internal and external recovery paths for Critical mode."""

    recovery_metadata: dict[str, object] = {
        "internal_model_retrieval": {
            "attempted": False,
            "used": False,
            "reason": None,
        },
        "external_fallback": {
            "attempted": False,
            "used": False,
            "reason": None,
            "sources_consulted": [],
        },
    }

    updated_evidence_package = evidence_package
    updated_response = response

    internal_result = run_internal_model_retrieval(
        namespace=namespace,
        evidence_package=evidence_package,
        fused_hits=getattr(retrieval_bundle, "fused_hits", None),
    )
    recovery_metadata["internal_model_retrieval"] = {
        "attempted": internal_result.attempted,
        "used": internal_result.used,
        "reason": internal_result.reason,
    }
    if internal_result.evidence_package is not None:
        updated_evidence_package = internal_result.evidence_package
        if internal_result.used:
            # Re-check the current response against the widened grounded evidence set.
            updated_response, updated_metadata = _apply_critical_verification(
                response=updated_response,
                evidence_package=updated_evidence_package,
            )
            critical_verifier_metadata = {
                **updated_metadata,
                "bounded_correction_attempted": critical_verifier_metadata.get(
                    "bounded_correction_attempted",
                    False,
                ),
            }

    updated_metadata = {
        **critical_verifier_metadata,
        "recovery_paths": recovery_metadata,
    }

    if critical_verifier_metadata.get("decision") in {"refuse", "degrade"}:
        external_result = await run_allowlisted_external_fallback(
            namespace=namespace,
            degraded_response=updated_response,
            query_text=query_text,
        )
        recovery_metadata["external_fallback"] = {
            "attempted": external_result.attempted,
            "used": external_result.used,
            "reason": external_result.reason,
            "sources_consulted": list(external_result.sources_consulted),
        }
        if external_result.response is not None and updated_response.verification_status == "degraded":
            updated_response = external_result.response

        updated_metadata = {
            **critical_verifier_metadata,
            "recovery_paths": recovery_metadata,
        }
    return updated_response, updated_evidence_package, updated_metadata, recovery_metadata


async def _update_query_trace_timings(
    *,
    session: AsyncSession,
    trace: QueryTrace,
    stage_latencies_ms: dict[str, int],
    total_latency_ms: int,
) -> None:
    """Persist finalized timing metadata after the initial trace write completes."""

    trace.stage_latencies_ms = stage_latencies_ms
    trace.total_latency_ms = total_latency_ms
    try:
        await session.commit()
    except SQLAlchemyError as exc:
        await session.rollback()
        raise QueryServiceError(
            "Failed to finalize query trace timings.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from exc


def _async_queued_response() -> GroundedAnswerResponse:
    """Return a placeholder response for one queued Verified request."""

    return GroundedAnswerResponse(
        answer="Verified request queued. Poll the run endpoint for the completed result.",
        citations=[],
        confidence_score=0.0,
        confidence_label="low",
        support_summary="insufficient",
        verification_status="degraded",
        degraded_reasons=["RUN_QUEUED"],
        generator_provider="async-verified-run-v1",
        provider_backend="async_verified_run_v1",
        provider_model=None,
        provider_fallback_used=False,
        provider_fallback_from=None,
    )


def _copy_trace_result(*, target: QueryTrace, source: QueryTrace, run_status: str) -> None:
    """Copy one completed trace result into an existing async run record."""

    target.namespace_id = source.namespace_id
    target.agent_id = source.agent_id
    target.conversation_id = source.conversation_id
    target.selected_mode = source.selected_mode
    target.requested_tier = source.requested_tier
    target.router_recommendation = source.router_recommendation
    target.effective_tier = source.effective_tier
    target.routing_reason = source.routing_reason
    target.query_redacted = source.query_redacted
    target.query_ciphertext = source.query_ciphertext
    target.retrieved_chunk_ids = list(source.retrieved_chunk_ids)
    target.selected_evidence_ids = list(source.selected_evidence_ids)
    target.generator_provider = source.generator_provider
    target.verifier_result = {
        **source.verifier_result,
        "run_status": run_status,
        "async_execution": True,
    }
    target.final_answer_redacted = source.final_answer_redacted
    target.citations = list(source.citations)
    target.overall_confidence = source.overall_confidence
    target.degraded_reasons = list(source.degraded_reasons)
    target.stage_latencies_ms = dict(source.stage_latencies_ms)
    target.total_latency_ms = source.total_latency_ms
    target.token_usage = dict(source.token_usage)


async def queue_async_verified_query(
    *,
    session: AsyncSession,
    tenant_context: TenantContext,
    query_request: QueryRequest,
) -> QueryExecutionResult:
    """Persist one queued Verified run to be completed asynchronously."""

    queued_response = _async_queued_response()
    retrieval_bundle = RetrievalBundle(sparse_hits=[], dense_hits=[], fused_hits=[])
    routing_decision = ExecutionRoutingDecision(
        requested_tier=ExecutionTier.CRITICAL,
        router_recommendation=ExecutionTier.CRITICAL,
        effective_tier=ExecutionTier.CRITICAL,
        routing_reason="critical_async_queued",
        request_source="query_request",
    )
    trace = await _persist_query_trace(
        session=session,
        tenant_context=tenant_context,
        query_request=query_request,
        response=queued_response,
        retrieval_bundle=retrieval_bundle,
        stage_latencies_ms={
            "retrieval_ms": 0,
            "evidence_packaging_ms": 0,
            "answering_ms": 0,
        },
        total_latency_ms=0,
        generator_provider=queued_response.generator_provider,
        routing_decision=routing_decision,
        run_status="queued",
    )
    return QueryExecutionResult(response=queued_response, trace_id=trace.trace_id)


async def process_async_verified_query(
    *,
    trace_id: uuid.UUID,
    tenant_context: TenantContext,
    query_request: QueryRequest,
) -> None:
    """Complete one queued Verified run in the background."""

    session_factory = get_session_factory()
    async with session_factory() as session:
        trace = await session.get(QueryTrace, trace_id)
        if trace is None:
            return
        trace.verifier_result = {
            **trace.verifier_result,
            "run_status": "running",
            "async_execution": True,
        }
        await session.commit()

        try:
            result = await execute_standard_query(
                session=session,
                tenant_context=tenant_context,
                query_request=query_request.model_copy(update={"prefer_async": False}),
            )
        except QueryServiceError as exc:
            trace = await session.get(QueryTrace, trace_id)
            if trace is None:
                return
            trace.verifier_result = {
                **trace.verifier_result,
                "run_status": "failed",
                "async_execution": True,
                "verification_reason": "ASYNC_EXECUTION_FAILED",
                "error_detail": exc.detail,
            }
            trace.final_answer_redacted = "Verified request failed before a final answer was produced."
            trace.generator_provider = "async-verified-run-v1"
            trace.overall_confidence = 0.0
            trace.degraded_reasons = ["ASYNC_EXECUTION_FAILED"]
            trace.selected_evidence_ids = []
            trace.retrieved_chunk_ids = []
            trace.citations = []
            trace.total_latency_ms = 0
            trace.stage_latencies_ms = {
                "retrieval_ms": 0,
                "evidence_packaging_ms": 0,
                "answering_ms": 0,
            }
            trace.token_usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            await session.commit()
            return

        completed_trace = await session.get(QueryTrace, result.trace_id)
        queued_trace = await session.get(QueryTrace, trace_id)
        if completed_trace is None or queued_trace is None:
            return
        _copy_trace_result(target=queued_trace, source=completed_trace, run_status="completed")
        await session.delete(completed_trace)
        await session.commit()


ProgressCallback = Callable[[dict], Awaitable[None]]


async def _emit(on_progress: ProgressCallback | None, event: dict) -> None:
    if on_progress is not None:
        await on_progress(event)


async def execute_standard_query(
    *,
    session: AsyncSession,
    tenant_context: TenantContext,
    query_request: QueryRequest,
    agent_id: uuid.UUID | None = None,
    conversation_id: uuid.UUID | None = None,
    selected_mode: UserFacingMode | None = None,
    agent_instructions: str = "",
    on_progress: ProgressCallback | None = None,
) -> QueryExecutionResult:
    """Run the Standard query path and persist a trace for the result."""

    namespace = await _get_namespace_for_tenant(
        session=session,
        tenant_id=tenant_context.tenant_id,
        namespace_id=query_request.namespace_id,
    )
    started_at = time.perf_counter()
    query_plan: QueryPlan | None = None

    await _emit(on_progress, {"type": "step_started", "step": "init", "label": "Workflow Steps"})

    conv_ms: int = 0
    check_ms: int = 0

    if _is_smalltalk_query(query_request.query):
        routing_decision = _resolve_execution_routing(
            query_request=query_request,
            selected_mode=selected_mode,
            namespace_min_execution_tier=namespace.min_execution_tier,
        )
        await _emit(on_progress, {"type": "step_completed", "step": "init", "label": "Workflow Steps", "duration_ms": int((time.perf_counter() - started_at) * 1000)})
    else:
        conv_started = time.perf_counter()
        await _emit(on_progress, {"type": "step_started", "step": "conversation_history", "label": "Create Message History"})
        conv_ctx = await _resolve_conversation_context(
            session=session,
            tenant_id=tenant_context.tenant_id,
            conversation_id=conversation_id,
            current_query=query_request.query,
        )
        conv_ms = int((time.perf_counter() - conv_started) * 1000)
        message_count = len(conv_ctx.recent_user_queries) if conv_ctx else 0
        await _emit(on_progress, {"type": "step_completed", "step": "conversation_history", "label": "Create Message History", "duration_ms": conv_ms, "message_count": message_count})

        check_started = time.perf_counter()
        await _emit(on_progress, {"type": "step_started", "step": "check_retrieval", "label": "Check Retrieval Need"})
        query_plan = build_query_plan(
            query_request.query,
            conversation_context=conv_ctx,
        )
        routing_decision = _resolve_execution_routing(
            query_request=query_request,
            selected_mode=selected_mode,
            namespace_min_execution_tier=namespace.min_execution_tier,
            query_plan=query_plan,
        )
        query_plan = refine_query_plan_for_execution_tier(
            query_plan,
            execution_tier=routing_decision.effective_tier,
        )
        _log_query_identification(query_plan=query_plan)
        check_ms = int((time.perf_counter() - check_started) * 1000)
        await _emit(on_progress, {"type": "step_completed", "step": "check_retrieval", "label": "Check Retrieval Need", "duration_ms": check_ms})
        await _emit(on_progress, {"type": "step_completed", "step": "init", "label": "Workflow Steps", "duration_ms": int((time.perf_counter() - started_at) * 1000)})

    if not _supports_execution_tier(
        available_tier=routing_decision.effective_tier,
        required_tier=namespace.min_execution_tier,
    ):
        raise QueryServiceError(
            "Namespace requires a higher execution tier than the resolved query path.",
            status_code=status.HTTP_409_CONFLICT,
        )

    bind_execution_context(
        requested_tier=routing_decision.requested_tier.value,
        router_recommendation=routing_decision.router_recommendation.value,
        effective_tier=routing_decision.effective_tier.value,
        routing_reason=routing_decision.routing_reason,
        selected_mode=selected_mode.value if selected_mode is not None else None,
    )

    if query_plan is None:
        retrieval_bundle = RetrievalBundle(sparse_hits=[], dense_hits=[], fused_hits=[])
        await _emit(on_progress, {"type": "step_started", "step": "generate", "label": "Generate"})
        response = _smalltalk_response()
        await _emit(on_progress, {"type": "step_completed", "step": "generate", "label": "Generate", "duration_ms": 0})
        stage_latencies_ms = {
            "retrieval_ms": 0,
            "evidence_packaging_ms": 0,
            "answering_ms": 0,
        }
        trace_started = time.perf_counter()
        trace = await _persist_query_trace(
            session=session,
            tenant_context=tenant_context,
            query_request=query_request,
            response=response,
            retrieval_bundle=retrieval_bundle,
            stage_latencies_ms=stage_latencies_ms,
            total_latency_ms=int((time.perf_counter() - started_at) * 1000),
            generator_provider="clarification-handler-v1",
            query_plan=None,
            evidence_debug=None,
            routing_decision=routing_decision,
            agent_id=agent_id,
            conversation_id=conversation_id,
            selected_mode=selected_mode,
        )
        trace_ms = int((time.perf_counter() - trace_started) * 1000)
        await _update_query_trace_timings(
            session=session,
            trace=trace,
            stage_latencies_ms={
                **stage_latencies_ms,
                "trace_persistence_ms": trace_ms,
            },
            total_latency_ms=int((time.perf_counter() - started_at) * 1000),
        )
        return QueryExecutionResult(
            response=response,
            trace_id=trace.trace_id,
        )

    research_started = time.perf_counter()
    await _emit(on_progress, {"type": "step_started", "step": "research", "label": "Research"})
    retrieval_started = time.perf_counter()
    _retrieval_bundle: RetrievalBundle | None = None
    _retrieval_error = False
    for _attempt in range(2):
        try:
            _retrieval_bundle = await retrieve_hybrid_candidates(
                session=session,
                tenant_id=tenant_context.tenant_id,
                namespace_id=query_request.namespace_id,
                query_text=query_request.query,
                query_plan=query_plan,
                execution_tier=routing_decision.effective_tier,
                freshness_profile=getattr(namespace, "freshness_profile", FreshnessProfile.BALANCED),
            )
            _retrieval_error = False
            break
        except RetrievalError:
            _retrieval_error = True
            if _attempt == 0:
                logger.warning("retrieval_failed_retrying", query=query_request.query)
                await asyncio.sleep(0.4)

    if _retrieval_error:
        retrieval_ms = int((time.perf_counter() - retrieval_started) * 1000)
        research_ms = int((time.perf_counter() - research_started) * 1000)
        await _emit(on_progress, {
            "type": "step_completed",
            "step": "research",
            "label": "Research",
            "duration_ms": research_ms,
            "evidence_count": 0,
        })
        await _emit(on_progress, {"type": "step_started", "step": "generate", "label": "Generate"})
        _empty_bundle = RetrievalBundle(sparse_hits=[], dense_hits=[], fused_hits=[])
        _empty_evidence = package_evidence(
            _empty_bundle,
            query_text=query_plan.resolved_query_text,
            execution_tier=routing_decision.effective_tier,
        )
        _degraded = shape_degraded_response(
            reason="RETRIEVAL_ERROR",
            answer_text="I was unable to search for grounded evidence for this query. Please try again.",
        )
        _retrieval_lats: dict[str, int] = {
            "conversation_history_ms": conv_ms,
            "check_retrieval_ms": check_ms,
            "retrieval_ms": retrieval_ms,
            "evidence_packaging_ms": 0,
            "answering_ms": 0,
        }
        await _emit(on_progress, {"type": "step_completed", "step": "generate", "label": "Generate", "duration_ms": 0})
        _dbg = _evidence_debug_summary(
            selected_evidence_ids=_empty_evidence.selected_evidence_ids,
            evidence_package=_empty_evidence,
        )
        _trace_t0 = time.perf_counter()
        _trace = await _persist_query_trace(
            session=session,
            tenant_context=tenant_context,
            query_request=query_request,
            response=_degraded,
            retrieval_bundle=_empty_bundle,
            stage_latencies_ms=_retrieval_lats,
            total_latency_ms=int((time.perf_counter() - started_at) * 1000),
            generator_provider="degraded-handler-v1",
            query_plan=query_plan,
            evidence_debug=_dbg,
            routing_decision=routing_decision,
            agent_id=agent_id,
            conversation_id=conversation_id,
            selected_mode=selected_mode,
            token_usage=None,
        )
        _trace_ms = int((time.perf_counter() - _trace_t0) * 1000)
        await _update_query_trace_timings(
            session=session,
            trace=_trace,
            stage_latencies_ms={**_retrieval_lats, "trace_persistence_ms": _trace_ms},
            total_latency_ms=int((time.perf_counter() - started_at) * 1000),
        )
        return QueryExecutionResult(response=_degraded, trace_id=_trace.trace_id)

    retrieval_ms = int((time.perf_counter() - retrieval_started) * 1000)
    retrieval_bundle = _retrieval_bundle  # type: ignore[assignment]

    evidence_started = time.perf_counter()
    evidence_package = package_evidence(
        retrieval_bundle,  # type: ignore[arg-type]
        query_text=query_plan.resolved_query_text,
        execution_tier=routing_decision.effective_tier,
    )
    _log_selected_evidence(
        query_text=query_plan.resolved_query_text,
        evidence_package=evidence_package,
    )
    evidence_ms = int((time.perf_counter() - evidence_started) * 1000)
    research_ms = int((time.perf_counter() - research_started) * 1000)
    await _emit(on_progress, {
        "type": "step_completed",
        "step": "research",
        "label": "Research",
        "duration_ms": research_ms,
        "evidence_count": len(evidence_package.items),
    })

    answering_started = time.perf_counter()
    await _emit(on_progress, {"type": "step_started", "step": "generate", "label": "Generate"})
    draft_token_usage: dict[str, int] | None = None
    if not evidence_package.items:
        response = shape_degraded_response(
            reason="NO_GROUNDED_EVIDENCE",
            answer_text="I could not find grounded evidence for this query.",
        )
        generator_provider = "degraded-handler-v1"
        critical_verifier_metadata = None
    else:
        try:
            draft = await generate_answer_from_evidence(
                query_text=query_request.query,
                evidence_package=evidence_package,
                agent_instructions=agent_instructions,
            )
            draft_token_usage = draft.token_usage
            response = shape_grounded_response(
                draft=draft,
                evidence_package=evidence_package,
            )
            generator_provider = draft.generator_provider
            if routing_decision.effective_tier is ExecutionTier.CRITICAL:
                response, critical_verifier_metadata = _apply_critical_verification(
                    response=response,
                    evidence_package=evidence_package,
                )
                retry_query_text = critical_verifier_metadata.get("retry_query_text")
                should_retry = (
                    critical_verifier_metadata.get("decision") in {"refuse", "degrade"}
                    and critical_verifier_metadata.get("reason") in {
                        "UNSUPPORTED_CLAIMS", "PARTIAL_SUPPORT", "CONTRADICTORY_EVIDENCE"
                    }
                    and isinstance(retry_query_text, str)
                    and retry_query_text.strip()
                )
                if should_retry:
                    corrective_result = await _run_critical_corrective_retry(
                        session=session,
                        tenant_context=tenant_context,
                        namespace=namespace,
                        query_request=query_request,
                        query_plan=query_plan,
                        retry_query_text=retry_query_text.strip(),
                        first_pass_verifier_metadata=critical_verifier_metadata,
                        agent_instructions=agent_instructions,
                    )
                    if corrective_result is not None:
                        (
                            retrieval_bundle,
                            evidence_package,
                            response,
                            generator_provider,
                            critical_verifier_metadata,
                        ) = corrective_result
                    else:
                        critical_verifier_metadata = {
                            **critical_verifier_metadata,
                            "bounded_correction_attempted": True,
                            "corrective_attempts_exhausted": True,
                        }
                if response.verification_status == "degraded":
                    response, evidence_package, critical_verifier_metadata, recovery_metadata = await _apply_critical_recovery_paths(
                        namespace=namespace,
                        response=response,
                        evidence_package=evidence_package,
                        critical_verifier_metadata=critical_verifier_metadata,
                        retrieval_bundle=retrieval_bundle,
                        query_text=query_request.query,
                    )
                response, critical_verifier_metadata = _apply_critical_conflict_disclosure(
                    namespace=namespace,
                    response=response,
                    evidence_package=evidence_package,
                    critical_verifier_metadata=critical_verifier_metadata,
                )
                response = _normalize_critical_response(response=response)
            else:
                critical_verifier_metadata = None
        except (GroundedGenerationError, ResponseShapingError) as exc:
            degraded_reason, answer_text = _degraded_reason_for_generation_exception(exc)
            response = shape_degraded_response(
                reason=degraded_reason,
                answer_text=answer_text,
            )
            generator_provider = "degraded-handler-v1"
            critical_verifier_metadata = None
    answering_ms = int((time.perf_counter() - answering_started) * 1000)
    await _emit(on_progress, {"type": "step_completed", "step": "generate", "label": "Generate", "duration_ms": answering_ms})

    stage_latencies_ms = {
        "conversation_history_ms": conv_ms,
        "check_retrieval_ms": check_ms,
        "retrieval_ms": retrieval_ms,
        "evidence_packaging_ms": evidence_ms,
        "answering_ms": answering_ms,
    }
    evidence_debug = _evidence_debug_summary(
        selected_evidence_ids=evidence_package.selected_evidence_ids,
        evidence_package=evidence_package,
    )
    if critical_verifier_metadata is not None:
        evidence_debug = {
            **evidence_debug,
            "critical_verifier": critical_verifier_metadata,
        }
    if routing_decision.effective_tier is ExecutionTier.CRITICAL:
        policy_metadata = _critical_policy_metadata(namespace=namespace)
        if critical_verifier_metadata is not None:
            recovery_paths = critical_verifier_metadata.get("recovery_paths")
            if isinstance(recovery_paths, dict):
                if isinstance(recovery_paths.get("external_fallback"), dict):
                    external_recovery = {
                        key: value
                        for key, value in recovery_paths["external_fallback"].items()
                        if value is not None and not (key == "sources_consulted" and value == [])
                    }
                    policy_metadata["external_fallback"] = {
                        **policy_metadata["external_fallback"],
                        **external_recovery,
                    }
                if isinstance(recovery_paths.get("internal_model_retrieval"), dict):
                    internal_recovery = {
                        key: value
                        for key, value in recovery_paths["internal_model_retrieval"].items()
                        if value is not None
                    }
                    policy_metadata["internal_model_retrieval"] = {
                        **policy_metadata["internal_model_retrieval"],
                        **internal_recovery,
                    }
        evidence_debug = {
            **evidence_debug,
            "critical_policy": policy_metadata,
        }

    trace_started = time.perf_counter()
    trace = await _persist_query_trace(
        session=session,
        tenant_context=tenant_context,
        query_request=query_request,
        response=response,
        retrieval_bundle=retrieval_bundle,
        stage_latencies_ms=stage_latencies_ms,
        total_latency_ms=int((time.perf_counter() - started_at) * 1000),
        generator_provider=generator_provider,
        query_plan=query_plan,
        evidence_debug=evidence_debug,
        routing_decision=routing_decision,
        agent_id=agent_id,
        conversation_id=conversation_id,
        selected_mode=selected_mode,
        token_usage=draft_token_usage,
    )
    trace_ms = int((time.perf_counter() - trace_started) * 1000)

    finalized_stage_latencies_ms = {
        **stage_latencies_ms,
        "trace_persistence_ms": trace_ms,
    }
    await _update_query_trace_timings(
        session=session,
        trace=trace,
        stage_latencies_ms=finalized_stage_latencies_ms,
        total_latency_ms=int((time.perf_counter() - started_at) * 1000),
    )

    return QueryExecutionResult(
        response=response,
        trace_id=trace.trace_id,
    )
