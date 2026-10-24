"""Sparse, dense, and fused retrieval helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime
from uuid import UUID

from sqlalchemy import or_, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.reranker import RerankerError, resolve_retrieval_reranker
from app.core.telemetry import get_logger
from app.core.query_analysis import (
    QueryPlan,
    build_query_plan,
    has_strong_intent_signal,
    is_action_query,
    is_collection_query,
    is_comparison_query,
    is_count_query,
    is_dataset_summary_query,
    is_entity_context_query,
    is_field_extraction_query,
    score_text_against_query,
    tokenize_meaningful_terms,
)
from app.core.embeddings import EmbeddingError, embed_texts
from app.models import Document, DocumentChunkRecord, ExecutionTier, FreshnessProfile
from app.core.qdrant_client import VectorStoreError, search_dense_points
from app.pipeline.contracts import FusedRetrievedChunk, RetrievedChunk


class RetrievalError(RuntimeError):
    """Raised when a retrieval path cannot complete safely."""


logger = get_logger("app.retrieval")


@dataclass(frozen=True)
class RetrievalBundle:
    """All retrieval candidates produced for one query."""

    sparse_hits: list[RetrievedChunk]
    dense_hits: list[RetrievedChunk]
    fused_hits: list[FusedRetrievedChunk]
    debug: dict[str, object] | None = None


def _query_term_coverage_score(*, text: str, query_plan: QueryPlan) -> float:
    """Reward chunks that cover more of the query's meaningful vocabulary."""

    profile = query_plan.profile
    text_terms = tokenize_meaningful_terms(text)
    if not text_terms:
        return 0.0

    direct_overlap = len(set(profile.terms) & text_terms)
    expanded_overlap = len(set(profile.expanded_terms) & text_terms)
    score = direct_overlap * 3.5 + max(expanded_overlap - direct_overlap, 0) * 1.25

    normalized_text = " ".join(text.lower().split())
    for attribute in profile.attribute_terms:
        if attribute in normalized_text:
            score += 3.0
    return score


def _section_match_bonus(
    *,
    section_title: str | None,
    section_slug: str | None,
    chunk_role: str,
    chunk_index: int,
    query_plan: QueryPlan,
) -> float:
    """Reward chunks whose structural metadata aligns with the query."""

    profile = query_plan.profile
    section_fragments = [
        fragment
        for fragment in (
            section_title,
            section_slug.replace("-", " ") if section_slug else None,
            chunk_role.replace("_", " "),
        )
        if fragment
    ]
    if not section_fragments:
        if is_dataset_summary_query(profile) and chunk_role == "document_header":
            return 4.0
        return 0.0

    section_terms = tokenize_meaningful_terms(" ".join(section_fragments))
    overlap = len(set(profile.expanded_terms) & section_terms)
    score = overlap * 4.0

    if is_collection_query(profile) and overlap > 0:
        score += 6.5 if chunk_role == "section_header" else 4.5 if chunk_role == "section_list" else 2.5
    elif is_field_extraction_query(profile) and overlap > 0:
        score += 4.0
    elif is_dataset_summary_query(profile) and chunk_role == "document_header":
        score += 4.0

    if chunk_role == "section_header":
        score += 2.0
    if section_title and chunk_index <= 1 and is_field_extraction_query(profile):
        score += 1.0
    return score


def _intent_bonus_for_hit(*, hit: FusedRetrievedChunk, query_plan: QueryPlan) -> float:
    """Apply stronger intent-focused bonuses and penalties after hybrid fusion."""

    profile = query_plan.profile
    strong_intent = has_strong_intent_signal(hit.text, profile=profile)
    line_count = len([line for line in hit.text.splitlines() if line.strip()])
    structured = ":" in hit.text or line_count >= 2

    if is_field_extraction_query(profile) and not is_collection_query(profile):
        return 10.0 if strong_intent else -5.0
    if is_action_query(profile):
        if strong_intent and structured:
            return 9.0
        if strong_intent:
            return 4.5
        return -3.0
    if is_comparison_query(profile):
        if strong_intent and structured:
            return 8.0
        if strong_intent:
            return 3.5
        return -2.0
    if is_entity_context_query(profile):
        return 7.0 if strong_intent else -2.5
    if is_count_query(profile):
        if strong_intent and structured:
            return 8.0
        if strong_intent:
            return 3.5
        return -3.0
    if is_collection_query(profile):
        if strong_intent and structured:
            return 8.0
        if strong_intent:
            return 3.0
        return -6.0
    if is_dataset_summary_query(profile):
        if strong_intent:
            return 4.0
        if hit.chunk_index <= 1:
            return 2.5
    return 0.0


def _coerce_uuid(value: UUID | str) -> UUID:
    """Coerce UUID-like payload values into UUID instances."""

    return value if isinstance(value, UUID) else UUID(str(value))


def _query_prefers_freshness(query_plan: QueryPlan) -> bool:
    """Return whether the query explicitly asks for newer or current evidence."""

    freshness_terms = {
        "current",
        "latest",
        "new",
        "newer",
        "recent",
        "update",
        "updated",
        "version",
    }
    profile = query_plan.profile
    if set(profile.terms) & freshness_terms:
        return True
    if set(profile.attribute_terms) & freshness_terms:
        return True
    if set(profile.context_terms) & freshness_terms:
        return True
    return False


def _freshness_boost_weight(freshness_profile: FreshnessProfile) -> float:
    """Return one small score weight for the namespace freshness profile."""

    if freshness_profile is FreshnessProfile.AGGRESSIVE:
        return 0.7
    if freshness_profile is FreshnessProfile.BALANCED:
        return 0.4
    return 0.15


async def _fetch_document_freshness_metadata(
    *,
    session: AsyncSession,
    tenant_id: UUID,
    namespace_id: UUID,
    document_ids: set[UUID],
) -> dict[UUID, datetime]:
    """Load effective freshness timestamps for the candidate documents."""

    if not document_ids:
        return {}

    statement = select(
        Document.doc_id,
        Document.published_at,
        Document.created_at,
    ).where(
        Document.tenant_id == tenant_id,
        Document.namespace_id == namespace_id,
        Document.doc_id.in_(sorted(document_ids, key=str)),
    )

    try:
        result = await session.execute(statement)
    except SQLAlchemyError as exc:
        raise RetrievalError("Freshness metadata lookup failed.") from exc

    rows = result.all() if hasattr(result, "all") else list(result)
    metadata: dict[UUID, datetime] = {}
    for row in rows:
        if isinstance(row, dict):
            document_id = _coerce_uuid(row["doc_id"])
            published_at = row.get("published_at")
            created_at = row.get("created_at")
        else:
            document_id = _coerce_uuid(getattr(row, "doc_id"))
            published_at = getattr(row, "published_at", None)
            created_at = getattr(row, "created_at", None)
        effective_timestamp = published_at or created_at
        if isinstance(effective_timestamp, datetime):
            metadata[document_id] = effective_timestamp
    return metadata


async def _apply_enterprise_freshness_scoring(
    fused_hits: list[FusedRetrievedChunk],
    *,
    session: AsyncSession,
    tenant_id: UUID,
    namespace_id: UUID,
    query_plan: QueryPlan,
    execution_tier: ExecutionTier,
    freshness_profile: FreshnessProfile,
) -> tuple[list[FusedRetrievedChunk], dict[str, object]]:
    """Prefer newer evidence when the Enterprise query explicitly needs freshness."""

    settings = get_settings()
    if (
        execution_tier is not ExecutionTier.ENTERPRISE
        or not getattr(settings, "enterprise_temporal_scoring_enabled", True)
        or not fused_hits
        or not _query_prefers_freshness(query_plan)
    ):
        return fused_hits, {
            "applied": False,
            "reason": "not_requested",
            "freshness_profile": freshness_profile.value,
        }

    freshness_by_document = await _fetch_document_freshness_metadata(
        session=session,
        tenant_id=tenant_id,
        namespace_id=namespace_id,
        document_ids={hit.document_id for hit in fused_hits},
    )
    if len(freshness_by_document) < 2:
        return fused_hits, {
            "applied": False,
            "reason": "insufficient_document_metadata",
            "freshness_profile": freshness_profile.value,
        }

    timestamps = [value.timestamp() for value in freshness_by_document.values()]
    oldest_timestamp = min(timestamps)
    newest_timestamp = max(timestamps)
    if newest_timestamp <= oldest_timestamp:
        return fused_hits, {
            "applied": False,
            "reason": "no_temporal_separation",
            "freshness_profile": freshness_profile.value,
        }

    boost_weight = _freshness_boost_weight(freshness_profile)
    rescored_hits: list[FusedRetrievedChunk] = []
    for hit in fused_hits:
        freshness_timestamp = freshness_by_document.get(hit.document_id)
        if freshness_timestamp is None:
            rescored_hits.append(hit)
            continue
        freshness_ratio = (
            (freshness_timestamp.timestamp() - oldest_timestamp)
            / (newest_timestamp - oldest_timestamp)
        )
        rescored_hits.append(
            replace(
                hit,
                fused_score=hit.fused_score + freshness_ratio * boost_weight,
            )
        )

    logger.info(
        "enterprise_freshness_scoring_applied",
        freshness_profile=freshness_profile.value,
        boosted_documents=len(freshness_by_document),
    )
    rescored = sorted(
        rescored_hits,
        key=lambda hit: (-hit.fused_score, hit.chunk_index, hit.chunk_id),
    )
    return rescored, {
        "applied": True,
        "freshness_profile": freshness_profile.value,
        "boosted_documents": len(freshness_by_document),
        "top_chunk_ids": [hit.chunk_id for hit in rescored[:3]],
    }


async def sparse_retrieve_chunks(
    *,
    session: AsyncSession,
    tenant_id: UUID,
    namespace_id: UUID,
    query_text: str,
    limit: int | None = None,
) -> list[RetrievedChunk]:
    """Retrieve lexical candidates from PostgreSQL full-text search."""

    candidate_limit = limit or get_settings().retrieval_candidate_limit
    statement = text(
        """
        select
            chunk_id,
            tenant_id,
            namespace_id,
            doc_id,
            chunk_index,
            chunk_text,
            section_title,
            section_slug,
            chunk_role,
            starts_with_heading,
            is_list_block,
            ts_rank_cd(search_vector, websearch_to_tsquery('english', :query_text)) as score
        from document_chunks
        where tenant_id = :tenant_id
          and namespace_id = :namespace_id
          and search_vector @@ websearch_to_tsquery('english', :query_text)
        order by score desc, chunk_index asc
        limit :limit
        """
    )

    try:
        result = await session.execute(
            statement,
            {
                "tenant_id": tenant_id,
                "namespace_id": namespace_id,
                "query_text": query_text,
                "limit": candidate_limit,
            },
        )
    except SQLAlchemyError as exc:
        raise RetrievalError("Sparse retrieval query failed.") from exc

    rows = result.mappings().all()
    hits: list[RetrievedChunk] = []
    for index, row in enumerate(rows, start=1):
        hits.append(
            RetrievedChunk(
                chunk_id=str(row["chunk_id"]),
                tenant_id=_coerce_uuid(row["tenant_id"]),
                namespace_id=_coerce_uuid(row["namespace_id"]),
                document_id=_coerce_uuid(row["doc_id"]),
                chunk_index=int(row["chunk_index"]),
                text=str(row["chunk_text"]),
                score=float(row["score"]),
                rank=index,
                source="sparse",
                section_title=(
                    str(row["section_title"])
                    if row.get("section_title") is not None
                    else None
                ),
                section_slug=(
                    str(row["section_slug"])
                    if row.get("section_slug") is not None
                    else None
                ),
                chunk_role=str(row.get("chunk_role") or "body"),
                starts_with_heading=bool(row.get("starts_with_heading", False)),
                is_list_block=bool(row.get("is_list_block", False)),
            )
        )
    return hits


async def dense_retrieve_chunks(
    *,
    tenant_id: UUID,
    namespace_id: UUID,
    query_text: str,
    limit: int | None = None,
) -> list[RetrievedChunk]:
    """Retrieve semantic candidates from Qdrant."""

    candidate_limit = limit or get_settings().retrieval_candidate_limit
    try:
        query_embedding = (await embed_texts([query_text], purpose="query"))[0]
        points = search_dense_points(
            query_vector=query_embedding.vector,
            tenant_id=tenant_id,
            namespace_id=namespace_id,
            limit=candidate_limit,
        )
    except (EmbeddingError, IndexError) as exc:
        raise RetrievalError("Dense retrieval failed to embed the query.") from exc
    except VectorStoreError as exc:
        raise RetrievalError("Dense retrieval query failed.") from exc

    hits: list[RetrievedChunk] = []
    for index, point in enumerate(points, start=1):
        payload = point.payload or {}
        hits.append(
            RetrievedChunk(
                chunk_id=str(payload["chunk_id"]),
                tenant_id=_coerce_uuid(payload["tenant_id"]),
                namespace_id=_coerce_uuid(payload["namespace_id"]),
                document_id=_coerce_uuid(payload["document_id"]),
                chunk_index=int(payload["chunk_index"]),
                text=str(payload["text"]),
                score=float(point.score),
                rank=index,
                source="dense",
                section_title=(
                    str(payload["section_title"])
                    if payload.get("section_title") is not None
                    else None
                ),
                section_slug=(
                    str(payload["section_slug"])
                    if payload.get("section_slug") is not None
                    else None
                ),
                chunk_role=str(payload.get("chunk_role") or "body"),
                starts_with_heading=bool(payload.get("starts_with_heading", False)),
                is_list_block=bool(payload.get("is_list_block", False)),
            )
        )
    return hits


def _dedupe_retrieval_hits(hits: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Keep the best occurrence of each chunk within one retrieval source."""

    if not hits:
        return []

    best_by_chunk: dict[str, RetrievedChunk] = {}
    for hit in hits:
        current = best_by_chunk.get(hit.chunk_id)
        if current is None or (hit.rank, -hit.score, hit.chunk_index, hit.chunk_id) < (
            current.rank,
            -current.score,
            current.chunk_index,
            current.chunk_id,
        ):
            best_by_chunk[hit.chunk_id] = hit

    ordered_hits = sorted(
        best_by_chunk.values(),
        key=lambda hit: (
            hit.rank,
            -hit.score,
            hit.chunk_index,
            hit.chunk_id,
        ),
    )
    return [
        replace(hit, rank=index)
        for index, hit in enumerate(ordered_hits, start=1)
    ]


def fuse_retrieval_hits(
    *hits_groups: Iterable[RetrievedChunk],
    limit: int | None = None,
    rrf_k: int | None = None,
) -> list[FusedRetrievedChunk]:
    """Fuse sparse and dense results using reciprocal rank fusion."""

    smoothing_constant = rrf_k or get_settings().rrf_smoothing_constant
    fused_by_chunk: dict[str, dict[str, object]] = {}

    for hits_group in hits_groups:
        for hit in hits_group:
            bucket = fused_by_chunk.setdefault(
                hit.chunk_id,
                {
                    "hit": hit,
                    "score": 0.0,
                    "sources": [],
                },
            )
            bucket["score"] = float(bucket["score"]) + 1.0 / (
                smoothing_constant + hit.rank
            )
            sources = bucket["sources"]
            if isinstance(sources, list) and hit.source not in sources:
                sources.append(hit.source)

    fused_hits = [
        FusedRetrievedChunk(
            chunk_id=chunk_id,
            tenant_id=entry["hit"].tenant_id,
            namespace_id=entry["hit"].namespace_id,
            document_id=entry["hit"].document_id,
            chunk_index=entry["hit"].chunk_index,
            text=entry["hit"].text,
            fused_score=float(entry["score"]),
            sources=tuple(sorted(entry["sources"])),
            section_title=entry["hit"].section_title,
            section_slug=entry["hit"].section_slug,
            chunk_role=entry["hit"].chunk_role,
            starts_with_heading=entry["hit"].starts_with_heading,
            is_list_block=entry["hit"].is_list_block,
        )
        for chunk_id, entry in fused_by_chunk.items()
    ]
    fused_hits.sort(
        key=lambda hit: (
            -hit.fused_score,
            hit.chunk_index,
            hit.chunk_id,
        )
    )
    if limit is None:
        return fused_hits
    return fused_hits[:limit]


def _rerank_fused_hits_for_query(
    fused_hits: list[FusedRetrievedChunk],
    *,
    query_plan: QueryPlan,
    limit: int,
) -> list[FusedRetrievedChunk]:
    """Rerank fused hits by query answerability before the final top-k cut."""

    if not fused_hits:
        return []

    profile = query_plan.profile
    top_fused_score = max(hit.fused_score for hit in fused_hits) or 1.0

    reranked = sorted(
        fused_hits,
        key=lambda hit: (
            -(
                (hit.fused_score / top_fused_score) * 8.5
                + score_text_against_query(
                    hit.text,
                    profile=profile,
                    chunk_index=hit.chunk_index,
                )
                + _query_term_coverage_score(text=hit.text, query_plan=query_plan)
                + _section_match_bonus(
                    section_title=hit.section_title,
                    section_slug=hit.section_slug,
                    chunk_role=hit.chunk_role,
                    chunk_index=hit.chunk_index,
                    query_plan=query_plan,
                )
                + _intent_bonus_for_hit(hit=hit, query_plan=query_plan)
                + len(hit.sources) * 0.9
            ),
            hit.chunk_index,
            hit.chunk_id,
        ),
    )
    return reranked[:limit]


async def _apply_enterprise_reranker(
    fused_hits: list[FusedRetrievedChunk],
    *,
    execution_tier: ExecutionTier,
    query_text: str,
    limit: int,
) -> list[FusedRetrievedChunk]:
    """Apply the Enterprise reranker when the active tier requests it."""

    if not fused_hits:
        return [], {
            "attempted": False,
            "applied": False,
            "backend": None,
            "reason": "no_candidates",
        }
    if execution_tier is not ExecutionTier.ENTERPRISE:
        return fused_hits[:limit], {
            "attempted": False,
            "applied": False,
            "backend": None,
            "reason": "non_enterprise_tier",
        }

    settings = get_settings()
    reranker = resolve_retrieval_reranker(
        execution_tier=execution_tier,
        settings=settings,
    )
    try:
        reranker_result = await reranker.rerank(
            query_text=query_text,
            hits=fused_hits[: settings.enterprise_reranker_candidate_limit],
            limit=limit,
        )
    except RerankerError as exc:
        logger.warning(
            "enterprise_reranker_failed",
            error=str(exc),
        )
        return fused_hits[:limit], {
            "attempted": True,
            "applied": False,
            "backend": getattr(reranker, "backend_name", "unknown"),
            "error": str(exc),
        }

    logger.info(
        "enterprise_reranker_applied",
        backend=reranker_result.backend_name,
        applied=reranker_result.applied,
        top_hits=[
            {
                "chunk_id": hit.chunk_id,
                "score": round(hit.score, 4),
            }
            for hit in reranker_result.hits[:3]
        ],
    )

    hit_by_chunk_id = {hit.chunk_id: hit for hit in fused_hits}
    reranked_hits: list[FusedRetrievedChunk] = []
    seen_chunk_ids: set[str] = set()

    for reranked_hit in reranker_result.hits:
        original_hit = hit_by_chunk_id.get(reranked_hit.chunk_id)
        if original_hit is None or reranked_hit.chunk_id in seen_chunk_ids:
            continue
        reranked_hits.append(
            replace(
                original_hit,
                fused_score=reranked_hit.score,
            )
        )
        seen_chunk_ids.add(reranked_hit.chunk_id)

    for hit in fused_hits:
        if hit.chunk_id in seen_chunk_ids:
            continue
        reranked_hits.append(hit)

    final_hits = reranked_hits[:limit]
    return final_hits, {
        "attempted": True,
        "applied": reranker_result.applied,
        "backend": reranker_result.backend_name,
        "top_chunk_ids": [hit.chunk_id for hit in final_hits[:3]],
        "debug": getattr(reranker_result, "debug", {}),
    }


async def _fetch_supporting_context_hits(
    *,
    session: AsyncSession,
    tenant_id: UUID,
    namespace_id: UUID,
    query_plan: QueryPlan,
    fused_hits: list[FusedRetrievedChunk],
) -> list[FusedRetrievedChunk]:
    """Recover nearby/header chunks from already-relevant documents."""

    if not fused_hits:
        return []

    profile = query_plan.profile
    target_indexes_by_doc: dict[UUID, set[int]] = {}
    target_sections_by_doc: dict[UUID, set[str]] = {}
    doc_priority_scores: dict[UUID, float] = {}

    for hit in fused_hits[:5]:
        doc_priority_scores[hit.document_id] = max(
            doc_priority_scores.get(hit.document_id, 0.0),
            hit.fused_score,
        )
        target_indexes = target_indexes_by_doc.setdefault(hit.document_id, set())
        target_indexes.update(
            index
            for index in (hit.chunk_index - 1, hit.chunk_index, hit.chunk_index + 1)
            if index >= 0
        )
        if hit.section_slug and (
            is_collection_query(profile)
            or (
                is_field_extraction_query(profile)
                and hit.chunk_role in {"section_header", "section_body", "section_list"}
            )
        ):
            target_sections_by_doc.setdefault(hit.document_id, set()).add(hit.section_slug)

    if is_field_extraction_query(profile):
        for document_id in list(target_indexes_by_doc)[:3]:
            target_indexes_by_doc[document_id].update({0, 1})

    clauses = [
        (
            (DocumentChunkRecord.doc_id == document_id)
            & (DocumentChunkRecord.chunk_index.in_(sorted(indexes)))
        )
        for document_id, indexes in target_indexes_by_doc.items()
        if indexes
    ]
    clauses.extend(
        (
            (DocumentChunkRecord.doc_id == document_id)
            & (DocumentChunkRecord.section_slug.in_(sorted(section_slugs)))
        )
        for document_id, section_slugs in target_sections_by_doc.items()
        if section_slugs
    )
    if not clauses:
        return []

    statement = (
        select(DocumentChunkRecord)
        .where(
            DocumentChunkRecord.tenant_id == tenant_id,
            DocumentChunkRecord.namespace_id == namespace_id,
            or_(*clauses),
        )
        .order_by(DocumentChunkRecord.doc_id.asc(), DocumentChunkRecord.chunk_index.asc())
    )

    try:
        result = await session.execute(statement)
    except SQLAlchemyError as exc:
        raise RetrievalError("Supporting context retrieval failed.") from exc

    existing_chunk_ids = {hit.chunk_id for hit in fused_hits}
    supplemental_hits: list[FusedRetrievedChunk] = []
    rows = result.scalars() if hasattr(result, "scalars") else []
    for row in rows:
        row_chunk_id = getattr(row, "chunk_id")
        if row_chunk_id in existing_chunk_ids:
            continue

        row_doc_id = getattr(row, "doc_id")
        row_chunk_index = int(getattr(row, "chunk_index"))
        row_section_slug = getattr(row, "section_slug", None)
        row_section_title = getattr(row, "section_title", None)
        row_chunk_role = getattr(row, "chunk_role", "body")
        row_starts_with_heading = bool(getattr(row, "starts_with_heading", False))
        row_is_list_block = bool(getattr(row, "is_list_block", False))
        doc_score = doc_priority_scores.get(row_doc_id, 0.0)
        proximity_score = 0.72
        if row_chunk_index <= 1 and is_field_extraction_query(profile):
            proximity_score = 0.9
        if row_section_slug and row_section_slug in target_sections_by_doc.get(row_doc_id, set()):
            proximity_score = max(proximity_score, 0.88)
            if row_chunk_role in {"section_header", "section_list"}:
                proximity_score = max(proximity_score, 0.94)

        supplemental_hits.append(
            FusedRetrievedChunk(
                chunk_id=row_chunk_id,
                tenant_id=getattr(row, "tenant_id"),
                namespace_id=getattr(row, "namespace_id"),
                document_id=row_doc_id,
                chunk_index=row_chunk_index,
                text=getattr(row, "chunk_text"),
                fused_score=max(doc_score * proximity_score, 0.0001),
                sources=(
                    ("section_context",)
                    if row_section_slug
                    and row_section_slug in target_sections_by_doc.get(row_doc_id, set())
                    else ("context",)
                ),
                section_title=row_section_title,
                section_slug=row_section_slug,
                chunk_role=row_chunk_role,
                starts_with_heading=row_starts_with_heading,
                is_list_block=row_is_list_block,
            )
        )

    return supplemental_hits


async def _fetch_namespace_lead_chunks(
    *,
    session: AsyncSession,
    tenant_id: UUID,
    namespace_id: UUID,
    limit: int,
) -> list[FusedRetrievedChunk]:
    """Fetch leading chunks across the namespace for broad dataset-summary questions."""

    statement = (
        select(DocumentChunkRecord)
        .where(
            DocumentChunkRecord.tenant_id == tenant_id,
            DocumentChunkRecord.namespace_id == namespace_id,
            DocumentChunkRecord.chunk_index <= 1,
        )
        .order_by(DocumentChunkRecord.doc_id.asc(), DocumentChunkRecord.chunk_index.asc())
        .limit(limit)
    )

    try:
        result = await session.execute(statement)
    except SQLAlchemyError as exc:
        raise RetrievalError("Dataset summary retrieval failed.") from exc

    rows = result.scalars() if hasattr(result, "scalars") else []
    return [
        FusedRetrievedChunk(
            chunk_id=getattr(row, "chunk_id"),
            tenant_id=getattr(row, "tenant_id"),
            namespace_id=getattr(row, "namespace_id"),
            document_id=getattr(row, "doc_id"),
            chunk_index=int(getattr(row, "chunk_index")),
            text=getattr(row, "chunk_text"),
            fused_score=1.0 if int(getattr(row, "chunk_index")) == 0 else 0.85,
            sources=("summary_context",),
            section_title=getattr(row, "section_title", None),
            section_slug=getattr(row, "section_slug", None),
            chunk_role=getattr(row, "chunk_role", "body"),
            starts_with_heading=bool(getattr(row, "starts_with_heading", False)),
            is_list_block=bool(getattr(row, "is_list_block", False)),
        )
        for row in rows
    ]


async def retrieve_hybrid_candidates(
    *,
    session: AsyncSession,
    tenant_id: UUID,
    namespace_id: UUID,
    query_text: str,
    query_plan: QueryPlan | None = None,
    limit: int | None = None,
    execution_tier: ExecutionTier = ExecutionTier.STANDARD,
    freshness_profile: FreshnessProfile = FreshnessProfile.BALANCED,
) -> RetrievalBundle:
    """Run configured retrieval paths, then shape candidates into fused-hit form."""

    settings = get_settings()
    retrieval_mode = getattr(settings, "retrieval_mode", "hybrid")
    final_limit = limit or settings.retrieval_candidate_limit
    overfetch_limit = max(
        final_limit,
        final_limit * settings.retrieval_overfetch_factor,
    )
    plan = query_plan or build_query_plan(query_text)
    profile = plan.profile

    sparse_hits: list[RetrievedChunk] = []
    dense_hits: list[RetrievedChunk] = []
    for retrieval_query in plan.retrieval_queries:
        if retrieval_mode == "hybrid":
            sparse_hits.extend(
                await sparse_retrieve_chunks(
                    session=session,
                    tenant_id=tenant_id,
                    namespace_id=namespace_id,
                    query_text=retrieval_query,
                    limit=overfetch_limit,
                )
            )
        dense_hits.extend(
            await dense_retrieve_chunks(
                tenant_id=tenant_id,
                namespace_id=namespace_id,
                query_text=retrieval_query,
                limit=overfetch_limit,
            )
        )
    sparse_hits = _dedupe_retrieval_hits(sparse_hits)
    dense_hits = _dedupe_retrieval_hits(dense_hits)
    fused_hits = fuse_retrieval_hits(
        sparse_hits,
        dense_hits,
        rrf_k=settings.rrf_smoothing_constant,
    )
    logger.debug(
        "retrieval_results",
        query_text=query_text,
        retrieval_mode=retrieval_mode,
        sparse_count=len(sparse_hits),
        dense_count=len(dense_hits),
        retrieved_count=len(fused_hits),
        retrieved_chunks=[
            {
                "chunk_id": hit.chunk_id,
                "rank": index,
                "score": hit.fused_score,
                "text_preview": hit.text[:200],
            }
            for index, hit in enumerate(fused_hits[:5], start=1)
        ],
    )
    supporting_hits = await _fetch_supporting_context_hits(
        session=session,
        tenant_id=tenant_id,
        namespace_id=namespace_id,
        query_plan=plan,
        fused_hits=fused_hits,
    )
    if supporting_hits:
        fused_hits = list(fused_hits) + supporting_hits
    if is_dataset_summary_query(profile):
        summary_hits = await _fetch_namespace_lead_chunks(
            session=session,
            tenant_id=tenant_id,
            namespace_id=namespace_id,
            limit=max(final_limit, settings.evidence_package_limit * 2),
        )
        if profile.document_reference_rank is not None:
            target_index = profile.document_reference_rank - 1
            adjusted_summary_hits: list[FusedRetrievedChunk] = []
            for index, hit in enumerate(summary_hits):
                scale = 1.5 if index == target_index else 0.75
                adjusted_summary_hits.append(
                    replace(hit, fused_score=max(hit.fused_score * scale, 0.0001))
                )
            summary_hits = adjusted_summary_hits
        existing_chunk_ids = {hit.chunk_id for hit in fused_hits}
        fused_hits = list(fused_hits) + [
            hit for hit in summary_hits if hit.chunk_id not in existing_chunk_ids
        ]
    fused_hits = _rerank_fused_hits_for_query(
        fused_hits,
        query_plan=plan,
        limit=max(final_limit, settings.enterprise_reranker_candidate_limit),
    )
    fused_hits, freshness_debug = await _apply_enterprise_freshness_scoring(
        fused_hits,
        session=session,
        tenant_id=tenant_id,
        namespace_id=namespace_id,
        query_plan=plan,
        execution_tier=execution_tier,
        freshness_profile=freshness_profile,
    )
    fused_hits, reranker_debug = await _apply_enterprise_reranker(
        fused_hits,
        execution_tier=execution_tier,
        query_text=plan.resolved_query_text,
        limit=final_limit,
    )
    return RetrievalBundle(
        sparse_hits=sparse_hits,
        dense_hits=dense_hits,
        fused_hits=fused_hits,
        debug={
            "execution_tier": execution_tier.value,
            "retrieval_queries": list(plan.retrieval_queries),
            "freshness": freshness_debug,
            "reranker": reranker_debug,
            "top_fused_hits": [
                {
                    "chunk_id": hit.chunk_id,
                    "score": round(hit.fused_score, 4),
                }
                for hit in fused_hits[:3]
            ],
        },
    )
