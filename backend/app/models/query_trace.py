"""Query trace persistence model."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.enums import ExecutionTier, UserFacingMode, sqlalchemy_enum

if TYPE_CHECKING:
    from app.models.namespace import Namespace
    from app.models.tenant import Tenant


class QueryTrace(Base):
    """Trace and provenance record for a grounded query."""

    __tablename__ = "query_traces"
    __table_args__ = (
        CheckConstraint(
            "overall_confidence >= 0 AND overall_confidence <= 1",
            name="ck_query_traces_overall_confidence",
        ),
        CheckConstraint(
            "total_latency_ms >= 0",
            name="ck_query_traces_total_latency_non_negative",
        ),
        ForeignKeyConstraint(
            ["tenant_id", "namespace_id"],
            ["namespaces.tenant_id", "namespaces.namespace_id"],
            name="fk_query_traces_tenant_namespace",
        ),
        ForeignKeyConstraint(
            ["tenant_id", "agent_id"],
            ["agents.tenant_id", "agents.agent_id"],
            name="fk_query_traces_tenant_agent",
        ),
        ForeignKeyConstraint(
            ["tenant_id", "conversation_id"],
            ["conversations.tenant_id", "conversations.conversation_id"],
            name="fk_query_traces_tenant_conversation",
        ),
        Index("ix_query_traces_tenant_id", "tenant_id"),
        Index("ix_query_traces_namespace_id", "namespace_id"),
        Index("ix_query_traces_agent_id", "agent_id"),
        Index("ix_query_traces_conversation_id", "conversation_id"),
        Index("ix_query_traces_created_at", "created_at"),
        Index("ix_query_traces_tenant_created_at", "tenant_id", "created_at"),
    )

    trace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tenants.tenant_id"),
        nullable=False,
    )
    namespace_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )
    agent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )
    selected_mode: Mapped[UserFacingMode | None] = mapped_column(
        sqlalchemy_enum(UserFacingMode, name="user_facing_mode_enum"),
        nullable=True,
    )
    requested_tier: Mapped[ExecutionTier | None] = mapped_column(
        sqlalchemy_enum(ExecutionTier, name="execution_tier_enum"),
        nullable=True,
    )
    router_recommendation: Mapped[ExecutionTier] = mapped_column(
        sqlalchemy_enum(ExecutionTier, name="execution_tier_enum"),
        nullable=False,
    )
    effective_tier: Mapped[ExecutionTier] = mapped_column(
        sqlalchemy_enum(ExecutionTier, name="execution_tier_enum"),
        nullable=False,
    )
    routing_reason: Mapped[str] = mapped_column(Text, nullable=False)
    query_redacted: Mapped[str] = mapped_column(Text, nullable=False)
    query_ciphertext: Mapped[bytes | None] = mapped_column(LargeBinary)
    retrieved_chunk_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    selected_evidence_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    generator_provider: Mapped[str] = mapped_column(String(100), nullable=False)
    verifier_result: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    final_answer_redacted: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    overall_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    degraded_reasons: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    stage_latencies_ms: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    total_latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    token_usage: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    feedback_rating: Mapped[str | None] = mapped_column(String(20), nullable=True)
    feedback_reasons: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    feedback_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    tenant: Mapped["Tenant"] = relationship(
        back_populates="query_traces",
        foreign_keys=[tenant_id],
        overlaps="namespace,query_traces",
    )
    namespace: Mapped["Namespace | None"] = relationship(
        back_populates="query_traces",
        primaryjoin=(
            "and_("
            "QueryTrace.tenant_id == Namespace.tenant_id, "
            "QueryTrace.namespace_id == Namespace.namespace_id"
            ")"
        ),
        foreign_keys="[QueryTrace.tenant_id, QueryTrace.namespace_id]",
        overlaps="tenant,query_traces",
    )
