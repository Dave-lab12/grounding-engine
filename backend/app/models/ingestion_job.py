"""Ingestion job persistence model."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.enums import IngestionJobStatus, sqlalchemy_enum

if TYPE_CHECKING:
    from app.models.document import Document
    from app.models.tenant import Tenant


class IngestionJob(Base):
    """Tracked ingestion execution for a document."""

    __tablename__ = "ingestion_jobs"
    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "doc_id"],
            ["documents.tenant_id", "documents.doc_id"],
            name="fk_ingestion_jobs_tenant_document",
        ),
        CheckConstraint("attempt_count >= 0", name="ck_ingestion_jobs_attempt_count"),
        Index("ix_ingestion_jobs_tenant_id", "tenant_id"),
        Index("ix_ingestion_jobs_doc_id", "doc_id"),
        Index("ix_ingestion_jobs_status", "status"),
    )

    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tenants.tenant_id"),
        nullable=False,
    )
    doc_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
    )
    status: Mapped[IngestionJobStatus] = mapped_column(
        sqlalchemy_enum(IngestionJobStatus, name="ingestion_job_status_enum"),
        nullable=False,
    )
    attempt_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("0"),
    )
    error_code: Mapped[str | None] = mapped_column(String(100))
    error_detail: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    tenant: Mapped["Tenant"] = relationship(
        back_populates="ingestion_jobs",
        foreign_keys=[tenant_id],
        overlaps="document,ingestion_jobs",
    )
    document: Mapped["Document"] = relationship(
        back_populates="ingestion_jobs",
        primaryjoin=(
            "and_("
            "IngestionJob.tenant_id == Document.tenant_id, "
            "IngestionJob.doc_id == Document.doc_id"
            ")"
        ),
        foreign_keys="[IngestionJob.tenant_id, IngestionJob.doc_id]",
        overlaps="tenant,ingestion_jobs,document",
    )
