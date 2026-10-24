"""Document persistence model."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.enums import DocumentStatus, sqlalchemy_enum

if TYPE_CHECKING:
    from app.models.ingestion_job import IngestionJob
    from app.models.namespace import Namespace
    from app.models.tenant import Tenant


class Document(Base):
    """Stored source document owned by a tenant namespace."""

    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "namespace_id",
            "checksum",
            name="uq_documents_tenant_namespace_checksum",
        ),
        UniqueConstraint("tenant_id", "doc_id", name="uq_documents_tenant_doc_id"),
        ForeignKeyConstraint(
            ["tenant_id", "namespace_id"],
            ["namespaces.tenant_id", "namespaces.namespace_id"],
            name="fk_documents_tenant_namespace",
        ),
        CheckConstraint(
            "file_size_bytes >= 0",
            name="ck_documents_file_size_non_negative",
        ),
        Index("ix_documents_tenant_id", "tenant_id"),
        Index("ix_documents_namespace_id", "namespace_id"),
        Index("ix_documents_status", "status"),
    )

    doc_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tenants.tenant_id"),
        nullable=False,
    )
    namespace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
    )
    object_key: Mapped[str] = mapped_column(Text, nullable=False)
    source_uri: Mapped[str | None] = mapped_column(Text)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    title: Mapped[str | None] = mapped_column(String(500))
    checksum: Mapped[str] = mapped_column(String(64), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    version: Mapped[str | None] = mapped_column(String(100))
    status: Mapped[DocumentStatus] = mapped_column(
        sqlalchemy_enum(DocumentStatus, name="document_status_enum"),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    tenant: Mapped["Tenant"] = relationship(
        back_populates="documents",
        foreign_keys=[tenant_id],
        overlaps="documents,namespace",
    )
    namespace: Mapped["Namespace"] = relationship(
        back_populates="documents",
        primaryjoin=(
            "and_("
            "Document.tenant_id == Namespace.tenant_id, "
            "Document.namespace_id == Namespace.namespace_id"
            ")"
        ),
        foreign_keys="[Document.tenant_id, Document.namespace_id]",
        overlaps="tenant,documents",
    )
    ingestion_jobs: Mapped[list["IngestionJob"]] = relationship(
        back_populates="document",
        primaryjoin=(
            "and_("
            "Document.tenant_id == IngestionJob.tenant_id, "
            "Document.doc_id == IngestionJob.doc_id"
            ")"
        ),
        foreign_keys="[IngestionJob.tenant_id, IngestionJob.doc_id]",
        overlaps="tenant,document,ingestion_jobs",
    )
