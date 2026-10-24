"""Integration tests for extraction-stage ingestion processing."""

from __future__ import annotations

import asyncio
import io
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

import psycopg
from docx import Document as DocxDocument

from app.config import get_settings
from app.core.database import dispose_database
from app.core.storage import StorageError, delete_object, download_bytes, upload_bytes
from app.models import (
    DocumentStatus,
    ExecutionTier,
    FreshnessProfile,
    IngestionJobStatus,
    SensitivityLevel,
    SubscriptionPlan,
)
from app.services.extraction import derive_extracted_text_key
from app.workers.extraction import run_extraction_job


@dataclass(frozen=True)
class SeededExtractionJobData:
    """Seeded document and queued job data for extraction worker tests."""

    tenant_id: uuid.UUID
    namespace_id: uuid.UUID
    document_id: uuid.UUID
    job_id: uuid.UUID
    object_key: str


def _sync_database_url() -> str:
    """Return a sync Postgres URL suitable for psycopg."""

    return get_settings().alembic_database_url.replace("+psycopg", "")


def _seed_extraction_fixture(
    *,
    mime_type: str,
    filename: str,
    payload: bytes,
    title: str,
) -> SeededExtractionJobData:
    """Seed a queued ingestion job and upload its source object."""

    tenant_id = uuid.uuid4()
    namespace_id = uuid.uuid4()
    document_id = uuid.uuid4()
    job_id = uuid.uuid4()
    object_key = (
        f"tenants/{tenant_id}/namespaces/{namespace_id}/documents/{document_id}/source/{filename}"
    )
    asyncio.run(
        upload_bytes(
            object_key,
            payload,
            content_type=mime_type,
            metadata={
                "tenant_id": str(tenant_id),
                "namespace_id": str(namespace_id),
                "document_id": str(document_id),
            },
        )
    )

    with psycopg.connect(_sync_database_url()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                insert into tenants (
                    tenant_id,
                    name,
                    subscription_plan,
                    max_execution_tier,
                    default_policy,
                    retention_days
                )
                values (%s, %s, %s, %s, %s::jsonb, %s)
                """,
                (
                    tenant_id,
                    f"tenant-{tenant_id}",
                    SubscriptionPlan.PRO.value,
                    ExecutionTier.ENTERPRISE.value,
                    json.dumps({"tier": "standard"}),
                    365,
                ),
            )
            cursor.execute(
                """
                insert into namespaces (
                    namespace_id,
                    tenant_id,
                    name,
                    domain,
                    sensitivity_level,
                    freshness_profile,
                    min_execution_tier,
                    allow_web_fallback,
                    allow_internal_model_retrieval
                )
                values (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    namespace_id,
                    tenant_id,
                    "primary",
                    "general",
                    SensitivityLevel.INTERNAL.value,
                    FreshnessProfile.BALANCED.value,
                    ExecutionTier.STANDARD.value,
                    False,
                    False,
                ),
            )
            cursor.execute(
                """
                insert into documents (
                    doc_id,
                    tenant_id,
                    namespace_id,
                    object_key,
                    source_uri,
                    mime_type,
                    title,
                    checksum,
                    file_size_bytes,
                    status
                )
                values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    document_id,
                    tenant_id,
                    namespace_id,
                    object_key,
                    f"s3://grounded-documents/{object_key}",
                    mime_type,
                    title,
                    uuid.uuid4().hex + uuid.uuid4().hex,
                    len(payload),
                    DocumentStatus.UPLOADED.value,
                ),
            )
            cursor.execute(
                """
                insert into ingestion_jobs (
                    job_id,
                    tenant_id,
                    doc_id,
                    status
                )
                values (%s, %s, %s, %s)
                """,
                (
                    job_id,
                    tenant_id,
                    document_id,
                    IngestionJobStatus.QUEUED.value,
                ),
            )

    return SeededExtractionJobData(
        tenant_id=tenant_id,
        namespace_id=namespace_id,
        document_id=document_id,
        job_id=job_id,
        object_key=object_key,
    )


def _cleanup_seeded_extraction_fixture(seed: SeededExtractionJobData) -> None:
    """Delete seeded DB rows and storage artifacts created by extraction tests."""

    extracted_key = derive_extracted_text_key(seed.object_key)
    for key in (seed.object_key, extracted_key):
        try:
            asyncio.run(delete_object(key))
        except StorageError:
            pass

    with psycopg.connect(_sync_database_url()) as connection:
        with connection.cursor() as cursor:
            cursor.execute("delete from ingestion_jobs where tenant_id = %s", (seed.tenant_id,))
            cursor.execute("delete from documents where tenant_id = %s", (seed.tenant_id,))
            cursor.execute("delete from namespaces where tenant_id = %s", (seed.tenant_id,))
            cursor.execute("delete from tenants where tenant_id = %s", (seed.tenant_id,))
    asyncio.run(dispose_database())


def _fetch_document_and_job(seed: SeededExtractionJobData) -> tuple[tuple, tuple]:
    """Return the document and ingestion job rows after extraction runs."""

    with psycopg.connect(_sync_database_url()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                select title, published_at, status
                from documents
                where doc_id = %s and tenant_id = %s
                """,
                (seed.document_id, seed.tenant_id),
            )
            document_row = cursor.fetchone()
            cursor.execute(
                """
                select status, attempt_count, error_code, error_detail
                from ingestion_jobs
                where job_id = %s and tenant_id = %s
                """,
                (seed.job_id, seed.tenant_id),
            )
            job_row = cursor.fetchone()
    assert document_row is not None
    assert job_row is not None
    return document_row, job_row


def test_run_extraction_job_extracts_txt_artifact() -> None:
    """TXT extraction should create a normalized artifact and keep processing active."""

    seed = _seed_extraction_fixture(
        mime_type="text/plain",
        filename="manual.txt",
        payload=b"line one\r\nline two\r\n",
        title="manual",
    )
    try:
        result = asyncio.run(run_extraction_job(seed.job_id))

        assert result.run.status is IngestionJobStatus.RUNNING
        assert result.run.document_status is DocumentStatus.PROCESSING
        assert result.extracted_document.artifact_key.endswith("/artifacts/extracted/text.txt")

        extracted_payload = asyncio.run(download_bytes(result.extracted_document.artifact_key))
        assert extracted_payload.decode("utf-8") == "line one\nline two"

        document_row, job_row = _fetch_document_and_job(seed)
        assert document_row == ("manual", None, "processing")
        assert job_row == ("running", 1, None, None)
    finally:
        _cleanup_seeded_extraction_fixture(seed)


def test_run_extraction_job_enriches_docx_metadata() -> None:
    """DOCX extraction should enrich stored title and publication metadata."""

    docx = DocxDocument()
    docx.add_paragraph("Docx body text")
    docx.core_properties.title = "DOCX Metadata Title"
    docx.core_properties.created = datetime(2024, 5, 2, 9, 30, tzinfo=UTC)
    buffer = io.BytesIO()
    docx.save(buffer)

    seed = _seed_extraction_fixture(
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename="docx-title.docx",
        payload=buffer.getvalue(),
        title="docx-title",
    )
    try:
        result = asyncio.run(run_extraction_job(seed.job_id))

        assert result.extracted_document.title == "DOCX Metadata Title"
        assert result.extracted_document.published_at == datetime(
            2024,
            5,
            2,
            9,
            30,
            tzinfo=UTC,
        )

        extracted_payload = asyncio.run(download_bytes(result.extracted_document.artifact_key))
        assert extracted_payload.decode("utf-8") == "Docx body text"

        document_row, job_row = _fetch_document_and_job(seed)
        assert document_row == (
            "DOCX Metadata Title",
            datetime(2024, 5, 2, 9, 30, tzinfo=UTC),
            "processing",
        )
        assert job_row == ("running", 1, None, None)
    finally:
        _cleanup_seeded_extraction_fixture(seed)


def test_run_extraction_job_fails_empty_text_documents() -> None:
    """Empty extracted content should fail the ingestion job cleanly."""

    seed = _seed_extraction_fixture(
        mime_type="text/plain",
        filename="empty.txt",
        payload=b"   \r\n\t",
        title="empty",
    )
    try:
        result = asyncio.run(run_extraction_job(seed.job_id))

        assert result.run.status is IngestionJobStatus.FAILED
        assert result.run.document_status is DocumentStatus.FAILED

        document_row, job_row = _fetch_document_and_job(seed)
        assert document_row == ("empty", None, "failed")
        assert job_row == (
            "failed",
            1,
            "EMPTY_EXTRACTION",
            "Document extraction produced no usable text.",
        )
    finally:
        _cleanup_seeded_extraction_fixture(seed)
