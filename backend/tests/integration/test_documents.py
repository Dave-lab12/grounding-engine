"""Integration tests for document upload and ingestion job routes."""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from dataclasses import dataclass

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.core.database import dispose_database
from app.core.security import hash_api_key
from app.core.storage import StorageError, delete_object, download_bytes
from app.main import create_app
from app.models import ExecutionTier, FreshnessProfile, SensitivityLevel, SubscriptionPlan


@dataclass(frozen=True)
class SeededUploadData:
    """Seeded tenant and namespace data for upload integration tests."""

    tenant_id: uuid.UUID
    namespace_id: uuid.UUID
    foreign_namespace_id: uuid.UUID
    raw_api_key: str


@pytest.fixture()
def upload_auth_env(monkeypatch) -> str:
    """Configure a deterministic API key salt for upload route tests."""

    salt = "integration-test-api-key-salt"
    monkeypatch.setenv("API_KEY_SALT", salt)
    get_settings.cache_clear()
    yield salt
    get_settings.cache_clear()


@pytest.fixture()
def autorun_disabled(monkeypatch) -> None:
    """Disable background ingestion so upload-only tests stay deterministic."""

    monkeypatch.setenv("INGESTION_AUTORUN_ENABLED", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture()
def upload_client(upload_auth_env: str, autorun_disabled: None) -> TestClient:
    """Create a test client after upload-related settings are configured."""

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    asyncio.run(dispose_database())


@pytest.fixture()
def autorun_upload_client(upload_auth_env: str, monkeypatch) -> TestClient:
    """Create a client with automatic ingestion orchestration enabled."""

    monkeypatch.setenv("INGESTION_AUTORUN_ENABLED", "true")
    get_settings.cache_clear()
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    asyncio.run(dispose_database())
    get_settings.cache_clear()


def _sync_database_url() -> str:
    """Return a sync Postgres URL suitable for psycopg."""

    return get_settings().alembic_database_url.replace("+psycopg", "")


@pytest.fixture()
def seeded_upload_data(upload_auth_env: str) -> SeededUploadData:
    """Insert the tenants, namespace, and API key needed by upload tests."""

    tenant_id = uuid.uuid4()
    tenant_name = f"tenant-{tenant_id}"
    namespace_id = uuid.uuid4()
    foreign_tenant_id = uuid.uuid4()
    foreign_namespace_id = uuid.uuid4()
    raw_api_key = f"grd_upload_{uuid.uuid4().hex}"

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
                    tenant_name,
                    SubscriptionPlan.PRO.value,
                    ExecutionTier.ENTERPRISE.value,
                    json.dumps({"tier": "standard"}),
                    365,
                ),
            )
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
                    foreign_tenant_id,
                    f"tenant-{foreign_tenant_id}",
                    SubscriptionPlan.FREE.value,
                    ExecutionTier.STANDARD.value,
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
                    foreign_namespace_id,
                    foreign_tenant_id,
                    "foreign",
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
                insert into api_keys (
                    key_id,
                    tenant_id,
                    key_hash,
                    label
                )
                values (%s, %s, %s, %s)
                """,
                (
                    uuid.uuid4(),
                    tenant_id,
                    hash_api_key(raw_api_key, upload_auth_env),
                    "integration-upload",
                ),
            )

    yield SeededUploadData(
        tenant_id=tenant_id,
        namespace_id=namespace_id,
        foreign_namespace_id=foreign_namespace_id,
        raw_api_key=raw_api_key,
    )

    with psycopg.connect(_sync_database_url()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "select object_key from documents where tenant_id in (%s, %s)",
                (tenant_id, foreign_tenant_id),
            )
            object_keys = [row[0] for row in cursor.fetchall()]

    for object_key in object_keys:
        try:
            asyncio.run(delete_object(object_key))
        except StorageError:
            pass

    with psycopg.connect(_sync_database_url()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "delete from query_traces where tenant_id in (%s, %s)",
                (tenant_id, foreign_tenant_id),
            )
            cursor.execute(
                "delete from ingestion_jobs where tenant_id in (%s, %s)",
                (tenant_id, foreign_tenant_id),
            )
            cursor.execute(
                "delete from document_chunks where tenant_id in (%s, %s)",
                (tenant_id, foreign_tenant_id),
            )
            cursor.execute(
                "delete from documents where tenant_id in (%s, %s)",
                (tenant_id, foreign_tenant_id),
            )
            cursor.execute(
                "delete from api_keys where tenant_id in (%s, %s)",
                (tenant_id, foreign_tenant_id),
            )
            cursor.execute(
                "delete from namespaces where tenant_id in (%s, %s)",
                (tenant_id, foreign_tenant_id),
            )
            cursor.execute(
                "delete from tenants where tenant_id in (%s, %s)",
                (tenant_id, foreign_tenant_id),
            )


def test_document_upload_requires_api_key(
    upload_client: TestClient,
    seeded_upload_data: SeededUploadData,
) -> None:
    """Upload route should reject unauthenticated callers."""

    response = upload_client.post(
        "/v1/documents/upload",
        data={"namespace_id": str(seeded_upload_data.namespace_id)},
        files={"file": ("note.txt", b"hello grounded", "text/plain")},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "API key is required."}


def test_document_upload_rejects_foreign_namespace(
    upload_client: TestClient,
    seeded_upload_data: SeededUploadData,
) -> None:
    """Uploads should not be allowed into namespaces owned by another tenant."""

    response = upload_client.post(
        "/v1/documents/upload",
        headers={"X-API-Key": seeded_upload_data.raw_api_key},
        data={"namespace_id": str(seeded_upload_data.foreign_namespace_id)},
        files={"file": ("note.txt", b"hello grounded", "text/plain")},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Namespace not found for tenant."}


def test_document_upload_rejects_unsupported_file_type(
    upload_client: TestClient,
    seeded_upload_data: SeededUploadData,
) -> None:
    """Uploads should reject file types outside the initial Phase 1 allowlist."""

    response = upload_client.post(
        "/v1/documents/upload",
        headers={"X-API-Key": seeded_upload_data.raw_api_key},
        data={"namespace_id": str(seeded_upload_data.namespace_id)},
        files={"file": ("note.csv", b"a,b,c", "text/csv")},
    )

    assert response.status_code == 415
    assert response.json() == {
        "detail": "Unsupported file type. Allowed types: .docx, .pdf, .txt."
    }


def test_document_upload_creates_document_job_and_storage_object(
    upload_client: TestClient,
    seeded_upload_data: SeededUploadData,
) -> None:
    """Successful uploads should persist the document, job, and source object."""

    payload = b"hello grounded"
    checksum = hashlib.sha256(payload).hexdigest()

    response = upload_client.post(
        "/v1/documents/upload",
        headers={"X-API-Key": seeded_upload_data.raw_api_key},
        data={
            "namespace_id": str(seeded_upload_data.namespace_id),
            "title": "Operations Manual",
        },
        files={"file": ("manual.txt", payload, "text/plain")},
    )

    assert response.status_code == 201
    assert response.json()["already_exists"] is False
    assert response.json()["document_status"] == "uploaded"
    assert response.json()["job_status"] == "queued"
    assert response.json()["namespace_id"] == str(seeded_upload_data.namespace_id)
    assert response.json()["filename"] == "manual.txt"
    assert response.json()["title"] == "Operations Manual"
    assert response.json()["mime_type"] == "text/plain"
    assert response.json()["file_size_bytes"] == len(payload)

    document_id = response.json()["document_id"]
    job_id = response.json()["job_id"]

    with psycopg.connect(_sync_database_url()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                select title, mime_type, checksum, file_size_bytes, status, object_key
                from documents
                where doc_id = %s and tenant_id = %s
                """,
                (document_id, seeded_upload_data.tenant_id),
            )
            document_row = cursor.fetchone()
            cursor.execute(
                """
                select status, attempt_count, doc_id
                from ingestion_jobs
                where job_id = %s and tenant_id = %s
                """,
                (job_id, seeded_upload_data.tenant_id),
            )
            job_row = cursor.fetchone()

    assert document_row is not None
    assert document_row[:5] == (
        "Operations Manual",
        "text/plain",
        checksum,
        len(payload),
        "uploaded",
    )
    assert document_row[5].startswith(
        (
            f"tenants/{seeded_upload_data.tenant_id}/"
            f"namespaces/{seeded_upload_data.namespace_id}/documents/{document_id}/"
        )
    )
    assert document_row[5].endswith("/manual.txt")
    assert job_row == ("queued", 0, uuid.UUID(document_id))

    stored_payload = asyncio.run(download_bytes(document_row[5]))
    assert stored_payload == payload

    status_response = upload_client.get(
        f"/v1/ingestion-jobs/{job_id}",
        headers={"X-API-Key": seeded_upload_data.raw_api_key},
    )

    assert status_response.status_code == 200
    assert status_response.json() == {
        "job_id": job_id,
        "document_id": document_id,
        "status": "queued",
        "attempt_count": 0,
        "error_code": None,
        "error_detail": None,
        "started_at": None,
        "completed_at": None,
        "created_at": status_response.json()["created_at"],
    }


def test_document_upload_is_idempotent_for_duplicate_content_in_same_namespace(
    upload_client: TestClient,
    seeded_upload_data: SeededUploadData,
) -> None:
    """Uploading the same file bytes twice should reuse the existing document and job."""

    payload = b"duplicate-safe handbook"

    first_response = upload_client.post(
        "/v1/documents/upload",
        headers={"X-API-Key": seeded_upload_data.raw_api_key},
        data={"namespace_id": str(seeded_upload_data.namespace_id)},
        files={"file": ("handbook.txt", payload, "text/plain")},
    )

    assert first_response.status_code == 201
    assert first_response.json()["already_exists"] is False

    duplicate_response = upload_client.post(
        "/v1/documents/upload",
        headers={"X-API-Key": seeded_upload_data.raw_api_key},
        data={"namespace_id": str(seeded_upload_data.namespace_id)},
        files={"file": ("handbook-copy.txt", payload, "text/plain")},
    )

    assert duplicate_response.status_code == 200
    duplicate_payload = duplicate_response.json()
    assert duplicate_payload["already_exists"] is True
    assert duplicate_payload["document_id"] == first_response.json()["document_id"]
    assert duplicate_payload["job_id"] == first_response.json()["job_id"]
    assert duplicate_payload["document_status"] == "uploaded"
    assert duplicate_payload["job_status"] == "queued"

    with psycopg.connect(_sync_database_url()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                select count(*)
                from documents
                where tenant_id = %s and namespace_id = %s
                """,
                (seeded_upload_data.tenant_id, seeded_upload_data.namespace_id),
            )
            document_count = cursor.fetchone()
            cursor.execute(
                """
                select count(*)
                from ingestion_jobs
                where tenant_id = %s
                """,
                (seeded_upload_data.tenant_id,),
            )
            job_count = cursor.fetchone()

    assert document_count == (1,)
    assert job_count == (1,)


def test_document_upload_autoruns_pipeline_and_query_returns_grounded_answer(
    autorun_upload_client: TestClient,
    seeded_upload_data: SeededUploadData,
) -> None:
    """Automatic ingestion should index the upload and enable grounded querying."""

    payload = (
        b"Maintenance window: Friday at 22:00 UTC. "
        b"Escalation contact: Platform Team."
    )

    upload_response = autorun_upload_client.post(
        "/v1/documents/upload",
        headers={"X-API-Key": seeded_upload_data.raw_api_key},
        data={
            "namespace_id": str(seeded_upload_data.namespace_id),
            "title": "Operations Manual",
        },
        files={"file": ("manual.txt", payload, "text/plain")},
    )

    assert upload_response.status_code == 201
    document_id = upload_response.json()["document_id"]
    job_id = upload_response.json()["job_id"]

    status_response = autorun_upload_client.get(
        f"/v1/ingestion-jobs/{job_id}",
        headers={"X-API-Key": seeded_upload_data.raw_api_key},
    )

    assert status_response.status_code == 200
    assert status_response.json()["status"] == "queued"
    assert status_response.json()["attempt_count"] == 0
    assert status_response.json()["error_code"] is None

    with psycopg.connect(_sync_database_url()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                select status
                from documents
                where tenant_id = %s and doc_id = %s
                """,
                (seeded_upload_data.tenant_id, document_id),
            )
            document_row = cursor.fetchone()
            cursor.execute(
                """
                select count(*)
                from document_chunks
                where tenant_id = %s and doc_id = %s
                """,
                (seeded_upload_data.tenant_id, document_id),
            )
            assert cursor.fetchone() is not None

    assert document_row == ("uploaded",)
