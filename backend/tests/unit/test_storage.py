"""Unit tests for object storage helpers."""

from __future__ import annotations

import asyncio
from collections import defaultdict

from botocore.exceptions import ClientError

from app.config import get_settings
from app.core.storage import (
    StoredObject,
    delete_object,
    download_bytes,
    ensure_bucket_exists,
    ensure_storage_ready,
    get_storage_client,
    reset_storage_client,
    upload_bytes,
)


class FakeBody:
    """Simple streaming body stand-in for boto3 responses."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return self.payload


class FakeS3Client:
    """Minimal S3 client stub for storage unit tests."""

    def __init__(self, *, bucket_exists: bool = True) -> None:
        self.bucket_exists = bucket_exists
        self.calls: dict[str, list[dict[str, object]]] = defaultdict(list)
        self.objects: dict[str, bytes] = {}

    def head_bucket(self, **kwargs: object) -> None:
        self.calls["head_bucket"].append(kwargs)
        if not self.bucket_exists:
            raise ClientError(
                {
                    "Error": {"Code": "NoSuchBucket"},
                    "ResponseMetadata": {"HTTPStatusCode": 404},
                },
                "HeadBucket",
            )

    def create_bucket(self, **kwargs: object) -> None:
        self.calls["create_bucket"].append(kwargs)
        self.bucket_exists = True

    def put_object(self, **kwargs: object) -> dict[str, str]:
        self.calls["put_object"].append(kwargs)
        body = kwargs["Body"]
        assert isinstance(body, bytes)
        self.objects[str(kwargs["Key"])] = body
        return {"ETag": '"etag-123"'}

    def get_object(self, **kwargs: object) -> dict[str, FakeBody]:
        self.calls["get_object"].append(kwargs)
        key = str(kwargs["Key"])
        return {"Body": FakeBody(self.objects[key])}

    def delete_object(self, **kwargs: object) -> None:
        self.calls["delete_object"].append(kwargs)
        key = str(kwargs["Key"])
        self.objects.pop(key, None)


def test_get_storage_client_uses_configured_settings(monkeypatch) -> None:
    """The storage client should honor the configured endpoint and credentials."""

    captured: dict[str, object] = {}

    def fake_client(service_name: str, **kwargs: object) -> object:
        captured["service_name"] = service_name
        captured.update(kwargs)
        return object()

    monkeypatch.setenv("S3_ENDPOINT_URL", "http://localhost:9000")
    monkeypatch.setenv("S3_BUCKET", "grounded-documents")
    monkeypatch.setenv("S3_ACCESS_KEY", "minioadmin")
    monkeypatch.setenv("S3_SECRET_KEY", "minioadmin")
    monkeypatch.setenv("S3_SECURE", "false")
    monkeypatch.setattr("app.core.storage.boto3.client", fake_client)
    get_settings.cache_clear()
    reset_storage_client()

    client = get_storage_client()

    assert client is not None
    assert captured["service_name"] == "s3"
    assert captured["endpoint_url"] == "http://localhost:9000/"
    assert captured["aws_access_key_id"] == "minioadmin"
    assert captured["aws_secret_access_key"] == "minioadmin"
    assert captured["use_ssl"] is False

    reset_storage_client()
    get_settings.cache_clear()


def test_ensure_bucket_exists_creates_missing_bucket(monkeypatch) -> None:
    """Missing buckets should be created automatically for the configured backend."""

    client = FakeS3Client(bucket_exists=False)
    monkeypatch.setenv("S3_BUCKET", "grounded-documents")
    monkeypatch.setattr("app.core.storage.get_storage_client", lambda: client)
    get_settings.cache_clear()

    ensure_bucket_exists()

    assert client.calls["head_bucket"] == [{"Bucket": "grounded-documents"}]
    assert client.calls["create_bucket"] == [{"Bucket": "grounded-documents"}]
    get_settings.cache_clear()


def test_upload_download_and_delete_object(monkeypatch) -> None:
    """Storage helpers should write, read, and delete payloads against the bucket."""

    client = FakeS3Client()
    monkeypatch.setenv("S3_BUCKET", "grounded-documents")
    monkeypatch.setattr("app.core.storage.get_storage_client", lambda: client)
    get_settings.cache_clear()

    stored_object = asyncio.run(
        upload_bytes(
            "docs/sample.txt",
            b"hello grounded",
            content_type="text/plain",
            metadata={"tenant": "demo"},
        )
    )
    downloaded = asyncio.run(download_bytes("docs/sample.txt"))
    asyncio.run(delete_object("docs/sample.txt"))

    assert stored_object == StoredObject(
        bucket="grounded-documents",
        key="docs/sample.txt",
        etag="etag-123",
        size=14,
    )
    assert downloaded == b"hello grounded"
    assert client.calls["put_object"][0]["Metadata"] == {"tenant": "demo"}
    assert client.calls["delete_object"] == [
        {"Bucket": "grounded-documents", "Key": "docs/sample.txt"}
    ]
    get_settings.cache_clear()


def test_ensure_storage_ready_returns_false_on_storage_error(monkeypatch) -> None:
    """Storage readiness should fail closed when the client cannot verify the bucket."""

    def fail() -> None:
        raise ClientError(
            {
                "Error": {"Code": "AccessDenied"},
                "ResponseMetadata": {"HTTPStatusCode": 403},
            },
            "HeadBucket",
        )

    monkeypatch.setattr("app.core.storage.ensure_bucket_exists", fail)

    assert asyncio.run(ensure_storage_ready()) is False
