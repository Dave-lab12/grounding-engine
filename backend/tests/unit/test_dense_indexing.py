"""Unit tests for dense indexing helpers."""

from __future__ import annotations

import json
import uuid

import pytest

from app.core.embeddings import build_dense_embedding, embed_texts
from app.core.gemini_embeddings import GeminiEmbeddingError
from app.core.openai_embeddings import OpenAICompatibleEmbeddingError
from app.pipeline.contracts import ChunkManifest
from app.services.dense_indexing import (
    _dense_point_id,
    _embedding_text_for_chunk,
    dense_index_document,
)
from app.services.ingestion import IngestionJobContext, IngestionProcessorError


def test_build_dense_embedding_is_deterministic() -> None:
    """The same text should always produce the same dense vector."""

    vector_one = build_dense_embedding("alpha beta gamma", dimensions=16)
    vector_two = build_dense_embedding("alpha beta gamma", dimensions=16)

    assert vector_one == vector_two
    assert len(vector_one) == 16


@pytest.mark.asyncio()
async def test_embed_texts_falls_back_from_openai_backend(monkeypatch) -> None:
    """Provider-backed embeddings should fall back to the local deterministic backend."""

    monkeypatch.setenv("EMBEDDING_BACKEND", "openai_compatible_v1")
    monkeypatch.setenv("DENSE_EMBEDDING_DIMENSIONS", "8")
    monkeypatch.setenv("EMBEDDING_PROVIDER_FALLBACK_ENABLED", "true")

    from app.config import get_settings

    get_settings.cache_clear()
    try:
        async def fake_embed_texts_openai_compatible(texts: list[str], *, purpose: str = "generic"):
            del texts
            del purpose
            raise OpenAICompatibleEmbeddingError("provider unavailable")

        monkeypatch.setattr(
            "app.core.openai_embeddings.embed_texts_openai_compatible",
            fake_embed_texts_openai_compatible,
        )

        embeddings = await embed_texts(["alpha beta"])
    finally:
        get_settings.cache_clear()

    assert len(embeddings) == 1
    assert embeddings[0].text == "alpha beta"
    assert len(embeddings[0].vector) == 8


@pytest.mark.asyncio()
async def test_embed_texts_falls_back_from_gemini_backend(monkeypatch) -> None:
    """Gemini-backed embeddings should fall back to the local deterministic backend."""

    monkeypatch.setenv("EMBEDDING_BACKEND", "gemini_v1")
    monkeypatch.setenv("DENSE_EMBEDDING_DIMENSIONS", "8")
    monkeypatch.setenv("EMBEDDING_PROVIDER_FALLBACK_ENABLED", "true")

    from app.config import get_settings

    get_settings.cache_clear()
    try:
        async def fake_embed_texts_gemini(texts: list[str], *, purpose: str = "generic"):
            del texts
            del purpose
            raise GeminiEmbeddingError("provider unavailable")

        monkeypatch.setattr(
            "app.core.gemini_embeddings.embed_texts_gemini",
            fake_embed_texts_gemini,
        )

        embeddings = await embed_texts(["alpha beta"])
    finally:
        get_settings.cache_clear()

    assert len(embeddings) == 1
    assert embeddings[0].text == "alpha beta"
    assert len(embeddings[0].vector) == 8


def test_dense_point_id_is_stable_uuid() -> None:
    """Dense point ids should be deterministic and Qdrant-compatible."""

    point_id = _dense_point_id("chunk-1")

    assert point_id == _dense_point_id("chunk-1")
    assert uuid.UUID(point_id)


def test_embedding_text_for_chunk_includes_light_structural_context() -> None:
    """Document embeddings should include useful title/section context without changing stored text."""

    context = IngestionJobContext(
        job_id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        namespace_id=uuid.uuid4(),
        object_key="tenants/t1/namespaces/n1/documents/d1/source/manual.txt",
        mime_type="text/plain",
        title="Employee Retention Policy",
        source_uri=None,
        attempt_count=1,
    )
    manifest = ChunkManifest.from_payload(
        {
            "document_id": str(context.document_id),
            "source_artifact_key": "artifact.txt",
            "chunking_strategy": "structure_aware_v1",
            "chunks": [
                {
                    "chunk_id": "chunk-1",
                    "chunk_index": 0,
                    "text": "Employee records must be retained for 7 years.",
                    "token_count": 8,
                    "character_count": 46,
                    "start_token": 0,
                    "end_token": 7,
                    "section_title": "RETENTION REQUIREMENTS",
                    "section_slug": "retention-requirements",
                    "chunk_role": "section_list",
                    "starts_with_heading": True,
                    "is_list_block": True,
                }
            ],
        }
    )

    embedding_text = _embedding_text_for_chunk(
        context=context,
        manifest=manifest,
        chunk_index=0,
    )

    assert "Document title: Employee Retention Policy" in embedding_text
    assert "Section: RETENTION REQUIREMENTS" in embedding_text
    assert "Chunk role: section list" in embedding_text
    assert "Structure: list" in embedding_text
    assert embedding_text.endswith("Employee records must be retained for 7 years.")


@pytest.mark.asyncio()
async def test_dense_index_document_reads_manifest_and_upserts_points(
    monkeypatch,
) -> None:
    """Dense indexing should embed manifest chunks and upsert one point per chunk."""

    context = IngestionJobContext(
        job_id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        namespace_id=uuid.uuid4(),
        object_key="tenants/t1/namespaces/n1/documents/d1/source/manual.txt",
        mime_type="text/plain",
        title="Manual",
        source_uri=None,
        attempt_count=1,
    )
    manifest = ChunkManifest.from_payload(
        {
            "document_id": str(context.document_id),
            "source_artifact_key": (
                "tenants/t1/namespaces/n1/documents/d1/artifacts/extracted/text.txt"
            ),
            "chunking_strategy": "deterministic_token_window_v1",
            "chunks": [
                {
                    "chunk_id": "chunk-1",
                    "chunk_index": 0,
                    "text": "alpha beta",
                    "token_count": 2,
                    "character_count": 10,
                    "start_token": 0,
                    "end_token": 1,
                    "section_title": "OVERVIEW",
                    "section_slug": "overview",
                    "chunk_role": "section_header",
                    "starts_with_heading": True,
                    "is_list_block": False,
                },
                {
                    "chunk_id": "chunk-2",
                    "chunk_index": 1,
                    "text": "gamma delta",
                    "token_count": 2,
                    "character_count": 11,
                    "start_token": 2,
                    "end_token": 3,
                    "section_title": "OVERVIEW",
                    "section_slug": "overview",
                    "chunk_role": "section_body",
                    "starts_with_heading": False,
                    "is_list_block": False,
                },
            ],
        }
    )

    async def fake_download_bytes(key: str) -> bytes:
        assert key.endswith("/artifacts/chunks/manifest.json")
        return json.dumps(manifest.to_payload()).encode("utf-8")

    captured: dict[str, object] = {}
    order: list[str] = []

    def fake_ensure_qdrant_collection(*, vector_size: int) -> None:
        captured["ensured_vector_size"] = vector_size
        order.append("ensure")

    def fake_upsert_dense_points(*, points, vector_size: int) -> int:
        captured["points"] = points
        captured["vector_size"] = vector_size
        order.append("upsert")
        return len(points)

    def fake_delete_dense_points_for_document(*, tenant_id, document_id) -> None:
        captured["deleted_tenant_id"] = tenant_id
        captured["deleted_document_id"] = document_id
        order.append("delete")

    monkeypatch.setattr("app.services.dense_indexing.download_bytes", fake_download_bytes)
    async def fake_embed_texts(texts: list[str], *, purpose: str = "generic"):
        from app.core.embeddings import DenseEmbedding

        assert texts[0].endswith("alpha beta")
        assert texts[1].endswith("gamma delta")
        assert any(text.startswith("Document title: Manual") for text in texts)
        assert purpose == "document"
        return [
            DenseEmbedding(text="alpha beta", vector=[0.1] * 16),
            DenseEmbedding(text="gamma delta", vector=[0.2] * 16),
        ]

    monkeypatch.setattr("app.services.dense_indexing.embed_texts", fake_embed_texts)
    monkeypatch.setattr(
        "app.services.dense_indexing.ensure_qdrant_collection",
        fake_ensure_qdrant_collection,
    )
    monkeypatch.setattr(
        "app.services.dense_indexing.delete_dense_points_for_document",
        fake_delete_dense_points_for_document,
    )
    monkeypatch.setattr(
        "app.services.dense_indexing.upsert_dense_points",
        fake_upsert_dense_points,
    )
    monkeypatch.setenv("DENSE_EMBEDDING_DIMENSIONS", "16")

    from app.config import get_settings

    get_settings.cache_clear()
    try:
        result = await dense_index_document(context)
    finally:
        get_settings.cache_clear()

    assert result.collection_name == "grounded_chunks"
    assert result.points_indexed == 2
    assert result.vector_dimensions == 16
    assert captured["deleted_tenant_id"] == context.tenant_id
    assert captured["deleted_document_id"] == context.document_id
    # the collection exists before the old points are deleted from it: a fresh store's first ingestion works
    assert order == ["ensure", "delete", "upsert"]
    assert captured["ensured_vector_size"] == 16
    assert captured["vector_size"] == 16
    points = captured["points"]
    assert isinstance(points, list)
    assert len(points) == 2
    assert points[0].id == _dense_point_id("chunk-1")
    assert points[0].payload["tenant_id"] == str(context.tenant_id)
    assert points[0].payload["text"] == "alpha beta"
    assert points[0].payload["section_title"] == "OVERVIEW"
    assert points[0].payload["section_slug"] == "overview"
    assert points[0].payload["chunk_role"] == "section_header"


@pytest.mark.asyncio()
async def test_dense_index_document_embeds_chunks_as_documents(monkeypatch) -> None:
    """Dense indexing should request document-style embeddings for chunk text."""

    context = IngestionJobContext(
        job_id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        namespace_id=uuid.uuid4(),
        object_key="tenants/t1/namespaces/n1/documents/d1/source/manual.txt",
        mime_type="text/plain",
        title="Manual",
        source_uri=None,
        attempt_count=1,
    )
    manifest = ChunkManifest.from_payload(
        {
            "document_id": str(context.document_id),
            "source_artifact_key": "tenants/t1/namespaces/n1/documents/d1/artifacts/extracted/text.txt",
            "chunking_strategy": "deterministic_token_window_v1",
            "chunks": [
                {
                    "chunk_id": "chunk-1",
                    "chunk_index": 0,
                    "text": "alpha beta",
                    "token_count": 2,
                    "character_count": 10,
                    "start_token": 0,
                    "end_token": 1,
                    "section_title": None,
                    "section_slug": None,
                    "chunk_role": "body",
                    "starts_with_heading": False,
                    "is_list_block": False,
                }
            ],
        }
    )

    async def fake_download_bytes(key: str) -> bytes:
        del key
        return json.dumps(manifest.to_payload()).encode("utf-8")

    async def fake_embed_texts(texts: list[str], *, purpose: str = "generic"):
        from app.core.embeddings import DenseEmbedding

        assert texts[0].endswith("alpha beta")
        assert "Document title: Manual" in texts[0]
        assert purpose == "document"
        return [DenseEmbedding(text="alpha beta", vector=[0.1] * 8)]

    monkeypatch.setattr("app.services.dense_indexing.download_bytes", fake_download_bytes)
    monkeypatch.setattr("app.services.dense_indexing.embed_texts", fake_embed_texts)
    monkeypatch.setattr(
        "app.services.dense_indexing.delete_dense_points_for_document",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "app.services.dense_indexing.upsert_dense_points",
        lambda **kwargs: 1,
    )
    monkeypatch.setenv("DENSE_EMBEDDING_DIMENSIONS", "8")

    from app.config import get_settings

    get_settings.cache_clear()
    try:
        result = await dense_index_document(context)
    finally:
        get_settings.cache_clear()

    assert result.points_indexed == 1


@pytest.mark.asyncio()
async def test_embed_texts_uses_retrieval_task_types_for_gemini(monkeypatch) -> None:
    """Gemini embeddings should use retrieval-specific task types for docs and queries."""

    monkeypatch.setenv("EMBEDDING_BACKEND", "gemini_v1")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("DENSE_EMBEDDING_DIMENSIONS", "8")

    from app.config import get_settings

    requests: list[dict[str, object]] = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

        def read(self) -> bytes:
            return json.dumps({"embedding": {"values": [0.1] * 8}}).encode("utf-8")

    def fake_urlopen(request, timeout):
        del timeout
        requests.append(json.loads(request.data.decode("utf-8")))
        return FakeResponse()

    get_settings.cache_clear()
    try:
        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        await embed_texts(["document text"], purpose="document")
        await embed_texts(["query text"], purpose="query")
    finally:
        get_settings.cache_clear()

    assert requests[0]["taskType"] == "RETRIEVAL_DOCUMENT"
    assert requests[1]["taskType"] == "RETRIEVAL_QUERY"


@pytest.mark.asyncio()
async def test_embed_texts_raises_when_provider_backend_is_strict(monkeypatch) -> None:
    """Configured remote embeddings should fail cleanly when provider fallback is disabled."""

    monkeypatch.setenv("EMBEDDING_BACKEND", "gemini_v1")
    monkeypatch.setenv("EMBEDDING_PROVIDER_FALLBACK_ENABLED", "false")
    monkeypatch.setenv("DENSE_EMBEDDING_DIMENSIONS", "8")

    from app.config import get_settings

    get_settings.cache_clear()
    try:
        async def fake_embed_texts_gemini(texts: list[str], *, purpose: str = "generic"):
            del texts, purpose
            raise GeminiEmbeddingError("provider unavailable")

        monkeypatch.setattr(
            "app.core.gemini_embeddings.embed_texts_gemini",
            fake_embed_texts_gemini,
        )

        with pytest.raises(GeminiEmbeddingError, match="provider unavailable"):
            await embed_texts(["alpha beta"], purpose="query")
    finally:
        get_settings.cache_clear()


@pytest.mark.asyncio()
async def test_dense_index_document_rejects_invalid_manifest(monkeypatch) -> None:
    """Invalid manifest payloads should fail with a clean ingestion error."""

    context = IngestionJobContext(
        job_id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        namespace_id=uuid.uuid4(),
        object_key="tenants/t1/namespaces/n1/documents/d1/source/manual.txt",
        mime_type="text/plain",
        title="Manual",
        source_uri=None,
        attempt_count=1,
    )

    async def fake_download_bytes(key: str) -> bytes:
        del key
        return b"{not valid json"

    monkeypatch.setattr("app.services.dense_indexing.download_bytes", fake_download_bytes)

    with pytest.raises(IngestionProcessorError, match="could not be parsed"):
        await dense_index_document(context)
