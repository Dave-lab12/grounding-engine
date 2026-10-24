"""AWS Bedrock embedding helpers (Amazon Titan Text Embeddings V2 request body)."""

from __future__ import annotations

import asyncio
import json

from botocore.client import BaseClient
from botocore.exceptions import BotoCoreError, ClientError

from app.config import Settings, get_settings
from app.core import bedrock_client
from app.core.embeddings import DenseEmbedding, EmbeddingError, EmbeddingPurpose
from app.core.provider_retry import run_with_retries


class BedrockEmbeddingError(EmbeddingError):
    """Raised when the Bedrock embedding backend cannot return vectors."""


async def _embed_one(text: str, *, client: BaseClient, settings: Settings) -> DenseEmbedding:
    body = json.dumps(
        {
            "inputText": text,
            "dimensions": settings.dense_embedding_dimensions,
            "normalize": True,
        }
    )

    def _perform_request() -> object:
        response = client.invoke_model(
            modelId=settings.bedrock_embedding_model,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        return json.loads(response["body"].read())

    try:
        payload = await run_with_retries(
            lambda: asyncio.to_thread(_perform_request),
            max_retries=settings.provider_max_retries,
            backoff_ms=settings.provider_retry_backoff_ms,
            should_retry=bedrock_client.is_retryable_bedrock_error,
        )
    except ClientError as exc:
        raise BedrockEmbeddingError(
            f"Bedrock embeddings failed: {bedrock_client.error_detail(exc)}"
        ) from exc
    except BotoCoreError as exc:
        raise BedrockEmbeddingError("Bedrock embedding request failed.") from exc
    except json.JSONDecodeError as exc:
        raise BedrockEmbeddingError("Bedrock embeddings returned invalid JSON.") from exc

    vector = payload.get("embedding") if isinstance(payload, dict) else None
    if not isinstance(vector, list) or not all(isinstance(value, (int, float)) for value in vector):
        raise BedrockEmbeddingError("Bedrock embeddings returned a malformed vector.")
    if len(vector) != settings.dense_embedding_dimensions:
        raise BedrockEmbeddingError(
            f"Embedding dimension mismatch: expected {settings.dense_embedding_dimensions}, got {len(vector)}."
        )
    return DenseEmbedding(text=text, vector=[float(value) for value in vector])


async def embed_texts_bedrock(
    texts: list[str],
    *,
    purpose: EmbeddingPurpose = "generic",
) -> list[DenseEmbedding]:
    """Embed texts with Bedrock, preserving order.

    Titan V2 takes one text per call and has no task type, so `purpose` is part of the shared signature only.
    """

    del purpose
    settings = get_settings()
    client = bedrock_client.get_bedrock_runtime_client()
    return [await _embed_one(text, client=client, settings=settings) for text in texts]
