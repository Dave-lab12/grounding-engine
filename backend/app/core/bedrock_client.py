"""AWS Bedrock runtime client shared by the Bedrock generation, embedding and reranking backends."""

from __future__ import annotations

from functools import lru_cache

import boto3
from botocore.client import BaseClient
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError

from app.config import get_settings


# Transient failures worth a retry; access and validation errors are final.
_RETRYABLE_ERROR_CODES = {
    "ThrottlingException",
    "ServiceUnavailableException",
    "InternalServerException",
    "ModelTimeoutException",
    "ModelNotReadyException",
}


@lru_cache
def get_bedrock_runtime_client() -> BaseClient:
    """Create and cache the bedrock-runtime client.

    Credentials come from the AWS default chain (environment, shared profile, instance role), never from the
    S3 settings, which point at MinIO. botocore's own retries are off: provider_retry.run_with_retries owns
    them, with the configured backoff.
    """

    settings = get_settings()
    return boto3.client(
        "bedrock-runtime",
        region_name=settings.bedrock_region,
        config=BotoConfig(
            read_timeout=settings.bedrock_timeout_seconds,
            retries={"max_attempts": 1},
        ),
    )


def reset_bedrock_runtime_client() -> None:
    """Clear the cached client for test isolation and settings changes."""

    get_bedrock_runtime_client.cache_clear()


def is_retryable_bedrock_error(exc: Exception) -> bool:
    """Return whether a Bedrock call failed in a way a retry can fix."""

    if isinstance(exc, ClientError):
        return exc.response.get("Error", {}).get("Code") in _RETRYABLE_ERROR_CODES
    return isinstance(exc, BotoCoreError)


def error_detail(exc: ClientError) -> str:
    """The error code and message of a failed Bedrock call, for the raised provider error."""

    error = exc.response.get("Error", {})
    return f"{error.get('Code', 'ClientError')}: {error.get('Message', str(exc))}"
