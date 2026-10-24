"""Application settings for the Grounded backend."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AnyHttpUrl, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from app.pipeline.contracts import ChunkingStrategy


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]

AppEnv = Literal["development", "test", "staging", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
GeneratorBackend = Literal[
    "local_grounded_v1", "gemini_v1", "openai_compatible_v1", "bedrock_v1"
]
EmbeddingBackend = Literal[
    "local_hash_v1", "gemini_v1", "openai_compatible_v1", "bedrock_v1"
]
RerankerBackend = Literal["disabled", "stub", "gemini_v1", "bedrock_v1"]
RetrievalMode = Literal["hybrid", "dense_only"]


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    app_name: str = "Grounded Backend"
    app_version: str = "0.1.0"
    app_env: AppEnv = "development"
    log_level: LogLevel = "INFO"
    api_base_url: AnyHttpUrl = "http://localhost:8000"
    cors_allowed_origins: str = (
        "http://localhost:8080,"
        "http://127.0.0.1:8080,"
        "http://localhost:5173,"
        "http://127.0.0.1:5173,"
        "http://localhost:3000,"
        "http://127.0.0.1:3000"
    )
    database_url: str = "postgresql+asyncpg://grounded:grounded@localhost:5433/grounded"
    alembic_database_url: str = (
        "postgresql+psycopg://grounded:grounded@localhost:5433/grounded"
    )
    redis_url: str = "redis://localhost:6379/0"
    document_upload_max_bytes: int = 25 * 1024 * 1024
    ingestion_autorun_enabled: bool = True
    chunking_strategy: ChunkingStrategy = "deterministic_token_window_v1"
    chunk_max_tokens: int = 256
    chunk_overlap_tokens: int = 40
    generator_backend: GeneratorBackend = "local_grounded_v1"
    gemini_api_key: str | None = None
    gemini_base_url: AnyHttpUrl = "https://generativelanguage.googleapis.com/v1beta"
    gemini_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "models/gemini-embedding-001"
    openai_api_key: str | None = None
    openai_base_url: AnyHttpUrl = "https://api.openai.com/v1"
    openai_model: str = "gpt-4.1-mini"
    openai_timeout_seconds: int = 30
    # AWS Bedrock. Credentials come from the AWS default chain (environment, profile, role); only the region
    # and the model ids are set here. The embedding model takes Titan Text Embeddings V2's request body, whose
    # output size must be 256, 512 or 1024 (DENSE_EMBEDDING_DIMENSIONS).
    bedrock_region: str | None = None
    bedrock_model: str | None = None
    bedrock_embedding_model: str = "amazon.titan-embed-text-v2:0"
    bedrock_timeout_seconds: int = 60
    # Explicit AWS credentials for Bedrock, for a .env without a shared AWS profile; unset, boto3's default
    # chain decides. The standard names, so a shell export and a .env line are the same thing.
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None
    provider_max_retries: int = 2
    provider_retry_backoff_ms: int = 250
    qdrant_url: AnyHttpUrl = "http://localhost:6333"
    qdrant_collection: str = "grounded_chunks"
    qdrant_check_compatibility: bool = False
    embedding_backend: EmbeddingBackend = "local_hash_v1"
    embedding_provider_fallback_enabled: bool = False
    dense_embedding_dimensions: int = 128
    openai_embedding_model: str = "text-embedding-3-small"
    retrieval_candidate_limit: int = 8
    retrieval_overfetch_factor: int = 4
    retrieval_mode: RetrievalMode = "dense_only"
    rrf_smoothing_constant: int = 60
    evidence_package_limit: int = 3
    enterprise_enabled: bool = True
    enterprise_trace_metadata_enabled: bool = True
    enterprise_auto_routing_enabled: bool = True
    enterprise_reranker_enabled: bool = True
    enterprise_reranker_backend: RerankerBackend = "gemini_v1"
    enterprise_reranker_candidate_limit: int = 24
    enterprise_temporal_scoring_enabled: bool = True
    critical_enabled: bool = False
    api_key_salt: str = "replace-in-local-env"
    resend_api_key: str = ""
    frontend_url: str = "http://localhost:5173"
    google_client_id: str = ""
    google_client_secret: str = ""
    s3_endpoint_url: AnyHttpUrl = "http://localhost:9000"
    s3_bucket: str = "grounded-documents"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_secure: bool = False

    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        """Normalize log level values before validation."""

        if isinstance(value, str):
            return value.upper()
        return value

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, value: str) -> str:
        """Ensure the runtime database URL uses the async driver."""

        if not value.startswith("postgresql+asyncpg://"):
            raise ValueError("DATABASE_URL must use the postgresql+asyncpg driver.")
        return value

    @field_validator("alembic_database_url")
    @classmethod
    def validate_alembic_database_url(cls, value: str) -> str:
        """Ensure the Alembic URL uses the sync psycopg driver."""

        if not value.startswith("postgresql+psycopg://"):
            raise ValueError(
                "ALEMBIC_DATABASE_URL must use the postgresql+psycopg driver."
            )
        return value

    @field_validator("document_upload_max_bytes")
    @classmethod
    def validate_document_upload_max_bytes(cls, value: int) -> int:
        """Ensure the upload size limit is a positive number of bytes."""

        if value <= 0:
            raise ValueError("DOCUMENT_UPLOAD_MAX_BYTES must be greater than zero.")
        return value

    @field_validator("chunk_max_tokens")
    @classmethod
    def validate_chunk_max_tokens(cls, value: int) -> int:
        """Ensure chunk token windows are positive."""

        if value <= 0:
            raise ValueError("CHUNK_MAX_TOKENS must be greater than zero.")
        return value

    @field_validator("chunking_strategy")
    @classmethod
    def validate_chunking_strategy(cls, value: str) -> str:
        """Ensure the configured chunking strategy is non-empty."""

        normalized = value.strip()
        if not normalized:
            raise ValueError("CHUNKING_STRATEGY must not be empty.")
        return normalized

    @field_validator("generator_backend")
    @classmethod
    def validate_generator_backend(cls, value: str) -> str:
        """Ensure the configured generation backend is non-empty."""

        normalized = value.strip()
        if not normalized:
            raise ValueError("GENERATOR_BACKEND must not be empty.")
        return normalized

    @field_validator(
        "gemini_api_key",
        "openai_api_key",
        "bedrock_region",
        "bedrock_model",
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
    )
    @classmethod
    def normalize_optional_secret(cls, value: str | None) -> str | None:
        """Normalize optional provider settings: a blank value is unset."""

        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("chunk_overlap_tokens")
    @classmethod
    def validate_chunk_overlap_tokens(cls, value: int, info: ValidationInfo) -> int:
        """Ensure chunk overlap stays non-negative and below the window size."""

        if value < 0:
            raise ValueError("CHUNK_OVERLAP_TOKENS must not be negative.")
        max_tokens = info.data.get("chunk_max_tokens")
        if isinstance(max_tokens, int) and value >= max_tokens:
            raise ValueError(
                "CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_MAX_TOKENS."
            )
        return value

    @field_validator("qdrant_collection")
    @classmethod
    def validate_qdrant_collection(cls, value: str) -> str:
        """Ensure the Qdrant collection name is not blank."""

        normalized = value.strip()
        if not normalized:
            raise ValueError("QDRANT_COLLECTION must not be empty.")
        return normalized

    @field_validator("embedding_backend")
    @classmethod
    def validate_embedding_backend(cls, value: str) -> str:
        """Ensure the embedding backend name is non-empty."""

        normalized = value.strip()
        if not normalized:
            raise ValueError("EMBEDDING_BACKEND must not be empty.")
        return normalized

    @field_validator("retrieval_mode")
    @classmethod
    def validate_retrieval_mode(cls, value: str) -> str:
        """Ensure retrieval mode is explicit and supported."""

        normalized = value.strip().lower()
        if normalized not in {"hybrid", "dense_only"}:
            raise ValueError("RETRIEVAL_MODE must be either 'hybrid' or 'dense_only'.")
        return normalized

    @field_validator("dense_embedding_dimensions")
    @classmethod
    def validate_dense_embedding_dimensions(cls, value: int) -> int:
        """Ensure dense embedding vectors have a positive configured size."""

        if value <= 0:
            raise ValueError("DENSE_EMBEDDING_DIMENSIONS must be greater than zero.")
        return value

    @field_validator(
        "retrieval_candidate_limit",
        "retrieval_overfetch_factor",
        "rrf_smoothing_constant",
        "evidence_package_limit",
        "provider_max_retries",
        "enterprise_reranker_candidate_limit",
    )
    @classmethod
    def validate_positive_retrieval_settings(
        cls, value: int, info: ValidationInfo
    ) -> int:
        """Ensure retrieval limits and fusion constants remain positive."""

        if value <= 0:
            raise ValueError(f"{info.field_name.upper()} must be greater than zero.")
        return value

    @field_validator("provider_retry_backoff_ms")
    @classmethod
    def validate_provider_retry_backoff_ms(cls, value: int) -> int:
        """Ensure provider retry backoff stays non-negative."""

        if value < 0:
            raise ValueError("PROVIDER_RETRY_BACKOFF_MS must not be negative.")
        return value

    @field_validator("evidence_package_limit")
    @classmethod
    def validate_evidence_package_limit(cls, value: int, info: ValidationInfo) -> int:
        """Keep packaged evidence at or below the retrieved candidate pool."""

        candidate_limit = info.data.get("retrieval_candidate_limit")
        if isinstance(candidate_limit, int) and value > candidate_limit:
            raise ValueError(
                "EVIDENCE_PACKAGE_LIMIT must not exceed RETRIEVAL_CANDIDATE_LIMIT."
            )
        return value

    @field_validator("api_key_salt")
    @classmethod
    def validate_api_key_salt(cls, value: str, info: ValidationInfo) -> str:
        """Ensure the API key salt is not empty or an unsafe placeholder."""

        normalized = value.strip()
        if not normalized:
            raise ValueError("API_KEY_SALT must not be empty.")

        app_env = (info.data.get("app_env") or "development").lower()
        if normalized == "replace-in-local-env" and app_env not in {
            "development",
            "test",
        }:
            raise ValueError(
                "API_KEY_SALT must be set to a secure non-default value outside "
                "development and test."
            )
        return normalized

    @field_validator("s3_bucket", "s3_access_key", "s3_secret_key")
    @classmethod
    def validate_storage_settings(cls, value: str, info: ValidationInfo) -> str:
        """Ensure required storage settings are not blank."""

        if not value.strip():
            field_name = info.field_name.upper()
            raise ValueError(f"{field_name} must not be empty.")
        return value

    @field_validator("s3_secure")
    @classmethod
    def validate_storage_tls_setting(
        cls,
        value: bool,
        info: ValidationInfo,
    ) -> bool:
        """Keep the secure flag aligned with the configured endpoint scheme."""

        endpoint = info.data.get("s3_endpoint_url")
        if endpoint is not None and value != (endpoint.scheme == "https"):
            raise ValueError("S3_SECURE must match the scheme used by S3_ENDPOINT_URL.")
        return value

    def startup_warnings(self) -> list[str]:
        """Return non-fatal startup warnings for fallback-prone Standard config."""

        warnings: list[str] = []
        if self.generator_backend == "gemini_v1" and not self.gemini_api_key:
            warnings.append(
                "GENERATOR_BACKEND=gemini_v1 is configured without GEMINI_API_KEY; "
                "Standard generation will fail until it is set."
            )
        if self.generator_backend == "bedrock_v1" and not self.bedrock_model:
            warnings.append(
                "GENERATOR_BACKEND=bedrock_v1 is configured without BEDROCK_MODEL; "
                "Standard generation will fail until it is set."
            )
        if self.embedding_backend == "gemini_v1" and not self.gemini_api_key:
            message = (
                "EMBEDDING_BACKEND=gemini_v1 is configured without GEMINI_API_KEY; "
            )
            if self.embedding_provider_fallback_enabled:
                warnings.append(
                    message + "dense retrieval will fall back to local hash embeddings."
                )
            else:
                warnings.append(
                    message
                    + "dense retrieval will fail until Gemini embeddings are configured."
                )
        if self.generator_backend == "openai_compatible_v1" and not self.openai_api_key:
            warnings.append(
                "GENERATOR_BACKEND=openai_compatible_v1 is configured without OPENAI_API_KEY; "
                "Standard generation will fail until it is set."
            )
        if self.embedding_backend == "openai_compatible_v1" and not self.openai_api_key:
            message = "EMBEDDING_BACKEND=openai_compatible_v1 is configured without OPENAI_API_KEY; "
            if self.embedding_provider_fallback_enabled:
                warnings.append(
                    message + "dense retrieval will fall back to local hash embeddings."
                )
            else:
                warnings.append(
                    message
                    + "dense retrieval will fail until provider embeddings are configured."
                )
        if (
            self.enterprise_enabled
            and self.enterprise_reranker_enabled
            and self.enterprise_reranker_backend == "gemini_v1"
            and not self.gemini_api_key
        ):
            warnings.append(
                "Enterprise reranking is enabled with the Gemini backend, but GEMINI_API_KEY is missing; "
                "Enterprise retrieval will stay enabled and fall back to fused retrieval ordering."
            )
        if (
            self.enterprise_enabled
            and self.enterprise_reranker_enabled
            and self.enterprise_reranker_backend == "bedrock_v1"
            and not self.bedrock_model
        ):
            warnings.append(
                "Enterprise reranking is enabled with the Bedrock backend, but BEDROCK_MODEL is missing; "
                "Enterprise retrieval will stay enabled and fall back to fused retrieval ordering."
            )
        if self.chunking_strategy == "deterministic_token_window_v1":
            warnings.append(
                "CHUNKING_STRATEGY is using the deterministic token-window baseline. "
                "Consider structure_aware_v1 for stronger Standard retrieval quality."
            )
        return warnings


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""

    return Settings()
