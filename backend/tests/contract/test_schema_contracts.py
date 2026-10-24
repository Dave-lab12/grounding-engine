"""Schema contract tests.

These tests verify that the Pydantic response models enforce the field
presence, types, and constraints that API consumers depend on. They do
NOT hit the database or network — they construct valid and invalid payloads
directly against the schema classes and assert the contract behaviour.

If a response schema field is removed, renamed, or its type is relaxed/tightened
in a breaking way, at least one test here will fail — catching the regression
before it reaches integration or production.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.schemas.agents import AgentResponse
from app.schemas.api_keys import APIKeyResponse
from app.schemas.datasets import DatasetResponse, DatasetUploadResponse
from app.models.enums import IngestionJobStatus
from app.schemas.query import CitationResponse, GroundedAnswerResponse, QueryRequest
from app.models.enums import (
    AgentStatus,
    DocumentStatus,
    ExecutionTier,
    FreshnessProfile,
    SensitivityLevel,
    UserFacingMode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uuid() -> uuid.UUID:
    return uuid.uuid4()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _citation(**overrides) -> dict:
    base = {
        "citation_id": "E001",
        "chunk_id": "chunk-1",
        "document_id": str(_uuid()),
        "chunk_index": 0,
        "quote": "some grounded quote",
    }
    return {**base, **overrides}


def _grounded_answer(**overrides) -> dict:
    base = {
        "answer": "The answer is here.",
        "citations": [_citation()],
        "confidence_score": 0.85,
        "confidence_label": "high",
        "support_summary": "grounded",
        "verification_status": "passed",
        "degraded_reasons": [],
        "generator_provider": "local-grounded-v1",
        "provider_backend": "local_grounded_v1",
        "provider_model": None,
        "provider_fallback_used": False,
        "provider_fallback_from": None,
    }
    return {**base, **overrides}


# ---------------------------------------------------------------------------
# QueryRequest
# ---------------------------------------------------------------------------

class TestQueryRequestContract:
    def test_valid_request(self):
        req = QueryRequest(namespace_id=_uuid(), query="What is grounding?")
        assert req.query == "What is grounding?"

    def test_empty_query_rejected(self):
        with pytest.raises(ValidationError):
            QueryRequest(namespace_id=_uuid(), query="")

    def test_single_char_query_accepted(self):
        # min_length=1 — a single character is valid at schema level
        req = QueryRequest(namespace_id=_uuid(), query="x")
        assert req.query == "x"

    def test_namespace_id_required(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="hello")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# CitationResponse
# ---------------------------------------------------------------------------

class TestCitationResponseContract:
    def test_valid_citation(self):
        c = CitationResponse(**_citation())
        assert c.citation_id == "E001"
        assert c.chunk_index == 0

    def test_empty_citation_id_rejected(self):
        with pytest.raises(ValidationError):
            CitationResponse(**_citation(citation_id=""))

    def test_empty_quote_rejected(self):
        with pytest.raises(ValidationError):
            CitationResponse(**_citation(quote=""))

    def test_negative_chunk_index_rejected(self):
        with pytest.raises(ValidationError):
            CitationResponse(**_citation(chunk_index=-1))

    def test_document_id_must_be_uuid(self):
        with pytest.raises(ValidationError):
            CitationResponse(**_citation(document_id="not-a-uuid"))


# ---------------------------------------------------------------------------
# GroundedAnswerResponse
# ---------------------------------------------------------------------------

class TestGroundedAnswerResponseContract:
    def test_valid_response(self):
        r = GroundedAnswerResponse(**_grounded_answer())
        assert r.verification_status == "passed"
        assert r.provider_fallback_used is False

    def test_confidence_score_bounds(self):
        with pytest.raises(ValidationError):
            GroundedAnswerResponse(**_grounded_answer(confidence_score=1.1))
        with pytest.raises(ValidationError):
            GroundedAnswerResponse(**_grounded_answer(confidence_score=-0.1))

    def test_invalid_confidence_label_rejected(self):
        with pytest.raises(ValidationError):
            GroundedAnswerResponse(**_grounded_answer(confidence_label="uncertain"))

    def test_invalid_support_summary_rejected(self):
        with pytest.raises(ValidationError):
            GroundedAnswerResponse(**_grounded_answer(support_summary="unknown"))

    def test_invalid_verification_status_rejected(self):
        with pytest.raises(ValidationError):
            GroundedAnswerResponse(**_grounded_answer(verification_status="pending"))

    def test_empty_answer_rejected(self):
        with pytest.raises(ValidationError):
            GroundedAnswerResponse(**_grounded_answer(answer=""))

    def test_empty_generator_provider_rejected(self):
        with pytest.raises(ValidationError):
            GroundedAnswerResponse(**_grounded_answer(generator_provider=""))

    def test_degraded_response_accepted(self):
        r = GroundedAnswerResponse(**_grounded_answer(
            verification_status="degraded",
            support_summary="insufficient",
            degraded_reasons=["UNSUPPORTED_CLAIMS"],
            confidence_label="low",
        ))
        assert r.degraded_reasons == ["UNSUPPORTED_CLAIMS"]

    def test_required_fields_present(self):
        r = GroundedAnswerResponse(**_grounded_answer())
        for field in (
            "answer", "citations", "confidence_score", "confidence_label",
            "support_summary", "verification_status", "degraded_reasons",
            "generator_provider", "provider_backend", "provider_fallback_used",
        ):
            assert hasattr(r, field), f"Missing required field: {field}"


# ---------------------------------------------------------------------------
# DatasetResponse
# ---------------------------------------------------------------------------

class TestDatasetResponseContract:
    def _valid(self, **overrides) -> dict:
        base = {
            "dataset_id": _uuid(),
            "workspace_id": _uuid(),
            "name": "Policy Library",
            "domain": "compliance",
            "sensitivity_level": SensitivityLevel.INTERNAL,
            "freshness_profile": FreshnessProfile.BALANCED,
            "min_execution_tier": ExecutionTier.STANDARD,
            "allow_web_fallback": False,
            "allow_internal_model_retrieval": False,
            "created_at": _now(),
        }
        return {**base, **overrides}

    def test_valid_dataset(self):
        d = DatasetResponse(**self._valid())
        assert d.name == "Policy Library"
        assert d.allow_web_fallback is False

    def test_all_required_fields_present(self):
        d = DatasetResponse(**self._valid())
        for field in (
            "dataset_id", "name", "domain", "sensitivity_level",
            "freshness_profile", "min_execution_tier", "allow_web_fallback",
            "allow_internal_model_retrieval", "created_at",
        ):
            assert hasattr(d, field), f"Missing required field: {field}"

    def test_invalid_sensitivity_level_rejected(self):
        with pytest.raises(ValidationError):
            DatasetResponse(**self._valid(sensitivity_level="top_secret"))


# ---------------------------------------------------------------------------
# AgentResponse
# ---------------------------------------------------------------------------

class TestAgentResponseContract:
    def _valid(self, **overrides) -> dict:
        base = {
            "agent_id": _uuid(),
            "workspace_id": _uuid(),
            "name": "Policy Analyst",
            "description": "Answers policy questions.",
            "system_instructions": "Be concise.",
            "default_mode": UserFacingMode.AUTO,
            "allowed_modes": [UserFacingMode.AUTO, UserFacingMode.INSTANT],
            "status": AgentStatus.ACTIVE,
            "dataset_ids": [_uuid()],
            "created_at": _now(),
            "updated_at": _now(),
        }
        return {**base, **overrides}

    def test_valid_agent(self):
        a = AgentResponse(**self._valid())
        assert a.name == "Policy Analyst"
        assert a.system_instructions == "Be concise."

    def test_system_instructions_always_present(self):
        a = AgentResponse(**self._valid(system_instructions=""))
        assert a.system_instructions == ""

    def test_dataset_ids_is_list(self):
        a = AgentResponse(**self._valid(dataset_ids=[]))
        assert isinstance(a.dataset_ids, list)

    def test_invalid_status_rejected(self):
        with pytest.raises(ValidationError):
            AgentResponse(**self._valid(status="deleted"))


# ---------------------------------------------------------------------------
# APIKeyResponse
# ---------------------------------------------------------------------------

class TestAPIKeyResponseContract:
    def _valid(self, **overrides) -> dict:
        base = {
            "key_id": _uuid(),
            "label": "Frontend key",
            "created_at": _now(),
            "last_used_at": None,
            "revoked_at": None,
        }
        return {**base, **overrides}

    def test_valid_key(self):
        k = APIKeyResponse(**self._valid())
        assert k.label == "Frontend key"
        assert k.revoked_at is None

    def test_revoked_key_accepted(self):
        k = APIKeyResponse(**self._valid(revoked_at=_now()))
        assert k.revoked_at is not None

    def test_all_required_fields_present(self):
        k = APIKeyResponse(**self._valid())
        for field in ("key_id", "label", "created_at", "last_used_at", "revoked_at"):
            assert hasattr(k, field), f"Missing field: {field}"


# ---------------------------------------------------------------------------
# DatasetUploadResponse
# ---------------------------------------------------------------------------

class TestDatasetUploadResponseContract:
    def _valid(self, **overrides) -> dict:
        base = {
            "document_id": _uuid(),
            "dataset_id": _uuid(),
            "job_id": _uuid(),
            "filename": "policy_memo.pdf",
            "title": "Policy memo",
            "mime_type": "application/pdf",
            "file_size_bytes": 10240,
            "document_status": DocumentStatus.UPLOADED,
            "job_status": IngestionJobStatus.QUEUED,
            "already_exists": False,
        }
        return {**base, **overrides}

    def test_valid_upload_response(self):
        r = DatasetUploadResponse(**self._valid())
        assert r.already_exists is False
        assert r.document_status == DocumentStatus.UPLOADED

    def test_already_exists_flag(self):
        r = DatasetUploadResponse(**self._valid(already_exists=True))
        assert r.already_exists is True


# ---------------------------------------------------------------------------
# Rate limit: verify 429 response shape is consistent across headers
# ---------------------------------------------------------------------------

class TestRateLimitResponseShape:
    """Verify that the rate limiter module exports the expected public symbols."""

    def test_rate_limit_public_api(self):
        from app.core.rate_limit import (
            QUERY_RPM,
            INGEST_RPM,
            DEFAULT_RPM,
            check_rate_limit,
            query_rate_limit,
            ingest_rate_limit,
            default_rate_limit,
        )
        assert QUERY_RPM >= 1
        assert INGEST_RPM >= 1
        assert DEFAULT_RPM >= 1
        assert callable(check_rate_limit)
        assert callable(query_rate_limit)
        assert callable(ingest_rate_limit)
        assert callable(default_rate_limit)

    def test_check_rate_limit_returns_result(self):
        from app.core.rate_limit import check_rate_limit, RateLimitResult
        result = check_rate_limit(
            api_key_id="test-key-contract",
            limit_key="contract_test",
            max_requests=100,
        )
        assert isinstance(result, RateLimitResult)
        assert result.allowed is True
        assert result.limit == 100
        assert result.remaining >= 0

    def test_rate_limit_blocks_after_exhaustion(self):
        from app.core.rate_limit import check_rate_limit
        key = f"contract-exhaust-{uuid.uuid4()}"
        # Exhaust the bucket
        for _ in range(3):
            check_rate_limit(api_key_id=key, limit_key="exhaust_test", max_requests=3)
        # Next call must be blocked
        result = check_rate_limit(api_key_id=key, limit_key="exhaust_test", max_requests=3)
        assert result.allowed is False
        assert result.remaining == 0
