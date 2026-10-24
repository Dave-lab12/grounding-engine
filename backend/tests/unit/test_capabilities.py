"""Unit tests for product capability resolution."""

from __future__ import annotations

import uuid

from app.api.deps import TenantContext
from app.models import ExecutionTier, SubscriptionPlan, UserFacingMode
from app.services.capabilities import build_capabilities_response
from app.services import capabilities as capabilities_module


def _tenant_context(
    *,
    subscription_plan: SubscriptionPlan,
    max_execution_tier: ExecutionTier,
) -> TenantContext:
    return TenantContext(
        tenant_id=uuid.uuid4(),
        tenant_name="tenant",
        subscription_plan=subscription_plan,
        max_execution_tier=max_execution_tier,
        api_key_id=uuid.uuid4(),
        api_key_label="test-key",
    )


def test_capabilities_enable_only_standard_backed_modes_for_free_plan() -> None:
    """Free tenants should only receive Auto and Instant at this stage."""

    response = build_capabilities_response(
        tenant_context=_tenant_context(
            subscription_plan=SubscriptionPlan.FREE,
            max_execution_tier=ExecutionTier.STANDARD,
        )
    )

    modes = {item.mode: item for item in response.modes}

    assert response.default_mode is UserFacingMode.AUTO
    assert modes[UserFacingMode.AUTO].enabled is True
    assert modes[UserFacingMode.INSTANT].enabled is True
    assert modes[UserFacingMode.THINKING].enabled is False
    assert modes[UserFacingMode.THINKING].availability_reason == "plan_restricted"
    assert modes[UserFacingMode.VERIFIED].enabled is False
    assert modes[UserFacingMode.VERIFIED].availability_reason == "plan_restricted"


def test_capabilities_leave_verified_coming_soon_until_critical_is_enabled(monkeypatch) -> None:
    """With Critical off (the CRITICAL_ENABLED default), Enterprise-capable tenants receive Thinking, and
    Verified stays disabled as coming_soon."""

    monkeypatch.setattr(
        capabilities_module,
        "get_settings",
        lambda: type(
            "SettingsStub",
            (),
            {"enterprise_enabled": True, "critical_enabled": False},
        )(),
    )

    response = build_capabilities_response(
        tenant_context=_tenant_context(
            subscription_plan=SubscriptionPlan.ENTERPRISE,
            max_execution_tier=ExecutionTier.CRITICAL,
        )
    )

    modes = {item.mode: item for item in response.modes}

    assert modes[UserFacingMode.AUTO].enabled is True
    assert modes[UserFacingMode.INSTANT].enabled is True
    assert modes[UserFacingMode.THINKING].enabled is True
    assert modes[UserFacingMode.THINKING].availability_reason is None
    assert modes[UserFacingMode.VERIFIED].enabled is False
    assert modes[UserFacingMode.VERIFIED].availability_reason == "coming_soon"


def test_capabilities_respect_max_execution_tier_before_future_implementation() -> None:
    """Tenant tier ceilings should still block modes before implementation state."""

    response = build_capabilities_response(
        tenant_context=_tenant_context(
            subscription_plan=SubscriptionPlan.PRO,
            max_execution_tier=ExecutionTier.STANDARD,
        )
    )

    modes = {item.mode: item for item in response.modes}

    assert modes[UserFacingMode.THINKING].enabled is False
    assert modes[UserFacingMode.THINKING].availability_reason == "tier_restricted"


def test_capabilities_include_current_and_future_product_shell_features() -> None:
    """Capabilities should describe what the backend supports now versus later."""

    response = build_capabilities_response(
        tenant_context=_tenant_context(
            subscription_plan=SubscriptionPlan.BUSINESS,
            max_execution_tier=ExecutionTier.CRITICAL,
        )
    )

    features = {item.key: item for item in response.features}

    assert features["document_upload"].enabled is True
    assert features["ingestion_jobs"].enabled is True
    assert features["grounded_query"].enabled is True
    assert features["workspaces"].enabled is True
    assert features["datasets"].enabled is True
    assert features["agents"].enabled is True
    assert features["agent_chat"].enabled is True
    assert features["conversations"].enabled is True
    assert features["run_history"].enabled is True
    assert features["dashboard"].enabled is True
    assert features["dashboard"].availability_reason is None
    assert features["api_key_management"].enabled is True
    assert features["api_key_management"].availability_reason is None


def test_capabilities_enable_verified_when_critical_is_live(monkeypatch) -> None:
    """Critical-capable tenants should receive Verified once Critical is enabled."""

    monkeypatch.setattr(
        capabilities_module,
        "get_settings",
        lambda: type(
            "SettingsStub",
            (),
            {"enterprise_enabled": True, "critical_enabled": True},
        )(),
    )

    response = build_capabilities_response(
        tenant_context=_tenant_context(
            subscription_plan=SubscriptionPlan.ENTERPRISE,
            max_execution_tier=ExecutionTier.CRITICAL,
        )
    )

    modes = {item.mode: item for item in response.modes}
    assert modes[UserFacingMode.VERIFIED].enabled is True
    assert modes[UserFacingMode.VERIFIED].availability_reason is None
