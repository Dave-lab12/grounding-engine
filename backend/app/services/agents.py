"""Agent CRUD and dataset attachment services."""

from __future__ import annotations

import uuid

from fastapi import status
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import Agent, AgentDataset, AgentStatus, Namespace, UserFacingMode, Workspace
from app.services.capabilities import get_current_supported_modes
from app.schemas.agents import AgentCreateRequest, AgentUpdateRequest


_ORDERED_AGENT_MODES: tuple[UserFacingMode, ...] = (
    UserFacingMode.AUTO,
    UserFacingMode.INSTANT,
    UserFacingMode.THINKING,
    UserFacingMode.VERIFIED,
)


class AgentServiceError(RuntimeError):
    """Raised when an agent operation cannot be completed."""

    def __init__(self, detail: str, *, status_code: int) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


def _normalize_name(name: str) -> str:
    """Normalize an agent name while rejecting blank values."""

    normalized = name.strip()
    if not normalized:
        raise AgentServiceError(
            "Agent name must not be empty.",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )
    return normalized


def _normalize_optional_text(value: str | None) -> str | None:
    """Normalize optional free-text fields while allowing clearing."""

    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _normalize_system_instructions(value: str | None) -> str:
    """Normalize system instructions while allowing a blank default."""

    if value is None:
        return ""
    return value.strip()


def _normalize_allowed_modes(
    *,
    default_mode: UserFacingMode,
    allowed_modes: list[UserFacingMode] | None,
) -> list[str]:
    """Validate and normalize allowed agent modes for the current backend state."""

    current_modes = get_current_supported_modes()
    if default_mode not in current_modes:
        raise AgentServiceError(
            f"Agent default mode '{default_mode.value}' is not available yet.",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    requested_modes = allowed_modes or [
        mode for mode in _ORDERED_AGENT_MODES if mode in current_modes
    ]
    normalized_values: list[str] = []
    seen: set[str] = set()
    for mode in requested_modes:
        if mode not in current_modes:
            raise AgentServiceError(
                f"Agent mode '{mode.value}' is not available yet.",
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            )
        if mode.value in seen:
            continue
        seen.add(mode.value)
        normalized_values.append(mode.value)

    if default_mode.value not in seen:
        normalized_values.append(default_mode.value)

    return normalized_values


async def _get_workspace_for_tenant(
    *,
    session: AsyncSession,
    tenant_id: uuid.UUID,
    workspace_id: uuid.UUID,
) -> Workspace:
    """Resolve one workspace only if it belongs to the authenticated tenant."""

    statement = select(Workspace).where(
        Workspace.tenant_id == tenant_id,
        Workspace.workspace_id == workspace_id,
    )
    result = await session.execute(statement)
    workspace = result.scalar_one_or_none()
    if workspace is None:
        raise AgentServiceError(
            "Workspace not found for tenant.",
            status_code=status.HTTP_404_NOT_FOUND,
        )
    return workspace


async def _get_dataset_for_tenant(
    *,
    session: AsyncSession,
    tenant_id: uuid.UUID,
    dataset_id: uuid.UUID,
) -> Namespace:
    """Resolve one dataset only if it belongs to the authenticated tenant."""

    statement = select(Namespace).where(
        Namespace.tenant_id == tenant_id,
        Namespace.namespace_id == dataset_id,
    )
    result = await session.execute(statement)
    dataset = result.scalar_one_or_none()
    if dataset is None:
        raise AgentServiceError(
            "Dataset not found for tenant.",
            status_code=status.HTTP_404_NOT_FOUND,
        )
    return dataset


async def _get_agent_for_tenant(
    *,
    session: AsyncSession,
    tenant_id: uuid.UUID,
    agent_id: uuid.UUID,
) -> Agent:
    """Resolve one agent with dataset attachments only if it belongs to the tenant."""

    session.expire_all()
    statement = (
        select(Agent)
        .options(selectinload(Agent.dataset_links))
        .where(
            Agent.tenant_id == tenant_id,
            Agent.agent_id == agent_id,
        )
    )
    result = await session.execute(statement)
    agent = result.scalar_one_or_none()
    if agent is None:
        raise AgentServiceError(
            "Agent not found for tenant.",
            status_code=status.HTTP_404_NOT_FOUND,
        )
    return agent


async def _ensure_agent_name_is_unique(
    *,
    session: AsyncSession,
    tenant_id: uuid.UUID,
    workspace_id: uuid.UUID,
    name: str,
    exclude_agent_id: uuid.UUID | None = None,
) -> None:
    """Ensure agent names remain unique within a workspace."""

    statement = select(Agent).where(
        Agent.tenant_id == tenant_id,
        Agent.workspace_id == workspace_id,
        Agent.name == name,
    )
    result = await session.execute(statement)
    existing = result.scalar_one_or_none()
    if existing is None:
        return
    if exclude_agent_id is not None and existing.agent_id == exclude_agent_id:
        return
    raise AgentServiceError(
        "Agent name already exists in workspace.",
        status_code=status.HTTP_409_CONFLICT,
    )


async def create_agent(
    *,
    session: AsyncSession,
    tenant_id: uuid.UUID,
    agent_request: AgentCreateRequest,
) -> Agent:
    """Create one agent inside a workspace."""

    await _get_workspace_for_tenant(
        session=session,
        tenant_id=tenant_id,
        workspace_id=agent_request.workspace_id,
    )

    name = _normalize_name(agent_request.name)
    description = _normalize_optional_text(agent_request.description)
    system_instructions = _normalize_system_instructions(agent_request.system_instructions)
    allowed_modes = _normalize_allowed_modes(
        default_mode=agent_request.default_mode,
        allowed_modes=agent_request.allowed_modes,
    )

    await _ensure_agent_name_is_unique(
        session=session,
        tenant_id=tenant_id,
        workspace_id=agent_request.workspace_id,
        name=name,
    )

    agent = Agent(
        tenant_id=tenant_id,
        workspace_id=agent_request.workspace_id,
        name=name,
        description=description,
        system_instructions=system_instructions,
        default_mode=agent_request.default_mode,
        allowed_modes=allowed_modes,
        status=AgentStatus.ACTIVE,
    )
    session.add(agent)
    try:
        await session.commit()
    except SQLAlchemyError as exc:
        await session.rollback()
        raise AgentServiceError(
            "Failed to create agent.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from exc

    await session.refresh(agent)
    return await _get_agent_for_tenant(
        session=session,
        tenant_id=tenant_id,
        agent_id=agent.agent_id,
    )


async def list_agents_for_tenant(
    *,
    session: AsyncSession,
    tenant_id: uuid.UUID,
    workspace_id: uuid.UUID | None = None,
) -> list[Agent]:
    """List agents owned by one tenant, optionally filtered by workspace."""

    if workspace_id is not None:
        await _get_workspace_for_tenant(
            session=session,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )

    statement = select(Agent).options(selectinload(Agent.dataset_links)).where(
        Agent.tenant_id == tenant_id
    )
    if workspace_id is not None:
        statement = statement.where(Agent.workspace_id == workspace_id)
    statement = statement.order_by(Agent.created_at.asc(), Agent.agent_id.asc())
    result = await session.execute(statement)
    return list(result.scalars().all())


async def get_agent_for_tenant(
    *,
    session: AsyncSession,
    tenant_id: uuid.UUID,
    agent_id: uuid.UUID,
) -> Agent:
    """Return one agent only if it belongs to the authenticated tenant."""

    return await _get_agent_for_tenant(
        session=session,
        tenant_id=tenant_id,
        agent_id=agent_id,
    )


async def update_agent(
    *,
    session: AsyncSession,
    tenant_id: uuid.UUID,
    agent_id: uuid.UUID,
    agent_request: AgentUpdateRequest,
) -> Agent:
    """Patch one tenant-scoped agent."""

    agent = await _get_agent_for_tenant(
        session=session,
        tenant_id=tenant_id,
        agent_id=agent_id,
    )

    fields_set = agent_request.model_fields_set

    if "name" in fields_set and agent_request.name is not None:
        name = _normalize_name(agent_request.name)
        await _ensure_agent_name_is_unique(
            session=session,
            tenant_id=tenant_id,
            workspace_id=agent.workspace_id,
            name=name,
            exclude_agent_id=agent.agent_id,
        )
        agent.name = name

    if "description" in fields_set:
        agent.description = _normalize_optional_text(agent_request.description)
    if "system_instructions" in fields_set:
        agent.system_instructions = _normalize_system_instructions(
            agent_request.system_instructions
        )

    next_default_mode = agent.default_mode
    if "default_mode" in fields_set and agent_request.default_mode is not None:
        next_default_mode = agent_request.default_mode

    if "default_mode" in fields_set or "allowed_modes" in fields_set:
        requested_allowed_modes = (
            agent_request.allowed_modes
            if "allowed_modes" in fields_set
            else [UserFacingMode(mode) for mode in agent.allowed_modes]
        )
        agent.allowed_modes = _normalize_allowed_modes(
            default_mode=next_default_mode,
            allowed_modes=requested_allowed_modes,
        )
        agent.default_mode = next_default_mode

    if "status" in fields_set and agent_request.status is not None:
        agent.status = agent_request.status

    try:
        await session.commit()
    except SQLAlchemyError as exc:
        await session.rollback()
        raise AgentServiceError(
            "Failed to update agent.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from exc

    return await _get_agent_for_tenant(
        session=session,
        tenant_id=tenant_id,
        agent_id=agent.agent_id,
    )


async def attach_dataset_to_agent(
    *,
    session: AsyncSession,
    tenant_id: uuid.UUID,
    agent_id: uuid.UUID,
    dataset_id: uuid.UUID,
) -> Agent:
    """Attach one dataset to one agent in the same workspace."""

    agent = await _get_agent_for_tenant(
        session=session,
        tenant_id=tenant_id,
        agent_id=agent_id,
    )
    dataset = await _get_dataset_for_tenant(
        session=session,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
    )

    if dataset.workspace_id != agent.workspace_id:
        raise AgentServiceError(
            "Dataset must belong to the same workspace as the agent.",
            status_code=status.HTTP_409_CONFLICT,
        )

    statement = select(AgentDataset).where(
        AgentDataset.tenant_id == tenant_id,
        AgentDataset.agent_id == agent_id,
        AgentDataset.dataset_id == dataset_id,
    )
    result = await session.execute(statement)
    existing = result.scalar_one_or_none()
    if existing is None:
        session.add(
            AgentDataset(
                tenant_id=tenant_id,
                agent_id=agent_id,
                dataset_id=dataset_id,
            )
        )
        try:
            await session.commit()
        except SQLAlchemyError as exc:
            await session.rollback()
            raise AgentServiceError(
                "Failed to attach dataset to agent.",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            ) from exc

    return await _get_agent_for_tenant(
        session=session,
        tenant_id=tenant_id,
        agent_id=agent_id,
    )


async def detach_dataset_from_agent(
    *,
    session: AsyncSession,
    tenant_id: uuid.UUID,
    agent_id: uuid.UUID,
    dataset_id: uuid.UUID,
) -> None:
    """Remove one dataset attachment from one agent."""

    await _get_agent_for_tenant(
        session=session,
        tenant_id=tenant_id,
        agent_id=agent_id,
    )

    statement = select(AgentDataset).where(
        AgentDataset.tenant_id == tenant_id,
        AgentDataset.agent_id == agent_id,
        AgentDataset.dataset_id == dataset_id,
    )
    result = await session.execute(statement)
    attachment = result.scalar_one_or_none()
    if attachment is None:
        raise AgentServiceError(
            "Dataset attachment not found for agent.",
            status_code=status.HTTP_404_NOT_FOUND,
        )

    await session.delete(attachment)
    try:
        await session.commit()
    except SQLAlchemyError as exc:
        await session.rollback()
        raise AgentServiceError(
            "Failed to detach dataset from agent.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from exc


async def delete_agent_for_tenant(
    *,
    session: AsyncSession,
    tenant_id: uuid.UUID,
    agent_id: uuid.UUID,
) -> None:
    """Delete one tenant-scoped agent."""

    agent = await _get_agent_for_tenant(
        session=session,
        tenant_id=tenant_id,
        agent_id=agent_id,
    )
    await session.delete(agent)
    try:
        await session.commit()
    except SQLAlchemyError as exc:
        await session.rollback()
        raise AgentServiceError(
            "Failed to delete agent.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from exc
