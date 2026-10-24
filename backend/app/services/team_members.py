"""Team member service — invite, list, update, remove workspace members."""

from __future__ import annotations

import uuid
import hashlib
import secrets

from fastapi import status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.telemetry import get_logger
from app.models.enums import WorkspaceMemberStatus
from app.models.workspace_member import WorkspaceMember
from app.schemas.team_members import TeamMemberInviteRequest, TeamMemberUpdateRequest

logger = get_logger("app.team_members")


class TeamMemberServiceError(RuntimeError):
    def __init__(self, detail: str, *, status_code: int) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


async def list_workspace_members(
    *,
    session: AsyncSession,
    tenant_id: uuid.UUID,
    workspace_id: uuid.UUID,
) -> list[WorkspaceMember]:
    result = await session.execute(
        select(WorkspaceMember).where(
            WorkspaceMember.tenant_id == tenant_id,
            WorkspaceMember.workspace_id == workspace_id,
            WorkspaceMember.status != WorkspaceMemberStatus.REMOVED,
        )
    )
    return list(result.scalars().all())


async def invite_workspace_member(
    *,
    session: AsyncSession,
    tenant_id: uuid.UUID,
    workspace_id: uuid.UUID,
    invite_request: TeamMemberInviteRequest,
) -> tuple[WorkspaceMember, str]:
    raw_token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()

    member = WorkspaceMember(
        member_id=uuid.uuid4(),
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        email=invite_request.email.lower(),
        role=invite_request.role,
        status=WorkspaceMemberStatus.PENDING,
        invitation_token_hash=token_hash,
    )
    session.add(member)
    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise TeamMemberServiceError(
            "A member with that email already exists in this workspace.",
            status_code=status.HTTP_409_CONFLICT,
        ) from exc
    except SQLAlchemyError as exc:
        await session.rollback()
        raise TeamMemberServiceError(
            "Failed to invite workspace member.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from exc
    await session.refresh(member)
    return member, raw_token


async def update_workspace_member(
    *,
    session: AsyncSession,
    tenant_id: uuid.UUID,
    workspace_id: uuid.UUID,
    member_id: uuid.UUID,
    update_request: TeamMemberUpdateRequest,
) -> WorkspaceMember:
    result = await session.execute(
        select(WorkspaceMember).where(
            WorkspaceMember.tenant_id == tenant_id,
            WorkspaceMember.workspace_id == workspace_id,
            WorkspaceMember.member_id == member_id,
        )
    )
    member = result.scalar_one_or_none()
    if member is None:
        raise TeamMemberServiceError(
            "Workspace member not found.", status_code=status.HTTP_404_NOT_FOUND
        )
    member.role = update_request.role
    try:
        await session.commit()
    except SQLAlchemyError as exc:
        await session.rollback()
        raise TeamMemberServiceError(
            "Failed to update workspace member.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from exc
    await session.refresh(member)
    return member


async def remove_workspace_member(
    *,
    session: AsyncSession,
    tenant_id: uuid.UUID,
    workspace_id: uuid.UUID,
    member_id: uuid.UUID,
) -> None:
    result = await session.execute(
        select(WorkspaceMember).where(
            WorkspaceMember.tenant_id == tenant_id,
            WorkspaceMember.workspace_id == workspace_id,
            WorkspaceMember.member_id == member_id,
        )
    )
    member = result.scalar_one_or_none()
    if member is None:
        raise TeamMemberServiceError(
            "Workspace member not found.", status_code=status.HTTP_404_NOT_FOUND
        )
    member.status = WorkspaceMemberStatus.REMOVED
    try:
        await session.commit()
    except SQLAlchemyError as exc:
        await session.rollback()
        raise TeamMemberServiceError(
            "Failed to remove workspace member.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from exc
