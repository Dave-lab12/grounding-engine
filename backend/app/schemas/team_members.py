"""Team member API schemas."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr

from app.models.enums import WorkspaceMemberRole, WorkspaceMemberStatus


class TeamMemberInviteRequest(BaseModel):
    """Request to invite a new member to a workspace."""

    email: EmailStr
    role: WorkspaceMemberRole = WorkspaceMemberRole.MEMBER


class TeamMemberUpdateRequest(BaseModel):
    """Request to change the role of an existing member."""

    role: WorkspaceMemberRole


class TeamMemberResponse(BaseModel):
    """Workspace member resource returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    member_id: UUID
    workspace_id: UUID
    email: str
    role: WorkspaceMemberRole
    status: WorkspaceMemberStatus
    invited_at: datetime
    joined_at: datetime | None
