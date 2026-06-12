"""Schema definitions for synthetic JSON extraction."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TicketExtraction(BaseModel):
    """Target JSON schema for the extraction task."""

    intent: str = Field(..., description="Canonical intent label")
    priority: str = Field(..., description="One of: low, medium, high")
    product: str = Field(..., description="Product area referenced in the ticket")
    needs_human: bool = Field(..., description="Whether this requires human escalation")
