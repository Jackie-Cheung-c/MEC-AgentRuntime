"""Structured operation names shared by Travel tools and state adapters."""

from enum import StrEnum


class TravelStateOperation(StrEnum):
    STATE_CREATED = "state_created"
    PROPOSAL_CREATED = "proposal_created"
    STATE_CONFIRMED = "state_confirmed"