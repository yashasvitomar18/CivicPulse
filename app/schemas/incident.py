
from datetime import datetime, timezone
from math import isfinite
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class IncidentAlert(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    incident_type: str = Field(min_length=1, max_length=100)
    location: str = Field(min_length=1, max_length=200)

    severity: Literal["low", "medium", "high", "critical"]
    source: Literal["vision", "risk_model", "emergency_call"]

    confidence: float = Field(ge=0.0, le=1.0)

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    description: str = Field(min_length=1, max_length=2000)

    @field_validator("incident_type", "location", "description")
    @classmethod
    def reject_blank_strings(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("This field cannot be blank")
        return value

    @field_validator("confidence")
    @classmethod
    def validate_finite_confidence(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("Confidence must be a finite number")
        return value


class IncidentResponse(IncidentAlert):
    id: int
