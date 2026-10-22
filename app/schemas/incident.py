from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class IncidentAlert(BaseModel):
    incident_type: str
    location: str
    severity: Literal["low", "medium", "high", "critical"]
    source: Literal["vision", "risk_model", "emergency_call"]
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    description: str

class IncidentResponse(IncidentAlert):
    id: int    