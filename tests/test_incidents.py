
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app
from app.schemas.incident import IncidentAlert

client = TestClient(app)


@pytest.fixture
def valid_payload():
    return {
        "incident_type": "possible_road_collision",
        "location": "Test Intersection",
        "severity": "high",
        "source": "vision",
        "confidence": 0.92,
        "description": "Two vehicles stopped unexpectedly.",
    }


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_create_valid_incident(valid_payload):
    response = client.post("/api/v1/incidents", json=valid_payload)

    assert response.status_code == 200
    assert response.json()["severity"] == "high"
    assert response.json()["confidence"] == 0.92
    assert "timestamp" in response.json()
    assert "id" in response.json()


def test_invalid_severity(valid_payload):
    valid_payload["severity"] = "extreme"

    response = client.post("/api/v1/incidents", json=valid_payload)

    assert response.status_code == 422


def test_invalid_confidence(valid_payload):
    valid_payload["confidence"] = 1.5

    response = client.post("/api/v1/incidents", json=valid_payload)

    assert response.status_code == 422


def test_invalid_severity_filter():
    response = client.get(
        "/api/v1/incidents",
        params={"severity": "extreme"},
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    "field",
    ["incident_type", "location", "description"],
)
def test_reject_blank_required_fields(valid_payload, field):
    valid_payload[field] = "   "

    response = client.post("/api/v1/incidents", json=valid_payload)

    assert response.status_code == 422


@pytest.mark.parametrize(
    "confidence",
    [float("inf"), float("-inf"), float("nan")],
)
def test_reject_non_finite_confidence(confidence, valid_payload):
    payload = {**valid_payload, "confidence": confidence}

    with pytest.raises(ValidationError):
        IncidentAlert.model_validate(payload)
