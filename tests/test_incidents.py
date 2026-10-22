from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_create_valid_incident():
    payload = {
        "incident_type": "possible_road_collision",
        "location": "Test Intersection",
        "severity": "high",
        "source": "vision",
        "confidence": 0.92,
        "description": "Two vehicles stopped unexpectedly.",
    }

    response = client.post("/api/v1/incidents", json=payload)

    assert response.status_code == 200
    assert response.json()["severity"] == "high"
    assert response.json()["confidence"] == 0.92
    assert "timestamp" in response.json()


def test_invalid_severity():
    payload = {
        "incident_type": "possible_road_collision",
        "location": "Test Intersection",
        "severity": "extreme",
        "source": "vision",
        "confidence": 0.92,
        "description": "Testing invalid severity.",
    }

    response = client.post("/api/v1/incidents", json=payload)

    assert response.status_code == 422


def test_invalid_confidence():
    payload = {
        "incident_type": "possible_road_collision",
        "location": "Test Intersection",
        "severity": "high",
        "source": "vision",
        "confidence": 1.5,
        "description": "Testing invalid confidence.",
    }

    response = client.post("/api/v1/incidents", json=payload)

    assert response.status_code == 422

def test_invalid_severity_filter():
    response = client.get(
        "/api/v1/incidents",
        params={"severity": "extreme"},
    )

    assert response.status_code == 422    
