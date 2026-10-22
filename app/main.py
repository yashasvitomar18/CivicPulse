
from typing import Annotated, Literal

from fastapi import FastAPI, Query, HTTPException

from app.schemas.incident import IncidentAlert, IncidentResponse
from app.core.database import Base, engine, SessionLocal
from app.models import Incident

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Smart City Safety System",
    description="AI-powered traffic monitoring and public safety API",
    version="0.1.0",
)


@app.get("/")
def home():
    return {
        "project": "Smart City Safety System",
        "status": "running",
        "message": "Welcome to the Smart City Control Tower",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/api/v1/incidents", response_model=IncidentResponse)
def create_incident(incident: IncidentAlert):
    db = SessionLocal()

    try:
        db_incident = Incident(
            incident_type=incident.incident_type,
            location=incident.location,
            severity=incident.severity,
            source=incident.source,
            confidence=incident.confidence,
            description=incident.description,
        )

        db.add(db_incident)
        db.commit()
        db.refresh(db_incident)

        return db_incident

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


@app.get(
    "/api/v1/incidents",
    response_model=list[IncidentResponse],
)
def get_incidents(
    severity: Literal["low", "medium", "high", "critical"] | None = None,
    source: Literal["vision", "risk_model", "emergency_call"] | None = None,
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
):
    db = SessionLocal()

    try:
        query = db.query(Incident)

        if severity is not None:
            query = query.filter(Incident.severity == severity)

        if source is not None:
            query = query.filter(Incident.source == source)

        return (
            query.order_by(Incident.id.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )

    finally:
        db.close()


@app.get(
    "/api/v1/incidents/{incident_id}",
    response_model=IncidentResponse,
)
def get_incident(incident_id: int):
    db = SessionLocal()

    try:
        incident = (
            db.query(Incident)
            .filter(Incident.id == incident_id)
            .first()
        )

        if incident is None:
            raise HTTPException(
                status_code=404,
                detail="Incident not found",
            )

        return incident

    finally:
        db.close()
