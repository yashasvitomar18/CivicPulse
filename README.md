
# CivicPulse 🚦
### AI-Powered Smart City Infrastructure & Public Safety System

CivicPulse is a smart-city safety project that uses computer vision to analyze traffic videos, detect vehicles, estimate traffic density, and send potential traffic incidents to a FastAPI backend.

> **Project status:** Prototype / Development

## Features

- **Vehicle Detection:** Detects cars, motorcycles, buses, and trucks using YOLO.
- **Vehicle Tracking:** Tracks detected vehicles across video frames using ByteTrack.
- **Traffic Analytics:** Counts vehicle detections by category.
- **Traffic Density Classification:** Classifies sampled frames as Low, Medium, or High using configurable demo thresholds.
- **Incident API:** Sends potential traffic congestion alerts to a FastAPI backend.
- **Multi-Frame Validation:** Requires consecutive sampled frames to meet the configured vehicle-count threshold before creating an alert.
- **API Retry Logic:** Retries failed incident requests up to three attempts.
- **Automated Tests:** Includes backend API tests using pytest.

## Tech Stack

- Python
- YOLO / Ultralytics
- OpenCV
- ByteTrack
- FastAPI
- SQLAlchemy
- SQLite
- Pydantic
- Pytest
- Requests

## Project Structure

```text
smart-city-safety/
├── app/
│   ├── main.py
│   ├── models.py
│   ├── core/
│   │   └── database.py
│   └── schemas/
│       └── incident.py
├── engine1/
│   ├── vehicle_detection.py
│   ├── vehicle_tracking.py
│   ├── traffic_analytics.py
│   ├── traffic_density.py
│   ├── incident_detector.py
│   └── traffic_real.mp4
├── tests/
│   └── test_incidents.py
├── requirements.txt
├── .gitignore
└── README.md
```

*Note: The structure above documents the current intended layout. Keep filenames aligned with the files present in your repository.*

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/yashasvitomar18/CivicPulse.git
cd CivicPulse
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

The computer-vision scripts also require the relevant Ultralytics, OpenCV, and Requests packages. Install them if they are not already included in your requirements:

```powershell
python -m pip install ultralytics opencv-python requests
```

## Run the Backend

From the project root, activate the virtual environment and run:

```powershell
uvicorn app.main:app --reload
```

API documentation:

- Swagger UI: http://127.0.0.1:8000/docs
- Health endpoint: http://127.0.0.1:8000/health

## Run the Computer Vision Engine

Make sure the input video exists at `engine1/traffic_real.mp4`.

### Vehicle Detection

```powershell
python engine1/vehicle_detection.py
```

### Vehicle Tracking

```powershell
python engine1/vehicle_tracking.py
```

### Traffic Analytics

```powershell
python engine1/traffic_analytics.py
```

### Traffic Density Classification

```powershell
python engine1/traffic_density.py
```

### Incident Detection and API Integration

Start the FastAPI backend in a separate terminal first:

```powershell
uvicorn app.main:app --reload
```

Then run the detector:

```powershell
python engine1/incident_detector.py
```

Example test configuration in PowerShell:

```powershell
$env:CONGESTION_THRESHOLD="20"
$env:REQUIRED_CONGESTED_FRAMES="2"
python engine1/incident_detector.py
```

These environment variables apply to the current PowerShell session. Open a new session or clear them to return to the default settings.

## Configuration

The incident detector supports these environment variables:

| Variable | Default | Purpose |
|---|---:|---|
| `TRAFFIC_VIDEO_PATH` | `engine1/traffic_real.mp4` | Input video |
| `INCIDENT_API_URL` | `http://127.0.0.1:8000/api/v1/incidents` | Backend endpoint |
| `YOLO_MODEL_PATH` | `yolov8n.pt` | YOLO model |
| `CONGESTION_THRESHOLD` | `25` | Minimum vehicle detections in a sampled frame |
| `FRAME_INTERVAL` | `24` | Analyze every Nth frame |
| `REQUIRED_CONGESTED_FRAMES` | `3` | Consecutive sampled frames required |
| `DETECTION_CONFIDENCE` | `0.35` | YOLO detection confidence threshold |

## Run Tests

```powershell
python -m pytest -v
```

The current backend test suite has passed five tests in the development environment.

## Important Limitations

- Traffic density thresholds are demonstration values and have not been calibrated for a specific road or camera.
- Vehicle count in a frame is not the same as road occupancy, vehicle speed, or a verified congestion measurement.
- Summing detections across frames does not represent unique vehicles.
- Congestion alerts are potential incidents and require human review.
- The current congestion detector does not independently confirm collisions or accidents.
- API retries do not guarantee duplicate prevention if a request is saved by the server but its response is lost.
- Performance and reliability must be validated using varied, representative videos before real-world deployment.

## Future Improvements

- Road accident and collision detection
- Camera-specific density calibration
- Better incident deduplication and idempotency
- Structured logging and monitoring
- More automated tests for computer-vision and API integration
- Dashboard for viewing traffic conditions and incident history

## Author

**Yashasvi Tomar**

GitHub: [@yashasvitomar18](https://github.com/yashasvitomar18)
