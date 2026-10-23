
import os
import time
import requests
import cv2
from ultralytics import YOLO

VIDEO_PATH = os.getenv(
    "TRAFFIC_VIDEO_PATH",
    "engine1/traffic_real.mp4"
)
API_URL = os.getenv(
    "INCIDENT_API_URL",
    "http://127.0.0.1:8000/api/v1/incidents"
)
MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")

CONGESTION_THRESHOLD = int(
    os.getenv("CONGESTION_THRESHOLD", "25")
)
FRAME_INTERVAL = int(os.getenv("FRAME_INTERVAL", "24"))
REQUIRED_CONGESTED_FRAMES = int(
    os.getenv("REQUIRED_CONGESTED_FRAMES", "3")
)
CONFIDENCE_THRESHOLD = float(
    os.getenv("DETECTION_CONFIDENCE", "0.35")
)

MAX_API_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 2

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def send_incident(payload):
    """Send an incident to the backend with limited retries."""
    for attempt in range(1, MAX_API_ATTEMPTS + 1):
        try:
            response = requests.post(
                API_URL,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            return response.json()

        except (requests.RequestException, ValueError) as exc:
            print(
                f"API attempt {attempt}/{MAX_API_ATTEMPTS} failed: "
                f"{exc}"
            )

            if attempt < MAX_API_ATTEMPTS:
                time.sleep(RETRY_DELAY_SECONDS)

    return None


def main():
    if CONGESTION_THRESHOLD < 1:
        raise ValueError("CONGESTION_THRESHOLD must be at least 1")

    if FRAME_INTERVAL < 1:
        raise ValueError("FRAME_INTERVAL must be at least 1")

    if REQUIRED_CONGESTED_FRAMES < 1:
        raise ValueError(
            "REQUIRED_CONGESTED_FRAMES must be at least 1"
        )

    if not 0 < CONFIDENCE_THRESHOLD <= 1:
        raise ValueError(
            "DETECTION_CONFIDENCE must be between 0 and 1"
        )

    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise FileNotFoundError(
            f"Cannot open video: {VIDEO_PATH}"
        )

    print(f"Video: {VIDEO_PATH}")
    print(f"Vehicle threshold: {CONGESTION_THRESHOLD}")
    print(f"Frame interval: {FRAME_INTERVAL}")
    print(
        "Required consecutive sampled frames: "
        f"{REQUIRED_CONGESTED_FRAMES}"
    )

    frame_number = 0
    alert_sent = False
    congested_frames = 0
    sampled_frames = 0

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            if frame_number % FRAME_INTERVAL != 0:
                continue

            sampled_frames += 1

            result = model(
                frame,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False
            )[0]

            counts = {
                name: 0
                for name in VEHICLE_CLASSES.values()
            }

            for box in result.boxes:
                class_id = int(box.cls[0])

                if class_id in VEHICLE_CLASSES:
                    counts[VEHICLE_CLASSES[class_id]] += 1

            total_vehicles = sum(counts.values())

            print(
                f"\nFrame {frame_number}: "
                f"{total_vehicles} vehicles detected"
            )
            print(f"Vehicle breakdown: {counts}")

            if total_vehicles >= CONGESTION_THRESHOLD:
                congested_frames += 1
            else:
                congested_frames = 0

            print(
                f"Consecutive congested sampled frames: "
                f"{congested_frames}/"
                f"{REQUIRED_CONGESTED_FRAMES}"
            )

            if (
                congested_frames >= REQUIRED_CONGESTED_FRAMES
                and not alert_sent
            ):
                payload = {
                    "incident_type": "possible_traffic_congestion",
                    "location": "Demo Traffic Camera",
                    "severity": "medium",
                    "source": "vision",
                    "confidence": 0.60,
                    "description": (
                        f"Traffic congestion candidate: "
                        f"{total_vehicles} vehicles detected in "
                        f"{REQUIRED_CONGESTED_FRAMES} consecutive "
                        "sampled frames. Automated review required."
                    ),
                }

                incident = send_incident(payload)

                if incident is not None:
                    print("\nIncident successfully saved to API.")
                    print(incident)
                    alert_sent = True
                else:
                    print(
                        "Incident could not be confirmed after retries. "
                        "Continuing video analysis."
                    )

        print("\n--- Detection Summary ---")
        print(f"Total frames processed: {frame_number}")
        print(f"Sampled frames analyzed: {sampled_frames}")
        print(f"Incident sent: {'Yes' if alert_sent else 'No'}")

    finally:
        cap.release()

    print("Incident detector finished.")


if __name__ == "__main__":
    main()
