
import requests
from ultralytics import YOLO
import cv2

VIDEO_PATH = "engine1/traffic_real.mp4"
API_URL = "http://127.0.0.1:8000/api/v1/incidents"
MODEL_PATH = "yolov8n.pt"

# Demo threshold: vehicles detected in one sampled frame
CONGESTION_THRESHOLD = 25
FRAME_INTERVAL = 24

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

frame_number = 0
alert_sent = False

try:
    while True:
        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        if frame_number % FRAME_INTERVAL != 0:
            continue

        result = model(frame, conf=0.35, verbose=False)[0]

        counts = {name: 0 for name in VEHICLE_CLASSES.values()}

        for box in result.boxes:
            class_id = int(box.cls[0])
            if class_id in VEHICLE_CLASSES:
                counts[VEHICLE_CLASSES[class_id]] += 1

        total_vehicles = sum(counts.values())

        print(
            f"Frame {frame_number}: "
            f"{total_vehicles} vehicles detected"
        )

        if total_vehicles >= CONGESTION_THRESHOLD and not alert_sent:
            payload = {
                "incident_type": "possible_traffic_congestion",
                "location": "Demo Traffic Camera",
                "severity": "medium",
                "source": "vision",
                "confidence": 0.60,
                "description": (
                    f"Possible congestion candidate: {total_vehicles} "
                    f"vehicles detected in frame {frame_number}. "
                    "Automated review required."
                ),
            }

            try:
                response = requests.post(
                    API_URL,
                    json=payload,
                    timeout=5,
                )
                response.raise_for_status()
                print("Incident saved:", response.json())
                alert_sent = True

            except requests.RequestException as exc:
                print(f"Incident API request failed: {exc}")

finally:
    cap.release()

print(f"Processed frames: {frame_number}")
print("Incident detector finished.")
