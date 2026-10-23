
from ultralytics import YOLO
from collections import Counter
import cv2

MODEL_PATH = "yolov8n.pt"
VIDEO_PATH = "engine1/traffic_real.mp4"
OUTPUT_PATH = "engine1/traffic_analytics.mp4"

VEHICLES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def get_density(count):
    if count < 10:
        return "LOW"
    elif count < 25:
        return "MEDIUM"
    return "HIGH"


model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or 24
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

frame_count = 0
unique_ids = set()
vehicle_totals = Counter()

try:
    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=list(VEHICLES.keys()),
            conf=0.35,
            verbose=False,
        )

        result = results[0]
        current_counts = Counter()

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                current_counts[VEHICLES[class_id]] += 1

            if result.boxes.id is not None:
                ids = result.boxes.id.int().cpu().tolist()
                unique_ids.update(ids)

        total = sum(current_counts.values())
        density = get_density(total)

        vehicle_totals.update(current_counts)

        annotated = result.plot()

        cv2.putText(
            annotated,
            f"Vehicles: {total} | Density: {density}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        writer.write(annotated)
        frame_count += 1

finally:
    cap.release()
    writer.release()

print("\n===== CivicPulse Traffic Analytics =====")
print(f"Frames processed: {frame_count}")
print(f"Unique tracking IDs observed: {len(unique_ids)}")
print("\nVehicle detections across all frames:")

for name in VEHICLES.values():
    print(f"{name.capitalize()}: {vehicle_totals[name]}")

print(f"\nAnalytics video saved: {OUTPUT_PATH}")
