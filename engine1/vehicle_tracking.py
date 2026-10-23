
from ultralytics import YOLO
import cv2

MODEL_PATH = "yolov8n.pt"
VIDEO_PATH = "engine1/traffic_real.mp4"
OUTPUT_PATH = "engine1/traffic_tracked.mp4"

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

fps = cap.get(cv2.CAP_PROP_FPS) or 24
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

unique_ids = set()
frame_count = 0

try:
    while True:
        success, frame = cap.read()

        if not success:
            break

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=list(VEHICLE_CLASSES.keys()),
            conf=0.35,
            verbose=False,
        )

        result = results[0]

        if result.boxes is not None and result.boxes.id is not None:
            track_ids = result.boxes.id.int().cpu().tolist()
            unique_ids.update(track_ids)

        annotated = result.plot()

        cv2.putText(
            annotated,
            f"Unique vehicle IDs observed: {len(unique_ids)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        writer.write(annotated)
        frame_count += 1

        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames...")

finally:
    cap.release()
    writer.release()

print("\n===== CivicPulse Tracking Report =====")
print(f"Frames processed: {frame_count}")
print(f"Unique tracking IDs observed: {len(unique_ids)}")
print(f"Tracked video saved: {OUTPUT_PATH}")
