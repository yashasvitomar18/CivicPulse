
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

video_path = "engine1/traffic_demo.mp4"
output_path = "engine1/traffic_detected.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError(f"Video not found: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS) or 20
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

vehicle_classes = {2, 3, 5, 7}
frame_count = 0

while True:
    success, frame = cap.read()

    if not success:
        break

    results = model(frame, conf=0.35, verbose=False)
    result = results[0]

    vehicle_count = sum(
        1 for box in result.boxes
        if int(box.cls[0]) in vehicle_classes
    )

    annotated = result.plot()

    cv2.putText(
        annotated,
        f"Vehicles detected: {vehicle_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    writer.write(annotated)
    frame_count += 1

cap.release()
writer.release()

print(f"Processed frames: {frame_count}")
print(f"Output saved: {output_path}")
