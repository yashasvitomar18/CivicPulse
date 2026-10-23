
import cv2

image_path = "engine1/test_traffic.png"
output_path = "engine1/traffic_demo.mp4"

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

height, width = image.shape[:2]

fps = 20
duration = 5
total_frames = fps * duration

writer = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

for i in range(total_frames):
    frame = image.copy()

    cv2.putText(
        frame,
        f"CivicPulse Demo | Frame: {i + 1}/{total_frames}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    writer.write(frame)

writer.release()
print(f"Demo video saved: {output_path}")
