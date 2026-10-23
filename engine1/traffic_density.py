
from collections import Counter

import cv2
from ultralytics import YOLO

VIDEO_PATH = "engine1/traffic_real.mp4"
MODEL_PATH = "yolov8n.pt"

FRAME_INTERVAL = 24
CONFIDENCE_THRESHOLD = 0.35

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def classify_density(vehicle_count):
    """Demo classification based on vehicle count per sampled frame."""
    if vehicle_count < 10:
        return "Low"
    elif vehicle_count < 20:
        return "Medium"
    return "High"


def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise FileNotFoundError(
            f"Cannot open video: {VIDEO_PATH}"
        )

    frame_number = 0
    sampled_frames = 0
    density_counts = Counter()
    total_vehicles_detected = 0

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
                verbose=False,
            )[0]

            vehicle_counts = Counter()

            for box in result.boxes:
                class_id = int(box.cls[0])

                if class_id in VEHICLE_CLASSES:
                    vehicle_counts[VEHICLE_CLASSES[class_id]] += 1

            counts = {
                name: vehicle_counts.get(name, 0)
                for name in VEHICLE_CLASSES.values()
            }

            total_vehicles = sum(counts.values())
            density = classify_density(total_vehicles)

            density_counts[density] += 1
            total_vehicles_detected += total_vehicles

            print(f"\nFrame: {frame_number}")
            print(f"Vehicle counts: {counts}")
            print(f"Total vehicles: {total_vehicles}")
            print(f"Traffic density: {density}")

    finally:
        cap.release()

    print("\n========== TRAFFIC DENSITY SUMMARY ==========")
    print(f"Total video frames: {frame_number}")
    print(f"Sampled frames analyzed: {sampled_frames}")
    print(f"Total vehicle detections across sampled frames: {total_vehicles_detected}")

    for level in ("Low", "Medium", "High"):
        count = density_counts[level]
        percentage = (
            count / sampled_frames * 100
            if sampled_frames
            else 0
        )
        print(f"{level}: {count} frames ({percentage:.1f}%)")

    if sampled_frames:
        average = total_vehicles_detected / sampled_frames
        most_common = density_counts.most_common(1)[0][0]

        print(f"Average vehicles per sampled frame: {average:.1f}")
        print(f"Most common density: {most_common}")
    else:
        print("No sampled frames were analyzed.")

    print("Density analysis completed.")


if __name__ == "__main__":
    main()
