import cv2
import time
from ultralytics import YOLO

# ================= CONFIG =================
MODEL_PATH = "yolov8n.pt"          # or yolov8s.pt for better accuracy
INPUT_VIDEO = "data/videos/input.mp4"
OUTPUT_VIDEO = "outputs/videos/output.mp4"
CONF_THRESHOLD = 0.4
# =========================================

# Load YOLO model
model = YOLO(MODEL_PATH)

# Open video
cap = cv2.VideoCapture(INPUT_VIDEO)
assert cap.isOpened(), "Error opening video file"

# Video properties
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_in = cap.get(cv2.CAP_PROP_FPS)

# Video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

import os
os.makedirs("outputs/videos", exist_ok=True)

out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps_in, (width, height))

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO inference
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)

    # Draw detections
    annotated_frame = results[0].plot()

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    # Display FPS
    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Show & save
    cv2.imshow("YOLO Video Detection", annotated_frame)
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Video processing complete")
print(f"📁 Saved as: {OUTPUT_VIDEO}")
