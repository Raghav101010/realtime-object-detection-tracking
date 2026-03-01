# scripts/run_video.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
from src.detector import Detector
from src.tracker import Tracker
from src.counter import LineCounter
from src.performance import PerformanceMonitor
from src.pipeline import Pipeline

video_path = "data/videos/input_video3.mp4"
cap = cv2.VideoCapture(video_path)

fps_in = cap.get(cv2.CAP_PROP_FPS)
scale = 0.5  # 50% of original
frame_skip = 3
frame_count = 0

detector = Detector()
tracker = Tracker()
performance = PerformanceMonitor()
ret, frame = cap.read()
if not ret:
    raise ValueError("Cannot read video")
# Resize first frame
frame = cv2.resize(frame, None, fx=scale, fy=scale)
height, width = frame.shape[:2]
line_position = frame.shape[0] // 2
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    "outputs/videos/Bi_counted_output3.mp4",
    fourcc,
    fps_in,
    (width, height)
)
counter = LineCounter(line_position=line_position)
pipeline = Pipeline(detector, tracker, counter, performance)

while ret:

    frame_count += 1

    if frame_count % frame_skip == 0:

        tracked_objects, up, down, inf_fps, disp_fps = pipeline.process_frame(frame)

        for obj in tracked_objects:
            x1, y1, x2, y2 = obj["bbox"]
            track_id = obj["id"]

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, f"UP: {up}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, f"DOWN: {down}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.putText(frame, f"INF FPS: {inf_fps:.2f}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.putText(frame, f"DISP FPS: {disp_fps:.2f}", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.line(frame, (0, line_position),
                 (width, line_position),
                 (0, 0, 255), 3)

        out.write(frame)
        cv2.imshow("Video", frame)

    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, None, fx=scale, fy=scale)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()