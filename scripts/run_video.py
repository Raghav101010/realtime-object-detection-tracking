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

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_in = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("outputs/videos/Bi_counted_output3.mp4", fourcc, fps_in, (width, height))

detector = Detector()
tracker = Tracker()
performance = PerformanceMonitor()
ret, frame = cap.read()
if not ret:
    raise ValueError("Cannot read video")

line_position = frame.shape[0] // 2
counter = LineCounter(line_position=line_position)
pipeline = Pipeline(detector, tracker, counter, performance)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    tracked_objects, up, down, inf_fps, disp_fps = pipeline.process_frame(frame)

    for obj in tracked_objects:
        x1, y1, x2, y2 = obj["bbox"]
        track_id = obj["id"]

        # Calculate centroid
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
    
    cv2.line(
    frame,
    (0, line_position),
    (frame.shape[1], line_position),
    (0, 0, 255),3)
    

    out.write(frame)
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()