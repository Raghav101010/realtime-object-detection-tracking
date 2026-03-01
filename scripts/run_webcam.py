# scripts/run_webcam.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
from src.detector import Detector
from src.tracker import Tracker
from src.counter import LineCounter
from src.performance import PerformanceMonitor
from src.pipeline import Pipeline

cap = cv2.VideoCapture(0)

detector = Detector()
tracker = Tracker()
counter = LineCounter(line_position=220)
performance = PerformanceMonitor()

pipeline = Pipeline(detector, tracker, counter, performance)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    tracked_objects, up, down, inf_fps, disp_fps = pipeline.process_frame(frame)

    for obj in tracked_objects:
        x1, y1, x2, y2 = obj["bbox"]
        track_id = obj["id"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(frame, f"UP: {up}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"DOWN: {down}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()