import cv2
from src.detector import Detector
from src.tracker import Tracker
from src.counter import LineCounter
from src.performance import PerformanceMonitor
from src.pipeline import Pipeline

def process_video(input_path, output_path, scale=0.5, frame_skip=3):

    cap = cv2.VideoCapture(input_path)
    fps_in = cap.get(cv2.CAP_PROP_FPS)

    detector = Detector()
    tracker = Tracker()
    performance = PerformanceMonitor()

    ret, frame = cap.read()
    if not ret:
        raise ValueError("Cannot read video")

    frame = cv2.resize(frame, None, fx=scale, fy=scale)
    height, width = frame.shape[:2]

    line_position = height // 2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps_in, (width, height))

    counter = LineCounter(line_position=line_position)
    pipeline = Pipeline(detector, tracker, counter, performance)

    frame_count = 0

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

            out.write(frame)

        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, None, fx=scale, fy=scale)

    cap.release()
    out.release()