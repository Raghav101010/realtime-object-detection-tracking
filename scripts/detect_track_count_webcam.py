from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import time

def main():
    model = YOLO("yolov8n.pt")

    tracker = DeepSort(
        max_age=30,
        n_init=3,
        max_iou_distance=0.7,
    )

    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), "Cannot open webcam"

    # Counting variables
    line_position = 300
    track_history = {}      # track_id -> previous center
    counted_ids = set()
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.4, classes=[0], verbose=False)[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            detections.append(([int(x1), int(y1), int(x2-x1), int(y2-y1)], conf, cls))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())

            # Compute center
            center_x = int((l + r) / 2)
            center_y = int((t + b) / 2)

            # Draw bounding box
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (l, t-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # Store previous position
            if track_id in track_history:
                prev_center = track_history[track_id]

                # Check line crossing (top → bottom)
                if (prev_center[1] < line_position and
                    center_y >= line_position and
                    track_id not in counted_ids):

                    count += 1
                    counted_ids.add(track_id)

            track_history[track_id] = (center_x, center_y)

            # Draw center point
            cv2.circle(frame, (center_x, center_y), 4, (0,0,255), -1)

        # Draw counting line
        cv2.line(frame, (0, line_position),
                 (frame.shape[1], line_position),
                 (255, 0, 0), 2)

        # Show count
        cv2.putText(frame, f"Count: {count}",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    3)

        cv2.imshow("Line Crossing Counter", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
