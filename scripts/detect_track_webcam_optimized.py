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

    # -------- Optimization parameters --------
    FRAME_SKIP = 2          # run YOLO every 2 frames
    YOLO_WIDTH = 640        # resize width for YOLO
    # -----------------------------------------

    frame_count = 0
    prev_time = 0

    total_inference_time = 0.0
    inference_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        original_h, original_w = frame.shape[:2]

        detections = []

        # -------- Frame skipping logic --------
        if frame_count % FRAME_SKIP == 0:
            # Resize frame for YOLO
            frame_small = cv2.resize(
                frame,
                (YOLO_WIDTH, int(original_h * YOLO_WIDTH / original_w))
            )

            scale_x = original_w / frame_small.shape[1]
            scale_y = original_h / frame_small.shape[0]

            start_inf = time.time()
            results = model(frame_small, conf=0.4, verbose=False)[0]
            end_inf = time.time()

            total_inference_time += (end_inf - start_inf)
            inference_count += 1

            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0]

                # Scale boxes back to original frame size
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)

                conf = float(box.conf[0])
                cls = int(box.cls[0])

                detections.append(
                    ([x1, y1, x2 - x1, y2 - y1], conf, cls)
                )
        # --------------------------------------

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())

            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {track_id}",
                (l, t - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # -------- Inference FPS (model-bound) calculation --------    
        inference_fps = (
        inference_count / total_inference_time
        if total_inference_time > 0 else 0
        )

        cv2.putText(
            frame,
            f"Inference FPS: {inference_fps:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )

        # -------- Display FPS (system-bound) calculation --------
        curr_time = time.time()
        display_fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time

        cv2.putText(
            frame,
            f"Display FPS: {int(display_fps)}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("YOLO + DeepSORT Tracking (Optimized)", frame)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
