from ultralytics import YOLO
import cv2
import time

def main():
    model = YOLO("yolov8n.pt")  # nano = fastest for CPU

    cap = cv2.VideoCapture(0)  # default webcam
    assert cap.isOpened(), "Cannot open webcam"

    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO inference
        results = model(frame, conf=0.4, verbose=False)
        annotated_frame = results[0].plot()

        # FPS calculation (real-time)
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time

        cv2.putText(
            annotated_frame,
            f"FPS: {int(fps)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("YOLOv8 Webcam Detection", annotated_frame)

        # Graceful exit (ESC or q)
        if cv2.waitKey(1) & 0xFF in [27, ord("q")]:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
