from ultralytics import YOLO
import cv2
import os

# Paths
IMAGE_PATH = "data/images/input.jpg"
OUTPUT_DIR = "outputs/images"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "output.jpg")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    # Load YOLOv8 pretrained model
    model = YOLO("yolov8n.pt")  # nano model (fast, CPU-friendly)

    # Run detection
    results = model(IMAGE_PATH)

    # Load original image
    image = cv2.imread(IMAGE_PATH)

    # Process detections
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = f"{model.names[class_id]} {confidence:.2f}"

            # Draw bounding box
            if confidence > 0.6:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label
                cv2.putText(
                    image,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (0, 25, 155),
                    2
                )

    # Save output image
    cv2.imwrite(OUTPUT_PATH, image)
    print(f"✅ Output saved at: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
