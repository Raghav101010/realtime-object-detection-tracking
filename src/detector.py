# src/detector.py

from ultralytics import YOLO

# detector/model.py

from ultralytics import YOLO

class Detector:
    def __init__(self, model_path="yolov8n.pt", conf=0.4, classes=None):
        self.model = YOLO(model_path)
        self.conf = conf
        self.classes = classes

    def detect(self, frame):
        results = self.model(
            frame,
            conf=self.conf,
            classes=self.classes,
            verbose=False
        )[0]

        detections = []

        for box in results.boxes:
            #x1, y1, x2, y2 = box.xyxy[0]
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = self.model.names[cls]

            if label != "person":
                continue

            if conf < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append(
                ([int(x1), int(y1), int(x2-x1), int(y2-y1)], conf, cls)
            )

        return detections
