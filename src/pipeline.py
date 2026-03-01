# src/pipeline.py

import cv2


class Pipeline:
    def __init__(self, detector, tracker, counter, performance):
        self.detector = detector
        self.tracker = tracker
        self.counter = counter
        self.performance = performance

    def process_frame(self, frame):
        self.performance.start_inference_timer()
        detections = self.detector.detect(frame)
        self.performance.stop_inference_timer()

        tracked_objects = self.tracker.update(detections, frame)
        self.counter.update(tracked_objects)

        up, down = self.counter.get_counts()
        inf_fps = self.performance.get_inference_fps()
        disp_fps = self.performance.get_display_fps()

        return tracked_objects, up, down, inf_fps, disp_fps