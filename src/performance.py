# src/performance.py

import time


class PerformanceMonitor:
    def __init__(self):
        self.prev_time = None
        self.inference_time = 0
        self.inference_count = 0

    def start_inference_timer(self):
        self._start = time.time()

    def stop_inference_timer(self):
        elapsed = time.time() - self._start
        self.inference_time += elapsed
        self.inference_count += 1

    def get_inference_fps(self):
        if self.inference_time == 0:
            return 0
        return self.inference_count / self.inference_time

    def get_display_fps(self):
        curr_time = time.time()
        if self.prev_time is None:
            self.prev_time = curr_time
            return 0

        fps = 1 / (curr_time - self.prev_time)
        self.prev_time = curr_time
        return fps