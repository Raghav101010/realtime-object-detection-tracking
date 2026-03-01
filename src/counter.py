# src/counter.py

class LineCounter:
    def __init__(self, line_position):
        self.line_position = line_position
        self.count_up = 0
        self.count_down = 0
        self.previous_positions = {}
 
    def update(self, tracked_objects):
        for obj in tracked_objects:
            track_id = obj["id"]
            x1, y1, x2, y2 = obj["bbox"]

            centroid_y = (y1 + y2) // 2

            if track_id not in self.previous_positions:
                self.previous_positions[track_id] = centroid_y
                continue

            prev_y = self.previous_positions[track_id]

            # Crossing downward
            if prev_y < self.line_position <= centroid_y:
                self.count_down += 1

            # Crossing upward
            elif prev_y > self.line_position >= centroid_y:
                self.count_up += 1

            self.previous_positions[track_id] = centroid_y

    def get_counts(self):
        return self.count_up, self.count_down