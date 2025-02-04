import numpy as np

class IOUTracker:
    def __init__(self, iou_threshold=0.7):
        # 1. Initialize the IOU threshold and required variables like trackers and next_id.
        self.next_id=0
        pass

    def _compute_iou(self, box1, box2):
        """
        2. Implement the IOU computation between two bounding boxes.
        - This involves calculating the intersection and union areas of the boxes.
        - Return the IOU score (intersection / union).
        """
        pass

    def update(self, detections):
        """
        3. Implement the update method to update the trackers with new detections.
        - Perform greedy matching using IOU: compare each tracker with new detections.
        - If IOU > threshold, match the detection to the tracker.
        - If no match is found for a detection, create a new tracker.
        - Remove trackers that don't match any detection from the previous frame.
        - Input Format : [xmax,ymin,xmax,ymax,score,class]
        - Return the updated list of trackers in the format: [xmin, ymin, xmax, ymax, track id, class, score].
        """
        pass