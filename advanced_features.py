"""
Advanced footfall counter with additional features
Includes region-based counting, heatmaps, and analytics
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
from typing import Dict, Tuple, List, Optional
import time
from datetime import datetime
import json


class PolygonROI:
    """
    Polygon-based Region of Interest for counting
    Supports complex shapes beyond simple lines
    """

    def __init__(self, points: List[Tuple[int, int]]):
        """
        Initialize polygon ROI

        Args:
            points: List of (x, y) coordinates defining the polygon
        """
        self.points = np.array(points, dtype=np.int32)
        self.centroid = self.calculate_centroid()

    def calculate_centroid(self):
        """Calculate the centroid of the polygon"""
        return tuple(np.mean(self.points, axis=0).astype(int))

    def contains_point(self, point: Tuple[int, int]) -> bool:
        """Check if a point is inside the polygon"""
        result = cv2.pointPolygonTest(self.points, point, False)
        return result >= 0

    def draw(self, frame, color=(0, 255, 255), thickness=2):
        """Draw the polygon on the frame"""
        cv2.polylines(frame, [self.points], True, color, thickness)


class AdvancedFootfallCounter:
    """
    Advanced footfall counter with multiple ROIs and analytics
    """

    def __init__(self, model_path='yolov8n.pt'):
        print("Loading YOLO model...")
        self.model = YOLO(model_path)

        # ROI and counting setup
        self.rois = {}  # name: PolygonROI
        self.line_rois = {}  # name: (y_position, direction)

        # Tracking
        self.tracks = {}  # track_id: {'positions': deque, 'counted': set, 'last_seen': frame}
        self.next_track_id = 0
        self.max_distance = 80

        # Counting
        self.entry_count = 0
        self.exit_count = 0
        self.roi_counts = defaultdict(int)

        # Analytics
        self.heatmap = None
        self.timestamps = []
        self.hourly_counts = defaultdict(int)

        # Performance
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

    def add_line_roi(self, name: str, y_position: int, direction='vertical'):
        """Add a line-based ROI for counting"""
        self.line_rois[name] = (y_position, direction)
        print(f"Added line ROI: {name} at y={y_position}")

    def add_polygon_roi(self, name: str, points: List[Tuple[int, int]]):
        """Add a polygon-based ROI for counting"""
        self.rois[name] = PolygonROI(points)
        print(f"Added polygon ROI: {name}")

    def detect_and_track(self, frame):
        """Detect and track people in the frame"""
        results = self.model.track(frame, classes=[0], persist=True, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes.id is not None:
                for box, track_id in zip(boxes, boxes.id):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()

                    if conf > 0.5:
                        x, y = int(x1), int(y1)
                        w, h = int(x2 - x1), int(y2 - y1)
                        cx, cy = int(x + w/2), int(y + h/2)

                        track_id = int(track_id.cpu().numpy())
                        detections.append({
                            'bbox': (x, y, w, h),
                            'centroid': (cx, cy),
                            'track_id': track_id,
                            'conf': conf
                        })

        return detections

    def update_heatmap(self, frame_shape, centroids):
        """Update heatmap with current centroids"""
        if self.heatmap is None:
            self.heatmap = np.zeros(frame_shape[:2], dtype=np.float32)

        for cx, cy in centroids:
            cv2.circle(self.heatmap, (cx, cy), 20, 1, -1)

    def draw_analytics(self, frame):
        """Draw analytics overlay"""
        # Draw heatmap overlay
        if self.heatmap is not None:
            heatmap_normalized = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
            heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_JET)
            frame = cv2.addWeighted(frame, 0.7, heatmap_colored, 0.3, 0)

        return frame

    def generate_report(self, output_path='footfall_report.json'):
        """Generate analytics report"""
        report = {
            'summary': {
                'total_entries': self.entry_count,
                'total_exits': self.exit_count,
                'net_count': self.entry_count - self.exit_count,
                'frames_processed': self.frame_count
            },
            'roi_counts': dict(self.roi_counts),
            'timestamp': datetime.now().isoformat()
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Report saved to: {output_path}")
        return report


def create_sample_video():
    """
    Create a simple test video with moving rectangles simulating people
    Useful for testing when you don't have a real video
    """
    print("Creating sample test video...")

    width, height = 640, 480
    fps = 30
    duration = 10  # seconds

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('sample_test.mp4', fourcc, fps, (width, height))

    class Person:
        def __init__(self, x, y, vx, vy):
            self.x = x
            self.y = y
            self.vx = vx
            self.vy = vy
            self.w = 40
            self.h = 80

        def update(self):
            self.x += self.vx
            self.y += self.vy

            # Bounce off edges
            if self.x < 0 or self.x > width - self.w:
                self.vx *= -1
            if self.y < 0 or self.y > height - self.h:
                self.vy *= -1

        def draw(self, frame):
            cv2.rectangle(frame, (int(self.x), int(self.y)),
                         (int(self.x + self.w), int(self.y + self.h)),
                         (0, 255, 0), -1)

    # Create some "people"
    people = [
        Person(100, 50, 2, 3),
        Person(300, 400, -1, -2),
        Person(500, 200, -2, 1),
        Person(200, 300, 1, -1),
    ]

    for frame_num in range(fps * duration):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw counting line
        line_y = height // 2
        cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), 2)

        # Update and draw people
        for person in people:
            person.update()
            person.draw(frame)

        out.write(frame)

    out.release()
    print("Sample video created: sample_test.mp4")
    print("You can now test with: python footfall_counter.py --video sample_test.mp4")


if __name__ == "__main__":
    create_sample_video()
