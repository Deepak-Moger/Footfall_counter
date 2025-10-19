"""
Footfall Counter using Computer Vision
Detects, tracks, and counts people crossing a virtual line
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
from typing import Dict, Tuple, List
import time


class CentroidTracker:
    """
    Simple centroid-based object tracker
    Tracks objects based on Euclidean distance between centroids
    """

    def __init__(self, max_disappeared=30, max_distance=50):
        self.next_object_id = 0
        self.objects = {}  # objectID: centroid
        self.disappeared = {}  # objectID: frames_disappeared
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.trajectories = defaultdict(lambda: deque(maxlen=30))

    def register(self, centroid):
        """Register a new object with the next available ID"""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.trajectories[self.next_object_id].append(centroid)
        self.next_object_id += 1

    def deregister(self, object_id):
        """Remove an object that has disappeared"""
        del self.objects[object_id]
        del self.disappeared[object_id]
        if object_id in self.trajectories:
            del self.trajectories[object_id]

    def update(self, detections):
        """
        Update tracked objects with new detections
        detections: list of (x, y, w, h) bounding boxes
        """
        # If no detections, mark all objects as disappeared
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # Calculate centroids from detections
        input_centroids = np.zeros((len(detections), 2), dtype="int")
        for i, (x, y, w, h) in enumerate(detections):
            cx = int(x + w / 2.0)
            cy = int(y + h / 2.0)
            input_centroids[i] = (cx, cy)

        # If no objects are being tracked, register all
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            # Get current tracked objects
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Compute distance between each pair of centroids
            D = np.zeros((len(object_centroids), len(input_centroids)))
            for i, oc in enumerate(object_centroids):
                for j, ic in enumerate(input_centroids):
                    D[i, j] = np.linalg.norm(np.array(oc) - np.array(ic))

            # Find minimum distances
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                # Check if distance is within threshold
                if D[row, col] > self.max_distance:
                    continue

                # Update the centroid
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                self.trajectories[object_id].append(tuple(input_centroids[col]))

                used_rows.add(row)
                used_cols.add(col)

            # Check which objects have disappeared
            unused_rows = set(range(D.shape[0])) - used_rows
            unused_cols = set(range(D.shape[1])) - used_cols

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # Register new objects
            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects


class FootfallCounter:
    """
    Main footfall counter class
    Handles detection, tracking, and counting
    """

    def __init__(self, model_path='yolov8n.pt', line_position=0.5, display_width=1280):
        """
        Initialize footfall counter

        Args:
            model_path: Path to YOLO model
            line_position: Relative position of counting line (0-1)
            display_width: Target width for display window (None for original size)
        """
        print("Loading YOLO model...")
        self.model = YOLO(model_path)
        self.tracker = CentroidTracker(max_disappeared=40, max_distance=80)

        self.line_position = line_position
        self.line_y = None
        self.display_width = display_width

        # Counting variables
        self.counted_ids = set()
        self.entry_count = 0
        self.exit_count = 0
        self.crossing_status = {}  # objectID: 'above' or 'below'

        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

    def set_line_position(self, frame_height):
        """Set the counting line position based on frame height"""
        self.line_y = int(frame_height * self.line_position)

    def detect_people(self, frame):
        """
        Detect people in the frame using YOLO

        Returns:
            List of bounding boxes (x, y, w, h)
        """
        results = self.model(frame, classes=[0], verbose=False)  # class 0 = person
        detections = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()

                # Filter by confidence
                if conf > 0.5:
                    x, y = int(x1), int(y1)
                    w, h = int(x2 - x1), int(y2 - y1)
                    detections.append((x, y, w, h))

        return detections

    def check_line_crossing(self, object_id, centroid):
        """
        Check if an object has crossed the counting line

        Args:
            object_id: Unique ID of tracked object
            centroid: Current centroid position (x, y)
        """
        cx, cy = centroid

        # Determine current position relative to line
        current_position = 'above' if cy < self.line_y else 'below'

        # Check if this is a new object
        if object_id not in self.crossing_status:
            self.crossing_status[object_id] = current_position
            return

        # Get previous position
        previous_position = self.crossing_status[object_id]

        # Check for crossing
        if previous_position != current_position and object_id not in self.counted_ids:
            if previous_position == 'above' and current_position == 'below':
                # Crossed from top to bottom (Entry)
                self.entry_count += 1
                self.counted_ids.add(object_id)
                print(f"Entry detected! ID: {object_id}, Total entries: {self.entry_count}")
            elif previous_position == 'below' and current_position == 'above':
                # Crossed from bottom to top (Exit)
                self.exit_count += 1
                self.counted_ids.add(object_id)
                print(f"Exit detected! ID: {object_id}, Total exits: {self.exit_count}")

        # Update position
        self.crossing_status[object_id] = current_position

    def draw_info(self, frame, detections):
        """Draw bounding boxes, tracking info, and counters"""
        # Draw counting line
        cv2.line(frame, (0, self.line_y), (frame.shape[1], self.line_y),
                (0, 255, 255), 3)
        cv2.putText(frame, "COUNTING LINE", (10, self.line_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Draw bounding boxes and trajectories
        for (x, y, w, h) in detections:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw tracked centroids and IDs
        for object_id, centroid in self.tracker.objects.items():
            cx, cy = centroid

            # Draw centroid
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # Draw trajectory
            if object_id in self.tracker.trajectories:
                trajectory = list(self.tracker.trajectories[object_id])
                for i in range(1, len(trajectory)):
                    if trajectory[i - 1] is None or trajectory[i] is None:
                        continue
                    cv2.line(frame, trajectory[i - 1], trajectory[i],
                            (255, 0, 255), 2)

            # Draw ID
            text = f"ID {object_id}"
            cv2.putText(frame, text, (cx - 20, cy - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Draw counter panel
        panel_height = 150
        panel = np.zeros((panel_height, frame.shape[1], 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)

        # Entry count
        cv2.putText(panel, f"ENTRIES: {self.entry_count}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # Exit count
        cv2.putText(panel, f"EXITS: {self.exit_count}", (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 100, 255), 3)

        # Total count
        total = self.entry_count - self.exit_count
        cv2.putText(panel, f"CURRENT: {total}", (400, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

        # FPS
        cv2.putText(panel, f"FPS: {self.fps:.1f}", (400, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # Active tracks
        cv2.putText(panel, f"TRACKING: {len(self.tracker.objects)}", (700, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 100), 2)

        # Combine panel and frame
        output = np.vstack([panel, frame])
        return output

    def update_fps(self):
        """Update FPS calculation"""
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            elapsed = time.time() - self.start_time
            self.fps = 30 / elapsed
            self.start_time = time.time()

    def process_video(self, video_path, output_path=None):
        """
        Process video and count footfall

        Args:
            video_path: Path to input video or 'webcam' for live camera
            output_path: Path to save output video (optional)
        """
        # Check if using webcam
        if video_path.lower() == 'webcam':
            cap = cv2.VideoCapture(0)
            is_webcam = True
            print("Using webcam...")
        else:
            cap = cv2.VideoCapture(video_path)
            is_webcam = False

        if not cap.isOpened():
            print(f"Error: Could not open video source")
            return

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) if not is_webcam else 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else 0

        if is_webcam:
            print(f"Webcam started: {frame_width}x{frame_height}")
            print("Press 'q' to quit, 'r' to reset counters")
        else:
            print(f"Video loaded: {frame_width}x{frame_height} @ {fps} FPS")
            print(f"Total frames: {total_frames}")

        # Set counting line position
        self.set_line_position(frame_height)

        # Setup video writer if output path is provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps,
                                 (frame_width, frame_height + 150))
            print(f"Output will be saved to: {output_path}")

        frame_number = 0

        print("\nProcessing video... Press 'q' to quit, 'p' to pause, 'r' to reset counters")

        while True:
            ret, frame = cap.read()

            if not ret:
                if is_webcam:
                    print("\nWebcam error")
                    break
                else:
                    print("\nEnd of video reached")
                    break

            frame_number += 1

            # Detect people
            detections = self.detect_people(frame)

            # Update tracker
            objects = self.tracker.update(detections)

            # Check for line crossings
            for object_id, centroid in objects.items():
                self.check_line_crossing(object_id, centroid)

            # Update FPS
            self.update_fps()

            # Draw visualization
            output_frame = self.draw_info(frame, detections)

            # Save frame if output writer is set
            if out:
                out.write(output_frame)

            # Resize for display if needed
            display_frame = output_frame
            if self.display_width and output_frame.shape[1] > self.display_width:
                aspect_ratio = output_frame.shape[0] / output_frame.shape[1]
                display_height = int(self.display_width * aspect_ratio)
                display_frame = cv2.resize(output_frame, (self.display_width, display_height))

            # Display
            cv2.imshow('Footfall Counter', display_frame)

            # Progress indicator
            if not is_webcam and frame_number % 30 == 0:
                progress = (frame_number / total_frames) * 100
                print(f"Progress: {progress:.1f}% | "
                      f"Entries: {self.entry_count} | "
                      f"Exits: {self.exit_count} | "
                      f"Current: {self.entry_count - self.exit_count}")

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nStopped by user")
                break
            elif key == ord('p'):
                print("Paused. Press any key to continue...")
                cv2.waitKey(0)
            elif key == ord('r'):
                # Reset counters
                self.entry_count = 0
                self.exit_count = 0
                self.counted_ids.clear()
                self.crossing_status.clear()
                print("Counters reset!")

        # Cleanup
        cap.release()
        if out:
            out.release()
            print(f"\n✓ Video saved successfully: {output_path}")
        cv2.destroyAllWindows()

        # Print final statistics
        print("\n" + "="*50)
        print("FINAL STATISTICS")
        print("="*50)
        print(f"Total Entries: {self.entry_count}")
        print(f"Total Exits: {self.exit_count}")
        print(f"Current Count: {self.entry_count - self.exit_count}")
        print(f"Frames Processed: {frame_number}")
        if output_path:
            print(f"Output Video: {output_path}")
        print("="*50)


def main():
    """Main function to run the footfall counter"""
    import argparse

    parser = argparse.ArgumentParser(description='Footfall Counter using Computer Vision')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to input video file or "webcam"')
    parser.add_argument('--output', type=str, default='footfall_output.mp4',
                       help='Path to save output video (default: footfall_output.mp4, use "none" to disable)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='Path to YOLO model (default: yolov8n.pt)')
    parser.add_argument('--line', type=float, default=0.5,
                       help='Counting line position (0-1, default: 0.5)')
    parser.add_argument('--display-width', type=int, default=1280,
                       help='Display window width in pixels (default: 1280, use 0 for original size)')

    args = parser.parse_args()

    # Create footfall counter
    display_width = None if args.display_width == 0 else args.display_width
    counter = FootfallCounter(model_path=args.model, line_position=args.line, display_width=display_width)

    # Determine output path
    output_path = None if args.output.lower() == 'none' else args.output

    # Process video
    counter.process_video(args.video, output_path)


if __name__ == "__main__":
    main()
