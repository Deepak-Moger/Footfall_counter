"""
FlowLens AI analytics engine.

The module keeps the original project purpose intact: detect, track, and count
people crossing a virtual line. It also exposes richer dashboard metrics that
make the project useful as a modern footfall intelligence product.
"""

from __future__ import annotations

import math
import random
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Deque, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

if TYPE_CHECKING:
    from ultralytics import YOLO


Point = Tuple[int, int]
Box = Tuple[int, int, int, int]


@dataclass
class Detection:
    """Person detection bounding box and confidence."""

    box: Box
    confidence: float


@dataclass
class TrackSnapshot:
    """A tracked person's current dashboard state."""

    track_id: int
    centroid: Point
    zone: str
    dwell_seconds: float
    status: str


@dataclass
class AnalyticsSnapshot:
    """Current state exported to the web dashboard and CLI."""

    entries: int = 0
    exits: int = 0
    occupancy: int = 0
    active_tracks: int = 0
    fps: float = 0.0
    confidence: float = 0.0
    line_position: float = 0.5
    line_y: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    processed_frames: int = 0
    uptime_seconds: float = 0.0
    peak_occupancy: int = 0
    average_dwell_seconds: float = 0.0
    conversion_rate: float = 0.0
    heatmap: List[Dict[str, float]] = field(default_factory=list)
    zones: List[Dict[str, float]] = field(default_factory=list)
    tracks: List[TrackSnapshot] = field(default_factory=list)
    events: List[Dict[str, str]] = field(default_factory=list)
    forecast: List[Dict[str, float]] = field(default_factory=list)
    ai_recommendations: List[str] = field(default_factory=list)
    alerts: List[Dict[str, str]] = field(default_factory=list)
    camera_health: str = "Idle"
    mode: str = "idle"
    source: str = ""
    timestamp: float = field(default_factory=time.time)


class CentroidTracker:
    """Simple centroid tracker with stable IDs and trajectory history."""

    def __init__(self, max_disappeared: int = 40, max_distance: int = 80) -> None:
        self.next_object_id = 0
        self.objects: Dict[int, Point] = {}
        self.disappeared: Dict[int, int] = {}
        self.first_seen: Dict[int, float] = {}
        self.last_seen: Dict[int, float] = {}
        self.trajectories: Dict[int, Deque[Point]] = defaultdict(lambda: deque(maxlen=42))
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid: Point) -> None:
        object_id = self.next_object_id
        now = time.time()
        self.objects[object_id] = centroid
        self.disappeared[object_id] = 0
        self.first_seen[object_id] = now
        self.last_seen[object_id] = now
        self.trajectories[object_id].append(centroid)
        self.next_object_id += 1

    def deregister(self, object_id: int) -> None:
        self.objects.pop(object_id, None)
        self.disappeared.pop(object_id, None)
        self.first_seen.pop(object_id, None)
        self.last_seen.pop(object_id, None)
        self.trajectories.pop(object_id, None)

    def update(self, detections: Iterable[Detection]) -> Dict[int, Point]:
        boxes = [detection.box for detection in detections]

        if not boxes:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(boxes), 2), dtype="int")
        for i, (x, y, w, h) in enumerate(boxes):
            input_centroids[i] = (int(x + w / 2.0), int(y + h / 2.0))

        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register((int(centroid[0]), int(centroid[1])))
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())
        distances = np.zeros((len(object_centroids), len(input_centroids)))

        for i, object_centroid in enumerate(object_centroids):
            for j, input_centroid in enumerate(input_centroids):
                distances[i, j] = np.linalg.norm(np.array(object_centroid) - input_centroid)

        rows = distances.min(axis=1).argsort()
        cols = distances.argmin(axis=1)[rows]
        used_rows = set()
        used_cols = set()
        now = time.time()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if distances[row, col] > self.max_distance:
                continue

            object_id = object_ids[row]
            centroid = (int(input_centroids[col][0]), int(input_centroids[col][1]))
            self.objects[object_id] = centroid
            self.disappeared[object_id] = 0
            self.last_seen[object_id] = now
            self.trajectories[object_id].append(centroid)
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(distances.shape[0])) - used_rows
        unused_cols = set(range(distances.shape[1])) - used_cols

        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        for col in unused_cols:
            centroid = (int(input_centroids[col][0]), int(input_centroids[col][1]))
            self.register(centroid)

        return self.objects


class FlowLensEngine:
    """Detection, tracking, counting, forecasting, and visualization service."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        line_position: float = 0.5,
        confidence_threshold: float = 0.5,
    ) -> None:
        self.model_path = model_path
        self.line_position = min(max(line_position, 0.05), 0.95)
        self.confidence_threshold = confidence_threshold
        self.model: Optional[YOLO] = None
        self.tracker = CentroidTracker(max_disappeared=40, max_distance=86)
        self.lock = threading.Lock()

        self.entry_count = 0
        self.exit_count = 0
        self.crossing_status: Dict[int, str] = {}
        self.counted_crossings: set[Tuple[int, str]] = set()
        self.events: Deque[Dict[str, str]] = deque(maxlen=12)
        self.heatmap_points: Deque[Dict[str, float]] = deque(maxlen=260)
        self.line_y = 0
        self.frame_width = 1280
        self.frame_height = 720
        self.last_frame: Optional[np.ndarray] = None
        self.last_detections: List[Detection] = []
        self.last_snapshot = AnalyticsSnapshot(line_position=self.line_position)
        self.last_frame_time = time.time()
        self.start_time = time.time()
        self.processed_frames = 0
        self.peak_occupancy = 0
        self.mode = "idle"
        self.source = ""

    def load_model(self) -> None:
        if self.model is None:
            from ultralytics import YOLO

            self.model = YOLO(self.model_path)

    def reset(self) -> None:
        with self.lock:
            self.entry_count = 0
            self.exit_count = 0
            self.crossing_status.clear()
            self.counted_crossings.clear()
            self.events.clear()
            self.heatmap_points.clear()
            self.tracker = CentroidTracker(max_disappeared=40, max_distance=86)
            self.processed_frames = 0
            self.peak_occupancy = 0
            self.start_time = time.time()

    def detect_people(self, frame: np.ndarray) -> List[Detection]:
        self.load_model()
        assert self.model is not None
        results = self.model(frame, classes=[0], verbose=False)
        detections: List[Detection] = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                if confidence < self.confidence_threshold:
                    continue
                detections.append(
                    Detection(
                        box=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                        confidence=confidence,
                    )
                )

        return detections

    def process_frame(self, frame: np.ndarray, source: str = "camera", mode: str = "live") -> np.ndarray:
        frame = frame.copy()
        self.frame_height, self.frame_width = frame.shape[:2]
        self.line_y = int(self.frame_height * self.line_position)

        detections = self.detect_people(frame)
        objects = self.tracker.update(detections)
        now = time.time()

        for object_id, centroid in objects.items():
            self._record_heatmap_point(centroid)
            self._check_line_crossing(object_id, centroid)

        fps = self._update_fps(now)
        occupancy = max(0, self.entry_count - self.exit_count)
        self.peak_occupancy = max(self.peak_occupancy, occupancy)
        overlay = self.draw_overlay(frame, detections, fps=fps)

        snapshot = self._build_snapshot(
            fps=fps,
            detections=detections,
            source=source,
            mode=mode,
        )

        with self.lock:
            self.last_frame = overlay
            self.last_detections = detections
            self.last_snapshot = snapshot
            self.processed_frames += 1
            self.mode = mode
            self.source = source

        return overlay

    def process_video_file(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        display: bool = False,
        display_width: Optional[int] = 1280,
    ) -> AnalyticsSnapshot:
        cap = cv2.VideoCapture(0 if video_path.lower() == "webcam" else video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        source_name = "webcam" if video_path.lower() == "webcam" else Path(video_path).name
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                overlay = self.process_frame(frame, source=source_name, mode="analysis")
                if writer:
                    writer.write(overlay)
                if display:
                    display_frame = overlay
                    if display_width and display_frame.shape[1] > display_width:
                        scale = display_width / display_frame.shape[1]
                        display_frame = cv2.resize(
                            display_frame,
                            (display_width, int(display_frame.shape[0] * scale)),
                        )
                    cv2.imshow("FlowLens AI", display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("r"):
                        self.reset()
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()

        return self.get_snapshot()

    def draw_overlay(self, frame: np.ndarray, detections: List[Detection], fps: float) -> np.ndarray:
        overlay = frame.copy()
        self._draw_counting_line(overlay)
        self._draw_detections(overlay, detections)
        self._draw_tracks(overlay)
        self._draw_hud(overlay, fps)
        return overlay

    def get_snapshot(self) -> AnalyticsSnapshot:
        with self.lock:
            return self.last_snapshot

    def get_jpeg(self) -> Optional[bytes]:
        with self.lock:
            if self.last_frame is None:
                return None
            ok, buffer = cv2.imencode(".jpg", self.last_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 84])
        if not ok:
            return None
        return buffer.tobytes()

    def get_placeholder_frame(self, width: int = 1280, height: int = 720) -> bytes:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            color = int(18 + y / height * 22)
            frame[y, :] = (color, color + 4, color + 10)
        cv2.putText(
            frame,
            "FlowLens AI",
            (48, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (230, 240, 245),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Waiting for video stream",
            (52, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (140, 155, 170),
            2,
            cv2.LINE_AA,
        )
        cv2.line(frame, (0, height // 2), (width, height // 2), (0, 229, 255), 2)
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        return buffer.tobytes() if ok else b""

    def build_demo_snapshot(self) -> AnalyticsSnapshot:
        """Animated synthetic data for the dashboard before a camera is started."""

        t = time.time()
        entries = 118 + int(16 * math.sin(t / 28))
        exits = 87 + int(11 * math.sin(t / 34 + 1.1))
        occupancy = max(0, entries - exits)
        heatmap = []
        for index in range(52):
            angle = t / 12 + index * 0.45
            x = 0.5 + 0.34 * math.sin(angle) + random.uniform(-0.025, 0.025)
            y = 0.5 + 0.24 * math.cos(angle * 0.8) + random.uniform(-0.025, 0.025)
            heatmap.append({"x": _clamp(x), "y": _clamp(y), "value": 0.45 + random.random() * 0.55})

        zones = [
            {"name": "Entrance", "occupancy": 18, "dwell": 31, "pressure": 0.78},
            {"name": "Aisle A", "occupancy": 9, "dwell": 47, "pressure": 0.42},
            {"name": "Checkout", "occupancy": 14, "dwell": 86, "pressure": 0.84},
            {"name": "Exit Lane", "occupancy": 6, "dwell": 22, "pressure": 0.28},
        ]
        forecast = self._build_forecast(occupancy, 24)
        recommendations = [
            "Open one assisted counter in the next 12 minutes.",
            "Entrance traffic is trending above the normal lunch window.",
            "Move a staff member to Checkout before queue pressure peaks.",
        ]
        alerts = [
            {"level": "warning", "title": "Queue risk", "message": "Checkout pressure is above 80%."},
            {"level": "info", "title": "Privacy mode", "message": "Only anonymous tracks are retained."},
        ]

        return AnalyticsSnapshot(
            entries=entries,
            exits=exits,
            occupancy=occupancy,
            active_tracks=7 + int(3 * math.sin(t / 9)),
            fps=28.4 + 2.2 * math.sin(t / 13),
            confidence=0.91,
            line_position=self.line_position,
            line_y=int(720 * self.line_position),
            frame_width=1280,
            frame_height=720,
            processed_frames=4200 + int(t) % 1000,
            uptime_seconds=3600 + int(t) % 1500,
            peak_occupancy=max(occupancy + 9, 42),
            average_dwell_seconds=52 + 8 * math.sin(t / 18),
            conversion_rate=0.72 + 0.04 * math.sin(t / 20),
            heatmap=heatmap,
            zones=zones,
            events=list(self.events) or [
                {"time": "now", "type": "Entry", "message": "Group of 2 entered"},
                {"time": "-1m", "type": "Exit", "message": "Single visitor exited"},
                {"time": "-3m", "type": "AI", "message": "Flow normalized near entrance"},
            ],
            forecast=forecast,
            ai_recommendations=recommendations,
            alerts=alerts,
            camera_health="Demo Mode",
            mode="demo",
            source="synthetic showroom",
            timestamp=t,
        )

    def _check_line_crossing(self, object_id: int, centroid: Point) -> None:
        _, cy = centroid
        current_position = "above" if cy < self.line_y else "below"
        previous_position = self.crossing_status.get(object_id)
        self.crossing_status[object_id] = current_position

        if previous_position is None or previous_position == current_position:
            return

        direction = "entry" if previous_position == "above" and current_position == "below" else "exit"
        crossing_key = (object_id, direction)
        if crossing_key in self.counted_crossings:
            return

        self.counted_crossings.add(crossing_key)
        if direction == "entry":
            self.entry_count += 1
            event_type = "Entry"
            message = f"Track {object_id} entered"
        else:
            self.exit_count += 1
            event_type = "Exit"
            message = f"Track {object_id} exited"

        self.events.appendleft(
            {
                "time": time.strftime("%H:%M:%S"),
                "type": event_type,
                "message": message,
            }
        )

    def _record_heatmap_point(self, centroid: Point) -> None:
        if self.frame_width <= 0 or self.frame_height <= 0:
            return
        x, y = centroid
        self.heatmap_points.append(
            {
                "x": _clamp(x / self.frame_width),
                "y": _clamp(y / self.frame_height),
                "value": 0.7,
            }
        )

    def _update_fps(self, now: float) -> float:
        elapsed = max(now - self.last_frame_time, 1e-6)
        instantaneous = 1.0 / elapsed
        self.last_frame_time = now
        if self.last_snapshot.fps <= 0:
            return min(instantaneous, 120.0)
        return self.last_snapshot.fps * 0.88 + min(instantaneous, 120.0) * 0.12

    def _build_snapshot(
        self,
        fps: float,
        detections: List[Detection],
        source: str,
        mode: str,
    ) -> AnalyticsSnapshot:
        occupancy = max(0, self.entry_count - self.exit_count)
        average_confidence = (
            sum(d.confidence for d in detections) / len(detections) if detections else 0.0
        )
        average_dwell = self._average_dwell()
        zones = self._build_zones()
        forecast = self._build_forecast(occupancy, len(self.tracker.objects))
        recommendations = self._build_recommendations(occupancy, zones, fps)
        alerts = self._build_alerts(occupancy, zones, fps)
        tracks = self._build_tracks()
        conversion_rate = self._conversion_rate()

        return AnalyticsSnapshot(
            entries=self.entry_count,
            exits=self.exit_count,
            occupancy=occupancy,
            active_tracks=len(self.tracker.objects),
            fps=fps,
            confidence=average_confidence,
            line_position=self.line_position,
            line_y=self.line_y,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            processed_frames=self.processed_frames,
            uptime_seconds=time.time() - self.start_time,
            peak_occupancy=self.peak_occupancy,
            average_dwell_seconds=average_dwell,
            conversion_rate=conversion_rate,
            heatmap=list(self.heatmap_points),
            zones=zones,
            tracks=tracks,
            events=list(self.events),
            forecast=forecast,
            ai_recommendations=recommendations,
            alerts=alerts,
            camera_health="Live" if fps > 5 else "Low FPS",
            mode=mode,
            source=source,
            timestamp=time.time(),
        )

    def _average_dwell(self) -> float:
        if not self.tracker.objects:
            return 0.0
        now = time.time()
        dwell_times = [now - self.tracker.first_seen.get(object_id, now) for object_id in self.tracker.objects]
        return sum(dwell_times) / len(dwell_times)

    def _conversion_rate(self) -> float:
        total_crossings = self.entry_count + self.exit_count
        if total_crossings == 0:
            return 0.0
        return min(1.0, self.entry_count / total_crossings)

    def _build_tracks(self) -> List[TrackSnapshot]:
        tracks: List[TrackSnapshot] = []
        now = time.time()
        for object_id, centroid in self.tracker.objects.items():
            dwell = now - self.tracker.first_seen.get(object_id, now)
            zone = self._zone_for_point(centroid)
            status = "Crossing" if abs(centroid[1] - self.line_y) < self.frame_height * 0.08 else "Moving"
            tracks.append(
                TrackSnapshot(
                    track_id=object_id,
                    centroid=centroid,
                    zone=zone,
                    dwell_seconds=dwell,
                    status=status,
                )
            )
        return tracks

    def _build_zones(self) -> List[Dict[str, float]]:
        names = ["Entrance", "Front Zone", "Middle Floor", "Exit Lane"]
        counts = {name: 0 for name in names}
        dwell = {name: [] for name in names}
        now = time.time()

        for object_id, centroid in self.tracker.objects.items():
            zone = self._zone_for_point(centroid)
            counts[zone] += 1
            dwell[zone].append(now - self.tracker.first_seen.get(object_id, now))

        zones: List[Dict[str, float]] = []
        for index, name in enumerate(names):
            occupancy = counts[name]
            average_dwell = sum(dwell[name]) / len(dwell[name]) if dwell[name] else 0.0
            pressure = min(1.0, occupancy / (3 + index * 2))
            zones.append(
                {
                    "name": name,
                    "occupancy": occupancy,
                    "dwell": average_dwell,
                    "pressure": pressure,
                }
            )
        return zones

    def _zone_for_point(self, centroid: Point) -> str:
        _, y = centroid
        if self.frame_height <= 0:
            return "Middle Floor"
        ratio = y / self.frame_height
        if ratio < 0.25:
            return "Entrance"
        if ratio < 0.5:
            return "Front Zone"
        if ratio < 0.75:
            return "Middle Floor"
        return "Exit Lane"

    def _build_forecast(self, occupancy: int, active_tracks: int) -> List[Dict[str, float]]:
        now = time.time()
        base = max(occupancy, active_tracks)
        trend = math.sin(now / 50) * 3 + max(0, self.entry_count - self.exit_count) * 0.04
        return [
            {"label": "+15m", "occupancy": max(0, round(base + trend + 2))},
            {"label": "+30m", "occupancy": max(0, round(base + trend * 1.4 + 4))},
            {"label": "+60m", "occupancy": max(0, round(base + trend * 2.1 + 7))},
        ]

    def _build_recommendations(
        self,
        occupancy: int,
        zones: List[Dict[str, float]],
        fps: float,
    ) -> List[str]:
        recommendations = []
        busiest = max(zones, key=lambda item: item["pressure"]) if zones else None
        if busiest and busiest["pressure"] > 0.65:
            recommendations.append(f"Shift staff toward {busiest['name']} before flow pressure increases.")
        if occupancy > 20:
            recommendations.append("Prepare queue support; occupancy is above the comfort threshold.")
        elif occupancy == 0:
            recommendations.append("Keep monitoring; the space is clear and ready for a new traffic wave.")
        else:
            recommendations.append("Traffic is stable; maintain current staffing and entrance flow.")
        if fps < 8:
            recommendations.append("Reduce stream resolution or use a smaller model to improve live accuracy.")
        recommendations.append("Privacy mode is active: only anonymous movement tracks are stored.")
        return recommendations[:3]

    def _build_alerts(
        self,
        occupancy: int,
        zones: List[Dict[str, float]],
        fps: float,
    ) -> List[Dict[str, str]]:
        alerts: List[Dict[str, str]] = []
        if fps and fps < 8:
            alerts.append(
                {
                    "level": "warning",
                    "title": "Low frame rate",
                    "message": "Live processing may miss fast crossings.",
                }
            )
        if occupancy > 30:
            alerts.append(
                {
                    "level": "critical",
                    "title": "Capacity pressure",
                    "message": "Occupancy is above the configured operating threshold.",
                }
            )
        crowded_zone = next((zone for zone in zones if zone["pressure"] > 0.8), None)
        if crowded_zone:
            alerts.append(
                {
                    "level": "warning",
                    "title": "Zone congestion",
                    "message": f"{crowded_zone['name']} needs attention.",
                }
            )
        if not alerts:
            alerts.append(
                {
                    "level": "info",
                    "title": "System normal",
                    "message": "Counts, tracks, and camera health are within range.",
                }
            )
        return alerts

    def _draw_counting_line(self, frame: np.ndarray) -> None:
        glow_color = (255, 229, 0)
        line_color = (255, 245, 190)
        cv2.line(frame, (0, self.line_y - 2), (self.frame_width, self.line_y - 2), glow_color, 2)
        cv2.line(frame, (0, self.line_y + 2), (self.frame_width, self.line_y + 2), (55, 242, 163), 2)
        cv2.line(frame, (0, self.line_y), (self.frame_width, self.line_y), line_color, 1)
        cv2.putText(
            frame,
            "SMART COUNTING LINE",
            (24, max(34, self.line_y - 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            line_color,
            2,
            cv2.LINE_AA,
        )

    def _draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> None:
        for detection in detections:
            x, y, w, h = detection.box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (53, 242, 163), 2)
            cv2.putText(
                frame,
                f"{detection.confidence:.0%}",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (225, 245, 242),
                1,
                cv2.LINE_AA,
            )

    def _draw_tracks(self, frame: np.ndarray) -> None:
        for object_id, centroid in self.tracker.objects.items():
            cx, cy = centroid
            trajectory = list(self.tracker.trajectories.get(object_id, []))
            for i in range(1, len(trajectory)):
                cv2.line(frame, trajectory[i - 1], trajectory[i], (255, 93, 115), 2)
            cv2.circle(frame, (cx, cy), 6, (0, 229, 255), -1)
            cv2.circle(frame, (cx, cy), 12, (0, 229, 255), 1)
            cv2.putText(
                frame,
                f"#{object_id}",
                (cx + 12, cy - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    def _draw_hud(self, frame: np.ndarray, fps: float) -> None:
        panel_height = 96
        panel = frame[:panel_height].copy()
        dark = np.zeros_like(panel)
        dark[:] = (10, 13, 20)
        blended = cv2.addWeighted(panel, 0.32, dark, 0.68, 0)
        frame[:panel_height] = blended

        metrics = [
            ("ENTRY", self.entry_count, (53, 242, 163)),
            ("EXIT", self.exit_count, (255, 184, 107)),
            ("INSIDE", max(0, self.entry_count - self.exit_count), (0, 229, 255)),
            ("FPS", f"{fps:.1f}", (231, 236, 242)),
        ]
        x = 24
        for label, value, color in metrics:
            cv2.putText(
                frame,
                label,
                (x, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (150, 164, 180),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                str(value),
                (x, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.02,
                color,
                2,
                cv2.LINE_AA,
            )
            x += 172


def snapshot_to_dict(snapshot: AnalyticsSnapshot) -> Dict[str, object]:
    """Serialize dataclass snapshots without leaking numpy values."""

    return {
        "entries": snapshot.entries,
        "exits": snapshot.exits,
        "occupancy": snapshot.occupancy,
        "activeTracks": snapshot.active_tracks,
        "fps": round(snapshot.fps, 1),
        "confidence": round(snapshot.confidence, 3),
        "linePosition": snapshot.line_position,
        "lineY": snapshot.line_y,
        "frameWidth": snapshot.frame_width,
        "frameHeight": snapshot.frame_height,
        "processedFrames": snapshot.processed_frames,
        "uptimeSeconds": round(snapshot.uptime_seconds, 1),
        "peakOccupancy": snapshot.peak_occupancy,
        "averageDwellSeconds": round(snapshot.average_dwell_seconds, 1),
        "conversionRate": round(snapshot.conversion_rate, 3),
        "heatmap": snapshot.heatmap,
        "zones": [
            {
                "name": zone["name"],
                "occupancy": int(zone["occupancy"]),
                "dwell": round(float(zone["dwell"]), 1),
                "pressure": round(float(zone["pressure"]), 3),
            }
            for zone in snapshot.zones
        ],
        "tracks": [
            {
                "id": track.track_id,
                "x": track.centroid[0],
                "y": track.centroid[1],
                "zone": track.zone,
                "dwellSeconds": round(track.dwell_seconds, 1),
                "status": track.status,
            }
            for track in snapshot.tracks
        ],
        "events": snapshot.events,
        "forecast": snapshot.forecast,
        "aiRecommendations": snapshot.ai_recommendations,
        "alerts": snapshot.alerts,
        "cameraHealth": snapshot.camera_health,
        "mode": snapshot.mode,
        "source": snapshot.source,
        "timestamp": snapshot.timestamp,
    }


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(max(value, low), high)
