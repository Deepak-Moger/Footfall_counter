"""Verify that FlowLens AI can run locally."""

from __future__ import annotations

import sys

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI
from ultralytics import YOLO

from core.analytics import FlowLensEngine, snapshot_to_dict


def main() -> int:
    print("=" * 64)
    print("FlowLens AI - Installation Test")
    print("=" * 64)

    print(f"\nPython: {sys.version.split()[0]}")
    if sys.version_info < (3, 8):
        raise RuntimeError("Python 3.8+ is required.")

    print(f"OpenCV: {cv2.__version__}")
    print(f"FastAPI: {FastAPI.__module__.split('.')[0]}")
    print(f"Uvicorn: {uvicorn.__version__}")

    print("\nLoading YOLO model...")
    YOLO("yolov8n.pt")
    print("Model loaded.")

    print("\nTesting analytics engine on a synthetic frame...")
    frame = np.zeros((480, 854, 3), dtype=np.uint8)
    cv2.rectangle(frame, (390, 150), (470, 420), (255, 255, 255), -1)
    engine = FlowLensEngine(model_path="yolov8n.pt")
    engine.process_frame(frame, source="synthetic test", mode="test")
    payload = snapshot_to_dict(engine.get_snapshot())

    required_keys = {"entries", "exits", "occupancy", "heatmap", "forecast", "aiRecommendations"}
    missing = required_keys - payload.keys()
    if missing:
        raise RuntimeError(f"Missing dashboard metrics: {sorted(missing)}")

    print("Dashboard metrics available.")
    print("\nAll tests passed.")
    print("\nRun the app:")
    print("  python footfall_counter.py dashboard")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

