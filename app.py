"""FlowLens AI web application."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.analytics import FlowLensEngine, snapshot_to_dict


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
SAMPLE_VIDEO = ROOT / "test_video3.mp4"

app = FastAPI(title="FlowLens AI", version="2.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

engine = FlowLensEngine(model_path=str(ROOT / "yolov8n.pt"))


class StreamRequest(BaseModel):
    source: str = "sample"
    loop: bool = True


class StreamController:
    """Owns the background video capture loop for the web dashboard."""

    def __init__(self, analytics_engine: FlowLensEngine) -> None:
        self.engine = analytics_engine
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.current_source = "demo"
        self.running = False
        self.error: Optional[str] = None

    def start(self, source: str, loop: bool = True) -> None:
        with self.lock:
            self.stop()
            self.stop_event.clear()
            self.error = None
            resolved_source = self._resolve_source(source)
            self.current_source = source
            self.running = True
            self.engine.reset()
            self.thread = threading.Thread(
                target=self._run,
                args=(resolved_source, source, loop),
                daemon=True,
            )
            self.thread.start()

    def stop(self) -> None:
        if self.thread and self.thread.is_alive():
            self.stop_event.set()
            self.thread.join(timeout=2.5)
        self.running = False
        self.thread = None

    def status(self) -> dict[str, object]:
        return {
            "running": self.running,
            "source": self.current_source,
            "error": self.error,
        }

    def _resolve_source(self, source: str) -> str | int:
        normalized = source.strip().lower()
        if normalized in {"webcam", "camera", "0"}:
            return 0
        if normalized in {"sample", "video", "test"}:
            if not SAMPLE_VIDEO.exists():
                raise HTTPException(status_code=404, detail="Sample video not found.")
            return str(SAMPLE_VIDEO)

        video_path = Path(source)
        if not video_path.is_absolute():
            video_path = ROOT / video_path
        if not video_path.exists():
            raise HTTPException(status_code=404, detail=f"Video not found: {source}")
        return str(video_path)

    def _run(self, resolved_source: str | int, label: str, loop: bool) -> None:
        cap = cv2.VideoCapture(resolved_source)
        if not cap.isOpened():
            self.error = "Could not open video source."
            self.running = False
            return

        source_name = "webcam" if isinstance(resolved_source, int) else Path(str(resolved_source)).name
        try:
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    if loop and not isinstance(resolved_source, int):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break

                self.engine.process_frame(frame, source=source_name, mode=label)
                time.sleep(0.001)
        except Exception as exc:  # pragma: no cover - surfaced to UI
            self.error = str(exc)
        finally:
            cap.release()
            self.running = False


controller = StreamController(engine)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/metrics")
def metrics() -> JSONResponse:
    if controller.running or engine.get_jpeg():
        payload = snapshot_to_dict(engine.get_snapshot())
    else:
        payload = snapshot_to_dict(engine.build_demo_snapshot())
    payload["stream"] = controller.status()
    return JSONResponse(payload)


@app.post("/api/start")
def start_stream(request: StreamRequest) -> JSONResponse:
    controller.start(request.source, loop=request.loop)
    return JSONResponse({"ok": True, **controller.status()})


@app.post("/api/stop")
def stop_stream() -> JSONResponse:
    controller.stop()
    return JSONResponse({"ok": True, **controller.status()})


@app.post("/api/reset")
def reset() -> JSONResponse:
    engine.reset()
    return JSONResponse({"ok": True})


@app.get("/api/status")
def status() -> JSONResponse:
    return JSONResponse(controller.status())


@app.get("/video")
def video_feed() -> StreamingResponse:
    return StreamingResponse(_mjpeg_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


def _mjpeg_frames():
    while True:
        frame = engine.get_jpeg() or engine.get_placeholder_frame()
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.08)


@app.on_event("shutdown")
def shutdown_event() -> None:
    controller.stop()

