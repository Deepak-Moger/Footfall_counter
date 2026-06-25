"""Command-line entry point for FlowLens AI."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from core.analytics import FlowLensEngine


ROOT = Path(__file__).resolve().parent
COMMANDS = {"dashboard", "video"}


def run_dashboard(host: str, port: int) -> int:
    env_port = os.getenv("PORT")
    resolved_port = int(env_port) if env_port else port
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "app:app",
        "--host",
        host,
        "--port",
        str(resolved_port),
    ]
    return subprocess.call(command, cwd=ROOT)


def run_video(args: argparse.Namespace) -> int:
    output = None if args.output.lower() == "none" else args.output
    engine = FlowLensEngine(
        model_path=args.model,
        line_position=args.line,
        confidence_threshold=args.confidence,
    )
    snapshot = engine.process_video_file(
        video_path=args.video,
        output_path=output,
        display=args.display,
        display_width=None if args.display_width == 0 else args.display_width,
    )
    print("\nFlowLens AI final statistics")
    print("=" * 34)
    print(f"Entries: {snapshot.entries}")
    print(f"Exits: {snapshot.exits}")
    print(f"Current occupancy: {snapshot.occupancy}")
    print(f"Peak occupancy: {snapshot.peak_occupancy}")
    print(f"Average dwell: {snapshot.average_dwell_seconds:.1f}s")
    print(f"Frames processed: {snapshot.processed_frames}")
    if output:
        print(f"Output video: {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FlowLens AI footfall intelligence")
    subparsers = parser.add_subparsers(dest="command")

    dashboard = subparsers.add_parser("dashboard", help="Launch the premium web dashboard")
    dashboard.add_argument("--host", default="127.0.0.1", help="Dashboard host")
    dashboard.add_argument("--port", default=8000, type=int, help="Dashboard port")

    video = subparsers.add_parser("video", help="Process a webcam or video file from the CLI")
    video.add_argument("--video", required=True, help='Path to video file or "webcam"')
    video.add_argument("--output", default="footfall_output.mp4", help='Output MP4 path or "none"')
    video.add_argument("--model", default=str(ROOT / "yolov8n.pt"), help="YOLO model path")
    video.add_argument("--line", default=0.5, type=float, help="Counting line position from 0 to 1")
    video.add_argument("--confidence", default=0.5, type=float, help="Detection confidence threshold")
    video.add_argument("--display", action="store_true", help="Show annotated video while processing")
    video.add_argument("--display-width", default=1280, type=int, help="Display width, or 0 for original")

    return parser


def normalize_argv(argv: list[str]) -> list[str]:
    """Keep the original flag-only video CLI working after adding subcommands."""

    if argv and argv[0] not in {"-h", "--help"} and argv[0] not in COMMANDS and argv[0].startswith("-"):
        return ["video", *argv]
    return argv


def main() -> int:
    parser = build_parser()
    args = parser.parse_args(normalize_argv(sys.argv[1:]))

    if args.command == "dashboard" or args.command is None:
        host = os.getenv("HOST", getattr(args, "host", "127.0.0.1"))
        port = getattr(args, "port", 8000)
        return run_dashboard(host, port)

    if args.command == "video":
        return run_video(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
