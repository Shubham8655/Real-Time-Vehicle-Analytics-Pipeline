"""Loop a local MP4 or publish a camera to MediaMTX using FFmpeg."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish video to MediaMTX RTSP.")
    parser.add_argument("--input", default="traffic.mp4", help="MP4 path (or a Windows camera dshow input)")
    parser.add_argument("--url", default="rtsp://localhost:8554/traffic")
    parser.add_argument("--camera", action="store_true", help="Use --input as a DirectShow camera name")
    args = parser.parse_args()
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("FFmpeg is required. Install it with: winget install Gyan.FFmpeg.Essentials")
    if not args.camera and not Path(args.input).is_file():
        raise SystemExit(f"Video not found: {args.input}")
    command = [ffmpeg, "-re"]
    command += ["-f", "dshow", "-i", f"video={args.input}"] if args.camera else ["-stream_loop", "-1", "-i", args.input]
    command += ["-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-f", "rtsp", "-rtsp_transport", "tcp", args.url]
    print("Publishing to", args.url, "(Ctrl+C to stop)")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
