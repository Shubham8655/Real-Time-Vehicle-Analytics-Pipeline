"""Publish an MP4 or Windows camera to MediaMTX using imageio's FFmpeg binary."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import imageio_ffmpeg
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Publish a local source to a MediaMTX RTSP path.")
    parser.add_argument("--input", default="traffic.mp4", help="MP4 file or DirectShow camera name")
    parser.add_argument("--url", default=os.getenv("RTSP_URL", "rtsp://localhost:8554/traffic"))
    parser.add_argument("--camera", action="store_true", help="Interpret --input as a camera name")
    args = parser.parse_args()
    source = Path(args.input)
    if not args.camera and not source.is_file():
        parser.error(f"video file not found: {source}")

    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-re"]
    if args.camera:
        command.extend(["-f", "dshow", "-i", f"video={args.input}"])
    else:
        command.extend(["-stream_loop", "-1", "-i", str(source.resolve())])
    command.extend([
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
        "-f", "rtsp", "-rtsp_transport", "tcp", args.url,
    ])
    print(f"Publishing to {args.url}. Press Ctrl+C to stop.")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
