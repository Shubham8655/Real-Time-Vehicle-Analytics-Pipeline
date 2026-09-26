"""Publish a looped MP4 or Windows camera to a MediaMTX RTSP endpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import imageio_ffmpeg

# Allow `python scripts/publish_video.py` from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_settings


def build_command(input_value: str, target_url: str, camera: bool) -> list[str]:
    """Build the FFmpeg invocation without running it, for predictable testing."""
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-re"]
    if camera:
        command.extend(["-f", "dshow", "-i", f"video={input_value}"])
    else:
        command.extend(["-stream_loop", "-1", "-i", str(Path(input_value).resolve())])
    return [
        *command,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-tune",
        "zerolatency",
        "-f",
        "rtsp",
        "-rtsp_transport",
        "tcp",
        target_url,
    ]


def main() -> None:
    """Parse publisher arguments and run FFmpeg until interrupted."""
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Publish a source to MediaMTX over RTSP.")
    parser.add_argument("--input", default="traffic.mp4", help="MP4 file or DirectShow camera name")
    parser.add_argument("--url", default=settings.rtsp_url, help="MediaMTX RTSP publish URL")
    parser.add_argument("--camera", action="store_true", help="Interpret --input as a Windows camera name")
    args = parser.parse_args()
    if not args.camera and not Path(args.input).is_file():
        parser.error(f"video file not found: {args.input}")

    print(f"Publishing to {args.url}. Press Ctrl+C to stop.")
    subprocess.run(build_command(args.input, args.url, args.camera), check=True)


if __name__ == "__main__":
    main()
