# Real-Time Vehicle Analytics Pipeline

A Python-only vehicle analytics demo that publishes a traffic video over RTSP, detects and tracks vehicles, classifies their visible color, records one event per line crossing in PostgreSQL, and displays live totals in a browser dashboard.

## Architecture

```text
traffic.mp4 ──► FFmpeg (Python-managed) ──► MediaMTX / RTSP
                                                   │
                                                   ▼
                       YOLO detector ─► ByteTrack IDs ─► crop color classifier
                                                   │
                                      line crossing event
                                                   ▼
                                             PostgreSQL
                                                   │
                                                   ▼
                                      Streamlit dashboard
```

The task mentions NVIDIA DeepStream and React. This implementation keeps the full inference and user interface in Python: Ultralytics YOLO with ByteTrack provides detection and persistent tracking, and Streamlit provides the dashboard. That makes the project runnable on a CPU-only Windows host as well as CUDA-enabled Python environments, without requiring the Linux/NVIDIA-specific DeepStream runtime or a separate JavaScript build chain.

## Included components

- `scripts/publish_video.py` publishes the included MP4 in a loop to MediaMTX. Its FFmpeg executable is supplied by the Python `imageio-ffmpeg` dependency.
- `pipeline.py` filters YOLO detections to cars, motorcycles, buses, and trucks; uses persistent ByteTrack IDs; classifies the central region of each vehicle crop; and writes the first line crossing per track asynchronously.
- `app.py` defines the PostgreSQL event schema and serves the dashboard. Metrics are queried from the database, so they survive process restarts.
- `docker-compose.yml` runs PostgreSQL 16 and MediaMTX. The database volume is retained when containers stop.

## Requirements

- Windows 10/11, Python 3.11+, and Git.
- Docker Desktop installed and running. Docker Compose provides PostgreSQL and MediaMTX, so separate installations of those services are not needed.
- The first YOLO run downloads `yolo11n.pt` from Ultralytics. CPU inference works without NVIDIA hardware; a compatible CUDA-enabled PyTorch installation can accelerate inference.

## Setup

Run these commands in PowerShell from the repository directory:

```powershell
Copy-Item .env.example .env
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
docker compose up -d postgres mediamtx
```

Wait for PostgreSQL to report healthy:

```powershell
docker compose ps
```

## Run the pipeline

Use three terminals, each with the repository as its working directory. Start the video publisher:

```powershell
.\.venv\Scripts\python scripts/publish_video.py
```

The source video is looped until the process is stopped with Ctrl+C. MediaMTX exposes the RTSP stream at `rtsp://localhost:8554/traffic` and HLS at `http://localhost:8888/traffic/index.m3u8`. The RTSP URL can be opened in VLC while the publisher is running.

Start inference in a second terminal:

```powershell
.\.venv\Scripts\python pipeline.py --show
```

Press `q` in the preview window to stop inference. Omit `--show` to run without an annotated preview. To choose a different source or counting line:

```powershell
.\.venv\Scripts\python pipeline.py --source "rtsp://localhost:8554/traffic" --line-ratio 0.60
```

The line ratio is a fraction of the frame height and must be between 0 and 1. The default is `0.55`. For a bounded sample run, add `--max-frames 300`. Set `RTSP_URL`, `YOLO_MODEL`, `LINE_RATIO`, and `DATABASE_URL` in `.env` to configure the app. The included video is expected at `traffic.mp4`; pass a different file to the publisher with `--input`.

Start the dashboard in a third terminal:

```powershell
.\.venv\Scripts\python -m streamlit run app.py
```

Streamlit prints the local dashboard URL (normally `http://localhost:8501`). It refreshes the database totals and recent events every two seconds. The page includes the total crossing count, color breakdown, and latest event table.

## Event schema and counting behavior

`detection_events` stores the unique database ID, a run-namespaced tracker ID, vehicle type, estimated color, crossing direction, detector confidence, and timezone-aware UTC timestamp. The horizontal line is positioned at `LINE_RATIO` of the frame height. The pipeline compares each track's center between frames and emits `north_to_south` or `south_to_north` the first time that ID changes sides. A new run receives a distinct ID namespace. Inserts use a bounded two-worker thread pool so a database write does not block frame inference; write failures are logged.

YOLO only processes the COCO vehicle classes (`car`, `motorcycle`, `bus`, and `truck`). ByteTrack assigns IDs between frames. The color estimate uses HSV statistics from the center of each detected vehicle crop to reduce road and background pixels. It is a visual estimate and can return `unknown` for an empty crop.

## Verification

```powershell
.\.venv\Scripts\python -m pytest -q
docker compose ps
```

For a complete live check, run the publisher, pipeline, and dashboard in separate terminals. Confirm the RTSP stream opens in VLC, then verify that crossing events appear in the dashboard. The pure unit tests cover crossing direction and color classification.

## Stop services

```powershell
docker compose down       # stop containers and retain database records
docker compose down -v    # also remove the local PostgreSQL volume and records
```
