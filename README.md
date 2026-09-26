# Real-Time Vehicle Analytics Pipeline

A Python-only, end-to-end vehicle analytics project. It loops the supplied traffic video into an RTSP endpoint, detects and tracks vehicles, counts each tracked vehicle once as it crosses a configurable line, stores the event in PostgreSQL, and shows the results in a live web dashboard.

The implementation intentionally uses a Python YOLO/ByteTrack pipeline rather than C++ or a hardware-bound DeepStream binary. It preserves the task's detector → tracker → cropped-colour-classifier → database flow, runs on a CPU-only Windows machine, and will use CUDA automatically when PyTorch is installed with CUDA support.

## Architecture

```text
traffic.mp4 / camera
        │  FFmpeg
        ▼
MediaMTX RTSP (rtsp://localhost:8554/traffic)
        │
        ▼
pipeline.py: YOLO detector → ByteTrack IDs → HSV crop colour classifier
        │                         │
        │                  one crossing event per ID
        ▼
PostgreSQL ───────────────► FastAPI dashboard (http://localhost:8000)
```

## What is included

- `docker-compose.yml` starts PostgreSQL 16 and MediaMTX locally.
- `scripts/publish_video.py` loops `traffic.mp4` into RTSP, or can publish a named Windows camera.
- `pipeline.py` runs the primary detector, persistent ByteTrack tracker, colour classifier, crossing logic, and non-blocking database writes.
- `app.py` contains the database schema, REST API, and dashboard server.
- `static/index.html` is a responsive, dependency-free live dashboard.

## Prerequisites

- Python 3.11+
- Docker Desktop (running)
- FFmpeg on `PATH` (`winget install Gyan.FFmpeg.Essentials` on Windows)

Docker provides PostgreSQL and MediaMTX; no separate local PostgreSQL installation is needed. The first inference run downloads the small `yolo11n.pt` model through Ultralytics.

## Setup and run

PowerShell commands from the repository root:

```powershell
Copy-Item .env.example .env
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
docker compose up -d
```

Open three terminals (activate the virtual environment in the last two):

```powershell
# Terminal 1: publish the included video as an endlessly looping RTSP source
.\.venv\Scripts\python scripts/publish_video.py

# Terminal 2: run detector, tracker, colour classification, and event persistence
.\.venv\Scripts\python pipeline.py --show

# Terminal 3: start the dashboard
.\.venv\Scripts\python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Visit `http://localhost:8000`. Press `q` in the annotated pipeline window to stop it. The RTSP stream can also be checked in VLC at `rtsp://localhost:8554/traffic`; while it is being published, MediaMTX HLS is at `http://localhost:8888/traffic/index.m3u8`.

To use a Windows camera instead of the MP4, first find its DirectShow name with `ffmpeg -list_devices true -f dshow -i dummy`, then run:

```powershell
.\.venv\Scripts\python scripts/publish_video.py --camera --input "Your Camera Name"
```

## Configuration

Copy `.env.example` to `.env` to change the database connection, RTSP input, or model file. The counting boundary defaults to 55% of the image height and can be adjusted without code changes:

```powershell
.\.venv\Scripts\python pipeline.py --line-ratio 0.65
```

Only COCO vehicle classes are passed to the detector (`car`, `motorcycle`, `bus`, and `truck`). ByteTrack supplies persistent IDs. Each ID is marked counted after its first side-to-side transition over the line, which prevents repeated counts while the object remains visible. The HSV colour classifier uses only the central portion of the vehicle crop to reduce road/background influence; it returns a deliberately conservative `unknown` when a colour cannot be classified.

## Data model and API

`detection_events` stores `vehicle_id`, `vehicle_type`, `color`, `direction`, `confidence`, and an UTC `crossed_at` timestamp. The FastAPI server creates this table on startup.

- `GET /health` checks database connectivity.
- `GET /api/metrics` returns the persisted total and colour breakdown.
- `GET /api/events?limit=30` returns the newest crossings.
- `POST /api/events` accepts a validated event; useful for an integration check or external producer.

## Verification

```powershell
.\.venv\Scripts\python -m pytest -q
docker compose ps
curl http://localhost:8000/health
```

The included unit tests cover line-crossing direction and neutral-colour classification. The intended end-to-end check is to run the three processes above, confirm `GET /api/metrics` increments, and observe the live table update.

## Engineering decisions

- **Python only:** No C++ source or build chain is used. Ultralytics, OpenCV, SQLAlchemy, and FastAPI are all called directly from Python.
- **Portable local services:** Compose makes PostgreSQL and MediaMTX reproducible and keeps operational configuration to two small files.
- **Durable source of truth:** dashboard counters are query-time aggregates over PostgreSQL, so a dashboard or pipeline restart does not reset analytics.
- **Responsive ingestion:** database inserts run in a small worker pool so a slow database does not block video-frame processing.
- **Small review surface:** the application logic is kept in `pipeline.py` and `app.py`; comments explain non-obvious vision and persistence decisions.

## Stop and reset

```powershell
docker compose down                 # stop services, retain database data
docker compose down -v              # stop services and remove the local database volume
```
