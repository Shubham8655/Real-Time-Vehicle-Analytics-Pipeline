# Real-Time Vehicle Analytics Pipeline

A Python-only, end-to-end vehicle analytics system for an RTSP traffic stream. It publishes a local video through MediaMTX, detects and tracks vehicles, estimates each crossing vehicle's visible color from its bounding-box crop, stores crossing events in PostgreSQL, and displays durable live analytics in a browser.

## What it delivers

- **RTSP ingestion:** MediaMTX brokers an RTSP publisher at `rtsp://localhost:8554/traffic` and exposes HLS at `http://localhost:8888/traffic/index.m3u8`.
- **Vehicle inference:** Ultralytics YOLO detects COCO vehicle classes: car, motorcycle, bus, and truck.
- **Persistent tracking:** ByteTrack assigns object IDs across frames.
- **Color classification:** The pipeline classifies the central pixels of every crossing vehicle crop in HSV space.
- **Accurate counting:** A horizontal boundary emits an event only when a tracked center crosses sides. In-memory state and a PostgreSQL unique constraint ensure each tracker ID is counted once per pipeline run.
- **Analytics dashboard:** Streamlit shows total crossings, color counts, and the most recent persistent events, refreshing every two seconds.

The original brief proposes NVIDIA DeepStream. This implementation deliberately uses Python-only, cross-platform components as requested: Ultralytics YOLO + ByteTrack provide the detector and tracker; the cropped color classifier is the secondary classification stage. It runs on Windows and CPU-only machines, while automatically using CUDA if the installed PyTorch build supports it.

## Architecture

```text
traffic.mp4 / camera
        |
        v
FFmpeg publisher (Python-managed) ---> MediaMTX RTSP/HLS
                                           |
                                           v
                       YOLO vehicle detector -> ByteTrack IDs
                                           |
                                           v
                         cropped HSV color classifier -> boundary crossing
                                                                |
                                          async SQLAlchemy writes v
                                                         PostgreSQL
                                                                |
                                                                v
                                                   Streamlit dashboard
```

## Repository layout

| Path | Purpose |
| --- | --- |
| `pipeline.py` | Inference loop, ByteTrack state, line-crossing logic, and asynchronous persistence. |
| `db.py` | SQLAlchemy schema and transactional event repository. |
| `app.py` | Streamlit live dashboard. |
| `scripts/publish_video.py` | Loops an MP4 or publishes a Windows DirectShow camera to RTSP. |
| `docker-compose.yml` | PostgreSQL 16 and MediaMTX services. |
| `mediamtx.yml` | RTSP/HLS path configuration. |
| `tests/` | Unit tests for color, boundary, single-count, and persistence behavior. |

## Prerequisites

- Windows 10/11 or another Python-supported OS
- Python 3.11+
- Docker Desktop running (Docker Compose supplies PostgreSQL and MediaMTX)
- Git

The first inference run downloads `yolo11n.pt` from Ultralytics. The included `traffic.mp4` is the demonstration input. A CUDA-enabled PyTorch installation is optional; CPU inference works but is slower.

## Setup

In PowerShell at the repository root:

```powershell
Copy-Item .env.example .env
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
docker compose up -d postgres mediamtx
docker compose ps
```

Wait until the `postgres` service reports `healthy`. The default credentials are local development credentials in `.env`; use a different `DATABASE_URL` in any shared environment.

## Run the system

Open three PowerShell terminals at the repository root.

**1. Publish the sample video**

```powershell
.\.venv\Scripts\python scripts\publish_video.py
```

The video loops indefinitely. Open `rtsp://localhost:8554/traffic` in VLC to verify RTSP delivery. The HLS stream is at `http://localhost:8888/traffic/index.m3u8`.

To publish a different file or a Windows DirectShow camera:

```powershell
.\.venv\Scripts\python scripts\publish_video.py --input "C:\videos\junction.mp4"
.\.venv\Scripts\python scripts\publish_video.py --camera --input "Integrated Camera"
```

**2. Start inference and counting**

```powershell
.\.venv\Scripts\python pipeline.py --show
```

Press `q` in the annotated video window to stop. `--show` is optional. Use `--max-frames 300` for a bounded smoke run. The pipeline reads the following settings from `.env` or the environment:

| Setting | Default | Description |
| --- | --- | --- |
| `DATABASE_URL` | local PostgreSQL DSN | SQLAlchemy PostgreSQL connection URL. |
| `RTSP_URL` | `rtsp://localhost:8554/traffic` | Input stream and publisher target. |
| `YOLO_MODEL` | `yolo11n.pt` | Ultralytics model name or local path. |
| `LINE_RATIO` | `0.55` | Counting line as a fraction of frame height, exclusive of 0 and 1. |

**3. Start the dashboard**

```powershell
.\.venv\Scripts\python -m streamlit run app.py
```

Open the URL Streamlit prints, usually `http://localhost:8501`.

## Implementation details

### Counting behavior

For each ByteTrack ID, the pipeline retains its last vertical center. A crossing is recorded when the center moves from above to on/below the horizontal line (`north_to_south`) or from below to on/above it (`south_to_north`). The track state flips to counted before the database job is queued. Stale tracks are removed after 60 seconds.

Each pipeline process has a random `run_id`. The `detection_events` table has a unique `(run_id, tracker_id)` constraint, which makes a duplicate write harmless even if a retry occurs. Database insertion runs in a bounded two-worker thread pool so slow I/O does not block frame inference.

### Schema

`detection_events` stores the primary key, run and tracker IDs, vehicle class, color, direction, detector confidence, and a timezone-aware UTC crossing timestamp. The schema is created idempotently at service start. Database data persists in Docker's named `postgres_data` volume.

### Color estimates

The classifier crops the central 60% of the vehicle bounding box to reduce road and background pixels. It uses median HSV saturation/value for black, white, and gray, then median hue for red, orange, yellow, green, blue, and purple. This is intentionally interpretable and lightweight; color labels are visual estimates and can be affected by lighting, reflections, and occlusion.

## Test and verification

Run the unit suite:

```powershell
.\.venv\Scripts\python -m pytest -q
```

The tests cover boundary directions, duplicate prevention for a re-crossing tracker, neutral and colored crop classification, and the database uniqueness/dashboard query path using a temporary SQLite database.

For an end-to-end check, start both Docker services, run the publisher and pipeline, then confirm that events appear in the dashboard after vehicles cross the drawn yellow line. `docker compose ps` should show PostgreSQL as healthy.

## Stop or reset services

```powershell
docker compose down       # stop services and keep event data
docker compose down -v    # stop services and remove the local PostgreSQL volume
```

## Known operational limits

- Stable tracking depends on camera angle, frame rate, and scene congestion. Adjust the detector model, confidence threshold, or counting line for a deployed camera.
- The demo starts with the small `yolo11n` model to keep local setup practical. A larger compatible YOLO model can improve accuracy at additional compute cost.
- The dashboard reads persisted events and does not embed annotated video. The RTSP/HLS endpoints remain available for a dedicated player.
