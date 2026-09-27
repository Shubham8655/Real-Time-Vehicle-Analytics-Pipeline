# Real-Time Vehicle Analytics Pipeline

Python-only vehicle analytics for a live traffic RTSP stream. It publishes a video through MediaMTX, detects COCO vehicles with YOLO, tracks them with ByteTrack, estimates color from the cropped vehicle image, records one event per boundary crossing in PostgreSQL, and presents live totals plus recent events in Streamlit.

## Architecture

```text
MP4 / camera -> FFmpeg -> MediaMTX RTSP/HLS -> YOLO + ByteTrack -> HSV crop color classifier
                                                               -> PostgreSQL -> Streamlit dashboard
```

The original brief names DeepStream. This project intentionally uses only Python, as required: Ultralytics YOLO is the primary vehicle detector, ByteTrack supplies persistent track IDs, and `classify_color` is the secondary crop-level classifier. A track is marked counted before its asynchronous database write is queued; `(run_id, tracker_id)` is also unique in PostgreSQL as durable duplicate protection.

## Components

- `scripts/publish_video.py` loops `traffic.mp4` (or a Windows DirectShow camera) to `rtsp://localhost:8554/traffic`.
- `pipeline.py` detects cars, motorcycles, buses, and trucks; tracks them; counts crossings on a configurable horizontal line; and writes events asynchronously.
- `db.py` owns the PostgreSQL schema and dashboard queries.
- `app.py` is the responsive Streamlit dashboard, automatically refreshing every two seconds.
- `docker-compose.yml` starts PostgreSQL 16 and MediaMTX, with a persistent database volume.

## Setup

Docker Desktop, Python 3.11+, and FFmpeg are required. They are installed in the prepared environment. From the repository root:

```powershell
Copy-Item .env.example .env
.\.venv\Scripts\python -m pip install -r requirements.txt
docker compose up -d postgres mediamtx
docker compose ps
```

PostgreSQL is ready when its container is `healthy`. The development database URL, RTSP URL, model name, and line position can be changed in `.env`.

## Run

Use three terminals:

```powershell
# Terminal 1: publish the demo video (loops until Ctrl+C)
.\.venv\Scripts\python scripts\publish_video.py

# Terminal 2: detect, track, classify and count
.\.venv\Scripts\python pipeline.py --show

# Terminal 3: dashboard
.\.venv\Scripts\python -m streamlit run app.py
```

Open `rtsp://localhost:8554/traffic` in VLC to inspect RTSP, or use `http://localhost:8888/traffic/index.m3u8` for HLS. Use `--source path\to\video.mp4` to run inference directly from a file and `--max-frames 300` for a bounded smoke test. For a camera publisher, use:

```powershell
.\.venv\Scripts\python scripts\publish_video.py --camera --input "Integrated Camera"
```

## Verification

```powershell
.\.venv\Scripts\python -m pytest -q
docker compose ps
```

Tests cover boundary directions, one-count-per-track behavior, crop color labels, uniqueness in the persistence layer, dashboard queries, and a mocked tracked crossing through the inference loop.

## Notes

The included small model (`yolo11n.pt`) downloads on its first inference run. CPU inference works; a compatible CUDA-enabled PyTorch installation is used automatically when available. Color labels are visible-light estimates and may vary with reflections or poor lighting. Adjust `LINE_RATIO`, detector confidence in `pipeline.py`, or the YOLO model for a production camera.

Stop local services with `docker compose down`; use `docker compose down -v` only when intentionally discarding the local analytics database.
