## Assignment – Computer Vision Intern

## Internship Task Specification: Real-Time Vehicle Analytics Pipeline

Welcome to the Vision Engineering team at Elansol Technologies. Over this sprint, you will architect and deploy an end-to-end computer vision pipeline that ingests live video, runs real-time inference to detect and classify vehicles, and surfaces the resulting analytics on a live web dashboard.

- Ingestion — Capture and distribute live RTSP streams using MediaMTX.

- Inference — Build an NVIDIA DeepStream pipeline to detect vehicles, classify their color, and track counts across a defined boundary.

- Persistence — Log every detection event, with precise timestamps and attributes, to a relational database.

- Visualization — Develop a responsive frontend dashboard displaying live stream analytics and historical data.

## 1. Core Objectives

## 2. Technical Stack

## Layer

Video broker

Computer vision engine NVIDIA DeepStream SDK (GStreamer-based); Python or C++

Database

Frontend UI

Version control

## Technology

MediaMTX (RTSP/HLS stream multiplexing)

PostgreSQL

React.js, or a native application

Git / GitHub

## 3. System Architecture

Data flows from the camera source through MediaMTX into the DeepStream inference pipeline, which writes detection events to PostgreSQL; the React dashboard reads from the database (and optionally the live stream) to render analytics in real time.

Key design requirement: the DeepStream pipeline must use a primary detector for vehicles (e.g., YOLO, TrafficCamNet) and a secondary classifier that runs on cropped vehicle bounding boxes to determine color. A tracker (NvDCF or SORT) is strictly required so the same vehicle is not counted more than once as it crosses the boundary.


## 4. Execution Milestone

## Phase 1 — Video Ingestion & Setup

- 1. Deploy MediaMTX locally or via Docker.

- 2. Publish a live camera feed, or loop an MP4 traffic video, to an RTSP endpoint using FFmpeg.

- 3. Verify the RTSP stream is accessible via a media player (e.g., VLC).

- This assignment is strictly for recruitment purposes and must not be shared, reproduced, or disclose 2. From the DeepStream Python/C++ probe, insert records asynchronously on each line-crossing event — either via a direct database connector or through a ZeroMQ message broker.

- 1. Configure nvstreammux to ingest the RTSP stream.

- 2. Integrate a primary inference engine (nvinfer) to detect vehicles.

- 3. Implement a multi-object tracker (nvtracker) to assign unique IDs to detected vehicles.

- 4. Integrate a secondary inference engine (nvinfer), configured to run only on vehicle bounding boxes, to classify color.

- 5. Implement line-crossing logic using nvdsanalytics or custom probe functions to trigger a "count" event.

## Phase 2 — DeepStream Vision Pipeline

## Phase 3 — Data Persistence

- 1. Set up a PostgreSQL database with a schema structured for analytics (e.g., id, vehicle_id, color, timestamp, direction).

## Phase 4 — Frontend Dashboard (React.js or Native)

- 1. Build a React.js interface — or a native application — to visualize the data.

- 2. Create a live metrics panel showing total vehicle count and a breakdown by color.

- 3. Display a live data table of the most recent database entries.

- 4. (Bonus) Embed a WebRTC or HLS player in the React app to display the live feed with drawn bounding boxes.

## 5 Final Deliverables

## 1. GitHub Repository

- Clean, well-commented code.

- A comprehensive README.md covering prerequisite installation steps (DeepStream, PostgreSQL, Node.js), execution commands, and pipeline configuration details.


## Assignment Walkthrough Instructions

During the assignment walkthrough, please keep your camera switched on throughout the recording session.

The walkthrough should cover the following:

- Introduction: Briefly introduce yourself and explain the project goals.

- Architecture Review: Explain how data moves from MediaMTX to React.

- Code Walkthrough: Highlight the core DeepStream logic, including bounding box extraction, tracking, and database insertion.

- Live Demonstration: Show the video feeding into MediaMTX, the DeepStream pipeline running, the database populating in real time, and the React UI updating dynamically.
