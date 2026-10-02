# 💡 How This Multi-Camera Person Tracking & ReID System Works

Welcome! This document explains in **simple, human-friendly terms** how this system works under the hood, what was modernized, and how we achieve **high HOTA (Higher Order Tracking Accuracy)** without training any models from scratch.

---

## 🎯 The Big Picture (Short & Simple)

Imagine you have 2 or 3 security cameras watching a hallway from different angles.
When a person walks through:
1. **Camera 1** sees them from the front.
2. **Camera 2** sees them from the side/back.

### What This System Does:
* It detects the person in every video frame.
* It follows (tracks) them continuously in Video 1 so their bounding box doesn't jump or change ID.
* It extracts their "visual fingerprint" (face, height, clothing color, body features).
* It matches their fingerprint across Video 2 and assigns them the **EXACT SAME Global Person ID**.

---

## 🚀 What We Rebuilt (Old vs New Architecture)

| Feature | Old Legacy Codebase 👴 | New Modern Architecture 🚀 |
| :--- | :--- | :--- |
| **Framework** | TensorFlow 1.x & Keras | **PyTorch 2.x** |
| **Object Detector** | YOLOv3 / YOLOv4 (Old `.h5` weights) | **Ultralytics YOLOv8 / YOLOv11** |
| **Single-Camera Tracker** | DeepSORT (`mars-small128.pb`) | **ByteTrack / BoT-SORT** |
| **Person Re-ID** | Old custom Torchreid scripts | **OSNet Deep ReID Backbone** |
| **Trajectory Smoothing** | None (Frequent box flickering) | **Linear Interpolation (HOTA Boost)** |
| **Deployment & UI** | Terminal only (`demo.py`) | **Hugging Face Spaces + Gradio Web UI** |

---

## 🧠 The 3-Step Magic Pipeline

```
[ Input Videos ] ──> Step 1: Detect ──> Step 2: Track ──> Step 3: Re-Identify ──> [ Annotated Video ]
```

### 1️⃣ Detection (Finding People)
We use pre-trained **YOLOv8** / **YOLOv11** models. Because these models are trained on millions of real-world images (COCO Dataset), **Class 0 ("person")** detection is already super sharp out of the box. *No model fine-tuning needed!*

### 2️⃣ Tracking (ByteTrack Logic)
Standard trackers throw away low-confidence detections, causing track IDs to break whenever a person is partially hidden behind a pole or another person.
**ByteTrack** keeps low-score detections in a temporal buffer and uses **Kalman Filter motion prediction** to maintain track continuity, keeping **IDF1 (Identity F1 Score)** ultra high.

### 3️⃣ Re-Identification (OSNet Visual Fingerprint)
When a tracklet is formed, our **OSNet PyTorch model** crops person patches and converts them into a 512-dimensional vector (a numerical visual fingerprint).
We compute the **Cosine Distance** between fingerprints across different cameras. If the distance is below the threshold (e.g. `< 0.25`), the system concludes: *"Hey! This is the same person!"* and fuses their global ID.

---

## 🌐 Hosting on Hugging Face Spaces (ZeroGPU Free)

This project is built to run directly on **Hugging Face Spaces**:
1. Create a **Gradio Space** on Hugging Face.
2. Upload this repository (`app.py`, `requirements.txt`, `src/`, `config/`, `README.md`).
3. If using **HF ZeroGPU**, `app.py` automatically utilizes `@spaces.GPU` for lightning-fast GPU inference for **FREE**.

---

## 💻 Quick Usage Commands

### Web App (Local or HF Space):
```bash
python app.py
```

### CLI Command (Terminal Batch Processing):
```bash
python main.py --videos videos/init/Double1.mp4 videos/init/Single1.mp4 --model yolov8m.pt --tracker bytetrack
```

Enjoy your modern, high-accuracy multi-camera tracking system! 🎉
