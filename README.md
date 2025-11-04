# Content-Based FoV Prediction with Grad-CAM (360° Videos)

This repo implements **content-based Field-of-View (FoV) prediction** for **360° (equirectangular) videos** using **Grad-CAM** heatmaps from an ImageNet backbone (default: ResNet50).  
We treat Grad-CAM activation as a **content saliency proxy** and compute per-frame FoV center(s) from the heatmap.

> This version avoids overlay video rendering (no blend), and saves per-frame heatmaps plus a CSV of predicted FoV centers.

---

## 🧱 Repo Structure
```
video-fov-gradcam/
├── data/
│   ├── frames/            # optionally store extracted frames here
│   └── videos/            # place input 360 videos here (equirectangular)
├── outputs/
│   ├── heatmaps/          # saved per-frame heatmaps (.npy / .png)
│   └── fov/               # CSV with FoV centers per frame
├── scripts/
│   └── run.ps1            # PowerShell helper for Windows
├── src/
│   ├── eval/metrics.py    # (placeholder) metrics for evaluation
│   ├── models/backbones.py
│   ├── utils/spherical.py
│   ├── extract_frames.py
│   ├── fov_from_heatmap.py
│   └── run_gradcam.py     # ENTRYPOINT: runs Grad-CAM + FoV estimation
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
# 1) Create venv
python -m venv venv
# 2) Activate (PowerShell on Windows)
venv\Scripts\Activate.ps1
# 3) Install
pip install -r requirements.txt
```

> **Note:** The first run may download torchvision weights for ResNet50 (~100 MB). If you prefer offline/no-download, set `--pretrained false` and pass your own `--weights` path to a local `.pth` file compatible with torchvision ResNet50.

---

## 🎬 Usage

### Option A: Run directly on a video file
```bash
python -m src.run_gradcam   --video data/videos/my360.mp4   --outdir outputs   --backbone resnet50   --pretrained true   --topk 1   --save-png true
```

### Option B: Run on a directory of frames
```bash
python -m src.run_gradcam   --frames data/frames   --outdir outputs   --backbone resnet50   --pretrained true   --topk 1   --save-png true
```

### Optional: Extract frames from a video
```bash
python -m src.extract_frames --video data/videos/my360.mp4 --outdir data/frames --fps 10
```

---

## 📦 Outputs
- `outputs/heatmaps/frame_XXXXX.npy` : Grad-CAM heatmap (float32, normalized 0..1)
- `outputs/heatmaps/frame_XXXXX.png` : (optional) saved heatmap visualization
- `outputs/fov/fov_centers.csv` : per-frame FoV center in spherical coords (lon, lat in degrees), plus pixel coordinates

**FoV center computation:** We use a weighted centroid over the heatmap in equirectangular pixel space, then map to spherical coordinates (longitude ∈ [-180, 180], latitude ∈ [-90, 90]). You can switch to top-k peaks with `--topk N`.

---

## 🧪 Notes / Assumptions
- Input frames are **equirectangular** (360°). If your video is cubic or fisheye, convert to equirectangular first.
- This baseline is **content-only**. It ignores motion and temporal cues by design; you can extend with temporal smoothing.
- No heavy overlay video is produced; this keeps the pipeline simple and fast.

---

## 📈 Roadmap (nice-to-have)
- Temporal smoothing of FoV centers (EMA/Kalman).
- Compare with gaze ground truth if available.
- Multi-backbone ablations (VGG, EfficientNet, Swin).
- Peak clustering for multi-attention FoV.

---

## 🔒 Large Files
Large artifacts (videos, weights) are excluded via `.gitignore` to keep the repo GitHub-friendly (no LFS required).