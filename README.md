# Deepfake Detector — Project Structure
use python 3.11
py -3.11 -m venv venv
venv\Scripts\activate
python --version
```
DEEPFAKE-DETECTOR/
├── backend/
│   ├── app/
│   │   │
│   │   ├── api/
│   │   │   └── v1/
│   │   │       ├── images.py          # POST /analyze/image
│   │   │       └── videos.py          # POST /analyze/video
│   │   │
│   │   ├── core/
│   │   │   ├── config.py              # Pydantic settings (.env)
│   │   │   └── database.py            # SQLAlchemy engine + session
│   │   │
│   │   ├── ml/
│   │   │   ├── image/
│   │   │   │   ├── model.py           # EfficientNet-B0 DeepfakeDetector
│   │   │   │   ├── inference.py       # predict_image + compute_final_score
│   │   │   │   ├── ela.py             # Error Level Analysis
│   │   │   │   ├── fft_analysis.py    # Frequency domain analysis
│   │   │   │   ├── noise_analysis.py  # Sensor noise pattern analysis
│   │   │   │   └── metadata_analysis.py # EXIF + AI keywords + dimensions
│   │   │   │
│   │   │   ├── video/                 # NEW ───────────────────────────────
│   │   │   │   ├── frame_extractor.py # Extract frames from video
│   │   │   │   ├── temporal_analysis.py # Temporal consistency score
│   │   │   │   └── video_inference.py # Full video detection pipeline
│   │   │   │
│   │   │   ├── train.py               # Training script (image classifier)
│   │   │   ├── dataset_loader.py      # GenImage dataset loader
│   │   │   └── organize_dataset.py    # Dataset split organizer
│   │   │
│   │   ├── schemas/
│   │   │   └── result.py              # Unified ImageAnalysisResult / VideoAnalysisResult
│   │   │
│   │   ├── services/
│   │   │   ├── image_service.py       # Thin service wrapper for image pipeline
│   │   │   └── video_service.py       # Thin service wrapper for video pipeline
│   │   │
│   │   └── db/
│   │       └── models/
│   │           ├── analysis.py        # Analysis DB model
│   │           └── media.py           # Uploaded media record

        ├── frontend/
        │   ├── static/css/style.css        ✓
        │   └── templates/
        │       ├── base.html               ✓
        │       ├── index.html              ✓
        │       └── partials/result.html    ✓
│   │
│   ├── main.py                        # FastAPI app + router registration
│   ├── requirements.txt
│   └── .env
│
├── dataset/
│   ├── raw/                           # Original dataset folders
│   ├── train/                         # After organize_dataset.py
│   └── val/
│
├── frontend/
│   └── ...                            # fastapi
│
└── docs/
    └── api.md
```

## Quick Start

```bash
# 1. Install dependencies
cd backend
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit DATABASE_URL, ML_MODELS_DIR etc.

# 3. Run API
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 
# 4. Open Swagger docs
open http://localhost:8000/docs
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/analyze/image` | Analyze image for AI generation |
| POST | `/api/v1/analyze/video` | Analyze video for deepfake |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI |

## Image Analysis Layers

| Layer | Signal | Weight (dynamic) |
|-------|--------|-----------------|
| ML (EfficientNet-B0) | Learned visual features | 30–55% |
| ELA | Compression fingerprint uniformity | 10–14% |
| FFT | Frequency domain artifacts | 8–12% |
| Noise | Sensor noise pattern | 7–11% |
| Metadata | EXIF, AI keywords, dimensions | 8–45% |

## Video Analysis Pipeline

```
Video File
    ↓
Frame Extractor (1fps + scene-change keyframes)
    ↓
Per-frame: ELA + FFT + Noise → frame_score
    ↓
Temporal Analysis → temporal_score (variance, gradients, flicker)
    ↓
Final Score = frame_aggregate × 0.60 + temporal_score × 0.40
```