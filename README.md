# AI Engineer — Object Detection System

A full-stack web application for real-time object detection using YOLO-based models. Supports video frame analysis and Google Maps Street View capture with two detection pipelines:

- **Rubbish Detection** — identifies rubbish accumulation areas and classifies waste types (plastic, paper, metal, glass, cardboard, organic, etc.)
- **Damaged Sign Detection** — detects and classifies damaged traffic signs using a two-stage pipeline with Explainable AI support

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, TypeScript, Vite, Zustand |
| Backend | FastAPI, Python 3.10+ |
| ML | PyTorch, YOLOv8, EfficientNetV2-S, FAISS, CLIP |
| XAI | Grad-CAM, SHAP, Zennit (LRP) |
| Database | SQLite |

---

## Quick Start

### 1. Backend

```bash
cd backend

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt   # includes shap, zennit, matplotlib

python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

> Place model files in `backend/app/models_data/` before starting.
> See the **Models** section below for the expected directory structure.

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

> The Vite dev server proxies `/api` → `http://localhost:8000` automatically.

---

## Models

Model files are **not included** in this repository (large binaries). Place them at:

```
backend/app/models_data/
├── damaged_sign_detection/
│   ├── yolov26_1_class.pt              ← Traffic sign YOLO detector
│   ├── EffnetV2_multilabel.pth         ← Sign damage classifier (EfficientNetV2-S)
│   ├── traffic_signs_3.index           ← FAISS retrieval index
│   └── traffic_signs_metadata_3.json   ← Sign descriptions metadata
└── rubbish_detection/
    ├── stage1_best.pt                  ← Rubbish area detector
    └── stage2_best.pt                  ← Rubbish type classifier
```
https://drive.google.com/drive/u/0/folders/1oQSm2lsku0twFJGKKZzH-e5WjdGONYEU
---

## Damaged Sign Detection Pipeline

The `traffic_sign` model uses a two-stage approach:

1. **YOLO detector** — locates traffic signs in the frame (bounding boxes)
2. **EfficientNetV2-S classifier** — predicts damage labels for each cropped sign

**Damage classes:** `bent`, `broken_sheet`, `crack`, `graffiti`, `normal`, `paint_loss`, `rust`, `scratch`

**Per-class thresholds** (tuned during evaluation):

| Class | Threshold |
|-------|-----------|
| bent | 0.65 |
| broken_sheet | 0.70 |
| crack | 0.75 |
| graffiti | 0.45 |
| normal | 0.80 |
| paint_loss | 0.90 |
| rust | 0.75 |
| scratch | 0.85 |

**Normal exclusivity:** if `normal` fires, all other damage classes are suppressed for that detection.

---

## Explainable AI (XAI)

The `/api/explain` endpoint returns visualisations explaining what the classifier focused on. Upload a sign image and choose a method:

| Method | Description | Output |
|--------|-------------|--------|
| `grad_cam` | Gradient-weighted Class Activation Map | Heatmap overlay per class (JET colormap) |
| `shap` | SHAP pixel-importance (blur masker) | Composite red/blue pixel map for all active classes |
| `zennit` | Layer-wise Relevance Propagation (LRP) | Side-by-side original + bwr heatmap per class |

Each result contains the class name, sigmoid probability, and a base64-encoded PNG image ready to embed:

```html
<img src="data:image/png;base64,{image_base64}">
```

> All three methods are installed and ready — `shap 0.51.0`, `zennit 1.0.0`, `matplotlib 3.10.8` are included in `requirements.txt`.

---

## Features

**Video Analysis**
- Upload MP4, AVI, MOV, or WebM files
- Play video and capture individual frames
- Run detection on any captured frame
- Annotated results with bounding boxes, labels, and confidence scores
- Click a detected object thumbnail to highlight it on the full image

**Map Analysis**
- Google Maps integration (Roadmap + Street View)
- Capture the current map view and run detection on it
- Works with both satellite/roadmap and Street View panoramas

**Detection UI**
- Model selector (choose between rubbish or damaged sign pipeline)
- Annotated canvas with per-class color-coded bounding boxes
- Grouped detection summary panel with crop thumbnails
- Inference time display

---

## Project Structure

```
ai-engineer/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry point
│   │   ├── config.py            # Settings (model paths, thresholds)
│   │   ├── models/              # SQLAlchemy models & Pydantic schemas
│   │   ├── routers/
│   │   │   ├── detect.py        # POST /api/detect, /api/detect-base64
│   │   │   ├── explain.py       # POST /api/explain  ← XAI endpoint
│   │   │   ├── models.py        # GET /api/models
│   │   │   └── results.py       # GET /api/results
│   │   ├── services/
│   │   │   ├── damaged_sign_pipeline.py  # Two-stage sign detector
│   │   │   ├── xai_service.py            # Grad-CAM / SHAP / Zennit  ← new
│   │   │   ├── rubbish_detection_pipeline.py
│   │   │   ├── model_loader.py
│   │   │   └── inference.py
│   │   ├── utils/               # Exceptions, logger
│   │   └── models_data/         # Model files (gitignored)
│   └── requirements.txt
│
├── updated_damaged_sign_detection/   # Reference XAI & inference scripts
│   ├── inference.py
│   ├── explain_grad-cam.py
│   ├── explain_shap.py
│   └── explain_zennit.py
│
└── frontend/
    ├── src/
    │   ├── components/          # Shared UI components
    │   ├── pages/               # VideoUploader, VideoPlayer, MapViewer
    │   ├── hooks/               # useDetectionState, useMapCapture
    │   ├── services/            # Axios API client
    │   └── types/               # TypeScript types
    ├── vite.config.ts
    └── package.json
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/detect` | Detect from uploaded image/video file |
| `POST` | `/api/detect-base64` | Detect from base64-encoded image |
| `POST` | `/api/explain` | XAI visualisation for a sign image |
| `GET` | `/api/models` | List loaded models |
| `GET` | `/api/results` | Detection history |
| `GET` | `/health` | Health check |

Interactive docs at `http://localhost:8000/docs`.

### POST /api/explain

```
Content-Type: multipart/form-data
Fields:
  method  (string)  — grad_cam | shap | zennit
  file    (binary)  — sign image (jpg, png, bmp)
```

Response:
```json
{
  "method": "grad_cam",
  "results": [
    {
      "class_name": "rust",
      "probability": 0.834,
      "image_base64": "<base64 PNG>"
    }
  ],
  "inference_time_ms": 142.5
}
```

---

## Environment Variables

**Frontend** (`frontend/.env.local`):

```env
# Required only for the Map tab
VITE_GOOGLE_MAPS_API_KEY=your_key_here
```

Google Maps requires the **Maps JavaScript API**, **Static Maps API**, and **Street View Static API** enabled in your Google Cloud project.

---

## License

See [LICENSE](LICENSE).
