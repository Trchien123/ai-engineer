# AI Engineer — Object Detection System

A full-stack web application for real-time object detection using YOLO-based models. Supports video frame analysis and Google Maps Street View capture with two detection pipelines:

- **Rubbish Detection** — identifies rubbish accumulation areas and classifies waste types (plastic, paper, metal, glass, cardboard, organic, etc.)
- **Damaged Sign Detection** — detects and identifies damaged or missing traffic signs

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, TypeScript, Vite, Zustand |
| Backend | FastAPI, Python 3.10+ |
| ML | PyTorch, YOLOv8, EfficientNetV2, FAISS |
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

pip install -r requirements.txt

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
├── yolov26_1_class.pt                  ← Traffic sign YOLO detector
├── damaged_sign_detection/
│   ├── EffnetV2_multilabel.pth         ← Sign classifier
│   ├── traffic_signs_3.index           ← FAISS retrieval index
│   └── traffic_signs_metadata_3.json
└── rubbish_detection/
    ├── stage1_best.pt                  ← Rubbish area detector
    └── stage2_best.pt                  ← Rubbish type classifier
```

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
│   │   ├── models/              # SQLAlchemy models & schemas
│   │   ├── routers/             # API route handlers
│   │   ├── services/            # Inference pipelines, storage
│   │   ├── utils/               # Exceptions, helpers
│   │   └── models_data/         # Model files (gitignored)
│   └── requirements.txt
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
| `POST` | `/api/detect` | Detect from uploaded image file |
| `POST` | `/api/detect-base64` | Detect from base64-encoded image |
| `GET` | `/api/models` | List loaded models |
| `GET` | `/api/results` | Detection history |
| `GET` | `/health` | Health check |

Interactive docs at `http://localhost:8000/docs`.

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
