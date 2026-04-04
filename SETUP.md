# Object Detection System - Complete Setup Guide

This guide provides comprehensive instructions for setting up, configuring, and deploying the full-stack Object Detection System.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Prerequisites](#prerequisites)
3. [Backend Setup](#backend-setup)
4. [Frontend Setup](#frontend-setup)
5. [Model Configuration](#model-configuration)
6. [Running the Application](#running-the-application)
7. [Verification](#verification)
8. [Deployment](#deployment)
9. [Troubleshooting](#troubleshooting)

## System Requirements

### Hardware
- **CPU**: Minimum 4 cores (8+ recommended)
- **RAM**: 8GB minimum (16GB+ recommended for larger models)
- **GPU**: NVIDIA GPU (CUDA 11.8+) optional but recommended for fast inference

### Operating System
- Windows 10/11
- macOS 10.14+
- Linux (Ubuntu 18.04+)

### Software
- Python 3.8 or higher
- Node.js 16.0 or higher
- npm 7.0 or higher (or yarn)

## Prerequisites

### 1. Install Python
Download from [python.org](https://www.python.org/downloads/) and ensure it's added to PATH.

Verify:
```bash
python --version  # Should be 3.8+
pip --version
```

### 2. Install Node.js
Download from [nodejs.org](https://nodejs.org/) (LTS version recommended).

Verify:
```bash
node --version
npm --version
```

### 3. Clone Repository
```bash
git clone <your-repo-url>
cd ai-engineer
```

### 4. Prepare Model Files
Ensure your YOLO .pt model files are available:
- `rubbish_area.pt` - Rubbish region detection model
- `rubbish_classification.pt` - Rubbish type classification model
- `traffic_sign.pt` - Damaged traffic sign detection model

## Backend Setup

### Step 1: Create Virtual Environment

**Windows:**
```bash
cd backend
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- FastAPI - Web framework
- Uvicorn - ASGI server
- SQLAlchemy - ORM
- PyTorch - Deep learning framework
- OpenCV - Image processing
- Pydantic - Data validation
- And other utilities...

### Step 3: Place Model Files

Create the models directory:
```bash
mkdir -p app/models_data
```

Copy your model files:
```bash
# Windows
copy "C:\path\to\rubbish_area.pt" app\models_data\
copy "C:\path\to\rubbish_classification.pt" app\models_data\
copy "C:\path\to\traffic_sign.pt" app\models_data\

# macOS/Linux
cp ~/models/rubbish_area.pt app/models_data/
cp ~/models/rubbish_classification.pt app/models_data/
cp ~/models/traffic_sign.pt app/models_data/
```

### Step 4: Configure Environment

Edit or create `.env` file in the `backend` directory:

```env
# API Settings
DEBUG=True
API_TITLE=Object Detection API

# Database
DATABASE_URL=sqlite:///./detection_results.db

# Model paths (relative to app/models_data/)
RUBBISH_AREA_MODEL=rubbish_area.pt
RUBBISH_CLASSIFICATION_MODEL=rubbish_classification.pt
TRAFFIC_SIGN_MODEL=traffic_sign.pt

# Image processing
MAX_IMAGE_SIZE=52428800
IMAGE_RESIZE_SIZE=640,640
ALLOWED_IMAGE_FORMATS=jpg,jpeg,png,bmp

# Inference
CONFIDENCE_THRESHOLD=0.5
IOU_THRESHOLD=0.45
```

### Step 5: Initialize Database

```bash
cd backend
python -c "from app.models.database import init_db; init_db(); print('✓ Database initialized')"
```

This creates `detection_results.db` in the backend directory.

## Frontend Setup

### Step 1: Install Dependencies

```bash
cd frontend
npm install
```

### Step 2: Configure Environment

Create `.env.local` file in the `frontend` directory:

```env
VITE_API_BASE_URL=http://localhost:8000
VITE_API_TIMEOUT=30000
```

### Step 3: Build Frontend (Optional)

For production build:
```bash
npm run build
```

This creates optimized files in `dist/` directory.

## Model Configuration

### Model Loading Process

1. Backend identifies available models from `app/models_data/` directory
2. On startup, backend attempts to load all configured models
3. Frontend queries `/api/models` endpoint to get loaded models list
4. User selects a model from the dropdown
5. Inference runs using selected model

### Adding New Models

To add a new YOLO model:

1. Add model file to `backend/app/models_data/sample_name.pt`
2. Update `backend/app/config.py` to add model configuration:
   ```python
   YOUR_MODEL: str = "sample_name.pt"
   ```
3. Update `backend/app/services/model_loader.py` model configs dictionary:
   ```python
   self.model_configs = {
       "your_model_key": {
           "file": "sample_name.pt",
           "description": "Description of your model"
       },
       # ... existing models
   }
   ```
4. Update `backend/app/routers/detect.py` if your model requires special post-processing

### Model Performance Tuning

Adjust thresholds in `backend/app/config.py`:

```python
# Lower for more detections (more false positives)
CONFIDENCE_THRESHOLD=0.3  # Default: 0.5

# Lower for more overlapping boxes to survive NMS
IOU_THRESHOLD=0.3  # Default: 0.45
```

## Running the Application

### Option 1: Development Mode (Recommended)

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate  # or venv\Scripts\activate on Windows
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

Expected output:
```
VITE v5.0.8  ready in XXX ms

➜  Local:   http://localhost:3000/
```

Open browser to http://localhost:3000

### Option 2: Production Mode

**Build Frontend:**
```bash
cd frontend
npm run build
```

**Start Backend:**
```bash
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Serve frontend from a static server or integrate with backend.

## Verification

### Step 1: Health Check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "ok",
  "models_loaded": 3,
  "total_models": 3
}
```

### Step 2: List Models

```bash
curl http://localhost:8000/api/models
```

Expected response:
```json
{
  "models": [
    {
      "name": "rubbish_area",
      "type": "rubbish_area",
      "description": "Detects rubbish area regions in images",
      "loaded": true
    },
    ...
  ],
  "count": 3
}
```

### Step 3: Test Detection API

Create a test image file, then:

```bash
curl -X POST -F "file=@test_image.jpg" -F "model_type=rubbish_area" \
  http://localhost:8000/api/detect
```

Expected response:
```json
{
  "id": 1,
  "model_type": "rubbish_area",
  "detections": [
    {
      "x_min": 100.5,
      "y_min": 150.2,
      "x_max": 250.8,
      "y_max": 300.5,
      "label": "rubbish_area",
      "confidence": 0.95
    }
  ],
  "image_height": 480,
  "image_width": 640,
  "inference_time_ms": 125.45,
  "created_at": "2026-04-04T12:00:00"
}
```

### Step 4: Test Frontend

1. Open http://localhost:3000
2. Wait for models to load (should show 3 models)
3. Upload a video file
4. Select a model
5. Click play and capture a frame
6. Click "Capture & Detect"
7. Should see detection results with bounding boxes

## Deployment

### Local Network Access

To access the application from other machines:

**Update Frontend API URL:**
Edit `frontend/.env.local`:
```env
VITE_API_BASE_URL=http://YOUR_IP_ADDRESS:8000
```

**Start Backend with public binding:**
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker Deployment (Optional)

Create `Dockerfile` (backend):
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app
COPY backend/app/models_data ./app/models_data

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t object-detection-backend .
docker run -p 8000:8000 object-detection-backend
```

### Cloud Deployment (AWS/Azure/GCP)

General steps:
1. Set up a cloud VM or App Service
2. Install Python, Node.js
3. Clone repository
4. Follow backend and frontend setup steps above
5. Use gunicorn for production ASGI server:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app
   ```
6. Use nginx as reverse proxy
7. Set up HTTPS with Let's Encrypt

## Troubleshooting

### Backend Issues

**Error: Module not found**
```
ModuleNotFoundError: No module named 'torch'
```
Solution:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# Or for GPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Error: Model not loaded**
```
Model 'rubbish_area' is not loaded or unavailable.
```
Solution:
- Check model file exists in `backend/app/models_data/`
- Check filename matches config in `app/config.py`
- Check disk space (models are large files)
- Check file permissions

**Error: Database locked**
```
sqlite3.OperationalError: database is locked
```
Solution:
- Close all connections to the database
- Delete `detection_results.db` and restart backend:
  ```bash
  rm detection_results.db
  python -m uvicorn app.main:app --reload
  ```

### Frontend Issues

**Error: Cannot connect to API**
```
AxiosError: connect ECONNREFUSED 127.0.0.1:8000
```
Solution:
- Ensure backend is running on port 8000
- Check `VITE_API_BASE_URL` in `.env.local`
- Check browser CORS headers in console

**Error: Video won't play**
Solution:
- Check video format is supported (MP4, AVI, MOV, WebM)
- Check video codec is compatible with browser
- Try converting: `ffmpeg -i input.avi -codec:v libx264 output.mp4`

**Error: Models list not loading**
Solution:
- Open browser DevTools (F12)
  - Check Network tab for `/api/models` request
  - Check Console for JavaScript errors
- Verify backend health at `/health` endpoint

### GPU/Performance Issues

**Out of memory during inference**
```
RuntimeError: CUDA out of memory
```
Solution:
- Reduce image size in `app/config.py`: `IMAGE_RESIZE_SIZE = (416, 416)`
- Use CPU instead: Edit models to run on CPU in `inference.py`
- Use smaller model variants

**Slow inference**
Solution:
- Check if GPU is being used: `nvidia-smi` (NVIDIA GPUs)
- Reduce `IMAGE_RESIZE_SIZE` for faster processing
- Use model quantization or pruning

## Support & Documentation

- **API Docs**: http://localhost:8000/docs (Swagger UI)
- **Redoc Docs**: http://localhost:8000/redoc
- **Architecture**: See [ARCHITECTURE.md](ARCHITECTURE.md)
- **Issues**: Create GitHub issue with error logs

---

**Last Updated**: April 2026  
**Version**: 1.0.0
