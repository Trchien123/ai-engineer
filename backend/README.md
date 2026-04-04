# Backend Documentation

The backend is a FastAPI-based REST API that handles all machine learning inference, model management, and result storage.

## Quick Start

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

API available at: http://localhost:8000  
Docs at: http://localhost:8000/docs

## Architecture

### Core Components

1. **FastAPI Application** (`app/main.py`)
   - ASGI server with uvicorn
   - CORS middleware for frontend communication
   - Startup/shutdown lifecycle management
   - Health check endpoint

2. **Configuration** (`app/config.py`)
   - Environment-based settings
   - Model paths and inference thresholds
   - Database configuration
   - Image processing parameters

3. **Models & Schemas** (`app/models/`)
   - SQLAlchemy ORM models (database)
   - Pydantic validation schemas (API)

4. **Services** (`app/services/`)
   - `model_loader.py` - Load and cache YOLO models
   - `inference.py` - Image preprocessing and inference pipeline
   - `storage.py` - Image validation and handling

5. **Routers** (`app/routers/`)
   - `detect.py` - Detection endpoints
   - `models.py` - Model management endpoints
   - `results.py` - Detection history endpoints

## API Endpoints

### Detection

#### POST `/api/detect`
Upload image file and run detection.

**Request:**
```bash
curl -X POST -F "file=@image.jpg" -F "model_type=rubbish_area" \
  http://localhost:8000/api/detect
```

**Response:**
```json
{
  "id": 1,
  "model_type": "rubbish_area",
  "detections": [
    {
      "x_min": 100,
      "y_min": 150,
      "x_max": 250,
      "y_max": 300,
      "label": "rubbish",
      "confidence": 0.95
    }
  ],
  "image_height": 480,
  "image_width": 640,
  "inference_time_ms": 125.45,
  "created_at": "2026-04-04T12:00:00"
}
```

#### POST `/api/detect-base64`
Run detection on base64-encoded image.

**Request:**
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{
    "image_base64": "data:image/jpeg;base64,...",
    "model_type": "traffic_sign"
  }' \
  http://localhost:8000/api/detect-base64
```

**Response:** Same as above

### Models

#### GET `/api/models`
List available detection models and their status.

**Response:**
```json
{
  "models": [
    {
      "name": "rubbish_area",
      "type": "rubbish_area",
      "description": "Detects rubbish area regions in images",
      "loaded": true
    },
    {
      "name": "rubbish_classification",
      "type": "rubbish_classification",
      "description": "Classifies rubbish into ~10 types",
      "loaded": true
    },
    {
      "name": "traffic_sign",
      "type": "traffic_sign",
      "description": "Detects damaged traffic signs",
      "loaded": true
    }
  ],
  "count": 3
}
```

### Results

#### GET `/api/results`
Get detection history with pagination.

**Query Parameters:**
- `model_type` (optional) - Filter by model type
- `limit` (default: 100, max: 1000) - Results per page
- `offset` (default: 0) - Pagination offset

**Response:**
```json
{
  "results": [
    {
      "id": 1,
      "model_type": "rubbish_area",
      "detections": [...],
      "image_height": 480,
      "image_width": 640,
      "inference_time_ms": 125.45,
      "created_at": "2026-04-04T12:00:00"
    }
  ],
  "total": 42,
  "limit": 100
}
```

#### GET `/api/results/{result_id}`
Get specific detection result by ID.

**Response:** Single DetectionResult object (same structure as above)

### Health Check

#### GET `/health`
Check API health and model status.

**Response:**
```json
{
  "status": "ok",
  "models_loaded": 3,
  "total_models": 3
}
```

#### GET `/`
Get API information.

**Response:**
```json
{
  "name": "Object Detection API",
  "version": "1.0.0",
  "endpoints": {
    "health": "/health",
    "docs": "/docs",
    "openapi": "/openapi.json",
    "models_list": "/api/models",
    "detection": "/api/detect",
    "detection_base64": "/api/detect-base64",
    "results_history": "/api/results",
    "result_by_id": "/api/results/{result_id}"
  }
}
```

## Configuration

### Environment Variables (.env)

```env
# API Settings
DEBUG=True                          # Enable debug mode
API_TITLE=Object Detection API      # API name

# Database
DATABASE_URL=sqlite:///./detection_results.db

# Model Files (relative to app/models_data/)
RUBBISH_AREA_MODEL=rubbish_area.pt
RUBBISH_CLASSIFICATION_MODEL=rubbish_classification.pt
TRAFFIC_SIGN_MODEL=traffic_sign.pt

# Image Processing
MAX_IMAGE_SIZE=52428800             # 50MB in bytes
IMAGE_RESIZE_SIZE=640,640           # YOLO standard size
ALLOWED_IMAGE_FORMATS=jpg,jpeg,png,bmp

# Inference
CONFIDENCE_THRESHOLD=0.5            # Minimum confidence to keep detection
IOU_THRESHOLD=0.45                  # NMS intersection-over-union threshold
```

## Services

### ModelLoader

Dynamically loads YOLO .pt model files and manages caching.

```python
from app.services.model_loader import model_loader

# Load single model
model = model_loader.load_model("rubbish_area")

# Load all configured models
results = model_loader.load_all_models()
# Returns: {"rubbish_area": True, "rubbish_classification": True, ...}

# Check if model is loaded
is_loaded = model_loader.is_loaded("rubbish_area")  # True/False

# Get all model info
info = model_loader.get_all_model_info()
```

### InferenceEngine

Orchestrates the complete inference pipeline: preprocessing → inference → postprocessing.

```python
from app.services.inference import InferenceEngine
import cv2

engine = InferenceEngine(model_loader)

# Load image
image = cv2.imread("image.jpg")

# Run inference
result = engine.infer(image, "rubbish_area")
# Returns: {
#   'detections': [DetectionBox, ...],
#   'inference_time_ms': 125.45,
#   'original_size': (640, 480),
#   'error': None
# }
```

### ImageStorageService

Handles image validation, loading, and storage.

```python
from app.services.storage import ImageStorageService

# Load from base64
image_array = ImageStorageService.load_image_from_base64(b64_string)

# Load from bytes
image_array = ImageStorageService.load_image_from_bytes(file_bytes)

# Save temporary image
filepath = ImageStorageService.save_temp_image(image_array)

# Cleanup
ImageStorageService.cleanup_temp_images()
```

## Database Models

### DetectionResult

Stores detection inference results.

```python
class DetectionResult(Base):
    id: int                      # Primary key
    model_type: str              # Model that ran (indexed)
    image_path: str              # Path to stored image
    detections: JSON             # List of detections
    image_height: int            # Original image height
    image_width: int             # Original image width
    inference_time_ms: float     # Execution time in milliseconds
    created_at: DateTime         # When result was stored (indexed)
    notes: str                   # Optional metadata
```

## Development

### Adding a New Model

1. Place .pt file in `backend/app/models_data/`
2. Update `app/config.py`:
   ```python
   class Settings(BaseSettings):
       NEW_MODEL_NAME: str = "new_model.pt"
   ```
3. Update `app/services/model_loader.py`:
   ```python
   def __init__(self):
       self.model_configs = {
           "new_model": {
               "file": "new_model.pt",
               "description": "Description of new model"
           }
       }
   ```

### Custom Post-Processing

Modify `app/services/inference.py`:

```python
def extract_detections(self, model_output, image_size, model_type):
    # Custom parsing for your model's output format
    detections = []
    # ... your parsing logic ...
    return detections
```

### Testing

```bash
# Test health check
curl http://localhost:8000/health

# Test models endpoint
curl http://localhost:8000/api/models

# Test detection with file
curl -X POST -F "file=@test.jpg" -F "model_type=rubbish_area" \
  http://localhost:8000/api/detect

# Test detection with base64
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "...", "model_type": "rubbish_area"}' \
  http://localhost:8000/api/detect-base64
```

## Error Handling

Common error responses:

```json
// 400 - Invalid input
{
  "detail": "Invalid image: Format 'gif' not allowed. Allowed: jpg, jpeg, png, bmp"
}

// 413 - File too large
{
  "detail": "File size exceeds maximum allowed size of 50MB."
}

// 503 - Model not available
{
  "detail": "Model 'rubbish_area' is not loaded or unavailable."
}

// 500 - Inference failed
{
  "detail": "Inference error: CUDA out of memory"
}
```

## Performance Tuning

### Inference Optimization

In `app/config.py`:
```python
# Faster inference, more detections
CONFIDENCE_THRESHOLD = 0.3  # Lower = more detections
IOU_THRESHOLD = 0.4         # Lower = keep more overlapping boxes

# Smaller image size = faster but less accurate
IMAGE_RESIZE_SIZE = (416, 416)  # Default: (640, 640)
```

### Database Optimization

For production with many results, migrate to PostgreSQL:

```env
DATABASE_URL=postgresql://user:password@localhost/detection_db
```

Then run alembic migrations (when added).

## Logging

Structured logging throughout the application:

```python
from app.utils.logger import get_logger

logger = get_logger(__name__)

logger.info("Processing image")
logger.error("Failed to load model")
logger.debug("Model inference time: 125ms")
```

Logs include timestamps, module names, and log levels.

---

**Backend Version**: 1.0.0  
**Python Version**: 3.8+  
**FastAPI Version**: 0.104.1  
**Last Updated**: April 2026
