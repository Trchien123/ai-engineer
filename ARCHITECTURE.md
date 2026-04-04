# Architecture Overview

## System Design

The Object Detection System is built with a clean separation of concerns:

- **Frontend**: React TypeScript React + Vite (SPA)
- **Backend**: FastAPI + SQLAlchemy (REST API)
- **ML Pipeline**: PyTorch-based inference engine
- **Database**: SQLite (local dev) / PostgreSQL (production)

## Directory Structure & Modules

### Backend Architecture

```
backend/
├── app/
│   ├── main.py                    # FastAPI application entry point
│   │                             # - CORS middleware setup
│   │                             # - Lifespan events (startup/shutdown)
│   │                             # - Route registration
│   │                             # - Health check endpoint
│   │
│   ├── config.py                 # Configuration management
│   │                             # - Settings from environment variables
│   │                             # - Model paths, inference thresholds
│   │                             # - Database URL
│   │
│   ├── models/
│   │   ├── database.py          # SQLAlchemy ORM models & database setup
│   │   │                        # - DetectionResult model (schema)
│   │   │                        # - Database session management
│   │   │                        # - Database initialization
│   │   │
│   │   ├── schemas.py           # Pydantic request/response schemas
│   │   │                        # - DetectionObject (bbox + label + confidence)
│   │   │                        # - DetectionResultResponse (API response)
│   │   │                        # - ModelInfo, ModelsListResponse
│   │   │                        # - HealthCheckResponse
│   │   │
│   │   └── __init__.py
│   │
│   ├── services/
│   │   ├── model_loader.py      # Model management service
│   │   │                        # - ModelLoader class: load .pt files dynamically
│   │   │                        # - Model caching (in-memory)
│   │   │                        # - Model availability checking
│   │   │
│   │   ├── inference.py         # ML inference pipeline
│   │   │                        # - DetectionPipeline class:
│   │   │                        # •  preprocess_image() - resize, normalize
│   │   │                        # •  extract_detections() - parse model output
│   │   │                        # •  apply_nms() - non-maximum suppression
│   │   │                        # •  calculate_iou() - intersection over union
│   │   │                        # - InferenceEngine class:
│   │   │                        # •  infer() - complete pipeline orchestration
│   │   │
│   │   ├── storage.py           # Image handling service
│   │   │                        # - ImageStorageService class:
│   │   │                        # •  validate_file_size()
│   │   │                        # •  validate_image_format()
│   │   │                        # •  load_image_from_base64()
│   │   │                        # •  load_image_from_bytes()
│   │   │                        # •  save_temp_image()
│   │   │                        # •  cleanup_temp_images()
│   │   │
│   │   └── __init__.py
│   │
│   ├── routers/
│   │   ├── detect.py            # Detection endpoints
│   │   │                        # - POST /api/detect (file upload)
│   │   │                        # - POST /api/detect-base64 (base64 image)
│   │   │                        # - Both endpoints: run inference + store results
│   │   │
│   │   ├── models.py            # Model management endpoints
│   │   │                        # - GET /api/models (list available models)
│   │   │
│   │   ├── results.py           # Detection history endpoints
│   │   │                        # - GET /api/results (paginated history)
│   │   │                        # - GET /api/results/{id} (specific result)
│   │   │
│   │   └── __init__.py
│   │
│   ├── utils/
│   │   ├── exceptions.py        # Custom exception classes
│   │   │                        # - ModelNotLoadedError
│   │   │                        # - InvalidImageError
│   │   │                        # - FileSizeExceededError
│   │   │                        # - ModelInferenceError
│   │   │
│   │   ├── logger.py            # Logging configuration
│   │   │                        # - get_logger() - structured logging
│   │   │
│   │   └── __init__.py
│   │
│   ├── models_data/             # YOLO model storage
│   │   ├── rubbish_area.pt
│   │   ├── rubbish_classification.pt
│   │   └── traffic_sign.pt
│   │
│   └── __init__.py
│
├── requirements.txt             # Python dependencies
├── .env                        # Environment configuration
└── README.md                   # Backend documentation
```

### Frontend Architecture

```
frontend/
├── src/
│   ├── types/
│   │   └── detection.ts         # TypeScript type definitions
│   │                           # - DetectionBox, DetectionResult
│   │                           # - ModelInfo, ModelType
│   │                           # - API response types
│   │
│   ├── services/
│   │   └── apiClient.ts        # API client service
│   │                           # - DetectionService (detect endpoints)
│   │                           # - ModelService (model management)
│   │                           # - HistoryService (results retrieval)
│   │                           # - HealthService (health checks)
│   │                           # - Uses axios for HTTP requests
│   │
│   ├── hooks/
│   │   ├── useVideoCapture.ts  # Video capture hook
│   │   │                       # - videoRef, canvasRef management
│   │   │                       # - captureFrame() - extract frame from video
│   │   │                       # - playVideo(), pauseVideo(), stopVideo()
│   │   │                       # - loadVideo() - load video file
│   │   │
│   │   ├── useMapCapture.ts    # Map capture hook
│   │   │                       # - mapContainerRef management
│   │   │                       # - captureMapView() - screenshot using html2canvas
│   │   │
│   │   ├── useDetectionState.ts # State management (Zustand store)
│   │   │                        # State:
│   │   │                        # - availableModels, selectedModel
│   │   │                        # - currentImage, detectionResult
│   │   │                        # - detecting (loading state), errors
│   │   │                        # - activeTab (video/map), showHistory
│   │   │                        # Actions: setters, resetters
│   │   │
│   │   └── index.ts
│   │
│   ├── components/
│   │   ├── ModelSelector.tsx    # Model dropdown selector
│   │   │ + ModelSelector.css   # - Display available models
│   │   │                       # - Show model status (loaded/not loaded)
│   │   │                       # - Model description
│   │   │
│   │   ├── DetectionResults.tsx  # Results visualization
│   │   │ + DetectionResults.css # - Canvas overlay rendering
│   │   │                        # - Draw bounding boxes with labels
│   │   │                        # - Display confidence scores
│   │   │                        # - Show detection summary list
│   │   │
│   │   ├── LoadingSpinner.tsx   # Loading indicator
│   │   │ + LoadingSpinner.css  # - Animated spinner
│   │   │                        # - Configurable size & message
│   │   │
│   │   ├── ErrorMessage.tsx     # Error display component
│   │   │ + ErrorMessage.css    # - Styled error alert
│   │   │                        # - Dismissible
│   │   │
│   │   └── index.ts
│   │
│   ├── pages/
│   │   ├── VideoUploader.tsx    # Video file upload
│   │   │ + VideoUploader.css   # - Drag-and-drop area
│   │   │                        # - File validation (format, size)
│   │   │                        # - Upload button
│   │   │
│   │   ├── VideoPlayer.tsx      # Video playback & frame capture
│   │   │ + VideoPlayer.css     # - HTML5 video player
│   │   │                        # - Play/pause/stop buttons
│   │   │                        # - Capture button with detection
│   │   │                        # - Frame preview
│   │   │
│   │   ├── MapViewer.tsx        # Map view & capture
│   │   │ + MapViewer.css       # - Google Maps placeholder
│   │   │                        # - Capture map view button
│   │   │                        # - Map capture preview
│   │   │
│   │   └── __init__.ts
│   │
│   ├── App.tsx                 # Main application component
│   │                           # - Route between video/map tabs
│   │                           # - Initialize models on mount
│   │                           # - Health check & error handling
│   │                           # - Display detection results
│   │
│   ├── App.css                 # Global app styles
│   │                           # - Layout, header, footer
│   │                           # - Tab navigation
│   │                           # - Responsive design
│   │
│   ├── index.css               # Global styles
│   │                           # - Button, input, form styles
│   │
│   ├── main.tsx                # React entry point
│   │                           # - ReactDOM.createRoot()
│   │                           # - Render App component
│   │
│   └── vite-env.d.ts           # Vite type definitions
│
├── public/                     # Static assets
│
├── index.html                  # HTML entry point
│
├── package.json                # Node dependencies
├── vite.config.ts              # Vite configuration
├── tsconfig.json               # TypeScript config
├── tsconfig.node.json          # Vite config TypeScript
└── README.md                   # Frontend documentation
```

## Data Flow Diagrams

### Video Detection Flow

```
User Action: Upload Video
        ↓
[VideoUploader] - Validates file size & format
        ↓
[VideoPlayer] - Load & display video using <video> element
        ↓
User Action: Click "Capture & Detect" button
        ↓
[useVideoCapture] - captureFrame()
  └─→ Canvas draws current video frame
  └─→ Canvas.toDataURL() → base64 string
        ↓
[useDetectionStore] - Store captured image
  └─→ setCurrentImage()
        ↓
[apiClient] - DetectionService.detectFromBase64()
  └─→ POST /api/detect-base64
  └─→ Send: { image_base64, model_type }
        ↓
[Backend: app/routers/detect.py]
  └─→ Validate input
  └─→ Load image from base64 (ImageStorageService)
  └─→ Run inference (InferenceEngine.infer())
          ├─→ Preprocess (resize, normalize)
          ├─→ Forward pass (load model, run inference)
          ├─→ Post-process (NMS, filtering)
        ↓
  └─→ Save result to database (DetectionResult model)
  └─→ Return: { id, model_type, detections, inference_time_ms, ... }
        ↓
[Frontend: setDetectionResult()]
  └─→ Store result in Zustand state
        ↓
[DetectionResults] - Render results
  └─→ Draw bounding boxes on canvas overlay
  └─→ Display detection list with confidence scores
```

### Model Selection Flow

```
User Action: Select Model from Dropdown
        ↓
[ModelSelector] - onChange event
        ↓
[useDetectionStore] - setSelectedModel(modelType)
        ↓
Next detection request will use new model
        ↓
[Backend] - Load selected model on demand
  └─→ ModelLoader.get_model(model_name)
  └─→ Returns cached model or None
  └─→ If None, returns error
```

### State Management

**Frontend State (Zustand Store):**
```javascript
{
  // Model management
  availableModels: ModelsListResponse | null
  selectedModel: 'rubbish_area' | 'rubbish_classification' | 'traffic_sign'
  modelsLoading: boolean
  modelsError: string | null
  
  // Current detection
  currentImage: string | null  // base64
  currentImageSize: { width, height } | null
  detectionResult: DetectionResult | null
  detecting: boolean
  detectionError: string | null
  
  // UI state
  activeTab: 'video' | 'map'
  showHistory: boolean
}
```

## API Endpoint Design

### Detection Endpoints

**POST /api/detect**
- Content-Type: multipart/form-data
- Parameters:
  - `file` (binary) - Image file
  - `model_type` (string) - Model selection
- Response: DetectionResult

**POST /api/detect-base64**
- Content-Type: application/json
- Body:
  ```json
  {
    "image_base64": "data:image/jpeg;base64,...",
    "model_type": "rubbish_area"
  }
  ```
- Response: DetectionResult

### Response Format

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

## Error Handling

### Backend Error Responses

```json
{
  "detail": "Error message describing what went wrong"
}
```

HTTP Status Codes:
- `200 OK` - Successful detection
- `400 Bad Request` - Invalid input (image format, size)
- `404 Not Found` - Result ID not found
- `413 Payload Too Large` - File size exceeded
- `500 Internal Server Error` - Server error
- `503 Service Unavailable` - Model not loaded

### Frontend Error Handling

- API errors caught in try-catch blocks
- Error stored in Zustand state: `setDetectionError()`
- `<ErrorMessage>` component displays errors
- User can dismiss errors

## Performance Optimization Strategies

### Backend
- **Model Caching**: Models loaded once and cached in memory
- **Image Preprocessing**: Efficient OpenCV operations
- **Batch Processing**: Can extend for batch detection
- **Database Indexing**: Indexes on `model_type` and `created_at`

### Frontend
- **Lazy Loading**: Zustand store setup enables code splitting
- **Canvas Rendering**: Efficient 2D canvas API for overlays
- **Image Compression**: JPEG compression before sending to backend
- **Memoization**: React.memo() for component optimization (can add)

## Extension Points

### Add New Detection Model

1. Place .pt file in `backend/app/models_data/`
2. Update `backend/app/config.py` with model path
3. Add config to `model_loader.py` model_configs dictionary
4. Model automatically appears in GET /api/models

### Add Custom Post-Processing

In `backend/app/services/inference.py`:
- Override `DetectionPipeline.extract_detections()` for custom output parsing
- Modify NMS parameters or add custom filtering logic

### Add Detection History UI

Create `HistoryViewer.tsx` component:
- Fetch history using `HistoryService.getHistory()`
- Display paginated results in grid/table
- Allow filtering by model type and date

## Security Considerations

- File size limits (50MB max)
- File type validation (image formats only)
- Input sanitization (Pydantic schemas)
- CORS configured (can restrict to specific origins)
- No authentication (add if needed)

## Database Schema

```sql
CREATE TABLE detection_results (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  model_type VARCHAR NOT NULL,
  image_path VARCHAR,
  detections JSON,
  image_height INTEGER,
  image_width INTEGER,
  inference_time_ms FLOAT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  notes TEXT
);

CREATE INDEX idx_model_type ON detection_results(model_type);
CREATE INDEX idx_created_at ON detection_results(created_at);
```

---

**Last Updated**: April 2026  
**Version**: 1.0.0
