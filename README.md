# Object Detection System

A full-stack web application for real-time object detection and classification. Features video analysis and Google Maps integration for detecting:

- **Rubbish Areas**: Identify regions with rubbish accumulation
- **Rubbish Classification**: Classify rubbish into ~10 types (plastic, paper, metal, glass, etc.)
- **Damaged Traffic Signs**: Detect and flag damaged or defaced traffic signs

## Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn
- YOLO model files (.pt format)

### Setup

1. **Clone the repository**
```bash
git clone <repo-url>
cd ai-engineer
```

2. **Backend Setup** (see [backend/README.md](backend/README.md))
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m app.main
```

3. **Frontend Setup** (see [frontend/README.md](frontend/README.md))
```bash
cd frontend
npm install
npm run dev
```

4. **Access the app**: Open http://localhost:3000 in your browser

## Features

### Video Analysis
- Upload video files (MP4, AVI, MOV, WebM)
- Play/pause/stop video playback
- Capture individual frames during playback
- Run detection on captured frames
- View results with bounding boxes and labels

### Map Analysis
- View street maps (Google Maps integration)
- Capture current map view
- Run detection on map views
- Identify objects in street scenes

### Detection Models
- Model selection dropdown
- Real-time inference
- Confidence score display
- Bounding box overlay rendering
- Detection history tracking

## Project Structure

```
ai-engineer/
├── backend/                          # FastAPI application
│   ├── app/
│   │   ├── main.py                  # FastAPI entry point
│   │   ├── config.py                # Configuration
│   │   ├── models/                  # Database & schemas
│   │   ├── services/                # Business logic
│   │   ├── routers/                 # API endpoints
│   │   ├── utils/                   # Utilities
│   │   └── models_data/             # Model storage
│   ├── requirements.txt             # Python dependencies
│   └── README.md                    # Backend documentation
│
├── frontend/                         # React TypeScript app
│   ├── src/
│   │   ├── components/              # Reusable components
│   │   ├── pages/                   # Feature pages
│   │   ├── hooks/                   # Custom React hooks
│   │   ├── services/                # API client
│   │   ├── types/                   # TypeScript types
│   │   └── App.tsx                  # Main app
│   ├── package.json                 # Node dependencies
│   ├── vite.config.ts              # Vite config
│   └── README.md                    # Frontend documentation
│
├── shared/                          # Shared types (if needed)
├── .env                             # Environment variables
├── README.md                        # This file
├── SETUP.md                         # Detailed setup guide
└── ARCHITECTURE.md                  # Architecture documentation
```

## Architecture Overview

### High-Level Flow
```
React Frontend (TypeScript + Vite)
        ↓ (HTTP/JSON)
FastAPI Backend (Python 3.8+)
        ↓ (Model inference)
PyTorch YOLO Models
        ↓ (Results)
SQLite Database
```

## Documentation

- **[SETUP.md](SETUP.md)** - Complete setup and configuration guide
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Detailed architecture and design
- **[backend/README.md](backend/README.md)** - Backend API documentation
- **[frontend/README.md](frontend/README.md)** - Frontend documentation

## Development

### Backend Development
```bash
cd backend
source venv/bin/activate
python -m uvicorn app.main:app --reload
```

### Frontend Development
```bash
cd frontend
npm run dev
```

### Code Quality

```bash
# Backend type checking (once installed)
# cd backend && mypy app/

# Frontend type checking & linting
cd frontend
npm run lint
npm run type-check
```

## API Endpoints

### Detection
- **POST** `/api/detect` - Upload image and detect objects
- **POST** `/api/detect-base64` - Detect from base64-encoded image

### Models
- **GET** `/api/models` - List available models

### Results
- **GET** `/api/results` - Detection history
- **GET** `/api/results/{id}` - Specific result

### Health
- **GET** `/health` - API health status
- **GET** `/` - API information

Interactive docs: http://localhost:8000/docs (Swagger UI)

## Performance

- **Inference Time**: 50-200ms per image (depends on model)
- **Model Loading**: ~2-5 seconds on startup for all 3 models
- **Video Processing**: Real-time frame capture and detection
- **Database**: SQLite for local dev (PostgreSQL for production)

## Troubleshooting

### Backend Issues
- **Models not found**: Place .pt files in `backend/app/models_data/`
- **Import errors**: Run `pip install -r requirements.txt`
- **Database locked**: Delete `detection_results.db` and restart

### Frontend Issues
- **API connection error**: Check backend runs on http://localhost:8000
- **Video upload fails**: Ensure file size < 500MB and format is supported
- **Models not loading**: Check `/api/models` endpoint in browser Network tab

## Technology Stack

**Backend:**
- FastAPI 0.104+
- SQLAlchemy 2.0+
- PyTorch 2.1+
- OpenCV 4.8+
- Python 3.8+

**Frontend:**
- React 18.2+
- TypeScript 5.2+
- Vite 5.0+
- Zustand 4.4+
- Axios 1.6+

## Contributing

1. Create a feature branch
2. Follow the project's code style
3. Write clear commit messages
4. Submit a pull request

## License

See [LICENSE](LICENSE) file.

## Support

For issues, questions, or suggestions, please open a GitHub issue.

---

**Version**: 1.0.0  
**Status**: Active Development  
**Last Updated**: April 2026
