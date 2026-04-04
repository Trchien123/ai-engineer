"""
FastAPI Application Entry Point.
Main application setup and startup/shutdown logic.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import settings
from app.models.database import init_db
from app.services.model_loader import model_loader
from app.utils.logger import get_logger

from app.routers import detect, models, results

logger = get_logger(__name__)


# Startup and shutdown event handlers
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage app startup and shutdown.
    """
    # Startup
    logger.info("=" * 50)
    logger.info("Starting Object Detection API")
    logger.info("=" * 50)
    
    # Initialize database
    logger.info("Initializing database...")
    init_db()
    logger.info("Database initialized")
    
    # Load models
    logger.info("Loading detection models...")
    load_results = model_loader.load_all_models()
    for model_name, success in load_results.items():
        status = "✓ Loaded" if success else "✗ Failed to load"
        logger.info(f"  {status}: {model_name}")
    
    loaded_count = sum(1 for v in load_results.values() if v)
    logger.info(f"Total models loaded: {loaded_count}/{len(load_results)}")
    
    logger.info("Application ready!")
    logger.info("=" * 50)
    
    yield  # App is running here
    
    # Shutdown
    logger.info("Shutting down...")
    logger.info("Application stopped")


# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="Full-stack object detection API with support for rubbish detection and traffic sign analysis",
    lifespan=lifespan
)

# Add CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(detect.router)
app.include_router(models.router)
app.include_router(results.router)


# Health check endpoint
@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint."""
    loaded_models = sum(1 for m in model_loader.models.values() if m is not None)
    return {
        "status": "ok",
        "models_loaded": loaded_models,
        "total_models": len(model_loader.model_configs)
    }


@app.get("/", tags=["info"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.API_TITLE,
        "version": settings.API_VERSION,
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
