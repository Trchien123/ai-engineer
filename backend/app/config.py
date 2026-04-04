"""
Configuration management for the FastAPI application.
Loads settings from environment variables.
"""
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Settings
    API_TITLE: str = "Object Detection API"
    API_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Database
    DATABASE_URL: str = "sqlite:///./detection_results.db"
    
    # Model paths
    MODELS_DIR: Path = Path(__file__).parent.parent / "app" / "models_data"
    RUBBISH_AREA_MODEL: str = "rubbish_area.pt"
    RUBBISH_CLASSIFICATION_MODEL: str = "rubbish_classification.pt"
    TRAFFIC_SIGN_MODEL: str = "traffic_sign.pt"
    
    # Image processing
    MAX_IMAGE_SIZE: int = 50 * 1024 * 1024  # 50 MB
    ALLOWED_IMAGE_FORMATS: list = ["jpg", "jpeg", "png", "bmp"]
    IMAGE_RESIZE_SIZE: tuple = (640, 640)  # Standard for YOLO
    
    # Inference
    CONFIDENCE_THRESHOLD: float = 0.5
    IOU_THRESHOLD: float = 0.45
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
