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
    TRAFFIC_SIGN_MODEL: str = "yolov26_1_class.pt"

    DAMAGED_SIGN_DIR: Path = Path(__file__).parent.parent / "app" / "models_data" / "damaged_sign_detection"

    RUBBISH_DETECTION_DIR: Path = Path(__file__).parent.parent / "app" / "models_data" / "rubbish_detection"
    RUBBISH_STAGE1_MODEL: str = "stage1_best.pt"
    RUBBISH_STAGE2_MODEL: str = "stage2_best.pt"
    TRAFFIC_SIGN_CLASSIFIER_MODEL: str = "EffnetV2_multilabel.pth"
    TRAFFIC_SIGN_RETRIEVER_INDEX: str = "traffic_signs_3.index"
    TRAFFIC_SIGN_RETRIEVER_METADATA: str = "traffic_signs_metadata_3.json"

    # Image and video uploads
    MAX_IMAGE_SIZE: int = 50 * 1024 * 1024  # 50 MB
    MAX_VIDEO_SIZE: int = 500 * 1024 * 1024  # 500 MB
    ALLOWED_IMAGE_FORMATS: list = ["jpg", "jpeg", "png", "bmp"]
    ALLOWED_VIDEO_FORMATS: list = ["mp4", "avi", "mov", "webm", "mkv"]
    IMAGE_RESIZE_SIZE: tuple = (640, 640)  # Standard for YOLO
    
    # Inference
    CONFIDENCE_THRESHOLD: float = 0.5
    IOU_THRESHOLD: float = 0.45
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
