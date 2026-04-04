"""
Database setup and models using SQLAlchemy.
"""
import os
from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from app.config import settings

# Create database engine
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


class DetectionResult(Base):
    """Model for storing detection results."""
    
    __tablename__ = "detection_results"
    
    id = Column(Integer, primary_key=True, index=True)
    model_type = Column(String, index=True)  # rubbish_area, rubbish_classification, traffic_sign
    image_path = Column(String)  # Path to stored image
    detections = Column(JSON)  # List of detected objects with bbox, label, confidence
    image_height = Column(Integer)
    image_width = Column(Integer)
    inference_time_ms = Column(Float)  # Inference execution time
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    notes = Column(Text, nullable=True)  # Optional notes


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency for database session in FastAPI endpoints."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
