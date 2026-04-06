"""
Pydantic schemas for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class DetectionObject(BaseModel):
    """Single detected object with bounding box and classification."""
    
    class Config:
        from_attributes = True
    
    x_min: float = Field(..., description="Left edge of bounding box (pixels)")
    y_min: float = Field(..., description="Top edge of bounding box (pixels)")
    x_max: float = Field(..., description="Right edge of bounding box (pixels)")
    y_max: float = Field(..., description="Bottom edge of bounding box (pixels)")
    label: str = Field(..., description="Class label (e.g., 'plastic', 'damaged_sign')")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score [0, 1]")


class DetectionRequestBase(BaseModel):
    """Base class for detection requests."""
    model_type: str = Field(
        ...,
        description="Model type: rubbish_area, rubbish_classification, or traffic_sign"
    )


class DetectionResultResponse(BaseModel):
    """Response containing detection results."""
    
    class Config:
        from_attributes = True
    
    id: int
    model_type: str
    detections: List[DetectionObject]
    image_height: int
    image_width: int
    inference_time_ms: float
    created_at: datetime


class ModelInfo(BaseModel):
    """Information about an available model."""
    
    name: str = Field(..., description="Model identifier (unique)")
    type: str = Field(..., description="Model type (rubbish_area, rubbish_classification, traffic_sign)")
    description: str = Field(..., description="Human-readable description")
    loaded: bool = Field(..., description="Whether model is currently loaded")


class ModelsListResponse(BaseModel):
    """Response containing list of available models."""
    
    models: List[ModelInfo]
    count: int


class HealthCheckResponse(BaseModel):
    """Response for health check endpoint."""
    
    status: str = Field(default="ok", description="Service status")
    models_loaded: int = Field(..., description="Number of loaded models")
    database_connected: bool = Field(..., description="Database connection status")


class DetectionHistoryResponse(BaseModel):
    """Response for detection history."""

    results: List[DetectionResultResponse]
    total: int = Field(..., description="Total number of results")
    limit: int = Field(..., description="Results returned (pagination limit)")


class ExplainResult(BaseModel):
    """Single XAI result for one predicted damage class."""

    class_name: str = Field(..., description="Predicted damage class (e.g. 'rust')")
    probability: float = Field(..., ge=0.0, le=1.0, description="Sigmoid probability [0, 1]")
    image_base64: str = Field(
        ...,
        description=(
            "Base64-encoded PNG visualisation. "
            "Use as: <img src='data:image/png;base64,{image_base64}'>"
        ),
    )


class ExplainResponse(BaseModel):
    """Response from POST /api/explain."""

    method: str = Field(..., description="XAI method used: grad_cam | shap | zennit")
    results: List[ExplainResult] = Field(..., description="One entry per active damage class")
    inference_time_ms: float = Field(..., description="Time taken for the XAI pass (ms)")
