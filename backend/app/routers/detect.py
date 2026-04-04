"""
API router for detection endpoints.
"""
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional
import io

from app.models.database import get_db, DetectionResult
from app.models.schemas import DetectionResultResponse, DetectionRequestBase
from app.services.model_loader import model_loader
from app.services.inference import InferenceEngine
from app.services.storage import ImageStorageService
from app.utils.logger import get_logger
from app.utils.exceptions import InvalidImageError, ModelNotLoadedError

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["detection"])

# Initialize inference engine
inference_engine = InferenceEngine(model_loader)


@router.post("/detect", response_model=DetectionResultResponse)
async def detect(
    model_type: str = Form(..., description="Model type: rubbish_area, rubbish_classification, or traffic_sign"),
    file: UploadFile = File(..., description="Image file (jpg, png, bmp)"),
    db: Session = Depends(get_db)
):
    """
    Run object detection on an uploaded image.
    
    Args:
        model_type: Which detection model to use
        file: Image file to process
        db: Database session
    
    Returns:
        Detection results with bounding boxes and labels
    """
    try:
        # Read file
        contents = await file.read()
        
        logger.info(f"Received detection request: model={model_type}, file_size={len(contents)} bytes")
        
        # Load image
        image_array = ImageStorageService.load_image_from_bytes(contents)
        image_height, image_width = image_array.shape[:2]
        
        # Check if model is loaded
        if not model_loader.is_loaded(model_type):
            raise ModelNotLoadedError(model_type)
        
        # Run inference
        result = inference_engine.infer(image_array, model_type)
        
        if result.get('error'):
            logger.error(f"Inference error: {result['error']}")
            raise HTTPException(status_code=500, detail=result['error'])
        
        # Save to database
        detection_db = DetectionResult(
            model_type=model_type,
            image_path="",  # Could store base64 or file path here if needed
            detections=[det for det in result['detections']],
            image_height=image_height,
            image_width=image_width,
            inference_time_ms=result['inference_time_ms']
        )
        db.add(detection_db)
        db.commit()
        db.refresh(detection_db)
        
        logger.info(f"Detection completed: {len(result['detections'])} objects found")
        
        response = DetectionResultResponse(
            id=detection_db.id,
            model_type=model_type,
            detections=[
                {
                    'x_min': det['x_min'],
                    'y_min': det['y_min'],
                    'x_max': det['x_max'],
                    'y_max': det['y_max'],
                    'label': det['label'],
                    'confidence': det['confidence']
                }
                for det in result['detections']
            ],
            image_height=image_height,
            image_width=image_width,
            inference_time_ms=result['inference_time_ms'],
            created_at=detection_db.created_at
        )
        
        return response
    
    except (InvalidImageError, ModelNotLoadedError) as e:
        logger.error(f"Request validation error: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@router.post("/detect-base64", response_model=DetectionResultResponse)
async def detect_base64(
    model_type: str,
    image_base64: str,
    db: Session = Depends(get_db)
):
    """
    Run object detection on a base64-encoded image.
    Useful for frontend sending captured frames.
    
    Args:
        model_type: Which detection model to use
        image_base64: Base64-encoded image string
        db: Database session
    
    Returns:
        Detection results with bounding boxes and labels
    """
    try:
        logger.info(f"Received base64 detection request: model={model_type}")
        
        # Load image from base64
        image_array = ImageStorageService.load_image_from_base64(image_base64)
        image_height, image_width = image_array.shape[:2]
        
        # Check if model is loaded
        if not model_loader.is_loaded(model_type):
            raise ModelNotLoadedError(model_type)
        
        # Run inference
        result = inference_engine.infer(image_array, model_type)
        
        if result.get('error'):
            logger.error(f"Inference error: {result['error']}")
            raise HTTPException(status_code=500, detail=result['error'])
        
        # Save to database
        detection_db = DetectionResult(
            model_type=model_type,
            image_path="",
            detections=[det for det in result['detections']],
            image_height=image_height,
            image_width=image_width,
            inference_time_ms=result['inference_time_ms']
        )
        db.add(detection_db)
        db.commit()
        db.refresh(detection_db)
        
        logger.info(f"Base64 detection completed: {len(result['detections'])} objects found")
        
        response = DetectionResultResponse(
            id=detection_db.id,
            model_type=model_type,
            detections=[
                {
                    'x_min': det['x_min'],
                    'y_min': det['y_min'],
                    'x_max': det['x_max'],
                    'y_max': det['y_max'],
                    'label': det['label'],
                    'confidence': det['confidence']
                }
                for det in result['detections']
            ],
            image_height=image_height,
            image_width=image_width,
            inference_time_ms=result['inference_time_ms'],
            created_at=detection_db.created_at
        )
        
        return response
    
    except (InvalidImageError, ModelNotLoadedError) as e:
        logger.error(f"Request validation error: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Base64 detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
