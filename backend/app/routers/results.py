"""
API router for detection results history endpoints.
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import List

from app.models.database import get_db, DetectionResult
from app.models.schemas import DetectionResultResponse, DetectionHistoryResponse
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["results"])


@router.get("/results", response_model=DetectionHistoryResponse)
async def get_detection_history(
    model_type: str = Query(None, description="Filter by model type (optional)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results to return"),
    offset: int = Query(0, ge=0, description="Results offset for pagination"),
    db: Session = Depends(get_db)
):
    """
    Get detection history.
    
    Args:
        model_type: Optional filter by model type
        limit: Maximum number of results
        offset: Pagination offset
        db: Database session
    
    Returns:
        List of past detection results
    """
    try:
        # Build query
        query = db.query(DetectionResult).order_by(DetectionResult.created_at.desc())
        
        # Apply filter if specified
        if model_type:
            query = query.filter(DetectionResult.model_type == model_type)
        
        # Get total count
        total = query.count()
        
        # Apply pagination
        results_db = query.offset(offset).limit(limit).all()
        
        # Convert to response schema
        results = [
            DetectionResultResponse(
                id=r.id,
                model_type=r.model_type,
                detections=r.detections if r.detections else [],
                image_height=r.image_height,
                image_width=r.image_width,
                inference_time_ms=r.inference_time_ms,
                created_at=r.created_at
            )
            for r in results_db
        ]
        
        logger.info(f"Retrieved {len(results)} detection results (total: {total})")
        
        return DetectionHistoryResponse(
            results=results,
            total=total,
            limit=limit
        )
    
    except Exception as e:
        logger.error(f"Failed to retrieve detection history: {str(e)}")
        raise


@router.get("/results/{result_id}", response_model=DetectionResultResponse)
async def get_detection_result(
    result_id: int,
    db: Session = Depends(get_db)
):
    """
    Get a specific detection result by ID.
    
    Args:
        result_id: ID of detection result
        db: Database session
    
    Returns:
        Detection result details
    """
    try:
        result_db = db.query(DetectionResult).filter(DetectionResult.id == result_id).first()
        
        if not result_db:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail=f"Result {result_id} not found")
        
        result = DetectionResultResponse(
            id=result_db.id,
            model_type=result_db.model_type,
            detections=result_db.detections if result_db.detections else [],
            image_height=result_db.image_height,
            image_width=result_db.image_width,
            inference_time_ms=result_db.inference_time_ms,
            created_at=result_db.created_at
        )
        
        logger.info(f"Retrieved detection result {result_id}")
        
        return result
    
    except Exception as e:
        logger.error(f"Failed to retrieve result {result_id}: {str(e)}")
        raise
