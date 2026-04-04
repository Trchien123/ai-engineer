"""
API router for model information endpoints.
"""
from fastapi import APIRouter
from app.models.schemas import ModelsListResponse, ModelInfo
from app.services.model_loader import model_loader
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["models"])


@router.get("/models", response_model=ModelsListResponse)
async def list_models():
    """
    Get list of available detection models and their status.
    
    Returns:
        List of available models with load status
    """
    model_info = model_loader.get_all_model_info()
    
    models = [
        ModelInfo(
            name=name,
            type=name,
            description=info['description'],
            loaded=info['loaded']
        )
        for name, info in model_info.items()
    ]
    
    logger.info(f"Listed {len(models)} models")
    
    return ModelsListResponse(
        models=models,
        count=len(models)
    )
