"""
Model loading service for YOLO detection models.
Handles dynamic loading and caching of .pt model files.
"""
from typing import Dict, Optional
from pathlib import Path
import torch
from app.config import settings
from app.utils.logger import get_logger
    
logger = get_logger(__name__)


class ModelLoader:
    """Manages loading and caching of YOLO detection models."""
    
    def __init__(self):
        """Initialize model cache."""
        self.models: Dict[str, torch.nn.Module] = {}
        self.model_configs = {
            "rubbish_area": {
                "file": settings.RUBBISH_AREA_MODEL,
                "description": "Detects rubbish area regions in images"
            },
            "rubbish_classification": {
                "file": settings.RUBBISH_CLASSIFICATION_MODEL,
                "description": "Classifies rubbish into ~10 types"
            },
            "traffic_sign": {
                "file": settings.TRAFFIC_SIGN_MODEL,
                "description": "Detects damaged traffic signs"
            }
        }
    
    def load_model(self, model_name: str) -> Optional[torch.nn.Module]:
        """
        Load a model from disk. Uses cache if already loaded.
        
        Args:
            model_name: Name of model (rubbish_area, rubbish_classification, traffic_sign)
        
        Returns:
            Loaded model or None if loading failed
        """
        # Return from cache if already loaded
        if model_name in self.models:
            logger.info(f"Model '{model_name}' loaded from cache")
            return self.models[model_name]
        
        # Check if model config exists
        if model_name not in self.model_configs:
            logger.error(f"Unknown model: {model_name}")
            return None
        
        config = self.model_configs[model_name]
        model_path = settings.MODELS_DIR / config["file"]
        
        # Check if file exists
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            logger.info(f"To use '{model_name}', place '{config['file']}' in: {settings.MODELS_DIR}")
            return None
        
        try:
            logger.info(f"Loading model from: {model_path}")
            
            # Use torch.hub.load or direct torch.load based on your model format
            # For YOLO, typically: model = torch.load(model_path, weights_only=False)
            # But the standard YOLO format is loaded via ultralytics
            # For now, using a placeholder that works with standard PyTorch models
            
            model = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # If model is wrapped, extract it
            if hasattr(model, 'module'):
                model = model.module
            
            # Set to eval mode
            model.eval()
            
            # Cache the model
            self.models[model_name] = model
            logger.info(f"Successfully loaded model: {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {str(e)}")
            return None
    
    def load_all_models(self) -> Dict[str, bool]:
        """
        Attempt to load all configured models.
        
        Returns:
            Dictionary with model names as keys and load success as values
        """
        results = {}
        for model_name in self.model_configs.keys():
            model = self.load_model(model_name)
            results[model_name] = model is not None
        
        loaded_count = sum(1 for v in results.values() if v)
        total_count = len(results)
        logger.info(f"Model loading complete: {loaded_count}/{total_count} models loaded")
        
        return results
    
    def get_model(self, model_name: str) -> Optional[torch.nn.Module]:
        """Get a loaded model (returns None if not loaded)."""
        return self.models.get(model_name)
    
    def is_loaded(self, model_name: str) -> bool:
        """Check if a model is currently loaded."""
        return model_name in self.models
    
    def get_all_model_info(self) -> Dict[str, Dict]:
        """Get information about all configured models."""
        info = {}
        for name, config in self.model_configs.items():
            info[name] = {
                **config,
                "loaded": self.is_loaded(name)
            }
        return info


# Global model loader instance
model_loader = ModelLoader()
