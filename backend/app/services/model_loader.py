"""
Model loading service for YOLO detection models.
Handles dynamic loading and caching of .pt model files.
"""
from typing import Dict, Optional, Any
from pathlib import Path
import torch
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ModelLoader:
    """Manages loading and caching of detection models."""

    def __init__(self):
        """Initialize model cache."""
        self.models: Dict[str, Any] = {}
        self.model_configs = {
            "rubbish": {
                "file": settings.RUBBISH_STAGE1_MODEL,
                "description": "Detects and classifies rubbish objects (two-stage pipeline)",
                "dir": settings.RUBBISH_DETECTION_DIR,
            },
            "traffic_sign": {
                "file": settings.TRAFFIC_SIGN_MODEL,
                "description": "Detects damaged traffic signs using a two-stage pipeline",
                "dir": settings.DAMAGED_SIGN_DIR,
            },
        }

    def load_model(self, model_name: str) -> Optional[Any]:
        """Load a model from disk. Uses cache if already loaded."""
        if model_name in self.models:
            logger.info(f"Model '{model_name}' loaded from cache")
            return self.models[model_name]

        if model_name not in self.model_configs:
            logger.error(f"Unknown model: {model_name}")
            return None

        config = self.model_configs[model_name]
        model_dir = config.get("dir", settings.MODELS_DIR)
        model_path = Path(model_dir) / config["file"]

        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            logger.info(f"To use '{model_name}', place '{config['file']}' in: {model_dir}")
            return None

        try:
            logger.info(f"Loading model from: {model_path}")

            if model_name == "rubbish":
                from app.services.rubbish_detection_pipeline import RubbishDetectionPipeline

                stage1_path = Path(settings.RUBBISH_DETECTION_DIR) / settings.RUBBISH_STAGE1_MODEL
                stage2_path = Path(settings.RUBBISH_DETECTION_DIR) / settings.RUBBISH_STAGE2_MODEL

                if not stage2_path.exists():
                    logger.error(f"Rubbish stage2 model not found: {stage2_path}")
                    return None

                pipeline = RubbishDetectionPipeline(
                    stage1_model_path=stage1_path,
                    stage2_model_path=stage2_path,
                )
                self.models[model_name] = pipeline
                logger.info("Successfully loaded rubbish detection pipeline")
                return pipeline

            if model_name == "traffic_sign":
                from app.services.damaged_sign_pipeline import DamagedSignDetector

                classifier_path = Path(settings.DAMAGED_SIGN_DIR) / settings.TRAFFIC_SIGN_CLASSIFIER_MODEL
                retriever_index_path = Path(settings.DAMAGED_SIGN_DIR) / settings.TRAFFIC_SIGN_RETRIEVER_INDEX
                retriever_metadata_path = Path(settings.DAMAGED_SIGN_DIR) / settings.TRAFFIC_SIGN_RETRIEVER_METADATA

                detector = DamagedSignDetector(
                    yolo_path=model_path,
                    classifier_path=classifier_path,
                    retriever_index_path=retriever_index_path,
                    retriever_metadata_path=retriever_metadata_path,
                )
                self.models[model_name] = detector
                logger.info(f"Successfully loaded traffic sign detection bundle")
                return detector

            model = torch.load(model_path, map_location="cpu")
            if hasattr(model, "module"):
                model = model.module
            model.eval()
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
