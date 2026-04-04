"""
Image inference pipeline service.
Handles preprocessing, model inference, and post-processing.
"""
import cv2
import numpy as np
import time
from typing import List, Dict, Tuple
from PIL import Image
from io import BytesIO
import base64

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DetectionPipeline:
    """
    Pipeline for image inference:
    1. Preprocess (resize, normalize)
    2. Model inference
    3. Post-process (NMS, filtering, bbox extraction)
    """
    
    @staticmethod
    def preprocess_image(image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Preprocess image for model inference.
        
        Args:
            image: Input image as numpy array (BGR format from cv2)
        
        Returns:
            Tuple of (preprocessed_image, original_size)
        """
        original_height, original_width = image.shape[:2]
        original_size = (original_width, original_height)
        
        # Resize to model input size
        target_size = settings.IMAGE_RESIZE_SIZE
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert BGR to RGB
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = resized_rgb.astype(np.float32) / 255.0
        
        # Add batch dimension: (H, W, C) -> (1, C, H, W) for PyTorch
        tensor = np.transpose(normalized, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        
        return tensor, original_size
    
    @staticmethod
    def scale_bbox(bbox: Tuple[float, float, float, float], 
                   from_size: Tuple[int, int], 
                   to_size: Tuple[int, int]) -> Tuple[float, float, float, float]:
        """
        Scale bounding box from one image size to another.
        
        Args:
            bbox: (x_min, y_min, x_max, y_max) in from_size coordinates
            from_size: (width, height) of bbox coordinates
            to_size: (width, height) of target coordinates
        
        Returns:
            Scaled bbox in to_size coordinates
        """
        scale_x = to_size[0] / from_size[0]
        scale_y = to_size[1] / from_size[1]
        
        x_min, y_min, x_max, y_max = bbox
        return (
            x_min * scale_x,
            y_min * scale_y,
            x_max * scale_x,
            y_max * scale_y
        )
    
    @staticmethod
    def extract_detections(model_output: Dict, 
                          image_size: Tuple[int, int],
                          model_type: str) -> List[Dict]:
        """
        Extract detection boxes from model output.
        
        This is a flexible template that adapts to different model outputs.
        Override this method based on your actual model's output format.
        
        Args:
            model_output: Raw output from model (varies by architecture)
            image_size: Original image size (width, height)
            model_type: Type of model (for context-specific processing)
        
        Returns:
            List of detections with format:
            {
                'x_min', 'y_min', 'x_max', 'y_max': bbox coordinates,
                'label': class name,
                'confidence': confidence score
            }
        """
        detections = []
        
        # This is a template structure. Your actual model output parsing
        # depends on your model architecture (YOLO, Faster R-CNN, etc.)
        # 
        # Example for YOLO-like output (assuming model_output has detections):
        if isinstance(model_output, dict) and 'detections' in model_output:
            raw_detections = model_output['detections']
            
            for det in raw_detections:
                # Assume det format: [x_min, y_min, x_max, y_max, conf, class_id]
                if len(det) >= 6:
                    x_min, y_min, x_max, y_max, conf, class_id = det[:6]
                    
                    # Filter by confidence threshold
                    if conf >= settings.CONFIDENCE_THRESHOLD:
                        detections.append({
                            'x_min': float(x_min),
                            'y_min': float(y_min),
                            'x_max': float(x_max),
                            'y_max': float(y_max),
                            'label': f"class_{int(class_id)}",
                            'confidence': float(conf)
                        })
        
        logger.info(f"Extracted {len(detections)} detections from model output")
        return detections
    
    @staticmethod
    def apply_nms(detections: List[Dict], iou_threshold: float = None) -> List[Dict]:
        """
        Apply Non-Maximum Suppression to remove duplicate detections.
        
        Args:
            detections: List of detection dicts
            iou_threshold: IOU threshold for suppression
        
        Returns:
            Filtered detections after NMS
        """
        if iou_threshold is None:
            iou_threshold = settings.IOU_THRESHOLD
        
        if not detections:
            return []
        
        # Sort by confidence (descending)
        sorted_dets = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        # Simple NMS implementation
        keep = []
        while sorted_dets:
            # Take highest confidence detection
            current = sorted_dets.pop(0)
            keep.append(current)
            
            # Remove detections with high IOU to current
            remaining = []
            for det in sorted_dets:
                iou = DetectionPipeline.calculate_iou(current, det)
                if iou < iou_threshold:
                    remaining.append(det)
            sorted_dets = remaining
        
        logger.info(f"NMS: {len(detections)} detections -> {len(keep)} detections")
        return keep
    
    @staticmethod
    def calculate_iou(box1: Dict, box2: Dict) -> float:
        """Calculate Intersection over Union between two boxes."""
        # Extract coordinates
        x1_min, y1_min = box1['x_min'], box1['y_min']
        x1_max, y1_max = box1['x_max'], box1['y_max']
        x2_min, y2_min = box2['x_min'], box2['y_min']
        x2_max, y2_max = box2['x_max'], box2['y_max']
        
        # Calculate intersection area
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union area
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area


class InferenceEngine:
    """Main inference engine orchestrating preprocessing -> inference -> postprocessing."""
    
    def __init__(self, model_loader):
        """
        Initialize inference engine.
        
        Args:
            model_loader: ModelLoader instance for accessing models
        """
        self.model_loader = model_loader
        self.pipeline = DetectionPipeline()
    
    def infer(self, image_array: np.ndarray, model_name: str) -> Dict:
        """
        Run inference on an image.
        
        Args:
            image_array: Image as numpy array (BGR format)
            model_name: Name of model to use
        
        Returns:
            Dictionary containing:
            {
                'detections': List of detection dicts,
                'inference_time_ms': float,
                'original_size': (width, height)
            }
        """
        start_time = time.time()
        
        # Get model
        model = self.model_loader.get_model(model_name)
        if model is None:
            logger.error(f"Model not loaded: {model_name}")
            return {
                'detections': [],
                'inference_time_ms': 0,
                'original_size': (0, 0),
                'error': f"Model '{model_name}' not loaded"
            }
        
        try:
            # Preprocess
            processed_image, original_size = self.pipeline.preprocess_image(image_array)
            logger.info(f"Image preprocessed: {original_size} -> {settings.IMAGE_RESIZE_SIZE}")
            
            # Convert to tensor format expected by model
            import torch
            image_tensor = torch.from_numpy(processed_image).float()
            
            # Inference
            with torch.no_grad():
                model_output = model(image_tensor)
            
            logger.info(f"Model inference complete")
            
            # Post-process
            detections = self.pipeline.extract_detections(
                model_output if isinstance(model_output, dict) else {'detections': []},
                original_size,
                model_name
            )
            
            # Apply NMS
            detections = self.pipeline.apply_nms(detections)
            
            inference_time_ms = (time.time() - start_time) * 1000
            
            return {
                'detections': detections,
                'inference_time_ms': inference_time_ms,
                'original_size': original_size,
                'error': None
            }
        
        except Exception as e:
            inference_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Inference failed: {str(e)}")
            return {
                'detections': [],
                'inference_time_ms': inference_time_ms,
                'original_size': (0, 0),
                'error': str(e)
            }
