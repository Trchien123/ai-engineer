"""
Image storage and handling service.
Manages image uploads, validation, and file cleanup.
"""
import os
import base64
from io import BytesIO
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from typing import Tuple, Optional

from app.config import settings
from app.utils.logger import get_logger
from app.utils.exceptions import InvalidImageError, FileSizeExceededError

logger = get_logger(__name__)


class ImageStorageService:
    """Service for handling image uploads, validation, and storage."""
    
    TEMP_IMAGE_DIR = Path("temp_images")
    
    @classmethod
    def __init__(cls):
        """Initialize storage service."""
        cls.TEMP_IMAGE_DIR.mkdir(exist_ok=True)
    
    @staticmethod
    def validate_file_size(file_size: int) -> bool:
        """
        Check if file size is within limits.
        
        Args:
            file_size: File size in bytes
        
        Returns:
            True if valid, raises exception otherwise
        """
        if file_size > settings.MAX_IMAGE_SIZE:
            max_mb = settings.MAX_IMAGE_SIZE / (1024 * 1024)
            raise FileSizeExceededError(int(max_mb))
        return True
    
    @staticmethod
    def validate_image_format(image_bytes: bytes) -> bool:
        """
        Validate image format by checking file signature.
        
        Args:
            image_bytes: Raw image bytes
        
        Returns:
            True if valid format
        """
        try:
            image = Image.open(BytesIO(image_bytes))
            fmt = image.format.lower() if image.format else None
            
            if fmt not in [f.lower() for f in settings.ALLOWED_IMAGE_FORMATS]:
                raise InvalidImageError(
                    f"Format '{fmt}' not allowed. Allowed: {', '.join(settings.ALLOWED_IMAGE_FORMATS)}"
                )
            
            return True
        except Exception as e:
            if isinstance(e, InvalidImageError):
                raise
            raise InvalidImageError(f"Cannot open image: {str(e)}")
    
    @staticmethod
    def load_image_from_base64(b64_string: str) -> np.ndarray:
        """
        Load image from base64 encoded string.
        
        Args:
            b64_string: Base64 encoded image string
        
        Returns:
            Image as numpy array (BGR format for OpenCV)
        """
        try:
            # Decode base64
            image_bytes = base64.b64decode(b64_string)
            
            # Validate file size
            ImageStorageService.validate_file_size(len(image_bytes))
            
            # Validate format
            ImageStorageService.validate_image_format(image_bytes)
            
            # Load with PIL
            image_pil = Image.open(BytesIO(image_bytes)).convert('RGB')
            
            # Convert to numpy array and BGR format (for OpenCV)
            image_rgb = np.array(image_pil)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            logger.info(f"Successfully loaded image from base64: {image_bgr.shape}")
            return image_bgr
        
        except Exception as e:
            logger.error(f"Failed to load image from base64: {str(e)}")
            if isinstance(e, (InvalidImageError, FileSizeExceededError)):
                raise
            raise InvalidImageError(f"Failed to decode image: {str(e)}")
    
    @staticmethod
    def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
        """
        Load image from raw bytes.
        
        Args:
            image_bytes: Raw image bytes
        
        Returns:
            Image as numpy array (BGR format for OpenCV)
        """
        try:
            # Validate file size
            ImageStorageService.validate_file_size(len(image_bytes))
            
            # Validate format
            ImageStorageService.validate_image_format(image_bytes)
            
            # Load with PIL
            image_pil = Image.open(BytesIO(image_bytes)).convert('RGB')
            
            # Convert to numpy array and BGR format
            image_rgb = np.array(image_pil)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            logger.info(f"Successfully loaded image from bytes: {image_bgr.shape}")
            return image_bgr
        
        except Exception as e:
            logger.error(f"Failed to load image from bytes: {str(e)}")
            if isinstance(e, (InvalidImageError, FileSizeExceededError)):
                raise
            raise InvalidImageError(f"Failed to load image: {str(e)}")
    
    @staticmethod
    def save_temp_image(image_array: np.ndarray, filename: str = None) -> str:
        """
        Save image to temporary storage.
        
        Args:
            image_array: Image as numpy array (BGR format)
            filename: Optional filename (auto-generated if not provided)
        
        Returns:
            Path to saved image
        """
        try:
            if filename is None:
                import uuid
                filename = f"temp_{uuid.uuid4().hex}.png"
            
            filepath = ImageStorageService.TEMP_IMAGE_DIR / filename
            cv2.imwrite(str(filepath), image_array)
            
            logger.info(f"Saved temp image to: {filepath}")
            return str(filepath)
        
        except Exception as e:
            logger.error(f"Failed to save temp image: {str(e)}")
            raise InvalidImageError(f"Failed to save image: {str(e)}")
    
    @staticmethod
    def cleanup_temp_images():
        """Remove all temporary images."""
        try:
            import shutil
            if ImageStorageService.TEMP_IMAGE_DIR.exists():
                shutil.rmtree(ImageStorageService.TEMP_IMAGE_DIR)
                ImageStorageService.TEMP_IMAGE_DIR.mkdir(exist_ok=True)
                logger.info("Cleaned up temporary images")
        except Exception as e:
            logger.error(f"Failed to cleanup temp images: {str(e)}")
