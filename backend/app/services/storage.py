"""
Image storage and handling service.
Manages image uploads, validation, and file cleanup.
"""
import os
import base64
import tempfile
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
    def validate_file_size(file_size: int, max_size: int = None) -> bool:
        """
        Check if file size is within limits.
        
        Args:
            file_size: File size in bytes
            max_size: Optional maximum size in bytes
        
        Returns:
            True if valid, raises exception otherwise
        """
        if max_size is None:
            max_size = settings.MAX_IMAGE_SIZE

        if file_size > max_size:
            max_mb = max_size / (1024 * 1024)
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
        except Exception as e:
            raise InvalidImageError(f"Cannot open image: {str(e)}")

        if fmt not in [f.lower() for f in settings.ALLOWED_IMAGE_FORMATS]:
            raise InvalidImageError(
                f"Format '{fmt}' not allowed. Allowed: {', '.join(settings.ALLOWED_IMAGE_FORMATS)}"
            )

        return True

    @staticmethod
    def _decode_image_bytes(image_bytes: bytes) -> np.ndarray:
        """
        Decode raw image bytes into an OpenCV BGR image.
        """
        image_arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image_bgr = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)
        if image_bgr is not None:
            return image_bgr

        try:
            image_pil = Image.open(BytesIO(image_bytes)).convert('RGB')
            image_rgb = np.array(image_pil)
            return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
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
            if b64_string.startswith("data:"):
                b64_string = b64_string.split(",", 1)[1]

            # Decode base64
            image_bytes = base64.b64decode(b64_string)
            
            # Validate file size
            ImageStorageService.validate_file_size(len(image_bytes))
            
            # Validate format
            ImageStorageService.validate_image_format(image_bytes)
            
            # Decode with OpenCV or fallback to PIL
            image_bgr = ImageStorageService._decode_image_bytes(image_bytes)
            
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
            ImageStorageService.validate_file_size(len(image_bytes), settings.MAX_IMAGE_SIZE)
            
            # Validate format
            ImageStorageService.validate_image_format(image_bytes)
            
            # Decode with OpenCV or fallback to PIL
            image_bgr = ImageStorageService._decode_image_bytes(image_bytes)
            
            logger.info(f"Successfully loaded image from bytes: {image_bgr.shape}")
            return image_bgr
        
        except Exception as e:
            logger.error(f"Failed to load image from bytes: {str(e)}")
            if isinstance(e, (InvalidImageError, FileSizeExceededError)):
                raise
            raise InvalidImageError(f"Failed to load image: {str(e)}")

    @staticmethod
    def load_frame_from_video_bytes(video_bytes: bytes, file_suffix: str = ".mp4") -> np.ndarray:
        """
        Extract the first frame from a video file uploaded as raw bytes.
        """
        try:
            ImageStorageService.validate_file_size(len(video_bytes), settings.MAX_VIDEO_SIZE)

            with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
                tmp_file.write(video_bytes)
                temp_path = tmp_file.name

            capture = cv2.VideoCapture(temp_path)
            if not capture.isOpened():
                raise InvalidImageError("Cannot open video file for frame extraction.")

            success, frame = capture.read()
            capture.release()
            if not success or frame is None:
                raise InvalidImageError("Failed to extract a frame from the video file.")

            logger.info(f"Extracted frame from video: {frame.shape}")
            return frame
        except Exception as e:
            logger.error(f"Failed to load video frame: {str(e)}")
            if isinstance(e, (InvalidImageError, FileSizeExceededError)):
                raise
            raise InvalidImageError(f"Failed to load video frame: {str(e)}")
        finally:
            try:
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass
    
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
