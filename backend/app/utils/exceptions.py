"""
Custom exceptions for the FastAPI application.
"""
from fastapi import HTTPException, status


class ModelNotLoadedError(HTTPException):
    """Raised when a model fails to load or is not found."""
    def __init__(self, model_name: str, reason: str | None = None):
        detail = reason or f"Model '{model_name}' is not loaded or unavailable."
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail
        )


class InvalidImageError(HTTPException):
    """Raised when uploaded image is invalid."""
    def __init__(self, reason: str = "Invalid image format or file"):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image: {reason}"
        )


class FileSizeExceededError(HTTPException):
    """Raised when uploaded file exceeds size limit."""
    def __init__(self, max_size_mb: int):
        super().__init__(
            status_code=status.HTTP_413_PAYLOAD_TOO_LARGE,
            detail=f"File size exceeds maximum allowed size of {max_size_mb}MB."
        )


class ModelInferenceError(HTTPException):
    """Raised when inference execution fails."""
    def __init__(self, reason: str = "Model inference failed"):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {reason}"
        )
