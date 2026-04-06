"""
API router for Explainable AI (XAI) endpoints.

Supported methods: grad_cam, shap, zennit
"""
from __future__ import annotations

import time
from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from app.models.schemas import ExplainResponse, ExplainResult
from app.services.model_loader import model_loader
from app.services.xai_service import XAI_METHODS, run_xai
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["explainability"])


@router.post("/explain", response_model=ExplainResponse)
async def explain(
    method: str = Form(
        ...,
        description=f"XAI method to use. One of: {', '.join(XAI_METHODS)}",
    ),
    file: UploadFile = File(..., description="Sign image (jpg, png, bmp)"),
):
    """
    Generate an explainability visualisation for the damaged sign classifier.

    Upload a single sign image and choose an XAI method.  The endpoint returns
    one visualisation image per predicted damage class, encoded as base64 PNG.

    - **grad_cam** — Gradient-weighted Class Activation Maps overlaid on the image.
    - **shap**     — SHAP pixel-importance maps (blur masker, requires `shap` package).
    - **zennit**   — Layer-wise Relevance Propagation heatmaps (requires `zennit` package).
    """
    method = method.lower().strip()
    if method not in XAI_METHODS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown XAI method '{method}'. Choose from: {list(XAI_METHODS)}",
        )

    # Load the traffic_sign model (classifier lives inside DamagedSignDetector)
    if not model_loader.is_loaded("traffic_sign"):
        loaded = model_loader.load_model("traffic_sign")
        if not loaded:
            config = model_loader.model_configs.get("traffic_sign", {})
            model_path = (
                Path(config.get("dir", "")) / config.get("file", "")
                if config
                else "unknown"
            )
            raise HTTPException(
                status_code=503,
                detail=f"traffic_sign model not loaded. Place model file at: {model_path}",
            )

    detector = model_loader.get_model("traffic_sign")
    if detector is None or not hasattr(detector, "classifier"):
        raise HTTPException(
            status_code=503,
            detail="Damaged sign classifier is not available.",
        )

    classifier = detector.classifier
    import torch
    device = torch.device(detector.device)

    # Decode uploaded image
    try:
        contents = await file.read()
        pil_img = Image.open(BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}")

    # Run XAI
    start = time.time()
    try:
        raw_results = run_xai(pil_img, classifier, device, method)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail=f"Missing dependency for method '{method}': {exc}",
        )
    except Exception as exc:
        logger.error(f"XAI '{method}' failed: {exc}")
        raise HTTPException(status_code=500, detail=f"XAI explanation failed: {exc}")

    elapsed_ms = (time.time() - start) * 1000
    logger.info(
        f"XAI/{method}: {len(raw_results)} class(es) explained in {elapsed_ms:.1f} ms"
    )

    return ExplainResponse(
        method=method,
        results=[ExplainResult(**r) for r in raw_results],
        inference_time_ms=elapsed_ms,
    )
