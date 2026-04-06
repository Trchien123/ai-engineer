"""
Explainable AI (XAI) service for the damaged sign classifier.

Provides three explanation methods:
  - Grad-CAM  : gradient-weighted class activation maps (visualised as heatmap overlays)
  - SHAP      : SHapley Additive exPlanations using blur-masked image perturbations
  - Zennit    : Layer-wise Relevance Propagation via EpsilonGammaBox composite

All methods return a list of dicts:
    [{"class_name": str, "probability": float, "image_base64": str}, ...]

Images are PNG-encoded and base64-encoded so they can be embedded directly in JSON
and rendered in a browser as  <img src="data:image/png;base64,...">.
"""
from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional

import cv2
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from app.services.damaged_sign_pipeline import (
    CLASS_NAMES,
    CLASSIFIER_THRESHOLDS,
    NORMAL_CLASS_IDX,
    predict_crop,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pil_to_tensor(pil_img: Image.Image, img_size: int = 224) -> torch.Tensor:
    """Return a (1, 3, H, W) normalised float tensor."""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(pil_img).unsqueeze(0)


def _pil_to_numpy_01(pil_img: Image.Image, img_size: int = 224) -> np.ndarray:
    """Return an (H, W, 3) float32 array in [0, 1] used by SHAP."""
    img = pil_img.convert("RGB").resize((img_size, img_size))
    return np.array(img, dtype=np.float32) / 255.0


def _fig_to_base64(fig: plt.Figure) -> str:
    """Serialise a matplotlib figure to a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _ndarray_bgr_to_base64(img_bgr: np.ndarray) -> str:
    """Encode a BGR uint8 ndarray as a base64 PNG string."""
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _get_active_classes(
    pil_img: Image.Image, model: Any, device: torch.device
) -> tuple[list[int], torch.Tensor]:
    """Run inference and return (active_class_indices, prob_tensor)."""
    damage_preds, probs = predict_crop(pil_img, model, device, thresholds=CLASSIFIER_THRESHOLDS)
    # Reconstruct the active indices from the returned label names
    label_set = {name for name, _ in damage_preds}
    active_indices = [i for i, name in enumerate(CLASS_NAMES) if name in label_set]
    return active_indices, probs


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

class _GradCAM:
    """Minimal Grad-CAM implementation using forward/backward hooks."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inp, output):
        self.activations = output

    def _save_gradient(self, _module, _grad_inp, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        outputs = self.model(input_tensor)
        outputs[0, class_idx].backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # type: ignore[union-attr]
        cam = torch.relu((weights * self.activations).sum(dim=1))  # type: ignore[union-attr]
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        return cam


def _get_last_conv_layer(model: torch.nn.Module) -> Optional[torch.nn.Module]:
    if hasattr(model, "features") and model.features is not None:
        for m in reversed(list(model.features.modules())):
            if isinstance(m, torch.nn.Conv2d):
                return m
    for m in reversed(list(model.modules())):
        if isinstance(m, torch.nn.Conv2d):
            return m
    return None


def explain_gradcam(
    pil_img: Image.Image,
    model: Any,
    device: torch.device,
) -> List[Dict]:
    """
    Generate Grad-CAM heatmap overlays for each predicted damage class.

    Returns one result dict per active class.
    """
    target_layer = _get_last_conv_layer(model)
    if target_layer is None:
        raise RuntimeError("No Conv2d layer found in model for Grad-CAM.")

    input_tensor = _pil_to_tensor(pil_img).to(device)
    img_np = np.array(pil_img.convert("RGB").resize((224, 224)), dtype=np.uint8)

    active_indices, probs = _get_active_classes(pil_img, model, device)
    if not active_indices:
        logger.info("Grad-CAM: no active classes, nothing to explain.")
        return []

    gradcam = _GradCAM(model, target_layer)
    results: List[Dict] = []

    for class_idx in active_indices:
        cam = gradcam.generate(input_tensor, class_idx)

        # Resize CAM to image size and overlay
        cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
        heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = (heatmap * 0.4 + img_np[:, :, ::-1] * 0.6).astype(np.uint8)

        results.append({
            "class_name": CLASS_NAMES[class_idx],
            "probability": float(probs[class_idx].item()),
            "image_base64": _ndarray_bgr_to_base64(overlay),
        })

    return results


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def explain_shap(
    pil_img: Image.Image,
    model: Any,
    device: torch.device,
) -> List[Dict]:
    """
    Generate SHAP image explanations using blur-masked perturbations.

    Returns a single result dict whose image shows all active classes.
    """
    try:
        import shap  # optional dependency
    except ImportError as exc:
        raise ImportError(
            "shap is required for SHAP explanations. Install it with: pip install shap"
        ) from exc

    img_np = _pil_to_numpy_01(pil_img)
    active_indices, probs = _get_active_classes(pil_img, model, device)

    if not active_indices:
        logger.info("SHAP: no active classes, nothing to explain.")
        return []

    def _predict_wrapper(images: np.ndarray) -> np.ndarray:
        """Convert SHAP's (N, H, W, C) numpy batch → model probabilities."""
        model.eval()
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        tensors = torch.stack(
            [norm(torch.tensor(img).permute(2, 0, 1).float()) for img in images]
        ).to(device)
        with torch.no_grad():
            return torch.sigmoid(model(tensors)).cpu().numpy()

    masker = shap.maskers.Image("blur(64,64)", img_np.shape)
    explainer = shap.Explainer(_predict_wrapper, masker, output_names=CLASS_NAMES)
    shap_values = explainer(
        np.expand_dims(img_np, 0),
        max_evals=1500,
        batch_size=50,
        outputs=active_indices,
    )

    # shap.image_plot writes to the current matplotlib figure
    shap.image_plot(shap_values, show=False)
    fig = plt.gcf()
    img_b64 = _fig_to_base64(fig)

    # Build one result per active class (all share the same composite figure)
    results: List[Dict] = []
    for i, class_idx in enumerate(active_indices):
        results.append({
            "class_name": CLASS_NAMES[class_idx],
            "probability": float(probs[class_idx].item()),
            # The composite SHAP figure is the same for all — attach only once (first class)
            "image_base64": img_b64 if i == 0 else "",
        })

    return results


# ---------------------------------------------------------------------------
# Zennit (LRP)
# ---------------------------------------------------------------------------

def explain_zennit(
    pil_img: Image.Image,
    model: Any,
    device: torch.device,
) -> List[Dict]:
    """
    Generate LRP (Layer-wise Relevance Propagation) heatmaps via Zennit.

    Returns one result dict per active class with a side-by-side matplotlib figure
    (original image + bwr relevance heatmap).
    """
    try:
        from zennit.composites import EpsilonGammaBox  # optional dependency
        from zennit.attribution import Gradient
    except ImportError as exc:
        raise ImportError(
            "zennit is required for LRP explanations. Install it with: pip install zennit"
        ) from exc

    input_tensor = _pil_to_tensor(pil_img).to(device)
    input_tensor.requires_grad_(True)

    active_indices, probs = _get_active_classes(pil_img, model, device)
    if not active_indices:
        logger.info("Zennit: no active classes, nothing to explain.")
        return []

    # Pre-compute logits for the output mask
    with torch.no_grad():
        logits = model(input_tensor)

    # ImageNet normalisation bounds required by EpsilonGammaBox
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    low_bound = (0.0 - mean) / std
    high_bound = (1.0 - mean) / std

    composite = EpsilonGammaBox(low=low_bound, high=high_bound)

    # Denormalised original image for display
    orig_np = input_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    orig_np = np.clip(
        (orig_np - orig_np.min()) / (orig_np.max() - orig_np.min() + 1e-8), 0, 1
    )

    results: List[Dict] = []

    with Gradient(model=model, composite=composite) as attributor:
        for class_idx in active_indices:
            output_mask = torch.zeros_like(logits)
            output_mask[0, class_idx] = 1.0
            _out, attribution = attributor(input_tensor, output_mask)
            relevance = attribution.sum(dim=1).squeeze(0).detach().cpu().numpy()

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            axes[0].imshow(orig_np)
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            vmax = np.max(np.abs(relevance)) + 1e-8
            im = axes[1].imshow(relevance, cmap="bwr", vmin=-vmax, vmax=vmax)
            axes[1].set_title(
                f"{CLASS_NAMES[class_idx]}  (prob: {probs[class_idx]:.3f})"
            )
            axes[1].axis("off")
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            plt.tight_layout()

            results.append({
                "class_name": CLASS_NAMES[class_idx],
                "probability": float(probs[class_idx].item()),
                "image_base64": _fig_to_base64(fig),
            })

    return results


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

XAI_METHODS = ("grad_cam", "shap", "zennit")


def run_xai(
    pil_img: Image.Image,
    model: Any,
    device: torch.device,
    method: str,
) -> List[Dict]:
    """
    Dispatch to the requested XAI method.

    Args:
        pil_img: PIL image to explain (should be a sign crop or a single sign photo).
        model: Loaded EfficientNetV2 classifier.
        device: Torch device.
        method: One of 'grad_cam', 'shap', or 'zennit'.

    Returns:
        List of result dicts with keys: class_name, probability, image_base64.
    """
    method = method.lower().strip()
    model.eval()

    if method == "grad_cam":
        return explain_gradcam(pil_img, model, device)
    elif method == "shap":
        return explain_shap(pil_img, model, device)
    elif method == "zennit":
        return explain_zennit(pil_img, model, device)
    else:
        raise ValueError(
            f"Unknown XAI method '{method}'. Choose from: {XAI_METHODS}"
        )
