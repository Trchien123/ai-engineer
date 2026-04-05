"""
Two-stage damaged traffic sign detection pipeline.

This module loads all assets needed for the damaged sign model and provides
an inference method that returns detections suitable for the API.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

CLASS_NAMES = [
    "bent",
    "broken_sheet",
    "crack",
    "graffiti",
    "normal",
    "paint_loss",
    "rust",
    "scratch",
]

RETRIEVER_THRESHOLD = 0.82
YOLO_CONFIDENCE_THRESHOLD = 0.5
YOLO_PREDICT_CONF = 0.2
YOLO_IMG_SIZE = 640


def preprocess_for_live_feed(img: Image.Image) -> Image.Image:
    """Enhanced preprocessing specifically for tiny crops."""
    img = img.convert("RGB")

    w, h = img.size
    if w < 100 or h < 100:
        img = img.resize((128, 128), Image.Resampling.BICUBIC)

    img = img.filter(ImageFilter.SHARPEN)
    img = img.filter(ImageFilter.DETAIL)

    desired_size = 224
    img.thumbnail((desired_size, desired_size), Image.Resampling.LANCZOS)

    new_img = Image.new("RGB", (desired_size, desired_size), (128, 128, 128))
    upper_left = ((desired_size - img.size[0]) // 2, (desired_size - img.size[1]) // 2)
    new_img.paste(img, upper_left)

    return new_img


def load_retriever(index_path: Path, metadata_path: Path):
    """Load FAISS index and clip retriever metadata."""
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "Missing dependencies for damaged sign retriever. "
            "Install faiss-cpu and sentence-transformers."
        ) from e

    index = faiss.read_index(str(index_path))
    clip_model = SentenceTransformer("clip-ViT-B-32")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, clip_model, metadata


def get_sign_metadata(crop_pil: Image.Image, retriever_assets: Tuple[Any, Any, List[Dict[str, Any]]]):
    """Retrieve sign metadata using CLIP embeddings and FAISS search."""
    index, clip_model, metadata = retriever_assets
    processed_img = preprocess_for_live_feed(crop_pil)
    query_vector = clip_model.encode(processed_img).astype("float32").reshape(1, -1)

    import faiss

    faiss.normalize_L2(query_vector)
    distances, indices = index.search(query_vector, 1)

    dist = float(distances[0][0])
    idx = int(indices[0][0])

    if dist < RETRIEVER_THRESHOLD or idx < 0 or idx >= len(metadata):
        return {"Sign No": "Unknown", "Descriptions": "Unknown Sign Type"}, dist

    return metadata[idx], dist


def load_effnetv2_multilabel(model_path: Path, device=None):
    """Create and load the EfficientNet v2 classifier."""
    try:
        import torch
        import torch.nn as nn
        from torchvision import transforms
        from torchvision.models import efficientnet_v2_s
    except ImportError as e:
        raise ImportError(
            "Missing Torch/torchvision dependencies for damaged sign classifier."
        ) from e

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.Linear(128, len(CLASS_NAMES)),
    )

    state_dict = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model


def predict_crop(crop_pil: Image.Image, model, device, threshold=0.5):
    """Predict damage labels for a crop using the classifier."""
    from torchvision import transforms
    import torch

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    tensor = transform(crop_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.sigmoid(outputs).squeeze(0)

    preds = (probs > threshold).int()
    predicted_labels = [
        (CLASS_NAMES[i], float(probs[i].item()))
        for i in range(len(CLASS_NAMES))
        if preds[i] == 1
    ]
    return predicted_labels, probs.cpu()


class DamagedSignDetector:
    """Wraps the two-stage damaged traffic sign model and retrieval assets."""

    def __init__(
        self,
        yolo_path: Path,
        classifier_path: Path,
        retriever_index_path: Path,
        retriever_metadata_path: Path,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "Missing ultralytics dependency for damaged sign detection."
            ) from e

        self.device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        self.yolo = YOLO(str(yolo_path))
        self.classifier = load_effnetv2_multilabel(classifier_path)
        self.retriever_assets = load_retriever(retriever_index_path, retriever_metadata_path)

    def infer(self, image_array: np.ndarray) -> List[Dict[str, Any]]:
        """Run inference on a BGR image and return detection boxes.

        Args:
            image_array: numpy array in BGR format

        Returns:
            List of objects with x_min, y_min, x_max, y_max, label and confidence
        """
        rgb_frame = image_array[:, :, ::-1]
        results = self.yolo.predict(
            rgb_frame,
            conf=YOLO_PREDICT_CONF,
            imgsz=YOLO_IMG_SIZE,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []
        if not results or len(results) == 0:
            return detections

        boxes = results[0].boxes
        if boxes is None:
            return detections

        for box in boxes:
            coords = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords.tolist()
            yolo_conf = float(box.conf[0].cpu().item())
            if yolo_conf < YOLO_CONFIDENCE_THRESHOLD:
                continue

            crop_bgr = image_array[y1:y2, x1:x2]
            if crop_bgr.size == 0:
                continue

            crop_pil = Image.fromarray(crop_bgr[:, :, ::-1])
            damage_preds, _ = predict_crop(crop_pil, self.classifier, self.device)

            if len(damage_preds) == 0:
                damage_label = "normal"
            else:
                damage_label = ", ".join([name for name, _ in damage_preds])

            sign_info, _ = get_sign_metadata(crop_pil, self.retriever_assets)
            sign_label = sign_info.get("Descriptions", "Unknown Sign Type")

            label = f"{sign_label} | {damage_label}"
            detections.append(
                {
                    "x_min": float(x1),
                    "y_min": float(y1),
                    "x_max": float(x2),
                    "y_max": float(y2),
                    "label": label,
                    "confidence": float(yolo_conf),
                }
            )

        return detections
