"""
Two-stage rubbish detection pipeline.

Stage 1 (YOLO): detects rubbish area regions in an image.
Stage 2 (YOLO): classifies objects within each cropped rubbish area.
Coordinate offsets are applied so all returned boxes are in original image space.
"""
import numpy as np
from pathlib import Path
from typing import List, Dict

from app.utils.logger import get_logger

logger = get_logger(__name__)

_CONF_STAGE1 = 0.25
_CONF_STAGE2 = 0.20
_PAD = 10
_IMGSZ = 640
_RUBBISH_AREA_CLASS_ID = 0


def _clamp(value: float, lo: int, hi: int) -> int:
    return max(lo, min(int(value), hi))


class RubbishDetectionPipeline:
    """Two-stage YOLO pipeline for rubbish detection."""

    def __init__(self, stage1_model_path: Path, stage2_model_path: Path):
        from ultralytics import YOLO
        logger.info(f"Loading rubbish stage1 model: {stage1_model_path}")
        self.stage1 = YOLO(str(stage1_model_path))
        logger.info(f"Loading rubbish stage2 model: {stage2_model_path}")
        self.stage2 = YOLO(str(stage2_model_path))
        logger.info("RubbishDetectionPipeline ready")

    def infer(self, image_array: np.ndarray) -> List[Dict]:
        """
        Run two-stage detection on a BGR image array.

        Args:
            image_array: BGR numpy array from OpenCV.

        Returns:
            List of detections, each a dict with keys:
            x_min, y_min, x_max, y_max  – coordinates in original image space
            label                        – class name from stage2 model
            confidence                   – stage2 confidence score
        """
        h, w = image_array.shape[:2]
        detections: List[Dict] = []

        # ------------------------------------------------------------------
        # Stage 1: locate rubbish areas
        # ------------------------------------------------------------------
        s1_results = self.stage1.predict(
            source=image_array,
            conf=_CONF_STAGE1,
            imgsz=_IMGSZ,
            verbose=False,
        )
        r1 = s1_results[0]

        if r1.boxes is None or len(r1.boxes) == 0:
            logger.info("Stage 1: no rubbish areas detected")
            return detections

        boxes_xyxy = r1.boxes.xyxy.cpu().numpy()
        cls_ids = r1.boxes.cls.cpu().numpy().astype(int)
        confs1 = r1.boxes.conf.cpu().numpy()

        crop_images: List[np.ndarray] = []
        crop_offsets: List[tuple] = []  # (x1, y1) for each crop
        area_boxes: List[tuple] = []    # (x1_orig, y1_orig, x2_orig, y2_orig) in original space

        for box, cls_id, score in zip(boxes_xyxy, cls_ids, confs1):
            if cls_id != _RUBBISH_AREA_CLASS_ID:
                continue

            # Record original-space area box for output
            area_boxes.append((float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(score)))

            x1 = _clamp(box[0] - _PAD, 0, w - 1)
            y1 = _clamp(box[1] - _PAD, 0, h - 1)
            x2 = _clamp(box[2] + _PAD, 0, w)
            y2 = _clamp(box[3] + _PAD, 0, h)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image_array[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_images.append(crop)
            crop_offsets.append((x1, y1))

        # Emit rubbish_area detections so the UI can display the stage-1 boxes
        for ax1, ay1, ax2, ay2, ascore in area_boxes:
            detections.append({
                "x_min": ax1,
                "y_min": ay1,
                "x_max": ax2,
                "y_max": ay2,
                "label": "rubbish_area",
                "confidence": ascore,
            })

        if not crop_images:
            return detections

        # ------------------------------------------------------------------
        # Stage 2: classify objects inside each crop
        # ------------------------------------------------------------------
        s2_results = self.stage2.predict(
            source=crop_images,
            conf=_CONF_STAGE2,
            imgsz=_IMGSZ,
            verbose=False,
        )

        for (cx1, cy1), r2 in zip(crop_offsets, s2_results):
            if r2.boxes is None or len(r2.boxes) == 0:
                continue

            boxes2 = r2.boxes.xyxy.cpu().numpy()
            cls_ids2 = r2.boxes.cls.cpu().numpy().astype(int)
            confs2 = r2.boxes.conf.cpu().numpy()

            for box2, cls_id2, score2 in zip(boxes2, cls_ids2, confs2):
                detections.append({
                    "x_min": float(box2[0]) + cx1,
                    "y_min": float(box2[1]) + cy1,
                    "x_max": float(box2[2]) + cx1,
                    "y_max": float(box2[3]) + cy1,
                    "label": self.stage2.names[int(cls_id2)],
                    "confidence": float(score2),
                })

        logger.info(f"Rubbish detection complete: {len(detections)} object(s) found")
        return detections
