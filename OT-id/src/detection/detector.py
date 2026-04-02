"""
YOLOv8-based IoT/OT device detector.

Strategy (zero-shot first, fine-tune later):
  - Use YOLOv8x (largest general model) which already knows: laptop, tv,
    cell phone, keyboard, mouse, remote, microwave, oven, toaster, refrigerator.
  - Augment with a custom class list for OT/ICS devices.
  - When a custom fine-tuned model is available, swap via config (model_path).

Detected region crops are passed downstream to OCR + classification.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


# COCO classes that are plausibly IoT/tech devices
COCO_DEVICE_CLASSES = {
    "laptop", "tv", "cell phone", "keyboard", "mouse", "remote",
    "microwave", "oven", "toaster", "refrigerator", "clock", "monitor",
}

# OT/ICS specific labels expected from a fine-tuned model
OT_DEVICE_CLASSES = {
    "plc", "hmi", "rtu", "scada_terminal", "ip_camera", "network_switch",
    "router", "access_point", "industrial_panel", "sensor_node",
    "smart_meter", "ups", "serial_converter", "barcode_scanner",
    "badge_reader", "intercom", "fire_panel", "hvac_controller",
}

ALL_DEVICE_CLASSES = COCO_DEVICE_CLASSES | OT_DEVICE_CLASSES


@dataclass
class Detection:
    """A single device bounding box detection."""
    label: str
    confidence: float
    bbox_xyxy: List[int]          # [x1, y1, x2, y2] pixel coords
    crop: Optional[object] = None  # numpy ndarray of the cropped region
    frame_index: int = 0
    timestamp: float = 0.0


class DeviceDetector:
    """
    Runs YOLOv8 inference on a frame and returns detected device regions.

    Args:
        model_path: Path to .pt weights file, or a YOLOv8 model name
                    e.g. 'yolov8x.pt' (auto-downloaded on first run).
                    Point to a custom fine-tuned model once available.
        conf_threshold: Minimum confidence to report (default 0.35)
        device:     'cpu', 'cuda', or 'mps'
    """

    def __init__(
        self,
        model_path: str = "yolov8x.pt",
        conf_threshold: float = 0.35,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.device = device
        self._model = None

    def load(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics is required: pip install ultralytics")
        print(f"[Detector] Loading model: {self.model_path}")
        self._model = YOLO(self.model_path)
        print("[Detector] Model loaded.")

    def detect(self, frame) -> List[Detection]:
        """
        Args:
            frame: Frame object from ingestion layer
        Returns:
            List of Detection objects for device-class hits only
        """
        if self._model is None:
            self.load()

        img = frame.image
        results = self._model(img, conf=self.conf_threshold, device=self.device, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            names = result.names
            for i, box in enumerate(boxes):
                label = names[int(box.cls[0])].lower()

                # Filter to device-relevant classes only
                # (for general COCO model, skip people, cars, etc.)
                if not self._is_device(label):
                    continue

                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                crop = img[y1:y2, x1:x2].copy()

                detections.append(Detection(
                    label=label,
                    confidence=float(box.conf[0]),
                    bbox_xyxy=[x1, y1, x2, y2],
                    crop=crop,
                    frame_index=frame.frame_index,
                    timestamp=frame.timestamp,
                ))

        return detections

    def _is_device(self, label: str) -> bool:
        # Exact match or substring match for OT labels
        if label in ALL_DEVICE_CLASSES:
            return True
        for cls in ALL_DEVICE_CLASSES:
            if cls in label or label in cls:
                return True
        return False
