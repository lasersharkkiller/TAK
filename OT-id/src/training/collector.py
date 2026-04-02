"""
Training Data Collector

Every time the pipeline successfully classifies a device, this module
saves the full frame + YOLO-format annotation so the detections
automatically accumulate into a fine-tuning dataset.

Output layout:
    data/training/
    ├── images/          ← full frames as JPEG (what YOLO trains on)
    ├── labels/          ← YOLO annotation .txt (one per image)
    ├── crops/           ← device crop only (for human review / Roboflow)
    ├── dataset.yaml     ← YOLO dataset config, ready for ultralytics train
    └── manifest.csv     ← human-readable log of every saved example

YOLO annotation format (one line per detection per image):
    <class_id> <x_center> <y_center> <width> <height>   (all 0.0–1.0)

Optionally uploads to a Roboflow project if ROBOFLOW_API_KEY is set.
"""
from __future__ import annotations
import csv
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

import cv2


# ── Class map ─────────────────────────────────────────────────────────────────
# Maps Claude category strings → YOLO integer class IDs.
# Add / reorder carefully — changing IDs invalidates existing labels.
CLASS_MAP: dict[str, int] = {
    "IP Camera":          0,   # generic / unclassified IP camera
    "camera_dome":        0,
    "camera_bullet":      1,
    "camera_ptz":         2,
    "PTZ":                2,
    "camera_fisheye":     3,
    "Access Point":       4,
    "outdoor_ap":         4,
    "cellular_antenna":   5,
    "utility_enclosure":  6,
    "RTU":                6,
    "satellite_dish":     7,
    "HVAC Controller":    8,
    "hvac_unit":          8,
    "solar_array":        9,
    "server_rack_top":   10,
    "PLC":               11,
    "HMI":               12,
    "industrial_panel":  12,
    "Network Switch":    13,
    "Router":            14,
    "Smart Meter":       15,
    "Badge Reader":      16,
    "Fire Panel":        17,
    "UPS":               18,
    "Unknown":           19,
}

CLASS_NAMES = [
    "camera_ip",          # 0
    "camera_bullet",      # 1
    "camera_ptz",         # 2
    "camera_fisheye",     # 3
    "outdoor_ap",         # 4
    "cellular_antenna",   # 5
    "utility_enclosure",  # 6
    "satellite_dish",     # 7
    "hvac_unit",          # 8
    "solar_array",        # 9
    "server_rack_top",    # 10
    "plc",                # 11
    "industrial_panel",   # 12
    "network_switch",     # 13
    "router",             # 14
    "smart_meter",        # 15
    "badge_reader",       # 16
    "fire_panel",         # 17
    "ups",                # 18
    "unknown_device",     # 19
]


def _resolve_class(category: str) -> int:
    """Fuzzy map a Claude category string to a class ID."""
    # Exact match first
    if category in CLASS_MAP:
        return CLASS_MAP[category]
    # Substring match
    cat_l = category.lower()
    for key, cid in CLASS_MAP.items():
        if key.lower() in cat_l or cat_l in key.lower():
            return cid
    return CLASS_MAP["Unknown"]


def _yolo_bbox(bbox_xyxy: list[int], img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Convert [x1,y1,x2,y2] pixel coords to YOLO normalized (cx,cy,w,h)."""
    x1, y1, x2, y2 = bbox_xyxy
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return cx, cy, w, h


class TrainingCollector:
    """
    Saves detection+classification results as a YOLO training dataset.

    Args:
        base_dir:           Root of data/training/ directory
        min_confidence:     Only save examples where Claude confidence >= this
        save_crops:         Also save per-device crop images (for review)
        roboflow_api_key:   If set, upload to Roboflow after each session
        roboflow_project:   Roboflow project name (e.g. "ot-id-drone")
        roboflow_workspace: Roboflow workspace slug
    """

    def __init__(
        self,
        base_dir: str,
        min_confidence: float = 0.55,
        save_crops: bool = True,
        roboflow_api_key: Optional[str] = None,
        roboflow_project: Optional[str] = None,
        roboflow_workspace: Optional[str] = None,
    ):
        self.base_dir        = base_dir
        self.min_confidence  = min_confidence
        self.save_crops      = save_crops
        self.rf_api_key      = roboflow_api_key
        self.rf_project      = roboflow_project
        self.rf_workspace    = roboflow_workspace

        self._images_dir  = os.path.join(base_dir, "images")
        self._labels_dir  = os.path.join(base_dir, "labels")
        self._crops_dir   = os.path.join(base_dir, "crops")
        self._manifest    = os.path.join(base_dir, "manifest.csv")

        os.makedirs(self._images_dir, exist_ok=True)
        os.makedirs(self._labels_dir, exist_ok=True)
        if save_crops:
            os.makedirs(self._crops_dir, exist_ok=True)

        self._saved_count = 0
        self._skipped_count = 0

        # Pending annotations for the current frame — flushed when frame changes
        self._pending_frame_id: Optional[str] = None
        self._pending_frame_img = None
        self._pending_annotations: list[str] = []   # YOLO annotation lines

        self._ensure_manifest_header()
        self._write_dataset_yaml()

    # ── Public API ────────────────────────────────────────────────────────────

    def collect(self, frame, detection, device_id) -> bool:
        """
        Record one detection for training.

        Args:
            frame:      Frame object from ingestion layer
            detection:  Detection object from detector
            device_id:  DeviceID from classifier

        Returns:
            True if the example was saved, False if skipped (low confidence).
        """
        if device_id.confidence < self.min_confidence:
            self._skipped_count += 1
            return False

        img = frame.image
        img_h, img_w = img.shape[:2]
        class_id = _resolve_class(device_id.category)
        cx, cy, w, h = _yolo_bbox(detection.bbox_xyxy, img_w, img_h)
        annotation_line = f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

        # Group multiple detections on the same frame into one image+label file
        frame_id = f"frame_{frame.frame_index:08d}"
        if frame_id != self._pending_frame_id:
            self._flush_pending_frame()
            self._pending_frame_id  = frame_id
            self._pending_frame_img = img.copy()
            self._pending_annotations = []

        self._pending_annotations.append(annotation_line)

        # Save crop
        if self.save_crops and detection.crop is not None:
            crop_name = f"{frame_id}_{class_id}_{uuid.uuid4().hex[:6]}.jpg"
            cv2.imwrite(os.path.join(self._crops_dir, crop_name), detection.crop)

        # Append to manifest
        self._append_manifest(frame, detection, device_id, class_id, frame_id)

        self._saved_count += 1
        return True

    def flush(self) -> None:
        """Call at end of session to flush any pending frame."""
        self._flush_pending_frame()

    def upload_to_roboflow(self) -> bool:
        """
        Upload the images/ folder to a Roboflow project.
        Returns True on success.
        """
        if not self.rf_api_key or not self.rf_project:
            print("[Collector] Roboflow upload skipped — no API key or project set.")
            return False
        try:
            from roboflow import Roboflow
        except ImportError:
            raise ImportError("roboflow is required: pip install roboflow")

        rf = Roboflow(api_key=self.rf_api_key)
        workspace = rf.workspace(self.rf_workspace) if self.rf_workspace else rf.workspace()
        project = workspace.project(self.rf_project)

        uploaded = 0
        for fname in os.listdir(self._images_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(self._images_dir, fname)
            label_path = os.path.join(self._labels_dir, os.path.splitext(fname)[0] + ".txt")
            if not os.path.exists(label_path):
                continue
            project.upload(
                image_path=img_path,
                annotation_path=label_path,
                annotation_labelmap={str(i): name for i, name in enumerate(CLASS_NAMES)},
                split="train",
                num_retry_uploads=3,
            )
            uploaded += 1

        print(f"[Collector] Uploaded {uploaded} images to Roboflow project '{self.rf_project}'.")
        return True

    @property
    def stats(self) -> dict:
        return {
            "saved": self._saved_count,
            "skipped_low_confidence": self._skipped_count,
            "dataset_dir": self.base_dir,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _flush_pending_frame(self) -> None:
        if self._pending_frame_id is None or not self._pending_annotations:
            return

        img_path   = os.path.join(self._images_dir, f"{self._pending_frame_id}.jpg")
        label_path = os.path.join(self._labels_dir, f"{self._pending_frame_id}.txt")

        cv2.imwrite(img_path, self._pending_frame_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(label_path, "w") as f:
            f.write("\n".join(self._pending_annotations) + "\n")

        self._pending_frame_id  = None
        self._pending_frame_img = None
        self._pending_annotations = []

    def _ensure_manifest_header(self) -> None:
        if not os.path.exists(self._manifest):
            with open(self._manifest, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "frame_id", "frame_index", "source_uri",
                    "class_id", "class_name", "manufacturer", "model",
                    "category", "confidence", "id_source",
                    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                    "lat", "lon", "alt_m",
                ])

    def _append_manifest(self, frame, detection, device_id, class_id: int, frame_id: str) -> None:
        x1, y1, x2, y2 = detection.bbox_xyxy
        with open(self._manifest, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now(timezone.utc).isoformat(),
                frame_id,
                frame.frame_index,
                frame.source_uri,
                class_id,
                CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else "unknown",
                device_id.manufacturer,
                device_id.model,
                device_id.category,
                f"{device_id.confidence:.3f}",
                device_id.id_source,
                x1, y1, x2, y2,
                frame.lat, frame.lon, frame.alt_m,
            ])

    def _write_dataset_yaml(self) -> None:
        """Write/refresh the YOLO dataset.yaml every time we init."""
        yaml_path = os.path.join(self.base_dir, "dataset.yaml")
        content = f"""# OT-id YOLO training dataset
# Generated by OT-id training collector
# Usage: yolo train data=dataset.yaml model=yolov8x.pt epochs=100 imgsz=640

path: {os.path.abspath(self.base_dir)}
train: images
val:   images   # split manually or use Roboflow for train/val split

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
        with open(yaml_path, "w") as f:
            f.write(content)
