"""
Main analysis pipeline.

Orchestrates: ingestion → detection → OCR → classification → recon → output.

De-duplicates devices across frames using a simple fingerprint cache
(manufacturer + model + approximate grid cell) to avoid re-running expensive
API calls for the same device seen in multiple frames.
"""
from __future__ import annotations
import os
import uuid
import time
from typing import List, Optional, Dict

from .ingestion.base import Frame
from .detection.detector import DeviceDetector, Detection
from .detection.ocr import OCRReader
from .classification.classifier import DeviceClassifier, DeviceID
from .recon.cve import search_cves
from .recon.shodan_lookup import search_by_product, lookup_ip
from .recon.creds import get_default_creds
from .recon.exploits import searchsploit_cve, get_ics_techniques
from .output.cot import build_cot_event, send_cot_udp
from .output.report import DeviceRecord, build_record, save_json_report, save_html_report
from .training.collector import TrainingCollector


class OTidPipeline:
    """
    Full OT-id analysis pipeline.

    Args:
        detector:       DeviceDetector instance
        ocr:            OCRReader instance
        classifier:     DeviceClassifier instance
        config:         dict from config.yaml (or any namespace)
        output_dir:     Where to write reports
        template_dir:   Where Jinja2 templates live
        collect_training: If True, save every classified detection as training data
    """

    def __init__(self, detector, ocr, classifier, config: dict, output_dir: str, template_dir: str,
                 collect_training: bool = False):
        self.detector = detector
        self.ocr = ocr
        self.classifier = classifier
        self.config = config
        self.output_dir = output_dir
        self.template_dir = template_dir

        self._seen: Dict[str, DeviceRecord] = {}   # fingerprint → record
        self.records: List[DeviceRecord] = []

        # API keys
        self._shodan_key = config.get("shodan_api_key") or os.environ.get("SHODAN_API_KEY")
        self._nvd_key = config.get("nvd_api_key") or os.environ.get("NVD_API_KEY")

        # CoT delivery config
        self._cot_enabled = config.get("cot_enabled", True)
        self._cot_host = config.get("cot_host", "239.2.3.1")
        self._cot_port = int(config.get("cot_port", 6969))

        # Training data collection
        self._collector: Optional[TrainingCollector] = None
        if collect_training or config.get("collect_training_data", False):
            training_dir = os.path.join(os.path.dirname(output_dir), "data", "training")
            self._collector = TrainingCollector(
                base_dir=training_dir,
                min_confidence=config.get("training_min_confidence", 0.55),
                save_crops=config.get("training_save_crops", True),
                roboflow_api_key=config.get("roboflow_api_key") or os.environ.get("ROBOFLOW_API_KEY"),
                roboflow_project=config.get("roboflow_project"),
                roboflow_workspace=config.get("roboflow_workspace"),
            )
            print(f"[Pipeline] Training collection ON → {training_dir}")

    def process_frame(self, frame: Frame) -> List[DeviceRecord]:
        """
        Run the full pipeline on one frame.
        Returns list of NEW device records produced (not duplicates).
        """
        detections = self.detector.detect(frame)
        new_records = []

        for det in detections:
            try:
                record = self._process_detection(frame, det)
                if record:
                    new_records.append(record)
            except Exception as e:
                print(f"[Pipeline] Error processing detection {det.label}: {e}")

        return new_records

    def _process_detection(self, frame: Frame, det: Detection) -> Optional[DeviceRecord]:
        # ── OCR ────────────────────────────────────────────────────────────────
        ocr_result = self.ocr.read_crop(det.crop) if det.crop is not None else None
        ocr_text = ocr_result.raw_text if ocr_result else ""

        # ── Classification ─────────────────────────────────────────────────────
        device_id: DeviceID = self.classifier.classify(
            crop=det.crop,
            ocr_text=ocr_text,
            yolo_label=det.label,
        )

        # ── Training data collection (before dedup — save every good detection) ─
        if self._collector is not None:
            saved = self._collector.collect(frame, det, device_id)
            if saved:
                print(f"[Pipeline]   → Training sample saved ({device_id.category}, conf={device_id.confidence:.2f})")

        # ── De-duplication ─────────────────────────────────────────────────────
        fingerprint = f"{device_id.manufacturer}::{device_id.model}"
        if fingerprint in self._seen:
            return None   # already processed this device

        self._seen[fingerprint] = True
        uid = uuid.uuid4().hex[:12]
        print(f"[Pipeline] New device: {device_id.manufacturer} {device_id.model} ({device_id.category})")

        # ── CVE lookup ─────────────────────────────────────────────────────────
        cves = search_cves(
            device_id.manufacturer,
            device_id.model,
            api_key=self._nvd_key,
        )
        print(f"[Pipeline]   → {len(cves)} CVEs found")

        # ── Shodan lookup ──────────────────────────────────────────────────────
        shodan_result = None
        if self._shodan_key:
            shodan_result = search_by_product(self._shodan_key, device_id.manufacturer, device_id.model)
            # If OCR found an IP, do direct lookup too
            if ocr_result and ocr_result.ip_candidates:
                ip = ocr_result.ip_candidates[0]
                ip_data = lookup_ip(self._shodan_key, ip)
                if ip_data:
                    print(f"[Pipeline]   → Shodan IP hit for {ip}")

        # ── Default credentials ────────────────────────────────────────────────
        creds = get_default_creds(device_id.manufacturer, device_id.model, device_id.category)
        print(f"[Pipeline]   → {len(creds)} default credential sets")

        # ── Exploit refs ───────────────────────────────────────────────────────
        exploit_refs = []
        for cve in cves[:5]:   # limit searchsploit calls
            exploit_refs.extend(searchsploit_cve(cve.cve_id))

        # ── MITRE ATT&CK ICS ──────────────────────────────────────────────────
        ics_techniques = get_ics_techniques(device_id.category)

        # ── Build record ───────────────────────────────────────────────────────
        record = build_record(
            uid=uid,
            frame=frame,
            device_id=device_id,
            cves=cves,
            creds=creds,
            shodan_result=shodan_result,
            exploit_refs=exploit_refs,
            ics_techniques=ics_techniques,
        )
        self.records.append(record)

        # ── CoT output ─────────────────────────────────────────────────────────
        if self._cot_enabled and frame.lat is not None:
            cot_xml = build_cot_event(
                device_id=device_id,
                cves=cves,
                creds=creds,
                lat=frame.lat,
                lon=frame.lon,
                alt_m=frame.alt_m or 0.0,
                frame_index=frame.frame_index,
                source_uri=frame.source_uri,
                ics_techniques=ics_techniques,
            )
            send_cot_udp(cot_xml, self._cot_host, self._cot_port)
            print(f"[Pipeline]   → CoT sent to {self._cot_host}:{self._cot_port}")

        return record

    def finalize(self, upload_to_roboflow: bool = False) -> dict:
        """Write reports, flush training data, return paths."""
        result: dict = {}

        if not self.records:
            print("[Pipeline] No devices identified — no report generated.")
        else:
            json_path = save_json_report(self.records, self.output_dir)
            html_path = save_html_report(self.records, self.output_dir, self.template_dir)
            result.update({"json": json_path, "html": html_path, "device_count": len(self.records)})

        if self._collector is not None:
            self._collector.flush()
            stats = self._collector.stats
            print(f"[Pipeline] Training samples: {stats['saved']} saved, "
                  f"{stats['skipped_low_confidence']} skipped (low confidence)")
            print(f"[Pipeline] Dataset dir: {stats['dataset_dir']}")
            result["training"] = stats
            if upload_to_roboflow:
                self._collector.upload_to_roboflow()

        return result
