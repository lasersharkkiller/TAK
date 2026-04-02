"""
Device make/model classifier using Claude's vision API.

Given a cropped image of a detected device + any OCR text already extracted,
asks Claude to identify:
  - Manufacturer / brand
  - Model name / number
  - Device category (PLC, IP camera, router, HMI, etc.)
  - Confidence estimate
  - Brief description

Uses claude-haiku for speed and cost efficiency on the per-frame pipeline.
Falls back to OCR hints if the API is unavailable.
"""
from __future__ import annotations
import base64
import json
import re
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class DeviceID:
    """Structured make/model identification result."""
    manufacturer: str = "Unknown"
    model: str = "Unknown"
    category: str = "Unknown"          # e.g. "PLC", "IP Camera", "Router"
    confidence: float = 0.0            # 0.0 – 1.0
    description: str = ""
    source: str = "claude"             # 'claude' | 'ocr_only' | 'manual'


_SYSTEM_PROMPT = """You are an expert in industrial control systems (ICS/OT/SCADA),
IoT devices, and network equipment. You are performing authorized penetration testing
reconnaissance. Analyze device images and extract make/model information.

Respond ONLY with valid JSON in this exact format:
{
  "manufacturer": "string",
  "model": "string",
  "category": "string",
  "confidence": 0.0-1.0,
  "description": "string"
}

Category must be one of: PLC, HMI, RTU, SCADA Terminal, IP Camera, Network Switch,
Router, Access Point, Industrial Panel, Sensor, Smart Meter, Badge Reader,
Fire Panel, HVAC Controller, UPS, Unknown.

If you cannot identify the device, use "Unknown" for manufacturer/model and low confidence."""

_USER_PROMPT_TEMPLATE = """Identify this device for penetration testing reconnaissance.

OCR text found on device: {ocr_text}
YOLO detection label: {yolo_label}

What is the manufacturer, model, and category of this device?"""


class DeviceClassifier:
    """
    Uses Claude claude-haiku-4-5-20251001 vision to identify device make/model from
    a cropped image + OCR hints.

    Args:
        api_key:    Anthropic API key (or set ANTHROPIC_API_KEY env var)
        model:      Claude model to use (default: claude-haiku-4-5-20251001 for speed)
        max_tokens: Max tokens in response (default 256 — JSON is small)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 256,
    ):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError("anthropic is required: pip install anthropic")
            import os
            key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            self._client = anthropic.Anthropic(api_key=key)
        return self._client

    def classify(
        self,
        crop,          # numpy ndarray (BGR)
        ocr_text: str = "",
        yolo_label: str = "device",
    ) -> DeviceID:
        """
        Classify a device crop image.

        Falls back to OCR-only identification if Claude API is unavailable.
        """
        try:
            return self._classify_with_claude(crop, ocr_text, yolo_label)
        except Exception as e:
            print(f"[Classifier] Claude API error: {e}. Falling back to OCR.")
            return self._classify_from_ocr(ocr_text, yolo_label)

    def _classify_with_claude(self, crop, ocr_text: str, yolo_label: str) -> DeviceID:
        client = self._get_client()

        # Encode crop as base64 JPEG
        _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = base64.standard_b64encode(buf.tobytes()).decode("utf-8")

        user_text = _USER_PROMPT_TEMPLATE.format(
            ocr_text=ocr_text or "(none)",
            yolo_label=yolo_label,
        )

        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img_b64,
                            },
                        },
                        {"type": "text", "text": user_text},
                    ],
                }
            ],
        )

        raw = response.content[0].text.strip()
        # Strip markdown code fences if present
        raw = re.sub(r"^```json?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)

        return DeviceID(
            manufacturer=data.get("manufacturer", "Unknown"),
            model=data.get("model", "Unknown"),
            category=data.get("category", "Unknown"),
            confidence=float(data.get("confidence", 0.0)),
            description=data.get("description", ""),
            source="claude",
        )

    def _classify_from_ocr(self, ocr_text: str, yolo_label: str) -> DeviceID:
        """Minimal fallback: use OCR text as best guess."""
        return DeviceID(
            manufacturer="Unknown",
            model=ocr_text[:60] if ocr_text else "Unknown",
            category=yolo_label.replace("_", " ").title(),
            confidence=0.2,
            description="Identified from OCR only (Claude API unavailable)",
            source="ocr_only",
        )
