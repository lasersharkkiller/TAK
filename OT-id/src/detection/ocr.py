"""
OCR module — reads text from detected device crops.

Extracts:
  - Brand/manufacturer names
  - Model numbers (e.g. "Siemens S7-1200", "Axis P3245-V")
  - Serial numbers
  - IP addresses printed on labels
  - Firmware version stickers

Uses EasyOCR (GPU-optional, good on embedded device labels).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import re


@dataclass
class OCRResult:
    raw_text: str               # All text found in the crop, joined
    lines: List[str]            # Individual text detections
    model_candidates: List[str] # Strings that look like model numbers
    ip_candidates: List[str]    # Strings that look like IP addresses
    brand_candidates: List[str] # Known brand names found


# Regex patterns
_MODEL_RE = re.compile(
    r"\b([A-Z]{1,6}[-_]?\d{2,8}[A-Z0-9\-/_.]{0,10})\b"
)
_IP_RE = re.compile(
    r"\b(\d{1,3}(?:\.\d{1,3}){3}(?::\d{2,5})?)\b"
)

# Subset of known OT/IoT brands for quick matching
KNOWN_BRANDS = {
    "siemens", "schneider", "allen-bradley", "rockwell", "honeywell",
    "emerson", "abb", "ge", "omron", "mitsubishi", "delta", "beckhoff",
    "axis", "hikvision", "dahua", "bosch", "hanwha", "pelco", "vivotek",
    "cisco", "netgear", "ubiquiti", "mikrotik", "fortinet", "palo alto",
    "advantech", "moxa", "phoenix contact", "wago", "pilz", "ifm",
    "pepperl+fuchs", "banner", "keyence",
}


class OCRReader:
    """
    Wraps EasyOCR and provides device-label-aware text extraction.

    Args:
        languages:  List of language codes (default ['en'])
        gpu:        Use GPU if available (default False for portability)
    """

    def __init__(self, languages: List[str] = None, gpu: bool = False):
        self.languages = languages or ["en"]
        self.gpu = gpu
        self._reader = None

    def load(self) -> None:
        try:
            import easyocr
        except ImportError:
            raise ImportError("easyocr is required: pip install easyocr")
        print("[OCR] Loading EasyOCR model…")
        self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
        print("[OCR] Ready.")

    def read_crop(self, crop) -> OCRResult:
        """
        Args:
            crop: numpy ndarray (BGR) of a detected device region
        Returns:
            OCRResult with parsed text fields
        """
        if self._reader is None:
            self.load()

        results = self._reader.readtext(crop, detail=0, paragraph=False)
        lines = [r.strip() for r in results if r.strip()]
        raw = " ".join(lines)

        return OCRResult(
            raw_text=raw,
            lines=lines,
            model_candidates=self._extract_models(raw),
            ip_candidates=self._extract_ips(raw),
            brand_candidates=self._extract_brands(raw),
        )

    def _extract_models(self, text: str) -> List[str]:
        return list(set(_MODEL_RE.findall(text.upper())))

    def _extract_ips(self, text: str) -> List[str]:
        return list(set(_IP_RE.findall(text)))

    def _extract_brands(self, text: str) -> List[str]:
        lower = text.lower()
        return [b for b in KNOWN_BRANDS if b in lower]
