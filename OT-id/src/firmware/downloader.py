"""
Phase 2 — Firmware Downloader

Given an identified device (manufacturer + model), attempts to locate and
download firmware binaries for offline analysis (binary diffing, CVE
validation, hardcoded credential extraction, etc.).

Sources tried in order:
  1. Vendor direct download portals (scrape known URL patterns)
  2. firmware.re — community firmware repository
  3. IoT Inspector database (FKIE)
  4. NIST NVD CPE → vendor advisory page scrape

All downloads are stored in data/firmware/<vendor>/<model>/<version>/

STATUS: Phase 2 — stubs in place, implement vendor scrapers incrementally.
"""
from __future__ import annotations
import os
import hashlib
import json
from dataclasses import dataclass, field
from typing import List, Optional

import requests
from bs4 import BeautifulSoup


FIRMWARE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "firmware"
)


@dataclass
class FirmwareRecord:
    vendor: str
    model: str
    version: str
    source_url: str
    local_path: Optional[str] = None
    sha256: Optional[str] = None
    size_bytes: int = 0
    download_ok: bool = False
    notes: str = ""


# ── Vendor portal URL patterns ────────────────────────────────────────────────
# Add entries as vendor portals are mapped.
# Format: (vendor_pattern, search_url_template, link_selector)
VENDOR_PORTALS = {
    "axis": {
        "search": "https://www.axis.com/support/firmware?query={model}",
        "notes": "Axis firmware portal — requires scraping product page",
    },
    "hikvision": {
        "search": "https://www.hikvision.com/en/support/download/firmware/?model={model}",
        "notes": "Hikvision firmware download page",
    },
    "dahua": {
        "search": "https://www.dahuasecurity.com/support/downloadCenter/",
        "notes": "Dahua download center — POST-based search",
    },
    "siemens": {
        "search": "https://support.industry.siemens.com/cs/search?query={model}+firmware",
        "notes": "Siemens Industry Support portal",
    },
    "moxa": {
        "search": "https://www.moxa.com/en/support/product-support/search?keyword={model}",
        "notes": "Moxa product support search",
    },
    "cisco": {
        "search": "https://software.cisco.com/download/home",
        "notes": "Cisco Software Download Center — requires CCO login for most",
    },
}


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_file(url: str, dest_path: str, timeout: int = 60) -> bool:
    """Stream-download a file to dest_path. Returns True on success."""
    try:
        resp = requests.get(url, stream=True, timeout=timeout,
                            headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(65536):
                f.write(chunk)
        return True
    except requests.RequestException as e:
        print(f"[Firmware] Download failed: {e}")
        return False


class FirmwareDownloader:
    """
    Attempts to find and download firmware for an identified device.

    Usage:
        dl = FirmwareDownloader()
        records = dl.download("Axis", "M3106-L")
        for r in records:
            if r.download_ok:
                print(f"Downloaded: {r.local_path} ({r.sha256})")
    """

    def __init__(self, firmware_dir: str = FIRMWARE_DIR):
        self.firmware_dir = firmware_dir

    def download(self, manufacturer: str, model: str) -> List[FirmwareRecord]:
        """
        Attempt firmware download from all known sources.
        Returns list of FirmwareRecords (successful + failed attempts).
        """
        records = []

        # 1. Try vendor portal
        vendor_record = self._try_vendor_portal(manufacturer, model)
        if vendor_record:
            records.append(vendor_record)

        # 2. Try firmware.re
        fwre_records = self._try_firmware_re(manufacturer, model)
        records.extend(fwre_records)

        if not any(r.download_ok for r in records):
            print(f"[Firmware] No firmware found for {manufacturer} {model}")

        return records

    def _try_vendor_portal(self, manufacturer: str, model: str) -> Optional[FirmwareRecord]:
        """
        Phase 2 TODO: implement per-vendor scrapers.
        This stub logs the portal URL for manual retrieval.
        """
        mfr_l = manufacturer.lower()
        for key, portal in VENDOR_PORTALS.items():
            if key in mfr_l:
                search_url = portal["search"].format(model=model.replace(" ", "+"))
                print(f"[Firmware] Vendor portal for {manufacturer}: {search_url}")
                return FirmwareRecord(
                    vendor=manufacturer,
                    model=model,
                    version="unknown",
                    source_url=search_url,
                    download_ok=False,
                    notes=f"Manual retrieval required. {portal['notes']}",
                )
        return None

    def _try_firmware_re(self, manufacturer: str, model: str) -> List[FirmwareRecord]:
        """
        Query firmware.re for matching firmware images.
        firmware.re hosts community-contributed firmware binaries.

        Phase 2 TODO: implement actual search + download.
        """
        # Stub — log intent
        print(f"[Firmware] TODO: query firmware.re for {manufacturer} {model}")
        return []

    def save_index(self, records: List[FirmwareRecord]) -> str:
        """Write a JSON index of all firmware download attempts."""
        os.makedirs(self.firmware_dir, exist_ok=True)
        path = os.path.join(self.firmware_dir, "firmware_index.json")
        existing = []
        if os.path.exists(path):
            with open(path) as f:
                existing = json.load(f)

        for r in records:
            existing.append({
                "vendor": r.vendor, "model": r.model, "version": r.version,
                "source_url": r.source_url, "local_path": r.local_path,
                "sha256": r.sha256, "size_bytes": r.size_bytes,
                "download_ok": r.download_ok, "notes": r.notes,
            })

        with open(path, "w") as f:
            json.dump(existing, f, indent=2)
        return path
