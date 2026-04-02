"""
Default credential lookup.

Loads a compiled default credentials database (CSV) and returns matching
entries for a given manufacturer/model/category.

The data/default_creds.csv file is populated from public sources:
  - SecLists/Passwords/Default-Credentials/
  - RouterPasswords.com exports
  - NIST ICS-CERT advisories
  - Vendor documentation (publicly available)
  - CISA ICS advisories

CSV schema: vendor,model,service,username,password,notes
"""
from __future__ import annotations
import csv
import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Credential:
    vendor: str
    model: str
    service: str       # e.g. "web", "ssh", "telnet", "ftp", "snmp"
    username: str
    password: str
    notes: str = ""


_DEFAULT_CSV_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "default_creds.csv"
)

_LOADED_CREDS: Optional[List[Credential]] = None


def _load_creds(csv_path: str) -> List[Credential]:
    creds = []
    if not os.path.exists(csv_path):
        print(f"[Creds] Credential DB not found at {csv_path}. Run scripts/fetch_creds.py to populate.")
        return creds
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            creds.append(Credential(
                vendor=row.get("vendor", "").strip(),
                model=row.get("model", "").strip(),
                service=row.get("service", "").strip(),
                username=row.get("username", "").strip(),
                password=row.get("password", "").strip(),
                notes=row.get("notes", "").strip(),
            ))
    return creds


def get_default_creds(
    manufacturer: str,
    model: str,
    category: str = "",
    csv_path: str = _DEFAULT_CSV_PATH,
) -> List[Credential]:
    """
    Return default credentials for the identified device.

    Matches on (vendor OR model) substring — case insensitive.
    Prioritises exact model matches over vendor-only matches.

    Args:
        manufacturer:   Device manufacturer (from DeviceID)
        model:          Device model (from DeviceID)
        category:       Device category (used as fallback search term)
        csv_path:       Path to credentials CSV file

    Returns:
        List of matching Credential objects, deduplicated
    """
    global _LOADED_CREDS
    if _LOADED_CREDS is None:
        _LOADED_CREDS = _load_creds(csv_path)

    mfr_l = manufacturer.lower()
    mdl_l = model.lower()
    cat_l = category.lower()

    exact: List[Credential] = []
    vendor_only: List[Credential] = []

    for cred in _LOADED_CREDS:
        v = cred.vendor.lower()
        m = cred.model.lower()

        model_hit = mdl_l not in ("unknown", "") and (mdl_l in m or m in mdl_l)
        vendor_hit = mfr_l not in ("unknown", "") and (mfr_l in v or v in mfr_l)

        if model_hit and vendor_hit:
            exact.append(cred)
        elif model_hit or vendor_hit:
            vendor_only.append(cred)

    # Deduplicate by (service, username, password)
    seen = set()
    results = []
    for cred in (exact + vendor_only):
        key = (cred.service, cred.username, cred.password)
        if key not in seen:
            seen.add(key)
            results.append(cred)

    return results
