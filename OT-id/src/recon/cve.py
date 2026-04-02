"""
CVE lookup via NVD REST API v2.

Queries the National Vulnerability Database for CVEs matching a device's
manufacturer + model. Returns structured CVE records with CVSS scores,
severity, and descriptions.

Rate limits: 5 req/30s without API key, 50 req/30s with key.
Set NVD_API_KEY in .env to increase throughput.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import List, Optional

import httpx


NVD_BASE = "https://services.nvd.nist.gov/rest/json/cves/2.0"
_LAST_REQUEST_TIME = 0.0
_MIN_INTERVAL = 0.6   # seconds between requests (unauthenticated)


@dataclass
class CVE:
    cve_id: str
    description: str
    cvss_score: Optional[float]       # 0.0 – 10.0
    severity: str                     # CRITICAL / HIGH / MEDIUM / LOW / NONE
    vector: str                       # CVSS attack vector string
    published: str
    references: List[str] = field(default_factory=list)


def _throttle():
    global _LAST_REQUEST_TIME
    elapsed = time.time() - _LAST_REQUEST_TIME
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _LAST_REQUEST_TIME = time.time()


def _parse_cve(vuln: dict) -> CVE:
    cve = vuln.get("cve", {})
    cve_id = cve.get("id", "")
    published = cve.get("published", "")

    # Description (English preferred)
    descs = cve.get("descriptions", [])
    description = next(
        (d["value"] for d in descs if d.get("lang") == "en"),
        descs[0]["value"] if descs else "",
    )

    # CVSS v3.1 preferred, fall back to v2
    metrics = cve.get("metrics", {})
    cvss_score = None
    severity = "NONE"
    vector = ""

    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        entries = metrics.get(key)
        if entries:
            m = entries[0].get("cvssData", {})
            cvss_score = m.get("baseScore")
            severity = m.get("baseSeverity", "NONE")
            vector = m.get("vectorString", "")
            break

    refs = [r["url"] for r in cve.get("references", []) if "url" in r]

    return CVE(
        cve_id=cve_id,
        description=description[:400],   # truncate for readability
        cvss_score=cvss_score,
        severity=severity,
        vector=vector,
        published=published[:10],
        references=refs[:5],
    )


def search_cves(
    manufacturer: str,
    model: str,
    api_key: Optional[str] = None,
    max_results: int = 20,
) -> List[CVE]:
    """
    Query NVD for CVEs matching the given device.

    Args:
        manufacturer:   e.g. "Siemens"
        model:          e.g. "S7-1200"
        api_key:        NVD API key (optional but recommended)
        max_results:    Max CVEs to return

    Returns:
        List of CVE objects sorted by CVSS score descending
    """
    if manufacturer == "Unknown" and model == "Unknown":
        return []

    # Build keyword query — NVD keyword search is OR-based between words
    keyword = f"{manufacturer} {model}".strip()

    headers = {}
    if api_key:
        headers["apiKey"] = api_key

    params = {
        "keywordSearch": keyword,
        "resultsPerPage": min(max_results, 2000),
    }

    _throttle()
    try:
        resp = httpx.get(NVD_BASE, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as e:
        print(f"[CVE] NVD request failed: {e}")
        return []

    vulns = data.get("vulnerabilities", [])
    cves = [_parse_cve(v) for v in vulns]

    # Sort by CVSS score descending
    cves.sort(key=lambda c: c.cvss_score or 0.0, reverse=True)
    return cves[:max_results]
