"""
Report generation — JSON + HTML output.

Produces per-session reports containing all identified devices, their
vulnerability profiles, default credentials, and MITRE ATT&CK ICS mappings.
"""
from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


@dataclass
class DeviceRecord:
    """Fully enriched record for one identified device instance."""
    uid: str
    manufacturer: str
    model: str
    category: str
    confidence: float
    id_source: str
    lat: Optional[float]
    lon: Optional[float]
    alt_m: Optional[float]
    frame_index: int
    timestamp: float
    source_uri: str

    # Recon results
    cves: List[Dict] = field(default_factory=list)
    credentials: List[Dict] = field(default_factory=list)
    shodan: Optional[Dict] = None
    exploits: List[Dict] = field(default_factory=list)
    ics_techniques: List[List[str]] = field(default_factory=list)

    # Risk summary
    highest_severity: str = "NONE"
    cve_count: int = 0
    cred_count: int = 0


def build_record(
    uid: str,
    frame,
    device_id,
    cves,
    creds,
    shodan_result,
    exploit_refs,
    ics_techniques,
) -> DeviceRecord:
    severity_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"]
    highest = "NONE"
    for sev in severity_order:
        if any(c.severity == sev for c in cves):
            highest = sev
            break

    return DeviceRecord(
        uid=uid,
        manufacturer=device_id.manufacturer,
        model=device_id.model,
        category=device_id.category,
        confidence=device_id.confidence,
        id_source=device_id.source,
        lat=frame.lat,
        lon=frame.lon,
        alt_m=frame.alt_m,
        frame_index=frame.frame_index,
        timestamp=frame.timestamp,
        source_uri=frame.source_uri,
        cves=[{
            "id": c.cve_id, "severity": c.severity,
            "score": c.cvss_score, "description": c.description,
            "published": c.published, "vector": c.vector,
        } for c in cves],
        credentials=[{
            "service": cr.service, "username": cr.username,
            "password": cr.password, "notes": cr.notes,
        } for cr in creds],
        shodan=_shodan_summary(shodan_result),
        exploits=[{
            "cve_id": e.cve_id, "title": e.title,
            "edb_id": e.exploit_db_id, "url": e.url,
        } for e in exploit_refs],
        ics_techniques=[[t[0], t[1]] for t in ics_techniques],
        highest_severity=highest,
        cve_count=len(cves),
        cred_count=len(creds),
    )


def _shodan_summary(result) -> Optional[Dict]:
    if result is None:
        return None
    return {
        "query": result.query,
        "total_exposed": result.total_results,
        "ports": result.exposed_ports,
        "vulns": result.shodan_vulns,
        "orgs": result.orgs[:5],
        "countries": result.countries[:5],
    }


def save_json_report(records: List[DeviceRecord], output_dir: str) -> str:
    """Write a JSON report. Returns the file path."""
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"OT-id_report_{ts}.json")

    payload = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "device_count": len(records),
        "devices": [asdict(r) for r in records],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"[Report] JSON report saved: {path}")
    return path


def save_html_report(records: List[DeviceRecord], output_dir: str, template_dir: str) -> str:
    """Render an HTML report using Jinja2. Returns the file path."""
    try:
        from jinja2 import Environment, FileSystemLoader
    except ImportError:
        raise ImportError("jinja2 is required: pip install jinja2")

    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"OT-id_report_{ts}.html")

    env = Environment(loader=FileSystemLoader(template_dir), autoescape=True)
    tmpl = env.get_template("report.html.j2")

    html = tmpl.render(
        generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        records=records,
        severity_colors={
            "CRITICAL": "#dc3545",
            "HIGH": "#fd7e14",
            "MEDIUM": "#ffc107",
            "LOW": "#28a745",
            "NONE": "#6c757d",
        }
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[Report] HTML report saved: {path}")
    return path
