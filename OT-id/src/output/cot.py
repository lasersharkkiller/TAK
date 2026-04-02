"""
Cursor on Target (CoT) XML output for ATAK overlay.

Each identified device becomes a CoT event that ATAK renders as a map marker.
Risk level drives the icon color:
  CRITICAL/HIGH  → red    (argb -65536)
  MEDIUM         → yellow (argb -256)
  LOW/NONE       → green  (argb -16711936)

Transmission options:
  - UDP multicast to 239.2.3.1:6969  (TAK multicast — LAN only)
  - UDP unicast to a specific TAK Server or ATAK EUD
  - Write to .cot file for later import as a data package

CoT type used: "b-m-p-s-m"  (track / sensor)
Custom detail block: <__OT_id> contains full recon data.
"""
from __future__ import annotations
import socket
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from typing import List, Optional
from dataclasses import dataclass


# ── Risk colour map ───────────────────────────────────────────────────────────
_RISK_COLORS = {
    "CRITICAL": "-65536",    # red
    "HIGH":     "-65536",    # red
    "MEDIUM":   "-256",      # yellow
    "LOW":      "-16711936", # green
    "NONE":     "-16711936", # green
}

TAK_MULTICAST_ADDR = "239.2.3.1"
TAK_MULTICAST_PORT = 6969


def _stale_time(minutes: int = 60) -> str:
    t = datetime.now(timezone.utc) + timedelta(minutes=minutes)
    return t.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _highest_severity(cves) -> str:
    order = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"]
    for sev in order:
        for cve in cves:
            if cve.severity == sev:
                return sev
    return "NONE"


def build_cot_event(
    device_id,           # DeviceID from classifier
    cves,                # List[CVE]
    creds,               # List[Credential]
    lat: float,
    lon: float,
    alt_m: float = 0.0,
    frame_index: int = 0,
    source_uri: str = "",
    ics_techniques=None,
) -> str:
    """
    Build a CoT XML string for a single identified device.

    Returns UTF-8 XML string.
    """
    event_uid = f"OT-id-{uuid.uuid4().hex[:12]}"
    now = _now()
    stale = _stale_time(60)
    severity = _highest_severity(cves)
    color = _RISK_COLORS.get(severity, "-256")

    # ── <event> ────────────────────────────────────────────────────────────────
    event = ET.Element("event", {
        "version": "2.0",
        "uid": event_uid,
        "type": "b-m-p-s-m",
        "time": now,
        "start": now,
        "stale": stale,
        "how": "m-g",
    })

    # ── <point> ────────────────────────────────────────────────────────────────
    ET.SubElement(event, "point", {
        "lat": str(lat),
        "lon": str(lon),
        "hae": str(alt_m),
        "ce": "9999999",
        "le": "9999999",
    })

    # ── <detail> ───────────────────────────────────────────────────────────────
    detail = ET.SubElement(event, "detail")

    callsign = f"OT-{device_id.category.replace(' ', '_')}-{event_uid[-4:].upper()}"
    ET.SubElement(detail, "contact", {"callsign": callsign})

    cve_count = len(cves)
    cred_count = len(creds)
    remarks_text = (
        f"[OT-id] {device_id.manufacturer} {device_id.model} | "
        f"Cat: {device_id.category} | CVEs: {cve_count} | "
        f"Risk: {severity} | Creds: {cred_count} | "
        f"Frame: {frame_index}"
    )
    ET.SubElement(detail, "remarks").text = remarks_text
    ET.SubElement(detail, "color", {"argb": color})
    ET.SubElement(detail, "precisionlocation", {
        "geopointsrc": "GPS",
        "altsrc": "GPS",
    })

    # ── <__OT_id> custom block ─────────────────────────────────────────────────
    ot = ET.SubElement(detail, "__OT_id", {"version": "1.0"})

    ET.SubElement(ot, "device", {
        "manufacturer": device_id.manufacturer,
        "model": device_id.model,
        "category": device_id.category,
        "confidence": f"{device_id.confidence:.2f}",
        "id_source": device_id.source,
    })

    vulns_el = ET.SubElement(ot, "vulnerabilities", {"count": str(cve_count), "risk": severity})
    for cve in cves[:10]:   # cap at 10 in the CoT packet; full list in report
        ET.SubElement(vulns_el, "cve", {
            "id": cve.cve_id,
            "severity": cve.severity,
            "score": str(cve.cvss_score or 0),
            "published": cve.published,
        }).text = cve.description[:120]

    creds_el = ET.SubElement(ot, "credentials", {"count": str(cred_count)})
    for cred in creds[:20]:
        ET.SubElement(creds_el, "cred", {
            "service": cred.service,
            "username": cred.username,
            "password": cred.password,
        })

    if ics_techniques:
        ics_el = ET.SubElement(ot, "ics_techniques")
        for tid, tname in ics_techniques:
            ET.SubElement(ics_el, "technique", {"id": tid, "name": tname})

    ET.SubElement(ot, "source", {"uri": source_uri})

    ET.indent(event, space="  ")
    return '<?xml version=\'1.0\' encoding=\'UTF-8\' standalone=\'yes\'?>\n' + \
           ET.tostring(event, encoding="unicode")


def send_cot_udp(
    xml_str: str,
    host: str = TAK_MULTICAST_ADDR,
    port: int = TAK_MULTICAST_PORT,
) -> None:
    """Send a CoT XML packet via UDP (multicast or unicast)."""
    data = xml_str.encode("utf-8")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        if host == TAK_MULTICAST_ADDR:
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 32)
        sock.sendto(data, (host, port))
    finally:
        sock.close()
