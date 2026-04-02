"""
Shodan recon module.

Given a device make/model or an IP address (extracted via OCR from the feed),
queries Shodan for:
  - Exposed services / open ports
  - Banner information
  - Known vulnerabilities (from Shodan's vuln feed)
  - Geographic exposure data
  - Organization / ISP

Requires SHODAN_API_KEY in .env
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class ShodanResult:
    query: str
    total_results: int
    sample: List[Dict]             # raw Shodan host records (first N)
    exposed_ports: List[int]
    banners: List[str]
    shodan_vulns: List[str]        # CVE IDs from Shodan vuln tag
    orgs: List[str]
    countries: List[str]


def search_by_product(
    api_key: str,
    manufacturer: str,
    model: str,
    max_results: int = 10,
) -> Optional[ShodanResult]:
    """
    Search Shodan for internet-exposed instances of the identified device.

    Args:
        api_key:        Shodan API key
        manufacturer:   Device manufacturer
        model:          Device model
        max_results:    Max host records to return

    Returns:
        ShodanResult or None on error
    """
    try:
        import shodan
    except ImportError:
        raise ImportError("shodan is required: pip install shodan")

    if manufacturer == "Unknown" and model == "Unknown":
        return None

    api = shodan.Shodan(api_key)

    # Build a Shodan dork
    parts = []
    if manufacturer.lower() not in ("unknown", ""):
        parts.append(f'"{manufacturer}"')
    if model.lower() not in ("unknown", ""):
        parts.append(f'"{model}"')
    query = " ".join(parts)

    try:
        results = api.search(query, limit=max_results)
    except shodan.APIError as e:
        print(f"[Shodan] API error: {e}")
        return None

    matches = results.get("matches", [])
    total = results.get("total", 0)

    ports = []
    banners = []
    vuln_ids = []
    orgs = []
    countries = []

    for host in matches:
        if "port" in host:
            ports.append(host["port"])
        if "data" in host:
            banners.append(host["data"][:200])
        for v in host.get("vulns", {}).keys():
            vuln_ids.append(v)
        if host.get("org"):
            orgs.append(host["org"])
        if host.get("location", {}).get("country_name"):
            countries.append(host["location"]["country_name"])

    return ShodanResult(
        query=query,
        total_results=total,
        sample=matches[:max_results],
        exposed_ports=sorted(set(ports)),
        banners=banners[:5],
        shodan_vulns=list(set(vuln_ids)),
        orgs=list(set(orgs))[:10],
        countries=list(set(countries))[:10],
    )


def lookup_ip(api_key: str, ip: str) -> Optional[Dict]:
    """
    Direct IP lookup on Shodan — use when OCR extracts an IP from a label.

    Returns the raw Shodan host dict or None.
    """
    try:
        import shodan
    except ImportError:
        raise ImportError("shodan is required: pip install shodan")

    api = shodan.Shodan(api_key)
    try:
        return api.host(ip)
    except shodan.APIError as e:
        print(f"[Shodan] IP lookup failed for {ip}: {e}")
        return None
