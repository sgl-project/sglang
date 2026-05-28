#!/usr/bin/env python3
"""
fm_osintaas.py — OSINT-as-a-Service Agent (Port 7826)
FractalMesh OMEGA Titan | Person/Domain/WiFi/Social intelligence gathering.
All credentials sourced from ~/.secrets/fractal.env at runtime.
Samuel James Hiotis | ABN 56 628 117 363
"""
import base64
import csv
import io
import json
import logging
import os
import re
import signal
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ── vault ─────────────────────────────────────────────────────────────────────
_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _line in _vault.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

# ── config ────────────────────────────────────────────────────────────────────
PORT                   = int(os.getenv("OSINTAAS_PORT", "7826"))
ROOT                   = Path(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")))
DB                     = ROOT / "database" / "sovereign.db"
LOG                    = ROOT / "logs" / "osintaas.log"
CRAWLBASE_NORMAL_TOKEN = os.getenv("CRAWLBASE_NORMAL_TOKEN", "")
CRAWLBASE_JS_TOKEN     = os.getenv("CRAWLBASE_JS_TOKEN", "")
WIGLE_API_NAME         = os.getenv("WIGLE_API_NAME", "")
WIGLE_API_TOKEN        = os.getenv("WIGLE_API_TOKEN", "")
GOOGLE_CSE_API_KEY     = os.getenv("GOOGLE_CSE_API_KEY", "")
GOOGLE_CSE_ID          = os.getenv("GOOGLE_CSE_ID", "")
SCRAPERAPI_KEY         = os.getenv("SCRAPERAPI_KEY", "")

ROOT.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)
DB.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [OSINTAAS] %(message)s",
    handlers=[logging.FileHandler(LOG), logging.StreamHandler()],
)
log = logging.getLogger("osintaas")

# ── false-positive email domains to strip ─────────────────────────────────────
_FP_DOMAINS = {
    "example.com", "test.com", "sentry.io", "domain.com", "email.com",
    "yoursite.com", "company.com", "noreply.com", "no-reply.com",
    "placeholder.com", "wixpress.com", "amazonaws.com", "cloudfront.net",
}

# ── database ──────────────────────────────────────────────────────────────────

def _db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB), timeout=15)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def _db_init():
    conn = _db_connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS osint_reports (
            id          INTEGER PRIMARY KEY,
            target      TEXT,
            report_type TEXT,
            status      TEXT,
            payload     TEXT,
            created_at  REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS osint_leads (
            id         INTEGER PRIMARY KEY,
            report_id  INTEGER,
            name       TEXT,
            email      TEXT,
            phone      TEXT,
            social     TEXT,
            location   TEXT,
            score      REAL,
            created_at REAL
        )
    """)
    conn.commit()
    conn.close()
    log.info("DB tables verified at %s", DB)


def _insert_report(target: str, report_type: str, status: str, payload: dict) -> int:
    conn = _db_connect()
    cur = conn.execute(
        "INSERT INTO osint_reports (target, report_type, status, payload, created_at) VALUES (?,?,?,?,?)",
        (target, report_type, status, json.dumps(payload), time.time()),
    )
    rid = cur.lastrowid
    conn.commit()
    conn.close()
    return rid


def _update_report(report_id: int, status: str, payload: dict):
    conn = _db_connect()
    conn.execute(
        "UPDATE osint_reports SET status=?, payload=? WHERE id=?",
        (status, json.dumps(payload), report_id),
    )
    conn.commit()
    conn.close()


def _insert_lead(report_id: int, name: str, email: str, phone: str,
                 social: str, location: str, score: float):
    conn = _db_connect()
    conn.execute(
        "INSERT INTO osint_leads (report_id,name,email,phone,social,location,score,created_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (report_id, name, email, phone, social, location, score, time.time()),
    )
    conn.commit()
    conn.close()


# ── helpers ───────────────────────────────────────────────────────────────────

def _google_cse(query: str, num: int = 10) -> list:
    """Query Google Custom Search Engine; returns list of {link, title, snippet}."""
    if not GOOGLE_CSE_API_KEY or not GOOGLE_CSE_ID:
        log.warning("Google CSE credentials not set")
        return []
    try:
        params = urllib.parse.urlencode({
            "key": GOOGLE_CSE_API_KEY,
            "cx":  GOOGLE_CSE_ID,
            "q":   query,
            "num": min(num, 10),
        })
        url = f"https://customsearch.googleapis.com/customsearch/v1?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "FractalMesh-OSINT/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        items = data.get("items", [])
        return [
            {"link": i.get("link", ""), "title": i.get("title", ""), "snippet": i.get("snippet", "")}
            for i in items
        ]
    except Exception as exc:
        log.warning("Google CSE error for %r: %s", query, exc)
        return []


def _crawlbase_fetch(url: str, js: bool = False) -> str:
    """Fetch a URL via Crawlbase; returns body text or '' on error."""
    token = CRAWLBASE_JS_TOKEN if js else CRAWLBASE_NORMAL_TOKEN
    if not token:
        log.warning("Crawlbase token not set (js=%s)", js)
        return ""
    try:
        params  = urllib.parse.urlencode({"token": token, "url": url})
        api_url = f"https://api.crawlbase.com/?{params}"
        req = urllib.request.Request(api_url, headers={"User-Agent": "FractalMesh-OSINT/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        log.warning("Crawlbase fetch error for %s: %s", url, exc)
        return ""


def _extract_emails(text: str) -> list:
    """Extract unique, filtered emails from raw text."""
    raw = re.findall(r"[\w.+\-]+@[\w\-]+\.[a-zA-Z]{2,}", text)
    seen = set()
    result = []
    for e in raw:
        e = e.lower()
        domain = e.split("@")[-1]
        if domain in _FP_DOMAINS:
            continue
        if e not in seen:
            seen.add(e)
            result.append(e)
    return result


def _extract_phones(text: str) -> list:
    """Extract unique phone numbers from raw text."""
    raw = re.findall(r"\+?[\d\s\-(). ]{7,20}", text)
    seen = set()
    result = []
    for p in raw:
        p = p.strip()
        digits = re.sub(r"\D", "", p)
        if len(digits) < 7 or len(digits) > 15:
            continue
        if digits not in seen:
            seen.add(digits)
            result.append(p)
    return result


def _extract_social_handles(text: str) -> list:
    """Extract social profile URLs from raw text."""
    patterns = [
        r"https?://(?:www\.)?twitter\.com/[\w]+",
        r"https?://(?:www\.)?linkedin\.com/in/[\w\-]+",
        r"https?://(?:www\.)?github\.com/[\w\-]+",
        r"https?://(?:www\.)?facebook\.com/[\w.\-]+",
        r"https?://(?:www\.)?instagram\.com/[\w.\-]+",
    ]
    seen = set()
    result = []
    for pat in patterns:
        for m in re.findall(pat, text, re.IGNORECASE):
            m = m.rstrip("/.,;)")
            if m not in seen:
                seen.add(m)
                result.append(m)
    return result


def _extract_subdomains(text: str, domain: str) -> list:
    """Extract subdomains of a given domain from raw text."""
    escaped = re.escape(domain)
    pattern = rf"[a-z0-9](?:[a-z0-9\-]*[a-z0-9])?\.{escaped}"
    raw = re.findall(pattern, text, re.IGNORECASE)
    seen = set()
    result = []
    for s in raw:
        s = s.lower()
        if s != domain and s not in seen:
            seen.add(s)
            result.append(s)
    return result


def _score_lead(email: str, phone: str, social: str) -> float:
    """Score a lead 0.0–1.0 based on data completeness."""
    score = 0.0
    if email and email.strip():
        score += 0.5
    if phone and phone.strip():
        score += 0.3
    if social and social.strip():
        score += 0.2
    return round(min(score, 1.0), 2)


def _crawl_urls(urls: list, js: bool = False) -> str:
    """Fetch up to 5 URLs and concatenate body text."""
    combined = []
    for url in urls[:5]:
        body = _crawlbase_fetch(url, js=js)
        if body:
            combined.append(body)
    return "\n".join(combined)


# ── scan handlers ─────────────────────────────────────────────────────────────

def _scan_person(body: dict) -> dict:
    target   = body.get("target", "").strip()
    location = body.get("location", "").strip()
    depth    = body.get("depth", "standard")

    if not target:
        return {"error": "target is required"}

    query = f'"{target}" site:linkedin.com OR site:twitter.com OR site:facebook.com'
    if location:
        query += f' "{location}"'

    results = _google_cse(query, num=10)
    urls    = [r["link"] for r in results]

    # Deep: also search without site restriction
    if depth == "deep":
        extra = _google_cse(f'"{target}" {location} email contact'.strip(), num=5)
        urls += [r["link"] for r in extra]

    raw_text = _crawl_urls(urls, js=False)
    # Also pull snippets as fallback text
    snippet_text = " ".join(r.get("snippet", "") for r in results)
    full_text = raw_text + "\n" + snippet_text

    emails  = _extract_emails(full_text)
    phones  = _extract_phones(full_text)
    socials = _extract_social_handles(full_text)

    payload = {
        "target":  target,
        "emails":  emails,
        "phones":  phones,
        "socials": socials,
        "sources": urls[:5],
        "depth":   depth,
    }
    rid = _insert_report(target, "person", "complete", payload)

    # Store individual leads
    for email in emails:
        score = _score_lead(email, phones[0] if phones else "", socials[0] if socials else "")
        _insert_lead(rid, target, email, phones[0] if phones else "",
                     socials[0] if socials else "", location, score)

    if not emails and phones:
        score = _score_lead("", phones[0], socials[0] if socials else "")
        _insert_lead(rid, target, "", phones[0], socials[0] if socials else "", location, score)

    return {
        "report_id": rid,
        "target":    target,
        "emails":    emails,
        "phones":    phones,
        "socials":   socials,
        "status":    "complete",
    }


def _scan_domain(body: dict) -> dict:
    domain = body.get("domain", "").strip().lower()
    depth  = body.get("depth", "standard")

    if not domain:
        return {"error": "domain is required"}

    query   = f"site:{domain} email OR contact OR team"
    results = _google_cse(query, num=10)
    urls    = [r["link"] for r in results]

    extra_results = []
    if depth == "deep":
        deep_q        = f'"@{domain}" -site:{domain}'
        extra_results = _google_cse(deep_q, num=10)
        urls          += [r["link"] for r in extra_results]

    raw_text = _crawl_urls(urls, js=True)
    snippet_text = " ".join(
        r.get("snippet", "") for r in (results + extra_results)
    )
    full_text = raw_text + "\n" + snippet_text

    emails     = _extract_emails(full_text)
    # Keep only emails that belong to the target domain plus any found
    subdomains = _extract_subdomains(full_text, domain)

    payload = {
        "domain":     domain,
        "emails":     emails,
        "subdomains": subdomains,
        "sources":    urls[:5],
        "depth":      depth,
    }
    rid = _insert_report(domain, "domain", "complete", payload)

    for email in emails:
        score = _score_lead(email, "", "")
        _insert_lead(rid, domain, email, "", "", domain, score)

    return {
        "report_id":  rid,
        "domain":     domain,
        "emails":     emails,
        "subdomains": subdomains,
        "status":     "complete",
    }


def _scan_wifi(body: dict) -> dict:
    lat       = body.get("lat")
    lon       = body.get("lon")
    radius_km = body.get("radius_km", 1.0)

    if lat is None or lon is None:
        return {"error": "lat and lon are required"}

    lat       = float(lat)
    lon       = float(lon)
    radius_km = float(radius_km)

    if not WIGLE_API_NAME or not WIGLE_API_TOKEN:
        return {"error": "WiGLE credentials not configured"}

    deg_offset = radius_km / 111.0
    params = urllib.parse.urlencode({
        "latrange1":      lat - deg_offset,
        "latrange2":      lat + deg_offset,
        "longrange1":     lon - deg_offset,
        "longrange2":     lon + deg_offset,
        "resultsPerPage": 50,
    })
    wigle_url = f"https://api.wigle.net/api/v2/network/search?{params}"
    creds     = base64.b64encode(f"{WIGLE_API_NAME}:{WIGLE_API_TOKEN}".encode()).decode()
    req       = urllib.request.Request(
        wigle_url,
        headers={"Authorization": f"Basic {creds}", "Accept": "application/json"},
    )

    networks = []
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode())
        for net in data.get("results", []):
            networks.append({
                "ssid":       net.get("ssid", ""),
                "lat":        net.get("trilat", lat),
                "lon":        net.get("trilong", lon),
                "encryption": net.get("encryption", "unknown"),
                "bssid":      net.get("netid", ""),
            })
    except Exception as exc:
        log.warning("WiGLE API error: %s", exc)
        return {"error": f"WiGLE API error: {exc}"}

    payload = {
        "lat":       lat,
        "lon":       lon,
        "radius_km": radius_km,
        "networks":  networks,
    }
    rid = _insert_report(f"{lat},{lon}", "wifi", "complete", payload)

    return {
        "report_id": rid,
        "networks":  networks,
        "count":     len(networks),
    }


def _scan_social(body: dict) -> dict:
    username  = body.get("username", "").strip()
    platforms = body.get("platforms", ["twitter", "github", "linkedin"])

    if not username:
        return {"error": "username is required"}

    platform_domains = {
        "twitter":   "twitter.com",
        "github":    "github.com",
        "linkedin":  "linkedin.com",
        "facebook":  "facebook.com",
        "instagram": "instagram.com",
        "reddit":    "reddit.com",
    }

    profiles = {}
    for platform in platforms:
        domain = platform_domains.get(platform.lower())
        if not domain:
            profiles[platform] = {"error": "unsupported platform"}
            continue

        query   = f'"{username}" site:{domain}'
        results = _google_cse(query, num=5)
        if not results:
            profiles[platform] = {"found": False}
            continue

        urls     = [r["link"] for r in results]
        raw_text = _crawl_urls(urls[:3], js=False)
        snippet_text = " ".join(r.get("snippet", "") for r in results)
        full_text = raw_text + "\n" + snippet_text

        emails  = _extract_emails(full_text)
        phones  = _extract_phones(full_text)
        socials = _extract_social_handles(full_text)

        profiles[platform] = {
            "found":    True,
            "url":      urls[0] if urls else "",
            "emails":   emails,
            "phones":   phones,
            "socials":  socials,
            "snippets": [r.get("snippet", "") for r in results[:3]],
        }

    payload = {"username": username, "platforms": platforms, "profiles": profiles}
    rid     = _insert_report(username, "social", "complete", payload)

    # Store leads from discovered profiles
    all_emails  = []
    all_phones  = []
    all_socials = []
    for pdata in profiles.values():
        if isinstance(pdata, dict) and pdata.get("found"):
            all_emails  += pdata.get("emails", [])
            all_phones  += pdata.get("phones", [])
            all_socials += pdata.get("socials", [])

    for email in list(dict.fromkeys(all_emails)):
        score = _score_lead(email, all_phones[0] if all_phones else "",
                            all_socials[0] if all_socials else "")
        _insert_lead(rid, username, email, all_phones[0] if all_phones else "",
                     all_socials[0] if all_socials else "", "", score)

    return {
        "report_id": rid,
        "username":  username,
        "profiles":  profiles,
    }


# ── HTTP handler ──────────────────────────────────────────────────────────────

class OSINTHandler(BaseHTTPRequestHandler):
    server_version = "FractalMesh-OSINTaaS/1.0"

    def log_message(self, fmt, *args):
        log.info("%s - %s", self.address_string(), fmt % args)

    # ── response helpers ──────────────────────────────────────────────────────

    def _send_json(self, data: dict | list, status: int = 200):
        body = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_csv(self, text: str, status: int = 200):
        body = text.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/csv")
        self.send_header("Content-Disposition", "attachment; filename=leads.csv")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode())
        except Exception:
            return {}

    def _parse_path(self):
        parsed = urllib.parse.urlparse(self.path)
        return parsed.path, urllib.parse.parse_qs(parsed.query)

    # ── GET ───────────────────────────────────────────────────────────────────

    def do_GET(self):
        path, qs = self._parse_path()

        if path == "/health":
            self._send_json({"status": "ok", "service": "fm-osintaas", "port": PORT})
            return

        if path == "/reports":
            conn  = _db_connect()
            rows  = conn.execute(
                "SELECT id, target, report_type, status, created_at FROM osint_reports ORDER BY id DESC"
            ).fetchall()
            conn.close()
            self._send_json([dict(r) for r in rows])
            return

        m = re.fullmatch(r"/reports/(\d+)", path)
        if m:
            rid  = int(m.group(1))
            conn = _db_connect()
            row  = conn.execute("SELECT * FROM osint_reports WHERE id=?", (rid,)).fetchone()
            conn.close()
            if not row:
                self._send_json({"error": "not found"}, 404)
                return
            data = dict(row)
            try:
                data["payload"] = json.loads(data["payload"] or "{}")
            except Exception:
                pass
            self._send_json(data)
            return

        if path == "/leads":
            min_score = float(qs.get("min_score", [0.0])[0])
            conn      = _db_connect()
            rows      = conn.execute(
                "SELECT * FROM osint_leads WHERE score >= ? ORDER BY score DESC", (min_score,)
            ).fetchall()
            conn.close()
            self._send_json([dict(r) for r in rows])
            return

        self._send_json({"error": "not found"}, 404)

    # ── POST ──────────────────────────────────────────────────────────────────

    def do_POST(self):
        path, _ = self._parse_path()
        body    = self._read_body()

        if path == "/scan/person":
            result = _scan_person(body)
            status = 400 if "error" in result else 200
            self._send_json(result, status)
            return

        if path == "/scan/domain":
            result = _scan_domain(body)
            status = 400 if "error" in result else 200
            self._send_json(result, status)
            return

        if path == "/scan/wifi":
            result = _scan_wifi(body)
            status = 400 if "error" in result else 200
            self._send_json(result, status)
            return

        if path == "/scan/social":
            result = _scan_social(body)
            status = 400 if "error" in result else 200
            self._send_json(result, status)
            return

        if path == "/leads/export":
            fmt       = body.get("format", "json")
            min_score = float(body.get("min_score", 0.0))
            conn      = _db_connect()
            rows      = conn.execute(
                "SELECT * FROM osint_leads WHERE score >= ? ORDER BY score DESC", (min_score,)
            ).fetchall()
            conn.close()
            leads = [dict(r) for r in rows]

            if fmt == "csv":
                output = io.StringIO()
                if leads:
                    writer = csv.DictWriter(output, fieldnames=leads[0].keys())
                    writer.writeheader()
                    writer.writerows(leads)
                else:
                    output.write("id,report_id,name,email,phone,social,location,score,created_at\n")
                self._send_csv(output.getvalue())
            else:
                self._send_json(leads)
            return

        if path == "/report/generate":
            report_id = body.get("report_id")
            if report_id is None:
                self._send_json({"error": "report_id is required"}, 400)
                return

            conn   = _db_connect()
            report = conn.execute(
                "SELECT * FROM osint_reports WHERE id=?", (int(report_id),)
            ).fetchone()
            leads_rows = conn.execute(
                "SELECT * FROM osint_leads WHERE report_id=? ORDER BY score DESC", (int(report_id),)
            ).fetchall()
            conn.close()

            if not report:
                self._send_json({"error": "report not found"}, 404)
                return

            report_dict = dict(report)
            try:
                raw_payload = json.loads(report_dict.get("payload") or "{}")
            except Exception:
                raw_payload = {}

            leads_list = [dict(r) for r in leads_rows]
            emails_found  = len(raw_payload.get("emails", []))
            phones_found  = len(raw_payload.get("phones", []))
            socials_found = len(raw_payload.get("socials", []))

            # Log this as a paid export
            _insert_report(
                report_dict.get("target", ""),
                "paid_export",
                "complete",
                {"source_report_id": report_id},
            )

            result = {
                "title":   "OSINT Report",
                "target":  report_dict.get("target", ""),
                "type":    report_dict.get("report_type", ""),
                "created": report_dict.get("created_at"),
                "summary": {
                    "emails_found":  emails_found,
                    "phones_found":  phones_found,
                    "socials_found": socials_found,
                    "leads_count":   len(leads_list),
                },
                "leads":    leads_list,
                "raw_data": raw_payload,
            }
            self._send_json(result)
            return

        self._send_json({"error": "not found"}, 404)


# ── entrypoint ────────────────────────────────────────────────────────────────

_running = True


def _sigterm(sig, frame):
    global _running
    log.info("Received signal %s — shutting down", sig)
    _running = False


def main():
    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    _db_init()

    server = HTTPServer(("0.0.0.0", PORT), OSINTHandler)
    log.info("fm-osintaas running on port %d | db=%s", PORT, DB)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        log.info("fm-osintaas stopped")


if __name__ == "__main__":
    main()
