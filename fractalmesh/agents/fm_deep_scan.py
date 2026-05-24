#!/usr/bin/env python3
"""
fm_deep_scan.py — Deep Scan Agent for FractalMesh OMEGA Titan (Port 7834)
Comprehensive domain/IP/email/keyword intelligence scanning.
All credentials sourced from ~/.secrets/fractal.env at runtime.
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import json
import time
import re
import socket
import sqlite3
import logging
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from html.parser import HTMLParser

# ── vault ──────────────────────────────────────────────────────────────────────
_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _line in _vault.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

# ── config ─────────────────────────────────────────────────────────────────────
PORT                   = int(os.getenv("DEEP_SCAN_PORT", "7834"))
ROOT                   = Path(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")))
DB                     = ROOT / "database" / "sovereign.db"
LOG                    = ROOT / "logs" / "deep_scan.log"
GOOGLE_CSE_API_KEY     = os.getenv("GOOGLE_CSE_API_KEY", "")
GOOGLE_CSE_ID          = os.getenv("GOOGLE_CSE_ID", "")
CRAWLBASE_NORMAL_TOKEN = os.getenv("CRAWLBASE_NORMAL_TOKEN", "")
CRAWLBASE_JS_TOKEN     = os.getenv("CRAWLBASE_JS_TOKEN", "")
WIGLE_API_NAME         = os.getenv("WIGLE_API_NAME", "")
WIGLE_API_TOKEN        = os.getenv("WIGLE_API_TOKEN", "")
SHODAN_API_KEY         = os.getenv("SHODAN_API_KEY", "")
ALCHEMY_API_KEY        = os.getenv("ALCHEMY_API_KEY", "")

ROOT.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)
DB.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [DEEP_SCAN] %(message)s",
    handlers=[logging.FileHandler(LOG), logging.StreamHandler()],
)
log = logging.getLogger("deep_scan")

# ── database ───────────────────────────────────────────────────────────────────

def _db_conn():
    conn = sqlite3.connect(str(DB), timeout=15)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def _db_init():
    conn = _db_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS deep_scans (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            target      TEXT,
            scan_type   TEXT,
            depth       TEXT,
            status      TEXT,
            findings    TEXT,
            risk_score  REAL,
            created_at  REAL,
            finished_at REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scan_findings (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id    INTEGER,
            category   TEXT,
            severity   TEXT,
            title      TEXT,
            detail     TEXT,
            evidence   TEXT,
            created_at REAL
        )
    """)
    conn.commit()
    conn.close()


def _scan_create(target: str, scan_type: str, depth: str) -> int:
    conn = _db_conn()
    cur = conn.execute(
        "INSERT INTO deep_scans (target, scan_type, depth, status, created_at) VALUES (?,?,?,?,?)",
        (target, scan_type, depth, "running", time.time()),
    )
    scan_id = cur.lastrowid
    conn.commit()
    conn.close()
    return scan_id


def _scan_finish(scan_id: int, findings: list, risk_score: float):
    conn = _db_conn()
    now = time.time()
    conn.execute(
        "UPDATE deep_scans SET status=?, findings=?, risk_score=?, finished_at=? WHERE id=?",
        ("completed", json.dumps(findings), risk_score, now, scan_id),
    )
    for f in findings:
        conn.execute(
            "INSERT INTO scan_findings (scan_id, category, severity, title, detail, evidence, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                scan_id,
                f.get("category", ""),
                f.get("severity", "info"),
                f.get("title", ""),
                f.get("detail", ""),
                f.get("evidence", ""),
                now,
            ),
        )
    conn.commit()
    conn.close()


def _scan_error(scan_id: int, msg: str):
    conn = _db_conn()
    conn.execute(
        "UPDATE deep_scans SET status=?, finished_at=? WHERE id=?",
        ("error", time.time(), scan_id),
    )
    conn.commit()
    conn.close()


# ── helpers ────────────────────────────────────────────────────────────────────

def _dns_resolve(domain: str) -> dict:
    result = {"A": [], "MX": [], "TXT": [], "NS": [], "CNAME": []}
    try:
        info = socket.getaddrinfo(domain, None)
        result["A"] = list({r[4][0] for r in info if r[0] == socket.AF_INET})
    except Exception:
        pass
    try:
        _, aliases, addresses = socket.gethostbyname_ex(domain)
        result["CNAME"] = [a for a in aliases if a and a != domain]
        for addr in addresses:
            if addr not in result["A"]:
                result["A"].append(addr)
    except Exception:
        pass
    # MX / TXT / NS via nslookup-style approach using urllib to a DoH endpoint
    for rtype in ("MX", "TXT", "NS"):
        try:
            url = (
                f"https://dns.google/resolve?name={urllib.parse.quote(domain)}&type={rtype}"
            )
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = json.loads(resp.read().decode())
            answers = data.get("Answer", [])
            for ans in answers:
                val = ans.get("data", "").strip().strip('"')
                if val and val not in result[rtype]:
                    result[rtype].append(val)
        except Exception:
            pass
    return result


def _http_probe(url: str, timeout: int = 5):
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; FractalMesh/1.0)",
                "Accept": "text/html,application/xhtml+xml,*/*",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            headers = dict(resp.headers)
            body = resp.read(65536).decode("utf-8", errors="replace")
            return (status, headers, body)
    except urllib.error.HTTPError as e:
        try:
            headers = dict(e.headers)
            body = e.read(4096).decode("utf-8", errors="replace")
            return (e.code, headers, body)
        except Exception:
            return (e.code, {}, "")
    except Exception:
        return None


def _google_cse(query: str, num: int = 10) -> list:
    if not GOOGLE_CSE_API_KEY or not GOOGLE_CSE_ID:
        return []
    params = urllib.parse.urlencode({
        "key": GOOGLE_CSE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "num": min(num, 10),
    })
    url = f"https://www.googleapis.com/customsearch/v1?{params}"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        items = data.get("items", [])
        return [
            {
                "link": item.get("link", ""),
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
            }
            for item in items
        ]
    except Exception as exc:
        log.warning("Google CSE error for %r: %s", query, exc)
        return []


def _crawlbase_get(url: str, js: bool = False) -> str:
    token = CRAWLBASE_JS_TOKEN if js else CRAWLBASE_NORMAL_TOKEN
    if not token:
        # fall back to direct fetch
        result = _http_probe(url, timeout=10)
        return result[2] if result else ""
    params = urllib.parse.urlencode({"token": token, "url": url})
    api_url = f"https://api.crawlbase.com/?{params}"
    try:
        req = urllib.request.Request(
            api_url,
            headers={"User-Agent": "FractalMesh/1.0"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read(524288).decode("utf-8", errors="replace")
    except Exception as exc:
        log.warning("Crawlbase error for %r: %s", url, exc)
        return ""


def _risk_score(findings_list: list) -> float:
    score = 10.0
    for f in findings_list:
        sev = f.get("severity", "info")
        if sev == "critical":
            score -= 2.0
        elif sev == "high":
            score -= 1.5
        elif sev == "medium":
            score -= 1.0
        elif sev == "low":
            score -= 0.5
    return max(0.0, round(score, 2))


class _LinkExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.forms = []
        self.meta = {}
        self._in_form = False
        self._form_action = ""

    def handle_starttag(self, tag, attrs):
        attrs_d = dict(attrs)
        if tag == "a":
            href = attrs_d.get("href", "")
            if href:
                self.links.append(href)
        elif tag == "form":
            self._in_form = True
            self._form_action = attrs_d.get("action", "")
        elif tag == "meta":
            name = attrs_d.get("name") or attrs_d.get("property") or ""
            content = attrs_d.get("content", "")
            if name:
                self.meta[name.lower()] = content

    def handle_endtag(self, tag):
        if tag == "form" and self._in_form:
            self.forms.append(self._form_action)
            self._in_form = False
            self._form_action = ""


def _extract_subdomains(results: list, base_domain: str) -> list:
    subs = set()
    pattern = re.compile(
        r"(?:https?://)?([a-zA-Z0-9._-]+\." + re.escape(base_domain) + r")"
    )
    for item in results:
        text = item.get("link", "") + " " + item.get("snippet", "")
        for match in pattern.findall(text):
            sub = match.lower()
            if sub != base_domain:
                subs.add(sub)
    return list(subs)


def _detect_tech_stack(body: str) -> list:
    techs = []
    checks = [
        ("WordPress", [r"wp-content", r"wp-includes", r"/wp-json/"]),
        ("Drupal", [r"Drupal\.settings", r"/sites/default/files", r"drupal\.js"]),
        ("Shopify", [r"cdn\.shopify\.com", r"Shopify\.theme"]),
        ("React", [r"__REACT_DEVTOOLS", r"react\.production\.min\.js", r"data-reactroot"]),
        ("Angular", [r"ng-version=", r"angular\.min\.js", r"ng-app="]),
        ("Next.js", [r"__NEXT_DATA__", r"/_next/static"]),
        ("jQuery", [r"jquery\.min\.js", r"jquery-\d"]),
        ("Bootstrap", [r"bootstrap\.min\.css", r"bootstrap\.bundle"]),
    ]
    for name, patterns in checks:
        for pat in patterns:
            if re.search(pat, body, re.IGNORECASE):
                techs.append(name)
                break
    return list(set(techs))


# ── scan logic ─────────────────────────────────────────────────────────────────

def _run_domain_scan(target: str, depth: str) -> tuple:
    findings = []
    summary = {}

    # 1. DNS enumeration
    dns = _dns_resolve(target)
    summary["dns"] = dns
    if not dns["A"]:
        findings.append({
            "category": "dns",
            "severity": "high",
            "title": "No A records found",
            "detail": f"Domain {target} did not resolve to any IPv4 address.",
            "evidence": "",
        })

    # Check SPF / DKIM / DMARC in TXT records
    txt_combined = " ".join(dns.get("TXT", []))
    if "v=spf1" not in txt_combined:
        findings.append({
            "category": "email_security",
            "severity": "medium",
            "title": "Missing SPF record",
            "detail": "No SPF TXT record detected. Domain may be used for email spoofing.",
            "evidence": "",
        })
    if "_dmarc" not in txt_combined.lower():
        findings.append({
            "category": "email_security",
            "severity": "medium",
            "title": "Missing DMARC record",
            "detail": "No DMARC policy detected.",
            "evidence": "",
        })
    dkim_found = any("dkim" in t.lower() for t in dns.get("TXT", []))
    if not dkim_found:
        findings.append({
            "category": "email_security",
            "severity": "low",
            "title": "DKIM not detected in TXT records",
            "detail": "DKIM selector TXT records were not found via standard enumeration.",
            "evidence": "",
        })

    # 2. HTTP headers
    has_https = False
    has_hsts = False
    has_csp = False
    has_xframe = False
    server_info = {}

    for scheme in ("http", "https"):
        url = f"{scheme}://{target}"
        probe = _http_probe(url, timeout=6)
        if probe is None:
            findings.append({
                "category": "http",
                "severity": "info",
                "title": f"{scheme.upper()} unreachable",
                "detail": f"Could not connect to {url}.",
                "evidence": "",
            })
            continue
        status, headers, body = probe
        h_lower = {k.lower(): v for k, v in headers.items()}
        if scheme == "https":
            has_https = True
        if "strict-transport-security" in h_lower:
            has_hsts = True
        if "content-security-policy" in h_lower:
            has_csp = True
        if "x-frame-options" in h_lower:
            has_xframe = True
        for banner_hdr in ("server", "x-powered-by"):
            if banner_hdr in h_lower:
                server_info[banner_hdr] = h_lower[banner_hdr]
                findings.append({
                    "category": "http_headers",
                    "severity": "low",
                    "title": f"Server banner disclosure: {banner_hdr}",
                    "detail": f"{banner_hdr}: {h_lower[banner_hdr]}",
                    "evidence": h_lower[banner_hdr],
                })

    summary["server_info"] = server_info

    if not has_https:
        findings.append({
            "category": "tls",
            "severity": "critical",
            "title": "HTTPS not available",
            "detail": "The target does not respond over HTTPS.",
            "evidence": "",
        })
    if not has_hsts:
        findings.append({
            "category": "http_headers",
            "severity": "high",
            "title": "Missing Strict-Transport-Security header",
            "detail": "HSTS not set; browsers may allow downgrade attacks.",
            "evidence": "",
        })
    if not has_csp:
        findings.append({
            "category": "http_headers",
            "severity": "medium",
            "title": "Missing Content-Security-Policy header",
            "detail": "No CSP header detected; XSS attack surface is elevated.",
            "evidence": "",
        })
    if not has_xframe:
        findings.append({
            "category": "http_headers",
            "severity": "medium",
            "title": "Missing X-Frame-Options header",
            "detail": "No clickjacking protection header present.",
            "evidence": "",
        })

    # 3. Google CSE recon
    indexed_pages = []
    exposed_docs = []
    leak_results = []

    if GOOGLE_CSE_API_KEY and GOOGLE_CSE_ID:
        indexed_pages = _google_cse(f"site:{target}", num=10)
        summary["indexed_pages_count"] = len(indexed_pages)

        exposed_docs = _google_cse(
            f"site:{target} filetype:pdf OR filetype:xlsx OR filetype:docx", num=10
        )
        for doc in exposed_docs:
            findings.append({
                "category": "exposed_data",
                "severity": "medium",
                "title": f"Exposed document: {doc['title'][:80]}",
                "detail": doc.get("snippet", ""),
                "evidence": doc.get("link", ""),
            })

        leak_results = _google_cse(
            f'"{target}" password OR credentials OR api_key', num=10
        )
        if leak_results:
            findings.append({
                "category": "credential_leak",
                "severity": "critical",
                "title": "Potential credential exposure detected via Google",
                "detail": f"{len(leak_results)} results mentioning credentials/passwords alongside the domain.",
                "evidence": json.dumps([r["link"] for r in leak_results[:5]]),
            })

    # 4. Crawlbase page scan (standard + deep)
    tech_stack = []
    if depth in ("standard", "deep"):
        home_body = _crawlbase_get(f"https://{target}", js=False)
        if not home_body:
            home_body = _crawlbase_get(f"http://{target}", js=False)

        if home_body:
            tech_stack = _detect_tech_stack(home_body)
            summary["tech_stack"] = tech_stack

            parser = _LinkExtractor()
            parser.feed(home_body)
            external_links = [l for l in parser.links if l.startswith("http") and target not in l]
            summary["external_links_count"] = len(external_links)
            summary["forms"] = parser.forms
            summary["meta_tags"] = dict(list(parser.meta.items())[:10])

            if parser.forms:
                for action in parser.forms:
                    if action and action.startswith("http") and target not in action:
                        findings.append({
                            "category": "form_security",
                            "severity": "high",
                            "title": "Form submits to external domain",
                            "detail": f"Form action points to: {action}",
                            "evidence": action,
                        })

        robots = _crawlbase_get(f"https://{target}/robots.txt", js=False)
        if robots:
            disallowed = [l for l in robots.splitlines() if l.lower().startswith("disallow")]
            summary["robots_disallowed_count"] = len(disallowed)

        sitemap = _crawlbase_get(f"https://{target}/sitemap.xml", js=False)
        if sitemap:
            sm_urls = re.findall(r"<loc>(.*?)</loc>", sitemap)
            summary["sitemap_urls_count"] = len(sm_urls)

    # 5. Subdomain discovery (deep)
    subdomains = []
    if depth == "deep" and GOOGLE_CSE_API_KEY and GOOGLE_CSE_ID:
        sub_results = _google_cse(f"site:*.{target}", num=10)
        subdomains = _extract_subdomains(sub_results, target)
        if subdomains:
            summary["subdomains"] = subdomains
            findings.append({
                "category": "recon",
                "severity": "info",
                "title": f"{len(subdomains)} subdomain(s) discovered",
                "detail": "Subdomains found via Google CSE enumeration.",
                "evidence": json.dumps(subdomains),
            })

    summary["tech_stack"] = tech_stack
    risk = _risk_score(findings)
    return findings, risk, summary


def _run_ip_scan(target: str, depth: str) -> tuple:
    findings = []
    result = {"hostname": "", "geo": {}, "shodan": {}, "open_ports": []}

    # 1. Reverse DNS
    try:
        hostname, aliases, _ = socket.gethostbyaddr(target)
        result["hostname"] = hostname
        findings.append({
            "category": "dns",
            "severity": "info",
            "title": "Reverse DNS resolved",
            "detail": f"{target} → {hostname}",
            "evidence": hostname,
        })
    except Exception:
        result["hostname"] = ""

    # 2. Shodan (optional)
    if SHODAN_API_KEY:
        try:
            url = f"https://api.shodan.io/shodan/host/{target}?key={SHODAN_API_KEY}"
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            result["shodan"] = {
                "ports": data.get("ports", []),
                "org": data.get("org", ""),
                "os": data.get("os", ""),
                "vulns": list(data.get("vulns", {}).keys()),
                "hostnames": data.get("hostnames", []),
            }
            vulns = result["shodan"].get("vulns", [])
            if vulns:
                findings.append({
                    "category": "vulnerabilities",
                    "severity": "critical",
                    "title": f"Shodan reports {len(vulns)} CVE(s)",
                    "detail": "Known vulnerabilities detected by Shodan.",
                    "evidence": json.dumps(vulns[:10]),
                })
        except Exception as exc:
            log.warning("Shodan lookup failed for %s: %s", target, exc)

    # 3. HTTP probe on common ports
    common_ports = [80, 443, 8080, 8443, 3000, 8000]
    open_ports = []
    for port in common_ports:
        try:
            with socket.create_connection((target, port), timeout=2):
                open_ports.append(port)
                findings.append({
                    "category": "open_ports",
                    "severity": "info",
                    "title": f"Port {port} open",
                    "detail": f"TCP connection established to {target}:{port}.",
                    "evidence": f"{target}:{port}",
                })
        except Exception:
            pass
    result["open_ports"] = open_ports

    # 4. Geo lookup (ip-api.com, no key required)
    try:
        url = f"http://ip-api.com/json/{target}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            geo = json.loads(resp.read().decode())
        result["geo"] = {
            "country": geo.get("country", ""),
            "region": geo.get("regionName", ""),
            "city": geo.get("city", ""),
            "isp": geo.get("isp", ""),
            "org": geo.get("org", ""),
            "lat": geo.get("lat"),
            "lon": geo.get("lon"),
        }
    except Exception as exc:
        log.warning("Geo lookup failed for %s: %s", target, exc)

    risk = _risk_score(findings)
    return findings, risk, result


def _run_email_scan(target: str) -> tuple:
    findings = []
    result = {
        "email": target,
        "exists": None,
        "domain_has_mx": False,
        "smtp_banner": "",
        "exposure_results": [],
    }

    if "@" not in target:
        return findings, 5.0, result

    _, domain = target.rsplit("@", 1)

    # 1. MX records
    dns = _dns_resolve(domain)
    mx_records = dns.get("MX", [])
    result["domain_has_mx"] = bool(mx_records)
    if not mx_records:
        findings.append({
            "category": "email",
            "severity": "high",
            "title": "No MX records found",
            "detail": f"Domain {domain} has no MX records; cannot receive mail.",
            "evidence": "",
        })
        risk = _risk_score(findings)
        return findings, risk, result

    # Pick first MX host (strip priority prefix like "10 mail.example.com.")
    mx_host = mx_records[0].split()[-1].rstrip(".") if mx_records else domain

    # 2. SMTP banner grab
    smtp_banner = ""
    smtp_sock = None
    try:
        smtp_sock = socket.create_connection((mx_host, 25), timeout=5)
        banner_data = smtp_sock.recv(1024).decode("utf-8", errors="replace").strip()
        smtp_banner = banner_data
        result["smtp_banner"] = smtp_banner

        # 3. EHLO → MAIL FROM → RCPT TO
        smtp_sock.sendall(b"EHLO deep-scan.fractalmesh.local\r\n")
        time.sleep(0.3)
        smtp_sock.recv(4096)

        smtp_sock.sendall(b"MAIL FROM:<scan@deep-scan.fractalmesh.local>\r\n")
        time.sleep(0.3)
        smtp_sock.recv(4096)

        rcpt_cmd = f"RCPT TO:<{target}>\r\n".encode()
        smtp_sock.sendall(rcpt_cmd)
        time.sleep(0.5)
        rcpt_resp = smtp_sock.recv(1024).decode("utf-8", errors="replace")
        if rcpt_resp.startswith("250"):
            result["exists"] = True
            findings.append({
                "category": "email",
                "severity": "info",
                "title": "Email address appears to exist (SMTP 250)",
                "detail": f"RCPT TO accepted by {mx_host}.",
                "evidence": rcpt_resp[:120],
            })
        elif rcpt_resp.startswith("550"):
            result["exists"] = False
            findings.append({
                "category": "email",
                "severity": "info",
                "title": "Email address rejected (SMTP 550)",
                "detail": f"RCPT TO rejected by {mx_host}.",
                "evidence": rcpt_resp[:120],
            })
        else:
            result["exists"] = None

        smtp_sock.sendall(b"QUIT\r\n")
    except Exception as exc:
        log.info("SMTP probe for %s failed: %s", target, exc)
        result["smtp_banner"] = smtp_banner
    finally:
        if smtp_sock:
            try:
                smtp_sock.close()
            except Exception:
                pass

    # 4. Google CSE breach check
    exposure = _google_cse(
        f'"{target}" site:pastebin.com OR site:github.com', num=10
    )
    result["exposure_results"] = exposure
    if exposure:
        findings.append({
            "category": "data_exposure",
            "severity": "high",
            "title": f"Email found in {len(exposure)} public paste/repo result(s)",
            "detail": "Email address appears in public breach sources.",
            "evidence": json.dumps([r["link"] for r in exposure[:5]]),
        })

    risk = _risk_score(findings)
    return findings, risk, result


def _run_keyword_scan(keywords: list, platforms: list, depth: str) -> tuple:
    findings = []
    aggregated = []
    email_pattern = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
    phone_pattern = re.compile(r"(\+?\d[\d\s\-().]{7,}\d)")

    platform_site_map = {
        "pastebin": "site:pastebin.com",
        "github":   "site:github.com",
        "reddit":   "site:reddit.com",
        "twitter":  "site:twitter.com OR site:x.com",
        "linkedin": "site:linkedin.com",
        "google":   "",
    }

    for kw in keywords:
        for platform in platforms:
            site_filter = platform_site_map.get(platform.lower(), "")
            query = f'"{kw}" {site_filter}'.strip()
            results = _google_cse(query, num=10)
            emails_found = []
            phones_found = []
            for r in results:
                text = r.get("snippet", "") + " " + r.get("title", "")
                emails_found.extend(email_pattern.findall(text))
                phones_found.extend(phone_pattern.findall(text))

            entry = {
                "keyword": kw,
                "platform": platform,
                "query": query,
                "results": results,
                "emails_found": list(set(emails_found)),
                "phones_found": list(set(phones_found)),
            }
            aggregated.append(entry)

            if emails_found:
                findings.append({
                    "category": "pii",
                    "severity": "high",
                    "title": f"Email(s) found for keyword '{kw}' on {platform}",
                    "detail": f"{len(set(emails_found))} unique email(s) in search snippets.",
                    "evidence": json.dumps(list(set(emails_found))[:10]),
                })
            if results:
                findings.append({
                    "category": "keyword_hit",
                    "severity": "medium",
                    "title": f"Keyword '{kw}' found on {platform}",
                    "detail": f"{len(results)} result(s) returned.",
                    "evidence": json.dumps([r["link"] for r in results[:5]]),
                })

    risk = _risk_score(findings)
    return findings, risk, aggregated


# ── HTTP server ────────────────────────────────────────────────────────────────

class DeepScanHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # silence default access log
        log.debug("HTTP %s", fmt % args)

    def _send(self, code: int, body: dict):
        data = json.dumps(body, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode())
        except Exception:
            return {}

    def _parse_path(self):
        parsed = urllib.parse.urlparse(self.path)
        return parsed.path, urllib.parse.parse_qs(parsed.query)

    # ── GET ────────────────────────────────────────────────────────────────────

    def do_GET(self):
        path, qs = self._parse_path()

        if path == "/health":
            self._send(200, {"status": "ok", "service": "fm-deep-scan", "port": PORT})
            return

        if path == "/scans":
            try:
                conn = _db_conn()
                rows = conn.execute(
                    "SELECT id, target, scan_type, status, risk_score, created_at "
                    "FROM deep_scans ORDER BY created_at DESC LIMIT 200"
                ).fetchall()
                conn.close()
                self._send(200, {"scans": [dict(r) for r in rows]})
            except Exception as exc:
                self._send(500, {"error": str(exc)})
            return

        scan_match = re.match(r"^/scans/(\d+)$", path)
        if scan_match:
            scan_id = int(scan_match.group(1))
            try:
                conn = _db_conn()
                row = conn.execute(
                    "SELECT * FROM deep_scans WHERE id=?", (scan_id,)
                ).fetchone()
                if not row:
                    conn.close()
                    self._send(404, {"error": "scan not found"})
                    return
                scan = dict(row)
                findings = conn.execute(
                    "SELECT * FROM scan_findings WHERE scan_id=? ORDER BY id",
                    (scan_id,),
                ).fetchall()
                conn.close()
                scan["findings_detail"] = [dict(f) for f in findings]
                self._send(200, scan)
            except Exception as exc:
                self._send(500, {"error": str(exc)})
            return

        if path == "/findings":
            try:
                severity   = qs.get("severity", [None])[0]
                scan_type  = qs.get("scan_type", [None])[0]
                limit      = int(qs.get("limit", ["50"])[0])
                clauses, params = [], []
                if severity:
                    clauses.append("sf.severity = ?")
                    params.append(severity)
                if scan_type:
                    clauses.append("ds.scan_type = ?")
                    params.append(scan_type)
                where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
                params.append(limit)
                conn = _db_conn()
                rows = conn.execute(
                    f"SELECT sf.*, ds.scan_type, ds.target FROM scan_findings sf "
                    f"JOIN deep_scans ds ON ds.id = sf.scan_id "
                    f"{where} ORDER BY sf.created_at DESC LIMIT ?",
                    params,
                ).fetchall()
                conn.close()
                self._send(200, {"findings": [dict(r) for r in rows]})
            except Exception as exc:
                self._send(500, {"error": str(exc)})
            return

        if path == "/analytics":
            try:
                conn = _db_conn()
                total = conn.execute("SELECT COUNT(*) FROM deep_scans").fetchone()[0]
                by_type = {}
                for r in conn.execute(
                    "SELECT scan_type, COUNT(*) cnt FROM deep_scans GROUP BY scan_type"
                ).fetchall():
                    by_type[r[0]] = r[1]
                avg_risk = conn.execute(
                    "SELECT AVG(risk_score) FROM deep_scans WHERE status='completed'"
                ).fetchone()[0]
                by_severity = {}
                for r in conn.execute(
                    "SELECT severity, COUNT(*) cnt FROM scan_findings GROUP BY severity"
                ).fetchall():
                    by_severity[r[0]] = r[1]
                common_issues = []
                for r in conn.execute(
                    "SELECT title, COUNT(*) cnt FROM scan_findings "
                    "GROUP BY title ORDER BY cnt DESC LIMIT 10"
                ).fetchall():
                    common_issues.append({"title": r[0], "count": r[1]})
                conn.close()
                self._send(200, {
                    "total_scans": total,
                    "scans_by_type": by_type,
                    "avg_risk_score": round(avg_risk or 0.0, 2),
                    "findings_by_severity": by_severity,
                    "most_common_issues": common_issues,
                })
            except Exception as exc:
                self._send(500, {"error": str(exc)})
            return

        self._send(404, {"error": "not found"})

    # ── POST ───────────────────────────────────────────────────────────────────

    def do_POST(self):
        path, _ = self._parse_path()
        body = self._read_body()

        if path == "/scan/domain":
            target = body.get("target", "").strip().lower()
            depth  = body.get("depth", "standard")
            if not target:
                self._send(400, {"error": "target required"})
                return
            if depth not in ("quick", "standard", "deep"):
                depth = "standard"
            scan_id = _scan_create(target, "domain", depth)
            try:
                findings, risk, summary = _run_domain_scan(target, depth)
                _scan_finish(scan_id, findings, risk)
                self._send(200, {
                    "scan_id": scan_id,
                    "target": target,
                    "risk_score": risk,
                    "findings_count": len(findings),
                    "findings": findings,
                    "summary": summary,
                })
            except Exception as exc:
                _scan_error(scan_id, str(exc))
                log.exception("domain scan error for %s", target)
                self._send(500, {"error": str(exc)})
            return

        if path == "/scan/ip":
            target = body.get("target", "").strip()
            depth  = body.get("depth", "quick")
            if not target:
                self._send(400, {"error": "target required"})
                return
            scan_id = _scan_create(target, "ip", depth)
            try:
                findings, risk, result = _run_ip_scan(target, depth)
                _scan_finish(scan_id, findings, risk)
                self._send(200, {
                    "scan_id": scan_id,
                    "target": target,
                    "risk_score": risk,
                    "open_ports": result["open_ports"],
                    "hostname": result["hostname"],
                    "geo": result["geo"],
                    "shodan": result["shodan"],
                    "findings": findings,
                })
            except Exception as exc:
                _scan_error(scan_id, str(exc))
                log.exception("ip scan error for %s", target)
                self._send(500, {"error": str(exc)})
            return

        if path == "/scan/email":
            target = body.get("target", "").strip()
            if not target or "@" not in target:
                self._send(400, {"error": "valid email required"})
                return
            scan_id = _scan_create(target, "email", "standard")
            try:
                findings, risk, result = _run_email_scan(target)
                _scan_finish(scan_id, findings, risk)
                self._send(200, {
                    "scan_id": scan_id,
                    "email": target,
                    "exists": result["exists"],
                    "domain_has_mx": result["domain_has_mx"],
                    "smtp_banner": result["smtp_banner"],
                    "exposure_results": result["exposure_results"],
                    "risk_score": risk,
                    "findings": findings,
                })
            except Exception as exc:
                _scan_error(scan_id, str(exc))
                log.exception("email scan error for %s", target)
                self._send(500, {"error": str(exc)})
            return

        if path == "/scan/keyword":
            keywords  = body.get("keywords", [])
            platforms = body.get("platforms", ["google"])
            depth     = body.get("depth", "standard")
            if not keywords or not isinstance(keywords, list):
                self._send(400, {"error": "keywords list required"})
                return
            target_repr = "|".join(keywords[:3])
            scan_id = _scan_create(target_repr, "keyword", depth)
            try:
                findings, risk, aggregated = _run_keyword_scan(keywords, platforms, depth)
                _scan_finish(scan_id, findings, risk)
                self._send(200, {
                    "scan_id": scan_id,
                    "keywords": keywords,
                    "platforms": platforms,
                    "risk_score": risk,
                    "findings_count": len(findings),
                    "findings": findings,
                    "results": aggregated,
                })
            except Exception as exc:
                _scan_error(scan_id, str(exc))
                log.exception("keyword scan error")
                self._send(500, {"error": str(exc)})
            return

        if path == "/scan/bulk":
            targets   = body.get("targets", [])
            scan_type = body.get("scan_type", "domain")
            depth     = body.get("depth", "quick")
            if not targets or not isinstance(targets, list):
                self._send(400, {"error": "targets list required"})
                return
            results = []
            for t in targets:
                t = str(t).strip()
                if not t:
                    continue
                s_id = _scan_create(t, scan_type, depth)
                try:
                    if scan_type == "domain":
                        findings, risk, summary = _run_domain_scan(t, depth)
                        _scan_finish(s_id, findings, risk)
                        results.append({
                            "scan_id": s_id, "target": t, "risk_score": risk,
                            "findings_count": len(findings), "summary": summary,
                        })
                    elif scan_type == "ip":
                        findings, risk, res = _run_ip_scan(t, depth)
                        _scan_finish(s_id, findings, risk)
                        results.append({
                            "scan_id": s_id, "target": t, "risk_score": risk,
                            "open_ports": res["open_ports"], "geo": res["geo"],
                        })
                    elif scan_type == "email":
                        findings, risk, res = _run_email_scan(t)
                        _scan_finish(s_id, findings, risk)
                        results.append({
                            "scan_id": s_id, "target": t, "risk_score": risk,
                            "exists": res["exists"],
                        })
                    else:
                        _scan_error(s_id, "unsupported scan_type for bulk")
                        results.append({"target": t, "error": "unsupported scan_type"})
                except Exception as exc:
                    _scan_error(s_id, str(exc))
                    results.append({"scan_id": s_id, "target": t, "error": str(exc)})
            self._send(200, {"total": len(results), "results": results})
            return

        self._send(404, {"error": "not found"})


# ── entrypoint ─────────────────────────────────────────────────────────────────

def main():
    _db_init()
    server = HTTPServer(("0.0.0.0", PORT), DeepScanHandler)
    log.info("Deep Scan Agent listening on port %d", PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
