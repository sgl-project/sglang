#!/usr/bin/env python3
"""
fm_crawlbase.py — Crawlbase (ProxyCrawl) Web Scraping Agent (Port 7824)
Scraping, screenshots, Google/LinkedIn, batch scraping, lead generation.
All credentials sourced from ~/.secrets/fractal.env at runtime.
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import json
import time
import signal
import sqlite3
import logging
import re
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Optional
from html.parser import HTMLParser

# ── vault ─────────────────────────────────────────────────────────────────────
_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _line in _vault.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

# ── config ────────────────────────────────────────────────────────────────────
PORT                    = int(os.getenv("CRAWLBASE_PORT", "7824"))
ROOT                    = Path(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")))
DB                      = ROOT / "database" / "sovereign.db"
LOG                     = ROOT / "logs" / "crawlbase.log"
CRAWLBASE_NORMAL_TOKEN  = os.getenv("CRAWLBASE_NORMAL_TOKEN", "")
CRAWLBASE_JS_TOKEN      = os.getenv("CRAWLBASE_JS_TOKEN", "")
CB_BASE                 = "https://api.crawlbase.com"

ROOT.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)
DB.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [CRAWLBASE] %(message)s",
    handlers=[logging.FileHandler(LOG), logging.StreamHandler()],
)
log = logging.getLogger("crawlbase")

# ── database ──────────────────────────────────────────────────────────────────

def _db_init():
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS crawlbase_jobs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            url         TEXT,
            status      TEXT,
            tokens_used TEXT,
            content_len INTEGER,
            ts          DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def _log_job(url: str, status: str, tokens_used: str, content_len: int):
    try:
        conn = sqlite3.connect(DB, timeout=5)
        conn.execute(
            "INSERT INTO crawlbase_jobs (url, status, tokens_used, content_len) VALUES (?, ?, ?, ?)",
            (url[:1024], status, tokens_used, content_len),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning("db_log error: %s", e)

def _recent_jobs(limit: int = 50) -> list:
    try:
        conn = sqlite3.connect(DB, timeout=5)
        rows = conn.execute(
            "SELECT id, url, status, tokens_used, content_len, ts FROM crawlbase_jobs "
            "ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [
            {"id": r[0], "url": r[1], "status": r[2],
             "tokens_used": r[3], "content_len": r[4], "ts": r[5]}
            for r in rows
        ]
    except Exception:
        return []

# ── HTML parsing helpers ──────────────────────────────────────────────────────

class _TextExtractor(HTMLParser):
    """Strip all HTML tags and collect visible text and links."""
    def __init__(self):
        super().__init__()
        self._text_parts: list = []
        self._links: list      = []
        self._title: str       = ""
        self._in_title: bool   = False
        self._skip_tags        = {"script", "style", "noscript", "head"}
        self._skip_depth: int  = 0

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in self._skip_tags:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True
        if tag == "a":
            href = dict(attrs).get("href", "")
            if href and href.startswith("http"):
                self._links.append(href)

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in self._skip_tags:
            self._skip_depth = max(0, self._skip_depth - 1)
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._skip_depth:
            return
        stripped = data.strip()
        if stripped:
            if self._in_title:
                self._title = stripped
            else:
                self._text_parts.append(stripped)

    @property
    def text(self) -> str:
        return " ".join(self._text_parts)

    @property
    def links(self) -> list:
        return list(dict.fromkeys(self._links))[:50]  # deduplicate, cap at 50

    @property
    def title(self) -> str:
        return self._title


def _parse_html(html: str) -> dict:
    """Extract title, text, and links from raw HTML."""
    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception:
        pass
    return {
        "title": parser.title,
        "body_text": parser.text[:8000],
        "links": parser.links,
    }

# ── Crawlbase API helper ──────────────────────────────────────────────────────

def _cb_get(params: dict, timeout: int = 30) -> dict:
    """
    GET https://api.crawlbase.com/?param=value&...
    Returns dict with: {status, pc_status, body, headers_raw, error}
    """
    qs  = urllib.parse.urlencode(params)
    url = f"{CB_BASE}/?{qs}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "FractalMesh-Crawlbase/1.0"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw         = r.read()
            pc_status   = r.getheader("original_status", r.getheader("pc_status", ""))
            content_type = r.getheader("Content-Type", "")
            # Try to decode body as UTF-8; fall back to latin-1
            try:
                body = raw.decode("utf-8")
            except UnicodeDecodeError:
                body = raw.decode("latin-1", errors="replace")
            return {
                "http_status":  r.status,
                "pc_status":    pc_status,
                "content_type": content_type,
                "body":         body,
                "body_bytes":   len(raw),
                "error":        None,
            }
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        log.warning("crawlbase HTTP %d: %s", e.code, detail[:120])
        return {"http_status": e.code, "pc_status": "", "body": "",
                "body_bytes": 0, "error": f"http_{e.code}: {detail}"}
    except Exception as e:
        log.error("crawlbase request error: %s", e)
        return {"http_status": 0, "pc_status": "", "body": "",
                "body_bytes": 0, "error": str(e)}


def _scrape_url(target_url: str, use_js: bool = False,
                autoparse: bool = True, extra_params: Optional[dict] = None) -> dict:
    """Core scrape logic. Selects token, calls Crawlbase, parses HTML."""
    token = CRAWLBASE_JS_TOKEN if use_js else CRAWLBASE_NORMAL_TOKEN
    if not token:
        tok_name = "CRAWLBASE_JS_TOKEN" if use_js else "CRAWLBASE_NORMAL_TOKEN"
        return {"error": f"{tok_name} not configured"}

    params: dict = {
        "token": token,
        "url":   target_url,
    }
    if autoparse:
        params["autoparse"] = "true"
    if use_js:
        params["javascript"] = "true"
    if extra_params:
        params.update(extra_params)

    token_type = "js" if use_js else "normal"
    r = _cb_get(params, timeout=60)

    parsed = {}
    if not r["error"] and r["body"]:
        parsed = _parse_html(r["body"])

    _log_job(
        target_url,
        str(r.get("pc_status") or r.get("http_status", "?")),
        token_type,
        r.get("body_bytes", 0),
    )

    return {
        "url":          target_url,
        "status":       r.get("pc_status") or r.get("http_status"),
        "token_type":   token_type,
        "title":        parsed.get("title", ""),
        "body_text":    parsed.get("body_text", ""),
        "links":        parsed.get("links", []),
        "content_bytes": r.get("body_bytes", 0),
        "error":        r.get("error"),
    }

# ── query string parser ───────────────────────────────────────────────────────

def _qs(path: str) -> dict:
    if "?" not in path:
        return {}
    return dict(urllib.parse.parse_qsl(path.split("?", 1)[1]))

# ── HTTP handler ──────────────────────────────────────────────────────────────

class CrawlbaseHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _respond(self, code: int, body: Any):
        payload = json.dumps(body, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def _base_path(self) -> str:
        return self.path.split("?")[0]

    # ── GET ───────────────────────────────────────────────────────────────────

    def do_GET(self):
        base = self._base_path()
        qs   = _qs(self.path)

        try:
            if base == "/health":
                token = CRAWLBASE_NORMAL_TOKEN
                if not token:
                    self._respond(200, {
                        "status": "error",
                        "error":  "CRAWLBASE_NORMAL_TOKEN not configured",
                        "normal_token": False,
                        "js_token":     bool(CRAWLBASE_JS_TOKEN),
                    })
                    return
                params = {
                    "token":       token,
                    "url":         "https://example.com",
                    "get_headers": "true",
                }
                r = _cb_get(params, timeout=20)
                self._respond(200, {
                    "status":       "ok" if not r["error"] else "error",
                    "normal_token": bool(CRAWLBASE_NORMAL_TOKEN),
                    "js_token":     bool(CRAWLBASE_JS_TOKEN),
                    "http_status":  r.get("http_status"),
                    "pc_status":    r.get("pc_status"),
                    "error":        r.get("error"),
                })

            elif base == "/jobs":
                rows = _recent_jobs()
                self._respond(200, {"count": len(rows), "jobs": rows})

            elif base == "/leads":
                keyword  = qs.get("keyword", "")
                location = qs.get("location", "")
                if not keyword:
                    self._respond(400, {"error": "keyword param required"})
                    return
                self._handle_leads(keyword, location)

            else:
                self._respond(404, {"error": "not_found", "path": base})

        except Exception as e:
            log.error("GET %s error: %s", base, e)
            self._respond(500, {"error": str(e)})

    def _handle_leads(self, keyword: str, location: str):
        """Scrape Google Maps-style search and LinkedIn for leads."""
        leads  = []
        errors = []

        # Strategy 1: Google search for LinkedIn profiles
        query      = f"{keyword} {location} site:linkedin.com/in" if location else f"{keyword} site:linkedin.com/in"
        google_url = "https://www.google.com/search?q=" + urllib.parse.quote(query)
        g_result   = _scrape_url(google_url, use_js=False, autoparse=True)
        if not g_result.get("error"):
            # Extract LinkedIn profile URLs from links
            linkedin_links = [
                lnk for lnk in g_result.get("links", [])
                if "linkedin.com/in/" in lnk
            ]
            for lnk in linkedin_links[:5]:
                leads.append({
                    "source":   "google_linkedin",
                    "url":      lnk,
                    "keyword":  keyword,
                    "location": location,
                })
        else:
            errors.append({"source": "google", "error": g_result["error"]})

        # Strategy 2: Google Maps search for local businesses
        if location:
            maps_query = urllib.parse.quote(f"{keyword} {location}")
            maps_url   = f"https://www.google.com/maps/search/{maps_query}"
            m_result   = _scrape_url(maps_url, use_js=True, autoparse=False)
            if not m_result.get("error"):
                # Extract business names from text heuristically
                text = m_result.get("body_text", "")
                leads.append({
                    "source":    "google_maps",
                    "keyword":   keyword,
                    "location":  location,
                    "body_text": text[:2000],
                })
            else:
                errors.append({"source": "google_maps", "error": m_result["error"]})

        self._respond(200, {
            "keyword":  keyword,
            "location": location,
            "count":    len(leads),
            "leads":    leads,
            "errors":   errors,
        })

    # ── POST ──────────────────────────────────────────────────────────────────

    def do_POST(self):
        base = self._base_path()
        try:
            data = self._read_body()
        except json.JSONDecodeError:
            self._respond(400, {"error": "invalid_json"})
            return
        except Exception as e:
            self._respond(400, {"error": str(e)})
            return

        try:
            if base == "/scrape":
                self._handle_scrape(data)
            elif base == "/scrape_batch":
                self._handle_scrape_batch(data)
            elif base == "/screenshot":
                self._handle_screenshot(data)
            elif base == "/google":
                self._handle_google(data)
            elif base == "/linkedin":
                self._handle_linkedin(data)
            else:
                self._respond(404, {"error": "unknown_path", "path": base})
        except Exception as e:
            log.error("POST %s error: %s", base, e)
            self._respond(500, {"error": str(e)})

    def _handle_scrape(self, data: dict):
        url    = data.get("url", "")
        use_js = bool(data.get("js", False))
        if not url:
            self._respond(400, {"error": "url required"})
            return
        result = _scrape_url(url, use_js=use_js, autoparse=True)
        code   = 200 if not result.get("error") else 502
        self._respond(code, result)

    def _handle_scrape_batch(self, data: dict):
        urls   = data.get("urls", [])
        use_js = bool(data.get("js", False))
        if not urls or not isinstance(urls, list):
            self._respond(400, {"error": "urls array required"})
            return
        if len(urls) > 20:
            self._respond(400, {"error": "maximum 20 URLs per batch"})
            return

        results = []
        for i, url in enumerate(urls):
            if i > 0:
                time.sleep(1)  # 1 second delay between requests
            result = _scrape_url(str(url), use_js=use_js, autoparse=True)
            results.append(result)
            log.info("batch scrape %d/%d: %s", i + 1, len(urls), url)

        ok_count  = sum(1 for r in results if not r.get("error"))
        err_count = len(results) - ok_count
        self._respond(200, {
            "total":     len(results),
            "success":   ok_count,
            "errors":    err_count,
            "results":   results,
        })

    def _handle_screenshot(self, data: dict):
        url = data.get("url", "")
        if not url:
            self._respond(400, {"error": "url required"})
            return
        if not CRAWLBASE_JS_TOKEN:
            self._respond(400, {"error": "CRAWLBASE_JS_TOKEN not configured"})
            return

        params = {
            "token":      CRAWLBASE_JS_TOKEN,
            "url":        url,
            "screenshot": "true",
        }
        r = _cb_get(params, timeout=45)
        _log_job(url, str(r.get("pc_status") or r.get("http_status")), "js_screenshot",
                 r.get("body_bytes", 0))

        if r.get("error"):
            self._respond(502, {"error": r["error"]})
            return

        import base64
        body_b64 = ""
        if r.get("body"):
            try:
                # Crawlbase returns raw binary for screenshots; encode to base64
                body_b64 = base64.b64encode(r["body"].encode("latin-1")).decode()
            except Exception:
                body_b64 = ""

        self._respond(200, {
            "url":          url,
            "status":       r.get("pc_status") or r.get("http_status"),
            "screenshot_b64": body_b64,
            "content_bytes": r.get("body_bytes", 0),
        })

    def _handle_google(self, data: dict):
        query = data.get("query", "")
        if not query:
            self._respond(400, {"error": "query required"})
            return
        token = CRAWLBASE_NORMAL_TOKEN
        if not token:
            self._respond(400, {"error": "CRAWLBASE_NORMAL_TOKEN not configured"})
            return

        google_url = "https://www.google.com/search?q=" + urllib.parse.quote(query)
        params = {
            "token":      token,
            "url":        google_url,
            "autoparse":  "true",
        }
        r = _cb_get(params, timeout=30)
        _log_job(google_url, str(r.get("pc_status") or r.get("http_status")),
                 "normal_google", r.get("body_bytes", 0))

        if r.get("error"):
            self._respond(502, {"error": r["error"]})
            return

        parsed = _parse_html(r["body"]) if r.get("body") else {}
        self._respond(200, {
            "query":       query,
            "status":      r.get("pc_status") or r.get("http_status"),
            "title":       parsed.get("title", ""),
            "body_text":   parsed.get("body_text", ""),
            "links":       parsed.get("links", []),
            "content_bytes": r.get("body_bytes", 0),
        })

    def _handle_linkedin(self, data: dict):
        linkedin_url = data.get("url", "")
        if not linkedin_url:
            self._respond(400, {"error": "url required"})
            return
        if not CRAWLBASE_JS_TOKEN:
            self._respond(400, {"error": "CRAWLBASE_JS_TOKEN not configured"})
            return

        result = _scrape_url(linkedin_url, use_js=True, autoparse=False)
        code   = 200 if not result.get("error") else 502
        self._respond(code, result)

# ── shutdown ───────────────────────────────────────────────────────────────────

_running = True

def _shutdown(*_):
    global _running
    log.info("shutdown signal — exiting cleanly")
    _running = False

signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT,  _shutdown)

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    _db_init()
    server = HTTPServer(("0.0.0.0", PORT), CrawlbaseHandler)
    log.info("Crawlbase agent listening on port %d", PORT)
    log.info("Normal token: %s | JS token: %s",
             "configured" if CRAWLBASE_NORMAL_TOKEN else "NOT SET",
             "configured" if CRAWLBASE_JS_TOKEN else "NOT SET")
    try:
        while _running:
            server.handle_request()
    finally:
        server.server_close()
        log.info("Crawlbase agent stopped")

if __name__ == "__main__":
    main()
