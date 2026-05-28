#!/usr/bin/env python3
"""
fm_seo_engine.py — FractalMesh OMEGA Titan SEO & Content Optimisation Engine
Port: 7893

SEO analysis and content optimisation platform. Analyse URLs for on-page SEO,
track keyword rankings, generate meta tags, create sitemaps, and get
AI-powered content recommendations via Anthropic claude-haiku-4-5.

Samuel James Hiotis | ABN 56 628 117 363
"""

import hashlib
import hmac
import html
import json
import os
import re
import secrets
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ---------------------------------------------------------------------------
# Vault loading — MUST be before any os.getenv calls
# ---------------------------------------------------------------------------
_ENV_FILE = Path.home() / ".secrets" / "fractal.env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PORT              = int(os.environ.get("SEO_ENGINE_PORT", "7893"))
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ADMIN_SECRET      = os.environ.get("ADMIN_SECRET", "")

ROOT     = Path.home() / "fmsaas"
DB_PATH  = ROOT / "database" / "sovereign.db"
LOG_PATH = ROOT / "logs" / "seo_engine.log"

for _d in (DB_PATH.parent, LOG_PATH.parent):
    _d.mkdir(parents=True, exist_ok=True)

START_TIME = time.time()

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
_log_lock = threading.Lock()

def _log(level: str, msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    entry = f"{ts} [{level.upper()}] seo_engine: {msg}"
    with _log_lock:
        try:
            with open(LOG_PATH, "a") as _f:
                _f.write(entry + "\n")
        except Exception:
            pass
        print(entry, flush=True)

def log_info(msg: str): _log("INFO",  msg)
def log_warn(msg: str): _log("WARN",  msg)
def log_err(msg: str):  _log("ERROR", msg)

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def _db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=15, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def db_init() -> None:
    conn = _db_connect()
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS pages (
            id                  INTEGER PRIMARY KEY,
            page_id             TEXT UNIQUE,
            url                 TEXT UNIQUE,
            title               TEXT,
            meta_description    TEXT,
            h1                  TEXT,
            word_count          INTEGER,
            keyword_density     TEXT    DEFAULT '{}',
            readability_score   REAL,
            seo_score           INTEGER DEFAULT 0,
            issues              TEXT    DEFAULT '[]',
            last_crawled        REAL,
            created_at          REAL,
            updated_at          REAL
        );

        CREATE TABLE IF NOT EXISTS keywords (
            id                  INTEGER PRIMARY KEY,
            kw_id               TEXT UNIQUE,
            keyword             TEXT,
            target_url          TEXT,
            search_volume       INTEGER DEFAULT 0,
            difficulty          INTEGER DEFAULT 0,
            position            INTEGER,
            previous_position   INTEGER,
            position_change     INTEGER DEFAULT 0,
            last_checked        REAL,
            created_at          REAL
        );

        CREATE TABLE IF NOT EXISTS meta_templates (
            id                      INTEGER PRIMARY KEY,
            template_id             TEXT UNIQUE,
            name                    TEXT,
            title_template          TEXT,
            description_template    TEXT,
            og_title_template       TEXT,
            og_description_template TEXT,
            page_type               TEXT,
            active                  INTEGER DEFAULT 1,
            created_at              REAL
        );

        CREATE TABLE IF NOT EXISTS sitemaps (
            id          INTEGER PRIMARY KEY,
            sitemap_id  TEXT UNIQUE,
            name        TEXT,
            base_url    TEXT,
            urls        TEXT DEFAULT '[]',
            generated_at REAL,
            url_count   INTEGER DEFAULT 0,
            created_at  REAL
        );

        CREATE TABLE IF NOT EXISTS backlinks (
            id                  INTEGER PRIMARY KEY,
            link_id             TEXT UNIQUE,
            target_url          TEXT,
            source_url          TEXT,
            anchor_text         TEXT,
            do_follow           INTEGER DEFAULT 1,
            domain_authority    INTEGER DEFAULT 0,
            discovered_at       REAL
        );
    """)
    conn.commit()
    conn.close()
    log_info("Database initialised.")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _uid(prefix: str = "") -> str:
    return prefix + secrets.token_hex(10)


def _now() -> float:
    return time.time()


def _json_response(handler: BaseHTTPRequestHandler, code: int, data) -> None:
    body = json.dumps(data, ensure_ascii=False).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _xml_response(handler: BaseHTTPRequestHandler, body: bytes) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", "application/xml; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_body(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", 0))
    if length == 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return {}


def _require_admin(handler: BaseHTTPRequestHandler) -> bool:
    """Return True if request carries valid admin secret, else send 403."""
    if not ADMIN_SECRET:
        return True  # no secret configured → open
    provided = handler.headers.get("X-Admin-Secret", "")
    try:
        ok = hmac.compare_digest(provided, ADMIN_SECRET)
    except Exception:
        ok = False
    if not ok:
        _json_response(handler, 403, {"error": "Forbidden"})
    return ok

# ---------------------------------------------------------------------------
# HTML fetching & parsing
# ---------------------------------------------------------------------------
_STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "by","from","as","is","was","are","were","be","been","being","have",
    "has","had","do","does","did","will","would","could","should","may",
    "might","shall","can","this","that","these","those","it","its","they",
    "them","their","we","our","you","your","he","his","she","her","i","my",
    "me","us","so","if","up","out","no","not","all","than","then","about",
    "into","over","after","also","more","what","which","who","how","when",
    "there","here","some","just","like","also","other","new","one","two",
}

def _fetch_html(url: str) -> tuple:
    """Fetch URL, return (html_str, final_url, status_code, error)."""
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "FractalMesh-SEOBot/1.0 (+https://fractalmesh.io/bot)"
                )
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            charset = "utf-8"
            ct = resp.headers.get_content_type() or ""
            params = resp.headers.get_params() or []
            for p in params:
                if p[0].lower() == "charset":
                    charset = p[1]
                    break
            body = resp.read(1_000_000)  # max 1 MB
            try:
                text = body.decode(charset, errors="replace")
            except Exception:
                text = body.decode("utf-8", errors="replace")
            return text, resp.geturl(), resp.status, None
    except urllib.error.HTTPError as exc:
        return "", url, exc.code, str(exc)
    except Exception as exc:
        return "", url, 0, str(exc)


def _strip_tags(text: str) -> str:
    """Remove HTML tags from text."""
    return re.sub(r"<[^>]+>", " ", text)


def _extract_title(html_text: str) -> str:
    m = re.search(r"<title[^>]*>(.*?)</title>", html_text, re.IGNORECASE | re.DOTALL)
    if m:
        return html.unescape(_strip_tags(m.group(1))).strip()
    return ""


def _extract_meta_description(html_text: str) -> str:
    patterns = [
        r'<meta\s+name=["\']description["\']\s+content=["\'](.*?)["\']',
        r'<meta\s+content=["\'](.*?)["\']\s+name=["\']description["\']',
    ]
    for pat in patterns:
        m = re.search(pat, html_text, re.IGNORECASE | re.DOTALL)
        if m:
            return html.unescape(m.group(1)).strip()
    return ""


def _extract_h1(html_text: str) -> str:
    m = re.search(r"<h1[^>]*>(.*?)</h1>", html_text, re.IGNORECASE | re.DOTALL)
    if m:
        return html.unescape(_strip_tags(m.group(1))).strip()
    return ""


def _extract_all_headings(html_text: str) -> dict:
    result = {}
    for level in range(1, 7):
        tag = f"h{level}"
        matches = re.findall(
            rf"<{tag}[^>]*>(.*?)</{tag}>", html_text, re.IGNORECASE | re.DOTALL
        )
        if matches:
            result[tag] = [
                html.unescape(_strip_tags(m)).strip() for m in matches
            ]
    return result


def _extract_body_text(html_text: str) -> str:
    # Remove script/style blocks
    text = re.sub(
        r"<(script|style|noscript|svg|head)[^>]*>.*?</\1>",
        " ", html_text, flags=re.IGNORECASE | re.DOTALL
    )
    return _strip_tags(text)


def _count_words(text: str) -> int:
    words = re.findall(r"[a-zA-Z']{2,}", text)
    return len(words)


def _keyword_density(text: str, top_n: int = 20) -> dict:
    words = re.findall(r"[a-zA-Z']{3,}", text.lower())
    freq = {}
    total = len(words) if words else 1
    for w in words:
        if w not in _STOPWORDS:
            freq[w] = freq.get(w, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return {w: round(c / total * 100, 2) for w, c in sorted_words}


def _extract_links(html_text: str, base_url: str) -> tuple:
    """Return (internal_links, external_links) as lists."""
    # Parse base domain
    m = re.match(r"(https?://[^/]+)", base_url)
    base_domain = m.group(1) if m else ""

    hrefs = re.findall(r'href=["\']([^"\'#\s]+)["\']', html_text, re.IGNORECASE)
    internal, external = [], []
    for h in hrefs:
        if h.startswith("http"):
            if base_domain and h.startswith(base_domain):
                internal.append(h)
            else:
                external.append(h)
        elif h.startswith("/"):
            internal.append(base_domain + h)
    return list(set(internal)), list(set(external))


def _readability_score(text: str) -> float:
    """Approximate Flesch Reading Ease (0-100, higher = easier)."""
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = re.findall(r"[a-zA-Z']{1,}", text)
    if not sentences or not words:
        return 0.0
    # Approximate syllables: vowel groups
    def _syllables(word: str) -> int:
        count = len(re.findall(r"[aeiouy]+", word.lower()))
        return max(1, count)

    total_syllables = sum(_syllables(w) for w in words)
    avg_words_per_sentence = len(words) / len(sentences)
    avg_syllables_per_word = total_syllables / len(words)
    score = 206.835 - 1.015 * avg_words_per_sentence - 84.6 * avg_syllables_per_word
    return round(max(0.0, min(100.0, score)), 1)


# ---------------------------------------------------------------------------
# SEO Score calculation
# ---------------------------------------------------------------------------
def _calculate_seo_score(
    url: str,
    title: str,
    meta_desc: str,
    h1: str,
    word_count: int,
    keyword_density: dict,
    existing_titles: list,
) -> tuple:
    """Return (score: int, issues: list[str])."""
    score = 0
    issues = []

    # Title
    if title:
        t_len = len(title)
        if 30 <= t_len <= 60:
            score += 20
        elif t_len > 0:
            score += 10
            if t_len < 30:
                issues.append("Title is too short (< 30 chars)")
            else:
                issues.append("Title is too long (> 60 chars)")
    else:
        issues.append("Missing title tag")
        score -= 20

    # Meta description
    if meta_desc:
        d_len = len(meta_desc)
        if 120 <= d_len <= 160:
            score += 20
        elif d_len > 0:
            score += 10
            if d_len < 120:
                issues.append("Meta description is too short (< 120 chars)")
            else:
                issues.append("Meta description is too long (> 160 chars)")
    else:
        issues.append("Missing meta description")
        score -= 15

    # H1
    if h1:
        score += 15
    else:
        issues.append("Missing H1 tag")
        score -= 10

    # Word count
    if word_count > 300:
        score += 15
    else:
        issues.append(f"Thin content — only {word_count} words (< 300)")
        score -= 10

    # HTTPS
    if url.startswith("https://"):
        score += 10
    else:
        issues.append("Page is not served over HTTPS")

    # Duplicate title check
    if title and title in existing_titles:
        issues.append("Duplicate title detected across analysed pages")
        score -= 10
    elif title:
        score += 10  # no duplicate

    # Keyword in title
    if title and keyword_density:
        top_kw = list(keyword_density.keys())[:3]
        title_lower = title.lower()
        if any(kw in title_lower for kw in top_kw):
            score += 10
        else:
            issues.append("Top keywords not found in title")

    return max(0, min(100, score)), issues


# ---------------------------------------------------------------------------
# Analyse URL
# ---------------------------------------------------------------------------
def analyse_url(url: str) -> dict:
    """Fetch and analyse a URL, store/update in DB, return analysis dict."""
    now = _now()
    html_text, final_url, status_code, error = _fetch_html(url)

    if error or not html_text:
        return {
            "error": error or "Empty response",
            "url": url,
            "status_code": status_code,
        }

    title = _extract_title(html_text)
    meta_desc = _extract_meta_description(html_text)
    h1 = _extract_h1(html_text)
    headings = _extract_all_headings(html_text)
    body_text = _extract_body_text(html_text)
    word_count = _count_words(body_text)
    kw_density = _keyword_density(body_text)
    readability = _readability_score(body_text)
    internal_links, external_links = _extract_links(html_text, final_url)

    # Gather existing titles for duplicate check
    conn = _db_connect()
    cur = conn.cursor()
    cur.execute("SELECT title FROM pages WHERE url != ?", (url,))
    existing_titles = [r["title"] for r in cur.fetchall() if r["title"]]

    seo_score, issues = _calculate_seo_score(
        final_url, title, meta_desc, h1, word_count, kw_density, existing_titles
    )

    # Upsert page record
    cur.execute("SELECT page_id FROM pages WHERE url = ?", (url,))
    row = cur.fetchone()
    if row:
        page_id = row["page_id"]
        cur.execute(
            """UPDATE pages SET title=?, meta_description=?, h1=?, word_count=?,
               keyword_density=?, readability_score=?, seo_score=?, issues=?,
               last_crawled=?, updated_at=?
               WHERE url=?""",
            (
                title, meta_desc, h1, word_count,
                json.dumps(kw_density), readability, seo_score,
                json.dumps(issues), now, now, url,
            ),
        )
    else:
        page_id = _uid("pg_")
        cur.execute(
            """INSERT INTO pages
               (page_id, url, title, meta_description, h1, word_count,
                keyword_density, readability_score, seo_score, issues,
                last_crawled, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                page_id, url, title, meta_desc, h1, word_count,
                json.dumps(kw_density), readability, seo_score,
                json.dumps(issues), now, now, now,
            ),
        )
    conn.commit()
    conn.close()

    return {
        "page_id": page_id,
        "url": final_url,
        "status_code": status_code,
        "title": title,
        "title_length": len(title),
        "meta_description": meta_desc,
        "meta_description_length": len(meta_desc),
        "h1": h1,
        "headings": headings,
        "word_count": word_count,
        "keyword_density": kw_density,
        "readability_score": readability,
        "seo_score": seo_score,
        "issues": issues,
        "internal_links_count": len(internal_links),
        "external_links_count": len(external_links),
        "internal_links": internal_links[:20],
        "external_links": external_links[:20],
        "last_crawled": now,
    }


# ---------------------------------------------------------------------------
# Sitemap XML generation
# ---------------------------------------------------------------------------
def _generate_sitemap_xml(base_url: str, urls: list) -> str:
    """Generate valid XML sitemap string from url entries."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ]
    for entry in urls:
        raw_url = entry.get("url", "")
        if not raw_url:
            continue
        # Resolve relative URLs
        if not raw_url.startswith("http"):
            raw_url = base_url.rstrip("/") + "/" + raw_url.lstrip("/")
        escaped_url = html.escape(raw_url)
        lines.append("  <url>")
        lines.append(f"    <loc>{escaped_url}</loc>")
        priority = entry.get("priority")
        if priority is not None:
            lines.append(f"    <priority>{float(priority):.1f}</priority>")
        changefreq = entry.get("changefreq")
        if changefreq:
            lines.append(f"    <changefreq>{html.escape(str(changefreq))}</changefreq>")
        lastmod = entry.get("lastmod")
        if lastmod:
            lines.append(f"    <lastmod>{html.escape(str(lastmod))}</lastmod>")
        lines.append("  </url>")
    lines.append("</urlset>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Meta tag helpers
# ---------------------------------------------------------------------------
def _apply_template(template: str, variables: dict) -> str:
    """Replace {{key}} placeholders with values from variables dict."""
    def _replacer(m):
        key = m.group(1).strip()
        return str(variables.get(key, m.group(0)))
    return re.sub(r"\{\{([^}]+)\}\}", _replacer, template)


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def _auto_generate_meta(title_vars: dict, description_vars: dict) -> dict:
    """Auto-generate meta tags from provided variable dicts."""
    # Build title from vars
    parts = [str(v) for v in title_vars.values() if v]
    title_raw = " | ".join(parts) if parts else "Untitled"
    title = _truncate(title_raw, 60)

    # Build description from vars
    desc_parts = [str(v) for v in description_vars.values() if v]
    desc_raw = " ".join(desc_parts) if desc_parts else ""
    meta_desc = _truncate(desc_raw, 160)

    return {
        "title": title,
        "meta_description": meta_desc,
        "og_title": title,
        "og_description": meta_desc,
    }


# ---------------------------------------------------------------------------
# Anthropic AI suggest
# ---------------------------------------------------------------------------
def _ai_suggest(url: str = "", content: str = "", topic: str = "") -> dict:
    """Call Anthropic claude-haiku-4-5 for SEO suggestions."""
    if not ANTHROPIC_API_KEY:
        return {"error": "ANTHROPIC_API_KEY not configured"}

    prompt_parts = []
    if url:
        prompt_parts.append(f"URL: {url}")
    if topic:
        prompt_parts.append(f"Topic: {topic}")
    if content:
        snippet = content[:3000]
        prompt_parts.append(f"Content snippet:\n{snippet}")

    if not prompt_parts:
        return {"error": "Provide at least one of: url, content, topic"}

    system_msg = (
        "You are an SEO expert. Analyse the provided information and return "
        "a JSON object with these keys:\n"
        "  keywords: list of 10 high-value keyword suggestions\n"
        "  title_ideas: list of 5 optimised page title ideas (max 60 chars each)\n"
        "  content_gaps: list of 5 content topics or sections that are missing\n"
        "  meta_description: one optimised meta description (120-160 chars)\n"
        "  recommendations: list of 5 actionable SEO improvement recommendations\n"
        "Respond with valid JSON only, no markdown fences."
    )

    user_msg = "\n".join(prompt_parts)
    payload = {
        "model": "claude-haiku-4-5",
        "max_tokens": 1024,
        "system": system_msg,
        "messages": [{"role": "user", "content": user_msg}],
    }

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
            text = resp_data["content"][0]["text"]
            try:
                return json.loads(text)
            except Exception:
                return {"raw": text}
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        return {"error": f"Anthropic API error {exc.code}: {err_body[:300]}"}
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Background daemon — re-analyse stale pages
# ---------------------------------------------------------------------------
def _background_daemon() -> None:
    """Re-analyse pages older than 24 hours, loop every 3600 seconds."""
    while True:
        try:
            conn = _db_connect()
            cur = conn.cursor()
            cutoff = _now() - 86400  # 24 hours
            cur.execute(
                "SELECT url FROM pages WHERE last_crawled < ? OR last_crawled IS NULL",
                (cutoff,),
            )
            stale = [r["url"] for r in cur.fetchall()]
            conn.close()
            for url in stale:
                try:
                    log_info(f"Background re-analysis: {url}")
                    analyse_url(url)
                    time.sleep(2)  # polite crawl delay
                except Exception as exc:
                    log_err(f"Background re-analysis failed for {url}: {exc}")
        except Exception as exc:
            log_err(f"Background daemon error: {exc}")
        time.sleep(3600)


# ---------------------------------------------------------------------------
# HTTP Request Handler
# ---------------------------------------------------------------------------
class SEOHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        log_info(fmt % args)

    # ---- routing ----

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")

        if path == "/health":
            self._handle_health()
        elif path == "/pages":
            self._handle_list_pages()
        elif path.startswith("/pages/"):
            page_id = path[len("/pages/"):]
            self._handle_get_page(page_id)
        elif path == "/keywords":
            self._handle_list_keywords()
        elif path == "/meta/templates":
            self._handle_list_templates()
        elif path.startswith("/sitemap/"):
            sitemap_id = path[len("/sitemap/"):]
            self._handle_get_sitemap(sitemap_id)
        elif path.startswith("/backlinks/"):
            target_url = urllib.parse_unquote(path[len("/backlinks/"):])
            self._handle_list_backlinks(target_url)
        elif path == "/dashboard":
            self._handle_dashboard()
        else:
            _json_response(self, 404, {"error": "Not found"})

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")

        if path == "/analyse":
            self._handle_analyse()
        elif path == "/keywords":
            self._handle_create_keyword()
        elif re.match(r"^/keywords/[^/]+/check$", path):
            kw_id = path.split("/")[2]
            self._handle_check_keyword(kw_id)
        elif path == "/meta/generate":
            self._handle_meta_generate()
        elif path == "/meta/templates":
            self._handle_create_template()
        elif path == "/sitemap":
            self._handle_create_sitemap()
        elif path == "/backlinks":
            self._handle_create_backlink()
        elif path == "/ai/suggest":
            self._handle_ai_suggest()
        else:
            _json_response(self, 404, {"error": "Not found"})

    # ---- GET handlers ----

    def _handle_health(self):
        _json_response(self, 200, {
            "status": "ok",
            "service": "seo_engine",
            "port": PORT,
            "uptime_seconds": round(_now() - START_TIME, 1),
        })

    def _handle_list_pages(self):
        if not _require_admin(self):
            return
        conn = _db_connect()
        cur = conn.cursor()
        cur.execute(
            "SELECT page_id, url, title, seo_score, word_count, last_crawled, created_at "
            "FROM pages ORDER BY updated_at DESC LIMIT 200"
        )
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        _json_response(self, 200, {"pages": rows, "count": len(rows)})

    def _handle_get_page(self, page_id: str):
        conn = _db_connect()
        cur = conn.cursor()
        cur.execute("SELECT * FROM pages WHERE page_id = ?", (page_id,))
        row = cur.fetchone()
        conn.close()
        if not row:
            _json_response(self, 404, {"error": "Page not found"})
            return
        data = dict(row)
        for field in ("keyword_density", "issues"):
            try:
                data[field] = json.loads(data[field])
            except Exception:
                pass
        _json_response(self, 200, data)

    def _handle_list_keywords(self):
        conn = _db_connect()
        cur = conn.cursor()
        cur.execute("SELECT * FROM keywords ORDER BY created_at DESC LIMIT 500")
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        _json_response(self, 200, {"keywords": rows, "count": len(rows)})

    def _handle_list_templates(self):
        conn = _db_connect()
        cur = conn.cursor()
        cur.execute("SELECT * FROM meta_templates WHERE active = 1 ORDER BY created_at DESC")
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        _json_response(self, 200, {"templates": rows, "count": len(rows)})

    def _handle_get_sitemap(self, sitemap_id: str):
        conn = _db_connect()
        cur = conn.cursor()
        cur.execute("SELECT * FROM sitemaps WHERE sitemap_id = ?", (sitemap_id,))
        row = cur.fetchone()
        conn.close()
        if not row:
            _json_response(self, 404, {"error": "Sitemap not found"})
            return
        row_dict = dict(row)
        try:
            urls = json.loads(row_dict["urls"])
        except Exception:
            urls = []
        xml_str = _generate_sitemap_xml(row_dict["base_url"], urls)
        _xml_response(self, xml_str.encode("utf-8"))

    def _handle_list_backlinks(self, target_url: str):
        # URL decode the target_url
        try:
            import urllib.parse as _up
            target_url = _up.unquote(target_url)
        except Exception:
            pass
        conn = _db_connect()
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM backlinks WHERE target_url = ? ORDER BY discovered_at DESC",
            (target_url,),
        )
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        _json_response(self, 200, {
            "target_url": target_url,
            "backlinks": rows,
            "count": len(rows),
        })

    def _handle_dashboard(self):
        if not _require_admin(self):
            return
        conn = _db_connect()
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) AS c FROM pages")
        total_pages = cur.fetchone()["c"]

        cur.execute("SELECT AVG(seo_score) AS a FROM pages")
        avg_row = cur.fetchone()
        avg_score = round(avg_row["a"] or 0, 1)

        cur.execute("SELECT COUNT(*) AS c FROM keywords")
        kw_count = cur.fetchone()["c"]

        cur.execute("SELECT COUNT(*) AS c FROM backlinks")
        bl_count = cur.fetchone()["c"]

        # Aggregate issues across all pages
        cur.execute("SELECT issues FROM pages WHERE issues IS NOT NULL AND issues != '[]'")
        issue_counts = {}
        for r in cur.fetchall():
            try:
                issue_list = json.loads(r["issues"])
                for issue in issue_list:
                    issue_counts[issue] = issue_counts.get(issue, 0) + 1
            except Exception:
                pass
        top_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        top_issues = [{"issue": i, "count": c} for i, c in top_issues]

        cur.execute("SELECT AVG(readability_score) AS a FROM pages")
        avg_read_row = cur.fetchone()
        avg_readability = round(avg_read_row["a"] or 0, 1)

        cur.execute("SELECT COUNT(*) AS c FROM sitemaps")
        sitemap_count = cur.fetchone()["c"]

        cur.execute("SELECT COUNT(*) AS c FROM meta_templates WHERE active=1")
        template_count = cur.fetchone()["c"]

        conn.close()
        _json_response(self, 200, {
            "total_pages": total_pages,
            "avg_seo_score": avg_score,
            "avg_readability_score": avg_readability,
            "keyword_count": kw_count,
            "backlink_count": bl_count,
            "sitemap_count": sitemap_count,
            "active_template_count": template_count,
            "top_issues": top_issues,
        })

    # ---- POST handlers ----

    def _handle_analyse(self):
        body = _read_body(self)
        url = body.get("url", "").strip()
        if not url:
            _json_response(self, 400, {"error": "url is required"})
            return
        if not re.match(r"^https?://", url):
            _json_response(self, 400, {"error": "url must start with http:// or https://"})
            return
        result = analyse_url(url)
        if "error" in result:
            _json_response(self, 502, result)
        else:
            _json_response(self, 200, result)

    def _handle_create_keyword(self):
        body = _read_body(self)
        keyword = body.get("keyword", "").strip()
        target_url = body.get("target_url", "").strip()
        if not keyword or not target_url:
            _json_response(self, 400, {"error": "keyword and target_url are required"})
            return
        search_volume = int(body.get("search_volume", 0))
        difficulty = int(body.get("difficulty", 0))
        now = _now()

        conn = _db_connect()
        cur = conn.cursor()
        # Check for duplicate keyword+url
        cur.execute(
            "SELECT kw_id FROM keywords WHERE keyword = ? AND target_url = ?",
            (keyword, target_url),
        )
        existing = cur.fetchone()
        if existing:
            conn.close()
            _json_response(self, 409, {
                "error": "Keyword already tracked for this URL",
                "kw_id": existing["kw_id"],
            })
            return

        kw_id = _uid("kw_")
        cur.execute(
            """INSERT INTO keywords
               (kw_id, keyword, target_url, search_volume, difficulty,
                position, previous_position, position_change, last_checked, created_at)
               VALUES (?,?,?,?,?,NULL,NULL,0,NULL,?)""",
            (kw_id, keyword, target_url, search_volume, difficulty, now),
        )
        conn.commit()
        conn.close()
        _json_response(self, 201, {
            "kw_id": kw_id,
            "keyword": keyword,
            "target_url": target_url,
            "search_volume": search_volume,
            "difficulty": difficulty,
            "created_at": now,
        })

    def _handle_check_keyword(self, kw_id: str):
        """Check if target_url page content contains the keyword."""
        conn = _db_connect()
        cur = conn.cursor()
        cur.execute("SELECT * FROM keywords WHERE kw_id = ?", (kw_id,))
        kw_row = cur.fetchone()
        if not kw_row:
            conn.close()
            _json_response(self, 404, {"error": "Keyword not found"})
            return

        keyword = kw_row["keyword"]
        target_url = kw_row["target_url"]

        # Fetch the target URL
        html_text, final_url, status_code, error = _fetch_html(target_url)
        if error or not html_text:
            conn.close()
            _json_response(self, 502, {
                "error": error or "Empty response",
                "url": target_url,
            })
            return

        body_text = _extract_body_text(html_text).lower()
        title = _extract_title(html_text).lower()
        meta_desc = _extract_meta_description(html_text).lower()
        kw_lower = keyword.lower()

        in_body = kw_lower in body_text
        in_title = kw_lower in title
        in_meta = kw_lower in meta_desc
        occurrences = body_text.count(kw_lower)
        word_count = _count_words(body_text)
        density = round(occurrences / max(word_count, 1) * 100, 2) if in_body else 0.0

        now = _now()
        prev_pos = kw_row["position"]
        # Assign a rough pseudo-position based on keyword presence
        new_position = 1 if in_title else (5 if in_meta else (10 if in_body else 100))
        position_change = (prev_pos - new_position) if prev_pos else 0

        cur.execute(
            """UPDATE keywords SET position=?, previous_position=?,
               position_change=?, last_checked=? WHERE kw_id=?""",
            (new_position, prev_pos, position_change, now, kw_id),
        )
        conn.commit()
        conn.close()

        _json_response(self, 200, {
            "kw_id": kw_id,
            "keyword": keyword,
            "target_url": target_url,
            "found_in_body": in_body,
            "found_in_title": in_title,
            "found_in_meta_description": in_meta,
            "occurrences": occurrences,
            "keyword_density_percent": density,
            "estimated_position": new_position,
            "position_change": position_change,
            "checked_at": now,
        })

    def _handle_meta_generate(self):
        body = _read_body(self)
        title_vars = body.get("title_vars", {})
        desc_vars = body.get("description_vars", {})
        template_id = body.get("template_id", "")
        page_type = body.get("page_type", "")

        if template_id:
            conn = _db_connect()
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM meta_templates WHERE template_id = ? AND active = 1",
                (template_id,),
            )
            tmpl = cur.fetchone()
            conn.close()
            if not tmpl:
                _json_response(self, 404, {"error": "Template not found or inactive"})
                return
            all_vars = {**title_vars, **desc_vars}
            raw_title = _apply_template(tmpl["title_template"] or "", all_vars)
            raw_desc  = _apply_template(tmpl["description_template"] or "", all_vars)
            raw_og_title = _apply_template(tmpl["og_title_template"] or raw_title, all_vars)
            raw_og_desc  = _apply_template(tmpl["og_description_template"] or raw_desc, all_vars)
        elif page_type:
            # Try to find a template by page_type
            conn = _db_connect()
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM meta_templates WHERE page_type = ? AND active = 1 LIMIT 1",
                (page_type,),
            )
            tmpl = cur.fetchone()
            conn.close()
            if tmpl:
                all_vars = {**title_vars, **desc_vars}
                raw_title = _apply_template(tmpl["title_template"] or "", all_vars)
                raw_desc  = _apply_template(tmpl["description_template"] or "", all_vars)
                raw_og_title = _apply_template(tmpl["og_title_template"] or raw_title, all_vars)
                raw_og_desc  = _apply_template(tmpl["og_description_template"] or raw_desc, all_vars)
            else:
                result = _auto_generate_meta(title_vars, desc_vars)
                _json_response(self, 200, result)
                return
        else:
            result = _auto_generate_meta(title_vars, desc_vars)
            _json_response(self, 200, result)
            return

        _json_response(self, 200, {
            "title": _truncate(raw_title, 60),
            "meta_description": _truncate(raw_desc, 160),
            "og_title": _truncate(raw_og_title, 60),
            "og_description": _truncate(raw_og_desc, 160),
        })

    def _handle_create_template(self):
        if not _require_admin(self):
            return
        body = _read_body(self)
        name = body.get("name", "").strip()
        title_template = body.get("title_template", "").strip()
        description_template = body.get("description_template", "").strip()
        if not name or not title_template:
            _json_response(self, 400, {"error": "name and title_template are required"})
            return
        og_title = body.get("og_title_template", title_template)
        og_desc  = body.get("og_description_template", description_template)
        page_type = body.get("page_type", "")
        now = _now()
        template_id = _uid("tmpl_")

        conn = _db_connect()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO meta_templates
               (template_id, name, title_template, description_template,
                og_title_template, og_description_template, page_type, active, created_at)
               VALUES (?,?,?,?,?,?,?,1,?)""",
            (template_id, name, title_template, description_template,
             og_title, og_desc, page_type, now),
        )
        conn.commit()
        conn.close()
        _json_response(self, 201, {
            "template_id": template_id,
            "name": name,
            "title_template": title_template,
            "description_template": description_template,
            "og_title_template": og_title,
            "og_description_template": og_desc,
            "page_type": page_type,
            "created_at": now,
        })

    def _handle_create_sitemap(self):
        body = _read_body(self)
        name = body.get("name", "").strip()
        base_url = body.get("base_url", "").strip()
        urls = body.get("urls", [])
        if not name or not base_url:
            _json_response(self, 400, {"error": "name and base_url are required"})
            return
        if not isinstance(urls, list):
            _json_response(self, 400, {"error": "urls must be a list"})
            return

        now = _now()
        sitemap_id = _uid("sm_")

        conn = _db_connect()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO sitemaps
               (sitemap_id, name, base_url, urls, generated_at, url_count, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (sitemap_id, name, base_url, json.dumps(urls), now, len(urls), now),
        )
        conn.commit()
        conn.close()

        xml_str = _generate_sitemap_xml(base_url, urls)
        _json_response(self, 201, {
            "sitemap_id": sitemap_id,
            "name": name,
            "base_url": base_url,
            "url_count": len(urls),
            "generated_at": now,
            "xml_preview": xml_str[:500] + ("..." if len(xml_str) > 500 else ""),
            "access_url": f"/sitemap/{sitemap_id}",
        })

    def _handle_create_backlink(self):
        body = _read_body(self)
        target_url = body.get("target_url", "").strip()
        source_url = body.get("source_url", "").strip()
        anchor_text = body.get("anchor_text", "").strip()
        do_follow = int(body.get("do_follow", 1))
        domain_authority = int(body.get("domain_authority", 0))

        if not target_url or not source_url:
            _json_response(self, 400, {
                "error": "target_url and source_url are required"
            })
            return

        now = _now()
        link_id = _uid("bl_")
        conn = _db_connect()
        cur = conn.cursor()
        try:
            cur.execute(
                """INSERT INTO backlinks
                   (link_id, target_url, source_url, anchor_text, do_follow,
                    domain_authority, discovered_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (link_id, target_url, source_url, anchor_text,
                 do_follow, domain_authority, now),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            conn.close()
            _json_response(self, 409, {"error": "Backlink already recorded"})
            return
        conn.close()
        _json_response(self, 201, {
            "link_id": link_id,
            "target_url": target_url,
            "source_url": source_url,
            "anchor_text": anchor_text,
            "do_follow": bool(do_follow),
            "domain_authority": domain_authority,
            "discovered_at": now,
        })

    def _handle_ai_suggest(self):
        body = _read_body(self)
        url     = body.get("url", "").strip()
        content = body.get("content", "").strip()
        topic   = body.get("topic", "").strip()
        if not url and not content and not topic:
            _json_response(self, 400, {
                "error": "Provide at least one of: url, content, topic"
            })
            return
        result = _ai_suggest(url=url, content=content, topic=topic)
        if "error" in result:
            _json_response(self, 502, result)
        else:
            _json_response(self, 200, result)


# ---------------------------------------------------------------------------
# Compatibility shim for urllib.parse import in handler
# ---------------------------------------------------------------------------
import urllib.parse as _urllib_parse


def _url_unquote(s: str) -> str:
    return _urllib_parse.unquote(s)

# Patch _handle_list_backlinks to use the proper unquote
_orig_list_backlinks = SEOHandler._handle_list_backlinks

def _patched_list_backlinks(self, target_url: str):
    target_url = _url_unquote(target_url)
    conn = _db_connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM backlinks WHERE target_url = ? ORDER BY discovered_at DESC",
        (target_url,),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    _json_response(self, 200, {
        "target_url": target_url,
        "backlinks": rows,
        "count": len(rows),
    })

SEOHandler._handle_list_backlinks = _patched_list_backlinks

# Similarly patch do_GET to use _url_unquote properly
_orig_do_get = SEOHandler.do_GET

def _patched_do_get(self):
    path = self.path.split("?")[0].rstrip("/")

    if path == "/health":
        self._handle_health()
    elif path == "/pages":
        self._handle_list_pages()
    elif path.startswith("/pages/"):
        page_id = path[len("/pages/"):]
        self._handle_get_page(page_id)
    elif path == "/keywords":
        self._handle_list_keywords()
    elif path == "/meta/templates":
        self._handle_list_templates()
    elif path.startswith("/sitemap/"):
        sitemap_id = path[len("/sitemap/"):]
        self._handle_get_sitemap(sitemap_id)
    elif path.startswith("/backlinks/"):
        raw = path[len("/backlinks/"):]
        target_url = _urllib_parse.unquote(raw)
        self._handle_list_backlinks(target_url)
    elif path == "/dashboard":
        self._handle_dashboard()
    else:
        _json_response(self, 404, {"error": "Not found"})

SEOHandler.do_GET = _patched_do_get


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    db_init()

    # Start background daemon thread
    daemon = threading.Thread(target=_background_daemon, daemon=True, name="seo-daemon")
    daemon.start()
    log_info(f"Background re-analysis daemon started.")

    server = HTTPServer(("0.0.0.0", PORT), SEOHandler)
    log_info(f"SEO Engine listening on port {PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log_info("SEO Engine shutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
