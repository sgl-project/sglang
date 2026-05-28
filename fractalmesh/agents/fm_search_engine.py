#!/usr/bin/env python3
"""
fm_search_engine.py — Sovereign Search Engine for FractalMesh OMEGA Titan (Port 7862)
Unified search aggregator: Google CSE, Bing v7, Crawlbase scraper fallback.
Deduplicates, ranks, and caches results in SQLite (WAL mode).
All credentials sourced from ~/.secrets/fractal.env at runtime.
Samuel James Hiotis | ABN 56 628 117 363
"""
import collections
import hashlib
import json
import math
import os
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ── vault ─────────────────────────────────────────────────────────────────────
_ENV_FILE = Path.home() / ".secrets" / "fractal.env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

# ── config ────────────────────────────────────────────────────────────────────
PORT                  = int(os.getenv("SEARCH_ENGINE_PORT", "7862"))
GOOGLE_CSE_API_KEY    = os.getenv("GOOGLE_CSE_API_KEY", "")
GOOGLE_CSE_ID         = os.getenv("GOOGLE_CSE_ID", "")
BING_SEARCH_API_KEY   = os.getenv("BING_SEARCH_API_KEY", "")
CRAWLBASE_NORMAL_TOKEN = os.getenv("CRAWLBASE_NORMAL_TOKEN", "")
ADMIN_SECRET          = os.getenv("ADMIN_SECRET", "")

ROOT = Path(os.getenv("FRACTALMESH_HOME", str(Path.home() / "fmsaas")))
DB   = ROOT / "database" / "sovereign.db"
LOG  = ROOT / "logs" / "search_engine.log"

ROOT.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)
DB.parent.mkdir(parents=True, exist_ok=True)

CACHE_TTL        = 3600          # seconds
MAINTENANCE_INTERVAL = 3600      # seconds between background sweeps
SOURCE_WEIGHTS   = {"google": 1.0, "bing": 0.9, "crawlbase": 0.7}
START_TIME       = time.time()

# ── simple logger ─────────────────────────────────────────────────────────────
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SEARCH] %(message)s",
    handlers=[logging.FileHandler(str(LOG)), logging.StreamHandler()],
)
log = logging.getLogger("search_engine")

# ── database ──────────────────────────────────────────────────────────────────

def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB), timeout=15, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def db_init():
    conn = db_connect()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS searches (
            id              INTEGER PRIMARY KEY,
            query           TEXT,
            backend         TEXT,
            results_count   INTEGER,
            cached          INTEGER DEFAULT 0,
            latency_ms      REAL,
            searched_at     REAL
        );
        CREATE TABLE IF NOT EXISTS results (
            id              INTEGER PRIMARY KEY,
            search_id       INTEGER,
            rank            INTEGER,
            title           TEXT,
            url             TEXT,
            snippet         TEXT,
            domain          TEXT,
            relevance_score REAL,
            source          TEXT
        );
        CREATE TABLE IF NOT EXISTS cache (
            id              INTEGER PRIMARY KEY,
            query_hash      TEXT UNIQUE,
            query           TEXT,
            results         TEXT,
            result_count    INTEGER,
            backends_used   TEXT,
            cached_at       REAL,
            expires_at      REAL,
            hit_count       INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS domains (
            id              INTEGER PRIMARY KEY,
            domain          TEXT UNIQUE,
            trust_score     REAL DEFAULT 0.5,
            block_listed    INTEGER DEFAULT 0,
            result_count    INTEGER DEFAULT 0,
            last_seen       REAL
        );
        CREATE INDEX IF NOT EXISTS idx_searches_query   ON searches(query);
        CREATE INDEX IF NOT EXISTS idx_searches_backend ON searches(backend);
        CREATE INDEX IF NOT EXISTS idx_results_search   ON results(search_id);
        CREATE INDEX IF NOT EXISTS idx_cache_hash       ON cache(query_hash);
        CREATE INDEX IF NOT EXISTS idx_domains_domain   ON domains(domain);
    """)
    conn.commit()
    conn.close()
    log.info("Database initialised at %s", DB)

# ── helpers ───────────────────────────────────────────────────────────────────

def query_hash(query: str) -> str:
    return hashlib.md5(query.lower().strip().encode()).hexdigest()


def extract_domain(url: str) -> str:
    try:
        parsed = urllib.parse.urlparse(url)
        host = parsed.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def json_response(handler, status: int, data: dict):
    body = json.dumps(data, indent=2).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def parse_query_string(path: str) -> dict:
    if "?" in path:
        qs = path.split("?", 1)[1]
        return dict(urllib.parse.parse_qsl(qs))
    return {}

# ── domain trust helpers ──────────────────────────────────────────────────────

def get_domain_trust(domain: str) -> float:
    if not domain:
        return 0.5
    try:
        conn = db_connect()
        row = conn.execute(
            "SELECT trust_score, block_listed FROM domains WHERE domain=?", (domain,)
        ).fetchone()
        conn.close()
        if row:
            if row["block_listed"]:
                return 0.0
            return float(row["trust_score"])
        return 0.5
    except Exception:
        return 0.5


def upsert_domain(domain: str, increment: bool = True):
    if not domain:
        return
    try:
        conn = db_connect()
        conn.execute("""
            INSERT INTO domains (domain, trust_score, block_listed, result_count, last_seen)
            VALUES (?, 0.5, 0, ?, ?)
            ON CONFLICT(domain) DO UPDATE SET
                result_count = result_count + ?,
                last_seen    = excluded.last_seen
        """, (domain, 1 if increment else 0, time.time(), 1 if increment else 0))
        conn.commit()
        conn.close()
    except Exception as exc:
        log.warning("upsert_domain error: %s", exc)

# ── search backends ───────────────────────────────────────────────────────────

def search_google(query: str) -> list:
    """Query Google Custom Search Engine API."""
    if not GOOGLE_CSE_API_KEY or not GOOGLE_CSE_ID:
        return []
    params = urllib.parse.urlencode({
        "q":   query,
        "key": GOOGLE_CSE_API_KEY,
        "cx":  GOOGLE_CSE_ID,
        "num": 10,
    })
    url = f"https://www.googleapis.com/customsearch/v1?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "FractalMesh-SearchEngine/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        items = data.get("items", [])
        results = []
        for i, item in enumerate(items):
            raw_url = item.get("link", "")
            domain  = extract_domain(raw_url)
            if not domain:
                continue
            results.append({
                "rank":            i + 1,
                "title":           item.get("title", ""),
                "url":             raw_url,
                "snippet":         item.get("snippet", ""),
                "domain":          domain,
                "relevance_score": 1.0 - (i * 0.05),
                "source":          "google",
            })
        log.info("Google CSE returned %d results for '%s'", len(results), query)
        return results
    except urllib.error.HTTPError as exc:
        log.warning("Google CSE HTTP error %s for '%s': %s", exc.code, query, exc.reason)
        return []
    except Exception as exc:
        log.warning("Google CSE error for '%s': %s", query, exc)
        return []


def search_bing(query: str) -> list:
    """Query Microsoft Bing Search v7 API."""
    if not BING_SEARCH_API_KEY:
        return []
    params = urllib.parse.urlencode({"q": query, "count": 10})
    url = f"https://api.bing.microsoft.com/v7.0/search?{params}"
    try:
        req = urllib.request.Request(
            url,
            headers={
                "Ocp-Apim-Subscription-Key": BING_SEARCH_API_KEY,
                "User-Agent": "FractalMesh-SearchEngine/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        items = data.get("webPages", {}).get("value", [])
        results = []
        for i, item in enumerate(items):
            raw_url = item.get("url", "")
            domain  = extract_domain(raw_url)
            if not domain:
                continue
            results.append({
                "rank":            i + 1,
                "title":           item.get("name", ""),
                "url":             raw_url,
                "snippet":         item.get("snippet", ""),
                "domain":          domain,
                "relevance_score": 1.0 - (i * 0.05),
                "source":          "bing",
            })
        log.info("Bing returned %d results for '%s'", len(results), query)
        return results
    except urllib.error.HTTPError as exc:
        log.warning("Bing HTTP error %s for '%s': %s", exc.code, query, exc.reason)
        return []
    except Exception as exc:
        log.warning("Bing error for '%s': %s", query, exc)
        return []


def search_crawlbase(query: str) -> list:
    """Scrape Google search results via Crawlbase proxy."""
    if not CRAWLBASE_NORMAL_TOKEN:
        return []
    target = urllib.parse.urlencode({"q": query, "num": 10})
    target_url = f"https://www.google.com/search?{target}"
    params = urllib.parse.urlencode({
        "token": CRAWLBASE_NORMAL_TOKEN,
        "url":   target_url,
    })
    scrape_url = f"https://api.crawlbase.com/?{params}"
    try:
        req = urllib.request.Request(
            scrape_url,
            headers={"User-Agent": "FractalMesh-SearchEngine/1.0"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        # Extract search result blocks: anchor href + h3 title + snippet text
        # Google SERP structure: <a href="/url?q=ACTUAL_URL&..."><h3>TITLE</h3></a>
        results = []
        seen_urls = set()

        # Pattern 1: extract <a href="/url?q=..."> blocks containing an <h3>
        block_pattern = re.compile(
            r'<a\s+[^>]*href=["\'](?:/url\?q=)?([^"\'#\s][^"\']*)["\'][^>]*>'
            r'(?:(?!<a\s).)*?<h3[^>]*>(.*?)</h3>',
            re.DOTALL | re.IGNORECASE,
        )
        # Pattern 2: simpler fallback — find h3 text adjacent to nearby anchor
        h3_pattern   = re.compile(r'<h3[^>]*>(.*?)</h3>', re.DOTALL | re.IGNORECASE)
        href_pattern = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)
        tag_strip    = re.compile(r'<[^>]+>')

        for match in block_pattern.finditer(html):
            raw_url = urllib.parse.unquote(match.group(1))
            # Google wraps real URLs in /url?q=REAL&... — unwrap if needed
            if raw_url.startswith("/url?"):
                parsed = urllib.parse.parse_qs(urllib.parse.urlparse(raw_url).query)
                raw_url = parsed.get("q", [""])[0]
            if not raw_url.startswith("http"):
                continue
            raw_url = raw_url.split("&")[0]  # trim tracking params in href
            if raw_url in seen_urls:
                continue
            domain = extract_domain(raw_url)
            if not domain or domain in ("google.com", "accounts.google.com"):
                continue
            title = tag_strip.sub("", match.group(2)).strip()
            if not title:
                continue
            seen_urls.add(raw_url)
            results.append({
                "rank":            len(results) + 1,
                "title":           title,
                "url":             raw_url,
                "snippet":         "",
                "domain":          domain,
                "relevance_score": max(0.3, 1.0 - (len(results) * 0.07)),
                "source":          "crawlbase",
            })
            if len(results) >= 10:
                break

        # Fallback: if block_pattern found nothing, try h3 + nearest preceding href
        if not results:
            h3_matches   = list(h3_pattern.finditer(html))
            href_matches = list(href_pattern.finditer(html))
            for h3m in h3_matches[:10]:
                title = tag_strip.sub("", h3m.group(1)).strip()
                if not title or len(title) < 5:
                    continue
                # find the href that ends just before this h3
                best_href = ""
                h3_start  = h3m.start()
                for hm in reversed(href_matches):
                    if hm.end() < h3_start and (h3_start - hm.end()) < 500:
                        candidate = urllib.parse.unquote(hm.group(1))
                        if candidate.startswith("http"):
                            best_href = candidate
                            break
                if not best_href or best_href in seen_urls:
                    continue
                domain = extract_domain(best_href)
                if not domain or domain in ("google.com", "accounts.google.com"):
                    continue
                seen_urls.add(best_href)
                results.append({
                    "rank":            len(results) + 1,
                    "title":           title,
                    "url":             best_href,
                    "snippet":         "",
                    "domain":          domain,
                    "relevance_score": max(0.3, 1.0 - (len(results) * 0.07)),
                    "source":          "crawlbase",
                })

        log.info("Crawlbase returned %d results for '%s'", len(results), query)
        return results
    except urllib.error.HTTPError as exc:
        log.warning("Crawlbase HTTP error %s for '%s': %s", exc.code, query, exc.reason)
        return []
    except Exception as exc:
        log.warning("Crawlbase error for '%s': %s", query, exc)
        return []

# ── deduplication & ranking ───────────────────────────────────────────────────

def deduplicate_and_rank(all_results: list, limit: int = 10) -> list:
    """
    Merge results by URL, boost relevance when seen in multiple backends,
    then rank by: (relevance_score * 0.6) + (source_weight * 0.3) + (domain_trust * 0.1)
    """
    grouped: dict = {}   # url -> merged result dict

    for item in all_results:
        url = item["url"].rstrip("/")
        if url not in grouped:
            grouped[url] = {
                "title":           item["title"],
                "url":             url,
                "snippet":         item["snippet"],
                "domain":          item["domain"],
                "sources":         [item["source"]],
                "relevance_score": item["relevance_score"],
                "source":          item["source"],
            }
        else:
            existing = grouped[url]
            # multi-source boost
            if item["source"] not in existing["sources"]:
                existing["sources"].append(item["source"])
                existing["relevance_score"] = min(
                    1.0, existing["relevance_score"] + 0.1
                )
            # prefer longer snippet
            if len(item.get("snippet", "")) > len(existing.get("snippet", "")):
                existing["snippet"] = item["snippet"]
            # prefer highest source weight
            if SOURCE_WEIGHTS.get(item["source"], 0) > SOURCE_WEIGHTS.get(existing["source"], 0):
                existing["source"] = item["source"]
                existing["title"]  = item["title"]

    ranked = []
    for url, item in grouped.items():
        domain       = item["domain"]
        trust        = get_domain_trust(domain)
        src_weight   = SOURCE_WEIGHTS.get(item["source"], 0.5)
        final_score  = (
            item["relevance_score"] * 0.6
            + src_weight            * 0.3
            + trust                 * 0.1
        )
        ranked.append({
            "title":           item["title"],
            "url":             item["url"],
            "snippet":         item["snippet"],
            "domain":          domain,
            "source":          item["source"],
            "sources":         item["sources"],
            "relevance_score": round(final_score, 4),
        })

    ranked.sort(key=lambda x: x["relevance_score"], reverse=True)

    # assign final rank
    output = []
    for i, item in enumerate(ranked[:limit]):
        item["rank"] = i + 1
        output.append(item)
    return output

# ── cache helpers ─────────────────────────────────────────────────────────────

def cache_get(qhash: str):
    try:
        conn = db_connect()
        row  = conn.execute(
            "SELECT * FROM cache WHERE query_hash=? AND expires_at > ?",
            (qhash, time.time()),
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE cache SET hit_count = hit_count + 1 WHERE query_hash=?",
                (qhash,),
            )
            conn.commit()
        conn.close()
        return dict(row) if row else None
    except Exception as exc:
        log.warning("cache_get error: %s", exc)
        return None


def cache_set(qhash: str, query: str, results: list, backends_used: list):
    try:
        now = time.time()
        conn = db_connect()
        conn.execute("""
            INSERT INTO cache (query_hash, query, results, result_count,
                               backends_used, cached_at, expires_at, hit_count)
            VALUES (?,?,?,?,?,?,?,0)
            ON CONFLICT(query_hash) DO UPDATE SET
                results       = excluded.results,
                result_count  = excluded.result_count,
                backends_used = excluded.backends_used,
                cached_at     = excluded.cached_at,
                expires_at    = excluded.expires_at,
                hit_count     = 0
        """, (
            qhash,
            query,
            json.dumps(results),
            len(results),
            json.dumps(backends_used),
            now,
            now + CACHE_TTL,
        ))
        conn.commit()
        conn.close()
    except Exception as exc:
        log.warning("cache_set error: %s", exc)


def cache_clear_all():
    try:
        conn = db_connect()
        conn.execute("DELETE FROM cache")
        conn.commit()
        conn.close()
    except Exception as exc:
        log.warning("cache_clear_all error: %s", exc)

# ── search orchestrator ───────────────────────────────────────────────────────

def run_search(query: str, backends: list, limit: int = 10, fresh: bool = False):
    """
    Orchestrate multi-backend search, deduplication, ranking, caching.
    Returns (results, was_cached, latency_ms, backends_used).
    """
    t0    = time.time()
    qhash = query_hash(query)

    # Determine which backends are actually available
    available = []
    if "google" in backends and GOOGLE_CSE_API_KEY and GOOGLE_CSE_ID:
        available.append("google")
    if "bing" in backends and BING_SEARCH_API_KEY:
        available.append("bing")
    if "crawlbase" in backends and CRAWLBASE_NORMAL_TOKEN:
        available.append("crawlbase")

    # Cache lookup (skip if fresh=True)
    if not fresh:
        cached = cache_get(qhash)
        if cached:
            latency = round((time.time() - t0) * 1000, 2)
            results = json.loads(cached["results"])[:limit]
            log.info("Cache HIT for '%s' (hash=%s)", query, qhash)
            _record_search(query, "cache", len(results), 1, latency)
            return results, True, latency, json.loads(cached["backends_used"])

    if not available:
        latency = round((time.time() - t0) * 1000, 2)
        return [], False, latency, []

    # Fan out to backends
    all_raw = []
    threads  = []
    lock     = threading.Lock()

    def _fetch(backend):
        if backend == "google":
            res = search_google(query)
        elif backend == "bing":
            res = search_bing(query)
        else:
            res = search_crawlbase(query)
        with lock:
            all_raw.extend(res)

    for b in available:
        t = threading.Thread(target=_fetch, args=(b,), daemon=True)
        threads.append(t)
        t.start()
    for t in threads:
        t.join(timeout=25)

    ranked = deduplicate_and_rank(all_raw, limit=limit)

    # Persist to cache
    cache_set(qhash, query, ranked, available)

    # Update domain table
    for item in ranked:
        upsert_domain(item["domain"])

    latency = round((time.time() - t0) * 1000, 2)
    _record_search(query, ",".join(available), len(ranked), 0, latency)

    return ranked, False, latency, available


def _record_search(query: str, backend: str, count: int, cached: int, latency_ms: float):
    try:
        conn = db_connect()
        conn.execute(
            "INSERT INTO searches (query, backend, results_count, cached, latency_ms, searched_at) "
            "VALUES (?,?,?,?,?,?)",
            (query, backend, count, cached, latency_ms, time.time()),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        log.warning("_record_search error: %s", exc)

# ── background maintenance ────────────────────────────────────────────────────

def _maintenance_loop():
    while True:
        time.sleep(MAINTENANCE_INTERVAL)
        try:
            conn = db_connect()
            # Expire stale cache entries
            deleted = conn.execute(
                "DELETE FROM cache WHERE expires_at < ?", (time.time(),)
            ).rowcount
            # Recalculate domain trust scores based on result frequency
            # trust = clamp(0.1..0.9) proportional to log(result_count + 1) / log(max + 1)
            rows = conn.execute(
                "SELECT domain, result_count FROM domains WHERE block_listed=0"
            ).fetchall()
            if rows:
                max_count = max(r["result_count"] for r in rows) or 1
                for row in rows:
                    score = 0.1 + 0.8 * (
                        math.log1p(row["result_count"]) / math.log1p(max_count)
                    )
                    conn.execute(
                        "UPDATE domains SET trust_score=? WHERE domain=?",
                        (round(score, 4), row["domain"]),
                    )
            conn.commit()
            conn.close()
            log.info(
                "Maintenance: expired %d cache entries, updated %d domain scores",
                deleted,
                len(rows) if rows else 0,
            )
        except Exception as exc:
            log.warning("Maintenance error: %s", exc)


# ── stats helpers ─────────────────────────────────────────────────────────────

def _total_searches() -> int:
    try:
        conn = db_connect()
        n = conn.execute("SELECT COUNT(*) FROM searches").fetchone()[0]
        conn.close()
        return n
    except Exception:
        return 0


def _cache_hit_rate() -> float:
    try:
        conn = db_connect()
        total  = conn.execute("SELECT COUNT(*) FROM searches").fetchone()[0]
        hits   = conn.execute("SELECT COUNT(*) FROM searches WHERE cached=1").fetchone()[0]
        conn.close()
        return round(hits / total, 4) if total else 0.0
    except Exception:
        return 0.0

# ── admin auth helper ─────────────────────────────────────────────────────────

def _is_admin(handler) -> bool:
    """Check Authorization header or X-Admin-Secret against ADMIN_SECRET."""
    if not ADMIN_SECRET:
        return True  # no secret configured → open
    auth = handler.headers.get("Authorization", "")
    secret_header = handler.headers.get("X-Admin-Secret", "")
    if secret_header == ADMIN_SECRET:
        return True
    if auth.startswith("Bearer ") and auth[7:] == ADMIN_SECRET:
        return True
    return False

# ── HTTP handler ──────────────────────────────────────────────────────────────

class SearchHandler(BaseHTTPRequestHandler):
    server_version = "FractalMesh-SearchEngine/1.0"

    def log_message(self, fmt, *args):  # suppress default access log noise
        log.debug(fmt, *args)

    # ── routing ────────────────────────────────────────────────────────────────

    def do_GET(self):
        path = self.path
        base = path.split("?")[0].rstrip("/")

        if base == "/health":
            return self._handle_health()
        if base == "/search":
            return self._handle_search()
        if base == "/searches":
            return self._handle_searches()
        if base == "/cache":
            return self._handle_cache_list()
        if re.match(r"^/cache/[a-f0-9]{32}$", base):
            qhash = base.split("/")[-1]
            return self._handle_cache_detail(qhash)
        if base == "/domains":
            return self._handle_domains()
        json_response(self, 404, {"error": "not found"})

    def do_POST(self):
        path = self.path
        base = path.split("?")[0].rstrip("/")

        m_block = re.match(r"^/domains/([^/]+)/block$", base)
        m_trust = re.match(r"^/domains/([^/]+)/trust$", base)

        if m_block:
            return self._handle_domain_block(urllib.parse.unquote(m_block.group(1)))
        if m_trust:
            return self._handle_domain_trust(urllib.parse.unquote(m_trust.group(1)))
        json_response(self, 404, {"error": "not found"})

    def do_DELETE(self):
        base = self.path.split("?")[0].rstrip("/")
        if base == "/cache":
            return self._handle_cache_clear()
        json_response(self, 404, {"error": "not found"})

    # ── GET /health ────────────────────────────────────────────────────────────

    def _handle_health(self):
        json_response(self, 200, {
            "status":         "ok",
            "uptime_seconds": round(time.time() - START_TIME, 1),
            "total_searches": _total_searches(),
            "cache_hit_rate": _cache_hit_rate(),
            "backends_configured": {
                "google":    bool(GOOGLE_CSE_API_KEY and GOOGLE_CSE_ID),
                "bing":      bool(BING_SEARCH_API_KEY),
                "crawlbase": bool(CRAWLBASE_NORMAL_TOKEN),
            },
            "port": PORT,
        })

    # ── GET /search ────────────────────────────────────────────────────────────

    def _handle_search(self):
        params = parse_query_string(self.path)
        query  = params.get("q", "").strip()
        if not query:
            return json_response(self, 400, {"error": "query param 'q' is required"})

        raw_backends = params.get("backends", "google,bing,crawlbase")
        backends     = [b.strip().lower() for b in raw_backends.split(",") if b.strip()]
        valid_set    = {"google", "bing", "crawlbase"}
        backends     = [b for b in backends if b in valid_set] or list(valid_set)

        try:
            limit = int(params.get("limit", "10"))
            limit = max(1, min(limit, 50))
        except (ValueError, TypeError):
            limit = 10

        fresh = params.get("fresh", "").lower() in ("1", "true", "yes")

        results, was_cached, latency_ms, backends_used = run_search(
            query, backends, limit=limit, fresh=fresh
        )
        json_response(self, 200, {
            "query":         query,
            "results":       results,
            "cached":        was_cached,
            "latency_ms":    latency_ms,
            "backends_used": backends_used,
            "result_count":  len(results),
        })

    # ── GET /searches ──────────────────────────────────────────────────────────

    def _handle_searches(self):
        params  = parse_query_string(self.path)
        backend = params.get("backend", "")
        try:
            limit = int(params.get("limit", "50"))
            limit = max(1, min(limit, 500))
        except (ValueError, TypeError):
            limit = 50
        try:
            conn = db_connect()
            if backend:
                rows = conn.execute(
                    "SELECT * FROM searches WHERE backend LIKE ? ORDER BY searched_at DESC LIMIT ?",
                    (f"%{backend}%", limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM searches ORDER BY searched_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            conn.close()
            json_response(self, 200, {
                "searches": [dict(r) for r in rows],
                "count":    len(rows),
            })
        except Exception as exc:
            json_response(self, 500, {"error": str(exc)})

    # ── GET /cache ────────────────────────────────────────────────────────────

    def _handle_cache_list(self):
        params = parse_query_string(self.path)
        try:
            limit = int(params.get("limit", "50"))
            limit = max(1, min(limit, 500))
        except (ValueError, TypeError):
            limit = 50
        try:
            conn = db_connect()
            rows = conn.execute(
                "SELECT id, query_hash, query, result_count, backends_used, "
                "cached_at, expires_at, hit_count FROM cache "
                "ORDER BY cached_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            conn.close()
            entries = []
            now = time.time()
            for r in rows:
                d = dict(r)
                d["expired"]  = d["expires_at"] < now
                d["ttl_left"] = max(0, round(d["expires_at"] - now))
                entries.append(d)
            json_response(self, 200, {"cache": entries, "count": len(entries)})
        except Exception as exc:
            json_response(self, 500, {"error": str(exc)})

    # ── GET /cache/{query_hash} ───────────────────────────────────────────────

    def _handle_cache_detail(self, qhash: str):
        try:
            conn = db_connect()
            row  = conn.execute(
                "SELECT * FROM cache WHERE query_hash=?", (qhash,)
            ).fetchone()
            conn.close()
            if not row:
                return json_response(self, 404, {"error": "cache entry not found"})
            d = dict(row)
            d["results"]       = json.loads(d.get("results", "[]"))
            d["backends_used"] = json.loads(d.get("backends_used", "[]"))
            d["expired"]       = d["expires_at"] < time.time()
            d["ttl_left"]      = max(0, round(d["expires_at"] - time.time()))
            json_response(self, 200, d)
        except Exception as exc:
            json_response(self, 500, {"error": str(exc)})

    # ── GET /domains ──────────────────────────────────────────────────────────

    def _handle_domains(self):
        params = parse_query_string(self.path)
        try:
            min_trust    = float(params.get("min_trust", "0"))
        except (ValueError, TypeError):
            min_trust = 0.0
        show_blocked = params.get("block_listed", "").lower() in ("1", "true", "yes", "")
        try:
            conn = db_connect()
            if params.get("block_listed") in ("1", "true", "yes"):
                rows = conn.execute(
                    "SELECT * FROM domains WHERE block_listed=1 AND trust_score >= ? "
                    "ORDER BY result_count DESC LIMIT 200",
                    (min_trust,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM domains WHERE trust_score >= ? "
                    "ORDER BY result_count DESC LIMIT 200",
                    (min_trust,),
                ).fetchall()
            conn.close()
            json_response(self, 200, {
                "domains": [dict(r) for r in rows],
                "count":   len(rows),
            })
        except Exception as exc:
            json_response(self, 500, {"error": str(exc)})

    # ── POST /domains/{domain}/block ──────────────────────────────────────────

    def _handle_domain_block(self, domain: str):
        if not _is_admin(self):
            return json_response(self, 403, {"error": "forbidden"})
        try:
            conn = db_connect()
            conn.execute("""
                INSERT INTO domains (domain, trust_score, block_listed, result_count, last_seen)
                VALUES (?, 0.0, 1, 0, ?)
                ON CONFLICT(domain) DO UPDATE SET block_listed=1, trust_score=0.0
            """, (domain, time.time()))
            conn.commit()
            conn.close()
            json_response(self, 200, {"ok": True, "domain": domain, "block_listed": True})
        except Exception as exc:
            json_response(self, 500, {"error": str(exc)})

    # ── POST /domains/{domain}/trust ──────────────────────────────────────────

    def _handle_domain_trust(self, domain: str):
        if not _is_admin(self):
            return json_response(self, 403, {"error": "forbidden"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body   = json.loads(self.rfile.read(length).decode()) if length else {}
        except Exception:
            return json_response(self, 400, {"error": "invalid JSON body"})
        score = body.get("score")
        if score is None:
            return json_response(self, 400, {"error": "body must contain {score}"})
        try:
            score = float(score)
            score = max(0.0, min(1.0, score))
        except (ValueError, TypeError):
            return json_response(self, 400, {"error": "score must be a float 0.0–1.0"})
        try:
            conn = db_connect()
            conn.execute("""
                INSERT INTO domains (domain, trust_score, block_listed, result_count, last_seen)
                VALUES (?, ?, 0, 0, ?)
                ON CONFLICT(domain) DO UPDATE SET trust_score=excluded.trust_score
            """, (domain, score, time.time()))
            conn.commit()
            conn.close()
            json_response(self, 200, {"ok": True, "domain": domain, "trust_score": score})
        except Exception as exc:
            json_response(self, 500, {"error": str(exc)})

    # ── DELETE /cache ─────────────────────────────────────────────────────────

    def _handle_cache_clear(self):
        if not _is_admin(self):
            return json_response(self, 403, {"error": "forbidden"})
        cache_clear_all()
        json_response(self, 200, {"ok": True, "message": "cache cleared"})

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    db_init()

    # Start background maintenance thread
    maint = threading.Thread(target=_maintenance_loop, daemon=True)
    maint.start()
    log.info("Maintenance thread started (interval=%ds)", MAINTENANCE_INTERVAL)

    # Determine which backends are ready at startup
    ready = []
    if GOOGLE_CSE_API_KEY and GOOGLE_CSE_ID:
        ready.append("google")
    if BING_SEARCH_API_KEY:
        ready.append("bing")
    if CRAWLBASE_NORMAL_TOKEN:
        ready.append("crawlbase")

    log.info(
        "Sovereign Search Engine starting on port %d | backends ready: %s",
        PORT,
        ready or ["none — set API keys in ~/.secrets/fractal.env"],
    )

    server = HTTPServer(("0.0.0.0", PORT), SearchHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
