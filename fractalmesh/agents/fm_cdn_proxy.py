#!/usr/bin/env python3
"""
fm_cdn_proxy.py — CDN Cache Proxy & Edge Accelerator (Port 7901)
FractalMesh OMEGA Titan | Samuel James Hiotis | ABN 56 628 117 363
Credentials sourced from ~/.secrets/fractal.env — never hardcoded.
"""
import os
import json
import sqlite3
import time
import hashlib
import hmac
import secrets
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs, urlencode
import urllib.request
import urllib.error

# ── vault ─────────────────────────────────────────────────────────────────────
_ENV_FILE = Path.home() / ".secrets" / "fractal.env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

PORT         = int(os.getenv("CDN_PROXY_PORT", "7901"))
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")
CF_ZONE_ID   = os.getenv("CLOUDFLARE_ZONE_ID", "")
CF_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "")

ROOT = Path(os.getenv("FRACTALMESH_HOME", str(Path.home() / "fmsaas")))
DB   = ROOT / "database" / "sovereign.db"
CACHE_DIR = ROOT / "cdn_cache"
ROOT.mkdir(parents=True, exist_ok=True)
DB.parent.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_CACHE_TTL = 3600  # 1 hour default
_cache_lock = threading.Lock()
_mem_cache = {}  # url_hash -> (body, headers, expires)

def _db():
    con = sqlite3.connect(str(DB), check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")
    con.row_factory = sqlite3.Row
    return con

def init_db():
    con = _db()
    con.executescript("""
        CREATE TABLE IF NOT EXISTS cdn_origins (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            origin_id    TEXT UNIQUE NOT NULL,
            name         TEXT NOT NULL,
            base_url     TEXT NOT NULL,
            ttl_seconds  INTEGER NOT NULL DEFAULT 3600,
            headers      TEXT NOT NULL DEFAULT '{}',
            active       INTEGER NOT NULL DEFAULT 1,
            created_at   REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS cdn_cache_entries (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            url_hash     TEXT UNIQUE NOT NULL,
            origin_url   TEXT NOT NULL,
            content_type TEXT NOT NULL DEFAULT 'text/plain',
            size_bytes   INTEGER NOT NULL DEFAULT 0,
            hit_count    INTEGER NOT NULL DEFAULT 0,
            cached_at    REAL NOT NULL,
            expires_at   REAL NOT NULL,
            status_code  INTEGER NOT NULL DEFAULT 200
        );
        CREATE TABLE IF NOT EXISTS cdn_rules (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            rule_id      TEXT UNIQUE NOT NULL,
            pattern      TEXT NOT NULL,
            ttl_seconds  INTEGER NOT NULL DEFAULT 3600,
            cache_control TEXT,
            priority     INTEGER NOT NULL DEFAULT 0,
            active       INTEGER NOT NULL DEFAULT 1,
            created_at   REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS cdn_stats (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            date         TEXT NOT NULL,
            hits         INTEGER NOT NULL DEFAULT 0,
            misses       INTEGER NOT NULL DEFAULT 0,
            bytes_served INTEGER NOT NULL DEFAULT 0,
            UNIQUE(date)
        );
        CREATE INDEX IF NOT EXISTS idx_cdn_cache_expires ON cdn_cache_entries(expires_at);
    """)
    con.commit()
    _seed_rules(con)
    con.close()

def _seed_rules(con):
    if con.execute("SELECT COUNT(*) FROM cdn_rules").fetchone()[0] > 0:
        return
    now = time.time()
    rules = [
        ("rule_static", r".*\.(css|js|png|jpg|gif|ico|woff2?)$", 86400, "public, max-age=86400", 10),
        ("rule_api",    r"^/api/",                                 60,    "no-cache",               5),
        ("rule_html",   r".*\.html?$",                             300,   "public, max-age=300",    1),
    ]
    for rid, pattern, ttl, cc, prio in rules:
        con.execute(
            "INSERT INTO cdn_rules(rule_id,pattern,ttl_seconds,cache_control,priority,created_at) VALUES(?,?,?,?,?,?)",
            (rid, pattern, ttl, cc, prio, now)
        )
    con.commit()

def _url_hash(url):
    return hashlib.sha256(url.encode()).hexdigest()

def _get_cached(url):
    h = _url_hash(url)
    now = time.time()
    with _cache_lock:
        if h in _mem_cache:
            body, ct, exp = _mem_cache[h]
            if exp > now:
                return body, ct, True
            del _mem_cache[h]
    # check disk cache
    cache_file = CACHE_DIR / f"{h}.bin"
    meta_file  = CACHE_DIR / f"{h}.json"
    if cache_file.exists() and meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text())
            if meta["expires_at"] > now:
                body = cache_file.read_bytes()
                with _cache_lock:
                    _mem_cache[h] = (body, meta["content_type"], meta["expires_at"])
                return body, meta["content_type"], True
        except Exception:
            pass
    return None, None, False

def _set_cached(url, body, content_type, ttl):
    h = _url_hash(url)
    expires = time.time() + ttl
    with _cache_lock:
        _mem_cache[h] = (body, content_type, expires)
    cache_file = CACHE_DIR / f"{h}.bin"
    meta_file  = CACHE_DIR / f"{h}.json"
    try:
        cache_file.write_bytes(body)
        meta_file.write_text(json.dumps({
            "url": url, "content_type": content_type,
            "expires_at": expires, "cached_at": time.time(), "size": len(body)
        }))
    except Exception:
        pass

def _fetch_origin(url, extra_headers=None):
    req = urllib.request.Request(url)
    if extra_headers:
        for k, v in extra_headers.items():
            req.add_header(k, v)
    req.add_header("User-Agent", "FractalMesh-CDN/1.0")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            body = r.read()
            ct = r.headers.get("Content-Type", "application/octet-stream")
            status = r.status
            return body, ct, status
    except urllib.error.HTTPError as e:
        return e.read(), "text/plain", e.code
    except Exception as e:
        return str(e).encode(), "text/plain", 502

def _record_stat(hit):
    try:
        con = _db()
        date = time.strftime("%Y-%m-%d")
        if hit:
            con.execute("INSERT INTO cdn_stats(date,hits,misses) VALUES(?,1,0) ON CONFLICT(date) DO UPDATE SET hits=hits+1", (date,))
        else:
            con.execute("INSERT INTO cdn_stats(date,hits,misses) VALUES(?,0,1) ON CONFLICT(date) DO UPDATE SET misses=misses+1", (date,))
        con.commit()
        con.close()
    except Exception:
        pass

def _eviction_daemon():
    while True:
        time.sleep(600)
        try:
            now = time.time()
            for f in CACHE_DIR.glob("*.json"):
                try:
                    meta = json.loads(f.read_text())
                    if meta["expires_at"] < now:
                        f.unlink(missing_ok=True)
                        (CACHE_DIR / f"{f.stem}.bin").unlink(missing_ok=True)
                except Exception:
                    pass
            with _cache_lock:
                expired = [k for k, v in _mem_cache.items() if v[2] < now]
                for k in expired:
                    del _mem_cache[k]
        except Exception:
            pass

threading.Thread(target=_eviction_daemon, daemon=True).start()

def _j(data, status=200):
    return status, json.dumps(data, default=str).encode()

def _err(msg, code=400):
    return _j({"error": msg}, code)

def _admin(h):
    v = h.get("X-Admin-Secret", "")
    return not ADMIN_SECRET or hmac.compare_digest(v, ADMIN_SECRET)

class CDNHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _read_body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n)) if n else {}

    def _send_json(self, code, body):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_content(self, code, body, ct, cache_hit):
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Cache", "HIT" if cache_hit else "MISS")
        self.send_header("X-Served-By", "FractalMesh-CDN")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type,X-Admin-Secret,X-Origin-Url")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        p = parsed.path.strip("/").split("/")
        qs = parse_qs(parsed.query)
        try:
            if p == ["health"]:
                code, body = _j({"status": "ok", "port": PORT, "agent": "fm_cdn_proxy",
                                  "cache_entries": len(_mem_cache)})
                self._send_json(code, body)
                return

            if p == ["origins"]:
                if not _admin(self.headers):
                    self._send_json(*_err("Unauthorized", 403)); return
                con = _db()
                rows = con.execute("SELECT * FROM cdn_origins WHERE active=1").fetchall()
                con.close()
                self._send_json(*_j([dict(r) for r in rows]))
                return

            if p == ["rules"]:
                if not _admin(self.headers):
                    self._send_json(*_err("Unauthorized", 403)); return
                con = _db()
                rows = con.execute("SELECT * FROM cdn_rules WHERE active=1 ORDER BY priority DESC").fetchall()
                con.close()
                self._send_json(*_j([dict(r) for r in rows]))
                return

            if p == ["stats"]:
                if not _admin(self.headers):
                    self._send_json(*_err("Unauthorized", 403)); return
                con = _db()
                rows = con.execute("SELECT * FROM cdn_stats ORDER BY date DESC LIMIT 30").fetchall()
                total_entries = con.execute("SELECT COUNT(*) FROM cdn_cache_entries").fetchone()[0]
                total_size = con.execute("SELECT COALESCE(SUM(size_bytes),0) FROM cdn_cache_entries").fetchone()[0]
                con.close()
                self._send_json(*_j({
                    "daily": [dict(r) for r in rows],
                    "total_cached_entries": total_entries,
                    "total_cached_bytes": total_size,
                    "mem_cache_entries": len(_mem_cache),
                }))
                return

            # proxy request: /proxy?url=https://...
            if p == ["proxy"]:
                url = qs.get("url", [None])[0]
                if not url:
                    self._send_json(*_err("url parameter required")); return
                ttl = int(qs.get("ttl", [str(_CACHE_TTL)])[0])
                body, ct, hit = _get_cached(url)
                if hit:
                    _record_stat(True)
                    self._send_content(200, body, ct, True)
                    return
                body, ct, status = _fetch_origin(url)
                if status == 200:
                    _set_cached(url, body, ct, ttl)
                _record_stat(False)
                self._send_content(status, body, ct, False)
                return

            self._send_json(*_err("Not found", 404))
        except Exception as e:
            self._send_json(*_err(str(e), 500))

    def do_POST(self):
        p = self.path.strip("/").split("/")
        try:
            data = self._read_body()
            code, body = self._post(p, data)
        except Exception as e:
            code, body = _err(str(e), 500)
        self._send_json(code, body)

    def do_DELETE(self):
        p = self.path.strip("/").split("/")
        try:
            if not _admin(self.headers):
                self._send_json(*_err("Unauthorized", 403)); return
            if p == ["cache"]:
                with _cache_lock:
                    _mem_cache.clear()
                for f in CACHE_DIR.glob("*"):
                    f.unlink(missing_ok=True)
                self._send_json(*_j({"cleared": True}))
                return
            if len(p) == 2 and p[0] == "cache":
                h = _url_hash(p[1]) if not p[1].startswith("sha:") else p[1][4:]
                with _cache_lock:
                    _mem_cache.pop(h, None)
                (CACHE_DIR / f"{h}.bin").unlink(missing_ok=True)
                (CACHE_DIR / f"{h}.json").unlink(missing_ok=True)
                self._send_json(*_j({"deleted": h}))
                return
            self._send_json(*_err("Not found", 404))
        except Exception as e:
            self._send_json(*_err(str(e), 500))

    def _post(self, p, data):
        con = _db()
        now = time.time()
        try:
            if p == ["origins"]:
                if not _admin(self.headers):
                    return _err("Unauthorized", 403)
                oid = "orig_" + secrets.token_hex(8)
                con.execute(
                    "INSERT INTO cdn_origins(origin_id,name,base_url,ttl_seconds,headers,created_at) VALUES(?,?,?,?,?,?)",
                    (oid, data.get("name",""), data.get("base_url",""),
                     data.get("ttl_seconds", 3600), json.dumps(data.get("headers",{})), now)
                )
                con.commit()
                return _j({"origin_id": oid}, 201)

            if p == ["rules"]:
                if not _admin(self.headers):
                    return _err("Unauthorized", 403)
                rid = "rule_" + secrets.token_hex(8)
                con.execute(
                    "INSERT INTO cdn_rules(rule_id,pattern,ttl_seconds,cache_control,priority,created_at) VALUES(?,?,?,?,?,?)",
                    (rid, data.get("pattern",".*"), data.get("ttl_seconds",3600),
                     data.get("cache_control","public"), data.get("priority",0), now)
                )
                con.commit()
                return _j({"rule_id": rid}, 201)

            if p == ["purge"]:
                if not _admin(self.headers):
                    return _err("Unauthorized", 403)
                urls = data.get("urls", [])
                purged = []
                for url in urls:
                    h = _url_hash(url)
                    with _cache_lock:
                        _mem_cache.pop(h, None)
                    (CACHE_DIR / f"{h}.bin").unlink(missing_ok=True)
                    (CACHE_DIR / f"{h}.json").unlink(missing_ok=True)
                    purged.append(url)
                # Also purge from Cloudflare if configured
                if CF_ZONE_ID and CF_API_TOKEN and urls:
                    try:
                        payload = json.dumps({"files": urls[:30]}).encode()
                        req = urllib.request.Request(
                            f"https://api.cloudflare.com/client/v4/zones/{CF_ZONE_ID}/purge_cache",
                            data=payload, method="POST"
                        )
                        req.add_header("Authorization", f"Bearer {CF_API_TOKEN}")
                        req.add_header("Content-Type", "application/json")
                        urllib.request.urlopen(req, timeout=10)
                    except Exception:
                        pass
                return _j({"purged": len(purged), "urls": purged})

            if p == ["warm"]:
                if not _admin(self.headers):
                    return _err("Unauthorized", 403)
                urls = data.get("urls", [])
                ttl = data.get("ttl", _CACHE_TTL)
                def _warm():
                    for url in urls[:50]:
                        body, ct, status = _fetch_origin(url)
                        if status == 200:
                            _set_cached(url, body, ct, ttl)
                threading.Thread(target=_warm, daemon=True).start()
                return _j({"warming": len(urls[:50])})

            return _err("Not found", 404)
        finally:
            con.close()


def run():
    init_db()
    server = HTTPServer(("0.0.0.0", PORT), CDNHandler)
    print(f"[fm_cdn_proxy] listening on port {PORT}")
    server.serve_forever()


if __name__ == "__main__":
    run()
