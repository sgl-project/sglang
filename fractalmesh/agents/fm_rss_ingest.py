#!/usr/bin/env python3
"""
fm_rss_ingest.py — RSS / Atom Feed Ingestion Pipeline (Port 7908)
Polls registered feeds, parses items, stores in sovereign.db, deduplicates
by GUID/URL hash, and exposes a REST API for consumers.
Credentials from ~/.secrets/fractal.env — never hardcoded.
Samuel James Hiotis | ABN 56 628 117 363
"""
import hashlib
import hmac
import json
import os
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

# ── vault ─────────────────────────────────────────────────────────────────────
_VAULT = Path.home() / ".secrets" / "fractal.env"
if _VAULT.exists():
    for _ln in _VAULT.read_text().splitlines():
        if "=" in _ln and not _ln.startswith("#"):
            _k, _, _v = _ln.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

PORT         = int(os.getenv("RSS_INGEST_PORT", "7908"))
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")
POLL_INTERVAL= int(os.getenv("RSS_POLL_INTERVAL", "900"))  # 15 min default
ROOT         = Path(os.getenv("FRACTALMESH_HOME", str(Path.home() / "fmsaas")))
DB           = ROOT / "database" / "sovereign.db"
ROOT.mkdir(parents=True, exist_ok=True)
DB.parent.mkdir(parents=True, exist_ok=True)

# ── DB ─────────────────────────────────────────────────────────────────────────
def _init_db():
    con = sqlite3.connect(str(DB), timeout=10)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript("""
        CREATE TABLE IF NOT EXISTS rss_feeds (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            url          TEXT UNIQUE NOT NULL,
            title        TEXT,
            category     TEXT NOT NULL DEFAULT 'general',
            active       INTEGER NOT NULL DEFAULT 1,
            last_polled  REAL,
            poll_count   INTEGER NOT NULL DEFAULT 0,
            error_count  INTEGER NOT NULL DEFAULT 0,
            created_at   REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS rss_items (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            feed_id      INTEGER NOT NULL,
            guid_hash    TEXT UNIQUE NOT NULL,
            title        TEXT,
            link         TEXT,
            summary      TEXT,
            author       TEXT,
            published    TEXT,
            category     TEXT,
            raw_content  TEXT,
            ingested_at  REAL NOT NULL,
            FOREIGN KEY(feed_id) REFERENCES rss_feeds(id)
        );
        CREATE INDEX IF NOT EXISTS idx_rss_items_feed  ON rss_items(feed_id);
        CREATE INDEX IF NOT EXISTS idx_rss_items_date  ON rss_items(ingested_at);
    """)
    # Seed default tech/AI feeds
    seeds = [
        ("https://feeds.feedburner.com/oreilly/radar/atom", "O'Reilly Radar", "tech"),
        ("https://techcrunch.com/feed/", "TechCrunch", "tech"),
        ("https://hnrss.org/frontpage", "Hacker News Front Page", "tech"),
    ]
    for url, title, cat in seeds:
        try:
            con.execute(
                "INSERT OR IGNORE INTO rss_feeds(url,title,category,created_at) VALUES(?,?,?,?)",
                (url, title, cat, time.time())
            )
        except Exception:
            pass
    con.commit()
    con.close()

def _db():
    con = sqlite3.connect(str(DB), check_same_thread=False, timeout=10)
    con.execute("PRAGMA journal_mode=WAL")
    con.row_factory = sqlite3.Row
    return con

# ── RSS parser ─────────────────────────────────────────────────────────────────
NS = {
    "atom":    "http://www.w3.org/2005/Atom",
    "content": "http://purl.org/rss/1.0/modules/content/",
    "dc":      "http://purl.org/dc/elements/1.1/",
}

def _text(el, *paths):
    for path in paths:
        node = el.find(path, NS)
        if node is not None and node.text:
            return node.text.strip()
    return ""

def _parse_feed(xml_bytes: bytes, feed_id: int) -> list:
    items = []
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return items

    ns_atom = root.tag.startswith("{http://www.w3.org/2005/Atom}")
    if ns_atom:
        entries = root.findall("atom:entry", NS)
        for e in entries:
            link_el = e.find("atom:link[@rel='alternate']", NS) or e.find("atom:link", NS)
            link = link_el.get("href", "") if link_el is not None else ""
            guid = _text(e, "atom:id") or link
            items.append({
                "feed_id":   feed_id,
                "guid_hash": hashlib.sha256(guid.encode()).hexdigest(),
                "title":     _text(e, "atom:title"),
                "link":      link,
                "summary":   _text(e, "atom:summary", "atom:content"),
                "author":    _text(e, "atom:author/atom:name"),
                "published": _text(e, "atom:published", "atom:updated"),
                "category":  _text(e, "atom:category"),
            })
    else:
        channel = root.find("channel")
        if channel is None:
            channel = root
        for item in channel.findall("item"):
            link = _text(item, "link")
            guid = _text(item, "guid") or link
            items.append({
                "feed_id":   feed_id,
                "guid_hash": hashlib.sha256(guid.encode()).hexdigest(),
                "title":     _text(item, "title"),
                "link":      link,
                "summary":   _text(item, "description", "content:encoded"),
                "author":    _text(item, "author", "dc:creator"),
                "published": _text(item, "pubDate"),
                "category":  _text(item, "category"),
            })
    return items[:50]

def _poll_feed(row):
    feed_id = row["id"]
    url     = row["url"]
    try:
        req = Request(url, headers={"User-Agent": "FractalMesh-RSS/1.0"})
        with urlopen(req, timeout=15) as r:
            xml_bytes = r.read(1_000_000)
        items = _parse_feed(xml_bytes, feed_id)
        con   = _db()
        saved = 0
        now   = time.time()
        for it in items:
            try:
                con.execute(
                    "INSERT OR IGNORE INTO rss_items(feed_id,guid_hash,title,link,summary,author,published,category,ingested_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?)",
                    (it["feed_id"], it["guid_hash"], it["title"][:500], it["link"][:1000],
                     it["summary"][:2000], it["author"][:200], it["published"][:100],
                     it["category"][:100], now)
                )
                if con.execute("SELECT changes()").fetchone()[0]:
                    saved += 1
            except Exception:
                pass
        con.execute(
            "UPDATE rss_feeds SET last_polled=?, poll_count=poll_count+1 WHERE id=?",
            (now, feed_id)
        )
        con.commit()
        con.close()
        return saved
    except Exception as exc:
        try:
            con = _db()
            con.execute("UPDATE rss_feeds SET error_count=error_count+1 WHERE id=?", (feed_id,))
            con.commit()
            con.close()
        except Exception:
            pass
        return 0

def _poll_daemon():
    time.sleep(30)
    while True:
        try:
            con = _db()
            feeds = con.execute(
                "SELECT * FROM rss_feeds WHERE active=1 AND (last_polled IS NULL OR last_polled < ?)",
                (time.time() - POLL_INTERVAL,)
            ).fetchall()
            con.close()
            for row in feeds:
                _poll_feed(row)
        except Exception:
            pass
        time.sleep(60)

threading.Thread(target=_poll_daemon, daemon=True).start()

# ── helpers ────────────────────────────────────────────────────────────────────
def _admin(headers) -> bool:
    if not ADMIN_SECRET:
        return True
    return hmac.compare_digest(headers.get("X-Admin-Secret", ""), ADMIN_SECRET)

def _j(data, code=200):
    return code, json.dumps(data, default=str).encode()

def _err(msg, code=400):
    return _j({"error": msg}, code)

# ── HTTP handler ───────────────────────────────────────────────────────────────
class RSSHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type,X-Admin-Secret")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        p = parsed.path.strip("/").split("/")
        qs = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        try:
            code, body = self._get(p, qs)
        except Exception as e:
            code, body = _err(str(e), 500)
        self._send(code, body)

    def _get(self, p, qs):
        con = _db()
        try:
            if p in (["health"], [""]):
                feed_count = con.execute("SELECT COUNT(*) FROM rss_feeds WHERE active=1").fetchone()[0]
                item_count = con.execute("SELECT COUNT(*) FROM rss_items").fetchone()[0]
                return _j({"status": "ok", "port": PORT, "feeds": feed_count, "items": item_count})

            if p == ["feeds"]:
                rows = con.execute("SELECT * FROM rss_feeds ORDER BY title").fetchall()
                return _j([dict(r) for r in rows])

            if p == ["items"]:
                limit  = min(int(qs.get("limit", 50)), 200)
                offset = int(qs.get("offset", 0))
                cat    = qs.get("category")
                if cat:
                    rows = con.execute(
                        "SELECT i.*, f.title as feed_title, f.category as feed_category "
                        "FROM rss_items i JOIN rss_feeds f ON i.feed_id=f.id "
                        "WHERE f.category=? ORDER BY i.ingested_at DESC LIMIT ? OFFSET ?",
                        (cat, limit, offset)
                    ).fetchall()
                else:
                    rows = con.execute(
                        "SELECT i.*, f.title as feed_title, f.category as feed_category "
                        "FROM rss_items i JOIN rss_feeds f ON i.feed_id=f.id "
                        "ORDER BY i.ingested_at DESC LIMIT ? OFFSET ?",
                        (limit, offset)
                    ).fetchall()
                return _j([dict(r) for r in rows])

            return _err("not found", 404)
        finally:
            con.close()

    def do_POST(self):
        p = self.path.strip("/").split("/")
        try:
            n = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(n) if n else b""
            data = json.loads(raw) if raw else {}
            code, body = self._post(p, data)
        except Exception as e:
            code, body = _err(str(e), 500)
        self._send(code, body)

    def _post(self, p, data):
        if p == ["feeds"]:
            if not _admin(self.headers):
                return _err("Unauthorized", 403)
            url = data.get("url", "").strip()
            if not url:
                return _err("url required")
            cat   = data.get("category", "general")
            title = data.get("title", url)
            con   = _db()
            try:
                con.execute(
                    "INSERT OR IGNORE INTO rss_feeds(url,title,category,created_at) VALUES(?,?,?,?)",
                    (url, title, cat, time.time())
                )
                con.commit()
                row = con.execute("SELECT * FROM rss_feeds WHERE url=?", (url,)).fetchone()
                return _j(dict(row), 201)
            finally:
                con.close()

        if p[0] == "feeds" and len(p) == 3 and p[2] == "poll":
            if not _admin(self.headers):
                return _err("Unauthorized", 403)
            feed_id = int(p[1])
            con = _db()
            try:
                row = con.execute("SELECT * FROM rss_feeds WHERE id=?", (feed_id,)).fetchone()
                if not row:
                    return _err("feed not found", 404)
            finally:
                con.close()
            saved = _poll_feed(row)
            return _j({"polled": True, "feed_id": feed_id, "new_items": saved})

        return _err("not found", 404)

    def do_DELETE(self):
        p = self.path.strip("/").split("/")
        if not _admin(self.headers):
            body = json.dumps({"error": "Unauthorized"}).encode()
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("WWW-Authenticate", 'X-Admin-Secret realm="FractalMesh"')
            self.end_headers()
            self.wfile.write(body)
            return
        try:
            if p[0] == "feeds" and len(p) == 2:
                feed_id = int(p[1])
                con = _db()
                con.execute("UPDATE rss_feeds SET active=0 WHERE id=?", (feed_id,))
                con.commit()
                con.close()
                code, body = _j({"deleted": True, "feed_id": feed_id})
            else:
                code, body = _err("not found", 404)
        except Exception as e:
            code, body = _err(str(e), 500)
        self._send(code, body)


def run():
    _init_db()
    server = HTTPServer(("0.0.0.0", PORT), RSSHandler)
    print(f"[fm_rss_ingest] listening on port {PORT}")
    server.serve_forever()


if __name__ == "__main__":
    run()
