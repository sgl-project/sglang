#!/usr/bin/env python3
"""
FractalMesh OMEGA Titan — Podcast & Audio Content Engine
Port: 7881

Full podcast management: create shows, generate episodes (text or AI TTS via
ElevenLabs), publish RSS 2.0 feeds with iTunes namespace, manage subscribers,
send notifications via SendGrid, store audio on Cloudflare R2.

Samuel James Hiotis | ABN 56 628 117 363
"""

import os
import json
import sqlite3
import time
import hashlib
import base64
import gzip
import threading
import xml.etree.ElementTree as ET
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# ---------------------------------------------------------------------------
# Vault loading
# ---------------------------------------------------------------------------
_ENV_FILE = Path.home() / ".secrets" / "fractal.env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PORT                = int(os.getenv("PODCAST_ENGINE_PORT", "7881"))
ROOT                = Path(os.getenv("FRACTALMESH_HOME", str(Path.home() / "fmsaas")))
DB_PATH             = ROOT / "database" / "sovereign.db"
LOG_PATH            = ROOT / "logs" / "fm_podcast_engine.log"

ELEVENLABS_API_KEY  = os.getenv("ELEVENLABS_API_KEY", "")
SENDGRID_API_KEY    = os.getenv("SENDGRID_API_KEY", "")
SENDGRID_FROM_EMAIL = os.getenv("SENDGRID_FROM_EMAIL", "")
R2_ACCOUNT_ID       = os.getenv("R2_ACCOUNT_ID", "")
R2_ACCESS_KEY_ID    = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET_NAME      = os.getenv("R2_BUCKET_NAME", "")
R2_PUBLIC_URL       = os.getenv("R2_PUBLIC_URL", "").rstrip("/")
ADMIN_SECRET        = os.getenv("ADMIN_SECRET", "")

DEFAULT_VOICE_ID    = "21m00Tcm4TlvDq8ikWAM"  # Rachel
ELEVENLABS_TTS_URL  = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
SENDGRID_SEND_URL   = "https://api.sendgrid.com/v3/mail/send"
R2_ENDPOINT         = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

for _d in (DB_PATH.parent, LOG_PATH.parent):
    _d.mkdir(parents=True, exist_ok=True)

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PODCAST] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("podcast_engine")

START_TIME = time.time()

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def _db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def _db_init():
    conn = _db_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS shows (
            id               INTEGER PRIMARY KEY,
            show_id          TEXT UNIQUE NOT NULL,
            title            TEXT NOT NULL,
            description      TEXT,
            author           TEXT,
            email            TEXT,
            category         TEXT DEFAULT 'Technology',
            language         TEXT DEFAULT 'en-au',
            image_url        TEXT,
            website          TEXT,
            explicit         INTEGER DEFAULT 0,
            status           TEXT DEFAULT 'active',
            episode_count    INTEGER DEFAULT 0,
            subscriber_count INTEGER DEFAULT 0,
            created_at       REAL
        );

        CREATE TABLE IF NOT EXISTS episodes (
            id               INTEGER PRIMARY KEY,
            episode_id       TEXT UNIQUE NOT NULL,
            show_id          TEXT NOT NULL,
            title            TEXT NOT NULL,
            description      TEXT,
            content          TEXT,
            audio_url        TEXT,
            audio_duration   INTEGER DEFAULT 0,
            audio_size       INTEGER DEFAULT 0,
            season           INTEGER DEFAULT 1,
            episode_number   INTEGER,
            episode_type     TEXT DEFAULT 'full',
            status           TEXT DEFAULT 'draft',
            published_at     REAL,
            scheduled_at     REAL,
            listen_count     INTEGER DEFAULT 0,
            created_at       REAL
        );

        CREATE TABLE IF NOT EXISTS subscribers (
            id            INTEGER PRIMARY KEY,
            email         TEXT NOT NULL,
            show_id       TEXT NOT NULL,
            subscribed_at REAL,
            active        INTEGER DEFAULT 1,
            UNIQUE (email, show_id)
        );

        CREATE TABLE IF NOT EXISTS listens (
            id              INTEGER PRIMARY KEY,
            episode_id      TEXT NOT NULL,
            listener_ip_hash TEXT,
            listen_duration INTEGER DEFAULT 0,
            completed       INTEGER DEFAULT 0,
            listened_at     REAL
        );
    """)
    conn.commit()
    conn.close()
    log.info("Database initialised")


def _seed_shows():
    conn = _db_conn()
    count = conn.execute("SELECT COUNT(*) FROM shows").fetchone()[0]
    if count == 0:
        show_id = _make_id("show")
        conn.execute(
            """INSERT OR IGNORE INTO shows
               (show_id, title, description, author, category, language, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                show_id,
                "FractalMesh Tech Talks",
                "Technical insights on AI automation, sovereign infrastructure, "
                "and the future of autonomous systems.",
                "FractalMesh OMEGA Titan",
                "Technology",
                "en-au",
                "active",
                time.time(),
            ),
        )
        conn.commit()
        log.info("Seeded default show: FractalMesh Tech Talks (%s)", show_id)
    conn.close()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_id(prefix: str = "id") -> str:
    raw = f"{prefix}-{time.time()}-{os.urandom(8).hex()}"
    return f"{prefix}_{hashlib.sha256(raw.encode()).hexdigest()[:16]}"


def _hash_ip(ip: str) -> str:
    return hashlib.sha256(ip.encode()).hexdigest()[:32]


def _now() -> float:
    return time.time()


def _json_row(row) -> dict:
    if row is None:
        return {}
    return dict(row)


def _json_rows(rows) -> list:
    return [dict(r) for r in rows]

# ---------------------------------------------------------------------------
# ElevenLabs TTS
# ---------------------------------------------------------------------------

def _tts_generate(text: str, voice_id: str = DEFAULT_VOICE_ID) -> bytes:
    """Call ElevenLabs TTS and return raw MP3 bytes."""
    if not ELEVENLABS_API_KEY:
        raise RuntimeError("ELEVENLABS_API_KEY not configured")
    url = ELEVENLABS_TTS_URL.format(voice_id=voice_id)
    payload = json.dumps({
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()

# ---------------------------------------------------------------------------
# Cloudflare R2 upload (HMAC-SHA256 / AWS SigV4-lite)
# ---------------------------------------------------------------------------

def _hmac_sha256(key: bytes, msg: bytes) -> bytes:
    import hmac
    return hmac.new(key, msg, "sha256").digest()


def _r2_upload(key: str, data: bytes, content_type: str = "audio/mpeg") -> str:
    """Upload bytes to R2 and return the public URL."""
    if not all([R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME]):
        raise RuntimeError("R2 credentials not fully configured")
    import hmac as _hmac
    import hashlib as _hs

    t = time.gmtime()
    datestamp = time.strftime("%Y%m%d", t)
    amz_date  = time.strftime("%Y%m%dT%H%M%SZ", t)
    host      = f"{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    region    = "auto"
    service   = "s3"

    payload_hash = _hs.sha256(data).hexdigest()
    canonical_headers = (
        f"content-type:{content_type}\n"
        f"host:{host}\n"
        f"x-amz-content-sha256:{payload_hash}\n"
        f"x-amz-date:{amz_date}\n"
    )
    signed_headers = "content-type;host;x-amz-content-sha256;x-amz-date"
    canonical_request = "\n".join([
        "PUT",
        f"/{R2_BUCKET_NAME}/{key}",
        "",
        canonical_headers,
        signed_headers,
        payload_hash,
    ])
    credential_scope = f"{datestamp}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join([
        "AWS4-HMAC-SHA256",
        amz_date,
        credential_scope,
        _hs.sha256(canonical_request.encode()).hexdigest(),
    ])
    signing_key = _hmac_sha256(
        _hmac_sha256(
            _hmac_sha256(
                _hmac_sha256(
                    f"AWS4{R2_SECRET_ACCESS_KEY}".encode(),
                    datestamp.encode(),
                ),
                region.encode(),
            ),
            service.encode(),
        ),
        b"aws4_request",
    )
    signature = _hmac.new(signing_key, string_to_sign.encode(), "sha256").hexdigest()
    auth_header = (
        f"AWS4-HMAC-SHA256 Credential={R2_ACCESS_KEY_ID}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    url = f"{R2_ENDPOINT}/{R2_BUCKET_NAME}/{key}"
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": auth_header,
            "Content-Type": content_type,
            "x-amz-content-sha256": payload_hash,
            "x-amz-date": amz_date,
        },
        method="PUT",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        resp.read()
    public = f"{R2_PUBLIC_URL}/{key}" if R2_PUBLIC_URL else url
    return public

# ---------------------------------------------------------------------------
# SendGrid email
# ---------------------------------------------------------------------------

def _send_email(to: str, subject: str, body_html: str) -> bool:
    if not SENDGRID_API_KEY or not SENDGRID_FROM_EMAIL:
        log.warning("SendGrid not configured; skipping email to %s", to)
        return False
    payload = json.dumps({
        "personalizations": [{"to": [{"email": to}]}],
        "from": {"email": SENDGRID_FROM_EMAIL},
        "subject": subject,
        "content": [{"type": "text/html", "value": body_html}],
    }).encode()
    req = urllib.request.Request(
        SENDGRID_SEND_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {SENDGRID_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status in (200, 202)
    except urllib.error.HTTPError as exc:
        log.error("SendGrid error %s for %s", exc.code, to)
        return False


def _notify_subscribers(show_id: str, episode: dict):
    conn = _db_conn()
    subs = conn.execute(
        "SELECT email FROM subscribers WHERE show_id=? AND active=1", (show_id,)
    ).fetchall()
    show = conn.execute("SELECT * FROM shows WHERE show_id=?", (show_id,)).fetchone()
    conn.close()
    if not subs or not show:
        return
    show = dict(show)
    subject = f"New episode: {episode['title']} — {show['title']}"
    for sub in subs:
        html = (
            f"<h2>{show['title']}</h2>"
            f"<p>A new episode is now available:</p>"
            f"<h3>{episode['title']}</h3>"
            f"<p>{episode.get('description', '')}</p>"
            f"<p><a href='{episode.get('audio_url', '#')}'>Listen now</a></p>"
            f"<hr><small>You are subscribed to {show['title']}.</small>"
        )
        _send_email(sub["email"], subject, html)


def _send_subscribe_confirmation(email: str, show_title: str):
    subject = f"Subscribed to {show_title}"
    html = (
        f"<p>You have successfully subscribed to <strong>{show_title}</strong>.</p>"
        f"<p>You will receive an email each time a new episode is published.</p>"
    )
    _send_email(email, subject, html)

# ---------------------------------------------------------------------------
# RSS feed generation
# ---------------------------------------------------------------------------

def _generate_rss(show_id: str) -> str:
    conn = _db_conn()
    show = conn.execute("SELECT * FROM shows WHERE show_id=?", (show_id,)).fetchone()
    if not show:
        conn.close()
        return ""
    show = dict(show)
    episodes = conn.execute(
        "SELECT * FROM episodes WHERE show_id=? AND status='published' ORDER BY published_at DESC",
        (show_id,),
    ).fetchall()
    conn.close()

    ET.register_namespace("itunes", "http://www.itunes.com/dtds/podcast-1.0.dtd")
    ET.register_namespace("content", "http://purl.org/rss/1.0/modules/content/")

    ITUNES = "http://www.itunes.com/dtds/podcast-1.0.dtd"

    rss = ET.Element("rss", {
        "version": "2.0",
        "xmlns:itunes": ITUNES,
        "xmlns:content": "http://purl.org/rss/1.0/modules/content/",
    })
    channel = ET.SubElement(rss, "channel")

    def _sub(parent, tag, text="", **attrs):
        el = ET.SubElement(parent, tag, attrs)
        if text:
            el.text = str(text)
        return el

    _sub(channel, "title", show["title"])
    _sub(channel, "description", show.get("description") or "")
    _sub(channel, "language", show.get("language") or "en-au")
    _sub(channel, "link", show.get("website") or "https://fractalmesh.net")
    _sub(channel, f"{{{ITUNES}}}author", show.get("author") or "")
    _sub(channel, f"{{{ITUNES}}}explicit", "yes" if show.get("explicit") else "no")

    if show.get("image_url"):
        img = ET.SubElement(channel, f"{{{ITUNES}}}image")
        img.set("href", show["image_url"])
        img2 = ET.SubElement(channel, "image")
        _sub(img2, "url", show["image_url"])
        _sub(img2, "title", show["title"])
        _sub(img2, "link", show.get("website") or "https://fractalmesh.net")

    if show.get("category"):
        cat = ET.SubElement(channel, f"{{{ITUNES}}}category")
        cat.set("text", show["category"])

    if show.get("email"):
        owner = ET.SubElement(channel, f"{{{ITUNES}}}owner")
        _sub(owner, f"{{{ITUNES}}}email", show["email"])

    for ep in episodes:
        ep = dict(ep)
        item = ET.SubElement(channel, "item")
        _sub(item, "title", ep["title"])
        _sub(item, "description", ep.get("description") or "")
        _sub(item, "guid", ep["episode_id"], isPermaLink="false")
        if ep.get("published_at"):
            pub_dt = time.strftime(
                "%a, %d %b %Y %H:%M:%S +0000", time.gmtime(ep["published_at"])
            )
            _sub(item, "pubDate", pub_dt)
        if ep.get("audio_url"):
            ET.SubElement(item, "enclosure", {
                "url": ep["audio_url"],
                "length": str(ep.get("audio_size") or 0),
                "type": "audio/mpeg",
            })
        if ep.get("audio_duration"):
            secs = int(ep["audio_duration"])
            _sub(item, f"{{{ITUNES}}}duration",
                 f"{secs // 3600:02d}:{(secs % 3600) // 60:02d}:{secs % 60:02d}")
        if ep.get("episode_number"):
            _sub(item, f"{{{ITUNES}}}episode", str(ep["episode_number"]))
        if ep.get("season"):
            _sub(item, f"{{{ITUNES}}}season", str(ep["season"]))
        _sub(item, f"{{{ITUNES}}}episodeType", ep.get("episode_type") or "full")
        if ep.get("content"):
            _sub(item, "content:encoded", ep["content"])

    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(rss, encoding="unicode")

# ---------------------------------------------------------------------------
# Episode publishing
# ---------------------------------------------------------------------------

def _publish_episode(episode_id: str) -> bool:
    conn = _db_conn()
    ep = conn.execute("SELECT * FROM episodes WHERE episode_id=?", (episode_id,)).fetchone()
    if not ep:
        conn.close()
        return False
    ep = dict(ep)
    now = time.time()
    conn.execute(
        "UPDATE episodes SET status='published', published_at=? WHERE episode_id=?",
        (now, episode_id),
    )
    conn.execute(
        "UPDATE shows SET episode_count = episode_count + 1 WHERE show_id=?",
        (ep["show_id"],),
    )
    conn.commit()
    conn.close()
    ep["published_at"] = now
    _notify_subscribers(ep["show_id"], ep)
    return True

# ---------------------------------------------------------------------------
# Background scheduler thread
# ---------------------------------------------------------------------------

def _scheduler_loop():
    while True:
        try:
            _run_scheduler()
        except Exception as exc:
            log.error("Scheduler error: %s", exc)
        time.sleep(3600)


def _run_scheduler():
    now = time.time()
    conn = _db_conn()
    pending = conn.execute(
        """SELECT episode_id FROM episodes
           WHERE status='scheduled' AND scheduled_at IS NOT NULL AND scheduled_at <= ?""",
        (now,),
    ).fetchall()
    conn.close()
    for row in pending:
        eid = row["episode_id"]
        log.info("Scheduler publishing episode %s", eid)
        _publish_episode(eid)


def _start_scheduler():
    t = threading.Thread(target=_scheduler_loop, daemon=True, name="podcast-scheduler")
    t.start()
    log.info("Scheduler thread started")

# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class PodcastHandler(BaseHTTPRequestHandler):
    server_version = "PodcastEngine/1.0"

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------

    def _parse_path(self):
        """Return (path_str, query_dict)."""
        if "?" in self.path:
            p, q = self.path.split("?", 1)
        else:
            p, q = self.path, ""
        params = {}
        for part in q.split("&"):
            if "=" in part:
                pk, pv = part.split("=", 1)
                params[pk] = urllib.parse.unquote_plus(pv)
        return p.rstrip("/"), params

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def _is_admin(self) -> bool:
        if not ADMIN_SECRET:
            return True
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            return auth[7:] == ADMIN_SECRET
        body_secret = self.headers.get("X-Admin-Secret", "")
        return body_secret == ADMIN_SECRET

    def _send(self, status: int, data, content_type: str = "application/json"):
        body = json.dumps(data, default=str).encode() if not isinstance(data, bytes) else data
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_xml(self, status: int, xml_str: str):
        body = xml_str.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/rss+xml; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _ok(self, data):        self._send(200, data)
    def _created(self, data):   self._send(201, data)
    def _bad(self, msg):        self._send(400, {"error": msg})
    def _unauth(self):          self._send(401, {"error": "Unauthorized"})
    def _not_found(self, msg="Not found"): self._send(404, {"error": msg})
    def _err(self, msg):        self._send(500, {"error": msg})

    def log_message(self, fmt, *args):  # silence default access log
        log.debug("HTTP %s", fmt % args)

    # ------------------------------------------------------------------
    # OPTIONS (CORS preflight)
    # ------------------------------------------------------------------

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Admin-Secret")
        self.end_headers()

    # ------------------------------------------------------------------
    # GET
    # ------------------------------------------------------------------

    def do_GET(self):
        path, params = self._parse_path()
        parts = [p for p in path.split("/") if p]

        # GET /health
        if path in ("/health", ""):
            return self._handle_health()

        # GET /shows
        if path == "/shows" and not parts[1:]:
            return self._handle_list_shows()

        # GET /shows/{show_id}
        if len(parts) == 2 and parts[0] == "shows":
            return self._handle_get_show(parts[1])

        # GET /shows/{show_id}/rss
        if len(parts) == 3 and parts[0] == "shows" and parts[2] == "rss":
            return self._handle_show_rss(parts[1])

        # GET /shows/{show_id}/episodes
        if len(parts) == 3 and parts[0] == "shows" and parts[2] == "episodes":
            return self._handle_show_episodes(parts[1], params)

        # GET /episodes/{episode_id}
        if len(parts) == 2 and parts[0] == "episodes":
            return self._handle_get_episode(parts[1])

        # GET /episodes/{episode_id}/audio
        if len(parts) == 3 and parts[0] == "episodes" and parts[2] == "audio":
            return self._handle_episode_audio(parts[1])

        # GET /analytics
        if path == "/analytics":
            if not self._is_admin():
                return self._unauth()
            return self._handle_analytics()

        self._not_found()

    # ------------------------------------------------------------------
    # POST
    # ------------------------------------------------------------------

    def do_POST(self):
        path, _ = self._parse_path()
        parts = [p for p in path.split("/") if p]
        body = self._read_body()

        # POST /shows
        if path == "/shows":
            if not self._is_admin():
                return self._unauth()
            return self._handle_create_show(body)

        # POST /shows/{show_id}/subscribe
        if len(parts) == 3 and parts[0] == "shows" and parts[2] == "subscribe":
            return self._handle_subscribe(parts[1], body)

        # POST /episodes
        if path == "/episodes":
            if not self._is_admin():
                return self._unauth()
            return self._handle_create_episode(body)

        # POST /episodes/{episode_id}/publish
        if len(parts) == 3 and parts[0] == "episodes" and parts[2] == "publish":
            if not self._is_admin():
                return self._unauth()
            return self._handle_publish_episode(parts[1])

        self._not_found()

    # ------------------------------------------------------------------
    # PUT
    # ------------------------------------------------------------------

    def do_PUT(self):
        path, _ = self._parse_path()
        parts = [p for p in path.split("/") if p]
        body = self._read_body()

        # PUT /episodes/{episode_id}
        if len(parts) == 2 and parts[0] == "episodes":
            if not self._is_admin():
                return self._unauth()
            return self._handle_update_episode(parts[1], body)

        self._not_found()

    # ------------------------------------------------------------------
    # DELETE
    # ------------------------------------------------------------------

    def do_DELETE(self):
        path, _ = self._parse_path()
        parts = [p for p in path.split("/") if p]
        body = self._read_body()

        # DELETE /shows/{show_id}/subscribe
        if len(parts) == 3 and parts[0] == "shows" and parts[2] == "subscribe":
            return self._handle_unsubscribe(parts[1], body)

        self._not_found()

    # ------------------------------------------------------------------
    # Handler implementations
    # ------------------------------------------------------------------

    def _handle_health(self):
        conn = _db_conn()
        shows      = conn.execute("SELECT COUNT(*) FROM shows").fetchone()[0]
        episodes   = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        subscribers = conn.execute("SELECT COUNT(*) FROM subscribers WHERE active=1").fetchone()[0]
        conn.close()
        self._ok({
            "status": "healthy",
            "service": "podcast_engine",
            "port": PORT,
            "uptime_seconds": round(time.time() - START_TIME, 1),
            "shows": shows,
            "episodes": episodes,
            "subscribers": subscribers,
        })

    def _handle_list_shows(self):
        conn = _db_conn()
        rows = conn.execute(
            "SELECT * FROM shows WHERE status='active' ORDER BY created_at DESC"
        ).fetchall()
        conn.close()
        self._ok({"shows": _json_rows(rows), "count": len(rows)})

    def _handle_get_show(self, show_id: str):
        conn = _db_conn()
        show = conn.execute("SELECT * FROM shows WHERE show_id=?", (show_id,)).fetchone()
        if not show:
            conn.close()
            return self._not_found("Show not found")
        show = dict(show)
        episodes = conn.execute(
            "SELECT * FROM episodes WHERE show_id=? ORDER BY created_at DESC LIMIT 10",
            (show_id,),
        ).fetchall()
        conn.close()
        show["recent_episodes"] = _json_rows(episodes)
        self._ok(show)

    def _handle_show_rss(self, show_id: str):
        xml_str = _generate_rss(show_id)
        if not xml_str:
            return self._not_found("Show not found")
        self._send_xml(200, xml_str)

    def _handle_show_episodes(self, show_id: str, params: dict):
        status = params.get("status", "")
        limit  = int(params.get("limit", "50"))
        conn = _db_conn()
        show = conn.execute("SELECT show_id FROM shows WHERE show_id=?", (show_id,)).fetchone()
        if not show:
            conn.close()
            return self._not_found("Show not found")
        if status:
            rows = conn.execute(
                "SELECT * FROM episodes WHERE show_id=? AND status=? ORDER BY created_at DESC LIMIT ?",
                (show_id, status, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM episodes WHERE show_id=? ORDER BY created_at DESC LIMIT ?",
                (show_id, limit),
            ).fetchall()
        conn.close()
        self._ok({"episodes": _json_rows(rows), "count": len(rows)})

    def _handle_get_episode(self, episode_id: str):
        conn = _db_conn()
        ep = conn.execute("SELECT * FROM episodes WHERE episode_id=?", (episode_id,)).fetchone()
        conn.close()
        if not ep:
            return self._not_found("Episode not found")
        self._ok(dict(ep))

    def _handle_episode_audio(self, episode_id: str):
        conn = _db_conn()
        ep = conn.execute("SELECT * FROM episodes WHERE episode_id=?", (episode_id,)).fetchone()
        if not ep:
            conn.close()
            return self._not_found("Episode not found")
        ep = dict(ep)
        # Record the listen
        ip_hash = _hash_ip(self.client_address[0])
        conn.execute(
            "INSERT INTO listens (episode_id, listener_ip_hash, listened_at) VALUES (?,?,?)",
            (episode_id, ip_hash, time.time()),
        )
        conn.execute(
            "UPDATE episodes SET listen_count = listen_count + 1 WHERE episode_id=?",
            (episode_id,),
        )
        conn.commit()
        conn.close()

        if ep.get("audio_url") and ep["audio_url"].startswith("http"):
            # Redirect to R2 / external URL
            self.send_response(302)
            self.send_header("Location", ep["audio_url"])
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
        elif ep.get("audio_url") and ep["audio_url"].startswith("data:"):
            # Inline base64 audio
            header, encoded = ep["audio_url"].split(",", 1)
            audio_bytes = base64.b64decode(encoded)
            self.send_response(200)
            self.send_header("Content-Type", "audio/mpeg")
            self.send_header("Content-Length", str(len(audio_bytes)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(audio_bytes)
        else:
            self._not_found("Audio not available")

    def _handle_analytics(self):
        conn = _db_conn()
        ep_stats = conn.execute(
            """SELECT e.episode_id, e.title, e.show_id, e.listen_count,
                      COUNT(l.id) AS raw_listens,
                      SUM(l.completed) AS completions,
                      AVG(l.listen_duration) AS avg_duration
               FROM episodes e
               LEFT JOIN listens l ON l.episode_id = e.episode_id
               GROUP BY e.episode_id
               ORDER BY e.listen_count DESC"""
        ).fetchall()
        sub_growth = conn.execute(
            """SELECT show_id, COUNT(*) AS total,
                      SUM(CASE WHEN active=1 THEN 1 ELSE 0 END) AS active_count
               FROM subscribers GROUP BY show_id"""
        ).fetchall()
        total_listens = conn.execute("SELECT COUNT(*) FROM listens").fetchone()[0]
        conn.close()
        self._ok({
            "total_listens": total_listens,
            "episodes": _json_rows(ep_stats),
            "subscriber_growth": _json_rows(sub_growth),
        })

    def _handle_create_show(self, body: dict):
        title = body.get("title", "").strip()
        if not title:
            return self._bad("title is required")
        show_id = _make_id("show")
        now = time.time()
        conn = _db_conn()
        try:
            conn.execute(
                """INSERT INTO shows
                   (show_id, title, description, author, email, category, language,
                    image_url, website, explicit, status, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    show_id,
                    title,
                    body.get("description", ""),
                    body.get("author", "FractalMesh OMEGA Titan"),
                    body.get("email", ""),
                    body.get("category", "Technology"),
                    body.get("language", "en-au"),
                    body.get("image_url", ""),
                    body.get("website", ""),
                    1 if body.get("explicit") else 0,
                    "active",
                    now,
                ),
            )
            conn.commit()
        except sqlite3.IntegrityError as exc:
            conn.close()
            return self._err(str(exc))
        show = conn.execute("SELECT * FROM shows WHERE show_id=?", (show_id,)).fetchone()
        conn.close()
        log.info("Created show: %s (%s)", title, show_id)
        self._created(dict(show))

    def _handle_subscribe(self, show_id: str, body: dict):
        email = body.get("email", "").strip().lower()
        if not email or "@" not in email:
            return self._bad("Valid email required")
        conn = _db_conn()
        show = conn.execute("SELECT * FROM shows WHERE show_id=?", (show_id,)).fetchone()
        if not show:
            conn.close()
            return self._not_found("Show not found")
        show = dict(show)
        try:
            conn.execute(
                "INSERT OR IGNORE INTO subscribers (email, show_id, subscribed_at, active) VALUES (?,?,?,1)",
                (email, show_id, time.time()),
            )
            conn.execute(
                "UPDATE subscribers SET active=1, subscribed_at=? WHERE email=? AND show_id=?",
                (time.time(), email, show_id),
            )
            conn.execute(
                "UPDATE shows SET subscriber_count = (SELECT COUNT(*) FROM subscribers WHERE show_id=? AND active=1) WHERE show_id=?",
                (show_id, show_id),
            )
            conn.commit()
        finally:
            conn.close()
        threading.Thread(
            target=_send_subscribe_confirmation,
            args=(email, show["title"]),
            daemon=True,
        ).start()
        self._ok({"subscribed": True, "email": email, "show_id": show_id})

    def _handle_unsubscribe(self, show_id: str, body: dict):
        email = body.get("email", "").strip().lower()
        if not email:
            return self._bad("email required")
        conn = _db_conn()
        conn.execute(
            "UPDATE subscribers SET active=0 WHERE email=? AND show_id=?",
            (email, show_id),
        )
        conn.execute(
            "UPDATE shows SET subscriber_count = (SELECT COUNT(*) FROM subscribers WHERE show_id=? AND active=1) WHERE show_id=?",
            (show_id, show_id),
        )
        conn.commit()
        conn.close()
        self._ok({"unsubscribed": True, "email": email, "show_id": show_id})

    def _handle_create_episode(self, body: dict):
        show_id = body.get("show_id", "").strip()
        title   = body.get("title", "").strip()
        if not show_id or not title:
            return self._bad("show_id and title are required")
        conn = _db_conn()
        show = conn.execute("SELECT show_id FROM shows WHERE show_id=?", (show_id,)).fetchone()
        if not show:
            conn.close()
            return self._not_found("Show not found")
        conn.close()

        episode_id    = _make_id("ep")
        content       = body.get("content", "")
        do_tts        = bool(body.get("tts", False))
        voice_id      = body.get("voice_id", DEFAULT_VOICE_ID)
        scheduled_at  = body.get("scheduled_at")
        status        = "scheduled" if scheduled_at else "draft"
        audio_url     = ""
        audio_size    = 0

        if do_tts and content:
            try:
                mp3_bytes = _tts_generate(content, voice_id)
                audio_size = len(mp3_bytes)
                r2_key = f"podcasts/{show_id}/{episode_id}.mp3"
                if all([R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME]):
                    audio_url = _r2_upload(r2_key, mp3_bytes, "audio/mpeg")
                    log.info("TTS audio uploaded to R2: %s", audio_url)
                else:
                    b64 = base64.b64encode(mp3_bytes).decode()
                    audio_url = f"data:audio/mpeg;base64,{b64}"
                    log.info("TTS audio stored as base64 (no R2 configured)")
            except Exception as exc:
                log.error("TTS generation failed: %s", exc)
                return self._err(f"TTS generation failed: {exc}")

        now = time.time()
        conn = _db_conn()
        conn.execute(
            """INSERT INTO episodes
               (episode_id, show_id, title, description, content,
                audio_url, audio_size, season, episode_number,
                episode_type, status, scheduled_at, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                episode_id,
                show_id,
                title,
                body.get("description", ""),
                content,
                audio_url,
                audio_size,
                int(body.get("season", 1)),
                body.get("episode_number"),
                body.get("episode_type", "full"),
                status,
                float(scheduled_at) if scheduled_at else None,
                now,
            ),
        )
        conn.commit()
        ep = conn.execute("SELECT * FROM episodes WHERE episode_id=?", (episode_id,)).fetchone()
        conn.close()
        log.info("Created episode: %s (%s) status=%s", title, episode_id, status)
        self._created(dict(ep))

    def _handle_publish_episode(self, episode_id: str):
        conn = _db_conn()
        ep = conn.execute("SELECT * FROM episodes WHERE episode_id=?", (episode_id,)).fetchone()
        conn.close()
        if not ep:
            return self._not_found("Episode not found")
        ok = _publish_episode(episode_id)
        if not ok:
            return self._err("Could not publish episode")
        conn = _db_conn()
        ep = conn.execute("SELECT * FROM episodes WHERE episode_id=?", (episode_id,)).fetchone()
        conn.close()
        log.info("Published episode %s", episode_id)
        self._ok(dict(ep))

    def _handle_update_episode(self, episode_id: str, body: dict):
        conn = _db_conn()
        ep = conn.execute("SELECT * FROM episodes WHERE episode_id=?", (episode_id,)).fetchone()
        if not ep:
            conn.close()
            return self._not_found("Episode not found")
        ep = dict(ep)
        allowed = {
            "title", "description", "content", "audio_url", "audio_duration",
            "audio_size", "season", "episode_number", "episode_type",
            "status", "scheduled_at",
        }
        updates = {k: v for k, v in body.items() if k in allowed}
        if not updates:
            conn.close()
            return self._bad("No valid fields to update")
        set_clause = ", ".join(f"{k}=?" for k in updates)
        vals = list(updates.values()) + [episode_id]
        conn.execute(f"UPDATE episodes SET {set_clause} WHERE episode_id=?", vals)
        conn.commit()
        ep = conn.execute("SELECT * FROM episodes WHERE episode_id=?", (episode_id,)).fetchone()
        conn.close()
        log.info("Updated episode %s fields: %s", episode_id, list(updates.keys()))
        self._ok(dict(ep))


# ---------------------------------------------------------------------------
# Patch urllib.parse import used in handler
# ---------------------------------------------------------------------------
import urllib.parse

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _db_init()
    _seed_shows()
    _start_scheduler()
    server = HTTPServer(("0.0.0.0", PORT), PodcastHandler)
    log.info("Podcast Engine listening on port %d", PORT)
    server.serve_forever()


if __name__ == "__main__":
    main()
