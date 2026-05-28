#!/usr/bin/env python3
"""
fm_video_platform.py — Video Platform & Streaming Manager (Port 7894)
FractalMesh OMEGA Titan: upload videos to R2, manage metadata, track views /
watch time, generate thumbnail URLs, handle video access control, create
playlists and channels.  AWS Signature Version 4 for all R2 operations.
stdlib only — no hardcoded credentials.
Samuel James Hiotis | ABN 56 628 117 363
"""
import base64
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ── vault ──────────────────────────────────────────────────────────────────────
_ENV_FILE = Path.home() / ".secrets" / "fractal.env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

# ── config ─────────────────────────────────────────────────────────────────────
PORT                 = int(os.getenv("VIDEO_PLATFORM_PORT", "7894"))
R2_ACCOUNT_ID        = os.getenv("R2_ACCOUNT_ID", "")
R2_ACCESS_KEY_ID     = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET_NAME       = os.getenv("R2_BUCKET_NAME", "fractalmesh-media")
SENDGRID_API_KEY     = os.getenv("SENDGRID_API_KEY", "")
SENDGRID_FROM_EMAIL  = os.getenv("SENDGRID_FROM_EMAIL", "")
ADMIN_SECRET         = os.getenv("ADMIN_SECRET", "")

ROOT    = Path(os.getenv("FRACTALMESH_HOME", str(Path.home() / "fmsaas")))
DB_PATH = ROOT / "database" / "sovereign.db"
LOG_DIR = ROOT / "logs"

for _p in (ROOT, DB_PATH.parent, LOG_DIR):
    _p.mkdir(parents=True, exist_ok=True)

START_TIME = time.time()
_db_lock   = threading.Lock()

# ── logging ────────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [VIDEO-PLATFORM] {msg}", flush=True)

# ── database ───────────────────────────────────────────────────────────────────

def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _db_lock:
        conn = _get_db()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS videos (
                id                   INTEGER PRIMARY KEY,
                video_id             TEXT UNIQUE NOT NULL,
                title                TEXT,
                description          TEXT,
                uploader_id          TEXT,
                channel_id           TEXT,
                r2_key               TEXT UNIQUE,
                thumbnail_key        TEXT,
                duration_seconds     REAL    DEFAULT 0,
                file_size_bytes      INTEGER DEFAULT 0,
                content_type         TEXT    DEFAULT 'video/mp4',
                status               TEXT    DEFAULT 'processing',
                visibility           TEXT    DEFAULT 'public',
                tags                 TEXT    DEFAULT '[]',
                view_count           INTEGER DEFAULT 0,
                like_count           INTEGER DEFAULT 0,
                total_watch_seconds  REAL    DEFAULT 0,
                created_at           REAL,
                updated_at           REAL,
                published_at         REAL
            );
            CREATE TABLE IF NOT EXISTS channels (
                id               INTEGER PRIMARY KEY,
                channel_id       TEXT UNIQUE NOT NULL,
                name             TEXT,
                description      TEXT,
                owner_email      TEXT,
                owner_id         TEXT,
                subscriber_count INTEGER DEFAULT 0,
                video_count      INTEGER DEFAULT 0,
                banner_key       TEXT,
                avatar_key       TEXT,
                created_at       REAL,
                updated_at       REAL
            );
            CREATE TABLE IF NOT EXISTS playlists (
                id          INTEGER PRIMARY KEY,
                playlist_id TEXT UNIQUE NOT NULL,
                channel_id  TEXT,
                title       TEXT,
                description TEXT,
                visibility  TEXT    DEFAULT 'public',
                video_ids   TEXT    DEFAULT '[]',
                video_count INTEGER DEFAULT 0,
                created_at  REAL,
                updated_at  REAL
            );
            CREATE TABLE IF NOT EXISTS watch_sessions (
                id            INTEGER PRIMARY KEY,
                session_id    TEXT UNIQUE NOT NULL,
                video_id      TEXT,
                viewer_id     TEXT,
                ip_hash       TEXT,
                watch_seconds REAL    DEFAULT 0,
                completed     INTEGER DEFAULT 0,
                created_at    REAL,
                updated_at    REAL
            );
            CREATE TABLE IF NOT EXISTS comments (
                id          INTEGER PRIMARY KEY,
                comment_id  TEXT UNIQUE NOT NULL,
                video_id    TEXT,
                author_id   TEXT,
                author_name TEXT,
                content     TEXT,
                parent_id   TEXT,
                like_count  INTEGER DEFAULT 0,
                status      TEXT    DEFAULT 'active',
                created_at  REAL
            );
            CREATE INDEX IF NOT EXISTS idx_videos_channel    ON videos(channel_id);
            CREATE INDEX IF NOT EXISTS idx_videos_status     ON videos(status);
            CREATE INDEX IF NOT EXISTS idx_videos_visibility ON videos(visibility);
            CREATE INDEX IF NOT EXISTS idx_watch_video       ON watch_sessions(video_id);
            CREATE INDEX IF NOT EXISTS idx_watch_viewer      ON watch_sessions(viewer_id);
            CREATE INDEX IF NOT EXISTS idx_comments_video    ON comments(video_id);
            CREATE INDEX IF NOT EXISTS idx_comments_parent   ON comments(parent_id);
        """)
        conn.commit()
        conn.close()
    _log("Database initialised")


# ── HTTP helpers ────────────────────────────────────────────────────────────────

def _json_response(handler: "BaseHTTPRequestHandler", code: int, data: dict) -> None:
    body = json.dumps(data, indent=2).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def _read_body(handler: "BaseHTTPRequestHandler") -> bytes:
    length = int(handler.headers.get("Content-Length", 0))
    return handler.rfile.read(length) if length else b""


def _require_admin(handler: "BaseHTTPRequestHandler") -> bool:
    secret = handler.headers.get("X-Admin-Secret", "")
    if not ADMIN_SECRET or not hmac.compare_digest(secret, ADMIN_SECRET):
        _json_response(handler, 403, {"error": "forbidden – X-Admin-Secret required"})
        return False
    return True


def _parse_qs(path: str) -> tuple:
    if "?" in path:
        base, qs = path.split("?", 1)
        return base, dict(urllib.parse.parse_qsl(qs))
    return path, {}


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    for field in ("tags", "video_ids"):
        if field in d and isinstance(d[field], str):
            try:
                d[field] = json.loads(d[field])
            except (json.JSONDecodeError, TypeError):
                d[field] = []
    return d


# ── AWS SigV4 helpers ──────────────────────────────────────────────────────────

def _hmac_sha256(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _get_signing_key(secret: str, date_stamp: str, region: str, service: str) -> bytes:
    k_date    = _hmac_sha256(("AWS4" + secret).encode("utf-8"), date_stamp)
    k_region  = _hmac_sha256(k_date, region)
    k_service = _hmac_sha256(k_region, service)
    k_signing = _hmac_sha256(k_service, "aws4_request")
    return k_signing


def _r2_request(
    method: str,
    key: str,
    body: bytes = None,
    content_type: str = "application/octet-stream",
    params: dict = None,
) -> urllib.request.Request:
    """Sign and build a urllib Request for Cloudflare R2 using AWS SigV4."""
    account = R2_ACCOUNT_ID
    bucket  = R2_BUCKET_NAME
    host    = f"{bucket}.{account}.r2.cloudflarestorage.com"
    region  = "auto"
    service = "s3"

    now        = time.gmtime()
    amz_date   = time.strftime("%Y%m%dT%H%M%SZ", now)
    date_stamp = time.strftime("%Y%m%d", now)

    safe_key      = "/".join(urllib.parse.quote(seg, safe="") for seg in key.lstrip("/").split("/"))
    canonical_uri = f"/{safe_key}"

    qs = urllib.parse.urlencode(sorted(params.items())) if params else ""

    payload_bytes = body or b""
    payload_hash  = hashlib.sha256(payload_bytes).hexdigest()

    canonical_headers = (
        f"content-type:{content_type}\n"
        f"host:{host}\n"
        f"x-amz-content-sha256:{payload_hash}\n"
        f"x-amz-date:{amz_date}\n"
    )
    signed_headers = "content-type;host;x-amz-content-sha256;x-amz-date"

    canonical_request = "\n".join([
        method.upper(),
        canonical_uri,
        qs,
        canonical_headers,
        signed_headers,
        payload_hash,
    ])

    credential_scope = f"{date_stamp}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join([
        "AWS4-HMAC-SHA256",
        amz_date,
        credential_scope,
        hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
    ])

    signing_key = _get_signing_key(R2_SECRET_ACCESS_KEY, date_stamp, region, service)
    signature   = hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    authorization = (
        f"AWS4-HMAC-SHA256 Credential={R2_ACCESS_KEY_ID}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )

    endpoint = f"https://{host}{canonical_uri}"
    if qs:
        endpoint = f"{endpoint}?{qs}"

    req = urllib.request.Request(endpoint, data=payload_bytes, method=method.upper())
    req.add_header("Authorization",        authorization)
    req.add_header("x-amz-date",           amz_date)
    req.add_header("x-amz-content-sha256", payload_hash)
    req.add_header("Content-Type",         content_type)
    req.add_header("Host",                 host)
    if payload_bytes:
        req.add_header("Content-Length", str(len(payload_bytes)))
    return req


def _r2_put(key: str, data: bytes, content_type: str) -> bool:
    """Upload object to R2.  Returns True on success."""
    if not (R2_ACCOUNT_ID and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY):
        _log("R2 credentials not configured – skipping upload")
        return False
    try:
        req = _r2_request("PUT", key, body=data, content_type=content_type)
        with urllib.request.urlopen(req, timeout=120) as resp:
            status = resp.status
        _log(f"R2 PUT {key} → {status}")
        return status in (200, 201, 204)
    except urllib.error.HTTPError as exc:
        _log(f"R2 PUT error {exc.code}: {exc.read()[:200]}")
        return False
    except Exception as exc:
        _log(f"R2 PUT exception: {exc}")
        return False


def _r2_delete(key: str) -> bool:
    """Delete object from R2.  Returns True on success."""
    if not (R2_ACCOUNT_ID and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY):
        return False
    try:
        req = _r2_request("DELETE", key, body=b"", content_type="application/octet-stream")
        with urllib.request.urlopen(req, timeout=30) as resp:
            status = resp.status
        _log(f"R2 DELETE {key} → {status}")
        return status in (200, 204)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return True  # already gone
        _log(f"R2 DELETE error {exc.code}: {exc.read()[:200]}")
        return False
    except Exception as exc:
        _log(f"R2 DELETE exception: {exc}")
        return False


def _r2_public_url(key: str) -> str:
    """Return the public R2 URL for a key."""
    account = R2_ACCOUNT_ID
    bucket  = R2_BUCKET_NAME
    safe_key = "/".join(urllib.parse.quote(seg, safe="") for seg in key.lstrip("/").split("/"))
    return f"https://{bucket}.{account}.r2.cloudflarestorage.com/{safe_key}"


def _generate_signed_token(video_id: str, ttl: int = 3600) -> str:
    """Generate a time-limited signed access token for private videos."""
    expires  = int(time.time()) + ttl
    raw      = f"{video_id}:{expires}:{secrets.token_hex(8)}"
    sig      = hmac.new(
        (ADMIN_SECRET or "fallback-secret").encode(),
        raw.encode(),
        hashlib.sha256,
    ).hexdigest()[:16]
    token = base64.urlsafe_b64encode(f"{raw}:{sig}".encode()).decode().rstrip("=")
    return token


def _get_stream_url(video: dict) -> str:
    """Return stream URL: direct R2 URL for public, signed token URL for private."""
    r2_key = video.get("r2_key", "")
    if not r2_key:
        return ""
    visibility = video.get("visibility", "public")
    if visibility == "public":
        return _r2_public_url(r2_key)
    token = _generate_signed_token(video.get("video_id", ""))
    return f"https://api.fractalmesh.com/videos/{video.get('video_id', '')}/stream?token={token}"


# ── channel helpers ────────────────────────────────────────────────────────────

def _create_channel(name: str, description: str, owner_email: str, owner_id: str = "") -> dict:
    channel_id = f"ch_{secrets.token_hex(10)}"
    now = time.time()
    with _db_lock:
        conn = _get_db()
        try:
            conn.execute(
                """INSERT INTO channels
                   (channel_id, name, description, owner_email, owner_id, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (channel_id, name, description, owner_email, owner_id, now, now),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM channels WHERE channel_id=?", (channel_id,)
            ).fetchone()
            return _row_to_dict(row)
        finally:
            conn.close()


def _get_channel(channel_id: str) -> dict | None:
    with _db_lock:
        conn = _get_db()
        try:
            row = conn.execute(
                "SELECT * FROM channels WHERE channel_id=?", (channel_id,)
            ).fetchone()
            if not row:
                return None
            channel = _row_to_dict(row)
            videos = conn.execute(
                """SELECT * FROM videos WHERE channel_id=? AND status='active'
                   ORDER BY created_at DESC LIMIT 50""",
                (channel_id,),
            ).fetchall()
            channel["videos"] = [_row_to_dict(v) for v in videos]
            return channel
        finally:
            conn.close()


# ── video helpers ──────────────────────────────────────────────────────────────

def _list_videos(channel_id: str = "", tag: str = "", visibility: str = "") -> list:
    clauses = ["status='active'"]
    params  = []
    if channel_id:
        clauses.append("channel_id=?")
        params.append(channel_id)
    if visibility:
        clauses.append("visibility=?")
        params.append(visibility)
    where = " AND ".join(clauses)
    sql   = f"SELECT * FROM videos WHERE {where} ORDER BY created_at DESC LIMIT 100"
    with _db_lock:
        conn = _get_db()
        try:
            rows = conn.execute(sql, params).fetchall()
            results = [_row_to_dict(r) for r in rows]
            if tag:
                results = [v for v in results if tag in v.get("tags", [])]
            return results
        finally:
            conn.close()


def _get_video(video_id: str) -> dict | None:
    with _db_lock:
        conn = _get_db()
        try:
            row = conn.execute(
                "SELECT * FROM videos WHERE video_id=?", (video_id,)
            ).fetchone()
            return _row_to_dict(row) if row else None
        finally:
            conn.close()


def _upload_video(
    channel_id: str,
    title: str,
    description: str,
    tags: list,
    visibility: str,
    filename: str,
    content_type: str,
    data_bytes: bytes,
    uploader_id: str = "",
    duration_seconds: float = 0.0,
) -> dict:
    video_id   = f"vid_{secrets.token_hex(12)}"
    ext        = filename.rsplit(".", 1)[-1] if "." in filename else "mp4"
    r2_key     = f"videos/{channel_id}/{video_id}.{ext}"
    now        = time.time()

    ok = _r2_put(r2_key, data_bytes, content_type)
    status = "active" if ok else "processing"

    with _db_lock:
        conn = _get_db()
        try:
            conn.execute(
                """INSERT INTO videos
                   (video_id, title, description, uploader_id, channel_id,
                    r2_key, duration_seconds, file_size_bytes, content_type,
                    status, visibility, tags, created_at, updated_at, published_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    video_id, title, description, uploader_id, channel_id,
                    r2_key, duration_seconds, len(data_bytes), content_type,
                    status, visibility, json.dumps(tags), now, now, now,
                ),
            )
            # Bump channel video_count
            conn.execute(
                "UPDATE channels SET video_count=video_count+1, updated_at=? WHERE channel_id=?",
                (now, channel_id),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM videos WHERE video_id=?", (video_id,)
            ).fetchone()
            return _row_to_dict(row)
        finally:
            conn.close()


def _record_view(video_id: str, viewer_id: str = "", ip: str = "") -> dict:
    session_id = f"ws_{secrets.token_hex(12)}"
    now        = time.time()
    ip_hash    = hashlib.sha256(ip.encode()).hexdigest()[:16] if ip else ""
    with _db_lock:
        conn = _get_db()
        try:
            conn.execute(
                """INSERT INTO watch_sessions
                   (session_id, video_id, viewer_id, ip_hash, created_at, updated_at)
                   VALUES (?,?,?,?,?,?)""",
                (session_id, video_id, viewer_id, ip_hash, now, now),
            )
            conn.execute(
                "UPDATE videos SET view_count=view_count+1, updated_at=? WHERE video_id=?",
                (now, video_id),
            )
            conn.commit()
            return {"session_id": session_id, "created_at": now}
        finally:
            conn.close()


def _update_watch_session(session_id: str, watch_seconds: float, completed: bool) -> bool:
    now = time.time()
    with _db_lock:
        conn = _get_db()
        try:
            row = conn.execute(
                "SELECT video_id FROM watch_sessions WHERE session_id=?", (session_id,)
            ).fetchone()
            if not row:
                return False
            video_id = row["video_id"]
            conn.execute(
                """UPDATE watch_sessions
                   SET watch_seconds=?, completed=?, updated_at=?
                   WHERE session_id=?""",
                (watch_seconds, 1 if completed else 0, now, session_id),
            )
            conn.execute(
                """UPDATE videos
                   SET total_watch_seconds=total_watch_seconds+?, updated_at=?
                   WHERE video_id=?""",
                (watch_seconds, now, video_id),
            )
            conn.commit()
            return True
        finally:
            conn.close()


def _like_video(video_id: str) -> int:
    now = time.time()
    with _db_lock:
        conn = _get_db()
        try:
            conn.execute(
                "UPDATE videos SET like_count=like_count+1, updated_at=? WHERE video_id=?",
                (now, video_id),
            )
            conn.commit()
            row = conn.execute(
                "SELECT like_count FROM videos WHERE video_id=?", (video_id,)
            ).fetchone()
            return row["like_count"] if row else 0
        finally:
            conn.close()


def _delete_video(video_id: str) -> bool:
    with _db_lock:
        conn = _get_db()
        try:
            row = conn.execute(
                "SELECT r2_key, thumbnail_key, channel_id FROM videos WHERE video_id=?",
                (video_id,),
            ).fetchone()
            if not row:
                return False
            r2_key        = row["r2_key"]
            thumbnail_key = row["thumbnail_key"]
            channel_id    = row["channel_id"]
            if r2_key:
                _r2_delete(r2_key)
            if thumbnail_key:
                _r2_delete(thumbnail_key)
            conn.execute("DELETE FROM videos WHERE video_id=?", (video_id,))
            conn.execute("DELETE FROM watch_sessions WHERE video_id=?", (video_id,))
            conn.execute("DELETE FROM comments WHERE video_id=?", (video_id,))
            if channel_id:
                conn.execute(
                    "UPDATE channels SET video_count=MAX(0,video_count-1), updated_at=? WHERE channel_id=?",
                    (time.time(), channel_id),
                )
            conn.commit()
            return True
        finally:
            conn.close()


# ── playlist helpers ───────────────────────────────────────────────────────────

def _create_playlist(channel_id: str, title: str, description: str, visibility: str) -> dict:
    playlist_id = f"pl_{secrets.token_hex(10)}"
    now         = time.time()
    with _db_lock:
        conn = _get_db()
        try:
            conn.execute(
                """INSERT INTO playlists
                   (playlist_id, channel_id, title, description, visibility, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (playlist_id, channel_id, title, description, visibility, now, now),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM playlists WHERE playlist_id=?", (playlist_id,)
            ).fetchone()
            return _row_to_dict(row)
        finally:
            conn.close()


def _add_to_playlist(playlist_id: str, video_id: str) -> dict | None:
    now = time.time()
    with _db_lock:
        conn = _get_db()
        try:
            row = conn.execute(
                "SELECT * FROM playlists WHERE playlist_id=?", (playlist_id,)
            ).fetchone()
            if not row:
                return None
            playlist = _row_to_dict(row)
            ids = playlist.get("video_ids", [])
            if video_id not in ids:
                ids.append(video_id)
            conn.execute(
                "UPDATE playlists SET video_ids=?, video_count=?, updated_at=? WHERE playlist_id=?",
                (json.dumps(ids), len(ids), now, playlist_id),
            )
            conn.commit()
            updated = conn.execute(
                "SELECT * FROM playlists WHERE playlist_id=?", (playlist_id,)
            ).fetchone()
            return _row_to_dict(updated)
        finally:
            conn.close()


def _get_playlist(playlist_id: str) -> dict | None:
    with _db_lock:
        conn = _get_db()
        try:
            row = conn.execute(
                "SELECT * FROM playlists WHERE playlist_id=?", (playlist_id,)
            ).fetchone()
            if not row:
                return None
            playlist = _row_to_dict(row)
            video_ids = playlist.get("video_ids", [])
            videos = []
            for vid_id in video_ids:
                vrow = conn.execute(
                    "SELECT * FROM videos WHERE video_id=?", (vid_id,)
                ).fetchone()
                if vrow:
                    videos.append(_row_to_dict(vrow))
            playlist["videos"] = videos
            return playlist
        finally:
            conn.close()


# ── comment helpers ────────────────────────────────────────────────────────────

def _add_comment(
    video_id: str,
    author_id: str,
    author_name: str,
    content: str,
    parent_id: str = "",
) -> dict:
    comment_id = f"cmt_{secrets.token_hex(10)}"
    now        = time.time()
    with _db_lock:
        conn = _get_db()
        try:
            conn.execute(
                """INSERT INTO comments
                   (comment_id, video_id, author_id, author_name, content, parent_id, created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (comment_id, video_id, author_id, author_name, content, parent_id or "", now),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM comments WHERE comment_id=?", (comment_id,)
            ).fetchone()
            return dict(row)
        finally:
            conn.close()


def _get_comments(video_id: str) -> list:
    """Return comments in threaded structure (top-level + replies)."""
    with _db_lock:
        conn = _get_db()
        try:
            rows = conn.execute(
                """SELECT * FROM comments WHERE video_id=? AND status='active'
                   ORDER BY created_at ASC""",
                (video_id,),
            ).fetchall()
            all_comments = [dict(r) for r in rows]
            top_level = [c for c in all_comments if not c.get("parent_id")]
            replies_by_parent = {}
            for c in all_comments:
                pid = c.get("parent_id", "")
                if pid:
                    replies_by_parent.setdefault(pid, []).append(c)
            for c in top_level:
                c["replies"] = replies_by_parent.get(c["comment_id"], [])
            return top_level
        finally:
            conn.close()


# ── analytics helpers ──────────────────────────────────────────────────────────

def _video_analytics(video_id: str) -> dict:
    with _db_lock:
        conn = _get_db()
        try:
            vrow = conn.execute(
                "SELECT * FROM videos WHERE video_id=?", (video_id,)
            ).fetchone()
            if not vrow:
                return {}
            v = _row_to_dict(vrow)

            # completion rate
            total_sessions = conn.execute(
                "SELECT COUNT(*) as c FROM watch_sessions WHERE video_id=?", (video_id,)
            ).fetchone()["c"]
            completed = conn.execute(
                "SELECT COUNT(*) as c FROM watch_sessions WHERE video_id=? AND completed=1",
                (video_id,),
            ).fetchone()["c"]
            completion_rate = round(completed / total_sessions, 4) if total_sessions else 0.0

            # views per day (last 30 days)
            cutoff = time.time() - 30 * 86400
            day_rows = conn.execute(
                """SELECT strftime('%Y-%m-%d', created_at, 'unixepoch') as day,
                          COUNT(*) as views
                   FROM watch_sessions
                   WHERE video_id=? AND created_at>=?
                   GROUP BY day ORDER BY day""",
                (video_id, cutoff),
            ).fetchall()
            views_per_day = {r["day"]: r["views"] for r in day_rows}

            return {
                "video_id":             video_id,
                "title":                v.get("title"),
                "view_count":           v.get("view_count", 0),
                "like_count":           v.get("like_count", 0),
                "total_watch_seconds":  v.get("total_watch_seconds", 0),
                "total_watch_hours":    round(v.get("total_watch_seconds", 0) / 3600, 3),
                "total_sessions":       total_sessions,
                "completed_sessions":   completed,
                "completion_rate":      completion_rate,
                "views_per_day":        views_per_day,
            }
        finally:
            conn.close()


def _channel_analytics(channel_id: str) -> dict:
    with _db_lock:
        conn = _get_db()
        try:
            crow = conn.execute(
                "SELECT * FROM channels WHERE channel_id=?", (channel_id,)
            ).fetchone()
            if not crow:
                return {}
            channel = _row_to_dict(crow)

            totals = conn.execute(
                """SELECT SUM(view_count) as total_views,
                          SUM(total_watch_seconds) as total_watch_seconds,
                          COUNT(*) as video_count
                   FROM videos WHERE channel_id=? AND status='active'""",
                (channel_id,),
            ).fetchone()

            top_videos = conn.execute(
                """SELECT video_id, title, view_count, like_count, total_watch_seconds
                   FROM videos WHERE channel_id=? AND status='active'
                   ORDER BY view_count DESC LIMIT 10""",
                (channel_id,),
            ).fetchall()

            return {
                "channel_id":         channel_id,
                "name":               channel.get("name"),
                "total_views":        totals["total_views"] or 0,
                "total_watch_seconds": totals["total_watch_seconds"] or 0,
                "total_watch_hours":  round((totals["total_watch_seconds"] or 0) / 3600, 3),
                "video_count":        totals["video_count"] or 0,
                "subscriber_count":   channel.get("subscriber_count", 0),
                "top_videos":         [dict(r) for r in top_videos],
            }
        finally:
            conn.close()


# ── background daemon ──────────────────────────────────────────────────────────

def _background_daemon() -> None:
    """Periodically refresh channels.video_count from the videos table."""
    while True:
        try:
            time.sleep(3600)
            _log("Background sync: updating channel video counts")
            with _db_lock:
                conn = _get_db()
                try:
                    conn.execute(
                        """UPDATE channels SET video_count=(
                               SELECT COUNT(*) FROM videos
                               WHERE videos.channel_id=channels.channel_id
                               AND videos.status='active'
                           ), updated_at=?""",
                        (time.time(),),
                    )
                    conn.commit()
                    _log("Channel video counts refreshed")
                finally:
                    conn.close()
        except Exception as exc:
            _log(f"Background daemon error: {exc}")


# ── request handler ────────────────────────────────────────────────────────────

class VideoPlatformHandler(BaseHTTPRequestHandler):
    server_version = "VideoPlatform/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args) -> None:  # suppress default access log
        pass

    # ── CORS pre-flight ────────────────────────────────────────────────────────
    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Admin-Secret, X-Uploader-Id")
        self.send_header("Content-Length", "0")
        self.end_headers()

    # ── GET ────────────────────────────────────────────────────────────────────
    def do_GET(self) -> None:
        path, qs = _parse_qs(self.path)
        parts = [p for p in path.split("/") if p]

        # GET /health
        if path == "/health":
            _json_response(self, 200, {
                "status":  "ok",
                "service": "video-platform",
                "port":    PORT,
                "uptime":  round(time.time() - START_TIME, 1),
            })
            return

        # GET /videos
        if path == "/videos":
            videos = _list_videos(
                channel_id=qs.get("channel_id", ""),
                tag=qs.get("tag", ""),
                visibility=qs.get("visibility", ""),
            )
            _json_response(self, 200, {"videos": videos, "count": len(videos)})
            return

        # GET /videos/{video_id}
        if len(parts) == 2 and parts[0] == "videos":
            video = _get_video(parts[1])
            if not video:
                _json_response(self, 404, {"error": "video not found"})
                return
            _json_response(self, 200, video)
            return

        # GET /videos/{video_id}/comments
        if len(parts) == 3 and parts[0] == "videos" and parts[2] == "comments":
            video_id = parts[1]
            video    = _get_video(video_id)
            if not video:
                _json_response(self, 404, {"error": "video not found"})
                return
            comments = _get_comments(video_id)
            _json_response(self, 200, {"comments": comments, "count": len(comments)})
            return

        # GET /channels/{channel_id}
        if len(parts) == 2 and parts[0] == "channels":
            channel = _get_channel(parts[1])
            if not channel:
                _json_response(self, 404, {"error": "channel not found"})
                return
            _json_response(self, 200, channel)
            return

        # GET /playlists/{playlist_id}
        if len(parts) == 2 and parts[0] == "playlists":
            pl = _get_playlist(parts[1])
            if not pl:
                _json_response(self, 404, {"error": "playlist not found"})
                return
            _json_response(self, 200, pl)
            return

        # GET /analytics/{video_id}
        if len(parts) == 2 and parts[0] == "analytics":
            analytics = _video_analytics(parts[1])
            if not analytics:
                _json_response(self, 404, {"error": "video not found"})
                return
            _json_response(self, 200, analytics)
            return

        # GET /analytics/channel/{channel_id}
        if len(parts) == 3 and parts[0] == "analytics" and parts[1] == "channel":
            analytics = _channel_analytics(parts[2])
            if not analytics:
                _json_response(self, 404, {"error": "channel not found"})
                return
            _json_response(self, 200, analytics)
            return

        _json_response(self, 404, {"error": "not found"})

    # ── POST ───────────────────────────────────────────────────────────────────
    def do_POST(self) -> None:
        path, _qs = _parse_qs(self.path)
        parts = [p for p in path.split("/") if p]

        raw = _read_body(self)
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            _json_response(self, 400, {"error": "invalid JSON body"})
            return

        # POST /channels
        if path == "/channels":
            name        = body.get("name", "").strip()
            description = body.get("description", "")
            owner_email = body.get("owner_email", "").strip()
            owner_id    = body.get("owner_id", "")
            if not name or not owner_email:
                _json_response(self, 400, {"error": "name and owner_email are required"})
                return
            channel = _create_channel(name, description, owner_email, owner_id)
            _log(f"Channel created: {channel['channel_id']} by {owner_email}")
            _json_response(self, 201, channel)
            return

        # POST /videos/upload
        if path == "/videos/upload":
            channel_id       = body.get("channel_id", "").strip()
            title            = body.get("title", "").strip()
            description      = body.get("description", "")
            tags             = body.get("tags", [])
            visibility       = body.get("visibility", "public")
            filename         = body.get("filename", "video.mp4").strip()
            content_type     = body.get("content_type", "video/mp4")
            data_b64         = body.get("data_b64", "")
            duration_seconds = float(body.get("duration_seconds", 0))
            uploader_id      = body.get("uploader_id", "")

            if not channel_id or not title or not data_b64:
                _json_response(self, 400, {"error": "channel_id, title, and data_b64 are required"})
                return

            try:
                data_bytes = base64.b64decode(data_b64)
            except Exception:
                _json_response(self, 400, {"error": "invalid base64 in data_b64"})
                return

            if visibility not in ("public", "unlisted", "private"):
                visibility = "public"

            if isinstance(tags, str):
                try:
                    tags = json.loads(tags)
                except Exception:
                    tags = [t.strip() for t in tags.split(",") if t.strip()]

            video = _upload_video(
                channel_id=channel_id,
                title=title,
                description=description,
                tags=tags,
                visibility=visibility,
                filename=filename,
                content_type=content_type,
                data_bytes=data_bytes,
                uploader_id=uploader_id,
                duration_seconds=duration_seconds,
            )
            stream_url = _get_stream_url(video)
            video["stream_url"] = stream_url
            _log(f"Video uploaded: {video['video_id']} to channel {channel_id}")
            _json_response(self, 201, video)
            return

        # POST /videos/{video_id}/view
        if len(parts) == 3 and parts[0] == "videos" and parts[2] == "view":
            video_id  = parts[1]
            video     = _get_video(video_id)
            if not video:
                _json_response(self, 404, {"error": "video not found"})
                return
            viewer_id = body.get("viewer_id", "")
            ip        = body.get("ip", self.client_address[0] if self.client_address else "")
            session   = _record_view(video_id, viewer_id, ip)
            stream_url = _get_stream_url(video)
            _json_response(self, 200, {
                "session_id": session["session_id"],
                "stream_url": stream_url,
                "view_count": (video.get("view_count", 0) + 1),
            })
            return

        # POST /videos/{video_id}/watch
        if len(parts) == 3 and parts[0] == "videos" and parts[2] == "watch":
            video_id      = parts[1]
            session_id    = body.get("session_id", "")
            watch_seconds = float(body.get("watch_seconds", 0))
            completed     = bool(body.get("completed", False))
            if not session_id:
                _json_response(self, 400, {"error": "session_id is required"})
                return
            ok = _update_watch_session(session_id, watch_seconds, completed)
            if not ok:
                _json_response(self, 404, {"error": "watch session not found"})
                return
            _json_response(self, 200, {
                "session_id":    session_id,
                "watch_seconds": watch_seconds,
                "completed":     completed,
                "updated":       True,
            })
            return

        # POST /videos/{video_id}/like
        if len(parts) == 3 and parts[0] == "videos" and parts[2] == "like":
            video_id = parts[1]
            video    = _get_video(video_id)
            if not video:
                _json_response(self, 404, {"error": "video not found"})
                return
            new_count = _like_video(video_id)
            _json_response(self, 200, {"video_id": video_id, "like_count": new_count})
            return

        # POST /videos/{video_id}/comments
        if len(parts) == 3 and parts[0] == "videos" and parts[2] == "comments":
            video_id = parts[1]
            video    = _get_video(video_id)
            if not video:
                _json_response(self, 404, {"error": "video not found"})
                return
            author_id   = body.get("author_id", "").strip()
            author_name = body.get("author_name", "Anonymous").strip()
            content     = body.get("content", "").strip()
            parent_id   = body.get("parent_id", "")
            if not content:
                _json_response(self, 400, {"error": "content is required"})
                return
            comment = _add_comment(video_id, author_id, author_name, content, parent_id)
            _json_response(self, 201, comment)
            return

        # POST /playlists
        if path == "/playlists":
            channel_id  = body.get("channel_id", "").strip()
            title       = body.get("title", "").strip()
            description = body.get("description", "")
            visibility  = body.get("visibility", "public")
            if not channel_id or not title:
                _json_response(self, 400, {"error": "channel_id and title are required"})
                return
            if visibility not in ("public", "unlisted", "private"):
                visibility = "public"
            pl = _create_playlist(channel_id, title, description, visibility)
            _log(f"Playlist created: {pl['playlist_id']} in channel {channel_id}")
            _json_response(self, 201, pl)
            return

        # POST /playlists/{playlist_id}/videos
        if len(parts) == 3 and parts[0] == "playlists" and parts[2] == "videos":
            playlist_id = parts[1]
            video_id    = body.get("video_id", "").strip()
            if not video_id:
                _json_response(self, 400, {"error": "video_id is required"})
                return
            pl = _add_to_playlist(playlist_id, video_id)
            if not pl:
                _json_response(self, 404, {"error": "playlist not found"})
                return
            _json_response(self, 200, pl)
            return

        _json_response(self, 404, {"error": "not found"})

    # ── DELETE ─────────────────────────────────────────────────────────────────
    def do_DELETE(self) -> None:
        path, _qs = _parse_qs(self.path)
        parts = [p for p in path.split("/") if p]

        # DELETE /videos/{video_id}
        if len(parts) == 2 and parts[0] == "videos":
            video_id    = parts[1]
            video       = _get_video(video_id)
            if not video:
                _json_response(self, 404, {"error": "video not found"})
                return

            # Check admin OR owner
            admin_secret  = self.headers.get("X-Admin-Secret", "")
            uploader_id   = self.headers.get("X-Uploader-Id", "")
            is_admin      = bool(
                ADMIN_SECRET
                and admin_secret
                and hmac.compare_digest(admin_secret, ADMIN_SECRET)
            )
            is_owner      = bool(
                uploader_id
                and video.get("uploader_id")
                and hmac.compare_digest(uploader_id, video["uploader_id"])
            )

            if not is_admin and not is_owner:
                _json_response(self, 403, {"error": "forbidden – admin or owner required"})
                return

            ok = _delete_video(video_id)
            if not ok:
                _json_response(self, 500, {"error": "delete failed"})
                return
            _log(f"Video deleted: {video_id}")
            _json_response(self, 200, {"deleted": True, "video_id": video_id})
            return

        _json_response(self, 404, {"error": "not found"})


# ── server entry point ─────────────────────────────────────────────────────────

def _run_server() -> None:
    _init_db()

    # start background daemon
    daemon = threading.Thread(target=_background_daemon, daemon=True, name="video-platform-daemon")
    daemon.start()

    server = HTTPServer(("0.0.0.0", PORT), VideoPlatformHandler)
    _log(f"Video Platform & Streaming Manager listening on port {PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log("Shutting down")
    finally:
        server.server_close()


if __name__ == "__main__":
    _run_server()
