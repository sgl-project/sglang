#!/usr/bin/env python3
"""
FractalMesh OMEGA Titan — Social Media Manager
Port: 7858

Unified social media management hub. Schedules and publishes posts across
Twitter/X, Dev.to, and Reddit. Tracks engagement metrics. Content queue
with approval workflow.

Samuel James Hiotis | ABN 56 628 117 363
"""

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ---------------------------------------------------------------------------
# Vault loading — never hardcode credentials
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
PORT                       = int(os.environ.get("SOCIAL_MANAGER_PORT", "7858"))
TWITTER_API_KEY            = os.environ.get("TWITTER_API_KEY", "")
TWITTER_API_SECRET         = os.environ.get("TWITTER_API_SECRET", "")
TWITTER_ACCESS_TOKEN       = os.environ.get("TWITTER_ACCESS_TOKEN", "")
TWITTER_ACCESS_TOKEN_SECRET = os.environ.get("TWITTER_ACCESS_TOKEN_SECRET", "")
TWITTER_BEARER_TOKEN       = os.environ.get("TWITTER_BEARER_TOKEN", "")
DEVTO_API_KEY              = os.environ.get("DEVTO_API_KEY", "")
REDDIT_CLIENT_ID           = os.environ.get("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET       = os.environ.get("REDDIT_CLIENT_SECRET", "")
REDDIT_USERNAME            = os.environ.get("REDDIT_USERNAME", "")
REDDIT_PASSWORD            = os.environ.get("REDDIT_PASSWORD", "")
ADMIN_SECRET               = os.environ.get("ADMIN_SECRET", "")

BASE_DIR = Path.home() / "fmsaas"
DB_PATH  = BASE_DIR / "database" / "sovereign.db"
LOG_PATH = BASE_DIR / "logs" / "fm_social_manager.log"

for _d in (DB_PATH.parent, LOG_PATH.parent):
    _d.mkdir(parents=True, exist_ok=True)

START_TIME = time.time()

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS posts (
            id               INTEGER PRIMARY KEY,
            platform         TEXT,
            content          TEXT,
            title            TEXT,
            tags             TEXT,
            subreddit        TEXT,
            scheduled_for    REAL,
            published_at     REAL,
            status           TEXT DEFAULT 'draft',
            platform_post_id TEXT,
            engagement_score REAL DEFAULT 0,
            created_at       REAL
        );

        CREATE TABLE IF NOT EXISTS metrics (
            id         INTEGER PRIMARY KEY,
            post_id    INTEGER,
            platform   TEXT,
            likes      INTEGER DEFAULT 0,
            comments   INTEGER DEFAULT 0,
            shares     INTEGER DEFAULT 0,
            views      INTEGER DEFAULT 0,
            fetched_at REAL
        );

        CREATE TABLE IF NOT EXISTS accounts (
            id          INTEGER PRIMARY KEY,
            platform    TEXT UNIQUE,
            username    TEXT,
            followers   INTEGER DEFAULT 0,
            following   INTEGER DEFAULT 0,
            verified    INTEGER DEFAULT 0,
            last_synced REAL
        );

        CREATE INDEX IF NOT EXISTS idx_posts_status    ON posts(status);
        CREATE INDEX IF NOT EXISTS idx_posts_platform  ON posts(platform);
        CREATE INDEX IF NOT EXISTS idx_metrics_post_id ON metrics(post_id);
    """)
    conn.commit()
    conn.close()

# ---------------------------------------------------------------------------
# Twitter OAuth 1.0a helper
# ---------------------------------------------------------------------------

def _percent_encode(s: str) -> str:
    return urllib.parse.quote(str(s), safe="")


def _twitter_oauth_header(method: str, url: str, body_params: dict) -> str:
    """Build OAuth 1.0a Authorization header for Twitter API v2."""
    oauth_params = {
        "oauth_consumer_key":     TWITTER_API_KEY,
        "oauth_nonce":            secrets.token_hex(16),
        "oauth_signature_method": "HMAC-SHA1",
        "oauth_timestamp":        str(int(time.time())),
        "oauth_token":            TWITTER_ACCESS_TOKEN,
        "oauth_version":          "1.0",
    }

    # Merge oauth params + body params for signature
    all_params = {}
    all_params.update(oauth_params)
    all_params.update(body_params)

    # Sort and encode
    sorted_params = "&".join(
        f"{_percent_encode(k)}={_percent_encode(v)}"
        for k, v in sorted(all_params.items())
    )

    base_string = "&".join([
        _percent_encode(method.upper()),
        _percent_encode(url),
        _percent_encode(sorted_params),
    ])

    signing_key = (
        _percent_encode(TWITTER_API_SECRET) + "&" +
        _percent_encode(TWITTER_ACCESS_TOKEN_SECRET)
    )

    signature = base64.b64encode(
        hmac.new(
            signing_key.encode("utf-8"),
            base_string.encode("utf-8"),
            hashlib.sha1,
        ).digest()
    ).decode("utf-8")

    oauth_params["oauth_signature"] = signature

    header_parts = ", ".join(
        f'{_percent_encode(k)}="{_percent_encode(v)}"'
        for k, v in sorted(oauth_params.items())
    )
    return f"OAuth {header_parts}"


def publish_twitter(content: str, **_kwargs) -> tuple[bool, str]:
    """Post a tweet via Twitter API v2 with OAuth 1.0a."""
    if not all([TWITTER_API_KEY, TWITTER_API_SECRET,
                TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET]):
        return False, "Twitter credentials not configured"

    url = "https://api.twitter.com/2/tweets"
    payload = json.dumps({"text": content}).encode("utf-8")

    auth_header = _twitter_oauth_header("POST", url, {})

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization":  auth_header,
            "Content-Type":   "application/json",
            "Accept":         "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            tweet_id = data.get("data", {}).get("id", "")
            return True, tweet_id
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return False, f"HTTP {exc.code}: {body}"
    except Exception as exc:
        return False, str(exc)

# ---------------------------------------------------------------------------
# Dev.to publisher
# ---------------------------------------------------------------------------

def publish_devto(content: str, title: str = "", tags=None, **_kwargs) -> tuple[bool, str]:
    """Publish an article to Dev.to."""
    if not DEVTO_API_KEY:
        return False, "DEVTO_API_KEY not configured"

    tags_list = tags if isinstance(tags, list) else (tags.split(",") if tags else [])
    tags_list = [t.strip() for t in tags_list if t.strip()]

    article = {
        "article": {
            "title":         title or "Untitled",
            "body_markdown": content,
            "published":     True,
            "tags":          tags_list,
        }
    }
    payload = json.dumps(article).encode("utf-8")

    req = urllib.request.Request(
        "https://dev.to/api/articles",
        data=payload,
        headers={
            "api-key":      DEVTO_API_KEY,
            "Content-Type": "application/json",
            "Accept":       "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            article_id = str(data.get("id", ""))
            return True, article_id
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return False, f"HTTP {exc.code}: {body}"
    except Exception as exc:
        return False, str(exc)

# ---------------------------------------------------------------------------
# Reddit publisher
# ---------------------------------------------------------------------------

def _reddit_access_token() -> tuple[bool, str]:
    """Obtain Reddit OAuth2 bearer token via password grant."""
    if not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET,
                REDDIT_USERNAME, REDDIT_PASSWORD]):
        return False, "Reddit credentials not configured"

    credentials = base64.b64encode(
        f"{REDDIT_CLIENT_ID}:{REDDIT_CLIENT_SECRET}".encode()
    ).decode()

    body = urllib.parse.urlencode({
        "grant_type": "password",
        "username":   REDDIT_USERNAME,
        "password":   REDDIT_PASSWORD,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://www.reddit.com/api/v1/access_token",
        data=body,
        headers={
            "Authorization": f"Basic {credentials}",
            "Content-Type":  "application/x-www-form-urlencoded",
            "User-Agent":    "FractalMesh/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            token = data.get("access_token", "")
            if not token:
                return False, f"No access_token in response: {data}"
            return True, token
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return False, f"HTTP {exc.code}: {body}"
    except Exception as exc:
        return False, str(exc)


def publish_reddit(content: str, title: str = "", subreddit: str = "", **_kwargs) -> tuple[bool, str]:
    """Submit a self post to Reddit."""
    if not subreddit:
        return False, "subreddit is required for Reddit posts"

    ok, token = _reddit_access_token()
    if not ok:
        return False, token

    body = urllib.parse.urlencode({
        "kind":  "self",
        "sr":    subreddit,
        "title": title or "Untitled",
        "text":  content,
        "api_type": "json",
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://oauth.reddit.com/api/submit",
        data=body,
        headers={
            "Authorization": f"bearer {token}",
            "Content-Type":  "application/x-www-form-urlencoded",
            "User-Agent":    "FractalMesh/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            # Reddit wraps response in json.data
            json_data = data.get("json", {})
            errors = json_data.get("errors", [])
            if errors:
                return False, str(errors)
            post_url = json_data.get("data", {}).get("url", "")
            post_id  = json_data.get("data", {}).get("id", "")
            return True, post_id or post_url
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return False, f"HTTP {exc.code}: {body}"
    except Exception as exc:
        return False, str(exc)

# ---------------------------------------------------------------------------
# Dispatch publish by platform
# ---------------------------------------------------------------------------

_PUBLISHERS = {
    "twitter": publish_twitter,
    "devto":   publish_devto,
    "reddit":  publish_reddit,
}


def publish_post(post_id: int) -> tuple[bool, str]:
    """Look up post and dispatch to correct publisher."""
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT * FROM posts WHERE id=?", (post_id,)
        ).fetchone()
        if not row:
            return False, "Post not found"

        platform = (row["platform"] or "").lower()
        publisher = _PUBLISHERS.get(platform)
        if not publisher:
            return False, f"Unknown platform: {platform}"

        tags_raw = row["tags"] or ""
        try:
            tags_parsed = json.loads(tags_raw)
        except Exception:
            tags_parsed = [t.strip() for t in tags_raw.split(",") if t.strip()]

        ok, platform_post_id = publisher(
            content   = row["content"] or "",
            title     = row["title"] or "",
            tags      = tags_parsed,
            subreddit = row["subreddit"] or "",
        )

        now = time.time()
        if ok:
            conn.execute(
                """UPDATE posts
                   SET status='published', published_at=?, platform_post_id=?
                   WHERE id=?""",
                (now, platform_post_id, post_id),
            )
        else:
            conn.execute(
                "UPDATE posts SET status='failed' WHERE id=?",
                (post_id,),
            )
        conn.commit()
        return ok, platform_post_id
    finally:
        conn.close()

# ---------------------------------------------------------------------------
# Account follower sync
# ---------------------------------------------------------------------------

def _sync_twitter_account() -> None:
    """Fetch Twitter account info via v2 /users/me."""
    if not TWITTER_BEARER_TOKEN:
        return
    req = urllib.request.Request(
        "https://api.twitter.com/2/users/me?user.fields=public_metrics,verified",
        headers={
            "Authorization": f"Bearer {TWITTER_BEARER_TOKEN}",
            "Accept":        "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read()).get("data", {})
            metrics = data.get("public_metrics", {})
            conn = get_db()
            conn.execute(
                """INSERT INTO accounts (platform, username, followers, following, verified, last_synced)
                   VALUES ('twitter', ?, ?, ?, ?, ?)
                   ON CONFLICT(platform) DO UPDATE SET
                       username=excluded.username,
                       followers=excluded.followers,
                       following=excluded.following,
                       verified=excluded.verified,
                       last_synced=excluded.last_synced""",
                (
                    data.get("username", REDDIT_USERNAME),
                    metrics.get("followers_count", 0),
                    metrics.get("following_count", 0),
                    1 if data.get("verified") else 0,
                    time.time(),
                ),
            )
            conn.commit()
            conn.close()
    except Exception:
        pass


def _sync_devto_account() -> None:
    """Fetch Dev.to profile follower count."""
    if not DEVTO_API_KEY:
        return
    req = urllib.request.Request(
        "https://dev.to/api/users/me",
        headers={"api-key": DEVTO_API_KEY, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            conn = get_db()
            conn.execute(
                """INSERT INTO accounts (platform, username, followers, following, verified, last_synced)
                   VALUES ('devto', ?, ?, 0, 0, ?)
                   ON CONFLICT(platform) DO UPDATE SET
                       username=excluded.username,
                       followers=excluded.followers,
                       last_synced=excluded.last_synced""",
                (
                    data.get("username", ""),
                    data.get("followers_count", 0),
                    time.time(),
                ),
            )
            conn.commit()
            conn.close()
    except Exception:
        pass


def _sync_reddit_account() -> None:
    """Fetch Reddit account karma as proxy for reach."""
    ok, token = _reddit_access_token()
    if not ok:
        return
    req = urllib.request.Request(
        "https://oauth.reddit.com/api/v1/me",
        headers={
            "Authorization": f"bearer {token}",
            "User-Agent":    "FractalMesh/1.0",
            "Accept":        "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            conn = get_db()
            conn.execute(
                """INSERT INTO accounts (platform, username, followers, following, verified, last_synced)
                   VALUES ('reddit', ?, ?, 0, ?, ?)
                   ON CONFLICT(platform) DO UPDATE SET
                       username=excluded.username,
                       followers=excluded.followers,
                       verified=excluded.verified,
                       last_synced=excluded.last_synced""",
                (
                    data.get("name", REDDIT_USERNAME),
                    data.get("link_karma", 0) + data.get("comment_karma", 0),
                    1 if data.get("verified") else 0,
                    time.time(),
                ),
            )
            conn.commit()
            conn.close()
    except Exception:
        pass


def sync_all_accounts() -> None:
    _sync_twitter_account()
    _sync_devto_account()
    _sync_reddit_account()

# ---------------------------------------------------------------------------
# Background scheduler
# ---------------------------------------------------------------------------

def _scheduler_loop() -> None:
    last_account_sync = 0.0
    while True:
        try:
            now = time.time()

            # Publish due scheduled posts
            conn = get_db()
            due = conn.execute(
                "SELECT id FROM posts WHERE status='scheduled' AND scheduled_for <= ?",
                (now,),
            ).fetchall()
            conn.close()

            for row in due:
                try:
                    publish_post(row["id"])
                except Exception:
                    pass

            # Sync accounts every hour
            if now - last_account_sync >= 3600:
                try:
                    sync_all_accounts()
                except Exception:
                    pass
                last_account_sync = now

        except Exception:
            pass

        time.sleep(60)


def start_scheduler() -> None:
    t = threading.Thread(target=_scheduler_loop, daemon=True, name="social-scheduler")
    t.start()

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _json_response(handler: BaseHTTPRequestHandler, code: int, data: object) -> None:
    body = json.dumps(data, default=str).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_body(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", 0))
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _check_admin(handler: BaseHTTPRequestHandler) -> bool:
    if not ADMIN_SECRET:
        return True
    auth = handler.headers.get("X-Admin-Secret", "")
    if hmac.compare_digest(auth, ADMIN_SECRET):
        return True
    # Also accept Bearer token
    bearer = handler.headers.get("Authorization", "")
    if bearer.startswith("Bearer ") and hmac.compare_digest(bearer[7:], ADMIN_SECRET):
        return True
    return False


def _parse_iso(ts: str) -> float:
    """Parse ISO 8601 timestamp to Unix float (best-effort)."""
    ts = ts.strip()
    # Remove trailing Z and replace T with space
    ts = ts.replace("Z", "+00:00").replace("T", " ")
    # Try various formats
    for fmt in (
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M%z",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    ):
        try:
            import datetime
            dt = datetime.datetime.strptime(ts.replace("+00:00", ""), fmt.replace("%z", ""))
            return dt.timestamp()
        except Exception:
            pass
    return float(ts)  # fallback: assume already numeric

# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

def handle_health(handler: BaseHTTPRequestHandler) -> None:
    conn = get_db()
    counts_raw = conn.execute(
        "SELECT platform, COUNT(*) as cnt FROM posts GROUP BY platform"
    ).fetchall()
    sched = conn.execute(
        "SELECT COUNT(*) as cnt FROM posts WHERE status='scheduled'"
    ).fetchone()["cnt"]
    conn.close()

    counts = {row["platform"]: row["cnt"] for row in counts_raw}
    _json_response(handler, 200, {
        "status":          "ok",
        "uptime_seconds":  round(time.time() - START_TIME, 1),
        "port":            PORT,
        "posts_by_platform": counts,
        "scheduled_count": sched,
    })


def handle_get_posts(handler: BaseHTTPRequestHandler) -> None:
    parsed = urllib.parse.urlparse(handler.path)
    qs = urllib.parse.parse_qs(parsed.query)

    platform = qs.get("platform", [None])[0]
    status   = qs.get("status", [None])[0]
    limit    = int(qs.get("limit", ["50"])[0])

    where_clauses = []
    params: list = []
    if platform:
        where_clauses.append("platform=?")
        params.append(platform)
    if status:
        where_clauses.append("status=?")
        params.append(status)

    where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    params.append(limit)

    conn = get_db()
    rows = conn.execute(
        f"SELECT * FROM posts {where} ORDER BY created_at DESC LIMIT ?",
        params,
    ).fetchall()
    conn.close()

    _json_response(handler, 200, [dict(r) for r in rows])


def handle_get_post(handler: BaseHTTPRequestHandler, post_id: int) -> None:
    conn = get_db()
    row = conn.execute("SELECT * FROM posts WHERE id=?", (post_id,)).fetchone()
    if not row:
        conn.close()
        _json_response(handler, 404, {"error": "Post not found"})
        return

    post = dict(row)
    metrics_rows = conn.execute(
        "SELECT * FROM metrics WHERE post_id=? ORDER BY fetched_at DESC",
        (post_id,),
    ).fetchall()
    conn.close()

    post["metrics"] = [dict(m) for m in metrics_rows]
    _json_response(handler, 200, post)


def handle_get_accounts(handler: BaseHTTPRequestHandler) -> None:
    conn = get_db()
    rows = conn.execute("SELECT * FROM accounts ORDER BY platform").fetchall()
    conn.close()
    _json_response(handler, 200, [dict(r) for r in rows])


def handle_get_metrics(handler: BaseHTTPRequestHandler) -> None:
    conn = get_db()
    rows = conn.execute(
        """SELECT m.platform,
                  SUM(m.likes)    AS total_likes,
                  SUM(m.comments) AS total_comments,
                  SUM(m.shares)   AS total_shares,
                  SUM(m.views)    AS total_views,
                  COUNT(*)        AS data_points
           FROM metrics m
           GROUP BY m.platform"""
    ).fetchall()
    conn.close()
    _json_response(handler, 200, [dict(r) for r in rows])


def handle_get_scheduled(handler: BaseHTTPRequestHandler) -> None:
    conn = get_db()
    rows = conn.execute(
        """SELECT * FROM posts
           WHERE status='scheduled'
           ORDER BY scheduled_for ASC
           LIMIT 100"""
    ).fetchall()
    conn.close()
    _json_response(handler, 200, [dict(r) for r in rows])


def handle_create_post(handler: BaseHTTPRequestHandler) -> None:
    body = _read_body(handler)

    platform  = body.get("platform", "").lower().strip()
    content   = body.get("content", "").strip()
    title     = body.get("title", "")
    tags      = body.get("tags", [])
    subreddit = body.get("subreddit", "")
    sched_raw = body.get("scheduled_for", None)

    if not platform:
        _json_response(handler, 400, {"error": "platform is required"})
        return
    if platform not in _PUBLISHERS:
        _json_response(handler, 400, {"error": f"Unknown platform: {platform}. Valid: {list(_PUBLISHERS)}"})
        return
    if not content:
        _json_response(handler, 400, {"error": "content is required"})
        return

    tags_json = json.dumps(tags) if isinstance(tags, list) else str(tags)

    scheduled_for = None
    if sched_raw:
        try:
            scheduled_for = _parse_iso(str(sched_raw))
        except Exception:
            _json_response(handler, 400, {"error": "Invalid scheduled_for timestamp"})
            return

    now = time.time()
    status = "scheduled" if scheduled_for else "draft"

    conn = get_db()
    cur = conn.execute(
        """INSERT INTO posts
               (platform, content, title, tags, subreddit,
                scheduled_for, status, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (platform, content, title, tags_json, subreddit,
         scheduled_for, status, now),
    )
    post_id = cur.lastrowid
    conn.commit()

    row = conn.execute("SELECT * FROM posts WHERE id=?", (post_id,)).fetchone()
    conn.close()

    _json_response(handler, 201, dict(row))


def handle_publish_post(handler: BaseHTTPRequestHandler, post_id: int) -> None:
    if not _check_admin(handler):
        _json_response(handler, 403, {"error": "Forbidden"})
        return

    ok, result = publish_post(post_id)
    if ok:
        _json_response(handler, 200, {"success": True, "platform_post_id": result})
    else:
        _json_response(handler, 502, {"success": False, "error": result})


def handle_approve_post(handler: BaseHTTPRequestHandler, post_id: int) -> None:
    if not _check_admin(handler):
        _json_response(handler, 403, {"error": "Forbidden"})
        return

    conn = get_db()
    row = conn.execute("SELECT * FROM posts WHERE id=?", (post_id,)).fetchone()
    if not row:
        conn.close()
        _json_response(handler, 404, {"error": "Post not found"})
        return

    if row["status"] not in ("draft",):
        conn.close()
        _json_response(handler, 400, {"error": f"Cannot approve post with status '{row['status']}'"})
        return

    now = time.time()
    if row["scheduled_for"] and row["scheduled_for"] > now:
        # Keep the schedule, just mark approved (still 'scheduled')
        conn.execute(
            "UPDATE posts SET status='scheduled' WHERE id=?", (post_id,)
        )
        conn.commit()
        updated = conn.execute("SELECT * FROM posts WHERE id=?", (post_id,)).fetchone()
        conn.close()
        _json_response(handler, 200, {"message": "approved, will publish at scheduled time", "post": dict(updated)})
    else:
        # Publish immediately
        conn.close()
        ok, result = publish_post(post_id)
        if ok:
            _json_response(handler, 200, {"success": True, "platform_post_id": result})
        else:
            _json_response(handler, 502, {"success": False, "error": result})


def handle_update_post(handler: BaseHTTPRequestHandler, post_id: int) -> None:
    conn = get_db()
    row = conn.execute("SELECT * FROM posts WHERE id=?", (post_id,)).fetchone()
    if not row:
        conn.close()
        _json_response(handler, 404, {"error": "Post not found"})
        return

    if row["status"] not in ("draft", "scheduled"):
        conn.close()
        _json_response(handler, 400, {"error": f"Cannot edit post with status '{row['status']}'"})
        return

    body = _read_body(handler)

    content       = body.get("content", row["content"])
    title         = body.get("title", row["title"])
    tags          = body.get("tags", None)
    subreddit     = body.get("subreddit", row["subreddit"])
    sched_raw     = body.get("scheduled_for", None)

    tags_json = row["tags"]
    if tags is not None:
        tags_json = json.dumps(tags) if isinstance(tags, list) else str(tags)

    scheduled_for = row["scheduled_for"]
    if sched_raw is not None:
        if sched_raw == "" or sched_raw is None:
            scheduled_for = None
        else:
            try:
                scheduled_for = _parse_iso(str(sched_raw))
            except Exception:
                conn.close()
                _json_response(handler, 400, {"error": "Invalid scheduled_for timestamp"})
                return

    # Re-evaluate status
    now = time.time()
    if scheduled_for and scheduled_for > now:
        new_status = "scheduled"
    else:
        new_status = row["status"]

    conn.execute(
        """UPDATE posts
           SET content=?, title=?, tags=?, subreddit=?,
               scheduled_for=?, status=?
           WHERE id=?""",
        (content, title, tags_json, subreddit, scheduled_for, new_status, post_id),
    )
    conn.commit()
    updated = conn.execute("SELECT * FROM posts WHERE id=?", (post_id,)).fetchone()
    conn.close()
    _json_response(handler, 200, dict(updated))


def handle_delete_post(handler: BaseHTTPRequestHandler, post_id: int) -> None:
    if not _check_admin(handler):
        _json_response(handler, 403, {"error": "Forbidden"})
        return

    conn = get_db()
    row = conn.execute("SELECT id FROM posts WHERE id=?", (post_id,)).fetchone()
    if not row:
        conn.close()
        _json_response(handler, 404, {"error": "Post not found"})
        return

    conn.execute("DELETE FROM metrics WHERE post_id=?", (post_id,))
    conn.execute("DELETE FROM posts WHERE id=?", (post_id,))
    conn.commit()
    conn.close()
    _json_response(handler, 200, {"deleted": post_id})

# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

# Compiled route patterns
_RE_POST_ID        = re.compile(r"^/posts/(\d+)$")
_RE_POST_PUBLISH   = re.compile(r"^/posts/(\d+)/publish$")
_RE_POST_APPROVE   = re.compile(r"^/posts/(\d+)/approve$")


class SocialManagerHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):  # silence default access log
        pass

    def _path(self) -> str:
        return urllib.parse.urlparse(self.path).path.rstrip("/") or "/"

    # ------------------------------------------------------------------ GET

    def do_GET(self):
        path = self._path()

        if path == "/health":
            handle_health(self)
            return

        if path == "/posts":
            handle_get_posts(self)
            return

        m = _RE_POST_ID.match(path)
        if m:
            handle_get_post(self, int(m.group(1)))
            return

        if path == "/accounts":
            handle_get_accounts(self)
            return

        if path == "/metrics":
            handle_get_metrics(self)
            return

        if path == "/scheduled":
            handle_get_scheduled(self)
            return

        _json_response(self, 404, {"error": "Not found", "path": path})

    # ------------------------------------------------------------------ POST

    def do_POST(self):
        path = self._path()

        if path == "/posts":
            handle_create_post(self)
            return

        m = _RE_POST_PUBLISH.match(path)
        if m:
            handle_publish_post(self, int(m.group(1)))
            return

        m = _RE_POST_APPROVE.match(path)
        if m:
            handle_approve_post(self, int(m.group(1)))
            return

        _json_response(self, 404, {"error": "Not found", "path": path})

    # ------------------------------------------------------------------ PUT

    def do_PUT(self):
        path = self._path()

        m = _RE_POST_ID.match(path)
        if m:
            handle_update_post(self, int(m.group(1)))
            return

        _json_response(self, 404, {"error": "Not found", "path": path})

    # ------------------------------------------------------------------ DELETE

    def do_DELETE(self):
        path = self._path()

        m = _RE_POST_ID.match(path)
        if m:
            handle_delete_post(self, int(m.group(1)))
            return

        _json_response(self, 404, {"error": "Not found", "path": path})

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    init_db()
    start_scheduler()

    server = HTTPServer(("0.0.0.0", PORT), SocialManagerHandler)
    server.socket.setsockopt(6, 1, 1)  # TCP_NODELAY via IPPROTO_TCP=6, TCP_NODELAY=1
    print(f"[SocialManager] Listening on port {PORT}", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[SocialManager] Shutting down.", flush=True)
        server.server_close()


if __name__ == "__main__":
    main()
