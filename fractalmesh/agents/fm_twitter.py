"""
FractalMesh OMEGA Titan — Twitter/X API v2 Agent
Port: 7816
"""

import os
import json
import sqlite3
import logging
import signal
import time
import hmac
import hashlib
import base64
import urllib.request
import urllib.error
import urllib.parse
import secrets
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# ---------------------------------------------------------------------------
# Vault loading
# ---------------------------------------------------------------------------
_VAULT = Path.home() / ".secrets" / "fractal.env"
if _VAULT.exists():
    with open(_VAULT) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
BASE_DIR = Path.home() / "fmsaas"
DB_PATH  = BASE_DIR / "database" / "sovereign.db"
LOG_PATH = BASE_DIR / "logs" / "fm_twitter.log"

for _d in (DB_PATH.parent, LOG_PATH.parent):
    _d.mkdir(parents=True, exist_ok=True)

PORT                    = int(os.environ.setdefault("TWITTER_PORT", "7816"))
TWITTER_API_KEY         = os.getenv("TWITTER_API_KEY", "")
TWITTER_API_SECRET      = os.getenv("TWITTER_API_SECRET", "")
TWITTER_ACCESS_TOKEN    = os.getenv("TWITTER_ACCESS_TOKEN", "")
TWITTER_ACCESS_SECRET   = os.getenv("TWITTER_ACCESS_TOKEN_SECRET", "")
TWITTER_BEARER_TOKEN    = os.getenv("TWITTER_BEARER_TOKEN", "")
BASE_URL                = "https://api.twitter.com/2"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("fm_twitter")

# ---------------------------------------------------------------------------
# SQLite — WAL mode
# ---------------------------------------------------------------------------
def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS twitter_posts (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            tweet_id  TEXT,
            text      TEXT,
            likes     INTEGER DEFAULT 0,
            retweets  INTEGER DEFAULT 0,
            ts        TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    return conn


_db = _get_db()

# ---------------------------------------------------------------------------
# OAuth 1.0a helpers
# ---------------------------------------------------------------------------
def _pct(value: str) -> str:
    """RFC 3986 percent-encode."""
    return urllib.parse.quote(str(value), safe="")


def _oauth_header(method: str, url: str, params: dict = None, body: dict = None) -> str:
    """
    Generate an OAuth 1.0a Authorization header using HMAC-SHA256.

    method  : HTTP verb (GET, POST, …)
    url     : full endpoint URL (no query string)
    params  : query-string parameters already on the URL
    body    : form-encoded body parameters (for POST with form body)
    """
    oauth_params = {
        "oauth_consumer_key":     TWITTER_API_KEY,
        "oauth_token":            TWITTER_ACCESS_TOKEN,
        "oauth_signature_method": "HMAC-SHA256",
        "oauth_version":          "1.0",
        "oauth_nonce":            secrets.token_hex(16),
        "oauth_timestamp":        str(int(time.time())),
    }

    # Merge all parameters for signature base string
    all_params = {}
    if params:
        all_params.update(params)
    if body:
        all_params.update(body)
    all_params.update(oauth_params)

    # Percent-encode keys and values, then sort
    encoded_params = sorted(
        (_pct(k), _pct(v)) for k, v in all_params.items()
    )
    param_string = "&".join(f"{k}={v}" for k, v in encoded_params)

    # Signature base string
    base_string = "&".join([
        method.upper(),
        _pct(url),
        _pct(param_string),
    ])

    # Signing key
    signing_key = f"{_pct(TWITTER_API_SECRET)}&{_pct(TWITTER_ACCESS_SECRET)}"

    # HMAC-SHA256
    sig = hmac.new(
        signing_key.encode(),
        base_string.encode(),
        hashlib.sha256,
    ).digest()
    oauth_params["oauth_signature"] = base64.b64encode(sig).decode()

    # Build Authorization header — only oauth_ params go here
    auth_parts = ", ".join(
        f'{_pct(k)}="{_pct(v)}"'
        for k, v in sorted(oauth_params.items())
    )
    return f"OAuth {auth_parts}"


def _bearer_header() -> dict:
    return {"Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"}


def _oauth1_headers(method: str, url: str, params: dict = None, body: dict = None) -> dict:
    return {"Authorization": _oauth_header(method, url, params=params, body=body)}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def _tw_get(path: str, params: dict = None, bearer: bool = False) -> dict:
    qs = ("?" + urllib.parse.urlencode(params)) if params else ""
    url_base = f"{BASE_URL}{path}"
    url = url_base + qs
    if bearer:
        headers = _bearer_header()
    else:
        headers = _oauth1_headers("GET", url_base, params=params)
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _tw_post(path: str, payload: dict, user_id: str = None) -> dict:
    url = f"{BASE_URL}{path}"
    data = json.dumps(payload).encode()
    headers = _oauth1_headers("POST", url)
    headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _get_my_id() -> str:
    """Fetch the authenticated user's numeric ID."""
    data = _tw_get("/users/me", params={"user.fields": "id,name,username"})
    return data.get("data", {}).get("id", "")


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------
_running = True


def _handle_signal(signum, frame):
    global _running
    log.info("Received signal %s — shutting down.", signum)
    _running = False


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------
def _json_body(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", 0))
    if length == 0:
        return {}
    return json.loads(handler.rfile.read(length).decode())


def _send_json(handler: BaseHTTPRequestHandler, code: int, payload: dict):
    body = json.dumps(payload, default=str).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class TwitterHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        log.debug("HTTP %s", fmt % args)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")
        qs = urllib.parse.parse_qs(parsed.query)

        try:
            if path == "/health":
                self._health()
            elif path == "/timeline":
                self._timeline()
            elif path == "/mentions":
                self._mentions()
            elif path == "/followers":
                self._followers()
            elif path == "/search":
                query = qs.get("q", [""])[0] or qs.get("query", [""])[0]
                self._search(query)
            else:
                _send_json(self, 404, {"error": "not found"})
        except urllib.error.HTTPError as exc:
            body = exc.read().decode() if hasattr(exc, "read") else str(exc)
            _send_json(self, exc.code, {"error": str(exc), "detail": body})
        except Exception as exc:
            log.exception("Unhandled error in GET %s", path)
            _send_json(self, 500, {"error": str(exc)})

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")
        try:
            if path == "/tweet":
                self._tweet()
            elif path == "/reply":
                self._reply()
            elif path == "/like":
                self._like()
            elif path == "/retweet":
                self._retweet()
            else:
                _send_json(self, 404, {"error": "not found"})
        except urllib.error.HTTPError as exc:
            body = exc.read().decode() if hasattr(exc, "read") else str(exc)
            _send_json(self, exc.code, {"error": str(exc), "detail": body})
        except Exception as exc:
            log.exception("Unhandled error in POST %s", path)
            _send_json(self, 500, {"error": str(exc)})

    # ---- handlers ----

    def _health(self):
        me = _tw_get("/users/me", params={"user.fields": "id,name,username,public_metrics"})
        _send_json(self, 200, {
            "status": "ok",
            "agent": "fm_twitter",
            "port": PORT,
            "account": me.get("data"),
        })

    def _tweet(self):
        body = _json_body(self)
        text = body.get("text", "")
        if not text:
            _send_json(self, 400, {"error": "text required"})
            return
        result = _tw_post("/tweets", {"text": text})
        tweet_data = result.get("data", {})
        tweet_id = tweet_data.get("id", "")
        _db.execute(
            "INSERT INTO twitter_posts (tweet_id, text) VALUES (?,?)",
            (tweet_id, text),
        )
        _db.commit()
        log.info("Tweet posted: %s", tweet_id)
        _send_json(self, 200, {
            "tweet_id": tweet_id,
            "url": f"https://twitter.com/i/web/status/{tweet_id}",
            "data": tweet_data,
        })

    def _reply(self):
        body = _json_body(self)
        text = body.get("text", "")
        reply_to = body.get("in_reply_to_tweet_id", "")
        if not text or not reply_to:
            _send_json(self, 400, {"error": "text and in_reply_to_tweet_id required"})
            return
        payload = {"text": text, "reply": {"in_reply_to_tweet_id": reply_to}}
        result = _tw_post("/tweets", payload)
        tweet_data = result.get("data", {})
        tweet_id = tweet_data.get("id", "")
        _db.execute(
            "INSERT INTO twitter_posts (tweet_id, text) VALUES (?,?)",
            (tweet_id, text),
        )
        _db.commit()
        log.info("Reply posted: %s -> %s", tweet_id, reply_to)
        _send_json(self, 200, {
            "tweet_id": tweet_id,
            "reply_to": reply_to,
            "url": f"https://twitter.com/i/web/status/{tweet_id}",
        })

    def _timeline(self):
        uid = _get_my_id()
        params = {
            "max_results": "20",
            "tweet.fields": "public_metrics,created_at",
        }
        data = _tw_get(f"/users/{uid}/tweets", params=params)
        _send_json(self, 200, data)

    def _like(self):
        body = _json_body(self)
        tweet_id = body.get("tweet_id", "")
        if not tweet_id:
            _send_json(self, 400, {"error": "tweet_id required"})
            return
        uid = _get_my_id()
        result = _tw_post(f"/users/{uid}/likes", {"tweet_id": tweet_id})
        log.info("Liked tweet: %s", tweet_id)
        _send_json(self, 200, result)

    def _retweet(self):
        body = _json_body(self)
        tweet_id = body.get("tweet_id", "")
        if not tweet_id:
            _send_json(self, 400, {"error": "tweet_id required"})
            return
        uid = _get_my_id()
        result = _tw_post(f"/users/{uid}/retweets", {"tweet_id": tweet_id})
        log.info("Retweeted: %s", tweet_id)
        _send_json(self, 200, result)

    def _mentions(self):
        uid = _get_my_id()
        params = {
            "max_results": "20",
            "tweet.fields": "public_metrics,created_at,author_id",
        }
        data = _tw_get(f"/users/{uid}/mentions", params=params)
        _send_json(self, 200, data)

    def _followers(self):
        uid = _get_my_id()
        params = {"max_results": "100", "user.fields": "name,username,public_metrics"}
        data = _tw_get(f"/users/{uid}/followers", params=params)
        _send_json(self, 200, data)

    def _search(self, query: str):
        if not query:
            _send_json(self, 400, {"error": "q or query parameter required"})
            return
        params = {
            "query": query,
            "max_results": "20",
            "tweet.fields": "public_metrics,created_at,author_id",
        }
        data = _tw_get("/tweets/search/recent", params=params, bearer=True)
        _send_json(self, 200, data)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    server = HTTPServer(("0.0.0.0", PORT), TwitterHandler)
    log.info("fm_twitter listening on port %d", PORT)
    while _running:
        server.handle_request()
    log.info("fm_twitter stopped.")


if __name__ == "__main__":
    main()
