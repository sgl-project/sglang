#!/usr/bin/env python3
"""
fm_content_delivery.py — Content Delivery & Media Management Agent (Port 7887)
FractalMesh OMEGA Titan: upload, organise, transform, and serve media assets
(images, videos, documents, audio) backed by Cloudflare R2 object storage.
Signed URLs for private assets. Usage analytics. Folder organisation.
AWS Signature Version 4 for all R2 operations — stdlib only, no credentials
hardcoded.
Samuel James Hiotis | ABN 56 628 117 363
"""
import base64
import hashlib
import hmac
import json
import mimetypes
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
PORT                 = int(os.getenv("CONTENT_DELIVERY_PORT", "7887"))
R2_ACCOUNT_ID        = os.getenv("R2_ACCOUNT_ID", "")
R2_ACCESS_KEY_ID     = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET_NAME       = os.getenv("R2_BUCKET_NAME", "fractalmesh-media")
CLOUDFLARE_ZONE_ID   = os.getenv("CLOUDFLARE_ZONE_ID", "")
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "")
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
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [CONTENT-DELIVERY] {msg}", flush=True)

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
            CREATE TABLE IF NOT EXISTS assets (
                id                INTEGER PRIMARY KEY,
                asset_id          TEXT UNIQUE NOT NULL,
                name              TEXT,
                original_filename TEXT,
                content_type      TEXT,
                size_bytes        INTEGER,
                r2_key            TEXT UNIQUE,
                r2_bucket         TEXT,
                folder            TEXT DEFAULT '/',
                tags              TEXT DEFAULT '[]',
                alt_text          TEXT,
                description       TEXT,
                is_public         INTEGER DEFAULT 1,
                download_count    INTEGER DEFAULT 0,
                width             INTEGER,
                height            INTEGER,
                duration_seconds  REAL,
                checksum          TEXT,
                uploaded_by       TEXT,
                created_at        REAL,
                updated_at        REAL
            );
            CREATE TABLE IF NOT EXISTS folders (
                id           INTEGER PRIMARY KEY,
                path         TEXT UNIQUE NOT NULL,
                name         TEXT,
                parent_path  TEXT,
                asset_count  INTEGER DEFAULT 0,
                created_at   REAL
            );
            CREATE TABLE IF NOT EXISTS access_tokens (
                id         INTEGER PRIMARY KEY,
                token      TEXT UNIQUE NOT NULL,
                asset_id   TEXT NOT NULL,
                expires_at REAL,
                max_uses   INTEGER DEFAULT 1,
                use_count  INTEGER DEFAULT 0,
                created_at REAL
            );
            CREATE TABLE IF NOT EXISTS usage_stats (
                id              INTEGER PRIMARY KEY,
                asset_id        TEXT NOT NULL,
                date            TEXT NOT NULL,
                views           INTEGER DEFAULT 0,
                downloads       INTEGER DEFAULT 0,
                bandwidth_bytes INTEGER DEFAULT 0,
                UNIQUE(asset_id, date)
            );
            CREATE INDEX IF NOT EXISTS idx_assets_folder      ON assets(folder);
            CREATE INDEX IF NOT EXISTS idx_assets_content_type ON assets(content_type);
            CREATE INDEX IF NOT EXISTS idx_access_tokens_token ON access_tokens(token);
            CREATE INDEX IF NOT EXISTS idx_usage_stats_asset   ON usage_stats(asset_id, date);
            INSERT OR IGNORE INTO folders(path, name, parent_path, asset_count, created_at)
                VALUES('/', 'root', NULL, 0, ?);
        """, (time.time(),))
        conn.commit()
        conn.close()
    _log("Database initialised")


# ── HTTP helpers ───────────────────────────────────────────────────────────────

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


def _parse_qs(path: str) -> tuple[str, dict]:
    if "?" in path:
        base, qs = path.split("?", 1)
        return base, dict(urllib.parse.parse_qsl(qs))
    return path, {}


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    for field in ("tags",):
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
    """
    Sign and build a urllib Request for Cloudflare R2 using AWS SigV4.
    Returns a configured urllib.request.Request ready for urlopen().
    """
    account  = R2_ACCOUNT_ID
    bucket   = R2_BUCKET_NAME
    host     = f"{bucket}.{account}.r2.cloudflarestorage.com"
    region   = "auto"
    service  = "s3"

    now        = time.gmtime()
    amz_date   = time.strftime("%Y%m%dT%H%M%SZ", now)
    date_stamp = time.strftime("%Y%m%d", now)

    # URL-encode the object key per S3 spec (preserve slashes)
    safe_key = "/".join(urllib.parse.quote(seg, safe="") for seg in key.lstrip("/").split("/"))
    canonical_uri = f"/{safe_key}"

    # Query string
    qs = urllib.parse.urlencode(sorted(params.items())) if params else ""

    # Payload hash
    payload_bytes    = body or b""
    payload_hash     = hashlib.sha256(payload_bytes).hexdigest()

    # Canonical headers — must be sorted
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
    req.add_header("Authorization",         authorization)
    req.add_header("x-amz-date",            amz_date)
    req.add_header("x-amz-content-sha256",  payload_hash)
    req.add_header("Content-Type",          content_type)
    req.add_header("Host",                  host)
    if payload_bytes:
        req.add_header("Content-Length", str(len(payload_bytes)))
    return req


def _r2_put(key: str, data: bytes, content_type: str) -> bool:
    """Upload object to R2. Returns True on success."""
    if not (R2_ACCOUNT_ID and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY):
        _log("R2 credentials not configured – skipping upload")
        return False
    try:
        req = _r2_request("PUT", key, body=data, content_type=content_type)
        with urllib.request.urlopen(req, timeout=60) as resp:
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
    """Delete object from R2. Returns True on success."""
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


def _r2_get(key: str) -> tuple[bytes, str, int]:
    """
    Fetch object from R2.
    Returns (body_bytes, content_type, status_code).
    """
    if not (R2_ACCOUNT_ID and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY):
        return b"", "application/octet-stream", 503
    try:
        req = _r2_request("GET", key, body=b"", content_type="application/octet-stream")
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read()
            ct   = resp.headers.get("Content-Type", "application/octet-stream")
            return body, ct, resp.status
    except urllib.error.HTTPError as exc:
        return b"", "application/octet-stream", exc.code
    except Exception as exc:
        _log(f"R2 GET exception: {exc}")
        return b"", "application/octet-stream", 500


def _r2_copy(src_key: str, dst_key: str, content_type: str) -> bool:
    """Server-side copy within R2 using x-amz-copy-source header."""
    if not (R2_ACCOUNT_ID and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY):
        return False
    bucket   = R2_BUCKET_NAME
    account  = R2_ACCOUNT_ID
    host     = f"{bucket}.{account}.r2.cloudflarestorage.com"
    region   = "auto"
    service  = "s3"

    now        = time.gmtime()
    amz_date   = time.strftime("%Y%m%dT%H%M%SZ", now)
    date_stamp = time.strftime("%Y%m%d", now)

    safe_dst = "/".join(urllib.parse.quote(seg, safe="") for seg in dst_key.lstrip("/").split("/"))
    canonical_uri = f"/{safe_dst}"
    copy_source   = f"/{bucket}/{src_key.lstrip('/')}"
    payload_hash  = hashlib.sha256(b"").hexdigest()

    canonical_headers = (
        f"content-type:{content_type}\n"
        f"host:{host}\n"
        f"x-amz-content-sha256:{payload_hash}\n"
        f"x-amz-copy-source:{copy_source}\n"
        f"x-amz-date:{amz_date}\n"
    )
    signed_headers    = "content-type;host;x-amz-content-sha256;x-amz-copy-source;x-amz-date"
    canonical_request = "\n".join([
        "PUT", canonical_uri, "",
        canonical_headers, signed_headers, payload_hash,
    ])
    credential_scope = f"{date_stamp}/{region}/{service}/aws4_request"
    string_to_sign   = "\n".join([
        "AWS4-HMAC-SHA256", amz_date, credential_scope,
        hashlib.sha256(canonical_request.encode()).hexdigest(),
    ])
    signing_key   = _get_signing_key(R2_SECRET_ACCESS_KEY, date_stamp, region, service)
    signature     = hmac.new(signing_key, string_to_sign.encode(), hashlib.sha256).hexdigest()
    authorization = (
        f"AWS4-HMAC-SHA256 Credential={R2_ACCESS_KEY_ID}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    endpoint = f"https://{host}{canonical_uri}"
    req = urllib.request.Request(endpoint, data=b"", method="PUT")
    req.add_header("Authorization",         authorization)
    req.add_header("x-amz-date",            amz_date)
    req.add_header("x-amz-content-sha256",  payload_hash)
    req.add_header("x-amz-copy-source",     copy_source)
    req.add_header("Content-Type",          content_type)
    req.add_header("Host",                  host)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            _log(f"R2 COPY {src_key} → {dst_key}: {resp.status}")
            return resp.status in (200, 201, 204)
    except urllib.error.HTTPError as exc:
        _log(f"R2 COPY error {exc.code}: {exc.read()[:200]}")
        return False
    except Exception as exc:
        _log(f"R2 COPY exception: {exc}")
        return False


# ── asset helpers ──────────────────────────────────────────────────────────────

def _make_asset_id() -> str:
    return "ast_" + secrets.token_urlsafe(16)


def _make_r2_key(folder: str, filename: str, asset_id: str) -> str:
    folder = folder.strip("/")
    safe_name = filename.replace(" ", "_")
    if folder:
        return f"{folder}/{asset_id}_{safe_name}"
    return f"{asset_id}_{safe_name}"


def _public_r2_url(r2_key: str) -> str:
    bucket  = R2_BUCKET_NAME
    account = R2_ACCOUNT_ID
    return f"https://{bucket}.{account}.r2.cloudflarestorage.com/{r2_key}"


def _record_view(asset_id: str, size_bytes: int = 0) -> None:
    today = time.strftime("%Y-%m-%d")
    try:
        with _db_lock:
            conn = _get_db()
            conn.execute("""
                INSERT INTO usage_stats(asset_id, date, views, bandwidth_bytes)
                VALUES(?, ?, 1, ?)
                ON CONFLICT(asset_id, date) DO UPDATE SET
                    views = views + 1,
                    bandwidth_bytes = bandwidth_bytes + ?
            """, (asset_id, today, size_bytes, size_bytes))
            conn.commit()
            conn.close()
    except Exception as exc:
        _log(f"_record_view error: {exc}")


def _record_download(asset_id: str, size_bytes: int = 0) -> None:
    today = time.strftime("%Y-%m-%d")
    try:
        with _db_lock:
            conn = _get_db()
            conn.execute("""
                INSERT INTO usage_stats(asset_id, date, downloads, bandwidth_bytes)
                VALUES(?, ?, 1, ?)
                ON CONFLICT(asset_id, date) DO UPDATE SET
                    downloads = downloads + 1,
                    bandwidth_bytes = bandwidth_bytes + ?
            """, (asset_id, today, size_bytes, size_bytes))
            conn.execute(
                "UPDATE assets SET download_count = download_count + 1 WHERE asset_id = ?",
                (asset_id,),
            )
            conn.commit()
            conn.close()
    except Exception as exc:
        _log(f"_record_download error: {exc}")


def _update_folder_count(folder: str, delta: int) -> None:
    try:
        with _db_lock:
            conn = _get_db()
            conn.execute(
                "UPDATE folders SET asset_count = MAX(0, asset_count + ?) WHERE path = ?",
                (delta, folder),
            )
            conn.commit()
            conn.close()
    except Exception:
        pass


# ── Cloudflare cache purge ──────────────────────────────────────────────────────

def _cf_purge_url(url: str) -> bool:
    """Purge a single URL from Cloudflare's cache via API."""
    if not (CLOUDFLARE_ZONE_ID and CLOUDFLARE_API_TOKEN):
        _log("Cloudflare credentials not configured – skipping cache purge")
        return False
    api_url = f"https://api.cloudflare.com/client/v4/zones/{CLOUDFLARE_ZONE_ID}/purge_cache"
    payload = json.dumps({"files": [url]}).encode()
    req = urllib.request.Request(api_url, data=payload, method="POST")
    req.add_header("Authorization", f"Bearer {CLOUDFLARE_API_TOKEN}")
    req.add_header("Content-Type",  "application/json")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read())
            success = body.get("success", False)
            _log(f"CF purge {url}: {'ok' if success else 'failed'}")
            return success
    except Exception as exc:
        _log(f"CF purge error: {exc}")
        return False


# ── background maintenance thread ──────────────────────────────────────────────

def _maintenance_loop() -> None:
    """
    Daemon thread: runs every hour.
    - Aggregates usage stats (no-op since stats are written on access, but
      this could compact or roll-up fine-grained data in future).
    - Deletes expired access_tokens.
    """
    while True:
        time.sleep(3600)
        _log("Running maintenance pass…")
        try:
            now = time.time()
            with _db_lock:
                conn = _get_db()
                deleted = conn.execute(
                    "DELETE FROM access_tokens WHERE expires_at < ?", (now,)
                ).rowcount
                conn.commit()
                conn.close()
            if deleted:
                _log(f"Pruned {deleted} expired access token(s)")
        except Exception as exc:
            _log(f"Maintenance error: {exc}")


# ── request handler ────────────────────────────────────────────────────────────

class ContentDeliveryHandler(BaseHTTPRequestHandler):
    server_version = "FractalMesh-ContentDelivery/1.0"

    def log_message(self, fmt, *args):  # suppress default access log
        pass

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Admin-Secret")
        self.end_headers()

    def do_GET(self):
        path, qs = _parse_qs(self.path)

        if path == "/health":
            return self._handle_health()

        if path == "/assets":
            return self._list_assets(qs)

        if path.startswith("/assets/") and path.endswith("/url"):
            asset_id = path[len("/assets/"):-len("/url")]
            return self._get_asset_url(asset_id)

        if path.startswith("/assets/"):
            asset_id = path[len("/assets/"):]
            return self._get_asset(asset_id)

        if path.startswith("/serve/"):
            token = path[len("/serve/"):]
            return self._serve_token(token)

        if path == "/folders":
            return self._list_folders()

        if path.startswith("/analytics/"):
            asset_id = path[len("/analytics/"):]
            return self._analytics_asset(asset_id)

        if path == "/analytics":
            return self._analytics_global()

        _json_response(self, 404, {"error": "not found"})

    def do_POST(self):
        path, _ = _parse_qs(self.path)

        if path == "/assets/upload":
            return self._upload_asset()

        if path.startswith("/assets/") and path.endswith("/move"):
            asset_id = path[len("/assets/"):-len("/move")]
            return self._move_asset(asset_id)

        if path.startswith("/assets/") and path.endswith("/tag"):
            asset_id = path[len("/assets/"):-len("/tag")]
            return self._tag_asset(asset_id)

        if path == "/folders":
            return self._create_folder()

        if path.startswith("/purge/"):
            asset_id = path[len("/purge/"):]
            return self._purge_asset(asset_id)

        _json_response(self, 404, {"error": "not found"})

    def do_DELETE(self):
        path, _ = _parse_qs(self.path)
        if path.startswith("/assets/"):
            asset_id = path[len("/assets/"):]
            return self._delete_asset(asset_id)
        _json_response(self, 404, {"error": "not found"})

    # ── handlers ──────────────────────────────────────────────────────────────

    def _handle_health(self):
        uptime = time.time() - START_TIME
        _json_response(self, 200, {
            "status":  "ok",
            "service": "fm_content_delivery",
            "port":    PORT,
            "uptime_seconds": round(uptime, 1),
            "r2_configured": bool(R2_ACCOUNT_ID and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY),
        })

    # ── asset upload ──────────────────────────────────────────────────────────

    def _upload_asset(self):
        raw = _read_body(self)
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return _json_response(self, 400, {"error": "invalid JSON body"})

        filename     = payload.get("filename", "")
        content_type = payload.get("content_type", "")
        data_b64     = payload.get("data_b64", "")
        folder       = payload.get("folder", "/")
        tags         = payload.get("tags", [])
        alt_text     = payload.get("alt_text", "")
        description  = payload.get("description", "")
        is_public    = int(payload.get("is_public", 1))
        uploaded_by  = payload.get("uploaded_by", "")

        if not filename:
            return _json_response(self, 400, {"error": "filename is required"})
        if not data_b64:
            return _json_response(self, 400, {"error": "data_b64 is required"})

        # Decode base64 data
        try:
            file_bytes = base64.b64decode(data_b64)
        except Exception:
            return _json_response(self, 400, {"error": "invalid base64 in data_b64"})

        # Guess content_type if not provided
        if not content_type:
            ct_guess, _ = mimetypes.guess_type(filename)
            content_type = ct_guess or "application/octet-stream"

        # Normalise folder
        folder = "/" + folder.strip("/") if folder.strip("/") else "/"

        asset_id  = _make_asset_id()
        r2_key    = _make_r2_key(folder, filename, asset_id)
        checksum  = hashlib.sha256(file_bytes).hexdigest()
        size_bytes = len(file_bytes)
        now       = time.time()

        # Upload to R2
        r2_ok = _r2_put(r2_key, file_bytes, content_type)

        # Persist asset record regardless (allow offline use)
        try:
            with _db_lock:
                conn = _get_db()
                conn.execute("""
                    INSERT INTO assets(
                        asset_id, name, original_filename, content_type,
                        size_bytes, r2_key, r2_bucket, folder, tags,
                        alt_text, description, is_public, download_count,
                        checksum, uploaded_by, created_at, updated_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0,?,?,?,?)
                """, (
                    asset_id, filename, filename, content_type,
                    size_bytes, r2_key, R2_BUCKET_NAME, folder,
                    json.dumps(tags if isinstance(tags, list) else []),
                    alt_text, description, is_public,
                    checksum, uploaded_by, now, now,
                ))
                # Ensure folder record exists
                conn.execute("""
                    INSERT OR IGNORE INTO folders(path, name, parent_path, asset_count, created_at)
                    VALUES(?, ?, ?, 0, ?)
                """, (
                    folder,
                    folder.split("/")[-1] or "root",
                    "/".join(folder.rstrip("/").split("/")[:-1]) or "/",
                    now,
                ))
                conn.execute(
                    "UPDATE folders SET asset_count = asset_count + 1 WHERE path = ?",
                    (folder,),
                )
                conn.commit()
                conn.close()
        except sqlite3.IntegrityError as exc:
            return _json_response(self, 409, {"error": f"duplicate r2_key or asset_id: {exc}"})
        except Exception as exc:
            return _json_response(self, 500, {"error": str(exc)})

        _log(f"Uploaded asset {asset_id} ({size_bytes} bytes, r2={'ok' if r2_ok else 'failed'})")

        response = {
            "asset_id":     asset_id,
            "r2_key":       r2_key,
            "r2_uploaded":  r2_ok,
            "size_bytes":   size_bytes,
            "content_type": content_type,
            "folder":       folder,
            "checksum":     checksum,
        }
        if is_public and r2_ok:
            response["url"] = _public_r2_url(r2_key)
        return _json_response(self, 201, response)

    # ── list assets ───────────────────────────────────────────────────────────

    def _list_assets(self, qs: dict):
        folder_filter  = qs.get("folder", "")
        tag_filter     = qs.get("tag", "")
        ct_filter      = qs.get("content_type", "")
        limit          = min(int(qs.get("limit", "100")), 500)
        offset         = int(qs.get("offset", "0"))

        conditions = []
        params     = []

        if folder_filter:
            conditions.append("folder = ?")
            params.append("/" + folder_filter.strip("/") if folder_filter.strip("/") else "/")
        if ct_filter:
            conditions.append("content_type LIKE ?")
            params.append(f"{ct_filter}%")

        where_sql = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        try:
            conn   = _get_db()
            rows   = conn.execute(
                f"SELECT * FROM assets {where_sql} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                params + [limit, offset],
            ).fetchall()
            total  = conn.execute(
                f"SELECT COUNT(*) FROM assets {where_sql}", params
            ).fetchone()[0]
            conn.close()
        except Exception as exc:
            return _json_response(self, 500, {"error": str(exc)})

        assets = [_row_to_dict(r) for r in rows]

        # Apply tag filter in Python (tags stored as JSON array)
        if tag_filter:
            assets = [a for a in assets if tag_filter in a.get("tags", [])]

        _json_response(self, 200, {
            "assets": assets,
            "total":  total,
            "limit":  limit,
            "offset": offset,
        })

    # ── get single asset ──────────────────────────────────────────────────────

    def _get_asset(self, asset_id: str):
        try:
            conn = _get_db()
            row  = conn.execute(
                "SELECT * FROM assets WHERE asset_id = ?", (asset_id,)
            ).fetchone()
            conn.close()
        except Exception as exc:
            return _json_response(self, 500, {"error": str(exc)})

        if not row:
            return _json_response(self, 404, {"error": "asset not found"})

        data = _row_to_dict(row)
        _json_response(self, 200, data)

    # ── delete asset ─────────────────────────────────────────────────────────

    def _delete_asset(self, asset_id: str):
        if not _require_admin(self):
            return

        try:
            conn = _get_db()
            row  = conn.execute(
                "SELECT * FROM assets WHERE asset_id = ?", (asset_id,)
            ).fetchone()
            conn.close()
        except Exception as exc:
            return _json_response(self, 500, {"error": str(exc)})

        if not row:
            return _json_response(self, 404, {"error": "asset not found"})

        asset = _row_to_dict(row)
        r2_key = asset["r2_key"]
        folder = asset["folder"]

        # Delete from R2
        r2_ok = _r2_delete(r2_key) if r2_key else True

        # Delete from DB
        try:
            with _db_lock:
                conn = _get_db()
                conn.execute("DELETE FROM assets WHERE asset_id = ?", (asset_id,))
                conn.execute("DELETE FROM access_tokens WHERE asset_id = ?", (asset_id,))
                conn.execute(
                    "UPDATE folders SET asset_count = MAX(0, asset_count - 1) WHERE path = ?",
                    (folder,),
                )
                conn.commit()
                conn.close()
        except Exception as exc:
            return _json_response(self, 500, {"error": str(exc)})

        _log(f"Deleted asset {asset_id} (r2_delete={r2_ok})")
        _json_response(self, 200, {"deleted": True, "asset_id": asset_id, "r2_deleted": r2_ok})

    # ── move asset ────────────────────────────────────────────────────────────

    def _move_asset(self, asset_id: str):
        raw = _read_body(self)
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return _json_response(self, 400, {"error": "invalid JSON"})

        new_folder = payload.get("folder", "/")
        new_folder = "/" + new_folder.strip("/") if new_folder.strip("/") else "/"

        try:
            conn = _get_db()
            row  = conn.execute(
                "SELECT * FROM assets WHERE asset_id = ?", (asset_id,)
            ).fetchone()
            conn.close()
        except Exception as exc:
            return _json_response(self, 500, {"error": str(exc)})

        if not row:
            return _json_response(self, 404, {"error": "asset not found"})

        asset       = _row_to_dict(row)
        old_r2_key  = asset["r2_key"]
        old_folder  = asset["folder"]
        content_type = asset["content_type"] or "application/octet-stream"
        filename    = asset["original_filename"] or "file"

        if old_folder == new_folder:
            return _json_response(self, 200, {"asset_id": asset_id, "folder": new_folder, "changed": False})

        new_r2_key = _make_r2_key(new_folder, filename, asset_id)

        # Server-side copy then delete old key
        copy_ok = _r2_copy(old_r2_key, new_r2_key, content_type)
        if copy_ok:
            _r2_delete(old_r2_key)
        else:
            # Fallback: download then re-upload
            body, _, status = _r2_get(old_r2_key)
            if status == 200 and body:
                copy_ok = _r2_put(new_r2_key, body, content_type)
                if copy_ok:
                    _r2_delete(old_r2_key)

        now = time.time()
        try:
            with _db_lock:
                conn = _get_db()
                conn.execute("""
                    UPDATE assets SET folder = ?, r2_key = ?, updated_at = ?
                    WHERE asset_id = ?
                """, (new_folder, new_r2_key if copy_ok else old_r2_key, now, asset_id))
                # Update folder counts
                conn.execute(
                    "UPDATE folders SET asset_count = MAX(0, asset_count - 1) WHERE path = ?",
                    (old_folder,),
                )
                conn.execute("""
                    INSERT OR IGNORE INTO folders(path, name, parent_path, asset_count, created_at)
                    VALUES(?, ?, ?, 0, ?)
                """, (
                    new_folder,
                    new_folder.split("/")[-1] or "root",
                    "/".join(new_folder.rstrip("/").split("/")[:-1]) or "/",
                    now,
                ))
                conn.execute(
                    "UPDATE folders SET asset_count = asset_count + 1 WHERE path = ?",
                    (new_folder,),
                )
                conn.commit()
                conn.close()
        except Exception as exc:
            return _json_response(self, 500, {"error": str(exc)})

        _log(f"Moved asset {asset_id} from {old_folder} to {new_folder}")
        _json_response(self, 200, {
            "asset_id":  asset_id,
            "folder":    new_folder,
            "r2_key":    new_r2_key if copy_ok else old_r2_key,
            "r2_moved":  copy_ok,
            "changed":   True,
        })

    # ── tag asset ─────────────────────────────────────────────────────────────

    def _tag_asset(self, asset_id: str):
        raw = _read_body(self)
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return _json_response(self, 400, {"error": "invalid JSON"})

        tags = payload.get("tags", [])
        if not isinstance(tags, list):
            return _json_response(self, 400, {"error": "tags must be a JSON array"})

        try:
            conn = _get_db()
            row  = conn.execute(
                "SELECT id FROM assets WHERE asset_id = ?", (asset_id,)
            ).fetchone()
            conn.close()
        except Exception as exc:
            return _json_response(self, 500, {"error": str(exc)})

        if not row:
            return _json_response(self, 404, {"error": "asset not found"})

        try:
            with _db_lock:
                conn = _get_db()
                conn.execute(
                    "UPDATE assets SET tags = ?, updated_at = ? WHERE asset_id = ?",
                    (json.dumps(tags), time.time(), asset_id),
                )
                conn.commit()
                conn.close()
        except Exception as exc:
            return _json_response(self, 500, {"error": str(exc)})

        _json_response(self, 200, {"asset_id": asset_id, "tags": tags})

    # ── get asset URL ─────────────────────────────────────────────────────────

    def _get_asset_url(self, asset_id: str):
        try:
            conn = _get_db()
            row  = conn.execute(
                "SELECT * FROM assets WHERE asset_id = ?", (asset_id,)
            ).fetchone()
            conn.close()
        except Exception as exc:
            return _json_response(self, 500, {"error": str(exc)})

        if not row:
            return _json_response(self, 404, {"error": "asset not found"})

        asset = _row_to_dict(row)

        if asset.get("is_public"):
            url = _public_r2_url(asset["r2_key"])
            return _json_response(self, 200, {
                "asset_id": asset_id,
                "url":      url,
                "type":     "public",
            })

        # Private: create signed access token, return /serve/{token}
        token      = secrets.token_urlsafe(32)
        expires_at = time.time() + 3600
        now        = time.time()

        try:
            with _db_lock:
                conn = _get_db()
                conn.execute("""
                    INSERT INTO access_tokens(token, asset_id, expires_at, max_uses, use_count, created_at)
                    VALUES(?, ?, ?, 1, 0, ?)
                """, (token, asset_id, expires_at, now))
                conn.commit()
                conn.close()
        except Exception as exc:
            return _json_response(self, 500, {"error": str(exc)})

        host    = self.headers.get("Host", f"localhost:{PORT}")
        scheme  = "http"
        serve_url = f"{scheme}://{host}/serve/{token}"

        _json_response(self, 200, {
            "asset_id":   asset_id,
            "url":        serve_url,
            "type":       "signed",
            "expires_at": expires_at,
            "expires_in": 3600,
        })

    # ── serve token ───────────────────────────────────────────────────────────

    def _serve_token(self, token: str):
        now = time.time()

        try:
            conn   = _get_db()
            trow   = conn.execute(
                "SELECT * FROM access_tokens WHERE token = ?", (token,)
            ).fetchone()
            conn.close()
        except Exception as exc:
            return _json_response(self, 500, {"error": str(exc)})

        if not trow:
            return _json_response(self, 404, {"error": "token not found"})

        tdata = dict(trow)

        if tdata["expires_at"] < now:
            return _json_response(self, 410, {"error": "token expired"})

        if tdata["use_count"] >= tdata["max_uses"]:
            return _json_response(self, 410, {"error": "token max uses reached"})

        asset_id = tdata["asset_id"]

        # Get asset record
        try:
            conn  = _get_db()
            arow  = conn.execute(
                "SELECT * FROM assets WHERE asset_id = ?", (asset_id,)
            ).fetchone()
            conn.close()
        except Exception as exc:
            return _json_response(self, 500, {"error": str(exc)})

        if not arow:
            return _json_response(self, 404, {"error": "asset not found"})

        asset = _row_to_dict(arow)
        r2_key = asset["r2_key"]

        # Increment token use_count
        try:
            with _db_lock:
                conn = _get_db()
                conn.execute(
                    "UPDATE access_tokens SET use_count = use_count + 1 WHERE token = ?",
                    (token,),
                )
                conn.commit()
                conn.close()
        except Exception:
            pass

        # Fetch from R2 and proxy to client
        body, content_type, status = _r2_get(r2_key)

        if status != 200:
            return _json_response(self, status, {"error": f"R2 returned {status}"})

        # Record view
        _record_view(asset_id, len(body))

        self.send_response(200)
        self.send_header("Content-Type",   content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control",  "private, no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        if asset.get("original_filename"):
            self.send_header(
                "Content-Disposition",
                f'inline; filename="{asset["original_filename"]}"',
            )
        self.end_headers()
        self.wfile.write(body)

    # ── folders ───────────────────────────────────────────────────────────────

    def _list_folders(self):
        try:
            conn  = _get_db()
            rows  = conn.execute(
                "SELECT * FROM folders ORDER BY path ASC"
            ).fetchall()
            conn.close()
        except Exception as exc:
            return _json_response(self, 500, {"error": str(exc)})

        _json_response(self, 200, {"folders": [dict(r) for r in rows]})

    def _create_folder(self):
        raw = _read_body(self)
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return _json_response(self, 400, {"error": "invalid JSON"})

        path = payload.get("path", "")
        name = payload.get("name", "")

        if not path:
            return _json_response(self, 400, {"error": "path is required"})

        path = "/" + path.strip("/") if path.strip("/") else "/"
        if not name:
            name = path.split("/")[-1] or "root"

        parent = "/".join(path.rstrip("/").split("/")[:-1]) or "/"
        now    = time.time()

        try:
            with _db_lock:
                conn = _get_db()
                conn.execute("""
                    INSERT OR IGNORE INTO folders(path, name, parent_path, asset_count, created_at)
                    VALUES(?, ?, ?, 0, ?)
                """, (path, name, parent, now))
                conn.commit()
                conn.close()
        except Exception as exc:
            return _json_response(self, 500, {"error": str(exc)})

        _json_response(self, 201, {"path": path, "name": name, "parent_path": parent})

    # ── analytics ─────────────────────────────────────────────────────────────

    def _analytics_asset(self, asset_id: str):
        # Verify asset exists
        try:
            conn = _get_db()
            arow = conn.execute(
                "SELECT asset_id, name FROM assets WHERE asset_id = ?", (asset_id,)
            ).fetchone()
            conn.close()
        except Exception as exc:
            return _json_response(self, 500, {"error": str(exc)})

        if not arow:
            return _json_response(self, 404, {"error": "asset not found"})

        cutoff = time.strftime("%Y-%m-%d", time.gmtime(time.time() - 30 * 86400))

        try:
            conn  = _get_db()
            rows  = conn.execute("""
                SELECT date, views, downloads, bandwidth_bytes
                FROM usage_stats
                WHERE asset_id = ? AND date >= ?
                ORDER BY date DESC
            """, (asset_id, cutoff)).fetchall()
            totals = conn.execute("""
                SELECT
                    SUM(views) as total_views,
                    SUM(downloads) as total_downloads,
                    SUM(bandwidth_bytes) as total_bandwidth
                FROM usage_stats
                WHERE asset_id = ? AND date >= ?
            """, (asset_id, cutoff)).fetchone()
            conn.close()
        except Exception as exc:
            return _json_response(self, 500, {"error": str(exc)})

        _json_response(self, 200, {
            "asset_id":       asset_id,
            "asset_name":     arow["name"],
            "period_days":    30,
            "totals": {
                "views":          totals["total_views"]     or 0,
                "downloads":      totals["total_downloads"] or 0,
                "bandwidth_bytes": totals["total_bandwidth"] or 0,
            },
            "daily": [dict(r) for r in rows],
        })

    def _analytics_global(self):
        if not _require_admin(self):
            return

        now_ts = time.time()
        month_start = time.strftime("%Y-%m-01", time.gmtime(now_ts))

        try:
            conn = _get_db()

            total_assets = conn.execute(
                "SELECT COUNT(*) FROM assets"
            ).fetchone()[0]

            total_size = conn.execute(
                "SELECT COALESCE(SUM(size_bytes), 0) FROM assets"
            ).fetchone()[0]

            top_downloaded = conn.execute("""
                SELECT asset_id, name, download_count
                FROM assets
                ORDER BY download_count DESC
                LIMIT 10
            """).fetchall()

            bandwidth_month = conn.execute("""
                SELECT COALESCE(SUM(bandwidth_bytes), 0)
                FROM usage_stats
                WHERE date >= ?
            """, (month_start,)).fetchone()[0]

            type_breakdown = conn.execute("""
                SELECT content_type, COUNT(*) as cnt, COALESCE(SUM(size_bytes), 0) as sz
                FROM assets
                GROUP BY content_type
                ORDER BY cnt DESC
                LIMIT 20
            """).fetchall()

            conn.close()
        except Exception as exc:
            return _json_response(self, 500, {"error": str(exc)})

        _json_response(self, 200, {
            "total_assets":       total_assets,
            "total_size_bytes":   total_size,
            "bandwidth_this_month_bytes": bandwidth_month,
            "top_downloaded":     [dict(r) for r in top_downloaded],
            "content_type_breakdown": [dict(r) for r in type_breakdown],
        })

    # ── purge Cloudflare cache ────────────────────────────────────────────────

    def _purge_asset(self, asset_id: str):
        if not _require_admin(self):
            return

        try:
            conn = _get_db()
            row  = conn.execute(
                "SELECT r2_key, is_public FROM assets WHERE asset_id = ?", (asset_id,)
            ).fetchone()
            conn.close()
        except Exception as exc:
            return _json_response(self, 500, {"error": str(exc)})

        if not row:
            return _json_response(self, 404, {"error": "asset not found"})

        r2_key    = row["r2_key"]
        is_public = row["is_public"]

        if not is_public:
            return _json_response(self, 400, {"error": "asset is private – no public URL to purge"})

        url    = _public_r2_url(r2_key)
        purged = _cf_purge_url(url)

        _json_response(self, 200, {
            "asset_id": asset_id,
            "url":      url,
            "purged":   purged,
        })


# ── server bootstrap ───────────────────────────────────────────────────────────

def _start_maintenance_thread() -> None:
    t = threading.Thread(target=_maintenance_loop, daemon=True, name="maintenance")
    t.start()
    _log("Maintenance daemon started")


def run_server() -> None:
    _init_db()
    _start_maintenance_thread()

    server = HTTPServer(("0.0.0.0", PORT), ContentDeliveryHandler)
    _log(f"Content Delivery & Media Management running on port {PORT}")
    _log(f"R2 bucket: {R2_BUCKET_NAME or '(not configured)'}")
    _log(f"DB: {DB_PATH}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log("Shutting down…")
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server()
