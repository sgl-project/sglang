#!/usr/bin/env python3
"""
fm_cloud_storage.py — Cloud Storage Abstraction Layer (Port 7861)
Unified upload/download across Cloudflare R2, Supabase Storage, and GitHub
Releases.  Files tracked in SQLite with full metadata.  CDN URL generation.
AWS Signature Version 4 used for R2.  No hardcoded credentials.
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import re
import io
import json
import time
import gzip
import hmac
import base64
import hashlib
import sqlite3
import mimetypes
import threading
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

# ── vault ──────────────────────────────────────────────────────────────────────
_ENV_FILE = Path.home() / ".secrets" / "fractal.env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

# ── config ─────────────────────────────────────────────────────────────────────
PORT                    = int(os.getenv("CLOUD_STORAGE_PORT", "7861"))
R2_ACCOUNT_ID           = os.getenv("R2_ACCOUNT_ID", "")
R2_ACCESS_KEY_ID        = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY    = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET_NAME          = os.getenv("R2_BUCKET_NAME", "")
R2_PUBLIC_URL           = os.getenv("R2_PUBLIC_URL", "")
SUPABASE_URL            = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY    = os.getenv("SUPABASE_SERVICE_KEY", "")
SUPABASE_STORAGE_BUCKET = os.getenv("SUPABASE_STORAGE_BUCKET", "")
GITHUB_TOKEN            = os.getenv("GITHUB_TOKEN", "")
GITHUB_ORG              = os.getenv("GITHUB_ORG", "")
GITHUB_REPO             = os.getenv("GITHUB_REPO", "")
ADMIN_SECRET            = os.getenv("ADMIN_SECRET", "")

ROOT    = Path(os.getenv("FRACTALMESH_HOME", str(Path.home() / "fmsaas")))
DB_PATH = ROOT / "database" / "sovereign.db"
LOG_DIR = ROOT / "logs"

ROOT.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

START_TIME = time.time()
_db_lock   = threading.Lock()

# ── helpers ────────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [CLOUD-STORAGE] {msg}", flush=True)


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
            CREATE TABLE IF NOT EXISTS files (
                id            INTEGER PRIMARY KEY,
                key           TEXT UNIQUE,
                filename      TEXT,
                content_type  TEXT,
                size_bytes    INTEGER,
                backend       TEXT,
                backend_url   TEXT,
                cdn_url       TEXT,
                checksum      TEXT,
                public        INTEGER DEFAULT 1,
                tags          TEXT,
                uploaded_at   REAL,
                last_accessed REAL,
                access_count  INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS upload_sessions (
                id               INTEGER PRIMARY KEY,
                session_id       TEXT UNIQUE,
                key              TEXT,
                backend          TEXT,
                status           TEXT DEFAULT 'pending',
                chunks_total     INTEGER DEFAULT 1,
                chunks_received  INTEGER DEFAULT 0,
                created_at       REAL,
                completed_at     REAL
            );
        """)
        conn.commit()
        conn.close()


def _json_response(handler: "BaseHTTPRequestHandler", code: int, data: dict) -> None:
    body = json.dumps(data, indent=2).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_body(handler: "BaseHTTPRequestHandler") -> bytes:
    length = int(handler.headers.get("Content-Length", 0))
    return handler.rfile.read(length) if length else b""


def _require_admin(handler: "BaseHTTPRequestHandler") -> bool:
    secret = handler.headers.get("X-Admin-Secret", "")
    if not ADMIN_SECRET or secret != ADMIN_SECRET:
        _json_response(handler, 403, {"error": "forbidden"})
        return False
    return True


def _guess_content_type(filename: str) -> str:
    ct, _ = mimetypes.guess_type(filename)
    return ct or "application/octet-stream"


def _checksum(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _choose_backend(key: str, content_type: str, size_bytes: int) -> str:
    """Select best backend based on file characteristics."""
    r2_ready = bool(R2_ACCOUNT_ID and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY and R2_BUCKET_NAME)
    sb_ready = bool(SUPABASE_URL and SUPABASE_SERVICE_KEY and SUPABASE_STORAGE_BUCKET)
    gh_ready = bool(GITHUB_TOKEN and GITHUB_ORG and GITHUB_REPO)

    is_media = any(content_type.startswith(p) for p in ("image/", "video/", "audio/"))
    is_text_json = content_type in (
        "text/plain", "application/json", "text/csv", "text/html",
        "text/markdown", "application/xml", "text/xml",
    )

    if gh_ready and size_bytes < 100_000 and is_text_json:
        return "github"
    if r2_ready and is_media:
        return "r2"
    if sb_ready and not r2_ready:
        return "supabase"
    if r2_ready:
        return "r2"
    if sb_ready:
        return "supabase"
    if gh_ready:
        return "github"
    return "r2"


# ── AWS Signature Version 4 (for R2) ──────────────────────────────────────────

def _hmac_sha256(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode(), hashlib.sha256).digest()


def _get_signing_key(secret: str, date_stamp: str, region: str, service: str) -> bytes:
    k_date    = _hmac_sha256(("AWS4" + secret).encode(), date_stamp)
    k_region  = _hmac_sha256(k_date, region)
    k_service = _hmac_sha256(k_region, service)
    k_signing = _hmac_sha256(k_service, "aws4_request")
    return k_signing


def _sign_r2_request(
    method: str,
    key: str,
    content_type: str,
    payload_hash: str,
    extra_headers: dict | None = None,
) -> tuple[dict, str]:
    """
    Returns (headers_dict, endpoint_url) with a valid AWS4-HMAC-SHA256 signature
    for Cloudflare R2.
    """
    region   = "auto"
    service  = "s3"
    host     = f"{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    endpoint = f"https://{host}/{R2_BUCKET_NAME}/{key.lstrip('/')}"

    now          = time.gmtime()
    amz_date     = time.strftime("%Y%m%dT%H%M%SZ", now)
    date_stamp   = time.strftime("%Y%m%d", now)

    # Canonical request
    canonical_uri   = f"/{R2_BUCKET_NAME}/{key.lstrip('/')}"
    query_string    = extra_headers.pop("_qs", "") if extra_headers else ""
    signed_headers  = "content-type;host;x-amz-content-sha256;x-amz-date"
    canonical_headers = (
        f"content-type:{content_type}\n"
        f"host:{host}\n"
        f"x-amz-content-sha256:{payload_hash}\n"
        f"x-amz-date:{amz_date}\n"
    )
    canonical_request = "\n".join([
        method.upper(),
        canonical_uri,
        query_string,
        canonical_headers,
        signed_headers,
        payload_hash,
    ])

    # String to sign
    credential_scope = f"{date_stamp}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join([
        "AWS4-HMAC-SHA256",
        amz_date,
        credential_scope,
        hashlib.sha256(canonical_request.encode()).hexdigest(),
    ])

    signing_key = _get_signing_key(R2_SECRET_ACCESS_KEY, date_stamp, region, service)
    signature   = hmac.new(signing_key, string_to_sign.encode(), hashlib.sha256).hexdigest()

    authorization = (
        f"AWS4-HMAC-SHA256 Credential={R2_ACCESS_KEY_ID}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    headers = {
        "Authorization":          authorization,
        "x-amz-date":             amz_date,
        "x-amz-content-sha256":   payload_hash,
        "Content-Type":           content_type,
        "Host":                   host,
    }
    return headers, endpoint


def _sign_r2_list(prefix: str = "") -> tuple[dict, str]:
    """Return (headers, url) for listing R2 bucket objects."""
    region   = "auto"
    service  = "s3"
    host     = f"{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    qs       = f"list-type=2&prefix={urllib.parse.quote(prefix)}"
    endpoint = f"https://{host}/{R2_BUCKET_NAME}?{qs}"

    now        = time.gmtime()
    amz_date   = time.strftime("%Y%m%dT%H%M%SZ", now)
    date_stamp = time.strftime("%Y%m%d", now)

    payload_hash    = hashlib.sha256(b"").hexdigest()
    signed_headers  = "host;x-amz-content-sha256;x-amz-date"
    canonical_headers = (
        f"host:{host}\n"
        f"x-amz-content-sha256:{payload_hash}\n"
        f"x-amz-date:{amz_date}\n"
    )
    canonical_uri = f"/{R2_BUCKET_NAME}"
    canonical_request = "\n".join([
        "GET", canonical_uri, qs,
        canonical_headers, signed_headers, payload_hash,
    ])
    credential_scope = f"{date_stamp}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join([
        "AWS4-HMAC-SHA256", amz_date, credential_scope,
        hashlib.sha256(canonical_request.encode()).hexdigest(),
    ])
    signing_key = _get_signing_key(R2_SECRET_ACCESS_KEY, date_stamp, region, service)
    signature   = hmac.new(signing_key, string_to_sign.encode(), hashlib.sha256).hexdigest()
    authorization = (
        f"AWS4-HMAC-SHA256 Credential={R2_ACCESS_KEY_ID}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    headers = {
        "Authorization":        authorization,
        "x-amz-date":           amz_date,
        "x-amz-content-sha256": payload_hash,
        "Host":                 host,
    }
    return headers, endpoint


# ── R2 backend ─────────────────────────────────────────────────────────────────

def _r2_upload(key: str, data: bytes, content_type: str) -> str:
    """Upload bytes to R2 and return backend URL."""
    payload_hash = hashlib.sha256(data).hexdigest()
    headers, endpoint = _sign_r2_request("PUT", key, content_type, payload_hash)
    req = urllib.request.Request(endpoint, data=data, headers=headers, method="PUT")
    with urllib.request.urlopen(req) as resp:
        if resp.status not in (200, 201, 204):
            raise RuntimeError(f"R2 upload failed: {resp.status}")
    return endpoint


def _r2_download(key: str) -> bytes:
    """Download object bytes from R2."""
    payload_hash = hashlib.sha256(b"").hexdigest()
    headers, endpoint = _sign_r2_request("GET", key, "application/octet-stream", payload_hash)
    req = urllib.request.Request(endpoint, headers=headers, method="GET")
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def _r2_delete(key: str) -> None:
    """Delete object from R2."""
    payload_hash = hashlib.sha256(b"").hexdigest()
    headers, endpoint = _sign_r2_request("DELETE", key, "application/octet-stream", payload_hash)
    req = urllib.request.Request(endpoint, headers=headers, method="DELETE")
    with urllib.request.urlopen(req) as resp:
        if resp.status not in (200, 204):
            raise RuntimeError(f"R2 delete failed: {resp.status}")


def _r2_list(prefix: str = "") -> list[str]:
    """List keys in R2 bucket matching prefix.  Returns list of key strings."""
    headers, endpoint = _sign_r2_list(prefix)
    req = urllib.request.Request(endpoint, headers=headers, method="GET")
    with urllib.request.urlopen(req) as resp:
        xml = resp.read().decode()
    return re.findall(r"<Key>(.*?)</Key>", xml)


def _r2_cdn_url(key: str) -> str:
    if R2_PUBLIC_URL:
        return f"{R2_PUBLIC_URL.rstrip('/')}/{key.lstrip('/')}"
    return f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com/{R2_BUCKET_NAME}/{key.lstrip('/')}"


# ── Supabase Storage backend ───────────────────────────────────────────────────

def _sb_upload(key: str, data: bytes, content_type: str) -> str:
    """Upload bytes to Supabase Storage and return backend URL."""
    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_STORAGE_BUCKET}/{key.lstrip('/')}"
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "Content-Type":  content_type,
            "x-upsert":      "true",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        if resp.status not in (200, 201):
            raise RuntimeError(f"Supabase upload failed: {resp.status}")
    return url


def _sb_download(key: str) -> bytes:
    """Download object from Supabase Storage (public endpoint)."""
    url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_STORAGE_BUCKET}/{key.lstrip('/')}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
    })
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def _sb_delete(key: str) -> None:
    """Delete object from Supabase Storage."""
    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_STORAGE_BUCKET}/{key.lstrip('/')}"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"},
        method="DELETE",
    )
    with urllib.request.urlopen(req) as resp:
        if resp.status not in (200, 204):
            raise RuntimeError(f"Supabase delete failed: {resp.status}")


def _sb_list(prefix: str = "") -> list[str]:
    """List objects in Supabase Storage bucket matching prefix."""
    url = f"{SUPABASE_URL}/storage/v1/object/list/{SUPABASE_STORAGE_BUCKET}"
    payload = json.dumps({"prefix": prefix, "limit": 1000}).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "Content-Type":  "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        items = json.loads(resp.read())
    return [item["name"] for item in items if "name" in item]


def _sb_cdn_url(key: str) -> str:
    return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_STORAGE_BUCKET}/{key.lstrip('/')}"


# ── GitHub Releases backend ────────────────────────────────────────────────────

def _gh_api(method: str, path: str, payload: dict | None = None,
            upload_data: bytes | None = None, upload_content_type: str = "") -> dict | list:
    """Generic GitHub API helper."""
    base = "https://api.github.com"
    url  = path if path.startswith("https://") else f"{base}{path}"
    data = json.dumps(payload).encode() if payload is not None else upload_data
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept":        "application/vnd.github+json",
        "User-Agent":    "FractalMesh-CloudStorage/1.0",
    }
    if payload is not None:
        headers["Content-Type"] = "application/json"
    elif upload_data is not None:
        headers["Content-Type"] = upload_content_type or "application/octet-stream"

    req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    with urllib.request.urlopen(req) as resp:
        body = resp.read()
        if body:
            return json.loads(body)
        return {}


def _gh_get_or_create_release() -> dict:
    """Return the latest release or create one named storage-YYYY-MM."""
    releases = _gh_api("GET", f"/repos/{GITHUB_ORG}/{GITHUB_REPO}/releases")
    if isinstance(releases, list) and releases:
        return releases[0]
    tag = time.strftime("storage-%Y-%m")
    return _gh_api("POST", f"/repos/{GITHUB_ORG}/{GITHUB_REPO}/releases", {
        "tag_name":   tag,
        "name":       tag,
        "body":       "FractalMesh cloud storage release",
        "draft":      False,
        "prerelease": False,
    })


def _gh_upload(key: str, data: bytes, content_type: str) -> tuple[str, str]:
    """Upload asset to GitHub release.  Returns (backend_url, cdn_url)."""
    release   = _gh_get_or_create_release()
    rid       = release["id"]
    filename  = key.replace("/", "_")
    upload_url = (
        f"https://uploads.github.com/repos/{GITHUB_ORG}/{GITHUB_REPO}"
        f"/releases/{rid}/assets?name={urllib.parse.quote(filename)}"
    )
    result = _gh_api("POST", upload_url, upload_data=data, upload_content_type=content_type)
    backend_url = result.get("url", "")
    cdn_url     = result.get("browser_download_url", backend_url)
    return backend_url, cdn_url


def _gh_download(key: str) -> bytes:
    """Download asset from GitHub Releases by matching filename."""
    releases = _gh_api("GET", f"/repos/{GITHUB_ORG}/{GITHUB_REPO}/releases")
    filename  = key.replace("/", "_")
    if not isinstance(releases, list):
        raise RuntimeError("GitHub releases not a list")
    for release in releases:
        for asset in release.get("assets", []):
            if asset["name"] == filename:
                durl = asset["browser_download_url"]
                req  = urllib.request.Request(durl, headers={
                    "Authorization": f"Bearer {GITHUB_TOKEN}",
                    "Accept":        "application/octet-stream",
                })
                with urllib.request.urlopen(req) as resp:
                    return resp.read()
    raise RuntimeError(f"GitHub asset not found for key: {key}")


def _gh_delete(key: str) -> None:
    """Delete asset from GitHub Releases by matching filename."""
    releases = _gh_api("GET", f"/repos/{GITHUB_ORG}/{GITHUB_REPO}/releases")
    filename  = key.replace("/", "_")
    if not isinstance(releases, list):
        return
    for release in releases:
        for asset in release.get("assets", []):
            if asset["name"] == filename:
                _gh_api("DELETE", f"/repos/{GITHUB_ORG}/{GITHUB_REPO}/releases/assets/{asset['id']}")
                return


def _gh_list() -> list[str]:
    """List all asset names across all releases, returned as keys."""
    releases = _gh_api("GET", f"/repos/{GITHUB_ORG}/{GITHUB_REPO}/releases")
    keys: list[str] = []
    if not isinstance(releases, list):
        return keys
    for release in releases:
        for asset in release.get("assets", []):
            keys.append(asset["name"].replace("_", "/", 1))
    return keys


# ── unified upload / download ──────────────────────────────────────────────────

def _upload_to_backend(
    backend: str, key: str, data: bytes, content_type: str
) -> tuple[str, str]:
    """Upload data to backend.  Returns (backend_url, cdn_url)."""
    if backend == "r2":
        burl = _r2_upload(key, data, content_type)
        curl = _r2_cdn_url(key)
        return burl, curl
    if backend == "supabase":
        burl = _sb_upload(key, data, content_type)
        curl = _sb_cdn_url(key)
        return burl, curl
    if backend == "github":
        return _gh_upload(key, data, content_type)
    raise ValueError(f"Unknown backend: {backend}")


def _download_from_backend(backend: str, key: str) -> bytes:
    if backend == "r2":
        return _r2_download(key)
    if backend == "supabase":
        return _sb_download(key)
    if backend == "github":
        return _gh_download(key)
    raise ValueError(f"Unknown backend: {backend}")


def _delete_from_backend(backend: str, key: str) -> None:
    if backend == "r2":
        _r2_delete(key)
    elif backend == "supabase":
        _sb_delete(key)
    elif backend == "github":
        _gh_delete(key)
    else:
        raise ValueError(f"Unknown backend: {backend}")


# ── SQLite helpers ─────────────────────────────────────────────────────────────

def _db_insert_file(conn: sqlite3.Connection, key: str, filename: str,
                    content_type: str, size_bytes: int, backend: str,
                    backend_url: str, cdn_url: str, checksum: str,
                    public: int, tags: list[str]) -> None:
    now = time.time()
    conn.execute("""
        INSERT INTO files
            (key, filename, content_type, size_bytes, backend,
             backend_url, cdn_url, checksum, public, tags,
             uploaded_at, last_accessed, access_count)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0)
        ON CONFLICT(key) DO UPDATE SET
            filename=excluded.filename,
            content_type=excluded.content_type,
            size_bytes=excluded.size_bytes,
            backend=excluded.backend,
            backend_url=excluded.backend_url,
            cdn_url=excluded.cdn_url,
            checksum=excluded.checksum,
            public=excluded.public,
            tags=excluded.tags,
            uploaded_at=excluded.uploaded_at,
            last_accessed=excluded.last_accessed
    """, (key, filename, content_type, size_bytes, backend,
          backend_url, cdn_url, checksum, public,
          json.dumps(tags or []), now, now))


def _db_get_file(conn: sqlite3.Connection, key: str) -> dict | None:
    row = conn.execute("SELECT * FROM files WHERE key=?", (key,)).fetchone()
    if not row:
        return None
    d = dict(row)
    try:
        d["tags"] = json.loads(d.get("tags") or "[]")
    except Exception:
        d["tags"] = []
    return d


def _db_touch(conn: sqlite3.Connection, key: str) -> None:
    conn.execute(
        "UPDATE files SET last_accessed=?, access_count=access_count+1 WHERE key=?",
        (time.time(), key),
    )


def _db_delete_file(conn: sqlite3.Connection, key: str) -> None:
    conn.execute("DELETE FROM files WHERE key=?", (key,))


def _db_list_files(conn: sqlite3.Connection,
                   backend: str = "", tags: list[str] | None = None,
                   limit: int = 200, public_only: bool = False) -> list[dict]:
    sql    = "SELECT * FROM files WHERE 1=1"
    params: list = []
    if backend:
        sql += " AND backend=?"
        params.append(backend)
    if public_only:
        sql += " AND public=1"
    sql += " ORDER BY uploaded_at DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        try:
            d["tags"] = json.loads(d.get("tags") or "[]")
        except Exception:
            d["tags"] = []
        if tags:
            row_tags = d["tags"] if isinstance(d["tags"], list) else []
            if not any(t in row_tags for t in tags):
                continue
        result.append(d)
    return result


def _db_stats(conn: sqlite3.Connection) -> dict:
    rows = conn.execute(
        "SELECT backend, COUNT(*) as cnt, SUM(size_bytes) as total FROM files GROUP BY backend"
    ).fetchall()
    file_count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    backends: dict[str, dict] = {}
    for row in rows:
        backends[row["backend"]] = {"files": row["cnt"], "bytes": row["total"] or 0}
    return {"file_count": file_count, "by_backend": backends}


# ── HTTP request handler ───────────────────────────────────────────────────────

class CloudStorageHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):  # silence default access log
        pass

    # ── routing ────────────────────────────────────────────────────────────────

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path   = parsed.path.rstrip("/")
        qs     = urllib.parse.parse_qs(parsed.query)

        if path == "/health":
            self._health()
        elif path == "/files":
            self._list_files(qs)
        elif re.match(r"^/files/[^/]+/download$", path):
            key = path[7:-9]  # strip /files/ and /download
            self._download(key)
        elif re.match(r"^/files/[^/]+/url$", path):
            key = path[7:-4]
            self._get_url(key)
        elif re.match(r"^/files/.+$", path):
            key = path[7:]
            self._file_meta(key)
        else:
            _json_response(self, 404, {"error": "not found"})

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path   = parsed.path.rstrip("/")

        if path == "/upload":
            self._upload()
        elif path == "/upload/url":
            self._upload_from_url()
        elif path == "/sync":
            self._sync()
        else:
            _json_response(self, 404, {"error": "not found"})

    def do_DELETE(self):
        parsed = urllib.parse.urlparse(self.path)
        path   = parsed.path.rstrip("/")
        if re.match(r"^/files/.+$", path):
            key = path[7:]
            self._delete_file(key)
        else:
            _json_response(self, 404, {"error": "not found"})

    def do_PUT(self):
        parsed = urllib.parse.urlparse(self.path)
        path   = parsed.path.rstrip("/")
        if re.match(r"^/files/.+$", path):
            key = path[7:]
            self._update_file(key)
        else:
            _json_response(self, 404, {"error": "not found"})

    # ── handlers ───────────────────────────────────────────────────────────────

    def _health(self):
        with _db_lock:
            conn = _get_db()
            stats = _db_stats(conn)
            conn.close()
        _json_response(self, 200, {
            "status":  "ok",
            "uptime":  round(time.time() - START_TIME, 2),
            "port":    PORT,
            "backends": {
                "r2":       bool(R2_ACCOUNT_ID and R2_ACCESS_KEY_ID),
                "supabase": bool(SUPABASE_URL and SUPABASE_SERVICE_KEY),
                "github":   bool(GITHUB_TOKEN and GITHUB_ORG and GITHUB_REPO),
            },
            **stats,
        })

    def _list_files(self, qs: dict):
        backend     = (qs.get("backend", [""])[0]).strip()
        tags_raw    = qs.get("tags", [""])[0]
        tags        = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else None
        limit       = min(int(qs.get("limit", ["200"])[0]), 1000)
        public_only = qs.get("public", [""])[0].lower() == "true"
        with _db_lock:
            conn  = _get_db()
            files = _db_list_files(conn, backend=backend, tags=tags,
                                   limit=limit, public_only=public_only)
            conn.close()
        _json_response(self, 200, {"files": files, "count": len(files)})

    def _file_meta(self, key: str):
        with _db_lock:
            conn = _get_db()
            meta = _db_get_file(conn, key)
            conn.close()
        if not meta:
            _json_response(self, 404, {"error": "file not found"})
            return
        _json_response(self, 200, meta)

    def _download(self, key: str):
        with _db_lock:
            conn = _get_db()
            meta = _db_get_file(conn, key)
            if meta:
                _db_touch(conn, key)
                conn.commit()
            conn.close()
        if not meta:
            _json_response(self, 404, {"error": "file not found"})
            return
        try:
            data = _download_from_backend(meta["backend"], key)
        except Exception as exc:
            _log(f"Download error for {key}: {exc}")
            _json_response(self, 502, {"error": str(exc)})
            return
        ct = meta.get("content_type") or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Disposition",
                         f'inline; filename="{meta.get("filename", key)}"')
        self.end_headers()
        self.wfile.write(data)

    def _get_url(self, key: str):
        with _db_lock:
            conn = _get_db()
            meta = _db_get_file(conn, key)
            conn.close()
        if not meta:
            _json_response(self, 404, {"error": "file not found"})
            return
        _json_response(self, 200, {
            "key":         key,
            "cdn_url":     meta.get("cdn_url") or meta.get("backend_url"),
            "backend_url": meta.get("backend_url"),
            "backend":     meta.get("backend"),
        })

    def _upload(self):
        raw = _read_body(self)
        try:
            body = json.loads(raw)
        except Exception:
            _json_response(self, 400, {"error": "invalid JSON body"})
            return

        key          = body.get("key", "")
        content_b64  = body.get("content_base64", "")
        content_type = body.get("content_type", "")
        backend_hint = body.get("backend", "")
        public       = 1 if body.get("public", True) else 0
        tags         = body.get("tags", [])

        if not key:
            _json_response(self, 400, {"error": "key is required"})
            return
        if not content_b64:
            _json_response(self, 400, {"error": "content_base64 is required"})
            return

        try:
            data = base64.b64decode(content_b64)
        except Exception:
            _json_response(self, 400, {"error": "invalid base64 content"})
            return

        filename = key.split("/")[-1]
        if not content_type:
            content_type = _guess_content_type(filename)

        backend = backend_hint if backend_hint in ("r2", "supabase", "github") else ""
        if not backend:
            backend = _choose_backend(key, content_type, len(data))

        try:
            backend_url, cdn_url = _upload_to_backend(backend, key, data, content_type)
        except Exception as exc:
            _log(f"Upload error [{backend}] {key}: {exc}")
            _json_response(self, 502, {"error": str(exc), "backend": backend})
            return

        checksum = _checksum(data)
        with _db_lock:
            conn = _get_db()
            _db_insert_file(conn, key, filename, content_type, len(data),
                            backend, backend_url, cdn_url, checksum, public, tags)
            conn.commit()
            meta = _db_get_file(conn, key)
            conn.close()

        _log(f"Uploaded {key} ({len(data)} bytes) to {backend}")
        _json_response(self, 201, {"status": "uploaded", "file": meta})

    def _upload_from_url(self):
        raw = _read_body(self)
        try:
            body = json.loads(raw)
        except Exception:
            _json_response(self, 400, {"error": "invalid JSON body"})
            return

        source_url   = body.get("source_url", "")
        key          = body.get("key", "")
        backend_hint = body.get("backend", "")

        if not source_url or not key:
            _json_response(self, 400, {"error": "source_url and key are required"})
            return

        try:
            req = urllib.request.Request(source_url, headers={"User-Agent": "FractalMesh/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data         = resp.read()
                content_type = resp.headers.get("Content-Type", "application/octet-stream").split(";")[0].strip()
        except Exception as exc:
            _json_response(self, 502, {"error": f"failed to fetch source URL: {exc}"})
            return

        filename = key.split("/")[-1]
        if not content_type:
            content_type = _guess_content_type(filename)

        backend = backend_hint if backend_hint in ("r2", "supabase", "github") else ""
        if not backend:
            backend = _choose_backend(key, content_type, len(data))

        try:
            backend_url, cdn_url = _upload_to_backend(backend, key, data, content_type)
        except Exception as exc:
            _log(f"URL-upload error [{backend}] {key}: {exc}")
            _json_response(self, 502, {"error": str(exc)})
            return

        checksum = _checksum(data)
        with _db_lock:
            conn = _get_db()
            _db_insert_file(conn, key, filename, content_type, len(data),
                            backend, backend_url, cdn_url, checksum, 1, [])
            conn.commit()
            meta = _db_get_file(conn, key)
            conn.close()

        _log(f"URL-uploaded {key} ({len(data)} bytes) to {backend}")
        _json_response(self, 201, {"status": "uploaded", "file": meta})

    def _delete_file(self, key: str):
        if not _require_admin(self):
            return
        with _db_lock:
            conn = _get_db()
            meta = _db_get_file(conn, key)
            conn.close()
        if not meta:
            _json_response(self, 404, {"error": "file not found"})
            return
        try:
            _delete_from_backend(meta["backend"], key)
        except Exception as exc:
            _log(f"Backend delete error [{meta['backend']}] {key}: {exc}")
        with _db_lock:
            conn = _get_db()
            _db_delete_file(conn, key)
            conn.commit()
            conn.close()
        _log(f"Deleted {key} from {meta['backend']}")
        _json_response(self, 200, {"status": "deleted", "key": key})

    def _update_file(self, key: str):
        if not _require_admin(self):
            return
        raw = _read_body(self)
        try:
            body = json.loads(raw)
        except Exception:
            _json_response(self, 400, {"error": "invalid JSON body"})
            return
        with _db_lock:
            conn = _get_db()
            meta = _db_get_file(conn, key)
            if not meta:
                conn.close()
                _json_response(self, 404, {"error": "file not found"})
                return
            updates: list[str] = []
            params: list       = []
            if "public" in body:
                updates.append("public=?")
                params.append(1 if body["public"] else 0)
            if "tags" in body:
                updates.append("tags=?")
                params.append(json.dumps(body["tags"]))
            if updates:
                params.append(key)
                conn.execute(f"UPDATE files SET {', '.join(updates)} WHERE key=?", params)
                conn.commit()
            meta = _db_get_file(conn, key)
            conn.close()
        _json_response(self, 200, {"status": "updated", "file": meta})

    def _sync(self):
        if not _require_admin(self):
            return
        report: dict[str, dict] = {"r2": {}, "supabase": {}, "github": {}, "summary": {}}
        added = deleted = errors = 0

        # ── R2 sync ────────────────────────────────────────────────────────────
        if R2_ACCOUNT_ID and R2_ACCESS_KEY_ID:
            try:
                remote_keys = set(_r2_list())
                with _db_lock:
                    conn = _get_db()
                    local_keys = {
                        r["key"] for r in
                        conn.execute("SELECT key FROM files WHERE backend='r2'").fetchall()
                    }
                    # keys on remote but not in DB → add record (minimal metadata)
                    for rk in remote_keys - local_keys:
                        ct = _guess_content_type(rk.split("/")[-1])
                        burl = _r2_cdn_url(rk)
                        _db_insert_file(conn, rk, rk.split("/")[-1], ct, 0,
                                        "r2", burl, _r2_cdn_url(rk), "", 1, [])
                        added += 1
                    # keys in DB but not on remote → remove DB record
                    for lk in local_keys - remote_keys:
                        _db_delete_file(conn, lk)
                        deleted += 1
                    conn.commit()
                    conn.close()
                report["r2"] = {"remote": len(remote_keys), "added": added, "removed": deleted}
            except Exception as exc:
                report["r2"] = {"error": str(exc)}
                errors += 1

        # ── Supabase sync ──────────────────────────────────────────────────────
        if SUPABASE_URL and SUPABASE_SERVICE_KEY:
            try:
                sb_added = sb_deleted = 0
                remote_keys = set(_sb_list())
                with _db_lock:
                    conn = _get_db()
                    local_keys = {
                        r["key"] for r in
                        conn.execute("SELECT key FROM files WHERE backend='supabase'").fetchall()
                    }
                    for rk in remote_keys - local_keys:
                        ct   = _guess_content_type(rk.split("/")[-1])
                        burl = _sb_cdn_url(rk)
                        _db_insert_file(conn, rk, rk.split("/")[-1], ct, 0,
                                        "supabase", burl, burl, "", 1, [])
                        sb_added += 1
                    for lk in local_keys - remote_keys:
                        _db_delete_file(conn, lk)
                        sb_deleted += 1
                    conn.commit()
                    conn.close()
                report["supabase"] = {
                    "remote": len(remote_keys),
                    "added":  sb_added,
                    "removed": sb_deleted,
                }
            except Exception as exc:
                report["supabase"] = {"error": str(exc)}
                errors += 1

        # ── GitHub sync ────────────────────────────────────────────────────────
        if GITHUB_TOKEN and GITHUB_ORG and GITHUB_REPO:
            try:
                gh_added = gh_deleted = 0
                remote_keys = set(_gh_list())
                with _db_lock:
                    conn = _get_db()
                    local_keys = {
                        r["key"] for r in
                        conn.execute("SELECT key FROM files WHERE backend='github'").fetchall()
                    }
                    for rk in remote_keys - local_keys:
                        ct = _guess_content_type(rk.split("/")[-1])
                        _db_insert_file(conn, rk, rk.split("/")[-1], ct, 0,
                                        "github", "", "", "", 1, [])
                        gh_added += 1
                    for lk in local_keys - remote_keys:
                        _db_delete_file(conn, lk)
                        gh_deleted += 1
                    conn.commit()
                    conn.close()
                report["github"] = {
                    "remote":  len(remote_keys),
                    "added":   gh_added,
                    "removed": gh_deleted,
                }
            except Exception as exc:
                report["github"] = {"error": str(exc)}
                errors += 1

        report["summary"] = {
            "total_added":   added,
            "total_deleted": deleted,
            "errors":        errors,
        }
        _log(f"Sync complete: {report['summary']}")
        _json_response(self, 200, {"status": "sync_complete", "report": report})


# ── server ─────────────────────────────────────────────────────────────────────

def _run_server() -> None:
    _init_db()
    _log(f"FractalMesh Cloud Storage starting on port {PORT}")
    server = HTTPServer(("0.0.0.0", PORT), CloudStorageHandler)
    server.serve_forever()


if __name__ == "__main__":
    _run_server()
