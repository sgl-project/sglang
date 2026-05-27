#!/usr/bin/env python3
"""
fm_auth_manager.py — FractalMesh Authentication & Authorization Manager (Port 7875)
Central auth system: registration, login, JWT-like tokens, roles, permissions, OAuth2.
Samuel James Hiotis | ABN 56 628 117 363
"""
import os, json, time, sqlite3, hashlib, hmac, secrets, base64, threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.request, urllib.error

# ---------------------------------------------------------------------------
# Vault loading — MUST appear before any os.getenv calls
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
PORT               = int(os.getenv("AUTH_MANAGER_PORT", "7875"))
_JWT_SECRET        = os.getenv("JWT_SECRET") or os.getenv("ADMIN_SECRET", "")
SENDGRID_API_KEY   = os.getenv("SENDGRID_API_KEY", "")
SENDGRID_FROM      = os.getenv("SENDGRID_FROM_EMAIL", "noreply@fractalmesh.io")
ADMIN_SECRET       = os.getenv("ADMIN_SECRET", "")

ROOT = Path(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")))
DB   = ROOT / "database" / "sovereign.db"
for _p in (ROOT, DB.parent):
    _p.mkdir(parents=True, exist_ok=True)

START_TIME = time.time()

# ---------------------------------------------------------------------------
# Password hashing helpers
# ---------------------------------------------------------------------------
PBKDF2_ITERATIONS = 260_000
PBKDF2_KEYLEN     = 32

def _hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """Return (hex_hash, hex_salt). Generate new salt when not provided."""
    if salt is None:
        salt = secrets.token_hex(32)
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"),
        bytes.fromhex(salt), PBKDF2_ITERATIONS, dklen=PBKDF2_KEYLEN
    )
    return dk.hex(), salt

def _verify_password(password: str, stored_hash: str, salt: str) -> bool:
    computed, _ = _hash_password(password, salt)
    return hmac.compare_digest(computed, stored_hash)

# ---------------------------------------------------------------------------
# JWT-like token helpers
# ---------------------------------------------------------------------------
_HEADER_B64 = base64.urlsafe_b64encode(
    json.dumps({"alg": "HS256", "typ": "FM-JWT"}, separators=(",", ":")).encode()
).rstrip(b"=").decode()

def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

def _b64url_decode(s: str) -> bytes:
    padding = 4 - len(s) % 4
    if padding != 4:
        s += "=" * padding
    return base64.urlsafe_b64decode(s)

def _make_token(user_id: str, email: str, role: str, session_id: str) -> tuple[str, float]:
    """Return (token_string, expires_at_epoch)."""
    now = time.time()
    exp = now + 3600
    payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "iat": now,
        "exp": exp,
        "sid": session_id,
    }
    payload_b64 = _b64url(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{_HEADER_B64}.{payload_b64}"
    sig = hmac.new(
        _JWT_SECRET.encode("utf-8"),
        signing_input.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    token = f"{signing_input}.{_b64url(sig)}"
    return token, exp

def _verify_token(token: str) -> dict | None:
    """Return decoded payload dict or None on failure."""
    parts = token.split(".")
    if len(parts) != 3:
        return None
    header_b64, payload_b64, sig_b64 = parts
    signing_input = f"{header_b64}.{payload_b64}"
    expected_sig = hmac.new(
        _JWT_SECRET.encode("utf-8"),
        signing_input.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    try:
        provided_sig = _b64url_decode(sig_b64)
    except Exception:
        return None
    if not hmac.compare_digest(expected_sig, provided_sig):
        return None
    try:
        payload = json.loads(_b64url_decode(payload_b64))
    except Exception:
        return None
    if payload.get("exp", 0) < time.time():
        return None
    return payload

def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()

# ---------------------------------------------------------------------------
# Database init and helpers
# ---------------------------------------------------------------------------
def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn

def _db_init():
    conn = _get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id              INTEGER PRIMARY KEY,
            user_id         TEXT UNIQUE NOT NULL,
            email           TEXT UNIQUE NOT NULL,
            password_hash   TEXT NOT NULL,
            salt            TEXT NOT NULL,
            name            TEXT,
            role            TEXT DEFAULT 'user',
            permissions     TEXT,
            status          TEXT DEFAULT 'active',
            email_verified  INTEGER DEFAULT 0,
            mfa_secret      TEXT,
            login_count     INTEGER DEFAULT 0,
            last_login_at   REAL,
            created_at      REAL,
            updated_at      REAL
        );
        CREATE TABLE IF NOT EXISTS sessions (
            id          INTEGER PRIMARY KEY,
            session_id  TEXT UNIQUE NOT NULL,
            user_id     TEXT NOT NULL,
            token_hash  TEXT UNIQUE NOT NULL,
            expires_at  REAL NOT NULL,
            ip_address  TEXT,
            user_agent  TEXT,
            revoked     INTEGER DEFAULT 0,
            created_at  REAL
        );
        CREATE TABLE IF NOT EXISTS password_resets (
            id          INTEGER PRIMARY KEY,
            user_id     TEXT NOT NULL,
            token_hash  TEXT UNIQUE NOT NULL,
            expires_at  REAL NOT NULL,
            used        INTEGER DEFAULT 0,
            created_at  REAL
        );
        CREATE TABLE IF NOT EXISTS audit_log (
            id          INTEGER PRIMARY KEY,
            user_id     TEXT,
            action      TEXT,
            ip_address  TEXT,
            metadata    TEXT,
            created_at  REAL
        );
        CREATE TABLE IF NOT EXISTS roles (
            id          INTEGER PRIMARY KEY,
            name        TEXT UNIQUE NOT NULL,
            permissions TEXT,
            description TEXT,
            created_at  REAL
        );
    """)
    conn.commit()
    _seed_roles(conn)
    conn.close()

_SEED_ROLES = [
    ("admin",     '["*"]',                                   "Full administrative access"),
    ("user",      '["read:own","write:own"]',                "Standard user"),
    ("developer", '["read:own","write:own","api:access"]',   "Developer API access"),
    ("viewer",    '["read:own"]',                            "Read-only access"),
]

def _seed_roles(conn: sqlite3.Connection):
    row = conn.execute("SELECT COUNT(*) FROM roles").fetchone()
    if row[0] > 0:
        return
    now = time.time()
    for name, perms, desc in _SEED_ROLES:
        conn.execute(
            "INSERT OR IGNORE INTO roles (name, permissions, description, created_at) VALUES (?,?,?,?)",
            (name, perms, desc, now),
        )
    conn.commit()

def _audit(conn: sqlite3.Connection, user_id: str | None, action: str,
           ip: str, meta: dict):
    conn.execute(
        "INSERT INTO audit_log (user_id, action, ip_address, metadata, created_at) VALUES (?,?,?,?,?)",
        (user_id, action, ip, json.dumps(meta), time.time()),
    )

def _get_user_permissions(conn: sqlite3.Connection, role: str) -> list[str]:
    row = conn.execute("SELECT permissions FROM roles WHERE name=?", (role,)).fetchone()
    if row:
        try:
            return json.loads(row["permissions"])
        except Exception:
            pass
    return []

# ---------------------------------------------------------------------------
# Background cleanup thread
# ---------------------------------------------------------------------------
def _cleanup_loop():
    while True:
        time.sleep(3600)
        try:
            conn = _get_db()
            now = time.time()
            ninety_days_ago = now - 86400 * 90
            conn.execute("DELETE FROM sessions WHERE expires_at < ? OR revoked=1", (now,))
            conn.execute("DELETE FROM password_resets WHERE expires_at < ? OR used=1", (now,))
            conn.execute("DELETE FROM audit_log WHERE created_at < ?", (ninety_days_ago,))
            conn.commit()
            conn.close()
        except Exception:
            pass

# ---------------------------------------------------------------------------
# SendGrid email helper
# ---------------------------------------------------------------------------
def _send_reset_email(to_email: str, reset_token: str) -> bool:
    if not SENDGRID_API_KEY:
        return False
    reset_url = f"https://app.fractalmesh.io/reset-password?token={reset_token}"
    body = {
        "personalizations": [{"to": [{"email": to_email}]}],
        "from": {"email": SENDGRID_FROM},
        "subject": "FractalMesh — Password Reset",
        "content": [{"type": "text/plain",
                      "value": f"Reset your password by visiting:\n\n{reset_url}\n\nThis link expires in 1 hour."}],
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        "https://api.sendgrid.com/v3/mail/send",
        data=data,
        headers={
            "Authorization": f"Bearer {SENDGRID_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status in (200, 202)
    except urllib.error.URLError:
        return False

# ---------------------------------------------------------------------------
# HTTP Request Handler
# ---------------------------------------------------------------------------
class AuthHandler(BaseHTTPRequestHandler):
    server_version = "FractalMesh-AuthManager/1.0"

    def log_message(self, fmt, *args):
        pass  # Suppress default access log noise

    # ------------------------------------------------------------------ utils
    def _send_json(self, status: int, data: dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict | None:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except Exception:
            return None

    def _get_bearer(self) -> str | None:
        auth = self.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return None

    def _require_auth(self) -> dict | None:
        """Validate bearer token; return payload dict or send 401 and return None."""
        token = self._get_bearer()
        if not token:
            self._send_json(401, {"error": "Missing bearer token"})
            return None
        payload = _verify_token(token)
        if not payload:
            self._send_json(401, {"error": "Invalid or expired token"})
            return None
        # Check session not revoked
        conn = _get_db()
        try:
            th = _token_hash(token)
            row = conn.execute(
                "SELECT revoked FROM sessions WHERE token_hash=?", (th,)
            ).fetchone()
            if row is None or row["revoked"]:
                self._send_json(401, {"error": "Session revoked"})
                return None
        finally:
            conn.close()
        return payload

    def _require_admin(self) -> dict | None:
        payload = self._require_auth()
        if payload is None:
            return None
        if payload.get("role") != "admin":
            self._send_json(403, {"error": "Admin role required"})
            return None
        return payload

    def _client_ip(self) -> str:
        return self.headers.get("X-Forwarded-For", self.client_address[0])

    # ------------------------------------------------------------------ routing
    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        if path == "/health":
            self._handle_health()
        elif path == "/me":
            self._handle_me_get()
        elif path == "/users":
            self._handle_users_list()
        elif path.startswith("/users/"):
            self._handle_user_get(path)
        elif path == "/roles":
            self._handle_roles()
        elif path == "/audit":
            self._handle_audit()
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        path = self.path.rstrip("/")
        if path == "/register":
            self._handle_register()
        elif path == "/login":
            self._handle_login()
        elif path == "/logout":
            self._handle_logout()
        elif path == "/refresh":
            self._handle_refresh()
        elif path == "/change_password":
            self._handle_change_password()
        elif path == "/forgot_password":
            self._handle_forgot_password()
        elif path == "/reset_password":
            self._handle_reset_password()
        elif path == "/verify_token":
            self._handle_verify_token()
        else:
            self._send_json(404, {"error": "Not found"})

    def do_PUT(self):
        path = self.path.rstrip("/")
        if path == "/me":
            self._handle_me_put()
        elif path.startswith("/users/") and path.endswith("/role"):
            self._handle_user_role(path)
        else:
            self._send_json(404, {"error": "Not found"})

    def do_DELETE(self):
        path = self.path.rstrip("/")
        if path.startswith("/users/"):
            self._handle_user_delete(path)
        else:
            self._send_json(404, {"error": "Not found"})

    # ------------------------------------------------------------------ GET /health
    def _handle_health(self):
        conn = _get_db()
        try:
            active_users = conn.execute(
                "SELECT COUNT(*) FROM users WHERE status='active'"
            ).fetchone()[0]
            active_sessions = conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE revoked=0 AND expires_at>?",
                (time.time(),)
            ).fetchone()[0]
        finally:
            conn.close()
        self._send_json(200, {
            "status": "ok",
            "service": "fm_auth_manager",
            "port": PORT,
            "uptime_seconds": round(time.time() - START_TIME, 1),
            "active_users": active_users,
            "active_sessions": active_sessions,
        })

    # ------------------------------------------------------------------ POST /register
    def _handle_register(self):
        body = self._read_json()
        if body is None:
            self._send_json(400, {"error": "Invalid JSON"}); return
        email    = (body.get("email") or "").strip().lower()
        password = body.get("password") or ""
        name     = (body.get("name") or "").strip()
        role     = (body.get("role") or "user").strip()
        if not email or "@" not in email:
            self._send_json(400, {"error": "Valid email required"}); return
        if len(password) < 8:
            self._send_json(400, {"error": "Password must be at least 8 characters"}); return
        if role not in ("admin", "user", "developer", "viewer"):
            role = "user"
        conn = _get_db()
        try:
            existing = conn.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()
            if existing:
                self._send_json(409, {"error": "Email already registered"}); return
            user_id = "u_" + secrets.token_urlsafe(16)
            pw_hash, salt = _hash_password(password)
            now = time.time()
            perms = _get_user_permissions(conn, role)
            conn.execute(
                """INSERT INTO users
                   (user_id,email,password_hash,salt,name,role,permissions,status,
                    email_verified,login_count,created_at,updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (user_id, email, pw_hash, salt, name, role, json.dumps(perms),
                 "active", 0, 0, now, now),
            )
            session_id = "s_" + secrets.token_urlsafe(24)
            token, expires_at = _make_token(user_id, email, role, session_id)
            conn.execute(
                """INSERT INTO sessions
                   (session_id,user_id,token_hash,expires_at,ip_address,user_agent,revoked,created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (session_id, user_id, _token_hash(token), expires_at,
                 self._client_ip(), self.headers.get("User-Agent", ""), 0, now),
            )
            _audit(conn, user_id, "register", self._client_ip(), {"email": email, "role": role})
            conn.commit()
        finally:
            conn.close()
        self._send_json(201, {
            "user_id": user_id,
            "token": token,
            "expires_at": expires_at,
        })

    # ------------------------------------------------------------------ POST /login
    def _handle_login(self):
        body = self._read_json()
        if body is None:
            self._send_json(400, {"error": "Invalid JSON"}); return
        email    = (body.get("email") or "").strip().lower()
        password = body.get("password") or ""
        if not email or not password:
            self._send_json(400, {"error": "Email and password required"}); return
        conn = _get_db()
        try:
            row = conn.execute(
                "SELECT * FROM users WHERE email=? AND status='active'", (email,)
            ).fetchone()
            if not row:
                self._send_json(401, {"error": "Invalid credentials"}); return
            if not _verify_password(password, row["password_hash"], row["salt"]):
                _audit(conn, row["user_id"], "login_fail", self._client_ip(), {"email": email})
                conn.commit()
                self._send_json(401, {"error": "Invalid credentials"}); return
            now = time.time()
            session_id = "s_" + secrets.token_urlsafe(24)
            token, expires_at = _make_token(row["user_id"], row["email"], row["role"], session_id)
            conn.execute(
                """INSERT INTO sessions
                   (session_id,user_id,token_hash,expires_at,ip_address,user_agent,revoked,created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (session_id, row["user_id"], _token_hash(token), expires_at,
                 self._client_ip(), self.headers.get("User-Agent", ""), 0, now),
            )
            conn.execute(
                "UPDATE users SET login_count=login_count+1, last_login_at=?, updated_at=? WHERE user_id=?",
                (now, now, row["user_id"]),
            )
            _audit(conn, row["user_id"], "login", self._client_ip(), {"email": email})
            conn.commit()
        finally:
            conn.close()
        self._send_json(200, {
            "token": token,
            "user_id": row["user_id"],
            "role": row["role"],
            "expires_at": expires_at,
        })

    # ------------------------------------------------------------------ POST /logout
    def _handle_logout(self):
        payload = self._require_auth()
        if payload is None:
            return
        token = self._get_bearer()
        conn = _get_db()
        try:
            conn.execute(
                "UPDATE sessions SET revoked=1 WHERE token_hash=?", (_token_hash(token),)
            )
            _audit(conn, payload["sub"], "logout", self._client_ip(), {})
            conn.commit()
        finally:
            conn.close()
        self._send_json(200, {"message": "Logged out successfully"})

    # ------------------------------------------------------------------ POST /refresh
    def _handle_refresh(self):
        payload = self._require_auth()
        if payload is None:
            return
        old_token = self._get_bearer()
        conn = _get_db()
        try:
            row = conn.execute(
                "SELECT * FROM users WHERE user_id=? AND status='active'", (payload["sub"],)
            ).fetchone()
            if not row:
                self._send_json(401, {"error": "User not found or inactive"}); return
            # Revoke old session
            conn.execute(
                "UPDATE sessions SET revoked=1 WHERE token_hash=?", (_token_hash(old_token),)
            )
            now = time.time()
            session_id = "s_" + secrets.token_urlsafe(24)
            token, expires_at = _make_token(row["user_id"], row["email"], row["role"], session_id)
            conn.execute(
                """INSERT INTO sessions
                   (session_id,user_id,token_hash,expires_at,ip_address,user_agent,revoked,created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (session_id, row["user_id"], _token_hash(token), expires_at,
                 self._client_ip(), self.headers.get("User-Agent", ""), 0, now),
            )
            _audit(conn, row["user_id"], "refresh", self._client_ip(), {})
            conn.commit()
        finally:
            conn.close()
        self._send_json(200, {
            "token": token,
            "user_id": row["user_id"],
            "role": row["role"],
            "expires_at": expires_at,
        })

    # ------------------------------------------------------------------ GET /me
    def _handle_me_get(self):
        payload = self._require_auth()
        if payload is None:
            return
        conn = _get_db()
        try:
            row = conn.execute(
                "SELECT user_id,email,name,role,permissions,status,email_verified,"
                "login_count,last_login_at,created_at FROM users WHERE user_id=?",
                (payload["sub"],),
            ).fetchone()
        finally:
            conn.close()
        if not row:
            self._send_json(404, {"error": "User not found"}); return
        self._send_json(200, {
            "user_id":        row["user_id"],
            "email":          row["email"],
            "name":           row["name"],
            "role":           row["role"],
            "permissions":    json.loads(row["permissions"] or "[]"),
            "status":         row["status"],
            "email_verified": bool(row["email_verified"]),
            "login_count":    row["login_count"],
            "last_login_at":  row["last_login_at"],
            "created_at":     row["created_at"],
        })

    # ------------------------------------------------------------------ PUT /me
    def _handle_me_put(self):
        payload = self._require_auth()
        if payload is None:
            return
        body = self._read_json()
        if body is None:
            self._send_json(400, {"error": "Invalid JSON"}); return
        updates = {}
        if "name" in body:
            updates["name"] = str(body["name"]).strip()
        if "email" in body:
            new_email = str(body["email"]).strip().lower()
            if "@" not in new_email:
                self._send_json(400, {"error": "Valid email required"}); return
            updates["email"] = new_email
        if not updates:
            self._send_json(400, {"error": "No valid fields to update"}); return
        updates["updated_at"] = time.time()
        set_clause = ", ".join(f"{k}=?" for k in updates)
        values = list(updates.values()) + [payload["sub"]]
        conn = _get_db()
        try:
            conn.execute(
                f"UPDATE users SET {set_clause} WHERE user_id=?", values
            )
            _audit(conn, payload["sub"], "update_profile", self._client_ip(),
                   {"fields": list(updates.keys())})
            conn.commit()
        except sqlite3.IntegrityError:
            self._send_json(409, {"error": "Email already in use"}); return
        finally:
            conn.close()
        self._send_json(200, {"message": "Profile updated"})

    # ------------------------------------------------------------------ POST /change_password
    def _handle_change_password(self):
        payload = self._require_auth()
        if payload is None:
            return
        body = self._read_json()
        if body is None:
            self._send_json(400, {"error": "Invalid JSON"}); return
        old_pw  = body.get("old_password") or ""
        new_pw  = body.get("new_password") or ""
        if not old_pw or not new_pw:
            self._send_json(400, {"error": "old_password and new_password required"}); return
        if len(new_pw) < 8:
            self._send_json(400, {"error": "New password must be at least 8 characters"}); return
        conn = _get_db()
        try:
            row = conn.execute(
                "SELECT password_hash, salt FROM users WHERE user_id=?", (payload["sub"],)
            ).fetchone()
            if not row or not _verify_password(old_pw, row["password_hash"], row["salt"]):
                self._send_json(401, {"error": "Current password incorrect"}); return
            new_hash, new_salt = _hash_password(new_pw)
            now = time.time()
            conn.execute(
                "UPDATE users SET password_hash=?, salt=?, updated_at=? WHERE user_id=?",
                (new_hash, new_salt, now, payload["sub"]),
            )
            # Revoke all other sessions for security
            conn.execute(
                "UPDATE sessions SET revoked=1 WHERE user_id=? AND session_id!=?",
                (payload["sub"], payload.get("sid", "")),
            )
            _audit(conn, payload["sub"], "change_password", self._client_ip(), {})
            conn.commit()
        finally:
            conn.close()
        self._send_json(200, {"message": "Password changed successfully"})

    # ------------------------------------------------------------------ POST /forgot_password
    def _handle_forgot_password(self):
        body = self._read_json()
        if body is None:
            self._send_json(400, {"error": "Invalid JSON"}); return
        email = (body.get("email") or "").strip().lower()
        if not email:
            self._send_json(400, {"error": "Email required"}); return
        conn = _get_db()
        try:
            row = conn.execute(
                "SELECT user_id FROM users WHERE email=? AND status='active'", (email,)
            ).fetchone()
            if row:
                reset_token = secrets.token_urlsafe(48)
                token_hash  = hashlib.sha256(reset_token.encode()).hexdigest()
                now = time.time()
                # Invalidate any existing reset tokens
                conn.execute(
                    "UPDATE password_resets SET used=1 WHERE user_id=?", (row["user_id"],)
                )
                conn.execute(
                    "INSERT INTO password_resets (user_id,token_hash,expires_at,used,created_at) "
                    "VALUES (?,?,?,?,?)",
                    (row["user_id"], token_hash, now + 3600, 0, now),
                )
                _audit(conn, row["user_id"], "forgot_password", self._client_ip(), {"email": email})
                conn.commit()
                _send_reset_email(email, reset_token)
            else:
                conn.commit()
        finally:
            conn.close()
        # Always return 200 to prevent email enumeration
        self._send_json(200, {"message": "If that email is registered, a reset link has been sent"})

    # ------------------------------------------------------------------ POST /reset_password
    def _handle_reset_password(self):
        body = self._read_json()
        if body is None:
            self._send_json(400, {"error": "Invalid JSON"}); return
        token  = body.get("token") or ""
        new_pw = body.get("new_password") or ""
        if not token or not new_pw:
            self._send_json(400, {"error": "token and new_password required"}); return
        if len(new_pw) < 8:
            self._send_json(400, {"error": "Password must be at least 8 characters"}); return
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        conn = _get_db()
        try:
            row = conn.execute(
                "SELECT * FROM password_resets WHERE token_hash=? AND used=0 AND expires_at>?",
                (token_hash, time.time()),
            ).fetchone()
            if not row:
                self._send_json(400, {"error": "Invalid or expired reset token"}); return
            new_hash, new_salt = _hash_password(new_pw)
            now = time.time()
            conn.execute(
                "UPDATE users SET password_hash=?, salt=?, updated_at=? WHERE user_id=?",
                (new_hash, new_salt, now, row["user_id"]),
            )
            conn.execute(
                "UPDATE password_resets SET used=1 WHERE token_hash=?", (token_hash,)
            )
            # Revoke all sessions for security
            conn.execute("UPDATE sessions SET revoked=1 WHERE user_id=?", (row["user_id"],))
            _audit(conn, row["user_id"], "reset_password", self._client_ip(), {})
            conn.commit()
        finally:
            conn.close()
        self._send_json(200, {"message": "Password reset successfully"})

    # ------------------------------------------------------------------ POST /verify_token
    def _handle_verify_token(self):
        body = self._read_json()
        if body is None:
            self._send_json(400, {"error": "Invalid JSON"}); return
        token = body.get("token") or ""
        if not token:
            self._send_json(400, {"error": "token required"}); return
        payload = _verify_token(token)
        if not payload:
            self._send_json(200, {"valid": False, "error": "Invalid or expired token"}); return
        # Check not revoked
        conn = _get_db()
        try:
            th  = _token_hash(token)
            row = conn.execute(
                "SELECT revoked FROM sessions WHERE token_hash=?", (th,)
            ).fetchone()
            if row is None or row["revoked"]:
                self._send_json(200, {"valid": False, "error": "Session revoked"}); return
            perms = _get_user_permissions(conn, payload["role"])
        finally:
            conn.close()
        expires_in = max(0, int(payload["exp"] - time.time()))
        self._send_json(200, {
            "valid":              True,
            "user_id":            payload["sub"],
            "email":              payload["email"],
            "role":               payload["role"],
            "permissions":        perms,
            "expires_in_seconds": expires_in,
        })

    # ------------------------------------------------------------------ GET /users  (admin)
    def _handle_users_list(self):
        if self._require_admin() is None:
            return
        conn = _get_db()
        try:
            rows = conn.execute(
                "SELECT user_id,email,name,role,status,email_verified,"
                "login_count,last_login_at,created_at FROM users ORDER BY created_at DESC"
            ).fetchall()
        finally:
            conn.close()
        users = [{
            "user_id":        r["user_id"],
            "email":          r["email"],
            "name":           r["name"],
            "role":           r["role"],
            "status":         r["status"],
            "email_verified": bool(r["email_verified"]),
            "login_count":    r["login_count"],
            "last_login_at":  r["last_login_at"],
            "created_at":     r["created_at"],
        } for r in rows]
        self._send_json(200, {"users": users, "total": len(users)})

    # ------------------------------------------------------------------ GET /users/{user_id}  (admin)
    def _handle_user_get(self, path: str):
        # Could also be /users/{id}/role  — only handle plain /users/{id} here
        parts = [p for p in path.split("/") if p]
        if len(parts) != 2:
            # Let other handlers deal with sub-paths
            self._send_json(404, {"error": "Not found"}); return
        if self._require_admin() is None:
            return
        target_uid = parts[1]
        conn = _get_db()
        try:
            row = conn.execute(
                "SELECT user_id,email,name,role,permissions,status,email_verified,"
                "login_count,last_login_at,created_at,updated_at FROM users WHERE user_id=?",
                (target_uid,),
            ).fetchone()
        finally:
            conn.close()
        if not row:
            self._send_json(404, {"error": "User not found"}); return
        self._send_json(200, {
            "user_id":        row["user_id"],
            "email":          row["email"],
            "name":           row["name"],
            "role":           row["role"],
            "permissions":    json.loads(row["permissions"] or "[]"),
            "status":         row["status"],
            "email_verified": bool(row["email_verified"]),
            "login_count":    row["login_count"],
            "last_login_at":  row["last_login_at"],
            "created_at":     row["created_at"],
            "updated_at":     row["updated_at"],
        })

    # ------------------------------------------------------------------ PUT /users/{user_id}/role  (admin)
    def _handle_user_role(self, path: str):
        admin_payload = self._require_admin()
        if admin_payload is None:
            return
        parts = [p for p in path.split("/") if p]
        # parts: ['users', '<uid>', 'role']
        if len(parts) != 3:
            self._send_json(404, {"error": "Not found"}); return
        target_uid = parts[1]
        body = self._read_json()
        if body is None:
            self._send_json(400, {"error": "Invalid JSON"}); return
        new_role = (body.get("role") or "").strip()
        if new_role not in ("admin", "user", "developer", "viewer"):
            self._send_json(400, {"error": "Invalid role. Must be admin/user/developer/viewer"}); return
        conn = _get_db()
        try:
            row = conn.execute("SELECT user_id FROM users WHERE user_id=?", (target_uid,)).fetchone()
            if not row:
                self._send_json(404, {"error": "User not found"}); return
            perms = _get_user_permissions(conn, new_role)
            conn.execute(
                "UPDATE users SET role=?, permissions=?, updated_at=? WHERE user_id=?",
                (new_role, json.dumps(perms), time.time(), target_uid),
            )
            _audit(conn, admin_payload["sub"], "set_role", self._client_ip(),
                   {"target": target_uid, "role": new_role})
            conn.commit()
        finally:
            conn.close()
        self._send_json(200, {"message": f"Role updated to {new_role}", "user_id": target_uid})

    # ------------------------------------------------------------------ DELETE /users/{user_id}  (admin)
    def _handle_user_delete(self, path: str):
        admin_payload = self._require_admin()
        if admin_payload is None:
            return
        parts = [p for p in path.split("/") if p]
        if len(parts) != 2:
            self._send_json(404, {"error": "Not found"}); return
        target_uid = parts[1]
        if target_uid == admin_payload["sub"]:
            self._send_json(400, {"error": "Cannot delete your own account"}); return
        conn = _get_db()
        try:
            row = conn.execute("SELECT user_id FROM users WHERE user_id=?", (target_uid,)).fetchone()
            if not row:
                self._send_json(404, {"error": "User not found"}); return
            now = time.time()
            conn.execute(
                "UPDATE users SET status='deleted', updated_at=? WHERE user_id=?",
                (now, target_uid),
            )
            conn.execute("UPDATE sessions SET revoked=1 WHERE user_id=?", (target_uid,))
            _audit(conn, admin_payload["sub"], "delete_user", self._client_ip(),
                   {"target": target_uid})
            conn.commit()
        finally:
            conn.close()
        self._send_json(200, {"message": "User deleted", "user_id": target_uid})

    # ------------------------------------------------------------------ GET /roles
    def _handle_roles(self):
        conn = _get_db()
        try:
            rows = conn.execute(
                "SELECT name, permissions, description, created_at FROM roles ORDER BY name"
            ).fetchall()
        finally:
            conn.close()
        roles = [{
            "name":        r["name"],
            "permissions": json.loads(r["permissions"] or "[]"),
            "description": r["description"],
            "created_at":  r["created_at"],
        } for r in rows]
        self._send_json(200, {"roles": roles})

    # ------------------------------------------------------------------ GET /audit  (admin)
    def _handle_audit(self):
        if self._require_admin() is None:
            return
        conn = _get_db()
        try:
            rows = conn.execute(
                "SELECT user_id, action, ip_address, metadata, created_at "
                "FROM audit_log ORDER BY created_at DESC LIMIT 500"
            ).fetchall()
        finally:
            conn.close()
        entries = [{
            "user_id":    r["user_id"],
            "action":     r["action"],
            "ip_address": r["ip_address"],
            "metadata":   json.loads(r["metadata"] or "{}"),
            "created_at": r["created_at"],
        } for r in rows]
        self._send_json(200, {"audit_log": entries, "total": len(entries)})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    _db_init()

    cleanup_thread = threading.Thread(target=_cleanup_loop, daemon=True)
    cleanup_thread.start()

    server = HTTPServer(("0.0.0.0", PORT), AuthHandler)
    print(f"[fm_auth_manager] Listening on port {PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[fm_auth_manager] Shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
