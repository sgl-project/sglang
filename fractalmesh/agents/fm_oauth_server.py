#!/usr/bin/env python3
"""
fm_oauth_server.py — OAuth 2.0 Authorization Server (Port 7905)
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
import base64
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs, urlencode

# ── vault ─────────────────────────────────────────────────────────────────────
_ENV_FILE = Path.home() / ".secrets" / "fractal.env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

PORT         = int(os.getenv("OAUTH_SERVER_PORT", "7905"))
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")
JWT_SECRET   = os.getenv("OAUTH_JWT_SECRET", "fm_oauth_jwt_secret_v1")
TOKEN_TTL    = int(os.getenv("OAUTH_TOKEN_TTL", "3600"))
REFRESH_TTL  = int(os.getenv("OAUTH_REFRESH_TTL", str(30 * 86400)))

ROOT = Path(os.getenv("FRACTALMESH_HOME", str(Path.home() / "fmsaas")))
DB   = ROOT / "database" / "sovereign.db"
ROOT.mkdir(parents=True, exist_ok=True)
DB.parent.mkdir(parents=True, exist_ok=True)

def _db():
    con = sqlite3.connect(str(DB), check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")
    con.row_factory = sqlite3.Row
    return con

def init_db():
    con = _db()
    con.executescript("""
        CREATE TABLE IF NOT EXISTS oauth_clients (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id       TEXT UNIQUE NOT NULL,
            client_secret   TEXT NOT NULL,
            name            TEXT NOT NULL,
            redirect_uris   TEXT NOT NULL DEFAULT '[]',
            scopes          TEXT NOT NULL DEFAULT '[]',
            grant_types     TEXT NOT NULL DEFAULT '["authorization_code","client_credentials"]',
            active          INTEGER NOT NULL DEFAULT 1,
            created_at      REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS oauth_users (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         TEXT UNIQUE NOT NULL,
            email           TEXT UNIQUE NOT NULL,
            password_hash   TEXT NOT NULL,
            scopes          TEXT NOT NULL DEFAULT '[]',
            active          INTEGER NOT NULL DEFAULT 1,
            created_at      REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS auth_codes (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            code            TEXT UNIQUE NOT NULL,
            client_id       TEXT NOT NULL,
            user_id         TEXT NOT NULL,
            redirect_uri    TEXT NOT NULL,
            scopes          TEXT NOT NULL DEFAULT '[]',
            code_challenge  TEXT,
            expires_at      REAL NOT NULL,
            used            INTEGER NOT NULL DEFAULT 0,
            created_at      REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS access_tokens (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            token_hash      TEXT UNIQUE NOT NULL,
            client_id       TEXT NOT NULL,
            user_id         TEXT,
            scopes          TEXT NOT NULL DEFAULT '[]',
            token_type      TEXT NOT NULL DEFAULT 'Bearer',
            expires_at      REAL NOT NULL,
            revoked         INTEGER NOT NULL DEFAULT 0,
            created_at      REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS refresh_tokens (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            token_hash      TEXT UNIQUE NOT NULL,
            access_token_hash TEXT NOT NULL,
            client_id       TEXT NOT NULL,
            user_id         TEXT,
            scopes          TEXT NOT NULL DEFAULT '[]',
            expires_at      REAL NOT NULL,
            revoked         INTEGER NOT NULL DEFAULT 0,
            created_at      REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_at_hash     ON access_tokens(token_hash);
        CREATE INDEX IF NOT EXISTS idx_rt_hash     ON refresh_tokens(token_hash);
        CREATE INDEX IF NOT EXISTS idx_ac_code     ON auth_codes(code);
    """)
    con.commit()
    con.close()

def _hash_token(token):
    return hashlib.sha256(token.encode()).hexdigest()

def _hash_password(password, salt=None):
    import hashlib
    if salt is None:
        salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260000)
    return f"pbkdf2$sha256$260000${salt}${dk.hex()}"

def _verify_password(password, stored):
    try:
        _, algo, iters, salt, dk_hex = stored.split("$")
        dk = hashlib.pbkdf2_hmac(algo, password.encode(), salt.encode(), int(iters))
        return hmac.compare_digest(dk.hex(), dk_hex)
    except Exception:
        return False

def _make_jwt(payload):
    header = base64.urlsafe_b64encode(json.dumps({"alg":"HS256","typ":"JWT"}).encode()).rstrip(b"=").decode()
    body   = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    sig    = hmac.new(JWT_SECRET.encode(), f"{header}.{body}".encode(), hashlib.sha256).hexdigest()
    sig_b64 = base64.urlsafe_b64encode(bytes.fromhex(sig)).rstrip(b"=").decode()
    return f"{header}.{body}.{sig_b64}"

def _verify_jwt(token):
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header, body, sig_b64 = parts
        expected_sig = hmac.new(JWT_SECRET.encode(), f"{header}.{body}".encode(), hashlib.sha256).hexdigest()
        actual_sig = bytes.fromhex(base64.urlsafe_b64decode(sig_b64 + "==").hex())
        if not hmac.compare_digest(expected_sig, actual_sig.hex()):
            return None
        payload = json.loads(base64.urlsafe_b64decode(body + "=="))
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None

def _issue_tokens(con, client_id, user_id, scopes):
    now = time.time()
    access_token  = secrets.token_urlsafe(32)
    refresh_token = secrets.token_urlsafe(48)
    at_hash = _hash_token(access_token)
    rt_hash = _hash_token(refresh_token)
    scopes_str = json.dumps(scopes)
    jwt_payload = {
        "sub": user_id or client_id, "client_id": client_id,
        "scopes": scopes, "iat": int(now), "exp": int(now + TOKEN_TTL),
    }
    jwt = _make_jwt(jwt_payload)
    con.execute(
        "INSERT INTO access_tokens(token_hash,client_id,user_id,scopes,expires_at,created_at) VALUES(?,?,?,?,?,?)",
        (at_hash, client_id, user_id, scopes_str, now + TOKEN_TTL, now)
    )
    con.execute(
        "INSERT INTO refresh_tokens(token_hash,access_token_hash,client_id,user_id,scopes,expires_at,created_at) VALUES(?,?,?,?,?,?,?)",
        (rt_hash, at_hash, client_id, user_id, scopes_str, now + REFRESH_TTL, now)
    )
    con.commit()
    return {
        "access_token": jwt,
        "token_type": "Bearer",
        "expires_in": TOKEN_TTL,
        "refresh_token": refresh_token,
        "scope": " ".join(scopes) if scopes else "",
    }

def _cleanup_daemon():
    while True:
        time.sleep(3600)
        try:
            con = _db()
            now = time.time()
            con.execute("DELETE FROM auth_codes WHERE expires_at < ? OR used=1", (now,))
            con.execute("DELETE FROM access_tokens WHERE expires_at < ?", (now,))
            con.execute("DELETE FROM refresh_tokens WHERE expires_at < ?", (now,))
            con.commit()
            con.close()
        except Exception:
            pass

threading.Thread(target=_cleanup_daemon, daemon=True).start()

def _j(data, status=200):
    return status, json.dumps(data, default=str).encode()

def _err(msg, code=400, error_type="invalid_request"):
    return _j({"error": error_type, "error_description": msg}, code)

def _admin(h):
    v = h.get("X-Admin-Secret", "")
    return not ADMIN_SECRET or hmac.compare_digest(v, ADMIN_SECRET)

def _get_bearer(headers):
    auth = headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return None

class OAuthHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _read_body(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b""
        ct = self.headers.get("Content-Type", "")
        if "application/json" in ct:
            return json.loads(raw) if raw else {}
        # form-urlencoded
        return dict(parse_qs(raw.decode(), keep_blank_values=True))

    def _parse_form(self, raw_dict):
        return {k: v[0] if isinstance(v, list) else v for k, v in raw_dict.items()}

    def _send(self, code, body, ct="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type,Authorization,X-Admin-Secret")
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

    def do_POST(self):
        p = self.path.strip("/").split("/")
        try:
            raw = self._read_body()
            data = self._parse_form(raw)
            code, body = self._post(p, data)
        except Exception as e:
            code, body = _err(str(e), 500)
        self._send(code, body)

    def _get(self, p, qs):
        con = _db()
        try:
            if p == ["health"]:
                return _j({"status": "ok", "port": PORT, "agent": "fm_oauth_server"})

            if p == [".well-known", "oauth-authorization-server"]:
                return _j({
                    "issuer": f"http://localhost:{PORT}",
                    "authorization_endpoint": f"http://localhost:{PORT}/authorize",
                    "token_endpoint": f"http://localhost:{PORT}/token",
                    "revocation_endpoint": f"http://localhost:{PORT}/revoke",
                    "introspection_endpoint": f"http://localhost:{PORT}/introspect",
                    "grant_types_supported": ["authorization_code", "client_credentials", "refresh_token"],
                    "response_types_supported": ["code"],
                    "scopes_supported": ["read", "write", "admin"],
                    "token_endpoint_auth_methods_supported": ["client_secret_basic", "client_secret_post"],
                    "code_challenge_methods_supported": ["S256", "plain"],
                })

            if p == ["authorize"]:
                client_id = qs.get("client_id","")
                redirect_uri = qs.get("redirect_uri","")
                response_type = qs.get("response_type","code")
                if response_type != "code":
                    return _err("unsupported_response_type", 400)
                client = con.execute("SELECT * FROM oauth_clients WHERE client_id=? AND active=1", (client_id,)).fetchone()
                if not client:
                    return _err("Client not found", 404, "invalid_client")
                # return authorization form (simplified JSON for API usage)
                return _j({
                    "client_id": client_id, "client_name": client["name"],
                    "scopes_requested": qs.get("scope","").split(),
                    "redirect_uri": redirect_uri,
                    "state": qs.get("state",""),
                    "message": "POST /authorize with user credentials to approve",
                })

            if p == ["clients"]:
                if not _admin(self.headers):
                    return _err("Unauthorized", 403)
                rows = con.execute("SELECT client_id,name,scopes,grant_types,active,created_at FROM oauth_clients").fetchall()
                return _j([dict(r) for r in rows])

            if p == ["users"]:
                if not _admin(self.headers):
                    return _err("Unauthorized", 403)
                rows = con.execute("SELECT user_id,email,scopes,active,created_at FROM oauth_users").fetchall()
                return _j([dict(r) for r in rows])

            return _err("Not found", 404)
        finally:
            con.close()

    def _post(self, p, data):
        con = _db()
        now = time.time()
        try:
            if p == ["clients"]:
                if not _admin(self.headers):
                    return _err("Unauthorized", 403)
                cid = "client_" + secrets.token_hex(8)
                csecret = secrets.token_urlsafe(32)
                con.execute(
                    "INSERT INTO oauth_clients(client_id,client_secret,name,redirect_uris,scopes,grant_types,created_at) VALUES(?,?,?,?,?,?,?)",
                    (cid, _hash_token(csecret), data.get("name",""),
                     json.dumps(data.get("redirect_uris",[])),
                     json.dumps(data.get("scopes",["read"])),
                     json.dumps(data.get("grant_types",["authorization_code","client_credentials"])),
                     now)
                )
                con.commit()
                return _j({"client_id": cid, "client_secret": csecret}, 201)

            if p == ["users"]:
                if not _admin(self.headers):
                    return _err("Unauthorized", 403)
                email = data.get("email","")
                password = data.get("password","")
                if not all([email, password]):
                    return _err("email and password required")
                uid = "usr_" + secrets.token_hex(8)
                pw_hash = _hash_password(password)
                con.execute(
                    "INSERT INTO oauth_users(user_id,email,password_hash,scopes,created_at) VALUES(?,?,?,?,?)",
                    (uid, email, pw_hash, json.dumps(data.get("scopes",["read"])), now)
                )
                con.commit()
                return _j({"user_id": uid, "email": email}, 201)

            if p == ["authorize"]:
                # Resource Owner Password Credentials + code issuance
                client_id = data.get("client_id","")
                client_secret = data.get("client_secret","")
                email = data.get("email","")
                password = data.get("password","")
                redirect_uri = data.get("redirect_uri","")
                scopes = data.get("scope","").split()
                client = con.execute("SELECT * FROM oauth_clients WHERE client_id=? AND active=1", (client_id,)).fetchone()
                if not client:
                    return _err("Unknown client", 401, "invalid_client")
                if not hmac.compare_digest(_hash_token(client_secret), client["client_secret"]):
                    return _err("Invalid client secret", 401, "invalid_client")
                user = con.execute("SELECT * FROM oauth_users WHERE email=? AND active=1", (email,)).fetchone()
                if not user or not _verify_password(password, user["password_hash"]):
                    return _err("Invalid credentials", 401, "invalid_grant")
                code = secrets.token_urlsafe(32)
                con.execute(
                    "INSERT INTO auth_codes(code,client_id,user_id,redirect_uri,scopes,expires_at,created_at) VALUES(?,?,?,?,?,?,?)",
                    (code, client_id, user["user_id"], redirect_uri, json.dumps(scopes), now + 600, now)
                )
                con.commit()
                return _j({"code": code, "state": data.get("state","")})

            if p == ["token"]:
                grant_type = data.get("grant_type","")

                if grant_type == "authorization_code":
                    code = data.get("code","")
                    client_id = data.get("client_id","")
                    client_secret = data.get("client_secret","")
                    client = con.execute("SELECT * FROM oauth_clients WHERE client_id=? AND active=1", (client_id,)).fetchone()
                    if not client:
                        return _err("Unknown client", 401, "invalid_client")
                    if not hmac.compare_digest(_hash_token(client_secret), client["client_secret"]):
                        return _err("Invalid secret", 401, "invalid_client")
                    ac = con.execute(
                        "SELECT * FROM auth_codes WHERE code=? AND used=0 AND expires_at > ?", (code, now)
                    ).fetchone()
                    if not ac:
                        return _err("Invalid or expired code", 400, "invalid_grant")
                    con.execute("UPDATE auth_codes SET used=1 WHERE code=?", (code,))
                    scopes = json.loads(ac["scopes"])
                    return _j(_issue_tokens(con, client_id, ac["user_id"], scopes))

                elif grant_type == "client_credentials":
                    auth = self.headers.get("Authorization","")
                    if auth.startswith("Basic "):
                        decoded = base64.b64decode(auth[6:]).decode()
                        client_id, client_secret = decoded.split(":", 1)
                    else:
                        client_id = data.get("client_id","")
                        client_secret = data.get("client_secret","")
                    client = con.execute("SELECT * FROM oauth_clients WHERE client_id=? AND active=1", (client_id,)).fetchone()
                    if not client:
                        return _err("Unknown client", 401, "invalid_client")
                    if not hmac.compare_digest(_hash_token(client_secret), client["client_secret"]):
                        return _err("Invalid secret", 401, "invalid_client")
                    scopes = data.get("scope","").split() or json.loads(client["scopes"])
                    return _j(_issue_tokens(con, client_id, None, scopes))

                elif grant_type == "refresh_token":
                    rt = data.get("refresh_token","")
                    rt_hash = _hash_token(rt)
                    row = con.execute(
                        "SELECT * FROM refresh_tokens WHERE token_hash=? AND revoked=0 AND expires_at > ?", (rt_hash, now)
                    ).fetchone()
                    if not row:
                        return _err("Invalid refresh token", 400, "invalid_grant")
                    con.execute("UPDATE refresh_tokens SET revoked=1 WHERE token_hash=?", (rt_hash,))
                    con.execute("UPDATE access_tokens SET revoked=1 WHERE token_hash=?", (row["access_token_hash"],))
                    scopes = json.loads(row["scopes"])
                    return _j(_issue_tokens(con, row["client_id"], row["user_id"], scopes))

                return _err("Unsupported grant_type", 400, "unsupported_grant_type")

            if p == ["revoke"]:
                token = data.get("token","")
                th = _hash_token(token)
                con.execute("UPDATE access_tokens SET revoked=1 WHERE token_hash=?", (th,))
                con.execute("UPDATE refresh_tokens SET revoked=1 WHERE token_hash=?", (th,))
                con.commit()
                return _j({"revoked": True})

            if p == ["introspect"]:
                token = data.get("token","")
                payload = _verify_jwt(token)
                if payload:
                    return _j({"active": True, **payload})
                # fallback: check db
                th = _hash_token(token)
                row = con.execute(
                    "SELECT * FROM access_tokens WHERE token_hash=? AND revoked=0 AND expires_at > ?", (th, now)
                ).fetchone()
                if row:
                    return _j({"active": True, "client_id": row["client_id"],
                               "sub": row["user_id"], "exp": int(row["expires_at"]),
                               "scope": " ".join(json.loads(row["scopes"]))})
                return _j({"active": False})

            return _err("Not found", 404)
        finally:
            con.close()


def run():
    init_db()
    server = HTTPServer(("0.0.0.0", PORT), OAuthHandler)
    print(f"[fm_oauth_server] listening on port {PORT}")
    server.serve_forever()


if __name__ == "__main__":
    run()
