#!/usr/bin/env python3
"""
FractalMesh OMEGA Titan — API Marketplace / Developer Portal
Port: 7870

Developer API key management portal. Developers register, get API keys,
subscribe to plans, and consume the FractalMesh API. Tracks usage per key,
enforces rate limits, bills metered usage monthly.

Author : Samuel James Hiotis | ABN 56 628 117 363
System : FractalMesh SaaS Platform
"""

# ---------------------------------------------------------------------------
# Vault loading — MUST run before any os.getenv calls
# ---------------------------------------------------------------------------
import os
from pathlib import Path

_ENV_FILE = Path.home() / ".secrets" / "fractal.env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

# ---------------------------------------------------------------------------
# stdlib imports
# ---------------------------------------------------------------------------
import hashlib
import hmac
import http.server
import json
import re
import secrets
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
AGENT_NAME = "fm_api_marketplace"
PORT = int(os.environ.get("API_MARKETPLACE_PORT", "7870"))

SENDGRID_API_KEY   = os.environ.get("SENDGRID_API_KEY", "")
SENDGRID_FROM_EMAIL = os.environ.get("SENDGRID_FROM_EMAIL", "noreply@fractalmesh.io")
STRIPE_SECRET_KEY  = os.environ.get("STRIPE_SECRET_KEY", "")
ADMIN_SECRET       = os.environ.get("ADMIN_SECRET", "")

DB_PATH = Path.home() / "fmsaas" / "database" / "sovereign.db"

# ---------------------------------------------------------------------------
# In-memory rate limit state  {key_id: until_timestamp}
# ---------------------------------------------------------------------------
_rate_limited: dict = {}
_rl_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Server start time
# ---------------------------------------------------------------------------
_START_TIME = time.time()

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    conn = _get_conn()
    with conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS developers (
                id               INTEGER PRIMARY KEY,
                email            TEXT    UNIQUE NOT NULL,
                name             TEXT    NOT NULL,
                company          TEXT,
                plan             TEXT    DEFAULT 'free',
                status           TEXT    DEFAULT 'active',
                stripe_customer_id TEXT,
                created_at       REAL    NOT NULL,
                last_seen        REAL
            );

            CREATE TABLE IF NOT EXISTS api_keys (
                id           INTEGER PRIMARY KEY,
                key_id       TEXT    UNIQUE NOT NULL,
                key_hash     TEXT    NOT NULL,
                developer_id INTEGER NOT NULL,
                name         TEXT    NOT NULL,
                scopes       TEXT    DEFAULT '[]',
                status       TEXT    DEFAULT 'active',
                expires_at   REAL,
                last_used_at REAL,
                created_at   REAL    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS usage_logs (
                id          INTEGER PRIMARY KEY,
                key_id      TEXT    NOT NULL,
                endpoint    TEXT    NOT NULL,
                method      TEXT    NOT NULL,
                status_code INTEGER NOT NULL,
                latency_ms  REAL    NOT NULL,
                tokens_used INTEGER DEFAULT 0,
                timestamp   REAL    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS quotas (
                id                    INTEGER PRIMARY KEY,
                plan                  TEXT    UNIQUE NOT NULL,
                requests_per_minute   INTEGER NOT NULL,
                requests_per_day      INTEGER NOT NULL,
                requests_per_month    INTEGER NOT NULL,
                tokens_per_month      INTEGER NOT NULL,
                price_per_1k_requests REAL    DEFAULT 0,
                monthly_fee           REAL    DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS billing_records (
                id               INTEGER PRIMARY KEY,
                developer_id     INTEGER NOT NULL,
                period           TEXT    NOT NULL,
                requests_count   INTEGER NOT NULL,
                tokens_used      INTEGER NOT NULL,
                amount_due       REAL    NOT NULL,
                status           TEXT    DEFAULT 'pending',
                stripe_invoice_id TEXT,
                created_at       REAL    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_usage_key_ts
                ON usage_logs(key_id, timestamp);
            CREATE INDEX IF NOT EXISTS idx_api_keys_hash
                ON api_keys(key_hash);
        """)
    conn.close()


def _seed_quotas() -> None:
    plans = [
        ("free",       60,   1_000,       10_000,    100_000,    0.0,     0.0),
        ("starter",   120,  10_000,      100_000,  1_000_000,   0.001,  29.0),
        ("pro",       300,  50_000,      500_000, 10_000_000,   0.0005, 99.0),
        ("enterprise",1000, 999_999_999, 999_999_999, 999_999_999, 0.0002, 299.0),
    ]
    conn = _get_conn()
    with conn:
        for row in plans:
            conn.execute("""
                INSERT OR IGNORE INTO quotas
                  (plan, requests_per_minute, requests_per_day,
                   requests_per_month, tokens_per_month,
                   price_per_1k_requests, monthly_fee)
                VALUES (?,?,?,?,?,?,?)
            """, row)
    conn.close()

# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

def _generate_api_key(env: str = "live") -> tuple[str, str, str]:
    """Return (key_id, plaintext_key, key_hash)."""
    raw = secrets.token_hex(24)
    plaintext = f"fm_{env}_{raw}"
    key_id = f"kid_{secrets.token_hex(8)}"
    key_hash = hashlib.sha256(plaintext.encode()).hexdigest()
    return key_id, plaintext, key_hash


def _hash_key(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode()).hexdigest()


def _lookup_key(plaintext: str) -> sqlite3.Row | None:
    h = _hash_key(plaintext)
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM api_keys WHERE key_hash=? AND status='active'", (h,)
    ).fetchone()
    conn.close()
    return row


def _is_rate_limited(key_id: str) -> bool:
    with _rl_lock:
        until = _rate_limited.get(key_id, 0)
        return time.time() < until

# ---------------------------------------------------------------------------
# Email helpers (SendGrid via urllib)
# ---------------------------------------------------------------------------

def _send_email(to_email: str, subject: str, body_html: str) -> None:
    if not SENDGRID_API_KEY:
        return
    import urllib.request
    payload = json.dumps({
        "personalizations": [{"to": [{"email": to_email}]}],
        "from": {"email": SENDGRID_FROM_EMAIL},
        "subject": subject,
        "content": [{"type": "text/html", "value": body_html}],
    }).encode()
    req = urllib.request.Request(
        "https://api.sendgrid.com/v3/mail/send",
        data=payload,
        headers={
            "Authorization": f"Bearer {SENDGRID_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass


def _send_welcome_email(name: str, email: str, api_key: str, plan: str) -> None:
    _send_email(
        email,
        "Welcome to FractalMesh API — Your API Key",
        f"""
        <h2>Welcome to FractalMesh, {name}!</h2>
        <p>Your developer account has been created on the <strong>{plan}</strong> plan.</p>
        <p>Your API key (shown once — store it securely):</p>
        <pre style="background:#f4f4f4;padding:12px">{api_key}</pre>
        <p>Use the <code>X-API-Key</code> header in all requests.</p>
        <p>Docs: https://api.fractalmesh.io/docs</p>
        """,
    )

# ---------------------------------------------------------------------------
# Stripe helpers (urllib only)
# ---------------------------------------------------------------------------

def _stripe_post(path: str, params: dict) -> dict:
    if not STRIPE_SECRET_KEY:
        return {}
    import urllib.parse, urllib.request, urllib.error
    data = urllib.parse.urlencode(params).encode()
    req = urllib.request.Request(
        f"https://api.stripe.com/v1{path}",
        data=data,
        headers={
            "Authorization": f"Bearer {STRIPE_SECRET_KEY}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except Exception:
        return {}


def _create_stripe_customer(email: str, name: str) -> str:
    result = _stripe_post("/customers", {"email": email, "name": name})
    return result.get("id", "")


def _create_stripe_invoice(
    stripe_customer_id: str, amount_cents: int, description: str
) -> str:
    if not stripe_customer_id or amount_cents <= 0:
        return ""
    item = _stripe_post("/invoiceitems", {
        "customer": stripe_customer_id,
        "amount": amount_cents,
        "currency": "usd",
        "description": description,
    })
    if not item.get("id"):
        return ""
    invoice = _stripe_post("/invoices", {
        "customer": stripe_customer_id,
        "auto_advance": "true",
    })
    return invoice.get("id", "")

# ---------------------------------------------------------------------------
# Background: monthly billing (runs once per calendar month)
# ---------------------------------------------------------------------------

def _billing_thread() -> None:
    last_billed_month: str = ""
    while True:
        now = time.localtime()
        current_month = f"{now.tm_year}-{now.tm_mon:02d}"
        if current_month != last_billed_month:
            # Only bill after the 1st of the month has started
            if now.tm_mday >= 1 and last_billed_month != "":
                _run_monthly_billing(last_billed_month)
            last_billed_month = current_month
            # Purge old usage logs (>90 days)
            cutoff = time.time() - 90 * 86400
            try:
                conn = _get_conn()
                with conn:
                    conn.execute("DELETE FROM usage_logs WHERE timestamp < ?", (cutoff,))
                conn.close()
            except Exception:
                pass
        time.sleep(3600)  # check once per hour


def _run_monthly_billing(period: str) -> None:
    year, month = period.split("-")
    year, month = int(year), int(month)
    # period start/end in epoch
    import calendar
    period_start = time.mktime((year, month, 1, 0, 0, 0, 0, 0, -1))
    # last day of month
    last_day = calendar.monthrange(year, month)[1]
    period_end = time.mktime((year, month, last_day, 23, 59, 59, 0, 0, -1))

    conn = _get_conn()
    try:
        developers = conn.execute(
            "SELECT d.*, q.price_per_1k_requests, q.monthly_fee "
            "FROM developers d "
            "JOIN quotas q ON q.plan = d.plan "
            "WHERE d.plan != 'free' AND d.status = 'active'"
        ).fetchall()

        for dev in developers:
            dev_id = dev["id"]
            # Get all key_ids for this developer
            keys = conn.execute(
                "SELECT key_id FROM api_keys WHERE developer_id=?", (dev_id,)
            ).fetchall()
            key_ids = [k["key_id"] for k in keys]
            if not key_ids:
                continue

            placeholders = ",".join("?" * len(key_ids))
            row = conn.execute(
                f"SELECT COUNT(*) as cnt, COALESCE(SUM(tokens_used),0) as tok "
                f"FROM usage_logs "
                f"WHERE key_id IN ({placeholders}) "
                f"AND timestamp BETWEEN ? AND ?",
                (*key_ids, period_start, period_end),
            ).fetchone()
            req_count = row["cnt"] or 0
            tokens = row["tok"] or 0

            amount = dev["monthly_fee"] + (req_count / 1000.0) * dev["price_per_1k_requests"]
            amount = round(amount, 4)

            stripe_invoice_id = ""
            if STRIPE_SECRET_KEY and dev["stripe_customer_id"] and amount > 0:
                stripe_invoice_id = _create_stripe_invoice(
                    dev["stripe_customer_id"],
                    int(amount * 100),
                    f"FractalMesh API usage — {period}",
                )

            with conn:
                conn.execute(
                    "INSERT INTO billing_records "
                    "(developer_id, period, requests_count, tokens_used, "
                    " amount_due, status, stripe_invoice_id, created_at) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (
                        dev_id, period, req_count, tokens,
                        amount,
                        "invoiced" if stripe_invoice_id else "pending",
                        stripe_invoice_id,
                        time.time(),
                    ),
                )
    finally:
        conn.close()

# ---------------------------------------------------------------------------
# Background: per-minute quota reset check
# ---------------------------------------------------------------------------

def _quota_reset_thread() -> None:
    while True:
        time.sleep(60)
        now = time.time()
        window_start = now - 60

        conn = _get_conn()
        try:
            # Get active keys with their plan quotas
            rows = conn.execute("""
                SELECT ak.key_id, q.requests_per_minute
                FROM api_keys ak
                JOIN developers d ON d.id = ak.developer_id
                JOIN quotas q ON q.plan = d.plan
                WHERE ak.status = 'active'
            """).fetchall()

            for row in rows:
                key_id = row["key_id"]
                rpm_limit = row["requests_per_minute"]
                count = conn.execute(
                    "SELECT COUNT(*) FROM usage_logs "
                    "WHERE key_id=? AND timestamp > ?",
                    (key_id, window_start),
                ).fetchone()[0]
                with _rl_lock:
                    if count > rpm_limit:
                        # Rate-limit for the remainder of this minute + 10s buffer
                        _rate_limited[key_id] = now + 70
                    else:
                        # Clear expired rate limits
                        if _rate_limited.get(key_id, 0) < now:
                            _rate_limited.pop(key_id, None)
        finally:
            conn.close()

# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class MarketplaceHandler(BaseHTTPRequestHandler):
    """API Marketplace request handler."""

    server_version = "FractalMesh-API-Marketplace/1.0"
    sys_version = ""

    def log_message(self, fmt, *args):  # suppress default access log
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def _send(self, code: int, data: dict) -> None:
        body = json.dumps(data, indent=2).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _ok(self, data: dict) -> None:
        self._send(200, data)

    def _created(self, data: dict) -> None:
        self._send(201, data)

    def _err(self, code: int, msg: str) -> None:
        self._send(code, {"error": msg})

    def _require_admin(self) -> bool:
        header = self.headers.get("X-Admin-Secret", "")
        if not ADMIN_SECRET or not hmac.compare_digest(header, ADMIN_SECRET):
            self._err(403, "admin access denied")
            return False
        return True

    def _require_auth(self) -> sqlite3.Row | None:
        """Validate X-API-Key header; return api_keys row or None."""
        key_plain = self.headers.get("X-API-Key", "").strip()
        if not key_plain:
            self._err(401, "missing X-API-Key header")
            return None
        row = _lookup_key(key_plain)
        if row is None:
            self._err(401, "invalid or revoked API key")
            return None
        if row["expires_at"] and time.time() > row["expires_at"]:
            self._err(401, "API key has expired")
            return None
        if _is_rate_limited(row["key_id"]):
            self._err(429, "rate limit exceeded")
            return None
        # Update last_used_at
        try:
            conn = _get_conn()
            with conn:
                conn.execute(
                    "UPDATE api_keys SET last_used_at=? WHERE key_id=?",
                    (time.time(), row["key_id"]),
                )
                conn.execute(
                    "UPDATE developers SET last_seen=? WHERE id=?",
                    (time.time(), row["developer_id"]),
                )
            conn.close()
        except Exception:
            pass
        return row

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")

        if path == "/health":
            self._handle_health()
        elif path == "/plans":
            self._handle_plans()
        elif path == "/keys":
            self._handle_list_keys()
        elif path == "/usage":
            self._handle_usage()
        elif path == "/billing":
            self._handle_billing()
        elif path == "/developers":
            self._handle_list_developers()
        elif re.match(r"^/developers/(\d+)$", path):
            m = re.match(r"^/developers/(\d+)$", path)
            self._handle_get_developer(int(m.group(1)))
        elif path == "/analytics":
            self._handle_analytics()
        else:
            self._err(404, "endpoint not found")

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")

        if path == "/register":
            self._handle_register()
        elif path == "/keys":
            self._handle_create_key()
        elif path == "/validate":
            self._handle_validate()
        else:
            self._err(404, "endpoint not found")

    def do_DELETE(self):
        path = self.path.split("?")[0].rstrip("/")
        m = re.match(r"^/keys/([a-zA-Z0-9_]+)$", path)
        if m:
            self._handle_revoke_key(m.group(1))
        else:
            self._err(404, "endpoint not found")

    def do_PUT(self):
        path = self.path.split("?")[0].rstrip("/")
        m = re.match(r"^/developers/(\d+)/plan$", path)
        if m:
            self._handle_update_plan(int(m.group(1)))
        else:
            self._err(404, "endpoint not found")

    # ------------------------------------------------------------------
    # GET /health
    # ------------------------------------------------------------------

    def _handle_health(self):
        conn = _get_conn()
        dev_count = conn.execute("SELECT COUNT(*) FROM developers").fetchone()[0]
        day_start = time.time() - 86400
        calls_today = conn.execute(
            "SELECT COUNT(*) FROM usage_logs WHERE timestamp > ?", (day_start,)
        ).fetchone()[0]
        conn.close()
        self._ok({
            "status": "ok",
            "agent": AGENT_NAME,
            "port": PORT,
            "uptime_seconds": round(time.time() - _START_TIME, 1),
            "developer_count": dev_count,
            "api_calls_today": calls_today,
        })

    # ------------------------------------------------------------------
    # GET /plans
    # ------------------------------------------------------------------

    def _handle_plans(self):
        conn = _get_conn()
        rows = conn.execute("SELECT * FROM quotas ORDER BY monthly_fee ASC").fetchall()
        conn.close()
        plans = []
        for r in rows:
            plans.append({
                "plan": r["plan"],
                "requests_per_minute": r["requests_per_minute"],
                "requests_per_day": r["requests_per_day"],
                "requests_per_month": r["requests_per_month"],
                "tokens_per_month": r["tokens_per_month"],
                "price_per_1k_requests": r["price_per_1k_requests"],
                "monthly_fee": r["monthly_fee"],
            })
        self._ok({"plans": plans})

    # ------------------------------------------------------------------
    # POST /register
    # ------------------------------------------------------------------

    def _handle_register(self):
        body = self._body()
        email   = (body.get("email") or "").strip().lower()
        name    = (body.get("name") or "").strip()
        company = (body.get("company") or "").strip()
        plan    = (body.get("plan") or "free").strip().lower()

        if not email or not name:
            self._err(400, "email and name are required")
            return
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
            self._err(400, "invalid email address")
            return

        # Validate plan
        conn = _get_conn()
        quota = conn.execute(
            "SELECT * FROM quotas WHERE plan=?", (plan,)
        ).fetchone()
        if quota is None:
            conn.close()
            self._err(400, f"unknown plan '{plan}'")
            return

        # Check duplicate email
        existing = conn.execute(
            "SELECT id FROM developers WHERE email=?", (email,)
        ).fetchone()
        if existing:
            conn.close()
            self._err(409, "email already registered")
            return

        now = time.time()
        try:
            with conn:
                conn.execute(
                    "INSERT INTO developers (email, name, company, plan, status, created_at, last_seen) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (email, name, company, plan, "active", now, now),
                )
            dev_row = conn.execute(
                "SELECT id FROM developers WHERE email=?", (email,)
            ).fetchone()
            dev_id = dev_row["id"]

            # Create Stripe customer if key available
            stripe_id = ""
            if STRIPE_SECRET_KEY:
                stripe_id = _create_stripe_customer(email, name)
                if stripe_id:
                    with conn:
                        conn.execute(
                            "UPDATE developers SET stripe_customer_id=? WHERE id=?",
                            (stripe_id, dev_id),
                        )

            # Generate first API key
            env = "live" if STRIPE_SECRET_KEY else "test"
            key_id, plaintext, key_hash = _generate_api_key(env)
            with conn:
                conn.execute(
                    "INSERT INTO api_keys "
                    "(key_id, key_hash, developer_id, name, scopes, status, created_at) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (key_id, key_hash, dev_id, "default", '["*"]', "active", now),
                )
        except Exception as exc:
            conn.close()
            self._err(500, f"registration failed: {exc}")
            return
        conn.close()

        # Send welcome email asynchronously
        threading.Thread(
            target=_send_welcome_email,
            args=(name, email, plaintext, plan),
            daemon=True,
        ).start()

        self._created({
            "developer_id": dev_id,
            "api_key": plaintext,
            "key_id": key_id,
            "plan": plan,
            "message": "API key shown once — store it securely.",
        })

    # ------------------------------------------------------------------
    # POST /keys  (auth required)
    # ------------------------------------------------------------------

    def _handle_create_key(self):
        key_row = self._require_auth()
        if key_row is None:
            return
        body = self._body()
        dev_id  = key_row["developer_id"]
        name    = (body.get("name") or "").strip()
        scopes  = body.get("scopes", ["*"])

        if not name:
            self._err(400, "key name is required")
            return
        if not isinstance(scopes, list):
            self._err(400, "scopes must be a list")
            return

        env = "live" if STRIPE_SECRET_KEY else "test"
        kid, plaintext, key_hash = _generate_api_key(env)
        now = time.time()

        conn = _get_conn()
        try:
            with conn:
                conn.execute(
                    "INSERT INTO api_keys "
                    "(key_id, key_hash, developer_id, name, scopes, status, created_at) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (kid, key_hash, dev_id, name, json.dumps(scopes), "active", now),
                )
        except Exception as exc:
            conn.close()
            self._err(500, f"key creation failed: {exc}")
            return
        conn.close()

        self._created({
            "key_id": kid,
            "api_key": plaintext,
            "name": name,
            "scopes": scopes,
            "message": "API key shown once — store it securely.",
        })

    # ------------------------------------------------------------------
    # DELETE /keys/{key_id}  (auth required)
    # ------------------------------------------------------------------

    def _handle_revoke_key(self, key_id: str):
        key_row = self._require_auth()
        if key_row is None:
            return
        dev_id = key_row["developer_id"]

        conn = _get_conn()
        target = conn.execute(
            "SELECT * FROM api_keys WHERE key_id=?", (key_id,)
        ).fetchone()
        if target is None:
            conn.close()
            self._err(404, "key not found")
            return
        if target["developer_id"] != dev_id:
            conn.close()
            self._err(403, "key does not belong to your account")
            return
        if target["key_id"] == key_row["key_id"]:
            conn.close()
            self._err(400, "cannot revoke the key used for this request")
            return
        with conn:
            conn.execute(
                "UPDATE api_keys SET status='revoked' WHERE key_id=?", (key_id,)
            )
        conn.close()
        self._ok({"revoked": key_id})

    # ------------------------------------------------------------------
    # GET /keys  (auth required)
    # ------------------------------------------------------------------

    def _handle_list_keys(self):
        key_row = self._require_auth()
        if key_row is None:
            return
        dev_id = key_row["developer_id"]

        conn = _get_conn()
        rows = conn.execute(
            "SELECT key_id, name, scopes, status, expires_at, last_used_at, created_at "
            "FROM api_keys WHERE developer_id=? ORDER BY created_at DESC",
            (dev_id,),
        ).fetchall()
        conn.close()
        keys = []
        for r in rows:
            keys.append({
                "key_id": r["key_id"],
                "name": r["name"],
                "scopes": json.loads(r["scopes"] or "[]"),
                "status": r["status"],
                "expires_at": r["expires_at"],
                "last_used_at": r["last_used_at"],
                "created_at": r["created_at"],
            })
        self._ok({"keys": keys})

    # ------------------------------------------------------------------
    # GET /usage  (auth required)
    # ------------------------------------------------------------------

    def _handle_usage(self):
        key_row = self._require_auth()
        if key_row is None:
            return
        dev_id = key_row["developer_id"]

        conn = _get_conn()
        # Gather all key_ids for this developer
        key_rows = conn.execute(
            "SELECT key_id FROM api_keys WHERE developer_id=?", (dev_id,)
        ).fetchall()
        key_ids = [r["key_id"] for r in key_rows]

        now = time.time()
        day_start = now - 86400
        # Month start (approx — first of current month)
        lt = time.localtime(now)
        month_start = time.mktime((lt.tm_year, lt.tm_mon, 1, 0, 0, 0, 0, 0, -1))

        result: dict = {
            "requests_today": 0,
            "requests_this_month": 0,
            "tokens_today": 0,
            "tokens_this_month": 0,
            "top_endpoints": [],
        }

        if key_ids:
            ph = ",".join("?" * len(key_ids))
            row_day = conn.execute(
                f"SELECT COUNT(*) as cnt, COALESCE(SUM(tokens_used),0) as tok "
                f"FROM usage_logs WHERE key_id IN ({ph}) AND timestamp > ?",
                (*key_ids, day_start),
            ).fetchone()
            row_month = conn.execute(
                f"SELECT COUNT(*) as cnt, COALESCE(SUM(tokens_used),0) as tok "
                f"FROM usage_logs WHERE key_id IN ({ph}) AND timestamp > ?",
                (*key_ids, month_start),
            ).fetchone()
            top_ep = conn.execute(
                f"SELECT endpoint, COUNT(*) as cnt "
                f"FROM usage_logs WHERE key_id IN ({ph}) AND timestamp > ? "
                f"GROUP BY endpoint ORDER BY cnt DESC LIMIT 10",
                (*key_ids, month_start),
            ).fetchall()
            result["requests_today"]       = row_day["cnt"]
            result["tokens_today"]         = row_day["tok"]
            result["requests_this_month"]  = row_month["cnt"]
            result["tokens_this_month"]    = row_month["tok"]
            result["top_endpoints"] = [
                {"endpoint": ep["endpoint"], "count": ep["cnt"]}
                for ep in top_ep
            ]
        conn.close()
        self._ok(result)

    # ------------------------------------------------------------------
    # GET /billing  (auth required)
    # ------------------------------------------------------------------

    def _handle_billing(self):
        key_row = self._require_auth()
        if key_row is None:
            return
        dev_id = key_row["developer_id"]

        conn = _get_conn()
        rows = conn.execute(
            "SELECT period, requests_count, tokens_used, amount_due, status, "
            "stripe_invoice_id, created_at "
            "FROM billing_records WHERE developer_id=? ORDER BY created_at DESC",
            (dev_id,),
        ).fetchall()
        conn.close()
        records = [dict(r) for r in rows]
        self._ok({"billing_records": records})

    # ------------------------------------------------------------------
    # POST /validate
    # ------------------------------------------------------------------

    def _handle_validate(self):
        body = self._body()
        api_key = (body.get("api_key") or "").strip()
        if not api_key:
            self._err(400, "api_key is required")
            return

        row = _lookup_key(api_key)
        if row is None:
            self._ok({"valid": False})
            return

        expired = bool(row["expires_at"] and time.time() > row["expires_at"])
        if expired:
            self._ok({"valid": False, "reason": "expired"})
            return

        rl = _is_rate_limited(row["key_id"])

        conn = _get_conn()
        dev = conn.execute(
            "SELECT d.*, q.* FROM developers d "
            "JOIN quotas q ON q.plan = d.plan "
            "WHERE d.id=?",
            (row["developer_id"],),
        ).fetchone()
        conn.close()

        if dev is None or dev["status"] != "active":
            self._ok({"valid": False, "reason": "developer inactive"})
            return

        quota = {
            "requests_per_minute": dev["requests_per_minute"],
            "requests_per_day": dev["requests_per_day"],
            "requests_per_month": dev["requests_per_month"],
            "tokens_per_month": dev["tokens_per_month"],
        }

        self._ok({
            "valid": True,
            "developer_id": row["developer_id"],
            "key_id": row["key_id"],
            "plan": dev["plan"],
            "scopes": json.loads(row["scopes"] or "[]"),
            "rate_limited": rl,
            "quota": quota,
        })

    # ------------------------------------------------------------------
    # GET /developers  (admin)
    # ------------------------------------------------------------------

    def _handle_list_developers(self):
        if not self._require_admin():
            return
        conn = _get_conn()
        rows = conn.execute(
            "SELECT id, email, name, company, plan, status, created_at, last_seen "
            "FROM developers ORDER BY created_at DESC"
        ).fetchall()
        conn.close()
        self._ok({"developers": [dict(r) for r in rows]})

    # ------------------------------------------------------------------
    # GET /developers/{id}  (admin)
    # ------------------------------------------------------------------

    def _handle_get_developer(self, dev_id: int):
        if not self._require_admin():
            return
        conn = _get_conn()
        dev = conn.execute(
            "SELECT * FROM developers WHERE id=?", (dev_id,)
        ).fetchone()
        if dev is None:
            conn.close()
            self._err(404, "developer not found")
            return
        keys = conn.execute(
            "SELECT key_id, name, scopes, status, expires_at, last_used_at, created_at "
            "FROM api_keys WHERE developer_id=?",
            (dev_id,),
        ).fetchall()

        now = time.time()
        lt = time.localtime(now)
        month_start = time.mktime((lt.tm_year, lt.tm_mon, 1, 0, 0, 0, 0, 0, -1))
        key_ids = [k["key_id"] for k in keys]
        month_reqs = 0
        if key_ids:
            ph = ",".join("?" * len(key_ids))
            month_reqs = conn.execute(
                f"SELECT COUNT(*) FROM usage_logs WHERE key_id IN ({ph}) AND timestamp > ?",
                (*key_ids, month_start),
            ).fetchone()[0]
        conn.close()

        self._ok({
            "developer": dict(dev),
            "api_keys": [
                {
                    "key_id": k["key_id"],
                    "name": k["name"],
                    "scopes": json.loads(k["scopes"] or "[]"),
                    "status": k["status"],
                    "expires_at": k["expires_at"],
                    "last_used_at": k["last_used_at"],
                    "created_at": k["created_at"],
                }
                for k in keys
            ],
            "requests_this_month": month_reqs,
        })

    # ------------------------------------------------------------------
    # PUT /developers/{id}/plan  (admin)
    # ------------------------------------------------------------------

    def _handle_update_plan(self, dev_id: int):
        if not self._require_admin():
            return
        body = self._body()
        new_plan = (body.get("plan") or "").strip().lower()
        if not new_plan:
            self._err(400, "plan is required")
            return

        conn = _get_conn()
        quota = conn.execute(
            "SELECT plan FROM quotas WHERE plan=?", (new_plan,)
        ).fetchone()
        if quota is None:
            conn.close()
            self._err(400, f"unknown plan '{new_plan}'")
            return
        dev = conn.execute(
            "SELECT id FROM developers WHERE id=?", (dev_id,)
        ).fetchone()
        if dev is None:
            conn.close()
            self._err(404, "developer not found")
            return
        with conn:
            conn.execute(
                "UPDATE developers SET plan=? WHERE id=?", (new_plan, dev_id)
            )
        conn.close()
        self._ok({"developer_id": dev_id, "plan": new_plan})

    # ------------------------------------------------------------------
    # GET /analytics  (admin)
    # ------------------------------------------------------------------

    def _handle_analytics(self):
        if not self._require_admin():
            return
        conn = _get_conn()

        now = time.time()
        lt = time.localtime(now)
        month_start = time.mktime((lt.tm_year, lt.tm_mon, 1, 0, 0, 0, 0, 0, -1))

        # Total revenue billed
        total_revenue = conn.execute(
            "SELECT COALESCE(SUM(amount_due),0) FROM billing_records"
        ).fetchone()[0]

        # Requests by plan this month
        rows_by_plan = conn.execute("""
            SELECT d.plan, COUNT(ul.id) as req_count
            FROM usage_logs ul
            JOIN api_keys ak ON ak.key_id = ul.key_id
            JOIN developers d ON d.id = ak.developer_id
            WHERE ul.timestamp > ?
            GROUP BY d.plan
            ORDER BY req_count DESC
        """, (month_start,)).fetchall()

        # Top 10 developers by requests this month
        top_devs = conn.execute("""
            SELECT d.id, d.email, d.name, d.plan, COUNT(ul.id) as req_count
            FROM usage_logs ul
            JOIN api_keys ak ON ak.key_id = ul.key_id
            JOIN developers d ON d.id = ak.developer_id
            WHERE ul.timestamp > ?
            GROUP BY d.id
            ORDER BY req_count DESC
            LIMIT 10
        """, (month_start,)).fetchall()

        # Developer counts by plan
        plan_dist = conn.execute(
            "SELECT plan, COUNT(*) as cnt FROM developers GROUP BY plan"
        ).fetchall()

        conn.close()

        self._ok({
            "total_revenue_billed": round(total_revenue, 2),
            "requests_by_plan_this_month": [
                {"plan": r["plan"], "requests": r["req_count"]}
                for r in rows_by_plan
            ],
            "top_developers_this_month": [
                {
                    "developer_id": r["id"],
                    "email": r["email"],
                    "name": r["name"],
                    "plan": r["plan"],
                    "requests": r["req_count"],
                }
                for r in top_devs
            ],
            "developer_distribution": [
                {"plan": r["plan"], "count": r["cnt"]}
                for r in plan_dist
            ],
        })


# ---------------------------------------------------------------------------
# Server bootstrap
# ---------------------------------------------------------------------------

def _start_background_threads() -> None:
    threading.Thread(target=_billing_thread, daemon=True, name="billing").start()
    threading.Thread(target=_quota_reset_thread, daemon=True, name="quota_reset").start()


def main() -> None:
    _init_db()
    _seed_quotas()
    _start_background_threads()

    server = HTTPServer(("0.0.0.0", PORT), MarketplaceHandler)
    print(f"[{AGENT_NAME}] listening on port {PORT}  db={DB_PATH}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"[{AGENT_NAME}] shutting down")
        server.server_close()


if __name__ == "__main__":
    main()
