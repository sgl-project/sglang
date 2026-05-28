#!/usr/bin/env python3
"""
FractalMesh OMEGA Titan — Stripe Payment Gateway
Port: 7854

Full-featured Stripe payment processing: checkout sessions, payment intents,
subscriptions, webhooks, and revenue tracking.

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
import base64
import hashlib
import hmac
import json
import signal
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
AGENT_NAME = "fm_stripe_gateway"
PORT = int(os.environ.get("STRIPE_GATEWAY_PORT", "7854"))

STRIPE_SECRET_KEY     = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PUBLISHABLE_KEY = os.environ.get("STRIPE_PUBLISHABLE_KEY", "")
ADMIN_SECRET          = os.environ.get("ADMIN_SECRET", "")

STRIPE_API_BASE = "https://api.stripe.com/v1"

ROOT    = Path(os.environ.get("FRACTALMESH_HOME", Path.home() / "fmsaas"))
DB_PATH = ROOT / "database" / "sovereign.db"

START_TIME = time.time()

# ---------------------------------------------------------------------------
# Database bootstrap
# ---------------------------------------------------------------------------

def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = _db()
    con.executescript("""
        CREATE TABLE IF NOT EXISTS payments (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            stripe_payment_id TEXT UNIQUE NOT NULL,
            amount            INTEGER NOT NULL,
            currency          TEXT    NOT NULL DEFAULT 'usd',
            status            TEXT    NOT NULL,
            customer_email    TEXT    NOT NULL DEFAULT '',
            description       TEXT    NOT NULL DEFAULT '',
            metadata          TEXT    NOT NULL DEFAULT '{}',
            created_at        REAL    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS subscriptions (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            stripe_sub_id       TEXT UNIQUE NOT NULL,
            customer_id         TEXT NOT NULL DEFAULT '',
            customer_email      TEXT NOT NULL DEFAULT '',
            plan                TEXT NOT NULL DEFAULT '',
            status              TEXT NOT NULL,
            current_period_end  REAL NOT NULL DEFAULT 0,
            created_at          REAL NOT NULL,
            updated_at          REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS webhook_events (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            stripe_event_id TEXT UNIQUE NOT NULL,
            event_type     TEXT NOT NULL,
            processed      INTEGER NOT NULL DEFAULT 0,
            payload        TEXT NOT NULL DEFAULT '{}',
            created_at     REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS customers (
            id                TEXT PRIMARY KEY,
            stripe_customer_id TEXT UNIQUE NOT NULL,
            email             TEXT NOT NULL DEFAULT '',
            name              TEXT NOT NULL DEFAULT '',
            metadata          TEXT NOT NULL DEFAULT '{}',
            created_at        REAL NOT NULL
        );
    """)
    # Replace the buggy primary key for customers (TEXT PK should be INTEGER)
    # Use a separate migration-safe approach:
    con.executescript("""
        CREATE TABLE IF NOT EXISTS customers2 (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            stripe_customer_id TEXT UNIQUE NOT NULL,
            email             TEXT NOT NULL DEFAULT '',
            name              TEXT NOT NULL DEFAULT '',
            metadata          TEXT NOT NULL DEFAULT '{}',
            created_at        REAL NOT NULL
        );
    """)
    # Check if the real customers table has INTEGER PK; if not, migrate
    cur = con.execute("PRAGMA table_info(customers)")
    cols = {row[1]: row[2] for row in cur.fetchall()}
    if cols.get("id") == "TEXT":
        con.execute("INSERT OR IGNORE INTO customers2 SELECT id, stripe_customer_id, email, name, metadata, created_at FROM customers")
        con.execute("DROP TABLE customers")
        con.execute("ALTER TABLE customers2 RENAME TO customers")
    else:
        con.execute("DROP TABLE customers2")

    con.commit()
    con.close()
    print(f"[{AGENT_NAME}] DB initialised at {DB_PATH}", flush=True)


def _db() -> sqlite3.Connection:
    con = sqlite3.connect(str(DB_PATH), timeout=15)
    con.execute("PRAGMA journal_mode=WAL")
    con.row_factory = sqlite3.Row
    return con

# ---------------------------------------------------------------------------
# Stripe API helpers
# ---------------------------------------------------------------------------

def _stripe_auth_header() -> str:
    """Return Basic auth header value for STRIPE_SECRET_KEY."""
    creds = base64.b64encode(f"{STRIPE_SECRET_KEY}:".encode()).decode()
    return f"Basic {creds}"


def _stripe_get(path: str, params: dict | None = None) -> dict:
    """GET from Stripe API.  Raises RuntimeError on HTTP error."""
    url = STRIPE_API_BASE + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "Authorization": _stripe_auth_header(),
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"Stripe GET {path} → {exc.code}: {body}") from exc


def _stripe_post(path: str, data: dict) -> dict:
    """POST application/x-www-form-urlencoded to Stripe API."""
    url = STRIPE_API_BASE + path
    encoded = _flatten_form(data).encode()
    req = urllib.request.Request(
        url,
        data=encoded,
        headers={
            "Authorization": _stripe_auth_header(),
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"Stripe POST {path} → {exc.code}: {body}") from exc


def _stripe_delete(path: str) -> dict:
    """DELETE on Stripe API (e.g. cancel subscription)."""
    url = STRIPE_API_BASE + path
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": _stripe_auth_header(),
            "Accept": "application/json",
        },
        method="DELETE",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"Stripe DELETE {path} → {exc.code}: {body}") from exc


def _flatten_form(data: dict, prefix: str = "") -> str:
    """
    Recursively flatten a dict to Stripe's form-encoding style.
    E.g. {"metadata": {"key": "val"}} → "metadata[key]=val"
    """
    parts: list[tuple[str, str]] = []
    for k, v in data.items():
        full_key = f"{prefix}[{k}]" if prefix else k
        if isinstance(v, dict):
            parts.append((_flatten_form(v, full_key), ""))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    parts.append((_flatten_form(item, f"{full_key}[{i}]"), ""))
                else:
                    parts.append((f"{full_key}[{i}]", str(item)))
        elif v is None:
            pass  # skip None values
        else:
            parts.append((full_key, str(v)))
    # Merge recursive results
    merged: list[tuple[str, str]] = []
    for key, val in parts:
        if val == "":
            # This was a recursive result already encoded as a string
            merged.append((key, ""))
        else:
            merged.append((key, val))
    # Filter out blanks that came from recursive calls
    result_pairs: list[tuple[str, str]] = []
    for key, val in parts:
        if val == "" and "[" in key:
            result_pairs.append((key, ""))
        elif val != "":
            result_pairs.append((key, val))
    return urllib.parse.urlencode(result_pairs)


def _flat_encode(data: dict) -> str:
    """Simplified flat encoder for Stripe forms (no nested dicts)."""
    return urllib.parse.urlencode({k: v for k, v in data.items() if v is not None})


def _stripe_form_encode(data: dict, prefix: str = "") -> list[tuple[str, str]]:
    """Return list of (key, value) pairs with Stripe bracket notation for nested dicts."""
    pairs: list[tuple[str, str]] = []
    for k, v in data.items():
        full_key = f"{prefix}[{k}]" if prefix else k
        if isinstance(v, dict):
            pairs.extend(_stripe_form_encode(v, full_key))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    pairs.extend(_stripe_form_encode(item, f"{full_key}[{i}]"))
                else:
                    pairs.append((f"{full_key}[{i}]", str(item)))
        elif v is not None:
            pairs.append((full_key, str(v)))
    return pairs


def stripe_encode(data: dict) -> bytes:
    """Encode a potentially nested dict for Stripe's API (application/x-www-form-urlencoded)."""
    pairs = _stripe_form_encode(data)
    return urllib.parse.urlencode(pairs).encode()


def _stripe_post_encoded(path: str, data: dict) -> dict:
    """POST with proper nested bracket encoding to Stripe API."""
    url = STRIPE_API_BASE + path
    encoded = stripe_encode(data)
    req = urllib.request.Request(
        url,
        data=encoded,
        headers={
            "Authorization": _stripe_auth_header(),
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"Stripe POST {path} → {exc.code}: {body}") from exc

# ---------------------------------------------------------------------------
# Webhook signature verification
# ---------------------------------------------------------------------------

def _verify_stripe_signature(payload: bytes, sig_header: str) -> bool:
    """
    Verify Stripe-Signature header using HMAC-SHA256.
    Header format: t=<timestamp>,v1=<signature>[,v1=<additional>]
    """
    if not STRIPE_WEBHOOK_SECRET:
        return False
    try:
        parts = {item.split("=", 1)[0]: item.split("=", 1)[1]
                 for item in sig_header.split(",") if "=" in item}
        timestamp = parts.get("t", "")
        v1_sig    = parts.get("v1", "")
        if not timestamp or not v1_sig:
            return False
        signed_payload = f"{timestamp}.".encode() + payload
        expected = hmac.new(
            STRIPE_WEBHOOK_SECRET.encode(),
            signed_payload,
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, v1_sig)
    except Exception:
        return False

# ---------------------------------------------------------------------------
# Background sync thread
# ---------------------------------------------------------------------------

def _sync_pending_payments() -> None:
    """
    Every 60 s: find payments with status='requires_confirmation' older than 5 minutes,
    query Stripe, and update their status in the DB.
    """
    while True:
        time.sleep(60)
        if not STRIPE_SECRET_KEY:
            continue
        cutoff = time.time() - 300  # 5 minutes
        try:
            con = _db()
            rows = con.execute(
                "SELECT stripe_payment_id FROM payments "
                "WHERE status='requires_confirmation' AND created_at < ?",
                (cutoff,),
            ).fetchall()
            con.close()

            for row in rows:
                pi_id = row["stripe_payment_id"]
                try:
                    pi = _stripe_get(f"/payment_intents/{pi_id}")
                    new_status = pi.get("status", "unknown")
                    con2 = _db()
                    con2.execute(
                        "UPDATE payments SET status=? WHERE stripe_payment_id=?",
                        (new_status, pi_id),
                    )
                    con2.commit()
                    con2.close()
                    print(f"[{AGENT_NAME}] sync: {pi_id} → {new_status}", flush=True)
                except Exception as exc:
                    print(f"[{AGENT_NAME}] sync error for {pi_id}: {exc}", flush=True)
        except Exception as exc:
            print(f"[{AGENT_NAME}] sync thread error: {exc}", flush=True)


def start_sync_thread() -> None:
    t = threading.Thread(target=_sync_pending_payments, daemon=True, name="stripe-sync")
    t.start()
    print(f"[{AGENT_NAME}] Background sync thread started.", flush=True)

# ---------------------------------------------------------------------------
# Revenue helpers
# ---------------------------------------------------------------------------

def _revenue_stats() -> dict:
    """Compute daily and monthly revenue totals from the payments table."""
    now = time.time()
    day_start   = now - 86400
    month_start = now - 86400 * 30

    con = _db()
    day_row = con.execute(
        "SELECT COALESCE(SUM(amount),0) FROM payments "
        "WHERE status='succeeded' AND created_at >= ?",
        (day_start,),
    ).fetchone()
    month_row = con.execute(
        "SELECT COALESCE(SUM(amount),0) FROM payments "
        "WHERE status='succeeded' AND created_at >= ?",
        (month_start,),
    ).fetchone()
    by_currency = con.execute(
        "SELECT currency, COALESCE(SUM(amount),0) AS total FROM payments "
        "WHERE status='succeeded' AND created_at >= ? GROUP BY currency",
        (month_start,),
    ).fetchall()
    con.close()

    return {
        "daily_revenue_cents": day_row[0],
        "monthly_revenue_cents": month_row[0],
        "monthly_by_currency": {r["currency"]: r["total"] for r in by_currency},
    }

# ---------------------------------------------------------------------------
# Customer helpers
# ---------------------------------------------------------------------------

def _find_or_create_customer(email: str, name: str = "") -> str:
    """Return existing Stripe customer_id for email, or create a new one."""
    con = _db()
    row = con.execute(
        "SELECT stripe_customer_id FROM customers WHERE email=? LIMIT 1",
        (email,),
    ).fetchone()
    con.close()

    if row:
        return row["stripe_customer_id"]

    # Create via Stripe API
    payload: dict = {"email": email}
    if name:
        payload["name"] = name
    customer = _stripe_post_encoded("/customers", payload)
    cust_id = customer["id"]

    con2 = _db()
    con2.execute(
        "INSERT OR IGNORE INTO customers (stripe_customer_id, email, name, metadata, created_at) "
        "VALUES (?,?,?,?,?)",
        (cust_id, email, name, json.dumps(customer.get("metadata") or {}), time.time()),
    )
    con2.commit()
    con2.close()
    return cust_id

# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------

class StripeGatewayHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        print(f"[{AGENT_NAME}] {self.address_string()} {fmt % args}", flush=True)

    # ---- Utilities ---------------------------------------------------------

    def _send(self, code: int, payload) -> None:
        if isinstance(payload, (dict, list)):
            body = json.dumps(payload, indent=2).encode()
            ct = "application/json"
        else:
            body = str(payload).encode()
            ct = "text/plain"
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length else b""

    def _read_json(self) -> dict:
        raw = self._read_body()
        return json.loads(raw) if raw else {}

    def _qs(self) -> dict:
        parsed = urllib.parse.urlparse(self.path)
        return dict(urllib.parse.parse_qsl(parsed.query))

    def _clean_path(self) -> str:
        return urllib.parse.urlparse(self.path).path.rstrip("/")

    def _require_admin(self) -> bool:
        """Return True if admin auth passes; otherwise send 403 and return False."""
        secret = self.headers.get("X-Admin-Secret", "")
        if not ADMIN_SECRET or not hmac.compare_digest(secret, ADMIN_SECRET):
            self._send(403, {"error": "forbidden: X-Admin-Secret required"})
            return False
        return True

    def _stripe_error(self, exc: Exception) -> None:
        """Send a 402 with the Stripe error message."""
        msg = str(exc)
        # Try to parse JSON from the error body
        try:
            # exc message format: "Stripe POST /x → 402: {json}"
            json_start = msg.find("{")
            if json_start != -1:
                err_obj = json.loads(msg[json_start:])
                stripe_msg = err_obj.get("error", {}).get("message", msg)
                self._send(402, {"error": stripe_msg, "stripe_error": err_obj.get("error", {})})
                return
        except Exception:
            pass
        self._send(402, {"error": msg})

    # ---- GET ---------------------------------------------------------------

    def do_GET(self):
        path = self._clean_path()
        qs   = self._qs()
        try:
            if path == "/health":
                self._handle_health()
            elif path == "/payments":
                self._handle_list_payments(qs)
            elif path.startswith("/payments/"):
                pi_id = path[len("/payments/"):]
                self._handle_get_payment(pi_id)
            elif path == "/subscriptions":
                self._handle_list_subscriptions(qs)
            elif path == "/customers":
                self._handle_list_customers(qs)
            elif path == "/revenue":
                self._handle_revenue()
            elif path == "/webhook_events":
                self._handle_list_webhook_events()
            else:
                self._send(404, {"error": "not found", "path": path})
        except Exception as exc:
            print(f"[{AGENT_NAME}] ERROR GET {path}: {exc}", flush=True)
            self._send(500, {"error": str(exc)})

    # ---- POST --------------------------------------------------------------

    def do_POST(self):
        path = self._clean_path()
        try:
            if path == "/create_payment_intent":
                self._handle_create_payment_intent()
            elif path == "/create_checkout_session":
                self._handle_create_checkout_session()
            elif path == "/create_subscription":
                self._handle_create_subscription()
            elif path == "/cancel_subscription":
                self._handle_cancel_subscription()
            elif path == "/webhook":
                self._handle_webhook()
            else:
                self._send(404, {"error": "not found", "path": path})
        except RuntimeError as exc:
            # Stripe API errors
            self._stripe_error(exc)
        except KeyError as exc:
            self._send(400, {"error": f"missing required field: {exc}"})
        except json.JSONDecodeError as exc:
            self._send(400, {"error": f"invalid JSON: {exc}"})
        except Exception as exc:
            print(f"[{AGENT_NAME}] ERROR POST {path}: {exc}", flush=True)
            self._send(500, {"error": str(exc)})

    # ---- GET handlers ------------------------------------------------------

    def _handle_health(self):
        con = _db()
        pay_count = con.execute("SELECT COUNT(*) FROM payments").fetchone()[0]
        sub_count = con.execute("SELECT COUNT(*) FROM subscriptions").fetchone()[0]
        con.close()
        self._send(200, {
            "status": "ok",
            "agent": AGENT_NAME,
            "port": PORT,
            "uptime_seconds": round(time.time() - START_TIME, 1),
            "payment_count": pay_count,
            "subscription_count": sub_count,
            "stripe_configured": bool(STRIPE_SECRET_KEY),
            "webhook_secret_configured": bool(STRIPE_WEBHOOK_SECRET),
        })

    def _handle_list_payments(self, qs: dict):
        limit         = min(int(qs.get("limit", "50")), 500)
        status_filter = qs.get("status", "")
        email_filter  = qs.get("customer_email", "")

        query  = "SELECT * FROM payments WHERE 1=1"
        params: list = []
        if status_filter:
            query += " AND status=?"
            params.append(status_filter)
        if email_filter:
            query += " AND customer_email=?"
            params.append(email_filter)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        con = _db()
        rows = con.execute(query, params).fetchall()
        con.close()
        self._send(200, [dict(r) for r in rows])

    def _handle_get_payment(self, stripe_payment_id: str):
        con = _db()
        row = con.execute(
            "SELECT * FROM payments WHERE stripe_payment_id=?",
            (stripe_payment_id,),
        ).fetchone()
        con.close()
        if row is None:
            self._send(404, {"error": "payment not found", "stripe_payment_id": stripe_payment_id})
        else:
            self._send(200, dict(row))

    def _handle_list_subscriptions(self, qs: dict):
        status_filter = qs.get("status", "")
        email_filter  = qs.get("customer_email", "")

        query  = "SELECT * FROM subscriptions WHERE 1=1"
        params: list = []
        if status_filter:
            query += " AND status=?"
            params.append(status_filter)
        if email_filter:
            query += " AND customer_email=?"
            params.append(email_filter)
        query += " ORDER BY created_at DESC LIMIT 200"

        con = _db()
        rows = con.execute(query, params).fetchall()
        con.close()
        self._send(200, [dict(r) for r in rows])

    def _handle_list_customers(self, qs: dict):
        limit = min(int(qs.get("limit", "50")), 500)
        con   = _db()
        rows  = con.execute(
            "SELECT * FROM customers ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        con.close()
        self._send(200, [dict(r) for r in rows])

    def _handle_revenue(self):
        self._send(200, _revenue_stats())

    def _handle_list_webhook_events(self):
        con  = _db()
        rows = con.execute(
            "SELECT * FROM webhook_events ORDER BY created_at DESC LIMIT 100"
        ).fetchall()
        con.close()
        self._send(200, [dict(r) for r in rows])

    # ---- POST handlers -----------------------------------------------------

    def _handle_create_payment_intent(self):
        body     = self._read_json()
        amount   = int(body["amount"])          # cents
        currency = body.get("currency", "usd").lower()
        desc     = body.get("description", "")
        email    = body.get("customer_email", "")
        metadata = body.get("metadata", {})

        if amount <= 0:
            self._send(400, {"error": "amount must be positive integer (cents)"})
            return

        payload: dict = {
            "amount":   amount,
            "currency": currency,
        }
        if desc:
            payload["description"] = desc
        if email:
            payload["receipt_email"] = email
        if metadata and isinstance(metadata, dict):
            payload["metadata"] = metadata

        pi = _stripe_post_encoded("/payment_intents", payload)

        pi_id  = pi["id"]
        status = pi.get("status", "requires_payment_method")

        con = _db()
        con.execute(
            "INSERT OR IGNORE INTO payments "
            "(stripe_payment_id, amount, currency, status, customer_email, description, metadata, created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                pi_id, amount, currency, status, email, desc,
                json.dumps(metadata), time.time(),
            ),
        )
        con.commit()
        con.close()

        print(f"[{AGENT_NAME}] PaymentIntent created: {pi_id} {amount}{currency} {email}", flush=True)
        self._send(201, {
            "payment_intent_id": pi_id,
            "client_secret":     pi.get("client_secret", ""),
            "status":            status,
            "amount":            amount,
            "currency":          currency,
        })

    def _handle_create_checkout_session(self):
        body        = self._read_json()
        amount      = int(body["amount"])
        currency    = body.get("currency", "usd").lower()
        desc        = body.get("description", "FractalMesh Payment")
        email       = body.get("customer_email", "")
        success_url = body["success_url"]
        cancel_url  = body["cancel_url"]

        if amount <= 0:
            self._send(400, {"error": "amount must be positive integer (cents)"})
            return

        payload: dict = {
            "mode":        "payment",
            "success_url": success_url,
            "cancel_url":  cancel_url,
            "line_items[0][price_data][currency]":             currency,
            "line_items[0][price_data][unit_amount]":          str(amount),
            "line_items[0][price_data][product_data][name]":   desc,
            "line_items[0][quantity]":                         "1",
        }
        if email:
            payload["customer_email"] = email

        session = _stripe_post_encoded("/checkout/sessions", payload)

        session_id  = session["id"]
        session_url = session.get("url", "")

        # Record as a pending payment in local DB
        con = _db()
        con.execute(
            "INSERT OR IGNORE INTO payments "
            "(stripe_payment_id, amount, currency, status, customer_email, description, metadata, created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                session_id, amount, currency,
                session.get("payment_status", "unpaid"),
                email, desc, json.dumps({"type": "checkout_session"}),
                time.time(),
            ),
        )
        con.commit()
        con.close()

        print(f"[{AGENT_NAME}] CheckoutSession created: {session_id} {amount}{currency}", flush=True)
        self._send(201, {
            "session_id": session_id,
            "url":        session_url,
            "amount":     amount,
            "currency":   currency,
        })

    def _handle_create_subscription(self):
        body     = self._read_json()
        email    = body["customer_email"]
        price_id = body["price_id"]
        name     = body.get("name", "")

        # Find or create Stripe customer
        cust_id = _find_or_create_customer(email, name)

        # Create subscription
        sub_payload: dict = {
            "customer": cust_id,
            "items[0][price]": price_id,
            "expand[0]": "latest_invoice.payment_intent",
        }
        sub = _stripe_post_encoded("/subscriptions", sub_payload)

        sub_id  = sub["id"]
        status  = sub.get("status", "incomplete")
        period_end = sub.get("current_period_end", 0)
        plan_name  = price_id  # Use price_id as plan identifier

        now = time.time()
        con = _db()
        con.execute(
            "INSERT OR REPLACE INTO subscriptions "
            "(stripe_sub_id, customer_id, customer_email, plan, status, current_period_end, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (sub_id, cust_id, email, plan_name, status, float(period_end), now, now),
        )
        con.commit()
        con.close()

        # Extract client_secret if available (for 3DS)
        client_secret = (
            sub.get("latest_invoice", {})
               .get("payment_intent", {})
               .get("client_secret", "")
        )

        print(f"[{AGENT_NAME}] Subscription created: {sub_id} for {email} plan={price_id}", flush=True)
        self._send(201, {
            "subscription_id":   sub_id,
            "customer_id":       cust_id,
            "customer_email":    email,
            "plan":              plan_name,
            "status":            status,
            "current_period_end": period_end,
            "client_secret":     client_secret,
        })

    def _handle_cancel_subscription(self):
        if not self._require_admin():
            return

        body      = self._read_json()
        sub_id    = body["stripe_sub_id"]

        # Cancel via Stripe API
        result = _stripe_delete(f"/subscriptions/{sub_id}")

        new_status = result.get("status", "canceled")
        now        = time.time()

        con = _db()
        con.execute(
            "UPDATE subscriptions SET status=?, updated_at=? WHERE stripe_sub_id=?",
            (new_status, now, sub_id),
        )
        con.commit()
        con.close()

        print(f"[{AGENT_NAME}] Subscription cancelled: {sub_id} → {new_status}", flush=True)
        self._send(200, {
            "subscription_id": sub_id,
            "status":          new_status,
            "cancelled_at":    now,
        })

    def _handle_webhook(self):
        raw_body   = self._read_body()
        sig_header = self.headers.get("Stripe-Signature", "")

        # Verify signature
        if STRIPE_WEBHOOK_SECRET:
            if not _verify_stripe_signature(raw_body, sig_header):
                print(f"[{AGENT_NAME}] Webhook signature verification FAILED", flush=True)
                self._send(400, {"error": "invalid stripe signature"})
                return

        try:
            event = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            self._send(400, {"error": f"invalid JSON: {exc}"})
            return

        event_id   = event.get("id", "")
        event_type = event.get("event", event.get("type", "unknown"))
        now        = time.time()

        # Idempotency: check if already processed
        con = _db()
        existing = con.execute(
            "SELECT id, processed FROM webhook_events WHERE stripe_event_id=?",
            (event_id,),
        ).fetchone()

        if existing:
            con.close()
            self._send(200, {"received": True, "duplicate": True, "event_id": event_id})
            return

        # Store raw event
        con.execute(
            "INSERT OR IGNORE INTO webhook_events "
            "(stripe_event_id, event_type, processed, payload, created_at) "
            "VALUES (?,?,0,?,?)",
            (event_id, event_type, json.dumps(event), now),
        )
        con.commit()
        con.close()

        # Process event
        data_obj = event.get("data", {}).get("object", {})
        try:
            if event_type == "payment_intent.succeeded":
                self._process_pi_succeeded(data_obj)
            elif event_type == "payment_intent.payment_failed":
                self._process_pi_failed(data_obj)
            elif event_type == "customer.subscription.updated":
                self._process_sub_updated(data_obj)
            elif event_type == "customer.subscription.deleted":
                self._process_sub_deleted(data_obj)
            else:
                print(f"[{AGENT_NAME}] Unhandled webhook event: {event_type}", flush=True)

            # Mark as processed
            con2 = _db()
            con2.execute(
                "UPDATE webhook_events SET processed=1 WHERE stripe_event_id=?",
                (event_id,),
            )
            con2.commit()
            con2.close()

        except Exception as exc:
            print(f"[{AGENT_NAME}] Webhook processing error ({event_type}): {exc}", flush=True)

        self._send(200, {"received": True, "event_id": event_id, "event_type": event_type})

    # ---- Webhook event processors ------------------------------------------

    def _process_pi_succeeded(self, pi: dict) -> None:
        pi_id  = pi.get("id", "")
        status = "succeeded"
        if not pi_id:
            return
        con = _db()
        existing = con.execute(
            "SELECT id FROM payments WHERE stripe_payment_id=?", (pi_id,)
        ).fetchone()
        if existing:
            con.execute(
                "UPDATE payments SET status=? WHERE stripe_payment_id=?",
                (status, pi_id),
            )
        else:
            # Insert fresh record from webhook data
            con.execute(
                "INSERT OR IGNORE INTO payments "
                "(stripe_payment_id, amount, currency, status, customer_email, description, metadata, created_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    pi_id,
                    int(pi.get("amount", 0)),
                    pi.get("currency", "usd"),
                    status,
                    pi.get("receipt_email") or "",
                    pi.get("description") or "",
                    json.dumps(pi.get("metadata") or {}),
                    float(pi.get("created", time.time())),
                ),
            )
        con.commit()
        con.close()
        print(f"[{AGENT_NAME}] payment_intent.succeeded: {pi_id}", flush=True)

    def _process_pi_failed(self, pi: dict) -> None:
        pi_id  = pi.get("id", "")
        status = "payment_failed"
        if not pi_id:
            return
        con = _db()
        con.execute(
            "UPDATE payments SET status=? WHERE stripe_payment_id=?",
            (status, pi_id),
        )
        con.commit()
        con.close()
        print(f"[{AGENT_NAME}] payment_intent.payment_failed: {pi_id}", flush=True)

    def _process_sub_updated(self, sub: dict) -> None:
        sub_id     = sub.get("id", "")
        status     = sub.get("status", "")
        period_end = float(sub.get("current_period_end", 0))
        now        = time.time()
        if not sub_id:
            return
        con = _db()
        existing = con.execute(
            "SELECT id FROM subscriptions WHERE stripe_sub_id=?", (sub_id,)
        ).fetchone()
        if existing:
            con.execute(
                "UPDATE subscriptions SET status=?, current_period_end=?, updated_at=? "
                "WHERE stripe_sub_id=?",
                (status, period_end, now, sub_id),
            )
        else:
            # Insert subscription from webhook if not seen before
            cust_id = sub.get("customer", "")
            plan    = ""
            items   = sub.get("items", {}).get("data", [])
            if items:
                plan = items[0].get("price", {}).get("id", "")
            con.execute(
                "INSERT OR IGNORE INTO subscriptions "
                "(stripe_sub_id, customer_id, customer_email, plan, status, "
                " current_period_end, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (sub_id, cust_id, "", plan, status, period_end, now, now),
            )
        con.commit()
        con.close()
        print(f"[{AGENT_NAME}] customer.subscription.updated: {sub_id} → {status}", flush=True)

    def _process_sub_deleted(self, sub: dict) -> None:
        sub_id = sub.get("id", "")
        if not sub_id:
            return
        now = time.time()
        con = _db()
        con.execute(
            "UPDATE subscriptions SET status='canceled', updated_at=? WHERE stripe_sub_id=?",
            (now, sub_id),
        )
        con.commit()
        con.close()
        print(f"[{AGENT_NAME}] customer.subscription.deleted: {sub_id}", flush=True)

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

_server: HTTPServer | None = None


def _shutdown(signum, _frame):
    print(f"[{AGENT_NAME}] Signal {signum} received — shutting down.", flush=True)
    if _server:
        threading.Thread(target=_server.shutdown, daemon=True).start()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    global _server

    init_db()
    start_sync_thread()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    _server = HTTPServer(("0.0.0.0", PORT), StripeGatewayHandler)
    configured = "yes" if STRIPE_SECRET_KEY else "NO — set STRIPE_SECRET_KEY"
    print(
        f"[{AGENT_NAME}] Listening on http://0.0.0.0:{PORT} | "
        f"stripe_key={configured} | "
        f"webhook_secret={'yes' if STRIPE_WEBHOOK_SECRET else 'no'}",
        flush=True,
    )
    _server.serve_forever()
    print(f"[{AGENT_NAME}] Stopped.", flush=True)


if __name__ == "__main__":
    main()
