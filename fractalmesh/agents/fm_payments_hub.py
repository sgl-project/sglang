#!/usr/bin/env python3
"""
FractalMesh OMEGA Titan — Multi-Gateway Payments Hub
Port: 7864

Unified payment hub supporting PayPal and Square alongside the existing Stripe
gateway.  Normalises payment data across providers into a single schema and
routes each charge to the optimal gateway based on amount, currency, and
live availability.

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
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

# ---------------------------------------------------------------------------
# Config — read from environment, never hard-coded
# ---------------------------------------------------------------------------
AGENT_NAME = "fm_payments_hub"
PORT = int(os.environ.get("PAYMENTS_HUB_PORT", "7864"))

PAYPAL_CLIENT_ID     = os.environ.get("PAYPAL_CLIENT_ID", "")
PAYPAL_CLIENT_SECRET = os.environ.get("PAYPAL_CLIENT_SECRET", "")
PAYPAL_MODE          = os.environ.get("PAYPAL_MODE", "sandbox").lower()  # "sandbox" | "live"

SQUARE_ACCESS_TOKEN = os.environ.get("SQUARE_ACCESS_TOKEN", "")
SQUARE_LOCATION_ID  = os.environ.get("SQUARE_LOCATION_ID", "")

ADMIN_SECRET = os.environ.get("ADMIN_SECRET", "")

# PayPal base URLs
if PAYPAL_MODE == "live":
    PAYPAL_BASE = "https://api.paypal.com"
else:
    PAYPAL_BASE = "https://api-m.sandbox.paypal.com"

# Square base URLs — sandbox token starts with "sandbox-"
if SQUARE_ACCESS_TOKEN.startswith("sandbox-"):
    SQUARE_BASE = "https://connect.squareupsandbox.com/v2"
else:
    SQUARE_BASE = "https://connect.squareup.com/v2"

SQUARE_VERSION = "2024-01-18"

ROOT    = Path(os.environ.get("FRACTALMESH_HOME", Path.home() / "fmsaas"))
DB_PATH = ROOT / "database" / "sovereign.db"

START_TIME = time.time()

# ---------------------------------------------------------------------------
# In-memory PayPal token cache
# ---------------------------------------------------------------------------
_PAYPAL_TOKEN: dict = {"access_token": None, "expires_at": 0.0}
_PAYPAL_TOKEN_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _db() -> sqlite3.Connection:
    con = sqlite3.connect(str(DB_PATH), timeout=10)
    con.execute("PRAGMA journal_mode=WAL")
    con.row_factory = sqlite3.Row
    return con


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = _db()
    con.executescript("""
        CREATE TABLE IF NOT EXISTS transactions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            tx_ref          TEXT    UNIQUE NOT NULL,
            gateway         TEXT    NOT NULL,
            gateway_tx_id   TEXT,
            amount          REAL    NOT NULL,
            currency        TEXT    NOT NULL DEFAULT 'USD',
            status          TEXT    NOT NULL DEFAULT 'pending',
            customer_email  TEXT    NOT NULL DEFAULT '',
            description     TEXT    NOT NULL DEFAULT '',
            gateway_fee     REAL,
            net_amount      REAL,
            metadata        TEXT    NOT NULL DEFAULT '{}',
            created_at      REAL    NOT NULL,
            updated_at      REAL    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS refunds (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            tx_ref            TEXT    NOT NULL,
            gateway           TEXT    NOT NULL,
            gateway_refund_id TEXT,
            amount            REAL    NOT NULL,
            reason            TEXT    NOT NULL DEFAULT '',
            status            TEXT    NOT NULL DEFAULT 'pending',
            created_at        REAL    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS gateway_status (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            gateway     TEXT    UNIQUE NOT NULL,
            enabled     INTEGER NOT NULL DEFAULT 1,
            last_check  REAL,
            latency_ms  REAL,
            error_count INTEGER NOT NULL DEFAULT 0
        );

        INSERT OR IGNORE INTO gateway_status (gateway) VALUES ('paypal');
        INSERT OR IGNORE INTO gateway_status (gateway) VALUES ('square');
    """)
    con.commit()
    con.close()


def _generate_tx_ref() -> str:
    """Generate a unique transaction reference."""
    raw = f"TX-{time.time()}-{os.urandom(8).hex()}"
    return "TX-" + hashlib.sha256(raw.encode()).hexdigest()[:16].upper()


def _generate_idempotency_key() -> str:
    return hashlib.sha256(os.urandom(32)).hexdigest()


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

def _is_admin(handler: "PaymentsHubHandler") -> bool:
    auth = handler.headers.get("Authorization", "")
    if not ADMIN_SECRET:
        return True
    if auth.startswith("Bearer "):
        token = auth[7:]
        return hmac.compare_digest(token, ADMIN_SECRET)
    return False


# ---------------------------------------------------------------------------
# PayPal OAuth2 token management
# ---------------------------------------------------------------------------

def _paypal_get_token() -> str:
    """Return a valid PayPal access token, refreshing if necessary."""
    with _PAYPAL_TOKEN_LOCK:
        now = time.time()
        if _PAYPAL_TOKEN["access_token"] and now < _PAYPAL_TOKEN["expires_at"]:
            return _PAYPAL_TOKEN["access_token"]

        credentials = f"{PAYPAL_CLIENT_ID}:{PAYPAL_CLIENT_SECRET}"
        encoded = base64.b64encode(credentials.encode()).decode()

        data = urllib.parse.urlencode({"grant_type": "client_credentials"}).encode()
        req = urllib.request.Request(
            f"{PAYPAL_BASE}/v1/oauth2/token",
            data=data,
            headers={
                "Authorization": f"Basic {encoded}",
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"PayPal token error {exc.code}: {exc.read().decode()}") from exc

        _PAYPAL_TOKEN["access_token"] = body["access_token"]
        # expire slightly early to avoid race at boundary
        _PAYPAL_TOKEN["expires_at"] = now + body.get("expires_in", 32400) - 60
        return _PAYPAL_TOKEN["access_token"]


def _paypal_request(method: str, path: str, payload: dict | None = None) -> dict:
    """Make an authenticated PayPal API request."""
    token = _paypal_get_token()
    url = f"{PAYPAL_BASE}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"PayPal API error {exc.code}: {exc.read().decode()}") from exc


# ---------------------------------------------------------------------------
# PayPal payment operations
# ---------------------------------------------------------------------------

def paypal_create_order(amount: float, currency: str, description: str) -> dict:
    """Create a PayPal order and return the order data including approval URL."""
    payload = {
        "intent": "CAPTURE",
        "purchase_units": [
            {
                "amount": {
                    "currency_code": currency.upper(),
                    "value": f"{amount:.2f}",
                },
                "description": description,
            }
        ],
        "application_context": {
            "return_url": "https://fractalmesh.io/payment/success",
            "cancel_url": "https://fractalmesh.io/payment/cancel",
        },
    }
    return _paypal_request("POST", "/v2/checkout/orders", payload)


def paypal_capture_order(order_id: str) -> dict:
    """Capture an approved PayPal order."""
    return _paypal_request("POST", f"/v2/checkout/orders/{order_id}/capture", {})


def paypal_refund_capture(capture_id: str, amount: float, currency: str, reason: str) -> dict:
    """Refund a captured PayPal payment."""
    payload: dict = {
        "note_to_payer": reason[:255] if reason else "Refund",
    }
    if amount:
        payload["amount"] = {
            "value": f"{amount:.2f}",
            "currency_code": currency.upper(),
        }
    return _paypal_request("POST", f"/v2/payments/captures/{capture_id}/refund", payload)


# ---------------------------------------------------------------------------
# Square payment operations
# ---------------------------------------------------------------------------

def _square_request(method: str, path: str, payload: dict | None = None) -> dict:
    """Make an authenticated Square API request."""
    url = f"{SQUARE_BASE}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {
        "Authorization": f"Bearer {SQUARE_ACCESS_TOKEN}",
        "Square-Version": SQUARE_VERSION,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Square API error {exc.code}: {exc.read().decode()}") from exc


def _to_cents(amount: float) -> int:
    """Convert a dollar amount to integer cents."""
    return int(round(amount * 100))


def square_create_payment(amount: float, currency: str, description: str) -> dict:
    """Create a Square payment using an EXTERNAL source."""
    payload = {
        "source_id": "EXTERNAL",
        "idempotency_key": _generate_idempotency_key(),
        "amount_money": {
            "amount": _to_cents(amount),
            "currency": currency.upper(),
        },
        "location_id": SQUARE_LOCATION_ID,
        "note": description[:500] if description else "",
        "external_details": {
            "type": "OTHER",
            "source": "FractalMesh Payments Hub",
        },
    }
    return _square_request("POST", "/payments", payload)


def square_refund_payment(payment_id: str, amount: float, currency: str, reason: str) -> dict:
    """Refund a Square payment."""
    payload = {
        "idempotency_key": _generate_idempotency_key(),
        "payment_id": payment_id,
        "amount_money": {
            "amount": _to_cents(amount),
            "currency": currency.upper(),
        },
        "reason": reason[:192] if reason else "Refund",
    }
    return _square_request("POST", "/refunds", payload)


# ---------------------------------------------------------------------------
# Gateway routing logic
# ---------------------------------------------------------------------------

def _gateway_enabled(gateway: str) -> bool:
    try:
        con = _db()
        row = con.execute(
            "SELECT enabled FROM gateway_status WHERE gateway = ?", (gateway,)
        ).fetchone()
        con.close()
        return bool(row and row["enabled"])
    except Exception:
        return False


def _select_gateway(amount: float, currency: str, preferred: str | None) -> str:
    """
    Select the optimal payment gateway.

    Rules (in priority order):
    1. If preferred gateway is specified and enabled → use it.
    2. AUD amounts under $10 → Square (lower fees for micro-transactions).
    3. Non-USD / non-AUD currencies → PayPal (wider international support).
    4. Default → PayPal if available, else Square.
    """
    currency_upper = currency.upper()

    if preferred:
        gw = preferred.lower()
        if gw in ("paypal", "square") and _gateway_enabled(gw):
            return gw

    if currency_upper == "AUD" and amount < 10.0:
        if _gateway_enabled("square"):
            return "square"

    if currency_upper not in ("USD", "AUD"):
        if _gateway_enabled("paypal"):
            return "paypal"
        if _gateway_enabled("square"):
            return "square"

    # Default path
    if _gateway_enabled("paypal"):
        return "paypal"
    if _gateway_enabled("square"):
        return "square"

    raise RuntimeError("No payment gateways are currently available")


# ---------------------------------------------------------------------------
# Background health-check thread
# ---------------------------------------------------------------------------

def _health_check_loop() -> None:
    """Daemon thread: checks each gateway every 300 s and updates gateway_status."""
    while True:
        _run_health_checks()
        time.sleep(300)


def _run_health_checks() -> None:
    for gateway in ("paypal", "square"):
        start = time.time()
        ok = False
        try:
            if gateway == "paypal":
                _paypal_request("GET", "/v1/identity/openidconnect/tokenservice")
            else:
                _square_request("GET", f"/locations/{SQUARE_LOCATION_ID}")
            ok = True
        except Exception:
            pass

        latency_ms = (time.time() - start) * 1000
        now = time.time()

        try:
            con = _db()
            if ok:
                con.execute(
                    """UPDATE gateway_status
                       SET enabled=1, last_check=?, latency_ms=?, error_count=0
                       WHERE gateway=?""",
                    (now, latency_ms, gateway),
                )
            else:
                con.execute(
                    """UPDATE gateway_status
                       SET last_check=?, latency_ms=?, error_count=error_count+1
                       WHERE gateway=?""",
                    (now, latency_ms, gateway),
                )
                # Disable if error_count exceeds threshold
                con.execute(
                    """UPDATE gateway_status SET enabled=0
                       WHERE gateway=? AND error_count >= 5""",
                    (gateway,),
                )
            con.commit()
            con.close()
        except Exception as exc:
            print(f"[{AGENT_NAME}] health-check DB write failed: {exc}", flush=True)


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

def _json_response(handler: "PaymentsHubHandler", code: int, body: dict) -> None:
    payload = json.dumps(body, default=str).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def _read_body(handler: "PaymentsHubHandler") -> dict:
    length = int(handler.headers.get("Content-Length", "0"))
    if length == 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


class PaymentsHubHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the Multi-Gateway Payments Hub."""

    server_version = "FractalMesh/PaymentsHub"
    sys_version = ""

    # ------------------------------------------------------------------ routing

    def do_GET(self) -> None:
        path = self.path.split("?")[0].rstrip("/")
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)

        if path == "/health":
            self._handle_health()
        elif path == "/transactions":
            self._handle_list_transactions(query)
        elif path.startswith("/transactions/"):
            tx_ref = path[len("/transactions/"):]
            self._handle_get_transaction(tx_ref)
        elif path == "/refunds":
            self._handle_list_refunds(query)
        elif path == "/gateways":
            self._handle_gateways()
        elif path == "/revenue":
            self._handle_revenue()
        else:
            _json_response(self, 404, {"error": "not found"})

    def do_POST(self) -> None:
        path = self.path.rstrip("/")

        if path == "/charge":
            self._handle_charge()
        elif path == "/refund":
            self._handle_refund()
        elif path.startswith("/gateways/") and path.endswith("/toggle"):
            parts = path.split("/")
            # /gateways/{gateway}/toggle
            if len(parts) == 4:
                self._handle_gateway_toggle(parts[2])
            else:
                _json_response(self, 404, {"error": "not found"})
        else:
            _json_response(self, 404, {"error": "not found"})

    def log_message(self, fmt: str, *args) -> None:  # noqa: ANN001
        print(f"[{AGENT_NAME}] {self.address_string()} {fmt % args}", flush=True)

    # ------------------------------------------------------------------ GET /health

    def _handle_health(self) -> None:
        try:
            con = _db()
            tx_count = con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
            gw_rows = con.execute(
                "SELECT gateway, enabled, last_check, latency_ms, error_count FROM gateway_status"
            ).fetchall()
            con.close()

            gateways = {
                r["gateway"]: {
                    "enabled": bool(r["enabled"]),
                    "last_check": r["last_check"],
                    "latency_ms": r["latency_ms"],
                    "error_count": r["error_count"],
                }
                for r in gw_rows
            }

            _json_response(
                self,
                200,
                {
                    "status": "ok",
                    "agent": AGENT_NAME,
                    "uptime_seconds": round(time.time() - START_TIME, 2),
                    "transaction_count": tx_count,
                    "gateways": gateways,
                    "paypal_mode": PAYPAL_MODE,
                },
            )
        except Exception as exc:
            _json_response(self, 500, {"error": str(exc)})

    # ------------------------------------------------------------------ GET /transactions

    def _handle_list_transactions(self, query: dict) -> None:
        try:
            gateway  = query.get("gateway", [None])[0]
            status   = query.get("status",  [None])[0]
            currency = query.get("currency",[None])[0]
            limit    = min(int(query.get("limit", ["100"])[0]), 500)

            sql    = "SELECT * FROM transactions WHERE 1=1"
            params: list = []

            if gateway:
                sql += " AND gateway = ?"
                params.append(gateway)
            if status:
                sql += " AND status = ?"
                params.append(status)
            if currency:
                sql += " AND UPPER(currency) = ?"
                params.append(currency.upper())

            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            con = _db()
            rows = [dict(r) for r in con.execute(sql, params).fetchall()]
            con.close()

            # Parse stored JSON metadata
            for r in rows:
                try:
                    r["metadata"] = json.loads(r["metadata"])
                except Exception:
                    pass

            _json_response(self, 200, {"transactions": rows, "count": len(rows)})
        except Exception as exc:
            _json_response(self, 500, {"error": str(exc)})

    # ------------------------------------------------------------------ GET /transactions/{tx_ref}

    def _handle_get_transaction(self, tx_ref: str) -> None:
        try:
            con = _db()
            row = con.execute(
                "SELECT * FROM transactions WHERE tx_ref = ?", (tx_ref,)
            ).fetchone()
            # Also fetch any refunds for this transaction
            refunds = [
                dict(r)
                for r in con.execute(
                    "SELECT * FROM refunds WHERE tx_ref = ? ORDER BY created_at DESC", (tx_ref,)
                ).fetchall()
            ]
            con.close()

            if not row:
                _json_response(self, 404, {"error": "transaction not found"})
                return

            data = dict(row)
            try:
                data["metadata"] = json.loads(data["metadata"])
            except Exception:
                pass
            data["refunds"] = refunds

            _json_response(self, 200, data)
        except Exception as exc:
            _json_response(self, 500, {"error": str(exc)})

    # ------------------------------------------------------------------ GET /refunds

    def _handle_list_refunds(self, query: dict) -> None:
        try:
            gateway = query.get("gateway", [None])[0]
            status  = query.get("status",  [None])[0]
            limit   = min(int(query.get("limit", ["100"])[0]), 500)

            sql    = "SELECT * FROM refunds WHERE 1=1"
            params: list = []

            if gateway:
                sql += " AND gateway = ?"
                params.append(gateway)
            if status:
                sql += " AND status = ?"
                params.append(status)

            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            con = _db()
            rows = [dict(r) for r in con.execute(sql, params).fetchall()]
            con.close()

            _json_response(self, 200, {"refunds": rows, "count": len(rows)})
        except Exception as exc:
            _json_response(self, 500, {"error": str(exc)})

    # ------------------------------------------------------------------ GET /gateways

    def _handle_gateways(self) -> None:
        try:
            con = _db()
            rows = con.execute("SELECT * FROM gateway_status").fetchall()
            con.close()
            gateways = [dict(r) for r in rows]
            _json_response(self, 200, {"gateways": gateways})
        except Exception as exc:
            _json_response(self, 500, {"error": str(exc)})

    # ------------------------------------------------------------------ GET /revenue

    def _handle_revenue(self) -> None:
        try:
            con = _db()

            # Daily totals (last 30 days) grouped by gateway and currency
            daily = [
                dict(r)
                for r in con.execute(
                    """
                    SELECT
                        strftime('%Y-%m-%d', datetime(created_at, 'unixepoch')) AS day,
                        gateway,
                        currency,
                        COUNT(*)          AS tx_count,
                        SUM(amount)       AS gross,
                        SUM(net_amount)   AS net
                    FROM transactions
                    WHERE status = 'completed'
                      AND created_at >= ?
                    GROUP BY day, gateway, currency
                    ORDER BY day DESC, gateway
                    """,
                    (time.time() - 30 * 86400,),
                ).fetchall()
            ]

            # Monthly totals (last 12 months) grouped by gateway and currency
            monthly = [
                dict(r)
                for r in con.execute(
                    """
                    SELECT
                        strftime('%Y-%m', datetime(created_at, 'unixepoch')) AS month,
                        gateway,
                        currency,
                        COUNT(*)          AS tx_count,
                        SUM(amount)       AS gross,
                        SUM(net_amount)   AS net
                    FROM transactions
                    WHERE status = 'completed'
                      AND created_at >= ?
                    GROUP BY month, gateway, currency
                    ORDER BY month DESC, gateway
                    """,
                    (time.time() - 365 * 86400,),
                ).fetchall()
            ]

            con.close()
            _json_response(self, 200, {"daily": daily, "monthly": monthly})
        except Exception as exc:
            _json_response(self, 500, {"error": str(exc)})

    # ------------------------------------------------------------------ POST /charge

    def _handle_charge(self) -> None:
        if not _is_admin(self):
            _json_response(self, 403, {"error": "forbidden"})
            return

        body = _read_body(self)

        amount         = body.get("amount")
        currency       = str(body.get("currency", "USD")).upper()
        customer_email = str(body.get("customer_email", ""))
        description    = str(body.get("description", ""))
        preferred_gw   = body.get("gateway")
        metadata       = body.get("metadata", {})

        if not amount or float(amount) <= 0:
            _json_response(self, 400, {"error": "amount must be a positive number"})
            return

        amount = float(amount)
        tx_ref = _generate_tx_ref()
        now    = time.time()

        try:
            gateway = _select_gateway(amount, currency, preferred_gw)
        except RuntimeError as exc:
            _json_response(self, 503, {"error": str(exc)})
            return

        checkout_url  = None
        gateway_tx_id = None
        status        = "pending"
        gateway_fee   = None
        net_amount    = None

        try:
            if gateway == "paypal":
                order = paypal_create_order(amount, currency, description)
                gateway_tx_id = order.get("id")
                status = "pending"
                # Extract the payer-approval URL
                for link in order.get("links", []):
                    if link.get("rel") == "approve":
                        checkout_url = link.get("href")
                        break

            elif gateway == "square":
                result  = square_create_payment(amount, currency, description)
                payment = result.get("payment", {})
                gateway_tx_id = payment.get("id")
                sq_status     = payment.get("status", "").upper()
                if sq_status == "COMPLETED":
                    status = "completed"
                    fee_money = payment.get("processing_fee", [{}])
                    if fee_money:
                        gateway_fee = fee_money[0].get("amount_money", {}).get("amount", 0) / 100
                        net_amount  = amount - (gateway_fee or 0)
                else:
                    status = "pending"

        except Exception as exc:
            # Log failure but still record the attempted transaction
            print(f"[{AGENT_NAME}] charge error on {gateway}: {exc}", flush=True)
            status = "failed"

        try:
            con = _db()
            con.execute(
                """INSERT INTO transactions
                   (tx_ref, gateway, gateway_tx_id, amount, currency, status,
                    customer_email, description, gateway_fee, net_amount, metadata,
                    created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    tx_ref, gateway, gateway_tx_id, amount, currency, status,
                    customer_email, description, gateway_fee, net_amount,
                    json.dumps(metadata), now, now,
                ),
            )
            con.commit()
            con.close()
        except Exception as exc:
            _json_response(self, 500, {"error": f"DB write failed: {exc}"})
            return

        response: dict = {
            "tx_ref": tx_ref,
            "gateway": gateway,
            "gateway_tx_id": gateway_tx_id,
            "status": status,
            "amount": amount,
            "currency": currency,
        }
        if checkout_url:
            response["checkout_url"] = checkout_url

        _json_response(self, 201 if status != "failed" else 500, response)

    # ------------------------------------------------------------------ POST /refund

    def _handle_refund(self) -> None:
        if not _is_admin(self):
            _json_response(self, 403, {"error": "forbidden"})
            return

        body   = _read_body(self)
        tx_ref = body.get("tx_ref", "").strip()
        amount = body.get("amount")
        reason = str(body.get("reason", "Refund"))

        if not tx_ref:
            _json_response(self, 400, {"error": "tx_ref is required"})
            return

        try:
            con = _db()
            row = con.execute(
                "SELECT * FROM transactions WHERE tx_ref = ?", (tx_ref,)
            ).fetchone()
            con.close()
        except Exception as exc:
            _json_response(self, 500, {"error": str(exc)})
            return

        if not row:
            _json_response(self, 404, {"error": "transaction not found"})
            return

        tx = dict(row)

        if tx["status"] not in ("completed", "captured"):
            _json_response(
                self, 400,
                {"error": f"cannot refund a transaction with status '{tx['status']}'"}
            )
            return

        refund_amount = float(amount) if amount else tx["amount"]
        if refund_amount <= 0 or refund_amount > tx["amount"]:
            _json_response(
                self, 400,
                {"error": "refund amount must be positive and ≤ original charge"}
            )
            return

        gateway           = tx["gateway"]
        gateway_tx_id     = tx["gateway_tx_id"] or ""
        currency          = tx["currency"]
        now               = time.time()
        gateway_refund_id = None
        refund_status     = "pending"

        try:
            if gateway == "paypal":
                result            = paypal_refund_capture(gateway_tx_id, refund_amount, currency, reason)
                gateway_refund_id = result.get("id")
                refund_status     = "completed" if result.get("status") == "COMPLETED" else "pending"

            elif gateway == "square":
                result            = square_refund_payment(gateway_tx_id, refund_amount, currency, reason)
                refund_data       = result.get("refund", {})
                gateway_refund_id = refund_data.get("id")
                sq_status         = refund_data.get("status", "").upper()
                refund_status     = "completed" if sq_status == "COMPLETED" else "pending"

        except Exception as exc:
            print(f"[{AGENT_NAME}] refund error on {gateway}: {exc}", flush=True)
            refund_status = "failed"

        try:
            con = _db()
            con.execute(
                """INSERT INTO refunds
                   (tx_ref, gateway, gateway_refund_id, amount, reason, status, created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (tx_ref, gateway, gateway_refund_id, refund_amount, reason, refund_status, now),
            )
            if refund_status == "completed":
                con.execute(
                    "UPDATE transactions SET status='refunded', updated_at=? WHERE tx_ref=?",
                    (now, tx_ref),
                )
            con.commit()
            con.close()
        except Exception as exc:
            _json_response(self, 500, {"error": f"DB write failed: {exc}"})
            return

        _json_response(
            self,
            200 if refund_status != "failed" else 500,
            {
                "tx_ref": tx_ref,
                "gateway": gateway,
                "gateway_refund_id": gateway_refund_id,
                "refund_amount": refund_amount,
                "currency": currency,
                "status": refund_status,
            },
        )

    # ------------------------------------------------------------------ POST /gateways/{gateway}/toggle

    def _handle_gateway_toggle(self, gateway: str) -> None:
        if not _is_admin(self):
            _json_response(self, 403, {"error": "forbidden"})
            return

        gateway = gateway.lower()
        if gateway not in ("paypal", "square"):
            _json_response(self, 400, {"error": f"unknown gateway: {gateway}"})
            return

        try:
            con = _db()
            row = con.execute(
                "SELECT enabled FROM gateway_status WHERE gateway = ?", (gateway,)
            ).fetchone()

            if not row:
                _json_response(self, 404, {"error": "gateway not found"})
                con.close()
                return

            new_state = 0 if row["enabled"] else 1
            con.execute(
                "UPDATE gateway_status SET enabled=?, last_check=? WHERE gateway=?",
                (new_state, time.time(), gateway),
            )
            con.commit()
            con.close()

            _json_response(
                self,
                200,
                {
                    "gateway": gateway,
                    "enabled": bool(new_state),
                    "message": f"{gateway} {'enabled' if new_state else 'disabled'}",
                },
            )
        except Exception as exc:
            _json_response(self, 500, {"error": str(exc)})


# ---------------------------------------------------------------------------
# Server bootstrap
# ---------------------------------------------------------------------------

def _start_health_thread() -> None:
    t = threading.Thread(target=_health_check_loop, name="gateway-healthcheck", daemon=True)
    t.start()
    print(f"[{AGENT_NAME}] Background health-check thread started (interval=300s)", flush=True)


def main() -> None:
    init_db()
    print(f"[{AGENT_NAME}] Database initialised at {DB_PATH}", flush=True)

    _start_health_thread()

    server = HTTPServer(("0.0.0.0", PORT), PaymentsHubHandler)
    print(
        f"[{AGENT_NAME}] Multi-Gateway Payments Hub listening on port {PORT} "
        f"(PayPal mode={PAYPAL_MODE}, Square base={SQUARE_BASE})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"[{AGENT_NAME}] Shutting down.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
