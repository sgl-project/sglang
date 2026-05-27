#!/usr/bin/env python3
"""
FractalMesh OMEGA Titan — Point-of-Sale System
Port: 7879

Full point-of-sale system for retail/service businesses. Manages products and
services catalogue, shopping carts, checkouts, receipts, daily cash
reconciliation, and staff management.

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
import json
import secrets
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
AGENT_NAME = "fm_pos_system"
PORT = int(os.environ.get("POS_SYSTEM_PORT", "7879"))

STRIPE_SECRET_KEY   = os.environ.get("STRIPE_SECRET_KEY", "")
SQUARE_ACCESS_TOKEN = os.environ.get("SQUARE_ACCESS_TOKEN", "")
SQUARE_LOCATION_ID  = os.environ.get("SQUARE_LOCATION_ID", "")
SENDGRID_API_KEY    = os.environ.get("SENDGRID_API_KEY", "")
SENDGRID_FROM_EMAIL = os.environ.get("SENDGRID_FROM_EMAIL", "noreply@fractalmesh.io")
ADMIN_SECRET        = os.environ.get("ADMIN_SECRET", "")

# Square base URL
if SQUARE_ACCESS_TOKEN.startswith("sandbox-"):
    SQUARE_BASE = "https://connect.squareupsandbox.com/v2"
else:
    SQUARE_BASE = "https://connect.squareup.com/v2"
SQUARE_VERSION = "2024-01-18"

ROOT    = Path(os.environ.get("FRACTALMESH_HOME", Path.home() / "fmsaas"))
DB_PATH = ROOT / "database" / "sovereign.db"

START_TIME = time.time()

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS catalogue (
                id          INTEGER PRIMARY KEY,
                sku         TEXT UNIQUE NOT NULL,
                name        TEXT NOT NULL,
                description TEXT,
                category    TEXT,
                price       REAL NOT NULL,
                cost        REAL,
                tax_rate    REAL DEFAULT 0.10,
                stock       INTEGER DEFAULT 0,
                barcode     TEXT,
                active      INTEGER DEFAULT 1,
                created_at  REAL
            );

            CREATE TABLE IF NOT EXISTS carts (
                id              INTEGER PRIMARY KEY,
                cart_id         TEXT UNIQUE NOT NULL,
                customer_email  TEXT,
                staff_id        TEXT,
                status          TEXT DEFAULT 'open',
                subtotal        REAL DEFAULT 0,
                tax             REAL DEFAULT 0,
                total           REAL DEFAULT 0,
                discount_amount REAL DEFAULT 0,
                notes           TEXT,
                created_at      REAL,
                updated_at      REAL,
                completed_at    REAL
            );

            CREATE TABLE IF NOT EXISTS cart_items (
                id          INTEGER PRIMARY KEY,
                cart_id     TEXT NOT NULL,
                sku         TEXT NOT NULL,
                name        TEXT NOT NULL,
                quantity    INTEGER NOT NULL,
                unit_price  REAL NOT NULL,
                tax_rate    REAL NOT NULL,
                line_total  REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS transactions (
                id             INTEGER PRIMARY KEY,
                tx_ref         TEXT UNIQUE NOT NULL,
                cart_id        TEXT,
                payment_method TEXT,
                gateway        TEXT,
                gateway_tx_id  TEXT,
                amount         REAL,
                currency       TEXT DEFAULT 'AUD',
                status         TEXT,
                customer_email TEXT,
                receipt_sent   INTEGER DEFAULT 0,
                created_at     REAL
            );

            CREATE TABLE IF NOT EXISTS staff (
                id         INTEGER PRIMARY KEY,
                staff_id   TEXT UNIQUE NOT NULL,
                name       TEXT NOT NULL,
                pin_hash   TEXT NOT NULL,
                role       TEXT DEFAULT 'cashier',
                active     INTEGER DEFAULT 1,
                created_at REAL
            );

            CREATE TABLE IF NOT EXISTS end_of_day (
                id                 INTEGER PRIMARY KEY,
                date               TEXT UNIQUE NOT NULL,
                total_sales        REAL,
                total_transactions INTEGER,
                total_refunds      REAL,
                cash_expected      REAL,
                cash_counted       REAL,
                variance           REAL,
                closed_by          TEXT,
                closed_at          REAL
            );
        """)
    _seed_catalogue()


def _seed_catalogue() -> None:
    """Seed catalogue with default products if empty."""
    seeds = [
        {
            "sku": "FM-STARTER",
            "name": "FractalMesh Starter License",
            "description": "Entry-level FractalMesh software license.",
            "category": "Software",
            "price": 29.00,
            "cost": 0.00,
            "tax_rate": 0.10,
            "stock": 9999,
        },
        {
            "sku": "FM-PRO",
            "name": "FractalMesh Pro License",
            "description": "Professional FractalMesh software license.",
            "category": "Software",
            "price": 99.00,
            "cost": 0.00,
            "tax_rate": 0.10,
            "stock": 9999,
        },
        {
            "sku": "SVC-CONSULT-1H",
            "name": "API Consultation (1hr)",
            "description": "One hour of API integration consultation.",
            "category": "Services",
            "price": 150.00,
            "cost": 0.00,
            "tax_rate": 0.10,
            "stock": 9999,
        },
        {
            "sku": "SVC-CUSTOM-INT",
            "name": "Custom Integration",
            "description": "Bespoke third-party integration development.",
            "category": "Services",
            "price": 500.00,
            "cost": 0.00,
            "tax_rate": 0.10,
            "stock": 9999,
        },
    ]
    with get_db() as conn:
        count = conn.execute("SELECT COUNT(*) FROM catalogue").fetchone()[0]
        if count == 0:
            now = time.time()
            for item in seeds:
                conn.execute(
                    """INSERT OR IGNORE INTO catalogue
                       (sku, name, description, category, price, cost,
                        tax_rate, stock, active, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?)""",
                    (
                        item["sku"], item["name"], item["description"],
                        item["category"], item["price"], item["cost"],
                        item["tax_rate"], item["stock"], now,
                    ),
                )

# ---------------------------------------------------------------------------
# Background: abandon stale open carts (older than 2 hours)
# ---------------------------------------------------------------------------

def _abandon_stale_carts() -> None:
    cutoff = time.time() - 7200
    with get_db() as conn:
        conn.execute(
            "UPDATE carts SET status='abandoned', updated_at=? "
            "WHERE status='open' AND created_at < ?",
            (time.time(), cutoff),
        )


def _background_loop() -> None:
    while True:
        try:
            _abandon_stale_carts()
        except Exception:
            pass
        time.sleep(300)


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def _is_admin(headers) -> bool:
    secret = headers.get("X-Admin-Secret", "")
    if not ADMIN_SECRET:
        return False
    return hmac.compare_digest(secret, ADMIN_SECRET)


def _hash_pin(pin: str) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", pin.encode(), salt.encode(), 260_000)
    return f"{salt}:{dk.hex()}"


def _verify_pin(pin: str, pin_hash: str) -> bool:
    try:
        salt, dk_hex = pin_hash.split(":", 1)
        dk = hashlib.pbkdf2_hmac("sha256", pin.encode(), salt.encode(), 260_000)
        return hmac.compare_digest(dk.hex(), dk_hex)
    except Exception:
        return False

# ---------------------------------------------------------------------------
# Cart totals helper
# ---------------------------------------------------------------------------

def _recalculate_cart(conn: sqlite3.Connection, cart_id: str) -> None:
    rows = conn.execute(
        "SELECT quantity, unit_price, tax_rate FROM cart_items WHERE cart_id=?",
        (cart_id,),
    ).fetchall()
    subtotal = sum(r["quantity"] * r["unit_price"] for r in rows)
    tax = sum(r["quantity"] * r["unit_price"] * r["tax_rate"] for r in rows)

    cart = conn.execute(
        "SELECT discount_amount FROM carts WHERE cart_id=?", (cart_id,)
    ).fetchone()
    discount = cart["discount_amount"] if cart else 0.0
    total = max(subtotal + tax - discount, 0.0)

    conn.execute(
        "UPDATE carts SET subtotal=?, tax=?, total=?, updated_at=? WHERE cart_id=?",
        (round(subtotal, 2), round(tax, 2), round(total, 2), time.time(), cart_id),
    )


def _update_line_total(conn: sqlite3.Connection, cart_id: str, sku: str) -> None:
    row = conn.execute(
        "SELECT quantity, unit_price, tax_rate FROM cart_items WHERE cart_id=? AND sku=?",
        (cart_id, sku),
    ).fetchone()
    if row:
        line_total = round(
            row["quantity"] * row["unit_price"] * (1 + row["tax_rate"]), 2
        )
        conn.execute(
            "UPDATE cart_items SET line_total=? WHERE cart_id=? AND sku=?",
            (line_total, cart_id, sku),
        )

# ---------------------------------------------------------------------------
# Payment gateways
# ---------------------------------------------------------------------------

def _stripe_charge(amount_aud: float, description: str, customer_email: str) -> dict:
    """Charge via Stripe. Returns {ok, gateway_tx_id, error}."""
    if not STRIPE_SECRET_KEY:
        return {"ok": False, "error": "STRIPE_SECRET_KEY not configured"}
    amount_cents = int(round(amount_aud * 100))
    payload = urllib.parse.urlencode({
        "amount": str(amount_cents),
        "currency": "aud",
        "description": description,
        "receipt_email": customer_email,
        "payment_method_types[]": "card",
    }).encode()
    req = urllib.request.Request(
        "https://api.stripe.com/v1/payment_intents",
        data=payload,
        method="POST",
    )
    cred = f"{STRIPE_SECRET_KEY}:"
    import base64 as _b64
    req.add_header(
        "Authorization",
        "Basic " + _b64.b64encode(cred.encode()).decode(),
    )
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        return {"ok": True, "gateway_tx_id": data.get("id", ""), "raw": data}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        return {"ok": False, "error": body}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _stripe_refund(gateway_tx_id: str, amount_aud: float) -> dict:
    """Refund a Stripe PaymentIntent."""
    if not STRIPE_SECRET_KEY:
        return {"ok": False, "error": "STRIPE_SECRET_KEY not configured"}
    amount_cents = int(round(amount_aud * 100))
    payload = urllib.parse.urlencode({
        "payment_intent": gateway_tx_id,
        "amount": str(amount_cents),
    }).encode()
    req = urllib.request.Request(
        "https://api.stripe.com/v1/refunds",
        data=payload,
        method="POST",
    )
    import base64 as _b64
    cred = f"{STRIPE_SECRET_KEY}:"
    req.add_header(
        "Authorization",
        "Basic " + _b64.b64encode(cred.encode()).decode(),
    )
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        return {"ok": True, "refund_id": data.get("id", ""), "raw": data}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        return {"ok": False, "error": body}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _square_charge(amount_aud: float, description: str, customer_email: str) -> dict:
    """Charge via Square. Returns {ok, gateway_tx_id, error}."""
    if not SQUARE_ACCESS_TOKEN or not SQUARE_LOCATION_ID:
        return {"ok": False, "error": "Square credentials not configured"}
    amount_cents = int(round(amount_aud * 100))
    idempotency_key = secrets.token_hex(16)
    body = json.dumps({
        "idempotency_key": idempotency_key,
        "amount_money": {"amount": amount_cents, "currency": "AUD"},
        "location_id": SQUARE_LOCATION_ID,
        "note": description,
        "buyer_email_address": customer_email,
        "source_id": "EXTERNAL",
        "external_details": {"type": "OTHER", "source": "FractalMesh POS"},
    }).encode()
    req = urllib.request.Request(
        f"{SQUARE_BASE}/payments",
        data=body,
        method="POST",
    )
    req.add_header("Authorization", f"Bearer {SQUARE_ACCESS_TOKEN}")
    req.add_header("Content-Type", "application/json")
    req.add_header("Square-Version", SQUARE_VERSION)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        payment = data.get("payment", {})
        return {"ok": True, "gateway_tx_id": payment.get("id", ""), "raw": data}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        return {"ok": False, "error": body}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _square_refund(gateway_tx_id: str, amount_aud: float) -> dict:
    """Refund a Square payment."""
    if not SQUARE_ACCESS_TOKEN:
        return {"ok": False, "error": "Square credentials not configured"}
    amount_cents = int(round(amount_aud * 100))
    idempotency_key = secrets.token_hex(16)
    body = json.dumps({
        "idempotency_key": idempotency_key,
        "payment_id": gateway_tx_id,
        "amount_money": {"amount": amount_cents, "currency": "AUD"},
    }).encode()
    req = urllib.request.Request(
        f"{SQUARE_BASE}/refunds",
        data=body,
        method="POST",
    )
    req.add_header("Authorization", f"Bearer {SQUARE_ACCESS_TOKEN}")
    req.add_header("Content-Type", "application/json")
    req.add_header("Square-Version", SQUARE_VERSION)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        refund = data.get("refund", {})
        return {"ok": True, "refund_id": refund.get("id", ""), "raw": data}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        return {"ok": False, "error": body}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

# ---------------------------------------------------------------------------
# SendGrid receipt email
# ---------------------------------------------------------------------------

def _send_receipt(customer_email: str, tx_ref: str, cart: dict, items: list) -> bool:
    if not SENDGRID_API_KEY or not customer_email:
        return False
    lines = "\n".join(
        f"  - {it['name']} x{it['quantity']}  @ ${it['unit_price']:.2f} ea  = ${it['line_total']:.2f}"
        for it in items
    )
    body_text = (
        f"Thank you for your purchase!\n\n"
        f"Receipt: {tx_ref}\n\n"
        f"Items:\n{lines}\n\n"
        f"Subtotal: ${cart['subtotal']:.2f}\n"
        f"Tax:      ${cart['tax']:.2f}\n"
        f"Discount: -${cart['discount_amount']:.2f}\n"
        f"Total:    ${cart['total']:.2f} AUD\n\n"
        f"FractalMesh OMEGA Titan | ABN 56 628 117 363"
    )
    payload = json.dumps({
        "personalizations": [{"to": [{"email": customer_email}]}],
        "from": {"email": SENDGRID_FROM_EMAIL},
        "subject": f"Your FractalMesh Receipt — {tx_ref}",
        "content": [{"type": "text/plain", "value": body_text}],
    }).encode()
    req = urllib.request.Request(
        "https://api.sendgrid.com/v3/mail/send",
        data=payload,
        method="POST",
    )
    req.add_header("Authorization", f"Bearer {SENDGRID_API_KEY}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status in (200, 202)
    except Exception:
        return False

# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

def _json_resp(handler, status: int, data) -> None:
    body = json.dumps(data, default=str).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_body(handler) -> dict:
    length = int(handler.headers.get("Content-Length", 0))
    if length == 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw)
    except Exception:
        return {}


class POSHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # noqa: silence default logging
        pass

    # ------------------------------------------------------------------ GET --
    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        qs = {}
        if "?" in self.path:
            qs = dict(urllib.parse.parse_qsl(self.path.split("?", 1)[1]))

        # GET /health
        if path == "/health":
            today = time.strftime("%Y-%m-%d")
            with get_db() as conn:
                sales_row = conn.execute(
                    "SELECT COALESCE(SUM(amount),0) FROM transactions "
                    "WHERE status='completed' AND date(created_at,'unixepoch')=?",
                    (today,),
                ).fetchone()
                open_carts = conn.execute(
                    "SELECT COUNT(*) FROM carts WHERE status='open'"
                ).fetchone()[0]
            _json_resp(self, 200, {
                "status": "ok",
                "agent": AGENT_NAME,
                "port": PORT,
                "uptime_seconds": round(time.time() - START_TIME, 1),
                "today_sales_aud": round(sales_row[0], 2),
                "open_carts": open_carts,
            })
            return

        # GET /catalogue
        if path == "/catalogue":
            clause, params = "WHERE 1=1", []
            if "category" in qs:
                clause += " AND category=?"; params.append(qs["category"])
            if "active" in qs:
                clause += " AND active=?"; params.append(int(qs["active"]))
            if qs.get("low_stock", "").lower() in ("1", "true", "yes"):
                clause += " AND stock < 10"
            with get_db() as conn:
                rows = conn.execute(
                    f"SELECT * FROM catalogue {clause} ORDER BY category, name",
                    params,
                ).fetchall()
            _json_resp(self, 200, {"items": [dict(r) for r in rows]})
            return

        # GET /catalogue/{sku}
        if path.startswith("/catalogue/") and len(path.split("/")) == 3:
            sku = path.split("/")[2]
            with get_db() as conn:
                row = conn.execute(
                    "SELECT * FROM catalogue WHERE sku=?", (sku,)
                ).fetchone()
            if not row:
                _json_resp(self, 404, {"error": "SKU not found"})
                return
            _json_resp(self, 200, dict(row))
            return

        # GET /carts/{cart_id}
        if path.startswith("/carts/") and len(path.split("/")) == 3:
            cart_id = path.split("/")[2]
            with get_db() as conn:
                cart = conn.execute(
                    "SELECT * FROM carts WHERE cart_id=?", (cart_id,)
                ).fetchone()
                if not cart:
                    _json_resp(self, 404, {"error": "Cart not found"}); return
                items = conn.execute(
                    "SELECT * FROM cart_items WHERE cart_id=?", (cart_id,)
                ).fetchall()
            result = dict(cart)
            result["items"] = [dict(i) for i in items]
            _json_resp(self, 200, result)
            return

        # GET /transactions
        if path == "/transactions":
            clause, params = "WHERE 1=1", []
            if "date" in qs:
                clause += " AND date(created_at,'unixepoch')=?"; params.append(qs["date"])
            if "payment_method" in qs:
                clause += " AND payment_method=?"; params.append(qs["payment_method"])
            limit = int(qs.get("limit", 100))
            with get_db() as conn:
                rows = conn.execute(
                    f"SELECT * FROM transactions {clause} ORDER BY created_at DESC LIMIT ?",
                    params + [limit],
                ).fetchall()
            _json_resp(self, 200, {"transactions": [dict(r) for r in rows]})
            return

        # GET /transactions/{tx_ref}
        if path.startswith("/transactions/") and len(path.split("/")) == 3:
            tx_ref = path.split("/")[2]
            with get_db() as conn:
                tx = conn.execute(
                    "SELECT * FROM transactions WHERE tx_ref=?", (tx_ref,)
                ).fetchone()
            if not tx:
                _json_resp(self, 404, {"error": "Transaction not found"}); return
            _json_resp(self, 200, dict(tx))
            return

        # GET /eod
        if path == "/eod":
            limit = int(qs.get("limit", 30))
            with get_db() as conn:
                rows = conn.execute(
                    "SELECT * FROM end_of_day ORDER BY date DESC LIMIT ?", (limit,)
                ).fetchall()
            _json_resp(self, 200, {"summaries": [dict(r) for r in rows]})
            return

        _json_resp(self, 404, {"error": "Not found"})

    # ---------------------------------------------------------------- POST --
    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        body = _read_body(self)
        parts = [p for p in path.split("/") if p]

        # POST /catalogue — admin-gated
        if path == "/catalogue":
            if not _is_admin(self.headers):
                _json_resp(self, 403, {"error": "Admin secret required"}); return
            required = ("sku", "name", "price")
            for field in required:
                if field not in body:
                    _json_resp(self, 400, {"error": f"Missing field: {field}"}); return
            now = time.time()
            try:
                with get_db() as conn:
                    conn.execute(
                        """INSERT INTO catalogue
                           (sku, name, description, category, price, cost,
                            tax_rate, stock, barcode, active, created_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)""",
                        (
                            body["sku"], body["name"],
                            body.get("description", ""),
                            body.get("category", "General"),
                            float(body["price"]),
                            float(body.get("cost", 0)),
                            float(body.get("tax_rate", 0.10)),
                            int(body.get("stock", 0)),
                            body.get("barcode", ""),
                            now,
                        ),
                    )
                _json_resp(self, 201, {"sku": body["sku"], "created": True})
            except sqlite3.IntegrityError:
                _json_resp(self, 409, {"error": "SKU already exists"})
            return

        # POST /carts
        if path == "/carts":
            cart_id = secrets.token_hex(12)
            now = time.time()
            with get_db() as conn:
                conn.execute(
                    """INSERT INTO carts
                       (cart_id, customer_email, staff_id, status,
                        subtotal, tax, total, discount_amount,
                        notes, created_at, updated_at)
                       VALUES (?, ?, ?, 'open', 0, 0, 0, 0, ?, ?, ?)""",
                    (
                        cart_id,
                        body.get("customer_email", ""),
                        body.get("staff_id", ""),
                        body.get("notes", ""),
                        now, now,
                    ),
                )
            _json_resp(self, 201, {"cart_id": cart_id, "status": "open"})
            return

        # POST /carts/{cart_id}/items
        if len(parts) == 3 and parts[0] == "carts" and parts[2] == "items":
            cart_id = parts[1]
            sku = body.get("sku", "")
            quantity = int(body.get("quantity", 1))
            if not sku or quantity < 1:
                _json_resp(self, 400, {"error": "sku and quantity >= 1 required"}); return
            with get_db() as conn:
                cart = conn.execute(
                    "SELECT * FROM carts WHERE cart_id=? AND status='open'", (cart_id,)
                ).fetchone()
                if not cart:
                    _json_resp(self, 404, {"error": "Open cart not found"}); return
                item = conn.execute(
                    "SELECT * FROM catalogue WHERE sku=? AND active=1", (sku,)
                ).fetchone()
                if not item:
                    _json_resp(self, 404, {"error": "Active catalogue item not found"}); return
                if item["stock"] < quantity:
                    _json_resp(self, 400, {"error": "Insufficient stock"}); return
                existing = conn.execute(
                    "SELECT * FROM cart_items WHERE cart_id=? AND sku=?",
                    (cart_id, sku),
                ).fetchone()
                if existing:
                    new_qty = existing["quantity"] + quantity
                    conn.execute(
                        "UPDATE cart_items SET quantity=? WHERE cart_id=? AND sku=?",
                        (new_qty, cart_id, sku),
                    )
                else:
                    line_total = round(
                        quantity * item["price"] * (1 + item["tax_rate"]), 2
                    )
                    conn.execute(
                        """INSERT INTO cart_items
                           (cart_id, sku, name, quantity, unit_price, tax_rate, line_total)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            cart_id, sku, item["name"], quantity,
                            item["price"], item["tax_rate"], line_total,
                        ),
                    )
                _update_line_total(conn, cart_id, sku)
                _recalculate_cart(conn, cart_id)
                cart_row = conn.execute(
                    "SELECT * FROM carts WHERE cart_id=?", (cart_id,)
                ).fetchone()
            _json_resp(self, 200, {
                "cart_id": cart_id,
                "subtotal": cart_row["subtotal"],
                "tax": cart_row["tax"],
                "total": cart_row["total"],
            })
            return

        # POST /carts/{cart_id}/discount — admin-gated
        if len(parts) == 3 and parts[0] == "carts" and parts[2] == "discount":
            if not _is_admin(self.headers):
                _json_resp(self, 403, {"error": "Admin secret required"}); return
            cart_id = parts[1]
            dtype = body.get("type", "fixed")
            value = float(body.get("value", 0))
            with get_db() as conn:
                cart = conn.execute(
                    "SELECT * FROM carts WHERE cart_id=? AND status='open'", (cart_id,)
                ).fetchone()
                if not cart:
                    _json_resp(self, 404, {"error": "Open cart not found"}); return
                if dtype == "percent":
                    discount = round(cart["subtotal"] * (value / 100), 2)
                else:
                    discount = round(value, 2)
                conn.execute(
                    "UPDATE carts SET discount_amount=?, updated_at=? WHERE cart_id=?",
                    (discount, time.time(), cart_id),
                )
                _recalculate_cart(conn, cart_id)
                cart_row = conn.execute(
                    "SELECT * FROM carts WHERE cart_id=?", (cart_id,)
                ).fetchone()
            _json_resp(self, 200, {
                "cart_id": cart_id,
                "discount_amount": cart_row["discount_amount"],
                "total": cart_row["total"],
            })
            return

        # POST /carts/{cart_id}/checkout
        if len(parts) == 3 and parts[0] == "carts" and parts[2] == "checkout":
            cart_id = parts[1]
            payment_method = body.get("payment_method", "cash").lower()
            customer_email = body.get("customer_email", "")
            if payment_method not in ("stripe", "square", "cash", "card"):
                _json_resp(self, 400, {"error": "Invalid payment_method"}); return
            with get_db() as conn:
                cart = conn.execute(
                    "SELECT * FROM carts WHERE cart_id=? AND status='open'", (cart_id,)
                ).fetchone()
                if not cart:
                    _json_resp(self, 404, {"error": "Open cart not found"}); return
                if cart["total"] <= 0:
                    _json_resp(self, 400, {"error": "Cart is empty or total is zero"}); return
                items = conn.execute(
                    "SELECT * FROM cart_items WHERE cart_id=?", (cart_id,)
                ).fetchall()
                items_list = [dict(i) for i in items]
                email = customer_email or cart["customer_email"] or ""
                total = cart["total"]

                # Process payment
                gateway = payment_method
                gateway_tx_id = ""
                status = "completed"

                if payment_method == "stripe":
                    result = _stripe_charge(total, f"FractalMesh POS — cart {cart_id}", email)
                    if not result["ok"]:
                        _json_resp(self, 502, {"error": result["error"]}); return
                    gateway_tx_id = result["gateway_tx_id"]
                elif payment_method == "square":
                    result = _square_charge(total, f"FractalMesh POS — cart {cart_id}", email)
                    if not result["ok"]:
                        _json_resp(self, 502, {"error": result["error"]}); return
                    gateway_tx_id = result["gateway_tx_id"]
                # cash/card: recorded locally, no external gateway call

                tx_ref = "TX-" + secrets.token_hex(8).upper()
                now = time.time()

                conn.execute(
                    """INSERT INTO transactions
                       (tx_ref, cart_id, payment_method, gateway, gateway_tx_id,
                        amount, currency, status, customer_email, receipt_sent, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, 'AUD', ?, ?, 0, ?)""",
                    (
                        tx_ref, cart_id, payment_method, gateway, gateway_tx_id,
                        round(total, 2), status, email, now,
                    ),
                )
                # Reduce stock
                for it in items_list:
                    conn.execute(
                        "UPDATE catalogue SET stock = MAX(stock - ?, 0) WHERE sku=?",
                        (it["quantity"], it["sku"]),
                    )
                # Mark cart completed
                conn.execute(
                    "UPDATE carts SET status='completed', completed_at=?, "
                    "updated_at=?, customer_email=? WHERE cart_id=?",
                    (now, now, email, cart_id),
                )
                cart_row = conn.execute(
                    "SELECT * FROM carts WHERE cart_id=?", (cart_id,)
                ).fetchone()

                # Send receipt
                receipt_sent = _send_receipt(email, tx_ref, dict(cart_row), items_list)
                if receipt_sent:
                    conn.execute(
                        "UPDATE transactions SET receipt_sent=1 WHERE tx_ref=?", (tx_ref,)
                    )

            _json_resp(self, 200, {
                "tx_ref": tx_ref,
                "cart_id": cart_id,
                "amount": round(total, 2),
                "currency": "AUD",
                "status": status,
                "gateway": gateway,
                "gateway_tx_id": gateway_tx_id,
                "receipt_sent": receipt_sent,
            })
            return

        # POST /transactions/{tx_ref}/refund — admin-gated
        if len(parts) == 3 and parts[0] == "transactions" and parts[2] == "refund":
            if not _is_admin(self.headers):
                _json_resp(self, 403, {"error": "Admin secret required"}); return
            tx_ref = parts[1]
            with get_db() as conn:
                tx = conn.execute(
                    "SELECT * FROM transactions WHERE tx_ref=?", (tx_ref,)
                ).fetchone()
                if not tx:
                    _json_resp(self, 404, {"error": "Transaction not found"}); return
                if tx["status"] not in ("completed",):
                    _json_resp(self, 400, {"error": "Transaction not eligible for refund"}); return
                refund_amount = float(body.get("amount", tx["amount"]))
                if refund_amount <= 0 or refund_amount > tx["amount"]:
                    _json_resp(self, 400, {"error": "Invalid refund amount"}); return

                gateway = tx["gateway"]
                refund_id = ""
                if gateway == "stripe":
                    r = _stripe_refund(tx["gateway_tx_id"], refund_amount)
                    if not r["ok"]:
                        _json_resp(self, 502, {"error": r["error"]}); return
                    refund_id = r["refund_id"]
                elif gateway == "square":
                    r = _square_refund(tx["gateway_tx_id"], refund_amount)
                    if not r["ok"]:
                        _json_resp(self, 502, {"error": r["error"]}); return
                    refund_id = r["refund_id"]

                new_status = "refunded" if refund_amount == tx["amount"] else "partial_refund"
                conn.execute(
                    "UPDATE transactions SET status=? WHERE tx_ref=?",
                    (new_status, tx_ref),
                )

            _json_resp(self, 200, {
                "tx_ref": tx_ref,
                "refund_amount": round(refund_amount, 2),
                "refund_id": refund_id,
                "status": new_status,
            })
            return

        # POST /eod — admin-gated
        if path == "/eod":
            if not _is_admin(self.headers):
                _json_resp(self, 403, {"error": "Admin secret required"}); return
            cash_counted = float(body.get("cash_counted", 0))
            closed_by = body.get("closed_by", "unknown")
            today = time.strftime("%Y-%m-%d")
            with get_db() as conn:
                existing = conn.execute(
                    "SELECT id FROM end_of_day WHERE date=?", (today,)
                ).fetchone()
                if existing:
                    _json_resp(self, 409, {"error": f"EOD already recorded for {today}"}); return
                sales_row = conn.execute(
                    "SELECT COALESCE(SUM(amount),0), COUNT(*) FROM transactions "
                    "WHERE status='completed' AND date(created_at,'unixepoch')=?",
                    (today,),
                ).fetchone()
                refund_row = conn.execute(
                    "SELECT COALESCE(SUM(amount),0) FROM transactions "
                    "WHERE status IN ('refunded','partial_refund') "
                    "AND date(created_at,'unixepoch')=?",
                    (today,),
                ).fetchone()
                cash_row = conn.execute(
                    "SELECT COALESCE(SUM(amount),0) FROM transactions "
                    "WHERE status='completed' AND payment_method='cash' "
                    "AND date(created_at,'unixepoch')=?",
                    (today,),
                ).fetchone()

                total_sales = round(sales_row[0], 2)
                total_tx = sales_row[1]
                total_refunds = round(refund_row[0], 2)
                cash_expected = round(cash_row[0], 2)
                variance = round(cash_counted - cash_expected, 2)
                now = time.time()

                conn.execute(
                    """INSERT INTO end_of_day
                       (date, total_sales, total_transactions, total_refunds,
                        cash_expected, cash_counted, variance, closed_by, closed_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        today, total_sales, total_tx, total_refunds,
                        cash_expected, cash_counted, variance, closed_by, now,
                    ),
                )
            _json_resp(self, 201, {
                "date": today,
                "total_sales": total_sales,
                "total_transactions": total_tx,
                "total_refunds": total_refunds,
                "cash_expected": cash_expected,
                "cash_counted": cash_counted,
                "variance": variance,
                "closed_by": closed_by,
            })
            return

        # POST /staff — admin-gated
        if path == "/staff":
            if not _is_admin(self.headers):
                _json_resp(self, 403, {"error": "Admin secret required"}); return
            name = body.get("name", "")
            pin = str(body.get("pin", ""))
            role = body.get("role", "cashier")
            if not name or not pin:
                _json_resp(self, 400, {"error": "name and pin are required"}); return
            staff_id = "STAFF-" + secrets.token_hex(4).upper()
            pin_hash = _hash_pin(pin)
            now = time.time()
            try:
                with get_db() as conn:
                    conn.execute(
                        "INSERT INTO staff (staff_id, name, pin_hash, role, active, created_at) "
                        "VALUES (?, ?, ?, ?, 1, ?)",
                        (staff_id, name, pin_hash, role, now),
                    )
                _json_resp(self, 201, {"staff_id": staff_id, "name": name, "role": role})
            except sqlite3.IntegrityError:
                _json_resp(self, 409, {"error": "Staff ID collision, retry"})
            return

        # POST /staff/auth
        if path == "/staff/auth":
            staff_id = body.get("staff_id", "")
            pin = str(body.get("pin", ""))
            if not staff_id or not pin:
                _json_resp(self, 400, {"error": "staff_id and pin required"}); return
            with get_db() as conn:
                row = conn.execute(
                    "SELECT * FROM staff WHERE staff_id=? AND active=1", (staff_id,)
                ).fetchone()
            if not row:
                _json_resp(self, 200, {"valid": False, "reason": "Unknown staff ID"}); return
            valid = _verify_pin(pin, row["pin_hash"])
            if valid:
                _json_resp(self, 200, {
                    "valid": True,
                    "staff_id": row["staff_id"],
                    "name": row["name"],
                    "role": row["role"],
                })
            else:
                _json_resp(self, 200, {"valid": False, "reason": "Incorrect PIN"})
            return

        _json_resp(self, 404, {"error": "Not found"})

    # --------------------------------------------------------------- PUT --
    def do_PUT(self):
        path = self.path.split("?")[0].rstrip("/")
        body = _read_body(self)
        parts = [p for p in path.split("/") if p]

        # PUT /catalogue/{sku} — admin-gated
        if len(parts) == 2 and parts[0] == "catalogue":
            if not _is_admin(self.headers):
                _json_resp(self, 403, {"error": "Admin secret required"}); return
            sku = parts[1]
            with get_db() as conn:
                row = conn.execute(
                    "SELECT * FROM catalogue WHERE sku=?", (sku,)
                ).fetchone()
                if not row:
                    _json_resp(self, 404, {"error": "SKU not found"}); return
                updatable = ("name", "description", "category", "price", "cost",
                             "tax_rate", "stock", "barcode", "active")
                sets, vals = [], []
                for field in updatable:
                    if field in body:
                        sets.append(f"{field}=?")
                        val = body[field]
                        if field in ("price", "cost", "tax_rate"):
                            val = float(val)
                        elif field in ("stock", "active"):
                            val = int(val)
                        vals.append(val)
                if not sets:
                    _json_resp(self, 400, {"error": "No updatable fields provided"}); return
                vals.append(sku)
                conn.execute(
                    f"UPDATE catalogue SET {', '.join(sets)} WHERE sku=?", vals
                )
                updated = conn.execute(
                    "SELECT * FROM catalogue WHERE sku=?", (sku,)
                ).fetchone()
            _json_resp(self, 200, dict(updated))
            return

        _json_resp(self, 404, {"error": "Not found"})

    # ------------------------------------------------------------ DELETE --
    def do_DELETE(self):
        path = self.path.split("?")[0].rstrip("/")
        parts = [p for p in path.split("/") if p]

        # DELETE /carts/{cart_id}/items/{sku}
        if len(parts) == 4 and parts[0] == "carts" and parts[2] == "items":
            cart_id = parts[1]
            sku = parts[3]
            with get_db() as conn:
                cart = conn.execute(
                    "SELECT * FROM carts WHERE cart_id=? AND status='open'", (cart_id,)
                ).fetchone()
                if not cart:
                    _json_resp(self, 404, {"error": "Open cart not found"}); return
                deleted = conn.execute(
                    "DELETE FROM cart_items WHERE cart_id=? AND sku=?", (cart_id, sku)
                ).rowcount
                if deleted == 0:
                    _json_resp(self, 404, {"error": "Item not in cart"}); return
                _recalculate_cart(conn, cart_id)
                cart_row = conn.execute(
                    "SELECT * FROM carts WHERE cart_id=?", (cart_id,)
                ).fetchone()
            _json_resp(self, 200, {
                "cart_id": cart_id,
                "removed_sku": sku,
                "subtotal": cart_row["subtotal"],
                "tax": cart_row["tax"],
                "total": cart_row["total"],
            })
            return

        _json_resp(self, 404, {"error": "Not found"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    init_db()

    bg = threading.Thread(target=_background_loop, daemon=True, name="pos-bg")
    bg.start()

    server = HTTPServer(("0.0.0.0", PORT), POSHandler)
    print(f"[{AGENT_NAME}] Listening on port {PORT} | DB: {DB_PATH}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"[{AGENT_NAME}] Shutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
