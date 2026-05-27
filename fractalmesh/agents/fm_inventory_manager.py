#!/usr/bin/env python3
"""
FractalMesh OMEGA Titan — Product Inventory & Fulfillment Manager
Port: 7859

Tracks digital and physical products, stock levels, orders, and fulfillment
status. Integrates with Printful for print-on-demand and Shopify for ecommerce.

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
import json
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
AGENT_NAME = "fm_inventory_manager"
PORT = int(os.environ.get("INVENTORY_MANAGER_PORT", "7859"))

PRINTFUL_API_KEY    = os.environ.get("PRINTFUL_API_KEY", "")
SHOPIFY_STORE_URL   = os.environ.get("SHOPIFY_STORE_URL", "")
SHOPIFY_ACCESS_TOKEN = os.environ.get("SHOPIFY_ACCESS_TOKEN", "")
ADMIN_SECRET        = os.environ.get("ADMIN_SECRET", "")

PRINTFUL_BASE = "https://api.printful.com"
SYNC_INTERVAL = 3600  # seconds

ROOT    = Path(os.environ.get("FRACTALMESH_HOME", Path.home() / "fmsaas"))
DB_PATH = ROOT / "database" / "sovereign.db"

START_TIME = time.time()

# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------
_LOG_DIR = ROOT / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / "fm_inventory_manager.log"


def _log(level: str, msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    line = f"{ts} [{AGENT_NAME}] {level.upper()} {msg}\n"
    try:
        with open(_LOG_FILE, "a") as fh:
            fh.write(line)
    except OSError:
        pass
    print(line, end="")


# ---------------------------------------------------------------------------
# Database bootstrap
# ---------------------------------------------------------------------------

def _db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB_PATH), timeout=10, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    con.row_factory = sqlite3.Row
    return con


def init_db() -> None:
    con = _db()
    con.executescript("""
        CREATE TABLE IF NOT EXISTS products (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            sku                 TEXT UNIQUE NOT NULL,
            name                TEXT NOT NULL,
            description         TEXT NOT NULL DEFAULT '',
            product_type        TEXT NOT NULL DEFAULT 'physical',
            price               REAL NOT NULL DEFAULT 0.0,
            cost                REAL NOT NULL DEFAULT 0.0,
            stock_quantity      INTEGER NOT NULL DEFAULT 0,
            low_stock_threshold INTEGER NOT NULL DEFAULT 5,
            printful_product_id TEXT,
            shopify_product_id  TEXT,
            active              INTEGER NOT NULL DEFAULT 1,
            created_at          REAL NOT NULL,
            updated_at          REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS orders (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            order_ref        TEXT UNIQUE NOT NULL,
            source           TEXT NOT NULL DEFAULT 'manual',
            customer_email   TEXT NOT NULL DEFAULT '',
            customer_name    TEXT NOT NULL DEFAULT '',
            status           TEXT NOT NULL DEFAULT 'pending',
            total_amount     REAL NOT NULL DEFAULT 0.0,
            currency         TEXT NOT NULL DEFAULT 'AUD',
            shopify_order_id TEXT,
            printful_order_id TEXT,
            created_at       REAL NOT NULL,
            updated_at       REAL NOT NULL,
            fulfilled_at     REAL
        );

        CREATE TABLE IF NOT EXISTS order_items (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id   INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            sku        TEXT NOT NULL,
            quantity   INTEGER NOT NULL,
            unit_price REAL NOT NULL,
            line_total  REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS fulfillments (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id        INTEGER NOT NULL,
            method          TEXT NOT NULL DEFAULT 'manual',
            tracking_number TEXT,
            carrier         TEXT,
            status          TEXT NOT NULL DEFAULT 'pending',
            shipped_at      REAL,
            delivered_at    REAL,
            created_at      REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS stock_movements (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id    INTEGER NOT NULL,
            movement_type TEXT NOT NULL,
            quantity      INTEGER NOT NULL,
            reason        TEXT NOT NULL DEFAULT '',
            reference_id  TEXT,
            created_at    REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_products_sku ON products(sku);
        CREATE INDEX IF NOT EXISTS idx_orders_ref ON orders(order_ref);
        CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
        CREATE INDEX IF NOT EXISTS idx_order_items_order ON order_items(order_id);
        CREATE INDEX IF NOT EXISTS idx_fulfillments_order ON fulfillments(order_id);
        CREATE INDEX IF NOT EXISTS idx_stock_movements_product ON stock_movements(product_id);
    """)
    con.commit()
    con.close()
    _log("info", "Database initialised")


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

def _check_admin(handler: "InventoryHandler") -> bool:
    token = handler.headers.get("X-Admin-Secret", "")
    if not ADMIN_SECRET:
        return True  # unconfigured → open
    if not token:
        return False
    expected = hashlib.sha256(ADMIN_SECRET.encode()).hexdigest()
    provided = hashlib.sha256(token.encode()).hexdigest()
    return expected == provided


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _http_get(url: str, headers: dict) -> dict:
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()
        _log("error", f"GET {url} → HTTP {exc.code}: {body[:200]}")
        return {}
    except Exception as exc:
        _log("error", f"GET {url} → {exc}")
        return {}


def _http_post(url: str, headers: dict, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()
        _log("error", f"POST {url} → HTTP {exc.code}: {body[:200]}")
        return {}
    except Exception as exc:
        _log("error", f"POST {url} → {exc}")
        return {}


# ---------------------------------------------------------------------------
# Printful helpers
# ---------------------------------------------------------------------------

def _printful_headers() -> dict:
    return {"Authorization": f"Bearer {PRINTFUL_API_KEY}", "Content-Type": "application/json"}


def _printful_get(path: str) -> dict:
    url = f"{PRINTFUL_BASE}/{path.lstrip('/')}"
    return _http_get(url, _printful_headers())


def _printful_post(path: str, payload: dict) -> dict:
    url = f"{PRINTFUL_BASE}/{path.lstrip('/')}"
    return _http_post(url, _printful_headers(), payload)


# ---------------------------------------------------------------------------
# Shopify helpers
# ---------------------------------------------------------------------------

def _shopify_base() -> str:
    store = SHOPIFY_STORE_URL.rstrip("/")
    if not store.startswith("http"):
        store = f"https://{store}"
    return f"{store}/admin/api/2024-01"


def _shopify_headers() -> dict:
    return {
        "X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN,
        "Content-Type": "application/json",
    }


def _shopify_get(path: str) -> dict:
    url = f"{_shopify_base()}/{path.lstrip('/')}"
    return _http_get(url, _shopify_headers())


def _shopify_post(path: str, payload: dict) -> dict:
    url = f"{_shopify_base()}/{path.lstrip('/')}"
    return _http_post(url, _shopify_headers(), payload)


# ---------------------------------------------------------------------------
# Sync logic (background thread)
# ---------------------------------------------------------------------------

def _sync_printful_products() -> None:
    if not PRINTFUL_API_KEY:
        _log("warning", "PRINTFUL_API_KEY not set — skipping Printful sync")
        return
    _log("info", "Syncing Printful product catalog …")
    data = _printful_get("/store/products")
    items = data.get("result", [])
    if not items:
        _log("warning", "Printful returned no products or error")
        return
    now = time.time()
    con = _db()
    upserted = 0
    for item in items:
        pid = str(item.get("id", ""))
        name = item.get("name", "Unknown")
        sku = f"PF-{pid}"
        try:
            con.execute("""
                INSERT INTO products (sku, name, description, product_type, price, cost,
                    printful_product_id, active, created_at, updated_at)
                VALUES (?, ?, ?, 'pod', 0.0, 0.0, ?, 1, ?, ?)
                ON CONFLICT(sku) DO UPDATE SET
                    name = excluded.name,
                    printful_product_id = excluded.printful_product_id,
                    updated_at = excluded.updated_at
            """, (sku, name, "", pid, now, now))
            upserted += 1
        except Exception as exc:
            _log("error", f"Upsert Printful product {pid}: {exc}")
    con.commit()
    con.close()
    _log("info", f"Printful sync complete — {upserted} products upserted")


def _check_low_stock() -> None:
    _log("info", "Checking low-stock levels …")
    con = _db()
    rows = con.execute("""
        SELECT sku, name, stock_quantity, low_stock_threshold
        FROM products
        WHERE active = 1 AND stock_quantity <= low_stock_threshold
        ORDER BY stock_quantity ASC
    """).fetchall()
    con.close()
    for row in rows:
        _log("warning",
             f"LOW STOCK: {row['sku']} '{row['name']}' — "
             f"qty={row['stock_quantity']} threshold={row['low_stock_threshold']}")


def _sync_shopify_orders() -> None:
    if not SHOPIFY_STORE_URL or not SHOPIFY_ACCESS_TOKEN:
        _log("warning", "Shopify credentials not set — skipping order sync")
        return
    _log("info", "Syncing Shopify orders …")
    data = _shopify_get("/orders.json?status=any&limit=50")
    orders = data.get("orders", [])
    if not orders:
        _log("info", "No Shopify orders returned")
        return
    now = time.time()
    con = _db()
    updated = 0
    for o in orders:
        shopify_id = str(o.get("id", ""))
        order_ref = f"SH-{shopify_id}"
        email = o.get("email", "")
        name = f"{o.get('billing_address', {}).get('first_name', '')} {o.get('billing_address', {}).get('last_name', '')}".strip()
        status = o.get("financial_status", "pending")
        total = float(o.get("total_price", 0))
        currency = o.get("currency", "AUD")
        try:
            con.execute("""
                INSERT INTO orders (order_ref, source, customer_email, customer_name,
                    status, total_amount, currency, shopify_order_id, created_at, updated_at)
                VALUES (?, 'shopify', ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(order_ref) DO UPDATE SET
                    status = excluded.status,
                    total_amount = excluded.total_amount,
                    customer_email = excluded.customer_email,
                    customer_name = excluded.customer_name,
                    updated_at = excluded.updated_at
            """, (order_ref, email, name, status, total, currency, shopify_id, now, now))
            updated += 1
        except Exception as exc:
            _log("error", f"Upsert Shopify order {shopify_id}: {exc}")
    con.commit()
    con.close()
    _log("info", f"Shopify order sync complete — {updated} orders processed")


def _run_sync() -> None:
    _sync_printful_products()
    _check_low_stock()
    _sync_shopify_orders()


def _background_worker() -> None:
    _log("info", "Background sync thread started")
    # Initial sync shortly after startup
    time.sleep(10)
    while True:
        try:
            _run_sync()
        except Exception as exc:
            _log("error", f"Background sync error: {exc}")
        time.sleep(SYNC_INTERVAL)


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------

def _json_resp(handler: "InventoryHandler", code: int, payload) -> None:
    body = json.dumps(payload, default=str).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _ok(handler: "InventoryHandler", data) -> None:
    _json_resp(handler, 200, {"status": "ok", "data": data})


def _created(handler: "InventoryHandler", data) -> None:
    _json_resp(handler, 201, {"status": "created", "data": data})


def _err(handler: "InventoryHandler", code: int, msg: str) -> None:
    _json_resp(handler, code, {"status": "error", "message": msg})


def _read_body(handler: "InventoryHandler") -> dict:
    length = int(handler.headers.get("Content-Length", 0))
    if length == 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw.decode())
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

def _handle_health(handler: "InventoryHandler") -> None:
    con = _db()
    product_count = con.execute("SELECT COUNT(*) FROM products WHERE active=1").fetchone()[0]
    order_count   = con.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    pending_count = con.execute("SELECT COUNT(*) FROM orders WHERE status='pending'").fetchone()[0]
    con.close()
    _ok(handler, {
        "agent": AGENT_NAME,
        "uptime_seconds": round(time.time() - START_TIME, 2),
        "products_active": product_count,
        "orders_total": order_count,
        "orders_pending": pending_count,
        "port": PORT,
    })


def _handle_list_products(handler: "InventoryHandler", qs: dict) -> None:
    clauses = ["1=1"]
    params: list = []

    if "product_type" in qs:
        clauses.append("product_type = ?")
        params.append(qs["product_type"])

    if "active" in qs:
        clauses.append("active = ?")
        params.append(1 if qs["active"].lower() in ("1", "true", "yes") else 0)
    else:
        clauses.append("active = 1")

    if qs.get("low_stock", "").lower() in ("1", "true", "yes"):
        clauses.append("stock_quantity <= low_stock_threshold")

    where = " AND ".join(clauses)
    con = _db()
    rows = con.execute(
        f"SELECT * FROM products WHERE {where} ORDER BY name ASC", params
    ).fetchall()
    con.close()
    _ok(handler, [dict(r) for r in rows])


def _handle_get_product(handler: "InventoryHandler", sku: str) -> None:
    con = _db()
    row = con.execute("SELECT * FROM products WHERE sku = ?", (sku,)).fetchone()
    if not row:
        con.close()
        _err(handler, 404, f"Product '{sku}' not found")
        return
    product = dict(row)
    movements = con.execute("""
        SELECT * FROM stock_movements WHERE product_id = ?
        ORDER BY created_at DESC LIMIT 20
    """, (product["id"],)).fetchall()
    con.close()
    product["recent_movements"] = [dict(m) for m in movements]
    _ok(handler, product)


def _handle_list_orders(handler: "InventoryHandler", qs: dict) -> None:
    clauses = ["1=1"]
    params: list = []

    if "status" in qs:
        clauses.append("status = ?")
        params.append(qs["status"])

    if "source" in qs:
        clauses.append("source = ?")
        params.append(qs["source"])

    try:
        limit = min(int(qs.get("limit", "50")), 200)
    except ValueError:
        limit = 50

    where = " AND ".join(clauses)
    con = _db()
    rows = con.execute(
        f"SELECT * FROM orders WHERE {where} ORDER BY created_at DESC LIMIT ?",
        params + [limit]
    ).fetchall()
    con.close()
    _ok(handler, [dict(r) for r in rows])


def _handle_get_order(handler: "InventoryHandler", order_ref: str) -> None:
    con = _db()
    row = con.execute("SELECT * FROM orders WHERE order_ref = ?", (order_ref,)).fetchone()
    if not row:
        con.close()
        _err(handler, 404, f"Order '{order_ref}' not found")
        return
    order = dict(row)
    items = con.execute(
        "SELECT * FROM order_items WHERE order_id = ?", (order["id"],)
    ).fetchall()
    fulfillments = con.execute(
        "SELECT * FROM fulfillments WHERE order_id = ? ORDER BY created_at DESC",
        (order["id"],)
    ).fetchall()
    con.close()
    order["items"] = [dict(i) for i in items]
    order["fulfillments"] = [dict(f) for f in fulfillments]
    _ok(handler, order)


def _handle_list_fulfillments(handler: "InventoryHandler", qs: dict) -> None:
    con = _db()
    rows = con.execute("""
        SELECT f.*, o.order_ref, o.customer_email, o.customer_name
        FROM fulfillments f
        JOIN orders o ON o.id = f.order_id
        WHERE f.status NOT IN ('delivered', 'cancelled')
        ORDER BY f.created_at DESC
        LIMIT 100
    """).fetchall()
    con.close()
    _ok(handler, [dict(r) for r in rows])


def _handle_list_stock_movements(handler: "InventoryHandler", qs: dict) -> None:
    clauses = ["1=1"]
    params: list = []

    if "product_id" in qs:
        try:
            clauses.append("product_id = ?")
            params.append(int(qs["product_id"]))
        except ValueError:
            pass

    try:
        limit = min(int(qs.get("limit", "50")), 500)
    except ValueError:
        limit = 50

    where = " AND ".join(clauses)
    con = _db()
    rows = con.execute(
        f"SELECT sm.*, p.sku, p.name FROM stock_movements sm "
        f"JOIN products p ON p.id = sm.product_id "
        f"WHERE {where} ORDER BY sm.created_at DESC LIMIT ?",
        params + [limit]
    ).fetchall()
    con.close()
    _ok(handler, [dict(r) for r in rows])


def _handle_analytics(handler: "InventoryHandler") -> None:
    con = _db()

    # Revenue by product (top 20)
    revenue_by_product = con.execute("""
        SELECT p.sku, p.name, SUM(oi.line_total) AS total_revenue,
               SUM(oi.quantity) AS units_sold
        FROM order_items oi
        JOIN products p ON p.id = oi.product_id
        JOIN orders o ON o.id = oi.order_id
        WHERE o.status NOT IN ('cancelled', 'refunded')
        GROUP BY p.id
        ORDER BY total_revenue DESC
        LIMIT 20
    """).fetchall()

    # Order volume by day (last 30 days)
    cutoff = time.time() - 30 * 86400
    order_volume = con.execute("""
        SELECT DATE(created_at, 'unixepoch') AS day, COUNT(*) AS order_count,
               SUM(total_amount) AS revenue
        FROM orders
        WHERE created_at >= ? AND status NOT IN ('cancelled', 'refunded')
        GROUP BY day
        ORDER BY day DESC
    """, (cutoff,)).fetchall()

    # Fulfillment rate
    total_orders = con.execute("SELECT COUNT(*) FROM orders WHERE status NOT IN ('cancelled')").fetchone()[0]
    fulfilled_orders = con.execute(
        "SELECT COUNT(*) FROM orders WHERE status IN ('fulfilled', 'delivered', 'completed')"
    ).fetchone()[0]
    fulfillment_rate = round(fulfilled_orders / total_orders * 100, 2) if total_orders else 0.0

    # Average order value
    avg_order = con.execute(
        "SELECT AVG(total_amount) FROM orders WHERE status NOT IN ('cancelled', 'refunded')"
    ).fetchone()[0] or 0.0

    con.close()
    _ok(handler, {
        "revenue_by_product": [dict(r) for r in revenue_by_product],
        "order_volume_by_day": [dict(r) for r in order_volume],
        "fulfillment_rate_pct": fulfillment_rate,
        "avg_order_value": round(avg_order, 2),
        "total_orders": total_orders,
        "fulfilled_orders": fulfilled_orders,
    })


def _handle_create_product(handler: "InventoryHandler") -> None:
    if not _check_admin(handler):
        _err(handler, 403, "Admin authentication required")
        return
    body = _read_body(handler)
    required = ["sku", "name"]
    for field in required:
        if not body.get(field):
            _err(handler, 400, f"Missing required field: {field}")
            return
    now = time.time()
    con = _db()
    try:
        con.execute("""
            INSERT INTO products (sku, name, description, product_type, price, cost,
                stock_quantity, low_stock_threshold, printful_product_id,
                shopify_product_id, active, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
        """, (
            body["sku"],
            body["name"],
            body.get("description", ""),
            body.get("product_type", "physical"),
            float(body.get("price", 0)),
            float(body.get("cost", 0)),
            int(body.get("stock_quantity", 0)),
            int(body.get("low_stock_threshold", 5)),
            body.get("printful_product_id"),
            body.get("shopify_product_id"),
            now, now,
        ))
        con.commit()
        row = con.execute("SELECT * FROM products WHERE sku = ?", (body["sku"],)).fetchone()
        con.close()
        _created(handler, dict(row))
    except sqlite3.IntegrityError:
        con.close()
        _err(handler, 409, f"Product with SKU '{body['sku']}' already exists")
    except Exception as exc:
        con.close()
        _err(handler, 500, str(exc))


def _handle_update_product(handler: "InventoryHandler", sku: str) -> None:
    if not _check_admin(handler):
        _err(handler, 403, "Admin authentication required")
        return
    body = _read_body(handler)
    if not body:
        _err(handler, 400, "Empty request body")
        return
    con = _db()
    row = con.execute("SELECT * FROM products WHERE sku = ?", (sku,)).fetchone()
    if not row:
        con.close()
        _err(handler, 404, f"Product '{sku}' not found")
        return

    allowed_fields = {
        "name", "description", "product_type", "price", "cost",
        "stock_quantity", "low_stock_threshold", "printful_product_id",
        "shopify_product_id", "active"
    }
    updates = {k: v for k, v in body.items() if k in allowed_fields}
    if not updates:
        con.close()
        _err(handler, 400, "No updatable fields provided")
        return

    # Record stock movement if stock_quantity changed
    old_qty = row["stock_quantity"]
    new_qty = updates.get("stock_quantity", old_qty)
    now = time.time()

    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [now, sku]
    con.execute(f"UPDATE products SET {set_clause}, updated_at = ? WHERE sku = ?", values)

    if "stock_quantity" in updates and int(new_qty) != old_qty:
        delta = int(new_qty) - old_qty
        movement_type = "adjustment"
        product_id = row["id"]
        con.execute("""
            INSERT INTO stock_movements (product_id, movement_type, quantity, reason, created_at)
            VALUES (?, ?, ?, 'manual update', ?)
        """, (product_id, movement_type, delta, now))

    con.commit()
    updated = con.execute("SELECT * FROM products WHERE sku = ?", (sku,)).fetchone()
    con.close()
    _ok(handler, dict(updated))


def _handle_create_order(handler: "InventoryHandler") -> None:
    body = _read_body(handler)
    required = ["customer_email", "items"]
    for field in required:
        if not body.get(field):
            _err(handler, 400, f"Missing required field: {field}")
            return

    items = body["items"]
    if not isinstance(items, list) or not items:
        _err(handler, 400, "items must be a non-empty list")
        return

    now = time.time()
    order_ref = f"MAN-{int(now)}-{hashlib.md5(body['customer_email'].encode()).hexdigest()[:6].upper()}"

    con = _db()
    total = 0.0
    resolved_items = []
    for item in items:
        sku = item.get("sku")
        qty = int(item.get("quantity", 1))
        unit_price = float(item.get("unit_price", 0))
        if not sku:
            con.close()
            _err(handler, 400, "Each item must have a 'sku'")
            return
        row = con.execute("SELECT * FROM products WHERE sku = ? AND active = 1", (sku,)).fetchone()
        if not row:
            con.close()
            _err(handler, 404, f"Product '{sku}' not found or inactive")
            return
        line_total = unit_price * qty
        total += line_total
        resolved_items.append({
            "product_id": row["id"],
            "sku": sku,
            "quantity": qty,
            "unit_price": unit_price,
            "line_total": line_total,
        })

    try:
        con.execute("""
            INSERT INTO orders (order_ref, source, customer_email, customer_name,
                status, total_amount, currency, created_at, updated_at)
            VALUES (?, 'manual', ?, ?, 'pending', ?, ?, ?, ?)
        """, (
            order_ref,
            body["customer_email"],
            body.get("customer_name", ""),
            round(total, 2),
            body.get("currency", "AUD"),
            now, now,
        ))
        order_id = con.execute("SELECT id FROM orders WHERE order_ref = ?", (order_ref,)).fetchone()["id"]

        for item in resolved_items:
            con.execute("""
                INSERT INTO order_items (order_id, product_id, sku, quantity, unit_price, line_total)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (order_id, item["product_id"], item["sku"],
                  item["quantity"], item["unit_price"], item["line_total"]))
            # Deduct stock for physical/pod products
            prod = con.execute("SELECT product_type, stock_quantity FROM products WHERE id = ?",
                               (item["product_id"],)).fetchone()
            if prod and prod["product_type"] in ("physical",):
                new_qty = max(0, prod["stock_quantity"] - item["quantity"])
                con.execute("UPDATE products SET stock_quantity = ?, updated_at = ? WHERE id = ?",
                            (new_qty, now, item["product_id"]))
                con.execute("""
                    INSERT INTO stock_movements (product_id, movement_type, quantity, reason, reference_id, created_at)
                    VALUES (?, 'sale', ?, 'order placed', ?, ?)
                """, (item["product_id"], -item["quantity"], order_ref, now))

        con.commit()
        order = con.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
        order_items = con.execute("SELECT * FROM order_items WHERE order_id = ?", (order_id,)).fetchall()
        con.close()
        result = dict(order)
        result["items"] = [dict(i) for i in order_items]
        _created(handler, result)
    except Exception as exc:
        con.close()
        _log("error", f"Create order error: {exc}")
        _err(handler, 500, str(exc))


def _handle_fulfill_order(handler: "InventoryHandler", order_ref: str) -> None:
    if not _check_admin(handler):
        _err(handler, 403, "Admin authentication required")
        return
    body = _read_body(handler)
    now = time.time()
    con = _db()
    order = con.execute("SELECT * FROM orders WHERE order_ref = ?", (order_ref,)).fetchone()
    if not order:
        con.close()
        _err(handler, 404, f"Order '{order_ref}' not found")
        return
    if order["status"] in ("fulfilled", "delivered", "cancelled"):
        con.close()
        _err(handler, 409, f"Order is already in status '{order['status']}'")
        return

    items = con.execute(
        "SELECT oi.*, p.product_type, p.printful_product_id FROM order_items oi "
        "JOIN products p ON p.id = oi.product_id WHERE oi.order_id = ?",
        (order["id"],)
    ).fetchall()
    con.close()

    pod_items = [i for i in items if i["product_type"] == "pod"]
    digital_items = [i for i in items if i["product_type"] == "digital"]

    printful_order_id = None
    fulfillment_method = "manual"

    # Attempt Printful fulfillment for POD items
    if pod_items and PRINTFUL_API_KEY:
        recipient = {
            "name": order["customer_name"] or order["customer_email"],
            "email": order["customer_email"],
            "address1": body.get("address1", ""),
            "city": body.get("city", ""),
            "state_code": body.get("state_code", ""),
            "country_code": body.get("country_code", "AU"),
            "zip": body.get("zip", ""),
        }
        pf_items = []
        for item in pod_items:
            pf_items.append({
                "sync_variant_id": item["printful_product_id"] or item["sku"],
                "quantity": item["quantity"],
            })
        pf_payload = {"recipient": recipient, "items": pf_items}
        pf_result = _printful_post("/orders", pf_payload)
        if pf_result.get("result"):
            printful_order_id = str(pf_result["result"].get("id", ""))
            fulfillment_method = "printful"
            _log("info", f"Printful order created: {printful_order_id} for {order_ref}")
        else:
            _log("warning", f"Printful order creation returned no result for {order_ref}")
    elif digital_items:
        fulfillment_method = "digital"

    # Update order and insert fulfillment record
    con = _db()
    try:
        status = "fulfilled"
        con.execute("""
            UPDATE orders SET status = ?, printful_order_id = ?, fulfilled_at = ?, updated_at = ?
            WHERE order_ref = ?
        """, (status, printful_order_id, now, now, order_ref))
        con.execute("""
            INSERT INTO fulfillments (order_id, method, status, created_at)
            VALUES (?, ?, 'shipped', ?)
        """, (order["id"], fulfillment_method, now))
        con.commit()
        updated_order = con.execute("SELECT * FROM orders WHERE order_ref = ?", (order_ref,)).fetchone()
        fulfillments = con.execute(
            "SELECT * FROM fulfillments WHERE order_id = ? ORDER BY created_at DESC",
            (order["id"],)
        ).fetchall()
        con.close()
        result = dict(updated_order)
        result["fulfillments"] = [dict(f) for f in fulfillments]
        result["fulfillment_method"] = fulfillment_method
        if printful_order_id:
            result["printful_order_id"] = printful_order_id
        _ok(handler, result)
    except Exception as exc:
        con.close()
        _log("error", f"Fulfill order error: {exc}")
        _err(handler, 500, str(exc))


def _handle_adjust_stock(handler: "InventoryHandler") -> None:
    if not _check_admin(handler):
        _err(handler, 403, "Admin authentication required")
        return
    body = _read_body(handler)
    sku = body.get("sku")
    quantity = body.get("quantity")
    reason = body.get("reason", "manual adjustment")

    if not sku:
        _err(handler, 400, "Missing required field: sku")
        return
    if quantity is None:
        _err(handler, 400, "Missing required field: quantity")
        return

    try:
        quantity = int(quantity)
    except (TypeError, ValueError):
        _err(handler, 400, "quantity must be an integer")
        return

    now = time.time()
    con = _db()
    row = con.execute("SELECT * FROM products WHERE sku = ?", (sku,)).fetchone()
    if not row:
        con.close()
        _err(handler, 404, f"Product '{sku}' not found")
        return

    new_qty = max(0, row["stock_quantity"] + quantity)
    movement_type = "adjustment" if quantity >= 0 else "deduction"

    try:
        con.execute("UPDATE products SET stock_quantity = ?, updated_at = ? WHERE sku = ?",
                    (new_qty, now, sku))
        con.execute("""
            INSERT INTO stock_movements (product_id, movement_type, quantity, reason, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (row["id"], movement_type, quantity, reason, now))
        con.commit()
        updated = con.execute("SELECT * FROM products WHERE sku = ?", (sku,)).fetchone()
        con.close()
        _ok(handler, {
            "sku": sku,
            "previous_quantity": row["stock_quantity"],
            "adjustment": quantity,
            "new_quantity": new_qty,
            "reason": reason,
            "product": dict(updated),
        })
    except Exception as exc:
        con.close()
        _log("error", f"Stock adjust error: {exc}")
        _err(handler, 500, str(exc))


def _handle_sync(handler: "InventoryHandler") -> None:
    if not _check_admin(handler):
        _err(handler, 403, "Admin authentication required")
        return
    _log("info", "Manual sync triggered")
    t = threading.Thread(target=_run_sync, daemon=True)
    t.start()
    _ok(handler, {"message": "Sync triggered in background"})


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

def _parse_qs(path: str) -> tuple[str, dict]:
    """Return (path_without_qs, query_dict)."""
    if "?" in path:
        p, qs = path.split("?", 1)
        params: dict = {}
        for pair in qs.split("&"):
            if "=" in pair:
                k, v = pair.split("=", 1)
                params[k] = urllib.request.pathname2url(v) if "%" in v else v
        return p, params
    return path, {}


class InventoryHandler(BaseHTTPRequestHandler):
    server_version = "FractalMesh/1.0"

    def log_message(self, fmt, *args):  # suppress default stderr logging
        _log("info", f"{self.address_string()} {fmt % args}")

    def do_GET(self):
        path, qs = _parse_qs(self.path)
        parts = [p for p in path.split("/") if p]

        try:
            if path == "/health":
                _handle_health(self)
            elif path == "/products":
                _handle_list_products(self, qs)
            elif len(parts) == 2 and parts[0] == "products":
                _handle_get_product(self, parts[1])
            elif path == "/orders":
                _handle_list_orders(self, qs)
            elif len(parts) == 2 and parts[0] == "orders":
                _handle_get_order(self, parts[1])
            elif path == "/fulfillments":
                _handle_list_fulfillments(self, qs)
            elif path == "/stock_movements":
                _handle_list_stock_movements(self, qs)
            elif path == "/analytics":
                _handle_analytics(self)
            else:
                _err(self, 404, f"Unknown route: {path}")
        except Exception as exc:
            _log("error", f"GET {path} unhandled: {exc}")
            _err(self, 500, "Internal server error")

    def do_POST(self):
        path, qs = _parse_qs(self.path)
        parts = [p for p in path.split("/") if p]

        try:
            if path == "/products":
                _handle_create_product(self)
            elif path == "/orders":
                _handle_create_order(self)
            elif len(parts) == 3 and parts[0] == "orders" and parts[2] == "fulfill":
                _handle_fulfill_order(self, parts[1])
            elif path == "/stock/adjust":
                _handle_adjust_stock(self)
            elif path == "/sync":
                _handle_sync(self)
            else:
                _err(self, 404, f"Unknown route: {path}")
        except Exception as exc:
            _log("error", f"POST {path} unhandled: {exc}")
            _err(self, 500, "Internal server error")

    def do_PUT(self):
        path, qs = _parse_qs(self.path)
        parts = [p for p in path.split("/") if p]

        try:
            if len(parts) == 2 and parts[0] == "products":
                _handle_update_product(self, parts[1])
            else:
                _err(self, 404, f"Unknown route: {path}")
        except Exception as exc:
            _log("error", f"PUT {path} unhandled: {exc}")
            _err(self, 500, "Internal server error")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _log("info", f"Starting {AGENT_NAME} on port {PORT}")
    init_db()

    # Start background sync thread
    t = threading.Thread(target=_background_worker, daemon=True)
    t.start()

    server = HTTPServer(("0.0.0.0", PORT), InventoryHandler)
    _log("info", f"{AGENT_NAME} listening on http://0.0.0.0:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log("info", "Shutting down")
        server.server_close()


if __name__ == "__main__":
    main()
