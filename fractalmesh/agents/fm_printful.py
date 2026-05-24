"""
FractalMesh OMEGA Titan — Printful Agent
Port: 7811
"""

# ---------------------------------------------------------------------------
# Vault loading
# ---------------------------------------------------------------------------
import os

_VAULT = os.path.expanduser("~/.secrets/fractal.env")
if os.path.isfile(_VAULT):
    with open(_VAULT) as _fh:
        for _line in _fh:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

# ---------------------------------------------------------------------------
# stdlib imports
# ---------------------------------------------------------------------------
import json
import logging
import signal
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
PRINTFUL_API_KEY = os.getenv("PRINTFUL_API_KEY", "")
PORT             = int(os.getenv("PRINTFUL_PORT", "7811"))

DB_PATH  = os.path.expanduser("~/fmsaas/database/sovereign.db")
LOG_PATH = os.path.expanduser("~/fmsaas/logs/fm_printful.log")

PRINTFUL_BASE = "https://api.printful.com"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [fm_printful] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("fm_printful")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS printful_orders (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id       TEXT UNIQUE,
            status         TEXT,
            recipient_name TEXT,
            total          REAL,
            ts             TEXT
        )
    """)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Printful API helpers
# ---------------------------------------------------------------------------

def _printful_request(path: str, method: str = "GET", payload=None):
    """Low-level Printful API call. Returns parsed JSON body or raises."""
    url = f"{PRINTFUL_BASE}{path}"
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()

    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {PRINTFUL_API_KEY}")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode()
        log.error("Printful HTTP %s %s: %s", exc.code, path, raw)
        raise
    except urllib.error.URLError as exc:
        log.error("Printful URL error %s: %s", path, exc.reason)
        raise


def pf_list_products():
    return _printful_request("/store/products")


def pf_get_product(product_id):
    return _printful_request(f"/store/products/{product_id}")


def pf_create_order(recipient: dict, items: list):
    return _printful_request("/orders", method="POST", payload={
        "recipient": recipient,
        "items": items,
    })


def pf_list_orders(status: str = ""):
    path = "/orders"
    if status:
        path += f"?status={urllib.parse.quote(status)}"
    return _printful_request(path)


def pf_get_order(order_id):
    return _printful_request(f"/orders/{order_id}")


def pf_estimate_shipping(recipient: dict, items: list):
    return _printful_request("/shipping/rates", method="POST", payload={
        "recipient": recipient,
        "items": items,
    })


def pf_list_countries():
    return _printful_request("/countries")


def pf_list_warehouses():
    return _printful_request("/warehouses")


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _upsert_order(conn: sqlite3.Connection, order: dict):
    oid    = str(order.get("id", ""))
    status = order.get("status", "")
    name   = order.get("recipient", {}).get("name", "")
    total  = float(order.get("retail_costs", {}).get("total", 0) or 0)
    ts     = order.get("created", "") or str(int(time.time()))
    conn.execute(
        "INSERT INTO printful_orders (order_id, status, recipient_name, total, ts) "
        "VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(order_id) DO UPDATE SET status=excluded.status, total=excluded.total",
        (oid, status, name, total, str(ts)),
    )


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

def _send_json(handler: BaseHTTPRequestHandler, status: int, payload):
    body = json.dumps(payload, indent=2).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _parse_qs(handler: BaseHTTPRequestHandler) -> dict:
    parsed = urllib.parse.urlparse(handler.path)
    return dict(urllib.parse.parse_qsl(parsed.query))


def _read_body(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", 0))
    if length == 0:
        return {}
    raw = handler.rfile.read(length)
    ct = handler.headers.get("Content-Type", "")
    if "application/json" in ct:
        return json.loads(raw.decode())
    return dict(urllib.parse.parse_qsl(raw.decode()))


class PrintfulHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        log.debug(fmt, *args)

    def _route(self) -> str:
        return urllib.parse.urlparse(self.path).path.rstrip("/") or "/"

    # ---- GET ---------------------------------------------------------------

    def do_GET(self):
        route = self._route()
        qs = _parse_qs(self)

        if route == "/health":
            conn = get_db()
            count = conn.execute("SELECT COUNT(*) FROM printful_orders").fetchone()[0]
            conn.close()
            _send_json(self, 200, {
                "status": "ok",
                "service": "fm_printful",
                "order_count": count,
            })

        elif route == "/products":
            try:
                data = pf_list_products()
                _send_json(self, 200, data)
            except Exception as exc:
                _send_json(self, 502, {"error": str(exc)})

        elif route == "/product":
            pid = qs.get("id", "")
            if not pid:
                _send_json(self, 400, {"error": "id required"})
                return
            try:
                data = pf_get_product(pid)
                _send_json(self, 200, data)
            except Exception as exc:
                _send_json(self, 502, {"error": str(exc)})

        elif route == "/orders":
            status = qs.get("status", "")
            try:
                data = pf_list_orders(status)
                # sync into local DB
                conn = get_db()
                for order in data.get("result", []):
                    _upsert_order(conn, order)
                conn.commit()
                conn.close()
                _send_json(self, 200, data)
            except Exception as exc:
                _send_json(self, 502, {"error": str(exc)})

        elif route == "/order":
            oid = qs.get("id", "")
            if not oid:
                _send_json(self, 400, {"error": "id required"})
                return
            try:
                data = pf_get_order(oid)
                # sync
                conn = get_db()
                result = data.get("result", {})
                if result:
                    _upsert_order(conn, result)
                    conn.commit()
                conn.close()
                _send_json(self, 200, data)
            except Exception as exc:
                _send_json(self, 502, {"error": str(exc)})

        elif route == "/countries":
            try:
                data = pf_list_countries()
                _send_json(self, 200, data)
            except Exception as exc:
                _send_json(self, 502, {"error": str(exc)})

        elif route == "/warehouses":
            try:
                data = pf_list_warehouses()
                _send_json(self, 200, data)
            except Exception as exc:
                _send_json(self, 502, {"error": str(exc)})

        else:
            _send_json(self, 404, {"error": "not found"})

    # ---- POST --------------------------------------------------------------

    def do_POST(self):
        route = self._route()
        body = _read_body(self)

        if route == "/order":
            recipient = body.get("recipient")
            items     = body.get("items")
            if not recipient or not items:
                _send_json(self, 400, {"error": "recipient and items required"})
                return
            try:
                data = pf_create_order(recipient, items)
                # store in DB
                result = data.get("result", {})
                if result:
                    conn = get_db()
                    _upsert_order(conn, result)
                    conn.commit()
                    conn.close()
                _send_json(self, 200, data)
            except Exception as exc:
                _send_json(self, 502, {"error": str(exc)})

        elif route == "/estimate":
            recipient = body.get("recipient")
            items     = body.get("items")
            if not recipient or not items:
                _send_json(self, 400, {"error": "recipient and items required"})
                return
            try:
                data = pf_estimate_shipping(recipient, items)
                _send_json(self, 200, data)
            except Exception as exc:
                _send_json(self, 502, {"error": str(exc)})

        else:
            _send_json(self, 404, {"error": "not found"})


# ---------------------------------------------------------------------------
# Signal handling + main
# ---------------------------------------------------------------------------

_running = True


def _shutdown(signum, frame):
    global _running
    log.info("Signal %s received — shutting down.", signum)
    _running = False


signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT, _shutdown)


def main():
    global _running
    # Warm-up: ensure DB schema exists
    conn = get_db()
    conn.close()
    server = HTTPServer(("0.0.0.0", PORT), PrintfulHandler)
    server.timeout = 1.0
    log.info("fm_printful listening on port %d", PORT)
    while _running:
        server.handle_request()
    server.server_close()
    log.info("fm_printful stopped.")


if __name__ == "__main__":
    main()
