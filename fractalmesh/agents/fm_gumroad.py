"""
FractalMesh OMEGA Titan — Gumroad Agent
Port: 7810
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
GUMROAD_ACCESS_TOKEN = os.getenv("GUMROAD_ACCESS_TOKEN", "")
GUMROAD_SELLER_ID    = os.getenv("GUMROAD_SELLER_ID", "")
PORT                 = int(os.getenv("GUMROAD_PORT", "7810"))

DB_PATH  = os.path.expanduser("~/fmsaas/database/sovereign.db")
LOG_PATH = os.path.expanduser("~/fmsaas/logs/fm_gumroad.log")

GUMROAD_BASE = "https://api.gumroad.com/v2"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [fm_gumroad] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("fm_gumroad")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS gumroad_sales (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id    TEXT,
            product_name  TEXT,
            amount        REAL,
            currency      TEXT,
            buyer_email   TEXT,
            ts            TEXT
        )
    """)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Gumroad API helpers
# ---------------------------------------------------------------------------

def _gumroad_request(path: str, method: str = "GET", params: dict | None = None,
                     body: dict | None = None):
    """Low-level Gumroad API call. Returns parsed JSON or raises."""
    base_params = {"access_token": GUMROAD_ACCESS_TOKEN}

    if method == "GET":
        if params:
            base_params.update(params)
        qs = urllib.parse.urlencode(base_params)
        url = f"{GUMROAD_BASE}{path}?{qs}"
        req = urllib.request.Request(url, method="GET")
    else:
        url = f"{GUMROAD_BASE}{path}"
        if body:
            base_params.update(body)
        data = urllib.parse.urlencode(base_params).encode()
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/x-www-form-urlencoded")

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode()
        log.error("Gumroad HTTP %s %s: %s", exc.code, path, raw)
        raise
    except urllib.error.URLError as exc:
        log.error("Gumroad URL error %s: %s", path, exc.reason)
        raise


def fetch_products():
    return _gumroad_request("/products")


def fetch_sales(product_id: str = "", page: int = 1):
    params = {"page": page}
    if product_id:
        params["product_id"] = product_id
    return _gumroad_request("/sales", params=params)


def fetch_subscribers(product_id: str):
    return _gumroad_request(f"/products/{product_id}/subscribers")


def create_product(name: str, description: str, price: int, currency: str = "usd"):
    return _gumroad_request("/products", method="POST", body={
        "name": name,
        "description": description,
        "price": price,
        "currency": currency,
    })


def enable_product(product_id: str):
    return _gumroad_request(f"/products/{product_id}/enable", method="PUT")


def disable_product(product_id: str):
    return _gumroad_request(f"/products/{product_id}/disable", method="PUT")


# ---------------------------------------------------------------------------
# Startup sync
# ---------------------------------------------------------------------------

def sync_sales():
    """Fetch last 50 sales from Gumroad and upsert into local DB."""
    log.info("Syncing sales from Gumroad...")
    try:
        result = fetch_sales()
        sales = result.get("sales", [])
        conn = get_db()
        cur = conn.cursor()
        inserted = 0
        for sale in sales[:50]:
            # Avoid duplicate by checking order_id mapped to product+email+ts
            cur.execute(
                "SELECT 1 FROM gumroad_sales WHERE product_id=? AND buyer_email=? AND ts=?",
                (sale.get("product_id"), sale.get("email"), sale.get("created_at", "")),
            )
            if not cur.fetchone():
                cur.execute(
                    "INSERT INTO gumroad_sales (product_id, product_name, amount, currency, buyer_email, ts) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        sale.get("product_id"),
                        sale.get("product_name"),
                        float(sale.get("price", 0)) / 100.0,
                        sale.get("currency", "usd"),
                        sale.get("email"),
                        sale.get("created_at", ""),
                    ),
                )
                inserted += 1
        conn.commit()
        conn.close()
        log.info("Sync complete — %d new sales inserted.", inserted)
    except Exception as exc:
        log.warning("Startup sync failed: %s", exc)


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


class GumroadHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # suppress default stderr log
        log.debug(fmt, *args)

    def _route(self) -> str:
        return urllib.parse.urlparse(self.path).path.rstrip("/") or "/"

    # ---- GET ---------------------------------------------------------------

    def do_GET(self):
        route = self._route()
        qs = _parse_qs(self)

        if route == "/health":
            conn = get_db()
            row = conn.execute("SELECT COALESCE(SUM(amount),0), COUNT(*) FROM gumroad_sales").fetchone()
            conn.close()
            _send_json(self, 200, {
                "status": "ok",
                "service": "fm_gumroad",
                "total_revenue": row[0],
                "sales_count": row[1],
            })

        elif route == "/products":
            try:
                data = fetch_products()
                _send_json(self, 200, data)
            except Exception as exc:
                _send_json(self, 502, {"error": str(exc)})

        elif route == "/sales":
            pid = qs.get("product_id", "")
            page = int(qs.get("page", "1"))
            try:
                data = fetch_sales(pid, page)
                _send_json(self, 200, data)
            except Exception as exc:
                _send_json(self, 502, {"error": str(exc)})

        elif route == "/subscribers":
            pid = qs.get("product_id", "")
            if not pid:
                _send_json(self, 400, {"error": "product_id required"})
                return
            try:
                data = fetch_subscribers(pid)
                _send_json(self, 200, data)
            except Exception as exc:
                _send_json(self, 502, {"error": str(exc)})

        elif route == "/analytics":
            conn = get_db()
            total_rev, sales_count = conn.execute(
                "SELECT COALESCE(SUM(amount),0), COUNT(*) FROM gumroad_sales"
            ).fetchone()
            per_product = []
            rows = conn.execute(
                "SELECT product_id, product_name, COUNT(*), SUM(amount) "
                "FROM gumroad_sales GROUP BY product_id"
            ).fetchall()
            for r in rows:
                per_product.append({
                    "product_id": r[0],
                    "product_name": r[1],
                    "sales_count": r[2],
                    "revenue": r[3],
                })
            conn.close()
            _send_json(self, 200, {
                "total_revenue": total_rev,
                "sales_count": sales_count,
                "per_product": per_product,
            })

        else:
            _send_json(self, 404, {"error": "not found"})

    # ---- POST --------------------------------------------------------------

    def do_POST(self):
        route = self._route()
        body = _read_body(self)

        if route == "/product":
            name        = body.get("name", "")
            description = body.get("description", "")
            price       = int(body.get("price", 0))
            currency    = body.get("currency", "usd")
            if not name:
                _send_json(self, 400, {"error": "name required"})
                return
            try:
                data = create_product(name, description, price, currency)
                _send_json(self, 200, data)
            except Exception as exc:
                _send_json(self, 502, {"error": str(exc)})

        elif route == "/enable":
            pid = body.get("product_id", "")
            if not pid:
                _send_json(self, 400, {"error": "product_id required"})
                return
            try:
                data = enable_product(pid)
                _send_json(self, 200, data)
            except Exception as exc:
                _send_json(self, 502, {"error": str(exc)})

        elif route == "/disable":
            pid = body.get("product_id", "")
            if not pid:
                _send_json(self, 400, {"error": "product_id required"})
                return
            try:
                data = disable_product(pid)
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
    sync_sales()
    server = HTTPServer(("0.0.0.0", PORT), GumroadHandler)
    server.timeout = 1.0
    log.info("fm_gumroad listening on port %d", PORT)
    while _running:
        server.handle_request()
    server.server_close()
    log.info("fm_gumroad stopped.")


if __name__ == "__main__":
    main()
