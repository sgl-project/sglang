"""
FractalMesh OMEGA Titan — Coinbase Advanced Trade Agent
Port: 7812
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
import base64
import hashlib
import hmac
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
COINBASE_API_KEY    = os.getenv("COINBASE_API_KEY", "")
COINBASE_SECRET_KEY = os.getenv("COINBASE_SECRET_KEY", "")
PORT                = int(os.getenv("COINBASE_PORT", "7812"))

DB_PATH  = os.path.expanduser("~/fmsaas/database/sovereign.db")
LOG_PATH = os.path.expanduser("~/fmsaas/logs/fm_coinbase.log")

CB_BASE = "https://api.coinbase.com/api/v3/brokerage"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [fm_coinbase] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("fm_coinbase")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS coinbase_orders (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id     TEXT UNIQUE,
            product_id   TEXT,
            side         TEXT,
            status       TEXT,
            filled_size  REAL,
            filled_value REAL,
            ts           TEXT
        )
    """)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# JWT / Auth helpers
# ---------------------------------------------------------------------------

def _b64url(data: bytes) -> str:
    """URL-safe base64 without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _make_jwt() -> str:
    """
    Build a simple HS256 JWT for Coinbase Advanced Trade.
    Header : {"alg":"HS256","typ":"JWT"}
    Payload: sub=API_KEY, iss="cdp", nbf/exp window of 120 s.
    """
    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": COINBASE_API_KEY,
        "iss": "cdp",
        "nbf": now,
        "exp": now + 120,
        "urn:coinbase:fc:requests:cb-access-key": COINBASE_API_KEY,
    }
    h = _b64url(json.dumps(header, separators=(",", ":")).encode())
    p = _b64url(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{h}.{p}".encode()
    secret = COINBASE_SECRET_KEY.encode()
    sig = hmac.new(secret, signing_input, hashlib.sha256).digest()
    return f"{h}.{p}.{_b64url(sig)}"


def _coinbase_request(path: str, method: str = "GET", payload=None):
    """Low-level Coinbase API call using Bearer JWT. Returns parsed JSON or raises."""
    url = f"{CB_BASE}{path}"
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()

    token = _make_jwt()
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode()
        log.error("Coinbase HTTP %s %s: %s", exc.code, path, raw)
        raise
    except urllib.error.URLError as exc:
        log.error("Coinbase URL error %s: %s", path, exc.reason)
        raise


# ---------------------------------------------------------------------------
# Coinbase API helpers
# ---------------------------------------------------------------------------

def cb_list_accounts():
    return _coinbase_request("/accounts")


def cb_get_account(account_uuid: str):
    return _coinbase_request(f"/accounts/{account_uuid}")


def cb_list_products(limit: int = 250):
    return _coinbase_request(f"/products?limit={limit}")


def cb_get_product(product_id: str):
    return _coinbase_request(f"/products/{product_id}")


def cb_list_orders(order_status: str = ""):
    path = "/orders/historical/batch"
    body: dict = {}
    if order_status:
        body["order_status"] = [order_status]
    return _coinbase_request(path, method="POST", payload=body)


def cb_get_order(order_id: str):
    return _coinbase_request(f"/orders/historical/{order_id}")


def cb_create_order(product_id: str, side: str, order_type: str,
                    base_size: str = "", quote_size: str = "",
                    limit_price: str = ""):
    """
    Create a market or limit order.
    order_type: 'market' or 'limit'
    side: 'BUY' or 'SELL'
    """
    order_config: dict = {}

    if order_type.lower() == "market":
        if side.upper() == "BUY":
            order_config = {"market_market_ioc": {"quote_size": quote_size or "0"}}
        else:
            order_config = {"market_market_ioc": {"base_size": base_size or "0"}}
    else:  # limit
        order_config = {
            "limit_limit_gtc": {
                "base_size": base_size or "0",
                "limit_price": limit_price or "0",
                "post_only": False,
            }
        }

    import uuid as _uuid
    client_order_id = str(_uuid.uuid4())

    return _coinbase_request("/orders", method="POST", payload={
        "client_order_id": client_order_id,
        "product_id": product_id,
        "side": side.upper(),
        "order_configuration": order_config,
    })


def cb_cancel_order(order_id: str):
    return _coinbase_request("/orders/batch_cancel", method="POST",
                             payload={"order_ids": [order_id]})


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _upsert_order(conn: sqlite3.Connection, order: dict):
    oid    = order.get("order_id", "")
    pid    = order.get("product_id", "")
    side   = order.get("side", "")
    status = order.get("status", "")
    fsz    = float(order.get("filled_size", 0) or 0)
    fval   = float(order.get("filled_value", 0) or 0)
    ts     = order.get("created_time", str(int(time.time())))
    conn.execute(
        "INSERT INTO coinbase_orders (order_id, product_id, side, status, filled_size, filled_value, ts) "
        "VALUES (?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(order_id) DO UPDATE SET status=excluded.status, "
        "filled_size=excluded.filled_size, filled_value=excluded.filled_value",
        (oid, pid, side, status, fsz, fval, str(ts)),
    )


def _portfolio_value_usd(accounts: list) -> float:
    """Sum USD-equivalent balances across accounts (USD, USDC treated 1:1)."""
    total = 0.0
    for acct in accounts:
        balance = acct.get("available_balance", {})
        value   = float(balance.get("value", 0) or 0)
        currency = balance.get("currency", "")
        # For USD / stablecoins count directly; others would need price lookup
        if currency in ("USD", "USDC", "USDT"):
            total += value
    return total


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


class CoinbaseHandler(BaseHTTPRequestHandler):
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
            count = conn.execute("SELECT COUNT(*) FROM coinbase_orders").fetchone()[0]
            conn.close()
            # Quick portfolio value from cached DB data
            try:
                accounts_resp = cb_list_accounts()
                accounts = accounts_resp.get("accounts", [])
                portfolio_usd = _portfolio_value_usd(accounts)
            except Exception:
                portfolio_usd = 0.0
            _send_json(self, 200, {
                "status": "ok",
                "service": "fm_coinbase",
                "accounts_count": len(accounts) if "accounts" in dir() else 0,
                "portfolio_usd": portfolio_usd,
                "local_orders": count,
            })

        elif route == "/accounts":
            try:
                data = cb_list_accounts()
                _send_json(self, 200, data)
            except Exception as exc:
                _send_json(self, 502, {"error": str(exc)})

        elif route == "/account":
            uuid = qs.get("uuid", "")
            if not uuid:
                _send_json(self, 400, {"error": "uuid required"})
                return
            try:
                data = cb_get_account(uuid)
                _send_json(self, 200, data)
            except Exception as exc:
                _send_json(self, 502, {"error": str(exc)})

        elif route == "/portfolio":
            try:
                data = cb_list_accounts()
                accounts = data.get("accounts", [])
                portfolio_usd = _portfolio_value_usd(accounts)
                per_account = []
                for acct in accounts:
                    bal = acct.get("available_balance", {})
                    per_account.append({
                        "uuid": acct.get("uuid"),
                        "name": acct.get("name"),
                        "currency": bal.get("currency"),
                        "balance": bal.get("value"),
                    })
                _send_json(self, 200, {
                    "portfolio_value_usd": portfolio_usd,
                    "accounts": per_account,
                })
            except Exception as exc:
                _send_json(self, 502, {"error": str(exc)})

        elif route == "/products":
            try:
                data = cb_list_products()
                _send_json(self, 200, data)
            except Exception as exc:
                _send_json(self, 502, {"error": str(exc)})

        elif route == "/product":
            pid = qs.get("id", "")
            if not pid:
                _send_json(self, 400, {"error": "id required"})
                return
            try:
                data = cb_get_product(pid)
                # Enrich with 24h price stats already in product response
                _send_json(self, 200, data)
            except Exception as exc:
                _send_json(self, 502, {"error": str(exc)})

        elif route == "/orders":
            status_filter = qs.get("status", "")
            try:
                data = cb_list_orders(status_filter)
                # sync into local DB
                conn = get_db()
                for order in data.get("orders", []):
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
                data = cb_get_order(oid)
                order = data.get("order", {})
                if order:
                    conn = get_db()
                    _upsert_order(conn, order)
                    conn.commit()
                    conn.close()
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
            product_id  = body.get("product_id", "")
            side        = body.get("side", "BUY")
            order_type  = body.get("order_type", "market")
            base_size   = str(body.get("base_size", ""))
            quote_size  = str(body.get("quote_size", ""))
            limit_price = str(body.get("limit_price", ""))

            if not product_id:
                _send_json(self, 400, {"error": "product_id required"})
                return
            if order_type.lower() == "limit" and not limit_price:
                _send_json(self, 400, {"error": "limit_price required for limit order"})
                return

            try:
                data = cb_create_order(
                    product_id, side, order_type,
                    base_size=base_size,
                    quote_size=quote_size,
                    limit_price=limit_price,
                )
                # store if successful
                order = data.get("success_response", {})
                if order:
                    conn = get_db()
                    _upsert_order(conn, {
                        "order_id": order.get("order_id", ""),
                        "product_id": product_id,
                        "side": side,
                        "status": "OPEN",
                        "filled_size": 0,
                        "filled_value": 0,
                        "created_time": str(int(time.time())),
                    })
                    conn.commit()
                    conn.close()
                _send_json(self, 200, data)
            except Exception as exc:
                _send_json(self, 502, {"error": str(exc)})

        elif route == "/cancel":
            order_id = body.get("order_id", "")
            if not order_id:
                _send_json(self, 400, {"error": "order_id required"})
                return
            try:
                data = cb_cancel_order(order_id)
                # update local status
                conn = get_db()
                conn.execute(
                    "UPDATE coinbase_orders SET status='CANCELLED' WHERE order_id=?",
                    (order_id,),
                )
                conn.commit()
                conn.close()
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
    # Ensure DB schema
    conn = get_db()
    conn.close()
    server = HTTPServer(("0.0.0.0", PORT), CoinbaseHandler)
    server.timeout = 1.0
    log.info("fm_coinbase listening on port %d", PORT)
    while _running:
        server.handle_request()
    server.server_close()
    log.info("fm_coinbase stopped.")


if __name__ == "__main__":
    main()
