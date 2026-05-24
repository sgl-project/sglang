#!/usr/bin/env python3
"""
FractalMesh OMEGA Titan — KuCoin Agent v2
KuCoin REST API v2 spot trading + portfolio agent.
Samuel James Hiotis | ABN 56628117363
Port 7814 | sovereign.db WAL
"""

# ── Vault load ────────────────────────────────────────────────────────────────
import os
_VAULT = os.path.join(os.path.expanduser("~"), ".secrets", "fractal.env")
try:
    with open(_VAULT) as _vf:
        for _line in _vf:
            _s = _line.strip()
            if _s and not _s.startswith("#") and "=" in _s:
                _k, _, _v = _s.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
except FileNotFoundError:
    pass

# ── stdlib imports ─────────────────────────────────────────────────────────────
import urllib.request
import urllib.error
import urllib.parse
import json
import sqlite3
import logging
import signal
import time
import hmac
import hashlib
import base64
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY        = os.getenv("KUCOIN_API_KEY", "")
API_SECRET     = os.getenv("KUCOIN_API_SECRET", "")
API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE", "")
PORT           = int(os.getenv("KUCOIN_PORT", "7814"))
BASE_URL       = "https://api.kucoin.com"

ROOT_DIR  = os.path.join(os.path.expanduser("~"), "fmsaas")
DB_PATH   = os.path.join(ROOT_DIR, "database", "sovereign.db")
LOG_PATH  = os.path.join(ROOT_DIR, "logs", "fm_kucoin.log")

# ── Logging ───────────────────────────────────────────────────────────────────
Path(os.path.dirname(LOG_PATH)).mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [fm_kucoin] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("fm_kucoin")

# ── Signal handling ───────────────────────────────────────────────────────────
_running = True

def _handle_signal(signum, frame):
    global _running
    log.info("Signal %s received — shutting down", signum)
    _running = False

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT,  _handle_signal)

# ── SQLite ────────────────────────────────────────────────────────────────────
def get_db() -> sqlite3.Connection:
    Path(os.path.dirname(DB_PATH)).mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn

def init_schema():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS kucoin_orders (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id   TEXT UNIQUE,
            client_oid TEXT,
            symbol     TEXT,
            side       TEXT,
            type       TEXT,
            status     TEXT,
            size       REAL,
            price      REAL,
            ts         INTEGER
        );
        CREATE TABLE IF NOT EXISTS kucoin_fills (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id TEXT,
            symbol   TEXT,
            side     TEXT,
            price    REAL,
            size     REAL,
            fee      REAL,
            ts       INTEGER
        );
    """)
    conn.commit()
    conn.close()
    log.info("SQLite schema ready: %s", DB_PATH)

# ── KuCoin auth helpers ───────────────────────────────────────────────────────
def _kucoin_headers(method: str, path: str, body: dict = None) -> dict:
    """Build KuCoin v2 signed headers."""
    timestamp  = str(int(time.time() * 1000))
    body_str   = json.dumps(body) if body else ""
    sign_str   = timestamp + method.upper() + path + body_str
    signature  = base64.b64encode(
        hmac.new(API_SECRET.encode(), sign_str.encode(), hashlib.sha256).digest()
    ).decode()
    passphrase_sig = base64.b64encode(
        hmac.new(API_SECRET.encode(), API_PASSPHRASE.encode(), hashlib.sha256).digest()
    ).decode()
    return {
        "KC-API-KEY":         API_KEY,
        "KC-API-SIGN":        signature,
        "KC-API-TIMESTAMP":   timestamp,
        "KC-API-PASSPHRASE":  passphrase_sig,
        "KC-API-KEY-VERSION": "2",
        "Content-Type":       "application/json",
    }

def kucoin_get(path: str, params: dict = None) -> dict:
    qs = urllib.parse.urlencode(params) if params else ""
    full_path = path + ("?" + qs if qs else "")
    headers = _kucoin_headers("GET", full_path)
    url = BASE_URL + full_path
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        log.error("KuCoin GET %s → HTTP %s: %s", path, e.code, body)
        return {"error": body, "code": e.code}
    except Exception as exc:
        log.error("KuCoin GET %s → %s", path, exc)
        return {"error": str(exc)}

def kucoin_post(path: str, body: dict) -> dict:
    headers = _kucoin_headers("POST", path, body)
    payload = json.dumps(body).encode()
    url = BASE_URL + path
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body_err = e.read().decode()
        log.error("KuCoin POST %s → HTTP %s: %s", path, e.code, body_err)
        return {"error": body_err, "code": e.code}
    except Exception as exc:
        log.error("KuCoin POST %s → %s", path, exc)
        return {"error": str(exc)}

def kucoin_delete(path: str, params: dict = None) -> dict:
    qs = urllib.parse.urlencode(params) if params else ""
    full_path = path + ("?" + qs if qs else "")
    headers = _kucoin_headers("DELETE", full_path)
    url = BASE_URL + full_path
    req = urllib.request.Request(url, headers=headers, method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        log.error("KuCoin DELETE %s → HTTP %s: %s", path, e.code, body)
        return {"error": body, "code": e.code}
    except Exception as exc:
        log.error("KuCoin DELETE %s → %s", path, exc)
        return {"error": str(exc)}

# ── UUID helper (stdlib only) ─────────────────────────────────────────────────
def _uuid4() -> str:
    """Generate a UUID4 string using os.urandom — no uuid module needed."""
    b = bytearray(os.urandom(16))
    b[6] = (b[6] & 0x0F) | 0x40
    b[8] = (b[8] & 0x3F) | 0x80
    h = b.hex()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"

# ── Route handlers ────────────────────────────────────────────────────────────
def handle_health() -> dict:
    accs = kucoin_get("/api/v2/accounts")
    acc_count = 0
    if "data" in accs:
        acc_count = len(accs["data"]) if isinstance(accs["data"], list) else 0
    return {
        "status":        "ok",
        "agent":         "fm_kucoin",
        "port":          PORT,
        "account_count": acc_count,
    }

def handle_accounts() -> dict:
    data = kucoin_get("/api/v2/accounts")
    if "error" in data:
        return data
    accounts = data.get("data", [])
    nonzero = [a for a in accounts if float(a.get("balance", 0)) > 0]
    return {"accounts": nonzero, "total": len(accounts), "nonzero": len(nonzero)}

def handle_ticker(symbol: str) -> dict:
    if not symbol:
        return {"error": "symbol query param required"}
    data = kucoin_get("/api/v1/market/orderbook/level1", {"symbol": symbol})
    if "error" in data:
        return data
    return {"symbol": symbol, "ticker": data.get("data")}

def handle_symbols() -> dict:
    data = kucoin_get("/api/v1/symbols")
    if "error" in data:
        return data
    symbols = data.get("data", [])
    return {"symbols": symbols, "count": len(symbols)}

def handle_place_order(req_body: dict) -> dict:
    required = ["side", "symbol", "type"]
    for r in required:
        if r not in req_body:
            return {"error": f"missing field: {r}"}
    order_type = req_body["type"].lower()
    client_oid = req_body.get("clientOid") or _uuid4()
    payload: dict = {
        "clientOid": client_oid,
        "side":      req_body["side"].lower(),
        "symbol":    req_body["symbol"],
        "type":      order_type,
    }
    if order_type == "limit":
        if "size" not in req_body or "price" not in req_body:
            return {"error": "size and price required for limit order"}
        payload["size"]  = str(req_body["size"])
        payload["price"] = str(req_body["price"])
    elif order_type == "market":
        if "funds" in req_body:
            payload["funds"] = str(req_body["funds"])
        elif "size" in req_body:
            payload["size"] = str(req_body["size"])
        else:
            return {"error": "size or funds required for market order"}
    data = kucoin_post("/api/v1/orders", payload)
    if "data" in data and "orderId" in data["data"]:
        order_id = data["data"]["orderId"]
        conn = get_db()
        conn.execute(
            "INSERT OR IGNORE INTO kucoin_orders "
            "(order_id, client_oid, symbol, side, type, status, size, price, ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (order_id, client_oid, req_body["symbol"], req_body["side"].lower(),
             order_type, "active",
             float(req_body.get("size", 0)),
             float(req_body.get("price", 0)),
             int(time.time())),
        )
        conn.commit()
        conn.close()
    return data

def handle_active_orders() -> dict:
    data = kucoin_get("/api/v1/orders", {"status": "active"})
    if "error" in data:
        return data
    orders = data.get("data", {}).get("items", [])
    return {"orders": orders, "count": len(orders)}

def handle_fills() -> dict:
    data = kucoin_get("/api/v1/fills")
    if "error" in data:
        return data
    fills = data.get("data", {}).get("items", [])
    # Persist new fills to local DB
    conn = get_db()
    for f in fills:
        conn.execute(
            "INSERT OR IGNORE INTO kucoin_fills (order_id, symbol, side, price, size, fee, ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (f.get("orderId", ""), f.get("symbol", ""), f.get("side", ""),
             float(f.get("price", 0)), float(f.get("size", 0)),
             float(f.get("fee", 0)), int(time.time())),
        )
    conn.commit()
    conn.close()
    return {"fills": fills, "count": len(fills)}

def handle_cancel_order(req_body: dict) -> dict:
    order_id = req_body.get("orderId") or req_body.get("order_id")
    if not order_id:
        return {"error": "orderId required"}
    data = kucoin_delete(f"/api/v1/orders/{order_id}")
    if "data" in data:
        conn = get_db()
        conn.execute("UPDATE kucoin_orders SET status='cancelled' WHERE order_id=?", (str(order_id),))
        conn.commit()
        conn.close()
    return data

def handle_cancel_all(req_body: dict) -> dict:
    symbol = req_body.get("symbol", "")
    params = {"symbol": symbol} if symbol else {}
    data = kucoin_delete("/api/v1/orders", params if params else None)
    return data

def handle_portfolio() -> dict:
    accs_resp = kucoin_get("/api/v2/accounts")
    if "error" in accs_resp:
        return accs_resp
    accounts = accs_resp.get("data", [])

    # Build per-currency balance map
    balances: dict = {}
    for a in accounts:
        currency = a.get("currency", "")
        bal = float(a.get("balance", 0))
        if bal > 0:
            balances[currency] = balances.get(currency, 0.0) + bal

    # Approximate USD value using last ticker for each non-USDT asset
    total_usd = balances.get("USDT", 0.0)
    asset_values = []
    for currency, amount in balances.items():
        if currency in ("USDT", "USDC", "TUSD", "BUSD"):
            asset_values.append({"currency": currency, "amount": amount, "usd_value": amount})
            continue
        symbol = f"{currency}-USDT"
        ticker_resp = kucoin_get("/api/v1/market/orderbook/level1", {"symbol": symbol})
        price = 0.0
        if "data" in ticker_resp and ticker_resp["data"]:
            try:
                price = float(ticker_resp["data"].get("price", 0))
            except (TypeError, ValueError):
                price = 0.0
        usd_value = amount * price
        total_usd += usd_value
        asset_values.append({
            "currency": currency,
            "amount":   amount,
            "price":    price,
            "usd_value": round(usd_value, 4),
        })

    return {
        "assets":            asset_values,
        "total_usd_approx":  round(total_usd, 2),
        "note":              "Approximate USD value using KuCoin level1 prices.",
    }

# ── HTTP handler ──────────────────────────────────────────────────────────────
class KuCoinHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        log.debug("HTTP %s", fmt % args)

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length:
            raw = self.rfile.read(length)
            try:
                return json.loads(raw.decode())
            except json.JSONDecodeError:
                return {}
        return {}

    def _route_get(self, path: str, qs: dict):
        if path == "/health":
            return handle_health()
        if path == "/accounts":
            return handle_accounts()
        if path == "/ticker":
            return handle_ticker(qs.get("symbol", ""))
        if path == "/symbols":
            return handle_symbols()
        if path == "/orders":
            return handle_active_orders()
        if path == "/fills":
            return handle_fills()
        if path == "/portfolio":
            return handle_portfolio()
        return None

    def _route_post(self, path: str, body: dict):
        if path == "/order":
            return handle_place_order(body)
        if path == "/cancel":
            return handle_cancel_order(body)
        if path == "/cancel_all":
            return handle_cancel_all(body)
        return None

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        qs = dict(urllib.parse.parse_qsl(parsed.query))
        result = self._route_get(parsed.path, qs)
        if result is None:
            self._send_json({"error": "not found"}, 404)
        else:
            self._send_json(result)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        body = self._read_body()
        result = self._route_post(parsed.path, body)
        if result is None:
            self._send_json({"error": "not found"}, 404)
        else:
            self._send_json(result)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global _running
    init_schema()
    server = HTTPServer(("0.0.0.0", PORT), KuCoinHandler)
    log.info("fm_kucoin listening on port %d | DB: %s", PORT, DB_PATH)
    while _running:
        server.handle_request()
    server.server_close()
    log.info("fm_kucoin stopped.")

if __name__ == "__main__":
    main()
