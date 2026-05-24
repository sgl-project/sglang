#!/usr/bin/env python3
"""
FractalMesh OMEGA Titan — Pionex Agent v1
Pionex REST API v1 grid-bot + spot trading agent.
Samuel James Hiotis | ABN 56628117363
Port 7813 | sovereign.db WAL
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
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY    = os.getenv("PIONEX_API_KEY", "")
API_SECRET = os.getenv("PIONEX_API_SECRET", "")
PORT       = int(os.getenv("PIONEX_PORT", "7813"))
BASE_URL   = "https://api.pionex.com"

ROOT_DIR   = os.path.join(os.path.expanduser("~"), "fmsaas")
DB_PATH    = os.path.join(ROOT_DIR, "database", "sovereign.db")
LOG_PATH   = os.path.join(ROOT_DIR, "logs", "fm_pionex.log")

# ── Logging ───────────────────────────────────────────────────────────────────
Path(os.path.dirname(LOG_PATH)).mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [fm_pionex] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("fm_pionex")

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
        CREATE TABLE IF NOT EXISTS pionex_orders (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id  TEXT UNIQUE,
            symbol    TEXT,
            side      TEXT,
            type      TEXT,
            status    TEXT,
            amount    REAL,
            price     REAL,
            ts        INTEGER
        );
        CREATE TABLE IF NOT EXISTS pionex_bots (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_id     TEXT UNIQUE,
            symbol     TEXT,
            invest     REAL,
            low_price  REAL,
            high_price REAL,
            status     TEXT,
            ts         INTEGER
        );
    """)
    conn.commit()
    conn.close()
    log.info("SQLite schema ready: %s", DB_PATH)

# ── Pionex auth helpers ────────────────────────────────────────────────────────
def _build_signature(method: str, path: str, params: dict, timestamp: str) -> str:
    """HMAC-SHA256 over method+path?sorted_params&timestamp=ts"""
    sorted_qs = urllib.parse.urlencode(sorted(params.items())) if params else ""
    path_with_query = path + ("?" + sorted_qs if sorted_qs else "")
    sign_str = f"{method}{path_with_query}&timestamp={timestamp}"
    return hmac.new(API_SECRET.encode(), sign_str.encode(), hashlib.sha256).hexdigest()

def _pionex_headers(method: str, path: str, params: dict) -> dict:
    timestamp = str(int(time.time() * 1000))
    sig = _build_signature(method, path, params, timestamp)
    return {
        "PIONEX-KEY":       API_KEY,
        "PIONEX-TIMESTAMP": timestamp,
        "PIONEX-SIGNATURE": sig,
        "Content-Type":     "application/json",
    }

def pionex_get(path: str, params: dict = None) -> dict:
    params = params or {}
    headers = _pionex_headers("GET", path, params)
    qs = urllib.parse.urlencode(sorted(params.items())) if params else ""
    url = BASE_URL + path + ("?" + qs if qs else "")
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        log.error("Pionex GET %s → HTTP %s: %s", path, e.code, body)
        return {"error": body, "code": e.code}
    except Exception as exc:
        log.error("Pionex GET %s → %s", path, exc)
        return {"error": str(exc)}

def pionex_post(path: str, body: dict, params: dict = None) -> dict:
    params = params or {}
    headers = _pionex_headers("POST", path, params)
    payload = json.dumps(body).encode()
    url = BASE_URL + path
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body_err = e.read().decode()
        log.error("Pionex POST %s → HTTP %s: %s", path, e.code, body_err)
        return {"error": body_err, "code": e.code}
    except Exception as exc:
        log.error("Pionex POST %s → %s", path, exc)
        return {"error": str(exc)}

def pionex_delete(path: str, params: dict = None) -> dict:
    params = params or {}
    headers = _pionex_headers("DELETE", path, params)
    qs = urllib.parse.urlencode(sorted(params.items())) if params else ""
    url = BASE_URL + path + ("?" + qs if qs else "")
    req = urllib.request.Request(url, headers=headers, method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        log.error("Pionex DELETE %s → HTTP %s: %s", path, e.code, body)
        return {"error": body, "code": e.code}
    except Exception as exc:
        log.error("Pionex DELETE %s → %s", path, exc)
        return {"error": str(exc)}

# ── Route handlers ────────────────────────────────────────────────────────────
def handle_health() -> dict:
    open_orders = pionex_get("/api/v1/trade/openOrders")
    count = 0
    if "data" in open_orders and "orders" in open_orders["data"]:
        count = len(open_orders["data"]["orders"])
    return {"status": "ok", "agent": "fm_pionex", "port": PORT, "open_orders": count}

def handle_account() -> dict:
    data = pionex_get("/api/v1/account/balances")
    if "error" in data:
        return data
    balances = data.get("data", {}).get("balances", [])
    nonzero = [b for b in balances if float(b.get("free", 0)) + float(b.get("frozen", 0)) > 0]
    return {"balances": nonzero}

def handle_symbols() -> dict:
    data = pionex_get("/api/v1/common/symbols")
    if "error" in data:
        return data
    symbols = data.get("data", {}).get("symbols", [])
    return {"symbols": symbols, "count": len(symbols)}

def handle_ticker(symbol: str) -> dict:
    if not symbol:
        return {"error": "symbol query param required"}
    data = pionex_get("/api/v1/market/tickers", {"symbol": symbol})
    if "error" in data:
        return data
    tickers = data.get("data", {}).get("tickers", [])
    return {"ticker": tickers[0] if tickers else None}

def handle_create_grid_bot(req_body: dict) -> dict:
    required = ["symbol", "invest_usdt", "low_price", "high_price", "grid_num"]
    for r in required:
        if r not in req_body:
            return {"error": f"missing field: {r}"}
    payload = {
        "symbol":      req_body["symbol"],
        "invest_usdt": float(req_body["invest_usdt"]),
        "low_price":   float(req_body["low_price"]),
        "high_price":  float(req_body["high_price"]),
        "grid_num":    int(req_body["grid_num"]),
    }
    if "take_profit_price" in req_body:
        payload["take_profit_price"] = float(req_body["take_profit_price"])
    data = pionex_post("/api/v1/trade/gridBot", payload)
    if "error" not in data:
        bot = data.get("data", {})
        bot_id = bot.get("orderId") or bot.get("botId") or bot.get("id", "")
        conn = get_db()
        conn.execute(
            "INSERT OR IGNORE INTO pionex_bots (bot_id, symbol, invest, low_price, high_price, status, ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (str(bot_id), req_body["symbol"], float(req_body["invest_usdt"]),
             float(req_body["low_price"]), float(req_body["high_price"]), "open", int(time.time())),
        )
        conn.commit()
        conn.close()
    return data

def handle_list_bots() -> dict:
    data = pionex_get("/api/v1/trade/gridBot/openOrders")
    if "error" in data:
        return data
    bots = data.get("data", {}).get("orders", [])
    return {"bots": bots, "count": len(bots)}

def handle_close_bot(req_body: dict) -> dict:
    bot_id = req_body.get("bot_id") or req_body.get("botId")
    if not bot_id:
        return {"error": "bot_id required"}
    data = pionex_delete("/api/v1/trade/gridBot", {"orderId": str(bot_id)})
    if "error" not in data:
        conn = get_db()
        conn.execute("UPDATE pionex_bots SET status='closed' WHERE bot_id=?", (str(bot_id),))
        conn.commit()
        conn.close()
    return data

def handle_place_order(req_body: dict) -> dict:
    required = ["symbol", "side", "type", "amount"]
    for r in required:
        if r not in req_body:
            return {"error": f"missing field: {r}"}
    payload = {
        "symbol": req_body["symbol"],
        "side":   req_body["side"].upper(),
        "type":   req_body["type"].upper(),
        "amount": str(req_body["amount"]),
    }
    if req_body["type"].upper() == "LIMIT":
        if "price" not in req_body:
            return {"error": "price required for LIMIT order"}
        payload["price"] = str(req_body["price"])
    data = pionex_post("/api/v1/trade/order", payload)
    if "error" not in data:
        order = data.get("data", {})
        order_id = order.get("orderId") or order.get("id", "")
        conn = get_db()
        conn.execute(
            "INSERT OR IGNORE INTO pionex_orders (order_id, symbol, side, type, status, amount, price, ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (str(order_id), req_body["symbol"], req_body["side"].upper(),
             req_body["type"].upper(), "open", float(req_body["amount"]),
             float(req_body.get("price", 0)), int(time.time())),
        )
        conn.commit()
        conn.close()
    return data

def handle_open_orders() -> dict:
    data = pionex_get("/api/v1/trade/openOrders")
    if "error" in data:
        return data
    orders = data.get("data", {}).get("orders", [])
    return {"orders": orders, "count": len(orders)}

def handle_cancel_order(req_body: dict) -> dict:
    order_id = req_body.get("orderId") or req_body.get("order_id")
    symbol   = req_body.get("symbol")
    if not order_id or not symbol:
        return {"error": "orderId and symbol required"}
    data = pionex_delete("/api/v1/trade/order", {"orderId": str(order_id), "symbol": symbol})
    if "error" not in data:
        conn = get_db()
        conn.execute("UPDATE pionex_orders SET status='cancelled' WHERE order_id=?", (str(order_id),))
        conn.commit()
        conn.close()
    return data

def handle_pnl() -> dict:
    conn = get_db()
    rows = conn.execute(
        "SELECT symbol, side, SUM(amount) as total_amt, AVG(price) as avg_price, COUNT(*) as trades "
        "FROM pionex_orders WHERE status != 'cancelled' GROUP BY symbol, side"
    ).fetchall()
    summary = [dict(r) for r in rows]
    bot_rows = conn.execute(
        "SELECT symbol, COUNT(*) as bot_count, SUM(invest) as total_invest "
        "FROM pionex_bots GROUP BY symbol"
    ).fetchall()
    conn.close()
    return {
        "order_summary": summary,
        "bot_summary":   [dict(r) for r in bot_rows],
        "note":          "Aggregate from local DB; actual P&L requires live price comparison.",
    }

# ── HTTP handler ──────────────────────────────────────────────────────────────
class PionexHandler(BaseHTTPRequestHandler):
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
        if path == "/account":
            return handle_account()
        if path == "/symbols":
            return handle_symbols()
        if path == "/ticker":
            return handle_ticker(qs.get("symbol", ""))
        if path == "/bots":
            return handle_list_bots()
        if path == "/orders":
            return handle_open_orders()
        if path == "/pnl":
            return handle_pnl()
        return None

    def _route_post(self, path: str, body: dict):
        if path == "/grid_bot":
            return handle_create_grid_bot(body)
        if path == "/close_bot":
            return handle_close_bot(body)
        if path == "/order":
            return handle_place_order(body)
        if path == "/cancel":
            return handle_cancel_order(body)
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
    server = HTTPServer(("0.0.0.0", PORT), PionexHandler)
    log.info("fm_pionex listening on port %d | DB: %s", PORT, DB_PATH)
    while _running:
        server.handle_request()
    server.server_close()
    log.info("fm_pionex stopped.")

if __name__ == "__main__":
    main()
