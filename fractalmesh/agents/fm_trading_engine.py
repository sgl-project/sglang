#!/usr/bin/env python3
"""
FractalMesh OMEGA Titan — Crypto/Forex Trading Engine
Port: 7855
Samuel James Hiotis | ABN 56628117363

Algorithmic trading engine supporting price feeds, signal generation,
paper trading simulation, and live order execution with safety rails.
Tracks portfolio P&L across Binance and Coinbase exchanges.

SAFETY RAILS:
  - LIVE_TRADING_ENABLED must be "true" explicitly; default is "false"
  - Max single order capped at min(TRADE_MAX_USD, $500) hard cap
  - No ETH transaction signing — exchange API only
  - All orders logged to SQLite BEFORE any API call
"""

import hashlib
import hmac
import http.server
import json
import math
import os
import sqlite3
import statistics
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Vault / env loading
# ---------------------------------------------------------------------------

_ENV_FILE = Path.home() / ".secrets" / "fractal.env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

PORT = int(os.environ.get("TRADING_ENGINE_PORT", "7855"))
DB_PATH = Path.home() / "fmsaas" / "database" / "sovereign.db"
START_TIME = time.time()

LIVE_TRADING_ENABLED = os.environ.get("LIVE_TRADING_ENABLED", "false").lower() == "true"
TRADE_MAX_USD = min(float(os.environ.get("TRADE_MAX_USD", "100")), 500.0)

BINANCE_BASE = "https://api.binance.com"
COINGECKO_URL = (
    "https://api.coingecko.com/api/v3/simple/price"
    "?ids=bitcoin,ethereum,solana,binancecoin"
    "&vs_currencies=usd"
    "&include_24hr_change=true"
    "&include_24hr_vol=true"
)

# CoinGecko id → trading symbol mapping
COINGECKO_ID_MAP = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "solana": "SOL",
    "binancecoin": "BNB",
}

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_db() -> sqlite3.Connection:
    """Return a WAL-mode connection to sovereign.db."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't already exist."""
    conn = get_db()
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS price_feeds (
            id          INTEGER PRIMARY KEY,
            symbol      TEXT    NOT NULL,
            price       REAL    NOT NULL,
            source      TEXT    NOT NULL,
            volume_24h  REAL,
            change_24h  REAL,
            fetched_at  REAL    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS signals (
            id              INTEGER PRIMARY KEY,
            symbol          TEXT    NOT NULL,
            signal_type     TEXT    NOT NULL,
            strength        REAL    NOT NULL,
            strategy        TEXT    NOT NULL,
            price_at_signal REAL    NOT NULL,
            triggered_at    REAL    NOT NULL,
            acted_on        INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS orders (
            id           INTEGER PRIMARY KEY,
            order_id     TEXT,
            symbol       TEXT    NOT NULL,
            side         TEXT    NOT NULL,
            quantity     REAL    NOT NULL,
            price        REAL    NOT NULL,
            order_type   TEXT    NOT NULL,
            status       TEXT    NOT NULL,
            exchange     TEXT    NOT NULL,
            paper_trade  INTEGER DEFAULT 1,
            filled_price REAL,
            pnl          REAL,
            created_at   REAL    NOT NULL,
            updated_at   REAL    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS portfolio (
            id              INTEGER PRIMARY KEY,
            asset           TEXT    NOT NULL UNIQUE,
            quantity        REAL    NOT NULL,
            avg_buy_price   REAL    NOT NULL,
            current_price   REAL,
            unrealized_pnl  REAL,
            updated_at      REAL    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS strategies (
            id         INTEGER PRIMARY KEY,
            name       TEXT    NOT NULL UNIQUE,
            config     TEXT    NOT NULL,
            enabled    INTEGER DEFAULT 1,
            created_at REAL    NOT NULL
        );
    """)

    # Seed default strategies if none exist
    cur.execute("SELECT COUNT(*) AS n FROM strategies")
    if cur.fetchone()["n"] == 0:
        now = time.time()
        default_strategies = [
            (
                "RSI_14",
                json.dumps({"period": 14, "oversold": 30, "overbought": 70,
                            "symbols": ["BTC", "ETH", "SOL", "BNB"]}),
                1, now,
            ),
            (
                "SMA_Crossover_7_14",
                json.dumps({"fast": 7, "slow": 14,
                            "symbols": ["BTC", "ETH", "SOL", "BNB"]}),
                1, now,
            ),
        ]
        cur.executemany(
            "INSERT INTO strategies (name, config, enabled, created_at) VALUES (?,?,?,?)",
            default_strategies,
        )

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Price feed helpers
# ---------------------------------------------------------------------------

def fetch_coingecko_prices() -> list[dict]:
    """Fetch current prices from CoinGecko public API. Returns list of dicts."""
    req = urllib.request.Request(
        COINGECKO_URL,
        headers={"User-Agent": "FractalMesh-TradingEngine/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
        return []

    now = time.time()
    rows = []
    for gecko_id, symbol in COINGECKO_ID_MAP.items():
        coin = data.get(gecko_id, {})
        price = coin.get("usd")
        if price is None:
            continue
        rows.append({
            "symbol": symbol,
            "price": float(price),
            "source": "coingecko",
            "volume_24h": float(coin.get("usd_24h_vol", 0) or 0),
            "change_24h": float(coin.get("usd_24h_change", 0) or 0),
            "fetched_at": now,
        })
    return rows


def store_price_feeds(rows: list[dict]) -> None:
    """Persist price feed rows to SQLite."""
    if not rows:
        return
    conn = get_db()
    conn.executemany(
        """INSERT INTO price_feeds (symbol, price, source, volume_24h, change_24h, fetched_at)
           VALUES (:symbol, :price, :source, :volume_24h, :change_24h, :fetched_at)""",
        rows,
    )
    conn.commit()
    # Keep portfolio current_price in sync
    for row in rows:
        conn.execute(
            """UPDATE portfolio SET current_price=?, updated_at=?,
               unrealized_pnl=(quantity * (? - avg_buy_price))
               WHERE asset=?""",
            (row["price"], row["fetched_at"], row["price"], row["symbol"]),
        )
    conn.commit()
    conn.close()


def get_price_history(symbol: str, limit: int = 100) -> list[float]:
    """Return up to `limit` historical close prices for RSI/SMA computation."""
    conn = get_db()
    rows = conn.execute(
        "SELECT price FROM price_feeds WHERE symbol=? ORDER BY fetched_at DESC LIMIT ?",
        (symbol.upper(), limit),
    ).fetchall()
    conn.close()
    return [r["price"] for r in reversed(rows)]


# ---------------------------------------------------------------------------
# Technical analysis
# ---------------------------------------------------------------------------

def compute_rsi(prices: list[float], period: int = 14) -> float | None:
    """Compute RSI-14 (Wilder's smoothed). Returns None if insufficient data."""
    if len(prices) < period + 1:
        return None

    gains = []
    losses = []
    for i in range(1, period + 1):
        delta = prices[i] - prices[i - 1]
        if delta >= 0:
            gains.append(delta)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(abs(delta))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    # Wilder smoothing for remaining prices
    for i in range(period + 1, len(prices)):
        delta = prices[i] - prices[i - 1]
        gain = max(delta, 0.0)
        loss = abs(min(delta, 0.0))
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_sma(prices: list[float], period: int) -> float | None:
    """Simple moving average over the last `period` prices."""
    if len(prices) < period:
        return None
    return statistics.mean(prices[-period:])


def rsi_strength(rsi: float, signal_type: str) -> float:
    """Map RSI value to signal strength 0.0-1.0."""
    if signal_type == "buy":
        # RSI 30 → 0.0, RSI 0 → 1.0
        return max(0.0, min(1.0, (30.0 - rsi) / 30.0))
    else:
        # RSI 70 → 0.0, RSI 100 → 1.0
        return max(0.0, min(1.0, (rsi - 70.0) / 30.0))


def sma_strength(fast_sma: float, slow_sma: float) -> float:
    """Map SMA crossover gap to signal strength 0.0-1.0."""
    if slow_sma == 0:
        return 0.0
    pct = abs(fast_sma - slow_sma) / slow_sma
    return min(1.0, pct * 50.0)  # 2% gap → strength 1.0


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def generate_signals(symbols: list[str] | None = None) -> None:
    """Run RSI-14 and SMA 7/14 crossover for all tracked symbols and store signals."""
    if symbols is None:
        symbols = list(COINGECKO_ID_MAP.values())

    conn = get_db()
    now = time.time()

    for symbol in symbols:
        prices = get_price_history(symbol, limit=100)
        if not prices:
            continue
        current_price = prices[-1]

        # -- RSI signal --
        rsi = compute_rsi(prices, period=14)
        if rsi is not None:
            if rsi < 30:
                strength = rsi_strength(rsi, "buy")
                conn.execute(
                    """INSERT INTO signals
                       (symbol, signal_type, strength, strategy, price_at_signal, triggered_at)
                       VALUES (?,?,?,?,?,?)""",
                    (symbol, "buy", strength, "RSI_14", current_price, now),
                )
            elif rsi > 70:
                strength = rsi_strength(rsi, "sell")
                conn.execute(
                    """INSERT INTO signals
                       (symbol, signal_type, strength, strategy, price_at_signal, triggered_at)
                       VALUES (?,?,?,?,?,?)""",
                    (symbol, "sell", strength, "RSI_14", current_price, now),
                )

        # -- SMA crossover signal --
        fast_sma = compute_sma(prices, 7)
        slow_sma = compute_sma(prices, 14)
        if fast_sma is not None and slow_sma is not None:
            # Compare to previous cross state using second-to-last reading
            prices_prev = prices[:-1]
            fast_prev = compute_sma(prices_prev, 7)
            slow_prev = compute_sma(prices_prev, 14)

            if fast_prev is not None and slow_prev is not None:
                crossed_above = (fast_prev <= slow_prev) and (fast_sma > slow_sma)
                crossed_below = (fast_prev >= slow_prev) and (fast_sma < slow_sma)

                if crossed_above:
                    strength = sma_strength(fast_sma, slow_sma)
                    conn.execute(
                        """INSERT INTO signals
                           (symbol, signal_type, strength, strategy, price_at_signal, triggered_at)
                           VALUES (?,?,?,?,?,?)""",
                        (symbol, "buy", strength, "SMA_Crossover_7_14", current_price, now),
                    )
                elif crossed_below:
                    strength = sma_strength(fast_sma, slow_sma)
                    conn.execute(
                        """INSERT INTO signals
                           (symbol, signal_type, strength, strategy, price_at_signal, triggered_at)
                           VALUES (?,?,?,?,?,?)""",
                        (symbol, "sell", strength, "SMA_Crossover_7_14", current_price, now),
                    )

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Background price feed thread
# ---------------------------------------------------------------------------

def price_feed_loop() -> None:
    """Daemon thread: fetch prices every 30 s, then regenerate signals."""
    while True:
        try:
            rows = fetch_coingecko_prices()
            if rows:
                store_price_feeds(rows)
                generate_signals()
        except Exception:
            pass  # never crash the daemon
        time.sleep(30)


# ---------------------------------------------------------------------------
# Binance API helpers (live orders only)
# ---------------------------------------------------------------------------

def _binance_sign(params: dict) -> str:
    """Append timestamp and HMAC-SHA256 signature. Returns query string."""
    secret = os.environ.get("BINANCE_SECRET_KEY", "")
    params["timestamp"] = int(time.time() * 1000)
    query = urllib.parse.urlencode(params)
    sig = hmac.new(
        secret.encode("utf-8"),
        query.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return query + "&signature=" + sig


def binance_place_order(
    symbol: str, side: str, quantity: float, order_type: str = "MARKET"
) -> dict:
    """
    Send a signed order to Binance.
    symbol: e.g. "BTCUSDT"
    side: "BUY" or "SELL"
    Returns parsed JSON response or raises on error.
    """
    api_key = os.environ.get("BINANCE_API_KEY", "")
    if not api_key:
        raise ValueError("BINANCE_API_KEY not configured")

    params = {
        "symbol": symbol.upper().replace("/", ""),
        "side": side.upper(),
        "type": order_type.upper(),
        "quantity": quantity,
    }
    if order_type.upper() == "MARKET":
        params.pop("timeInForce", None)
    else:
        params["timeInForce"] = "GTC"

    signed_qs = _binance_sign(params)
    url = f"{BINANCE_BASE}/api/v3/order?" + signed_qs

    req = urllib.request.Request(
        url,
        method="POST",
        headers={
            "X-MBX-APIKEY": api_key,
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()
        raise RuntimeError(f"Binance API error {exc.code}: {body}") from exc


# ---------------------------------------------------------------------------
# Order helpers
# ---------------------------------------------------------------------------

def _current_price_for_symbol(symbol: str) -> float | None:
    """Return latest known price for a symbol."""
    conn = get_db()
    row = conn.execute(
        "SELECT price FROM price_feeds WHERE symbol=? ORDER BY fetched_at DESC LIMIT 1",
        (symbol.upper(),),
    ).fetchone()
    conn.close()
    return row["price"] if row else None


def _generate_order_id(symbol: str, side: str) -> str:
    ts = str(time.time()).encode()
    return "FM-" + hashlib.sha256(ts + symbol.encode() + side.encode()).hexdigest()[:16].upper()


def create_paper_order(
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    order_type: str,
    strategy: str,
    exchange: str = "paper",
) -> dict:
    """Insert a paper trade order and update portfolio."""
    now = time.time()
    order_id = _generate_order_id(symbol, side)

    conn = get_db()
    cur = conn.execute(
        """INSERT INTO orders
           (order_id, symbol, side, quantity, price, order_type, status,
            exchange, paper_trade, filled_price, pnl, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            order_id, symbol.upper(), side.upper(), quantity, price,
            order_type.upper(), "FILLED", exchange, 1,
            price, 0.0, now, now,
        ),
    )
    inserted_id = cur.lastrowid

    # Update portfolio
    asset = symbol.upper().replace("USDT", "").replace("/", "")
    if side.upper() == "BUY":
        existing = conn.execute(
            "SELECT quantity, avg_buy_price FROM portfolio WHERE asset=?", (asset,)
        ).fetchone()
        if existing:
            old_qty = existing["quantity"]
            old_avg = existing["avg_buy_price"]
            new_qty = old_qty + quantity
            new_avg = ((old_qty * old_avg) + (quantity * price)) / new_qty
            conn.execute(
                """UPDATE portfolio SET quantity=?, avg_buy_price=?,
                   current_price=?, unrealized_pnl=(? * (? - ?)), updated_at=?
                   WHERE asset=?""",
                (new_qty, new_avg, price, new_qty, price, new_avg, now, asset),
            )
        else:
            conn.execute(
                """INSERT INTO portfolio (asset, quantity, avg_buy_price, current_price,
                   unrealized_pnl, updated_at)
                   VALUES (?,?,?,?,?,?)""",
                (asset, quantity, price, price, 0.0, now),
            )
    elif side.upper() == "SELL":
        existing = conn.execute(
            "SELECT quantity, avg_buy_price FROM portfolio WHERE asset=?", (asset,)
        ).fetchone()
        if existing:
            old_qty = existing["quantity"]
            old_avg = existing["avg_buy_price"]
            new_qty = max(0.0, old_qty - quantity)
            realized_pnl = quantity * (price - old_avg)
            # Update PnL on order
            conn.execute(
                "UPDATE orders SET pnl=? WHERE id=?", (realized_pnl, inserted_id)
            )
            if new_qty <= 0:
                conn.execute("DELETE FROM portfolio WHERE asset=?", (asset,))
            else:
                conn.execute(
                    """UPDATE portfolio SET quantity=?, current_price=?,
                       unrealized_pnl=(? * (? - avg_buy_price)), updated_at=?
                       WHERE asset=?""",
                    (new_qty, price, new_qty, price, now, asset),
                )

    conn.commit()
    conn.close()

    return {
        "order_id": order_id,
        "symbol": symbol.upper(),
        "side": side.upper(),
        "quantity": quantity,
        "price": price,
        "order_type": order_type.upper(),
        "status": "FILLED",
        "paper_trade": True,
        "exchange": exchange,
    }


def create_live_order(
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    order_type: str,
    strategy: str,
) -> dict:
    """
    Log order to SQLite FIRST, then submit to Binance.
    Hard-capped at TRADE_MAX_USD.
    """
    # Safety cap: recalculate quantity if order value exceeds cap
    order_value = quantity * price
    if order_value > TRADE_MAX_USD:
        quantity = TRADE_MAX_USD / price

    now = time.time()
    order_id = _generate_order_id(symbol, side)
    binance_symbol = symbol.upper().replace("/", "").replace("-", "")

    conn = get_db()
    # Log BEFORE API call
    cur = conn.execute(
        """INSERT INTO orders
           (order_id, symbol, side, quantity, price, order_type, status,
            exchange, paper_trade, filled_price, pnl, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            order_id, symbol.upper(), side.upper(), quantity, price,
            order_type.upper(), "PENDING", "binance", 0,
            None, None, now, now,
        ),
    )
    inserted_id = cur.lastrowid
    conn.commit()

    try:
        response = binance_place_order(binance_symbol, side, quantity, order_type)
        filled_price = float(response.get("fills", [{}])[0].get("price", price) if response.get("fills") else price)
        status = response.get("status", "FILLED")

        conn.execute(
            """UPDATE orders SET order_id=?, status=?, filled_price=?, updated_at=?
               WHERE id=?""",
            (
                str(response.get("orderId", order_id)),
                status,
                filled_price,
                time.time(),
                inserted_id,
            ),
        )
        conn.commit()
        conn.close()

        return {
            "order_id": str(response.get("orderId", order_id)),
            "symbol": symbol.upper(),
            "side": side.upper(),
            "quantity": quantity,
            "price": filled_price,
            "order_type": order_type.upper(),
            "status": status,
            "paper_trade": False,
            "exchange": "binance",
            "raw_response": response,
        }
    except Exception as exc:
        conn.execute(
            "UPDATE orders SET status='FAILED', updated_at=? WHERE id=?",
            (time.time(), inserted_id),
        )
        conn.commit()
        conn.close()
        raise exc


# ---------------------------------------------------------------------------
# HTTP utilities
# ---------------------------------------------------------------------------

def _ok(handler: http.server.BaseHTTPRequestHandler, data: dict, code: int = 200) -> None:
    body = json.dumps(data, default=str).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _err(handler: http.server.BaseHTTPRequestHandler, message: str, code: int = 400) -> None:
    _ok(handler, {"error": message}, code)


def _read_body(handler: http.server.BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", 0))
    if length == 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw.decode())
    except json.JSONDecodeError:
        return {}


def _check_admin(handler: http.server.BaseHTTPRequestHandler) -> bool:
    """Validate X-Admin-Secret header against ADMIN_SECRET env var."""
    secret = os.environ.get("ADMIN_SECRET", "")
    if not secret:
        _err(handler, "ADMIN_SECRET not configured", 500)
        return False
    provided = handler.headers.get("X-Admin-Secret", "")
    if not hmac.compare_digest(provided, secret):
        _err(handler, "Unauthorized", 403)
        return False
    return True


def _parse_qs(path: str) -> tuple[str, dict]:
    """Split path and query string, return (path, {param: value})."""
    if "?" in path:
        p, qs = path.split("?", 1)
        params = {}
        for part in qs.split("&"):
            if "=" in part:
                k, v = part.split("=", 1)
                params[urllib.parse.unquote_plus(k)] = urllib.parse.unquote_plus(v)
        return p, params
    return path, {}


# ---------------------------------------------------------------------------
# Endpoint handlers
# ---------------------------------------------------------------------------

def handle_health(handler: http.server.BaseHTTPRequestHandler) -> None:
    conn = get_db()
    order_count = conn.execute("SELECT COUNT(*) AS n FROM orders").fetchone()["n"]
    signal_count = conn.execute("SELECT COUNT(*) AS n FROM signals").fetchone()["n"]
    conn.close()
    _ok(handler, {
        "status": "ok",
        "service": "fm_trading_engine",
        "port": PORT,
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "live_trading_enabled": LIVE_TRADING_ENABLED,
        "trade_max_usd": TRADE_MAX_USD,
        "order_count": order_count,
        "signal_count": signal_count,
    })


def handle_get_prices(handler: http.server.BaseHTTPRequestHandler) -> None:
    """Latest price for each symbol (one row per symbol)."""
    conn = get_db()
    rows = conn.execute(
        """SELECT pf.* FROM price_feeds pf
           INNER JOIN (
               SELECT symbol, MAX(fetched_at) AS latest
               FROM price_feeds GROUP BY symbol
           ) latest ON pf.symbol=latest.symbol AND pf.fetched_at=latest.latest
           ORDER BY pf.symbol"""
    ).fetchall()
    conn.close()
    _ok(handler, {"prices": [dict(r) for r in rows]})


def handle_get_price_symbol(handler: http.server.BaseHTTPRequestHandler, symbol: str) -> None:
    """Price history for a specific symbol (last 100 readings)."""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM price_feeds WHERE symbol=? ORDER BY fetched_at DESC LIMIT 100",
        (symbol.upper(),),
    ).fetchall()
    conn.close()
    _ok(handler, {"symbol": symbol.upper(), "history": [dict(r) for r in rows]})


def handle_get_signals(handler: http.server.BaseHTTPRequestHandler, params: dict) -> None:
    """List recent signals with optional filters: symbol, acted_on, limit."""
    symbol = params.get("symbol")
    acted_on = params.get("acted_on")
    try:
        limit = int(params.get("limit", 50))
    except ValueError:
        limit = 50

    clauses = []
    args: list = []
    if symbol:
        clauses.append("symbol=?")
        args.append(symbol.upper())
    if acted_on is not None:
        clauses.append("acted_on=?")
        args.append(int(acted_on))

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    args.append(limit)

    conn = get_db()
    rows = conn.execute(
        f"SELECT * FROM signals {where} ORDER BY triggered_at DESC LIMIT ?", args
    ).fetchall()
    conn.close()
    _ok(handler, {"signals": [dict(r) for r in rows]})


def handle_get_orders(handler: http.server.BaseHTTPRequestHandler, params: dict) -> None:
    """List orders with optional filters: symbol, status, paper_trade."""
    symbol = params.get("symbol")
    status = params.get("status")
    paper_trade = params.get("paper_trade")

    clauses = []
    args: list = []
    if symbol:
        clauses.append("symbol=?")
        args.append(symbol.upper())
    if status:
        clauses.append("status=?")
        args.append(status.upper())
    if paper_trade is not None:
        clauses.append("paper_trade=?")
        args.append(int(paper_trade))

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

    conn = get_db()
    rows = conn.execute(
        f"SELECT * FROM orders {where} ORDER BY created_at DESC LIMIT 200"
    ).fetchall()
    conn.close()
    _ok(handler, {"orders": [dict(r) for r in rows]})


def handle_get_portfolio(handler: http.server.BaseHTTPRequestHandler) -> None:
    """Current portfolio with total value and unrealized P&L."""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM portfolio ORDER BY asset"
    ).fetchall()
    conn.close()

    positions = [dict(r) for r in rows]
    total_value = sum(
        (p["quantity"] * p["current_price"]) for p in positions if p["current_price"]
    )
    total_unrealized = sum(p["unrealized_pnl"] or 0.0 for p in positions)

    _ok(handler, {
        "portfolio": positions,
        "total_value_usd": round(total_value, 2),
        "total_unrealized_pnl_usd": round(total_unrealized, 2),
    })


def handle_get_strategies(handler: http.server.BaseHTTPRequestHandler) -> None:
    conn = get_db()
    rows = conn.execute("SELECT * FROM strategies ORDER BY name").fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        try:
            d["config"] = json.loads(d["config"])
        except (json.JSONDecodeError, TypeError):
            pass
        result.append(d)
    _ok(handler, {"strategies": result})


def handle_get_analytics(handler: http.server.BaseHTTPRequestHandler) -> None:
    """Win rate, total trades, P&L summary, best/worst trade."""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM orders WHERE status IN ('FILLED') ORDER BY created_at"
    ).fetchall()
    conn.close()

    orders = [dict(r) for r in rows]
    total_trades = len(orders)

    if total_trades == 0:
        _ok(handler, {
            "total_trades": 0,
            "win_rate_pct": 0.0,
            "total_pnl": 0.0,
            "best_trade_pnl": None,
            "worst_trade_pnl": None,
            "paper_trades": 0,
            "live_trades": 0,
        })
        return

    pnls = [o["pnl"] for o in orders if o["pnl"] is not None]
    wins = [p for p in pnls if p > 0]
    win_rate = (len(wins) / len(pnls) * 100.0) if pnls else 0.0
    total_pnl = sum(pnls)
    best = max(pnls) if pnls else None
    worst = min(pnls) if pnls else None
    paper_trades = sum(1 for o in orders if o["paper_trade"])
    live_trades = total_trades - paper_trades

    # Per-symbol breakdown
    symbol_pnl: dict[str, float] = {}
    for o in orders:
        if o["pnl"] is not None:
            symbol_pnl[o["symbol"]] = symbol_pnl.get(o["symbol"], 0.0) + o["pnl"]

    _ok(handler, {
        "total_trades": total_trades,
        "win_rate_pct": round(win_rate, 2),
        "total_pnl": round(total_pnl, 4),
        "best_trade_pnl": round(best, 4) if best is not None else None,
        "worst_trade_pnl": round(worst, 4) if worst is not None else None,
        "paper_trades": paper_trades,
        "live_trades": live_trades,
        "pnl_by_symbol": {k: round(v, 4) for k, v in symbol_pnl.items()},
    })


def handle_execute_order(handler: http.server.BaseHTTPRequestHandler) -> None:
    """POST /execute_order — admin-gated order execution."""
    if not _check_admin(handler):
        return

    body = _read_body(handler)
    symbol = body.get("symbol", "").strip()
    side = body.get("side", "").strip().upper()
    quantity = body.get("quantity")
    order_type = body.get("order_type", "MARKET").strip().upper()
    strategy = body.get("strategy", "manual").strip()

    if not symbol:
        _err(handler, "symbol is required")
        return
    if side not in ("BUY", "SELL"):
        _err(handler, "side must be BUY or SELL")
        return
    try:
        quantity = float(quantity)
        if quantity <= 0:
            raise ValueError
    except (TypeError, ValueError):
        _err(handler, "quantity must be a positive number")
        return

    # Get current price for USD cap check
    price = _current_price_for_symbol(symbol)
    if price is None:
        # Fallback: attempt to use a provided price
        price = float(body.get("price", 0))
    if price <= 0:
        _err(handler, f"No price data for {symbol}; refresh price feeds first")
        return

    # Enforce hard cap
    order_value = quantity * price
    if order_value > TRADE_MAX_USD:
        capped_qty = TRADE_MAX_USD / price
        quantity = round(capped_qty, 8)

    if LIVE_TRADING_ENABLED:
        try:
            result = create_live_order(symbol, side, quantity, price, order_type, strategy)
        except Exception as exc:
            _err(handler, f"Live order failed: {exc}", 502)
            return
    else:
        result = create_paper_order(symbol, side, quantity, price, order_type, strategy)

    _ok(handler, {"order": result, "live_trading": LIVE_TRADING_ENABLED}, 201)


def handle_post_strategies(handler: http.server.BaseHTTPRequestHandler) -> None:
    """POST /strategies — create or update a strategy config (admin-gated)."""
    if not _check_admin(handler):
        return

    body = _read_body(handler)
    name = body.get("name", "").strip()
    config = body.get("config")
    enabled = int(body.get("enabled", 1))

    if not name:
        _err(handler, "name is required")
        return
    if config is None:
        _err(handler, "config is required")
        return

    config_str = json.dumps(config) if not isinstance(config, str) else config
    now = time.time()

    conn = get_db()
    existing = conn.execute("SELECT id FROM strategies WHERE name=?", (name,)).fetchone()
    if existing:
        conn.execute(
            "UPDATE strategies SET config=?, enabled=? WHERE name=?",
            (config_str, enabled, name),
        )
        action = "updated"
    else:
        conn.execute(
            "INSERT INTO strategies (name, config, enabled, created_at) VALUES (?,?,?,?)",
            (name, config_str, enabled, now),
        )
        action = "created"
    conn.commit()
    conn.close()

    _ok(handler, {"action": action, "name": name, "enabled": bool(enabled)}, 201)


def handle_close_position(handler: http.server.BaseHTTPRequestHandler) -> None:
    """POST /close_position — admin-gated; sell entire position for an asset."""
    if not _check_admin(handler):
        return

    body = _read_body(handler)
    asset = body.get("asset", "").strip().upper()
    if not asset:
        _err(handler, "asset is required")
        return

    conn = get_db()
    row = conn.execute(
        "SELECT quantity, current_price FROM portfolio WHERE asset=?", (asset,)
    ).fetchone()
    conn.close()

    if not row:
        _err(handler, f"No open position for asset: {asset}", 404)
        return

    quantity = row["quantity"]
    price = row["current_price"]

    if quantity <= 0:
        _err(handler, f"Position for {asset} has zero quantity", 400)
        return
    if not price:
        _err(handler, f"No current price for {asset}", 400)
        return

    # Derive symbol (assume USDT pair)
    symbol = asset + "USDT"

    if LIVE_TRADING_ENABLED:
        try:
            result = create_live_order(symbol, "SELL", quantity, price, "MARKET", "close_position")
        except Exception as exc:
            _err(handler, f"Live close failed: {exc}", 502)
            return
    else:
        result = create_paper_order(symbol, "SELL", quantity, price, "MARKET", "close_position")

    _ok(handler, {"closed": result, "live_trading": LIVE_TRADING_ENABLED}, 200)


# ---------------------------------------------------------------------------
# HTTP Request Handler
# ---------------------------------------------------------------------------

class TradingEngineHandler(http.server.BaseHTTPRequestHandler):
    """Route all HTTP requests to the appropriate handler."""

    def log_message(self, fmt: str, *args) -> None:  # type: ignore[override]
        pass  # suppress default Apache-style logging

    def do_GET(self) -> None:
        path, params = _parse_qs(self.path)

        if path == "/health":
            handle_health(self)
        elif path == "/prices":
            handle_get_prices(self)
        elif path.startswith("/prices/"):
            symbol = path.split("/prices/", 1)[1].strip("/")
            if not symbol:
                _err(self, "Symbol required in path")
            else:
                handle_get_price_symbol(self, symbol)
        elif path == "/signals":
            handle_get_signals(self, params)
        elif path == "/orders":
            handle_get_orders(self, params)
        elif path == "/portfolio":
            handle_get_portfolio(self)
        elif path == "/strategies":
            handle_get_strategies(self)
        elif path == "/analytics":
            handle_get_analytics(self)
        else:
            _err(self, "Not found", 404)

    def do_POST(self) -> None:
        path, _ = _parse_qs(self.path)

        if path == "/execute_order":
            handle_execute_order(self)
        elif path == "/strategies":
            handle_post_strategies(self)
        elif path == "/close_position":
            handle_close_position(self)
        else:
            _err(self, "Not found", 404)


# ---------------------------------------------------------------------------
# Server bootstrap
# ---------------------------------------------------------------------------

class ThreadedHTTPServer(http.server.ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    init_db()

    # Start background price feed thread
    feed_thread = threading.Thread(target=price_feed_loop, daemon=True, name="price-feed")
    feed_thread.start()

    server = ThreadedHTTPServer(("0.0.0.0", PORT), TradingEngineHandler)
    print(
        f"[TradingEngine] Listening on port {PORT} | "
        f"live_trading={LIVE_TRADING_ENABLED} | max_order_usd=${TRADE_MAX_USD:.0f} | "
        f"db={DB_PATH}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[TradingEngine] Shutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
