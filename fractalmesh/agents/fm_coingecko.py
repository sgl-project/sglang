"""
FractalMesh OMEGA Titan — CoinGecko Market Data Agent
Port : 7820
DB   : ~/fmsaas/database/sovereign.db  (table: coingecko_queries)
Log  : ~/fmsaas/logs/fm_coingecko.log
"""

# ── Vault loading ──────────────────────────────────────────────────────────────
import os

_VAULT = os.path.expanduser("~/.secrets/fractal.env")
if os.path.isfile(_VAULT):
    with open(_VAULT) as _fh:
        for _line in _fh:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

# ── stdlib imports ─────────────────────────────────────────────────────────────
import json
import logging
import signal
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

# ── Configuration ──────────────────────────────────────────────────────────────
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY", "")   # optional — pro key
COINGECKO_BASE    = "https://api.coingecko.com/api/v3"
PORT              = int(os.getenv("COINGECKO_PORT", "7820"))

DB_PATH  = os.path.expanduser("~/fmsaas/database/sovereign.db")
LOG_PATH = os.path.expanduser("~/fmsaas/logs/fm_coingecko.log")

# ── Logging ────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(DB_PATH),  exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [fm_coingecko] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("fm_coingecko")

# ── In-memory price cache (60-second TTL) ──────────────────────────────────────
# Structure: { cache_key: (timestamp, data) }
_price_cache: dict[str, tuple[float, object]] = {}
_CACHE_TTL = 60  # seconds


def _cache_get(key: str):
    entry = _price_cache.get(key)
    if entry and (time.time() - entry[0]) < _CACHE_TTL:
        return entry[1]
    return None


def _cache_set(key: str, data) -> None:
    _price_cache[key] = (time.time(), data)

# ── SQLite setup ───────────────────────────────────────────────────────────────

def get_db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""
        CREATE TABLE IF NOT EXISTS coingecko_queries (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            coin_id    TEXT,
            price_usd  REAL,
            market_cap REAL,
            ts         INTEGER
        )
    """)
    con.commit()
    return con


def db_log_price(coin_id: str, price_usd: float, market_cap: float) -> None:
    try:
        con = get_db()
        con.execute(
            "INSERT INTO coingecko_queries (coin_id, price_usd, market_cap, ts) VALUES (?,?,?,?)",
            (coin_id, price_usd, market_cap, int(time.time())),
        )
        con.commit()
        con.close()
    except Exception as exc:
        log.error("db_log_price failed: %s", exc)

# ── CoinGecko API helper ───────────────────────────────────────────────────────

def _cg_get(path: str, params: dict | None = None) -> dict | list:
    """
    GET request to CoinGecko API v3.
    Adds demo API key header when COINGECKO_API_KEY is set.
    """
    url = COINGECKO_BASE.rstrip("/") + path
    if params:
        url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    headers = {"Accept": "application/json"}
    if COINGECKO_API_KEY:
        headers["x-cg-demo-api-key"] = COINGECKO_API_KEY
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        log.error("CoinGecko HTTP %s %s: %s", exc.code, path, body)
        raise
    except Exception as exc:
        log.error("CoinGecko request error %s: %s", path, exc)
        raise

# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_qs(query: str) -> dict:
    parsed = urllib.parse.parse_qs(query or "")
    return {k: v[0] for k, v in parsed.items()}

# ── Request handler ────────────────────────────────────────────────────────────

class CoinGeckoHandler(BaseHTTPRequestHandler):
    """HTTP handler for the CoinGecko agent."""

    server_version = "FractalMesh-CoinGecko/1.0"

    # ── utilities ──────────────────────────────────────────────────────────────

    def _send_json(self, data, status: int = 200) -> None:
        body = json.dumps(data, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, msg: str, status: int = 500) -> None:
        self._send_json({"error": msg}, status)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw)

    def log_message(self, fmt, *args):
        log.info("HTTP %s", fmt % args)

    # ── GET dispatcher ─────────────────────────────────────────────────────────

    def do_GET(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        qs     = _parse_qs(parsed.query)
        path   = parsed.path

        routes = {
            "/health":    self._health,
            "/price":     self._price,
            "/coin":      self._coin,
            "/market":    self._market,
            "/trending":  self._trending,
            "/history":   self._history,
            "/chart":     self._chart,
            "/exchanges": self._exchanges,
            "/global":    self._global,
        }
        handler = routes.get(path)
        if handler:
            try:
                handler(qs)
            except urllib.error.HTTPError as exc:
                self._error(f"CoinGecko API error: {exc.code}", exc.code if exc.code < 600 else 502)
            except Exception as exc:
                log.exception("Route %s error", path)
                self._error(str(exc))
        else:
            self._error("Not found", 404)

    def do_POST(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path   = parsed.path
        if path == "/portfolio":
            try:
                self._portfolio(self._read_body())
            except Exception as exc:
                log.exception("POST /portfolio error")
                self._error(str(exc))
        else:
            self._error("Not found", 404)

    # ── route implementations ──────────────────────────────────────────────────

    def _health(self, qs: dict) -> None:
        """GET /health — ping CoinGecko /ping endpoint."""
        data = _cg_get("/ping")
        self._send_json({
            "status":     "ok",
            "agent":      "fm_coingecko",
            "port":       PORT,
            "cg_status":  data.get("gecko_says", "unknown"),
            "pro_key":    bool(COINGECKO_API_KEY),
            "ts":         int(time.time()),
        })

    def _price(self, qs: dict) -> None:
        """
        GET /price?ids=bitcoin,ethereum&currencies=usd,eur
        Returns spot price with 24h change and market cap.
        Cached for 60 seconds.
        """
        ids        = qs.get("ids", "bitcoin")
        currencies = qs.get("currencies", "usd")
        cache_key  = f"price:{ids}:{currencies}"
        cached = _cache_get(cache_key)
        if cached is not None:
            self._send_json({"cached": True, "data": cached})
            return
        data = _cg_get("/simple/price", {
            "ids":                   ids,
            "vs_currencies":         currencies,
            "include_24hr_change":   "true",
            "include_market_cap":    "true",
        })
        _cache_set(cache_key, data)
        # Log each coin to DB
        for coin_id, vals in (data or {}).items():
            price_usd = vals.get("usd", 0.0)
            mcap      = vals.get("usd_market_cap", 0.0)
            db_log_price(coin_id, price_usd, mcap)
        self._send_json({"cached": False, "data": data})

    def _coin(self, qs: dict) -> None:
        """
        GET /coin?id=bitcoin
        Full coin data: name, symbol, price, market cap, ATH, etc.
        """
        coin_id = qs.get("id", "")
        if not coin_id:
            self._error("id param required", 400)
            return
        cache_key = f"coin:{coin_id}"
        cached = _cache_get(cache_key)
        if cached is not None:
            self._send_json({"cached": True, "data": cached})
            return
        data = _cg_get(f"/coins/{coin_id}")
        _cache_set(cache_key, data)
        # Flatten key fields for readability
        market = data.get("market_data", {})
        summary = {
            "id":                     data.get("id"),
            "name":                   data.get("name"),
            "symbol":                 data.get("symbol"),
            "current_price_usd":      market.get("current_price", {}).get("usd"),
            "market_cap_usd":         market.get("market_cap", {}).get("usd"),
            "total_volume_usd":       market.get("total_volume", {}).get("usd"),
            "price_change_24h_pct":   market.get("price_change_percentage_24h"),
            "ath_usd":                market.get("ath", {}).get("usd"),
            "ath_date":               market.get("ath_date", {}).get("usd"),
            "circulating_supply":     market.get("circulating_supply"),
            "max_supply":             market.get("max_supply"),
            "description_en":         (data.get("description", {}).get("en") or "")[:500],
            "homepage":               (data.get("links", {}).get("homepage") or [""])[0],
        }
        db_log_price(coin_id, summary.get("current_price_usd") or 0.0, summary.get("market_cap_usd") or 0.0)
        self._send_json({"cached": False, "summary": summary})

    def _market(self, qs: dict) -> None:
        """GET /market — top 50 coins by market cap."""
        cache_key = "market:top50"
        cached = _cache_get(cache_key)
        if cached is not None:
            self._send_json({"cached": True, "count": len(cached), "coins": cached})
            return
        data = _cg_get("/coins/markets", {
            "vs_currency": "usd",
            "order":       "market_cap_desc",
            "per_page":    "50",
            "page":        "1",
        })
        _cache_set(cache_key, data)
        self._send_json({"cached": False, "count": len(data), "coins": data})

    def _trending(self, qs: dict) -> None:
        """GET /trending — trending searches on CoinGecko."""
        cache_key = "trending"
        cached = _cache_get(cache_key)
        if cached is not None:
            self._send_json({"cached": True, "data": cached})
            return
        data = _cg_get("/search/trending")
        _cache_set(cache_key, data)
        self._send_json({"cached": False, "data": data})

    def _history(self, qs: dict) -> None:
        """
        GET /history?id=bitcoin&date=01-01-2024
        Historical price data for a specific date (DD-MM-YYYY).
        """
        coin_id = qs.get("id", "")
        date    = qs.get("date", "")  # format: DD-MM-YYYY
        if not coin_id or not date:
            self._error("id and date params required (date: DD-MM-YYYY)", 400)
            return
        data = _cg_get(f"/coins/{coin_id}/history", {"date": date, "localization": "false"})
        market = data.get("market_data", {})
        self._send_json({
            "id":              coin_id,
            "date":            date,
            "price_usd":       market.get("current_price", {}).get("usd"),
            "market_cap_usd":  market.get("market_cap", {}).get("usd"),
            "volume_usd":      market.get("total_volume", {}).get("usd"),
        })

    def _chart(self, qs: dict) -> None:
        """
        GET /chart?id=bitcoin&days=30
        OHLC price + volume chart data for past N days.
        """
        coin_id = qs.get("id", "")
        days    = qs.get("days", "30")
        if not coin_id:
            self._error("id param required", 400)
            return
        cache_key = f"chart:{coin_id}:{days}"
        cached = _cache_get(cache_key)
        if cached is not None:
            self._send_json({"cached": True, "id": coin_id, "days": days, "data": cached})
            return
        data = _cg_get(f"/coins/{coin_id}/market_chart", {"vs_currency": "usd", "days": days})
        _cache_set(cache_key, data)
        # Summarize: return count of data points, first/last prices
        prices = data.get("prices", [])
        self._send_json({
            "cached":       False,
            "id":           coin_id,
            "days":         days,
            "point_count":  len(prices),
            "first_price":  prices[0] if prices else None,
            "last_price":   prices[-1] if prices else None,
            "prices":       prices,
            "market_caps":  data.get("market_caps", []),
            "total_volumes": data.get("total_volumes", []),
        })

    def _exchanges(self, qs: dict) -> None:
        """GET /exchanges — top exchanges ranked by trade volume."""
        cache_key = "exchanges"
        cached = _cache_get(cache_key)
        if cached is not None:
            self._send_json({"cached": True, "count": len(cached), "exchanges": cached})
            return
        data = _cg_get("/exchanges", {"per_page": "20", "page": "1"})
        _cache_set(cache_key, data)
        self._send_json({"cached": False, "count": len(data), "exchanges": data})

    def _global(self, qs: dict) -> None:
        """GET /global — global market data: total market cap, BTC dominance, etc."""
        cache_key = "global"
        cached = _cache_get(cache_key)
        if cached is not None:
            self._send_json({"cached": True, "data": cached})
            return
        data = _cg_get("/global")
        _cache_set(cache_key, data)
        self._send_json({"cached": False, "data": data})

    def _portfolio(self, body: dict) -> None:
        """
        POST /portfolio
        Body: {"holdings": [{"coin_id": "bitcoin", "amount": 0.5}, ...]}
        Returns current USD value for each holding + grand total.
        """
        holdings = body.get("holdings", [])
        if not holdings:
            self._error("holdings list required", 400)
            return

        coin_ids = ",".join(h["coin_id"] for h in holdings if "coin_id" in h)
        if not coin_ids:
            self._error("no valid coin_id entries in holdings", 400)
            return

        # Use /simple/price — check cache first
        cache_key = f"price:{coin_ids}:usd"
        prices_data = _cache_get(cache_key)
        if prices_data is None:
            prices_data = _cg_get("/simple/price", {
                "ids":                  coin_ids,
                "vs_currencies":        "usd",
                "include_24hr_change":  "true",
                "include_market_cap":   "true",
            })
            _cache_set(cache_key, prices_data)

        result = []
        grand_total = 0.0
        for h in holdings:
            cid    = h.get("coin_id", "")
            amount = float(h.get("amount", 0))
            info   = (prices_data or {}).get(cid, {})
            usd    = float(info.get("usd", 0))
            value  = usd * amount
            grand_total += value
            result.append({
                "coin_id":         cid,
                "amount":          amount,
                "price_usd":       usd,
                "value_usd":       value,
                "change_24h_pct":  info.get("usd_24h_change"),
                "market_cap_usd":  info.get("usd_market_cap"),
            })
            db_log_price(cid, usd, info.get("usd_market_cap") or 0.0)

        self._send_json({
            "holdings":    result,
            "total_usd":   grand_total,
            "as_of":       int(time.time()),
        })


# ── Signal handling ────────────────────────────────────────────────────────────
_running = True

def _shutdown(signum, frame):
    global _running
    log.info("Signal %s received — shutting down fm_coingecko", signum)
    _running = False

signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT,  _shutdown)

# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    get_db().close()

    server = HTTPServer(("0.0.0.0", PORT), CoinGeckoHandler)
    log.info("fm_coingecko listening on port %d", PORT)

    global _running
    while _running:
        server.handle_request()

    log.info("fm_coingecko stopped")


if __name__ == "__main__":
    main()
