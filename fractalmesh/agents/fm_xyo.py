"""
FractalMesh OMEGA Titan — XYO Network Agent
Port : 7821
DB   : ~/fmsaas/database/sovereign.db  (table: xyo_witnesses)
Log  : ~/fmsaas/logs/fm_xyo.log

SECURITY NOTE: XYO_PHRASE (seed phrase) is loaded into memory ONLY.
It is NEVER logged, never transmitted in plaintext, never stored in the DB,
and never included in any HTTP response body.
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
XYO_API_KEY = os.getenv("XYO_API_KEY", "")
_XYO_PHRASE = os.getenv("XYO_PHRASE", "")   # NEVER log, print, or expose this variable
PORT        = int(os.getenv("XYO_PORT", "7821"))

# Primary and fallback base URLs
_XYO_PRIMARY  = "https://api.xyo.network/v1"
_XYO_FALLBACK = "https://beta.api.xyo.network/v1"

DB_PATH  = os.path.expanduser("~/fmsaas/database/sovereign.db")
LOG_PATH = os.path.expanduser("~/fmsaas/logs/fm_xyo.log")

# ── Logging ────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(DB_PATH),  exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [fm_xyo] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("fm_xyo")

# Sanity check — confirm phrase is loaded but NEVER reveal it
if _XYO_PHRASE:
    log.info("XYO_PHRASE loaded (redacted, %d words)", len(_XYO_PHRASE.split()))
else:
    log.warning("XYO_PHRASE not set — witness signing will be unavailable")

# ── SQLite setup ───────────────────────────────────────────────────────────────

def get_db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""
        CREATE TABLE IF NOT EXISTS xyo_witnesses (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            lat       REAL,
            lng       REAL,
            altitude  REAL,
            reward    REAL,
            tx_hash   TEXT,
            ts        INTEGER
        )
    """)
    con.commit()
    return con


def db_log_witness(lat: float, lng: float, altitude: float, reward: float, tx_hash: str) -> None:
    try:
        con = get_db()
        con.execute(
            "INSERT INTO xyo_witnesses (lat, lng, altitude, reward, tx_hash, ts) VALUES (?,?,?,?,?,?)",
            (lat, lng, altitude, reward, tx_hash, int(time.time())),
        )
        con.commit()
        con.close()
    except Exception as exc:
        log.error("db_log_witness failed: %s", exc)

# ── XYO API helper ─────────────────────────────────────────────────────────────

def _xyo_request(
    path: str,
    method: str = "GET",
    params: dict | None = None,
    body: dict | None = None,
) -> dict | list:
    """
    Authenticated request to XYO Network API.
    Tries primary URL; falls back to beta URL on connection errors.
    """
    for base_url in (_XYO_PRIMARY, _XYO_FALLBACK):
        url = base_url.rstrip("/") + path
        if params:
            url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        headers = {
            "Accept":        "application/json",
            "Authorization": f"Bearer {XYO_API_KEY}",
            "Content-Type":  "application/json",
        }
        data = json.dumps(body).encode() if body else None
        req  = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode(errors="replace")
            log.error("XYO HTTP %s %s: %s", exc.code, path, err_body)
            # 4xx errors are definitive — do not retry on fallback
            if 400 <= exc.code < 500:
                raise
            log.warning("Trying fallback URL for %s", path)
            continue  # attempt fallback
        except urllib.error.URLError as exc:
            log.warning("XYO connection error (%s), trying fallback: %s", base_url, exc.reason)
            continue
        except Exception as exc:
            log.error("XYO request error %s: %s", path, exc)
            raise
    raise RuntimeError(f"XYO API unreachable at both primary and fallback for {path}")

# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_qs(query: str) -> dict:
    parsed = urllib.parse.parse_qs(query or "")
    return {k: v[0] for k, v in parsed.items()}

# ── Request handler ────────────────────────────────────────────────────────────

class XYOHandler(BaseHTTPRequestHandler):
    """HTTP handler for the XYO Network agent."""

    server_version = "FractalMesh-XYO/1.0"

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
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def log_message(self, fmt, *args):
        log.info("HTTP %s", fmt % args)

    # ── GET dispatcher ─────────────────────────────────────────────────────────

    def do_GET(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        qs     = _parse_qs(parsed.query)
        path   = parsed.path

        routes = {
            "/health":      self._health,
            "/balance":     self._balance,
            "/rewards":     self._rewards,
            "/devices":     self._devices,
            "/geodata":     self._geodata,
            "/network":     self._network,
            "/leaderboard": self._leaderboard,
        }
        handler = routes.get(path)
        if handler:
            try:
                handler(qs)
            except urllib.error.HTTPError as exc:
                self._error(f"XYO API error: {exc.code}", exc.code if exc.code < 600 else 502)
            except RuntimeError as exc:
                self._error(str(exc), 503)
            except Exception as exc:
                log.exception("Route %s error", path)
                self._error(str(exc))
        else:
            self._error("Not found", 404)

    def do_POST(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path   = parsed.path
        if path == "/witness":
            try:
                self._witness(self._read_body())
            except urllib.error.HTTPError as exc:
                self._error(f"XYO API error: {exc.code}", exc.code if exc.code < 600 else 502)
            except Exception as exc:
                log.exception("POST /witness error")
                self._error(str(exc))
        else:
            self._error("Not found", 404)

    # ── route implementations ──────────────────────────────────────────────────

    def _health(self, qs: dict) -> None:
        """
        GET /health
        Checks node connectivity by probing the network/stats endpoint.
        """
        connected = False
        node_info = {}
        try:
            node_info = _xyo_request("/network/stats")
            connected = True
        except Exception as exc:
            log.warning("Health check failed: %s", exc)

        self._send_json({
            "status":        "ok" if connected else "degraded",
            "agent":         "fm_xyo",
            "port":          PORT,
            "node_connected": connected,
            "node_info":     node_info,
            "phrase_loaded": bool(_XYO_PHRASE),
            "ts":            int(time.time()),
        })

    def _balance(self, qs: dict) -> None:
        """GET /balance — XYO token balance for the authenticated account."""
        data = _xyo_request("/token/balance")
        self._send_json({
            "xyo_balance": data.get("balance"),
            "unit":        data.get("unit", "XYO"),
            "raw":         data,
        })

    def _rewards(self, qs: dict) -> None:
        """GET /rewards — reward history for the authenticated account."""
        data = _xyo_request("/rewards/history")
        history = data.get("history", data if isinstance(data, list) else [])
        self._send_json({
            "count":   len(history),
            "rewards": history,
        })

    def _devices(self, qs: dict) -> None:
        """GET /devices — registered XYO sentinel/bridge/archivist devices."""
        data = _xyo_request("/devices")
        devices = data.get("devices", data if isinstance(data, list) else [])
        self._send_json({
            "count":   len(devices),
            "devices": devices,
        })

    def _geodata(self, qs: dict) -> None:
        """
        GET /geodata?lat=&lng=&radius=
        Returns nearby XYO data points within radius meters.
        """
        lat    = qs.get("lat", "")
        lng    = qs.get("lng", "")
        radius = qs.get("radius", "1000")
        if not lat or not lng:
            self._error("lat and lng params required", 400)
            return
        params = {"lat": lat, "lng": lng, "radius": radius}
        data = _xyo_request("/geodata", params=params)
        points = data.get("data", data if isinstance(data, list) else [])
        self._send_json({
            "lat":    lat,
            "lng":    lng,
            "radius": radius,
            "count":  len(points),
            "points": points,
        })

    def _witness(self, body: dict) -> None:
        """
        POST /witness
        Body: {lat, lng, altitude, accuracy, timestamp}
        Submits a geo-witness to the XYO network.

        SECURITY: XYO_PHRASE is never transmitted in any field of this request.
        The API key (Bearer token) is the only credential sent over the wire.
        """
        # Validate required fields
        required = ["lat", "lng"]
        for field in required:
            if field not in body:
                self._error(f"'{field}' is required in witness payload", 400)
                return

        lat      = float(body["lat"])
        lng      = float(body["lng"])
        altitude = float(body.get("altitude", 0.0))
        accuracy = float(body.get("accuracy", 10.0))
        ts       = int(body.get("timestamp", time.time()))

        # Build witness payload — phrase is NEVER included
        payload = {
            "lat":       lat,
            "lng":       lng,
            "altitude":  altitude,
            "accuracy":  accuracy,
            "timestamp": ts,
        }

        log.info("Submitting witness lat=%.6f lng=%.6f alt=%.1f", lat, lng, altitude)
        data = _xyo_request("/witness", method="POST", body=payload)

        reward  = float(data.get("reward", 0.0))
        tx_hash = data.get("tx_hash", data.get("hash", ""))

        db_log_witness(lat, lng, altitude, reward, tx_hash)
        log.info("Witness accepted: reward=%.4f XYO tx=%s", reward, tx_hash or "none")

        self._send_json({
            "status":    "witnessed",
            "lat":       lat,
            "lng":       lng,
            "altitude":  altitude,
            "accuracy":  accuracy,
            "timestamp": ts,
            "reward":    reward,
            "tx_hash":   tx_hash,
            "raw":       data,
        })

    def _network(self, qs: dict) -> None:
        """GET /network — XYO network stats: node count, witness count, etc."""
        data = _xyo_request("/network/stats")
        self._send_json(data)

    def _leaderboard(self, qs: dict) -> None:
        """GET /leaderboard — top XYO earners by reward volume."""
        data = _xyo_request("/leaderboard")
        leaders = data.get("leaders", data if isinstance(data, list) else [])
        self._send_json({
            "count":   len(leaders),
            "leaders": leaders,
        })


# ── Signal handling ────────────────────────────────────────────────────────────
_running = True

def _shutdown(signum, frame):
    global _running
    log.info("Signal %s received — shutting down fm_xyo", signum)
    _running = False

signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT,  _shutdown)

# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    get_db().close()

    server = HTTPServer(("0.0.0.0", PORT), XYOHandler)
    log.info("fm_xyo listening on port %d", PORT)

    global _running
    while _running:
        server.handle_request()

    log.info("fm_xyo stopped")


if __name__ == "__main__":
    main()
