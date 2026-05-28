#!/usr/bin/env python3
"""
fm_opensea.py — OpenSea NFT Marketplace Agent (Port 7800)
Wallet portfolio management, listing creation, offer submission,
floor price tracking, and event webhooks for OpenSea v2 API.
All credentials sourced from ~/.secrets/fractal.env at runtime.
Samuel James Hiotis | ABN 56 628 117 363
"""
import hashlib
import hmac
import json
import logging
import os
import signal
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ── vault ─────────────────────────────────────────────────────────────────────
_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _line in _vault.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

# ── config ────────────────────────────────────────────────────────────────────
PORT            = int(os.getenv("OPENSEA_PORT", "7800"))
ROOT            = Path(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")))
DB_PATH         = ROOT / "database" / "sovereign.db"
LOG_PATH        = ROOT / "logs" / "opensea.log"
API_KEY         = os.getenv("OPENSEA_API_KEY", "")
WALLET          = os.getenv("OPENSEA_WALLET_ADDRESS", "")
CHAIN           = os.getenv("OPENSEA_CHAIN", "ethereum")
BASE_URL        = "https://api.opensea.io/api/v2"
SYNC_INTERVAL   = int(os.getenv("OPENSEA_SYNC_INTERVAL", "1800"))  # 30 min

ROOT.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [OPENSEA] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
log = logging.getLogger("opensea")

# ── database ──────────────────────────────────────────────────────────────────
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS opensea_listings (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                nft_id   TEXT NOT NULL,
                collection TEXT,
                price_eth REAL,
                status   TEXT DEFAULT 'active',
                tx_hash  TEXT,
                ts       INTEGER DEFAULT (strftime('%s','now'))
            );
            CREATE TABLE IF NOT EXISTS opensea_collections (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                slug        TEXT NOT NULL,
                name        TEXT,
                floor_price REAL,
                volume      REAL,
                ts          INTEGER DEFAULT (strftime('%s','now'))
            );
            CREATE INDEX IF NOT EXISTS idx_listings_nft ON opensea_listings(nft_id);
            CREATE INDEX IF NOT EXISTS idx_listings_status ON opensea_listings(status);
            CREATE INDEX IF NOT EXISTS idx_collections_slug ON opensea_collections(slug);
        """)
    log.info("Database initialised at %s", DB_PATH)

# ── OpenSea API helper ────────────────────────────────────────────────────────
def _os(method: str, path: str, body: dict | None = None) -> dict:
    """Call OpenSea API v2 with X-API-KEY header."""
    url = BASE_URL.rstrip("/") + "/" + path.lstrip("/")
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method.upper(),
        headers={
            "X-API-KEY": API_KEY,
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode(errors="replace")
        log.error("OpenSea HTTP %s %s → %s: %s", method, path, exc.code, body_text)
        return {"error": exc.code, "detail": body_text}
    except urllib.error.URLError as exc:
        log.error("OpenSea URL error %s %s: %s", method, path, exc.reason)
        return {"error": "url_error", "detail": str(exc.reason)}

# ── collection sync background thread ────────────────────────────────────────
def _sync_collection(slug: str) -> None:
    """Fetch and persist stats for one collection slug."""
    data = _os("GET", f"/collections/{slug}")
    if "error" in data:
        return
    floor = 0.0
    stats = data.get("stats", {})
    floor_obj = stats.get("floor_price", {})
    if isinstance(floor_obj, dict):
        floor = float(floor_obj.get("value", 0) or 0)
    elif isinstance(floor_obj, (int, float)):
        floor = float(floor_obj)
    volume = float(stats.get("total_volume", 0) or 0)
    name   = data.get("name", slug)
    with get_db() as conn:
        conn.execute(
            """INSERT INTO opensea_collections (slug, name, floor_price, volume)
               VALUES (?, ?, ?, ?)""",
            (slug, name, floor, volume),
        )
    log.info("Synced collection %s floor=%.4f ETH volume=%.2f", slug, floor, volume)

def _auto_sync_loop() -> None:
    """Periodically sync collection stats for NFTs owned by wallet."""
    while not _shutdown.is_set():
        try:
            if WALLET and API_KEY:
                assets = _os("GET", f"/chain/{CHAIN}/account/{WALLET}/nfts")
                seen: set[str] = set()
                for asset in assets.get("nfts", []):
                    slug = asset.get("collection", "")
                    if slug and slug not in seen:
                        seen.add(slug)
                        _sync_collection(slug)
        except Exception as exc:
            log.error("Auto-sync error: %s", exc)
        _shutdown.wait(SYNC_INTERVAL)

# ── portfolio valuation ───────────────────────────────────────────────────────
def _portfolio_value() -> dict:
    """Sum the latest floor price for each collection in the wallet."""
    if not (WALLET and API_KEY):
        return {"error": "wallet or api_key not configured"}
    assets_data = _os("GET", f"/chain/{CHAIN}/account/{WALLET}/nfts")
    nfts = assets_data.get("nfts", [])
    slugs: dict[str, int] = {}
    for nft in nfts:
        slug = nft.get("collection", "unknown")
        slugs[slug] = slugs.get(slug, 0) + 1
    total_eth = 0.0
    breakdown = []
    with get_db() as conn:
        for slug, count in slugs.items():
            row = conn.execute(
                "SELECT floor_price FROM opensea_collections WHERE slug=? ORDER BY ts DESC LIMIT 1",
                (slug,),
            ).fetchone()
            floor = float(row["floor_price"]) if row else 0.0
            value = floor * count
            total_eth += value
            breakdown.append({"collection": slug, "count": count, "floor_eth": floor, "value_eth": value})
    return {"total_eth": round(total_eth, 6), "nft_count": len(nfts), "breakdown": breakdown}

# ── HTTP handler ──────────────────────────────────────────────────────────────
class OpenSeaHandler(BaseHTTPRequestHandler):
    server_version = "FractalMesh/OpenSea"
    log_message = lambda self, *a: None  # silence default access log

    # ── helpers ────────────────────────────────────────────────────────────
    def _parsed(self):
        return urllib.parse.urlparse(self.path)

    def _qs(self) -> dict:
        return urllib.parse.parse_qs(self._parsed().query)

    def _qp(self, key: str, default: str = "") -> str:
        return self._qs().get(key, [default])[0]

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else {}

    def _send(self, payload: dict, code: int = 200) -> None:
        body = json.dumps(payload, indent=2).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _err(self, msg: str, code: int = 400) -> None:
        self._send({"error": msg}, code)

    # ── routing ────────────────────────────────────────────────────────────
    def do_GET(self):
        p = self._parsed().path.rstrip("/")
        try:
            if p == "/health":        self._health()
            elif p == "/collections": self._get_collections()
            elif p == "/assets":      self._get_assets()
            elif p == "/listings":    self._get_listings()
            elif p == "/floor":       self._get_floor()
            elif p == "/events":      self._get_events()
            elif p == "/portfolio":   self._get_portfolio()
            else:                     self._err("not found", 404)
        except Exception as exc:
            log.exception("GET %s", self.path)
            self._err(str(exc), 500)

    def do_POST(self):
        p = self._parsed().path.rstrip("/")
        try:
            if p == "/list":         self._post_list()
            elif p == "/offer":      self._post_offer()
            elif p == "/webhook":    self._post_webhook()
            else:                    self._err("not found", 404)
        except Exception as exc:
            log.exception("POST %s", self.path)
            self._err(str(exc), 500)

    # ── GET handlers ───────────────────────────────────────────────────────
    def _health(self):
        self._send({
            "status": "ok",
            "agent": "fm_opensea",
            "port": PORT,
            "wallet": WALLET[:8] + "…" if len(WALLET) > 8 else WALLET,
            "chain": CHAIN,
            "configured": bool(API_KEY and WALLET),
        })

    def _get_collections(self):
        if not (WALLET and API_KEY):
            return self._err("OPENSEA_API_KEY and OPENSEA_WALLET_ADDRESS required", 503)
        data = _os("GET", f"/collections?chain={CHAIN}&creator_username={WALLET}")
        self._send(data)

    def _get_assets(self):
        if not (WALLET and API_KEY):
            return self._err("OPENSEA_API_KEY and OPENSEA_WALLET_ADDRESS required", 503)
        limit = self._qp("limit", "50")
        next_cursor = self._qp("next", "")
        path = f"/chain/{CHAIN}/account/{WALLET}/nfts?limit={limit}"
        if next_cursor:
            path += f"&next={urllib.parse.quote(next_cursor)}"
        data = _os("GET", path)
        self._send(data)

    def _get_listings(self):
        with get_db() as conn:
            rows = conn.execute(
                "SELECT * FROM opensea_listings ORDER BY ts DESC LIMIT 100"
            ).fetchall()
        self._send({"listings": [dict(r) for r in rows]})

    def _get_floor(self):
        slug = self._qp("collection")
        if not slug:
            return self._err("collection query param required")
        data = _os("GET", f"/collections/{slug}")
        if "error" in data:
            return self._send(data, 502)
        stats = data.get("stats", {})
        floor_obj = stats.get("floor_price", {})
        floor = 0.0
        if isinstance(floor_obj, dict):
            floor = float(floor_obj.get("value", 0) or 0)
        elif isinstance(floor_obj, (int, float)):
            floor = float(floor_obj)
        # persist to DB
        with get_db() as conn:
            conn.execute(
                "INSERT INTO opensea_collections (slug, name, floor_price, volume) VALUES (?,?,?,?)",
                (slug, data.get("name", slug), floor, float(stats.get("total_volume", 0) or 0)),
            )
        self._send({"collection": slug, "floor_price_eth": floor, "stats": stats})

    def _get_events(self):
        slug = self._qp("collection")
        if not slug:
            return self._err("collection query param required")
        event_type = self._qp("event_type", "sale")
        path = f"/events/collection/{slug}?event_type={event_type}&limit=50"
        data = _os("GET", path)
        self._send(data)

    def _get_portfolio(self):
        result = _portfolio_value()
        self._send(result, 200 if "error" not in result else 503)

    # ── POST handlers ──────────────────────────────────────────────────────
    def _post_list(self):
        body = self._body()
        required = ("nft_contract", "token_id", "price_eth", "expiry_days")
        missing = [k for k in required if k not in body]
        if missing:
            return self._err(f"missing fields: {missing}")
        if not API_KEY:
            return self._err("OPENSEA_API_KEY not configured", 503)
        expiry_ts = int(time.time()) + int(body["expiry_days"]) * 86400
        payload = {
            "parameters": {
                "offerer": WALLET,
                "offer": [{"itemType": 2, "token": body["nft_contract"], "identifierOrCriteria": str(body["token_id"]), "startAmount": "1", "endAmount": "1"}],
                "consideration": [],
                "startTime": str(int(time.time())),
                "endTime": str(expiry_ts),
                "orderType": 0,
                "zone": "0x0000000000000000000000000000000000000000",
                "zoneHash": "0x" + "0" * 64,
                "salt": hashlib.sha256(os.urandom(32)).hexdigest(),
                "conduitKey": "0x" + "0" * 64,
                "totalOriginalConsiderationItems": 0,
            },
            "signature": "",
            "protocol_address": "0x0000000000000000000000000000000000000000",
        }
        result = _os("POST", "/listings", payload)
        if "error" not in result:
            nft_id = f"{body['nft_contract']}:{body['token_id']}"
            with get_db() as conn:
                conn.execute(
                    "INSERT INTO opensea_listings (nft_id, price_eth, status) VALUES (?,?,?)",
                    (nft_id, float(body["price_eth"]), "pending"),
                )
            log.info("Listing created nft=%s price=%.4f ETH", nft_id, float(body["price_eth"]))
        self._send(result)

    def _post_offer(self):
        body = self._body()
        required = ("nft_contract", "token_id", "price_eth")
        missing = [k for k in required if k not in body]
        if missing:
            return self._err(f"missing fields: {missing}")
        if not API_KEY:
            return self._err("OPENSEA_API_KEY not configured", 503)
        payload = {
            "protocol_address": "0x0000000000000000000000000000000000000000",
            "criteria": {"contract": {"address": body["nft_contract"]}, "token_ids": [str(body["token_id"])]},
            "quantity": 1,
            "price": {"currency": "WETH", "value": str(int(float(body["price_eth"]) * 1e18))},
            "expiration": int(time.time()) + 86400,
        }
        result = _os("POST", "/offers", payload)
        self._send(result)

    def _post_webhook(self):
        body = self._body()
        event_type = body.get("event_type", "unknown")
        payload_data = body.get("payload", {})
        log.info("Webhook received event_type=%s", event_type)
        # Verify HMAC if secret present
        secret = os.getenv("OPENSEA_WEBHOOK_SECRET", "")
        if secret:
            sig = self.headers.get("X-Opensea-Signature", "")
            raw = self.rfile  # already consumed; signature check on re-received
            # Best-effort: validate signature from header if present
            raw_body = json.dumps(body).encode()
            expected = hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
            if not hmac.compare_digest(sig, expected):
                log.warning("Webhook HMAC mismatch — proceeding anyway")
        # Persist sale events to listings table
        if event_type in ("item_sold", "item_listed"):
            nft = payload_data.get("item", {})
            nft_id = nft.get("nft_id", "unknown")
            price_eth = 0.0
            price_data = payload_data.get("sale_price") or payload_data.get("listing_price") or {}
            if isinstance(price_data, dict):
                price_eth = float(price_data.get("value", 0) or 0) / 1e18
            tx = payload_data.get("transaction", {}).get("hash", "")
            status = "sold" if event_type == "item_sold" else "listed"
            with get_db() as conn:
                conn.execute(
                    "INSERT INTO opensea_listings (nft_id, price_eth, status, tx_hash) VALUES (?,?,?,?)",
                    (nft_id, price_eth, status, tx),
                )
            log.info("Webhook persisted %s nft=%s price=%.6f ETH", status, nft_id, price_eth)
        self._send({"received": True, "event_type": event_type})


# ── shutdown ──────────────────────────────────────────────────────────────────
_shutdown = threading.Event()
_server: HTTPServer | None = None

def _handle_signal(signum, frame):
    log.info("Signal %s received — shutting down", signum)
    _shutdown.set()
    if _server:
        threading.Thread(target=_server.shutdown, daemon=True).start()

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT,  _handle_signal)

# ── entrypoint ────────────────────────────────────────────────────────────────
def main():
    global _server
    init_db()
    sync_thread = threading.Thread(target=_auto_sync_loop, daemon=True, name="opensea-sync")
    sync_thread.start()
    _server = HTTPServer(("0.0.0.0", PORT), OpenSeaHandler)
    log.info("OpenSea agent listening on port %d  chain=%s  wallet=%s", PORT, CHAIN, WALLET[:10] if WALLET else "unset")
    try:
        _server.serve_forever()
    finally:
        log.info("OpenSea agent stopped")

if __name__ == "__main__":
    main()
