"""
FractalMesh OMEGA Titan — Blockchain/Web3 Bridge
Port : 7866
DB   : ~/fmsaas/database/sovereign.db  (WAL mode)
Log  : ~/fmsaas/logs/fm_blockchain_bridge.log

Read-only blockchain data aggregator + unsigned transaction builder.
Fetches balances, transaction history, gas prices, token data, and NFT
holdings. Used by other agents to monitor wallets and prepare transactions
for user review before any signing occurs.

SAFETY GUARANTEE:
  - NEVER signs transactions
  - NEVER stores or logs private keys / seed phrases
  - /transfer/prepare returns an unsigned display object ONLY
  - ETH_PRIVATE_KEY and PRIVATE_KEY are intentionally never read
"""

# ── Vault loading ──────────────────────────────────────────────────────────────
import os
from pathlib import Path

_ENV_FILE = Path.home() / ".secrets" / "fractal.env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

# ── stdlib imports ─────────────────────────────────────────────────────────────
import base64
import binascii
import hashlib
import hmac
import json
import logging
import re
import sqlite3
import struct
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

# ── Configuration ──────────────────────────────────────────────────────────────
PORT               = int(os.getenv("BLOCKCHAIN_BRIDGE_PORT", "7866"))
ALCHEMY_API_KEY    = os.getenv("ALCHEMY_API_KEY", "")
ALCHEMY_NETWORK    = os.getenv("ALCHEMY_NETWORK", "eth-mainnet")
ETHERSCAN_API_KEY  = os.getenv("ETHERSCAN_API_KEY", "")
ADMIN_SECRET       = os.getenv("ADMIN_SECRET", "")

# Intentionally absent: ETH_PRIVATE_KEY, PRIVATE_KEY — never read from vault.

ALCHEMY_URL  = f"https://{ALCHEMY_NETWORK}.g.alchemy.com/v2/{ALCHEMY_API_KEY}"
ETHERSCAN_URL = "https://api.etherscan.io/api"
COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price"

DB_PATH  = Path.home() / "fmsaas" / "database" / "sovereign.db"
LOG_PATH = Path.home() / "fmsaas" / "logs" / "fm_blockchain_bridge.log"

REFRESH_INTERVAL = 300   # seconds between background balance refreshes
REQUEST_TIMEOUT  = 15    # HTTP timeout in seconds
GAS_HISTORY_LIMIT = 100  # rows returned by /gas/history

# ── Logging ────────────────────────────────────────────────────────────────────
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [fm_blockchain_bridge] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_PATH)),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("fm_blockchain_bridge")

_START_TIME = time.time()

# ── Private-key / seed-phrase safety patterns ──────────────────────────────────
_HEX64_RE    = re.compile(r'^(0x)?[0-9a-fA-F]{64}$')
_MNEMONIC_RE = re.compile(r'^(\w+\s+){11,23}\w+$')

def _looks_like_secret(value: str) -> bool:
    """Return True if value resembles a private key or seed phrase."""
    v = value.strip()
    if _HEX64_RE.match(v):
        return True
    words = v.split()
    if len(words) in (12, 24) and _MNEMONIC_RE.match(v):
        return True
    return False

# ── Database ───────────────────────────────────────────────────────────────────
def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn

def _init_db() -> None:
    conn = _get_db()
    with conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS wallet_watches (
                id               INTEGER PRIMARY KEY,
                address          TEXT    UNIQUE,
                label            TEXT,
                chain            TEXT    DEFAULT 'eth',
                last_balance_eth REAL,
                last_balance_usd REAL,
                last_checked     REAL,
                created_at       REAL
            );

            CREATE TABLE IF NOT EXISTS tx_cache (
                id              INTEGER PRIMARY KEY,
                address         TEXT,
                chain           TEXT,
                tx_hash         TEXT    UNIQUE,
                from_addr       TEXT,
                to_addr         TEXT,
                value_eth       REAL,
                gas_used        INTEGER,
                gas_price_gwei  REAL,
                block_number    INTEGER,
                status          TEXT,
                timestamp       REAL,
                cached_at       REAL
            );

            CREATE TABLE IF NOT EXISTS gas_history (
                id               INTEGER PRIMARY KEY,
                chain            TEXT,
                base_fee_gwei    REAL,
                priority_fee_gwei REAL,
                slow_gwei        REAL,
                standard_gwei    REAL,
                fast_gwei        REAL,
                fetched_at       REAL
            );

            CREATE TABLE IF NOT EXISTS nft_holdings (
                id          INTEGER PRIMARY KEY,
                address     TEXT,
                contract    TEXT,
                token_id    TEXT,
                name        TEXT,
                description TEXT,
                image_url   TEXT,
                chain       TEXT,
                fetched_at  REAL
            );

            CREATE INDEX IF NOT EXISTS idx_wallet_address ON wallet_watches(address);
            CREATE INDEX IF NOT EXISTS idx_tx_address    ON tx_cache(address);
            CREATE INDEX IF NOT EXISTS idx_gas_chain     ON gas_history(chain);
            CREATE INDEX IF NOT EXISTS idx_nft_address   ON nft_holdings(address);
        """)
    conn.close()
    log.info("Database initialised at %s", DB_PATH)

# ── Alchemy JSON-RPC helper ────────────────────────────────────────────────────
def _alchemy_rpc(method: str, params: list) -> dict:
    """Send a JSON-RPC request to Alchemy and return the parsed response."""
    if not ALCHEMY_API_KEY:
        raise RuntimeError("ALCHEMY_API_KEY not configured")
    payload = json.dumps({
        "jsonrpc": "2.0",
        "method":  method,
        "params":  params,
        "id":      1,
    }).encode()
    req = urllib.request.Request(
        ALCHEMY_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        data = json.loads(resp.read().decode())
    if "error" in data:
        raise RuntimeError(f"Alchemy RPC error: {data['error']}")
    return data

def _alchemy_get(path: str, params: dict) -> dict:
    """Send an Alchemy REST GET request (for NFT API etc.)."""
    if not ALCHEMY_API_KEY:
        raise RuntimeError("ALCHEMY_API_KEY not configured")
    base = f"https://{ALCHEMY_NETWORK}.g.alchemy.com/v2/{ALCHEMY_API_KEY}"
    qs   = "&".join(f"{k}={urllib.request.quote(str(v))}" for k, v in params.items())
    url  = f"{base}/{path}?{qs}"
    req  = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        return json.loads(resp.read().decode())

# ── Blockchain helpers ─────────────────────────────────────────────────────────
def _hex_to_eth(hex_val: str) -> float:
    """Convert a hex wei string to ETH float."""
    return int(hex_val, 16) / 1e18

def _hex_to_gwei(hex_val: str) -> float:
    """Convert a hex wei string to Gwei float."""
    return int(hex_val, 16) / 1e9

def _eth_to_hex(eth: float) -> str:
    """Convert an ETH float to a hex wei string."""
    wei = int(eth * 1e18)
    return hex(wei)

def _int_to_hex(n: int) -> str:
    return hex(n)

def _normalise_address(addr: str) -> str:
    """Return lower-case address, raise ValueError if invalid."""
    addr = addr.strip()
    if not re.match(r'^0x[0-9a-fA-F]{40}$', addr):
        raise ValueError(f"Invalid Ethereum address: {addr!r}")
    return addr.lower()

# ── ETH price from CoinGecko ───────────────────────────────────────────────────
def _get_eth_usd_price() -> float:
    url = f"{COINGECKO_URL}?ids=ethereum&vs_currencies=usd"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
        return float(data["ethereum"]["usd"])
    except Exception as exc:
        log.warning("CoinGecko price fetch failed: %s", exc)
        return 0.0

# ── Balance fetch ──────────────────────────────────────────────────────────────
def _fetch_balance(address: str) -> tuple[float, float]:
    """Return (balance_eth, balance_usd) for address."""
    result = _alchemy_rpc("eth_getBalance", [address, "latest"])
    eth    = _hex_to_eth(result["result"])
    price  = _get_eth_usd_price()
    usd    = eth * price
    return eth, usd

# ── Gas prices ─────────────────────────────────────────────────────────────────
def _fetch_gas_prices() -> dict:
    """Return gas price dict with slow/standard/fast in gwei."""
    gas_result  = _alchemy_rpc("eth_gasPrice", [])
    prio_result = _alchemy_rpc("eth_maxPriorityFeePerGas", [])

    base_gwei  = _hex_to_gwei(gas_result["result"])
    prio_gwei  = _hex_to_gwei(prio_result["result"])

    # Approximate tiers: slow = base, standard = base+priority, fast = base+2x priority
    slow_gwei     = round(base_gwei, 3)
    standard_gwei = round(base_gwei + prio_gwei, 3)
    fast_gwei     = round(base_gwei + prio_gwei * 2, 3)

    return {
        "base_fee_gwei":     round(base_gwei, 3),
        "priority_fee_gwei": round(prio_gwei, 3),
        "slow_gwei":         slow_gwei,
        "standard_gwei":     standard_gwei,
        "fast_gwei":         fast_gwei,
    }

def _store_gas_snapshot(gas: dict) -> None:
    conn = _get_db()
    with conn:
        conn.execute(
            """INSERT INTO gas_history
               (chain, base_fee_gwei, priority_fee_gwei,
                slow_gwei, standard_gwei, fast_gwei, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (ALCHEMY_NETWORK,
             gas["base_fee_gwei"], gas["priority_fee_gwei"],
             gas["slow_gwei"], gas["standard_gwei"], gas["fast_gwei"],
             time.time()),
        )
    conn.close()

# ── Latest block ───────────────────────────────────────────────────────────────
def _fetch_latest_block() -> dict:
    result = _alchemy_rpc("eth_blockNumber", [])
    block_hex = result["result"]
    block_num = int(block_hex, 16)
    return {"block_number": block_num, "block_hex": block_hex, "chain": ALCHEMY_NETWORK}

# ── Transaction history (Etherscan) ───────────────────────────────────────────
def _fetch_transactions(address: str) -> list[dict]:
    """Fetch recent transactions from Etherscan and cache them."""
    if not ETHERSCAN_API_KEY:
        raise RuntimeError("ETHERSCAN_API_KEY not configured")
    url = (
        f"{ETHERSCAN_URL}?module=account&action=txlist"
        f"&address={address}&apikey={ETHERSCAN_API_KEY}&sort=desc&offset=20"
    )
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        data = json.loads(resp.read().decode())

    if data.get("status") != "1":
        # No transactions or API error — return empty list gracefully
        return []

    txs = []
    conn = _get_db()
    with conn:
        for tx in data.get("result", []):
            value_eth      = int(tx.get("value", "0")) / 1e18
            gas_used       = int(tx.get("gasUsed", "0"))
            gas_price_gwei = int(tx.get("gasPrice", "0")) / 1e9
            block_number   = int(tx.get("blockNumber", "0"))
            ts             = float(tx.get("timeStamp", "0"))
            status         = "success" if tx.get("txreceipt_status") == "1" else "failed"
            tx_hash        = tx.get("hash", "")
            from_addr      = tx.get("from", "")
            to_addr        = tx.get("to", "")

            record = {
                "tx_hash":        tx_hash,
                "from_addr":      from_addr,
                "to_addr":        to_addr,
                "value_eth":      value_eth,
                "gas_used":       gas_used,
                "gas_price_gwei": gas_price_gwei,
                "block_number":   block_number,
                "status":         status,
                "timestamp":      ts,
            }
            txs.append(record)

            # Upsert into cache
            try:
                conn.execute(
                    """INSERT INTO tx_cache
                       (address, chain, tx_hash, from_addr, to_addr,
                        value_eth, gas_used, gas_price_gwei, block_number,
                        status, timestamp, cached_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(tx_hash) DO UPDATE SET
                           status    = excluded.status,
                           cached_at = excluded.cached_at""",
                    (address, ALCHEMY_NETWORK, tx_hash, from_addr, to_addr,
                     value_eth, gas_used, gas_price_gwei, block_number,
                     status, ts, time.time()),
                )
            except sqlite3.IntegrityError:
                pass
    conn.close()
    return txs

# ── NFT holdings (Alchemy alchemy_getNFTs) ────────────────────────────────────
def _fetch_nfts(address: str) -> list[dict]:
    """Fetch NFT holdings for address via Alchemy and cache them."""
    try:
        data = _alchemy_get("getNFTs", {"owner": address, "withMetadata": "true"})
    except Exception as exc:
        log.warning("NFT fetch failed for %s: %s", address, exc)
        return []

    nfts = []
    conn = _get_db()
    with conn:
        for nft in data.get("ownedNfts", []):
            contract  = nft.get("contract", {}).get("address", "")
            token_id  = nft.get("id", {}).get("tokenId", "")
            meta      = nft.get("metadata", {})
            name      = meta.get("name", "")
            desc      = meta.get("description", "")
            image_url = meta.get("image", "")

            record = {
                "contract":    contract,
                "token_id":    token_id,
                "name":        name,
                "description": desc,
                "image_url":   image_url,
                "chain":       ALCHEMY_NETWORK,
            }
            nfts.append(record)

            conn.execute(
                """INSERT INTO nft_holdings
                   (address, contract, token_id, name, description,
                    image_url, chain, fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (address, contract, token_id, name, desc,
                 image_url, ALCHEMY_NETWORK, time.time()),
            )
    conn.close()
    return nfts

# ── Unsigned TX builder ────────────────────────────────────────────────────────
def _build_unsigned_tx(from_addr: str, to_addr: str, value_eth: float,
                       data: str = "0x") -> dict:
    """
    Build a display-only unsigned transaction object.
    Fetches nonce, gas price, and gas estimate from Alchemy.
    NEVER signs or broadcasts. Returns a dict for user review only.
    """
    # Fetch nonce
    nonce_result = _alchemy_rpc("eth_getTransactionCount", [from_addr, "latest"])
    nonce = int(nonce_result["result"], 16)

    # Fetch gas price
    gas_price_result = _alchemy_rpc("eth_gasPrice", [])
    gas_price_wei    = int(gas_price_result["result"], 16)

    # Build a minimal TX object for gas estimation
    tx_for_estimate = {
        "from":  from_addr,
        "to":    to_addr,
        "value": _eth_to_hex(value_eth),
    }
    if data and data != "0x":
        tx_for_estimate["data"] = data

    # Estimate gas
    try:
        gas_result = _alchemy_rpc("eth_estimateGas", [tx_for_estimate])
        gas_limit  = int(gas_result["result"], 16)
        # Add 20% buffer
        gas_limit  = int(gas_limit * 1.2)
    except Exception as exc:
        log.warning("Gas estimation failed, using default 21000: %s", exc)
        gas_limit = 21000

    # Determine chainId
    chain_ids = {
        "eth-mainnet":       1,
        "polygon-mainnet":   137,
        "eth-goerli":        5,
        "eth-sepolia":       11155111,
        "polygon-mumbai":    80001,
    }
    chain_id = chain_ids.get(ALCHEMY_NETWORK, 1)

    unsigned_tx = {
        "from":     from_addr,
        "to":       to_addr,
        "value":    _eth_to_hex(value_eth),
        "gas":      hex(gas_limit),
        "gasPrice": hex(gas_price_wei),
        "nonce":    hex(nonce),
        "data":     data if data else "0x",
        "chainId":  chain_id,
        "_note":    "UNSIGNED TRANSACTION — for display/review only. Do not sign without user consent.",
    }
    return unsigned_tx

# ── Balance refresh (used by background thread + /refresh endpoint) ────────────
def _refresh_all_wallets() -> int:
    """Refresh balances for all watched addresses. Returns count updated."""
    conn = _get_db()
    rows = conn.execute("SELECT address FROM wallet_watches").fetchall()
    conn.close()

    updated = 0
    for row in rows:
        address = row["address"]
        try:
            eth, usd = _fetch_balance(address)
            conn2 = _get_db()
            with conn2:
                conn2.execute(
                    """UPDATE wallet_watches
                       SET last_balance_eth = ?, last_balance_usd = ?,
                           last_checked = ?
                       WHERE address = ?""",
                    (eth, usd, time.time(), address),
                )
            conn2.close()
            updated += 1
            log.info("Balance refreshed for %s: %.6f ETH ($%.2f)", address, eth, usd)
        except Exception as exc:
            log.warning("Balance refresh failed for %s: %s", address, exc)

    # Also store a gas snapshot
    try:
        gas = _fetch_gas_prices()
        _store_gas_snapshot(gas)
        log.info("Gas snapshot stored: %s", gas)
    except Exception as exc:
        log.warning("Gas snapshot failed: %s", exc)

    return updated

# ── Background refresh thread ──────────────────────────────────────────────────
def _background_loop() -> None:
    log.info("Background refresh thread started (interval=%ds)", REFRESH_INTERVAL)
    while True:
        time.sleep(REFRESH_INTERVAL)
        try:
            count = _refresh_all_wallets()
            log.info("Background refresh completed: %d wallets updated", count)
        except Exception as exc:
            log.error("Background refresh error: %s", exc)

# ── Auth helper ────────────────────────────────────────────────────────────────
def _is_admin(headers) -> bool:
    """Check X-Admin-Secret header against ADMIN_SECRET."""
    if not ADMIN_SECRET:
        return True  # If not configured, allow (development mode)
    secret = headers.get("X-Admin-Secret", "")
    return hmac.compare_digest(secret, ADMIN_SECRET)

# ── JSON response helpers ──────────────────────────────────────────────────────
def _json_response(handler: BaseHTTPRequestHandler, status: int,
                   data: dict | list) -> None:
    body = json.dumps(data, default=str).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)

def _bad_request(handler, message: str) -> None:
    _json_response(handler, 400, {"error": message})

def _not_found(handler) -> None:
    _json_response(handler, 404, {"error": "Not found"})

def _unauthorized(handler) -> None:
    _json_response(handler, 401, {"error": "Unauthorized: X-Admin-Secret required"})

def _server_error(handler, exc: Exception) -> None:
    log.error("Internal error: %s", exc, exc_info=True)
    _json_response(handler, 500, {"error": str(exc)})

# ── Request Handler ────────────────────────────────────────────────────────────
class BlockchainBridgeHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the FractalMesh Blockchain/Web3 Bridge."""

    def log_message(self, fmt, *args):  # silence default access log
        log.debug("HTTP %s", fmt % args)

    # ── Routing ────────────────────────────────────────────────────────────────

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")

        if path == "/health":
            self._handle_health()
        elif path == "/gas":
            self._handle_gas()
        elif path == "/gas/history":
            self._handle_gas_history()
        elif path == "/wallets":
            self._handle_wallets_list()
        elif path == "/block/latest":
            self._handle_latest_block()
        elif re.match(r'^/wallets/0x[0-9a-fA-F]{40}/transactions$', path):
            address = path.split("/")[2]
            self._handle_wallet_transactions(address)
        elif re.match(r'^/wallets/0x[0-9a-fA-F]{40}/nfts$', path):
            address = path.split("/")[2]
            self._handle_wallet_nfts(address)
        elif re.match(r'^/wallets/0x[0-9a-fA-F]{40}$', path):
            address = path.split("/")[2]
            self._handle_wallet_detail(address)
        else:
            _not_found(self)

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")

        if path == "/wallets":
            self._handle_add_wallet()
        elif path == "/transfer/prepare":
            self._handle_transfer_prepare()
        elif path == "/refresh":
            self._handle_force_refresh()
        else:
            _not_found(self)

    def do_DELETE(self):
        path = self.path.split("?")[0].rstrip("/")

        if re.match(r'^/wallets/0x[0-9a-fA-F]{40}$', path):
            address = path.split("/")[2]
            self._handle_delete_wallet(address)
        else:
            _not_found(self)

    # ── GET /health ────────────────────────────────────────────────────────────

    def _handle_health(self):
        conn = _get_db()
        row  = conn.execute("SELECT COUNT(*) AS cnt FROM wallet_watches").fetchone()
        watched = row["cnt"] if row else 0
        conn.close()

        latest_block = None
        try:
            info = _fetch_latest_block()
            latest_block = info["block_number"]
        except Exception as exc:
            log.warning("Health check block fetch failed: %s", exc)

        _json_response(self, 200, {
            "status":          "ok",
            "service":         "fm_blockchain_bridge",
            "port":            PORT,
            "uptime_seconds":  round(time.time() - _START_TIME, 1),
            "watched_wallets": watched,
            "latest_block":    latest_block,
            "chain":           ALCHEMY_NETWORK,
            "alchemy_configured":   bool(ALCHEMY_API_KEY),
            "etherscan_configured": bool(ETHERSCAN_API_KEY),
        })

    # ── GET /gas ───────────────────────────────────────────────────────────────

    def _handle_gas(self):
        try:
            gas = _fetch_gas_prices()
            gas["chain"]      = ALCHEMY_NETWORK
            gas["fetched_at"] = time.time()
            _json_response(self, 200, gas)
        except Exception as exc:
            _server_error(self, exc)

    # ── GET /gas/history ───────────────────────────────────────────────────────

    def _handle_gas_history(self):
        try:
            conn = _get_db()
            rows = conn.execute(
                """SELECT chain, base_fee_gwei, priority_fee_gwei,
                          slow_gwei, standard_gwei, fast_gwei, fetched_at
                   FROM gas_history
                   ORDER BY fetched_at DESC
                   LIMIT ?""",
                (GAS_HISTORY_LIMIT,),
            ).fetchall()
            conn.close()
            _json_response(self, 200, [dict(r) for r in rows])
        except Exception as exc:
            _server_error(self, exc)

    # ── GET /wallets ───────────────────────────────────────────────────────────

    def _handle_wallets_list(self):
        try:
            conn = _get_db()
            rows = conn.execute(
                """SELECT address, label, chain, last_balance_eth,
                          last_balance_usd, last_checked, created_at
                   FROM wallet_watches
                   ORDER BY created_at DESC"""
            ).fetchall()
            conn.close()
            _json_response(self, 200, [dict(r) for r in rows])
        except Exception as exc:
            _server_error(self, exc)

    # ── GET /wallets/{address} ─────────────────────────────────────────────────

    def _handle_wallet_detail(self, address: str):
        try:
            address = _normalise_address(address)
        except ValueError as exc:
            _bad_request(self, str(exc))
            return

        conn = _get_db()
        wallet = conn.execute(
            "SELECT * FROM wallet_watches WHERE address = ?", (address,)
        ).fetchone()

        if not wallet:
            conn.close()
            _not_found(self)
            return

        # Fetch recent cached TXs
        txs = conn.execute(
            """SELECT tx_hash, from_addr, to_addr, value_eth, gas_used,
                      gas_price_gwei, block_number, status, timestamp
               FROM tx_cache WHERE address = ?
               ORDER BY timestamp DESC LIMIT 20""",
            (address,),
        ).fetchall()
        conn.close()

        _json_response(self, 200, {
            "wallet":       dict(wallet),
            "recent_txs":   [dict(t) for t in txs],
        })

    # ── GET /wallets/{address}/transactions ────────────────────────────────────

    def _handle_wallet_transactions(self, address: str):
        try:
            address = _normalise_address(address)
        except ValueError as exc:
            _bad_request(self, str(exc))
            return

        try:
            txs = _fetch_transactions(address)
            _json_response(self, 200, {
                "address":      address,
                "chain":        ALCHEMY_NETWORK,
                "count":        len(txs),
                "transactions": txs,
            })
        except Exception as exc:
            _server_error(self, exc)

    # ── GET /wallets/{address}/nfts ────────────────────────────────────────────

    def _handle_wallet_nfts(self, address: str):
        try:
            address = _normalise_address(address)
        except ValueError as exc:
            _bad_request(self, str(exc))
            return

        try:
            # Clear stale NFT rows for this address before refreshing
            conn = _get_db()
            with conn:
                conn.execute("DELETE FROM nft_holdings WHERE address = ?", (address,))
            conn.close()

            nfts = _fetch_nfts(address)
            _json_response(self, 200, {
                "address": address,
                "chain":   ALCHEMY_NETWORK,
                "count":   len(nfts),
                "nfts":    nfts,
            })
        except Exception as exc:
            _server_error(self, exc)

    # ── GET /block/latest ──────────────────────────────────────────────────────

    def _handle_latest_block(self):
        try:
            info = _fetch_latest_block()
            info["fetched_at"] = time.time()
            _json_response(self, 200, info)
        except Exception as exc:
            _server_error(self, exc)

    # ── POST /wallets ──────────────────────────────────────────────────────────

    def _handle_add_wallet(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body   = json.loads(self.rfile.read(length).decode()) if length else {}
        except (json.JSONDecodeError, ValueError) as exc:
            _bad_request(self, f"Invalid JSON: {exc}")
            return

        raw_address = body.get("address", "")
        label       = str(body.get("label", ""))[:128]
        chain       = str(body.get("chain", "eth"))[:32]

        # Safety: reject anything that looks like a private key
        if _looks_like_secret(raw_address):
            _bad_request(self, "Rejected: value looks like a private key or seed phrase")
            return

        try:
            address = _normalise_address(raw_address)
        except ValueError as exc:
            _bad_request(self, str(exc))
            return

        try:
            conn = _get_db()
            with conn:
                conn.execute(
                    """INSERT INTO wallet_watches
                       (address, label, chain, created_at)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(address) DO UPDATE SET
                           label = excluded.label,
                           chain = excluded.chain""",
                    (address, label, chain, time.time()),
                )
            conn.close()
            log.info("Wallet added/updated: %s (%s)", address, label)
            _json_response(self, 200, {
                "status":  "ok",
                "address": address,
                "label":   label,
                "chain":   chain,
            })
        except Exception as exc:
            _server_error(self, exc)

    # ── DELETE /wallets/{address} ─────────────────────────────────────────────

    def _handle_delete_wallet(self, address: str):
        if not _is_admin(self.headers):
            _unauthorized(self)
            return

        try:
            address = _normalise_address(address)
        except ValueError as exc:
            _bad_request(self, str(exc))
            return

        conn = _get_db()
        with conn:
            conn.execute("DELETE FROM wallet_watches WHERE address = ?", (address,))
            conn.execute("DELETE FROM tx_cache WHERE address = ?", (address,))
            conn.execute("DELETE FROM nft_holdings WHERE address = ?", (address,))
        conn.close()
        log.info("Wallet removed from watchlist: %s", address)
        _json_response(self, 200, {"status": "ok", "removed": address})

    # ── POST /transfer/prepare ─────────────────────────────────────────────────

    def _handle_transfer_prepare(self):
        """
        Prepares an unsigned transaction payload for user review.
        NEVER signs or sends the transaction. Admin-gated.
        """
        if not _is_admin(self.headers):
            _unauthorized(self)
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            body   = json.loads(self.rfile.read(length).decode()) if length else {}
        except (json.JSONDecodeError, ValueError) as exc:
            _bad_request(self, f"Invalid JSON: {exc}")
            return

        from_raw   = body.get("from_address", "")
        to_raw     = body.get("to_address", "")
        value_eth  = body.get("value_eth", 0)
        tx_data    = body.get("data", "0x")

        # Reject any value that looks like a private key
        for field_name, field_val in [
            ("from_address", from_raw),
            ("to_address",   to_raw),
            ("data",         str(tx_data)),
        ]:
            if _looks_like_secret(str(field_val)):
                log.warning("Rejected potential secret in field %s", field_name)
                _bad_request(self, f"Rejected: '{field_name}' looks like a private key or seed phrase")
                return

        try:
            from_addr = _normalise_address(from_raw)
            to_addr   = _normalise_address(to_raw)
        except ValueError as exc:
            _bad_request(self, str(exc))
            return

        try:
            value_eth = float(value_eth)
        except (TypeError, ValueError):
            _bad_request(self, "value_eth must be a number")
            return

        if value_eth < 0:
            _bad_request(self, "value_eth must be non-negative")
            return

        # Validate data field (must be hex string)
        if tx_data and tx_data != "0x":
            if not re.match(r'^0x[0-9a-fA-F]*$', str(tx_data)):
                _bad_request(self, "data must be a hex string starting with 0x")
                return

        try:
            unsigned_tx = _build_unsigned_tx(from_addr, to_addr, value_eth,
                                             str(tx_data) if tx_data else "0x")
            log.info(
                "Unsigned TX prepared: from=%s to=%s value=%.6f ETH",
                from_addr, to_addr, value_eth,
            )
            _json_response(self, 200, {
                "status":       "prepared",
                "warning":      "This is an UNSIGNED transaction. It has NOT been signed or broadcast.",
                "unsigned_tx":  unsigned_tx,
                "value_eth":    value_eth,
                "chain":        ALCHEMY_NETWORK,
                "prepared_at":  time.time(),
            })
        except Exception as exc:
            _server_error(self, exc)

    # ── POST /refresh ──────────────────────────────────────────────────────────

    def _handle_force_refresh(self):
        if not _is_admin(self.headers):
            _unauthorized(self)
            return

        try:
            updated = _refresh_all_wallets()
            _json_response(self, 200, {
                "status":  "ok",
                "updated": updated,
                "chain":   ALCHEMY_NETWORK,
            })
        except Exception as exc:
            _server_error(self, exc)


# ── Server entry point ─────────────────────────────────────────────────────────
def main() -> None:
    _init_db()

    # Start background refresh daemon
    t = threading.Thread(target=_background_loop, daemon=True)
    t.start()

    server = HTTPServer(("0.0.0.0", PORT), BlockchainBridgeHandler)
    log.info(
        "FractalMesh Blockchain Bridge running on port %d (chain=%s)",
        PORT, ALCHEMY_NETWORK,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutdown requested")
    finally:
        server.server_close()
        log.info("Server stopped")


if __name__ == "__main__":
    main()
