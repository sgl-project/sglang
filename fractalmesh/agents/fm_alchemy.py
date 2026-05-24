"""
FractalMesh OMEGA Titan — Alchemy Web3 Agent
Port : 7818
DB   : ~/fmsaas/database/sovereign.db  (table: alchemy_events)
Log  : ~/fmsaas/logs/fm_alchemy.log
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
ALCHEMY_API_KEY       = os.getenv("ALCHEMY_API_KEY", "")
ALCHEMY_ETH_MAINNET   = os.getenv("ALCHEMY_ETH_MAINNET", f"https://eth-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}")
ALCHEMY_POLYGON       = os.getenv("ALCHEMY_POLYGON_MAINNET", f"https://polygon-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}")
ETH_ADDRESS           = os.getenv("ETH_ADDRESS", "")
PORT                  = int(os.getenv("ALCHEMY_PORT", "7818"))

DB_PATH  = os.path.expanduser("~/fmsaas/database/sovereign.db")
LOG_PATH = os.path.expanduser("~/fmsaas/logs/fm_alchemy.log")

# ── Logging ────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(DB_PATH),  exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [fm_alchemy] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("fm_alchemy")

# ── SQLite setup ───────────────────────────────────────────────────────────────
def get_db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""
        CREATE TABLE IF NOT EXISTS alchemy_events (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            address    TEXT,
            event_type TEXT,
            value_eth  REAL,
            tx_hash    TEXT,
            ts         INTEGER
        )
    """)
    con.commit()
    return con


def db_insert_event(address: str, event_type: str, value_eth: float, tx_hash: str) -> None:
    try:
        con = get_db()
        con.execute(
            "INSERT INTO alchemy_events (address, event_type, value_eth, tx_hash, ts) VALUES (?,?,?,?,?)",
            (address, event_type, value_eth, tx_hash, int(time.time())),
        )
        con.commit()
        con.close()
    except Exception as exc:
        log.error("db_insert_event failed: %s", exc)

# ── Alchemy helpers ────────────────────────────────────────────────────────────
CHAIN_URLS = {
    "eth":     ALCHEMY_ETH_MAINNET,
    "polygon": ALCHEMY_POLYGON,
}

def _rpc_url(chain: str) -> str:
    return CHAIN_URLS.get(chain.lower(), ALCHEMY_ETH_MAINNET)


def _jsonrpc(url: str, method: str, params: list) -> dict:
    """Execute a JSON-RPC 2.0 POST request."""
    payload = json.dumps({"jsonrpc": "2.0", "method": method, "params": params, "id": 1}).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        log.error("JSON-RPC HTTP error %s: %s", exc.code, body)
        raise
    except Exception as exc:
        log.error("JSON-RPC error: %s", exc)
        raise


def _alchemy_get(path: str, params: dict | None = None, chain: str = "eth") -> dict:
    """HTTP GET to Alchemy REST endpoints (NFT / Transfers APIs)."""
    base = _rpc_url(chain).rstrip("/")
    # NFT v3 and other REST endpoints share the key embedded in the base URL
    full_url = base + path
    if params:
        full_url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(full_url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def _wei_to_eth(wei_hex: str) -> float:
    try:
        return int(wei_hex, 16) / 1e18
    except Exception:
        return 0.0


def _hex_to_gwei(hex_val: str) -> float:
    try:
        return int(hex_val, 16) / 1e9
    except Exception:
        return 0.0


def _coingecko_eth_price() -> float:
    """Fetch current ETH/USD from CoinGecko free endpoint."""
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd"
        with urllib.request.urlopen(url, timeout=8) as resp:
            data = json.loads(resp.read())
            return float(data["ethereum"]["usd"])
    except Exception:
        return 0.0  # fallback — caller must handle 0

# ── Request handler ────────────────────────────────────────────────────────────

def _parse_qs(query: str) -> dict:
    parsed = urllib.parse.parse_qs(query or "")
    return {k: v[0] for k, v in parsed.items()}


class AlchemyHandler(BaseHTTPRequestHandler):
    """HTTP handler for the Alchemy agent."""

    server_version = "FractalMesh-Alchemy/1.0"

    # ── utilities ──────────────────────────────────────────────────────────────

    def _send_json(self, data: dict, status: int = 200) -> None:
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

    def log_message(self, fmt, *args):  # route to our logger
        log.info("HTTP %s", fmt % args)

    # ── GET dispatcher ─────────────────────────────────────────────────────────

    def do_GET(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        qs     = _parse_qs(parsed.query)
        path   = parsed.path

        routes = {
            "/health":    self._health,
            "/balance":   self._balance,
            "/nfts":      self._nfts,
            "/tokens":    self._tokens,
            "/tx":        self._tx,
            "/history":   self._history,
            "/block":     self._block,
            "/gas":       self._gas,
            "/portfolio": self._portfolio,
        }
        handler = routes.get(path)
        if handler:
            try:
                handler(qs)
            except Exception as exc:
                log.exception("Route %s error", path)
                self._error(str(exc))
        else:
            self._error("Not found", 404)

    def do_POST(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path   = parsed.path
        if path == "/transfer":
            try:
                self._transfer(self._read_body())
            except Exception as exc:
                log.exception("POST /transfer error")
                self._error(str(exc))
        else:
            self._error("Not found", 404)

    # ── route implementations ──────────────────────────────────────────────────

    def _health(self, qs: dict) -> None:
        balance_wei = "0x0"
        eth_bal = 0.0
        try:
            resp = _jsonrpc(ALCHEMY_ETH_MAINNET, "eth_getBalance", [ETH_ADDRESS, "latest"])
            balance_wei = resp.get("result", "0x0")
            eth_bal = _wei_to_eth(balance_wei)
        except Exception as exc:
            log.warning("health balance check failed: %s", exc)
        self._send_json({
            "status":      "ok",
            "agent":       "fm_alchemy",
            "port":        PORT,
            "wallet":      ETH_ADDRESS,
            "eth_balance": eth_bal,
            "ts":          int(time.time()),
        })

    def _balance(self, qs: dict) -> None:
        address = qs.get("address", ETH_ADDRESS)
        chain   = qs.get("chain", "eth")
        if not address:
            self._error("address param required", 400)
            return
        url  = _rpc_url(chain)
        resp = _jsonrpc(url, "eth_getBalance", [address, "latest"])
        result = resp.get("result", "0x0")
        eth = _wei_to_eth(result)
        db_insert_event(address, "balance_check", eth, "")
        self._send_json({
            "address": address,
            "chain":   chain,
            "wei":     int(result, 16),
            "eth":     eth,
        })

    def _nfts(self, qs: dict) -> None:
        address = qs.get("address", ETH_ADDRESS)
        chain   = qs.get("chain", "eth")
        if not address:
            self._error("address param required", 400)
            return
        # Alchemy NFT API v3 — REST endpoint
        nft_url = (
            f"https://eth-mainnet.g.alchemy.com/nft/v3/{ALCHEMY_API_KEY}"
            f"/getNFTsForOwner?owner={address}&withMetadata=false&pageSize=100"
        )
        if chain.lower() == "polygon":
            nft_url = (
                f"https://polygon-mainnet.g.alchemy.com/nft/v3/{ALCHEMY_API_KEY}"
                f"/getNFTsForOwner?owner={address}&withMetadata=false&pageSize=100"
            )
        req = urllib.request.Request(nft_url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        self._send_json({
            "address":    address,
            "chain":      chain,
            "total_count": data.get("totalCount", 0),
            "nfts":       data.get("ownedNfts", []),
        })

    def _tokens(self, qs: dict) -> None:
        address = qs.get("address", ETH_ADDRESS)
        chain   = qs.get("chain", "eth")
        if not address:
            self._error("address param required", 400)
            return
        url  = _rpc_url(chain)
        resp = _jsonrpc(url, "alchemy_getTokenBalances", [address, "erc20"])
        result = resp.get("result", {})
        balances = [
            {"contract": t["contractAddress"], "raw": t["tokenBalance"]}
            for t in result.get("tokenBalances", [])
            if t.get("tokenBalance") not in (None, "0x0000000000000000000000000000000000000000000000000000000000000000")
        ]
        self._send_json({"address": address, "chain": chain, "token_balances": balances})

    def _tx(self, qs: dict) -> None:
        tx_hash = qs.get("hash", "")
        if not tx_hash:
            self._error("hash param required", 400)
            return
        resp = _jsonrpc(ALCHEMY_ETH_MAINNET, "eth_getTransactionByHash", [tx_hash])
        self._send_json({"tx": resp.get("result")})

    def _history(self, qs: dict) -> None:
        address = qs.get("address", ETH_ADDRESS)
        if not address:
            self._error("address param required", 400)
            return
        params = {
            "fromBlock":  "0x0",
            "toBlock":    "latest",
            "toAddress":  address,
            "category":   ["external", "internal", "erc20"],
            "maxCount":   "0x14",  # 20
            "withMetadata": True,
            "excludeZeroValue": True,
            "order": "desc",
        }
        resp = _jsonrpc(ALCHEMY_ETH_MAINNET, "alchemy_getAssetTransfers", [params])
        transfers = resp.get("result", {}).get("transfers", [])
        self._send_json({"address": address, "transfers": transfers})

    def _block(self, qs: dict) -> None:
        resp = _jsonrpc(ALCHEMY_ETH_MAINNET, "eth_blockNumber", [])
        result = resp.get("result", "0x0")
        self._send_json({"block_hex": result, "block_number": int(result, 16)})

    def _gas(self, qs: dict) -> None:
        resp = _jsonrpc(ALCHEMY_ETH_MAINNET, "eth_gasPrice", [])
        result = resp.get("result", "0x0")
        gwei = _hex_to_gwei(result)
        self._send_json({"gas_price_hex": result, "gas_price_gwei": gwei})

    def _portfolio(self, qs: dict) -> None:
        address = qs.get("address", ETH_ADDRESS)
        if not address:
            self._error("address param required", 400)
            return

        # ETH balance
        eth_resp = _jsonrpc(ALCHEMY_ETH_MAINNET, "eth_getBalance", [address, "latest"])
        eth_bal  = _wei_to_eth(eth_resp.get("result", "0x0"))

        # ERC-20 token balances
        tok_resp = _jsonrpc(ALCHEMY_ETH_MAINNET, "alchemy_getTokenBalances", [address, "erc20"])
        token_balances = tok_resp.get("result", {}).get("tokenBalances", [])

        # ETH price
        eth_usd = _coingecko_eth_price()
        eth_usd_value = eth_bal * eth_usd if eth_usd else None

        db_insert_event(address, "portfolio_check", eth_bal, "")
        self._send_json({
            "address":        address,
            "eth":            eth_bal,
            "eth_usd_price":  eth_usd or "unavailable",
            "eth_usd_value":  eth_usd_value,
            "erc20_count":    len(token_balances),
            "erc20_balances": [
                {"contract": t["contractAddress"], "raw": t["tokenBalance"]}
                for t in token_balances
                if t.get("tokenBalance") not in (
                    None,
                    "0x0000000000000000000000000000000000000000000000000000000000000000",
                )
            ],
        })

    def _transfer(self, body: dict) -> None:
        """
        Prepare (NOT send) a raw ETH transfer transaction.
        Returns hex-encoded unsigned tx data for external signing.
        """
        from_addr = body.get("from", ETH_ADDRESS)
        to_addr   = body.get("to", "")
        amount_eth = float(body.get("amount_eth", 0))

        if not to_addr:
            self._error("'to' address required", 400)
            return
        if amount_eth <= 0:
            self._error("amount_eth must be positive", 400)
            return

        value_wei = int(amount_eth * 1e18)

        # Fetch nonce
        nonce_resp = _jsonrpc(ALCHEMY_ETH_MAINNET, "eth_getTransactionCount", [from_addr, "latest"])
        nonce = int(nonce_resp.get("result", "0x0"), 16)

        # Fetch gas price
        gas_resp = _jsonrpc(ALCHEMY_ETH_MAINNET, "eth_gasPrice", [])
        gas_price_hex = gas_resp.get("result", "0x77359400")  # 2 gwei fallback

        # Standard ETH transfer gas limit
        gas_limit = 21000

        # Build unsigned tx fields (EIP-155 legacy format, no signing)
        raw_tx_fields = {
            "nonce":    hex(nonce),
            "gasPrice": gas_price_hex,
            "gasLimit": hex(gas_limit),
            "to":       to_addr,
            "value":    hex(value_wei),
            "data":     "0x",
            "chainId":  1,  # Ethereum mainnet
        }

        log.info("Transfer prepared (NOT broadcast): from=%s to=%s value_eth=%s", from_addr, to_addr, amount_eth)
        self._send_json({
            "status":      "prepared_not_sent",
            "warning":     "Sign this transaction offline with your private key before broadcasting.",
            "from":        from_addr,
            "to":          to_addr,
            "amount_eth":  amount_eth,
            "value_wei":   value_wei,
            "unsigned_tx": raw_tx_fields,
        })


# ── Signal handling ────────────────────────────────────────────────────────────
_running = True

def _shutdown(signum, frame):
    global _running
    log.info("Signal %s received — shutting down fm_alchemy", signum)
    _running = False

signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT,  _shutdown)

# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    # Ensure DB + tables exist at startup
    get_db().close()

    server = HTTPServer(("0.0.0.0", PORT), AlchemyHandler)
    log.info("fm_alchemy listening on port %d", PORT)

    global _running
    while _running:
        server.handle_request()

    log.info("fm_alchemy stopped")


if __name__ == "__main__":
    main()
