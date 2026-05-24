"""
FractalMesh OMEGA Titan — Moralis Web3 Data Agent
Port : 7819
DB   : ~/fmsaas/database/sovereign.db  (table: moralis_queries)
Log  : ~/fmsaas/logs/fm_moralis.log
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
MORALIS_API_KEY = os.getenv("MORALIS_API_KEY", "")
MORALIS_BASE    = "https://deep-index.moralis.io/api/v2.2"
PORT            = int(os.getenv("MORALIS_PORT", "7819"))

DB_PATH  = os.path.expanduser("~/fmsaas/database/sovereign.db")
LOG_PATH = os.path.expanduser("~/fmsaas/logs/fm_moralis.log")

# ── Logging ────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(DB_PATH),  exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [fm_moralis] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("fm_moralis")

# ── SQLite setup ───────────────────────────────────────────────────────────────
def get_db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""
        CREATE TABLE IF NOT EXISTS moralis_queries (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            query_type   TEXT,
            address      TEXT,
            chain        TEXT,
            result_count INTEGER,
            ts           INTEGER
        )
    """)
    con.commit()
    return con


def db_log_query(query_type: str, address: str, chain: str, result_count: int) -> None:
    try:
        con = get_db()
        con.execute(
            "INSERT INTO moralis_queries (query_type, address, chain, result_count, ts) VALUES (?,?,?,?,?)",
            (query_type, address, chain, result_count, int(time.time())),
        )
        con.commit()
        con.close()
    except Exception as exc:
        log.error("db_log_query failed: %s", exc)

# ── Moralis API helper ─────────────────────────────────────────────────────────

def _moralis_get(path: str, params: dict | None = None) -> dict | list:
    """
    Authenticated GET to the Moralis deep-index API.
    Raises urllib.error.HTTPError on non-2xx.
    """
    url = MORALIS_BASE.rstrip("/") + path
    if params:
        url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    req = urllib.request.Request(
        url,
        headers={
            "Accept":    "application/json",
            "X-API-Key": MORALIS_API_KEY,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            # Capture rate-limit headers for /health
            _rl_remaining = resp.headers.get("x-rate-limit-remaining", "")
            _rl_reset      = resp.headers.get("x-rate-limit-reset", "")
            data = json.loads(resp.read())
            # Attach rate-limit metadata if present
            if isinstance(data, dict):
                if _rl_remaining:
                    data["_rl_remaining"] = _rl_remaining
                if _rl_reset:
                    data["_rl_reset"] = _rl_reset
            return data
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        log.error("Moralis HTTP %s %s: %s", exc.code, path, body)
        raise
    except Exception as exc:
        log.error("Moralis request error %s: %s", path, exc)
        raise

# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_qs(query: str) -> dict:
    parsed = urllib.parse.parse_qs(query or "")
    return {k: v[0] for k, v in parsed.items()}


def _count(data) -> int:
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        return int(data.get("total", data.get("result_count", len(data.get("result", [])))))
    return 0

# ── Request handler ────────────────────────────────────────────────────────────

class MoralisHandler(BaseHTTPRequestHandler):
    """HTTP handler for the Moralis agent."""

    server_version = "FractalMesh-Moralis/1.0"

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

    def log_message(self, fmt, *args):
        log.info("HTTP %s", fmt % args)

    # ── GET dispatcher ─────────────────────────────────────────────────────────

    def do_GET(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        qs     = _parse_qs(parsed.query)
        path   = parsed.path

        routes = {
            "/health":         self._health,
            "/nfts":           self._nfts,
            "/nft":            self._nft,
            "/transfers":      self._transfers,
            "/tokens":         self._tokens,
            "/token_price":    self._token_price,
            "/native":         self._native,
            "/wallet_history": self._wallet_history,
            "/defi":           self._defi,
            "/collections":    self._collections,
        }
        handler = routes.get(path)
        if handler:
            try:
                handler(qs)
            except urllib.error.HTTPError as exc:
                self._error(f"Moralis API error: {exc.code}", exc.code if exc.code < 600 else 502)
            except Exception as exc:
                log.exception("Route %s error", path)
                self._error(str(exc))
        else:
            self._error("Not found", 404)

    # ── route implementations ──────────────────────────────────────────────────

    def _health(self, qs: dict) -> None:
        """
        GET /health
        Calls a lightweight endpoint to verify API key and capture rate-limit info.
        """
        try:
            # /web3/version is a minimal authenticated endpoint
            data = _moralis_get("/web3/version")
            self._send_json({
                "status":          "ok",
                "agent":           "fm_moralis",
                "port":            PORT,
                "moralis_version": data.get("version", "unknown"),
                "rl_remaining":    data.get("_rl_remaining", "unknown"),
                "rl_reset":        data.get("_rl_reset", "unknown"),
                "ts":              int(time.time()),
            })
        except Exception as exc:
            self._send_json({"status": "degraded", "error": str(exc), "ts": int(time.time())}, 200)

    def _nfts(self, qs: dict) -> None:
        """
        GET /nfts?address=&chain=
        Returns all NFTs for a wallet.
        """
        address = qs.get("address", "")
        chain   = qs.get("chain", "eth")
        if not address:
            self._error("address param required", 400)
            return
        data = _moralis_get(f"/{address}/nft", {"chain": chain, "format": "decimal"})
        result = data.get("result", []) if isinstance(data, dict) else data
        db_log_query("nfts", address, chain, len(result))
        self._send_json({"address": address, "chain": chain, "count": len(result), "nfts": result})

    def _nft(self, qs: dict) -> None:
        """
        GET /nft?address=&token_id=&chain=
        Returns metadata for a specific NFT.
        """
        address  = qs.get("address", "")
        token_id = qs.get("token_id", "")
        chain    = qs.get("chain", "eth")
        if not address or not token_id:
            self._error("address and token_id params required", 400)
            return
        data = _moralis_get(f"/nft/{address}/{token_id}", {"chain": chain})
        db_log_query("nft_single", address, chain, 1)
        self._send_json(data)

    def _transfers(self, qs: dict) -> None:
        """
        GET /transfers?address=&chain=
        Returns NFT transfer history for a wallet.
        """
        address = qs.get("address", "")
        chain   = qs.get("chain", "eth")
        if not address:
            self._error("address param required", 400)
            return
        data = _moralis_get(f"/{address}/nft/transfers", {"chain": chain, "format": "decimal"})
        result = data.get("result", []) if isinstance(data, dict) else data
        db_log_query("nft_transfers", address, chain, len(result))
        self._send_json({"address": address, "chain": chain, "count": len(result), "transfers": result})

    def _tokens(self, qs: dict) -> None:
        """
        GET /tokens?address=&chain=
        Returns ERC-20 token balances for a wallet.
        """
        address = qs.get("address", "")
        chain   = qs.get("chain", "eth")
        if not address:
            self._error("address param required", 400)
            return
        data = _moralis_get(f"/{address}/erc20", {"chain": chain})
        result = data if isinstance(data, list) else data.get("result", [])
        db_log_query("erc20_balances", address, chain, len(result))
        self._send_json({"address": address, "chain": chain, "count": len(result), "tokens": result})

    def _token_price(self, qs: dict) -> None:
        """
        GET /token_price?address=&chain=
        Returns current USD/ETH price for an ERC-20 contract address.
        """
        address = qs.get("address", "")
        chain   = qs.get("chain", "eth")
        if not address:
            self._error("address param required", 400)
            return
        data = _moralis_get(f"/erc20/{address}/price", {"chain": chain})
        db_log_query("token_price", address, chain, 1)
        self._send_json(data)

    def _native(self, qs: dict) -> None:
        """
        GET /native?address=&chain=
        Returns native token (ETH/MATIC/etc.) balance.
        """
        address = qs.get("address", "")
        chain   = qs.get("chain", "eth")
        if not address:
            self._error("address param required", 400)
            return
        data = _moralis_get(f"/{address}/balance", {"chain": chain})
        db_log_query("native_balance", address, chain, 1)
        balance_wei = int(data.get("balance", "0"))
        self._send_json({
            "address":     address,
            "chain":       chain,
            "balance_wei": balance_wei,
            "balance_eth": balance_wei / 1e18,
        })

    def _wallet_history(self, qs: dict) -> None:
        """
        GET /wallet_history?address=&chain=
        Returns wallet net worth and transaction history summary.
        """
        address = qs.get("address", "")
        chain   = qs.get("chain", "eth")
        if not address:
            self._error("address param required", 400)
            return
        # Wallet net worth endpoint
        try:
            net_worth = _moralis_get(f"/wallets/{address}/net-worth", {"chains": chain})
        except Exception:
            net_worth = {}

        # Wallet history (transactions)
        try:
            history = _moralis_get(f"/{address}", {"chain": chain})
        except Exception:
            history = {}

        result = history.get("result", []) if isinstance(history, dict) else []
        db_log_query("wallet_history", address, chain, len(result))
        self._send_json({
            "address":     address,
            "chain":       chain,
            "net_worth":   net_worth,
            "tx_count":    len(result),
            "history":     result,
        })

    def _defi(self, qs: dict) -> None:
        """
        GET /defi?address=&chain=
        Returns DeFi positions (Uniswap, Aave, Compound, etc.) for the wallet.
        """
        address = qs.get("address", "")
        chain   = qs.get("chain", "eth")
        if not address:
            self._error("address param required", 400)
            return
        data   = _moralis_get(f"/wallets/{address}/defi/positions", {"chain": chain})
        result = data if isinstance(data, list) else data.get("result", [])
        db_log_query("defi_positions", address, chain, len(result))
        self._send_json({
            "address":   address,
            "chain":     chain,
            "count":     len(result),
            "positions": result,
        })

    def _collections(self, qs: dict) -> None:
        """
        GET /collections?address=
        Returns unique NFT collections owned by a wallet.
        """
        address = qs.get("address", "")
        if not address:
            self._error("address param required", 400)
            return
        # Aggregate NFTs and deduplicate by token_address
        data   = _moralis_get(f"/{address}/nft", {"chain": "eth", "format": "decimal"})
        nfts   = data.get("result", []) if isinstance(data, dict) else []
        seen   = {}
        for nft in nfts:
            ca = nft.get("token_address", "")
            if ca and ca not in seen:
                seen[ca] = {
                    "token_address": ca,
                    "name":          nft.get("name", ""),
                    "symbol":        nft.get("symbol", ""),
                    "contract_type": nft.get("contract_type", ""),
                }
        collections = list(seen.values())
        db_log_query("collections", address, "eth", len(collections))
        self._send_json({
            "address":     address,
            "count":       len(collections),
            "collections": collections,
        })


# ── Signal handling ────────────────────────────────────────────────────────────
_running = True

def _shutdown(signum, frame):
    global _running
    log.info("Signal %s received — shutting down fm_moralis", signum)
    _running = False

signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT,  _shutdown)

# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    get_db().close()

    server = HTTPServer(("0.0.0.0", PORT), MoralisHandler)
    log.info("fm_moralis listening on port %d", PORT)

    global _running
    while _running:
        server.handle_request()

    log.info("fm_moralis stopped")


if __name__ == "__main__":
    main()
