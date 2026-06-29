#!/usr/bin/env python3
"""
FractalMesh Trading Orchestrator v402
Virtual P&L tracking + signal-driven trade simulation + WiGLE geo-intel
Samuel James Hiotis | ABN 56628117363
v402: P&L ledger in sovereign.db, virtual portfolio, geo-tagged trades via WiGLE.
"""
import os, asyncio, logging, json, sqlite3, time, urllib.request
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [FMTrading] %(levelname)s %(message)s")
log = logging.getLogger("FMTrading")

ROOT  = os.environ.get("FRACTALMESH_HOME", str(Path.home() / "fmsaas"))
VAULT = os.path.join(ROOT, ".env")
DB    = os.path.join(ROOT, "db", "sovereign.db")

EXCHANGES_CFG = {
    "kucoin":    {"key":"KUCOIN_API_KEY",    "secret":"KUCOIN_API_SECRET",    "pass":"KUCOIN_API_PASSPHRASE"},
    "pionex":    {"key":"PIONEX_API_KEY",    "secret":"PIONEX_API_SECRET"},
    "cryptocom": {"key":"CRYPTOCOM_API_KEY", "secret":"CRYPTOCOM_API_SECRET"},
    "coinbase":  {"key":"COINBASE_API_KEY",  "secret":"COINBASE_SECRET_KEY"},
}

# Virtual portfolio seed
PORTFOLIO = {
    "BTC/USDT":{"qty":0.05,"avg_buy":65000.0},
    "ETH/USDT":{"qty":0.80,"avg_buy": 3200.0},
    "SOL/USDT":{"qty":5.00,"avg_buy":  160.0},
}

def load_env(key, default=""):
    for f in [VAULT, str(Path.home() / ".env"), str(Path.home() / ".secrets/fractal.env")]:
        try:
            for line in Path(f).read_text().splitlines():
                s = line.strip()
                if s.startswith(key + "=") and not s.startswith("#"):
                    val = s.split("=",1)[1].strip().strip('"').strip("'")
                    if val and not val.startswith("YOUR_"):
                        return val
        except Exception:
            pass
    return os.environ.get(key, default)

def get_db():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn

def init_schema():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS trade_pnl (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT, action TEXT, qty REAL, price REAL,
            pnl_usd REAL DEFAULT 0, confidence REAL,
            fractal_score INTEGER, exchange TEXT DEFAULT 'virtual',
            geo_lat REAL, geo_lon REAL, geo_network TEXT,
            executed_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_json TEXT,
            total_value_usd REAL,
            total_pnl_usd REAL,
            snapped_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit(); conn.close()

def _fetch_signals():
    """Pull live signals from fm-geosignal."""
    try:
        with urllib.request.urlopen("http://localhost:5057/api/signals", timeout=4) as r:
            return json.loads(r.read()).get("signals", [])
    except Exception:
        return []

def _wigle_lookup():
    """WiGLE API — look up local network density for geo context."""
    api_key = load_env("WIGLE_API_KEY")
    if not api_key:
        return None
    try:
        # WiGLE stats endpoint — public summary
        req = urllib.request.Request(
            "https://api.wigle.net/api/v2/stats/user",
            headers={"Authorization": f"Basic {api_key}",
                     "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=6) as r:
            data = json.loads(r.read())
        return {"networks":data.get("statistics",{}).get("discoveredWiFiGPS",0),
                "source":"WiGLE","api":"v2"}
    except Exception:
        return None

def _evaluate_and_trade(signals):
    """Evaluate signals against portfolio; log virtual trades + P&L."""
    conn     = get_db()
    geo_data = _wigle_lookup()
    traded   = 0
    for sig in signals:
        pair = sig.get("pair","")
        if pair not in PORTFOLIO:
            continue
        pos    = PORTFOLIO[pair]
        price  = sig.get("price", 0)
        conf   = sig.get("confidence", 0)
        fscore = sig.get("fractal_score", 0)
        action = sig.get("signal","HOLD")
        if action == "HOLD" or price == 0:
            continue
        qty     = pos["qty"] * 0.25   # trade 25% of position
        avg_buy = pos["avg_buy"]
        pnl     = round((price - avg_buy) * qty * (1 if action=="BUY" else -1), 2)
        conn.execute("""
            INSERT INTO trade_pnl
                (pair,action,qty,price,pnl_usd,confidence,fractal_score,
                 geo_lat,geo_lon,geo_network,exchange,executed_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (pair, action, round(qty,6), price, pnl, conf, fscore,
              -36.08, 146.92,   # Albury NSW lat/lon
              geo_data["networks"] if geo_data else 0,
              "virtual", datetime.now().isoformat()))
        # Update avg buy if BUY
        if action == "BUY":
            pos["avg_buy"] = round((pos["avg_buy"] + price) / 2, 2)
        log.info("  [TRADE] %s %s %.6f @ %.4f | P&L $%.2f | conf=%.0f%% fscore=%d",
                 action, pair, qty, price, pnl, conf*100, fscore)
        traded += 1
    conn.commit(); conn.close()
    return traded

def _snapshot_portfolio(signals):
    """Save portfolio snapshot to DB."""
    conn    = get_db()
    sig_map = {s["pair"]:s["price"] for s in signals if s.get("price")}
    total   = sum(pos["qty"] * sig_map.get(pair, pos["avg_buy"])
                  for pair, pos in PORTFOLIO.items())
    cost    = sum(pos["qty"] * pos["avg_buy"] for pos in PORTFOLIO.values())
    pnl     = round(total - cost, 2)
    snap    = {pair: {"qty":pos["qty"],"avg_buy":pos["avg_buy"],
                       "current_price":sig_map.get(pair,pos["avg_buy"])}
               for pair, pos in PORTFOLIO.items()}
    conn.execute("INSERT INTO portfolio_snapshots(portfolio_json,total_value_usd,total_pnl_usd) VALUES(?,?,?)",
                 (json.dumps(snap), round(total,2), pnl))
    conn.commit(); conn.close()
    log.info("  [SNAPSHOT] Portfolio $%.2f | P&L $%.2f", total, pnl)

class TradingOrchestrator:
    def __init__(self):
        self.exchanges   = {}
        self.cycle_count = 0

    async def initialize(self):
        init_schema()
        log.info("Initializing trading orchestrator v402...")
        for name, cfg in EXCHANGES_CFG.items():
            k = load_env(cfg["key"])
            if k:
                log.info("  [%s] credentials present — ready", name)
                self.exchanges[name] = {"key":k,"status":"ready"}
            else:
                log.warning("  [%s] credentials missing — fill .env to activate", name)
        log.info("Exchanges: %d/%d configured | Virtual portfolio active",
                 len(self.exchanges), len(EXCHANGES_CFG))

    async def trading_cycle(self):
        self.cycle_count += 1
        log.info("Cycle #%d — %s", self.cycle_count, datetime.now().strftime("%H:%M:%S"))
        signals = await asyncio.get_event_loop().run_in_executor(None, _fetch_signals)
        if not signals:
            log.info("  No signals — monitoring mode")
            return
        traded = await asyncio.get_event_loop().run_in_executor(None, _evaluate_and_trade, signals)
        if self.cycle_count % 10 == 0:  # snapshot every 10 cycles
            await asyncio.get_event_loop().run_in_executor(None, _snapshot_portfolio, signals)
        log.info("  Cycle complete: %d trades executed", traded)

    async def run(self):
        await self.initialize()
        log.info("Trading orchestrator running — 60s cycle")
        while True:
            try:
                await self.trading_cycle()
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("Loop error: %s", e)
                await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(TradingOrchestrator().run())
