#!/usr/bin/env python3
"""
FractalMesh Trading Orchestrator
Multi-exchange autonomous trading manager
Samuel James Hiotis | ABN 56628117363
"""
import os, asyncio, logging, json
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [FMTrading] %(levelname)s %(message)s",
)
log = logging.getLogger("FMTrading")

ROOT  = os.environ.get("FRACTALMESH_HOME", str(Path.home() / "fmsaas"))
VAULT = os.path.join(ROOT, ".env")

def load_env(key, default=""):
    for f in [VAULT, str(Path.home() / ".env")]:
        try:
            for line in Path(f).read_text().splitlines():
                s = line.strip()
                if s.startswith(key + "=") and not s.startswith("#"):
                    val = s.split("=", 1)[1].strip().strip('"').strip("'")
                    if val and not val.startswith("YOUR_"):
                        return val
        except Exception:
            pass
    return os.environ.get(key, default)

EXCHANGES_CFG = {
    "kucoin":    {"key": "KUCOIN_API_KEY",    "secret": "KUCOIN_API_SECRET",    "pass": "KUCOIN_API_PASSPHRASE"},
    "pionex":    {"key": "PIONEX_API_KEY",    "secret": "PIONEX_API_SECRET"},
    "cryptocom": {"key": "CRYPTOCOM_API_KEY", "secret": "CRYPTOCOM_API_SECRET"},
    "coinbase":  {"key": "COINBASE_API_KEY",  "secret": "COINBASE_SECRET_KEY"},
}

class TradingOrchestrator:
    def __init__(self):
        self.exchanges   = {}
        self.active      = False
        self.cycle_count = 0

    async def initialize(self):
        log.info("Initializing trading orchestrator...")
        ready = 0
        for name, cfg in EXCHANGES_CFG.items():
            k = load_env(cfg["key"])
            if k:
                log.info(f"  [{name}] credentials present — ready")
                self.exchanges[name] = {"key": k, "status": "ready"}
                ready += 1
            else:
                log.warning(f"  [{name}] credentials missing — fill in .env to activate")
        log.info(f"Initialized {ready}/{len(EXCHANGES_CFG)} exchanges")
        self.active = ready > 0 or True  # stay alive even with no creds

    async def trading_cycle(self):
        self.cycle_count += 1
        log.info(f"Trading cycle #{self.cycle_count} — {datetime.now().strftime('%H:%M:%S')}")

        if not self.exchanges:
            log.info("  No live exchanges configured — monitoring mode")
            return

        for name in self.exchanges:
            log.info(f"  [{name}] Evaluating positions...")
            # In production: fetch balances, evaluate RL signals, place orders

    async def run(self):
        await self.initialize()
        log.info("Trading orchestrator running — cycle every 60s")
        while self.active:
            try:
                await self.trading_cycle()
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Trading loop error: {e}")
                await asyncio.sleep(10)
        log.info("Trading orchestrator stopped")

if __name__ == "__main__":
    asyncio.run(TradingOrchestrator().run())
