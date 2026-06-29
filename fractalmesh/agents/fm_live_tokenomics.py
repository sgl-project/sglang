"""
FractalMesh Live Tokenomics Agent
Tracks royalty pools, φ-harmonic balances, and tokenomics state in sovereign.db
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import math
import time
import signal
import sqlite3
from datetime import datetime

ROOT     = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB       = os.path.join(ROOT, "database", "sovereign.db")
INTERVAL = int(os.getenv("TOKENOMICS_INTERVAL", "300"))
PHI      = 1.6180339887

# Royalty pool allocation (% of revenue)
POOLS = [
    {"id": "operator",    "label": "Operator (Samuel Hiotis)", "pct": 60.0},
    {"id": "reinvest",    "label": "Infrastructure Reinvest",  "pct": 20.0},
    {"id": "compliance",  "label": "Compliance Reserve",       "pct": 10.0},
    {"id": "depin",       "label": "DePIN Node Rewards",       "pct": 5.0},
    {"id": "bounty",      "label": "Bounty Pool",              "pct": 5.0},
]

_running = True
_epoch   = 0


def _db_init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS tokenomics_state (
        id INTEGER PRIMARY KEY, epoch INTEGER, pool_id TEXT,
        pool_label TEXT, pct REAL, phi_weight REAL, aud_balance REAL,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS royalty_pools (
        pool_id TEXT PRIMARY KEY, label TEXT, pct REAL, aud_balance REAL,
        updated DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    for p in POOLS:
        conn.execute("""INSERT OR IGNORE INTO royalty_pools (pool_id,label,pct,aud_balance)
            VALUES (?,?,?,?)""", (p["id"], p["label"], p["pct"], 0.0))
    conn.commit(); conn.close()


def _current_balances() -> dict:
    conn = sqlite3.connect(DB, timeout=10)
    rows = conn.execute("SELECT pool_id, aud_balance FROM royalty_pools").fetchall()
    conn.close()
    return {r[0]: r[1] for r in rows}


def _phi_weight(pct: float, idx: int) -> float:
    return round(pct * (PHI ** idx) / 100, 6)


def _log(epoch, pool, phi_w, bal):
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("""INSERT INTO tokenomics_state (epoch,pool_id,pool_label,pct,phi_weight,aud_balance)
        VALUES (?,?,?,?,?,?)""", (epoch, pool["id"], pool["label"], pool["pct"], phi_w, bal))
    conn.commit(); conn.close()


def run_cycle():
    global _epoch
    _epoch += 1
    balances = _current_balances()
    ts       = datetime.utcnow().isoformat()
    total    = sum(balances.values())
    print(f"[fm-live-tokenomics] Epoch {_epoch} | {ts} | total=${total:.2f} AUD")
    for i, p in enumerate(POOLS):
        bal    = balances.get(p["id"], 0.0)
        phi_w  = _phi_weight(p["pct"], i)
        phi_bal = round(bal * PHI, 4)
        print(f"   → {p['id']:<12} {p['pct']:>5.1f}% | ${bal:>8.2f} AUD | φ={phi_w:.6f} | φ-bal={phi_bal}")
        _log(_epoch, p, phi_w, bal)


def _sigterm(s, f):
    global _running
    _running = False


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    _db_init()
    print(f"[fm-live-tokenomics] Active | interval={INTERVAL}s | pools={len(POOLS)}")
    while _running:
        try:
            run_cycle()
        except Exception as e:
            print(f"[fm-live-tokenomics] ERR {e}")
        for _ in range(INTERVAL):
            if not _running: break
            time.sleep(1)
    print("[fm-live-tokenomics] Stopped.")
