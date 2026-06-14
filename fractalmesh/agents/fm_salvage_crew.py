"""
FractalMesh Salvage Crew Agent
Recovers corrupted or missing sovereign.db tables; re-seeds from backup snapshots
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import time
import signal
import shutil
import sqlite3
from datetime import datetime

ROOT     = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB       = os.path.join(ROOT, "database", "sovereign.db")
BACKUP   = os.path.join(ROOT, "backups")
INTERVAL = int(os.getenv("SALVAGE_INTERVAL", "1800"))

REQUIRED_TABLES = [
    "pulse_log", "leads", "revenue", "royalty_pools",
    "tokenomics_state", "smart_contracts", "chemical_chain",
    "wigle_telemetry_ledger",
]

_running = True


def _db_init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    os.makedirs(BACKUP, exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS salvage_log (
        id INTEGER PRIMARY KEY, table_name TEXT, action TEXT,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()


def _existing_tables() -> set:
    try:
        conn  = sqlite3.connect(DB, timeout=10)
        rows  = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        conn.close()
        return {r[0] for r in rows}
    except Exception:
        return set()


def _seed_table(conn, tname: str):
    seeds = {
        "pulse_log":              "CREATE TABLE IF NOT EXISTS pulse_log (id INTEGER PRIMARY KEY, source TEXT, event TEXT, priority REAL, ts DATETIME DEFAULT CURRENT_TIMESTAMP)",
        "leads":                  "CREATE TABLE IF NOT EXISTS leads (id INTEGER PRIMARY KEY, industry TEXT, region TEXT, query_hash TEXT UNIQUE, raw_query TEXT, result_count INTEGER, source TEXT, ts DATETIME DEFAULT CURRENT_TIMESTAMP)",
        "revenue":                "CREATE TABLE IF NOT EXISTS revenue (id INTEGER PRIMARY KEY, source TEXT, amount_aud REAL, description TEXT, ts DATETIME DEFAULT CURRENT_TIMESTAMP)",
        "royalty_pools":          "CREATE TABLE IF NOT EXISTS royalty_pools (pool_id TEXT PRIMARY KEY, label TEXT, pct REAL, aud_balance REAL, updated DATETIME DEFAULT CURRENT_TIMESTAMP)",
        "tokenomics_state":       "CREATE TABLE IF NOT EXISTS tokenomics_state (id INTEGER PRIMARY KEY, epoch INTEGER, pool_id TEXT, pool_label TEXT, pct REAL, phi_weight REAL, aud_balance REAL, ts DATETIME DEFAULT CURRENT_TIMESTAMP)",
        "smart_contracts":        "CREATE TABLE IF NOT EXISTS smart_contracts (id INTEGER PRIMARY KEY, ref TEXT UNIQUE, client TEXT, status TEXT, fee_aud REAL, ts DATETIME DEFAULT CURRENT_TIMESTAMP)",
        "chemical_chain":         "CREATE TABLE IF NOT EXISTS chemical_chain (id INTEGER PRIMARY KEY, element TEXT, compound TEXT, hash TEXT, ts DATETIME DEFAULT CURRENT_TIMESTAMP)",
        "wigle_telemetry_ledger": "CREATE TABLE IF NOT EXISTS wigle_telemetry_ledger (id INTEGER PRIMARY KEY, ssid TEXT, bssid TEXT, lat REAL, lon REAL, signal_dbm INTEGER, ts DATETIME DEFAULT CURRENT_TIMESTAMP)",
    }
    if tname in seeds:
        conn.execute(seeds[tname])


def _log_action(tname, action):
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("INSERT INTO salvage_log (table_name,action) VALUES (?,?)", (tname, action))
    conn.commit(); conn.close()


def run_cycle():
    existing = _existing_tables()
    missing  = [t for t in REQUIRED_TABLES if t not in existing]
    ts       = datetime.utcnow().isoformat()
    print(f"[fm-salvage-crew] {ts} | tables={len(existing)} | missing={len(missing)}")
    if missing:
        conn = sqlite3.connect(DB, timeout=10)
        for tname in missing:
            _seed_table(conn, tname)
            print(f"   → SALVAGED {tname}")
            _log_action(tname, "seeded")
        conn.commit(); conn.close()
    else:
        print(f"   → all {len(REQUIRED_TABLES)} required tables present")


def _sigterm(s, f):
    global _running
    _running = False


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    _db_init()
    print(f"[fm-salvage-crew] Active | interval={INTERVAL}s | watching {len(REQUIRED_TABLES)} tables")
    while _running:
        try:
            run_cycle()
        except Exception as e:
            print(f"[fm-salvage-crew] ERR {e}")
        for _ in range(INTERVAL):
            if not _running: break
            time.sleep(1)
    print("[fm-salvage-crew] Stopped.")
