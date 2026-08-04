#!/usr/bin/env python3
"""
rf_bridge.py — Syncs RF/GBWiGLE telemetry sales into FractalMesh sovereign.db
Runs every 5 minutes, reads ~/ai-mesh/sales/rf_sales.json → writes to SQLite
Samuel James Hiotis | ABN 56628117363 | Albury NSW
"""
import os, json, sqlite3, time, logging
from datetime import datetime
from pathlib import Path

for vault in [Path(os.path.expanduser("~/.secrets/fractal.env")),
              Path(os.path.expanduser("~/fmsaas/.env"))]:
    if vault.exists():
        for line in vault.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [RF-BRIDGE] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("rf_bridge")

ROOT         = os.environ.get("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB           = os.path.join(ROOT, "db", "sovereign.db")
RF_SALES     = Path(os.path.expanduser("~/ai-mesh/sales/rf_sales.json"))

def ensure_dirs():
    RF_SALES.parent.mkdir(parents=True, exist_ok=True)
    if not RF_SALES.exists():
        RF_SALES.write_text("[]")

def ensure_schema(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stripe_session_id TEXT UNIQUE,
            product TEXT,
            customer_email TEXT,
            amount_aud REAL,
            status TEXT DEFAULT 'completed',
            payment_confirmed INTEGER DEFAULT 1,
            created_at TEXT
        )
    """)
    conn.commit()

def sync() -> int:
    try:
        sales = json.loads(RF_SALES.read_text())
    except Exception as e:
        log.warning("Could not read rf_sales.json: %s", e)
        return 0
    if not sales:
        return 0
    if not Path(DB).exists():
        log.warning("sovereign.db not found at %s", DB)
        return 0
    synced = 0
    try:
        conn = sqlite3.connect(DB, timeout=10)
        ensure_schema(conn)
        for i, sale in enumerate(sales):
            sid = sale.get("session_id", f"RF_{int(time.time())}_{i}")
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO orders
                    (stripe_session_id, product, customer_email,
                     amount_aud, status, payment_confirmed, created_at)
                    VALUES (?, ?, ?, ?, 'completed', 1, ?)
                """, (
                    sid,
                    sale.get("product", "RF-Telemetry-PRO"),
                    sale.get("email", "rf@local"),
                    float(sale.get("amount_aud", 9.00)),
                    datetime.now().isoformat(),
                ))
                synced += 1
            except Exception as e:
                log.warning("Row insert: %s", e)
        conn.commit()
        conn.close()
        RF_SALES.write_text("[]")   # clear after sync
        if synced:
            log.info("Synced %d RF sales to sovereign.db", synced)
    except Exception as e:
        log.error("DB error: %s", e)
    return synced

def main():
    ensure_dirs()
    log.info("rf-bridge started | sovereign=%s | rf_sales=%s", DB, RF_SALES)
    while True:
        try:
            sync()
        except Exception as e:
            log.error("Cycle error: %s", e)
        time.sleep(300)   # 5 minutes

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
