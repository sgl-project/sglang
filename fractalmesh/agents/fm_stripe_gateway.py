"""
FractalMesh Stripe Gateway Agent
Polls Stripe balance, recent charges, and logs revenue to sovereign.db
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import json
import time
import signal
import sqlite3
import urllib.request
import urllib.parse
import base64
from datetime import datetime

ROOT     = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB       = os.path.join(ROOT, "database", "sovereign.db")
INTERVAL = int(os.getenv("STRIPE_GATEWAY_INTERVAL", "300"))
DRY_RUN  = os.getenv("ENABLE_STRIPE_GATEWAY", "false").lower() != "true"

STRIPE_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_API = "https://api.stripe.com/v1"

_running = True


def _db_init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS revenue (
        id INTEGER PRIMARY KEY, source TEXT, charge_id TEXT UNIQUE,
        amount_aud REAL, currency TEXT, description TEXT, status TEXT,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()


def _stripe_get(path: str) -> dict:
    if not STRIPE_KEY:
        return {}
    try:
        creds = base64.b64encode(f"{STRIPE_KEY}:".encode()).decode()
        req   = urllib.request.Request(
            f"{STRIPE_API}{path}",
            headers={"Authorization": f"Basic {creds}"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"   [stripe-gateway] api err: {e}")
        return {}


def _log_charge(charge: dict):
    amount_aud = charge.get("amount", 0) / 100.0
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("""INSERT OR IGNORE INTO revenue
        (source,charge_id,amount_aud,currency,description,status)
        VALUES (?,?,?,?,?,?)""",
        ("stripe", charge.get("id", ""), amount_aud,
         charge.get("currency", "aud").upper(),
         (charge.get("description") or "")[:200],
         charge.get("status", "")))
    conn.commit(); conn.close()


def run_cycle():
    ts = datetime.utcnow().isoformat()
    if DRY_RUN or not STRIPE_KEY:
        print(f"[fm-stripe-gateway] {ts} | dry={DRY_RUN} | key={'set' if STRIPE_KEY else 'missing'}")
        return

    balance = _stripe_get("/balance")
    avail   = balance.get("available", [{}])
    avail_aud = next((e["amount"] / 100 for e in avail if e.get("currency") == "aud"), 0)
    print(f"[fm-stripe-gateway] {ts} | balance AUD=${avail_aud:.2f}")

    charges = _stripe_get("/charges?limit=5")
    for charge in charges.get("data", []):
        _log_charge(charge)
        amt = charge.get("amount", 0) / 100
        print(f"   → {charge.get('id','?')[:16]} ${amt:.2f} {charge.get('status','')}")


def _sigterm(s, f):
    global _running
    _running = False


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    _db_init()
    print(f"[fm-stripe-gateway] Active | interval={INTERVAL}s | dry={DRY_RUN} | key={'set' if STRIPE_KEY else 'MISSING'}")
    while _running:
        try:
            run_cycle()
        except Exception as e:
            print(f"[fm-stripe-gateway] ERR {e}")
        for _ in range(INTERVAL):
            if not _running: break
            time.sleep(1)
    print("[fm-stripe-gateway] Stopped.")
