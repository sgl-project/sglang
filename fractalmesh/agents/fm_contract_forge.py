"""
FractalMesh Contract Forge Agent
Generates sovereign MSA contracts as markdown documents in dist/
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import time
import signal
import sqlite3
from datetime import datetime

ROOT     = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB       = os.path.join(ROOT, "database", "sovereign.db")
DIST     = os.path.join(ROOT, "dist")
INTERVAL = int(os.getenv("CONTRACT_FORGE_INTERVAL", "3600"))
DRY_RUN  = os.getenv("ENABLE_CONTRACT_FORGE", "false").lower() != "true"

CONTRACTS = [
    {"ref": "FRACTAL-OBR-7729", "client": "O'Brien Logistics",
     "scope": "NCC 2025 + SOC 2 compliance automation, sovereign node deployment",
     "fee": 4500.0, "retainer": 997.0, "output": "obrien_msa_final.md"},
    {"ref": "FRACTAL-GEN-0001", "client": "General Client Template",
     "scope": "Infrastructure audit, sovereign automation setup",
     "fee": 750.0, "retainer": 0.0, "output": "general_msa_template.md"},
]

_running = True


def _db_init():
    os.makedirs(DIST, exist_ok=True)
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS contract_log (
        id INTEGER PRIMARY KEY, ref TEXT, client TEXT,
        fee_aud REAL, path TEXT, status TEXT,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()


def _render(c: dict) -> str:
    now      = datetime.utcnow().strftime("%Y-%m-%d")
    retainer = f"\n- **Monthly Retainer:** ${c['retainer']:,.2f} AUD/month" if c["retainer"] > 0 else ""
    return f"""# Master Services Agreement
**Reference:** {c['ref']}  **Date:** {now}
**Operator:** Samuel James Hiotis | ABN 56 628 117 363 | Albury NSW
**Client:** {c['client']}

## Scope of Services
{c['scope']}

## Commercial Terms
- **Project Fee:** ${c['fee']:,.2f} AUD (ex GST){retainer}
- **Payment Terms:** 50% upfront, 50% on delivery

## Deliverables
- Sovereign infrastructure deployment or audit
- Compliance documentation (NCC 2025 / SOC 2 as scoped)
- HMAC-signed audit trail in sovereign.db
- PM2-managed agent swarm configuration

## Governing Law
New South Wales, Australia.

---
*FractalMesh Sovereign Contract Forge v1*
"""


def _forge(c: dict):
    content = _render(c)
    path    = os.path.join(DIST, c["output"])
    if not DRY_RUN:
        with open(path, "w") as f:
            f.write(content)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("INSERT INTO contract_log (ref,client,fee_aud,path,status) VALUES (?,?,?,?,?)",
                 (c["ref"], c["client"], c["fee"], path, "dry_run" if DRY_RUN else "forged"))
    conn.commit(); conn.close()
    return path


def run_cycle():
    print(f"[fm-contract-forge] {datetime.utcnow().isoformat()} | dry={DRY_RUN}")
    for c in CONTRACTS:
        path = _forge(c)
        print(f"   → {c['ref']:<25} {c['client']:<25} ${c['fee']:,.2f} | {'dry' if DRY_RUN else path}")


def _sigterm(s, f):
    global _running
    _running = False


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    _db_init()
    print(f"[fm-contract-forge] Active | interval={INTERVAL}s | dry={DRY_RUN}")
    while _running:
        try:
            run_cycle()
        except Exception as e:
            print(f"[fm-contract-forge] ERR {e}")
        for _ in range(INTERVAL):
            if not _running: break
            time.sleep(1)
    print("[fm-contract-forge] Stopped.")
