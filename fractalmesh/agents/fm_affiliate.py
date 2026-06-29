"""
FractalMesh Affiliate Agent
Tracks affiliate link health, logs conversions to sovereign.db
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import time
import signal
import sqlite3
import urllib.request
from datetime import datetime

ROOT     = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB       = os.path.join(ROOT, "database", "sovereign.db")
INTERVAL = int(os.getenv("AFFILIATE_POLL_INTERVAL", "600"))
DRY_RUN  = os.getenv("ENABLE_AFFILIATE_LIVE", "false").lower() != "true"
PHI      = 1.6180339887

LINKS = [
    {"id": "proton",       "url": os.getenv("AFF_PROTON",       ""), "label": "Proton Mail/VPN"},
    {"id": "mullvad",      "url": os.getenv("AFF_MULLVAD",      ""), "label": "Mullvad VPN"},
    {"id": "digitalocean", "url": os.getenv("AFF_DIGITALOCEAN", ""), "label": "DigitalOcean"},
    {"id": "vultr",        "url": os.getenv("AFF_VULTR",        ""), "label": "Vultr Cloud"},
    {"id": "kucoin",       "url": os.getenv("AFF_KUCOIN",       ""), "label": "KuCoin Exchange"},
    {"id": "pionex",       "url": os.getenv("AFF_PIONEX",       ""), "label": "Pionex Bots"},
    {"id": "helium",       "url": os.getenv("AFF_HELIUM",       ""), "label": "Helium DePIN"},
    {"id": "xyo",          "url": os.getenv("AFF_XYO",          ""), "label": "XYO Network"},
    {"id": "blofin",       "url": os.getenv("BLOFIN_REFERRAL_URL", ""), "label": "BloFin Referral"},
]

_running = True


def _db_init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS affiliate_log (
        id INTEGER PRIMARY KEY, link_id TEXT, label TEXT,
        action TEXT, phi_score REAL, ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()


def _log(link_id, label, action):
    phi_score = round(len(link_id) * PHI, 4)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("INSERT INTO affiliate_log (link_id, label, action, phi_score) VALUES (?,?,?,?)",
                 (link_id, label, action, phi_score))
    conn.commit(); conn.close()


def _ping(url: str) -> str:
    if not url:
        return "no_url"
    try:
        req = urllib.request.Request(url, method="HEAD")
        req.add_header("User-Agent", "FractalMesh-AffiliateBot/1.0")
        with urllib.request.urlopen(req, timeout=8) as r:
            return f"ok:{r.status}"
    except Exception as e:
        return f"err:{e}"


def run_cycle():
    active = sum(1 for l in LINKS if l["url"])
    print(f"[fm-affiliate] {datetime.utcnow().isoformat()} | {active}/{len(LINKS)} configured | dry={DRY_RUN}")
    for l in LINKS:
        status = "dry_run" if DRY_RUN else _ping(l["url"])
        print(f"   → {l['id']:<14} {l['label']:<22} {status}")
        _log(l["id"], l["label"], status)


def _sigterm(s, f):
    global _running
    _running = False


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    _db_init()
    print(f"[fm-affiliate] Active | interval={INTERVAL}s | dry={DRY_RUN}")
    while _running:
        try:
            run_cycle()
        except Exception as e:
            print(f"[fm-affiliate] ERR {e}")
        for _ in range(INTERVAL):
            if not _running: break
            time.sleep(1)
    print("[fm-affiliate] Stopped.")
