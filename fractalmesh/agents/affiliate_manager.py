"""
FractalMesh Affiliate Manager v2.1.0
Tracks affiliate program performance, click attribution,
conversion revenue. Seeds 10 partner programs on first run.
Samuel James Hiotis | ABN 56 628 117 363 | Sole Trader
"""
import os
import json
import time
import signal
import sqlite3
from datetime import datetime, timezone

ROOT     = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB       = os.path.join(ROOT, "database", "sovereign.db")
INTERVAL = int(os.getenv("AFFILIATE_INTERVAL", "3600"))

PHI      = 1.6180339887
_running = True

SEED_PROGRAMS = [
    # (program, network, ref_code, ref_url, commission_type, commission_value,
    #  cookie_days, payout_threshold, status, notes)
    ("Manus.im",      "Direct", "XDCMWO3VETC7FV",
     "https://manus.im/invitation/XDCMWO3VETC7FV",
     "per_signup", "Unlocked credits", 30, "N/A", "active", "AI agent platform"),
    ("Vultr",         "Direct", "",
     os.getenv("AFF_VULTR_URL", "https://vultr.com"),
     "per_signup", "A$100 credit + A$35", 30, "A$50", "active", "Cloud hosting"),
    ("DigitalOcean",  "Impact", "",
     os.getenv("AFF_DO_URL", "https://m.do.co/c/"),
     "per_signup", "A$25-A$200", 30, "A$50", "active", "Dev hosting"),
    ("Hostinger",     "Direct", "",
     "https://hostinger.com",
     "rev_share", "60% first payment", 30, "A$100", "active", "Web hosting"),
    ("Cursor AI",     "Direct", "",
     "https://cursor.sh",
     "per_trial", "A$20-40 per conv", 30, "A$50", "active", "AI IDE"),
    ("Notion",        "Direct", "",
     "https://notion.so",
     "rev_share", "variable", 90, "A$10", "active", "Workspace"),
    ("Perplexity AI", "Direct", "",
     "https://perplexity.ai",
     "per_pro", "A$10 per Pro", 30, "A$25", "active", "AI search"),
    ("Fiverr",        "Direct", "",
     "https://go.fiverr.com",
     "hybrid", "A$15-A$150 CPA", 30, "A$100", "active", "Freelance marketplace"),
    ("Gumroad",       "Direct", "",
     "https://gumroad.com/affiliates",
     "rev_share", "10% of sale", 30, "A$10", "active", "Digital products"),
    ("Namecheap",     "Impact", "",
     "https://namecheap.com",
     "rev_share", "20%-35%", 30, "A$50", "active", "Domains + hosting"),
    # DePIN / crypto from existing vault
    ("KuCoin",        "Direct", "",
     os.getenv("AFF_KUCOIN_URL", "https://www.kucoin.com"),
     "rev_share", "20% fee share", 30, "A$25", "active", "Crypto exchange"),
    ("Pionex",        "Direct", "",
     os.getenv("AFF_PIONEX_URL", "https://www.pionex.com"),
     "rev_share", "20% lifetime", 30, "A$25", "active", "Crypto trading bots"),
    ("BloFin",        "Direct", "",
     os.getenv("BLOFIN_REFERRAL_URL", "https://blofin.com"),
     "rev_share", "20% rebate", 30, "A$25", "active", "Crypto futures"),
    ("Proton",        "Direct", "",
     os.getenv("AFF_PROTON_URL", "https://pr.tn/ref/"),
     "rev_share", "30-100% first year", 30, "A$25", "active", "Privacy suite"),
    ("Mullvad VPN",   "Direct", "",
     os.getenv("AFF_MULLVAD_URL", "https://mullvad.net"),
     "flat", "Flat referral", 30, "A$25", "active", "Privacy VPN"),
    ("Helium",        "Direct", "",
     os.getenv("AFF_HELIUM_URL", "https://helium.com"),
     "per_node", "Node deploy ref", 90, "N/A", "active", "DePIN IoT"),
    ("XYO Network",   "Direct", "",
     os.getenv("AFF_XYO_URL", "https://xyo.network"),
     "per_sentinel", "Sentinel referral", 90, "N/A", "active", "DePIN geo-proof"),
]


def _db_init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS affiliates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        program TEXT NOT NULL UNIQUE,
        network TEXT, ref_code TEXT, ref_url TEXT,
        commission_type TEXT, commission_value TEXT,
        cookie_days INTEGER DEFAULT 30,
        payout_threshold TEXT, status TEXT DEFAULT 'active',
        notes TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS affiliate_clicks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        program TEXT, source TEXT, medium TEXT,
        content_id TEXT, ip_hash TEXT,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE INDEX IF NOT EXISTS idx_aff_clicks_ts
        ON affiliate_clicks(ts DESC)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS affiliate_conversions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        program TEXT, amount REAL, currency TEXT DEFAULT 'AUD',
        ref TEXT, status TEXT DEFAULT 'pending',
        phi_score REAL,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS content_pieces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        platform TEXT, title TEXT, body TEXT,
        affiliates_embedded TEXT, url TEXT,
        status TEXT DEFAULT 'draft',
        views INTEGER DEFAULT 0, clicks INTEGER DEFAULT 0,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS drip_sequences (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lead_email TEXT, sequence_name TEXT,
        step INTEGER DEFAULT 0, last_sent DATETIME,
        status TEXT DEFAULT 'active')""")
    conn.commit(); conn.close()


def _seed_programs():
    conn = sqlite3.connect(DB, timeout=10)
    added = 0
    for prog in SEED_PROGRAMS:
        exists = conn.execute(
            "SELECT id FROM affiliates WHERE program=?", (prog[0],)).fetchone()
        if not exists:
            conn.execute("""INSERT INTO affiliates
                (program,network,ref_code,ref_url,commission_type,commission_value,
                 cookie_days,payout_threshold,status,notes)
                VALUES (?,?,?,?,?,?,?,?,?,?)""", prog)
            added += 1
    conn.commit(); conn.close()
    if added:
        print(f"[affiliate-mgr] Seeded {added} programs")


def _report():
    conn = sqlite3.connect(DB, timeout=10)

    def safe(sql, default=0):
        try:
            r = conn.execute(sql).fetchone()
            return r[0] if r and r[0] is not None else default
        except Exception:
            return default

    total    = float(safe(
        "SELECT COALESCE(SUM(amount),0) FROM affiliate_conversions "
        "WHERE status!='rejected'"))
    pending  = float(safe(
        "SELECT COALESCE(SUM(amount),0) FROM affiliate_conversions "
        "WHERE status='pending'"))
    clicks   = safe(
        "SELECT COUNT(*) FROM affiliate_clicks "
        "WHERE ts>datetime('now','-1 day')")
    programs = safe("SELECT COUNT(*) FROM affiliates WHERE status='active'")
    content  = safe("SELECT COUNT(*) FROM content_pieces")
    drip_act = safe("SELECT COUNT(*) FROM drip_sequences WHERE status='active'")

    phi = round(total * PHI / 100, 4) if total else 0.0
    conn.close()

    print(f"[affiliate-mgr] Report: A${total:.2f} earned | "
          f"A${pending:.2f} pending | {clicks} clicks/24h | "
          f"{programs} programs | {content} content pieces | "
          f"{drip_act} active drip sequences | φ={phi}")


def run_cycle():
    ts = datetime.now(tz=timezone.utc).isoformat()
    print(f"[affiliate-mgr] {ts}")
    _seed_programs()
    _report()


def _sigterm(s, f):
    global _running
    _running = False


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT,  _sigterm)
    _db_init()
    print(f"[affiliate-mgr] Active | interval={INTERVAL}s | "
          f"programs={len(SEED_PROGRAMS)}")
    while _running:
        try:
            run_cycle()
        except Exception as e:
            print(f"[affiliate-mgr] ERR {e}")
        for _ in range(INTERVAL):
            if not _running: break
            time.sleep(1)
    print("[affiliate-mgr] Stopped.")
