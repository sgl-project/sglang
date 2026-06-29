"""
FractalMesh Auto Advert Agent
Rotates sovereign ad copy to configured channels on a cooldown schedule
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import time
import signal
import sqlite3
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

ROOT       = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB         = os.path.join(ROOT, "database", "sovereign.db")
INTERVAL   = int(os.getenv("AUTO_ADVERT_INTERVAL", "3600"))
COOLDOWN   = int(os.getenv("AUTO_ADVERT_COOLDOWN", "14400"))
DRY_RUN    = os.getenv("ENABLE_AUTO_ADVERT", "false").lower() != "true"
GMAIL_USER = os.getenv("GMAIL_USER", "")
GMAIL_PASS = os.getenv("GMAIL_APP_PASS", "")

ADS = [
    {"id": "blueprint",    "subject": "Free AI Node Blueprint — Sovereign Infrastructure",
     "body": "Grab the free FractalMesh blueprint:\nhttps://fractalmesh.net/blueprint.html\n26-node swarm | HMAC-signed | AFSL-ready"},
    {"id": "consulting",   "subject": "Sovereign Infra Audit — Limited Slots Open",
     "body": "Audit + harden Linux/Android edge stacks:\nhttps://fractalmesh.net/consulting.html\nNCC 2025 + SOC 2 from $750 AUD"},
    {"id": "automation",   "subject": "Automate Ops — FractalMesh Automation Pack",
     "body": "PM2 agent swarms for revenue & compliance:\nhttps://fractalmesh.net/products.html\n$49–$79 one-time"},
    {"id": "intelligence", "subject": "April 2026 Resilience Briefing",
     "body": "Latest sovereign briefing:\nhttps://fractalmesh.net/intelligence.html\nEdge hardening | DePIN | AFSL automation"},
]

_running = True


def _db_init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS advert_log (
        id INTEGER PRIMARY KEY, ad_id TEXT, channel TEXT, status TEXT,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()


def _last_sent(ad_id: str) -> float:
    conn = sqlite3.connect(DB, timeout=10)
    row  = conn.execute(
        "SELECT ts FROM advert_log WHERE ad_id=? AND status='sent' ORDER BY ts DESC LIMIT 1",
        (ad_id,)).fetchone()
    conn.close()
    if not row: return 0.0
    try:
        return datetime.fromisoformat(row[0]).timestamp()
    except Exception:
        return 0.0


def _log(ad_id, channel, status):
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("INSERT INTO advert_log (ad_id, channel, status) VALUES (?,?,?)",
                 (ad_id, channel, status))
    conn.commit(); conn.close()


def _send(ad: dict) -> str:
    if not GMAIL_USER or not GMAIL_PASS:
        return "no_creds"
    try:
        msg = MIMEText(ad["body"])
        msg["Subject"] = ad["subject"]
        msg["From"] = msg["To"] = GMAIL_USER
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as s:
            s.login(GMAIL_USER, GMAIL_PASS)
            s.sendmail(GMAIL_USER, [GMAIL_USER], msg.as_string())
        return "sent"
    except Exception as e:
        return f"err:{e}"


def run_cycle():
    now = time.time()
    print(f"[fm-auto-advert] {datetime.utcnow().isoformat()} dry={DRY_RUN}")
    for ad in ADS:
        age = now - _last_sent(ad["id"])
        if age < COOLDOWN:
            print(f"   → {ad['id']:<14} cooldown {int(COOLDOWN-age)}s"); continue
        status = "dry_run" if DRY_RUN else _send(ad)
        print(f"   → {ad['id']:<14} {status}")
        _log(ad["id"], "gmail", status)


def _sigterm(s, f):
    global _running
    _running = False


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    _db_init()
    print(f"[fm-auto-advert] Active | interval={INTERVAL}s | dry={DRY_RUN}")
    while _running:
        try:
            run_cycle()
        except Exception as e:
            print(f"[fm-auto-advert] ERR {e}")
        for _ in range(INTERVAL):
            if not _running: break
            time.sleep(1)
    print("[fm-auto-advert] Stopped.")
