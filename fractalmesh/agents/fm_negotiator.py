"""
FractalMesh Negotiator Agent v2.0.0
4A Signature: payment handling, proposal generation, negotiation specialist.
Includes satellite intelligence report tiers (methane, AIS, crop yield).
Logs pipeline stages to sovereign.db; dry-run safe.
Samuel James Hiotis | ABN 56 628 117 363 | Sole Trader
"""
import os
import json
import time
import signal
import sqlite3
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime

ROOT       = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB         = os.path.join(ROOT, "database", "sovereign.db")
INTERVAL   = int(os.getenv("NEGOTIATOR_INTERVAL", "3600"))
DRY_RUN    = os.getenv("ENABLE_NEGOTIATOR", "false").lower() != "true"
GMAIL_USER = os.getenv("GMAIL_USER", "")
GMAIL_PASS = os.getenv("GMAIL_APP_PASS", "")

OPERATOR_NAME  = "Samuel James Hiotis"
OPERATOR_ABN   = "56 628 117 363"
OPERATOR_EMAIL = GMAIL_USER or os.getenv("OPERATOR_EMAIL", "")
OPERATOR_PHONE = os.getenv("OPERATOR_PHONE", "0439 008 640")

# ── 4A Signature tiers ─────────────────────────────────────────────────────────
TIERS = {
    # Infrastructure consulting
    "audit":           {"label": "Infrastructure Audit",              "aud":  750.0, "days": 5},
    "compliance":      {"label": "Compliance Automation",             "aud": 1500.0, "days": 10},
    "deploy":          {"label": "Sovereign Node Deploy",             "aud": 2500.0, "days": 14},
    "bundle":          {"label": "Full Sovereign Bundle",             "aud": 4500.0, "days": 21},
    # Satellite intelligence reports
    "ch4_single":      {"label": "Methane Super-Emitter Report",      "aud": 2000.0, "days": 3},
    "ch4_cluster":     {"label": "Methane Cluster Report (5 sites)",  "aud": 8000.0, "days": 5},
    "ch4_retainer":    {"label": "Monthly Methane Monitoring",        "aud": 3500.0, "days": 30},
    "ais_dark":        {"label": "Dark Fleet Intelligence (30 days)", "aud": 4500.0, "days": 7},
    "crop_yield":      {"label": "Early Yield Signal Report",         "aud": 1500.0, "days": 5},
    "cross_verify":    {"label": "WiFi + Satellite Ground-Truth",     "aud": 6000.0, "days": 7},
}

NEGOTIATION_ANCHORS = {
    "open":       1.20,   # initial ask = 120% of target
    "concession": 0.95,   # first concession to 95%
    "floor":      0.85,   # absolute floor at 85%
    "urgency_disc": 0.10, # 10% urgency discount if they need within 48h
}

_running = True


def _db_init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS negotiation_log (
        id INTEGER PRIMARY KEY, prospect TEXT, tier TEXT, stage TEXT,
        ask_aud REAL, target_aud REAL, status TEXT, notes TEXT,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS proposals (
        id INTEGER PRIMARY KEY, prospect TEXT, tier TEXT, content TEXT,
        ask_aud REAL, sent INTEGER DEFAULT 0,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()


def _log_negotiation(prospect, tier, stage, ask, target, status, notes=""):
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("""INSERT INTO negotiation_log (prospect,tier,stage,ask_aud,target_aud,status,notes)
        VALUES (?,?,?,?,?,?,?)""",
        (prospect, tier, stage, ask, target, status, notes[:500]))
    conn.commit(); conn.close()


def _proposal(prospect: str, tier_key: str, urgency: bool = False) -> dict:
    t       = TIERS.get(tier_key, TIERS["audit"])
    anchor  = NEGOTIATION_ANCHORS
    target  = t["aud"]
    ask     = round(target * anchor["open"], 2)
    disc    = round(target * anchor["urgency_disc"], 2) if urgency else 0
    final   = round(ask - disc, 2)
    now     = datetime.utcnow().strftime("%d %B %Y")
    payment_link = os.getenv("STRIPE_PAYMENT_LINK", "https://fractalmesh.net/products.html")

    body = f"""Dear {prospect},

Thank you for your interest in FractalMesh sovereign infrastructure services.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PROPOSAL — {t['label'].upper()}
  Reference: FM-{datetime.utcnow().strftime('%Y%m')}-{abs(hash(prospect)) % 9999:04d}
  Date: {now}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCOPE
  • {t['label']}
  • Delivery: {t['days']} business days
  • HMAC-signed audit trail in sovereign.db
  • NCC 2025 + SOC 2 control documentation (where scoped)
  • PM2-managed agent swarm configuration

INVESTMENT
  • Project Fee: ${final:,.2f} AUD (ex GST)
  • Payment Terms: 50% upfront, 50% on delivery
  • Secure payment: {payment_link}{"" if not disc else f"\n  • Urgency discount applied: -${disc:.2f} AUD"}

NEXT STEP
  Reply to confirm scope, and I'll send the MSA for e-signature.
  Or book a 30-min call: https://fractalmesh.net/contact.html

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{OPERATOR_NAME}
ABN {OPERATOR_ABN} | {OPERATOR_EMAIL} | {OPERATOR_PHONE}
FractalMesh Omega Titan — Sovereign Infrastructure
"""
    return {"prospect": prospect, "tier": tier_key, "ask": final, "target": target,
            "body": body, "subject": f"FractalMesh Proposal — {t['label']} for {prospect}"}


def _send_proposal(prop: dict) -> str:
    if not GMAIL_USER or not GMAIL_PASS:
        return "no_creds"
    if DRY_RUN:
        return "dry_run"
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = prop["subject"]
        msg["From"]    = GMAIL_USER
        msg["To"]      = GMAIL_USER   # self-send; replace with prospect email in production
        msg.attach(MIMEText(prop["body"], "plain"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as s:
            s.login(GMAIL_USER, GMAIL_PASS)
            s.sendmail(GMAIL_USER, [GMAIL_USER], msg.as_string())
        conn = sqlite3.connect(DB, timeout=10)
        conn.execute("INSERT INTO proposals (prospect,tier,content,ask_aud,sent) VALUES (?,?,?,?,1)",
                     (prop["prospect"], prop["tier"], prop["body"][:3000], prop["ask"]))
        conn.commit(); conn.close()
        return "sent"
    except Exception as e:
        return f"err:{e}"


def _process_pipeline():
    """Check for pending leads and generate proposals for high-priority ones."""
    if not os.path.exists(DB):
        return []
    conn = sqlite3.connect(DB, timeout=10)
    # High-result leads from OSINT spider → qualify as warm prospects
    rows = conn.execute("""
        SELECT raw_query, result_count, industry FROM leads
        WHERE result_count > 50 AND source='osint_spider'
        ORDER BY result_count DESC LIMIT 5""").fetchall()
    conn.close()
    prospects = []
    for q, count, industry in rows:
        # Extract company signal from query
        m = re.search(r'"([^"]+)"', q) if __import__("re").search(r'"([^"]+)"', q) else None
        label = m.group(1) if m else industry
        prospects.append({"name": label, "industry": industry, "signals": count})
    return prospects


def run_cycle():
    ts  = datetime.utcnow().isoformat()
    print(f"[fm-negotiator] {ts} | dry={DRY_RUN}")

    # Demo proposal generation (one per cycle)
    demo_props = [
        ("O'Brien Logistics",            "bundle",       False),
        ("Visy Albury",                  "compliance",   True),
        ("Local Construction",           "audit",        False),
        ("Carbon Trading Desk",          "ch4_single",   False),
        ("Meridian Marine Insurance",    "ais_dark",     True),
        ("Macquarie Commodities Desk",   "cross_verify", False),
    ]
    for prospect, tier, urgency in demo_props:
        prop   = _proposal(prospect, tier, urgency)
        status = _send_proposal(prop)
        print(f"   → {prospect:<25} {tier:<12} ${prop['ask']:>8,.2f} AUD | {status}")
        _log_negotiation(prospect, tier, "proposal_sent" if status == "sent" else "staged",
                         prop["ask"], prop["target"], status)

    # Check pipeline
    warm_leads = _process_pipeline()
    if warm_leads:
        print(f"   Pipeline: {len(warm_leads)} warm leads from OSINT")
        for lead in warm_leads:
            print(f"      ⟶ {lead['name']:<30} [{lead['industry']}] signals={lead['signals']}")


def _sigterm(s, f):
    global _running
    _running = False


if __name__ == "__main__":
    import re
    signal.signal(signal.SIGTERM, _sigterm)
    _db_init()
    print(f"[fm-negotiator] 4A Negotiation Specialist active | interval={INTERVAL}s | dry={DRY_RUN}")
    while _running:
        try:
            run_cycle()
        except Exception as e:
            print(f"[fm-negotiator] ERR {e}")
        for _ in range(INTERVAL):
            if not _running: break
            time.sleep(1)
    print("[fm-negotiator] Stopped.")
