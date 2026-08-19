"""
FractalMesh Campaign Manager v2.0.0
Multi-channel outreach campaign lifecycle:
prospect → warm → proposal → negotiation → closed/lost.
A/B testing, ROI tracking, φ-prioritised follow-up queue.
Samuel James Hiotis | ABN 56 628 117 363 | Sole Trader
"""
import os
import re
import json
import time
import signal
import sqlite3
import smtplib
import hashlib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta

ROOT       = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB         = os.path.join(ROOT, "database", "sovereign.db")
INTERVAL   = int(os.getenv("CAMPAIGN_INTERVAL", "3600"))
DRY_RUN    = os.getenv("ENABLE_CAMPAIGNS", "false").lower() != "true"

GMAIL_USER   = os.getenv("GMAIL_USER",    "")
GMAIL_PASS   = os.getenv("GMAIL_APP_PASS", "")
OPERATOR     = "Samuel James Hiotis"
ABN          = "56 628 117 363"
PHONE        = os.getenv("OPERATOR_PHONE", "0439 008 640")
SITE         = "https://fractalmesh.net"
PAYMENT_LINK = os.getenv("STRIPE_PAYMENT_LINK", f"{SITE}/products.html")

PHI       = 1.6180339887
_running  = True

# ── Campaign state machine ────────────────────────────────────────────────────
STATES     = ["prospect", "email_sent", "warm", "proposal_sent",
              "negotiating", "closed_won", "closed_lost"]

NEXT_STATE = {
    "prospect":      "email_sent",
    "email_sent":    "warm",
    "warm":          "proposal_sent",
    "proposal_sent": "negotiating",
    "negotiating":   "closed_won",
}

# Follow-up delays (days) per stage
FOLLOWUP_DAYS = {
    "prospect":      0,
    "email_sent":    3,
    "warm":          7,
    "proposal_sent": 5,
    "negotiating":   2,
}

# A/B template sets
TEMPLATES = {
    "A_intro": {
        "subject": "Quick question about infrastructure automation — {company}",
        "body": """Hi {name},

I noticed {company} might benefit from sovereign infrastructure automation.

I'm Samuel James Hiotis (ABN {abn}), a Albury-based automation specialist.
I've helped businesses in the Murray-Darling corridor reduce compliance overhead
by 40-70% through self-hosted AI agent meshes.

Would a 15-min call be worth your time this week?

{name_op}
{phone} | {site}
""",
    },
    "B_intro": {
        "subject": "Automation audit offer for {company} — no cost",
        "body": """Hello,

Offering a complimentary 30-minute infrastructure automation audit for {company}.

Outcomes: identify 3-5 manual processes ripe for AI-agent replacement,
estimate ROI, and map a deployment path to NCC 2025 compliance.

No obligation. Secure payment if you proceed: {payment_link}

{name_op} | ABN {abn}
{phone} | {site}
""",
    },
    "followup_warm": {
        "subject": "Following up — automation for {company}",
        "body": """Hi {name},

Just checking in on my earlier message about automation for {company}.

Have you had a chance to think about it?

Happy to share a brief outline of what a deployment looks like.
Reply or book a call: {site}/contact.html

{name_op}
""",
    },
    "proposal_reminder": {
        "subject": "Proposal reminder — {company} | FractalMesh",
        "body": """Hi {name},

I sent through a proposal for {company} a few days ago.

Quick summary:
  • Scope: Infrastructure automation + compliance documentation
  • Investment: from $750 AUD (ex GST)
  • Timeline: 5-21 business days

To proceed or ask questions: reply to this email or call {phone}.
Secure payment: {payment_link}

{name_op} | ABN {abn}
""",
    },
}


# ── DB ────────────────────────────────────────────────────────────────────────

def _db_init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS campaigns (
        id INTEGER PRIMARY KEY, name TEXT UNIQUE, template_set TEXT,
        status TEXT DEFAULT 'active', created DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS campaign_prospects (
        id INTEGER PRIMARY KEY, campaign_id INTEGER, name TEXT, company TEXT,
        email TEXT, phone TEXT, industry TEXT, region TEXT,
        state TEXT DEFAULT 'prospect', template_variant TEXT DEFAULT 'A',
        last_action DATETIME, next_action DATETIME,
        phi_score REAL DEFAULT 1.0, notes TEXT,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS campaign_events (
        id INTEGER PRIMARY KEY, prospect_id INTEGER, event TEXT,
        template TEXT, channel TEXT, status TEXT, notes TEXT,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS campaign_roi (
        id INTEGER PRIMARY KEY, campaign_id INTEGER, prospects INTEGER,
        emails_sent INTEGER, warm_count INTEGER, proposals INTEGER,
        closed_won INTEGER, revenue_aud REAL, roi_pct REAL,
        phi_score REAL, ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()


def _seed_demo_prospects():
    """Seed initial demo prospects if none exist."""
    conn = sqlite3.connect(DB, timeout=10)
    count = conn.execute("SELECT COUNT(*) FROM campaign_prospects").fetchone()[0]
    if count > 0:
        conn.close()
        return

    # Ensure default campaign exists
    conn.execute("""INSERT OR IGNORE INTO campaigns (name,template_set,status)
        VALUES ('albury_q2_2026','AB','active')""")
    conn.commit()
    cam_id = conn.execute(
        "SELECT id FROM campaigns WHERE name='albury_q2_2026'").fetchone()[0]

    prospects = [
        ("Operations Manager", "O'Brien Logistics",    "logistics",   "albury"),
        ("IT Manager",         "Visy Albury",           "manufacturing","albury"),
        ("Director",           "Border Trust Law",      "professional","albury"),
        ("CEO",                "Albury City Council",   "government",  "albury"),
        ("Owner",              "Murray Fresh Produce",  "agriculture", "wodonga"),
    ]
    now  = datetime.utcnow().isoformat()
    next_day = (datetime.utcnow() + timedelta(hours=1)).isoformat()
    for i, (role, company, industry, region) in enumerate(prospects):
        phi = round((i + 1) * PHI, 4)
        conn.execute("""INSERT INTO campaign_prospects
            (campaign_id,name,company,industry,region,state,template_variant,
             last_action,next_action,phi_score)
            VALUES (?,?,?,?,?,'prospect',?,?,?,?)""",
            (cam_id, role, company, industry, region,
             "A" if i % 2 == 0 else "B", now, next_day, phi))
    conn.commit(); conn.close()


def _render(template_key: str, prospect: dict) -> tuple:
    tmpl    = TEMPLATES.get(template_key, TEMPLATES["A_intro"])
    ctx     = {
        "name":         prospect.get("name", "there"),
        "company":      prospect.get("company", "your company"),
        "name_op":      OPERATOR,
        "abn":          ABN,
        "phone":        PHONE,
        "site":         SITE,
        "payment_link": PAYMENT_LINK,
    }
    subject = tmpl["subject"].format(**ctx)
    body    = tmpl["body"].format(**ctx)
    return subject, body


def _send_email(to_addr: str, subject: str, body: str) -> str:
    if not GMAIL_USER or not GMAIL_PASS:
        return "no_creds"
    if DRY_RUN:
        return "dry_run"
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = GMAIL_USER
        msg["To"]      = to_addr or GMAIL_USER
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as s:
            s.login(GMAIL_USER, GMAIL_PASS)
            s.sendmail(GMAIL_USER, [to_addr or GMAIL_USER], msg.as_string())
        return "sent"
    except Exception as e:
        return f"err:{e}"


def _process_due_actions():
    now  = datetime.utcnow().isoformat()
    conn = sqlite3.connect(DB, timeout=10)
    conn.row_factory = sqlite3.Row
    due  = conn.execute("""
        SELECT p.*, c.name as campaign_name
        FROM campaign_prospects p
        JOIN campaigns c ON c.id = p.campaign_id
        WHERE p.next_action <= ? AND p.state NOT IN ('closed_won','closed_lost')
        AND c.status = 'active'
        ORDER BY p.phi_score DESC
        LIMIT 20""", (now,)).fetchall()
    conn.close()

    acted = 0
    for p in due:
        state = p["state"]
        var   = p["template_variant"]

        # Choose template
        if state == "prospect":
            tkey = f"{var}_intro"
        elif state == "email_sent":
            tkey = "followup_warm"
        elif state == "warm":
            tkey = "A_intro"  # stronger CTA
        elif state == "proposal_sent":
            tkey = "proposal_reminder"
        else:
            continue

        subject, body = _render(tkey, dict(p))
        email_addr    = p["email"] or ""
        status        = _send_email(email_addr, subject, body)

        next_state  = NEXT_STATE.get(state, state)
        delay_days  = FOLLOWUP_DAYS.get(next_state, 7)
        next_action = (datetime.utcnow() + timedelta(days=delay_days)).isoformat()

        conn = sqlite3.connect(DB, timeout=10)
        conn.execute("""UPDATE campaign_prospects
            SET state=?, last_action=?, next_action=?
            WHERE id=?""", (next_state, now, next_action, p["id"]))
        conn.execute("""INSERT INTO campaign_events
            (prospect_id,event,template,channel,status)
            VALUES (?,?,?,?,?)""",
            (p["id"], f"action_{state}", tkey, "email", status))
        conn.commit(); conn.close()

        print(f"   [{status}] {p['company']:<30} {state}→{next_state} | {tkey}")
        acted += 1

    return len(due), acted


def _roi_snapshot():
    conn = sqlite3.connect(DB, timeout=10)
    conn.row_factory = sqlite3.Row
    cams = conn.execute("SELECT id FROM campaigns WHERE status='active'").fetchall()
    conn.close()

    for cam in cams:
        cid = cam["id"]
        conn = sqlite3.connect(DB, timeout=10)
        row = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN state='email_sent' OR state>'email_sent' THEN 1 ELSE 0 END) as sent,
                SUM(CASE WHEN state='warm' OR state>'warm' THEN 1 ELSE 0 END) as warm,
                SUM(CASE WHEN state='proposal_sent' OR state>'proposal_sent' THEN 1 ELSE 0 END) as props,
                SUM(CASE WHEN state='closed_won' THEN 1 ELSE 0 END) as won
            FROM campaign_prospects WHERE campaign_id=?""", (cid,)).fetchone()
        conn.close()

        revenue = (row["won"] or 0) * 2500.0  # avg deal $2500 AUD
        roi_pct = round((revenue / max((row["total"] or 1) * 5.0, 1)) * 100, 1)
        phi     = round(revenue * PHI / 1000, 4)

        conn = sqlite3.connect(DB, timeout=10)
        conn.execute("""INSERT INTO campaign_roi
            (campaign_id,prospects,emails_sent,warm_count,proposals,closed_won,
             revenue_aud,roi_pct,phi_score)
            VALUES (?,?,?,?,?,?,?,?,?)""",
            (cid, row["total"] or 0, row["sent"] or 0, row["warm"] or 0,
             row["props"] or 0, row["won"] or 0, revenue, roi_pct, phi))
        conn.commit(); conn.close()
        print(f"   ROI: prospects={row['total']} sent={row['sent']} "
              f"warm={row['warm']} won={row['won']} "
              f"revenue=${revenue:.0f} roi={roi_pct}%")


def run_cycle():
    ts = datetime.utcnow().isoformat()
    print(f"[fm-campaign-manager] {ts} | dry={DRY_RUN}")

    _seed_demo_prospects()
    due, acted = _process_due_actions()
    _roi_snapshot()
    print(f"   Processed {due} due | {acted} actions taken")


def _sigterm(s, f):
    global _running
    _running = False


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT,  _sigterm)
    _db_init()
    print(f"[fm-campaign-manager] Active | interval={INTERVAL}s | dry={DRY_RUN}")
    while _running:
        try:
            run_cycle()
        except Exception as e:
            print(f"[fm-campaign-manager] ERR {e}")
        for _ in range(INTERVAL):
            if not _running: break
            time.sleep(1)
    print("[fm-campaign-manager] Stopped.")
