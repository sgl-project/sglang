"""
FractalMesh Drip Agent v2.1.0
5-step email drip sequence for warm leads.
Enrolls leads from leads table; 3-day step spacing; Gmail SMTP_SSL.
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
from datetime import datetime, timedelta, timezone

ROOT       = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB         = os.path.join(ROOT, "database", "sovereign.db")
INTERVAL   = int(os.getenv("DRIP_INTERVAL", "3600"))
DRY_RUN    = os.getenv("ENABLE_DRIP", "false").lower() != "true"

GMAIL_USER = os.getenv("GMAIL_USER",     "")
GMAIL_PASS = os.getenv("GMAIL_APP_PASS", "")
OPERATOR   = "Samuel James Hiotis"
ABN        = "56 628 117 363"
PHONE      = os.getenv("OPERATOR_PHONE", "0439 008 640")
SITE       = "https://fractalmesh.net"
PAYMENT    = os.getenv("STRIPE_PAYMENT_LINK", f"{SITE}/products.html")
CALENDAR   = f"{SITE}/contact.html"

PHI      = 1.6180339887
_running = True

STEP_GAP_DAYS = 3

SEQUENCE = [
    # step 0 — welcome
    {
        "subject": "Welcome to FractalMesh — sovereign infrastructure for solo operators",
        "body": """\
Hi {name},

Thanks for your interest in FractalMesh.

I'm Samuel James Hiotis — a sole trader building sovereign AI infrastructure
from a single Android phone + VPS stack in regional NSW.

What FractalMesh can do for you:
  • Infrastructure audits (NCC 2025-aligned)
  • Automated compliance documentation
  • PM2-managed AI agent swarms
  • Satellite intelligence reports (methane, AIS, crop yield)

If any of that is relevant to your operation, I'd love to show you.

Book a free 30-min call: {calendar}

— {operator}
ABN {abn} | {phone}
{site}
""",
    },
    # step 1 — proof
    {
        "subject": "FractalMesh — what a sovereign node actually looks like",
        "body": """\
Hi {name},

Quick follow-up.

The stack runs 44+ PM2 agents on a $6/month VPS + Android Termux node.
Everything is auditable — HMAC-signed events, SQLite WAL, no black boxes.

One thing clients ask about: the satellite intelligence reports.
We pull Sentinel-5P methane readings and AIS vessel tracking data
and package them into client-ready PDF/Markdown reports.

Sample use cases:
  • Carbon traders verifying super-emitter sites ($2,000 AUD/report)
  • Marine insurers tracking dark-fleet events ($4,500 AUD/report)
  • Commodity desks cross-checking NDVI crop yield signals ($1,500 AUD)

Want to see a sample report? Just reply.

— {operator}
ABN {abn} | {phone} | {site}
""",
    },
    # step 2 — offer
    {
        "subject": "FractalMesh Infrastructure Audit — A$750 flat",
        "body": """\
Hi {name},

I'd like to offer you a flat-rate infrastructure audit for your operation.

INFRASTRUCTURE AUDIT — A$750 AUD (ex GST)
  • 5 business days
  • NCC 2025 + SOC 2 control gap analysis
  • HMAC-signed audit trail
  • Written report with remediation steps
  • 50% upfront, 50% on delivery

This is the entry point for most clients before we scope compliance
automation or a full sovereign node deployment.

Secure payment: {payment}
Or reply to book a scoping call: {calendar}

— {operator}
ABN {abn} | {phone} | {site}
""",
    },
    # step 3 — urgency
    {
        "subject": "Last follow-up from FractalMesh",
        "body": """\
Hi {name},

I've sent a couple of notes — I won't keep filling your inbox.

If your infrastructure or compliance situation has changed,
or you'd like to explore the satellite intelligence reports,
I'm available for a quick call any time.

Book here: {calendar}

If the timing isn't right, no hard feelings — I'll remove you from follow-ups.

— {operator}
ABN {abn} | {phone} | {site}
""",
    },
    # step 4 — re-engage or close
    {
        "subject": "FractalMesh — leaving this here",
        "body": """\
Hi {name},

Final note.

If you ever need:
  • A quick infrastructure audit (A$750)
  • Compliance automation (A$1,500)
  • Sovereign node deployment (A$2,500–A$4,500)
  • Methane / AIS / crop yield intelligence reports

…I'm here: {site} | {phone}

Wishing you well.

— {operator} | ABN {abn}
""",
    },
]


def _db_init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS drip_sequences (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lead_email TEXT, sequence_name TEXT,
        step INTEGER DEFAULT 0, last_sent DATETIME,
        status TEXT DEFAULT 'active')""")
    conn.execute("""CREATE TABLE IF NOT EXISTS drip_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lead_email TEXT, step INTEGER, subject TEXT,
        status TEXT, ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()


def _enroll_new_leads():
    """Add leads that aren't yet in any drip sequence."""
    conn    = sqlite3.connect(DB, timeout=10)
    added   = 0
    try:
        leads = conn.execute("""
            SELECT email FROM leads
            WHERE email IS NOT NULL AND email != ''
            AND email NOT IN (
                SELECT lead_email FROM drip_sequences
            )
            LIMIT 50""").fetchall()
        for (email,) in leads:
            conn.execute("""INSERT INTO drip_sequences
                (lead_email, sequence_name, step, status)
                VALUES (?,?,0,?)""",
                (email, "main_v1", "active"))
            added += 1
        conn.commit()
    except Exception:
        pass
    finally:
        conn.close()
    if added:
        print(f"[drip] Enrolled {added} new leads")
    return added


def _due_sequences() -> list:
    """Return drip records where next step is due."""
    cutoff = (datetime.now(tz=timezone.utc)
              - timedelta(days=STEP_GAP_DAYS)).isoformat()
    conn   = sqlite3.connect(DB, timeout=10)
    conn.row_factory = sqlite3.Row
    rows   = conn.execute("""
        SELECT * FROM drip_sequences
        WHERE status = 'active'
        AND step < ?
        AND (last_sent IS NULL OR last_sent < ?)
        LIMIT 20""",
        (len(SEQUENCE), cutoff)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _lead_name(email: str) -> str:
    """Best-effort first name from leads table."""
    conn = sqlite3.connect(DB, timeout=10)
    try:
        row = conn.execute(
            "SELECT first_name FROM leads WHERE email=? LIMIT 1", (email,)).fetchone()
        if row and row[0]:
            return row[0].split()[0]
    except Exception:
        pass
    finally:
        conn.close()
    return "there"


def _send_email(to_addr: str, subject: str, body: str) -> str:
    if not GMAIL_USER or not GMAIL_PASS:
        return "no_creds"
    if DRY_RUN:
        return "dry_run"
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = GMAIL_USER
        msg["To"]      = to_addr
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as s:
            s.login(GMAIL_USER, GMAIL_PASS)
            s.sendmail(GMAIL_USER, [to_addr], msg.as_string())
        return "sent"
    except Exception as e:
        return f"err:{e}"


def _advance_sequence(seq_id: int, step: int, email: str, subject: str, status: str):
    now  = datetime.now(tz=timezone.utc).isoformat()
    next_step = step + 1
    new_status = "completed" if next_step >= len(SEQUENCE) else "active"
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("UPDATE drip_sequences SET step=?, last_sent=?, status=? WHERE id=?",
                 (next_step, now, new_status, seq_id))
    conn.execute("INSERT INTO drip_log (lead_email,step,subject,status) VALUES (?,?,?,?)",
                 (email, step, subject[:200], status))
    conn.commit(); conn.close()


def run_cycle():
    ts = datetime.now(tz=timezone.utc).isoformat()
    print(f"[drip] {ts} | dry={DRY_RUN}")

    enrolled = _enroll_new_leads()
    due      = _due_sequences()
    print(f"   Due: {len(due)} sequences")

    for seq in due:
        step  = seq["step"]
        email = seq["lead_email"]
        tpl   = SEQUENCE[step]
        name  = _lead_name(email)

        subject = tpl["subject"]
        body    = tpl["body"].format(
            name=name, operator=OPERATOR, abn=ABN,
            phone=PHONE, site=SITE, payment=PAYMENT, calendar=CALENDAR,
        )

        status = _send_email(email, subject, body)
        _advance_sequence(seq["id"], step, email, subject, status)
        print(f"   → {email:<35} step={step} | {status}")

    # Summary
    conn = sqlite3.connect(DB, timeout=10)
    try:
        active    = conn.execute(
            "SELECT COUNT(*) FROM drip_sequences WHERE status='active'").fetchone()[0]
        completed = conn.execute(
            "SELECT COUNT(*) FROM drip_sequences WHERE status='completed'").fetchone()[0]
        sent_24h  = conn.execute(
            "SELECT COUNT(*) FROM drip_log WHERE ts>datetime('now','-1 day') "
            "AND status IN ('sent','dry_run')").fetchone()[0]
    except Exception:
        active = completed = sent_24h = 0
    finally:
        conn.close()
    phi = round(active * PHI / max(active, 1), 4)
    print(f"   Sequences: {active} active | {completed} completed | "
          f"{sent_24h} sent/24h | φ={phi}")


def _sigterm(s, f):
    global _running
    _running = False


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT,  _sigterm)
    _db_init()
    print(f"[drip] Active | interval={INTERVAL}s | dry={DRY_RUN} | "
          f"steps={len(SEQUENCE)} | gap={STEP_GAP_DAYS}d")
    while _running:
        try:
            run_cycle()
        except Exception as e:
            print(f"[drip] ERR {e}")
        for _ in range(INTERVAL):
            if not _running:
                break
            time.sleep(1)
    print("[drip] Stopped.")
