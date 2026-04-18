#!/usr/bin/env python3
"""
fm_email_listener.py — Gmail IMAP watcher
Polls inbox for compliance signals and client emails; logs summaries to sovereign.db
Vault keys: GMAIL_USER, GMAIL_APP_PASS
Samuel James Hiotis | ABN 56628117363
"""
import os
import imaplib
import email
import json
import time
import signal
import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timezone

_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _line in _vault.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [EMAIL-LISTENER] %(message)s")
log = logging.getLogger("email_listener")

ROOT          = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB            = os.path.join(ROOT, "database", "sovereign.db")
GMAIL_USER    = os.getenv("GMAIL_USER", "")
GMAIL_PASS    = os.getenv("GMAIL_APP_PASS", "")
POLL_INTERVAL = int(os.getenv("EMAIL_POLL_INTERVAL", "300"))
MAX_FETCH     = int(os.getenv("EMAIL_MAX_FETCH", "10"))

KEYWORDS = ["compliance", "invoice", "audit", "fractal", "sovereign", "obrien", "blueprint"]

_running = True


def _db_init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS email_log (
        id INTEGER PRIMARY KEY, msg_id TEXT UNIQUE, subject TEXT,
        sender TEXT, matched_keyword TEXT, ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()


def _log_email(msg_id, subject, sender, keyword):
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("""INSERT OR IGNORE INTO email_log (msg_id,subject,sender,matched_keyword)
        VALUES (?,?,?,?)""", (msg_id, subject[:200], sender[:100], keyword))
    conn.commit(); conn.close()


def _fetch_emails() -> list:
    if not GMAIL_USER or not GMAIL_PASS:
        return []
    results = []
    try:
        imap = imaplib.IMAP4_SSL("imap.gmail.com", 993)
        imap.login(GMAIL_USER, GMAIL_PASS)
        imap.select("INBOX")
        _, data = imap.search(None, "UNSEEN")
        uids = data[0].split()[-MAX_FETCH:]
        for uid in reversed(uids):
            _, msg_data = imap.fetch(uid, "(RFC822.HEADER)")
            raw = msg_data[0][1] if msg_data and msg_data[0] else b""
            msg = email.message_from_bytes(raw)
            subject = str(msg.get("Subject", ""))
            sender  = str(msg.get("From", ""))
            msg_id  = str(msg.get("Message-ID", uid.decode()))
            kw      = next((k for k in KEYWORDS if k in subject.lower() or k in sender.lower()), "")
            results.append({"msg_id": msg_id, "subject": subject, "sender": sender, "keyword": kw})
        imap.logout()
    except Exception as e:
        log.warning("IMAP error: %s", e)
    return results


def poll():
    ts      = datetime.now(timezone.utc).isoformat()
    emails  = _fetch_emails()
    matched = [e for e in emails if e["keyword"]]
    status  = "polling" if GMAIL_USER else "credentials_not_set"
    out     = {"agent": "fm-email-listener", "status": status, "fetched": len(emails),
               "matched": len(matched), "ts": ts}
    print(json.dumps(out), flush=True)
    for e in emails:
        _log_email(e["msg_id"], e["subject"], e["sender"], e["keyword"])
        if e["keyword"]:
            log.info("MATCH [%s] subject=%s from=%s", e["keyword"], e["subject"][:60], e["sender"][:40])


def _sigterm(s, f):
    global _running
    _running = False


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    _db_init()
    log.info("fm-email-listener online | poll every %ds | key=%s", POLL_INTERVAL, "GMAIL_USER" if GMAIL_USER else "MISSING")
    while _running:
        try:
            poll()
        except Exception as e:
            log.error("ERR %s", e)
        for _ in range(POLL_INTERVAL):
            if not _running: break
            time.sleep(1)
    log.info("fm-email-listener stopped.")
