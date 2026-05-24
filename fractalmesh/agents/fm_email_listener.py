#!/usr/bin/env python3
"""
fm_email_listener.py — Email Listener / Inbound Processing Agent
FractalMesh OMEGA Titan | Port 7845

Polls Gmail via IMAP, parses inbound emails, applies rule-based automation
(tag_lead, auto_reply, forward_to_mcp, ignore), exposes REST API for email
management. SQLite WAL at ~/fmsaas/database/sovereign.db.

Vault keys: GMAIL_USER, GMAIL_APP_PASS, IMAP_HOST, IMAP_PORT,
            EMAIL_CHECK_INTERVAL, EMAIL_LISTENER_PORT, ADMIN_SECRET
Samuel James Hiotis | ABN 56628117363
"""

import os
import json
import re
import sqlite3
import time
import email
import email.header
import imaplib
import smtplib
import threading
import logging
import urllib.request
import urllib.parse
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import quopri
import base64

# ---------------------------------------------------------------------------
# Vault loader
# ---------------------------------------------------------------------------
_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _line in _vault.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

# ---------------------------------------------------------------------------
# Config (from env / vault — never hardcoded)
# ---------------------------------------------------------------------------
PORT                 = int(os.getenv("EMAIL_LISTENER_PORT", "7845"))
GMAIL_USER           = os.getenv("GMAIL_USER", "")
GMAIL_APP_PASS       = os.getenv("GMAIL_APP_PASS", "")
IMAP_HOST            = os.getenv("IMAP_HOST", "imap.gmail.com")
IMAP_PORT            = int(os.getenv("IMAP_PORT", "993"))
EMAIL_CHECK_INTERVAL = int(os.getenv("EMAIL_CHECK_INTERVAL", "120"))
ADMIN_SECRET         = os.getenv("ADMIN_SECRET", "")

ROOT = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB   = os.path.join(ROOT, "database", "sovereign.db")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [EMAIL-LISTENER] %(levelname)s %(message)s",
)
log = logging.getLogger("fm_email_listener")

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB, timeout=15, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _init_db():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS inbound_emails (
            id              INTEGER PRIMARY KEY,
            message_id      TEXT UNIQUE,
            from_addr       TEXT,
            subject         TEXT,
            body_text       TEXT,
            received_at     REAL,
            processed       INTEGER DEFAULT 0,
            category        TEXT,
            intent          TEXT,
            lead_extracted  INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS email_rules (
            id          INTEGER PRIMARY KEY,
            name        TEXT UNIQUE,
            match_field TEXT,
            pattern     TEXT,
            action      TEXT,
            priority    INTEGER DEFAULT 5,
            enabled     INTEGER DEFAULT 1,
            match_count INTEGER DEFAULT 0,
            created_at  REAL
        );
        CREATE TABLE IF NOT EXISTS email_replies (
            id       INTEGER PRIMARY KEY,
            email_id INTEGER,
            to_addr  TEXT,
            subject  TEXT,
            body     TEXT,
            sent_at  REAL,
            status   TEXT
        );
        CREATE TABLE IF NOT EXISTS leads (
            id         INTEGER PRIMARY KEY,
            name       TEXT,
            email      TEXT UNIQUE,
            phone      TEXT,
            company    TEXT,
            source     TEXT DEFAULT 'email',
            created_at REAL
        );
    """)
    conn.commit()
    conn.close()


def _row_to_dict(row) -> dict:
    if row is None:
        return None
    return dict(row)

# ---------------------------------------------------------------------------
# Email parsing helpers
# ---------------------------------------------------------------------------

def _decode_header_value(value: str) -> str:
    """Decode RFC2047-encoded email headers."""
    if not value:
        return ""
    parts = email.header.decode_header(value)
    decoded = []
    for chunk, charset in parts:
        if isinstance(chunk, bytes):
            try:
                decoded.append(chunk.decode(charset or "utf-8", errors="replace"))
            except Exception:
                decoded.append(chunk.decode("utf-8", errors="replace"))
        else:
            decoded.append(chunk)
    return "".join(decoded)


def _decode_body(part) -> str:
    """Decode a message part, handling base64, quoted-printable, and plain."""
    payload = part.get_payload(decode=True)
    if payload is None:
        return ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except Exception:
        return payload.decode("utf-8", errors="replace")


def _parse_email(raw_bytes: bytes):
    """
    Parse raw RFC822 bytes.
    Returns (message_id, from_addr, subject, date, body_text).
    """
    msg = email.message_from_bytes(raw_bytes)

    message_id = msg.get("Message-ID", "").strip()
    from_addr  = _decode_header_value(msg.get("From", ""))
    subject    = _decode_header_value(msg.get("Subject", ""))
    date_str   = msg.get("Date", "")

    # Convert date to epoch float
    try:
        from email.utils import parsedate_to_datetime
        received_at = parsedate_to_datetime(date_str).timestamp()
    except Exception:
        received_at = time.time()

    # Extract plain-text body
    body_text = ""
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            cd = str(part.get("Content-Disposition", ""))
            if ct == "text/plain" and "attachment" not in cd:
                body_text = _decode_body(part)
                break
    else:
        if msg.get_content_type() == "text/plain":
            body_text = _decode_body(msg)

    return message_id, from_addr, subject, received_at, body_text.strip()


def _extract_contacts(text: str) -> dict:
    """
    Scan text for email, phone, and name patterns.
    Returns dict with keys: email, phone, name.
    """
    result = {"email": "", "phone": "", "name": ""}

    # Email
    email_match = re.search(
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text
    )
    if email_match:
        result["email"] = email_match.group(0)

    # Phone (various formats)
    phone_match = re.search(
        r"(?:\+?1[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}", text
    )
    if phone_match:
        result["phone"] = phone_match.group(0).strip()

    # Name patterns
    name_patterns = [
        r"[Mm]y name is ([A-Z][a-z]+(?: [A-Z][a-z]+)+)",
        r"[Ii] am ([A-Z][a-z]+(?: [A-Z][a-z]+)+)",
        r"[Tt]his is ([A-Z][a-z]+(?: [A-Z][a-z]+)+)",
        r"^([A-Z][a-z]+(?: [A-Z][a-z]+)+),?\s*$",
        r"[Rr]egards?,?\s*\n([A-Z][a-z]+(?: [A-Z][a-z]+)+)",
        r"[Tt]hanks?,?\s*\n([A-Z][a-z]+(?: [A-Z][a-z]+)+)",
    ]
    for pat in name_patterns:
        m = re.search(pat, text, re.MULTILINE)
        if m:
            result["name"] = m.group(1).strip()
            break

    return result

# ---------------------------------------------------------------------------
# SMTP helper
# ---------------------------------------------------------------------------

def _smtp_reply(to: str, subject: str, body: str) -> str:
    """Send reply via Gmail SMTP SSL. Returns sent Message-ID."""
    if not GMAIL_USER or not GMAIL_APP_PASS:
        raise RuntimeError("SMTP credentials not configured")

    msg = MIMEMultipart("alternative")
    msg["From"]    = GMAIL_USER
    msg["To"]      = to
    msg["Subject"] = subject
    sent_id        = f"<fm-{int(time.time())}@fractalmesh>"
    msg["Message-ID"] = sent_id
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_USER, GMAIL_APP_PASS)
        server.sendmail(GMAIL_USER, [to], msg.as_string())

    log.info("SMTP reply sent to %s subject=%s", to, subject[:60])
    return sent_id

# ---------------------------------------------------------------------------
# Rule engine
# ---------------------------------------------------------------------------

def _run_rules(conn: sqlite3.Connection, email_row: dict) -> list:
    """
    Apply all enabled email_rules to email_row (ordered by priority asc).
    Returns list of dicts describing actions taken.
    """
    rules = conn.execute(
        "SELECT * FROM email_rules WHERE enabled=1 ORDER BY priority ASC"
    ).fetchall()

    actions_taken = []

    for rule in rules:
        rule = dict(rule)
        field   = rule["match_field"]  # from, subject, body
        pattern = rule["pattern"]
        action  = rule["action"]

        target = ""
        if field == "from":
            target = email_row.get("from_addr", "")
        elif field == "subject":
            target = email_row.get("subject", "")
        elif field == "body":
            target = email_row.get("body_text", "")

        try:
            matched = bool(re.search(pattern, target, re.IGNORECASE))
        except re.error:
            log.warning("Invalid regex in rule %s: %s", rule["name"], pattern)
            continue

        if not matched:
            continue

        # Increment match_count
        conn.execute(
            "UPDATE email_rules SET match_count=match_count+1 WHERE id=?",
            (rule["id"],),
        )

        log.info(
            "Rule '%s' matched email id=%s action=%s",
            rule["name"], email_row.get("id"), action,
        )

        action_result = {"rule": rule["name"], "action": action, "status": "ok"}

        if action == "ignore":
            actions_taken.append(action_result)
            break  # stop processing further rules

        elif action == "tag_lead":
            try:
                contacts = _extract_contacts(email_row.get("body_text", ""))
                lead_email = contacts["email"] or email_row.get("from_addr", "")
                # Clean angle-bracket notation
                m = re.search(r"<([^>]+)>", lead_email)
                if m:
                    lead_email = m.group(1)
                conn.execute(
                    """INSERT OR IGNORE INTO leads
                       (name, email, phone, company, source, created_at)
                       VALUES (?,?,?,?,?,?)""",
                    (
                        contacts["name"],
                        lead_email,
                        contacts["phone"],
                        "",
                        "email",
                        time.time(),
                    ),
                )
                conn.execute(
                    "UPDATE inbound_emails SET lead_extracted=1, category='lead' WHERE id=?",
                    (email_row["id"],),
                )
                action_result["lead_email"] = lead_email
            except Exception as exc:
                action_result["status"] = f"error: {exc}"
            actions_taken.append(action_result)

        elif action == "auto_reply":
            try:
                from_raw = email_row.get("from_addr", "")
                m = re.search(r"<([^>]+)>", from_raw)
                to_addr = m.group(1) if m else from_raw
                subj = "Re: " + email_row.get("subject", "")
                body = (
                    "Thank you for your message. We'll be in touch shortly.\n\n"
                    "Best,\nFractalMesh Team"
                )
                sent_id = _smtp_reply(to_addr, subj, body)
                conn.execute(
                    """INSERT INTO email_replies
                       (email_id, to_addr, subject, body, sent_at, status)
                       VALUES (?,?,?,?,?,?)""",
                    (email_row["id"], to_addr, subj, body, time.time(), "sent"),
                )
                action_result["sent_to"] = to_addr
                action_result["message_id"] = sent_id
            except Exception as exc:
                action_result["status"] = f"error: {exc}"
            actions_taken.append(action_result)

        elif action == "forward_to_mcp":
            try:
                payload = json.dumps(
                    {
                        "source": "email_listener",
                        "email_id": email_row.get("id"),
                        "from": email_row.get("from_addr"),
                        "subject": email_row.get("subject"),
                        "body": email_row.get("body_text", "")[:1000],
                    }
                ).encode()
                req = urllib.request.Request(
                    "http://localhost:7785/",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    action_result["mcp_status"] = resp.status
            except Exception as exc:
                action_result["status"] = f"error: {exc}"
            actions_taken.append(action_result)

    # Mark email as processed
    if actions_taken:
        conn.execute(
            "UPDATE inbound_emails SET processed=1 WHERE id=?",
            (email_row["id"],),
        )

    conn.commit()
    return actions_taken

# ---------------------------------------------------------------------------
# IMAP poller (background daemon thread)
# ---------------------------------------------------------------------------

def _imap_poll_loop():
    if not GMAIL_USER or not GMAIL_APP_PASS:
        log.warning("IMAP credentials not configured — polling disabled")
        while True:
            time.sleep(3600)
        return

    log.info(
        "IMAP poller started | host=%s port=%d interval=%ds user=%s",
        IMAP_HOST, IMAP_PORT, EMAIL_CHECK_INTERVAL, GMAIL_USER,
    )

    while True:
        try:
            _poll_imap_once()
        except Exception as exc:
            log.error("IMAP poll error: %s", exc)
        time.sleep(EMAIL_CHECK_INTERVAL)


def _poll_imap_once():
    imap = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
    try:
        imap.login(GMAIL_USER, GMAIL_APP_PASS)
        imap.select("INBOX")
        status, data = imap.search(None, "UNSEEN")
        if status != "OK":
            return
        uids = data[0].split()
        if not uids:
            return

        log.info("IMAP: %d unseen messages found", len(uids))
        conn = _get_conn()

        for uid in uids:
            try:
                _, msg_data = imap.fetch(uid, "(RFC822)")
                if not msg_data or not msg_data[0]:
                    continue
                raw = msg_data[0][1]
                if not isinstance(raw, bytes):
                    continue

                message_id, from_addr, subject, received_at, body_text = _parse_email(raw)

                if not message_id:
                    message_id = f"<fm-gen-{uid.decode()}-{int(time.time())}>"

                # Skip if already stored
                existing = conn.execute(
                    "SELECT id FROM inbound_emails WHERE message_id=?",
                    (message_id,),
                ).fetchone()
                if existing:
                    continue

                cursor = conn.execute(
                    """INSERT OR IGNORE INTO inbound_emails
                       (message_id, from_addr, subject, body_text, received_at)
                       VALUES (?,?,?,?,?)""",
                    (message_id, from_addr, subject, body_text, received_at),
                )
                conn.commit()
                new_id = cursor.lastrowid

                if new_id:
                    email_row = _row_to_dict(
                        conn.execute(
                            "SELECT * FROM inbound_emails WHERE id=?", (new_id,)
                        ).fetchone()
                    )
                    if email_row:
                        actions = _run_rules(conn, email_row)
                        log.info(
                            "Stored email id=%d from=%s subject=%s rules=%d",
                            new_id, from_addr[:40], subject[:60], len(actions),
                        )

                # Mark as SEEN
                imap.store(uid, "+FLAGS", "\\Seen")

            except Exception as exc:
                log.error("Error processing uid %s: %s", uid, exc)

        conn.close()
    finally:
        try:
            imap.logout()
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Default rules seeder
# ---------------------------------------------------------------------------

DEFAULT_RULES = [
    {
        "name": "new_inquiry",
        "match_field": "subject",
        "pattern": r"(?i)interested|quote|pricing|how much",
        "action": "tag_lead",
        "priority": 1,
    },
    {
        "name": "unsubscribe",
        "match_field": "body",
        "pattern": r"(?i)unsubscribe|opt.?out|remove me",
        "action": "ignore",
        "priority": 2,
    },
    {
        "name": "spam_filter",
        "match_field": "from",
        "pattern": r"(?i)noreply|no-reply|donotreply",
        "action": "ignore",
        "priority": 3,
    },
    {
        "name": "auto_ack_leads",
        "match_field": "subject",
        "pattern": r"(?i)contact|enquiry|hello|hi",
        "action": "auto_reply",
        "priority": 4,
    },
    {
        "name": "mcp_forward",
        "match_field": "subject",
        "pattern": r"@fractalmesh|COMMAND|#mesh",
        "action": "forward_to_mcp",
        "priority": 5,
    },
]


def _seed_rules(conn: sqlite3.Connection) -> int:
    count = 0
    for r in DEFAULT_RULES:
        cursor = conn.execute(
            """INSERT OR IGNORE INTO email_rules
               (name, match_field, pattern, action, priority, enabled, match_count, created_at)
               VALUES (?,?,?,?,?,1,0,?)""",
            (r["name"], r["match_field"], r["pattern"], r["action"], r["priority"], time.time()),
        )
        if cursor.rowcount:
            count += 1
    conn.commit()
    return count

# ---------------------------------------------------------------------------
# HTTP Request Handler
# ---------------------------------------------------------------------------

def _require_admin(handler) -> bool:
    secret = handler.headers.get("X-Admin-Secret", "")
    if not ADMIN_SECRET or secret != ADMIN_SECRET:
        handler._send_json({"error": "Unauthorized"}, 401)
        return False
    return True


def _read_body(handler) -> dict:
    length = int(handler.headers.get("Content-Length", 0))
    if length == 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw)
    except Exception:
        return {}


class EmailListenerHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        log.debug("HTTP %s", fmt % args)

    def _send_json(self, data, code=200):
        body = json.dumps(data, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ------------------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        qs   = urllib.parse.parse_qs(self.path.split("?")[1]) if "?" in self.path else {}

        # GET /health
        if path == "/health":
            self._send_json({
                "status": "ok",
                "service": "fm-email-listener",
                "port": PORT,
                "imap_configured": bool(GMAIL_USER and GMAIL_APP_PASS),
            })

        # GET /emails
        elif path == "/emails":
            self._handle_get_emails(qs)

        # GET /emails/{id}
        elif re.match(r"^/emails/\d+$", path):
            email_id = int(path.split("/")[-1])
            self._handle_get_email(email_id)

        # GET /rules
        elif path == "/rules":
            conn = _get_conn()
            rows = [_row_to_dict(r) for r in conn.execute(
                "SELECT * FROM email_rules ORDER BY priority ASC"
            ).fetchall()]
            conn.close()
            self._send_json(rows)

        # GET /replies
        elif path == "/replies":
            conn = _get_conn()
            rows = [_row_to_dict(r) for r in conn.execute(
                "SELECT * FROM email_replies ORDER BY sent_at DESC"
            ).fetchall()]
            conn.close()
            self._send_json(rows)

        # GET /analytics
        elif path == "/analytics":
            self._handle_analytics()

        else:
            self._send_json({"error": "Not found"}, 404)

    # ------------------------------------------------------------------
    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")

        # POST /emails/{id}/process
        if re.match(r"^/emails/\d+/process$", path):
            email_id = int(path.split("/")[-2])
            self._handle_process_email(email_id)

        # POST /emails/{id}/reply
        elif re.match(r"^/emails/\d+/reply$", path):
            email_id = int(path.split("/")[-2])
            self._handle_reply_email(email_id)

        # POST /emails/{id}/extract_lead
        elif re.match(r"^/emails/\d+/extract_lead$", path):
            email_id = int(path.split("/")[-2])
            self._handle_extract_lead(email_id)

        # POST /rules/create
        elif path == "/rules/create":
            if not _require_admin(self):
                return
            self._handle_create_rule()

        # POST /rules/seed
        elif path == "/rules/seed":
            conn = _get_conn()
            seeded = _seed_rules(conn)
            conn.close()
            self._send_json({"seeded": seeded})

        else:
            self._send_json({"error": "Not found"}, 404)

    # ------------------------------------------------------------------
    def do_PUT(self):
        path = self.path.split("?")[0].rstrip("/")

        # PUT /rules/{id}
        if re.match(r"^/rules/\d+$", path):
            if not _require_admin(self):
                return
            rule_id = int(path.split("/")[-1])
            self._handle_update_rule(rule_id)
        else:
            self._send_json({"error": "Not found"}, 404)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _handle_get_emails(self, qs: dict):
        conditions = []
        params     = []

        processed = qs.get("processed", [None])[0]
        if processed is not None:
            val = 0 if processed.lower() in ("false", "0") else 1
            conditions.append("processed=?")
            params.append(val)

        category = qs.get("category", [None])[0]
        if category:
            conditions.append("category=?")
            params.append(category)

        limit = int(qs.get("limit", ["50"])[0])

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql   = f"SELECT * FROM inbound_emails {where} ORDER BY received_at DESC LIMIT ?"
        params.append(limit)

        conn = _get_conn()
        rows = conn.execute(sql, params).fetchall()
        conn.close()

        result = []
        for r in rows:
            d = _row_to_dict(r)
            if d.get("body_text"):
                d["body_text"] = d["body_text"][:500]
            result.append(d)
        self._send_json(result)

    def _handle_get_email(self, email_id: int):
        conn = _get_conn()
        row  = conn.execute(
            "SELECT * FROM inbound_emails WHERE id=?", (email_id,)
        ).fetchone()
        conn.close()
        if row is None:
            self._send_json({"error": "Not found"}, 404)
            return
        self._send_json(_row_to_dict(row))

    def _handle_process_email(self, email_id: int):
        conn = _get_conn()
        row  = conn.execute(
            "SELECT * FROM inbound_emails WHERE id=?", (email_id,)
        ).fetchone()
        if row is None:
            conn.close()
            self._send_json({"error": "Not found"}, 404)
            return
        email_row   = _row_to_dict(row)
        actions     = _run_rules(conn, email_row)
        conn.close()
        self._send_json({"rules_matched": len(actions), "actions_taken": actions})

    def _handle_reply_email(self, email_id: int):
        body = _read_body(self)
        reply_body = body.get("body", "")
        if not reply_body:
            self._send_json({"error": "body required"}, 400)
            return

        conn = _get_conn()
        row  = conn.execute(
            "SELECT * FROM inbound_emails WHERE id=?", (email_id,)
        ).fetchone()
        if row is None:
            conn.close()
            self._send_json({"error": "Not found"}, 404)
            return

        email_row = _row_to_dict(row)
        from_raw  = email_row.get("from_addr", "")
        m         = re.search(r"<([^>]+)>", from_raw)
        to_addr   = m.group(1) if m else from_raw
        subject   = "Re: " + email_row.get("subject", "")

        try:
            sent_id = _smtp_reply(to_addr, subject, reply_body)
            conn.execute(
                """INSERT INTO email_replies
                   (email_id, to_addr, subject, body, sent_at, status)
                   VALUES (?,?,?,?,?,?)""",
                (email_id, to_addr, subject, reply_body, time.time(), "sent"),
            )
            conn.commit()
            conn.close()
            self._send_json({"sent": True, "to": to_addr, "message_id": sent_id})
        except Exception as exc:
            conn.close()
            self._send_json({"sent": False, "error": str(exc)}, 500)

    def _handle_extract_lead(self, email_id: int):
        conn = _get_conn()
        row  = conn.execute(
            "SELECT * FROM inbound_emails WHERE id=?", (email_id,)
        ).fetchone()
        if row is None:
            conn.close()
            self._send_json({"error": "Not found"}, 404)
            return

        email_row = _row_to_dict(row)
        body      = email_row.get("body_text", "")
        contacts  = _extract_contacts(body)

        from_raw = email_row.get("from_addr", "")
        m = re.search(r"<([^>]+)>", from_raw)
        lead_email = m.group(1) if m else from_raw

        # Prefer extracted email if found, else use from_addr
        if contacts["email"]:
            lead_email = contacts["email"]

        try:
            cursor = conn.execute(
                """INSERT OR IGNORE INTO leads
                   (name, email, phone, company, source, created_at)
                   VALUES (?,?,?,?,?,?)""",
                (
                    contacts["name"],
                    lead_email,
                    contacts["phone"],
                    "",
                    "email",
                    time.time(),
                ),
            )
            lead_id = cursor.lastrowid
            conn.execute(
                "UPDATE inbound_emails SET lead_extracted=1, category='lead' WHERE id=?",
                (email_id,),
            )
            conn.commit()
            conn.close()
            self._send_json({
                "lead_id": lead_id,
                "name":    contacts["name"],
                "email":   lead_email,
                "phone":   contacts["phone"],
            })
        except Exception as exc:
            conn.close()
            self._send_json({"error": str(exc)}, 500)

    def _handle_create_rule(self):
        body = _read_body(self)
        required = ["name", "match_field", "pattern", "action"]
        for f in required:
            if not body.get(f):
                self._send_json({"error": f"Missing field: {f}"}, 400)
                return

        conn = _get_conn()
        try:
            cursor = conn.execute(
                """INSERT INTO email_rules
                   (name, match_field, pattern, action, priority, enabled, match_count, created_at)
                   VALUES (?,?,?,?,?,1,0,?)""",
                (
                    body["name"],
                    body["match_field"],
                    body["pattern"],
                    body["action"],
                    int(body.get("priority", 5)),
                    time.time(),
                ),
            )
            conn.commit()
            rule_id = cursor.lastrowid
            row = conn.execute(
                "SELECT * FROM email_rules WHERE id=?", (rule_id,)
            ).fetchone()
            conn.close()
            self._send_json(_row_to_dict(row), 201)
        except sqlite3.IntegrityError as exc:
            conn.close()
            self._send_json({"error": str(exc)}, 409)

    def _handle_update_rule(self, rule_id: int):
        body = _read_body(self)
        conn = _get_conn()

        row = conn.execute(
            "SELECT * FROM email_rules WHERE id=?", (rule_id,)
        ).fetchone()
        if row is None:
            conn.close()
            self._send_json({"error": "Not found"}, 404)
            return

        current = _row_to_dict(row)
        updatable = ["match_field", "pattern", "action", "priority", "enabled", "name"]
        for key in updatable:
            if key in body:
                current[key] = body[key]

        conn.execute(
            """UPDATE email_rules
               SET name=?, match_field=?, pattern=?, action=?,
                   priority=?, enabled=?
               WHERE id=?""",
            (
                current["name"],
                current["match_field"],
                current["pattern"],
                current["action"],
                current["priority"],
                current["enabled"],
                rule_id,
            ),
        )
        conn.commit()
        updated = _row_to_dict(
            conn.execute("SELECT * FROM email_rules WHERE id=?", (rule_id,)).fetchone()
        )
        conn.close()
        self._send_json(updated)

    def _handle_analytics(self):
        conn  = _get_conn()
        today = time.time() - 86400

        emails_total     = conn.execute("SELECT COUNT(*) FROM inbound_emails").fetchone()[0]
        emails_processed = conn.execute(
            "SELECT COUNT(*) FROM inbound_emails WHERE processed=1"
        ).fetchone()[0]
        emails_today     = conn.execute(
            "SELECT COUNT(*) FROM inbound_emails WHERE received_at>=?", (today,)
        ).fetchone()[0]
        leads_extracted  = conn.execute(
            "SELECT COUNT(*) FROM inbound_emails WHERE lead_extracted=1"
        ).fetchone()[0]
        replies_sent     = conn.execute("SELECT COUNT(*) FROM email_replies").fetchone()[0]

        rules_by_action = {}
        for row in conn.execute(
            "SELECT action, COUNT(*) as cnt FROM email_rules GROUP BY action"
        ).fetchall():
            rules_by_action[row[0]] = row[1]

        conn.close()
        self._send_json({
            "emails_total":     emails_total,
            "emails_processed": emails_processed,
            "emails_today":     emails_today,
            "leads_extracted":  leads_extracted,
            "replies_sent":     replies_sent,
            "rules_by_action":  rules_by_action,
        })

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    log.info("fm-email-listener starting | port=%d", PORT)
    _init_db()

    conn = _get_conn()
    seeded = _seed_rules(conn)
    conn.close()
    if seeded:
        log.info("Seeded %d default email rules", seeded)

    # Start IMAP poller in background daemon thread
    poller = threading.Thread(target=_imap_poll_loop, name="imap-poller", daemon=True)
    poller.start()

    server = HTTPServer(("0.0.0.0", PORT), EmailListenerHandler)
    log.info(
        "HTTP listening on 0.0.0.0:%d | IMAP=%s",
        PORT,
        "enabled" if (GMAIL_USER and GMAIL_APP_PASS) else "disabled",
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("fm-email-listener stopped.")


if __name__ == "__main__":
    main()
