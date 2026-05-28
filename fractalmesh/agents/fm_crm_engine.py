#!/usr/bin/env python3
"""
fm_crm_engine.py — FractalMesh OMEGA Titan Customer Relationship Management Engine
Port: 7860

Full CRM for managing leads, contacts, companies, deals, and interactions.
Tracks the entire customer lifecycle from lead to paying customer.
Includes deal pipeline, activity logging, and drip campaign scheduling.

Samuel James Hiotis | ABN 56 628 117 363
"""

import hashlib
import hmac
import json
import math
import os
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ---------------------------------------------------------------------------
# Vault loading
# ---------------------------------------------------------------------------
_ENV_FILE = Path.home() / ".secrets" / "fractal.env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PORT             = int(os.environ.get("CRM_ENGINE_PORT", "7860"))
SENDGRID_API_KEY = os.environ.get("SENDGRID_API_KEY", "")
FROM_EMAIL       = os.environ.get("SENDGRID_FROM_EMAIL", "")
ADMIN_SECRET     = os.environ.get("ADMIN_SECRET", "")
MCP_SECRET       = os.environ.get("MCP_SECRET", "")
MCP_PORT         = int(os.environ.get("MCP_PORT", "7785"))

ROOT     = Path.home() / "fmsaas"
DB_PATH  = ROOT / "database" / "sovereign.db"
LOG_PATH = ROOT / "logs" / "crm_engine.log"

for _d in (DB_PATH.parent, LOG_PATH.parent):
    _d.mkdir(parents=True, exist_ok=True)

START_TIME = time.time()

# ---------------------------------------------------------------------------
# Deal stages
# ---------------------------------------------------------------------------
DEAL_STAGES = ["prospecting", "qualification", "proposal", "negotiation", "closed_won", "closed_lost"]

# ---------------------------------------------------------------------------
# Simple logger (no external deps)
# ---------------------------------------------------------------------------
_log_lock = threading.Lock()

def _log(level: str, msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    entry = f"{ts} [{level.upper()}] crm_engine: {msg}"
    with _log_lock:
        try:
            with open(LOG_PATH, "a") as _f:
                _f.write(entry + "\n")
        except Exception:
            pass
        print(entry, flush=True)

def log_info(msg: str):  _log("INFO",  msg)
def log_warn(msg: str):  _log("WARN",  msg)
def log_err(msg: str):   _log("ERROR", msg)

# ---------------------------------------------------------------------------
# Database initialisation
# ---------------------------------------------------------------------------
def _db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=10, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def db_init() -> None:
    conn = _db_connect()
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS contacts (
            id              INTEGER PRIMARY KEY,
            email           TEXT UNIQUE NOT NULL,
            first_name      TEXT,
            last_name       TEXT,
            company         TEXT,
            phone           TEXT,
            title           TEXT,
            source          TEXT,
            lifecycle_stage TEXT    DEFAULT 'lead',
            lead_score      INTEGER DEFAULT 0,
            tags            TEXT,
            notes           TEXT,
            created_at      REAL,
            updated_at      REAL,
            last_activity_at REAL
        );

        CREATE TABLE IF NOT EXISTS companies (
            id              INTEGER PRIMARY KEY,
            name            TEXT UNIQUE NOT NULL,
            domain          TEXT,
            industry        TEXT,
            size            TEXT,
            annual_revenue  REAL,
            country         TEXT,
            city            TEXT,
            notes           TEXT,
            created_at      REAL,
            updated_at      REAL
        );

        CREATE TABLE IF NOT EXISTS deals (
            id              INTEGER PRIMARY KEY,
            title           TEXT NOT NULL,
            contact_id      INTEGER,
            company_id      INTEGER,
            stage           TEXT    DEFAULT 'prospecting',
            value           REAL    DEFAULT 0,
            currency        TEXT    DEFAULT 'AUD',
            probability     REAL    DEFAULT 0.1,
            expected_close  REAL,
            won_at          REAL,
            lost_at         REAL,
            lost_reason     TEXT,
            created_at      REAL,
            updated_at      REAL
        );

        CREATE TABLE IF NOT EXISTS activities (
            id              INTEGER PRIMARY KEY,
            contact_id      INTEGER,
            deal_id         INTEGER,
            activity_type   TEXT,
            subject         TEXT,
            body            TEXT,
            outcome         TEXT,
            scheduled_at    REAL,
            completed_at    REAL,
            created_at      REAL
        );

        CREATE TABLE IF NOT EXISTS email_sequences (
            id          INTEGER PRIMARY KEY,
            name        TEXT NOT NULL,
            status      TEXT DEFAULT 'active',
            created_at  REAL
        );

        CREATE TABLE IF NOT EXISTS sequence_steps (
            id              INTEGER PRIMARY KEY,
            sequence_id     INTEGER NOT NULL,
            step_order      INTEGER NOT NULL,
            delay_hours     INTEGER NOT NULL,
            subject         TEXT,
            body_template   TEXT
        );

        CREATE TABLE IF NOT EXISTS sequence_enrollments (
            id              INTEGER PRIMARY KEY,
            sequence_id     INTEGER NOT NULL,
            contact_id      INTEGER NOT NULL,
            current_step    INTEGER DEFAULT 0,
            status          TEXT    DEFAULT 'active',
            enrolled_at     REAL,
            next_send_at    REAL
        );
    """)

    conn.commit()
    conn.close()
    log_info("Database initialised (WAL mode)")


# ---------------------------------------------------------------------------
# Lead scoring
# ---------------------------------------------------------------------------
def recalculate_lead_score(contact_id: int) -> int:
    """Recalculate and persist lead_score for a contact."""
    now = time.time()
    try:
        conn = _db_connect()
        cur  = conn.cursor()

        score = 0

        # +10 for each completed activity
        row = cur.execute(
            "SELECT COUNT(*) FROM activities WHERE contact_id=? AND completed_at IS NOT NULL",
            (contact_id,)
        ).fetchone()
        score += (row[0] or 0) * 10

        # +5 for email_opened activities
        row = cur.execute(
            "SELECT COUNT(*) FROM activities WHERE contact_id=? AND activity_type='email_opened'",
            (contact_id,)
        ).fetchone()
        score += (row[0] or 0) * 5

        # Deal-based scoring
        deals = cur.execute(
            "SELECT stage FROM deals WHERE contact_id=?",
            (contact_id,)
        ).fetchall()
        for d in deals:
            stage = d["stage"]
            if stage in ("proposal", "negotiation"):
                score += 20
            elif stage == "closed_won":
                score += 50

        cur.execute(
            "UPDATE contacts SET lead_score=?, updated_at=? WHERE id=?",
            (score, now, contact_id)
        )
        conn.commit()
        conn.close()
        return score
    except Exception as exc:
        log_err(f"lead score calc failed for contact {contact_id}: {exc}")
        return 0


# ---------------------------------------------------------------------------
# SendGrid helper
# ---------------------------------------------------------------------------
def send_email(to_email: str, to_name: str, subject: str, body: str) -> bool:
    """Send a plain-text email via SendGrid API v3."""
    if not SENDGRID_API_KEY or not FROM_EMAIL:
        log_warn("SendGrid not configured — skipping email send")
        return False
    payload = json.dumps({
        "personalizations": [{"to": [{"email": to_email, "name": to_name}]}],
        "from": {"email": FROM_EMAIL},
        "subject": subject,
        "content": [{"type": "text/plain", "value": body}],
    }).encode()
    req = urllib.request.Request(
        "https://api.sendgrid.com/v3/mail/send",
        data=payload,
        headers={
            "Authorization": f"Bearer {SENDGRID_API_KEY}",
            "Content-Type":  "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            ok = resp.status in (200, 202)
            log_info(f"Email sent to {to_email}: status={resp.status}")
            return ok
    except urllib.error.HTTPError as exc:
        log_err(f"SendGrid HTTP {exc.code} sending to {to_email}: {exc.read()[:200]}")
        return False
    except Exception as exc:
        log_err(f"SendGrid error sending to {to_email}: {exc}")
        return False


# ---------------------------------------------------------------------------
# Drip campaign background thread
# ---------------------------------------------------------------------------
def _render_template(template: str, contact: dict) -> str:
    """Substitute {first_name}, {last_name}, {email}, {company} in template."""
    for key in ("first_name", "last_name", "email", "company"):
        template = template.replace("{" + key + "}", str(contact.get(key) or ""))
    return template


def _drip_tick() -> None:
    """Process one tick of drip campaign sends."""
    now = time.time()
    try:
        conn = _db_connect()
        cur  = conn.cursor()

        enrollments = cur.execute(
            """SELECT e.id, e.sequence_id, e.contact_id, e.current_step
               FROM sequence_enrollments e
               WHERE e.status='active' AND e.next_send_at <= ?""",
            (now,)
        ).fetchall()

        for enrollment in enrollments:
            eid        = enrollment["id"]
            seq_id     = enrollment["sequence_id"]
            contact_id = enrollment["contact_id"]
            step_idx   = enrollment["current_step"]

            # Fetch step
            step = cur.execute(
                """SELECT * FROM sequence_steps
                   WHERE sequence_id=? AND step_order=?""",
                (seq_id, step_idx)
            ).fetchone()

            if step is None:
                # No more steps — mark complete
                cur.execute(
                    "UPDATE sequence_enrollments SET status='completed' WHERE id=?",
                    (eid,)
                )
                conn.commit()
                continue

            # Fetch contact
            contact = cur.execute(
                "SELECT * FROM contacts WHERE id=?",
                (contact_id,)
            ).fetchone()
            if contact is None:
                cur.execute(
                    "UPDATE sequence_enrollments SET status='cancelled' WHERE id=?",
                    (eid,)
                )
                conn.commit()
                continue

            contact_dict = dict(contact)
            subject      = step["subject"] or ""
            body         = _render_template(step["body_template"] or "", contact_dict)
            to_name      = f"{contact_dict.get('first_name','')} {contact_dict.get('last_name','')}".strip()

            sent = send_email(contact_dict["email"], to_name, subject, body)

            # Log activity
            cur.execute(
                """INSERT INTO activities
                   (contact_id, deal_id, activity_type, subject, body, outcome, completed_at, created_at)
                   VALUES (?, NULL, 'email_sent', ?, ?, ?, ?, ?)""",
                (
                    contact_id,
                    subject,
                    body[:500],
                    "sent" if sent else "failed",
                    now,
                    now,
                )
            )

            # Advance enrollment
            next_step = cur.execute(
                """SELECT delay_hours FROM sequence_steps
                   WHERE sequence_id=? AND step_order=?""",
                (seq_id, step_idx + 1)
            ).fetchone()

            if next_step is not None:
                next_send = now + next_step["delay_hours"] * 3600
                cur.execute(
                    """UPDATE sequence_enrollments
                       SET current_step=?, next_send_at=?
                       WHERE id=?""",
                    (step_idx + 1, next_send, eid)
                )
            else:
                cur.execute(
                    "UPDATE sequence_enrollments SET status='completed', current_step=? WHERE id=?",
                    (step_idx + 1, eid)
                )

            conn.commit()

        conn.close()

    except Exception as exc:
        log_err(f"Drip tick error: {exc}")


def _drip_loop() -> None:
    """Daemon thread: run drip campaign every 300 seconds."""
    log_info("Drip campaign thread started (interval=300s)")
    while True:
        try:
            _drip_tick()
        except Exception as exc:
            log_err(f"Drip loop unhandled: {exc}")
        time.sleep(300)


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------
def _is_admin(headers) -> bool:
    if not ADMIN_SECRET:
        return True
    provided = headers.get("X-Admin-Secret", "") or headers.get("x-admin-secret", "")
    return hmac.compare_digest(provided, ADMIN_SECRET)


def _is_mcp(headers) -> bool:
    if not MCP_SECRET:
        return True
    return headers.get("X-MCP-Secret", "") == MCP_SECRET or \
           headers.get("x-mcp-secret", "") == MCP_SECRET


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def _json_response(handler: "CRMHandler", code: int, data) -> None:
    body = json.dumps(data, default=str).encode()
    handler.send_response(code)
    handler.send_header("Content-Type",   "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_body(handler: "CRMHandler") -> dict:
    length = int(handler.headers.get("Content-Length", 0))
    if length == 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _parse_qs(path: str) -> tuple[str, dict]:
    """Return (path_without_qs, params_dict)."""
    if "?" not in path:
        return path, {}
    p, q = path.split("?", 1)
    params: dict = {}
    for part in q.split("&"):
        if "=" in part:
            k, v = part.split("=", 1)
            params[urllib.parse.unquote_plus(k)] = urllib.parse.unquote_plus(v)
        else:
            params[urllib.parse.unquote_plus(part)] = ""
    return p, params


# need urllib.parse for unquote_plus
import urllib.parse


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------
class CRMHandler(BaseHTTPRequestHandler):
    server_version = "FractalMesh-CRM/1.0"

    def log_message(self, fmt, *args):
        log_info(f"{self.client_address[0]} - {fmt % args}")

    # ------------------------------------------------------------------ routing
    def do_GET(self):
        path, params = _parse_qs(self.path)
        parts = [p for p in path.strip("/").split("/") if p]

        try:
            if not parts or parts == [""]:
                _json_response(self, 200, {"service": "CRM Engine", "status": "ok"})

            elif parts[0] == "health":
                self._health()

            elif parts[0] == "contacts":
                if len(parts) == 1:
                    self._list_contacts(params)
                elif len(parts) == 2 and parts[1] == "search":
                    self._search_contacts(params)
                elif len(parts) == 2:
                    self._get_contact(parts[1])
                else:
                    _json_response(self, 404, {"error": "not found"})

            elif parts[0] == "companies":
                if len(parts) == 1:
                    self._list_companies(params)
                elif len(parts) == 2:
                    self._get_company(parts[1])
                else:
                    _json_response(self, 404, {"error": "not found"})

            elif parts[0] == "deals":
                if len(parts) == 1:
                    self._list_deals(params)
                elif len(parts) == 2:
                    self._get_deal(parts[1])
                else:
                    _json_response(self, 404, {"error": "not found"})

            elif parts[0] == "pipeline":
                self._pipeline()

            elif parts[0] == "activities":
                self._list_activities(params)

            elif parts[0] == "sequences":
                if len(parts) == 1:
                    self._list_sequences()
                elif len(parts) == 2:
                    self._get_sequence(parts[1])
                else:
                    _json_response(self, 404, {"error": "not found"})

            else:
                _json_response(self, 404, {"error": "not found"})

        except Exception as exc:
            log_err(f"GET {path}: {exc}")
            _json_response(self, 500, {"error": str(exc)})

    def do_POST(self):
        path, params = _parse_qs(self.path)
        parts = [p for p in path.strip("/").split("/") if p]
        body  = _read_body(self)

        try:
            if parts[0] == "contacts":
                if len(parts) == 1:
                    self._create_contact(body)
                elif len(parts) == 3 and parts[2] == "activities":
                    self._log_activity(parts[1], body)
                else:
                    _json_response(self, 404, {"error": "not found"})

            elif parts[0] == "companies" and len(parts) == 1:
                self._create_company(body)

            elif parts[0] == "deals":
                if len(parts) == 1:
                    self._create_deal(body)
                else:
                    _json_response(self, 404, {"error": "not found"})

            elif parts[0] == "sequences":
                if len(parts) == 1:
                    if not _is_admin(self.headers):
                        _json_response(self, 403, {"error": "admin required"})
                        return
                    self._create_sequence(body)
                elif len(parts) == 3 and parts[2] == "enroll":
                    self._enroll_contact(parts[1], body)
                elif len(parts) == 3 and parts[2] == "unenroll":
                    if not _is_admin(self.headers):
                        _json_response(self, 403, {"error": "admin required"})
                        return
                    self._unenroll_contact(parts[1], body)
                else:
                    _json_response(self, 404, {"error": "not found"})

            else:
                _json_response(self, 404, {"error": "not found"})

        except Exception as exc:
            log_err(f"POST {path}: {exc}")
            _json_response(self, 500, {"error": str(exc)})

    def do_PUT(self):
        path, params = _parse_qs(self.path)
        parts = [p for p in path.strip("/").split("/") if p]
        body  = _read_body(self)

        try:
            if parts[0] == "contacts" and len(parts) == 2:
                self._update_contact(parts[1], body)
            elif parts[0] == "deals" and len(parts) == 2:
                self._update_deal(parts[1], body)
            else:
                _json_response(self, 404, {"error": "not found"})

        except Exception as exc:
            log_err(f"PUT {path}: {exc}")
            _json_response(self, 500, {"error": str(exc)})

    # ------------------------------------------------------------------ GET /health
    def _health(self):
        try:
            conn = _db_connect()
            cur  = conn.cursor()
            contacts  = cur.execute("SELECT COUNT(*) FROM contacts").fetchone()[0]
            deals     = cur.execute("SELECT COUNT(*) FROM deals").fetchone()[0]
            companies = cur.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
            conn.close()
        except Exception:
            contacts = deals = companies = 0

        _json_response(self, 200, {
            "status":          "ok",
            "uptime_seconds":  round(time.time() - START_TIME, 1),
            "port":            PORT,
            "contacts":        contacts,
            "deals":           deals,
            "companies":       companies,
        })

    # ------------------------------------------------------------------ GET /contacts
    def _list_contacts(self, params: dict):
        conn  = _db_connect()
        cur   = conn.cursor()

        clauses: list[str] = []
        args:    list      = []

        if params.get("lifecycle_stage"):
            clauses.append("lifecycle_stage=?")
            args.append(params["lifecycle_stage"])
        if params.get("company"):
            clauses.append("company LIKE ?")
            args.append(f"%{params['company']}%")
        if params.get("tags"):
            clauses.append("tags LIKE ?")
            args.append(f"%{params['tags']}%")
        if params.get("min_lead_score"):
            clauses.append("lead_score >= ?")
            args.append(int(params["min_lead_score"]))

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        limit = min(int(params.get("limit", 100)), 500)

        rows = cur.execute(
            f"SELECT * FROM contacts {where} ORDER BY updated_at DESC LIMIT ?",
            args + [limit]
        ).fetchall()
        conn.close()
        _json_response(self, 200, {"contacts": [dict(r) for r in rows], "count": len(rows)})

    # ------------------------------------------------------------------ GET /contacts/search
    def _search_contacts(self, params: dict):
        q = params.get("q", "").strip()
        if not q:
            _json_response(self, 400, {"error": "query param 'q' required"})
            return
        like = f"%{q}%"
        conn = _db_connect()
        cur  = conn.cursor()
        rows = cur.execute(
            """SELECT * FROM contacts
               WHERE email LIKE ? OR first_name LIKE ? OR last_name LIKE ? OR company LIKE ?
               ORDER BY lead_score DESC LIMIT 100""",
            (like, like, like, like)
        ).fetchall()
        conn.close()
        _json_response(self, 200, {"contacts": [dict(r) for r in rows], "count": len(rows)})

    # ------------------------------------------------------------------ GET /contacts/{id}
    def _get_contact(self, contact_id: str):
        try:
            cid = int(contact_id)
        except ValueError:
            _json_response(self, 400, {"error": "invalid contact id"})
            return

        conn = _db_connect()
        cur  = conn.cursor()
        contact = cur.execute("SELECT * FROM contacts WHERE id=?", (cid,)).fetchone()
        if not contact:
            conn.close()
            _json_response(self, 404, {"error": "contact not found"})
            return

        deals = cur.execute(
            "SELECT * FROM deals WHERE contact_id=? ORDER BY updated_at DESC",
            (cid,)
        ).fetchall()
        activities = cur.execute(
            """SELECT * FROM activities WHERE contact_id=?
               ORDER BY created_at DESC LIMIT 20""",
            (cid,)
        ).fetchall()
        conn.close()

        result = dict(contact)
        result["deals"]      = [dict(d) for d in deals]
        result["activities"] = [dict(a) for a in activities]
        _json_response(self, 200, result)

    # ------------------------------------------------------------------ POST /contacts
    def _create_contact(self, body: dict):
        email = (body.get("email") or "").strip().lower()
        if not email:
            _json_response(self, 400, {"error": "email required"})
            return

        now = time.time()
        try:
            conn = _db_connect()
            cur  = conn.cursor()
            cur.execute(
                """INSERT INTO contacts
                   (email, first_name, last_name, company, phone, title, source,
                    lifecycle_stage, lead_score, tags, notes, created_at, updated_at, last_activity_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    email,
                    body.get("first_name", ""),
                    body.get("last_name",  ""),
                    body.get("company",    ""),
                    body.get("phone",      ""),
                    body.get("title",      ""),
                    body.get("source",     ""),
                    body.get("lifecycle_stage", "lead"),
                    0,
                    body.get("tags",  ""),
                    body.get("notes", ""),
                    now, now, now,
                )
            )
            cid = cur.lastrowid
            conn.commit()
            conn.close()
            _json_response(self, 201, {"id": cid, "email": email, "created": True})
        except sqlite3.IntegrityError:
            _json_response(self, 409, {"error": "contact with that email already exists"})

    # ------------------------------------------------------------------ PUT /contacts/{id}
    def _update_contact(self, contact_id: str, body: dict):
        try:
            cid = int(contact_id)
        except ValueError:
            _json_response(self, 400, {"error": "invalid contact id"})
            return

        allowed = {
            "first_name", "last_name", "company", "phone", "title", "source",
            "lifecycle_stage", "tags", "notes", "email",
        }
        updates = {k: v for k, v in body.items() if k in allowed}
        if not updates:
            _json_response(self, 400, {"error": "no updatable fields provided"})
            return

        now = time.time()
        updates["updated_at"] = now

        set_clause = ", ".join(f"{k}=?" for k in updates)
        vals       = list(updates.values()) + [cid]

        conn = _db_connect()
        cur  = conn.cursor()
        cur.execute(f"UPDATE contacts SET {set_clause} WHERE id=?", vals)
        if cur.rowcount == 0:
            conn.close()
            _json_response(self, 404, {"error": "contact not found"})
            return
        conn.commit()
        conn.close()

        score = recalculate_lead_score(cid)
        _json_response(self, 200, {"id": cid, "updated": True, "lead_score": score})

    # ------------------------------------------------------------------ POST /contacts/{id}/activities
    def _log_activity(self, contact_id: str, body: dict):
        try:
            cid = int(contact_id)
        except ValueError:
            _json_response(self, 400, {"error": "invalid contact id"})
            return

        now  = time.time()
        conn = _db_connect()
        cur  = conn.cursor()

        contact = cur.execute("SELECT id FROM contacts WHERE id=?", (cid,)).fetchone()
        if not contact:
            conn.close()
            _json_response(self, 404, {"error": "contact not found"})
            return

        completed_at = body.get("completed_at")
        if completed_at is not None:
            try:
                completed_at = float(completed_at)
            except (TypeError, ValueError):
                completed_at = now

        cur.execute(
            """INSERT INTO activities
               (contact_id, deal_id, activity_type, subject, body, outcome,
                scheduled_at, completed_at, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                cid,
                body.get("deal_id"),
                body.get("activity_type", "note"),
                body.get("subject", ""),
                body.get("body", ""),
                body.get("outcome", ""),
                body.get("scheduled_at"),
                completed_at,
                now,
            )
        )
        aid = cur.lastrowid

        # update last_activity_at on contact
        cur.execute(
            "UPDATE contacts SET last_activity_at=?, updated_at=? WHERE id=?",
            (now, now, cid)
        )
        conn.commit()
        conn.close()

        score = recalculate_lead_score(cid)
        _json_response(self, 201, {"id": aid, "contact_id": cid, "created": True, "lead_score": score})

    # ------------------------------------------------------------------ GET /companies
    def _list_companies(self, params: dict):
        conn  = _db_connect()
        cur   = conn.cursor()
        limit = min(int(params.get("limit", 100)), 500)
        rows  = cur.execute(
            "SELECT * FROM companies ORDER BY updated_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
        conn.close()
        _json_response(self, 200, {"companies": [dict(r) for r in rows], "count": len(rows)})

    # ------------------------------------------------------------------ GET /companies/{id}
    def _get_company(self, company_id: str):
        try:
            coid = int(company_id)
        except ValueError:
            _json_response(self, 400, {"error": "invalid company id"})
            return

        conn    = _db_connect()
        cur     = conn.cursor()
        company = cur.execute("SELECT * FROM companies WHERE id=?", (coid,)).fetchone()
        if not company:
            conn.close()
            _json_response(self, 404, {"error": "company not found"})
            return

        contacts = cur.execute(
            "SELECT * FROM contacts WHERE company=? ORDER BY lead_score DESC",
            (company["name"],)
        ).fetchall()
        deals = cur.execute(
            "SELECT * FROM deals WHERE company_id=? ORDER BY updated_at DESC",
            (coid,)
        ).fetchall()
        conn.close()

        result = dict(company)
        result["contacts"] = [dict(c) for c in contacts]
        result["deals"]    = [dict(d) for d in deals]
        _json_response(self, 200, result)

    # ------------------------------------------------------------------ POST /companies
    def _create_company(self, body: dict):
        name = (body.get("name") or "").strip()
        if not name:
            _json_response(self, 400, {"error": "company name required"})
            return

        now = time.time()
        try:
            conn = _db_connect()
            cur  = conn.cursor()
            cur.execute(
                """INSERT INTO companies
                   (name, domain, industry, size, annual_revenue, country, city, notes,
                    created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    name,
                    body.get("domain",         ""),
                    body.get("industry",        ""),
                    body.get("size",            ""),
                    body.get("annual_revenue",  0),
                    body.get("country",         ""),
                    body.get("city",            ""),
                    body.get("notes",           ""),
                    now, now,
                )
            )
            coid = cur.lastrowid
            conn.commit()
            conn.close()
            _json_response(self, 201, {"id": coid, "name": name, "created": True})
        except sqlite3.IntegrityError:
            _json_response(self, 409, {"error": "company with that name already exists"})

    # ------------------------------------------------------------------ GET /deals
    def _list_deals(self, params: dict):
        conn     = _db_connect()
        cur      = conn.cursor()
        clauses: list[str] = []
        args:    list      = []

        if params.get("stage"):
            clauses.append("stage=?")
            args.append(params["stage"])
        if params.get("contact_id"):
            clauses.append("contact_id=?")
            args.append(int(params["contact_id"]))
        if params.get("min_value"):
            clauses.append("value >= ?")
            args.append(float(params["min_value"]))

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        limit = min(int(params.get("limit", 100)), 500)

        rows = cur.execute(
            f"SELECT * FROM deals {where} ORDER BY updated_at DESC LIMIT ?",
            args + [limit]
        ).fetchall()
        conn.close()
        _json_response(self, 200, {"deals": [dict(r) for r in rows], "count": len(rows)})

    # ------------------------------------------------------------------ GET /deals/{id}
    def _get_deal(self, deal_id: str):
        try:
            did = int(deal_id)
        except ValueError:
            _json_response(self, 400, {"error": "invalid deal id"})
            return

        conn = _db_connect()
        cur  = conn.cursor()
        deal = cur.execute("SELECT * FROM deals WHERE id=?", (did,)).fetchone()
        if not deal:
            conn.close()
            _json_response(self, 404, {"error": "deal not found"})
            return

        activities = cur.execute(
            """SELECT * FROM activities WHERE deal_id=?
               ORDER BY created_at DESC LIMIT 20""",
            (did,)
        ).fetchall()
        conn.close()

        result = dict(deal)
        result["activities"] = [dict(a) for a in activities]
        _json_response(self, 200, result)

    # ------------------------------------------------------------------ POST /deals
    def _create_deal(self, body: dict):
        title = (body.get("title") or "").strip()
        if not title:
            _json_response(self, 400, {"error": "deal title required"})
            return

        stage = body.get("stage", "prospecting")
        if stage not in DEAL_STAGES:
            _json_response(self, 400, {"error": f"invalid stage; must be one of {DEAL_STAGES}"})
            return

        # Default probability by stage
        prob_defaults = {
            "prospecting":   0.10,
            "qualification": 0.20,
            "proposal":      0.40,
            "negotiation":   0.60,
            "closed_won":    1.00,
            "closed_lost":   0.00,
        }

        now = time.time()
        conn = _db_connect()
        cur  = conn.cursor()
        cur.execute(
            """INSERT INTO deals
               (title, contact_id, company_id, stage, value, currency, probability,
                expected_close, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                title,
                body.get("contact_id"),
                body.get("company_id"),
                stage,
                float(body.get("value", 0)),
                body.get("currency", "AUD"),
                float(body.get("probability", prob_defaults[stage])),
                body.get("expected_close"),
                now, now,
            )
        )
        did = cur.lastrowid
        conn.commit()
        conn.close()

        # Rescore contact if linked
        if body.get("contact_id"):
            recalculate_lead_score(int(body["contact_id"]))

        _json_response(self, 201, {"id": did, "title": title, "created": True})

    # ------------------------------------------------------------------ PUT /deals/{id}
    def _update_deal(self, deal_id: str, body: dict):
        try:
            did = int(deal_id)
        except ValueError:
            _json_response(self, 400, {"error": "invalid deal id"})
            return

        now  = time.time()
        conn = _db_connect()
        cur  = conn.cursor()

        deal = cur.execute("SELECT * FROM deals WHERE id=?", (did,)).fetchone()
        if not deal:
            conn.close()
            _json_response(self, 404, {"error": "deal not found"})
            return

        new_stage = body.get("stage", deal["stage"])
        if new_stage not in DEAL_STAGES:
            conn.close()
            _json_response(self, 400, {"error": f"invalid stage; must be one of {DEAL_STAGES}"})
            return

        # Admin gate for won/lost
        if new_stage in ("closed_won", "closed_lost") and new_stage != deal["stage"]:
            if not _is_admin(self.headers):
                conn.close()
                _json_response(self, 403, {"error": "admin required to close deals"})
                return

        won_at   = deal["won_at"]
        lost_at  = deal["lost_at"]
        lost_reason = body.get("lost_reason", deal["lost_reason"])

        if new_stage == "closed_won" and deal["stage"] != "closed_won":
            won_at = now
        if new_stage == "closed_lost" and deal["stage"] != "closed_lost":
            lost_at = now

        cur.execute(
            """UPDATE deals
               SET stage=?, value=?, probability=?, expected_close=?,
                   won_at=?, lost_at=?, lost_reason=?, updated_at=?
               WHERE id=?""",
            (
                new_stage,
                float(body.get("value", deal["value"])),
                float(body.get("probability", deal["probability"])),
                body.get("expected_close", deal["expected_close"]),
                won_at,
                lost_at,
                lost_reason,
                now,
                did,
            )
        )
        conn.commit()
        contact_id = deal["contact_id"]
        conn.close()

        if contact_id:
            recalculate_lead_score(int(contact_id))

        _json_response(self, 200, {"id": did, "updated": True, "stage": new_stage})

    # ------------------------------------------------------------------ GET /pipeline
    def _pipeline(self):
        conn = _db_connect()
        cur  = conn.cursor()
        rows = cur.execute(
            """SELECT stage, COUNT(*) AS count, COALESCE(SUM(value),0) AS total_value,
                      COALESCE(AVG(probability),0) AS avg_probability
               FROM deals
               GROUP BY stage"""
        ).fetchall()
        conn.close()

        # Build full stage map including empty stages
        stage_map = {s: {"stage": s, "count": 0, "total_value": 0.0, "avg_probability": 0.0}
                     for s in DEAL_STAGES}
        for r in rows:
            stage_map[r["stage"]] = {
                "stage":           r["stage"],
                "count":           r["count"],
                "total_value":     round(r["total_value"], 2),
                "avg_probability": round(r["avg_probability"], 4),
            }

        _json_response(self, 200, {
            "pipeline": [stage_map[s] for s in DEAL_STAGES],
        })

    # ------------------------------------------------------------------ GET /activities
    def _list_activities(self, params: dict):
        conn     = _db_connect()
        cur      = conn.cursor()
        clauses: list[str] = []
        args:    list      = []

        if params.get("contact_id"):
            clauses.append("contact_id=?")
            args.append(int(params["contact_id"]))
        if params.get("activity_type"):
            clauses.append("activity_type=?")
            args.append(params["activity_type"])

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        limit = min(int(params.get("limit", 50)), 200)

        rows = cur.execute(
            f"SELECT * FROM activities {where} ORDER BY created_at DESC LIMIT ?",
            args + [limit]
        ).fetchall()
        conn.close()
        _json_response(self, 200, {"activities": [dict(r) for r in rows], "count": len(rows)})

    # ------------------------------------------------------------------ GET /sequences
    def _list_sequences(self):
        conn = _db_connect()
        cur  = conn.cursor()
        rows = cur.execute(
            """SELECT s.*, COUNT(e.id) AS enrollments
               FROM email_sequences s
               LEFT JOIN sequence_enrollments e ON e.sequence_id=s.id
               GROUP BY s.id
               ORDER BY s.created_at DESC"""
        ).fetchall()
        conn.close()
        _json_response(self, 200, {"sequences": [dict(r) for r in rows], "count": len(rows)})

    # ------------------------------------------------------------------ GET /sequences/{id}
    def _get_sequence(self, seq_id: str):
        try:
            sid = int(seq_id)
        except ValueError:
            _json_response(self, 400, {"error": "invalid sequence id"})
            return

        conn = _db_connect()
        cur  = conn.cursor()
        seq  = cur.execute("SELECT * FROM email_sequences WHERE id=?", (sid,)).fetchone()
        if not seq:
            conn.close()
            _json_response(self, 404, {"error": "sequence not found"})
            return

        steps = cur.execute(
            "SELECT * FROM sequence_steps WHERE sequence_id=? ORDER BY step_order",
            (sid,)
        ).fetchall()

        stats = cur.execute(
            """SELECT status, COUNT(*) AS count
               FROM sequence_enrollments WHERE sequence_id=?
               GROUP BY status""",
            (sid,)
        ).fetchall()

        conn.close()

        result = dict(seq)
        result["steps"]  = [dict(s) for s in steps]
        result["enrollment_stats"] = {r["status"]: r["count"] for r in stats}
        _json_response(self, 200, result)

    # ------------------------------------------------------------------ POST /sequences
    def _create_sequence(self, body: dict):
        name = (body.get("name") or "").strip()
        if not name:
            _json_response(self, 400, {"error": "sequence name required"})
            return

        steps_raw = body.get("steps", [])
        if not isinstance(steps_raw, list):
            _json_response(self, 400, {"error": "'steps' must be a list"})
            return

        now  = time.time()
        conn = _db_connect()
        cur  = conn.cursor()

        cur.execute(
            "INSERT INTO email_sequences (name, status, created_at) VALUES (?,?,?)",
            (name, body.get("status", "active"), now)
        )
        sid = cur.lastrowid

        for idx, step in enumerate(steps_raw):
            cur.execute(
                """INSERT INTO sequence_steps
                   (sequence_id, step_order, delay_hours, subject, body_template)
                   VALUES (?,?,?,?,?)""",
                (
                    sid,
                    step.get("step_order", idx),
                    int(step.get("delay_hours", 24)),
                    step.get("subject", ""),
                    step.get("body_template", ""),
                )
            )

        conn.commit()
        conn.close()
        _json_response(self, 201, {"id": sid, "name": name, "steps": len(steps_raw), "created": True})

    # ------------------------------------------------------------------ POST /sequences/{id}/enroll
    def _enroll_contact(self, seq_id: str, body: dict):
        try:
            sid = int(seq_id)
        except ValueError:
            _json_response(self, 400, {"error": "invalid sequence id"})
            return

        contact_id = body.get("contact_id")
        if contact_id is None:
            _json_response(self, 400, {"error": "contact_id required"})
            return

        try:
            contact_id = int(contact_id)
        except (TypeError, ValueError):
            _json_response(self, 400, {"error": "invalid contact_id"})
            return

        now  = time.time()
        conn = _db_connect()
        cur  = conn.cursor()

        seq = cur.execute("SELECT id FROM email_sequences WHERE id=?", (sid,)).fetchone()
        if not seq:
            conn.close()
            _json_response(self, 404, {"error": "sequence not found"})
            return

        contact = cur.execute("SELECT id FROM contacts WHERE id=?", (contact_id,)).fetchone()
        if not contact:
            conn.close()
            _json_response(self, 404, {"error": "contact not found"})
            return

        # Get the first step's delay to set next_send_at
        first_step = cur.execute(
            "SELECT delay_hours FROM sequence_steps WHERE sequence_id=? AND step_order=0",
            (sid,)
        ).fetchone()
        next_send = now + (first_step["delay_hours"] * 3600 if first_step else 0)

        # Check if already enrolled and active
        existing = cur.execute(
            """SELECT id FROM sequence_enrollments
               WHERE sequence_id=? AND contact_id=? AND status='active'""",
            (sid, contact_id)
        ).fetchone()
        if existing:
            conn.close()
            _json_response(self, 409, {"error": "contact already enrolled in this sequence"})
            return

        cur.execute(
            """INSERT INTO sequence_enrollments
               (sequence_id, contact_id, current_step, status, enrolled_at, next_send_at)
               VALUES (?,?,?,?,?,?)""",
            (sid, contact_id, 0, "active", now, next_send)
        )
        eid = cur.lastrowid
        conn.commit()
        conn.close()
        _json_response(self, 201, {"enrollment_id": eid, "contact_id": contact_id,
                                    "sequence_id": sid, "enrolled": True})

    # ------------------------------------------------------------------ POST /sequences/{id}/unenroll
    def _unenroll_contact(self, seq_id: str, body: dict):
        try:
            sid = int(seq_id)
        except ValueError:
            _json_response(self, 400, {"error": "invalid sequence id"})
            return

        contact_id = body.get("contact_id")
        if contact_id is None:
            _json_response(self, 400, {"error": "contact_id required"})
            return

        try:
            contact_id = int(contact_id)
        except (TypeError, ValueError):
            _json_response(self, 400, {"error": "invalid contact_id"})
            return

        conn = _db_connect()
        cur  = conn.cursor()
        cur.execute(
            """UPDATE sequence_enrollments
               SET status='cancelled'
               WHERE sequence_id=? AND contact_id=? AND status='active'""",
            (sid, contact_id)
        )
        affected = cur.rowcount
        conn.commit()
        conn.close()

        if affected == 0:
            _json_response(self, 404, {"error": "no active enrollment found"})
        else:
            _json_response(self, 200, {"unenrolled": True, "contact_id": contact_id,
                                        "sequence_id": sid})


# ---------------------------------------------------------------------------
# Server bootstrap
# ---------------------------------------------------------------------------
def main() -> None:
    db_init()

    # Start drip campaign background thread
    drip_thread = threading.Thread(target=_drip_loop, daemon=True, name="drip-campaign")
    drip_thread.start()

    server = HTTPServer(("0.0.0.0", PORT), CRMHandler)
    log_info(f"CRM Engine running on port {PORT}")
    log_info(f"Database: {DB_PATH}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log_info("Shutting down CRM Engine")
        server.server_close()


if __name__ == "__main__":
    main()
