#!/usr/bin/env python3
"""
FractalMesh OMEGA Titan — Advanced Lead Generation Pipeline (Q-ALSM v2)
Port: 7827
Samuel James Hiotis | ABN 56 628 117 363 | Sole Trader
All credentials sourced from ~/.secrets/fractal.env at runtime. Never hardcoded.
"""

import json
import logging
import os
import re
import signal
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ---------------------------------------------------------------------------
# Vault loading
# ---------------------------------------------------------------------------
_VAULT = Path.home() / ".secrets" / "fractal.env"
if _VAULT.exists():
    for _line in _VAULT.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PORT              = int(os.getenv("LEADGEN_PORT", "7827"))
ROOT              = Path(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")))
DB_PATH           = ROOT / "database" / "sovereign.db"
LOG_PATH          = ROOT / "logs" / "fm_leadgen.log"

GOOGLE_CSE_API_KEY   = os.getenv("GOOGLE_CSE_API_KEY", "")
GOOGLE_CSE_ID        = os.getenv("GOOGLE_CSE_ID", "")
SENDGRID_API_KEY     = os.getenv("SENDGRID_API_KEY", "")
SENDGRID_FROM_EMAIL  = os.getenv("SENDGRID_FROM_EMAIL", "")
CRAWLBASE_TOKEN      = os.getenv("CRAWLBASE_NORMAL_TOKEN", "")

for _d in (DB_PATH.parent, LOG_PATH.parent):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [LEADGEN] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("fm_leadgen")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS leads (
    id          INTEGER PRIMARY KEY,
    name        TEXT,
    email       TEXT,
    phone       TEXT,
    company     TEXT,
    location    TEXT,
    source      TEXT,
    score       REAL DEFAULT 0.0,
    status      TEXT DEFAULT 'new',
    tags        TEXT,
    notes       TEXT,
    created_at  REAL,
    updated_at  REAL
);

CREATE TABLE IF NOT EXISTS sequences (
    id           INTEGER PRIMARY KEY,
    lead_id      INTEGER,
    step         INTEGER,
    action       TEXT,
    scheduled_at REAL,
    executed_at  REAL,
    result       TEXT
);

CREATE TABLE IF NOT EXISTS campaigns (
    id               INTEGER PRIMARY KEY,
    name             TEXT,
    target_industry  TEXT,
    target_location  TEXT,
    keywords         TEXT,
    status           TEXT DEFAULT 'active',
    leads_found      INTEGER DEFAULT 0,
    created_at       REAL
);
"""


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    conn = get_db()
    conn.executescript(DDL)
    conn.commit()
    conn.close()
    log.info("Database initialised at %s", DB_PATH)


# ---------------------------------------------------------------------------
# Helpers: external APIs
# ---------------------------------------------------------------------------

def _google_cse(query: str, num: int = 10) -> list:
    """Query Google Custom Search Engine. Returns list of {link, title, snippet}."""
    if not GOOGLE_CSE_API_KEY or not GOOGLE_CSE_ID:
        log.warning("Google CSE credentials not set")
        return []
    params = urllib.parse.urlencode({
        "key": GOOGLE_CSE_API_KEY,
        "cx":  GOOGLE_CSE_ID,
        "q":   query,
        "num": min(num, 10),
    })
    url = f"https://customsearch.googleapis.com/customsearch/v1?{params}"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        items = data.get("items", [])
        return [
            {"link": it.get("link", ""), "title": it.get("title", ""), "snippet": it.get("snippet", "")}
            for it in items
        ]
    except Exception as exc:
        log.error("Google CSE error: %s", exc)
        return []


def _crawlbase_fetch(url: str) -> str:
    """Fetch a URL via Crawlbase normal token. Returns body text or empty string."""
    if not CRAWLBASE_TOKEN:
        log.warning("CRAWLBASE_NORMAL_TOKEN not set")
        return ""
    params = urllib.parse.urlencode({"token": CRAWLBASE_TOKEN, "url": url})
    cb_url = f"https://api.crawlbase.com/?{params}"
    try:
        req = urllib.request.Request(cb_url, headers={"Accept": "text/html"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
        try:
            return raw.decode("utf-8", errors="replace")
        except Exception:
            return raw.decode("latin-1", errors="replace")
    except Exception as exc:
        log.error("Crawlbase fetch error for %s: %s", url, exc)
        return ""


def _extract_contacts(html_text: str) -> dict:
    """Extract emails and phones from raw HTML / text.
    Returns {"emails": [...], "phones": [...]}
    """
    # Emails
    email_re = re.compile(r"[\w.+\-]+@[\w\-]+\.[a-zA-Z]{2,}", re.IGNORECASE)
    raw_emails = email_re.findall(html_text)
    # Filter obvious false positives (image filenames, schemeless paths, etc.)
    skip_exts = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".css", ".js", ".woff"}
    emails = []
    seen_e = set()
    for e in raw_emails:
        el = e.lower()
        if any(el.endswith(x) for x in skip_exts):
            continue
        if el in seen_e:
            continue
        seen_e.add(el)
        emails.append(e.lower())

    # Phones  (7-20 chars, must contain at least 7 digits)
    phone_re = re.compile(r"\+?[\d\s\-(). ]{7,20}")
    raw_phones = phone_re.findall(html_text)
    phones = []
    seen_p = set()
    for p in raw_phones:
        digits = re.sub(r"\D", "", p)
        if len(digits) < 7:
            continue
        norm = p.strip()
        if norm in seen_p:
            continue
        seen_p.add(norm)
        phones.append(norm)

    return {"emails": emails[:20], "phones": phones[:20]}


def _send_email(to_email: str, to_name: str, company: str, step_type: str) -> str:
    """Send an email via SendGrid v3 Mail Send API.
    Returns the SendGrid message-id header or raises on failure.
    """
    if not SENDGRID_API_KEY or not SENDGRID_FROM_EMAIL:
        raise RuntimeError("SendGrid credentials not configured")

    company_str = company or "your business"
    name_str    = to_name or "there"

    if step_type in ("email", "cold_outreach"):
        subject = f"Quick question about {company_str}"
        body    = (
            f"Hi {name_str},\n\n"
            f"I came across {company_str} and wanted to reach out — "
            f"we help businesses like yours grow faster with smarter systems.\n\n"
            f"Would you be open to a quick 15-minute call this week?\n\n"
            f"Best,\nSamuel James Hiotis\n"
            f"FractalMesh | fractalmesh.net\nABN 56 628 117 363"
        )
    elif step_type == "follow_up":
        subject = f"Following up — {company_str}"
        body    = (
            f"Hi {name_str},\n\n"
            f"Just following up on my earlier note about {company_str}.\n\n"
            f"Happy to keep it brief — even 10 minutes could be valuable.\n\n"
            f"Let me know if you'd like to connect.\n\n"
            f"Best,\nSamuel\nFractalMesh | fractalmesh.net"
        )
    elif step_type == "final":
        subject = f"Last note — {company_str}"
        body    = (
            f"Hi {name_str},\n\n"
            f"This will be my last reach-out regarding {company_str}.\n\n"
            f"If the timing isn't right, no worries at all — "
            f"feel free to get in touch whenever it makes sense.\n\n"
            f"Wishing {company_str} continued success.\n\n"
            f"Best,\nSamuel\nFractalMesh | fractalmesh.net"
        )
    elif step_type == "value_email":
        subject = f"One thing that could help {company_str}"
        body    = (
            f"Hi {name_str},\n\n"
            f"Wanted to share something that's been moving the needle for "
            f"businesses similar to {company_str}.\n\n"
            f"[Short insight or value-add relevant to their industry]\n\n"
            f"Happy to walk you through it — just reply and we'll set something up.\n\n"
            f"Best,\nSamuel\nFractalMesh | fractalmesh.net"
        )
    elif step_type == "case_study":
        subject = f"How we helped a business like {company_str}"
        body    = (
            f"Hi {name_str},\n\n"
            f"Thought you'd find this relevant — we recently worked with a client "
            f"in a similar space and helped them streamline operations significantly.\n\n"
            f"Could share the full case study if you're interested.\n\n"
            f"Best,\nSamuel\nFractalMesh | fractalmesh.net"
        )
    elif step_type == "offer":
        subject = f"Special offer for {company_str}"
        body    = (
            f"Hi {name_str},\n\n"
            f"We're currently offering new clients a complimentary infrastructure audit "
            f"— no strings attached.\n\n"
            f"It's a 30-minute session where we map out quick wins for {company_str}.\n\n"
            f"Reply to claim your spot.\n\n"
            f"Best,\nSamuel\nFractalMesh | fractalmesh.net"
        )
    elif step_type == "close":
        subject = f"Ready to move forward, {name_str}?"
        body    = (
            f"Hi {name_str},\n\n"
            f"I've enjoyed staying in touch. Whenever {company_str} is ready to explore "
            f"what we can do together, I'm here.\n\n"
            f"Book a call anytime: fractalmesh.net/contact\n\n"
            f"Best,\nSamuel\nFractalMesh | fractalmesh.net"
        )
    else:
        subject = f"Touching base — {company_str}"
        body    = (
            f"Hi {name_str},\n\nJust checking in.\n\n"
            f"Best,\nSamuel\nFractalMesh | fractalmesh.net"
        )

    payload = json.dumps({
        "personalizations": [{"to": [{"email": to_email, "name": to_name or ""}]}],
        "from": {"email": SENDGRID_FROM_EMAIL, "name": "Samuel Hiotis | FractalMesh"},
        "subject": subject,
        "content": [{"type": "text/plain", "value": body}],
    }).encode("utf-8")

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
        with urllib.request.urlopen(req, timeout=20) as resp:
            msg_id = resp.headers.get("X-Message-Id", "sent")
            return msg_id
    except urllib.error.HTTPError as exc:
        body_err = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"SendGrid HTTP {exc.code}: {body_err}") from exc


# ---------------------------------------------------------------------------
# Score helper
# ---------------------------------------------------------------------------

def _calculate_score(lead: dict) -> float:
    score = 0.0
    if lead.get("email"):
        score += 0.3
    if lead.get("phone"):
        score += 0.2
    if lead.get("company"):
        score += 0.15
    if lead.get("location"):
        score += 0.1
    status = (lead.get("status") or "").lower()
    if status == "contacted":
        score += 0.1
    elif status == "responded":
        score += 0.2
    return round(min(score, 1.0), 4)


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------

class LeadGenHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):  # suppress default access log spam
        log.debug("HTTP %s", fmt % args)

    # ------------------------------------------------------------------ utils

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def _send_json(self, data, status: int = 200):
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _not_found(self):
        self._send_json({"error": "not found"}, 404)

    def _bad_request(self, msg: str = "bad request"):
        self._send_json({"error": msg}, 400)

    # ------------------------------------------------------------------ routing

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")

        if path == "/health":
            self._send_json({"status": "ok", "service": "fm-leadgen", "port": 7827})

        elif path == "/analytics":
            self._handle_analytics()

        elif path == "/campaigns":
            self._handle_list_campaigns()

        elif re.match(r"^/leads/\d+$", path):
            lead_id = int(path.split("/")[-1])
            self._handle_get_lead(lead_id)

        elif re.match(r"^/sequences/\d+$", path):
            lead_id = int(path.split("/")[-1])
            self._handle_get_sequences(lead_id)

        else:
            self._not_found()

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        body = self._read_body()

        if path == "/campaign/create":
            self._handle_campaign_create(body)

        elif re.match(r"^/campaign/\d+/run$", path):
            campaign_id = int(path.split("/")[-2])
            self._handle_campaign_run(campaign_id, body)

        elif path == "/leads/search":
            self._handle_leads_search(body)

        elif path == "/leads/bulk_import":
            self._handle_bulk_import(body)

        elif path == "/leads/score":
            self._handle_score(body)

        elif path == "/sequence/start":
            self._handle_sequence_start(body)

        elif path == "/sequence/execute_due":
            self._handle_execute_due()

        else:
            self._not_found()

    def do_PUT(self):
        path = self.path.split("?")[0].rstrip("/")
        body = self._read_body()

        if re.match(r"^/leads/\d+$", path):
            lead_id = int(path.split("/")[-1])
            self._handle_update_lead(lead_id, body)
        else:
            self._not_found()

    def do_DELETE(self):
        path = self.path.split("?")[0].rstrip("/")

        if re.match(r"^/leads/\d+$", path):
            lead_id = int(path.split("/")[-1])
            self._handle_delete_lead(lead_id)
        else:
            self._not_found()

    # ------------------------------------------------------------------ handlers

    def _handle_campaign_create(self, body: dict):
        name     = body.get("name", "")
        industry = body.get("industry", "")
        location = body.get("location", "")
        keywords = body.get("keywords", [])
        if not name:
            return self._bad_request("name required")
        kw_str = json.dumps(keywords)
        now    = time.time()
        conn   = get_db()
        try:
            cur = conn.execute(
                "INSERT INTO campaigns (name, target_industry, target_location, keywords, status, created_at) "
                "VALUES (?, ?, ?, ?, 'active', ?)",
                (name, industry, location, kw_str, now),
            )
            conn.commit()
            campaign_id = cur.lastrowid
        finally:
            conn.close()
        log.info("Campaign created id=%s name=%s", campaign_id, name)
        self._send_json({"campaign_id": campaign_id})

    def _handle_campaign_run(self, campaign_id: int, body: dict):
        max_leads = int(body.get("max_leads", 50))
        conn      = get_db()
        try:
            row = conn.execute("SELECT * FROM campaigns WHERE id=?", (campaign_id,)).fetchone()
            if not row:
                conn.close()
                return self._not_found()
            campaign   = dict(row)
            keywords   = json.loads(campaign.get("keywords") or "[]")
            location   = campaign.get("target_location", "")
        finally:
            conn.close()

        new_leads   = 0
        total_leads = 0
        source_tag  = f"campaign:{campaign_id}"

        for kw in keywords:
            if new_leads >= max_leads:
                break
            query   = f"{kw} {location} email contact"
            results = _google_cse(query, num=10)

            for item in results:
                if new_leads >= max_leads:
                    break
                url = item.get("link", "")
                if not url:
                    continue

                html = _crawlbase_fetch(url)
                if not html:
                    continue

                contacts = _extract_contacts(html)

                # Try to extract a basic name from title/snippet heuristics
                snippet = item.get("snippet", "")
                title   = item.get("title", "")
                company_guess = title.split("|")[0].split("-")[0].strip() if title else ""

                for email in contacts["emails"]:
                    phone = contacts["phones"][0] if contacts["phones"] else None
                    conn  = get_db()
                    try:
                        exists = conn.execute(
                            "SELECT id FROM leads WHERE email=?", (email,)
                        ).fetchone()
                        if exists:
                            total_leads += 1
                            continue
                        now = time.time()
                        score = _calculate_score({
                            "email": email, "phone": phone,
                            "company": company_guess, "location": location,
                        })
                        conn.execute(
                            "INSERT INTO leads (name, email, phone, company, location, source, score, status, created_at, updated_at) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, 'new', ?, ?)",
                            ("", email, phone, company_guess, location, source_tag, score, now, now),
                        )
                        conn.commit()
                        new_leads   += 1
                        total_leads += 1
                    finally:
                        conn.close()

        # Update campaign leads_found
        conn = get_db()
        try:
            conn.execute(
                "UPDATE campaigns SET leads_found = leads_found + ? WHERE id=?",
                (new_leads, campaign_id),
            )
            conn.commit()
        finally:
            conn.close()

        log.info("Campaign %s run complete: new=%s total=%s", campaign_id, new_leads, total_leads)
        self._send_json({"campaign_id": campaign_id, "new_leads": new_leads, "total_leads": total_leads})

    def _handle_leads_search(self, body: dict):
        query      = body.get("query", "")
        location   = body.get("location", "")
        min_score  = float(body.get("min_score", 0.0))
        status     = body.get("status", "")
        limit      = int(body.get("limit", 50))

        like = f"%{query}%"
        loc_like = f"%{location}%"

        sql    = "SELECT * FROM leads WHERE status != 'deleted'"
        params = []

        if query:
            sql    += " AND (name LIKE ? OR company LIKE ? OR location LIKE ? OR tags LIKE ?)"
            params += [like, like, like, like]
        if location:
            sql    += " AND location LIKE ?"
            params.append(loc_like)
        if min_score:
            sql    += " AND score >= ?"
            params.append(min_score)
        if status:
            sql    += " AND status = ?"
            params.append(status)

        sql    += " ORDER BY score DESC LIMIT ?"
        params.append(limit)

        conn = get_db()
        try:
            rows = conn.execute(sql, params).fetchall()
            leads = [dict(r) for r in rows]
        finally:
            conn.close()

        self._send_json(leads)

    def _handle_get_lead(self, lead_id: int):
        conn = get_db()
        try:
            row = conn.execute("SELECT * FROM leads WHERE id=?", (lead_id,)).fetchone()
        finally:
            conn.close()
        if not row:
            return self._not_found()
        self._send_json(dict(row))

    def _handle_update_lead(self, lead_id: int, body: dict):
        allowed = {"status", "notes", "tags", "name", "email", "phone", "company", "location", "score"}
        updates = {k: v for k, v in body.items() if k in allowed}
        if not updates:
            return self._bad_request("no valid fields")
        updates["updated_at"] = time.time()
        set_clause = ", ".join(f"{k}=?" for k in updates)
        values     = list(updates.values()) + [lead_id]
        conn = get_db()
        try:
            conn.execute(f"UPDATE leads SET {set_clause} WHERE id=?", values)
            conn.commit()
            row = conn.execute("SELECT * FROM leads WHERE id=?", (lead_id,)).fetchone()
        finally:
            conn.close()
        if not row:
            return self._not_found()
        self._send_json(dict(row))

    def _handle_delete_lead(self, lead_id: int):
        conn = get_db()
        try:
            conn.execute(
                "UPDATE leads SET status='deleted', updated_at=? WHERE id=?",
                (time.time(), lead_id),
            )
            conn.commit()
        finally:
            conn.close()
        self._send_json({"deleted": True, "lead_id": lead_id})

    def _handle_bulk_import(self, body: dict):
        leads_data = body.get("leads", [])
        imported   = 0
        skipped    = 0
        now        = time.time()
        conn       = get_db()
        try:
            for lead in leads_data:
                email = (lead.get("email") or "").strip().lower()
                if email:
                    exists = conn.execute("SELECT id FROM leads WHERE email=?", (email,)).fetchone()
                    if exists:
                        skipped += 1
                        continue
                score = _calculate_score(lead)
                conn.execute(
                    "INSERT INTO leads (name, email, phone, company, location, source, score, status, tags, notes, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, 'new', ?, ?, ?, ?)",
                    (
                        lead.get("name", ""),
                        email,
                        lead.get("phone", ""),
                        lead.get("company", ""),
                        lead.get("location", ""),
                        lead.get("source", "bulk_import"),
                        score,
                        lead.get("tags", ""),
                        lead.get("notes", ""),
                        now,
                        now,
                    ),
                )
                imported += 1
            conn.commit()
        finally:
            conn.close()
        log.info("Bulk import: imported=%s skipped=%s", imported, skipped)
        self._send_json({"imported": imported, "skipped": skipped})

    def _handle_sequence_start(self, body: dict):
        lead_id       = body.get("lead_id")
        sequence_type = body.get("sequence_type", "cold_outreach")
        if not lead_id:
            return self._bad_request("lead_id required")

        SEQUENCE_PLANS = {
            "cold_outreach": [
                {"step": 0, "action": "email",       "offset_days": 0},
                {"step": 1, "action": "follow_up",   "offset_days": 3},
                {"step": 2, "action": "final",        "offset_days": 7},
            ],
            "follow_up": [
                {"step": 0, "action": "email",         "offset_days": 0},
                {"step": 1, "action": "call_reminder", "offset_days": 2},
            ],
            "nurture": [
                {"step": 0, "action": "email",       "offset_days": 0},
                {"step": 1, "action": "value_email", "offset_days": 5},
                {"step": 2, "action": "case_study",  "offset_days": 10},
                {"step": 3, "action": "offer",        "offset_days": 15},
                {"step": 4, "action": "close",        "offset_days": 20},
            ],
        }

        plan = SEQUENCE_PLANS.get(sequence_type)
        if not plan:
            return self._bad_request(f"unknown sequence_type: {sequence_type}")

        now  = time.time()
        conn = get_db()
        first_id = None
        try:
            for step_def in plan:
                scheduled_at = now + step_def["offset_days"] * 86400
                cur = conn.execute(
                    "INSERT INTO sequences (lead_id, step, action, scheduled_at) VALUES (?, ?, ?, ?)",
                    (lead_id, step_def["step"], step_def["action"], scheduled_at),
                )
                if first_id is None:
                    first_id = cur.lastrowid
            conn.commit()
        finally:
            conn.close()

        log.info("Sequence started: lead_id=%s type=%s steps=%s", lead_id, sequence_type, len(plan))
        self._send_json({"sequence_id": first_id, "steps": len(plan), "lead_id": lead_id})

    def _handle_execute_due(self):
        now  = time.time()
        conn = get_db()
        executed = 0
        errors   = 0
        try:
            due_rows = conn.execute(
                "SELECT s.*, l.email, l.name, l.company "
                "FROM sequences s "
                "LEFT JOIN leads l ON l.id = s.lead_id "
                "WHERE s.scheduled_at <= ? AND s.executed_at IS NULL",
                (now,),
            ).fetchall()

            for row in due_rows:
                seq_id = row["id"]
                action = row["action"]
                email  = row["email"] or ""
                name   = row["name"] or ""
                company = row["company"] or ""

                try:
                    if action == "call_reminder":
                        result = "reminder_logged"
                    else:
                        # email-type actions
                        if not email:
                            result = "skipped:no_email"
                        else:
                            msg_id = _send_email(email, name, company, action)
                            result = f"sent:{msg_id}"

                    conn.execute(
                        "UPDATE sequences SET executed_at=?, result=? WHERE id=?",
                        (now, result, seq_id),
                    )
                    conn.commit()
                    executed += 1
                    log.info("Sequence step executed: seq_id=%s action=%s result=%s", seq_id, action, result)

                except Exception as exc:
                    log.error("Sequence step error: seq_id=%s error=%s", seq_id, exc)
                    conn.execute(
                        "UPDATE sequences SET executed_at=?, result=? WHERE id=?",
                        (now, f"error:{exc}", seq_id),
                    )
                    conn.commit()
                    errors += 1

        finally:
            conn.close()

        self._send_json({"executed": executed, "errors": errors})

    def _handle_get_sequences(self, lead_id: int):
        conn = get_db()
        try:
            rows = conn.execute(
                "SELECT * FROM sequences WHERE lead_id=? ORDER BY step ASC",
                (lead_id,),
            ).fetchall()
            steps = [dict(r) for r in rows]
        finally:
            conn.close()
        self._send_json(steps)

    def _handle_score(self, body: dict):
        lead_id = body.get("lead_id")
        if not lead_id:
            return self._bad_request("lead_id required")
        conn = get_db()
        try:
            row = conn.execute("SELECT * FROM leads WHERE id=?", (lead_id,)).fetchone()
            if not row:
                conn.close()
                return self._not_found()
            lead  = dict(row)
            score = _calculate_score(lead)
            conn.execute(
                "UPDATE leads SET score=?, updated_at=? WHERE id=?",
                (score, time.time(), lead_id),
            )
            conn.commit()
        finally:
            conn.close()
        self._send_json({"lead_id": lead_id, "score": score})

    def _handle_analytics(self):
        conn = get_db()
        try:
            total = conn.execute(
                "SELECT COUNT(*) FROM leads WHERE status != 'deleted'"
            ).fetchone()[0]

            status_rows = conn.execute(
                "SELECT status, COUNT(*) AS cnt FROM leads WHERE status != 'deleted' GROUP BY status"
            ).fetchall()
            leads_by_status = {r["status"]: r["cnt"] for r in status_rows}

            source_rows = conn.execute(
                "SELECT source, COUNT(*) AS cnt FROM leads WHERE status != 'deleted' GROUP BY source"
            ).fetchall()
            leads_by_source = {r["source"]: r["cnt"] for r in source_rows}

            avg_row = conn.execute(
                "SELECT AVG(score) FROM leads WHERE status != 'deleted'"
            ).fetchone()
            avg_score = round(avg_row[0] or 0.0, 4)

            active_campaigns = conn.execute(
                "SELECT COUNT(*) FROM campaigns WHERE status='active'"
            ).fetchone()[0]
        finally:
            conn.close()

        self._send_json({
            "total_leads":      total,
            "leads_by_status":  leads_by_status,
            "leads_by_source":  leads_by_source,
            "avg_score":        avg_score,
            "campaigns_active": active_campaigns,
        })

    def _handle_list_campaigns(self):
        conn = get_db()
        try:
            rows = conn.execute(
                "SELECT * FROM campaigns ORDER BY created_at DESC"
            ).fetchall()
            campaigns = [dict(r) for r in rows]
        finally:
            conn.close()
        self._send_json(campaigns)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

_server: HTTPServer | None = None


def _shutdown(signum, frame):
    log.info("Shutdown signal received (signal %s)", signum)
    if _server:
        _server.shutdown()


def main():
    global _server
    init_db()
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    _server = HTTPServer(("0.0.0.0", PORT), LeadGenHandler)
    log.info("fm-leadgen (Q-ALSM v2) listening on port %s", PORT)
    _server.serve_forever()


if __name__ == "__main__":
    main()
