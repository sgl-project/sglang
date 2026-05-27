#!/usr/bin/env python3
"""
FractalMesh OMEGA Titan — Events & Ticketing Manager
Port: 7883

Full event management and ticketing system: create events, sell tickets,
manage attendees, send confirmation & reminder emails, and handle check-in
with QR-code validation.

Author : Samuel James Hiotis | ABN 56 628 117 363
System : FractalMesh SaaS Platform
"""

# ---------------------------------------------------------------------------
# Vault loading — MUST run before any os.getenv calls
# ---------------------------------------------------------------------------
import os
from pathlib import Path

_ENV_FILE = Path.home() / ".secrets" / "fractal.env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

# ---------------------------------------------------------------------------
# stdlib imports
# ---------------------------------------------------------------------------
import base64
import hashlib
import hmac
import json
import secrets
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
AGENT_NAME = "fm_events_manager"
PORT = int(os.environ.get("EVENTS_MANAGER_PORT", "7883"))

STRIPE_SECRET_KEY   = os.environ.get("STRIPE_SECRET_KEY", "")
SENDGRID_API_KEY    = os.environ.get("SENDGRID_API_KEY", "")
SENDGRID_FROM_EMAIL = os.environ.get("SENDGRID_FROM_EMAIL", "noreply@fractalmesh.io")
ADMIN_SECRET        = os.environ.get("ADMIN_SECRET", "")

STRIPE_API_BASE   = "https://api.stripe.com/v1"
SENDGRID_API_BASE = "https://api.sendgrid.com/v3"

ROOT    = Path(os.environ.get("FRACTALMESH_HOME", Path.home() / "fmsaas"))
DB_PATH = ROOT / "database" / "sovereign.db"

START_TIME = time.time()

# ---------------------------------------------------------------------------
# Database bootstrap
# ---------------------------------------------------------------------------

def _db() -> sqlite3.Connection:
    con = sqlite3.connect(str(DB_PATH), timeout=15)
    con.execute("PRAGMA journal_mode=WAL")
    con.row_factory = sqlite3.Row
    return con


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = _db()
    con.executescript("""
        CREATE TABLE IF NOT EXISTS events (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id     TEXT UNIQUE NOT NULL,
            title        TEXT NOT NULL,
            description  TEXT NOT NULL DEFAULT '',
            organizer    TEXT NOT NULL DEFAULT 'FractalMesh',
            venue        TEXT NOT NULL DEFAULT '',
            address      TEXT NOT NULL DEFAULT '',
            start_time   REAL NOT NULL,
            end_time     REAL NOT NULL,
            timezone     TEXT NOT NULL DEFAULT 'Australia/Sydney',
            category     TEXT NOT NULL DEFAULT 'general',
            image_url    TEXT NOT NULL DEFAULT '',
            status       TEXT NOT NULL DEFAULT 'draft',
            capacity     INTEGER NOT NULL DEFAULT 0,
            tickets_sold INTEGER NOT NULL DEFAULT 0,
            is_online    INTEGER NOT NULL DEFAULT 0,
            online_url   TEXT NOT NULL DEFAULT '',
            created_at   REAL NOT NULL,
            updated_at   REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS ticket_types (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id      TEXT NOT NULL,
            name          TEXT NOT NULL,
            description   TEXT NOT NULL DEFAULT '',
            price         REAL NOT NULL DEFAULT 0,
            currency      TEXT NOT NULL DEFAULT 'AUD',
            quantity      INTEGER NOT NULL DEFAULT 0,
            quantity_sold INTEGER NOT NULL DEFAULT 0,
            sale_starts   REAL,
            sale_ends     REAL,
            active        INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS tickets (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id         TEXT UNIQUE NOT NULL,
            event_id          TEXT NOT NULL,
            ticket_type_id    INTEGER NOT NULL,
            attendee_email    TEXT NOT NULL,
            attendee_name     TEXT NOT NULL DEFAULT '',
            attendee_phone    TEXT NOT NULL DEFAULT '',
            stripe_payment_id TEXT NOT NULL DEFAULT '',
            amount            REAL NOT NULL DEFAULT 0,
            status            TEXT NOT NULL DEFAULT 'valid',
            qr_code           TEXT NOT NULL DEFAULT '',
            checked_in        INTEGER NOT NULL DEFAULT 0,
            checked_in_at     REAL,
            created_at        REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS waitlist (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id       TEXT NOT NULL,
            email          TEXT NOT NULL,
            name           TEXT NOT NULL DEFAULT '',
            ticket_type_id INTEGER,
            added_at       REAL NOT NULL,
            notified       INTEGER NOT NULL DEFAULT 0
        );
    """)
    con.commit()
    con.close()
    print(f"[{AGENT_NAME}] DB initialised at {DB_PATH}", flush=True)


# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------

def seed_db() -> None:
    con = _db()
    try:
        row = con.execute("SELECT COUNT(*) AS cnt FROM events").fetchone()
        if row["cnt"] > 0:
            return

        now = time.time()
        event_id = "devsum2026"
        start = now + 30 * 86400
        end   = start + 8 * 3600

        con.execute(
            """INSERT INTO events
               (event_id, title, description, organizer, venue, address,
                start_time, end_time, timezone, category, status,
                capacity, is_online, online_url, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                event_id,
                "FractalMesh Developer Summit 2026",
                "The premier virtual gathering for FractalMesh developers. "
                "Join us for talks, workshops, and networking with the community.",
                "FractalMesh",
                "Online",
                "",
                start, end,
                "Australia/Sydney",
                "technology",
                "published",
                500,
                1,
                "https://summit.fractalmesh.io",
                now, now,
            ),
        )

        con.execute(
            """INSERT INTO ticket_types
               (event_id, name, description, price, currency, quantity, sale_starts, sale_ends, active)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (event_id, "General Admission",
             "Full access to all sessions and workshops.", 0.0, "AUD",
             450, now, start, 1),
        )

        con.execute(
            """INSERT INTO ticket_types
               (event_id, name, description, price, currency, quantity, sale_starts, sale_ends, active)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (event_id, "VIP Access",
             "Priority access, exclusive Q&A sessions, and a VIP networking lounge.",
             99.0, "AUD", 50, now, start, 1),
        )

        con.commit()
        print(f"[{AGENT_NAME}] Seeded developer summit event.", flush=True)
    finally:
        con.close()


# ---------------------------------------------------------------------------
# QR code generation
# ---------------------------------------------------------------------------

def generate_qr_code(event_id: str, ticket_id: str) -> str:
    """Generate a deterministic QR-like string for check-in validation."""
    digest = hashlib.sha256(
        (ticket_id + ADMIN_SECRET).encode()
    ).hexdigest()[:8].upper()
    return f"FM-EVT-{event_id[:8]}-{ticket_id[:8]}-{digest}"


def verify_qr_code(ticket_id: str, qr_code: str) -> bool:
    expected = generate_qr_code("", ticket_id)
    # Re-derive from stored ticket event_id agnostically — compare last token
    tokens = qr_code.split("-")
    if len(tokens) != 5:
        return False
    digest = hashlib.sha256(
        (ticket_id + ADMIN_SECRET).encode()
    ).hexdigest()[:8].upper()
    return hmac.compare_digest(tokens[-1], digest)


# ---------------------------------------------------------------------------
# Stripe helpers
# ---------------------------------------------------------------------------

def _stripe_request(method: str, path: str, data: dict | None = None) -> dict:
    url = f"{STRIPE_API_BASE}{path}"
    body = urllib.parse.urlencode(data).encode() if data else None
    creds = base64.b64encode(f"{STRIPE_SECRET_KEY}:".encode()).decode()
    req = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "Authorization": f"Basic {creds}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode()
        try:
            return json.loads(raw)
        except Exception:
            return {"error": {"message": raw}}


def stripe_charge(payment_method_id: str, amount_aud: float,
                  description: str, customer_email: str) -> dict:
    """Create a PaymentIntent and confirm it immediately."""
    amount_cents = int(round(amount_aud * 100))
    pi = _stripe_request("POST", "/payment_intents", {
        "amount": amount_cents,
        "currency": "aud",
        "payment_method": payment_method_id,
        "confirm": "true",
        "description": description,
        "receipt_email": customer_email,
    })
    return pi


def stripe_refund(payment_intent_id: str) -> dict:
    return _stripe_request("POST", "/refunds", {
        "payment_intent": payment_intent_id,
    })


# ---------------------------------------------------------------------------
# SendGrid helpers
# ---------------------------------------------------------------------------

def _sendgrid_send(to_email: str, to_name: str, subject: str, html: str) -> bool:
    if not SENDGRID_API_KEY:
        print(f"[{AGENT_NAME}] SendGrid key missing — skipping email to {to_email}", flush=True)
        return False
    payload = json.dumps({
        "personalizations": [{"to": [{"email": to_email, "name": to_name}]}],
        "from": {"email": SENDGRID_FROM_EMAIL, "name": "FractalMesh Events"},
        "subject": subject,
        "content": [{"type": "text/html", "value": html}],
    }).encode()
    req = urllib.request.Request(
        f"{SENDGRID_API_BASE}/mail/send",
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {SENDGRID_API_KEY}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status in (200, 202)
    except urllib.error.HTTPError as exc:
        print(f"[{AGENT_NAME}] SendGrid error {exc.code}: {exc.read().decode()}", flush=True)
        return False
    except Exception as exc:
        print(f"[{AGENT_NAME}] SendGrid exception: {exc}", flush=True)
        return False


def send_ticket_confirmation(ticket: sqlite3.Row, event: sqlite3.Row,
                              ticket_type: sqlite3.Row) -> None:
    event_start = time.strftime("%A %d %B %Y %H:%M", time.localtime(event["start_time"]))
    price_str = f"${ticket['amount']:.2f} AUD" if ticket["amount"] > 0 else "FREE"
    html = f"""
    <h2>Your ticket is confirmed!</h2>
    <p>Hi {ticket['attendee_name']},</p>
    <p>You're registered for <strong>{event['title']}</strong>.</p>
    <table border="0" cellpadding="6">
      <tr><td><strong>Date &amp; Time:</strong></td><td>{event_start} ({event['timezone']})</td></tr>
      <tr><td><strong>Venue:</strong></td><td>{'Online — ' + event['online_url'] if event['is_online'] else event['venue']}</td></tr>
      <tr><td><strong>Ticket Type:</strong></td><td>{ticket_type['name']}</td></tr>
      <tr><td><strong>Amount Paid:</strong></td><td>{price_str}</td></tr>
      <tr><td><strong>Ticket ID:</strong></td><td>{ticket['ticket_id']}</td></tr>
      <tr><td><strong>QR Code:</strong></td><td><code>{ticket['qr_code']}</code></td></tr>
    </table>
    <p>Present your QR code at the door. See you there!</p>
    <p>— The FractalMesh Events Team</p>
    """
    _sendgrid_send(
        ticket["attendee_email"],
        ticket["attendee_name"],
        f"Your ticket for {event['title']}",
        html,
    )


def send_reminder_email(ticket: sqlite3.Row, event: sqlite3.Row) -> None:
    event_start = time.strftime("%A %d %B %Y %H:%M", time.localtime(event["start_time"]))
    html = f"""
    <h2>Reminder: {event['title']} is tomorrow!</h2>
    <p>Hi {ticket['attendee_name']},</p>
    <p>Just a reminder that <strong>{event['title']}</strong> starts tomorrow.</p>
    <p><strong>When:</strong> {event_start} ({event['timezone']})</p>
    <p><strong>Where:</strong> {'Online — ' + event['online_url'] if event['is_online'] else event['venue'] + ', ' + event['address']}</p>
    <p><strong>Your QR Code:</strong> <code>{ticket['qr_code']}</code></p>
    <p>We look forward to seeing you!</p>
    <p>— The FractalMesh Events Team</p>
    """
    _sendgrid_send(
        ticket["attendee_email"],
        ticket["attendee_name"],
        f"Reminder: {event['title']} is tomorrow",
        html,
    )


def send_waitlist_notification(entry: sqlite3.Row, event: sqlite3.Row,
                               ticket_type_name: str) -> None:
    html = f"""
    <h2>Good news — a ticket is available!</h2>
    <p>Hi {entry['name']},</p>
    <p>A <strong>{ticket_type_name}</strong> ticket for <strong>{event['title']}</strong>
       has become available.</p>
    <p>Visit <a href="https://fractalmesh.io/events/{event['event_id']}">
       the event page</a> to secure your spot before it's gone!</p>
    <p>— The FractalMesh Events Team</p>
    """
    _sendgrid_send(
        entry["email"],
        entry["name"],
        f"Ticket available: {event['title']}",
        html,
    )


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

def _background_worker() -> None:
    """Daemon thread: send 24-hour reminders and notify waitlist on cancellations."""
    while True:
        try:
            _run_reminders()
            _run_waitlist_notifications()
        except Exception as exc:
            print(f"[{AGENT_NAME}] background worker error: {exc}", flush=True)
        time.sleep(3600)


def _run_reminders() -> None:
    now = time.time()
    window_start = now + 23 * 3600
    window_end   = now + 25 * 3600
    con = _db()
    try:
        events = con.execute(
            "SELECT * FROM events WHERE start_time BETWEEN ? AND ? AND status='published'",
            (window_start, window_end),
        ).fetchall()
        for event in events:
            tickets = con.execute(
                "SELECT * FROM tickets WHERE event_id=? AND status='valid' AND checked_in=0",
                (event["event_id"],),
            ).fetchall()
            for ticket in tickets:
                send_reminder_email(ticket, event)
    finally:
        con.close()


def _run_waitlist_notifications() -> None:
    con = _db()
    try:
        entries = con.execute(
            "SELECT w.*, tt.name AS tt_name FROM waitlist w "
            "LEFT JOIN ticket_types tt ON tt.id = w.ticket_type_id "
            "WHERE w.notified=0",
        ).fetchall()
        for entry in entries:
            # Check if capacity has opened up for this ticket type
            tt_row = con.execute(
                "SELECT * FROM ticket_types WHERE id=?", (entry["ticket_type_id"],)
            ).fetchone()
            if tt_row is None:
                continue
            available = tt_row["quantity"] - tt_row["quantity_sold"]
            if available <= 0:
                continue
            event = con.execute(
                "SELECT * FROM events WHERE event_id=?", (entry["event_id"],)
            ).fetchone()
            if event is None:
                continue
            tt_name = entry["tt_name"] or tt_row["name"]
            send_waitlist_notification(entry, event, tt_name)
            con.execute(
                "UPDATE waitlist SET notified=1 WHERE id=?", (entry["id"],)
            )
            con.commit()
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

def _is_admin(handler: "EventsHandler") -> bool:
    auth = handler.headers.get("X-Admin-Secret", "")
    if not ADMIN_SECRET:
        return False
    return hmac.compare_digest(auth, ADMIN_SECRET)


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------

def _json_response(handler: "EventsHandler", code: int, data: dict | list) -> None:
    body = json.dumps(data, default=str).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_body(handler: "EventsHandler") -> dict:
    length = int(handler.headers.get("Content-Length", "0"))
    if length == 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw.decode())
    except Exception:
        return {}


class EventsHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):  # suppress default access log spam
        pass

    # ------------------------------------------------------------------
    # Route dispatch
    # ------------------------------------------------------------------

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        qs   = urllib.parse.parse_qs(self.path.split("?")[1]) if "?" in self.path else {}

        if path == "/health":
            return self._handle_health()

        if path == "/events":
            return self._handle_list_events(qs)

        if path == "/analytics":
            return self._handle_analytics()

        parts = path.strip("/").split("/")

        if len(parts) == 2 and parts[0] == "events":
            return self._handle_get_event(parts[1])

        if len(parts) == 3 and parts[0] == "events" and parts[2] == "attendees":
            return self._handle_get_attendees(parts[1])

        if len(parts) == 2 and parts[0] == "tickets":
            return self._handle_get_ticket(parts[1])

        if len(parts) == 2 and parts[0] == "waitlist":
            return self._handle_get_waitlist(parts[1])

        _json_response(self, 404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        parts = path.strip("/").split("/")

        if path == "/events":
            return self._handle_create_event()

        if path == "/tickets/purchase":
            return self._handle_purchase_ticket(free=False)

        if path == "/tickets/purchase/free":
            return self._handle_purchase_ticket(free=True)

        if path == "/waitlist":
            return self._handle_add_waitlist()

        if len(parts) == 3 and parts[0] == "events" and parts[2] == "ticket_types":
            return self._handle_add_ticket_type(parts[1])

        if len(parts) == 3 and parts[0] == "events" and parts[2] == "publish":
            return self._handle_publish_event(parts[1])

        if len(parts) == 3 and parts[0] == "tickets" and parts[2] == "checkin":
            return self._handle_checkin(parts[1])

        if len(parts) == 3 and parts[0] == "tickets" and parts[2] == "cancel":
            return self._handle_cancel_ticket(parts[1])

        _json_response(self, 404, {"error": "not found"})

    # ------------------------------------------------------------------
    # GET handlers
    # ------------------------------------------------------------------

    def _handle_health(self):
        now = time.time()
        con = _db()
        try:
            event_count = con.execute("SELECT COUNT(*) AS c FROM events").fetchone()["c"]
            day_start   = now - (now % 86400)
            sold_today  = con.execute(
                "SELECT COUNT(*) AS c FROM tickets WHERE created_at >= ? AND status='valid'",
                (day_start,),
            ).fetchone()["c"]
        finally:
            con.close()
        _json_response(self, 200, {
            "status": "ok",
            "agent": AGENT_NAME,
            "port": PORT,
            "uptime_seconds": round(now - START_TIME, 2),
            "event_count": event_count,
            "tickets_sold_today": sold_today,
        })

    def _handle_list_events(self, qs: dict):
        con = _db()
        try:
            clauses = []
            params  = []

            status_filter = qs.get("status", [None])[0]
            if status_filter:
                clauses.append("status=?")
                params.append(status_filter)

            cat_filter = qs.get("category", [None])[0]
            if cat_filter:
                clauses.append("category=?")
                params.append(cat_filter)

            upcoming = qs.get("upcoming", [None])[0]
            if upcoming and upcoming.lower() in ("1", "true", "yes"):
                clauses.append("start_time > ?")
                params.append(time.time())

            limit = min(int(qs.get("limit", ["50"])[0]), 200)
            where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            rows = con.execute(
                f"SELECT * FROM events {where} ORDER BY start_time ASC LIMIT ?",
                params + [limit],
            ).fetchall()
            events = []
            for r in rows:
                e = dict(r)
                tts = con.execute(
                    "SELECT * FROM ticket_types WHERE event_id=? AND active=1",
                    (r["event_id"],),
                ).fetchall()
                e["ticket_types"] = [dict(t) for t in tts]
                events.append(e)
        finally:
            con.close()
        _json_response(self, 200, {"events": events, "count": len(events)})

    def _handle_get_event(self, event_id: str):
        con = _db()
        try:
            event = con.execute(
                "SELECT * FROM events WHERE event_id=?", (event_id,)
            ).fetchone()
            if not event:
                return _json_response(self, 404, {"error": "event not found"})
            tts = con.execute(
                "SELECT * FROM ticket_types WHERE event_id=?", (event_id,)
            ).fetchall()
            data = dict(event)
            data["ticket_types"] = []
            for tt in tts:
                t = dict(tt)
                t["available"] = max(0, t["quantity"] - t["quantity_sold"])
                data["ticket_types"].append(t)
        finally:
            con.close()
        _json_response(self, 200, data)

    def _handle_get_attendees(self, event_id: str):
        if not _is_admin(self):
            return _json_response(self, 403, {"error": "forbidden"})
        con = _db()
        try:
            event = con.execute(
                "SELECT * FROM events WHERE event_id=?", (event_id,)
            ).fetchone()
            if not event:
                return _json_response(self, 404, {"error": "event not found"})
            tickets = con.execute(
                "SELECT t.*, tt.name AS ticket_type_name FROM tickets t "
                "LEFT JOIN ticket_types tt ON tt.id = t.ticket_type_id "
                "WHERE t.event_id=? AND t.status='valid'",
                (event_id,),
            ).fetchall()
        finally:
            con.close()
        _json_response(self, 200, {
            "event_id": event_id,
            "attendees": [dict(t) for t in tickets],
            "count": len(tickets),
        })

    def _handle_get_ticket(self, ticket_id: str):
        con = _db()
        try:
            ticket = con.execute(
                "SELECT * FROM tickets WHERE ticket_id=?", (ticket_id,)
            ).fetchone()
            if not ticket:
                return _json_response(self, 404, {"error": "ticket not found"})
        finally:
            con.close()
        _json_response(self, 200, dict(ticket))

    def _handle_get_waitlist(self, event_id: str):
        if not _is_admin(self):
            return _json_response(self, 403, {"error": "forbidden"})
        con = _db()
        try:
            event = con.execute(
                "SELECT * FROM events WHERE event_id=?", (event_id,)
            ).fetchone()
            if not event:
                return _json_response(self, 404, {"error": "event not found"})
            entries = con.execute(
                "SELECT * FROM waitlist WHERE event_id=? ORDER BY added_at ASC",
                (event_id,),
            ).fetchall()
        finally:
            con.close()
        _json_response(self, 200, {
            "event_id": event_id,
            "waitlist": [dict(e) for e in entries],
            "count": len(entries),
        })

    def _handle_analytics(self):
        if not _is_admin(self):
            return _json_response(self, 403, {"error": "forbidden"})
        con = _db()
        try:
            # Revenue by event
            revenue_rows = con.execute(
                """SELECT e.event_id, e.title,
                          SUM(t.amount) AS total_revenue,
                          COUNT(t.id) AS tickets_sold
                   FROM events e
                   LEFT JOIN tickets t ON t.event_id = e.event_id AND t.status='valid'
                   GROUP BY e.event_id""",
            ).fetchall()

            # Attendance rate (checked_in / valid tickets per event)
            attendance_rows = con.execute(
                """SELECT event_id,
                          COUNT(*) AS total_valid,
                          SUM(checked_in) AS total_checked_in
                   FROM tickets WHERE status='valid'
                   GROUP BY event_id""",
            ).fetchall()

            # Popular ticket types
            popular_rows = con.execute(
                """SELECT tt.name, tt.event_id, tt.quantity_sold, tt.price,
                          SUM(t.amount) AS revenue
                   FROM ticket_types tt
                   LEFT JOIN tickets t ON t.ticket_type_id = tt.id AND t.status='valid'
                   GROUP BY tt.id
                   ORDER BY tt.quantity_sold DESC
                   LIMIT 20""",
            ).fetchall()

            # Total overview
            totals = con.execute(
                """SELECT COUNT(*) AS total_tickets,
                          SUM(amount) AS total_revenue
                   FROM tickets WHERE status='valid'""",
            ).fetchone()
        finally:
            con.close()

        attendance = {}
        for r in attendance_rows:
            rate = 0.0
            if r["total_valid"] and r["total_valid"] > 0:
                rate = round(r["total_checked_in"] / r["total_valid"] * 100, 1)
            attendance[r["event_id"]] = {
                "total_valid": r["total_valid"],
                "total_checked_in": r["total_checked_in"],
                "attendance_rate_pct": rate,
            }

        _json_response(self, 200, {
            "overview": {
                "total_tickets_sold": totals["total_tickets"] or 0,
                "total_revenue_aud": round(totals["total_revenue"] or 0, 2),
            },
            "revenue_by_event": [dict(r) for r in revenue_rows],
            "attendance_by_event": attendance,
            "popular_ticket_types": [dict(r) for r in popular_rows],
        })

    # ------------------------------------------------------------------
    # POST handlers
    # ------------------------------------------------------------------

    def _handle_create_event(self):
        if not _is_admin(self):
            return _json_response(self, 403, {"error": "forbidden"})
        body = _read_body(self)
        required = ["title", "start_time", "end_time"]
        missing = [f for f in required if not body.get(f)]
        if missing:
            return _json_response(self, 400, {"error": f"missing fields: {missing}"})

        event_id = secrets.token_hex(8)
        now = time.time()
        con = _db()
        try:
            con.execute(
                """INSERT INTO events
                   (event_id, title, description, organizer, venue, address,
                    start_time, end_time, timezone, category, image_url,
                    status, capacity, is_online, online_url, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    event_id,
                    body["title"],
                    body.get("description", ""),
                    body.get("organizer", "FractalMesh"),
                    body.get("venue", ""),
                    body.get("address", ""),
                    float(body["start_time"]),
                    float(body["end_time"]),
                    body.get("timezone", "Australia/Sydney"),
                    body.get("category", "general"),
                    body.get("image_url", ""),
                    body.get("status", "draft"),
                    int(body.get("capacity", 0)),
                    int(bool(body.get("is_online", False))),
                    body.get("online_url", ""),
                    now, now,
                ),
            )
            con.commit()
        finally:
            con.close()
        _json_response(self, 201, {"event_id": event_id, "status": "created"})

    def _handle_add_ticket_type(self, event_id: str):
        if not _is_admin(self):
            return _json_response(self, 403, {"error": "forbidden"})
        con = _db()
        try:
            event = con.execute(
                "SELECT * FROM events WHERE event_id=?", (event_id,)
            ).fetchone()
            if not event:
                return _json_response(self, 404, {"error": "event not found"})
            body = _read_body(self)
            required = ["name", "quantity"]
            missing = [f for f in required if body.get(f) is None]
            if missing:
                return _json_response(self, 400, {"error": f"missing fields: {missing}"})

            con.execute(
                """INSERT INTO ticket_types
                   (event_id, name, description, price, currency, quantity,
                    sale_starts, sale_ends, active)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    event_id,
                    body["name"],
                    body.get("description", ""),
                    float(body.get("price", 0)),
                    body.get("currency", "AUD"),
                    int(body["quantity"]),
                    body.get("sale_starts"),
                    body.get("sale_ends"),
                    int(body.get("active", 1)),
                ),
            )
            tt_id = con.execute("SELECT last_insert_rowid() AS lid").fetchone()["lid"]
            con.commit()
        finally:
            con.close()
        _json_response(self, 201, {"ticket_type_id": tt_id, "status": "created"})

    def _handle_publish_event(self, event_id: str):
        if not _is_admin(self):
            return _json_response(self, 403, {"error": "forbidden"})
        con = _db()
        try:
            event = con.execute(
                "SELECT * FROM events WHERE event_id=?", (event_id,)
            ).fetchone()
            if not event:
                return _json_response(self, 404, {"error": "event not found"})
            con.execute(
                "UPDATE events SET status='published', updated_at=? WHERE event_id=?",
                (time.time(), event_id),
            )
            con.commit()
        finally:
            con.close()
        _json_response(self, 200, {"event_id": event_id, "status": "published"})

    def _handle_purchase_ticket(self, free: bool = False):
        body = _read_body(self)
        required = ["event_id", "ticket_type_id", "attendee_email", "attendee_name"]
        if not free:
            required.append("payment_method_id")
        missing = [f for f in required if not body.get(f)]
        if missing:
            return _json_response(self, 400, {"error": f"missing fields: {missing}"})

        con = _db()
        try:
            event = con.execute(
                "SELECT * FROM events WHERE event_id=? AND status='published'",
                (body["event_id"],),
            ).fetchone()
            if not event:
                return _json_response(self, 404, {"error": "event not found or not published"})

            tt = con.execute(
                "SELECT * FROM ticket_types WHERE id=? AND event_id=? AND active=1",
                (int(body["ticket_type_id"]), body["event_id"]),
            ).fetchone()
            if not tt:
                return _json_response(self, 404, {"error": "ticket type not found"})

            now = time.time()

            # Check sale window
            if tt["sale_starts"] and now < tt["sale_starts"]:
                return _json_response(self, 400, {"error": "ticket sales have not started yet"})
            if tt["sale_ends"] and now > tt["sale_ends"]:
                return _json_response(self, 400, {"error": "ticket sales have ended"})

            # Check availability
            available = tt["quantity"] - tt["quantity_sold"]
            if available <= 0:
                return _json_response(self, 409, {"error": "sold out"})

            # Check overall capacity
            if event["capacity"] > 0 and event["tickets_sold"] >= event["capacity"]:
                return _json_response(self, 409, {"error": "event at capacity"})

            # Free-ticket path: reject payment_method_id if provided for paid ticket
            if free and tt["price"] > 0:
                return _json_response(self, 400,
                    {"error": "this ticket type is not free; use /tickets/purchase"})

            stripe_payment_id = ""
            if tt["price"] > 0 and not free:
                pi = stripe_charge(
                    body["payment_method_id"],
                    tt["price"],
                    f"{event['title']} — {tt['name']}",
                    body["attendee_email"],
                )
                if "error" in pi:
                    return _json_response(self, 402, {
                        "error": "payment failed",
                        "detail": pi["error"].get("message", "unknown error"),
                    })
                if pi.get("status") not in ("succeeded", "requires_capture"):
                    return _json_response(self, 402, {
                        "error": "payment not completed",
                        "stripe_status": pi.get("status"),
                    })
                stripe_payment_id = pi["id"]

            ticket_id = secrets.token_hex(12)
            qr_code   = generate_qr_code(body["event_id"], ticket_id)
            amount    = tt["price"] if not free else 0.0

            con.execute(
                """INSERT INTO tickets
                   (ticket_id, event_id, ticket_type_id, attendee_email,
                    attendee_name, attendee_phone, stripe_payment_id,
                    amount, status, qr_code, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    ticket_id,
                    body["event_id"],
                    int(body["ticket_type_id"]),
                    body["attendee_email"],
                    body["attendee_name"],
                    body.get("attendee_phone", ""),
                    stripe_payment_id,
                    amount,
                    "valid",
                    qr_code,
                    now,
                ),
            )
            con.execute(
                "UPDATE ticket_types SET quantity_sold = quantity_sold + 1 WHERE id=?",
                (tt["id"],),
            )
            con.execute(
                "UPDATE events SET tickets_sold = tickets_sold + 1, updated_at=? WHERE event_id=?",
                (now, body["event_id"]),
            )
            con.commit()

            # Fetch freshly for email
            ticket_row = con.execute(
                "SELECT * FROM tickets WHERE ticket_id=?", (ticket_id,)
            ).fetchone()
            tt_row = con.execute(
                "SELECT * FROM ticket_types WHERE id=?", (int(body["ticket_type_id"]),)
            ).fetchone()
        finally:
            con.close()

        # Send confirmation email in a fire-and-forget thread
        threading.Thread(
            target=send_ticket_confirmation,
            args=(ticket_row, event, tt_row),
            daemon=True,
        ).start()

        _json_response(self, 201, {
            "ticket_id": ticket_id,
            "qr_code": qr_code,
            "amount_charged": amount,
            "status": "valid",
        })

    def _handle_checkin(self, ticket_id: str):
        if not _is_admin(self):
            return _json_response(self, 403, {"error": "forbidden"})
        body = _read_body(self)
        qr_code = body.get("qr_code", "")
        if not qr_code:
            return _json_response(self, 400, {"error": "qr_code required"})

        con = _db()
        try:
            ticket = con.execute(
                "SELECT * FROM tickets WHERE ticket_id=?", (ticket_id,)
            ).fetchone()
            if not ticket:
                return _json_response(self, 404, {"error": "ticket not found"})
            if ticket["status"] != "valid":
                return _json_response(self, 409, {
                    "error": f"ticket status is '{ticket['status']}', not valid"
                })
            if ticket["checked_in"]:
                return _json_response(self, 409, {"error": "ticket already checked in"})

            # Verify QR
            if not verify_qr_code(ticket_id, qr_code):
                return _json_response(self, 400, {"error": "invalid QR code"})

            now = time.time()
            con.execute(
                "UPDATE tickets SET checked_in=1, checked_in_at=? WHERE ticket_id=?",
                (now, ticket_id),
            )
            con.commit()
        finally:
            con.close()

        _json_response(self, 200, {
            "ticket_id": ticket_id,
            "checked_in": True,
            "checked_in_at": now,
        })

    def _handle_cancel_ticket(self, ticket_id: str):
        if not _is_admin(self):
            return _json_response(self, 403, {"error": "forbidden"})
        con = _db()
        try:
            ticket = con.execute(
                "SELECT * FROM tickets WHERE ticket_id=?", (ticket_id,)
            ).fetchone()
            if not ticket:
                return _json_response(self, 404, {"error": "ticket not found"})
            if ticket["status"] == "cancelled":
                return _json_response(self, 409, {"error": "ticket already cancelled"})

            refund_id = ""
            refund_err = ""
            if ticket["stripe_payment_id"] and ticket["amount"] > 0:
                result = stripe_refund(ticket["stripe_payment_id"])
                if "error" in result:
                    refund_err = result["error"].get("message", "refund failed")
                else:
                    refund_id = result.get("id", "")

            now = time.time()
            con.execute(
                "UPDATE tickets SET status='cancelled' WHERE ticket_id=?", (ticket_id,)
            )
            # Decrement sold counts
            con.execute(
                "UPDATE ticket_types SET quantity_sold = MAX(0, quantity_sold - 1) WHERE id=?",
                (ticket["ticket_type_id"],),
            )
            con.execute(
                "UPDATE events SET tickets_sold = MAX(0, tickets_sold - 1), updated_at=? "
                "WHERE event_id=?",
                (now, ticket["event_id"]),
            )
            con.commit()
        finally:
            con.close()

        resp = {"ticket_id": ticket_id, "status": "cancelled"}
        if refund_id:
            resp["stripe_refund_id"] = refund_id
        if refund_err:
            resp["refund_warning"] = refund_err
        _json_response(self, 200, resp)

    def _handle_add_waitlist(self):
        body = _read_body(self)
        required = ["event_id", "email", "name"]
        missing = [f for f in required if not body.get(f)]
        if missing:
            return _json_response(self, 400, {"error": f"missing fields: {missing}"})

        con = _db()
        try:
            event = con.execute(
                "SELECT * FROM events WHERE event_id=?", (body["event_id"],)
            ).fetchone()
            if not event:
                return _json_response(self, 404, {"error": "event not found"})

            # Prevent duplicate entries
            existing = con.execute(
                "SELECT id FROM waitlist WHERE event_id=? AND email=?",
                (body["event_id"], body["email"]),
            ).fetchone()
            if existing:
                return _json_response(self, 409, {"error": "already on waitlist"})

            now = time.time()
            con.execute(
                "INSERT INTO waitlist (event_id, email, name, ticket_type_id, added_at) "
                "VALUES (?,?,?,?,?)",
                (
                    body["event_id"],
                    body["email"],
                    body["name"],
                    body.get("ticket_type_id"),
                    now,
                ),
            )
            con.commit()
            entry_id = con.execute(
                "SELECT last_insert_rowid() AS lid"
            ).fetchone()["lid"]
        finally:
            con.close()

        _json_response(self, 201, {
            "waitlist_id": entry_id,
            "event_id": body["event_id"],
            "email": body["email"],
            "status": "added",
        })


# ---------------------------------------------------------------------------
# Server bootstrap
# ---------------------------------------------------------------------------

def run() -> None:
    init_db()
    seed_db()

    bg = threading.Thread(target=_background_worker, daemon=True, name="events-bg")
    bg.start()

    server = HTTPServer(("0.0.0.0", PORT), EventsHandler)
    print(f"[{AGENT_NAME}] listening on port {PORT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"[{AGENT_NAME}] shutting down.", flush=True)


if __name__ == "__main__":
    run()
