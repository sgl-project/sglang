"""
FractalMesh OMEGA Titan — Email Campaign Manager
Port: 7867

Full email campaign management: create campaigns, manage subscriber lists,
send bulk emails via SendGrid, track opens/clicks via pixel tracking,
and analyse deliverability.
"""

import base64
import hashlib
import hmac
import html
import json
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
# Config
# ---------------------------------------------------------------------------
PORT             = int(os.environ.get("EMAIL_CAMPAIGN_PORT", "7867"))
SENDGRID_API_KEY = os.environ.get("SENDGRID_API_KEY", "")
FROM_EMAIL       = os.environ.get("SENDGRID_FROM_EMAIL", "")
FROM_NAME        = os.environ.get("SENDGRID_FROM_NAME", "FractalMesh")
ADMIN_SECRET     = os.environ.get("ADMIN_SECRET", "")
SENDGRID_URL     = "https://api.sendgrid.com/v3/mail/send"
BASE_DOMAIN      = "https://yourdomain.com"

DB_DIR = Path.home() / "fmsaas" / "database"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "sovereign.db"

START_TIME = time.time()

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS lists (
                id               INTEGER PRIMARY KEY,
                name             TEXT UNIQUE,
                description      TEXT,
                subscriber_count INTEGER DEFAULT 0,
                created_at       REAL
            );

            CREATE TABLE IF NOT EXISTS subscribers (
                id               INTEGER PRIMARY KEY,
                email            TEXT UNIQUE,
                first_name       TEXT,
                last_name        TEXT,
                list_ids         TEXT,
                status           TEXT DEFAULT 'subscribed',
                source           TEXT,
                tags             TEXT,
                custom_fields    TEXT,
                subscribed_at    REAL,
                unsubscribed_at  REAL,
                bounced_at       REAL
            );

            CREATE TABLE IF NOT EXISTS campaigns (
                id                INTEGER PRIMARY KEY,
                name              TEXT,
                subject           TEXT,
                from_name         TEXT,
                from_email        TEXT,
                body_html         TEXT,
                body_text         TEXT,
                list_ids          TEXT,
                status            TEXT DEFAULT 'draft',
                scheduled_at      REAL,
                sent_at           REAL,
                total_recipients  INTEGER DEFAULT 0,
                total_sent        INTEGER DEFAULT 0,
                created_at        REAL
            );

            CREATE TABLE IF NOT EXISTS sends (
                id             INTEGER PRIMARY KEY,
                campaign_id    INTEGER,
                subscriber_id  INTEGER,
                email          TEXT,
                status         TEXT DEFAULT 'pending',
                sg_message_id  TEXT,
                sent_at        REAL,
                opened_at      REAL,
                clicked_at     REAL,
                bounced_at     REAL
            );

            CREATE TABLE IF NOT EXISTS events (
                id             INTEGER PRIMARY KEY,
                campaign_id    INTEGER,
                subscriber_id  INTEGER,
                event_type     TEXT,
                metadata       TEXT,
                occurred_at    REAL
            );

            CREATE INDEX IF NOT EXISTS idx_subscribers_email
                ON subscribers(email);
            CREATE INDEX IF NOT EXISTS idx_subscribers_status
                ON subscribers(status);
            CREATE INDEX IF NOT EXISTS idx_campaigns_status
                ON campaigns(status);
            CREATE INDEX IF NOT EXISTS idx_sends_campaign
                ON sends(campaign_id);
            CREATE INDEX IF NOT EXISTS idx_events_campaign
                ON events(campaign_id);
        """)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def now() -> float:
    return time.time()


def json_resp(handler: "CampaignHandler", code: int, data: object) -> None:
    body = json.dumps(data, default=str).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def error_resp(handler: "CampaignHandler", code: int, message: str) -> None:
    json_resp(handler, code, {"error": message})


def parse_body(handler: "CampaignHandler") -> dict:
    length = int(handler.headers.get("Content-Length", 0))
    if length == 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw)
    except Exception:
        return {}


def parse_qs(query: str) -> dict:
    params: dict = {}
    if not query:
        return params
    for part in query.split("&"):
        if "=" in part:
            k, v = part.split("=", 1)
            params[urllib.parse.unquote_plus(k)] = urllib.parse.unquote_plus(v)
        else:
            params[urllib.parse.unquote_plus(part)] = ""
    return params


# keep urllib.parse available via a local import alias
try:
    import urllib.parse as urllib_parse
except ImportError:
    urllib_parse = None  # type: ignore


def _unquote(s: str) -> str:
    try:
        import urllib.parse as _up
        return _up.unquote_plus(s)
    except Exception:
        return s


def require_admin(handler: "CampaignHandler") -> bool:
    if not ADMIN_SECRET:
        return True
    auth = handler.headers.get("X-Admin-Secret", "")
    if auth != ADMIN_SECRET:
        error_resp(handler, 403, "Forbidden: invalid admin secret")
        return False
    return True


def compute_unsubscribe_token(email: str) -> str:
    key = SENDGRID_API_KEY[:16].encode() if SENDGRID_API_KEY else b"defaultkey12345!"
    return hmac.new(key, email.encode(), hashlib.sha256).hexdigest()


def build_unsubscribe_url(email: str) -> str:
    token = compute_unsubscribe_token(email)
    try:
        import urllib.parse as _up
        encoded_email = _up.quote(email)
    except Exception:
        encoded_email = email
    return f"{BASE_DOMAIN}/unsubscribe?email={encoded_email}&token={token}"


def list_ids_to_set(list_ids_str: str) -> set:
    if not list_ids_str:
        return set()
    try:
        return set(json.loads(list_ids_str))
    except Exception:
        return set(x.strip() for x in list_ids_str.split(",") if x.strip())


def update_list_subscriber_count(conn: sqlite3.Connection, list_id: int) -> None:
    count = conn.execute(
        """SELECT COUNT(*) FROM subscribers
           WHERE status='subscribed'
             AND list_ids LIKE ?""",
        (f"%{list_id}%",),
    ).fetchone()[0]
    conn.execute(
        "UPDATE lists SET subscriber_count=? WHERE id=?",
        (count, list_id),
    )


# ---------------------------------------------------------------------------
# SendGrid send logic
# ---------------------------------------------------------------------------

def _sg_send_batch(
    campaign_id: int,
    subject: str,
    from_name: str,
    from_email: str,
    body_html: str,
    body_text: str,
    batch: list,  # list of sqlite3.Row
) -> list:
    """Send a batch of subscribers via SendGrid. Returns list of (subscriber_id, status, sg_message_id)."""
    results = []
    personalizations = []
    for sub in batch:
        email = sub["email"]
        first = sub["first_name"] or ""
        last = sub["last_name"] or ""
        unsub_url = build_unsubscribe_url(email)
        personalizations.append({
            "to": [{"email": email, "name": f"{first} {last}".strip()}],
            "substitutions": {
                "{first_name}": first,
                "{last_name}": last,
                "{unsubscribe_url}": unsub_url,
            },
            "custom_args": {
                "campaign_id": str(campaign_id),
                "subscriber_id": str(sub["id"]),
            },
        })

    payload = {
        "personalizations": personalizations,
        "from": {"email": from_email, "name": from_name},
        "subject": subject,
        "content": [],
        "tracking_settings": {
            "open_tracking": {"enable": True},
            "click_tracking": {"enable": True},
        },
    }
    if body_text:
        payload["content"].append({"type": "text/plain", "value": body_text})
    if body_html:
        payload["content"].append({"type": "text/html", "value": body_html})
    if not payload["content"]:
        payload["content"].append({"type": "text/plain", "value": ""})

    body_bytes = json.dumps(payload).encode()
    req = urllib.request.Request(
        SENDGRID_URL,
        data=body_bytes,
        headers={
            "Authorization": f"Bearer {SENDGRID_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    sg_message_id = ""
    status = "sent"
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            sg_message_id = resp.headers.get("X-Message-Id", "")
    except urllib.error.HTTPError as exc:
        status = "failed"
        sg_message_id = f"error:{exc.code}"
    except Exception as exc:
        status = "failed"
        sg_message_id = f"error:{exc}"

    sent_ts = now()
    for sub in batch:
        results.append((sub["id"], status, sg_message_id, sent_ts))
    return results


def _do_send_campaign(campaign: sqlite3.Row) -> None:
    campaign_id = campaign["id"]
    list_ids = list_ids_to_set(campaign["list_ids"])
    if not list_ids:
        return

    with get_db() as conn:
        rows = conn.execute("SELECT * FROM subscribers WHERE status='subscribed'").fetchall()

    subscribers = [r for r in rows if list_ids_to_set(r["list_ids"]) & list_ids]
    if not subscribers:
        with get_db() as conn:
            conn.execute(
                "UPDATE campaigns SET status='sent', sent_at=?, total_recipients=0, total_sent=0 WHERE id=?",
                (now(), campaign_id),
            )
        return

    from_name  = campaign["from_name"] or FROM_NAME
    from_email = campaign["from_email"] or FROM_EMAIL
    subject    = campaign["subject"] or "(no subject)"
    body_html  = campaign["body_html"] or ""
    body_text  = campaign["body_text"] or ""

    total_sent = 0
    batch_size = 100
    all_results = []

    for i in range(0, len(subscribers), batch_size):
        batch = subscribers[i: i + batch_size]
        results = _sg_send_batch(
            campaign_id, subject, from_name, from_email, body_html, body_text, batch
        )
        all_results.extend(results)
        total_sent += sum(1 for r in results if r[1] == "sent")
        if i + batch_size < len(subscribers):
            time.sleep(1)

    with get_db() as conn:
        conn.execute(
            """UPDATE campaigns
               SET status='sent', sent_at=?, total_recipients=?, total_sent=?
               WHERE id=?""",
            (now(), len(subscribers), total_sent, campaign_id),
        )
        for sub_id, status, sg_msg_id, sent_ts in all_results:
            sub_email = next(
                (s["email"] for s in subscribers if s["id"] == sub_id), ""
            )
            conn.execute(
                """INSERT INTO sends
                   (campaign_id, subscriber_id, email, status, sg_message_id, sent_at)
                   VALUES (?,?,?,?,?,?)""",
                (campaign_id, sub_id, sub_email, status, sg_msg_id, sent_ts),
            )
            conn.execute(
                """INSERT INTO events (campaign_id, subscriber_id, event_type, metadata, occurred_at)
                   VALUES (?,?,?,?,?)""",
                (campaign_id, sub_id, status, json.dumps({"sg_message_id": sg_msg_id}), sent_ts),
            )


def _campaign_sender_loop() -> None:
    """Background daemon: check every 60s for scheduled campaigns to send."""
    while True:
        try:
            with get_db() as conn:
                due = conn.execute(
                    "SELECT * FROM campaigns WHERE status='scheduled' AND scheduled_at <= ?",
                    (now(),),
                ).fetchall()
            for campaign in due:
                with get_db() as conn:
                    conn.execute(
                        "UPDATE campaigns SET status='sending' WHERE id=? AND status='scheduled'",
                        (campaign["id"],),
                    )
                try:
                    _do_send_campaign(campaign)
                except Exception:
                    with get_db() as conn:
                        conn.execute(
                            "UPDATE campaigns SET status='error' WHERE id=?",
                            (campaign["id"],),
                        )
        except Exception:
            pass
        time.sleep(60)


# ---------------------------------------------------------------------------
# Tracking pixel bytes
# ---------------------------------------------------------------------------
_PIXEL_GIF_B64 = "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
_PIXEL_GIF = base64.b64decode(_PIXEL_GIF_B64)


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------

class CampaignHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # suppress default logging
        pass

    # ------------------------------------------------------------------
    # Router
    # ------------------------------------------------------------------
    def do_GET(self):
        path, _, query = self.path.partition("?")
        parts = [p for p in path.split("/") if p]
        params = parse_qs(query)

        try:
            if path == "/health":
                self._health()
            elif path == "/lists":
                self._get_lists()
            elif len(parts) == 2 and parts[0] == "lists":
                self._get_list(parts[1])
            elif path == "/subscribers":
                self._get_subscribers(params)
            elif len(parts) == 2 and parts[0] == "subscribers":
                self._get_subscriber(parts[1])
            elif path == "/campaigns":
                self._get_campaigns(params)
            elif len(parts) == 2 and parts[0] == "campaigns":
                self._get_campaign(parts[1])
            elif len(parts) == 3 and parts[0] == "campaigns" and parts[2] == "stats":
                self._get_campaign_stats(parts[1])
            elif len(parts) == 3 and parts[0] == "pixel":
                self._pixel(parts[1], parts[2])
            elif len(parts) == 3 and parts[0] == "click":
                self._click(parts[1], parts[2], params)
            elif path == "/unsubscribe":
                self._unsubscribe_get(params)
            else:
                error_resp(self, 404, "Not found")
        except Exception as exc:
            error_resp(self, 500, str(exc))

    def do_POST(self):
        path, _, _ = self.path.partition("?")
        parts = [p for p in path.split("/") if p]

        try:
            if path == "/lists":
                self._create_list()
            elif path == "/subscribers":
                self._create_subscriber()
            elif path == "/subscribers/import":
                self._import_subscribers()
            elif len(parts) == 3 and parts[0] == "subscribers" and parts[2] == "unsubscribe":
                self._unsubscribe_post(parts[1])
            elif path == "/campaigns":
                self._create_campaign()
            elif len(parts) == 3 and parts[0] == "campaigns" and parts[2] == "schedule":
                self._schedule_campaign(parts[1])
            elif len(parts) == 3 and parts[0] == "campaigns" and parts[2] == "send_test":
                self._send_test(parts[1])
            else:
                error_resp(self, 404, "Not found")
        except Exception as exc:
            error_resp(self, 500, str(exc))

    # ------------------------------------------------------------------
    # GET /health
    # ------------------------------------------------------------------
    def _health(self):
        with get_db() as conn:
            campaigns = conn.execute("SELECT COUNT(*) FROM campaigns").fetchone()[0]
            subscribers = conn.execute("SELECT COUNT(*) FROM subscribers").fetchone()[0]
            lists = conn.execute("SELECT COUNT(*) FROM lists").fetchone()[0]
        json_resp(self, 200, {
            "status": "ok",
            "uptime_seconds": round(now() - START_TIME, 1),
            "campaigns": campaigns,
            "subscribers": subscribers,
            "lists": lists,
            "port": PORT,
        })

    # ------------------------------------------------------------------
    # GET /lists
    # ------------------------------------------------------------------
    def _get_lists(self):
        with get_db() as conn:
            rows = conn.execute(
                "SELECT * FROM lists ORDER BY created_at DESC"
            ).fetchall()
        json_resp(self, 200, [dict(r) for r in rows])

    # ------------------------------------------------------------------
    # GET /lists/{id}
    # ------------------------------------------------------------------
    def _get_list(self, list_id_str: str):
        try:
            list_id = int(list_id_str)
        except ValueError:
            error_resp(self, 400, "Invalid list id")
            return
        with get_db() as conn:
            lst = conn.execute("SELECT * FROM lists WHERE id=?", (list_id,)).fetchone()
            if not lst:
                error_resp(self, 404, "List not found")
                return
            recent = conn.execute(
                """SELECT id, email, first_name, last_name, status, subscribed_at
                   FROM subscribers
                   WHERE list_ids LIKE ? AND status='subscribed'
                   ORDER BY subscribed_at DESC LIMIT 20""",
                (f"%{list_id}%",),
            ).fetchall()
        data = dict(lst)
        data["recent_subscribers"] = [dict(r) for r in recent]
        json_resp(self, 200, data)

    # ------------------------------------------------------------------
    # GET /subscribers
    # ------------------------------------------------------------------
    def _get_subscribers(self, params: dict):
        list_id = params.get("list_id", "")
        status  = params.get("status", "")
        tags    = params.get("tags", "")
        limit   = min(int(params.get("limit", "100")), 1000)

        clauses = []
        args: list = []
        if list_id:
            clauses.append("list_ids LIKE ?")
            args.append(f"%{list_id}%")
        if status:
            clauses.append("status=?")
            args.append(status)
        if tags:
            clauses.append("tags LIKE ?")
            args.append(f"%{tags}%")

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        args.append(limit)
        with get_db() as conn:
            rows = conn.execute(
                f"SELECT * FROM subscribers {where} ORDER BY subscribed_at DESC LIMIT ?",
                args,
            ).fetchall()
        json_resp(self, 200, [dict(r) for r in rows])

    # ------------------------------------------------------------------
    # GET /subscribers/{email}
    # ------------------------------------------------------------------
    def _get_subscriber(self, raw_email: str):
        email = _unquote(raw_email)
        with get_db() as conn:
            sub = conn.execute(
                "SELECT * FROM subscribers WHERE email=?", (email,)
            ).fetchone()
            if not sub:
                error_resp(self, 404, "Subscriber not found")
                return
            sends = conn.execute(
                "SELECT * FROM sends WHERE subscriber_id=? ORDER BY sent_at DESC LIMIT 50",
                (sub["id"],),
            ).fetchall()
        data = dict(sub)
        data["send_history"] = [dict(s) for s in sends]
        json_resp(self, 200, data)

    # ------------------------------------------------------------------
    # GET /campaigns
    # ------------------------------------------------------------------
    def _get_campaigns(self, params: dict):
        status = params.get("status", "")
        limit  = min(int(params.get("limit", "50")), 500)
        if status:
            with get_db() as conn:
                rows = conn.execute(
                    "SELECT * FROM campaigns WHERE status=? ORDER BY created_at DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
        else:
            with get_db() as conn:
                rows = conn.execute(
                    "SELECT * FROM campaigns ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        json_resp(self, 200, [dict(r) for r in rows])

    # ------------------------------------------------------------------
    # GET /campaigns/{id}
    # ------------------------------------------------------------------
    def _get_campaign(self, campaign_id_str: str):
        try:
            cid = int(campaign_id_str)
        except ValueError:
            error_resp(self, 400, "Invalid campaign id")
            return
        with get_db() as conn:
            camp = conn.execute("SELECT * FROM campaigns WHERE id=?", (cid,)).fetchone()
            if not camp:
                error_resp(self, 404, "Campaign not found")
                return
            total_sent    = conn.execute("SELECT COUNT(*) FROM sends WHERE campaign_id=?", (cid,)).fetchone()[0]
            total_opened  = conn.execute("SELECT COUNT(*) FROM sends WHERE campaign_id=? AND opened_at IS NOT NULL", (cid,)).fetchone()[0]
            total_clicked = conn.execute("SELECT COUNT(*) FROM sends WHERE campaign_id=? AND clicked_at IS NOT NULL", (cid,)).fetchone()[0]
        data = dict(camp)
        data["stats"] = {
            "total_sent": total_sent,
            "total_opened": total_opened,
            "total_clicked": total_clicked,
            "open_rate": round(total_opened / total_sent * 100, 2) if total_sent else 0,
            "click_rate": round(total_clicked / total_sent * 100, 2) if total_sent else 0,
        }
        json_resp(self, 200, data)

    # ------------------------------------------------------------------
    # GET /campaigns/{id}/stats
    # ------------------------------------------------------------------
    def _get_campaign_stats(self, campaign_id_str: str):
        try:
            cid = int(campaign_id_str)
        except ValueError:
            error_resp(self, 400, "Invalid campaign id")
            return
        with get_db() as conn:
            camp = conn.execute("SELECT id, name, subject, status FROM campaigns WHERE id=?", (cid,)).fetchone()
            if not camp:
                error_resp(self, 404, "Campaign not found")
                return
            total_sent    = conn.execute("SELECT COUNT(*) FROM sends WHERE campaign_id=?", (cid,)).fetchone()[0]
            total_opened  = conn.execute("SELECT COUNT(*) FROM sends WHERE campaign_id=? AND opened_at IS NOT NULL", (cid,)).fetchone()[0]
            total_clicked = conn.execute("SELECT COUNT(*) FROM sends WHERE campaign_id=? AND clicked_at IS NOT NULL", (cid,)).fetchone()[0]
            total_bounced = conn.execute("SELECT COUNT(*) FROM sends WHERE campaign_id=? AND bounced_at IS NOT NULL", (cid,)).fetchone()[0]
            total_pending = conn.execute("SELECT COUNT(*) FROM sends WHERE campaign_id=? AND status='pending'", (cid,)).fetchone()[0]
            total_failed  = conn.execute("SELECT COUNT(*) FROM sends WHERE campaign_id=? AND status='failed'", (cid,)).fetchone()[0]
            recent_events = conn.execute(
                "SELECT * FROM events WHERE campaign_id=? ORDER BY occurred_at DESC LIMIT 100",
                (cid,),
            ).fetchall()
        json_resp(self, 200, {
            "campaign_id": cid,
            "name": camp["name"],
            "subject": camp["subject"],
            "status": camp["status"],
            "counts": {
                "sent": total_sent,
                "opened": total_opened,
                "clicked": total_clicked,
                "bounced": total_bounced,
                "pending": total_pending,
                "failed": total_failed,
            },
            "rates": {
                "open_rate":    round(total_opened  / total_sent * 100, 2) if total_sent else 0,
                "click_rate":   round(total_clicked / total_sent * 100, 2) if total_sent else 0,
                "bounce_rate":  round(total_bounced / total_sent * 100, 2) if total_sent else 0,
                "failure_rate": round(total_failed  / total_sent * 100, 2) if total_sent else 0,
            },
            "recent_events": [dict(e) for e in recent_events],
        })

    # ------------------------------------------------------------------
    # GET /pixel/{campaign_id}/{subscriber_id}
    # ------------------------------------------------------------------
    def _pixel(self, campaign_id_str: str, subscriber_id_str: str):
        try:
            cid = int(campaign_id_str)
            sid = int(subscriber_id_str)
        except ValueError:
            self.send_response(400)
            self.end_headers()
            return
        ts = now()
        try:
            with get_db() as conn:
                conn.execute(
                    """UPDATE sends SET opened_at=?
                       WHERE campaign_id=? AND subscriber_id=? AND opened_at IS NULL""",
                    (ts, cid, sid),
                )
                conn.execute(
                    """INSERT INTO events (campaign_id, subscriber_id, event_type, metadata, occurred_at)
                       VALUES (?,?,?,?,?)""",
                    (cid, sid, "open", json.dumps({"source": "pixel"}), ts),
                )
        except Exception:
            pass

        self.send_response(200)
        self.send_header("Content-Type", "image/gif")
        self.send_header("Content-Length", str(len(_PIXEL_GIF)))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(_PIXEL_GIF)

    # ------------------------------------------------------------------
    # GET /click/{campaign_id}/{subscriber_id}
    # ------------------------------------------------------------------
    def _click(self, campaign_id_str: str, subscriber_id_str: str, params: dict):
        try:
            cid = int(campaign_id_str)
            sid = int(subscriber_id_str)
        except ValueError:
            error_resp(self, 400, "Invalid ids")
            return
        target_url = params.get("url", BASE_DOMAIN)
        ts = now()
        try:
            with get_db() as conn:
                conn.execute(
                    """UPDATE sends SET clicked_at=?
                       WHERE campaign_id=? AND subscriber_id=? AND clicked_at IS NULL""",
                    (ts, cid, sid),
                )
                conn.execute(
                    """INSERT INTO events (campaign_id, subscriber_id, event_type, metadata, occurred_at)
                       VALUES (?,?,?,?,?)""",
                    (cid, sid, "click", json.dumps({"url": target_url}), ts),
                )
        except Exception:
            pass

        self.send_response(302)
        self.send_header("Location", target_url)
        self.end_headers()

    # ------------------------------------------------------------------
    # GET /unsubscribe?email=...&token=...
    # ------------------------------------------------------------------
    def _unsubscribe_get(self, params: dict):
        email = params.get("email", "")
        token = params.get("token", "")
        if not email or not token:
            error_resp(self, 400, "Missing email or token")
            return
        expected = compute_unsubscribe_token(email)
        if not hmac.compare_digest(token, expected):
            error_resp(self, 403, "Invalid unsubscribe token")
            return
        ts = now()
        with get_db() as conn:
            sub = conn.execute("SELECT id FROM subscribers WHERE email=?", (email,)).fetchone()
            if not sub:
                error_resp(self, 404, "Subscriber not found")
                return
            conn.execute(
                "UPDATE subscribers SET status='unsubscribed', unsubscribed_at=? WHERE email=?",
                (ts, email),
            )
        body = b"<html><body><h2>You have been unsubscribed successfully.</h2></body></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ------------------------------------------------------------------
    # POST /lists
    # ------------------------------------------------------------------
    def _create_list(self):
        data = parse_body(self)
        name = (data.get("name") or "").strip()
        if not name:
            error_resp(self, 400, "name is required")
            return
        description = data.get("description", "")
        ts = now()
        try:
            with get_db() as conn:
                cur = conn.execute(
                    "INSERT INTO lists (name, description, subscriber_count, created_at) VALUES (?,?,0,?)",
                    (name, description, ts),
                )
                list_id = cur.lastrowid
                row = conn.execute("SELECT * FROM lists WHERE id=?", (list_id,)).fetchone()
        except sqlite3.IntegrityError:
            error_resp(self, 409, f"List '{name}' already exists")
            return
        json_resp(self, 201, dict(row))

    # ------------------------------------------------------------------
    # POST /subscribers
    # ------------------------------------------------------------------
    def _create_subscriber(self):
        data = parse_body(self)
        email = (data.get("email") or "").strip().lower()
        if not email or not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            error_resp(self, 400, "Valid email is required")
            return
        first_name    = data.get("first_name", "")
        last_name     = data.get("last_name", "")
        list_ids_raw  = data.get("list_ids", [])
        source        = data.get("source", "")
        tags_raw      = data.get("tags", [])
        custom_fields = data.get("custom_fields", {})
        ts = now()

        list_ids_str   = json.dumps(list_ids_raw) if isinstance(list_ids_raw, list) else str(list_ids_raw)
        tags_str       = json.dumps(tags_raw) if isinstance(tags_raw, list) else str(tags_raw)
        custom_fields_str = json.dumps(custom_fields) if isinstance(custom_fields, dict) else str(custom_fields)

        try:
            with get_db() as conn:
                cur = conn.execute(
                    """INSERT INTO subscribers
                       (email, first_name, last_name, list_ids, status, source, tags, custom_fields, subscribed_at)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (email, first_name, last_name, list_ids_str, "subscribed",
                     source, tags_str, custom_fields_str, ts),
                )
                sub_id = cur.lastrowid
                for lid in (list_ids_raw if isinstance(list_ids_raw, list) else []):
                    try:
                        update_list_subscriber_count(conn, int(lid))
                    except Exception:
                        pass
                row = conn.execute("SELECT * FROM subscribers WHERE id=?", (sub_id,)).fetchone()
        except sqlite3.IntegrityError:
            error_resp(self, 409, f"Subscriber '{email}' already exists")
            return
        json_resp(self, 201, dict(row))

    # ------------------------------------------------------------------
    # POST /subscribers/import  (admin-gated)
    # ------------------------------------------------------------------
    def _import_subscribers(self):
        if not require_admin(self):
            return
        data = parse_body(self)
        subscribers_raw = data.get("subscribers", [])
        list_id = data.get("list_id")
        if not isinstance(subscribers_raw, list):
            error_resp(self, 400, "subscribers must be an array")
            return

        imported = 0
        skipped  = 0
        ts = now()
        with get_db() as conn:
            for sub in subscribers_raw:
                email = (sub.get("email") or "").strip().lower()
                if not email or not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                    skipped += 1
                    continue
                first_name    = sub.get("first_name", "")
                last_name     = sub.get("last_name", "")
                source        = sub.get("source", "import")
                tags_raw      = sub.get("tags", [])
                custom_fields = sub.get("custom_fields", {})

                list_ids_list = sub.get("list_ids", [])
                if list_id and list_id not in list_ids_list:
                    list_ids_list.append(list_id)
                list_ids_str      = json.dumps(list_ids_list)
                tags_str          = json.dumps(tags_raw) if isinstance(tags_raw, list) else str(tags_raw)
                custom_fields_str = json.dumps(custom_fields) if isinstance(custom_fields, dict) else str(custom_fields)

                try:
                    conn.execute(
                        """INSERT INTO subscribers
                           (email, first_name, last_name, list_ids, status, source, tags, custom_fields, subscribed_at)
                           VALUES (?,?,?,?,?,?,?,?,?)""",
                        (email, first_name, last_name, list_ids_str, "subscribed",
                         source, tags_str, custom_fields_str, ts),
                    )
                    imported += 1
                except sqlite3.IntegrityError:
                    skipped += 1

            if list_id:
                try:
                    update_list_subscriber_count(conn, int(list_id))
                except Exception:
                    pass

        json_resp(self, 200, {"imported": imported, "skipped": skipped})

    # ------------------------------------------------------------------
    # POST /subscribers/{email}/unsubscribe
    # ------------------------------------------------------------------
    def _unsubscribe_post(self, raw_email: str):
        email = _unquote(raw_email).lower()
        ts = now()
        with get_db() as conn:
            sub = conn.execute("SELECT id FROM subscribers WHERE email=?", (email,)).fetchone()
            if not sub:
                error_resp(self, 404, "Subscriber not found")
                return
            conn.execute(
                "UPDATE subscribers SET status='unsubscribed', unsubscribed_at=? WHERE email=?",
                (ts, email),
            )
        json_resp(self, 200, {"status": "unsubscribed", "email": email})

    # ------------------------------------------------------------------
    # POST /campaigns
    # ------------------------------------------------------------------
    def _create_campaign(self):
        data = parse_body(self)
        name = (data.get("name") or "").strip()
        if not name:
            error_resp(self, 400, "name is required")
            return
        subject     = data.get("subject", "")
        from_name   = data.get("from_name", FROM_NAME)
        from_email  = data.get("from_email", FROM_EMAIL)
        body_html   = data.get("body_html", "")
        body_text   = data.get("body_text", "")
        list_ids    = data.get("list_ids", [])
        list_ids_str = json.dumps(list_ids) if isinstance(list_ids, list) else str(list_ids)
        ts = now()
        with get_db() as conn:
            cur = conn.execute(
                """INSERT INTO campaigns
                   (name, subject, from_name, from_email, body_html, body_text,
                    list_ids, status, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (name, subject, from_name, from_email, body_html, body_text,
                 list_ids_str, "draft", ts),
            )
            cid = cur.lastrowid
            row = conn.execute("SELECT * FROM campaigns WHERE id=?", (cid,)).fetchone()
        json_resp(self, 201, dict(row))

    # ------------------------------------------------------------------
    # POST /campaigns/{id}/schedule  (admin-gated)
    # ------------------------------------------------------------------
    def _schedule_campaign(self, campaign_id_str: str):
        if not require_admin(self):
            return
        try:
            cid = int(campaign_id_str)
        except ValueError:
            error_resp(self, 400, "Invalid campaign id")
            return
        data = parse_body(self)
        scheduled_at_raw = data.get("scheduled_at", "")
        if not scheduled_at_raw:
            error_resp(self, 400, "scheduled_at is required")
            return
        # Parse ISO 8601 to epoch float
        try:
            import time as _t

            # Handle both 'Z' suffix and offset-naive strings
            s = str(scheduled_at_raw).replace("Z", "+00:00")
            # Use strptime for stdlib-only parsing
            for fmt in (
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
            ):
                try:
                    import datetime as _dt
                    if "%z" in fmt:
                        parsed = _dt.datetime.strptime(scheduled_at_raw.replace("Z", "+0000"), fmt.replace("%z", "%z"))
                    else:
                        parsed = _dt.datetime.strptime(scheduled_at_raw.rstrip("Z"), fmt)
                    scheduled_ts = parsed.timestamp()
                    break
                except Exception:
                    continue
            else:
                scheduled_ts = float(scheduled_at_raw)
        except Exception:
            error_resp(self, 400, "Invalid scheduled_at format")
            return

        with get_db() as conn:
            camp = conn.execute("SELECT id, status FROM campaigns WHERE id=?", (cid,)).fetchone()
            if not camp:
                error_resp(self, 404, "Campaign not found")
                return
            conn.execute(
                "UPDATE campaigns SET status='scheduled', scheduled_at=? WHERE id=?",
                (scheduled_ts, cid),
            )
            row = conn.execute("SELECT * FROM campaigns WHERE id=?", (cid,)).fetchone()
        json_resp(self, 200, dict(row))

    # ------------------------------------------------------------------
    # POST /campaigns/{id}/send_test  (admin-gated)
    # ------------------------------------------------------------------
    def _send_test(self, campaign_id_str: str):
        if not require_admin(self):
            return
        try:
            cid = int(campaign_id_str)
        except ValueError:
            error_resp(self, 400, "Invalid campaign id")
            return
        data = parse_body(self)
        to_email = (data.get("to_email") or "").strip()
        if not to_email or not re.match(r"[^@]+@[^@]+\.[^@]+", to_email):
            error_resp(self, 400, "Valid to_email is required")
            return
        with get_db() as conn:
            camp = conn.execute("SELECT * FROM campaigns WHERE id=?", (cid,)).fetchone()
            if not camp:
                error_resp(self, 404, "Campaign not found")
                return

        from_name_val  = camp["from_name"] or FROM_NAME
        from_email_val = camp["from_email"] or FROM_EMAIL
        subject        = f"[TEST] {camp['subject'] or '(no subject)'}"
        body_html      = camp["body_html"] or ""
        body_text      = camp["body_text"] or ""
        unsub_url      = build_unsubscribe_url(to_email)

        payload = {
            "personalizations": [{
                "to": [{"email": to_email}],
                "substitutions": {
                    "{first_name}": "Test",
                    "{last_name}":  "User",
                    "{unsubscribe_url}": unsub_url,
                },
            }],
            "from": {"email": from_email_val, "name": from_name_val},
            "subject": subject,
            "content": [],
            "tracking_settings": {
                "open_tracking": {"enable": False},
                "click_tracking": {"enable": False},
            },
        }
        if body_text:
            payload["content"].append({"type": "text/plain", "value": body_text})
        if body_html:
            payload["content"].append({"type": "text/html", "value": body_html})
        if not payload["content"]:
            payload["content"].append({"type": "text/plain", "value": "(empty)"})

        body_bytes = json.dumps(payload).encode()
        req = urllib.request.Request(
            SENDGRID_URL,
            data=body_bytes,
            headers={
                "Authorization": f"Bearer {SENDGRID_API_KEY}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                sg_id = resp.headers.get("X-Message-Id", "")
            json_resp(self, 200, {"status": "sent", "to_email": to_email, "sg_message_id": sg_id})
        except urllib.error.HTTPError as exc:
            body_err = ""
            try:
                body_err = exc.read().decode()
            except Exception:
                pass
            error_resp(self, 502, f"SendGrid error {exc.code}: {body_err}")
        except Exception as exc:
            error_resp(self, 502, f"Send failed: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    init_db()

    sender = threading.Thread(target=_campaign_sender_loop, daemon=True)
    sender.start()

    server = HTTPServer(("0.0.0.0", PORT), CampaignHandler)
    print(f"[fm_email_campaign] Listening on port {PORT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
