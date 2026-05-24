#!/usr/bin/env python3
"""
FractalMesh OMEGA Titan — Multi-Channel Notification Dispatcher
Port: 7851
Channels: telegram, email, slack, log_only
DB: ~/fmsaas/database/sovereign.db (WAL mode)
"""

import json
import logging
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] fm-notifier %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("fm-notifier")

# ---------------------------------------------------------------------------
# Config from environment (no hardcoded credentials)
# ---------------------------------------------------------------------------
NOTIFIER_PORT = int(os.environ.get("NOTIFIER_PORT", 7851))
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
SENDGRID_API_KEY = os.environ.get("SENDGRID_API_KEY", "")
SENDGRID_FROM_EMAIL = os.environ.get("SENDGRID_FROM_EMAIL", "")
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")
ADMIN_SECRET = os.environ.get("ADMIN_SECRET", "")

DB_PATH = Path.home() / "fmsaas" / "database" / "sovereign.db"
DISPATCH_INTERVAL = 10  # seconds between dispatcher runs
MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_db()
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS notifications (
                id           INTEGER PRIMARY KEY,
                channel      TEXT,
                level        TEXT,
                subject      TEXT,
                body         TEXT,
                status       TEXT    DEFAULT 'pending',
                retry_count  INTEGER DEFAULT 0,
                sent_at      REAL,
                error        TEXT,
                created_at   REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS notification_rules (
                id            INTEGER PRIMARY KEY,
                event_pattern TEXT,
                channel       TEXT,
                level         TEXT,
                template      TEXT,
                enabled       INTEGER DEFAULT 1,
                created_at    REAL
            )
        """)
    conn.close()


# ---------------------------------------------------------------------------
# Channel dispatch helpers
# ---------------------------------------------------------------------------

def _send_telegram(subject: str, body: str, level: str) -> bool:
    """POST a message to Telegram via Bot API. Returns True on success."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram not configured (missing BOT_TOKEN or CHAT_ID)")
        return False

    text = f"<b>[{level.upper()}] FractalMesh</b>\n{subject}\n\n{body}"
    payload = json.dumps({
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
    }).encode()

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode())
            return result.get("ok", False)
    except Exception as exc:
        log.error("Telegram send failed: %s", exc)
        return False


def _send_email(subject: str, body: str, to: str = "") -> bool:
    """POST to SendGrid v3 /mail/send. Returns True on success."""
    if not SENDGRID_API_KEY or not SENDGRID_FROM_EMAIL:
        log.warning("Email not configured (missing SENDGRID_API_KEY or SENDGRID_FROM_EMAIL)")
        return False

    recipient = to if to else SENDGRID_FROM_EMAIL
    payload = json.dumps({
        "personalizations": [{"to": [{"email": recipient}]}],
        "from": {"email": SENDGRID_FROM_EMAIL},
        "subject": subject,
        "content": [{"type": "text/plain", "value": body}],
    }).encode()

    req = urllib.request.Request(
        "https://api.sendgrid.com/v3/mail/send",
        data=payload,
        headers={
            "Authorization": f"Bearer {SENDGRID_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status in (200, 202)
    except urllib.error.HTTPError as exc:
        log.error("SendGrid HTTP %s: %s", exc.code, exc.read().decode()[:200])
        return False
    except Exception as exc:
        log.error("Email send failed: %s", exc)
        return False


def _send_slack(subject: str, body: str, level: str) -> bool:
    """POST to Slack incoming webhook. Returns True on success."""
    if not SLACK_WEBHOOK_URL:
        log.warning("Slack not configured (missing SLACK_WEBHOOK_URL)")
        return False

    text = f"*[{level.upper()}] FractalMesh — {subject}*\n{body}"
    payload = json.dumps({"text": text}).encode()

    req = urllib.request.Request(
        SLACK_WEBHOOK_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.read().decode().strip() == "ok"
    except Exception as exc:
        log.error("Slack send failed: %s", exc)
        return False


def _dispatch(row: sqlite3.Row) -> tuple[bool, str]:
    """Dispatch a notification row to its channel. Returns (success, error_msg)."""
    channel = row["channel"]
    subject = row["subject"] or ""
    body = row["body"] or ""
    level = row["level"] or "info"
    to = ""  # email recipient stored in subject prefix convention — use body field has no to
    # 'to' address is not in schema; fall back to SENDGRID_FROM_EMAIL inside _send_email

    if channel == "telegram":
        ok = _send_telegram(subject, body, level)
    elif channel == "email":
        ok = _send_email(subject, body, to)
    elif channel == "slack":
        ok = _send_slack(subject, body, level)
    elif channel == "log_only":
        log.info("[log_only] [%s] %s — %s", level.upper(), subject, body)
        ok = True
    else:
        log.warning("Unknown channel: %s", channel)
        return False, f"unknown channel: {channel}"

    if ok:
        return True, ""
    return False, f"dispatch to {channel} returned failure"


# ---------------------------------------------------------------------------
# Background dispatcher thread
# ---------------------------------------------------------------------------

_db_lock = threading.Lock()


def _dispatcher_loop() -> None:
    log.info("Dispatcher thread started (interval=%ds)", DISPATCH_INTERVAL)
    while True:
        try:
            _process_pending()
        except Exception as exc:
            log.error("Dispatcher error: %s", exc)
        time.sleep(DISPATCH_INTERVAL)


def _process_pending() -> None:
    conn = get_db()
    try:
        with _db_lock:
            rows = conn.execute(
                "SELECT * FROM notifications WHERE status='pending' AND retry_count < ? ORDER BY created_at ASC",
                (MAX_RETRIES,),
            ).fetchall()

        for row in rows:
            nid = row["id"]
            success, error = _dispatch(row)
            now = time.time()
            with _db_lock:
                if success:
                    conn.execute(
                        "UPDATE notifications SET status='sent', sent_at=?, error=NULL WHERE id=?",
                        (now, nid),
                    )
                    conn.commit()
                    log.info("Notification %d sent via %s", nid, row["channel"])
                else:
                    new_retry = row["retry_count"] + 1
                    new_status = "failed" if new_retry >= MAX_RETRIES else "pending"
                    conn.execute(
                        "UPDATE notifications SET status=?, retry_count=?, error=? WHERE id=?",
                        (new_status, new_retry, error, nid),
                    )
                    conn.commit()
                    log.warning(
                        "Notification %d failed (retry %d/%d): %s",
                        nid, new_retry, MAX_RETRIES, error,
                    )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Rule seeding
# ---------------------------------------------------------------------------

DEFAULT_RULES = [
    ("security.*",       "telegram", "critical", "🚨 SECURITY: {subject}\n{body}"),
    ("agent.down",       "telegram", "error",    "⚠️ AGENT DOWN: {subject}\n{body}"),
    ("agent.down",       "email",    "error",    "Agent Down Alert: {subject}\n{body}"),
    ("revenue.milestone","telegram", "info",     "💰 REVENUE: {subject}\n{body}"),
    ("lead.new",         "telegram", "info",     "🎯 NEW LEAD: {subject}\n{body}"),
    ("license.issued",   "email",    "info",     "License Issued: {subject}\n{body}"),
]


def seed_rules() -> int:
    conn = get_db()
    seeded = 0
    now = time.time()
    with _db_lock:
        for pattern, channel, level, template in DEFAULT_RULES:
            cur = conn.execute(
                "SELECT id FROM notification_rules WHERE event_pattern=? AND channel=?",
                (pattern, channel),
            )
            if cur.fetchone() is None:
                conn.execute(
                    "INSERT INTO notification_rules (event_pattern, channel, level, template, enabled, created_at) VALUES (?,?,?,?,1,?)",
                    (pattern, channel, level, template, now),
                )
                seeded += 1
        conn.commit()
    conn.close()
    log.info("Seeded %d default notification rules", seeded)
    return seeded


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------

class NotifierHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):  # suppress default access log spam
        log.debug(fmt, *args)

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode())
        except json.JSONDecodeError:
            return {}

    def _send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _require_admin(self) -> bool:
        secret = self.headers.get("X-Admin-Secret", "")
        if not ADMIN_SECRET or secret != ADMIN_SECRET:
            self._send_json({"error": "unauthorized"}, 403)
            return False
        return True

    def _insert_notification(self, channel: str, level: str, subject: str,
                              body: str, status: str = "pending") -> int:
        conn = get_db()
        now = time.time()
        with _db_lock:
            cur = conn.execute(
                "INSERT INTO notifications (channel, level, subject, body, status, created_at) VALUES (?,?,?,?,?,?)",
                (channel, level, subject, body, status, now),
            )
            nid = cur.lastrowid
            conn.commit()
        conn.close()
        return nid

    # ------------------------------------------------------------------
    # GET routing
    # ------------------------------------------------------------------

    def do_GET(self):
        path = urllib.parse.urlparse(self.path)
        route = path.path.rstrip("/")
        query = dict(urllib.parse.parse_qsl(path.query))

        if route == "/health":
            self._handle_health()
        elif route == "/notifications":
            self._handle_list_notifications(query)
        elif route.startswith("/notifications/") and route.count("/") == 2:
            nid = route.split("/")[-1]
            self._handle_get_notification(nid)
        elif route == "/rules":
            self._handle_list_rules()
        elif route == "/analytics":
            self._handle_analytics()
        else:
            self._send_json({"error": "not found"}, 404)

    def _handle_health(self):
        channels = []
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            channels.append("telegram")
        if SENDGRID_API_KEY and SENDGRID_FROM_EMAIL:
            channels.append("email")
        if SLACK_WEBHOOK_URL:
            channels.append("slack")
        self._send_json({
            "status": "ok",
            "service": "fm-notifier",
            "port": NOTIFIER_PORT,
            "channels": channels,
        })

    def _handle_list_notifications(self, query: dict):
        filters = []
        params = []
        if "status" in query:
            filters.append("status=?")
            params.append(query["status"])
        if "channel" in query:
            filters.append("channel=?")
            params.append(query["channel"])
        limit = int(query.get("limit", 50))
        where = ("WHERE " + " AND ".join(filters)) if filters else ""
        sql = f"SELECT * FROM notifications {where} ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        conn = get_db()
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        self._send_json({"notifications": [dict(r) for r in rows], "count": len(rows)})

    def _handle_get_notification(self, nid: str):
        if not nid.isdigit():
            self._send_json({"error": "invalid id"}, 400)
            return
        conn = get_db()
        row = conn.execute("SELECT * FROM notifications WHERE id=?", (int(nid),)).fetchone()
        conn.close()
        if row is None:
            self._send_json({"error": "not found"}, 404)
        else:
            self._send_json(dict(row))

    def _handle_list_rules(self):
        conn = get_db()
        rows = conn.execute("SELECT * FROM notification_rules ORDER BY id").fetchall()
        conn.close()
        self._send_json({"rules": [dict(r) for r in rows], "count": len(rows)})

    def _handle_analytics(self):
        conn = get_db()
        day_start = time.time() - 86400
        sent_today = conn.execute(
            "SELECT COUNT(*) FROM notifications WHERE status='sent' AND sent_at>=?",
            (day_start,),
        ).fetchone()[0]

        by_channel = {}
        for r in conn.execute(
            "SELECT channel, COUNT(*) as cnt FROM notifications WHERE status='sent' GROUP BY channel"
        ).fetchall():
            by_channel[r["channel"]] = r["cnt"]

        by_level = {}
        for r in conn.execute(
            "SELECT level, COUNT(*) as cnt FROM notifications WHERE status='sent' GROUP BY level"
        ).fetchall():
            by_level[r["level"]] = r["cnt"]

        total = conn.execute("SELECT COUNT(*) FROM notifications").fetchone()[0]
        failed = conn.execute("SELECT COUNT(*) FROM notifications WHERE status='failed'").fetchone()[0]
        failure_rate = round(failed / total, 4) if total else 0.0

        avg_retry = conn.execute("SELECT AVG(retry_count) FROM notifications").fetchone()[0] or 0.0
        conn.close()

        self._send_json({
            "notifications_sent_today": sent_today,
            "by_channel": by_channel,
            "by_level": by_level,
            "failure_rate": failure_rate,
            "avg_retry_count": round(avg_retry, 3),
        })

    # ------------------------------------------------------------------
    # POST routing
    # ------------------------------------------------------------------

    def do_POST(self):
        path = urllib.parse.urlparse(self.path)
        route = path.path.rstrip("/")

        if route == "/notify":
            self._handle_notify()
        elif route == "/notify/alert":
            self._handle_notify_alert()
        elif route == "/notify/batch":
            self._handle_notify_batch()
        elif route.startswith("/notifications/") and route.endswith("/retry"):
            parts = route.split("/")
            if len(parts) == 4 and parts[2].isdigit():
                self._handle_retry(parts[2])
            else:
                self._send_json({"error": "invalid path"}, 400)
        elif route == "/rules/create":
            self._handle_rules_create()
        elif route == "/rules/seed":
            self._handle_rules_seed()
        elif route == "/send/telegram":
            self._handle_send_telegram_direct()
        else:
            self._send_json({"error": "not found"}, 404)

    def _handle_notify(self):
        data = self._read_json()
        channel = data.get("channel", "log_only")
        level = data.get("level", "info")
        subject = data.get("subject", "")
        body = data.get("body", "")
        nid = self._insert_notification(channel, level, subject, body)
        self._send_json({"notification_id": nid, "status": "pending"}, 201)

    def _handle_notify_alert(self):
        data = self._read_json()
        level = data.get("level", "critical")
        subject = data.get("subject", "Alert")
        body = data.get("body", "")

        sent_to = []
        results = {}

        def attempt(channel):
            if channel == "telegram" and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                ok = _send_telegram(subject, body, level)
                results[channel] = ok
            elif channel == "email" and SENDGRID_API_KEY and SENDGRID_FROM_EMAIL:
                ok = _send_email(subject, body)
                results[channel] = ok
            elif channel == "slack" and SLACK_WEBHOOK_URL:
                ok = _send_slack(subject, body, level)
                results[channel] = ok

        threads = []
        for ch in ("telegram", "email", "slack"):
            t = threading.Thread(target=attempt, args=(ch,), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join(timeout=20)

        now = time.time()
        conn = get_db()
        with _db_lock:
            for channel, ok in results.items():
                status = "sent" if ok else "failed"
                conn.execute(
                    "INSERT INTO notifications (channel, level, subject, body, status, sent_at, created_at) VALUES (?,?,?,?,?,?,?)",
                    (channel, level, subject, body, status, now if ok else None, now),
                )
                if ok:
                    sent_to.append(channel)
            conn.commit()
        conn.close()

        self._send_json({"sent_to": sent_to, "results": results})

    def _handle_notify_batch(self):
        data = self._read_json()
        notifications = data.get("notifications", [])
        if not isinstance(notifications, list):
            self._send_json({"error": "notifications must be a list"}, 400)
            return
        conn = get_db()
        now = time.time()
        count = 0
        with _db_lock:
            for n in notifications:
                channel = n.get("channel", "log_only")
                level = n.get("level", "info")
                subject = n.get("subject", "")
                body = n.get("body", "")
                conn.execute(
                    "INSERT INTO notifications (channel, level, subject, body, status, created_at) VALUES (?,?,?,?,?,?)",
                    (channel, level, subject, body, "pending", now),
                )
                count += 1
            conn.commit()
        conn.close()
        self._send_json({"queued": count}, 201)

    def _handle_retry(self, nid: str):
        conn = get_db()
        row = conn.execute("SELECT id FROM notifications WHERE id=?", (int(nid),)).fetchone()
        if row is None:
            conn.close()
            self._send_json({"error": "not found"}, 404)
            return
        with _db_lock:
            conn.execute(
                "UPDATE notifications SET status='pending', retry_count=0, error=NULL WHERE id=?",
                (int(nid),),
            )
            conn.commit()
        conn.close()
        self._send_json({"queued": True})

    def _handle_rules_create(self):
        if not self._require_admin():
            return
        data = self._read_json()
        pattern = data.get("event_pattern", "")
        channel = data.get("channel", "")
        level = data.get("level", "info")
        template = data.get("template", "{subject}\n{body}")
        if not pattern or not channel:
            self._send_json({"error": "event_pattern and channel required"}, 400)
            return
        conn = get_db()
        now = time.time()
        with _db_lock:
            cur = conn.execute(
                "INSERT INTO notification_rules (event_pattern, channel, level, template, enabled, created_at) VALUES (?,?,?,?,1,?)",
                (pattern, channel, level, template, now),
            )
            rule_id = cur.lastrowid
            conn.commit()
        conn.close()
        self._send_json({"rule_id": rule_id}, 201)

    def _handle_rules_seed(self):
        seeded = seed_rules()
        self._send_json({"seeded": seeded})

    def _handle_send_telegram_direct(self):
        data = self._read_json()
        text = data.get("text", "")
        parse_mode = data.get("parse_mode", "HTML")
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            self._send_json({"error": "Telegram not configured"}, 503)
            return

        payload = json.dumps({
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": parse_mode,
        }).encode()

        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode())
                if result.get("ok"):
                    msg_id = result.get("result", {}).get("message_id")
                    self._send_json({"sent": True, "message_id": msg_id})
                else:
                    self._send_json({"sent": False, "error": result.get("description", "unknown")}, 502)
        except Exception as exc:
            self._send_json({"sent": False, "error": str(exc)}, 502)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def startup_telegram_ping() -> None:
    """Send startup notification to Telegram if configured."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    ok = _send_telegram(
        subject="System Online",
        body="96 agents active",
        level="info",
    )
    # Override text for startup to match spec exactly
    payload = json.dumps({
        "chat_id": TELEGRAM_CHAT_ID,
        "text": "🚀 FractalMesh OMEGA Titan online — 96 agents active",
        "parse_mode": "HTML",
    }).encode()
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode())
            if result.get("ok"):
                log.info("Startup Telegram ping sent")
            else:
                log.warning("Startup Telegram ping failed: %s", result)
    except Exception as exc:
        log.warning("Startup Telegram ping error: %s", exc)


def main() -> None:
    log.info("Initialising fm-notifier on port %d", NOTIFIER_PORT)
    init_db()
    seed_rules()

    dispatcher = threading.Thread(target=_dispatcher_loop, daemon=True, name="dispatcher")
    dispatcher.start()

    startup_telegram_ping()

    server = HTTPServer(("0.0.0.0", NOTIFIER_PORT), NotifierHandler)
    log.info("fm-notifier listening on 0.0.0.0:%d", NOTIFIER_PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("fm-notifier shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
