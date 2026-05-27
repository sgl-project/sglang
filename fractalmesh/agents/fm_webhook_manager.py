#!/usr/bin/env python3
"""
fm_webhook_manager.py — Webhook & Event Bus Manager (Port 7889)
Central webhook registry and event dispatcher for FractalMesh OMEGA Titan.
External services register webhook endpoints; when FractalMesh fires events,
this agent fans out to all matching subscriptions with HMAC-signed payloads,
handles retries with exponential backoff, logs delivery history.
Credentials sourced from ~/.secrets/fractal.env at runtime — never hardcoded.
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import json
import time
import hmac
import hashlib
import secrets
import base64
import sqlite3
import threading
import urllib.request
import urllib.error
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

# ── vault loading (MUST be before any os.getenv calls) ─────────────────────────
_ENV_FILE = Path.home() / ".secrets" / "fractal.env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

# ── config ─────────────────────────────────────────────────────────────────────
PORT         = int(os.getenv("WEBHOOK_MANAGER_PORT", "7889"))
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")
ROOT         = Path(os.getenv("FRACTALMESH_HOME", str(Path.home() / "fmsaas")))
DB_PATH      = ROOT / "database" / "sovereign.db"
LOG_DIR      = ROOT / "logs"

ROOT.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# retry backoff intervals in seconds
RETRY_BACKOFFS = [5, 25, 125, 625, 3125]
MAX_ATTEMPTS   = 5
RETRY_DAEMON_SLEEP = 30
MAX_RESPONSE_BODY  = 1000

# ── database ───────────────────────────────────────────────────────────────────

def _get_db() -> sqlite3.Connection:
    """Open a WAL-mode connection to sovereign.db."""
    con = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    con.row_factory = sqlite3.Row
    return con


def _init_db() -> None:
    """Create tables if they do not exist."""
    with _get_db() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS subscriptions (
                id            INTEGER PRIMARY KEY,
                sub_id        TEXT UNIQUE NOT NULL,
                name          TEXT NOT NULL,
                endpoint_url  TEXT NOT NULL,
                secret        TEXT NOT NULL,
                events        TEXT DEFAULT '[]',
                status        TEXT DEFAULT 'active',
                created_by    TEXT,
                retry_count   INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                created_at    REAL,
                updated_at    REAL
            );

            CREATE TABLE IF NOT EXISTS events (
                id          INTEGER PRIMARY KEY,
                event_id    TEXT UNIQUE NOT NULL,
                event_type  TEXT NOT NULL,
                payload     TEXT,
                source      TEXT,
                created_at  REAL
            );

            CREATE TABLE IF NOT EXISTS deliveries (
                id            INTEGER PRIMARY KEY,
                delivery_id   TEXT UNIQUE NOT NULL,
                event_id      TEXT NOT NULL,
                sub_id        TEXT NOT NULL,
                status        TEXT DEFAULT 'pending',
                http_status   INTEGER,
                response_body TEXT,
                attempt       INTEGER DEFAULT 1,
                next_retry_at REAL,
                delivered_at  REAL,
                created_at    REAL
            );

            CREATE TABLE IF NOT EXISTS inbound_hooks (
                id                INTEGER PRIMARY KEY,
                hook_id           TEXT UNIQUE NOT NULL,
                name              TEXT NOT NULL,
                path              TEXT UNIQUE NOT NULL,
                secret            TEXT NOT NULL,
                event_type_prefix TEXT,
                created_at        REAL,
                payload_count     INTEGER DEFAULT 0
            );
        """)


# ── HMAC helpers ───────────────────────────────────────────────────────────────

def _sign_payload(secret: str, body: bytes) -> str:
    """Return sha256=<hex> HMAC signature for outbound delivery."""
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _verify_admin(headers) -> bool:
    """Check X-Admin-Secret header with constant-time compare."""
    if not ADMIN_SECRET:
        return False
    provided = headers.get("X-Admin-Secret", "")
    return hmac.compare_digest(provided, ADMIN_SECRET)


def _verify_inbound_signature(secret: str, body: bytes, sig_header: str) -> bool:
    """Verify X-Hub-Signature-256 or X-Signature header."""
    if not sig_header:
        return False
    expected = _sign_payload(secret, body)
    # support both sha256=... and raw hex
    if sig_header.startswith("sha256="):
        return hmac.compare_digest(sig_header, expected)
    # raw hex fallback
    raw_expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(sig_header, raw_expected)


# ── event pattern matching ─────────────────────────────────────────────────────

def _event_matches(pattern: str, event_type: str) -> bool:
    """
    Match event_type against pattern.
    Supports:
      - exact match: "payment.created"
      - wildcard suffix: "payment.*"
      - match-all: "*"
    """
    if pattern == "*":
        return True
    if pattern == event_type:
        return True
    if pattern.endswith(".*"):
        prefix = pattern[:-2]
        return event_type.startswith(prefix + ".")
    return False


def _subscription_matches(sub_events_json: str, event_type: str) -> bool:
    """Return True if any pattern in sub_events matches event_type."""
    try:
        patterns = json.loads(sub_events_json)
    except (json.JSONDecodeError, TypeError):
        patterns = []
    if not patterns:
        return True  # empty list = subscribe to all
    return any(_event_matches(p, event_type) for p in patterns)


# ── delivery ───────────────────────────────────────────────────────────────────

def _attempt_delivery(delivery_id: str, endpoint_url: str, secret: str,
                      body_bytes: bytes, attempt: int) -> tuple:
    """
    Try to POST body_bytes to endpoint_url.
    Returns (success: bool, http_status: int or None, response_body: str).
    """
    signature = _sign_payload(secret, body_bytes)
    req = urllib.request.Request(
        url=endpoint_url,
        data=body_bytes,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-FractalMesh-Signature": signature,
            "X-Delivery-ID": delivery_id,
            "X-Attempt": str(attempt),
            "User-Agent": "FractalMesh-WebhookManager/1.0",
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            http_status = resp.status
            response_body = resp.read(MAX_RESPONSE_BODY).decode("utf-8", errors="replace")
            success = 200 <= http_status < 300
            return success, http_status, response_body[:MAX_RESPONSE_BODY]
    except urllib.error.HTTPError as exc:
        http_status = exc.code
        try:
            response_body = exc.read(MAX_RESPONSE_BODY).decode("utf-8", errors="replace")
        except Exception:
            response_body = str(exc)
        return False, http_status, response_body[:MAX_RESPONSE_BODY]
    except urllib.error.URLError as exc:
        return False, None, str(exc)[:MAX_RESPONSE_BODY]
    except Exception as exc:
        return False, None, str(exc)[:MAX_RESPONSE_BODY]


def _perform_delivery(delivery_id: str) -> None:
    """
    Execute one delivery attempt. Updates the deliveries row and subscription stats.
    Schedules next retry on failure.
    """
    with _get_db() as con:
        row = con.execute(
            "SELECT d.*, s.endpoint_url, s.secret "
            "FROM deliveries d JOIN subscriptions s ON d.sub_id=s.sub_id "
            "WHERE d.delivery_id=?",
            (delivery_id,)
        ).fetchone()
        if not row:
            return

        ev_row = con.execute(
            "SELECT payload FROM events WHERE event_id=?",
            (row["event_id"],)
        ).fetchone()
        if not ev_row:
            return

        body_bytes = ev_row["payload"].encode("utf-8")
        attempt    = row["attempt"]
        endpoint   = row["endpoint_url"]
        secret     = row["secret"]
        sub_id     = row["sub_id"]

        success, http_status, response_body = _attempt_delivery(
            delivery_id, endpoint, secret, body_bytes, attempt
        )
        now = time.time()

        if success:
            con.execute(
                "UPDATE deliveries SET status='delivered', http_status=?, "
                "response_body=?, delivered_at=? WHERE delivery_id=?",
                (http_status, response_body, now, delivery_id)
            )
            con.execute(
                "UPDATE subscriptions SET success_count=success_count+1, updated_at=? "
                "WHERE sub_id=?",
                (now, sub_id)
            )
        else:
            if attempt >= MAX_ATTEMPTS:
                con.execute(
                    "UPDATE deliveries SET status='failed', http_status=?, "
                    "response_body=?, attempt=? WHERE delivery_id=?",
                    (http_status, response_body, attempt, delivery_id)
                )
                con.execute(
                    "UPDATE subscriptions SET failure_count=failure_count+1, "
                    "retry_count=retry_count+1, updated_at=? WHERE sub_id=?",
                    (now, sub_id)
                )
            else:
                backoff = RETRY_BACKOFFS[attempt - 1] if attempt - 1 < len(RETRY_BACKOFFS) else RETRY_BACKOFFS[-1]
                next_retry = now + backoff
                con.execute(
                    "UPDATE deliveries SET status='pending', http_status=?, "
                    "response_body=?, attempt=?, next_retry_at=? WHERE delivery_id=?",
                    (http_status, response_body, attempt + 1, next_retry, delivery_id)
                )
                con.execute(
                    "UPDATE subscriptions SET retry_count=retry_count+1, updated_at=? "
                    "WHERE sub_id=?",
                    (now, sub_id)
                )


def _dispatch_event(event_id: str, event_type: str, payload_json: str) -> int:
    """
    Find all active subscriptions matching event_type, create delivery rows,
    and kick off background delivery threads.
    Returns the number of subscriptions targeted.
    """
    with _get_db() as con:
        subs = con.execute(
            "SELECT sub_id, events FROM subscriptions WHERE status='active'"
        ).fetchall()

    matching = [s for s in subs if _subscription_matches(s["events"], event_type)]
    now = time.time()

    delivery_ids = []
    with _get_db() as con:
        for sub in matching:
            delivery_id = "dlv_" + secrets.token_hex(16)
            con.execute(
                "INSERT INTO deliveries "
                "(delivery_id, event_id, sub_id, status, attempt, next_retry_at, created_at) "
                "VALUES (?,?,?,'pending',1,?,?)",
                (delivery_id, event_id, sub["sub_id"], now, now)
            )
            delivery_ids.append(delivery_id)

    for did in delivery_ids:
        t = threading.Thread(target=_perform_delivery, args=(did,), daemon=True)
        t.start()

    return len(delivery_ids)


# ── background retry daemon ────────────────────────────────────────────────────

def _retry_daemon() -> None:
    """Daemon thread: every 30 s find pending deliveries whose retry time has come."""
    while True:
        try:
            now = time.time()
            with _get_db() as con:
                rows = con.execute(
                    "SELECT delivery_id FROM deliveries "
                    "WHERE status='pending' AND next_retry_at <= ?",
                    (now,)
                ).fetchall()
            for row in rows:
                t = threading.Thread(
                    target=_perform_delivery, args=(row["delivery_id"],), daemon=True
                )
                t.start()
        except Exception:
            pass
        time.sleep(RETRY_DAEMON_SLEEP)


# ── HTTP handler ───────────────────────────────────────────────────────────────

class WebhookManagerHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the Webhook & Event Bus Manager."""

    # ── util ────────────────────────────────────────────────────────────────────

    def _send_json(self, data, status: int = 200) -> None:
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, message: str, status: int = 400) -> None:
        self._send_json({"error": message}, status)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length > 0 else b""

    def _parse_json_body(self):
        raw = self._read_body()
        if not raw:
            return None, raw
        try:
            return json.loads(raw), raw
        except json.JSONDecodeError as exc:
            return None, raw

    def _path_parts(self):
        """Return path split into non-empty parts."""
        return [p for p in self.path.split("?")[0].split("/") if p]

    def _require_admin(self) -> bool:
        if not _verify_admin(self.headers):
            self._send_error("Forbidden: invalid or missing X-Admin-Secret", 403)
            return False
        return True

    def log_message(self, fmt, *args):
        pass  # suppress default stderr logging

    # ── routing ─────────────────────────────────────────────────────────────────

    def do_GET(self):
        parts = self._path_parts()
        try:
            if not parts or parts[0] == "health":
                self._handle_health()
            elif parts[0] == "subscriptions" and len(parts) == 1:
                self._handle_list_subscriptions()
            elif parts[0] == "subscriptions" and len(parts) == 2:
                self._handle_get_subscription(parts[1])
            elif parts[0] == "events" and len(parts) == 1:
                self._handle_list_events()
            elif parts[0] == "events" and len(parts) == 2:
                self._handle_get_event(parts[1])
            elif parts[0] == "deliveries" and len(parts) == 1:
                self._handle_list_deliveries()
            elif parts[0] == "deliveries" and len(parts) == 2:
                self._handle_sub_deliveries(parts[1])
            elif parts[0] == "stats" and len(parts) == 1:
                self._handle_stats()
            else:
                self._send_error("Not found", 404)
        except Exception as exc:
            self._send_error(f"Internal error: {exc}", 500)

    def do_POST(self):
        parts = self._path_parts()
        try:
            if parts[0] == "subscriptions" and len(parts) == 1:
                self._handle_create_subscription()
            elif parts[0] == "events" and len(parts) == 1:
                self._handle_fire_event()
            elif parts[0] == "inbound" and len(parts) == 1:
                self._handle_register_inbound()
            elif parts[0] == "receive" and len(parts) == 2:
                self._handle_receive_inbound(parts[1])
            elif parts[0] == "retry" and len(parts) == 2:
                self._handle_retry_delivery(parts[1])
            else:
                self._send_error("Not found", 404)
        except Exception as exc:
            self._send_error(f"Internal error: {exc}", 500)

    def do_PUT(self):
        parts = self._path_parts()
        try:
            if parts[0] == "subscriptions" and len(parts) == 2:
                self._handle_update_subscription(parts[1])
            else:
                self._send_error("Not found", 404)
        except Exception as exc:
            self._send_error(f"Internal error: {exc}", 500)

    def do_DELETE(self):
        parts = self._path_parts()
        try:
            if parts[0] == "subscriptions" and len(parts) == 2:
                self._handle_delete_subscription(parts[1])
            else:
                self._send_error("Not found", 404)
        except Exception as exc:
            self._send_error(f"Internal error: {exc}", 500)

    # ── GET /health ─────────────────────────────────────────────────────────────

    def _handle_health(self):
        with _get_db() as con:
            sub_count = con.execute(
                "SELECT COUNT(*) FROM subscriptions WHERE status='active'"
            ).fetchone()[0]
            pending = con.execute(
                "SELECT COUNT(*) FROM deliveries WHERE status='pending'"
            ).fetchone()[0]
        self._send_json({
            "status": "ok",
            "service": "fm_webhook_manager",
            "port": PORT,
            "active_subscriptions": sub_count,
            "pending_deliveries": pending,
            "timestamp": time.time(),
        })

    # ── POST /subscriptions ─────────────────────────────────────────────────────

    def _handle_create_subscription(self):
        data, _ = self._parse_json_body()
        if data is None:
            self._send_error("Invalid JSON body")
            return
        name         = data.get("name", "").strip()
        endpoint_url = data.get("endpoint_url", "").strip()
        events_list  = data.get("events", [])
        created_by   = data.get("created_by", "")

        if not name:
            self._send_error("'name' is required")
            return
        if not endpoint_url:
            self._send_error("'endpoint_url' is required")
            return
        if not isinstance(events_list, list):
            self._send_error("'events' must be a list")
            return

        sub_id      = "sub_" + secrets.token_hex(16)
        sub_secret  = secrets.token_hex(32)
        events_json = json.dumps(events_list)
        now         = time.time()

        with _get_db() as con:
            con.execute(
                "INSERT INTO subscriptions "
                "(sub_id, name, endpoint_url, secret, events, status, created_by, created_at, updated_at) "
                "VALUES (?,?,?,?,?,'active',?,?,?)",
                (sub_id, name, endpoint_url, sub_secret, events_json, created_by, now, now)
            )

        self._send_json({
            "sub_id": sub_id,
            "name": name,
            "endpoint_url": endpoint_url,
            "events": events_list,
            "secret": sub_secret,
            "status": "active",
            "created_by": created_by,
            "created_at": now,
            "message": "Subscription created. Store the secret securely — it will not be shown again.",
        }, 201)

    # ── GET /subscriptions ──────────────────────────────────────────────────────

    def _handle_list_subscriptions(self):
        if not self._require_admin():
            return
        with _get_db() as con:
            rows = con.execute(
                "SELECT sub_id, name, endpoint_url, events, status, created_by, "
                "retry_count, success_count, failure_count, created_at, updated_at "
                "FROM subscriptions ORDER BY created_at DESC"
            ).fetchall()
        self._send_json({"subscriptions": [dict(r) for r in rows]})

    # ── GET /subscriptions/{sub_id} ─────────────────────────────────────────────

    def _handle_get_subscription(self, sub_id: str):
        with _get_db() as con:
            row = con.execute(
                "SELECT sub_id, name, endpoint_url, events, status, created_by, "
                "retry_count, success_count, failure_count, created_at, updated_at "
                "FROM subscriptions WHERE sub_id=?",
                (sub_id,)
            ).fetchone()
        if not row:
            self._send_error("Subscription not found", 404)
            return
        self._send_json(dict(row))

    # ── PUT /subscriptions/{sub_id} ─────────────────────────────────────────────

    def _handle_update_subscription(self, sub_id: str):
        data, _ = self._parse_json_body()
        if data is None:
            self._send_error("Invalid JSON body")
            return
        with _get_db() as con:
            row = con.execute(
                "SELECT sub_id, endpoint_url, events FROM subscriptions WHERE sub_id=?",
                (sub_id,)
            ).fetchone()
            if not row:
                self._send_error("Subscription not found", 404)
                return

            new_url    = data.get("endpoint_url", row["endpoint_url"])
            new_events = data.get("events", json.loads(row["events"]))
            if not isinstance(new_events, list):
                self._send_error("'events' must be a list")
                return
            now = time.time()
            con.execute(
                "UPDATE subscriptions SET endpoint_url=?, events=?, updated_at=? "
                "WHERE sub_id=?",
                (new_url, json.dumps(new_events), now, sub_id)
            )
            updated = con.execute(
                "SELECT sub_id, name, endpoint_url, events, status, created_by, "
                "retry_count, success_count, failure_count, created_at, updated_at "
                "FROM subscriptions WHERE sub_id=?",
                (sub_id,)
            ).fetchone()
        self._send_json(dict(updated))

    # ── DELETE /subscriptions/{sub_id} ──────────────────────────────────────────

    def _handle_delete_subscription(self, sub_id: str):
        with _get_db() as con:
            row = con.execute(
                "SELECT sub_id FROM subscriptions WHERE sub_id=?", (sub_id,)
            ).fetchone()
            if not row:
                self._send_error("Subscription not found", 404)
                return
            now = time.time()
            con.execute(
                "UPDATE subscriptions SET status='inactive', updated_at=? WHERE sub_id=?",
                (now, sub_id)
            )
        self._send_json({"sub_id": sub_id, "status": "inactive", "message": "Subscription deactivated"})

    # ── POST /events ────────────────────────────────────────────────────────────

    def _handle_fire_event(self):
        data, raw = self._parse_json_body()
        if data is None:
            self._send_error("Invalid JSON body")
            return
        event_type = data.get("event_type", "").strip()
        payload    = data.get("payload", {})
        source     = data.get("source", "")

        if not event_type:
            self._send_error("'event_type' is required")
            return

        event_id    = "evt_" + secrets.token_hex(16)
        now         = time.time()
        payload_json = json.dumps({
            "event_id":   event_id,
            "event_type": event_type,
            "payload":    payload,
            "source":     source,
            "created_at": now,
        })

        with _get_db() as con:
            con.execute(
                "INSERT INTO events (event_id, event_type, payload, source, created_at) "
                "VALUES (?,?,?,?,?)",
                (event_id, event_type, payload_json, source, now)
            )

        dispatched = _dispatch_event(event_id, event_type, payload_json)

        self._send_json({
            "event_id":         event_id,
            "event_type":       event_type,
            "source":           source,
            "subscriptions_notified": dispatched,
            "created_at":       now,
        }, 202)

    # ── GET /events ─────────────────────────────────────────────────────────────

    def _handle_list_events(self):
        if not self._require_admin():
            return
        with _get_db() as con:
            rows = con.execute(
                "SELECT event_id, event_type, source, created_at "
                "FROM events ORDER BY created_at DESC LIMIT 100"
            ).fetchall()
        self._send_json({"events": [dict(r) for r in rows]})

    # ── GET /events/{event_id} ──────────────────────────────────────────────────

    def _handle_get_event(self, event_id: str):
        with _get_db() as con:
            ev = con.execute(
                "SELECT event_id, event_type, payload, source, created_at "
                "FROM events WHERE event_id=?",
                (event_id,)
            ).fetchone()
            if not ev:
                self._send_error("Event not found", 404)
                return
            deliveries = con.execute(
                "SELECT delivery_id, sub_id, status, http_status, attempt, "
                "next_retry_at, delivered_at, created_at "
                "FROM deliveries WHERE event_id=? ORDER BY created_at",
                (event_id,)
            ).fetchall()

        ev_dict = dict(ev)
        try:
            ev_dict["payload"] = json.loads(ev_dict["payload"])
        except Exception:
            pass
        self._send_json({
            "event": ev_dict,
            "deliveries": [dict(d) for d in deliveries],
        })

    # ── GET /deliveries ─────────────────────────────────────────────────────────

    def _handle_list_deliveries(self):
        if not self._require_admin():
            return
        with _get_db() as con:
            rows = con.execute(
                "SELECT delivery_id, event_id, sub_id, status, http_status, "
                "attempt, next_retry_at, delivered_at, created_at "
                "FROM deliveries ORDER BY created_at DESC LIMIT 100"
            ).fetchall()
        self._send_json({"deliveries": [dict(r) for r in rows]})

    # ── GET /deliveries/{sub_id} ────────────────────────────────────────────────

    def _handle_sub_deliveries(self, sub_id: str):
        with _get_db() as con:
            rows = con.execute(
                "SELECT delivery_id, event_id, sub_id, status, http_status, "
                "attempt, next_retry_at, delivered_at, created_at "
                "FROM deliveries WHERE sub_id=? ORDER BY created_at DESC LIMIT 200",
                (sub_id,)
            ).fetchall()
        self._send_json({"sub_id": sub_id, "deliveries": [dict(r) for r in rows]})

    # ── POST /inbound ───────────────────────────────────────────────────────────

    def _handle_register_inbound(self):
        data, _ = self._parse_json_body()
        if data is None:
            self._send_error("Invalid JSON body")
            return
        name              = data.get("name", "").strip()
        event_type_prefix = data.get("event_type_prefix", "inbound").strip()
        path_override     = data.get("path", "").strip()

        if not name:
            self._send_error("'name' is required")
            return

        hook_id    = "ihk_" + secrets.token_hex(16)
        hook_path  = path_override if path_override else ("/receive/" + hook_id)
        hook_secret = secrets.token_hex(32)
        now        = time.time()

        with _get_db() as con:
            # check path uniqueness
            existing = con.execute(
                "SELECT hook_id FROM inbound_hooks WHERE path=?", (hook_path,)
            ).fetchone()
            if existing:
                self._send_error(f"Path '{hook_path}' is already registered", 409)
                return
            con.execute(
                "INSERT INTO inbound_hooks "
                "(hook_id, name, path, secret, event_type_prefix, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (hook_id, name, hook_path, hook_secret, event_type_prefix, now)
            )

        self._send_json({
            "hook_id":           hook_id,
            "name":              name,
            "path":              hook_path,
            "secret":            hook_secret,
            "event_type_prefix": event_type_prefix,
            "created_at":        now,
            "message": "Store the secret securely. Sign payloads with X-Hub-Signature-256 or X-Signature.",
        }, 201)

    # ── POST /receive/{hook_id} ─────────────────────────────────────────────────

    def _handle_receive_inbound(self, hook_id: str):
        raw = self._read_body()
        with _get_db() as con:
            hook = con.execute(
                "SELECT hook_id, secret, event_type_prefix, path "
                "FROM inbound_hooks WHERE hook_id=?",
                (hook_id,)
            ).fetchone()
        if not hook:
            self._send_error("Inbound hook not found", 404)
            return

        # verify signature
        sig = (self.headers.get("X-Hub-Signature-256") or
               self.headers.get("X-Signature") or "")
        if not _verify_inbound_signature(hook["secret"], raw, sig):
            self._send_error("Signature verification failed", 401)
            return

        # parse payload
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {"raw": raw.decode("utf-8", errors="replace")}

        # determine event type
        inbound_event_type = hook["event_type_prefix"] + ".received"

        event_id   = "evt_" + secrets.token_hex(16)
        now        = time.time()
        payload_json = json.dumps({
            "event_id":   event_id,
            "event_type": inbound_event_type,
            "payload":    payload,
            "source":     f"inbound:{hook_id}",
            "created_at": now,
        })

        with _get_db() as con:
            con.execute(
                "INSERT INTO events (event_id, event_type, payload, source, created_at) "
                "VALUES (?,?,?,?,?)",
                (event_id, inbound_event_type, payload_json,
                 f"inbound:{hook_id}", now)
            )
            con.execute(
                "UPDATE inbound_hooks SET payload_count=payload_count+1 "
                "WHERE hook_id=?",
                (hook_id,)
            )

        dispatched = _dispatch_event(event_id, inbound_event_type, payload_json)

        self._send_json({
            "event_id":               event_id,
            "event_type":             inbound_event_type,
            "subscriptions_notified": dispatched,
            "accepted_at":            now,
        }, 202)

    # ── POST /retry/{delivery_id} ───────────────────────────────────────────────

    def _handle_retry_delivery(self, delivery_id: str):
        if not self._require_admin():
            return
        with _get_db() as con:
            row = con.execute(
                "SELECT delivery_id, status FROM deliveries WHERE delivery_id=?",
                (delivery_id,)
            ).fetchone()
        if not row:
            self._send_error("Delivery not found", 404)
            return

        # reset to pending with next_retry_at = now so daemon picks it up,
        # but also do it immediately in a background thread
        now = time.time()
        with _get_db() as con:
            con.execute(
                "UPDATE deliveries SET status='pending', next_retry_at=?, attempt=1 "
                "WHERE delivery_id=?",
                (now, delivery_id)
            )
        t = threading.Thread(target=_perform_delivery, args=(delivery_id,), daemon=True)
        t.start()
        self._send_json({
            "delivery_id": delivery_id,
            "message":     "Retry initiated",
            "triggered_at": now,
        })

    # ── GET /stats ──────────────────────────────────────────────────────────────

    def _handle_stats(self):
        if not self._require_admin():
            return
        with _get_db() as con:
            total_subs = con.execute(
                "SELECT COUNT(*) FROM subscriptions"
            ).fetchone()[0]
            active_subs = con.execute(
                "SELECT COUNT(*) FROM subscriptions WHERE status='active'"
            ).fetchone()[0]
            total_events = con.execute(
                "SELECT COUNT(*) FROM events"
            ).fetchone()[0]
            total_deliveries = con.execute(
                "SELECT COUNT(*) FROM deliveries"
            ).fetchone()[0]
            delivered = con.execute(
                "SELECT COUNT(*) FROM deliveries WHERE status='delivered'"
            ).fetchone()[0]
            failed = con.execute(
                "SELECT COUNT(*) FROM deliveries WHERE status='failed'"
            ).fetchone()[0]
            pending = con.execute(
                "SELECT COUNT(*) FROM deliveries WHERE status='pending'"
            ).fetchone()[0]
            total_inbound = con.execute(
                "SELECT COUNT(*) FROM inbound_hooks"
            ).fetchone()[0]
            inbound_payloads = con.execute(
                "SELECT COALESCE(SUM(payload_count),0) FROM inbound_hooks"
            ).fetchone()[0]

        success_rate = 0.0
        if total_deliveries > 0:
            success_rate = round(delivered / total_deliveries * 100, 2)

        self._send_json({
            "total_subscriptions":  total_subs,
            "active_subscriptions": active_subs,
            "total_events":         total_events,
            "total_deliveries":     total_deliveries,
            "delivered":            delivered,
            "failed":               failed,
            "pending_deliveries":   pending,
            "delivery_success_rate_pct": success_rate,
            "inbound_hooks":        total_inbound,
            "inbound_payloads_received": inbound_payloads,
            "timestamp":            time.time(),
        })


# ── server bootstrap ───────────────────────────────────────────────────────────

def _start_retry_daemon() -> None:
    """Start the background retry daemon thread."""
    t = threading.Thread(target=_retry_daemon, name="retry-daemon", daemon=True)
    t.start()


def run(host: str = "0.0.0.0", port: int = PORT) -> None:
    """Initialise DB, start retry daemon, and run the HTTP server."""
    _init_db()
    _start_retry_daemon()
    server = HTTPServer((host, port), WebhookManagerHandler)
    print(f"[fm_webhook_manager] Webhook & Event Bus Manager listening on {host}:{port}")
    print(f"[fm_webhook_manager] Database: {DB_PATH}")
    print(f"[fm_webhook_manager] Retry daemon: every {RETRY_DAEMON_SLEEP}s")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[fm_webhook_manager] Shutting down.")
        server.server_close()


if __name__ == "__main__":
    run()
