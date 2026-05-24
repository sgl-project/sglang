#!/usr/bin/env python3
"""
fm_webhook_hub.py — Webhook Hub / Event Bus Agent (Port 7837)
Central webhook receiver and event router for FractalMesh OMEGA Titan.
External services (Stripe, GitHub, Zapier, etc.) POST here; validated events
are normalised and routed to the right FractalMesh intent via MCP.
Credentials sourced from ~/.secrets/fractal.env at runtime — never hardcoded.
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import json
import time
import hmac
import hashlib
import signal
import sqlite3
import logging
import fnmatch
import urllib.request
import urllib.error
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

# ── vault ──────────────────────────────────────────────────────────────────────
_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _line in _vault.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

# ── config ─────────────────────────────────────────────────────────────────────
PORT               = int(os.getenv("WEBHOOK_HUB_PORT", "7837"))
MCP_PORT           = int(os.getenv("MCP_PORT", "7785"))
MCP_SECRET         = os.getenv("MCP_SECRET", "fm_mcp_internal")
WEBHOOK_HUB_SECRET = os.getenv("WEBHOOK_HUB_SECRET", "fm_webhook_hub_secret")
ROOT               = Path(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")))
DB                 = ROOT / "database" / "sovereign.db"
LOG                = ROOT / "logs" / "webhook_hub.log"

ROOT.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)
DB.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [WEBHOOK-HUB] %(message)s",
    handlers=[
        logging.FileHandler(LOG),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("webhook_hub")

# ── database ───────────────────────────────────────────────────────────────────
def _db_connect():
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn

def _db_init():
    conn = _db_connect()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS webhook_endpoints (
            id          INTEGER PRIMARY KEY,
            name        TEXT UNIQUE,
            path        TEXT UNIQUE,
            source      TEXT,
            secret      TEXT,
            enabled     INTEGER DEFAULT 1,
            event_count INTEGER DEFAULT 0,
            created_at  REAL
        );
        CREATE TABLE IF NOT EXISTS webhook_events (
            id              INTEGER PRIMARY KEY,
            endpoint_id     INTEGER,
            source          TEXT,
            event_type      TEXT,
            payload         TEXT,
            signature_valid INTEGER,
            routed_to       TEXT,
            route_result    TEXT,
            created_at      REAL
        );
        CREATE TABLE IF NOT EXISTS webhook_subscriptions (
            id              INTEGER PRIMARY KEY,
            event_pattern   TEXT,
            intent          TEXT,
            kwargs_template TEXT,
            enabled         INTEGER DEFAULT 1,
            trigger_count   INTEGER DEFAULT 0,
            created_at      REAL
        );
    """)
    conn.commit()
    conn.close()

def _seed_endpoints():
    now = time.time()
    defaults = [
        ("stripe",  "/webhooks/stripe",  "stripe",  os.getenv("STRIPE_SECRET_KEY", "")),
        ("github",  "/webhooks/github",  "github",  os.getenv("GITHUB_WEBHOOK_SECRET", "")),
        ("zapier",  "/webhooks/zapier",  "zapier",  os.getenv("ZAPIER_WEBHOOK_SECRET", "")),
        ("generic", "/webhooks/generic", "generic", WEBHOOK_HUB_SECRET),
        ("gumroad", "/webhooks/gumroad", "gumroad", ""),
    ]
    conn = _db_connect()
    for name, path, source, secret in defaults:
        conn.execute(
            """INSERT OR IGNORE INTO webhook_endpoints
               (name, path, source, secret, enabled, event_count, created_at)
               VALUES (?,?,?,?,1,0,?)""",
            (name, path, source, secret, now),
        )
    conn.commit()
    conn.close()

def _seed_subscriptions():
    now = time.time()
    defaults = [
        ("stripe.checkout.session.completed", "gumroad",       json.dumps({"op": "sync_sales"})),
        ("stripe.payment_intent.succeeded",   "sendgrid_send", json.dumps({"op": "send_receipt", "email": "{event.receipt_email}"})),
        ("github.push",                       "admin_query",   json.dumps({"op": "log_deploy", "ref": "{event.ref}", "repo": "{event.repository.full_name}"})),
        ("gumroad.sale",                      "leadgen",       json.dumps({"op": "import_lead", "email": "{event.email}", "name": "{event.full_name}"})),
        ("stripe.customer.subscription.created", "data_query", json.dumps({"op": "upgrade_api_tier", "customer": "{event.customer}"})),
        ("*.*.error",                         "admin_query",   json.dumps({"op": "alert_error", "source": "{event.source}", "message": "{event.message}"})),
    ]
    conn = _db_connect()
    for pattern, intent, kwargs_template in defaults:
        conn.execute(
            """INSERT OR IGNORE INTO webhook_subscriptions
               (event_pattern, intent, kwargs_template, enabled, trigger_count, created_at)
               SELECT ?,?,?,1,0,?
               WHERE NOT EXISTS (SELECT 1 FROM webhook_subscriptions WHERE event_pattern=? AND intent=?)""",
            (pattern, intent, kwargs_template, now, pattern, intent),
        )
    conn.commit()
    conn.close()

# ── signature helpers ──────────────────────────────────────────────────────────
def _validate_stripe_sig(body_bytes: bytes, sig_header: str, secret: str) -> bool:
    """Stripe-Signature: t=timestamp,v1=hexsig"""
    if not secret or not sig_header:
        return False
    try:
        parts = dict(item.split("=", 1) for item in sig_header.split(",") if "=" in item)
        ts  = parts.get("t", "")
        v1  = parts.get("v1", "")
        if not ts or not v1:
            return False
        signed_payload = f"{ts}.{body_bytes.decode('utf-8', errors='replace')}"
        expected = hmac.new(
            secret.encode(),
            signed_payload.encode(),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, v1)
    except Exception:
        return False

def _validate_github_sig(body_bytes: bytes, sig_header: str, secret: str) -> bool:
    """X-Hub-Signature-256: sha256=hexdigest"""
    if not secret or not sig_header:
        return False
    try:
        if not sig_header.startswith("sha256="):
            return False
        provided = sig_header[len("sha256="):]
        expected = hmac.new(secret.encode(), body_bytes, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, provided)
    except Exception:
        return False

def _validate_generic_sig(body_bytes: bytes, sig_header: str, secret: str) -> bool:
    """X-Hub-Secret or X-Zapier-Signature: HMAC-SHA256 hex"""
    if not secret or not sig_header:
        return False
    try:
        sig = sig_header.replace("sha256=", "")
        expected = hmac.new(secret.encode(), body_bytes, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, sig)
    except Exception:
        return False

# ── routing helpers ────────────────────────────────────────────────────────────
def _match_pattern(pattern: str, event_str: str) -> bool:
    """Glob matching where * matches any single segment (dot-delimited)."""
    return fnmatch.fnmatch(event_str, pattern)

def _substitute_template(template_dict: dict, event_data: dict) -> dict:
    """Replace {event.key} and {event.a.b.c} placeholders with actual values."""
    def _get_nested(data: dict, path: str):
        parts = path.split(".")
        cur = data
        for p in parts:
            if isinstance(cur, dict):
                cur = cur.get(p, "")
            else:
                return ""
        return str(cur) if cur is not None else ""

    def _replace_val(val):
        if not isinstance(val, str):
            return val
        result = val
        # Find all {event.*} placeholders
        import re
        for match in re.findall(r"\{event\.([\w.]+)\}", val):
            replacement = _get_nested(event_data, match)
            result = result.replace(f"{{event.{match}}}", replacement)
        return result

    return {k: _replace_val(v) for k, v in template_dict.items()}

def _dispatch_to_mcp(intent: str, kwargs: dict) -> dict:
    """POST to MCP router with HMAC authentication."""
    mcp_url = f"http://127.0.0.1:{MCP_PORT}/run"
    payload = json.dumps({"intent": intent, **kwargs}).encode()
    sig = hmac.new(MCP_SECRET.encode(), payload, hashlib.sha256).hexdigest()
    try:
        req = urllib.request.Request(
            mcp_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "X-MCP-Signature": sig,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return {"error": f"HTTP {e.code}: {body}"}
    except Exception as ex:
        return {"error": str(ex)}

def _route_event(event_id: int, source: str, event_type: str, payload: dict) -> int:
    """Find matching subscriptions and dispatch; returns number of routes triggered."""
    event_str = f"{source}.{event_type}"
    conn = _db_connect()
    subs = conn.execute(
        "SELECT * FROM webhook_subscriptions WHERE enabled=1"
    ).fetchall()
    conn.close()

    routes_triggered = 0
    for sub in subs:
        pattern = sub["event_pattern"]
        if not _match_pattern(pattern, event_str):
            continue
        try:
            tmpl = json.loads(sub["kwargs_template"]) if sub["kwargs_template"] else {}
        except (json.JSONDecodeError, TypeError):
            tmpl = {}
        kwargs = _substitute_template(tmpl, payload)
        result = _dispatch_to_mcp(sub["intent"], kwargs)
        route_label = sub["intent"]
        route_result = json.dumps(result)
        routes_triggered += 1
        # Update subscription trigger count
        conn2 = _db_connect()
        conn2.execute(
            "UPDATE webhook_subscriptions SET trigger_count=trigger_count+1 WHERE id=?",
            (sub["id"],),
        )
        conn2.execute(
            "UPDATE webhook_events SET routed_to=?, route_result=? WHERE id=?",
            (route_label, route_result, event_id),
        )
        conn2.commit()
        conn2.close()
        log.info("Routed event %d → intent=%s result=%s", event_id, route_label, route_result[:120])
    return routes_triggered

# ── HTTP handler ───────────────────────────────────────────────────────────────
class WebhookHubHandler(BaseHTTPRequestHandler):
    server_version = "FractalMesh-WebhookHub/1.0"

    def log_message(self, fmt, *args):
        log.info(fmt, *args)

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length > 0 else b""

    def _parse_path(self):
        """Return path without query string."""
        return self.path.split("?")[0].rstrip("/") or "/"

    # ── routing ────────────────────────────────────────────────────────────────
    def do_GET(self):
        path = self._parse_path()
        if path == "/health":
            self._handle_health()
        elif path == "/endpoints":
            self._handle_list_endpoints()
        elif path == "/subscriptions":
            self._handle_list_subscriptions()
        elif path == "/events":
            self._handle_list_events()
        elif path.startswith("/events/"):
            seg = path[len("/events/"):]
            if seg.isdigit():
                self._handle_get_event(int(seg))
            else:
                self._send_json({"error": "not found"}, 404)
        elif path == "/analytics":
            self._handle_analytics()
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        path = self._parse_path()
        if path.startswith("/webhooks/"):
            self._handle_webhook(path)
        elif path == "/endpoints/register":
            self._handle_register_endpoint()
        elif path == "/subscriptions/create":
            self._handle_create_subscription()
        elif path.startswith("/events/replay/"):
            seg = path[len("/events/replay/"):]
            if seg.isdigit():
                self._handle_replay_event(int(seg))
            else:
                self._send_json({"error": "not found"}, 404)
        else:
            self._send_json({"error": "not found"}, 404)

    def do_DELETE(self):
        path = self._parse_path()
        if path.startswith("/subscriptions/"):
            seg = path[len("/subscriptions/"):]
            if seg.isdigit():
                self._handle_delete_subscription(int(seg))
            else:
                self._send_json({"error": "not found"}, 404)
        else:
            self._send_json({"error": "not found"}, 404)

    # ── handlers ───────────────────────────────────────────────────────────────
    def _handle_health(self):
        self._send_json({"status": "ok", "service": "fm-webhook-hub", "port": PORT})

    def _handle_webhook(self, path: str):
        body_bytes = self._read_body()
        conn = _db_connect()
        row = conn.execute(
            "SELECT * FROM webhook_endpoints WHERE path=? AND enabled=1",
            (path,),
        ).fetchone()
        conn.close()
        if not row:
            self._send_json({"error": "unknown webhook path"}, 404)
            return

        source = row["source"]
        secret = row["secret"] or ""
        endpoint_id = row["id"]

        # ── signature validation ───────────────────────────────────────────────
        sig_valid = False
        if source == "stripe":
            sig_header = self.headers.get("Stripe-Signature", "")
            sig_valid = _validate_stripe_sig(body_bytes, sig_header, secret)
            if not secret:
                sig_valid = True  # no key configured, accept
        elif source == "github":
            sig_header = self.headers.get("X-Hub-Signature-256", "")
            sig_valid = _validate_github_sig(body_bytes, sig_header, secret)
            if not secret:
                sig_valid = True
        elif source == "zapier":
            sig_header = self.headers.get("X-Zapier-Signature", "")
            if secret:
                sig_valid = _validate_generic_sig(body_bytes, sig_header, secret)
            else:
                sig_valid = True  # no ZAPIER_WEBHOOK_SECRET set
        elif source == "generic":
            sig_header = self.headers.get("X-Hub-Secret", "")
            sig_valid = _validate_generic_sig(body_bytes, sig_header, secret)
        elif source == "gumroad":
            sig_valid = True  # Gumroad doesn't sign
        else:
            sig_header = self.headers.get("X-Hub-Secret", "")
            sig_valid = _validate_generic_sig(body_bytes, sig_header, secret) if secret else True

        # ── parse body ─────────────────────────────────────────────────────────
        try:
            payload = json.loads(body_bytes) if body_bytes else {}
        except (json.JSONDecodeError, ValueError):
            payload = {}

        # ── determine event_type ───────────────────────────────────────────────
        if source == "stripe":
            event_type = payload.get("type", "unknown")
        elif source == "github":
            event_type = self.headers.get("X-GitHub-Event", "unknown")
        elif source == "zapier":
            event_type = payload.get("event_type", "trigger")
        else:
            event_type = payload.get("event_type", "webhook")

        # ── store event ────────────────────────────────────────────────────────
        now = time.time()
        conn2 = _db_connect()
        cur = conn2.execute(
            """INSERT INTO webhook_events
               (endpoint_id, source, event_type, payload, signature_valid, routed_to, route_result, created_at)
               VALUES (?,?,?,?,?,NULL,NULL,?)""",
            (endpoint_id, source, event_type, json.dumps(payload), int(sig_valid), now),
        )
        event_id = cur.lastrowid
        conn2.execute(
            "UPDATE webhook_endpoints SET event_count=event_count+1 WHERE id=?",
            (endpoint_id,),
        )
        conn2.commit()
        conn2.close()

        # ── route event ────────────────────────────────────────────────────────
        routes_triggered = _route_event(event_id, source, event_type, payload)
        log.info(
            "Webhook %s → source=%s type=%s sig_valid=%s routes=%d",
            path, source, event_type, sig_valid, routes_triggered,
        )
        self._send_json({
            "received": True,
            "event_id": event_id,
            "routes_triggered": routes_triggered,
        })

    def _handle_register_endpoint(self):
        body_bytes = self._read_body()
        try:
            data = json.loads(body_bytes)
        except (json.JSONDecodeError, ValueError):
            self._send_json({"error": "invalid JSON"}, 400)
            return
        name   = data.get("name", "")
        path   = data.get("path", "")
        source = data.get("source", "generic")
        secret = data.get("secret", "")
        if not name or not path:
            self._send_json({"error": "name and path required"}, 400)
            return
        now = time.time()
        try:
            conn = _db_connect()
            cur = conn.execute(
                """INSERT INTO webhook_endpoints
                   (name, path, source, secret, enabled, event_count, created_at)
                   VALUES (?,?,?,?,1,0,?)""",
                (name, path, source, secret, now),
            )
            endpoint_id = cur.lastrowid
            conn.commit()
            conn.close()
        except sqlite3.IntegrityError as ex:
            self._send_json({"error": f"duplicate name or path: {ex}"}, 409)
            return
        log.info("Registered endpoint id=%d name=%s path=%s", endpoint_id, name, path)
        self._send_json({"endpoint_id": endpoint_id, "path": path})

    def _handle_list_endpoints(self):
        conn = _db_connect()
        rows = conn.execute(
            "SELECT id, name, path, source, enabled, event_count, created_at FROM webhook_endpoints ORDER BY id"
        ).fetchall()
        conn.close()
        self._send_json([dict(r) for r in rows])

    def _handle_create_subscription(self):
        body_bytes = self._read_body()
        try:
            data = json.loads(body_bytes)
        except (json.JSONDecodeError, ValueError):
            self._send_json({"error": "invalid JSON"}, 400)
            return
        pattern  = data.get("event_pattern", "")
        intent   = data.get("intent", "")
        tmpl     = data.get("kwargs_template", {})
        if not pattern or not intent:
            self._send_json({"error": "event_pattern and intent required"}, 400)
            return
        now = time.time()
        conn = _db_connect()
        cur = conn.execute(
            """INSERT INTO webhook_subscriptions
               (event_pattern, intent, kwargs_template, enabled, trigger_count, created_at)
               VALUES (?,?,?,1,0,?)""",
            (pattern, intent, json.dumps(tmpl), now),
        )
        sub_id = cur.lastrowid
        conn.commit()
        conn.close()
        log.info("Created subscription id=%d pattern=%s → %s", sub_id, pattern, intent)
        self._send_json({"subscription_id": sub_id})

    def _handle_list_subscriptions(self):
        conn = _db_connect()
        rows = conn.execute(
            "SELECT id, event_pattern, intent, kwargs_template, enabled, trigger_count, created_at FROM webhook_subscriptions ORDER BY id"
        ).fetchall()
        conn.close()
        self._send_json([dict(r) for r in rows])

    def _handle_delete_subscription(self, sub_id: int):
        conn = _db_connect()
        conn.execute("DELETE FROM webhook_subscriptions WHERE id=?", (sub_id,))
        conn.commit()
        conn.close()
        log.info("Deleted subscription id=%d", sub_id)
        self._send_json({"deleted": True, "subscription_id": sub_id})

    def _handle_list_events(self):
        conn = _db_connect()
        rows = conn.execute(
            """SELECT e.id, ep.name AS endpoint_name, e.source, e.event_type,
                      e.signature_valid, e.routed_to, e.route_result, e.created_at
               FROM webhook_events e
               LEFT JOIN webhook_endpoints ep ON ep.id = e.endpoint_id
               ORDER BY e.id DESC LIMIT 100"""
        ).fetchall()
        conn.close()
        self._send_json([dict(r) for r in rows])

    def _handle_get_event(self, event_id: int):
        conn = _db_connect()
        row = conn.execute(
            """SELECT e.*, ep.name AS endpoint_name
               FROM webhook_events e
               LEFT JOIN webhook_endpoints ep ON ep.id = e.endpoint_id
               WHERE e.id=?""",
            (event_id,),
        ).fetchone()
        conn.close()
        if not row:
            self._send_json({"error": "event not found"}, 404)
            return
        d = dict(row)
        # Deserialise payload for convenience
        try:
            d["payload"] = json.loads(d["payload"]) if d.get("payload") else {}
        except (json.JSONDecodeError, TypeError):
            pass
        self._send_json(d)

    def _handle_replay_event(self, event_id: int):
        conn = _db_connect()
        row = conn.execute(
            "SELECT * FROM webhook_events WHERE id=?", (event_id,)
        ).fetchone()
        conn.close()
        if not row:
            self._send_json({"error": "event not found"}, 404)
            return
        try:
            payload = json.loads(row["payload"]) if row["payload"] else {}
        except (json.JSONDecodeError, TypeError):
            payload = {}
        routes_triggered = _route_event(event_id, row["source"], row["event_type"], payload)
        log.info("Replayed event id=%d routes=%d", event_id, routes_triggered)
        self._send_json({
            "replayed": True,
            "event_id": event_id,
            "routes_triggered": routes_triggered,
        })

    def _handle_analytics(self):
        conn = _db_connect()
        total_row = conn.execute("SELECT COUNT(*) AS n FROM webhook_events").fetchone()
        events_total = total_row["n"] if total_row else 0

        by_source = conn.execute(
            "SELECT source, COUNT(*) AS n FROM webhook_events GROUP BY source ORDER BY n DESC"
        ).fetchall()
        events_by_source = {r["source"]: r["n"] for r in by_source}

        today_start = time.time() - (time.time() % 86400)
        today_row = conn.execute(
            "SELECT COUNT(*) AS n FROM webhook_events WHERE created_at >= ?",
            (today_start,),
        ).fetchone()
        events_today = today_row["n"] if today_row else 0

        success_row = conn.execute(
            "SELECT COUNT(*) AS n FROM webhook_events WHERE route_result IS NOT NULL AND route_result NOT LIKE '%error%'"
        ).fetchone()
        success_count = success_row["n"] if success_row else 0
        routing_success_rate = round(success_count / events_total, 4) if events_total > 0 else 0.0

        top_types = conn.execute(
            "SELECT event_type, COUNT(*) AS n FROM webhook_events GROUP BY event_type ORDER BY n DESC LIMIT 10"
        ).fetchall()
        top_event_types = [{"event_type": r["event_type"], "count": r["n"]} for r in top_types]

        conn.close()
        self._send_json({
            "events_total": events_total,
            "events_by_source": events_by_source,
            "events_today": events_today,
            "routing_success_rate": routing_success_rate,
            "top_event_types": top_event_types,
        })

# ── server ─────────────────────────────────────────────────────────────────────
_running = True

def _handle_signal(sig, frame):
    global _running
    log.info("Received signal %d, shutting down…", sig)
    _running = False

def main():
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    log.info("Initialising database at %s", DB)
    _db_init()
    _seed_endpoints()
    _seed_subscriptions()

    server = HTTPServer(("0.0.0.0", PORT), WebhookHubHandler)
    server.timeout = 1.0
    log.info("FractalMesh Webhook Hub listening on port %d", PORT)

    while _running:
        server.handle_request()

    server.server_close()
    log.info("Webhook Hub stopped.")

if __name__ == "__main__":
    main()
