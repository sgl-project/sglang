#!/usr/bin/env python3
"""
fm_stripe_worker.py — Stripe Webhook Worker & Payment Event Processor (Port 7909)
Receives Stripe webhooks, verifies HMAC-SHA256 signatures, persists events,
dispatches intents to the MCP bus, and exposes a payment event query API.
Credentials from ~/.secrets/fractal.env — never hardcoded.
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import json
import time
import hmac
import hashlib
import sqlite3
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from urllib.request import urlopen, Request

# ── vault ─────────────────────────────────────────────────────────────────────
_VAULT = Path.home() / ".secrets" / "fractal.env"
if _VAULT.exists():
    for _ln in _VAULT.read_text().splitlines():
        if "=" in _ln and not _ln.startswith("#"):
            _k, _, _v = _ln.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

PORT            = int(os.getenv("STRIPE_WORKER_PORT", "7909"))
ADMIN_SECRET    = os.getenv("ADMIN_SECRET", "")
WEBHOOK_SECRET  = os.getenv("STRIPE_WEBHOOK_SECRET", "")  # whsec_xxxx
MCP_URL         = os.getenv("MCP_URL", "http://127.0.0.1:7785")
ROOT            = Path(os.getenv("FRACTALMESH_HOME", str(Path.home() / "fmsaas")))
DB              = ROOT / "database" / "sovereign.db"
ROOT.mkdir(parents=True, exist_ok=True)
DB.parent.mkdir(parents=True, exist_ok=True)

# ── DB ─────────────────────────────────────────────────────────────────────────
def _init_db():
    con = sqlite3.connect(str(DB), timeout=10)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript("""
        CREATE TABLE IF NOT EXISTS stripe_events (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            stripe_id     TEXT UNIQUE NOT NULL,
            event_type    TEXT NOT NULL,
            amount_cents  INTEGER,
            currency      TEXT,
            customer_id   TEXT,
            payment_intent TEXT,
            status        TEXT NOT NULL DEFAULT 'received',
            payload       TEXT,
            processed_at  REAL,
            created_at    REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS stripe_customers (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            stripe_cid    TEXT UNIQUE NOT NULL,
            email         TEXT,
            name          TEXT,
            total_paid    REAL NOT NULL DEFAULT 0,
            last_payment  REAL,
            created_at    REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_se_type   ON stripe_events(event_type);
        CREATE INDEX IF NOT EXISTS idx_se_cid    ON stripe_events(customer_id);
        CREATE INDEX IF NOT EXISTS idx_sc_cid    ON stripe_customers(stripe_cid);
    """)
    con.commit()
    con.close()

def _db():
    con = sqlite3.connect(str(DB), check_same_thread=False, timeout=10)
    con.execute("PRAGMA journal_mode=WAL")
    con.row_factory = sqlite3.Row
    return con

# ── Stripe signature verification ──────────────────────────────────────────────
def _verify_stripe_sig(body: bytes, sig_header: str) -> bool:
    if not WEBHOOK_SECRET:
        return True  # accept all if no secret configured
    try:
        pairs = {k: v for k, v in (p.split("=", 1) for p in sig_header.split(",") if "=" in p)}
        ts    = pairs.get("t", "")
        v1    = pairs.get("v1", "")
        signed_payload = f"{ts}.".encode() + body
        expected = hmac.new(WEBHOOK_SECRET.encode(), signed_payload, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, v1)
    except Exception:
        return False

# ── event processor ────────────────────────────────────────────────────────────
_INTENT_MAP = {
    "payment_intent.succeeded":     "payment_success",
    "payment_intent.payment_failed":"payment_failed",
    "checkout.session.completed":   "checkout_complete",
    "customer.subscription.created":"subscription_created",
    "customer.subscription.deleted":"subscription_cancelled",
    "invoice.paid":                 "invoice_paid",
    "invoice.payment_failed":       "invoice_failed",
    "charge.refunded":              "charge_refunded",
    "charge.dispute.created":       "dispute_created",
}

def _dispatch_mcp(intent: str, kwargs: dict):
    try:
        payload = json.dumps({"intent": intent, "args": [], "kwargs": kwargs}).encode()
        req = Request(f"{MCP_URL}/", data=payload,
                      headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=5):
            pass
    except Exception:
        pass

def _process_event(event_id: int, event_type: str, payload: dict):
    intent = _INTENT_MAP.get(event_type)
    obj = payload.get("data", {}).get("object", {})
    kwargs = {
        "stripe_event_id": payload.get("id", ""),
        "event_type":      event_type,
        "amount":          obj.get("amount", obj.get("amount_total", 0)),
        "currency":        obj.get("currency", "aud"),
        "customer_id":     obj.get("customer", ""),
    }
    if intent:
        _dispatch_mcp(intent, kwargs)
    # update customer totals
    cid = obj.get("customer", "")
    amt = obj.get("amount_received", obj.get("amount_total", 0)) or 0
    if cid and event_type in ("payment_intent.succeeded", "checkout.session.completed"):
        try:
            con = _db()
            con.execute("""
                INSERT INTO stripe_customers(stripe_cid, email, total_paid, last_payment, created_at)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(stripe_cid) DO UPDATE SET
                    total_paid   = total_paid + excluded.total_paid,
                    last_payment = excluded.last_payment
            """, (cid, obj.get("receipt_email", ""), amt / 100, time.time(), time.time()))
            con.commit()
            con.close()
        except Exception:
            pass
    try:
        con = _db()
        con.execute("UPDATE stripe_events SET status='processed', processed_at=? WHERE id=?",
                    (time.time(), event_id))
        con.commit()
        con.close()
    except Exception:
        pass

# ── helpers ────────────────────────────────────────────────────────────────────
def _admin(headers) -> bool:
    if not ADMIN_SECRET:
        return True
    return hmac.compare_digest(headers.get("X-Admin-Secret", ""), ADMIN_SECRET)

def _j(data, code=200):
    return code, json.dumps(data, default=str).encode()

def _err(msg, code=400):
    return _j({"error": msg}, code)

# ── HTTP handler ───────────────────────────────────────────────────────────────
class StripeWorkerHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ct="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type,X-Admin-Secret,Stripe-Signature")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        p = parsed.path.strip("/").split("/")
        qs = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        try:
            code, body = self._get(p, qs)
        except Exception as e:
            code, body = _err(str(e), 500)
        self._send(code, body)

    def _get(self, p, qs):
        con = _db()
        try:
            if p in (["health"], [""]):
                total_events = con.execute("SELECT COUNT(*) FROM stripe_events").fetchone()[0]
                unprocessed  = con.execute(
                    "SELECT COUNT(*) FROM stripe_events WHERE status='received'"
                ).fetchone()[0]
                return _j({"status": "ok", "port": PORT,
                            "total_events": total_events, "unprocessed": unprocessed})

            if p == ["events"]:
                if not _admin(self.headers):
                    return _err("Unauthorized", 403)
                limit  = min(int(qs.get("limit", 50)), 200)
                offset = int(qs.get("offset", 0))
                etype  = qs.get("type")
                if etype:
                    rows = con.execute(
                        "SELECT * FROM stripe_events WHERE event_type=? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                        (etype, limit, offset)
                    ).fetchall()
                else:
                    rows = con.execute(
                        "SELECT * FROM stripe_events ORDER BY created_at DESC LIMIT ? OFFSET ?",
                        (limit, offset)
                    ).fetchall()
                return _j([dict(r) for r in rows])

            if p == ["customers"]:
                if not _admin(self.headers):
                    return _err("Unauthorized", 403)
                rows = con.execute(
                    "SELECT * FROM stripe_customers ORDER BY last_payment DESC LIMIT 100"
                ).fetchall()
                return _j([dict(r) for r in rows])

            if p == ["revenue"]:
                if not _admin(self.headers):
                    return _err("Unauthorized", 403)
                total = con.execute(
                    "SELECT SUM(amount_cents) FROM stripe_events "
                    "WHERE event_type IN ('payment_intent.succeeded','checkout.session.completed') "
                    "AND status='processed'"
                ).fetchone()[0] or 0
                return _j({
                    "total_aud": round(total / 100, 2),
                    "total_cents": int(total),
                    "currency": "AUD",
                })

            return _err("not found", 404)
        finally:
            con.close()

    def do_POST(self):
        p = self.path.strip("/").split("/")
        try:
            n    = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n) if n else b""
            code, resp = self._post(p, body)
        except Exception as e:
            code, resp = _err(str(e), 500)
        self._send(code, resp)

    def _post(self, p, raw: bytes):
        if p == ["webhook"] or p == ["stripe", "webhook"]:
            sig = self.headers.get("Stripe-Signature", "")
            if not _verify_stripe_sig(raw, sig):
                return _err("Invalid Stripe signature", 401)
            try:
                payload    = json.loads(raw)
                event_type = payload.get("type", "unknown")
                stripe_id  = payload.get("id", f"evt_{int(time.time())}")
                obj        = payload.get("data", {}).get("object", {})
                con        = _db()
                con.execute(
                    "INSERT OR IGNORE INTO stripe_events"
                    "(stripe_id,event_type,amount_cents,currency,customer_id,payment_intent,payload,created_at)"
                    " VALUES(?,?,?,?,?,?,?,?)",
                    (
                        stripe_id, event_type,
                        obj.get("amount", obj.get("amount_total")),
                        obj.get("currency", "aud"),
                        obj.get("customer", ""),
                        obj.get("payment_intent", ""),
                        raw.decode("utf-8", errors="replace")[:10000],
                        time.time(),
                    )
                )
                con.commit()
                row = con.execute("SELECT id FROM stripe_events WHERE stripe_id=?", (stripe_id,)).fetchone()
                con.close()
                if row:
                    threading.Thread(
                        target=_process_event,
                        args=(row["id"], event_type, payload),
                        daemon=True
                    ).start()
                return _j({"received": True, "stripe_id": stripe_id, "type": event_type})
            except Exception as e:
                return _err(f"processing error: {e}", 500)

        return _err("not found", 404)


def run():
    _init_db()
    server = HTTPServer(("0.0.0.0", PORT), StripeWorkerHandler)
    print(f"[fm_stripe_worker] listening on port {PORT}")
    server.serve_forever()


if __name__ == "__main__":
    run()
