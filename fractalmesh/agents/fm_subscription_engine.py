#!/usr/bin/env python3
"""
FractalMesh OMEGA Titan — Subscription & Recurring Revenue Engine
Port: 7869

Complete subscription lifecycle manager: free trials, plan upgrades/downgrades,
proration, dunning (retry failed payments), and MRR/ARR analytics.

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
import hashlib
import hmac
import json
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
AGENT_NAME = "fm_subscription_engine"
PORT = int(os.environ.get("SUBSCRIPTION_ENGINE_PORT", "7869"))

STRIPE_SECRET_KEY   = os.environ.get("STRIPE_SECRET_KEY", "")
SENDGRID_API_KEY    = os.environ.get("SENDGRID_API_KEY", "")
SENDGRID_FROM_EMAIL = os.environ.get("SENDGRID_FROM_EMAIL", "noreply@fractalmesh.io")
SENDGRID_FROM_NAME  = os.environ.get("SENDGRID_FROM_NAME", "FractalMesh")
ADMIN_SECRET        = os.environ.get("ADMIN_SECRET", "")

DB_PATH = Path.home() / "fmsaas" / "database" / "sovereign.db"
START_TIME = time.time()

# Dunning retry schedule offsets in seconds
DUNNING_SCHEDULE = [
    1 * 86400,   # +1 day
    3 * 86400,   # +3 days
    7 * 86400,   # +7 days
    14 * 86400,  # +14 days
]

# ---------------------------------------------------------------------------
# Pre-seeded plans
# ---------------------------------------------------------------------------
SEED_PLANS = [
    {
        "plan_id": "starter",
        "name": "Starter",
        "description": "Perfect for individuals and small teams getting started.",
        "price_monthly": 29.0,
        "price_annual": 290.0,
        "currency": "AUD",
        "features": json.dumps(["API Access", "10K requests/mo", "Basic analytics", "Email support", "1 user"]),
        "trial_days": 0,
        "active": 1,
        "stripe_price_id_monthly": "",
        "stripe_price_id_annual": "",
    },
    {
        "plan_id": "pro",
        "name": "Pro",
        "description": "For growing teams that need more power and flexibility.",
        "price_monthly": 99.0,
        "price_annual": 990.0,
        "currency": "AUD",
        "features": json.dumps(["Everything in Starter", "100K requests/mo", "Advanced analytics", "Priority support", "5 users", "Webhooks"]),
        "trial_days": 14,
        "active": 1,
        "stripe_price_id_monthly": "",
        "stripe_price_id_annual": "",
    },
    {
        "plan_id": "business",
        "name": "Business",
        "description": "Enterprise-grade features for scaling businesses.",
        "price_monthly": 299.0,
        "price_annual": 2990.0,
        "currency": "AUD",
        "features": json.dumps(["Everything in Pro", "Unlimited requests", "Custom analytics", "Dedicated support", "Unlimited users", "SLA guarantee", "White label"]),
        "trial_days": 14,
        "active": 1,
        "stripe_price_id_monthly": "",
        "stripe_price_id_annual": "",
    },
    {
        "plan_id": "enterprise",
        "name": "Enterprise",
        "description": "Custom pricing for large organisations. Contact us.",
        "price_monthly": 0.0,
        "price_annual": 0.0,
        "currency": "AUD",
        "features": json.dumps(["Contact us for custom pricing"]),
        "trial_days": 0,
        "active": 0,
        "stripe_price_id_monthly": "",
        "stripe_price_id_annual": "",
    },
]

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS plans (
                id                      INTEGER PRIMARY KEY,
                plan_id                 TEXT UNIQUE NOT NULL,
                name                    TEXT NOT NULL,
                description             TEXT,
                price_monthly           REAL NOT NULL DEFAULT 0,
                price_annual            REAL NOT NULL DEFAULT 0,
                currency                TEXT DEFAULT 'AUD',
                features                TEXT,
                trial_days              INTEGER DEFAULT 0,
                active                  INTEGER DEFAULT 1,
                stripe_price_id_monthly TEXT,
                stripe_price_id_annual  TEXT,
                created_at              REAL
            );

            CREATE TABLE IF NOT EXISTS subscriptions (
                id                      INTEGER PRIMARY KEY,
                sub_ref                 TEXT UNIQUE NOT NULL,
                customer_email          TEXT NOT NULL,
                customer_name           TEXT,
                plan_id                 TEXT NOT NULL,
                billing_cycle           TEXT DEFAULT 'monthly',
                status                  TEXT DEFAULT 'trialing',
                trial_ends_at           REAL,
                current_period_start    REAL,
                current_period_end      REAL,
                stripe_sub_id           TEXT,
                stripe_customer_id      TEXT,
                cancel_at_period_end    INTEGER DEFAULT 0,
                cancelled_at            REAL,
                created_at              REAL,
                updated_at              REAL
            );

            CREATE TABLE IF NOT EXISTS invoices (
                id                INTEGER PRIMARY KEY,
                sub_ref           TEXT NOT NULL,
                invoice_ref       TEXT UNIQUE NOT NULL,
                amount            REAL NOT NULL,
                currency          TEXT DEFAULT 'AUD',
                status            TEXT DEFAULT 'pending',
                due_date          REAL,
                paid_at           REAL,
                stripe_invoice_id TEXT,
                attempt_count     INTEGER DEFAULT 0,
                created_at        REAL
            );

            CREATE TABLE IF NOT EXISTS dunning_attempts (
                id              INTEGER PRIMARY KEY,
                invoice_ref     TEXT NOT NULL,
                attempt_number  INTEGER NOT NULL,
                attempted_at    REAL,
                result          TEXT,
                next_attempt_at REAL
            );

            CREATE TABLE IF NOT EXISTS mrr_snapshots (
                id           INTEGER PRIMARY KEY,
                snapshot_date TEXT NOT NULL,
                mrr          REAL,
                arr          REAL,
                active_subs  INTEGER,
                churned_subs INTEGER,
                new_subs     INTEGER,
                upgraded     INTEGER,
                downgraded   INTEGER,
                created_at   REAL
            );
        """)
        # Seed plans if empty
        row = conn.execute("SELECT COUNT(*) FROM plans").fetchone()
        if row[0] == 0:
            now = time.time()
            for p in SEED_PLANS:
                conn.execute(
                    """INSERT OR IGNORE INTO plans
                       (plan_id, name, description, price_monthly, price_annual,
                        currency, features, trial_days, active,
                        stripe_price_id_monthly, stripe_price_id_annual, created_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (p["plan_id"], p["name"], p["description"],
                     p["price_monthly"], p["price_annual"],
                     p["currency"], p["features"], p["trial_days"],
                     p["active"], p["stripe_price_id_monthly"],
                     p["stripe_price_id_annual"], now),
                )


def row_to_dict(row) -> dict:
    if row is None:
        return {}
    return dict(row)


def gen_ref(prefix: str) -> str:
    raw = f"{prefix}-{time.time()}-{os.urandom(8).hex()}"
    return f"{prefix}_{hashlib.sha256(raw.encode()).hexdigest()[:16].upper()}"

# ---------------------------------------------------------------------------
# Stripe helpers
# ---------------------------------------------------------------------------

def stripe_request(method: str, path: str, data: dict | None = None) -> dict:
    url = f"https://api.stripe.com{path}"
    body = None
    if data is not None:
        encoded = "&".join(f"{k}={urllib.request.quote(str(v))}" for k, v in data.items())
        body = encoded.encode()
    req = urllib.request.Request(url, data=body, method=method)
    token = f"{STRIPE_SECRET_KEY}:"
    import base64 as _b64
    req.add_header("Authorization", f"Basic {_b64.b64encode(token.encode()).decode()}")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body_bytes = exc.read()
        try:
            return json.loads(body_bytes.decode())
        except Exception:
            return {"error": {"message": str(exc)}}
    except Exception as exc:
        return {"error": {"message": str(exc)}}


def stripe_create_customer(email: str, name: str) -> dict:
    return stripe_request("POST", "/v1/customers", {"email": email, "name": name})


def stripe_create_subscription(customer_id: str, price_id: str,
                                payment_method_id: str,
                                trial_days: int = 0) -> dict:
    data: dict = {
        "customer": customer_id,
        "items[0][price]": price_id,
        "default_payment_method": payment_method_id,
        "expand[0]": "latest_invoice.payment_intent",
    }
    if trial_days > 0:
        data["trial_period_days"] = trial_days
    return stripe_request("POST", "/v1/subscriptions", data)


def stripe_update_subscription(stripe_sub_id: str, new_price_id: str,
                                proration_behavior: str = "create_prorations") -> dict:
    # First retrieve to get item id
    sub = stripe_request("GET", f"/v1/subscriptions/{stripe_sub_id}")
    if "error" in sub:
        return sub
    items = sub.get("items", {}).get("data", [])
    if not items:
        return {"error": {"message": "No subscription items found"}}
    item_id = items[0]["id"]
    return stripe_request("POST", f"/v1/subscriptions/{stripe_sub_id}", {
        f"items[0][id]": item_id,
        f"items[0][price]": new_price_id,
        "proration_behavior": proration_behavior,
    })


def stripe_cancel_subscription(stripe_sub_id: str, at_period_end: bool = True) -> dict:
    if at_period_end:
        return stripe_request("POST", f"/v1/subscriptions/{stripe_sub_id}",
                              {"cancel_at_period_end": "true"})
    return stripe_request("DELETE", f"/v1/subscriptions/{stripe_sub_id}")


def stripe_reactivate_subscription(stripe_sub_id: str) -> dict:
    return stripe_request("POST", f"/v1/subscriptions/{stripe_sub_id}",
                          {"cancel_at_period_end": "false"})


def stripe_retry_invoice(stripe_invoice_id: str) -> dict:
    return stripe_request("POST", f"/v1/invoices/{stripe_invoice_id}/pay", {})

# ---------------------------------------------------------------------------
# SendGrid helpers
# ---------------------------------------------------------------------------

def sendgrid_send(to_email: str, to_name: str, subject: str, html: str) -> bool:
    if not SENDGRID_API_KEY:
        return False
    payload = json.dumps({
        "personalizations": [{"to": [{"email": to_email, "name": to_name}]}],
        "from": {"email": SENDGRID_FROM_EMAIL, "name": SENDGRID_FROM_NAME},
        "subject": subject,
        "content": [{"type": "text/html", "value": html}],
    }).encode()
    req = urllib.request.Request(
        "https://api.sendgrid.com/v3/mail/send",
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
    except Exception:
        return False


def send_welcome_email(email: str, name: str, plan_name: str, trial_days: int) -> None:
    trial_note = f"<p>Your <strong>{trial_days}-day free trial</strong> has started. No charge until the trial ends.</p>" if trial_days > 0 else ""
    html = f"""
    <h2>Welcome to FractalMesh, {name}!</h2>
    <p>You are now subscribed to the <strong>{plan_name}</strong> plan.</p>
    {trial_note}
    <p>Thank you for choosing FractalMesh OMEGA Titan.</p>
    """
    sendgrid_send(email, name, f"Welcome to FractalMesh — {plan_name} Plan", html)


def send_cancellation_email(email: str, name: str, plan_name: str, reason: str = "") -> None:
    html = f"""
    <h2>Your FractalMesh subscription has been cancelled</h2>
    <p>Hello {name},</p>
    <p>Your <strong>{plan_name}</strong> subscription has been cancelled{' due to ' + reason if reason else ''}.</p>
    <p>You can reactivate at any time from your dashboard.</p>
    """
    sendgrid_send(email, name, "FractalMesh Subscription Cancelled", html)


def send_payment_failed_email(email: str, name: str, amount: float,
                               currency: str, attempt: int) -> None:
    html = f"""
    <h2>Payment Failed — Action Required</h2>
    <p>Hello {name},</p>
    <p>We were unable to process your payment of <strong>{currency} {amount:.2f}</strong>
       (attempt {attempt} of 4).</p>
    <p>Please update your payment method to avoid service interruption.</p>
    """
    sendgrid_send(email, name, "FractalMesh — Payment Failed", html)

# ---------------------------------------------------------------------------
# MRR computation
# ---------------------------------------------------------------------------

def compute_current_mrr() -> float:
    with get_db() as conn:
        rows = conn.execute(
            """SELECT s.billing_cycle, p.price_monthly, p.price_annual
               FROM subscriptions s
               JOIN plans p ON s.plan_id = p.plan_id
               WHERE s.status IN ('active', 'trialing')"""
        ).fetchall()
    mrr = 0.0
    for r in rows:
        if r["billing_cycle"] == "annual":
            mrr += (r["price_annual"] or 0) / 12.0
        else:
            mrr += r["price_monthly"] or 0
    return round(mrr, 2)


def take_mrr_snapshot() -> None:
    today = time.strftime("%Y-%m-%d", time.localtime())
    now = time.time()
    today_start = time.mktime(time.strptime(today, "%Y-%m-%d"))
    today_end = today_start + 86400

    with get_db() as conn:
        existing = conn.execute(
            "SELECT id FROM mrr_snapshots WHERE snapshot_date=?", (today,)
        ).fetchone()
        if existing:
            return

        mrr = compute_current_mrr()
        arr = round(mrr * 12, 2)

        active_count = conn.execute(
            "SELECT COUNT(*) FROM subscriptions WHERE status IN ('active','trialing')"
        ).fetchone()[0]

        new_count = conn.execute(
            "SELECT COUNT(*) FROM subscriptions WHERE created_at >= ? AND created_at < ?",
            (today_start, today_end)
        ).fetchone()[0]

        churned_count = conn.execute(
            "SELECT COUNT(*) FROM subscriptions WHERE cancelled_at >= ? AND cancelled_at < ?",
            (today_start, today_end)
        ).fetchone()[0]

        conn.execute(
            """INSERT INTO mrr_snapshots
               (snapshot_date, mrr, arr, active_subs, churned_subs,
                new_subs, upgraded, downgraded, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (today, mrr, arr, active_count, churned_count, new_count, 0, 0, now)
        )

# ---------------------------------------------------------------------------
# Dunning engine
# ---------------------------------------------------------------------------

def run_dunning() -> None:
    """Retry failed invoices according to dunning schedule."""
    now = time.time()
    with get_db() as conn:
        failed_invoices = conn.execute(
            """SELECT i.*, s.customer_email, s.customer_name, s.plan_id, s.stripe_sub_id,
                      p.name as plan_name
               FROM invoices i
               JOIN subscriptions s ON i.sub_ref = s.sub_ref
               JOIN plans p ON s.plan_id = p.plan_id
               WHERE i.status = 'failed'
                 AND i.attempt_count < 4
                 AND (
                     SELECT COALESCE(MAX(da.next_attempt_at), 0)
                     FROM dunning_attempts da
                     WHERE da.invoice_ref = i.invoice_ref
                 ) <= ?""",
            (now,)
        ).fetchall()

    for inv in failed_invoices:
        inv_dict = dict(inv)
        attempt_num = inv_dict["attempt_count"] + 1
        result = "failed"

        if inv_dict.get("stripe_invoice_id"):
            resp = stripe_retry_invoice(inv_dict["stripe_invoice_id"])
            if resp.get("status") == "paid" or resp.get("paid"):
                result = "paid"
        else:
            # No Stripe invoice ID — mark as simulated retry
            result = "no_stripe_id"

        attempted_at = time.time()
        next_offset = DUNNING_SCHEDULE[attempt_num - 1] if attempt_num <= len(DUNNING_SCHEDULE) else None
        next_attempt_at = attempted_at + next_offset if next_offset else None

        with get_db() as conn:
            conn.execute(
                """INSERT INTO dunning_attempts
                   (invoice_ref, attempt_number, attempted_at, result, next_attempt_at)
                   VALUES (?,?,?,?,?)""",
                (inv_dict["invoice_ref"], attempt_num, attempted_at,
                 result, next_attempt_at)
            )

            if result == "paid":
                conn.execute(
                    "UPDATE invoices SET status='paid', paid_at=?, attempt_count=? WHERE invoice_ref=?",
                    (attempted_at, attempt_num, inv_dict["invoice_ref"])
                )
                # Reactivate subscription
                conn.execute(
                    "UPDATE subscriptions SET status='active', updated_at=? WHERE sub_ref=?",
                    (attempted_at, inv_dict["sub_ref"])
                )
            else:
                conn.execute(
                    "UPDATE invoices SET attempt_count=? WHERE invoice_ref=?",
                    (attempt_num, inv_dict["invoice_ref"])
                )
                # Send payment failed email
                send_payment_failed_email(
                    inv_dict["customer_email"],
                    inv_dict["customer_name"] or "",
                    inv_dict["amount"],
                    inv_dict["currency"],
                    attempt_num,
                )
                # 4th failure: cancel subscription
                if attempt_num >= 4:
                    with get_db() as conn2:
                        conn2.execute(
                            """UPDATE subscriptions SET status='cancelled', cancelled_at=?,
                               updated_at=? WHERE sub_ref=?""",
                            (attempted_at, attempted_at, inv_dict["sub_ref"])
                        )
                    # Cancel on Stripe too
                    if inv_dict.get("stripe_sub_id"):
                        stripe_cancel_subscription(inv_dict["stripe_sub_id"], at_period_end=False)
                    send_cancellation_email(
                        inv_dict["customer_email"],
                        inv_dict["customer_name"] or "",
                        inv_dict["plan_name"],
                        reason="non-payment after 4 attempts",
                    )


def dunning_daemon() -> None:
    """Background thread: run dunning check every hour."""
    while True:
        try:
            run_dunning()
        except Exception as exc:
            pass
        time.sleep(3600)


def mrr_snapshot_daemon() -> None:
    """Background thread: take MRR snapshot daily at 00:05."""
    while True:
        now_struct = time.localtime()
        # Sleep until 00:05
        if now_struct.tm_hour == 0 and now_struct.tm_min >= 5:
            # Already past 00:05 today, wait until tomorrow
            seconds_until = 86400 - (now_struct.tm_hour * 3600 + now_struct.tm_min * 60 + now_struct.tm_sec) + 5 * 60
        else:
            seconds_until = (0 * 3600 + 5 * 60) - (now_struct.tm_hour * 3600 + now_struct.tm_min * 60 + now_struct.tm_sec)
            if seconds_until < 0:
                seconds_until += 86400
        time.sleep(max(seconds_until, 60))
        try:
            take_mrr_snapshot()
        except Exception:
            pass

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def json_response(handler: BaseHTTPRequestHandler, code: int, data: dict) -> None:
    body = json.dumps(data, default=str).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def parse_body(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", 0))
    if length == 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw.decode())
    except Exception:
        return {}


def parse_query(path: str) -> tuple[str, dict]:
    if "?" not in path:
        return path, {}
    base, qs = path.split("?", 1)
    params: dict = {}
    for part in qs.split("&"):
        if "=" in part:
            k, v = part.split("=", 1)
            params[urllib.request.unquote(k)] = urllib.request.unquote(v)
    return base, params


def verify_admin(handler: BaseHTTPRequestHandler) -> bool:
    if not ADMIN_SECRET:
        return True
    auth = handler.headers.get("X-Admin-Secret", "")
    return hmac.compare_digest(auth, ADMIN_SECRET)

# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class SubscriptionHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):  # suppress default logging
        pass

    # ------------------------------------------------------------------ GET

    def do_GET(self):
        path, params = parse_query(self.path)
        parts = [p for p in path.strip("/").split("/") if p]

        # GET /health
        if parts == ["health"]:
            with get_db() as conn:
                active = conn.execute(
                    "SELECT COUNT(*) FROM subscriptions WHERE status IN ('active','trialing')"
                ).fetchone()[0]
            mrr = compute_current_mrr()
            json_response(self, 200, {
                "status": "ok",
                "uptime_seconds": round(time.time() - START_TIME, 1),
                "active_subscriptions": active,
                "current_mrr_aud": mrr,
            })
            return

        # GET /plans
        if parts == ["plans"]:
            with get_db() as conn:
                rows = conn.execute(
                    "SELECT * FROM plans WHERE active=1 ORDER BY price_monthly"
                ).fetchall()
            plans = []
            for r in rows:
                d = row_to_dict(r)
                d["features"] = json.loads(d.get("features") or "[]")
                plans.append(d)
            json_response(self, 200, {"plans": plans, "count": len(plans)})
            return

        # GET /plans/{plan_id}
        if len(parts) == 2 and parts[0] == "plans":
            with get_db() as conn:
                row = conn.execute(
                    "SELECT * FROM plans WHERE plan_id=?", (parts[1],)
                ).fetchone()
            if not row:
                json_response(self, 404, {"error": "Plan not found"})
                return
            d = row_to_dict(row)
            d["features"] = json.loads(d.get("features") or "[]")
            json_response(self, 200, d)
            return

        # GET /subscriptions
        if parts == ["subscriptions"]:
            where, args = [], []
            if "status" in params:
                where.append("status=?")
                args.append(params["status"])
            if "plan_id" in params:
                where.append("plan_id=?")
                args.append(params["plan_id"])
            if "customer_email" in params:
                where.append("customer_email=?")
                args.append(params["customer_email"])
            limit = int(params.get("limit", 100))
            sql = "SELECT * FROM subscriptions"
            if where:
                sql += " WHERE " + " AND ".join(where)
            sql += " ORDER BY created_at DESC LIMIT ?"
            args.append(limit)
            with get_db() as conn:
                rows = conn.execute(sql, args).fetchall()
            json_response(self, 200, {
                "subscriptions": [row_to_dict(r) for r in rows],
                "count": len(rows),
            })
            return

        # GET /subscriptions/{sub_ref}
        if len(parts) == 2 and parts[0] == "subscriptions":
            sub_ref = parts[1]
            with get_db() as conn:
                sub = conn.execute(
                    "SELECT * FROM subscriptions WHERE sub_ref=?", (sub_ref,)
                ).fetchone()
                if not sub:
                    json_response(self, 404, {"error": "Subscription not found"})
                    return
                invs = conn.execute(
                    "SELECT * FROM invoices WHERE sub_ref=? ORDER BY created_at DESC",
                    (sub_ref,)
                ).fetchall()
            d = row_to_dict(sub)
            d["invoices"] = [row_to_dict(i) for i in invs]
            json_response(self, 200, d)
            return

        # GET /invoices
        if parts == ["invoices"]:
            where, args = [], []
            if "status" in params:
                where.append("status=?")
                args.append(params["status"])
            if "sub_ref" in params:
                where.append("sub_ref=?")
                args.append(params["sub_ref"])
            limit = int(params.get("limit", 100))
            sql = "SELECT * FROM invoices"
            if where:
                sql += " WHERE " + " AND ".join(where)
            sql += " ORDER BY created_at DESC LIMIT ?"
            args.append(limit)
            with get_db() as conn:
                rows = conn.execute(sql, args).fetchall()
            json_response(self, 200, {
                "invoices": [row_to_dict(r) for r in rows],
                "count": len(rows),
            })
            return

        # GET /mrr
        if parts == ["mrr"]:
            with get_db() as conn:
                rows = conn.execute(
                    "SELECT * FROM mrr_snapshots ORDER BY snapshot_date DESC LIMIT 30"
                ).fetchall()
            snapshots = [row_to_dict(r) for r in rows]
            growth_rate = None
            if len(snapshots) >= 2:
                latest = snapshots[0]["mrr"] or 0
                prev = snapshots[-1]["mrr"] or 0
                if prev > 0:
                    growth_rate = round(((latest - prev) / prev) * 100, 2)
            current_mrr = compute_current_mrr()
            json_response(self, 200, {
                "current_mrr": current_mrr,
                "current_arr": round(current_mrr * 12, 2),
                "snapshots": snapshots,
                "growth_rate_pct": growth_rate,
            })
            return

        # GET /analytics
        if parts == ["analytics"]:
            with get_db() as conn:
                total_subs = conn.execute("SELECT COUNT(*) FROM subscriptions").fetchone()[0]
                churned = conn.execute(
                    "SELECT COUNT(*) FROM subscriptions WHERE status='cancelled'"
                ).fetchone()[0]
                churn_rate = round((churned / total_subs * 100), 2) if total_subs > 0 else 0.0

                # Average subscription length (days) for cancelled subs
                avg_len_row = conn.execute(
                    """SELECT AVG((cancelled_at - created_at)/86400.0)
                       FROM subscriptions WHERE status='cancelled' AND cancelled_at IS NOT NULL"""
                ).fetchone()
                avg_length_days = round(avg_len_row[0] or 0, 1)

                # LTV by plan
                ltv_rows = conn.execute(
                    """SELECT s.plan_id,
                              COUNT(*) as total,
                              SUM(i.amount) as total_revenue
                       FROM subscriptions s
                       LEFT JOIN invoices i ON s.sub_ref = i.sub_ref AND i.status='paid'
                       GROUP BY s.plan_id"""
                ).fetchall()
                ltv_by_plan = {}
                for r in ltv_rows:
                    count = r["total"] or 1
                    rev = r["total_revenue"] or 0
                    ltv_by_plan[r["plan_id"]] = {
                        "subscribers": count,
                        "total_revenue": round(rev, 2),
                        "avg_ltv": round(rev / count, 2),
                    }

                # Trial conversion rate
                trials = conn.execute(
                    "SELECT COUNT(*) FROM subscriptions WHERE trial_ends_at IS NOT NULL"
                ).fetchone()[0]
                converted = conn.execute(
                    """SELECT COUNT(*) FROM subscriptions
                       WHERE trial_ends_at IS NOT NULL AND status='active'"""
                ).fetchone()[0]
                trial_conversion = round((converted / trials * 100), 2) if trials > 0 else 0.0

            json_response(self, 200, {
                "total_subscriptions": total_subs,
                "churn_rate_pct": churn_rate,
                "avg_subscription_length_days": avg_length_days,
                "ltv_by_plan": ltv_by_plan,
                "trial_conversion_rate_pct": trial_conversion,
            })
            return

        json_response(self, 404, {"error": "Not found"})

    # ----------------------------------------------------------------- POST

    def do_POST(self):
        path, _ = parse_query(self.path)
        parts = [p for p in path.strip("/").split("/") if p]

        # POST /subscribe
        if parts == ["subscribe"]:
            body = parse_body(self)
            email = body.get("customer_email", "").strip()
            name = body.get("customer_name", "").strip()
            plan_id = body.get("plan_id", "").strip()
            billing_cycle = body.get("billing_cycle", "monthly").strip()
            payment_method_id = body.get("payment_method_id", "").strip()

            if not email or not plan_id:
                json_response(self, 400, {"error": "customer_email and plan_id are required"})
                return

            with get_db() as conn:
                plan_row = conn.execute(
                    "SELECT * FROM plans WHERE plan_id=? AND active=1", (plan_id,)
                ).fetchone()
            if not plan_row:
                json_response(self, 404, {"error": "Plan not found or inactive"})
                return

            plan = row_to_dict(plan_row)
            now = time.time()
            trial_days = plan.get("trial_days", 0)
            trial_ends_at = now + trial_days * 86400 if trial_days > 0 else None

            # Determine period
            if billing_cycle == "annual":
                period_end = now + 365 * 86400
                price_id = plan.get("stripe_price_id_annual", "")
            else:
                period_end = now + 30 * 86400
                price_id = plan.get("stripe_price_id_monthly", "")

            stripe_customer_id = ""
            stripe_sub_id = ""
            status = "trialing" if trial_days > 0 else "active"

            # Stripe integration (if key present)
            if STRIPE_SECRET_KEY and payment_method_id:
                cust = stripe_create_customer(email, name)
                if "id" in cust:
                    stripe_customer_id = cust["id"]
                    if price_id:
                        sub_resp = stripe_create_subscription(
                            stripe_customer_id, price_id,
                            payment_method_id, trial_days
                        )
                        if "id" in sub_resp:
                            stripe_sub_id = sub_resp["id"]
                            status = sub_resp.get("status", status)
                            if sub_resp.get("current_period_start"):
                                now = sub_resp["current_period_start"]
                            if sub_resp.get("current_period_end"):
                                period_end = sub_resp["current_period_end"]

            sub_ref = gen_ref("SUB")
            with get_db() as conn:
                conn.execute(
                    """INSERT INTO subscriptions
                       (sub_ref, customer_email, customer_name, plan_id, billing_cycle,
                        status, trial_ends_at, current_period_start, current_period_end,
                        stripe_sub_id, stripe_customer_id, created_at, updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (sub_ref, email, name, plan_id, billing_cycle,
                     status, trial_ends_at, now, period_end,
                     stripe_sub_id, stripe_customer_id, now, now)
                )

            send_welcome_email(email, name, plan["name"], trial_days)
            json_response(self, 201, {
                "sub_ref": sub_ref,
                "status": status,
                "plan_id": plan_id,
                "billing_cycle": billing_cycle,
                "trial_ends_at": trial_ends_at,
                "current_period_end": period_end,
                "stripe_customer_id": stripe_customer_id,
                "stripe_sub_id": stripe_sub_id,
            })
            return

        # POST /subscriptions/{sub_ref}/upgrade
        if len(parts) == 3 and parts[0] == "subscriptions" and parts[2] == "upgrade":
            if not verify_admin(self):
                json_response(self, 403, {"error": "Forbidden"})
                return
            sub_ref = parts[1]
            body = parse_body(self)
            new_plan_id = body.get("new_plan_id", "").strip()
            new_cycle = body.get("billing_cycle", "").strip()

            if not new_plan_id:
                json_response(self, 400, {"error": "new_plan_id is required"})
                return

            with get_db() as conn:
                sub = conn.execute(
                    "SELECT * FROM subscriptions WHERE sub_ref=?", (sub_ref,)
                ).fetchone()
                plan_row = conn.execute(
                    "SELECT * FROM plans WHERE plan_id=? AND active=1", (new_plan_id,)
                ).fetchone()

            if not sub:
                json_response(self, 404, {"error": "Subscription not found"})
                return
            if not plan_row:
                json_response(self, 404, {"error": "New plan not found"})
                return

            sub = row_to_dict(sub)
            plan = row_to_dict(plan_row)
            billing_cycle = new_cycle or sub["billing_cycle"]
            price_id = plan.get("stripe_price_id_annual", "") if billing_cycle == "annual" \
                else plan.get("stripe_price_id_monthly", "")

            now = time.time()
            stripe_sub_id = sub.get("stripe_sub_id", "")

            if STRIPE_SECRET_KEY and stripe_sub_id and price_id:
                stripe_update_subscription(stripe_sub_id, price_id)

            with get_db() as conn:
                conn.execute(
                    """UPDATE subscriptions SET plan_id=?, billing_cycle=?, updated_at=?
                       WHERE sub_ref=?""",
                    (new_plan_id, billing_cycle, now, sub_ref)
                )

            json_response(self, 200, {
                "sub_ref": sub_ref,
                "new_plan_id": new_plan_id,
                "billing_cycle": billing_cycle,
                "updated_at": now,
            })
            return

        # POST /subscriptions/{sub_ref}/cancel
        if len(parts) == 3 and parts[0] == "subscriptions" and parts[2] == "cancel":
            if not verify_admin(self):
                json_response(self, 403, {"error": "Forbidden"})
                return
            sub_ref = parts[1]
            body = parse_body(self)
            immediate = bool(body.get("immediate", False))

            with get_db() as conn:
                sub = conn.execute(
                    "SELECT * FROM subscriptions WHERE sub_ref=?", (sub_ref,)
                ).fetchone()

            if not sub:
                json_response(self, 404, {"error": "Subscription not found"})
                return

            sub = row_to_dict(sub)
            now = time.time()
            stripe_sub_id = sub.get("stripe_sub_id", "")

            if STRIPE_SECRET_KEY and stripe_sub_id:
                stripe_cancel_subscription(stripe_sub_id, at_period_end=not immediate)

            with get_db() as conn:
                if immediate:
                    conn.execute(
                        """UPDATE subscriptions SET status='cancelled', cancelled_at=?,
                           cancel_at_period_end=0, updated_at=? WHERE sub_ref=?""",
                        (now, now, sub_ref)
                    )
                else:
                    conn.execute(
                        """UPDATE subscriptions SET cancel_at_period_end=1, updated_at=?
                           WHERE sub_ref=?""",
                        (now, sub_ref)
                    )

            with get_db() as conn:
                plan_row = conn.execute(
                    "SELECT name FROM plans WHERE plan_id=?", (sub["plan_id"],)
                ).fetchone()
            plan_name = plan_row["name"] if plan_row else sub["plan_id"]

            if immediate:
                send_cancellation_email(sub["customer_email"], sub["customer_name"] or "", plan_name)

            json_response(self, 200, {
                "sub_ref": sub_ref,
                "immediate": immediate,
                "status": "cancelled" if immediate else "cancel_at_period_end",
                "cancelled_at": now if immediate else None,
            })
            return

        # POST /subscriptions/{sub_ref}/reactivate
        if len(parts) == 3 and parts[0] == "subscriptions" and parts[2] == "reactivate":
            if not verify_admin(self):
                json_response(self, 403, {"error": "Forbidden"})
                return
            sub_ref = parts[1]

            with get_db() as conn:
                sub = conn.execute(
                    "SELECT * FROM subscriptions WHERE sub_ref=?", (sub_ref,)
                ).fetchone()

            if not sub:
                json_response(self, 404, {"error": "Subscription not found"})
                return

            sub = row_to_dict(sub)
            now = time.time()
            stripe_sub_id = sub.get("stripe_sub_id", "")

            if STRIPE_SECRET_KEY and stripe_sub_id:
                stripe_reactivate_subscription(stripe_sub_id)

            with get_db() as conn:
                conn.execute(
                    """UPDATE subscriptions SET status='active', cancel_at_period_end=0,
                       cancelled_at=NULL, updated_at=? WHERE sub_ref=?""",
                    (now, sub_ref)
                )

            json_response(self, 200, {
                "sub_ref": sub_ref,
                "status": "active",
                "reactivated_at": now,
            })
            return

        # POST /plans
        if parts == ["plans"]:
            if not verify_admin(self):
                json_response(self, 403, {"error": "Forbidden"})
                return
            body = parse_body(self)
            plan_id = body.get("plan_id", "").strip()
            name = body.get("name", "").strip()
            if not plan_id or not name:
                json_response(self, 400, {"error": "plan_id and name are required"})
                return

            now = time.time()
            features = body.get("features", [])
            with get_db() as conn:
                try:
                    conn.execute(
                        """INSERT INTO plans
                           (plan_id, name, description, price_monthly, price_annual,
                            currency, features, trial_days, active,
                            stripe_price_id_monthly, stripe_price_id_annual, created_at)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (plan_id, name,
                         body.get("description", ""),
                         float(body.get("price_monthly", 0)),
                         float(body.get("price_annual", 0)),
                         body.get("currency", "AUD"),
                         json.dumps(features if isinstance(features, list) else [features]),
                         int(body.get("trial_days", 0)),
                         int(body.get("active", 1)),
                         body.get("stripe_price_id_monthly", ""),
                         body.get("stripe_price_id_annual", ""),
                         now)
                    )
                except sqlite3.IntegrityError:
                    json_response(self, 409, {"error": "plan_id already exists"})
                    return

            json_response(self, 201, {"plan_id": plan_id, "created": True})
            return

        json_response(self, 404, {"error": "Not found"})

    # ------------------------------------------------------------------ PUT

    def do_PUT(self):
        path, _ = parse_query(self.path)
        parts = [p for p in path.strip("/").split("/") if p]

        # PUT /plans/{plan_id}
        if len(parts) == 2 and parts[0] == "plans":
            if not verify_admin(self):
                json_response(self, 403, {"error": "Forbidden"})
                return
            plan_id = parts[1]
            body = parse_body(self)

            with get_db() as conn:
                existing = conn.execute(
                    "SELECT * FROM plans WHERE plan_id=?", (plan_id,)
                ).fetchone()
                if not existing:
                    json_response(self, 404, {"error": "Plan not found"})
                    return

                ex = row_to_dict(existing)
                features = body.get("features", json.loads(ex.get("features") or "[]"))
                if isinstance(features, list):
                    features_json = json.dumps(features)
                else:
                    features_json = ex.get("features", "[]")

                conn.execute(
                    """UPDATE plans SET
                       name=?, description=?, price_monthly=?, price_annual=?,
                       currency=?, features=?, trial_days=?, active=?,
                       stripe_price_id_monthly=?, stripe_price_id_annual=?
                       WHERE plan_id=?""",
                    (body.get("name", ex["name"]),
                     body.get("description", ex.get("description", "")),
                     float(body.get("price_monthly", ex["price_monthly"])),
                     float(body.get("price_annual", ex["price_annual"])),
                     body.get("currency", ex.get("currency", "AUD")),
                     features_json,
                     int(body.get("trial_days", ex.get("trial_days", 0))),
                     int(body.get("active", ex.get("active", 1))),
                     body.get("stripe_price_id_monthly", ex.get("stripe_price_id_monthly", "")),
                     body.get("stripe_price_id_annual", ex.get("stripe_price_id_annual", "")),
                     plan_id)
                )

            json_response(self, 200, {"plan_id": plan_id, "updated": True})
            return

        json_response(self, 404, {"error": "Not found"})

# ---------------------------------------------------------------------------
# Server startup
# ---------------------------------------------------------------------------

def main() -> None:
    init_db()

    # Start background daemons
    t_dunning = threading.Thread(target=dunning_daemon, daemon=True, name="dunning")
    t_dunning.start()

    t_mrr = threading.Thread(target=mrr_snapshot_daemon, daemon=True, name="mrr_snapshot")
    t_mrr.start()

    server = HTTPServer(("0.0.0.0", PORT), SubscriptionHandler)
    print(f"[{AGENT_NAME}] Listening on port {PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"[{AGENT_NAME}] Shutting down")
        server.server_close()


if __name__ == "__main__":
    main()
