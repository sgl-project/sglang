"""
FractalMesh OMEGA Titan — Affiliate Network Manager
Full affiliate marketing network: registration, tracking links, commission
management, Stripe Connect payouts, SendGrid notifications.
Port: 7885
Samuel James Hiotis | ABN 56 628 117 363
"""

import base64
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ---------------------------------------------------------------------------
# Vault loading — MUST be before any os.getenv calls
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
PORT             = int(os.getenv("AFFILIATE_NETWORK_PORT", "7885"))
AGENT_NAME       = "fm_affiliate_network"
DB_PATH          = Path.home() / "fmsaas" / "database" / "sovereign.db"
MIN_PAYOUT       = 50.0
AUTO_PAYOUT_MIN  = 200.0
PAYOUT_DAEMON_INTERVAL = 86400  # 24 h
CONVERSION_APPROVE_DAYS = 7

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create all required tables if they do not already exist."""
    with _get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS affiliates (
                id                INTEGER PRIMARY KEY,
                affiliate_id      TEXT    UNIQUE NOT NULL,
                name              TEXT    NOT NULL,
                email             TEXT    UNIQUE NOT NULL,
                website           TEXT,
                status            TEXT    DEFAULT 'pending',
                commission_rate   REAL    DEFAULT 0.20,
                total_clicks      INTEGER DEFAULT 0,
                total_conversions INTEGER DEFAULT 0,
                total_earnings    REAL    DEFAULT 0,
                pending_payout    REAL    DEFAULT 0,
                stripe_account_id TEXT,
                joined_at         REAL,
                updated_at        REAL
            );

            CREATE TABLE IF NOT EXISTS affiliate_links (
                id           INTEGER PRIMARY KEY,
                link_id      TEXT    UNIQUE NOT NULL,
                affiliate_id TEXT    NOT NULL,
                campaign     TEXT,
                target_url   TEXT    NOT NULL,
                short_code   TEXT    UNIQUE NOT NULL,
                clicks       INTEGER DEFAULT 0,
                conversions  INTEGER DEFAULT 0,
                created_at   REAL
            );

            CREATE TABLE IF NOT EXISTS clicks (
                id           INTEGER PRIMARY KEY,
                link_id      TEXT    NOT NULL,
                affiliate_id TEXT    NOT NULL,
                ip_hash      TEXT,
                user_agent   TEXT,
                referrer     TEXT,
                converted    INTEGER DEFAULT 0,
                created_at   REAL
            );

            CREATE TABLE IF NOT EXISTS conversions (
                id            INTEGER PRIMARY KEY,
                conversion_id TEXT    UNIQUE NOT NULL,
                affiliate_id  TEXT    NOT NULL,
                link_id       TEXT,
                order_ref     TEXT,
                order_value   REAL,
                commission    REAL,
                status        TEXT    DEFAULT 'pending',
                paid_out      INTEGER DEFAULT 0,
                created_at    REAL
            );

            CREATE TABLE IF NOT EXISTS payouts (
                id                 INTEGER PRIMARY KEY,
                payout_id          TEXT    UNIQUE NOT NULL,
                affiliate_id       TEXT    NOT NULL,
                amount             REAL,
                stripe_transfer_id TEXT,
                status             TEXT    DEFAULT 'pending',
                created_at         REAL
            );
        """)


# ---------------------------------------------------------------------------
# SendGrid helpers
# ---------------------------------------------------------------------------

def _sendgrid_send(to_email: str, subject: str, body_text: str) -> bool:
    """Send a plain-text email via SendGrid. Returns True on success."""
    api_key  = os.getenv("SENDGRID_API_KEY", "")
    from_email = os.getenv("SENDGRID_FROM_EMAIL", "noreply@fractalmesh.net")
    if not api_key:
        return False

    payload = json.dumps({
        "personalizations": [{"to": [{"email": to_email}]}],
        "from": {"email": from_email},
        "subject": subject,
        "content": [{"type": "text/plain", "value": body_text}],
    }).encode()

    req = urllib.request.Request(
        "https://api.sendgrid.com/v3/mail/send",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status in (200, 202)
    except urllib.error.URLError:
        return False


def send_welcome_email(name: str, email: str, affiliate_id: str) -> None:
    subject = "Welcome to FractalMesh Affiliate Network!"
    body = (
        f"Hi {name},\n\n"
        "Thank you for joining the FractalMesh Affiliate Network.\n\n"
        f"Your affiliate ID is: {affiliate_id}\n\n"
        "Your application is currently under review. You will receive another "
        "email once your account is approved.\n\n"
        "Once approved, you can create tracking links and start earning commissions.\n\n"
        "Best regards,\nFractalMesh Team"
    )
    _sendgrid_send(email, subject, body)


def send_approval_email(name: str, email: str, affiliate_id: str,
                        commission_rate: float) -> None:
    subject = "Your FractalMesh Affiliate Account Has Been Approved!"
    body = (
        f"Hi {name},\n\n"
        "Great news — your affiliate account has been approved!\n\n"
        f"Affiliate ID: {affiliate_id}\n"
        f"Commission Rate: {commission_rate * 100:.1f}%\n\n"
        "You can now log in and create tracking links to start earning commissions.\n\n"
        "Minimum payout threshold: $50.00 AUD\n\n"
        "Best regards,\nFractalMesh Team"
    )
    _sendgrid_send(email, subject, body)


def send_conversion_email(name: str, email: str, order_ref: str,
                          order_value: float, commission: float) -> None:
    subject = "You Earned a Commission — FractalMesh Affiliate"
    body = (
        f"Hi {name},\n\n"
        "Congratulations! You just earned a commission.\n\n"
        f"Order Reference: {order_ref}\n"
        f"Order Value:     ${order_value:.2f} AUD\n"
        f"Your Commission: ${commission:.2f} AUD\n\n"
        "This commission is pending approval and will be available for payout "
        "after 7 days.\n\n"
        "Best regards,\nFractalMesh Team"
    )
    _sendgrid_send(email, subject, body)


def send_payout_email(name: str, email: str, amount: float,
                      payout_id: str) -> None:
    subject = "Payout Processed — FractalMesh Affiliate"
    body = (
        f"Hi {name},\n\n"
        "Your payout has been processed!\n\n"
        f"Payout ID: {payout_id}\n"
        f"Amount:    ${amount:.2f} AUD\n\n"
        "Funds will appear in your Stripe account within 1–3 business days.\n\n"
        "Best regards,\nFractalMesh Team"
    )
    _sendgrid_send(email, subject, body)


# ---------------------------------------------------------------------------
# Stripe Connect transfer
# ---------------------------------------------------------------------------

def stripe_transfer(amount_cents: int, stripe_account_id: str,
                    description: str) -> dict:
    """Execute a Stripe Connect transfer. Returns the Stripe transfer object."""
    payload = urllib.parse.urlencode({
        "amount": str(amount_cents),
        "currency": "aud",
        "destination": stripe_account_id,
        "description": description,
    }).encode()
    req = urllib.request.Request(
        "https://api.stripe.com/v1/transfers", data=payload
    )
    credentials = base64.b64encode(
        f"{os.getenv('STRIPE_SECRET_KEY', '')}:".encode()
    ).decode()
    req.add_header("Authorization", f"Basic {credentials}")
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------

def _check_admin(handler: "AffiliateHandler") -> bool:
    """Return True if the X-Admin-Secret header matches ADMIN_SECRET."""
    secret = os.getenv("ADMIN_SECRET", "")
    provided = handler.headers.get("X-Admin-Secret", "")
    if not secret:
        return False
    return hmac.compare_digest(
        secret.encode(), provided.encode()
    )


# ---------------------------------------------------------------------------
# Background payout daemon
# ---------------------------------------------------------------------------

def _payout_daemon() -> None:
    """
    Daemon thread that runs every 24 h:
    1. Approve conversions older than 7 days (status pending → approved).
    2. Execute batch Stripe transfers for affiliates with pending_payout >= 200.
    """
    while True:
        time.sleep(PAYOUT_DAEMON_INTERVAL)
        try:
            _run_auto_payouts()
        except Exception as exc:  # pylint: disable=broad-except
            # Log but do not crash the daemon
            print(f"[payout-daemon] Error: {exc}")


def _run_auto_payouts() -> None:
    cutoff = time.time() - (CONVERSION_APPROVE_DAYS * 86400)
    with _get_db() as conn:
        # Approve old pending conversions
        conn.execute(
            """
            UPDATE conversions
               SET status = 'approved'
             WHERE status = 'pending'
               AND created_at <= ?
            """,
            (cutoff,),
        )
        conn.commit()

        # Find affiliates eligible for auto-payout
        rows = conn.execute(
            """
            SELECT affiliate_id, pending_payout, stripe_account_id,
                   name, email
              FROM affiliates
             WHERE pending_payout >= ?
               AND status = 'active'
               AND stripe_account_id IS NOT NULL
               AND stripe_account_id != ''
            """,
            (AUTO_PAYOUT_MIN,),
        ).fetchall()

    for row in rows:
        try:
            _execute_payout(
                affiliate_id=row["affiliate_id"],
                pending_amount=row["pending_payout"],
                stripe_account_id=row["stripe_account_id"],
                name=row["name"],
                email=row["email"],
            )
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[payout-daemon] Payout failed for {row['affiliate_id']}: {exc}")


def _execute_payout(affiliate_id: str, pending_amount: float,
                    stripe_account_id: str, name: str, email: str) -> dict:
    """Create a Stripe transfer and record the payout. Returns payout record."""
    amount_cents = int(pending_amount * 100)
    payout_id = f"pay_{secrets.token_urlsafe(12)}"
    description = f"FractalMesh affiliate payout for {affiliate_id}"

    transfer = stripe_transfer(amount_cents, stripe_account_id, description)
    transfer_id = transfer.get("id", "")

    now = time.time()
    with _get_db() as conn:
        conn.execute(
            """
            INSERT INTO payouts (payout_id, affiliate_id, amount,
                                 stripe_transfer_id, status, created_at)
            VALUES (?, ?, ?, ?, 'completed', ?)
            """,
            (payout_id, affiliate_id, pending_amount, transfer_id, now),
        )
        conn.execute(
            """
            UPDATE affiliates
               SET pending_payout = 0,
                   updated_at = ?
             WHERE affiliate_id = ?
            """,
            (now, affiliate_id),
        )
        # Mark related approved conversions as paid out
        conn.execute(
            """
            UPDATE conversions
               SET paid_out = 1,
                   status = 'paid'
             WHERE affiliate_id = ?
               AND status = 'approved'
               AND paid_out = 0
            """,
            (affiliate_id,),
        )
        conn.commit()

    send_payout_email(name, email, pending_amount, payout_id)
    return {"payout_id": payout_id, "amount": pending_amount,
            "stripe_transfer_id": transfer_id}


# ---------------------------------------------------------------------------
# JSON / HTTP utility helpers
# ---------------------------------------------------------------------------

def _json_response(handler: "AffiliateHandler", code: int,
                   data: dict | list) -> None:
    body = json.dumps(data, default=str).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_body(handler: "AffiliateHandler") -> dict:
    length = int(handler.headers.get("Content-Length", 0))
    if length == 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}


def _client_ip(handler: "AffiliateHandler") -> str:
    forwarded = handler.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return handler.client_address[0]


def _ip_hash(ip: str) -> str:
    return hashlib.sha256(ip.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

def handle_health(handler: "AffiliateHandler") -> None:
    _json_response(handler, 200, {
        "status": "ok",
        "port": PORT,
        "agent": AGENT_NAME,
    })


# --- /affiliates ---

def handle_list_affiliates(handler: "AffiliateHandler") -> None:
    if not _check_admin(handler):
        _json_response(handler, 403, {"error": "forbidden"})
        return

    with _get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM affiliates ORDER BY joined_at DESC"
        ).fetchall()

    _json_response(handler, 200, [dict(r) for r in rows])


def handle_register_affiliate(handler: "AffiliateHandler") -> None:
    body = _read_body(handler)
    name    = (body.get("name") or "").strip()
    email   = (body.get("email") or "").strip().lower()
    website = (body.get("website") or "").strip()
    commission_rate = float(body.get("commission_rate", 0.20))

    if not name or not email:
        _json_response(handler, 400, {"error": "name and email are required"})
        return

    # Clamp commission rate to reasonable bounds
    commission_rate = max(0.01, min(0.50, commission_rate))

    affiliate_id = f"aff_{secrets.token_urlsafe(10)}"
    now = time.time()

    try:
        with _get_db() as conn:
            conn.execute(
                """
                INSERT INTO affiliates
                    (affiliate_id, name, email, website, status,
                     commission_rate, joined_at, updated_at)
                VALUES (?, ?, ?, ?, 'pending', ?, ?, ?)
                """,
                (affiliate_id, name, email, website,
                 commission_rate, now, now),
            )
            conn.commit()
    except sqlite3.IntegrityError:
        _json_response(handler, 409,
                       {"error": "email already registered"})
        return

    # Fire welcome email in background so it does not block the response
    threading.Thread(
        target=send_welcome_email,
        args=(name, email, affiliate_id),
        daemon=True,
    ).start()

    _json_response(handler, 201, {
        "affiliate_id": affiliate_id,
        "name": name,
        "email": email,
        "status": "pending",
        "commission_rate": commission_rate,
        "message": "Registration successful. Awaiting approval.",
    })


def handle_approve_affiliate(handler: "AffiliateHandler") -> None:
    if not _check_admin(handler):
        _json_response(handler, 403, {"error": "forbidden"})
        return

    body = _read_body(handler)
    affiliate_id = (body.get("affiliate_id") or "").strip()
    if not affiliate_id:
        _json_response(handler, 400, {"error": "affiliate_id is required"})
        return

    now = time.time()
    with _get_db() as conn:
        result = conn.execute(
            "SELECT * FROM affiliates WHERE affiliate_id = ?",
            (affiliate_id,),
        ).fetchone()

        if not result:
            _json_response(handler, 404, {"error": "affiliate not found"})
            return

        if result["status"] == "active":
            _json_response(handler, 200,
                           {"message": "already active", "affiliate_id": affiliate_id})
            return

        conn.execute(
            """
            UPDATE affiliates
               SET status = 'active', updated_at = ?
             WHERE affiliate_id = ?
            """,
            (now, affiliate_id),
        )
        conn.commit()

    threading.Thread(
        target=send_approval_email,
        args=(result["name"], result["email"], affiliate_id,
              result["commission_rate"]),
        daemon=True,
    ).start()

    _json_response(handler, 200, {
        "affiliate_id": affiliate_id,
        "status": "active",
        "message": "Affiliate approved successfully.",
    })


def handle_get_affiliate(handler: "AffiliateHandler",
                         affiliate_id: str) -> None:
    with _get_db() as conn:
        row = conn.execute(
            "SELECT * FROM affiliates WHERE affiliate_id = ?",
            (affiliate_id,),
        ).fetchone()

    if not row:
        _json_response(handler, 404, {"error": "affiliate not found"})
        return

    _json_response(handler, 200, dict(row))


# --- /links ---

def handle_create_link(handler: "AffiliateHandler") -> None:
    body = _read_body(handler)
    affiliate_id = (body.get("affiliate_id") or "").strip()
    campaign     = (body.get("campaign") or "default").strip()
    target_url   = (body.get("target_url") or "").strip()

    if not affiliate_id or not target_url:
        _json_response(handler, 400,
                       {"error": "affiliate_id and target_url are required"})
        return

    with _get_db() as conn:
        aff = conn.execute(
            "SELECT * FROM affiliates WHERE affiliate_id = ?",
            (affiliate_id,),
        ).fetchone()

        if not aff:
            _json_response(handler, 404, {"error": "affiliate not found"})
            return

        if aff["status"] != "active":
            _json_response(handler, 403,
                           {"error": "affiliate account is not active"})
            return

        link_id    = f"lnk_{secrets.token_urlsafe(10)}"
        short_code = secrets.token_urlsafe(8)
        now = time.time()

        # Ensure short_code uniqueness (extremely unlikely collision but guard it)
        while conn.execute(
            "SELECT 1 FROM affiliate_links WHERE short_code = ?",
            (short_code,),
        ).fetchone():
            short_code = secrets.token_urlsafe(8)

        conn.execute(
            """
            INSERT INTO affiliate_links
                (link_id, affiliate_id, campaign, target_url,
                 short_code, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (link_id, affiliate_id, campaign, target_url, short_code, now),
        )
        conn.commit()

    _json_response(handler, 201, {
        "link_id": link_id,
        "affiliate_id": affiliate_id,
        "campaign": campaign,
        "target_url": target_url,
        "short_code": short_code,
        "tracking_url": f"/track/{short_code}",
    })


def handle_list_links(handler: "AffiliateHandler",
                      affiliate_id: str) -> None:
    with _get_db() as conn:
        rows = conn.execute(
            """
            SELECT * FROM affiliate_links
             WHERE affiliate_id = ?
             ORDER BY created_at DESC
            """,
            (affiliate_id,),
        ).fetchall()

    _json_response(handler, 200, [dict(r) for r in rows])


# --- /track ---

def handle_track(handler: "AffiliateHandler", short_code: str) -> None:
    with _get_db() as conn:
        link = conn.execute(
            "SELECT * FROM affiliate_links WHERE short_code = ?",
            (short_code,),
        ).fetchone()

        if not link:
            _json_response(handler, 404, {"error": "tracking link not found"})
            return

        ip    = _client_ip(handler)
        now   = time.time()
        ua    = handler.headers.get("User-Agent", "")[:512]
        ref   = handler.headers.get("Referer", "")[:512]

        conn.execute(
            """
            INSERT INTO clicks
                (link_id, affiliate_id, ip_hash, user_agent,
                 referrer, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (link["link_id"], link["affiliate_id"],
             _ip_hash(ip), ua, ref, now),
        )
        conn.execute(
            "UPDATE affiliate_links SET clicks = clicks + 1 WHERE link_id = ?",
            (link["link_id"],),
        )
        conn.execute(
            """
            UPDATE affiliates
               SET total_clicks = total_clicks + 1, updated_at = ?
             WHERE affiliate_id = ?
            """,
            (now, link["affiliate_id"]),
        )
        conn.commit()
        target_url = link["target_url"]

    handler.send_response(302)
    handler.send_header("Location", target_url)
    handler.end_headers()


# --- /convert ---

def handle_convert(handler: "AffiliateHandler") -> None:
    body = _read_body(handler)
    short_code  = (body.get("short_code") or "").strip()
    order_ref   = (body.get("order_ref") or "").strip()
    order_value = float(body.get("order_value", 0))

    if not short_code or not order_ref or order_value <= 0:
        _json_response(handler, 400, {
            "error": "short_code, order_ref and positive order_value are required"
        })
        return

    with _get_db() as conn:
        link = conn.execute(
            "SELECT * FROM affiliate_links WHERE short_code = ?",
            (short_code,),
        ).fetchone()

        if not link:
            _json_response(handler, 404, {"error": "tracking link not found"})
            return

        aff = conn.execute(
            "SELECT * FROM affiliates WHERE affiliate_id = ?",
            (link["affiliate_id"],),
        ).fetchone()

        if not aff:
            _json_response(handler, 404, {"error": "affiliate not found"})
            return

        commission    = round(order_value * aff["commission_rate"], 4)
        conversion_id = f"conv_{secrets.token_urlsafe(12)}"
        now           = time.time()

        conn.execute(
            """
            INSERT INTO conversions
                (conversion_id, affiliate_id, link_id, order_ref,
                 order_value, commission, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
            """,
            (conversion_id, aff["affiliate_id"], link["link_id"],
             order_ref, order_value, commission, now),
        )
        conn.execute(
            """
            UPDATE affiliate_links
               SET conversions = conversions + 1
             WHERE link_id = ?
            """,
            (link["link_id"],),
        )
        conn.execute(
            """
            UPDATE affiliates
               SET total_conversions = total_conversions + 1,
                   total_earnings    = total_earnings + ?,
                   pending_payout    = pending_payout + ?,
                   updated_at        = ?
             WHERE affiliate_id = ?
            """,
            (commission, commission, now, aff["affiliate_id"]),
        )
        # Mark the most recent click for this link as converted
        conn.execute(
            """
            UPDATE clicks SET converted = 1
             WHERE link_id = ?
               AND converted = 0
             ORDER BY created_at DESC
             LIMIT 1
            """,
            (link["link_id"],),
        )
        conn.commit()
        aff_name  = aff["name"]
        aff_email = aff["email"]

    threading.Thread(
        target=send_conversion_email,
        args=(aff_name, aff_email, order_ref, order_value, commission),
        daemon=True,
    ).start()

    _json_response(handler, 201, {
        "conversion_id": conversion_id,
        "affiliate_id": aff["affiliate_id"],
        "order_ref": order_ref,
        "order_value": order_value,
        "commission": commission,
        "status": "pending",
    })


# --- /conversions ---

def handle_list_conversions(handler: "AffiliateHandler",
                            affiliate_id: str) -> None:
    with _get_db() as conn:
        rows = conn.execute(
            """
            SELECT * FROM conversions
             WHERE affiliate_id = ?
             ORDER BY created_at DESC
            """,
            (affiliate_id,),
        ).fetchall()

    _json_response(handler, 200, [dict(r) for r in rows])


# --- /payout ---

def handle_request_payout(handler: "AffiliateHandler") -> None:
    body = _read_body(handler)
    affiliate_id = (body.get("affiliate_id") or "").strip()
    if not affiliate_id:
        _json_response(handler, 400, {"error": "affiliate_id is required"})
        return

    with _get_db() as conn:
        aff = conn.execute(
            "SELECT * FROM affiliates WHERE affiliate_id = ?",
            (affiliate_id,),
        ).fetchone()

    if not aff:
        _json_response(handler, 404, {"error": "affiliate not found"})
        return

    if aff["status"] != "active":
        _json_response(handler, 403,
                       {"error": "affiliate account is not active"})
        return

    pending = aff["pending_payout"]
    if pending < MIN_PAYOUT:
        _json_response(handler, 400, {
            "error": f"Minimum payout is ${MIN_PAYOUT:.2f}. "
                     f"Current pending: ${pending:.2f}"
        })
        return

    stripe_account_id = aff["stripe_account_id"] or ""
    if not stripe_account_id:
        _json_response(handler, 400, {
            "error": "No Stripe account linked. Please add your Stripe account ID."
        })
        return

    try:
        result = _execute_payout(
            affiliate_id=affiliate_id,
            pending_amount=pending,
            stripe_account_id=stripe_account_id,
            name=aff["name"],
            email=aff["email"],
        )
    except urllib.error.URLError as exc:
        _json_response(handler, 502, {"error": f"Stripe transfer failed: {exc}"})
        return
    except Exception as exc:  # pylint: disable=broad-except
        _json_response(handler, 500, {"error": str(exc)})
        return

    _json_response(handler, 200, {
        "payout_id": result["payout_id"],
        "amount": result["amount"],
        "stripe_transfer_id": result["stripe_transfer_id"],
        "status": "completed",
    })


def handle_list_payouts(handler: "AffiliateHandler",
                        affiliate_id: str) -> None:
    with _get_db() as conn:
        rows = conn.execute(
            """
            SELECT * FROM payouts
             WHERE affiliate_id = ?
             ORDER BY created_at DESC
            """,
            (affiliate_id,),
        ).fetchall()

    _json_response(handler, 200, [dict(r) for r in rows])


# --- /analytics ---

def handle_analytics_global(handler: "AffiliateHandler") -> None:
    if not _check_admin(handler):
        _json_response(handler, 403, {"error": "forbidden"})
        return

    with _get_db() as conn:
        totals = conn.execute(
            """
            SELECT
                COUNT(*)                             AS total_affiliates,
                SUM(total_clicks)                    AS total_clicks,
                SUM(total_conversions)               AS total_conversions,
                SUM(total_earnings)                  AS total_commissions_paid
            FROM affiliates
            """
        ).fetchone()

        total_clicks_raw = totals["total_clicks"] or 0
        total_conv_raw   = totals["total_conversions"] or 0
        conversion_rate  = (
            round(total_conv_raw / total_clicks_raw, 4)
            if total_clicks_raw > 0 else 0.0
        )

    _json_response(handler, 200, {
        "total_affiliates":      totals["total_affiliates"] or 0,
        "total_clicks":          total_clicks_raw,
        "total_conversions":     total_conv_raw,
        "total_commissions_paid": round(totals["total_commissions_paid"] or 0, 4),
        "conversion_rate":       conversion_rate,
    })


def handle_analytics_affiliate(handler: "AffiliateHandler",
                                affiliate_id: str) -> None:
    with _get_db() as conn:
        aff = conn.execute(
            "SELECT * FROM affiliates WHERE affiliate_id = ?",
            (affiliate_id,),
        ).fetchone()

        if not aff:
            _json_response(handler, 404, {"error": "affiliate not found"})
            return

        # Top campaigns by conversions
        campaigns = conn.execute(
            """
            SELECT campaign,
                   SUM(clicks)       AS clicks,
                   SUM(conversions)  AS conversions
              FROM affiliate_links
             WHERE affiliate_id = ?
             GROUP BY campaign
             ORDER BY conversions DESC, clicks DESC
             LIMIT 10
            """,
            (affiliate_id,),
        ).fetchall()

    clicks = aff["total_clicks"] or 0
    convs  = aff["total_conversions"] or 0
    rate   = round(convs / clicks, 4) if clicks > 0 else 0.0

    _json_response(handler, 200, {
        "affiliate_id":   affiliate_id,
        "name":           aff["name"],
        "clicks":         clicks,
        "conversions":    convs,
        "earnings":       round(aff["total_earnings"] or 0, 4),
        "pending_payout": round(aff["pending_payout"] or 0, 4),
        "conversion_rate": rate,
        "top_campaigns": [dict(c) for c in campaigns],
    })


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------

class AffiliateHandler(BaseHTTPRequestHandler):
    """BaseHTTPRequestHandler implementation for the Affiliate Network."""

    # Silence access log noise in production
    def log_message(self, fmt: str, *args) -> None:  # type: ignore[override]
        pass

    def log_error(self, fmt: str, *args) -> None:  # type: ignore[override]
        pass

    # ------------------------------------------------------------------
    # GET dispatcher
    # ------------------------------------------------------------------
    def do_GET(self) -> None:
        path = self.path.split("?")[0].rstrip("/")

        # /health
        if path == "/health":
            handle_health(self)
            return

        # /affiliates  (admin list)
        if path == "/affiliates":
            handle_list_affiliates(self)
            return

        # /affiliates/<id>
        if path.startswith("/affiliates/"):
            parts = path.split("/")
            if len(parts) == 3 and parts[2]:
                handle_get_affiliate(self, parts[2])
                return

        # /links/<affiliate_id>
        if path.startswith("/links/"):
            parts = path.split("/")
            if len(parts) == 3 and parts[2]:
                handle_list_links(self, parts[2])
                return

        # /track/<short_code>
        if path.startswith("/track/"):
            parts = path.split("/")
            if len(parts) == 3 and parts[2]:
                handle_track(self, parts[2])
                return

        # /conversions/<affiliate_id>
        if path.startswith("/conversions/"):
            parts = path.split("/")
            if len(parts) == 3 and parts[2]:
                handle_list_conversions(self, parts[2])
                return

        # /payouts/<affiliate_id>
        if path.startswith("/payouts/"):
            parts = path.split("/")
            if len(parts) == 3 and parts[2]:
                handle_list_payouts(self, parts[2])
                return

        # /analytics   (global, admin)
        if path == "/analytics":
            handle_analytics_global(self)
            return

        # /analytics/<affiliate_id>
        if path.startswith("/analytics/"):
            parts = path.split("/")
            if len(parts) == 3 and parts[2]:
                handle_analytics_affiliate(self, parts[2])
                return

        _json_response(self, 404, {"error": "not found"})

    # ------------------------------------------------------------------
    # POST dispatcher
    # ------------------------------------------------------------------
    def do_POST(self) -> None:
        path = self.path.split("?")[0].rstrip("/")

        if path == "/affiliates/register":
            handle_register_affiliate(self)
            return

        if path == "/affiliates/approve":
            handle_approve_affiliate(self)
            return

        if path == "/links":
            handle_create_link(self)
            return

        if path == "/convert":
            handle_convert(self)
            return

        if path == "/payout/request":
            handle_request_payout(self)
            return

        _json_response(self, 404, {"error": "not found"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    init_db()

    # Start background payout daemon
    daemon = threading.Thread(target=_payout_daemon, daemon=True,
                              name="payout-daemon")
    daemon.start()

    server = HTTPServer(("0.0.0.0", PORT), AffiliateHandler)
    print(f"[{AGENT_NAME}] Listening on port {PORT}")
    print(f"[{AGENT_NAME}] Database: {DB_PATH}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\n[{AGENT_NAME}] Shutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
