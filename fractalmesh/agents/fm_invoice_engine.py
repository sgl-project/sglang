#!/usr/bin/env python3
"""
FractalMesh OMEGA Titan — Professional Invoice & Billing Engine
Port: 7888

Full-featured Australian GST invoicing system: create invoices, send HTML
invoice emails, accept Stripe payments, track payment status, automated
overdue reminders.

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
AGENT_NAME = "fm_invoice_engine"
PORT       = int(os.environ.get("INVOICE_ENGINE_PORT", "7888"))

STRIPE_SECRET_KEY    = os.environ.get("STRIPE_SECRET_KEY", "")
SENDGRID_API_KEY     = os.environ.get("SENDGRID_API_KEY", "")
SENDGRID_FROM_EMAIL  = os.environ.get("SENDGRID_FROM_EMAIL", "")
BUSINESS_NAME        = os.environ.get("BUSINESS_NAME", "IronVision Nexus")
BUSINESS_ABN         = os.environ.get("BUSINESS_ABN", "56 628 117 363")
BUSINESS_ADDRESS     = os.environ.get("BUSINESS_ADDRESS", "Albury NSW 2640 Australia")
ADMIN_SECRET         = os.environ.get("ADMIN_SECRET", "")

STRIPE_API_BASE = "https://api.stripe.com/v1"

ROOT    = Path(os.environ.get("FRACTALMESH_HOME", Path.home() / "fmsaas"))
DB_PATH = ROOT / "database" / "sovereign.db"

for _d in (DB_PATH.parent, ROOT / "logs"):
    _d.mkdir(parents=True, exist_ok=True)

START_TIME = time.time()

GST_RATE          = 0.10
DEFAULT_DUE_DAYS  = 30
REMINDER_INTERVAL = 3600  # 1 hour between reminder daemon sweeps

# ---------------------------------------------------------------------------
# Database init
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS clients (
                id              INTEGER PRIMARY KEY,
                client_id       TEXT UNIQUE NOT NULL,
                name            TEXT NOT NULL,
                email           TEXT UNIQUE NOT NULL,
                phone           TEXT,
                company         TEXT,
                abn             TEXT,
                address         TEXT,
                city            TEXT,
                state           TEXT,
                postcode        TEXT,
                country         TEXT DEFAULT 'Australia',
                currency        TEXT DEFAULT 'AUD',
                payment_terms   INTEGER DEFAULT 30,
                notes           TEXT,
                created_at      REAL NOT NULL,
                updated_at      REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS invoices (
                id                       INTEGER PRIMARY KEY,
                invoice_id               TEXT UNIQUE NOT NULL,
                invoice_number           TEXT UNIQUE NOT NULL,
                client_id                TEXT NOT NULL,
                status                   TEXT DEFAULT 'draft',
                subtotal                 REAL DEFAULT 0,
                gst_amount               REAL DEFAULT 0,
                total                    REAL DEFAULT 0,
                currency                 TEXT DEFAULT 'AUD',
                due_date                 REAL,
                issue_date               REAL,
                paid_at                  REAL,
                stripe_payment_intent_id TEXT,
                notes                    TEXT,
                footer                   TEXT,
                created_at               REAL NOT NULL,
                updated_at               REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS invoice_items (
                id           INTEGER PRIMARY KEY,
                invoice_id   TEXT NOT NULL,
                description  TEXT NOT NULL,
                quantity     REAL DEFAULT 1,
                unit_price   REAL NOT NULL,
                gst_rate     REAL DEFAULT 0.10,
                line_total   REAL NOT NULL,
                gst_included INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS payments (
                id               INTEGER PRIMARY KEY,
                payment_id       TEXT UNIQUE NOT NULL,
                invoice_id       TEXT NOT NULL,
                amount           REAL NOT NULL,
                currency         TEXT DEFAULT 'AUD',
                method           TEXT NOT NULL,
                stripe_charge_id TEXT,
                notes            TEXT,
                created_at       REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS reminders (
                id           INTEGER PRIMARY KEY,
                invoice_id   TEXT NOT NULL,
                sent_at      REAL NOT NULL,
                reminder_type TEXT NOT NULL,
                days_overdue INTEGER NOT NULL
            );
        """)
    print(f"[{AGENT_NAME}] database initialised at {DB_PATH}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uid(prefix: str = "") -> str:
    return (prefix + secrets.token_hex(10)).upper()


def _now() -> float:
    return time.time()


def _ts_to_date(ts: float) -> str:
    """Return ISO date string from unix timestamp."""
    if not ts:
        return ""
    t = time.gmtime(ts)
    return f"{t.tm_year:04d}-{t.tm_mon:02d}-{t.tm_mday:02d}"


def _fmt_currency(amount: float, currency: str = "AUD") -> str:
    return f"{currency} {amount:,.2f}"


def _next_invoice_number() -> str:
    year = time.gmtime().tm_year
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM invoices WHERE invoice_number LIKE ?",
            (f"INV-{year}-%",),
        ).fetchone()
        seq = (row["cnt"] if row else 0) + 1
    return f"INV-{year}-{seq:04d}"


def _require_admin(handler: "InvoiceHandler") -> bool:
    """Return True if request carries valid admin secret; send 403 otherwise."""
    secret = handler.headers.get("X-Admin-Secret", "")
    if not ADMIN_SECRET:
        return True  # not configured — open in dev
    if not hmac.compare_digest(secret, ADMIN_SECRET):
        handler._send_json({"error": "forbidden"}, 403)
        return False
    return True


def _row_to_dict(row) -> dict:
    if row is None:
        return {}
    return dict(row)


# ---------------------------------------------------------------------------
# SendGrid helpers
# ---------------------------------------------------------------------------

def _sendgrid_send(to_email: str, to_name: str, subject: str,
                   html_body: str, attachments: list | None = None) -> bool:
    if not SENDGRID_API_KEY:
        print(f"[{AGENT_NAME}] SENDGRID_API_KEY not set — skipping email to {to_email}")
        return False

    payload: dict = {
        "personalizations": [{"to": [{"email": to_email, "name": to_name}]}],
        "from": {"email": SENDGRID_FROM_EMAIL, "name": BUSINESS_NAME},
        "subject": subject,
        "content": [{"type": "text/html", "value": html_body}],
    }
    if attachments:
        payload["attachments"] = attachments

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        "https://api.sendgrid.com/v3/mail/send",
        data=data,
        headers={
            "Authorization": f"Bearer {SENDGRID_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            ok = resp.status in (200, 202)
            print(f"[{AGENT_NAME}] email sent to {to_email} — status {resp.status}")
            return ok
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        print(f"[{AGENT_NAME}] sendgrid error {exc.code}: {body}")
        return False
    except Exception as exc:
        print(f"[{AGENT_NAME}] sendgrid exception: {exc}")
        return False


# ---------------------------------------------------------------------------
# Stripe helpers
# ---------------------------------------------------------------------------

def _stripe_post(path: str, params: dict) -> dict:
    if not STRIPE_SECRET_KEY:
        return {"error": "STRIPE_SECRET_KEY not configured"}
    data = urllib.parse.urlencode(params).encode()
    creds = base64.b64encode(f"{STRIPE_SECRET_KEY}:".encode()).decode()
    req = urllib.request.Request(
        f"{STRIPE_API_BASE}{path}",
        data=data,
        headers={
            "Authorization": f"Basic {creds}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        print(f"[{AGENT_NAME}] stripe error {exc.code}: {body}")
        try:
            return json.loads(body)
        except Exception:
            return {"error": body}
    except Exception as exc:
        print(f"[{AGENT_NAME}] stripe exception: {exc}")
        return {"error": str(exc)}


def _create_payment_intent(amount_aud: float, invoice_id: str,
                            invoice_number: str) -> dict:
    """Create a Stripe PaymentIntent for the given AUD amount (converted to cents)."""
    amount_cents = int(round(amount_aud * 100))
    return _stripe_post("/payment_intents", {
        "amount": amount_cents,
        "currency": "aud",
        "description": f"Invoice {invoice_number}",
        "metadata[invoice_id]": invoice_id,
        "metadata[invoice_number]": invoice_number,
        "automatic_payment_methods[enabled]": "true",
    })


# ---------------------------------------------------------------------------
# Invoice HTML renderer
# ---------------------------------------------------------------------------

def _render_invoice_html(invoice: dict, items: list[dict], client: dict) -> str:
    """Generate a professional HTML invoice."""

    def _row(label: str, value: str, bold: bool = False) -> str:
        weight = "font-weight:600;" if bold else ""
        return (
            f"<tr>"
            f"<td style='padding:4px 12px 4px 0;color:#555;{weight}'>{label}</td>"
            f"<td style='padding:4px 0;{weight}'>{value}</td>"
            f"</tr>"
        )

    # Build items table rows
    item_rows = ""
    for item in items:
        qty        = item.get("quantity", 1)
        unit_price = item.get("unit_price", 0)
        gst_rate   = item.get("gst_rate", GST_RATE)
        line_total = item.get("line_total", qty * unit_price)
        gst_amt    = line_total * gst_rate
        item_rows += (
            f"<tr>"
            f"<td style='padding:10px 8px;border-bottom:1px solid #eee;'>"
            f"{item.get('description','')}</td>"
            f"<td style='padding:10px 8px;border-bottom:1px solid #eee;"
            f"text-align:right;'>{qty}</td>"
            f"<td style='padding:10px 8px;border-bottom:1px solid #eee;"
            f"text-align:right;'>${unit_price:,.2f}</td>"
            f"<td style='padding:10px 8px;border-bottom:1px solid #eee;"
            f"text-align:right;'>{gst_rate*100:.0f}%</td>"
            f"<td style='padding:10px 8px;border-bottom:1px solid #eee;"
            f"text-align:right;font-weight:600;'>${line_total:,.2f}</td>"
            f"</tr>"
        )

    subtotal   = invoice.get("subtotal", 0)
    gst_amount = invoice.get("gst_amount", 0)
    total      = invoice.get("total", 0)
    currency   = invoice.get("currency", "AUD")
    status     = invoice.get("status", "draft").upper()

    status_color = {
        "PAID": "#27ae60", "OVERDUE": "#e74c3c",
        "SENT": "#2980b9", "DRAFT": "#95a5a6", "VOID": "#7f8c8d",
    }.get(status, "#2c3e50")

    issue_date = _ts_to_date(invoice.get("issue_date") or _now())
    due_date   = _ts_to_date(invoice.get("due_date", 0))
    inv_number = invoice.get("invoice_number", "")

    client_name    = client.get("name", "")
    client_company = client.get("company", "")
    client_email   = client.get("email", "")
    client_addr    = ", ".join(filter(None, [
        client.get("address", ""),
        client.get("city", ""),
        client.get("state", ""),
        client.get("postcode", ""),
        client.get("country", "Australia"),
    ]))
    client_abn = client.get("abn", "")

    notes  = invoice.get("notes", "") or ""
    footer = invoice.get("footer", "") or ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Invoice {inv_number}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                 'Helvetica Neue', Arial, sans-serif;
    font-size: 14px; color: #2c3e50; background: #f5f7fa; line-height: 1.6;
  }}
  .page {{
    max-width: 800px; margin: 30px auto; background: #fff;
    border-radius: 8px; box-shadow: 0 2px 20px rgba(0,0,0,.1);
    overflow: hidden;
  }}
  .header {{
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    color: #fff; padding: 40px;
  }}
  .header-top {{
    display: flex; justify-content: space-between; align-items: flex-start;
  }}
  .business-name {{
    font-size: 26px; font-weight: 700; letter-spacing: -0.5px; color: #e8f4fd;
  }}
  .business-details {{
    font-size: 12px; color: #a0b4c8; margin-top: 6px; line-height: 1.7;
  }}
  .invoice-badge {{
    text-align: right;
  }}
  .invoice-title {{
    font-size: 32px; font-weight: 800; color: #fff; text-transform: uppercase;
    letter-spacing: 2px;
  }}
  .invoice-number {{
    font-size: 14px; color: #a0b4c8; margin-top: 4px;
  }}
  .status-badge {{
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1px; margin-top: 8px;
    background: {status_color}; color: #fff;
  }}
  .meta-strip {{
    background: #f8fafc; border-bottom: 1px solid #e8ecf0;
    padding: 20px 40px; display: flex; gap: 40px;
  }}
  .meta-item {{ }}
  .meta-label {{ font-size: 11px; text-transform: uppercase;
                 letter-spacing: .8px; color: #8899aa; font-weight: 600; }}
  .meta-value {{ font-size: 15px; font-weight: 600; color: #2c3e50; margin-top: 2px; }}
  .body {{ padding: 40px; }}
  .parties {{ display: flex; gap: 40px; margin-bottom: 32px; }}
  .party {{ flex: 1; }}
  .party-label {{ font-size: 11px; text-transform: uppercase;
                  letter-spacing: .8px; color: #8899aa; font-weight: 600;
                  margin-bottom: 8px; }}
  .party-name {{ font-size: 16px; font-weight: 700; color: #1a1a2e; }}
  .party-detail {{ font-size: 13px; color: #556; margin-top: 3px; }}
  table.items {{
    width: 100%; border-collapse: collapse; margin-bottom: 24px;
  }}
  table.items thead tr {{
    background: #1a1a2e; color: #fff;
  }}
  table.items thead th {{
    padding: 12px 8px; text-align: left; font-size: 12px;
    text-transform: uppercase; letter-spacing: .6px; font-weight: 600;
  }}
  table.items thead th:not(:first-child) {{ text-align: right; }}
  table.items tbody tr:hover {{ background: #f8fafc; }}
  .totals-wrap {{ display: flex; justify-content: flex-end; margin-bottom: 32px; }}
  .totals-table {{ width: 280px; border-collapse: collapse; }}
  .totals-table td {{ padding: 8px 12px; font-size: 14px; }}
  .totals-table tr.subtotal td {{ color: #556; }}
  .totals-table tr.gst td {{ color: #556; border-bottom: 2px solid #1a1a2e; }}
  .totals-table tr.total td {{
    font-size: 18px; font-weight: 700; color: #1a1a2e; padding-top: 12px;
  }}
  .totals-table td:last-child {{ text-align: right; font-weight: 600; }}
  .section-title {{
    font-size: 12px; text-transform: uppercase; letter-spacing: .8px;
    color: #8899aa; font-weight: 600; margin-bottom: 8px;
    padding-bottom: 6px; border-bottom: 1px solid #eee;
  }}
  .notes-box {{
    background: #f8fafc; border-left: 4px solid #1a1a2e; border-radius: 0 6px 6px 0;
    padding: 16px; margin-bottom: 24px; font-size: 13px; color: #445;
    white-space: pre-wrap;
  }}
  .payment-box {{
    background: #eafaf1; border: 1px solid #a9dfbf; border-radius: 6px;
    padding: 16px; margin-bottom: 24px; font-size: 13px; color: #1e8449;
  }}
  .footer-strip {{
    background: #f8fafc; border-top: 1px solid #e8ecf0; padding: 20px 40px;
    text-align: center; font-size: 12px; color: #8899aa;
  }}
  @media print {{
    body {{ background: #fff; }}
    .page {{ box-shadow: none; margin: 0; border-radius: 0; }}
  }}
</style>
</head>
<body>
<div class="page">

  <!-- Header -->
  <div class="header">
    <div class="header-top">
      <div>
        <div class="business-name">{BUSINESS_NAME}</div>
        <div class="business-details">
          ABN: {BUSINESS_ABN}<br>
          {BUSINESS_ADDRESS}<br>
          {SENDGRID_FROM_EMAIL or ""}
        </div>
      </div>
      <div class="invoice-badge">
        <div class="invoice-title">Invoice</div>
        <div class="invoice-number">{inv_number}</div>
        <div class="status-badge">{status}</div>
      </div>
    </div>
  </div>

  <!-- Meta strip -->
  <div class="meta-strip">
    <div class="meta-item">
      <div class="meta-label">Issue Date</div>
      <div class="meta-value">{issue_date}</div>
    </div>
    <div class="meta-item">
      <div class="meta-label">Due Date</div>
      <div class="meta-value">{due_date}</div>
    </div>
    <div class="meta-item">
      <div class="meta-label">Currency</div>
      <div class="meta-value">{currency}</div>
    </div>
    <div class="meta-item">
      <div class="meta-label">Amount Due</div>
      <div class="meta-value" style="color:{status_color};">
        ${total:,.2f} {currency}
      </div>
    </div>
  </div>

  <!-- Body -->
  <div class="body">

    <!-- Parties -->
    <div class="parties">
      <div class="party">
        <div class="party-label">Bill From</div>
        <div class="party-name">{BUSINESS_NAME}</div>
        <div class="party-detail">ABN: {BUSINESS_ABN}</div>
        <div class="party-detail">{BUSINESS_ADDRESS}</div>
      </div>
      <div class="party">
        <div class="party-label">Bill To</div>
        <div class="party-name">{client_company or client_name}</div>
        {'<div class="party-detail">' + client_name + '</div>' if client_company else ''}
        {'<div class="party-detail">ABN: ' + client_abn + '</div>' if client_abn else ''}
        <div class="party-detail">{client_email}</div>
        {'<div class="party-detail">' + client_addr + '</div>' if client_addr else ''}
      </div>
    </div>

    <!-- Line Items -->
    <table class="items">
      <thead>
        <tr>
          <th style="width:45%">Description</th>
          <th>Qty</th>
          <th>Unit Price</th>
          <th>GST</th>
          <th>Amount</th>
        </tr>
      </thead>
      <tbody>
        {item_rows}
      </tbody>
    </table>

    <!-- Totals -->
    <div class="totals-wrap">
      <table class="totals-table">
        <tr class="subtotal">
          <td>Subtotal (ex. GST)</td>
          <td>${subtotal:,.2f}</td>
        </tr>
        <tr class="gst">
          <td>GST (10%)</td>
          <td>${gst_amount:,.2f}</td>
        </tr>
        <tr class="total">
          <td>Total</td>
          <td style="color:{status_color};">${total:,.2f} {currency}</td>
        </tr>
      </table>
    </div>

    <!-- Payment Instructions -->
    <div class="payment-box">
      <div class="section-title" style="color:#1e8449;border-color:#a9dfbf;">
        Payment Instructions
      </div>
      Please pay by <strong>{due_date}</strong> via bank transfer or online payment.<br>
      Quote invoice number <strong>{inv_number}</strong> in your payment reference.
    </div>

    <!-- Notes -->
    {'<div class="section-title">Notes</div><div class="notes-box">' + notes + '</div>' if notes else ''}

  </div><!-- /body -->

  <!-- Footer -->
  <div class="footer-strip">
    {footer or f'Thank you for your business — {BUSINESS_NAME} | ABN {BUSINESS_ABN} | Tax Invoice'}
    <br>
    <small>Generated by FractalMesh OMEGA Titan Billing Engine</small>
  </div>

</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Business logic helpers
# ---------------------------------------------------------------------------

def _calculate_invoice_totals(items: list[dict]) -> tuple[float, float, float]:
    """Return (subtotal, gst_amount, total) for a list of item dicts."""
    subtotal = 0.0
    for item in items:
        qty        = float(item.get("quantity", 1))
        unit_price = float(item.get("unit_price", 0))
        line_total = qty * unit_price
        item["line_total"] = round(line_total, 2)
        subtotal += line_total
    subtotal   = round(subtotal, 2)
    gst_amount = round(subtotal * GST_RATE, 2)
    total      = round(subtotal + gst_amount, 2)
    return subtotal, gst_amount, total


def _get_client(client_id: str) -> dict | None:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM clients WHERE client_id=?", (client_id,)
        ).fetchone()
    return _row_to_dict(row) if row else None


def _get_invoice(invoice_id: str) -> dict | None:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM invoices WHERE invoice_id=?", (invoice_id,)
        ).fetchone()
    return _row_to_dict(row) if row else None


def _get_invoice_items(invoice_id: str) -> list[dict]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM invoice_items WHERE invoice_id=? ORDER BY id",
            (invoice_id,),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def _get_invoice_payments(invoice_id: str) -> list[dict]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM payments WHERE invoice_id=? ORDER BY created_at",
            (invoice_id,),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Reminder daemon
# ---------------------------------------------------------------------------

REMINDER_THRESHOLDS = [1, 7, 14, 30]  # days overdue to send reminders


def _reminder_daemon() -> None:
    print(f"[{AGENT_NAME}] reminder daemon started")
    while True:
        try:
            _run_reminder_sweep()
        except Exception as exc:
            print(f"[{AGENT_NAME}] reminder sweep error: {exc}")
        time.sleep(REMINDER_INTERVAL)


def _run_reminder_sweep() -> None:
    now = _now()
    with _get_conn() as conn:
        overdue = conn.execute(
            "SELECT * FROM invoices WHERE status='sent' AND due_date < ?",
            (now,),
        ).fetchall()

    for inv_row in overdue:
        inv         = _row_to_dict(inv_row)
        invoice_id  = inv["invoice_id"]
        due_ts      = inv["due_date"]
        days_over   = int((now - due_ts) / 86400)

        for threshold in REMINDER_THRESHOLDS:
            if days_over < threshold:
                break
            # Check if this threshold reminder was already sent
            with _get_conn() as conn:
                existing = conn.execute(
                    "SELECT id FROM reminders WHERE invoice_id=? AND days_overdue=?",
                    (invoice_id, threshold),
                ).fetchone()
            if existing:
                continue

            # Send reminder
            client = _get_client(inv["client_id"])
            if not client:
                continue
            _send_reminder_email(inv, client, days_over, threshold)
            with _get_conn() as conn:
                conn.execute(
                    "INSERT INTO reminders (invoice_id, sent_at, reminder_type, days_overdue) "
                    "VALUES (?,?,?,?)",
                    (invoice_id, now, f"overdue_{threshold}d", threshold),
                )
            print(f"[{AGENT_NAME}] reminder sent: {invoice_id} {threshold}d overdue")


def _send_reminder_email(inv: dict, client: dict, days_over: int, threshold: int) -> None:
    to_email = client.get("email", "")
    to_name  = client.get("name", "")
    inv_num  = inv.get("invoice_number", "")
    total    = inv.get("total", 0)
    currency = inv.get("currency", "AUD")
    due_date = _ts_to_date(inv.get("due_date", 0))

    urgency = "Payment Reminder" if threshold <= 7 else "URGENT: Overdue Payment"
    subject = f"{urgency} — Invoice {inv_num} ({days_over} days overdue)"

    html = f"""
    <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;">
      <div style="background:#e74c3c;color:#fff;padding:20px;border-radius:6px 6px 0 0;">
        <h2 style="margin:0;">{urgency}</h2>
      </div>
      <div style="background:#fff;padding:24px;border:1px solid #eee;border-radius:0 0 6px 6px;">
        <p>Dear {to_name},</p>
        <p>This is a reminder that invoice <strong>{inv_num}</strong> for
           <strong>{currency} {total:,.2f}</strong> was due on <strong>{due_date}</strong>
           and is now <strong>{days_over} days overdue</strong>.</p>
        <p>Please arrange payment at your earliest convenience to avoid any service
           interruption.</p>
        <table style="width:100%;background:#f8f9fa;border-radius:6px;padding:16px;
                      border-collapse:collapse;margin:20px 0;">
          <tr><td style="padding:6px;color:#555;">Invoice</td>
              <td style="padding:6px;font-weight:700;">{inv_num}</td></tr>
          <tr><td style="padding:6px;color:#555;">Amount Due</td>
              <td style="padding:6px;font-weight:700;color:#e74c3c;">
                {currency} {total:,.2f}</td></tr>
          <tr><td style="padding:6px;color:#555;">Days Overdue</td>
              <td style="padding:6px;font-weight:700;color:#e74c3c;">{days_over}</td></tr>
        </table>
        <p>If you have already made payment, please disregard this notice.</p>
        <p>Thank you,<br><strong>{BUSINESS_NAME}</strong></p>
      </div>
    </div>
    """
    _sendgrid_send(to_email, to_name, subject, html)


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------

class InvoiceHandler(BaseHTTPRequestHandler):
    server_version = "FractalMesh/2.0"

    # -------- routing helpers -----------------------------------------------

    def _parse_path(self) -> tuple[str, dict]:
        parsed = urllib.parse.urlparse(self.path)
        qs     = dict(urllib.parse.parse_qsl(parsed.query))
        return parsed.path.rstrip("/"), qs

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def _send_json(self, data: dict | list, status: int = 200) -> None:
        body = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str, status: int = 200) -> None:
        body = html.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        print(f"[{AGENT_NAME}] {self.address_string()} {fmt % args}")

    # -------- CORS pre-flight -----------------------------------------------

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers",
                         "Content-Type,X-Admin-Secret")
        self.end_headers()

    # -------- GET -----------------------------------------------------------

    def do_GET(self):
        path, qs = self._parse_path()

        if path == "/health":
            self._handle_health()
        elif path == "/clients":
            self._handle_list_clients(qs)
        elif path.startswith("/clients/"):
            client_id = path[len("/clients/"):]
            self._handle_get_client(client_id)
        elif path == "/invoices":
            self._handle_list_invoices(qs)
        elif path.startswith("/invoices/") and path.endswith("/html"):
            invoice_id = path[len("/invoices/"):-len("/html")]
            self._handle_invoice_html(invoice_id)
        elif path.startswith("/invoices/"):
            invoice_id = path[len("/invoices/"):]
            self._handle_get_invoice(invoice_id)
        elif path == "/dashboard":
            self._handle_dashboard()
        else:
            self._send_json({"error": "not found"}, 404)

    # -------- POST ----------------------------------------------------------

    def do_POST(self):
        path, _ = self._parse_path()
        body     = self._read_body()

        if path == "/clients":
            self._handle_create_client(body)
        elif path == "/invoices":
            self._handle_create_invoice(body)
        elif path.startswith("/invoices/") and path.endswith("/send"):
            invoice_id = path[len("/invoices/"):-len("/send")]
            self._handle_send_invoice(invoice_id)
        elif path.startswith("/invoices/") and path.endswith("/pay"):
            invoice_id = path[len("/invoices/"):-len("/pay")]
            self._handle_pay_invoice(invoice_id, body)
        elif path.startswith("/invoices/") and path.endswith("/stripe-pay"):
            invoice_id = path[len("/invoices/"):-len("/stripe-pay")]
            self._handle_stripe_pay(invoice_id)
        elif path.startswith("/invoices/") and path.endswith("/void"):
            invoice_id = path[len("/invoices/"):-len("/void")]
            self._handle_void_invoice(invoice_id)
        else:
            self._send_json({"error": "not found"}, 404)

    # ====================================================================
    # Handler implementations
    # ====================================================================

    # ---- /health -----------------------------------------------------------

    def _handle_health(self):
        self._send_json({
            "status": "ok",
            "agent": AGENT_NAME,
            "port": PORT,
            "uptime_seconds": round(_now() - START_TIME, 1),
            "business": BUSINESS_NAME,
            "abn": BUSINESS_ABN,
        })

    # ---- /clients ----------------------------------------------------------

    def _handle_list_clients(self, qs: dict):
        if not _require_admin(self):
            return
        with _get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM clients ORDER BY name"
            ).fetchall()
        self._send_json([_row_to_dict(r) for r in rows])

    def _handle_create_client(self, body: dict):
        name  = (body.get("name") or "").strip()
        email = (body.get("email") or "").strip()
        if not name or not email:
            self._send_json({"error": "name and email required"}, 400)
            return

        client_id = _uid("CLI")
        now       = _now()
        try:
            with _get_conn() as conn:
                conn.execute(
                    """INSERT INTO clients
                       (client_id,name,email,phone,company,abn,address,city,
                        state,postcode,country,currency,payment_terms,notes,
                        created_at,updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        client_id, name, email,
                        body.get("phone", ""),
                        body.get("company", ""),
                        body.get("abn", ""),
                        body.get("address", ""),
                        body.get("city", ""),
                        body.get("state", ""),
                        body.get("postcode", ""),
                        body.get("country", "Australia"),
                        body.get("currency", "AUD"),
                        int(body.get("payment_terms", DEFAULT_DUE_DAYS)),
                        body.get("notes", ""),
                        now, now,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            self._send_json({"error": str(exc)}, 409)
            return

        client = _get_client(client_id)
        self._send_json({"success": True, "client": client}, 201)

    def _handle_get_client(self, client_id: str):
        client = _get_client(client_id)
        if not client:
            self._send_json({"error": "client not found"}, 404)
            return

        # Invoice summary
        with _get_conn() as conn:
            invoices = conn.execute(
                "SELECT * FROM invoices WHERE client_id=? ORDER BY created_at DESC",
                (client_id,),
            ).fetchall()
            totals = conn.execute(
                """SELECT
                     SUM(total) as lifetime_revenue,
                     SUM(CASE WHEN status='paid' THEN total ELSE 0 END) as paid_total,
                     SUM(CASE WHEN status IN ('sent','overdue') THEN total ELSE 0 END)
                       as outstanding_total,
                     COUNT(*) as invoice_count
                   FROM invoices WHERE client_id=?""",
                (client_id,),
            ).fetchone()

        self._send_json({
            "client": client,
            "invoices": [_row_to_dict(r) for r in invoices],
            "summary": _row_to_dict(totals),
        })

    # ---- /invoices ---------------------------------------------------------

    def _handle_create_invoice(self, body: dict):
        client_id = body.get("client_id", "")
        items_raw = body.get("items", [])

        client = _get_client(client_id)
        if not client:
            self._send_json({"error": "client not found"}, 404)
            return
        if not items_raw:
            self._send_json({"error": "items required"}, 400)
            return

        # Build validated item list
        items = []
        for it in items_raw:
            desc       = str(it.get("description", "")).strip()
            qty        = float(it.get("quantity", 1))
            unit_price = float(it.get("unit_price", 0))
            gst_rate   = float(it.get("gst_rate", GST_RATE))
            if not desc:
                self._send_json({"error": "item description required"}, 400)
                return
            items.append({
                "description": desc,
                "quantity": qty,
                "unit_price": unit_price,
                "gst_rate": gst_rate,
                "gst_included": 1,
            })

        subtotal, gst_amount, total = _calculate_invoice_totals(items)

        now        = _now()
        due_days   = int(body.get("due_days", client.get("payment_terms", DEFAULT_DUE_DAYS)))
        due_date   = now + due_days * 86400
        invoice_id = _uid("INV")
        inv_number = _next_invoice_number()

        with _get_conn() as conn:
            conn.execute(
                """INSERT INTO invoices
                   (invoice_id,invoice_number,client_id,status,subtotal,
                    gst_amount,total,currency,due_date,issue_date,
                    notes,footer,created_at,updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    invoice_id, inv_number, client_id, "draft",
                    subtotal, gst_amount, total,
                    client.get("currency", "AUD"),
                    due_date, now,
                    body.get("notes", ""),
                    body.get("footer", ""),
                    now, now,
                ),
            )
            for item in items:
                conn.execute(
                    """INSERT INTO invoice_items
                       (invoice_id,description,quantity,unit_price,gst_rate,
                        line_total,gst_included)
                       VALUES (?,?,?,?,?,?,?)""",
                    (
                        invoice_id,
                        item["description"],
                        item["quantity"],
                        item["unit_price"],
                        item["gst_rate"],
                        item["line_total"],
                        item["gst_included"],
                    ),
                )

        invoice = _get_invoice(invoice_id)
        self._send_json({
            "success": True,
            "invoice": invoice,
            "items": _get_invoice_items(invoice_id),
        }, 201)

    def _handle_list_invoices(self, qs: dict):
        if not _require_admin(self):
            return
        where_parts = []
        params      = []
        if qs.get("status"):
            where_parts.append("status=?")
            params.append(qs["status"])
        if qs.get("client_id"):
            where_parts.append("client_id=?")
            params.append(qs["client_id"])

        where_clause = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""
        with _get_conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM invoices {where_clause} ORDER BY created_at DESC",
                params,
            ).fetchall()
        self._send_json([_row_to_dict(r) for r in rows])

    def _handle_get_invoice(self, invoice_id: str):
        invoice = _get_invoice(invoice_id)
        if not invoice:
            self._send_json({"error": "invoice not found"}, 404)
            return
        client   = _get_client(invoice["client_id"])
        items    = _get_invoice_items(invoice_id)
        payments = _get_invoice_payments(invoice_id)
        self._send_json({
            "invoice": invoice,
            "client": client,
            "items": items,
            "payments": payments,
        })

    def _handle_send_invoice(self, invoice_id: str):
        invoice = _get_invoice(invoice_id)
        if not invoice:
            self._send_json({"error": "invoice not found"}, 404)
            return
        if invoice["status"] == "void":
            self._send_json({"error": "cannot send voided invoice"}, 400)
            return

        client = _get_client(invoice["client_id"])
        if not client:
            self._send_json({"error": "client not found"}, 404)
            return

        items    = _get_invoice_items(invoice_id)
        html_inv = _render_invoice_html(invoice, items, client)

        to_email  = client["email"]
        to_name   = client["name"]
        inv_num   = invoice["invoice_number"]
        total     = invoice["total"]
        currency  = invoice["currency"]
        due_date  = _ts_to_date(invoice["due_date"])

        subject = f"Invoice {inv_num} from {BUSINESS_NAME} — {currency} {total:,.2f} due {due_date}"

        # Email body wraps the HTML invoice inline
        email_html = f"""
        <div style="font-family:Arial,sans-serif;max-width:800px;margin:0 auto;">
          <p>Dear {to_name},</p>
          <p>Please find your invoice <strong>{inv_num}</strong> for
             <strong>{currency} {total:,.2f}</strong> below.<br>
             Payment is due by <strong>{due_date}</strong>.</p>
          <p>If you have any questions, please reply to this email.</p>
          <p>Thank you,<br><strong>{BUSINESS_NAME}</strong></p>
          <hr style="margin:30px 0;border:none;border-top:1px solid #eee;">
          {html_inv}
        </div>
        """
        ok = _sendgrid_send(to_email, to_name, subject, email_html)

        # Update status to 'sent'
        now = _now()
        with _get_conn() as conn:
            conn.execute(
                "UPDATE invoices SET status='sent', updated_at=? WHERE invoice_id=?",
                (now, invoice_id),
            )

        self._send_json({
            "success": True,
            "email_sent": ok,
            "invoice_id": invoice_id,
            "invoice_number": inv_num,
            "sent_to": to_email,
        })

    def _handle_pay_invoice(self, invoice_id: str, body: dict):
        invoice = _get_invoice(invoice_id)
        if not invoice:
            self._send_json({"error": "invoice not found"}, 404)
            return
        if invoice["status"] in ("paid", "void"):
            self._send_json({"error": f"invoice already {invoice['status']}"}, 400)
            return

        amount = float(body.get("amount", 0))
        method = str(body.get("method", "bank_transfer")).strip()
        if amount <= 0:
            self._send_json({"error": "amount must be positive"}, 400)
            return

        payment_id = _uid("PAY")
        now        = _now()
        with _get_conn() as conn:
            conn.execute(
                """INSERT INTO payments
                   (payment_id,invoice_id,amount,currency,method,
                    stripe_charge_id,notes,created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    payment_id, invoice_id, amount,
                    invoice.get("currency", "AUD"),
                    method,
                    body.get("stripe_charge_id", ""),
                    body.get("notes", ""),
                    now,
                ),
            )

        # Check if fully paid
        with _get_conn() as conn:
            paid_row = conn.execute(
                "SELECT SUM(amount) as paid FROM payments WHERE invoice_id=?",
                (invoice_id,),
            ).fetchone()
        total_paid = paid_row["paid"] if paid_row and paid_row["paid"] else 0

        is_fully_paid = total_paid >= invoice["total"] - 0.01

        if is_fully_paid:
            with _get_conn() as conn:
                conn.execute(
                    "UPDATE invoices SET status='paid', paid_at=?, updated_at=? "
                    "WHERE invoice_id=?",
                    (now, now, invoice_id),
                )
            # Send confirmation email
            client = _get_client(invoice["client_id"])
            if client:
                _send_payment_confirmation(invoice, client, total_paid)

        self._send_json({
            "success": True,
            "payment_id": payment_id,
            "amount_paid": amount,
            "total_paid": total_paid,
            "invoice_total": invoice["total"],
            "is_fully_paid": is_fully_paid,
            "status": "paid" if is_fully_paid else invoice["status"],
        })

    def _handle_stripe_pay(self, invoice_id: str):
        invoice = _get_invoice(invoice_id)
        if not invoice:
            self._send_json({"error": "invoice not found"}, 404)
            return
        if invoice["status"] in ("paid", "void"):
            self._send_json({"error": f"invoice {invoice['status']}"}, 400)
            return

        result = _create_payment_intent(
            invoice["total"],
            invoice_id,
            invoice["invoice_number"],
        )
        if "error" in result and "client_secret" not in result:
            self._send_json({"error": result.get("error", "stripe error")}, 502)
            return

        # Store intent ID
        intent_id = result.get("id", "")
        if intent_id:
            with _get_conn() as conn:
                conn.execute(
                    "UPDATE invoices SET stripe_payment_intent_id=?, updated_at=? "
                    "WHERE invoice_id=?",
                    (intent_id, _now(), invoice_id),
                )

        self._send_json({
            "success": True,
            "client_secret": result.get("client_secret"),
            "payment_intent_id": intent_id,
            "amount": invoice["total"],
            "currency": invoice["currency"],
            "invoice_id": invoice_id,
            "invoice_number": invoice["invoice_number"],
        })

    def _handle_void_invoice(self, invoice_id: str):
        if not _require_admin(self):
            return
        invoice = _get_invoice(invoice_id)
        if not invoice:
            self._send_json({"error": "invoice not found"}, 404)
            return
        if invoice["status"] == "paid":
            self._send_json({"error": "cannot void a paid invoice"}, 400)
            return
        if invoice["status"] == "void":
            self._send_json({"error": "invoice already voided"}, 400)
            return

        with _get_conn() as conn:
            conn.execute(
                "UPDATE invoices SET status='void', updated_at=? WHERE invoice_id=?",
                (_now(), invoice_id),
            )
        self._send_json({
            "success": True,
            "invoice_id": invoice_id,
            "status": "void",
        })

    def _handle_invoice_html(self, invoice_id: str):
        invoice = _get_invoice(invoice_id)
        if not invoice:
            self._send_json({"error": "invoice not found"}, 404)
            return
        client = _get_client(invoice["client_id"]) or {}
        items  = _get_invoice_items(invoice_id)
        html   = _render_invoice_html(invoice, items, client)
        self._send_html(html)

    def _handle_dashboard(self):
        if not _require_admin(self):
            return
        now             = _now()
        month_start     = _month_start_ts(now)

        with _get_conn() as conn:
            # Unpaid total
            unpaid_row = conn.execute(
                "SELECT SUM(total) as v FROM invoices "
                "WHERE status IN ('sent','draft')"
            ).fetchone()
            unpaid_total = unpaid_row["v"] or 0 if unpaid_row else 0

            # Overdue count
            overdue_row = conn.execute(
                "SELECT COUNT(*) as cnt FROM invoices "
                "WHERE status='sent' AND due_date < ?",
                (now,),
            ).fetchone()
            overdue_count = overdue_row["cnt"] if overdue_row else 0

            # Paid this month
            paid_month_row = conn.execute(
                "SELECT SUM(total) as v FROM invoices "
                "WHERE status='paid' AND paid_at >= ?",
                (month_start,),
            ).fetchone()
            paid_this_month = paid_month_row["v"] or 0 if paid_month_row else 0

            # Top 5 clients by total revenue
            top_clients = conn.execute(
                """SELECT c.name, c.company, c.client_id,
                          SUM(i.total) as revenue, COUNT(i.id) as invoice_count
                   FROM invoices i
                   JOIN clients c ON c.client_id = i.client_id
                   WHERE i.status = 'paid'
                   GROUP BY c.client_id
                   ORDER BY revenue DESC
                   LIMIT 5""",
            ).fetchall()

            # Recent invoices
            recent = conn.execute(
                "SELECT invoice_id,invoice_number,client_id,status,"
                "total,currency,due_date,issue_date "
                "FROM invoices ORDER BY created_at DESC LIMIT 10"
            ).fetchall()

            # Invoice status breakdown
            breakdown = conn.execute(
                "SELECT status, COUNT(*) as cnt, SUM(total) as total "
                "FROM invoices GROUP BY status"
            ).fetchall()

        self._send_json({
            "unpaid_total": round(unpaid_total, 2),
            "overdue_count": overdue_count,
            "paid_this_month": round(paid_this_month, 2),
            "top_clients": [_row_to_dict(r) for r in top_clients],
            "recent_invoices": [_row_to_dict(r) for r in recent],
            "status_breakdown": [_row_to_dict(r) for r in breakdown],
        })


# ---------------------------------------------------------------------------
# Payment confirmation email
# ---------------------------------------------------------------------------

def _send_payment_confirmation(invoice: dict, client: dict, total_paid: float) -> None:
    to_email = client.get("email", "")
    to_name  = client.get("name", "")
    inv_num  = invoice.get("invoice_number", "")
    total    = invoice.get("total", 0)
    currency = invoice.get("currency", "AUD")
    subject  = f"Payment Received — Invoice {inv_num} — Thank You!"

    html = f"""
    <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;">
      <div style="background:#27ae60;color:#fff;padding:20px;border-radius:6px 6px 0 0;">
        <h2 style="margin:0;">Payment Received — Thank You!</h2>
      </div>
      <div style="background:#fff;padding:24px;border:1px solid #eee;border-radius:0 0 6px 6px;">
        <p>Dear {to_name},</p>
        <p>We have received your payment for invoice <strong>{inv_num}</strong>.
           Your account is now up to date.</p>
        <table style="width:100%;background:#f8f9fa;border-radius:6px;padding:16px;
                      border-collapse:collapse;margin:20px 0;">
          <tr><td style="padding:6px;color:#555;">Invoice</td>
              <td style="padding:6px;font-weight:700;">{inv_num}</td></tr>
          <tr><td style="padding:6px;color:#555;">Amount Paid</td>
              <td style="padding:6px;font-weight:700;color:#27ae60;">
                {currency} {total_paid:,.2f}</td></tr>
          <tr><td style="padding:6px;color:#555;">Invoice Total</td>
              <td style="padding:6px;">{currency} {total:,.2f}</td></tr>
          <tr><td style="padding:6px;color:#555;">Status</td>
              <td style="padding:6px;font-weight:700;color:#27ae60;">PAID</td></tr>
        </table>
        <p>Thank you for your prompt payment. We look forward to continuing to
           work with you.</p>
        <p>Warm regards,<br><strong>{BUSINESS_NAME}</strong><br>
           ABN: {BUSINESS_ABN}</p>
      </div>
    </div>
    """
    _sendgrid_send(to_email, to_name, subject, html)


# ---------------------------------------------------------------------------
# Utility: month start timestamp
# ---------------------------------------------------------------------------

def _month_start_ts(ts: float) -> float:
    t = time.gmtime(ts)
    return time.mktime((t.tm_year, t.tm_mon, 1, 0, 0, 0, 0, 0, 0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    init_db()

    # Start reminder daemon
    t = threading.Thread(target=_reminder_daemon, daemon=True, name="reminder-daemon")
    t.start()

    server = HTTPServer(("0.0.0.0", PORT), InvoiceHandler)
    print(f"[{AGENT_NAME}] listening on port {PORT}")
    print(f"[{AGENT_NAME}] business: {BUSINESS_NAME} | ABN: {BUSINESS_ABN}")
    print(f"[{AGENT_NAME}] database: {DB_PATH}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\n[{AGENT_NAME}] shutting down")
        server.server_close()


if __name__ == "__main__":
    main()
