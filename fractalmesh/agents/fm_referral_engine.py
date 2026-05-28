"""
FractalMesh OMEGA Titan — Referral & Affiliate Commission Engine
Port: 7878
Samuel James Hiotis | ABN 56 628 117 363

Full referral and affiliate marketing system: unique referral links, click
tracking, conversion recording, commission calculation, and Stripe payouts.
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
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

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
BASE_DIR = Path.home() / "fmsaas"
DB_PATH  = BASE_DIR / "database" / "sovereign.db"
LOG_PATH = BASE_DIR / "logs" / "fm_referral_engine.log"

for _d in (DB_PATH.parent, LOG_PATH.parent):
    _d.mkdir(parents=True, exist_ok=True)

PORT              = int(os.environ.get("REFERRAL_ENGINE_PORT", "7878"))
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
SENDGRID_API_KEY  = os.environ.get("SENDGRID_API_KEY", "")
SENDGRID_FROM     = os.environ.get("SENDGRID_FROM_EMAIL", "noreply@fractalmesh.io")
ADMIN_SECRET      = os.environ.get("ADMIN_SECRET", "")

START_TIME = time.time()

COMMISSION_TIERS = {
    "standard": 0.20,
    "premium":  0.30,
    "partner":  0.40,
}

PAYOUT_MIN_AUD    = 50.0
PAYOUT_INTERVAL_S = 86400  # 24 h

# 1x1 transparent GIF
_PIXEL_GIF = base64.b64decode(
    "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
)

# ---------------------------------------------------------------------------
# Logging (simple file + stdout)
# ---------------------------------------------------------------------------
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [REFERRAL] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("fm_referral_engine")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS affiliates (
                id                INTEGER PRIMARY KEY,
                affiliate_id      TEXT UNIQUE NOT NULL,
                email             TEXT UNIQUE NOT NULL,
                name              TEXT,
                company           TEXT,
                status            TEXT DEFAULT 'active',
                tier              TEXT DEFAULT 'standard',
                commission_rate   REAL DEFAULT 0.20,
                total_clicks      INTEGER DEFAULT 0,
                total_conversions INTEGER DEFAULT 0,
                total_earned      REAL DEFAULT 0,
                total_paid        REAL DEFAULT 0,
                stripe_account_id TEXT,
                created_at        REAL
            );

            CREATE TABLE IF NOT EXISTS referral_links (
                id              INTEGER PRIMARY KEY,
                link_code       TEXT UNIQUE NOT NULL,
                affiliate_id    TEXT NOT NULL,
                campaign        TEXT DEFAULT 'default',
                destination_url TEXT,
                clicks          INTEGER DEFAULT 0,
                conversions     INTEGER DEFAULT 0,
                created_at      REAL
            );

            CREATE TABLE IF NOT EXISTS clicks (
                id          INTEGER PRIMARY KEY,
                link_code   TEXT NOT NULL,
                ip_hash     TEXT,
                user_agent  TEXT,
                referrer    TEXT,
                landed_at   REAL,
                converted   INTEGER DEFAULT 0,
                converted_at REAL
            );

            CREATE TABLE IF NOT EXISTS conversions (
                id               INTEGER PRIMARY KEY,
                click_id         INTEGER,
                link_code        TEXT NOT NULL,
                affiliate_id     TEXT NOT NULL,
                conversion_type  TEXT,
                order_value      REAL DEFAULT 0,
                commission_rate  REAL,
                commission_amount REAL,
                status           TEXT DEFAULT 'pending',
                order_ref        TEXT,
                created_at       REAL,
                approved_at      REAL
            );

            CREATE TABLE IF NOT EXISTS payouts (
                id                 INTEGER PRIMARY KEY,
                affiliate_id       TEXT NOT NULL,
                amount             REAL,
                currency           TEXT DEFAULT 'AUD',
                period_start       REAL,
                period_end         REAL,
                conversions_count  INTEGER,
                status             TEXT DEFAULT 'pending',
                stripe_transfer_id TEXT,
                created_at         REAL,
                paid_at            REAL
            );

            CREATE INDEX IF NOT EXISTS idx_clicks_link  ON clicks(link_code);
            CREATE INDEX IF NOT EXISTS idx_conv_link    ON conversions(link_code);
            CREATE INDEX IF NOT EXISTS idx_conv_aff     ON conversions(affiliate_id);
            CREATE INDEX IF NOT EXISTS idx_payouts_aff  ON payouts(affiliate_id);
        """)
    log.info("Database initialised at %s", DB_PATH)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _json_resp(handler: "ReferralHandler", status: int, data: dict) -> None:
    body = json.dumps(data, default=str).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _require_admin(handler: "ReferralHandler") -> bool:
    """Return True if the request carries a valid ADMIN_SECRET."""
    token = handler.headers.get("X-Admin-Secret", "")
    if not ADMIN_SECRET:
        _json_resp(handler, 503, {"error": "admin auth not configured"})
        return False
    if not hmac.compare_digest(token, ADMIN_SECRET):
        _json_resp(handler, 401, {"error": "unauthorized"})
        return False
    return True


def _hash_ip(ip: str) -> str:
    return hashlib.sha256(ip.encode()).hexdigest()


def _affiliate_id_from_email(email: str) -> str:
    raw = f"{email}{time.time()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _generate_link_code() -> str:
    return f"fm_{secrets.token_urlsafe(8)}"


def _row_to_dict(row: sqlite3.Row) -> dict:
    return dict(row)

# ---------------------------------------------------------------------------
# Stripe helpers
# ---------------------------------------------------------------------------

def _stripe_transfer(affiliate_id: str, stripe_account: str,
                     amount_aud: float) -> tuple[bool, str]:
    """Call Stripe Transfers API. Returns (success, transfer_id_or_error)."""
    if not STRIPE_SECRET_KEY:
        return False, "stripe_not_configured"
    amount_cents = int(round(amount_aud * 100))
    payload = (
        f"amount={amount_cents}"
        f"&currency=aud"
        f"&destination={stripe_account}"
        f"&transfer_group=referral_{affiliate_id}"
    ).encode()
    req = urllib.request.Request(
        "https://api.stripe.com/v1/transfers",
        data=payload,
        headers={
            "Authorization": f"Bearer {STRIPE_SECRET_KEY}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode())
            return True, result.get("id", "")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()
        log.error("Stripe transfer error %s: %s", exc.code, body)
        return False, f"http_{exc.code}"
    except Exception as exc:  # noqa: BLE001
        log.error("Stripe transfer exception: %s", exc)
        return False, str(exc)

# ---------------------------------------------------------------------------
# SendGrid helpers
# ---------------------------------------------------------------------------

def _send_email(to_email: str, subject: str, text_body: str) -> bool:
    if not SENDGRID_API_KEY:
        log.warning("SendGrid not configured; skipping email to %s", to_email)
        return False
    payload = json.dumps({
        "personalizations": [{"to": [{"email": to_email}]}],
        "from": {"email": SENDGRID_FROM},
        "subject": subject,
        "content": [{"type": "text/plain", "value": text_body}],
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
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status in (200, 202)
    except urllib.error.HTTPError as exc:
        log.error("SendGrid error %s for %s", exc.code, to_email)
        return False
    except Exception as exc:  # noqa: BLE001
        log.error("SendGrid exception: %s", exc)
        return False


def _send_welcome_email(email: str, name: str, affiliate_id: str,
                        link_code: str) -> None:
    subject = "Welcome to the FractalMesh Affiliate Program!"
    body = (
        f"Hi {name},\n\n"
        f"Welcome to FractalMesh OMEGA Titan's affiliate program!\n\n"
        f"Your Affiliate ID: {affiliate_id}\n"
        f"Your first referral link code: {link_code}\n\n"
        f"Share your link to start earning commissions. "
        f"You will receive payouts once your balance exceeds $50 AUD.\n\n"
        f"— The FractalMesh Team"
    )
    _send_email(email, subject, body)


def _send_payout_email(email: str, name: str, amount: float,
                       transfer_id: str) -> None:
    subject = "FractalMesh Affiliate Payout Processed"
    body = (
        f"Hi {name},\n\n"
        f"Your affiliate payout of ${amount:.2f} AUD has been processed.\n"
        f"Stripe transfer ID: {transfer_id}\n\n"
        f"Thank you for being a FractalMesh affiliate!\n\n"
        f"— The FractalMesh Team"
    )
    _send_email(email, subject, body)

# ---------------------------------------------------------------------------
# Payout background thread
# ---------------------------------------------------------------------------

def _process_payouts() -> None:
    """Find affiliates with unpaid balance > threshold, create payouts."""
    now = time.time()
    try:
        with get_db() as conn:
            rows = conn.execute(
                """
                SELECT a.affiliate_id, a.email, a.name, a.stripe_account_id,
                       a.total_earned - a.total_paid AS pending_balance
                FROM affiliates a
                WHERE a.status = 'active'
                  AND (a.total_earned - a.total_paid) > ?
                """,
                (PAYOUT_MIN_AUD,),
            ).fetchall()

        for row in rows:
            aff_id      = row["affiliate_id"]
            balance     = row["pending_balance"]
            stripe_acct = row["stripe_account_id"]

            # Count approved conversions not yet included in a payout
            with get_db() as conn:
                conv_count = conn.execute(
                    """
                    SELECT COUNT(*) FROM conversions
                    WHERE affiliate_id = ?
                      AND status = 'approved'
                      AND id NOT IN (
                          SELECT COALESCE(id, -1) FROM payouts
                          WHERE affiliate_id = ? AND status IN ('pending','paid')
                      )
                    """,
                    (aff_id, aff_id),
                ).fetchone()[0]

                # Insert payout record
                conn.execute(
                    """
                    INSERT INTO payouts
                        (affiliate_id, amount, currency, period_start,
                         period_end, conversions_count, status, created_at)
                    VALUES (?, ?, 'AUD', ?, ?, ?, 'pending', ?)
                    """,
                    (aff_id, balance, 0.0, now, conv_count, now),
                )
                payout_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

            log.info("Payout created for affiliate %s: $%.2f AUD", aff_id, balance)

            # Attempt Stripe transfer
            if stripe_acct:
                success, transfer_id = _stripe_transfer(aff_id, stripe_acct, balance)
                if success:
                    with get_db() as conn:
                        conn.execute(
                            """
                            UPDATE payouts SET status='paid',
                                stripe_transfer_id=?, paid_at=?
                            WHERE id=?
                            """,
                            (transfer_id, now, payout_id),
                        )
                        conn.execute(
                            "UPDATE affiliates SET total_paid=total_paid+? WHERE affiliate_id=?",
                            (balance, aff_id),
                        )
                    log.info("Stripe transfer %s for affiliate %s", transfer_id, aff_id)
                    _send_payout_email(row["email"], row["name"] or "", balance, transfer_id)
                else:
                    log.error("Stripe transfer failed for affiliate %s: %s", aff_id, transfer_id)
            else:
                log.info("No Stripe account for affiliate %s; payout queued", aff_id)

    except Exception as exc:  # noqa: BLE001
        log.error("Payout thread error: %s", exc)


def _payout_loop() -> None:
    while True:
        time.sleep(PAYOUT_INTERVAL_S)
        log.info("Running scheduled payout check…")
        _process_payouts()


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class ReferralHandler(BaseHTTPRequestHandler):
    server_version = "FractalMeshReferral/1.0"

    def log_message(self, fmt, *args):  # silence default access log
        log.debug(fmt, *args)

    # ---- routing -----------------------------------------------------------

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path.rstrip("/")
        parts  = [p for p in path.split("/") if p]

        if path == "/health":
            return self._health()

        if path == "/affiliates":
            return self._list_affiliates()

        if path == "/conversions":
            return self._list_conversions()

        if path == "/analytics":
            return self._analytics()

        # /affiliates/{id}
        if len(parts) == 2 and parts[0] == "affiliates":
            return self._affiliate_detail(parts[1])

        # /affiliates/{id}/links|conversions|payouts
        if len(parts) == 3 and parts[0] == "affiliates":
            sub = parts[2]
            if sub == "links":
                return self._affiliate_links(parts[1])
            if sub == "conversions":
                return self._affiliate_conversions(parts[1])
            if sub == "payouts":
                return self._affiliate_payouts(parts[1])

        # /r/{code} or /r/{code}/pixel
        if len(parts) >= 2 and parts[0] == "r":
            if len(parts) == 3 and parts[2] == "pixel":
                return self._impression_pixel(parts[1])
            if len(parts) == 2:
                return self._click_redirect(parts[1])

        _json_resp(self, 404, {"error": "not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        path   = parsed.path.rstrip("/")
        parts  = [p for p in path.split("/") if p]

        if path == "/affiliates/register":
            return self._register_affiliate()

        # /affiliates/{id}/links
        if len(parts) == 3 and parts[0] == "affiliates" and parts[2] == "links":
            return self._create_link(parts[1])

        if path == "/conversions":
            return self._record_conversion()

        # /conversions/{id}/approve
        if len(parts) == 3 and parts[0] == "conversions" and parts[2] == "approve":
            return self._approve_conversion(parts[1])

        if path == "/payouts/process":
            return self._trigger_payouts()

        _json_resp(self, 404, {"error": "not found"})

    # ---- body helper -------------------------------------------------------

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode())
        except (ValueError, UnicodeDecodeError):
            return {}

    # ---- GET /health -------------------------------------------------------

    def _health(self):
        try:
            with get_db() as conn:
                aff_count = conn.execute(
                    "SELECT COUNT(*) FROM affiliates WHERE status='active'"
                ).fetchone()[0]
                pending = conn.execute(
                    "SELECT COALESCE(SUM(commission_amount),0) FROM conversions WHERE status='pending'"
                ).fetchone()[0]
        except Exception as exc:  # noqa: BLE001
            _json_resp(self, 500, {"error": str(exc)})
            return
        _json_resp(self, 200, {
            "service":             "fm_referral_engine",
            "port":                PORT,
            "uptime_seconds":      round(time.time() - START_TIME, 1),
            "active_affiliates":   aff_count,
            "pending_commissions": round(pending, 4),
        })

    # ---- GET /affiliates ---------------------------------------------------

    def _list_affiliates(self):
        if not _require_admin(self):
            return
        try:
            with get_db() as conn:
                rows = conn.execute(
                    "SELECT * FROM affiliates ORDER BY created_at DESC"
                ).fetchall()
        except Exception as exc:  # noqa: BLE001
            _json_resp(self, 500, {"error": str(exc)})
            return
        _json_resp(self, 200, {"affiliates": [_row_to_dict(r) for r in rows]})

    # ---- GET /affiliates/{id} ----------------------------------------------

    def _affiliate_detail(self, affiliate_id: str):
        # Admin or self (must send X-Affiliate-Email + X-Affiliate-Id)
        is_admin = (
            ADMIN_SECRET
            and hmac.compare_digest(
                self.headers.get("X-Admin-Secret", ""), ADMIN_SECRET
            )
        )
        if not is_admin:
            hdr_id    = self.headers.get("X-Affiliate-Id", "")
            hdr_email = self.headers.get("X-Affiliate-Email", "")
            try:
                with get_db() as conn:
                    row = conn.execute(
                        "SELECT * FROM affiliates WHERE affiliate_id=?",
                        (affiliate_id,),
                    ).fetchone()
            except Exception as exc:  # noqa: BLE001
                _json_resp(self, 500, {"error": str(exc)})
                return
            if not row:
                _json_resp(self, 404, {"error": "affiliate not found"})
                return
            if not (hdr_id == affiliate_id and hdr_email == row["email"]):
                _json_resp(self, 401, {"error": "unauthorized"})
                return
            _json_resp(self, 200, {"affiliate": _row_to_dict(row)})
            return

        try:
            with get_db() as conn:
                row = conn.execute(
                    "SELECT * FROM affiliates WHERE affiliate_id=?",
                    (affiliate_id,),
                ).fetchone()
        except Exception as exc:  # noqa: BLE001
            _json_resp(self, 500, {"error": str(exc)})
            return
        if not row:
            _json_resp(self, 404, {"error": "affiliate not found"})
            return
        _json_resp(self, 200, {"affiliate": _row_to_dict(row)})

    # ---- GET /affiliates/{id}/links ----------------------------------------

    def _affiliate_links(self, affiliate_id: str):
        if not _require_admin(self):
            # Also allow the affiliate themselves
            hdr_id = self.headers.get("X-Affiliate-Id", "")
            if hdr_id != affiliate_id:
                return  # _require_admin already sent 401
        try:
            with get_db() as conn:
                rows = conn.execute(
                    "SELECT * FROM referral_links WHERE affiliate_id=? ORDER BY created_at DESC",
                    (affiliate_id,),
                ).fetchall()
        except Exception as exc:  # noqa: BLE001
            _json_resp(self, 500, {"error": str(exc)})
            return
        _json_resp(self, 200, {"links": [_row_to_dict(r) for r in rows]})

    # ---- GET /affiliates/{id}/conversions ----------------------------------

    def _affiliate_conversions(self, affiliate_id: str):
        is_admin = (
            ADMIN_SECRET
            and hmac.compare_digest(
                self.headers.get("X-Admin-Secret", ""), ADMIN_SECRET
            )
        )
        hdr_id = self.headers.get("X-Affiliate-Id", "")
        if not is_admin and hdr_id != affiliate_id:
            _json_resp(self, 401, {"error": "unauthorized"})
            return
        try:
            with get_db() as conn:
                rows = conn.execute(
                    "SELECT * FROM conversions WHERE affiliate_id=? ORDER BY created_at DESC",
                    (affiliate_id,),
                ).fetchall()
        except Exception as exc:  # noqa: BLE001
            _json_resp(self, 500, {"error": str(exc)})
            return
        _json_resp(self, 200, {"conversions": [_row_to_dict(r) for r in rows]})

    # ---- GET /affiliates/{id}/payouts --------------------------------------

    def _affiliate_payouts(self, affiliate_id: str):
        is_admin = (
            ADMIN_SECRET
            and hmac.compare_digest(
                self.headers.get("X-Admin-Secret", ""), ADMIN_SECRET
            )
        )
        hdr_id = self.headers.get("X-Affiliate-Id", "")
        if not is_admin and hdr_id != affiliate_id:
            _json_resp(self, 401, {"error": "unauthorized"})
            return
        try:
            with get_db() as conn:
                rows = conn.execute(
                    "SELECT * FROM payouts WHERE affiliate_id=? ORDER BY created_at DESC",
                    (affiliate_id,),
                ).fetchall()
        except Exception as exc:  # noqa: BLE001
            _json_resp(self, 500, {"error": str(exc)})
            return
        _json_resp(self, 200, {"payouts": [_row_to_dict(r) for r in rows]})

    # ---- GET /r/{code} — click tracking redirect ---------------------------

    def _click_redirect(self, link_code: str):
        try:
            with get_db() as conn:
                link = conn.execute(
                    "SELECT * FROM referral_links WHERE link_code=?",
                    (link_code,),
                ).fetchone()
        except Exception as exc:  # noqa: BLE001
            log.error("Click DB error: %s", exc)
            self.send_response(302)
            self.send_header("Location", "/")
            self.end_headers()
            return

        if not link:
            _json_resp(self, 404, {"error": "link not found"})
            return

        client_ip  = self.client_address[0]
        ip_hash    = _hash_ip(client_ip)
        user_agent = self.headers.get("User-Agent", "")
        referrer   = self.headers.get("Referer", "")
        now        = time.time()

        try:
            with get_db() as conn:
                conn.execute(
                    """
                    INSERT INTO clicks (link_code, ip_hash, user_agent, referrer, landed_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (link_code, ip_hash, user_agent, referrer, now),
                )
                conn.execute(
                    "UPDATE referral_links SET clicks=clicks+1 WHERE link_code=?",
                    (link_code,),
                )
                conn.execute(
                    "UPDATE affiliates SET total_clicks=total_clicks+1 WHERE affiliate_id=?",
                    (link["affiliate_id"],),
                )
        except Exception as exc:  # noqa: BLE001
            log.error("Click record error: %s", exc)

        dest = link["destination_url"] or "/"
        self.send_response(302)
        self.send_header("Location", dest)
        self.send_header("Cache-Control", "no-store, no-cache")
        self.end_headers()

    # ---- GET /r/{code}/pixel — impression pixel ----------------------------

    def _impression_pixel(self, link_code: str):
        try:
            with get_db() as conn:
                link = conn.execute(
                    "SELECT affiliate_id FROM referral_links WHERE link_code=?",
                    (link_code,),
                ).fetchone()
        except Exception:  # noqa: BLE001
            link = None

        if link:
            client_ip  = self.client_address[0]
            ip_hash    = _hash_ip(client_ip)
            user_agent = self.headers.get("User-Agent", "")
            referrer   = self.headers.get("Referer", "")
            now        = time.time()
            try:
                with get_db() as conn:
                    conn.execute(
                        """
                        INSERT INTO clicks (link_code, ip_hash, user_agent, referrer, landed_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (link_code, ip_hash, user_agent, referrer, now),
                    )
                    conn.execute(
                        "UPDATE referral_links SET clicks=clicks+1 WHERE link_code=?",
                        (link_code,),
                    )
                    conn.execute(
                        "UPDATE affiliates SET total_clicks=total_clicks+1 WHERE affiliate_id=?",
                        (link["affiliate_id"],),
                    )
            except Exception as exc:  # noqa: BLE001
                log.error("Pixel impression record error: %s", exc)

        self.send_response(200)
        self.send_header("Content-Type", "image/gif")
        self.send_header("Content-Length", str(len(_PIXEL_GIF)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.end_headers()
        self.wfile.write(_PIXEL_GIF)

    # ---- POST /affiliates/register -----------------------------------------

    def _register_affiliate(self):
        body = self._read_body()
        email   = (body.get("email") or "").strip().lower()
        name    = (body.get("name") or "").strip()
        company = (body.get("company") or "").strip()

        if not email:
            _json_resp(self, 400, {"error": "email is required"})
            return

        affiliate_id = _affiliate_id_from_email(email)
        link_code    = _generate_link_code()
        now          = time.time()

        try:
            with get_db() as conn:
                existing = conn.execute(
                    "SELECT affiliate_id FROM affiliates WHERE email=?",
                    (email,),
                ).fetchone()
                if existing:
                    _json_resp(self, 409, {"error": "email already registered"})
                    return

                conn.execute(
                    """
                    INSERT INTO affiliates
                        (affiliate_id, email, name, company, status, tier,
                         commission_rate, created_at)
                    VALUES (?, ?, ?, ?, 'active', 'standard', 0.20, ?)
                    """,
                    (affiliate_id, email, name, company, now),
                )
                conn.execute(
                    """
                    INSERT INTO referral_links
                        (link_code, affiliate_id, campaign, destination_url, created_at)
                    VALUES (?, ?, 'default', '', ?)
                    """,
                    (link_code, affiliate_id, now),
                )
        except sqlite3.IntegrityError as exc:
            _json_resp(self, 409, {"error": str(exc)})
            return
        except Exception as exc:  # noqa: BLE001
            _json_resp(self, 500, {"error": str(exc)})
            return

        threading.Thread(
            target=_send_welcome_email,
            args=(email, name, affiliate_id, link_code),
            daemon=True,
        ).start()

        log.info("New affiliate registered: %s (%s)", affiliate_id, email)
        _json_resp(self, 201, {
            "affiliate_id": affiliate_id,
            "email":        email,
            "link_code":    link_code,
            "tier":         "standard",
            "commission_rate": 0.20,
        })

    # ---- POST /affiliates/{id}/links ---------------------------------------

    def _create_link(self, affiliate_id: str):
        hdr_id = self.headers.get("X-Affiliate-Id", "")
        if hdr_id != affiliate_id:
            is_admin = (
                ADMIN_SECRET
                and hmac.compare_digest(
                    self.headers.get("X-Admin-Secret", ""), ADMIN_SECRET
                )
            )
            if not is_admin:
                _json_resp(self, 401, {"error": "unauthorized"})
                return

        body    = self._read_body()
        campaign = (body.get("campaign") or "default").strip()
        dest_url = (body.get("destination_url") or "").strip()

        try:
            with get_db() as conn:
                exists = conn.execute(
                    "SELECT id FROM affiliates WHERE affiliate_id=?",
                    (affiliate_id,),
                ).fetchone()
                if not exists:
                    _json_resp(self, 404, {"error": "affiliate not found"})
                    return

                link_code = _generate_link_code()
                now       = time.time()
                conn.execute(
                    """
                    INSERT INTO referral_links
                        (link_code, affiliate_id, campaign, destination_url, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (link_code, affiliate_id, campaign, dest_url, now),
                )
        except Exception as exc:  # noqa: BLE001
            _json_resp(self, 500, {"error": str(exc)})
            return

        log.info("New link %s for affiliate %s (campaign=%s)", link_code, affiliate_id, campaign)
        _json_resp(self, 201, {
            "link_code":      link_code,
            "affiliate_id":   affiliate_id,
            "campaign":       campaign,
            "destination_url": dest_url,
        })

    # ---- POST /conversions -------------------------------------------------

    def _record_conversion(self):
        # Requires ADMIN_SECRET or MCP_SECRET
        mcp_secret   = os.environ.get("MCP_SECRET", "")
        token        = self.headers.get("X-Admin-Secret", "") or self.headers.get("X-Mcp-Secret", "")
        valid_admin  = ADMIN_SECRET  and hmac.compare_digest(token, ADMIN_SECRET)
        valid_mcp    = mcp_secret    and hmac.compare_digest(token, mcp_secret)
        if not valid_admin and not valid_mcp:
            _json_resp(self, 401, {"error": "unauthorized"})
            return

        body            = self._read_body()
        link_code       = (body.get("link_code") or "").strip()
        conversion_type = (body.get("conversion_type") or "sale").strip()
        order_value     = float(body.get("order_value") or 0)
        order_ref       = (body.get("order_ref") or "").strip()

        if not link_code:
            _json_resp(self, 400, {"error": "link_code required"})
            return

        now = time.time()

        try:
            with get_db() as conn:
                link = conn.execute(
                    "SELECT * FROM referral_links WHERE link_code=?",
                    (link_code,),
                ).fetchone()
                if not link:
                    _json_resp(self, 404, {"error": "link not found"})
                    return

                aff = conn.execute(
                    "SELECT * FROM affiliates WHERE affiliate_id=?",
                    (link["affiliate_id"],),
                ).fetchone()
                if not aff:
                    _json_resp(self, 404, {"error": "affiliate not found"})
                    return

                # Find the most recent unconverted click for this link
                click = conn.execute(
                    """
                    SELECT id FROM clicks
                    WHERE link_code=? AND converted=0
                    ORDER BY landed_at DESC LIMIT 1
                    """,
                    (link_code,),
                ).fetchone()
                click_id = click["id"] if click else None

                rate   = aff["commission_rate"]
                amount = round(order_value * rate, 4)

                conn.execute(
                    """
                    INSERT INTO conversions
                        (click_id, link_code, affiliate_id, conversion_type,
                         order_value, commission_rate, commission_amount,
                         status, order_ref, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                    """,
                    (click_id, link_code, aff["affiliate_id"], conversion_type,
                     order_value, rate, amount, order_ref, now),
                )
                conv_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

                # Mark click as converted
                if click_id:
                    conn.execute(
                        "UPDATE clicks SET converted=1, converted_at=? WHERE id=?",
                        (now, click_id),
                    )

                conn.execute(
                    "UPDATE referral_links SET conversions=conversions+1 WHERE link_code=?",
                    (link_code,),
                )
                conn.execute(
                    """
                    UPDATE affiliates
                    SET total_conversions=total_conversions+1
                    WHERE affiliate_id=?
                    """,
                    (aff["affiliate_id"],),
                )
        except Exception as exc:  # noqa: BLE001
            _json_resp(self, 500, {"error": str(exc)})
            return

        log.info(
            "Conversion %s recorded: affiliate=%s amount=%.4f AUD",
            conv_id, aff["affiliate_id"], amount,
        )
        _json_resp(self, 201, {
            "conversion_id":    conv_id,
            "affiliate_id":     aff["affiliate_id"],
            "commission_rate":  rate,
            "commission_amount": amount,
            "status":           "pending",
        })

    # ---- POST /conversions/{id}/approve ------------------------------------

    def _approve_conversion(self, conv_id_str: str):
        if not _require_admin(self):
            return
        try:
            conv_id = int(conv_id_str)
        except ValueError:
            _json_resp(self, 400, {"error": "invalid conversion id"})
            return

        now = time.time()
        try:
            with get_db() as conn:
                row = conn.execute(
                    "SELECT * FROM conversions WHERE id=?", (conv_id,)
                ).fetchone()
                if not row:
                    _json_resp(self, 404, {"error": "conversion not found"})
                    return
                if row["status"] != "pending":
                    _json_resp(self, 409, {"error": f"conversion already {row['status']}"})
                    return

                conn.execute(
                    "UPDATE conversions SET status='approved', approved_at=? WHERE id=?",
                    (now, conv_id),
                )
                conn.execute(
                    """
                    UPDATE affiliates
                    SET total_earned=total_earned+?
                    WHERE affiliate_id=?
                    """,
                    (row["commission_amount"], row["affiliate_id"]),
                )
        except Exception as exc:  # noqa: BLE001
            _json_resp(self, 500, {"error": str(exc)})
            return

        log.info("Conversion %s approved; +%.4f AUD for %s", conv_id,
                 row["commission_amount"], row["affiliate_id"])
        _json_resp(self, 200, {"conversion_id": conv_id, "status": "approved"})

    # ---- GET /conversions --------------------------------------------------

    def _list_conversions(self):
        if not _require_admin(self):
            return
        parsed = urlparse(self.path)
        qs     = parse_qs(parsed.query)
        status = qs.get("status", [None])[0]
        try:
            with get_db() as conn:
                if status:
                    rows = conn.execute(
                        "SELECT * FROM conversions WHERE status=? ORDER BY created_at DESC",
                        (status,),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM conversions ORDER BY created_at DESC LIMIT 500"
                    ).fetchall()
        except Exception as exc:  # noqa: BLE001
            _json_resp(self, 500, {"error": str(exc)})
            return
        _json_resp(self, 200, {"conversions": [_row_to_dict(r) for r in rows]})

    # ---- POST /payouts/process ---------------------------------------------

    def _trigger_payouts(self):
        if not _require_admin(self):
            return
        threading.Thread(target=_process_payouts, daemon=True).start()
        _json_resp(self, 202, {"message": "payout processing initiated"})

    # ---- GET /analytics ----------------------------------------------------

    def _analytics(self):
        if not _require_admin(self):
            return
        try:
            with get_db() as conn:
                total_clicks = conn.execute(
                    "SELECT COALESCE(SUM(clicks),0) FROM referral_links"
                ).fetchone()[0]
                total_conversions = conn.execute(
                    "SELECT COUNT(*) FROM conversions"
                ).fetchone()[0]
                total_commission = conn.execute(
                    "SELECT COALESCE(SUM(commission_amount),0) FROM conversions WHERE status='approved'"
                ).fetchone()[0]
                total_paid = conn.execute(
                    "SELECT COALESCE(SUM(amount),0) FROM payouts WHERE status='paid'"
                ).fetchone()[0]
                pending_commission = conn.execute(
                    "SELECT COALESCE(SUM(commission_amount),0) FROM conversions WHERE status='pending'"
                ).fetchone()[0]
                affiliate_count = conn.execute(
                    "SELECT COUNT(*) FROM affiliates WHERE status='active'"
                ).fetchone()[0]

                by_tier = conn.execute(
                    """
                    SELECT tier,
                           COUNT(*) AS count,
                           COALESCE(SUM(total_earned),0) AS earned,
                           COALESCE(SUM(total_paid),0) AS paid
                    FROM affiliates
                    GROUP BY tier
                    """
                ).fetchall()

                top_affiliates = conn.execute(
                    """
                    SELECT affiliate_id, name, tier, total_clicks,
                           total_conversions, total_earned
                    FROM affiliates
                    ORDER BY total_earned DESC
                    LIMIT 10
                    """
                ).fetchall()
        except Exception as exc:  # noqa: BLE001
            _json_resp(self, 500, {"error": str(exc)})
            return

        cvr = round(total_conversions / total_clicks * 100, 2) if total_clicks else 0.0

        _json_resp(self, 200, {
            "total_clicks":         total_clicks,
            "total_conversions":    total_conversions,
            "conversion_rate_pct":  cvr,
            "total_commission_aud": round(total_commission, 4),
            "total_paid_aud":       round(total_paid, 4),
            "pending_commission_aud": round(pending_commission, 4),
            "active_affiliates":    affiliate_count,
            "by_tier":              [_row_to_dict(r) for r in by_tier],
            "top_affiliates":       [_row_to_dict(r) for r in top_affiliates],
        })


# ---------------------------------------------------------------------------
# Server bootstrap
# ---------------------------------------------------------------------------

def run_server() -> None:
    init_db()

    payout_thread = threading.Thread(target=_payout_loop, daemon=True, name="payout-loop")
    payout_thread.start()
    log.info("Payout background thread started (interval=%ds)", PAYOUT_INTERVAL_S)

    server = HTTPServer(("0.0.0.0", PORT), ReferralHandler)
    log.info("FractalMesh Referral Engine listening on port %d", PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down referral engine.")
        server.server_close()


if __name__ == "__main__":
    run_server()
