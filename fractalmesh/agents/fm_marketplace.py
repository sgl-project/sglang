#!/usr/bin/env python3
"""
FractalMesh OMEGA Titan — Digital Goods Marketplace
Port: 7882

Gumroad/Envato-style marketplace where sellers list digital products
(templates, code, datasets, tools) and buyers purchase them with the
platform taking a configurable commission.

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
AGENT_NAME = "fm_marketplace"
PORT = int(os.environ.get("MARKETPLACE_PORT", "7882"))

STRIPE_SECRET_KEY  = os.environ.get("STRIPE_SECRET_KEY", "")
SENDGRID_API_KEY   = os.environ.get("SENDGRID_API_KEY", "")
SENDGRID_FROM_EMAIL = os.environ.get("SENDGRID_FROM_EMAIL", "noreply@fractalmesh.io")
ADMIN_SECRET       = os.environ.get("ADMIN_SECRET", "")

STRIPE_API_BASE   = "https://api.stripe.com/v1"
SENDGRID_API_BASE = "https://api.sendgrid.com/v3"

ROOT    = Path(os.environ.get("FRACTALMESH_HOME", Path.home() / "fmsaas"))
DB_PATH = ROOT / "database" / "sovereign.db"
LOG_PATH = ROOT / "logs" / f"{AGENT_NAME}.log"

START_TIME = time.time()

SEED_CATEGORIES = [
    ("Templates",          "templates",          "Document, web, and project templates"),
    ("Code & Scripts",     "code-scripts",       "Source code, scripts, and snippets"),
    ("Datasets",           "datasets",           "Curated data files and databases"),
    ("Design Assets",      "design-assets",      "Graphics, icons, UI kits, and fonts"),
    ("Courses",            "courses",            "Educational content and tutorials"),
    ("Tools & Utilities",  "tools-utilities",    "Software tools and utility programs"),
    ("APIs & Integrations","apis-integrations",  "API wrappers and integration packages"),
]

PAYOUT_INTERVAL = 86400  # 24 hours

# ---------------------------------------------------------------------------
# Logging — minimal stdlib
# ---------------------------------------------------------------------------
os.makedirs(str(LOG_PATH.parent), exist_ok=True)

def _log(level: str, msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    line = f"{ts} [{AGENT_NAME}] {level} {msg}"
    print(line, flush=True)
    try:
        with open(str(LOG_PATH), "a") as fh:
            fh.write(line + "\n")
    except OSError:
        pass

def log_info(msg: str)  -> None: _log("INFO ", msg)
def log_warn(msg: str)  -> None: _log("WARN ", msg)
def log_error(msg: str) -> None: _log("ERROR", msg)

# ---------------------------------------------------------------------------
# Database
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
        CREATE TABLE IF NOT EXISTS sellers (
            id               INTEGER PRIMARY KEY,
            seller_id        TEXT UNIQUE NOT NULL,
            email            TEXT UNIQUE NOT NULL,
            name             TEXT NOT NULL DEFAULT '',
            bio              TEXT NOT NULL DEFAULT '',
            website          TEXT NOT NULL DEFAULT '',
            stripe_account_id TEXT NOT NULL DEFAULT '',
            commission_rate  REAL NOT NULL DEFAULT 0.15,
            status           TEXT NOT NULL DEFAULT 'active',
            total_sales      REAL NOT NULL DEFAULT 0,
            created_at       REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS listings (
            id          INTEGER PRIMARY KEY,
            listing_id  TEXT UNIQUE NOT NULL,
            seller_id   TEXT NOT NULL,
            title       TEXT NOT NULL DEFAULT '',
            description TEXT NOT NULL DEFAULT '',
            category    TEXT NOT NULL DEFAULT '',
            tags        TEXT NOT NULL DEFAULT '',
            price       REAL NOT NULL DEFAULT 0,
            currency    TEXT NOT NULL DEFAULT 'AUD',
            file_key    TEXT NOT NULL DEFAULT '',
            file_name   TEXT NOT NULL DEFAULT '',
            file_size   INTEGER NOT NULL DEFAULT 0,
            preview_url TEXT NOT NULL DEFAULT '',
            status      TEXT NOT NULL DEFAULT 'draft',
            sales_count INTEGER NOT NULL DEFAULT 0,
            rating      REAL NOT NULL DEFAULT 0,
            created_at  REAL NOT NULL,
            updated_at  REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS purchases (
            id                INTEGER PRIMARY KEY,
            purchase_id       TEXT UNIQUE NOT NULL,
            listing_id        TEXT NOT NULL,
            buyer_email       TEXT NOT NULL,
            buyer_name        TEXT NOT NULL DEFAULT '',
            seller_id         TEXT NOT NULL,
            amount            REAL NOT NULL DEFAULT 0,
            commission        REAL NOT NULL DEFAULT 0,
            seller_payout     REAL NOT NULL DEFAULT 0,
            stripe_payment_id TEXT NOT NULL DEFAULT '',
            download_token    TEXT NOT NULL DEFAULT '',
            download_count    INTEGER NOT NULL DEFAULT 0,
            max_downloads     INTEGER NOT NULL DEFAULT 5,
            status            TEXT NOT NULL DEFAULT 'pending',
            created_at        REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS reviews (
            id          INTEGER PRIMARY KEY,
            listing_id  TEXT NOT NULL,
            buyer_email TEXT NOT NULL,
            rating      INTEGER NOT NULL DEFAULT 0,
            comment     TEXT NOT NULL DEFAULT '',
            created_at  REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS categories (
            id            INTEGER PRIMARY KEY,
            name          TEXT UNIQUE NOT NULL,
            slug          TEXT UNIQUE NOT NULL,
            description   TEXT NOT NULL DEFAULT '',
            listing_count INTEGER NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_listings_seller   ON listings(seller_id);
        CREATE INDEX IF NOT EXISTS idx_listings_category ON listings(category);
        CREATE INDEX IF NOT EXISTS idx_listings_status   ON listings(status);
        CREATE INDEX IF NOT EXISTS idx_purchases_listing ON purchases(listing_id);
        CREATE INDEX IF NOT EXISTS idx_purchases_buyer   ON purchases(buyer_email);
        CREATE INDEX IF NOT EXISTS idx_purchases_seller  ON purchases(seller_id);
        CREATE INDEX IF NOT EXISTS idx_purchases_token   ON purchases(download_token);
        CREATE INDEX IF NOT EXISTS idx_reviews_listing   ON reviews(listing_id);
    """)

    # Seed categories if empty
    cur = con.execute("SELECT COUNT(*) FROM categories")
    if cur.fetchone()[0] == 0:
        for name, slug, desc in SEED_CATEGORIES:
            con.execute(
                "INSERT OR IGNORE INTO categories(name, slug, description) VALUES (?,?,?)",
                (name, slug, desc),
            )
        log_info("Seeded default categories")

    con.commit()
    con.close()
    log_info(f"DB initialised at {DB_PATH}")

# ---------------------------------------------------------------------------
# Stripe helpers
# ---------------------------------------------------------------------------

def _stripe_auth() -> str:
    creds = base64.b64encode(f"{STRIPE_SECRET_KEY}:".encode()).decode()
    return f"Basic {creds}"


def _flatten_form(d: dict, prefix: str = "") -> dict:
    """Flatten nested dict into Stripe form-encoding format."""
    out = {}
    for k, v in d.items():
        full_key = f"{prefix}[{k}]" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_form(v, full_key))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    out.update(_flatten_form(item, f"{full_key}[{i}]"))
                else:
                    out[f"{full_key}[{i}]"] = str(item)
        elif v is not None:
            out[full_key] = str(v)
    return out


def _stripe_post(path: str, data: dict) -> dict:
    url = STRIPE_API_BASE + path
    encoded = urllib.parse.urlencode(_flatten_form(data)).encode()
    req = urllib.request.Request(
        url, data=encoded,
        headers={
            "Authorization": _stripe_auth(),
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"Stripe POST {path} → {exc.code}: {body}") from exc


def _stripe_get(path: str, params: dict | None = None) -> dict:
    url = STRIPE_API_BASE + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url,
        headers={"Authorization": _stripe_auth(), "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"Stripe GET {path} → {exc.code}: {body}") from exc


def stripe_create_payment_intent(amount_aud: float, currency: str,
                                  metadata: dict) -> dict:
    """Create a PaymentIntent and return {id, client_secret}."""
    amount_cents = int(round(amount_aud * 100))
    result = _stripe_post("/payment_intents", {
        "amount": amount_cents,
        "currency": currency.lower(),
        "metadata": metadata,
        "automatic_payment_methods": {"enabled": "true"},
    })
    return {"id": result["id"], "client_secret": result["client_secret"]}


def stripe_transfer_to_seller(amount_aud: float, currency: str,
                               stripe_account_id: str, metadata: dict) -> dict:
    """Transfer funds to a connected seller account."""
    amount_cents = int(round(amount_aud * 100))
    result = _stripe_post("/transfers", {
        "amount": amount_cents,
        "currency": currency.lower(),
        "destination": stripe_account_id,
        "metadata": metadata,
    })
    return result

# ---------------------------------------------------------------------------
# SendGrid email helper
# ---------------------------------------------------------------------------

def _send_email(to_email: str, subject: str, body_html: str) -> bool:
    if not SENDGRID_API_KEY:
        log_warn("SENDGRID_API_KEY not set — skipping email")
        return False
    payload = json.dumps({
        "personalizations": [{"to": [{"email": to_email}]}],
        "from": {"email": SENDGRID_FROM_EMAIL},
        "subject": subject,
        "content": [{"type": "text/html", "value": body_html}],
    }).encode()
    req = urllib.request.Request(
        f"{SENDGRID_API_BASE}/mail/send",
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
        log_error(f"SendGrid error {exc.code}: {exc.read().decode(errors='replace')}")
        return False
    except Exception as exc:
        log_error(f"SendGrid exception: {exc}")
        return False


def send_welcome_email(to_email: str, name: str, seller_id: str) -> None:
    _send_email(
        to_email,
        "Welcome to FractalMesh Marketplace!",
        f"""<h2>Welcome, {name}!</h2>
<p>Your seller account has been created. Your Seller ID is: <strong>{seller_id}</strong></p>
<p>You can now create listings and start selling digital goods on FractalMesh Marketplace.</p>
<p>Include your Seller ID as the <code>X-Seller-Id</code> header when calling the API.</p>""",
    )


def send_download_link_email(to_email: str, buyer_name: str, listing_title: str,
                              download_token: str, purchase_id: str) -> None:
    _send_email(
        to_email,
        f"Your purchase: {listing_title}",
        f"""<h2>Thank you, {buyer_name}!</h2>
<p>Your purchase of <strong>{listing_title}</strong> (ID: {purchase_id}) is confirmed.</p>
<p>Download your file using the token below (valid for 5 downloads):</p>
<p><strong>Download token:</strong> {download_token}</p>
<p>API endpoint: <code>GET /download/{download_token}</code></p>""",
    )


def send_payout_notification(to_email: str, seller_name: str,
                              amount: float, currency: str,
                              transfer_id: str) -> None:
    _send_email(
        to_email,
        "FractalMesh Marketplace — Payout Processed",
        f"""<h2>Payout Sent, {seller_name}!</h2>
<p>A payout of <strong>{currency} {amount:.2f}</strong> has been transferred to your Stripe account.</p>
<p>Transfer ID: <code>{transfer_id}</code></p>""",
    )

# ---------------------------------------------------------------------------
# Background payout thread
# ---------------------------------------------------------------------------

def _run_payouts() -> None:
    """Run once per PAYOUT_INTERVAL: batch completed purchases → Stripe transfers."""
    while True:
        try:
            _process_payouts()
        except Exception as exc:
            log_error(f"Payout thread error: {exc}")
        time.sleep(PAYOUT_INTERVAL)


def _process_payouts() -> None:
    log_info("Running payout sweep")
    con = _db()
    try:
        # Find completed purchases with unpaid seller payouts
        rows = con.execute("""
            SELECT p.purchase_id, p.seller_id, p.seller_payout, p.currency,
                   p.listing_id, p.stripe_payment_id,
                   s.stripe_account_id, s.email AS seller_email, s.name AS seller_name,
                   l.currency AS listing_currency
            FROM purchases p
            JOIN sellers s ON s.seller_id = p.seller_id
            JOIN listings l ON l.listing_id = p.listing_id
            WHERE p.status = 'completed'
              AND p.stripe_payment_id != ''
              AND s.stripe_account_id != ''
        """).fetchall()

        # Group by seller
        by_seller: dict[str, list] = {}
        for row in rows:
            sid = row["seller_id"]
            by_seller.setdefault(sid, []).append(dict(row))

        for seller_id, purchases in by_seller.items():
            total_payout = sum(p["seller_payout"] for p in purchases)
            if total_payout < 0.50:
                continue  # Below minimum transfer threshold
            seller_info = purchases[0]
            currency = seller_info.get("listing_currency", "aud")
            try:
                transfer = stripe_transfer_to_seller(
                    total_payout,
                    currency,
                    seller_info["stripe_account_id"],
                    {
                        "seller_id": seller_id,
                        "purchase_count": len(purchases),
                        "platform": "fractalmesh_marketplace",
                    },
                )
                transfer_id = transfer.get("id", "unknown")
                # Mark those purchases as payout-sent by storing transfer id
                for p in purchases:
                    con.execute(
                        "UPDATE purchases SET status='paid_out' WHERE purchase_id=?",
                        (p["purchase_id"],),
                    )
                con.commit()
                send_payout_notification(
                    seller_info["seller_email"],
                    seller_info["seller_name"],
                    total_payout,
                    currency.upper(),
                    transfer_id,
                )
                log_info(f"Payout {transfer_id} → seller {seller_id}: {currency.upper()} {total_payout:.2f}")
            except Exception as exc:
                log_error(f"Payout failed for seller {seller_id}: {exc}")
    finally:
        con.close()

# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def _check_admin(headers: dict) -> bool:
    if not ADMIN_SECRET:
        return False
    auth = headers.get("X-Admin-Secret", "") or headers.get("x-admin-secret", "")
    return hmac.compare_digest(auth, ADMIN_SECRET)


def _get_seller_id(headers: dict) -> str:
    return (headers.get("X-Seller-Id", "") or headers.get("x-seller-id", "")).strip()

# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

def _parse_qs(qs: str) -> dict:
    if not qs:
        return {}
    result = {}
    for part in qs.split("&"):
        if "=" in part:
            k, v = part.split("=", 1)
            result[urllib.parse.unquote_plus(k)] = urllib.parse.unquote_plus(v)
    return result


class MarketplaceHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # suppress default access log
        pass

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _ok(self, payload: dict) -> None:
        self._send(200, payload)

    def _err(self, code: int, msg: str) -> None:
        self._send(code, {"error": msg})

    # ------------------------------------------------------------------ GET
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path   = parsed.path.rstrip("/")
        qs     = _parse_qs(parsed.query)

        if path == "/health":
            return self._handle_health()
        if path == "/categories":
            return self._handle_list_categories()
        if path == "/listings":
            return self._handle_browse_listings(qs)
        if path == "/listings/search":
            return self._handle_search_listings(qs)
        if path.startswith("/listings/"):
            parts = path.split("/")
            if len(parts) == 3:
                return self._handle_listing_detail(parts[2])
        if path.startswith("/sellers/") and path.endswith("/analytics"):
            parts = path.split("/")
            if len(parts) == 4 and parts[3] == "analytics":
                return self._handle_seller_analytics(parts[2])
        if path.startswith("/sellers/"):
            parts = path.split("/")
            if len(parts) == 3:
                return self._handle_seller_profile(parts[2])
        if path.startswith("/download/"):
            parts = path.split("/")
            if len(parts) == 3:
                return self._handle_download(parts[2])
        if path == "/analytics":
            return self._handle_platform_analytics()

        self._err(404, "not found")

    # ----------------------------------------------------------------- POST
    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path   = parsed.path.rstrip("/")

        if path == "/sellers/register":
            return self._handle_register_seller()
        if path == "/listings":
            return self._handle_create_listing()
        if path.startswith("/listings/") and path.endswith("/publish"):
            parts = path.split("/")
            if len(parts) == 4 and parts[3] == "publish":
                return self._handle_publish_listing(parts[2])
        if path.startswith("/listings/") and path.endswith("/reviews"):
            parts = path.split("/")
            if len(parts) == 4 and parts[3] == "reviews":
                return self._handle_add_review(parts[2])
        if path == "/purchase":
            return self._handle_purchase()
        if path == "/purchase/free":
            return self._handle_free_purchase()

        self._err(404, "not found")

    # ----------------------------------------------------------------
    # GET handlers
    # ----------------------------------------------------------------

    def _handle_health(self):
        con = _db()
        try:
            listing_count  = con.execute("SELECT COUNT(*) FROM listings WHERE status='active'").fetchone()[0]
            seller_count   = con.execute("SELECT COUNT(*) FROM sellers WHERE status='active'").fetchone()[0]
            purchase_count = con.execute("SELECT COUNT(*) FROM purchases").fetchone()[0]
        finally:
            con.close()
        self._ok({
            "status": "ok",
            "service": AGENT_NAME,
            "port": PORT,
            "uptime": round(time.time() - START_TIME, 1),
            "listings": listing_count,
            "sellers": seller_count,
            "purchases": purchase_count,
        })

    def _handle_list_categories(self):
        con = _db()
        try:
            rows = con.execute(
                "SELECT name, slug, description, listing_count FROM categories ORDER BY name"
            ).fetchall()
        finally:
            con.close()
        self._ok({"categories": [dict(r) for r in rows]})

    def _handle_browse_listings(self, qs: dict):
        category  = qs.get("category", "")
        tags      = qs.get("tags", "")
        min_price = qs.get("min_price", "")
        max_price = qs.get("max_price", "")
        sort      = qs.get("sort", "newest")
        limit     = min(int(qs.get("limit", "50")), 200)

        where_clauses = ["l.status = 'active'"]
        params: list = []

        if category:
            where_clauses.append("l.category = ?")
            params.append(category)
        if tags:
            where_clauses.append("l.tags LIKE ?")
            params.append(f"%{tags}%")
        if min_price:
            try:
                where_clauses.append("l.price >= ?")
                params.append(float(min_price))
            except ValueError:
                pass
        if max_price:
            try:
                where_clauses.append("l.price <= ?")
                params.append(float(max_price))
            except ValueError:
                pass

        order = {
            "newest":    "l.created_at DESC",
            "price_asc": "l.price ASC",
            "rating":    "l.rating DESC",
        }.get(sort, "l.created_at DESC")

        where_sql = " AND ".join(where_clauses)
        sql = f"""
            SELECT l.listing_id, l.title, l.description, l.category, l.tags,
                   l.price, l.currency, l.preview_url, l.sales_count, l.rating,
                   l.created_at, s.name AS seller_name, s.seller_id
            FROM listings l
            JOIN sellers s ON s.seller_id = l.seller_id
            WHERE {where_sql}
            ORDER BY {order}
            LIMIT ?
        """
        params.append(limit)
        con = _db()
        try:
            rows = con.execute(sql, params).fetchall()
        finally:
            con.close()
        self._ok({"listings": [dict(r) for r in rows], "count": len(rows)})

    def _handle_search_listings(self, qs: dict):
        q = qs.get("q", "").strip()
        if not q:
            self._err(400, "q parameter required")
            return
        limit = min(int(qs.get("limit", "50")), 200)
        pattern = f"%{q}%"
        sql = """
            SELECT l.listing_id, l.title, l.description, l.category, l.tags,
                   l.price, l.currency, l.preview_url, l.sales_count, l.rating,
                   l.created_at, s.name AS seller_name, s.seller_id
            FROM listings l
            JOIN sellers s ON s.seller_id = l.seller_id
            WHERE l.status = 'active'
              AND (l.title LIKE ? OR l.description LIKE ? OR l.tags LIKE ?)
            ORDER BY l.rating DESC, l.sales_count DESC
            LIMIT ?
        """
        con = _db()
        try:
            rows = con.execute(sql, (pattern, pattern, pattern, limit)).fetchall()
        finally:
            con.close()
        self._ok({"listings": [dict(r) for r in rows], "count": len(rows), "query": q})

    def _handle_listing_detail(self, listing_id: str):
        con = _db()
        try:
            row = con.execute("""
                SELECT l.*, s.name AS seller_name, s.bio AS seller_bio,
                       s.website AS seller_website, s.total_sales AS seller_total_sales
                FROM listings l
                JOIN sellers s ON s.seller_id = l.seller_id
                WHERE l.listing_id = ?
            """, (listing_id,)).fetchone()
            if not row:
                self._err(404, "listing not found")
                return
            listing = dict(row)
            reviews = con.execute(
                "SELECT buyer_email, rating, comment, created_at FROM reviews WHERE listing_id=? ORDER BY created_at DESC LIMIT 20",
                (listing_id,),
            ).fetchall()
            listing["reviews"] = [dict(r) for r in reviews]
        finally:
            con.close()
        self._ok(listing)

    def _handle_seller_profile(self, seller_id: str):
        con = _db()
        try:
            row = con.execute(
                "SELECT seller_id, email, name, bio, website, total_sales, created_at FROM sellers WHERE seller_id=? AND status='active'",
                (seller_id,),
            ).fetchone()
            if not row:
                self._err(404, "seller not found")
                return
            profile = dict(row)
            listings = con.execute(
                "SELECT listing_id, title, category, price, currency, sales_count, rating, created_at FROM listings WHERE seller_id=? AND status='active' ORDER BY created_at DESC",
                (seller_id,),
            ).fetchall()
            profile["listings"] = [dict(r) for r in listings]
        finally:
            con.close()
        self._ok(profile)

    def _handle_seller_analytics(self, seller_id: str):
        # Require seller auth or admin
        req_seller = _get_seller_id(dict(self.headers))
        is_admin   = _check_admin(dict(self.headers))
        if not is_admin and req_seller != seller_id:
            self._err(403, "forbidden")
            return
        con = _db()
        try:
            seller = con.execute(
                "SELECT * FROM sellers WHERE seller_id=?", (seller_id,)
            ).fetchone()
            if not seller:
                self._err(404, "seller not found")
                return
            stats = con.execute("""
                SELECT COUNT(*) AS total_purchases,
                       SUM(amount) AS gmv,
                       SUM(seller_payout) AS total_payout,
                       SUM(commission) AS total_commission
                FROM purchases
                WHERE seller_id=? AND status IN ('completed','paid_out')
            """, (seller_id,)).fetchone()
            top_listings = con.execute("""
                SELECT l.listing_id, l.title, l.price, l.sales_count, l.rating,
                       COUNT(p.id) AS purchase_count,
                       SUM(p.seller_payout) AS revenue
                FROM listings l
                LEFT JOIN purchases p ON p.listing_id = l.listing_id
                    AND p.status IN ('completed','paid_out')
                WHERE l.seller_id=?
                GROUP BY l.listing_id
                ORDER BY revenue DESC
                LIMIT 10
            """, (seller_id,)).fetchall()
        finally:
            con.close()
        self._ok({
            "seller_id": seller_id,
            "name": seller["name"],
            "stats": dict(stats) if stats else {},
            "top_listings": [dict(r) for r in top_listings],
        })

    def _handle_platform_analytics(self):
        if not _check_admin(dict(self.headers)):
            self._err(403, "admin access required")
            return
        con = _db()
        try:
            totals = con.execute("""
                SELECT COUNT(*) AS total_purchases,
                       SUM(amount) AS gmv,
                       SUM(commission) AS platform_revenue,
                       SUM(seller_payout) AS seller_payouts
                FROM purchases WHERE status IN ('completed','paid_out')
            """).fetchone()
            top_listings = con.execute("""
                SELECT l.listing_id, l.title, l.category, l.price,
                       l.sales_count, l.rating
                FROM listings WHERE status='active'
                ORDER BY sales_count DESC LIMIT 10
            """).fetchall()
            leaderboard = con.execute("""
                SELECT s.seller_id, s.name, s.total_sales,
                       COUNT(DISTINCT l.listing_id) AS listing_count
                FROM sellers s
                LEFT JOIN listings l ON l.seller_id = s.seller_id AND l.status='active'
                WHERE s.status='active'
                GROUP BY s.seller_id
                ORDER BY s.total_sales DESC LIMIT 20
            """).fetchall()
            listing_count  = con.execute("SELECT COUNT(*) FROM listings WHERE status='active'").fetchone()[0]
            seller_count   = con.execute("SELECT COUNT(*) FROM sellers WHERE status='active'").fetchone()[0]
        finally:
            con.close()
        self._ok({
            "totals": dict(totals) if totals else {},
            "active_listings": listing_count,
            "active_sellers": seller_count,
            "top_listings": [dict(r) for r in top_listings],
            "seller_leaderboard": [dict(r) for r in leaderboard],
        })

    def _handle_download(self, download_token: str):
        con = _db()
        try:
            row = con.execute(
                "SELECT * FROM purchases WHERE download_token=?",
                (download_token,),
            ).fetchone()
            if not row:
                self._err(404, "invalid download token")
                return
            if row["download_count"] >= row["max_downloads"]:
                self._err(410, "download limit exhausted")
                return
            # Fetch listing file info
            listing = con.execute(
                "SELECT file_key, file_name FROM listings WHERE listing_id=?",
                (row["listing_id"],),
            ).fetchone()
            if not listing:
                self._err(404, "listing not found")
                return
            # Decrement remaining downloads
            new_count = row["download_count"] + 1
            con.execute(
                "UPDATE purchases SET download_count=? WHERE purchase_id=?",
                (new_count, row["purchase_id"]),
            )
            # Mark completed on first successful download if still pending
            if row["status"] == "pending":
                con.execute(
                    "UPDATE purchases SET status='completed' WHERE purchase_id=?",
                    (row["purchase_id"],),
                )
                # Update seller total_sales
                con.execute(
                    "UPDATE sellers SET total_sales = total_sales + ? WHERE seller_id=?",
                    (row["seller_payout"], row["seller_id"]),
                )
                # Increment listing sales count
                con.execute(
                    "UPDATE listings SET sales_count = sales_count + 1, updated_at=? WHERE listing_id=?",
                    (time.time(), row["listing_id"]),
                )
                # Update category listing count
                cat_row = con.execute(
                    "SELECT category FROM listings WHERE listing_id=?",
                    (row["listing_id"],),
                ).fetchone()
                if cat_row:
                    con.execute(
                        "UPDATE categories SET listing_count = listing_count WHERE name=?",
                        (cat_row["category"],),
                    )
            con.commit()
        finally:
            con.close()
        remaining = row["max_downloads"] - new_count
        self._ok({
            "purchase_id": row["purchase_id"],
            "file_key": listing["file_key"],
            "file_name": listing["file_name"],
            "downloads_remaining": remaining,
        })

    # ----------------------------------------------------------------
    # POST handlers
    # ----------------------------------------------------------------

    def _handle_register_seller(self):
        body = self._body()
        email   = body.get("email", "").strip().lower()
        name    = body.get("name", "").strip()
        bio     = body.get("bio", "")
        website = body.get("website", "")

        if not email or not name:
            self._err(400, "email and name required")
            return

        seller_id = "sel_" + secrets.token_urlsafe(16)
        now = time.time()
        con = _db()
        try:
            try:
                con.execute(
                    "INSERT INTO sellers(seller_id, email, name, bio, website, created_at) VALUES (?,?,?,?,?,?)",
                    (seller_id, email, name, bio, website, now),
                )
                con.commit()
            except sqlite3.IntegrityError:
                self._err(409, "email already registered")
                return
        finally:
            con.close()

        # Send welcome email in background to not block response
        threading.Thread(
            target=send_welcome_email, args=(email, name, seller_id), daemon=True
        ).start()

        log_info(f"Registered seller {seller_id} ({email})")
        self._send(201, {
            "seller_id": seller_id,
            "email": email,
            "name": name,
            "message": "Seller account created. Check your email for details.",
        })

    def _handle_create_listing(self):
        seller_id = _get_seller_id(dict(self.headers))
        is_admin  = _check_admin(dict(self.headers))

        if not seller_id and not is_admin:
            self._err(401, "X-Seller-Id header required")
            return

        body        = self._body()
        title       = body.get("title", "").strip()
        description = body.get("description", "").strip()
        category    = body.get("category", "").strip()
        tags        = body.get("tags", "")
        price       = body.get("price")
        file_key    = body.get("file_key", "")
        file_name   = body.get("file_name", "")
        file_size   = body.get("file_size", 0)
        preview_url = body.get("preview_url", "")
        currency    = body.get("currency", "AUD").upper()

        if not title or not description or not category or price is None:
            self._err(400, "title, description, category, and price required")
            return

        try:
            price = float(price)
        except (TypeError, ValueError):
            self._err(400, "price must be a number")
            return

        # If admin is creating, they must supply seller_id in body
        if is_admin and not seller_id:
            seller_id = body.get("seller_id", "")
        if not seller_id:
            self._err(400, "seller_id required")
            return

        con = _db()
        try:
            # Validate seller exists
            row = con.execute(
                "SELECT seller_id FROM sellers WHERE seller_id=? AND status='active'",
                (seller_id,),
            ).fetchone()
            if not row:
                self._err(404, "seller not found or inactive")
                return

            listing_id = "lst_" + secrets.token_urlsafe(16)
            now = time.time()
            con.execute(
                """INSERT INTO listings(listing_id, seller_id, title, description, category,
                   tags, price, currency, file_key, file_name, file_size, preview_url,
                   created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (listing_id, seller_id, title, description, category, tags, price,
                 currency, file_key, file_name, int(file_size), preview_url, now, now),
            )
            con.commit()
        finally:
            con.close()

        log_info(f"Created listing {listing_id} by seller {seller_id}")
        self._send(201, {
            "listing_id": listing_id,
            "status": "draft",
            "message": "Listing created in draft. Use /listings/{id}/publish to activate.",
        })

    def _handle_publish_listing(self, listing_id: str):
        seller_id = _get_seller_id(dict(self.headers))
        is_admin  = _check_admin(dict(self.headers))

        if not seller_id and not is_admin:
            self._err(401, "seller or admin authentication required")
            return

        con = _db()
        try:
            row = con.execute(
                "SELECT seller_id, status FROM listings WHERE listing_id=?",
                (listing_id,),
            ).fetchone()
            if not row:
                self._err(404, "listing not found")
                return
            if not is_admin and row["seller_id"] != seller_id:
                self._err(403, "forbidden: not your listing")
                return

            now = time.time()
            con.execute(
                "UPDATE listings SET status='active', updated_at=? WHERE listing_id=?",
                (now, listing_id),
            )
            # Update category count
            cat_row = con.execute(
                "SELECT category FROM listings WHERE listing_id=?", (listing_id,)
            ).fetchone()
            if cat_row:
                con.execute(
                    "UPDATE categories SET listing_count = listing_count + 1 WHERE name=?",
                    (cat_row["category"],),
                )
            con.commit()
        finally:
            con.close()

        log_info(f"Published listing {listing_id}")
        self._ok({"listing_id": listing_id, "status": "active"})

    def _handle_purchase(self):
        body               = self._body()
        listing_id         = body.get("listing_id", "").strip()
        buyer_email        = body.get("buyer_email", "").strip().lower()
        buyer_name         = body.get("buyer_name", "").strip()
        payment_method_id  = body.get("payment_method_id", "")

        if not listing_id or not buyer_email or not buyer_name:
            self._err(400, "listing_id, buyer_email, buyer_name required")
            return

        con = _db()
        try:
            listing = con.execute(
                "SELECT * FROM listings WHERE listing_id=? AND status='active'",
                (listing_id,),
            ).fetchone()
            if not listing:
                self._err(404, "listing not found or not active")
                return

            if listing["price"] <= 0:
                self._err(400, "use /purchase/free for free listings")
                return

            seller = con.execute(
                "SELECT * FROM sellers WHERE seller_id=? AND status='active'",
                (listing["seller_id"],),
            ).fetchone()
            if not seller:
                self._err(503, "seller account unavailable")
                return

            amount          = listing["price"]
            commission_rate = seller["commission_rate"]
            commission      = round(amount * commission_rate, 2)
            seller_payout   = round(amount - commission, 2)
            purchase_id     = "pur_" + secrets.token_urlsafe(16)
            download_token  = secrets.token_urlsafe(32)
            now             = time.time()

            # Create Stripe PaymentIntent
            try:
                intent = stripe_create_payment_intent(
                    amount,
                    listing["currency"],
                    {
                        "purchase_id": purchase_id,
                        "listing_id": listing_id,
                        "buyer_email": buyer_email,
                        "platform": "fractalmesh_marketplace",
                    },
                )
                stripe_payment_id = intent["id"]
                client_secret     = intent["client_secret"]
            except RuntimeError as exc:
                log_error(f"Stripe PaymentIntent failed: {exc}")
                self._err(502, "payment processing unavailable")
                return

            con.execute(
                """INSERT INTO purchases(purchase_id, listing_id, buyer_email, buyer_name,
                   seller_id, amount, commission, seller_payout, stripe_payment_id,
                   download_token, status, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,'pending',?)""",
                (purchase_id, listing_id, buyer_email, buyer_name,
                 listing["seller_id"], amount, commission, seller_payout,
                 stripe_payment_id, download_token, now),
            )
            con.commit()
        finally:
            con.close()

        # Send email with download link (after payment confirmed)
        threading.Thread(
            target=send_download_link_email,
            args=(buyer_email, buyer_name, listing["title"], download_token, purchase_id),
            daemon=True,
        ).start()

        log_info(f"Purchase {purchase_id} created for listing {listing_id} by {buyer_email}")
        self._send(201, {
            "purchase_id": purchase_id,
            "client_secret": client_secret,
            "download_token": download_token,
            "amount": amount,
            "currency": listing["currency"],
        })

    def _handle_free_purchase(self):
        body        = self._body()
        listing_id  = body.get("listing_id", "").strip()
        buyer_email = body.get("buyer_email", "").strip().lower()
        buyer_name  = body.get("buyer_name", "").strip()

        if not listing_id or not buyer_email or not buyer_name:
            self._err(400, "listing_id, buyer_email, buyer_name required")
            return

        con = _db()
        try:
            listing = con.execute(
                "SELECT * FROM listings WHERE listing_id=? AND status='active'",
                (listing_id,),
            ).fetchone()
            if not listing:
                self._err(404, "listing not found or not active")
                return
            if listing["price"] > 0:
                self._err(400, "listing is not free — use /purchase")
                return

            purchase_id    = "pur_" + secrets.token_urlsafe(16)
            download_token = secrets.token_urlsafe(32)
            now            = time.time()

            con.execute(
                """INSERT INTO purchases(purchase_id, listing_id, buyer_email, buyer_name,
                   seller_id, amount, commission, seller_payout, download_token,
                   status, created_at)
                   VALUES (?,?,?,?,?,0,0,0,?,'completed',?)""",
                (purchase_id, listing_id, buyer_email, buyer_name,
                 listing["seller_id"], download_token, now),
            )
            # Increment sales count for free listing
            con.execute(
                "UPDATE listings SET sales_count = sales_count + 1, updated_at=? WHERE listing_id=?",
                (now, listing_id),
            )
            con.commit()
        finally:
            con.close()

        threading.Thread(
            target=send_download_link_email,
            args=(buyer_email, buyer_name, listing["title"], download_token, purchase_id),
            daemon=True,
        ).start()

        log_info(f"Free purchase {purchase_id} for listing {listing_id} by {buyer_email}")
        self._send(201, {
            "purchase_id": purchase_id,
            "download_token": download_token,
            "message": "Free download ready.",
        })

    def _handle_add_review(self, listing_id: str):
        body        = self._body()
        buyer_email = body.get("buyer_email", "").strip().lower()
        rating      = body.get("rating")
        comment     = body.get("comment", "").strip()

        if not buyer_email or rating is None:
            self._err(400, "buyer_email and rating required")
            return

        try:
            rating = int(rating)
        except (TypeError, ValueError):
            self._err(400, "rating must be an integer 1-5")
            return

        if not 1 <= rating <= 5:
            self._err(400, "rating must be between 1 and 5")
            return

        con = _db()
        try:
            # Verify buyer has purchased this listing
            purchase = con.execute(
                "SELECT purchase_id FROM purchases WHERE listing_id=? AND buyer_email=? AND status IN ('completed','paid_out')",
                (listing_id, buyer_email),
            ).fetchone()
            if not purchase:
                self._err(403, "you must have purchased this listing to leave a review")
                return

            # Prevent duplicate review
            existing = con.execute(
                "SELECT id FROM reviews WHERE listing_id=? AND buyer_email=?",
                (listing_id, buyer_email),
            ).fetchone()
            if existing:
                self._err(409, "you have already reviewed this listing")
                return

            now = time.time()
            con.execute(
                "INSERT INTO reviews(listing_id, buyer_email, rating, comment, created_at) VALUES (?,?,?,?,?)",
                (listing_id, buyer_email, rating, comment, now),
            )
            # Recalculate average rating
            avg_row = con.execute(
                "SELECT AVG(rating) AS avg FROM reviews WHERE listing_id=?",
                (listing_id,),
            ).fetchone()
            new_avg = round(avg_row["avg"] or 0, 2)
            con.execute(
                "UPDATE listings SET rating=?, updated_at=? WHERE listing_id=?",
                (new_avg, now, listing_id),
            )
            con.commit()
        finally:
            con.close()

        self._send(201, {
            "listing_id": listing_id,
            "rating": rating,
            "new_average_rating": new_avg,
        })

# ---------------------------------------------------------------------------
# Server bootstrap
# ---------------------------------------------------------------------------

def start_payout_thread() -> None:
    t = threading.Thread(target=_run_payouts, name="payout-worker", daemon=True)
    t.start()
    log_info("Payout background thread started")


def run_server() -> None:
    init_db()
    start_payout_thread()
    server = HTTPServer(("0.0.0.0", PORT), MarketplaceHandler)
    log_info(f"FractalMesh Marketplace listening on :{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log_info("Shutting down")
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server()
