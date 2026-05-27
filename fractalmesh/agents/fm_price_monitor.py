#!/usr/bin/env python3
"""
FractalMesh OMEGA Titan — Product/Competitor Price Monitor
Port: 7877

Automated price monitoring agent. Tracks product/service prices from competitor
websites and APIs, detects price changes, fires alerts, and maintains price
history for trend analysis.

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
import html
import json
import re
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
AGENT_NAME = "fm_price_monitor"
PORT = int(os.environ.get("PRICE_MONITOR_PORT", "7877"))

CRAWLBASE_NORMAL_TOKEN = os.environ.get("CRAWLBASE_NORMAL_TOKEN", "")
SENDGRID_API_KEY       = os.environ.get("SENDGRID_API_KEY", "")
SENDGRID_FROM_EMAIL    = os.environ.get("SENDGRID_FROM_EMAIL", "")
ADMIN_SECRET           = os.environ.get("ADMIN_SECRET", "")

CRAWLBASE_API_BASE  = "https://api.crawlbase.com/"
SENDGRID_SEND_URL   = "https://api.sendgrid.com/v3/mail/send"
MONITOR_INTERVAL    = 300   # seconds between background sweep passes
DB_PATH = Path.home() / "fmsaas" / "database" / "sovereign.db"

START_TIME = time.time()

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = _get_conn()
    with conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS products (
                id          INTEGER PRIMARY KEY,
                name        TEXT    NOT NULL,
                category    TEXT,
                our_price   REAL    NOT NULL,
                currency    TEXT    DEFAULT 'AUD',
                created_at  REAL    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS competitors (
                id          INTEGER PRIMARY KEY,
                name        TEXT    NOT NULL,
                website     TEXT,
                created_at  REAL    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS price_watches (
                id                    INTEGER PRIMARY KEY,
                product_id            INTEGER NOT NULL,
                competitor_id         INTEGER NOT NULL,
                watch_url             TEXT    NOT NULL,
                price_selector        TEXT,
                extract_pattern       TEXT,
                current_price         REAL,
                last_checked          REAL,
                check_interval_hours  INTEGER DEFAULT 24,
                alert_on_change       INTEGER DEFAULT 1,
                alert_threshold_pct   REAL    DEFAULT 5.0,
                status                TEXT    DEFAULT 'active',
                created_at            REAL    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS price_history (
                id          INTEGER PRIMARY KEY,
                watch_id    INTEGER NOT NULL,
                price       REAL,
                currency    TEXT,
                raw_text    TEXT,
                checked_at  REAL    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS alerts (
                id          INTEGER PRIMARY KEY,
                watch_id    INTEGER NOT NULL,
                alert_type  TEXT,
                old_price   REAL,
                new_price   REAL,
                change_pct  REAL,
                message     TEXT,
                sent        INTEGER DEFAULT 0,
                created_at  REAL    NOT NULL
            );
        """)
    conn.close()


def _seed_data() -> None:
    conn = _get_conn()
    try:
        row = conn.execute("SELECT COUNT(*) FROM products").fetchone()
        if row[0] > 0:
            return
        now = time.time()
        with conn:
            conn.execute(
                "INSERT INTO products (name, category, our_price, currency, created_at) VALUES (?,?,?,?,?)",
                ("FractalMesh Starter Plan", "SaaS", 29.0, "AUD", now),
            )
            conn.execute(
                "INSERT INTO products (name, category, our_price, currency, created_at) VALUES (?,?,?,?,?)",
                ("FractalMesh Pro Plan", "SaaS", 99.0, "AUD", now),
            )
            conn.execute(
                "INSERT INTO competitors (name, website, created_at) VALUES (?,?,?)",
                ("Competitor A", "https://example.com", now),
            )
            conn.execute(
                "INSERT INTO competitors (name, website, created_at) VALUES (?,?,?)",
                ("Competitor B", "https://example.org", now),
            )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Price extraction
# ---------------------------------------------------------------------------

def _clean_price_string(raw: str) -> float | None:
    """Strip currency symbols/commas and parse as float."""
    cleaned = re.sub(r"[^\d.]", "", raw.replace(",", ""))
    try:
        val = float(cleaned)
        return val if val > 0 else None
    except (ValueError, TypeError):
        return None


def _extract_price_from_html(html_text: str, price_selector: str | None,
                              extract_pattern: str | None) -> tuple[float | None, str]:
    """
    Returns (price_float_or_None, raw_matched_text).

    Strategy:
      1. If extract_pattern set: use as regex on raw HTML, first group.
      2. Otherwise: scan ±500 chars around first occurrence of price_selector
         for a price-like string ($\d+[\d,.]*).
      3. Clean and parse.
    """
    raw_text = ""
    try:
        if extract_pattern:
            m = re.search(extract_pattern, html_text, re.IGNORECASE | re.DOTALL)
            if m:
                raw_text = m.group(1) if m.lastindex and m.lastindex >= 1 else m.group(0)
                price = _clean_price_string(raw_text)
                return price, raw_text
            return None, ""

        # Selector-based scan
        selector_hint = price_selector or ""
        search_text = html_text

        if selector_hint:
            idx = html_text.find(selector_hint)
            if idx == -1:
                # Try case-insensitive
                lower_html = html_text.lower()
                idx = lower_html.find(selector_hint.lower())
            if idx != -1:
                start = max(0, idx - 500)
                end = min(len(html_text), idx + len(selector_hint) + 500)
                search_text = html_text[start:end]

        # Look for price pattern: $ followed by digits/commas/dots
        price_pattern = r'\$\s*([\d,]+(?:\.\d{1,2})?)'
        m = re.search(price_pattern, search_text)
        if m:
            raw_text = m.group(0)
            price = _clean_price_string(m.group(1))
            return price, raw_text

        return None, ""
    except Exception:
        return None, ""


# ---------------------------------------------------------------------------
# Crawlbase fetch
# ---------------------------------------------------------------------------

def _crawlbase_fetch(url: str) -> str | None:
    """Fetch a URL via Crawlbase and return HTML string, or None on failure."""
    if not CRAWLBASE_NORMAL_TOKEN:
        return None
    api_url = f"{CRAWLBASE_API_BASE}?token={CRAWLBASE_NORMAL_TOKEN}&url={urllib.parse.quote(url, safe='')}"
    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "FractalMesh-PriceMonitor/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return None


# ---------------------------------------------------------------------------
# SendGrid alert email
# ---------------------------------------------------------------------------

def _send_alert_email(alert_id: int, watch_id: int, message: str,
                       old_price: float | None, new_price: float | None,
                       change_pct: float | None) -> bool:
    """Send a price alert email via SendGrid. Returns True on success."""
    if not SENDGRID_API_KEY or not SENDGRID_FROM_EMAIL:
        return False

    subject = f"[FractalMesh] Price Alert — Watch #{watch_id}"
    body = (
        f"Price Alert\n\n"
        f"Watch ID   : {watch_id}\n"
        f"Alert ID   : {alert_id}\n"
        f"Old Price  : {old_price}\n"
        f"New Price  : {new_price}\n"
        f"Change %   : {change_pct:.2f}%\n\n"
        f"{message}"
    ) if change_pct is not None else message

    payload = json.dumps({
        "personalizations": [{"to": [{"email": SENDGRID_FROM_EMAIL}]}],
        "from": {"email": SENDGRID_FROM_EMAIL},
        "subject": subject,
        "content": [{"type": "text/plain", "value": body}],
    }).encode("utf-8")

    req = urllib.request.Request(
        SENDGRID_SEND_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {SENDGRID_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status in (200, 201, 202)
    except (urllib.error.URLError, urllib.error.HTTPError):
        return False


# ---------------------------------------------------------------------------
# Single watch check
# ---------------------------------------------------------------------------

def _check_watch(watch: sqlite3.Row) -> None:
    """Fetch price for one watch, update history, fire alert if needed."""
    watch_id    = watch["id"]
    url         = watch["watch_url"]
    selector    = watch["price_selector"]
    pattern     = watch["extract_pattern"]
    old_price   = watch["current_price"]
    threshold   = watch["alert_threshold_pct"] or 5.0
    alert_on    = watch["alert_on_change"]
    now         = time.time()

    html_text = _crawlbase_fetch(url)
    if html_text is None:
        return

    new_price, raw_text = _extract_price_from_html(html_text, selector, pattern)
    if new_price is None:
        return

    conn = _get_conn()
    try:
        with conn:
            conn.execute(
                "INSERT INTO price_history (watch_id, price, currency, raw_text, checked_at) VALUES (?,?,?,?,?)",
                (watch_id, new_price, "AUD", raw_text, now),
            )
            conn.execute(
                "UPDATE price_watches SET current_price=?, last_checked=? WHERE id=?",
                (new_price, now, watch_id),
            )

        if alert_on and old_price is not None and old_price > 0:
            change_pct = abs((new_price - old_price) / old_price) * 100.0
            if change_pct >= threshold:
                direction  = "increased" if new_price > old_price else "decreased"
                alert_type = "price_increase" if new_price > old_price else "price_decrease"
                message = (
                    f"Competitor price {direction} by {change_pct:.2f}% "
                    f"(was {old_price}, now {new_price}) on watch #{watch_id}."
                )
                with conn:
                    cur = conn.execute(
                        "INSERT INTO alerts (watch_id, alert_type, old_price, new_price, change_pct, message, sent, created_at) "
                        "VALUES (?,?,?,?,?,?,0,?)",
                        (watch_id, alert_type, old_price, new_price, change_pct, message, now),
                    )
                    alert_id = cur.lastrowid

                sent = _send_alert_email(alert_id, watch_id, message, old_price, new_price, change_pct)
                if sent:
                    with conn:
                        conn.execute("UPDATE alerts SET sent=1 WHERE id=?", (alert_id,))
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Background monitor thread
# ---------------------------------------------------------------------------

def _monitor_loop() -> None:
    """Daemon thread: every MONITOR_INTERVAL seconds check due watches."""
    while True:
        try:
            now = time.time()
            conn = _get_conn()
            try:
                watches = conn.execute(
                    "SELECT * FROM price_watches "
                    "WHERE status='active' "
                    "  AND (last_checked IS NULL OR last_checked + check_interval_hours * 3600 <= ?)",
                    (now,),
                ).fetchall()
            finally:
                conn.close()

            for watch in watches:
                try:
                    _check_watch(watch)
                except Exception:
                    pass
        except Exception:
            pass
        time.sleep(MONITOR_INTERVAL)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _json_response(handler: BaseHTTPRequestHandler, code: int, data: object) -> None:
    body = json.dumps(data, default=str).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_body(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", 0))
    if length == 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return {}


def _check_admin(handler: BaseHTTPRequestHandler) -> bool:
    auth = handler.headers.get("X-Admin-Secret", "") or handler.headers.get("Authorization", "")
    token = auth.replace("Bearer ", "").strip()
    if not ADMIN_SECRET:
        return True  # no secret configured — open
    return hashlib.sha256(token.encode()).hexdigest() == hashlib.sha256(ADMIN_SECRET.encode()).hexdigest()


def _parse_qs(path: str) -> dict:
    if "?" in path:
        return dict(urllib.parse.parse_qsl(path.split("?", 1)[1]))
    return {}


def _row_to_dict(row: sqlite3.Row) -> dict:
    return dict(zip(row.keys(), tuple(row)))


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

def _handle_health(handler: BaseHTTPRequestHandler) -> None:
    conn = _get_conn()
    try:
        active  = conn.execute("SELECT COUNT(*) FROM price_watches WHERE status='active'").fetchone()[0]
        n_alerts = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
    finally:
        conn.close()
    _json_response(handler, 200, {
        "status": "ok",
        "agent": AGENT_NAME,
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "active_watches": active,
        "total_alerts": n_alerts,
        "port": PORT,
    })


def _handle_get_products(handler: BaseHTTPRequestHandler) -> None:
    conn = _get_conn()
    try:
        rows = conn.execute("SELECT * FROM products ORDER BY created_at").fetchall()
    finally:
        conn.close()
    _json_response(handler, 200, {"products": [_row_to_dict(r) for r in rows]})


def _handle_post_products(handler: BaseHTTPRequestHandler) -> None:
    if not _check_admin(handler):
        _json_response(handler, 403, {"error": "forbidden"})
        return
    body = _read_body(handler)
    name      = (body.get("name") or "").strip()
    category  = (body.get("category") or "").strip()
    our_price = body.get("our_price")
    currency  = (body.get("currency") or "AUD").strip()
    if not name or our_price is None:
        _json_response(handler, 400, {"error": "name and our_price are required"})
        return
    try:
        our_price = float(our_price)
    except (TypeError, ValueError):
        _json_response(handler, 400, {"error": "our_price must be numeric"})
        return
    now = time.time()
    conn = _get_conn()
    try:
        with conn:
            cur = conn.execute(
                "INSERT INTO products (name, category, our_price, currency, created_at) VALUES (?,?,?,?,?)",
                (name, category, our_price, currency, now),
            )
            row = conn.execute("SELECT * FROM products WHERE id=?", (cur.lastrowid,)).fetchone()
    finally:
        conn.close()
    _json_response(handler, 201, {"product": _row_to_dict(row)})


def _handle_get_competitors(handler: BaseHTTPRequestHandler) -> None:
    conn = _get_conn()
    try:
        rows = conn.execute("SELECT * FROM competitors ORDER BY created_at").fetchall()
    finally:
        conn.close()
    _json_response(handler, 200, {"competitors": [_row_to_dict(r) for r in rows]})


def _handle_post_competitors(handler: BaseHTTPRequestHandler) -> None:
    if not _check_admin(handler):
        _json_response(handler, 403, {"error": "forbidden"})
        return
    body    = _read_body(handler)
    name    = (body.get("name") or "").strip()
    website = (body.get("website") or "").strip()
    if not name:
        _json_response(handler, 400, {"error": "name is required"})
        return
    now = time.time()
    conn = _get_conn()
    try:
        with conn:
            cur = conn.execute(
                "INSERT INTO competitors (name, website, created_at) VALUES (?,?,?)",
                (name, website, now),
            )
            row = conn.execute("SELECT * FROM competitors WHERE id=?", (cur.lastrowid,)).fetchone()
    finally:
        conn.close()
    _json_response(handler, 201, {"competitor": _row_to_dict(row)})


def _handle_get_watches(handler: BaseHTTPRequestHandler, qs: dict) -> None:
    conditions = []
    params: list = []
    if "product_id" in qs:
        conditions.append("product_id = ?")
        params.append(int(qs["product_id"]))
    if "competitor_id" in qs:
        conditions.append("competitor_id = ?")
        params.append(int(qs["competitor_id"]))
    if "status" in qs:
        conditions.append("status = ?")
        params.append(qs["status"])
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    conn = _get_conn()
    try:
        rows = conn.execute(f"SELECT * FROM price_watches {where} ORDER BY created_at", params).fetchall()
    finally:
        conn.close()
    _json_response(handler, 200, {"watches": [_row_to_dict(r) for r in rows]})


def _handle_get_watch_detail(handler: BaseHTTPRequestHandler, watch_id: int) -> None:
    conn = _get_conn()
    try:
        watch = conn.execute("SELECT * FROM price_watches WHERE id=?", (watch_id,)).fetchone()
        if watch is None:
            _json_response(handler, 404, {"error": "watch not found"})
            return
        history = conn.execute(
            "SELECT * FROM price_history WHERE watch_id=? ORDER BY checked_at DESC LIMIT 30",
            (watch_id,),
        ).fetchall()
    finally:
        conn.close()
    _json_response(handler, 200, {
        "watch": _row_to_dict(watch),
        "history": [_row_to_dict(r) for r in history],
    })


def _handle_post_watches(handler: BaseHTTPRequestHandler) -> None:
    if not _check_admin(handler):
        _json_response(handler, 403, {"error": "forbidden"})
        return
    body = _read_body(handler)
    product_id    = body.get("product_id")
    competitor_id = body.get("competitor_id")
    watch_url     = (body.get("watch_url") or "").strip()
    if not product_id or not competitor_id or not watch_url:
        _json_response(handler, 400, {"error": "product_id, competitor_id, and watch_url are required"})
        return
    price_selector      = (body.get("price_selector") or "").strip() or None
    extract_pattern     = (body.get("extract_pattern") or "").strip() or None
    check_interval_h    = int(body.get("check_interval_hours", 24))
    alert_threshold_pct = float(body.get("alert_threshold_pct", 5.0))
    now = time.time()
    conn = _get_conn()
    try:
        with conn:
            cur = conn.execute(
                "INSERT INTO price_watches "
                "(product_id, competitor_id, watch_url, price_selector, extract_pattern, "
                " check_interval_hours, alert_threshold_pct, status, created_at) "
                "VALUES (?,?,?,?,?,?,?,'active',?)",
                (product_id, competitor_id, watch_url, price_selector, extract_pattern,
                 check_interval_h, alert_threshold_pct, now),
            )
            row = conn.execute("SELECT * FROM price_watches WHERE id=?", (cur.lastrowid,)).fetchone()
    finally:
        conn.close()
    _json_response(handler, 201, {"watch": _row_to_dict(row)})


def _handle_post_watch_check(handler: BaseHTTPRequestHandler, watch_id: int) -> None:
    if not _check_admin(handler):
        _json_response(handler, 403, {"error": "forbidden"})
        return
    conn = _get_conn()
    try:
        watch = conn.execute("SELECT * FROM price_watches WHERE id=?", (watch_id,)).fetchone()
    finally:
        conn.close()
    if watch is None:
        _json_response(handler, 404, {"error": "watch not found"})
        return
    threading.Thread(target=_check_watch, args=(watch,), daemon=True).start()
    _json_response(handler, 202, {"status": "check_queued", "watch_id": watch_id})


def _handle_put_watch(handler: BaseHTTPRequestHandler, watch_id: int) -> None:
    if not _check_admin(handler):
        _json_response(handler, 403, {"error": "forbidden"})
        return
    body = _read_body(handler)
    conn = _get_conn()
    try:
        watch = conn.execute("SELECT * FROM price_watches WHERE id=?", (watch_id,)).fetchone()
        if watch is None:
            _json_response(handler, 404, {"error": "watch not found"})
            return
        allowed = {
            "watch_url", "price_selector", "extract_pattern",
            "check_interval_hours", "alert_on_change",
            "alert_threshold_pct", "status",
        }
        updates = {k: v for k, v in body.items() if k in allowed}
        if not updates:
            _json_response(handler, 400, {"error": "no valid fields to update"})
            return
        set_clause = ", ".join(f"{k}=?" for k in updates)
        vals = list(updates.values()) + [watch_id]
        with conn:
            conn.execute(f"UPDATE price_watches SET {set_clause} WHERE id=?", vals)
            row = conn.execute("SELECT * FROM price_watches WHERE id=?", (watch_id,)).fetchone()
    finally:
        conn.close()
    _json_response(handler, 200, {"watch": _row_to_dict(row)})


def _handle_delete_watch(handler: BaseHTTPRequestHandler, watch_id: int) -> None:
    if not _check_admin(handler):
        _json_response(handler, 403, {"error": "forbidden"})
        return
    conn = _get_conn()
    try:
        watch = conn.execute("SELECT * FROM price_watches WHERE id=?", (watch_id,)).fetchone()
        if watch is None:
            _json_response(handler, 404, {"error": "watch not found"})
            return
        with conn:
            conn.execute("UPDATE price_watches SET status='inactive' WHERE id=?", (watch_id,))
    finally:
        conn.close()
    _json_response(handler, 200, {"status": "deactivated", "watch_id": watch_id})


def _handle_get_history(handler: BaseHTTPRequestHandler, qs: dict) -> None:
    conditions = []
    params: list = []
    if "watch_id" in qs:
        conditions.append("watch_id = ?")
        params.append(int(qs["watch_id"]))
    if "days" in qs:
        cutoff = time.time() - float(qs["days"]) * 86400
        conditions.append("checked_at >= ?")
        params.append(cutoff)
    limit = int(qs.get("limit", 100))
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    conn = _get_conn()
    try:
        rows = conn.execute(
            f"SELECT * FROM price_history {where} ORDER BY checked_at DESC LIMIT ?",
            params + [limit],
        ).fetchall()
    finally:
        conn.close()
    _json_response(handler, 200, {"history": [_row_to_dict(r) for r in rows]})


def _handle_get_alerts(handler: BaseHTTPRequestHandler, qs: dict) -> None:
    conditions = []
    params: list = []
    if "sent" in qs:
        conditions.append("sent = ?")
        params.append(int(qs["sent"]))
    if "watch_id" in qs:
        conditions.append("watch_id = ?")
        params.append(int(qs["watch_id"]))
    limit = int(qs.get("limit", 50))
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    conn = _get_conn()
    try:
        rows = conn.execute(
            f"SELECT * FROM alerts {where} ORDER BY created_at DESC LIMIT ?",
            params + [limit],
        ).fetchall()
    finally:
        conn.close()
    _json_response(handler, 200, {"alerts": [_row_to_dict(r) for r in rows]})


def _handle_get_analysis(handler: BaseHTTPRequestHandler) -> None:
    """
    For each product, compare our_price to competitor current_prices.
    Returns: cheapest competitor, most expensive competitor, our percentile rank.
    """
    conn = _get_conn()
    try:
        products = conn.execute("SELECT * FROM products ORDER BY id").fetchall()
        result = []
        for product in products:
            pid = product["id"]
            # Gather active watches with a known current_price for this product
            watches = conn.execute(
                "SELECT pw.*, c.name AS competitor_name FROM price_watches pw "
                "JOIN competitors c ON c.id = pw.competitor_id "
                "WHERE pw.product_id=? AND pw.status='active' AND pw.current_price IS NOT NULL",
                (pid,),
            ).fetchall()

            competitor_prices = [
                {"competitor_id": w["competitor_id"],
                 "competitor_name": w["competitor_name"],
                 "watch_id": w["id"],
                 "price": w["current_price"]}
                for w in watches
            ]

            our_price   = product["our_price"]
            all_prices  = [cp["price"] for cp in competitor_prices]

            cheapest        = None
            most_expensive  = None
            our_percentile  = None

            if all_prices:
                min_price = min(all_prices)
                max_price = max(all_prices)
                cheapest = next(
                    (cp for cp in competitor_prices if cp["price"] == min_price), None
                )
                most_expensive = next(
                    (cp for cp in competitor_prices if cp["price"] == max_price), None
                )
                # Percentile: fraction of competitor prices our price is above
                below = sum(1 for p in all_prices if our_price > p)
                our_percentile = round(below / len(all_prices) * 100, 1)

            result.append({
                "product_id":      pid,
                "product_name":    product["name"],
                "our_price":       our_price,
                "currency":        product["currency"],
                "competitor_count": len(competitor_prices),
                "competitor_prices": competitor_prices,
                "cheapest_competitor": cheapest,
                "most_expensive_competitor": most_expensive,
                "our_percentile_rank": our_percentile,
            })
    finally:
        conn.close()
    _json_response(handler, 200, {"analysis": result})


# ---------------------------------------------------------------------------
# Request router
# ---------------------------------------------------------------------------

_WATCH_ID_RE     = re.compile(r"^/watches/(\d+)$")
_WATCH_CHECK_RE  = re.compile(r"^/watches/(\d+)/check$")


class PriceMonitorHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # suppress default access log noise
        pass

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/") or "/"
        qs   = _parse_qs(self.path)

        if path == "/health":
            _handle_health(self)
        elif path == "/products":
            _handle_get_products(self)
        elif path == "/competitors":
            _handle_get_competitors(self)
        elif path == "/watches":
            _handle_get_watches(self, qs)
        elif m := _WATCH_ID_RE.match(path):
            _handle_get_watch_detail(self, int(m.group(1)))
        elif path == "/history":
            _handle_get_history(self, qs)
        elif path == "/alerts":
            _handle_get_alerts(self, qs)
        elif path == "/analysis":
            _handle_get_analysis(self)
        else:
            _json_response(self, 404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/") or "/"

        if path == "/products":
            _handle_post_products(self)
        elif path == "/competitors":
            _handle_post_competitors(self)
        elif path == "/watches":
            _handle_post_watches(self)
        elif m := _WATCH_CHECK_RE.match(path):
            _handle_post_watch_check(self, int(m.group(1)))
        else:
            _json_response(self, 404, {"error": "not found"})

    def do_PUT(self):
        path = self.path.split("?")[0].rstrip("/") or "/"
        if m := _WATCH_ID_RE.match(path):
            _handle_put_watch(self, int(m.group(1)))
        else:
            _json_response(self, 404, {"error": "not found"})

    def do_DELETE(self):
        path = self.path.split("?")[0].rstrip("/") or "/"
        if m := _WATCH_ID_RE.match(path):
            _handle_delete_watch(self, int(m.group(1)))
        else:
            _json_response(self, 404, {"error": "not found"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _init_db()
    _seed_data()

    monitor_thread = threading.Thread(target=_monitor_loop, daemon=True, name="price-monitor-bg")
    monitor_thread.start()

    server = HTTPServer(("0.0.0.0", PORT), PriceMonitorHandler)
    print(f"[{AGENT_NAME}] Listening on port {PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"[{AGENT_NAME}] Shutting down")
        server.server_close()


if __name__ == "__main__":
    main()
