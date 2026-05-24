"""
FractalMesh OMEGA Titan — Software Licensing & Key Management Agent
Port: 7846
Samuel James Hiotis | ABN 56 628 117 363 | Sole Trader
"""

import os
import json
import sqlite3
import time
import hmac
import hashlib
import secrets
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Configuration (from vault / environment)
# ---------------------------------------------------------------------------
PORT             = int(os.getenv("LICENSING_PORT", "7846"))
MASTER_SECRET    = os.getenv("LICENSE_MASTER_SECRET", "")
ADMIN_SECRET     = os.getenv("ADMIN_SECRET", "")
STRIPE_SECRET    = os.getenv("STRIPE_SECRET_KEY", "")

DB_PATH          = Path(os.path.expanduser("~/fmsaas/database/sovereign.db"))

VALID_TIERS      = {"FREE", "STARTER", "PRO", "ENTERPRISE", "DATA", "OSINT"}

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    conn = _get_conn()
    with conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS products (
                id          INTEGER PRIMARY KEY,
                name        TEXT UNIQUE,
                description TEXT,
                price_usd   REAL,
                tier        TEXT,
                features    TEXT,
                active      INTEGER DEFAULT 1,
                created_at  REAL
            );

            CREATE TABLE IF NOT EXISTS licenses (
                id                 INTEGER PRIMARY KEY,
                license_key        TEXT UNIQUE,
                product_id         INTEGER,
                customer_email     TEXT,
                customer_name      TEXT,
                tier               TEXT,
                status             TEXT DEFAULT 'active',
                max_activations    INTEGER DEFAULT 1,
                activations        INTEGER DEFAULT 0,
                expires_at         REAL,
                issued_at          REAL,
                stripe_payment_id  TEXT,
                metadata           TEXT
            );

            CREATE TABLE IF NOT EXISTS activations (
                id           INTEGER PRIMARY KEY,
                license_id   INTEGER,
                machine_id   TEXT,
                ip_hash      TEXT,
                activated_at REAL,
                last_seen    REAL,
                revoked      INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS license_events (
                id           INTEGER PRIMARY KEY,
                license_id   INTEGER,
                event_type   TEXT,
                detail       TEXT,
                created_at   REAL
            );
        """)
    conn.close()


def _seed_products() -> None:
    now = time.time()
    products = [
        (
            "FractalMesh Starter",
            "Lead generation and content engine with 10k API calls/month",
            29.0,
            "STARTER",
            json.dumps(["leadgen", "content_engine", "10k_api_calls"]),
        ),
        (
            "FractalMesh Pro",
            "All Starter features plus AIaaS, NFT engine, 50k API calls, priority support",
            99.0,
            "PRO",
            json.dumps(["all_starter", "aiaas", "nft_engine", "50k_api_calls", "priority_support"]),
        ),
        (
            "FractalMesh Enterprise",
            "Unlimited usage, custom agents, white-label, dedicated support",
            499.0,
            "ENTERPRISE",
            json.dumps(["unlimited", "custom_agents", "white_label", "dedicated_support"]),
        ),
        (
            "FractalMesh Data API Access",
            "Data API with 1k daily requests",
            19.0,
            "DATA",
            json.dumps(["data_api_1k_daily"]),
        ),
        (
            "FractalMesh OSINT Report",
            "One-time OSINT report pack",
            49.0,
            "OSINT",
            json.dumps(["osint_report_pack"]),
        ),
    ]
    conn = _get_conn()
    with conn:
        for name, desc, price, tier, features in products:
            conn.execute(
                """INSERT OR IGNORE INTO products
                   (name, description, price_usd, tier, features, active, created_at)
                   VALUES (?, ?, ?, ?, ?, 1, ?)""",
                (name, desc, price, tier, features, now),
            )
    conn.close()


# ---------------------------------------------------------------------------
# License key helpers
# ---------------------------------------------------------------------------

def _generate_key(tier: str) -> str:
    """Generate a signed license key: FM-{TIER}-{random_hex_12}-{hmac_check_4}"""
    rand_part = secrets.token_hex(6).upper()          # 12 hex chars
    stem      = f"FM-{tier}-{rand_part}"
    check     = _hmac_of(stem)[:4]
    return f"{stem}-{check}"


def _hmac_of(data: str) -> str:
    secret = MASTER_SECRET.encode() if MASTER_SECRET else b"insecure-default"
    return hmac.new(secret, data.encode(), hashlib.sha256).hexdigest()


def _verify_key_format(key: str) -> bool:
    """Return True if the key has valid format and correct HMAC checksum."""
    parts = key.split("-")
    # FM-TIER-RANDOM12-check4  → parts: ['FM', tier, rand_hex_12, check4]
    if len(parts) != 4:
        return False
    prefix, tier, rand_part, check = parts
    if prefix != "FM":
        return False
    if tier not in VALID_TIERS:
        return False
    if len(rand_part) != 12 or not _is_hex(rand_part):
        return False
    if len(check) != 4 or not _is_hex(check.lower()):
        return False
    stem     = f"FM-{tier}-{rand_part}"
    expected = _hmac_of(stem)[:4]
    return hmac.compare_digest(check.lower(), expected.lower())


def _is_hex(s: str) -> bool:
    try:
        int(s, 16)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

def _check_auth(handler: "LicensingHandler") -> bool:
    if not ADMIN_SECRET:
        return False
    provided = handler.headers.get("X-Admin-Secret", "")
    return hmac.compare_digest(provided, ADMIN_SECRET)


# ---------------------------------------------------------------------------
# Event logging
# ---------------------------------------------------------------------------

def _log_event(conn: sqlite3.Connection, license_id: int, event_type: str, detail: str) -> None:
    conn.execute(
        "INSERT INTO license_events (license_id, event_type, detail, created_at) VALUES (?, ?, ?, ?)",
        (license_id, event_type, detail, time.time()),
    )


# ---------------------------------------------------------------------------
# Internal license issuance (shared by POST /licenses/issue and webhooks)
# ---------------------------------------------------------------------------

def _issue_license(
    product_id: int,
    customer_email: str,
    customer_name: str,
    expires_days: int = 365,
    max_activations: int = 1,
    stripe_payment_id: str = "",
    metadata: dict | None = None,
) -> dict | None:
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT id, name, tier FROM products WHERE id=? AND active=1", (product_id,)
        ).fetchone()
        if not row:
            return None
        product_id_db, product_name, tier = row["id"], row["name"], row["tier"]

        key        = _generate_key(tier)
        issued_at  = time.time()
        expires_at = issued_at + expires_days * 86400

        with conn:
            cur = conn.execute(
                """INSERT INTO licenses
                   (license_key, product_id, customer_email, customer_name, tier,
                    status, max_activations, activations, expires_at, issued_at,
                    stripe_payment_id, metadata)
                   VALUES (?, ?, ?, ?, ?, 'active', ?, 0, ?, ?, ?, ?)""",
                (
                    key, product_id_db, customer_email, customer_name, tier,
                    max_activations, expires_at, issued_at,
                    stripe_payment_id, json.dumps(metadata or {}),
                ),
            )
            license_id = cur.lastrowid
            _log_event(conn, license_id, "issued", f"product={product_name} email={customer_email}")

        return {
            "license_key":    key,
            "product":        product_name,
            "tier":           tier,
            "expires_at":     expires_at,
            "customer_email": customer_email,
            "customer_name":  customer_name,
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------

class LicensingHandler(BaseHTTPRequestHandler):
    server_version = "FractalMeshLicensing/2.0"

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path.rstrip("/")

        if path == "/health":
            return self._ok({"status": "ok", "service": "fm-licensing", "port": PORT})

        if path == "/products":
            return self._handle_get_products()

        if path == "/licenses":
            if not _check_auth(self):
                return self._forbidden()
            return self._handle_list_licenses()

        if path == "/analytics":
            if not _check_auth(self):
                return self._forbidden()
            return self._handle_analytics()

        if path.startswith("/licenses/"):
            key = path[len("/licenses/"):]
            if not _check_auth(self):
                return self._forbidden()
            return self._handle_get_license(key)

        return self._not_found()

    def do_POST(self):
        parsed = urlparse(self.path)
        path   = parsed.path.rstrip("/")
        body   = self._read_body()

        if path == "/licenses/issue":
            if not _check_auth(self):
                return self._forbidden()
            return self._handle_issue(body)

        if path == "/licenses/verify":
            return self._handle_verify(body)

        if path == "/licenses/activate":
            return self._handle_activate(body)

        if path == "/licenses/deactivate":
            return self._handle_deactivate(body)

        if path == "/licenses/bulk_issue":
            if not _check_auth(self):
                return self._forbidden()
            return self._handle_bulk_issue(body)

        if path == "/webhooks/stripe":
            return self._handle_stripe_webhook(body)

        if path.startswith("/licenses/") and path.endswith("/revoke"):
            key = path[len("/licenses/"):-len("/revoke")]
            if not _check_auth(self):
                return self._forbidden()
            return self._handle_revoke(key)

        return self._not_found()

    def do_PUT(self):
        parsed = urlparse(self.path)
        path   = parsed.path.rstrip("/")
        body   = self._read_body()

        if path.startswith("/licenses/"):
            key = path[len("/licenses/"):]
            if not _check_auth(self):
                return self._forbidden()
            return self._handle_update_license(key, body)

        return self._not_found()

    # ------------------------------------------------------------------
    # Endpoint implementations
    # ------------------------------------------------------------------

    def _handle_get_products(self):
        conn = _get_conn()
        rows = conn.execute(
            "SELECT id, name, description, price_usd, tier, features FROM products WHERE active=1"
        ).fetchall()
        conn.close()
        products = [
            {
                "id":          r["id"],
                "name":        r["name"],
                "description": r["description"],
                "price_usd":   r["price_usd"],
                "tier":        r["tier"],
                "features":    json.loads(r["features"] or "[]"),
            }
            for r in rows
        ]
        return self._ok({"products": products})

    def _handle_issue(self, body: dict):
        product_id      = body.get("product_id")
        customer_email  = body.get("customer_email", "")
        customer_name   = body.get("customer_name", "")
        expires_days    = int(body.get("expires_days", 365))
        max_activations = int(body.get("max_activations", 1))
        stripe_id       = body.get("stripe_payment_id", "")

        if not product_id or not customer_email:
            return self._bad_request("product_id and customer_email required")

        result = _issue_license(
            product_id, customer_email, customer_name,
            expires_days, max_activations, stripe_id,
        )
        if result is None:
            return self._bad_request("product not found or inactive")

        return self._ok(result)

    def _handle_verify(self, body: dict):
        key        = body.get("license_key", "")
        machine_id = body.get("machine_id", "")

        if not _verify_key_format(key):
            return self._ok({"valid": False, "reason": "invalid_key"})

        conn = _get_conn()
        try:
            lic = conn.execute(
                """SELECT l.id, l.status, l.expires_at, l.tier, l.max_activations,
                          l.activations, l.customer_name, l.customer_email,
                          p.name AS product_name, p.features
                   FROM licenses l
                   JOIN products p ON p.id = l.product_id
                   WHERE l.license_key=?""",
                (key,),
            ).fetchone()

            if not lic:
                return self._ok({"valid": False, "reason": "invalid_key"})

            if lic["status"] == "revoked":
                return self._ok({"valid": False, "reason": "revoked"})
            if lic["status"] == "suspended":
                return self._ok({"valid": False, "reason": "revoked"})
            if lic["expires_at"] and time.time() > lic["expires_at"]:
                return self._ok({"valid": False, "reason": "expired"})

            # Check if this machine is already activated
            existing = conn.execute(
                "SELECT id FROM activations WHERE license_id=? AND machine_id=? AND revoked=0",
                (lic["id"], machine_id),
            ).fetchone()

            if not existing:
                if lic["activations"] >= lic["max_activations"]:
                    return self._ok({"valid": False, "reason": "max_activations"})
                # Register new activation
                now = time.time()
                with conn:
                    conn.execute(
                        "INSERT INTO activations (license_id, machine_id, activated_at, last_seen) VALUES (?, ?, ?, ?)",
                        (lic["id"], machine_id, now, now),
                    )
                    conn.execute(
                        "UPDATE licenses SET activations=activations+1 WHERE id=?",
                        (lic["id"],),
                    )
                    _log_event(conn, lic["id"], "verify_new_machine", f"machine={machine_id}")
            else:
                # Update last_seen
                with conn:
                    conn.execute(
                        "UPDATE activations SET last_seen=? WHERE license_id=? AND machine_id=? AND revoked=0",
                        (time.time(), lic["id"], machine_id),
                    )
                    _log_event(conn, lic["id"], "verify_seen", f"machine={machine_id}")

            return self._ok({
                "valid":      True,
                "tier":       lic["tier"],
                "product":    lic["product_name"],
                "features":   json.loads(lic["features"] or "[]"),
                "customer":   lic["customer_name"],
                "expires_at": lic["expires_at"],
            })
        finally:
            conn.close()

    def _handle_activate(self, body: dict):
        key        = body.get("license_key", "")
        machine_id = body.get("machine_id", "")
        ip_hash    = body.get("ip_hash", "")

        if not _verify_key_format(key):
            return self._bad_request("invalid license key format")

        conn = _get_conn()
        try:
            lic = conn.execute(
                "SELECT id, status, expires_at, max_activations, activations FROM licenses WHERE license_key=?",
                (key,),
            ).fetchone()

            if not lic:
                return self._bad_request("license not found")
            if lic["status"] != "active":
                return self._bad_request(f"license status={lic['status']}")
            if lic["expires_at"] and time.time() > lic["expires_at"]:
                return self._bad_request("license expired")

            existing = conn.execute(
                "SELECT id FROM activations WHERE license_id=? AND machine_id=? AND revoked=0",
                (lic["id"], machine_id),
            ).fetchone()

            if existing:
                return self._ok({
                    "activated":         False,
                    "detail":            "already_activated",
                    "activation_id":     existing["id"],
                    "activations_used":  lic["activations"],
                    "max_activations":   lic["max_activations"],
                })

            if lic["activations"] >= lic["max_activations"]:
                return self._ok({
                    "activated":        False,
                    "detail":           "max_activations_reached",
                    "activations_used": lic["activations"],
                    "max_activations":  lic["max_activations"],
                })

            now = time.time()
            with conn:
                cur = conn.execute(
                    "INSERT INTO activations (license_id, machine_id, ip_hash, activated_at, last_seen) VALUES (?, ?, ?, ?, ?)",
                    (lic["id"], machine_id, ip_hash, now, now),
                )
                activation_id = cur.lastrowid
                conn.execute(
                    "UPDATE licenses SET activations=activations+1 WHERE id=?",
                    (lic["id"],),
                )
                _log_event(conn, lic["id"], "activated", f"machine={machine_id} activation_id={activation_id}")

            new_count = lic["activations"] + 1
            return self._ok({
                "activated":        True,
                "activation_id":    activation_id,
                "activations_used": new_count,
                "max_activations":  lic["max_activations"],
            })
        finally:
            conn.close()

    def _handle_deactivate(self, body: dict):
        key        = body.get("license_key", "")
        machine_id = body.get("machine_id", "")

        if not _verify_key_format(key):
            return self._bad_request("invalid license key format")

        conn = _get_conn()
        try:
            lic = conn.execute(
                "SELECT id FROM licenses WHERE license_key=?", (key,)
            ).fetchone()
            if not lic:
                return self._bad_request("license not found")

            act = conn.execute(
                "SELECT id FROM activations WHERE license_id=? AND machine_id=? AND revoked=0",
                (lic["id"], machine_id),
            ).fetchone()
            if not act:
                return self._bad_request("activation not found")

            with conn:
                conn.execute(
                    "UPDATE activations SET revoked=1 WHERE id=?", (act["id"],)
                )
                conn.execute(
                    "UPDATE licenses SET activations=MAX(0, activations-1) WHERE id=?",
                    (lic["id"],),
                )
                _log_event(conn, lic["id"], "deactivated", f"machine={machine_id}")

            return self._ok({"deactivated": True})
        finally:
            conn.close()

    def _handle_list_licenses(self):
        conn = _get_conn()
        rows = conn.execute(
            """SELECT l.id, l.license_key, l.customer_email, l.customer_name,
                      l.tier, l.status, l.max_activations, l.activations,
                      l.expires_at, l.issued_at, p.name AS product_name
               FROM licenses l
               JOIN products p ON p.id = l.product_id
               ORDER BY l.issued_at DESC"""
        ).fetchall()
        conn.close()
        return self._ok({
            "licenses": [
                {
                    "id":              r["id"],
                    "license_key":     r["license_key"],
                    "customer_email":  r["customer_email"],
                    "customer_name":   r["customer_name"],
                    "tier":            r["tier"],
                    "status":          r["status"],
                    "product":         r["product_name"],
                    "max_activations": r["max_activations"],
                    "activations":     r["activations"],
                    "expires_at":      r["expires_at"],
                    "issued_at":       r["issued_at"],
                }
                for r in rows
            ]
        })

    def _handle_get_license(self, key: str):
        conn = _get_conn()
        try:
            lic = conn.execute(
                """SELECT l.*, p.name AS product_name, p.features, p.price_usd
                   FROM licenses l
                   JOIN products p ON p.id = l.product_id
                   WHERE l.license_key=?""",
                (key,),
            ).fetchone()
            if not lic:
                return self._not_found()

            acts = conn.execute(
                "SELECT id, machine_id, ip_hash, activated_at, last_seen, revoked FROM activations WHERE license_id=?",
                (lic["id"],),
            ).fetchall()

            events = conn.execute(
                "SELECT event_type, detail, created_at FROM license_events WHERE license_id=? ORDER BY created_at DESC LIMIT 50",
                (lic["id"],),
            ).fetchall()

            return self._ok({
                "license": {
                    "id":               lic["id"],
                    "license_key":      lic["license_key"],
                    "product":          lic["product_name"],
                    "features":         json.loads(lic["features"] or "[]"),
                    "price_usd":        lic["price_usd"],
                    "customer_email":   lic["customer_email"],
                    "customer_name":    lic["customer_name"],
                    "tier":             lic["tier"],
                    "status":           lic["status"],
                    "max_activations":  lic["max_activations"],
                    "activations":      lic["activations"],
                    "expires_at":       lic["expires_at"],
                    "issued_at":        lic["issued_at"],
                    "stripe_payment_id":lic["stripe_payment_id"],
                    "metadata":         json.loads(lic["metadata"] or "{}"),
                },
                "activations": [
                    {
                        "id":           a["id"],
                        "machine_id":   a["machine_id"],
                        "ip_hash":      a["ip_hash"],
                        "activated_at": a["activated_at"],
                        "last_seen":    a["last_seen"],
                        "revoked":      bool(a["revoked"]),
                    }
                    for a in acts
                ],
                "events": [
                    {"event_type": e["event_type"], "detail": e["detail"], "created_at": e["created_at"]}
                    for e in events
                ],
            })
        finally:
            conn.close()

    def _handle_update_license(self, key: str, body: dict):
        conn = _get_conn()
        try:
            lic = conn.execute(
                "SELECT id FROM licenses WHERE license_key=?", (key,)
            ).fetchone()
            if not lic:
                return self._not_found()

            updates = []
            params  = []

            if "status" in body:
                updates.append("status=?")
                params.append(body["status"])

            if "expires_days" in body:
                new_exp = time.time() + int(body["expires_days"]) * 86400
                updates.append("expires_at=?")
                params.append(new_exp)

            if "max_activations" in body:
                updates.append("max_activations=?")
                params.append(int(body["max_activations"]))

            if not updates:
                return self._bad_request("no valid fields to update")

            params.append(lic["id"])
            with conn:
                conn.execute(
                    f"UPDATE licenses SET {', '.join(updates)} WHERE id=?", params
                )
                _log_event(conn, lic["id"], "updated", json.dumps(body))

            return self._ok({"updated": True, "license_key": key})
        finally:
            conn.close()

    def _handle_revoke(self, key: str):
        conn = _get_conn()
        try:
            lic = conn.execute(
                "SELECT id FROM licenses WHERE license_key=?", (key,)
            ).fetchone()
            if not lic:
                return self._not_found()

            with conn:
                conn.execute(
                    "UPDATE licenses SET status='revoked' WHERE id=?", (lic["id"],)
                )
                _log_event(conn, lic["id"], "revoked", "admin revoke")

            return self._ok({"revoked": True, "license_key": key})
        finally:
            conn.close()

    def _handle_stripe_webhook(self, body: dict):
        event_type = body.get("type", "")
        data       = body.get("data", {}).get("object", {})

        if event_type not in ("payment_intent.succeeded", "checkout.session.completed"):
            return self._ok({"received": True})

        metadata = data.get("metadata", {})
        customer_email = (
            data.get("customer_email")
            or data.get("customer_details", {}).get("email", "")
            or metadata.get("customer_email", "")
        )
        customer_name = metadata.get("customer_name", "")
        product_id_str = metadata.get("product_id", "")

        if not customer_email or not product_id_str:
            return self._ok({"received": True, "detail": "missing metadata"})

        try:
            product_id = int(product_id_str)
        except (ValueError, TypeError):
            return self._ok({"received": True, "detail": "invalid product_id"})

        stripe_id = data.get("id", "")
        result    = _issue_license(
            product_id, customer_email, customer_name,
            expires_days=365, max_activations=1,
            stripe_payment_id=stripe_id, metadata={"source": "stripe_webhook"},
        )

        if result:
            return self._ok({"received": True, "license_key": result["license_key"]})
        return self._ok({"received": True, "detail": "product not found"})

    def _handle_analytics(self):
        conn = _get_conn()
        try:
            total      = conn.execute("SELECT COUNT(*) FROM licenses").fetchone()[0]
            active     = conn.execute("SELECT COUNT(*) FROM licenses WHERE status='active'").fetchone()[0]
            expired    = conn.execute(
                "SELECT COUNT(*) FROM licenses WHERE expires_at IS NOT NULL AND expires_at < ?",
                (time.time(),),
            ).fetchone()[0]
            revoked    = conn.execute("SELECT COUNT(*) FROM licenses WHERE status='revoked'").fetchone()[0]
            acts_total = conn.execute("SELECT COUNT(*) FROM activations WHERE revoked=0").fetchone()[0]

            revenue = conn.execute(
                """SELECT COALESCE(SUM(p.price_usd), 0)
                   FROM licenses l
                   JOIN products p ON p.id = l.product_id
                   WHERE l.status='active'"""
            ).fetchone()[0]

            tier_rows = conn.execute(
                """SELECT l.tier, COUNT(*) AS cnt
                   FROM licenses l
                   GROUP BY l.tier"""
            ).fetchall()

            return self._ok({
                "licenses_issued":   total,
                "active":            active,
                "expired":           expired,
                "revoked":           revoked,
                "activations_total": acts_total,
                "revenue_estimate":  round(revenue, 2),
                "by_tier":           {r["tier"]: r["cnt"] for r in tier_rows},
            })
        finally:
            conn.close()

    def _handle_bulk_issue(self, body: dict):
        product_id   = body.get("product_id")
        emails       = body.get("emails", [])
        expires_days = int(body.get("expires_days", 30))

        if not product_id or not emails:
            return self._bad_request("product_id and emails required")

        issued = 0
        keys   = []
        for email in emails:
            result = _issue_license(
                product_id, str(email), "",
                expires_days=expires_days, max_activations=1,
            )
            if result:
                issued += 1
                keys.append(result["license_key"])

        return self._ok({"issued": issued, "keys": keys})

    # ------------------------------------------------------------------
    # Low-level response helpers
    # ------------------------------------------------------------------

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, ValueError):
            return {}

    def _ok(self, payload: dict, status: int = 200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _bad_request(self, reason: str = "bad request"):
        self._ok({"error": reason}, 400)

    def _forbidden(self):
        self._ok({"error": "forbidden"}, 403)

    def _not_found(self):
        self._ok({"error": "not found"}, 404)

    def log_message(self, fmt, *args):
        print(f"[fm-licensing] {self.address_string()} - {fmt % args}")


# ---------------------------------------------------------------------------
# Startup demo license
# ---------------------------------------------------------------------------

def _seed_demo_license() -> None:
    """Generate one FREE demo license and log it to stdout."""
    conn = _get_conn()
    try:
        existing = conn.execute(
            "SELECT COUNT(*) FROM licenses WHERE tier='FREE'"
        ).fetchone()[0]
        if existing > 0:
            return
    finally:
        conn.close()

    key = _generate_key("FREE")
    now = time.time()
    conn = _get_conn()
    try:
        product = conn.execute(
            "SELECT id FROM products WHERE tier='STARTER' LIMIT 1"
        ).fetchone()
        if product:
            with conn:
                cur = conn.execute(
                    """INSERT INTO licenses
                       (license_key, product_id, customer_email, customer_name, tier,
                        status, max_activations, activations, expires_at, issued_at, metadata)
                       VALUES (?, ?, ?, ?, 'FREE', 'active', 1, 0, ?, ?, ?)""",
                    (
                        key,
                        product["id"],
                        "demo@fractalmesh.net",
                        "FractalMesh Demo",
                        now + 30 * 86400,
                        now,
                        json.dumps({"note": "startup demo license"}),
                    ),
                )
                _log_event(conn, cur.lastrowid, "issued", "startup demo")
    finally:
        conn.close()

    print(f"[fm-licensing] Demo FREE license generated: {key}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"[fm-licensing] Initialising database at {DB_PATH}")
    _init_db()
    _seed_products()
    _seed_demo_license()

    server = HTTPServer(("0.0.0.0", PORT), LicensingHandler)
    print(f"[fm-licensing] Listening on port {PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[fm-licensing] Stopped.")
    finally:
        server.server_close()
