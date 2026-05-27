#!/usr/bin/env python3
"""
fm_rate_limiter.py — FractalMesh OMEGA Titan Distributed Rate Limiter / API Gateway (Port 7848)
Token Bucket / Sliding Window rate limiting with SQLite WAL, admin controls, analytics.
Samuel James Hiotis | ABN 56 628 117 363
"""
import os, json, time, sqlite3, threading, hashlib, hmac, fcntl
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

# ---------------------------------------------------------------------------
# Vault / env bootstrap
# ---------------------------------------------------------------------------
_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _l in _vault.read_text().splitlines():
        if "=" in _l and not _l.startswith("#"):
            _k, _, _v = _l.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

PORT         = int(os.getenv("RATE_LIMITER_PORT", "7848"))
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")
ROOT         = Path(os.path.expanduser("~/fmsaas"))
DB           = ROOT / "database" / "sovereign.db"

for p in (ROOT, DB.parent):
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------
def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB, timeout=15, check_same_thread=False)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA busy_timeout=10000")
    c.row_factory = sqlite3.Row
    return c


def _db_init():
    c = _conn()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS rate_rules (
            id              INTEGER PRIMARY KEY,
            name            TEXT UNIQUE NOT NULL,
            identifier_type TEXT NOT NULL,
            max_requests    INTEGER NOT NULL,
            window_seconds  INTEGER NOT NULL,
            burst           INTEGER DEFAULT 0,
            action          TEXT DEFAULT 'reject',
            enabled         INTEGER DEFAULT 1,
            created_at      REAL
        );

        CREATE TABLE IF NOT EXISTS rate_counters (
            id              INTEGER PRIMARY KEY,
            rule_id         INTEGER NOT NULL,
            identifier      TEXT NOT NULL,
            window_start    REAL NOT NULL,
            request_count   INTEGER NOT NULL DEFAULT 0,
            last_request    REAL NOT NULL,
            UNIQUE(rule_id, identifier, window_start)
        );

        CREATE INDEX IF NOT EXISTS idx_counters_lookup
            ON rate_counters (rule_id, identifier, window_start);

        CREATE TABLE IF NOT EXISTS rate_events (
            id          INTEGER PRIMARY KEY,
            rule_id     INTEGER,
            identifier  TEXT,
            event_type  TEXT,
            detail      TEXT,
            created_at  REAL
        );
    """)
    c.commit()
    c.close()


# ---------------------------------------------------------------------------
# Core algorithm helpers
# ---------------------------------------------------------------------------
def _sliding_window_count(conn: sqlite3.Connection, rule_id: int,
                          identifier: str, window_seconds: int, now: float) -> int:
    """Count requests for identifier in the last window_seconds."""
    cutoff = now - window_seconds
    row = conn.execute(
        "SELECT COALESCE(SUM(request_count),0) FROM rate_counters "
        "WHERE rule_id=? AND identifier=? AND window_start >= ?",
        (rule_id, identifier, cutoff)
    ).fetchone()
    return int(row[0]) if row else 0


def _check_rate(conn: sqlite3.Connection, rule_row: sqlite3.Row,
                identifier: str, increment: bool) -> tuple:
    """
    Returns (allowed: bool, remaining: int, reset_at: float)
    Uses sliding window; burst allows up to max_requests+burst in first second.
    """
    now        = time.time()
    rule_id    = rule_row["id"]
    max_req    = rule_row["max_requests"]
    window     = rule_row["window_seconds"]
    burst      = rule_row["burst"]
    action     = rule_row["action"]

    # purge stale sub-windows for this identifier
    cutoff = now - window
    conn.execute(
        "DELETE FROM rate_counters WHERE rule_id=? AND identifier=? AND window_start < ?",
        (rule_id, identifier, cutoff)
    )

    current_count = _sliding_window_count(conn, rule_id, identifier, window, now)

    # determine effective limit (burst only relevant if log_only or first window)
    effective_limit = max_req
    if burst > 0:
        # check if identifier has any history at all in this window
        existing = conn.execute(
            "SELECT COUNT(*) FROM rate_counters WHERE rule_id=? AND identifier=?",
            (rule_id, identifier)
        ).fetchone()[0]
        if existing == 0:
            effective_limit = max_req + burst

    if action == "log_only":
        allowed = True
    else:
        allowed = current_count < effective_limit

    remaining = max(0, effective_limit - current_count - (1 if allowed and increment else 0))

    # compute when the oldest window expires
    oldest = conn.execute(
        "SELECT MIN(window_start) FROM rate_counters WHERE rule_id=? AND identifier=?",
        (rule_id, identifier)
    ).fetchone()[0]
    reset_at = (oldest + window) if oldest else (now + window)

    if allowed and increment:
        # upsert into current second bucket
        bucket = float(int(now))  # 1-second granularity bucket
        conn.execute(
            """INSERT INTO rate_counters (rule_id, identifier, window_start, request_count, last_request)
               VALUES (?, ?, ?, 1, ?)
               ON CONFLICT(rule_id, identifier, window_start)
               DO UPDATE SET request_count = request_count + 1, last_request = excluded.last_request""",
            (rule_id, identifier, bucket, now)
        )
        conn.commit()

    return allowed, remaining, reset_at


# ---------------------------------------------------------------------------
# Background cleanup thread
# ---------------------------------------------------------------------------
def _cleanup_loop():
    while True:
        time.sleep(60)
        try:
            c = _conn()
            now = time.time()
            # delete counters older than 2x max window — keep DB small
            # fetch all rules to get their window_seconds
            rules = c.execute("SELECT id, window_seconds FROM rate_rules").fetchall()
            deleted = 0
            for r in rules:
                cutoff = now - r["window_seconds"] * 2
                cur = c.execute(
                    "DELETE FROM rate_counters WHERE rule_id=? AND window_start < ?",
                    (r["id"], cutoff)
                )
                deleted += cur.rowcount
            c.commit()
            c.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Default rule seed
# ---------------------------------------------------------------------------
DEFAULT_RULES = [
    ("api_global",      "api_key",   10000, 3600, 100,  "reject"),
    ("per_ip",          "ip",          500, 3600,  20,  "reject"),
    ("aiaas_chat",      "api_key",     100, 3600,   0,  "reject"),
    ("osint_scan",      "api_key",      20, 3600,   0,  "reject"),
    ("data_export",     "api_key",      10, 3600,   0,  "reject"),
    ("webhook_inbound", "source_ip",  1000, 3600,   0,  "reject"),
]

def _seed_defaults() -> int:
    c = _conn()
    now = time.time()
    seeded = 0
    for name, id_type, max_req, window, burst, action in DEFAULT_RULES:
        cur = c.execute(
            """INSERT OR IGNORE INTO rate_rules
               (name, identifier_type, max_requests, window_seconds, burst, action, enabled, created_at)
               VALUES (?,?,?,?,?,?,1,?)""",
            (name, id_type, max_req, window, burst, action, now)
        )
        seeded += cur.rowcount
    c.commit()
    c.close()
    return seeded


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------
def _check_auth(handler: BaseHTTPRequestHandler) -> bool:
    if not ADMIN_SECRET:
        return True
    return hmac.compare_digest(handler.headers.get("X-Admin-Secret", ""), ADMIN_SECRET)


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class RateLimiterHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass  # silence default access log

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def _send(self, code: int, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_err(self, code: int, msg: str):
        self._send(code, {"error": msg})

    # ------------------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")

        if path == "/health":
            self._send(200, {"status": "ok", "service": "fm-rate-limiter", "port": PORT})

        elif path == "/rules":
            self._handle_rules_list()

        elif path == "/analytics":
            self._handle_analytics()

        elif path == "/events":
            self._handle_events()

        elif path.startswith("/status/"):
            parts = path[len("/status/"):].split("/", 1)
            if len(parts) != 2:
                self._send_err(400, "Usage: /status/{rule}/{identifier}")
            else:
                self._handle_status(parts[0], parts[1])

        else:
            self._send_err(404, "not found")

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")

        if path == "/check":
            self._handle_check()

        elif path == "/check/batch":
            self._handle_check_batch()

        elif path == "/rules/create":
            if not _check_auth(self):
                self._send_err(401, "unauthorized")
                return
            self._handle_rules_create()

        elif path == "/rules/seed":
            self._handle_rules_seed()

        elif path.startswith("/reset/"):
            if not _check_auth(self):
                self._send_err(401, "unauthorized")
                return
            parts = path[len("/reset/"):].split("/", 1)
            if len(parts) != 2:
                self._send_err(400, "Usage: /reset/{rule}/{identifier}")
            else:
                self._handle_reset(parts[0], parts[1])

        else:
            self._send_err(404, "not found")

    def do_PUT(self):
        path = self.path.split("?")[0].rstrip("/")
        if path.startswith("/rules/"):
            if not _check_auth(self):
                self._send_err(401, "unauthorized")
                return
            try:
                rule_id = int(path[len("/rules/"):])
            except ValueError:
                self._send_err(400, "invalid rule id")
                return
            self._handle_rule_update(rule_id)
        else:
            self._send_err(404, "not found")

    def do_DELETE(self):
        path = self.path.split("?")[0].rstrip("/")
        if path.startswith("/rules/"):
            if not _check_auth(self):
                self._send_err(401, "unauthorized")
                return
            try:
                rule_id = int(path[len("/rules/"):])
            except ValueError:
                self._send_err(400, "invalid rule id")
                return
            self._handle_rule_delete(rule_id)
        else:
            self._send_err(404, "not found")

    # ------------------------------------------------------------------
    # Endpoint implementations
    # ------------------------------------------------------------------

    def _handle_check(self):
        body = self._read_body()
        rule_name  = body.get("rule")
        identifier = body.get("identifier")
        increment  = bool(body.get("increment", True))

        if not rule_name or not identifier:
            self._send_err(400, "rule and identifier are required")
            return

        c = _conn()
        try:
            row = c.execute(
                "SELECT * FROM rate_rules WHERE name=? AND enabled=1", (rule_name,)
            ).fetchone()
            if not row:
                self._send_err(404, f"rule '{rule_name}' not found or disabled")
                return

            allowed, remaining, reset_at = _check_rate(c, row, identifier, increment)

            if not allowed:
                retry_after = max(0, int(reset_at - time.time()))
                c.execute(
                    "INSERT INTO rate_events (rule_id, identifier, event_type, detail, created_at) VALUES (?,?,?,?,?)",
                    (row["id"], identifier, "rejected",
                     json.dumps({"rule": rule_name, "limit": row["max_requests"]}), time.time())
                )
                c.commit()
                if row["action"] == "throttle":
                    time.sleep(min(retry_after, 5))
                self._send(429, {
                    "allowed": False,
                    "retry_after": retry_after,
                    "limit": row["max_requests"]
                })
            else:
                self._send(200, {
                    "allowed": True,
                    "remaining": remaining,
                    "reset_at": int(reset_at)
                })
        finally:
            c.close()

    def _handle_check_batch(self):
        body = self._read_body()
        checks        = body.get("checks", [])
        stop_on_reject = bool(body.get("stop_on_reject", False))

        if not isinstance(checks, list):
            self._send_err(400, "checks must be an array")
            return

        results = []
        c = _conn()
        try:
            for item in checks:
                rule_name  = item.get("rule")
                identifier = item.get("identifier")
                increment  = bool(item.get("increment", True))

                if not rule_name or not identifier:
                    results.append({"error": "rule and identifier required"})
                    continue

                row = c.execute(
                    "SELECT * FROM rate_rules WHERE name=? AND enabled=1", (rule_name,)
                ).fetchone()

                if not row:
                    results.append({"rule": rule_name, "error": "not found"})
                    continue

                allowed, remaining, reset_at = _check_rate(c, row, identifier, increment)

                if not allowed:
                    retry_after = max(0, int(reset_at - time.time()))
                    c.execute(
                        "INSERT INTO rate_events (rule_id, identifier, event_type, detail, created_at) VALUES (?,?,?,?,?)",
                        (row["id"], identifier, "rejected",
                         json.dumps({"rule": rule_name}), time.time())
                    )
                    c.commit()
                    result = {
                        "rule": rule_name,
                        "identifier": identifier,
                        "allowed": False,
                        "retry_after": retry_after,
                        "limit": row["max_requests"]
                    }
                    results.append(result)
                    if stop_on_reject:
                        break
                else:
                    results.append({
                        "rule": rule_name,
                        "identifier": identifier,
                        "allowed": True,
                        "remaining": remaining,
                        "reset_at": int(reset_at)
                    })
        finally:
            c.close()

        self._send(200, {"results": results})

    def _handle_rules_list(self):
        c = _conn()
        try:
            rows = c.execute("SELECT * FROM rate_rules ORDER BY id").fetchall()
            rules = [dict(r) for r in rows]
        finally:
            c.close()
        self._send(200, {"rules": rules})

    def _handle_rules_create(self):
        body = self._read_body()
        required = ("name", "identifier_type", "max_requests", "window_seconds")
        for field in required:
            if field not in body:
                self._send_err(400, f"missing field: {field}")
                return

        name       = str(body["name"])
        id_type    = str(body["identifier_type"])
        max_req    = int(body["max_requests"])
        window     = int(body["window_seconds"])
        burst      = int(body.get("burst", 0))
        action     = str(body.get("action", "reject"))
        enabled    = int(body.get("enabled", 1))

        if action not in ("reject", "throttle", "log_only"):
            self._send_err(400, "action must be reject, throttle, or log_only")
            return

        c = _conn()
        try:
            cur = c.execute(
                """INSERT INTO rate_rules
                   (name, identifier_type, max_requests, window_seconds, burst, action, enabled, created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (name, id_type, max_req, window, burst, action, enabled, time.time())
            )
            c.commit()
            self._send(201, {"rule_id": cur.lastrowid})
        except sqlite3.IntegrityError:
            self._send_err(409, f"rule '{name}' already exists")
        finally:
            c.close()

    def _handle_rule_update(self, rule_id: int):
        body = self._read_body()
        if not body:
            self._send_err(400, "empty body")
            return

        allowed_fields = {
            "name", "identifier_type", "max_requests", "window_seconds",
            "burst", "action", "enabled"
        }
        updates = {k: v for k, v in body.items() if k in allowed_fields}
        if not updates:
            self._send_err(400, "no valid fields to update")
            return

        if "action" in updates and updates["action"] not in ("reject", "throttle", "log_only"):
            self._send_err(400, "action must be reject, throttle, or log_only")
            return

        c = _conn()
        try:
            row = c.execute("SELECT id FROM rate_rules WHERE id=?", (rule_id,)).fetchone()
            if not row:
                self._send_err(404, "rule not found")
                return
            set_clause = ", ".join(f"{k}=?" for k in updates)
            vals = list(updates.values()) + [rule_id]
            c.execute(f"UPDATE rate_rules SET {set_clause} WHERE id=?", vals)
            c.commit()
            self._send(200, {"updated": True, "rule_id": rule_id})
        finally:
            c.close()

    def _handle_rule_delete(self, rule_id: int):
        c = _conn()
        try:
            row = c.execute("SELECT id FROM rate_rules WHERE id=?", (rule_id,)).fetchone()
            if not row:
                self._send_err(404, "rule not found")
                return
            c.execute("DELETE FROM rate_rules WHERE id=?", (rule_id,))
            c.execute("DELETE FROM rate_counters WHERE rule_id=?", (rule_id,))
            c.commit()
            self._send(200, {"deleted": True, "rule_id": rule_id})
        finally:
            c.close()

    def _handle_rules_seed(self):
        seeded = _seed_defaults()
        self._send(200, {"seeded": seeded})

    def _handle_status(self, rule_name: str, identifier: str):
        c = _conn()
        try:
            row = c.execute(
                "SELECT * FROM rate_rules WHERE name=?", (rule_name,)
            ).fetchone()
            if not row:
                self._send_err(404, f"rule '{rule_name}' not found")
                return

            now    = time.time()
            window = row["window_seconds"]
            count  = _sliding_window_count(c, row["id"], identifier, window, now)
            max_req = row["max_requests"]
            remaining = max(0, max_req - count)

            oldest = c.execute(
                "SELECT MIN(window_start) FROM rate_counters WHERE rule_id=? AND identifier=?",
                (row["id"], identifier)
            ).fetchone()[0]
            reset_at = (oldest + window) if oldest else (now + window)

            self._send(200, {
                "rule":                 rule_name,
                "identifier":          identifier,
                "requests_this_window": count,
                "limit":               max_req,
                "remaining":           remaining,
                "window_resets_at":    int(reset_at)
            })
        finally:
            c.close()

    def _handle_reset(self, rule_name: str, identifier: str):
        c = _conn()
        try:
            row = c.execute(
                "SELECT id FROM rate_rules WHERE name=?", (rule_name,)
            ).fetchone()
            if not row:
                self._send_err(404, f"rule '{rule_name}' not found")
                return
            c.execute(
                "DELETE FROM rate_counters WHERE rule_id=? AND identifier=?",
                (row["id"], identifier)
            )
            c.commit()
            self._send(200, {"reset": True})
        finally:
            c.close()

    def _handle_analytics(self):
        c = _conn()
        try:
            now        = time.time()
            day_start  = now - 86400

            total = c.execute(
                "SELECT COALESCE(SUM(request_count),0) FROM rate_counters WHERE window_start >= ?",
                (day_start,)
            ).fetchone()[0]

            rejections = c.execute(
                "SELECT COUNT(*) FROM rate_events WHERE event_type='rejected' AND created_at >= ?",
                (day_start,)
            ).fetchone()[0]

            rejection_rate = round((rejections / max(total, 1)) * 100, 2)

            top_throttled = c.execute(
                """SELECT identifier, COUNT(*) as hits
                   FROM rate_events
                   WHERE event_type='rejected' AND created_at >= ?
                   GROUP BY identifier
                   ORDER BY hits DESC LIMIT 10""",
                (day_start,)
            ).fetchall()

            top_rules = c.execute(
                """SELECT r.name, COUNT(e.id) as hits
                   FROM rate_events e
                   JOIN rate_rules r ON r.id = e.rule_id
                   WHERE e.created_at >= ?
                   GROUP BY e.rule_id
                   ORDER BY hits DESC LIMIT 10""",
                (day_start,)
            ).fetchall()

            self._send(200, {
                "total_checks_today":       int(total),
                "rejections_today":         int(rejections),
                "rejection_rate_pct":       rejection_rate,
                "top_throttled_identifiers": [dict(r) for r in top_throttled],
                "most_triggered_rules":     [dict(r) for r in top_rules]
            })
        finally:
            c.close()

    def _handle_events(self):
        c = _conn()
        try:
            rows = c.execute(
                """SELECT e.id, r.name as rule, e.identifier, e.event_type,
                          e.detail, e.created_at
                   FROM rate_events e
                   LEFT JOIN rate_rules r ON r.id = e.rule_id
                   ORDER BY e.id DESC LIMIT 100"""
            ).fetchall()
            self._send(200, {"events": [dict(r) for r in rows]})
        finally:
            c.close()


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
def main():
    _db_init()
    _seed_defaults()

    cleanup = threading.Thread(target=_cleanup_loop, daemon=True, name="rl-cleanup")
    cleanup.start()

    server = HTTPServer(("0.0.0.0", PORT), RateLimiterHandler)
    print(f"[fm-rate-limiter] Listening on port {PORT}  (db={DB})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
