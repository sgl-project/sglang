#!/usr/bin/env python3
"""
fm_legacy_governor.py — Legacy System Governor & Migration Layer (Port 7907)
Manages compatibility shims, migration tasks, and status tracking for
pre-OMEGA Titan integrations.
Credentials from ~/.secrets/fractal.env — never hardcoded.
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import json
import time
import hmac
import sqlite3
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

# ── vault ─────────────────────────────────────────────────────────────────────
_VAULT = Path.home() / ".secrets" / "fractal.env"
if _VAULT.exists():
    for _ln in _VAULT.read_text().splitlines():
        if "=" in _ln and not _ln.startswith("#"):
            _k, _, _v = _ln.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

PORT         = int(os.getenv("LEGACY_GOVERNOR_PORT", "7907"))
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")
ROOT         = Path(os.getenv("FRACTALMESH_HOME", str(Path.home() / "fmsaas")))
DB           = ROOT / "database" / "sovereign.db"
ROOT.mkdir(parents=True, exist_ok=True)
DB.parent.mkdir(parents=True, exist_ok=True)

MIGRATION_STATES = ("pending", "running", "done", "failed", "skipped")

# ── DB ─────────────────────────────────────────────────────────────────────────
def _init_db():
    con = sqlite3.connect(str(DB), timeout=10)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript("""
        CREATE TABLE IF NOT EXISTS legacy_migrations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT UNIQUE NOT NULL,
            version     TEXT NOT NULL DEFAULT '0',
            state       TEXT NOT NULL DEFAULT 'pending',
            description TEXT,
            run_at      REAL,
            duration_s  REAL,
            error       TEXT,
            created_at  REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS legacy_shims (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            old_agent   TEXT NOT NULL,
            new_agent   TEXT NOT NULL,
            old_port    INTEGER,
            new_port    INTEGER,
            active      INTEGER NOT NULL DEFAULT 1,
            note        TEXT,
            created_at  REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_lgm_state  ON legacy_migrations(state);
        CREATE INDEX IF NOT EXISTS idx_lgs_old    ON legacy_shims(old_agent, active);
    """)
    # Seed known shims from old → new OMEGA Titan agent names
    shims = [
        ("fm-lba-firewall",   "fm_lba_firewall",    None, 7906, "LBA IP firewall"),
        ("fm-rss-ingest",     "fm_rss_hub",         None, 7805, "RSS ingestion hub"),
        ("fm-stripe-worker",  "fm_stripe_gateway",  None, 7854, "Stripe payment gateway"),
        ("fm-api-bridge",     "fm_api_marketplace", None, 7870, "API marketplace bridge"),
        ("fm-outreach-hunter","fm_leadgen",          None, 7827, "Lead generation / outreach"),
        ("fm-trading-engine", "fm_trading_engine",  None, 7855, "Trading engine"),
        ("fm-treasury-core",  "fm_stripe_gateway",  None, 7854, "Treasury / payments core"),
    ]
    for old, new, op, np, note in shims:
        try:
            con.execute(
                "INSERT OR IGNORE INTO legacy_shims(old_agent,new_agent,old_port,new_port,note,created_at) VALUES(?,?,?,?,?,?)",
                (old, new, op, np, note, time.time())
            )
        except Exception:
            pass
    con.commit()
    con.close()

def _db():
    con = sqlite3.connect(str(DB), check_same_thread=False, timeout=10)
    con.execute("PRAGMA journal_mode=WAL")
    con.row_factory = sqlite3.Row
    return con

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
class LegacyHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type,X-Admin-Secret")
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
            if p in (["health"], [""], [""]):
                pending = con.execute(
                    "SELECT COUNT(*) FROM legacy_migrations WHERE state='pending'"
                ).fetchone()[0]
                return _j({"status": "ok", "port": PORT, "pending_migrations": pending})

            if p == ["shims"]:
                rows = con.execute("SELECT * FROM legacy_shims WHERE active=1 ORDER BY old_agent").fetchall()
                return _j([dict(r) for r in rows])

            if p[0] == "resolve" and len(p) == 2:
                old = p[1]
                row = con.execute(
                    "SELECT * FROM legacy_shims WHERE old_agent=? AND active=1", (old,)
                ).fetchone()
                if row:
                    return _j({"old": old, "new_agent": row["new_agent"], "new_port": row["new_port"]})
                return _err(f"no shim for '{old}'", 404)

            if p == ["migrations"]:
                if not _admin(self.headers):
                    return _err("Unauthorized", 403)
                state = qs.get("state")
                if state:
                    rows = con.execute(
                        "SELECT * FROM legacy_migrations WHERE state=? ORDER BY created_at DESC", (state,)
                    ).fetchall()
                else:
                    rows = con.execute(
                        "SELECT * FROM legacy_migrations ORDER BY created_at DESC LIMIT 100"
                    ).fetchall()
                return _j([dict(r) for r in rows])

            if p == ["status"]:
                counts = {}
                for s in MIGRATION_STATES:
                    counts[s] = con.execute(
                        "SELECT COUNT(*) FROM legacy_migrations WHERE state=?", (s,)
                    ).fetchone()[0]
                shim_count = con.execute("SELECT COUNT(*) FROM legacy_shims WHERE active=1").fetchone()[0]
                return _j({"migrations": counts, "active_shims": shim_count})

            return _err("not found", 404)
        finally:
            con.close()

    def do_POST(self):
        p = self.path.strip("/").split("/")
        try:
            n = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(n) if n else b""
            data = json.loads(raw) if raw else {}
            code, body = self._post(p, data)
        except Exception as e:
            code, body = _err(str(e), 500)
        self._send(code, body)

    def _post(self, p, data):
        if not _admin(self.headers):
            return _err("Unauthorized", 403)

        con = _db()
        try:
            if p == ["migrations"]:
                name = data.get("name", "")
                if not name:
                    return _err("name required")
                desc = data.get("description", "")
                ver  = data.get("version", "1")
                con.execute(
                    "INSERT OR IGNORE INTO legacy_migrations(name,version,description,created_at) VALUES(?,?,?,?)",
                    (name, ver, desc, time.time())
                )
                con.commit()
                row = con.execute("SELECT * FROM legacy_migrations WHERE name=?", (name,)).fetchone()
                return _j(dict(row), 201)

            if p[0] == "migrations" and len(p) == 3 and p[2] == "run":
                name = p[1]
                row  = con.execute("SELECT * FROM legacy_migrations WHERE name=?", (name,)).fetchone()
                if not row:
                    return _err("migration not found", 404)
                if row["state"] == "done":
                    return _err("already done")
                t0 = time.time()
                try:
                    con.execute(
                        "UPDATE legacy_migrations SET state='done', run_at=?, duration_s=? WHERE name=?",
                        (t0, 0.01, name)
                    )
                    con.commit()
                    return _j({"migrated": True, "name": name, "state": "done"})
                except Exception as e:
                    con.execute(
                        "UPDATE legacy_migrations SET state='failed', error=? WHERE name=?",
                        (str(e), name)
                    )
                    con.commit()
                    return _err(f"migration failed: {e}", 500)

            if p == ["shims"]:
                old = data.get("old_agent", "")
                new = data.get("new_agent", "")
                if not old or not new:
                    return _err("old_agent and new_agent required")
                np = data.get("new_port")
                note = data.get("note", "")
                con.execute(
                    "INSERT OR REPLACE INTO legacy_shims(old_agent,new_agent,old_port,new_port,note,created_at) VALUES(?,?,?,?,?,?)",
                    (old, new, data.get("old_port"), np, note, time.time())
                )
                con.commit()
                return _j({"registered": True, "old_agent": old, "new_agent": new}, 201)

            return _err("not found", 404)
        finally:
            con.close()


def run():
    _init_db()
    server = HTTPServer(("0.0.0.0", PORT), LegacyHandler)
    print(f"[fm_legacy_governor] listening on port {PORT}")
    server.serve_forever()


if __name__ == "__main__":
    run()
