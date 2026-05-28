#!/usr/bin/env python3
"""
fm_audit_log.py — Immutable Audit & Compliance Log (Port 7903)
FractalMesh OMEGA Titan | Samuel James Hiotis | ABN 56 628 117 363
Credentials sourced from ~/.secrets/fractal.env — never hardcoded.
"""
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# ── vault ─────────────────────────────────────────────────────────────────────
_ENV_FILE = Path.home() / ".secrets" / "fractal.env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

PORT         = int(os.getenv("AUDIT_LOG_PORT", "7903"))
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")
CHAIN_SECRET = os.getenv("AUDIT_CHAIN_SECRET", "fm_audit_chain_v1")

ROOT = Path(os.getenv("FRACTALMESH_HOME", str(Path.home() / "fmsaas")))
DB   = ROOT / "database" / "sovereign.db"
ROOT.mkdir(parents=True, exist_ok=True)
DB.parent.mkdir(parents=True, exist_ok=True)

_chain_lock = threading.Lock()

def _db():
    con = sqlite3.connect(str(DB), check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")
    con.row_factory = sqlite3.Row
    return con

def init_db():
    con = _db()
    con.executescript("""
        CREATE TABLE IF NOT EXISTS audit_entries (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_id     TEXT UNIQUE NOT NULL,
            sequence_num INTEGER NOT NULL,
            actor_id     TEXT NOT NULL,
            actor_type   TEXT NOT NULL DEFAULT 'user',
            action       TEXT NOT NULL,
            resource     TEXT NOT NULL,
            resource_id  TEXT,
            outcome      TEXT NOT NULL DEFAULT 'success',
            ip_hash      TEXT,
            details      TEXT NOT NULL DEFAULT '{}',
            prev_hash    TEXT NOT NULL,
            entry_hash   TEXT UNIQUE NOT NULL,
            created_at   REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS audit_alerts (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_id     TEXT UNIQUE NOT NULL,
            pattern      TEXT NOT NULL,
            description  TEXT NOT NULL,
            threshold    INTEGER NOT NULL DEFAULT 5,
            window_secs  INTEGER NOT NULL DEFAULT 300,
            active       INTEGER NOT NULL DEFAULT 1,
            created_at   REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS audit_reports (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id    TEXT UNIQUE NOT NULL,
            report_type  TEXT NOT NULL,
            period_start REAL NOT NULL,
            period_end   REAL NOT NULL,
            generated_at REAL NOT NULL,
            summary      TEXT NOT NULL DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_audit_actor    ON audit_entries(actor_id);
        CREATE INDEX IF NOT EXISTS idx_audit_action   ON audit_entries(action);
        CREATE INDEX IF NOT EXISTS idx_audit_created  ON audit_entries(created_at);
        CREATE INDEX IF NOT EXISTS idx_audit_resource ON audit_entries(resource);
    """)
    con.commit()
    _seed_alerts(con)
    con.close()

def _seed_alerts(con):
    if con.execute("SELECT COUNT(*) FROM audit_alerts").fetchone()[0] > 0:
        return
    now = time.time()
    alerts = [
        ("al_failed_logins", "login.failed",     "Multiple failed logins",         5,  300),
        ("al_data_export",   "data.export",       "Bulk data export",               3, 3600),
        ("al_admin_changes", "admin.permission",  "Permission changes",             3, 3600),
        ("al_delete_bulk",   "resource.delete",   "Bulk deletions",                 10, 600),
    ]
    for aid, pattern, desc, threshold, window in alerts:
        con.execute(
            "INSERT INTO audit_alerts(alert_id,pattern,description,threshold,window_secs,created_at) VALUES(?,?,?,?,?,?)",
            (aid, pattern, desc, threshold, window, now)
        )
    con.commit()

def _get_prev_hash(con):
    row = con.execute("SELECT entry_hash FROM audit_entries ORDER BY sequence_num DESC LIMIT 1").fetchone()
    return row["entry_hash"] if row else "GENESIS"

def _compute_entry_hash(entry_id, sequence_num, actor_id, action, resource, prev_hash, created_at):
    payload = f"{entry_id}|{sequence_num}|{actor_id}|{action}|{resource}|{prev_hash}|{created_at}"
    return hmac.new(CHAIN_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()

def _get_next_seq(con):
    row = con.execute("SELECT MAX(sequence_num) FROM audit_entries").fetchone()
    return (row[0] or 0) + 1

def _log_entry(con, actor_id, actor_type, action, resource, resource_id=None,
               outcome="success", ip_hash=None, details=None):
    with _chain_lock:
        entry_id = "ae_" + secrets.token_hex(8)
        now = time.time()
        seq = _get_next_seq(con)
        prev_hash = _get_prev_hash(con)
        entry_hash = _compute_entry_hash(entry_id, seq, actor_id, action, resource, prev_hash, now)
        con.execute(
            "INSERT INTO audit_entries(entry_id,sequence_num,actor_id,actor_type,action,resource,"
            "resource_id,outcome,ip_hash,details,prev_hash,entry_hash,created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (entry_id, seq, actor_id, actor_type, action, resource, resource_id,
             outcome, ip_hash, json.dumps(details or {}), prev_hash, entry_hash, now)
        )
        con.commit()
        return entry_id, entry_hash

def _verify_chain(con):
    rows = con.execute("SELECT * FROM audit_entries ORDER BY sequence_num").fetchall()
    if not rows:
        return True, 0
    prev = "GENESIS"
    for row in rows:
        expected = _compute_entry_hash(
            row["entry_id"], row["sequence_num"], row["actor_id"],
            row["action"], row["resource"], row["prev_hash"], row["created_at"]
        )
        if expected != row["entry_hash"]:
            return False, row["sequence_num"]
        if row["prev_hash"] != prev:
            return False, row["sequence_num"]
        prev = row["entry_hash"]
    return True, len(rows)

def _j(data, status=200):
    return status, json.dumps(data, default=str).encode()

def _err(msg, code=400):
    return _j({"error": msg}, code)

def _admin(h):
    v = h.get("X-Admin-Secret", "")
    return not ADMIN_SECRET or hmac.compare_digest(v, ADMIN_SECRET)

class AuditHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _read_body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n)) if n else {}

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
        qs = parse_qs(parsed.query)
        try:
            code, body = self._get(p, qs)
        except Exception as e:
            code, body = _err(str(e), 500)
        self._send(code, body)

    def do_POST(self):
        p = self.path.strip("/").split("/")
        try:
            data = self._read_body()
            code, body = self._post(p, data)
        except Exception as e:
            code, body = _err(str(e), 500)
        self._send(code, body)

    def _get(self, p, qs):
        con = _db()
        try:
            if p == ["health"]:
                total = con.execute("SELECT COUNT(*) FROM audit_entries").fetchone()[0]
                return _j({"status": "ok", "port": PORT, "agent": "fm_audit_log", "total_entries": total})

            if p == ["entries"]:
                if not _admin(self.headers):
                    return _err("Unauthorized", 403)
                actor = qs.get("actor_id", [None])[0]
                action = qs.get("action", [None])[0]
                resource = qs.get("resource", [None])[0]
                limit = int(qs.get("limit", ["100"])[0])
                since = float(qs.get("since", ["0"])[0])
                q = "SELECT * FROM audit_entries WHERE created_at >= ?"
                vals = [since]
                if actor:
                    q += " AND actor_id=?"; vals.append(actor)
                if action:
                    q += " AND action=?"; vals.append(action)
                if resource:
                    q += " AND resource=?"; vals.append(resource)
                q += " ORDER BY sequence_num DESC LIMIT ?"
                vals.append(limit)
                rows = con.execute(q, vals).fetchall()
                return _j([dict(r) for r in rows])

            if len(p) == 2 and p[0] == "entries":
                if not _admin(self.headers):
                    return _err("Unauthorized", 403)
                row = con.execute("SELECT * FROM audit_entries WHERE entry_id=?", (p[1],)).fetchone()
                if not row:
                    return _err("Entry not found", 404)
                return _j(dict(row))

            if p == ["verify"]:
                if not _admin(self.headers):
                    return _err("Unauthorized", 403)
                valid, count_or_seq = _verify_chain(con)
                if valid:
                    return _j({"valid": True, "entries_verified": count_or_seq})
                return _j({"valid": False, "tampered_at_sequence": count_or_seq}, 409)

            if p == ["alerts"]:
                if not _admin(self.headers):
                    return _err("Unauthorized", 403)
                rows = con.execute("SELECT * FROM audit_alerts WHERE active=1").fetchall()
                return _j([dict(r) for r in rows])

            if p == ["reports"]:
                if not _admin(self.headers):
                    return _err("Unauthorized", 403)
                rows = con.execute("SELECT * FROM audit_reports ORDER BY generated_at DESC LIMIT 20").fetchall()
                return _j([dict(r) for r in rows])

            if p == ["summary"]:
                if not _admin(self.headers):
                    return _err("Unauthorized", 403)
                now = time.time()
                since = now - 86400
                by_action = con.execute(
                    "SELECT action, COUNT(*) as cnt FROM audit_entries WHERE created_at >= ? GROUP BY action ORDER BY cnt DESC LIMIT 10",
                    (since,)
                ).fetchall()
                by_actor = con.execute(
                    "SELECT actor_id, COUNT(*) as cnt FROM audit_entries WHERE created_at >= ? GROUP BY actor_id ORDER BY cnt DESC LIMIT 10",
                    (since,)
                ).fetchall()
                failures = con.execute(
                    "SELECT COUNT(*) FROM audit_entries WHERE outcome='failure' AND created_at >= ?", (since,)
                ).fetchone()[0]
                total = con.execute("SELECT COUNT(*) FROM audit_entries").fetchone()[0]
                return _j({
                    "total_entries": total,
                    "entries_24h": con.execute("SELECT COUNT(*) FROM audit_entries WHERE created_at >= ?", (since,)).fetchone()[0],
                    "failures_24h": failures,
                    "top_actions": [{"action": r["action"], "count": r["cnt"]} for r in by_action],
                    "top_actors": [{"actor_id": r["actor_id"], "count": r["cnt"]} for r in by_actor],
                })

            return _err("Not found", 404)
        finally:
            con.close()

    def _post(self, p, data):
        con = _db()
        try:
            if p == ["log"]:
                actor_id = data.get("actor_id", "system")
                action = data.get("action", "")
                resource = data.get("resource", "")
                if not action:
                    return _err("action required")
                ip = self.client_address[0]
                ip_hash = hashlib.sha256(ip.encode()).hexdigest()[:16]
                entry_id, entry_hash = _log_entry(
                    con, actor_id, data.get("actor_type","user"),
                    action, resource, data.get("resource_id"),
                    data.get("outcome","success"), ip_hash, data.get("details",{})
                )
                return _j({"entry_id": entry_id, "entry_hash": entry_hash[:16]}, 201)

            if p == ["bulk"]:
                entries = data.get("entries", [])
                results = []
                for e in entries[:100]:
                    ip_hash = hashlib.sha256((e.get("ip","")).encode()).hexdigest()[:16]
                    eid, ehash = _log_entry(
                        con, e.get("actor_id","system"), e.get("actor_type","system"),
                        e.get("action",""), e.get("resource",""),
                        e.get("resource_id"), e.get("outcome","success"),
                        ip_hash, e.get("details",{})
                    )
                    results.append({"entry_id": eid, "hash": ehash[:16]})
                return _j({"logged": len(results), "entries": results}, 201)

            if p == ["reports", "generate"]:
                if not _admin(self.headers):
                    return _err("Unauthorized", 403)
                now = time.time()
                period_end = now
                period_start = now - float(data.get("period_days", 30)) * 86400
                con2 = _db()
                counts = con2.execute(
                    "SELECT action, COUNT(*) as cnt FROM audit_entries WHERE created_at BETWEEN ? AND ? GROUP BY action",
                    (period_start, period_end)
                ).fetchall()
                failures = con2.execute(
                    "SELECT COUNT(*) FROM audit_entries WHERE outcome='failure' AND created_at BETWEEN ? AND ?",
                    (period_start, period_end)
                ).fetchone()[0]
                summary = {
                    "by_action": {r["action"]: r["cnt"] for r in counts},
                    "failures": failures,
                    "period_days": data.get("period_days", 30),
                }
                rid = "rpt_" + secrets.token_hex(8)
                con2.execute(
                    "INSERT INTO audit_reports(report_id,report_type,period_start,period_end,generated_at,summary) VALUES(?,?,?,?,?,?)",
                    (rid, data.get("report_type","compliance"), period_start, period_end, now, json.dumps(summary))
                )
                con2.commit()
                con2.close()
                return _j({"report_id": rid, "summary": summary}, 201)

            if p == ["alerts"]:
                if not _admin(self.headers):
                    return _err("Unauthorized", 403)
                aid = "al_" + secrets.token_hex(8)
                con.execute(
                    "INSERT INTO audit_alerts(alert_id,pattern,description,threshold,window_secs,created_at) VALUES(?,?,?,?,?,?)",
                    (aid, data.get("pattern",""), data.get("description",""),
                     data.get("threshold",5), data.get("window_secs",300), time.time())
                )
                con.commit()
                return _j({"alert_id": aid}, 201)

            return _err("Not found", 404)
        finally:
            con.close()


def run():
    init_db()
    server = HTTPServer(("0.0.0.0", PORT), AuditHandler)
    print(f"[fm_audit_log] listening on port {PORT}")
    server.serve_forever()


if __name__ == "__main__":
    run()
