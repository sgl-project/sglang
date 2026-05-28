#!/usr/bin/env python3
"""
fm_lba_firewall.py — Licensed Breach Alert Firewall (Port 7906)
IP reputation, rate-gate, and threat-signal enforcement layer.
Credentials from ~/.secrets/fractal.env — never hardcoded.
Samuel James Hiotis | ABN 56 628 117 363
"""
import hashlib
import hmac
import json
import os
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# ── vault ─────────────────────────────────────────────────────────────────────
_VAULT = Path.home() / ".secrets" / "fractal.env"
if _VAULT.exists():
    for _ln in _VAULT.read_text().splitlines():
        if "=" in _ln and not _ln.startswith("#"):
            _k, _, _v = _ln.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

PORT         = int(os.getenv("LBA_FIREWALL_PORT", "7906"))
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")
ROOT         = Path(os.getenv("FRACTALMESH_HOME", str(Path.home() / "fmsaas")))
DB           = ROOT / "database" / "sovereign.db"
ROOT.mkdir(parents=True, exist_ok=True)
DB.parent.mkdir(parents=True, exist_ok=True)

# banned patterns — any payload containing these is blocked
BANNED_PATTERNS = [
    "sk_live_", "sk-ant-api", "ETH_PRIVATE_KEY", "PRIVATE_KEY=",
    "password=", "secret=",
]

# ── DB ─────────────────────────────────────────────────────────────────────────
def _init_db():
    con = sqlite3.connect(str(DB), timeout=10)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript("""
        CREATE TABLE IF NOT EXISTS lba_blocks (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            ip         TEXT NOT NULL,
            reason     TEXT NOT NULL,
            blocked_at REAL NOT NULL,
            expires_at REAL,
            active     INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS lba_events (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            ip         TEXT NOT NULL,
            event_type TEXT NOT NULL,
            detail     TEXT,
            ts         REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_lba_blocks_ip ON lba_blocks(ip, active);
        CREATE INDEX IF NOT EXISTS idx_lba_events_ip ON lba_events(ip);
    """)
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

def _lba_check(payload: str) -> bool:
    low = payload.lower()
    return not any(p.lower() in low for p in BANNED_PATTERNS)

def _is_blocked(ip: str) -> bool:
    try:
        con = _db()
        now = time.time()
        row = con.execute(
            "SELECT id FROM lba_blocks WHERE ip=? AND active=1 AND (expires_at IS NULL OR expires_at > ?)",
            (ip, now)
        ).fetchone()
        con.close()
        return row is not None
    except Exception:
        return False

def _block_ip(ip: str, reason: str, duration_s: int = 0):
    try:
        con = _db()
        expires = time.time() + duration_s if duration_s else None
        con.execute(
            "INSERT INTO lba_blocks(ip,reason,blocked_at,expires_at) VALUES(?,?,?,?)",
            (ip, reason, time.time(), expires)
        )
        con.commit()
        con.close()
    except Exception:
        pass

def _log_event(ip: str, event_type: str, detail: str = ""):
    try:
        con = _db()
        con.execute(
            "INSERT INTO lba_events(ip,event_type,detail,ts) VALUES(?,?,?,?)",
            (ip, event_type, detail[:500], time.time())
        )
        con.commit()
        con.close()
    except Exception:
        pass

def _expiry_daemon():
    while True:
        time.sleep(3600)
        try:
            con = _db()
            con.execute("UPDATE lba_blocks SET active=0 WHERE expires_at IS NOT NULL AND expires_at < ?", (time.time(),))
            con.commit()
            con.close()
        except Exception:
            pass

threading.Thread(target=_expiry_daemon, daemon=True).start()

# ── HTTP handler ───────────────────────────────────────────────────────────────
def _j(data, code=200):
    return code, json.dumps(data, default=str).encode()

def _err(msg, code=400):
    return _j({"error": msg}, code)

class LBAHandler(BaseHTTPRequestHandler):
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
        self.send_header("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS")
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
            if p == ["health"] or p == [""]:
                total = con.execute("SELECT COUNT(*) FROM lba_blocks WHERE active=1").fetchone()[0]
                return _j({"status": "ok", "active_blocks": total, "port": PORT})

            if p == ["blocks"]:
                if not _admin(self.headers):
                    return _err("Unauthorized", 403)
                ip_filter = qs.get("ip")
                if ip_filter:
                    rows = con.execute(
                        "SELECT * FROM lba_blocks WHERE ip=? ORDER BY blocked_at DESC", (ip_filter,)
                    ).fetchall()
                else:
                    rows = con.execute(
                        "SELECT * FROM lba_blocks WHERE active=1 ORDER BY blocked_at DESC LIMIT 200"
                    ).fetchall()
                return _j([dict(r) for r in rows])

            if p == ["events"]:
                if not _admin(self.headers):
                    return _err("Unauthorized", 403)
                ip_filter = qs.get("ip")
                limit = int(qs.get("limit", 100))
                if ip_filter:
                    rows = con.execute(
                        "SELECT * FROM lba_events WHERE ip=? ORDER BY ts DESC LIMIT ?",
                        (ip_filter, limit)
                    ).fetchall()
                else:
                    rows = con.execute(
                        "SELECT * FROM lba_events ORDER BY ts DESC LIMIT ?", (limit,)
                    ).fetchall()
                return _j([dict(r) for r in rows])

            if p[0] == "check" and len(p) == 2:
                ip = p[1]
                blocked = _is_blocked(ip)
                _log_event(ip, "check")
                return _j({"ip": ip, "blocked": blocked})

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
        if p == ["check"]:
            ip      = data.get("ip", "")
            payload = data.get("payload", "")
            if not ip:
                return _err("ip required")
            blocked = _is_blocked(ip)
            lba_ok  = _lba_check(str(payload))
            if not lba_ok:
                _block_ip(ip, "lba_pattern_match", duration_s=3600)
                _log_event(ip, "lba_block", "banned pattern in payload")
                return _j({"ip": ip, "blocked": True, "reason": "lba_pattern_match"})
            return _j({"ip": ip, "blocked": blocked, "lba_clean": True})

        if p == ["block"]:
            if not _admin(self.headers):
                return _err("Unauthorized", 403)
            ip       = data.get("ip", "")
            reason   = data.get("reason", "manual")
            duration = int(data.get("duration_s", 0))
            if not ip:
                return _err("ip required")
            _block_ip(ip, reason, duration)
            _log_event(ip, "manual_block", reason)
            return _j({"blocked": True, "ip": ip, "reason": reason})

        if p == ["unblock"]:
            if not _admin(self.headers):
                return _err("Unauthorized", 403)
            ip = data.get("ip", "")
            if not ip:
                return _err("ip required")
            con = _db()
            con.execute("UPDATE lba_blocks SET active=0 WHERE ip=?", (ip,))
            con.commit()
            con.close()
            _log_event(ip, "unblock")
            return _j({"unblocked": True, "ip": ip})

        if p == ["report"]:
            ip       = data.get("ip", "")
            event    = data.get("event_type", "report")
            detail   = data.get("detail", "")
            if not ip:
                return _err("ip required")
            _log_event(ip, event, detail)
            return _j({"logged": True, "ip": ip, "event_type": event})

        return _err("not found", 404)

    def do_DELETE(self):
        p = self.path.strip("/").split("/")
        if not _admin(self.headers):
            body = json.dumps({"error": "Unauthorized"}).encode()
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("WWW-Authenticate", 'X-Admin-Secret realm="FractalMesh"')
            self.end_headers()
            self.wfile.write(body)
            return
        try:
            if p[0] == "blocks" and len(p) == 2:
                block_id = int(p[1])
                con = _db()
                con.execute("UPDATE lba_blocks SET active=0 WHERE id=?", (block_id,))
                con.commit()
                con.close()
                code, body = _j({"deleted": True, "id": block_id})
            else:
                code, body = _err("not found", 404)
        except Exception as e:
            code, body = _err(str(e), 500)
        self._send(code, body)


def run():
    _init_db()
    server = HTTPServer(("0.0.0.0", PORT), LBAHandler)
    print(f"[fm_lba_firewall] listening on port {PORT}")
    server.serve_forever()


if __name__ == "__main__":
    run()
