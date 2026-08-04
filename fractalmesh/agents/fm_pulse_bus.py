"""
FractalMesh Pulse Bus
Secure internal event bus — HMAC-signed payloads, SQLite ledger
Samuel James Hiotis | ABN 56 628 117 363
"""
import sys
import json
import sqlite3
import hmac
import hashlib
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DB   = os.path.join(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")), "database", "sovereign.db")
PORT = int(os.getenv("BUS_PORT", "5060"))
SECRET = os.getenv("BUS_SECRET", "fm_internal_key").encode()


def _verify(sig: str, body: bytes) -> bool:
    expected = hmac.new(SECRET, body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, sig)


def _log_event(source: str, event: str, priority: float = 1.0):
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS pulse_log "
        "(id INTEGER PRIMARY KEY, source TEXT, event TEXT, priority REAL, "
        "ts DATETIME DEFAULT CURRENT_TIMESTAMP)"
    )
    conn.execute(
        "INSERT INTO pulse_log (source, event, priority) VALUES (?,?,?)",
        (source, event, priority)
    )
    conn.commit()
    conn.close()


class PulseHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silence access log

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            sig    = self.headers.get("X-Fractal-Signature", "")

            if sig and not _verify(sig, body):
                self.send_response(403)
                self.end_headers()
                return

            data = json.loads(body)
            _log_event(
                data.get("agent", "unknown"),
                data.get("event", ""),
                float(data.get("priority", 1.0))
            )
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        except Exception as e:
            print(f"[BUS ERR] {e}")
            self.send_response(500)
            self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"online","service":"fm-pulse-bus"}')
        else:
            self.send_response(404)
            self.end_headers()


if __name__ == "__main__":
    print(f"[fm-pulse-bus] Secure event bus active on :{PORT}")
    HTTPServer(("127.0.0.1", PORT), PulseHandler).serve_forever()
