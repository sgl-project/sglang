#!/usr/bin/env python3
"""
FractalMesh OMEGA Titan — Tunnel/IP Manager Agent
Samuel James Hiotis | ABN 56628117363 | Albury NSW
Port 7843 | stdlib only | SQLite WAL | Cloudflare / ngrok / darktunnel
"""

import json
import os
import re
import signal
import socket
import sqlite3
import subprocess
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ---------------------------------------------------------------------------
# Config from environment (no hardcoded credentials)
# ---------------------------------------------------------------------------
TUNNEL_PORT = int(os.environ.get("TUNNEL_PORT", 7843))
TUNNEL_PROVIDER = os.environ.get("TUNNEL_PROVIDER", "cloudflare")
CLOUDFLARE_TOKEN = os.environ.get("CLOUDFLARE_TOKEN", "")
NGROK_TOKEN = os.environ.get("NGROK_TOKEN", "")

ROOT = os.environ.get("FRACTALMESH_HOME", str(Path.home() / "fmsaas"))
DB_PATH = os.path.join(ROOT, "database", "sovereign.db")

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _db():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS tunnels (
                id          INTEGER PRIMARY KEY,
                name        TEXT UNIQUE,
                provider    TEXT,
                local_port  INTEGER,
                public_url  TEXT,
                pid         INTEGER,
                status      TEXT DEFAULT 'stopped',
                started_at  REAL,
                stopped_at  REAL
            );
            CREATE TABLE IF NOT EXISTS tunnel_events (
                id          INTEGER PRIMARY KEY,
                tunnel_id   INTEGER,
                event_type  TEXT,
                detail      TEXT,
                created_at  REAL
            );
            CREATE TABLE IF NOT EXISTS ip_registry (
                id          INTEGER PRIMARY KEY,
                hostname    TEXT,
                local_ip    TEXT,
                public_ip   TEXT,
                recorded_at REAL
            );
        """)


# ---------------------------------------------------------------------------
# IP detection helpers
# ---------------------------------------------------------------------------

def _get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("1.1.1.1", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _get_public_ip() -> str:
    try:
        req = urllib.request.Request(
            "https://api.ipify.org?format=json",
            headers={"User-Agent": "FractalMesh-TunnelManager/1.0"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            return data.get("ip", "unknown")
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# URL extraction / PID helpers
# ---------------------------------------------------------------------------

_URL_PATTERN = re.compile(
    r"https://[a-z0-9\-]+\.(trycloudflare\.com|ngrok\.io|darktunnel\.net)[^\s]*"
)


def _detect_url_from_output(text: str) -> str:
    match = _URL_PATTERN.search(text)
    return match.group(0).rstrip("/") if match else ""


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, OSError):
        return False


# ---------------------------------------------------------------------------
# Tunnel command builder
# ---------------------------------------------------------------------------

_INSTALL_HINTS = {
    "cloudflare": "pkg install cloudflared",
    "ngrok": "pkg install ngrok",
    "darktunnel": "pip install darktunnel",
}

_BINARIES = {
    "cloudflare": "cloudflared",
    "ngrok": "ngrok",
    "darktunnel": "darktunnel",
}


def _which(binary: str) -> str:
    result = subprocess.run(["which", binary], capture_output=True, text=True)
    return result.stdout.strip()


def _build_command(provider: str, local_port: int) -> list:
    if provider == "cloudflare":
        cmd = ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{local_port}"]
        if CLOUDFLARE_TOKEN:
            cmd += ["--token", CLOUDFLARE_TOKEN]
        return cmd
    if provider == "ngrok":
        cmd = ["ngrok", "http", str(local_port)]
        if NGROK_TOKEN:
            cmd += ["--authtoken", NGROK_TOKEN]
        return cmd
    if provider == "darktunnel":
        return ["darktunnel", "http", str(local_port)]
    raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# Core tunnel operations
# ---------------------------------------------------------------------------

def _log_event(conn, tunnel_id: int, event_type: str, detail: str):
    conn.execute(
        "INSERT INTO tunnel_events (tunnel_id, event_type, detail, created_at) VALUES (?,?,?,?)",
        (tunnel_id, event_type, detail, time.time()),
    )


def start_tunnel(name: str, local_port: int, provider: str) -> dict:
    binary = _BINARIES.get(provider)
    if not binary:
        return {"status": "error", "error": f"Unknown provider: {provider}"}

    if not _which(binary):
        return {
            "status": "not_installed",
            "provider": provider,
            "install_cmd": _INSTALL_HINTS.get(provider, f"install {binary}"),
        }

    cmd = _build_command(provider, local_port)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    public_url = ""
    deadline = time.time() + 15
    output_buf = []

    while time.time() < deadline:
        if proc.poll() is not None:
            break
        try:
            line = proc.stdout.readline()
            if line:
                output_buf.append(line)
                url = _detect_url_from_output(line)
                if url:
                    public_url = url
                    break
        except Exception:
            break
        time.sleep(0.2)

    if not public_url:
        combined = "".join(output_buf)
        public_url = _detect_url_from_output(combined)

    now = time.time()
    with _db() as conn:
        conn.execute(
            """INSERT INTO tunnels (name, provider, local_port, public_url, pid, status, started_at)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(name) DO UPDATE SET
                 provider=excluded.provider, local_port=excluded.local_port,
                 public_url=excluded.public_url, pid=excluded.pid,
                 status=excluded.status, started_at=excluded.started_at,
                 stopped_at=NULL""",
            (name, provider, local_port, public_url, proc.pid, "running", now),
        )
        row = conn.execute("SELECT id FROM tunnels WHERE name=?", (name,)).fetchone()
        tunnel_id = row["id"]
        _log_event(conn, tunnel_id, "started", f"pid={proc.pid} url={public_url}")

    return {
        "tunnel_id": tunnel_id,
        "public_url": public_url,
        "provider": provider,
        "local_port": local_port,
        "status": "running",
        "pid": proc.pid,
    }


def stop_tunnel(tunnel_id: int) -> dict:
    with _db() as conn:
        row = conn.execute("SELECT * FROM tunnels WHERE id=?", (tunnel_id,)).fetchone()
        if not row:
            return {"error": "not_found"}
        pid = row["pid"]
        if pid and _is_pid_alive(pid):
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
        now = time.time()
        conn.execute(
            "UPDATE tunnels SET status='stopped', stopped_at=? WHERE id=?",
            (now, tunnel_id),
        )
        _log_event(conn, tunnel_id, "stopped", f"pid={pid}")
    return {"stopped": True, "tunnel_id": tunnel_id}


# ---------------------------------------------------------------------------
# Background threads
# ---------------------------------------------------------------------------

_last_local_ip = ""
_last_public_ip = ""
_ip_lock = threading.Lock()


def _ip_monitor():
    global _last_local_ip, _last_public_ip
    while True:
        try:
            local_ip = _get_local_ip()
            public_ip = _get_public_ip()
            hostname = socket.gethostname()
            with _ip_lock:
                changed = (local_ip != _last_local_ip) or (public_ip != _last_public_ip)
                if changed:
                    _last_local_ip = local_ip
                    _last_public_ip = public_ip
                    with _db() as conn:
                        conn.execute(
                            "INSERT INTO ip_registry (hostname, local_ip, public_ip, recorded_at) VALUES (?,?,?,?)",
                            (hostname, local_ip, public_ip, time.time()),
                        )
                    print(f"[IP-MONITOR] IP changed → local={local_ip} public={public_ip}")
        except Exception as e:
            print(f"[IP-MONITOR] error: {e}")
        time.sleep(300)


def _tunnel_monitor():
    while True:
        try:
            with _db() as conn:
                rows = conn.execute(
                    "SELECT id, pid, name FROM tunnels WHERE status='running'"
                ).fetchall()
                for row in rows:
                    pid = row["pid"]
                    if pid and not _is_pid_alive(pid):
                        conn.execute(
                            "UPDATE tunnels SET status='crashed' WHERE id=?", (row["id"],)
                        )
                        _log_event(conn, row["id"], "crashed", f"pid={pid} no longer alive")
                        print(f"[TUNNEL-MONITOR] Tunnel '{row['name']}' (pid={pid}) crashed")
        except Exception as e:
            print(f"[TUNNEL-MONITOR] error: {e}")
        time.sleep(30)


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------

def _json_response(handler, code: int, data):
    body = json.dumps(data, default=str).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class TunnelHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        print(f"[HTTP] {self.address_string()} {fmt % args}")

    # ------------------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")

        if path == "/health":
            _json_response(self, 200, {
                "status": "ok",
                "service": "fm-tunnel-manager",
                "port": TUNNEL_PORT,
                "provider": TUNNEL_PROVIDER,
            })

        elif path == "/ip":
            local_ip = _get_local_ip()
            public_ip = _get_public_ip()
            hostname = socket.gethostname()
            with _db() as conn:
                conn.execute(
                    "INSERT INTO ip_registry (hostname, local_ip, public_ip, recorded_at) VALUES (?,?,?,?)",
                    (hostname, local_ip, public_ip, time.time()),
                )
            _json_response(self, 200, {
                "local_ip": local_ip,
                "public_ip": public_ip,
                "hostname": hostname,
            })

        elif path == "/ip/history":
            with _db() as conn:
                rows = conn.execute(
                    "SELECT * FROM ip_registry ORDER BY recorded_at DESC LIMIT 50"
                ).fetchall()
            _json_response(self, 200, [dict(r) for r in rows])

        elif path == "/tunnels":
            with _db() as conn:
                rows = conn.execute("SELECT * FROM tunnels ORDER BY id").fetchall()
            _json_response(self, 200, [dict(r) for r in rows])

        elif re.match(r"^/tunnels/(\d+)/url$", path):
            tid = int(re.match(r"^/tunnels/(\d+)/url$", path).group(1))
            with _db() as conn:
                row = conn.execute("SELECT public_url, status FROM tunnels WHERE id=?", (tid,)).fetchone()
            if not row:
                _json_response(self, 404, {"error": "not_found"})
            else:
                _json_response(self, 200, {"public_url": row["public_url"], "status": row["status"]})

        elif re.match(r"^/tunnels/(\d+)/events$", path):
            tid = int(re.match(r"^/tunnels/(\d+)/events$", path).group(1))
            with _db() as conn:
                rows = conn.execute(
                    "SELECT * FROM tunnel_events WHERE tunnel_id=? ORDER BY created_at DESC",
                    (tid,),
                ).fetchall()
            _json_response(self, 200, [dict(r) for r in rows])

        elif path == "/providers":
            result = {}
            for name, binary in _BINARIES.items():
                binary_path = _which(binary)
                result[name] = {
                    "installed": bool(binary_path),
                    "binary": binary,
                    "path": binary_path or None,
                }
            _json_response(self, 200, result)

        elif path == "/analytics":
            with _db() as conn:
                total = conn.execute("SELECT COUNT(*) FROM tunnels").fetchone()[0]
                running = conn.execute("SELECT COUNT(*) FROM tunnels WHERE status='running'").fetchone()[0]
                stopped = conn.execute("SELECT COUNT(*) FROM tunnels WHERE status='stopped'").fetchone()[0]
                crashed = conn.execute("SELECT COUNT(*) FROM tunnels WHERE status='crashed'").fetchone()[0]
                rows = conn.execute(
                    "SELECT started_at, stopped_at, status FROM tunnels WHERE started_at IS NOT NULL"
                ).fetchall()
            now = time.time()
            total_uptime = 0.0
            for r in rows:
                start = r["started_at"] or 0
                end = r["stopped_at"] if r["stopped_at"] else (now if r["status"] == "running" else start)
                total_uptime += max(0, end - start)
            _json_response(self, 200, {
                "total_tunnels": total,
                "running": running,
                "stopped": stopped,
                "crashed": crashed,
                "total_uptime_seconds": round(total_uptime, 2),
            })

        else:
            _json_response(self, 404, {"error": "not_found", "path": path})

    # ------------------------------------------------------------------
    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        length = int(self.headers.get("Content-Length", 0))
        body = {}
        if length:
            try:
                body = json.loads(self.rfile.read(length).decode())
            except Exception:
                _json_response(self, 400, {"error": "invalid_json"})
                return

        if path == "/tunnels/start":
            name = body.get("name", f"tunnel-{int(time.time())}")
            local_port = body.get("local_port")
            provider = body.get("provider", TUNNEL_PROVIDER)
            if not local_port:
                _json_response(self, 400, {"error": "local_port required"})
                return
            result = start_tunnel(name, int(local_port), provider)
            code = 200 if result.get("status") != "error" else 500
            _json_response(self, code, result)

        elif re.match(r"^/tunnels/(\d+)/stop$", path):
            tid = int(re.match(r"^/tunnels/(\d+)/stop$", path).group(1))
            result = stop_tunnel(tid)
            code = 404 if "error" in result else 200
            _json_response(self, code, result)

        elif path == "/tunnels/start_all":
            ports = body.get("ports", [])
            provider = body.get("provider", TUNNEL_PROVIDER)
            results = []
            for port in ports:
                name = f"auto-{port}"
                res = start_tunnel(name, int(port), provider)
                results.append(res)
            _json_response(self, 200, {"results": results})

        elif path == "/tunnels/expose_dashboard":
            dashboard_ports = [
                ("web-ide", 7777),
                ("admin-dashboard", 7833),
                ("mcp-router", 7785),
            ]
            tunnels = []
            for tname, tport in dashboard_ports:
                res = start_tunnel(tname, tport, TUNNEL_PROVIDER)
                tunnels.append(res)
            _json_response(self, 200, {"tunnels": tunnels})

        else:
            _json_response(self, 404, {"error": "not_found", "path": path})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    init_db()

    # Seed initial IP record
    try:
        local_ip = _get_local_ip()
        public_ip = _get_public_ip()
        hostname = socket.gethostname()
        global _last_local_ip, _last_public_ip
        _last_local_ip = local_ip
        _last_public_ip = public_ip
        with _db() as conn:
            conn.execute(
                "INSERT INTO ip_registry (hostname, local_ip, public_ip, recorded_at) VALUES (?,?,?,?)",
                (hostname, local_ip, public_ip, time.time()),
            )
        print(f"[STARTUP] local_ip={local_ip} public_ip={public_ip}")
    except Exception as e:
        print(f"[STARTUP] IP seed failed: {e}")

    # Start background threads
    t1 = threading.Thread(target=_ip_monitor, daemon=True, name="ip-monitor")
    t1.start()

    t2 = threading.Thread(target=_tunnel_monitor, daemon=True, name="tunnel-monitor")
    t2.start()

    # Start HTTP server
    server = HTTPServer(("0.0.0.0", TUNNEL_PORT), TunnelHandler)
    print(f"[TUNNEL-MANAGER] Listening on port {TUNNEL_PORT} | provider={TUNNEL_PROVIDER}")

    def _shutdown(signum, frame):
        print("[TUNNEL-MANAGER] Shutting down...")
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    server.serve_forever()


if __name__ == "__main__":
    main()
