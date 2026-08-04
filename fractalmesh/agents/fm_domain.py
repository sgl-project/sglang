"""
FractalMesh Domain Monitor Agent
Checks DNS, HTTP reachability, and SSL expiry for sovereign domains
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import ssl
import time
import signal
import socket
import sqlite3
import urllib.request
import urllib.error
from datetime import datetime, timezone

ROOT     = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB       = os.path.join(ROOT, "database", "sovereign.db")
INTERVAL = int(os.getenv("DOMAIN_CHECK_INTERVAL", "3600"))

DOMAINS = [
    {"id": "fractalmesh-net", "host": "fractalmesh.net",     "https": True},
    {"id": "fractalmesh-www", "host": "www.fractalmesh.net", "https": True},
    {"id": "api-health",      "host": "127.0.0.1",           "https": False, "port": 5057},
]

_running = True


def _db_init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS domain_log (
        id INTEGER PRIMARY KEY, domain_id TEXT, host TEXT,
        dns_ok INTEGER, http_status INTEGER, ssl_days_left REAL,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()


def _dns_ok(host: str) -> bool:
    try:
        socket.getaddrinfo(host, 80, proto=socket.IPPROTO_TCP); return True
    except Exception:
        return False


def _http(host: str, https: bool, port=None) -> int:
    scheme = "https" if https else "http"
    p      = port or (443 if https else 80)
    url    = f"{scheme}://{host}:{p}/health" if port else f"{scheme}://{host}/"
    try:
        ctx = ssl._create_unverified_context() if https else None
        req = urllib.request.Request(url, headers={"User-Agent": "FractalMesh-DomainBot/1.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=8) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return -1


def _ssl_days(host: str) -> float:
    try:
        ctx  = ssl.create_default_context()
        conn = ctx.wrap_socket(socket.create_connection((host, 443), timeout=8), server_hostname=host)
        exp  = conn.getpeercert()["notAfter"]; conn.close()
        exp_dt = datetime.strptime(exp, "%b %d %H:%M:%S %Y %Z").replace(tzinfo=timezone.utc)
        return round((exp_dt - datetime.now(timezone.utc)).total_seconds() / 86400, 1)
    except Exception:
        return -1.0


def _log(d_id, host, dns_ok, http_status, ssl_days):
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("INSERT INTO domain_log (domain_id,host,dns_ok,http_status,ssl_days_left) VALUES (?,?,?,?,?)",
                 (d_id, host, int(dns_ok), http_status, ssl_days))
    conn.commit(); conn.close()


def run_cycle():
    print(f"[fm-domain] {datetime.utcnow().isoformat()}")
    for d in DOMAINS:
        dns  = _dns_ok(d["host"])
        http = _http(d["host"], d["https"], d.get("port"))
        ssl  = _ssl_days(d["host"]) if d["https"] and not d.get("port") else -1.0
        ssl_str = f"SSL {ssl:.0f}d" if ssl >= 0 else ""
        print(f"   → {d['host']:<28} DNS={'ok' if dns else 'FAIL'} HTTP={http} {ssl_str}")
        _log(d["id"], d["host"], dns, http, ssl)


def _sigterm(s, f):
    global _running
    _running = False


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    _db_init()
    print(f"[fm-domain] Active | interval={INTERVAL}s | domains={len(DOMAINS)}")
    while _running:
        try:
            run_cycle()
        except Exception as e:
            print(f"[fm-domain] ERR {e}")
        for _ in range(INTERVAL):
            if not _running: break
            time.sleep(1)
    print("[fm-domain] Stopped.")
