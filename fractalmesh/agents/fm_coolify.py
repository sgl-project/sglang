#!/usr/bin/env python3
"""
fm_coolify.py — Coolify Self-Hosted PaaS Agent (Port 7796)
Deploy, restart, stop, update env vars via Coolify v4 API.
Samuel James Hiotis | ABN 56 628 117 363
"""
import json
import logging
import os
import signal
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _l in _vault.read_text().splitlines():
        if "=" in _l and not _l.startswith("#"):
            _k, _, _v = _l.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

PORT          = int(os.getenv("COOLIFY_PORT", "7796"))
ROOT          = Path(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")))
DB            = ROOT / "database" / "sovereign.db"
LOG           = ROOT / "logs" / "coolify.log"
COOLIFY_URL   = os.getenv("COOLIFY_URL", "").rstrip("/")
COOLIFY_TOKEN = os.getenv("COOLIFY_TOKEN", "")
CF_API        = f"{COOLIFY_URL}/api/v1" if COOLIFY_URL else ""

for p in (ROOT, LOG.parent, DB.parent): p.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [COOLIFY] %(message)s",
    handlers=[logging.FileHandler(LOG), logging.StreamHandler()])
log = logging.getLogger("coolify")

def _db_init():
    c = sqlite3.connect(DB, timeout=10); c.execute("PRAGMA journal_mode=WAL")
    c.execute("""CREATE TABLE IF NOT EXISTS coolify_deployments (
        id INTEGER PRIMARY KEY, app_name TEXT, deployment_id TEXT,
        status TEXT, triggered_by TEXT, latency_ms REAL,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    c.commit(); c.close()

def _db_log(app, dep_id, status, triggered_by, latency_ms):
    try:
        c = sqlite3.connect(DB, timeout=5)
        c.execute("INSERT INTO coolify_deployments (app_name,deployment_id,status,triggered_by,latency_ms) VALUES (?,?,?,?,?)",
                  (app, dep_id, status, triggered_by, latency_ms))
        c.commit(); c.close()
    except Exception as e: log.warning("db: %s", e)

def _cf(method, path, body=None):
    if not CF_API or not COOLIFY_TOKEN: return {"error":"COOLIFY_URL or COOLIFY_TOKEN not configured"}
    data = json.dumps(body).encode() if body else None
    req  = urllib.request.Request(f"{CF_API}/{path.lstrip('/')}", data=data, method=method,
        headers={"Authorization": f"Bearer {COOLIFY_TOKEN}", "Content-Type":"application/json", "Accept":"application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r: return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": f"http_{e.code}", "detail": e.read().decode()[:300]}
    except Exception as e: return {"error": str(e)}

def _pm2_status() -> list:
    try:
        out = subprocess.getoutput("pm2 jlist 2>/dev/null")
        procs = json.loads(out) if out.strip().startswith("[") else []
        return [{"name":p.get("name"), "status":p.get("pm2_env",{}).get("status"),
                 "memory_mb":round(p.get("monit",{}).get("memory",0)/1048576,1),
                 "cpu":p.get("monit",{}).get("cpu",0), "pid":p.get("pid")} for p in procs]
    except Exception: return []

class CoolifyHandler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _r(self, code, body):
        p = json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",str(len(p))); self.end_headers(); self.wfile.write(p)

    def do_GET(self):
        import urllib.parse
        qs = urllib.parse.parse_qs(self.path.split("?",1)[-1] if "?" in self.path else "")
        ep = self.path.split("?")[0]
        if ep == "/health":
            self._r(200, {"status":"ok","url":COOLIFY_URL,"configured":bool(COOLIFY_URL and COOLIFY_TOKEN)})
        elif ep == "/apps":      self._r(200, _cf("GET","applications"))
        elif ep == "/servers":   self._r(200, _cf("GET","servers"))
        elif ep == "/pm2_status":self._r(200, {"processes":_pm2_status(),"count":len(_pm2_status())})
        elif ep == "/deployments":
            app = qs.get("app",[""])[0]
            self._r(200, _cf("GET", f"applications/{app}/deployments") if app else {"error":"app required"})
        else:
            self._r(404, {"error":"not_found"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length",0)); d = json.loads(self.rfile.read(n))
            t0 = time.time(); ep = self.path.split("?")[0]
            if ep == "/deploy":
                uuid  = d.get("app_uuid","")
                force = d.get("force_rebuild", False)
                r     = _cf("POST", f"applications/{uuid}/restart" if not force else f"applications/{uuid}/deploy",
                             {"force_rebuild": force})
                dep_id = r.get("deployment_uuid","")
                lat    = (time.time()-t0)*1000
                _db_log(uuid, dep_id, "triggered", "api", lat)
                log.info("deploy app=%s dep=%s", uuid, dep_id)
                self._r(200, r)
            elif ep == "/restart":
                uuid = d.get("app_uuid","")
                r    = _cf("POST", f"applications/{uuid}/restart")
                _db_log(uuid, "", "restart", "api", (time.time()-t0)*1000); self._r(200, r)
            elif ep == "/stop":
                uuid = d.get("app_uuid","")
                r    = _cf("POST", f"applications/{uuid}/stop")
                _db_log(uuid, "", "stop", "api", (time.time()-t0)*1000); self._r(200, r)
            elif ep == "/env":
                uuid = d.get("app_uuid",""); envs = d.get("envs", [])
                r    = _cf("PATCH", f"applications/{uuid}", {"environment_variables": envs})
                self._r(200, r)
            elif ep == "/webhook":
                event = d.get("type", "unknown")
                _db_log(d.get("application",{}).get("name",""), d.get("deployment_uuid",""), event, "webhook", 0)
                self._r(200, {"status":"received","event":event})
            else:
                self._r(404, {"error":"unknown_path"})
        except json.JSONDecodeError: self._r(400, {"error":"invalid_json"})
        except Exception as e: self._r(500, {"error":str(e)})

_running = True
def _shutdown(*_): global _running; _running = False
signal.signal(signal.SIGTERM, _shutdown); signal.signal(signal.SIGINT, _shutdown)

def main():
    _db_init(); server = HTTPServer(("0.0.0.0", PORT), CoolifyHandler)
    log.info("Coolify agent on port %d | url=%s", PORT, COOLIFY_URL or "not_set")
    try:
        while _running: server.handle_request()
    finally: server.server_close()

if __name__ == "__main__": main()
