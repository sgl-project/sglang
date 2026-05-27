#!/usr/bin/env python3
"""
fm_admin_api.py — FractalMesh Unified Administration API (Port 7804)
Mesh management, agent control, config, audit, port scanning, DB stats.
Samuel James Hiotis | ABN 56 628 117 363
"""
import os, json, time, signal, sqlite3, logging, subprocess, socket, urllib.request, hmac
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _l in _vault.read_text().splitlines():
        if "=" in _l and not _l.startswith("#"):
            _k, _, _v = _l.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

PORT         = int(os.getenv("ADMIN_PORT", "7804"))
ROOT         = Path(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")))
DB           = ROOT / "database" / "sovereign.db"
LOG          = ROOT / "logs" / "admin_api.log"
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")
MCP_URL      = os.getenv("MCP_URL", "http://127.0.0.1:7785")
VERSION      = "2.1.0"

for p in (ROOT, LOG.parent, DB.parent): p.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [ADMIN] %(message)s",
    handlers=[logging.FileHandler(LOG), logging.StreamHandler()])
log = logging.getLogger("admin_api")

def _db_init():
    c = sqlite3.connect(DB, timeout=10); c.execute("PRAGMA journal_mode=WAL")
    c.execute("""CREATE TABLE IF NOT EXISTS admin_audit (
        id INTEGER PRIMARY KEY, action TEXT, actor TEXT,
        params TEXT, result TEXT, ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    c.execute("""CREATE TABLE IF NOT EXISTS admin_config (
        key TEXT PRIMARY KEY, value TEXT, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    c.commit(); c.close()

def _audit(action, params, result, actor="api"):
    try:
        c = sqlite3.connect(DB, timeout=5)
        c.execute("INSERT INTO admin_audit (action,actor,params,result) VALUES (?,?,?,?)",
                  (action, actor, json.dumps(params)[:500], str(result)[:500]))
        c.commit(); c.close()
    except Exception as e: log.warning("audit: %s", e)

def _auth(headers) -> bool:
    if not ADMIN_SECRET: return True  # open if no secret set
    return hmac.compare_digest(headers.get("X-Admin-Secret", ""), ADMIN_SECRET)

def _pm2_list() -> list:
    try:
        out = subprocess.getoutput("pm2 jlist 2>/dev/null")
        procs = json.loads(out) if out.strip().startswith("[") else []
        return [{"name":p.get("name",""), "status":p.get("pm2_env",{}).get("status",""),
                 "pid":p.get("pid"), "memory_mb":round(p.get("monit",{}).get("memory",0)/1048576,1),
                 "cpu":p.get("monit",{}).get("cpu",0), "restarts":p.get("pm2_env",{}).get("restart_time",0),
                 "uptime":p.get("pm2_env",{}).get("pm_uptime")} for p in procs]
    except Exception: return []

def _pm2_cmd(cmd, agent):
    try:
        out = subprocess.getoutput(f"pm2 {cmd} {agent} 2>&1")
        return {"status":"ok","output":out[:500]}
    except Exception as e: return {"error":str(e)}

def _tail_log(agent, lines=50) -> str:
    for suffix in ["-out.log","-error.log"]:
        p = ROOT / "logs" / f"{agent}{suffix}"
        if p.exists():
            try:
                out = subprocess.getoutput(f"tail -n {lines} '{p}'")
                return out
            except Exception: pass
    return "log not found"

def _port_scan() -> dict:
    ports = list(range(7785, 7815))
    results = {}
    for port in ports:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.3)
        try:
            s.connect(("127.0.0.1", port)); results[str(port)] = "LISTENING"
        except: results[str(port)] = "closed"
        finally: s.close()
    return results

def _db_stats() -> list:
    try:
        c = sqlite3.connect(DB, timeout=5)
        tables = c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        stats  = []
        for (t,) in tables:
            try:
                count = c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                stats.append({"table":t,"rows":count})
            except: pass
        c.close(); return sorted(stats, key=lambda x:-x["rows"])
    except Exception: return []

def _disk_usage() -> dict:
    try:
        out = subprocess.getoutput(f"du -sh '{ROOT}' 2>/dev/null")
        return {"root_dir":str(ROOT),"used":out.split()[0] if out else "unknown"}
    except: return {}

def _vault_keys() -> list:
    keys = []
    if _vault.exists():
        for line in _vault.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k = line.partition("=")[0].strip()
                if k: keys.append(k)
    return sorted(keys)

def _config_get(key):
    try:
        c = sqlite3.connect(DB, timeout=5)
        r = c.execute("SELECT value FROM admin_config WHERE key=?",(key,)).fetchone()
        c.close(); return r[0] if r else None
    except: return None

def _config_set(key, value):
    try:
        c = sqlite3.connect(DB, timeout=5)
        c.execute("INSERT OR REPLACE INTO admin_config (key,value,updated_at) VALUES (?,?,CURRENT_TIMESTAMP)",(key,str(value)))
        c.commit(); c.close()
    except Exception as e: log.warning("config_set: %s",e)

class AdminHandler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _r(self, code, body):
        p = json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",str(len(p))); self.end_headers(); self.wfile.write(p)

    def _auth_401(self):
        p = json.dumps({"error": "X-Admin-Secret required"}).encode()
        self.send_response(401)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(p)))
        self.send_header("WWW-Authenticate", 'X-Admin-Secret realm="FractalMesh"')
        self.end_headers()
        self.wfile.write(p)

    def do_GET(self):
        import urllib.parse
        qs = urllib.parse.parse_qs(self.path.split("?",1)[-1] if "?" in self.path else "")
        ep = self.path.split("?")[0]

        _open_get = {"/health"}
        if ep not in _open_get and not _auth(self.headers):
            self._auth_401(); return

        if ep == "/health":
            procs = _pm2_list()
            self._r(200,{"status":"ok","version":VERSION,"agents":len(procs)})
        elif ep == "/status":
            procs = _pm2_list()
            self._r(200,{"processes":procs,"disk":_disk_usage(),
                          "uptime":subprocess.getoutput("uptime -p 2>/dev/null"),
                          "ts":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())})
        elif ep == "/agents":
            self._r(200,{"agents":_pm2_list()})
        elif ep == "/logs":
            agent = qs.get("agent",[""])[0]; lines = int(qs.get("lines",["50"])[0])
            self._r(200,{"agent":agent,"log":_tail_log(agent,lines)})
        elif ep == "/config":
            try:
                c = sqlite3.connect(DB,timeout=5)
                rows = c.execute("SELECT key,value,updated_at FROM admin_config ORDER BY key").fetchall()
                c.close(); self._r(200,{"config":[{"key":r[0],"value":r[1],"updated":r[2]} for r in rows]})
            except Exception as e: self._r(500,{"error":str(e)})
        elif ep == "/ports":
            self._r(200,{"ports":_port_scan()})
        elif ep == "/audit":
            try:
                c = sqlite3.connect(DB,timeout=5)
                rows = c.execute("SELECT * FROM admin_audit ORDER BY ts DESC LIMIT 100").fetchall()
                c.close(); self._r(200,{"audit":rows})
            except Exception as e: self._r(500,{"error":str(e)})
        elif ep == "/db/stats":
            self._r(200,{"tables":_db_stats()})
        elif ep == "/vault/keys":
            self._r(200,{"keys":_vault_keys(),"count":len(_vault_keys())})
        else:
            self._r(404,{"error":"not_found"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length",0)); d = json.loads(self.rfile.read(n))
            ep = self.path.split("?")[0]

            if ep not in ("/health",) and not _auth(self.headers):
                self._auth_401(); return

            if ep == "/agent/restart":
                agent = d.get("agent_name",""); r = _pm2_cmd("restart",agent)
                _audit("restart",d,r); self._r(200,r)
            elif ep == "/agent/stop":
                agent = d.get("agent_name",""); r = _pm2_cmd("stop",agent)
                _audit("stop",d,r); self._r(200,r)
            elif ep == "/agent/start":
                agent = d.get("agent_name",""); r = _pm2_cmd("start",agent)
                _audit("start",d,r); self._r(200,r)
            elif ep == "/config/set":
                k = d.get("key",""); v = d.get("value","")
                _config_set(k,v); _audit("config_set",d,"ok")
                self._r(200,{"key":k,"value":v})
            elif ep == "/broadcast":
                intent = d.get("intent",""); kwargs = d.get("kwargs",{})
                payload = json.dumps({"intent":intent,"args":[],"kwargs":kwargs}).encode()
                req = urllib.request.Request(f"{MCP_URL}/",data=payload,
                    headers={"Content-Type":"application/json"},method="POST")
                try:
                    with urllib.request.urlopen(req,timeout=5) as r: result = json.loads(r.read())
                except Exception as e: result = {"error":str(e)}
                _audit("broadcast",d,result); self._r(200,result)
            else:
                self._r(404,{"error":"unknown_path"})
        except json.JSONDecodeError: self._r(400,{"error":"invalid_json"})
        except Exception as e: self._r(500,{"error":str(e)})

_running = True
def _shutdown(*_): global _running; _running = False
signal.signal(signal.SIGTERM, _shutdown); signal.signal(signal.SIGINT, _shutdown)

def main():
    _db_init(); server = HTTPServer(("0.0.0.0", PORT), AdminHandler)
    log.info("Admin API on port %d | secret=%s | version=%s", PORT, bool(ADMIN_SECRET), VERSION)
    try:
        while _running: server.handle_request()
    finally: server.server_close()

if __name__ == "__main__": main()
