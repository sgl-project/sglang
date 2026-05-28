#!/usr/bin/env python3
"""
fm_base44.py — base44 No-Code App Builder Integration (Port 7809)
Full CRUD for apps, pages, components, data-models, deployments, webhooks.
Samuel James Hiotis | ABN 56 628 117 363
"""
import json
import logging
import os
import signal
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _l in _vault.read_text().splitlines():
        if "=" in _l and not _l.startswith("#"):
            _k, _, _v = _l.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

PORT        = int(os.getenv("BASE44_PORT", "7809"))
API_KEY     = os.getenv("BASE44_API_KEY", "")
BASE_URL    = os.getenv("BASE44_BASE_URL", "https://api.base44.com/v1")
WORKSPACE   = os.getenv("BASE44_WORKSPACE", "")
ROOT        = Path(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")))
DB          = ROOT / "database" / "sovereign.db"
LOG         = ROOT / "logs" / "base44.log"

for p in (ROOT, LOG.parent, DB.parent): p.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [BASE44] %(message)s",
    handlers=[logging.FileHandler(LOG), logging.StreamHandler()])
log = logging.getLogger("base44")

# ── DB init ────────────────────────────────────────────────────────────────────
def _db_init():
    c = sqlite3.connect(DB, timeout=10); c.execute("PRAGMA journal_mode=WAL")
    c.execute("""CREATE TABLE IF NOT EXISTS base44_ops (
        id INTEGER PRIMARY KEY, op TEXT, resource TEXT, resource_id TEXT,
        status TEXT, ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    c.commit(); c.close()

def _log_op(op: str, resource: str, resource_id: str, status: str):
    try:
        c = sqlite3.connect(DB, timeout=5)
        c.execute("INSERT INTO base44_ops (op,resource,resource_id,status) VALUES (?,?,?,?)",
                  (op, resource, resource_id or "", status))
        c.commit(); c.close()
    except Exception: pass

# ── HTTP helpers ───────────────────────────────────────────────────────────────
def _req(method: str, path: str, body: dict = None) -> dict:
    if not API_KEY:
        return {"error": "BASE44_API_KEY not configured in vault"}
    url = f"{BASE_URL.rstrip('/')}/{path.lstrip('/')}"
    data = json.dumps(body).encode() if body else None
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
        "Accept":        "application/json",
        "X-Workspace":   WORKSPACE,
    }
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": e.reason, "status_code": e.code, "detail": e.read().decode("utf-8","replace")[:500]}
    except Exception as e:
        return {"error": str(e)}

def _get(path):    return _req("GET",    path)
def _post(path, body): return _req("POST",   path, body)
def _put(path, body):  return _req("PUT",    path, body)
def _patch(path, body):return _req("PATCH",  path, body)
def _del(path):    return _req("DELETE", path)

# ── App operations ─────────────────────────────────────────────────────────────
def list_apps() -> dict:
    return _get("/apps")

def get_app(app_id: str) -> dict:
    return _get(f"/apps/{app_id}")

def create_app(name: str, description: str = "", template: str = "") -> dict:
    body = {"name": name, "description": description}
    if template: body["template"] = template
    if WORKSPACE: body["workspace"] = WORKSPACE
    r = _post("/apps", body)
    _log_op("create", "app", r.get("id", ""), "ok" if "id" in r else "error")
    return r

def update_app(app_id: str, fields: dict) -> dict:
    r = _patch(f"/apps/{app_id}", fields)
    _log_op("update", "app", app_id, "ok" if "id" in r else "error")
    return r

def delete_app(app_id: str) -> dict:
    r = _del(f"/apps/{app_id}")
    _log_op("delete", "app", app_id, "ok")
    return r

def deploy_app(app_id: str, env: str = "production") -> dict:
    r = _post(f"/apps/{app_id}/deploy", {"environment": env})
    _log_op("deploy", "app", app_id, "ok" if "deployment_id" in r else "error")
    return r

def app_status(app_id: str) -> dict:
    return _get(f"/apps/{app_id}/status")

# ── Page / Component operations ───────────────────────────────────────────────
def list_pages(app_id: str) -> dict:
    return _get(f"/apps/{app_id}/pages")

def create_page(app_id: str, name: str, path: str = "", layout: str = "default") -> dict:
    r = _post(f"/apps/{app_id}/pages", {"name": name, "path": path or f"/{name.lower().replace(' ','-')}", "layout": layout})
    _log_op("create", "page", r.get("id", ""), "ok" if "id" in r else "error")
    return r

def list_components(app_id: str, page_id: str = "") -> dict:
    path = f"/apps/{app_id}/components"
    if page_id: path += f"?page_id={page_id}"
    return _get(path)

def add_component(app_id: str, page_id: str, component_type: str, props: dict) -> dict:
    r = _post(f"/apps/{app_id}/components", {"page_id": page_id, "type": component_type, "props": props})
    _log_op("add", "component", r.get("id", ""), "ok" if "id" in r else "error")
    return r

def update_component(app_id: str, component_id: str, props: dict) -> dict:
    return _patch(f"/apps/{app_id}/components/{component_id}", {"props": props})

def remove_component(app_id: str, component_id: str) -> dict:
    return _del(f"/apps/{app_id}/components/{component_id}")

# ── Data model / Entity operations ────────────────────────────────────────────
def list_entities(app_id: str) -> dict:
    return _get(f"/apps/{app_id}/entities")

def create_entity(app_id: str, name: str, fields: list) -> dict:
    """fields: [{"name":"col","type":"string|number|boolean|date|relation","required":bool}]"""
    r = _post(f"/apps/{app_id}/entities", {"name": name, "fields": fields})
    _log_op("create", "entity", r.get("id", ""), "ok" if "id" in r else "error")
    return r

def update_entity(app_id: str, entity_id: str, fields: list) -> dict:
    return _patch(f"/apps/{app_id}/entities/{entity_id}", {"fields": fields})

def delete_entity(app_id: str, entity_id: str) -> dict:
    return _del(f"/apps/{app_id}/entities/{entity_id}")

# ── Data records ──────────────────────────────────────────────────────────────
def list_records(app_id: str, entity_id: str, filters: dict = None, limit: int = 50) -> dict:
    params = {"limit": limit}
    if filters: params["filters"] = json.dumps(filters)
    qs = urllib.parse.urlencode(params)
    return _get(f"/apps/{app_id}/entities/{entity_id}/records?{qs}")

def create_record(app_id: str, entity_id: str, data: dict) -> dict:
    r = _post(f"/apps/{app_id}/entities/{entity_id}/records", data)
    _log_op("create", "record", r.get("id", ""), "ok" if "id" in r else "error")
    return r

def update_record(app_id: str, entity_id: str, record_id: str, data: dict) -> dict:
    return _patch(f"/apps/{app_id}/entities/{entity_id}/records/{record_id}", data)

def delete_record(app_id: str, entity_id: str, record_id: str) -> dict:
    return _del(f"/apps/{app_id}/entities/{entity_id}/records/{record_id}")

# ── Integration / Webhook operations ─────────────────────────────────────────
def list_integrations(app_id: str) -> dict:
    return _get(f"/apps/{app_id}/integrations")

def add_integration(app_id: str, provider: str, config: dict) -> dict:
    r = _post(f"/apps/{app_id}/integrations", {"provider": provider, "config": config})
    _log_op("add", "integration", r.get("id", ""), "ok" if "id" in r else "error")
    return r

def list_webhooks(app_id: str) -> dict:
    return _get(f"/apps/{app_id}/webhooks")

def create_webhook(app_id: str, event: str, url: str, secret: str = "") -> dict:
    body = {"event": event, "url": url}
    if secret: body["secret"] = secret
    r = _post(f"/apps/{app_id}/webhooks", body)
    _log_op("create", "webhook", r.get("id", ""), "ok" if "id" in r else "error")
    return r

def delete_webhook(app_id: str, webhook_id: str) -> dict:
    return _del(f"/apps/{app_id}/webhooks/{webhook_id}")

# ── Deployment history ────────────────────────────────────────────────────────
def list_deployments(app_id: str) -> dict:
    return _get(f"/apps/{app_id}/deployments")

def rollback_deployment(app_id: str, deployment_id: str) -> dict:
    r = _post(f"/apps/{app_id}/deployments/{deployment_id}/rollback", {})
    _log_op("rollback", "deployment", deployment_id, "ok" if "id" in r else "error")
    return r

# ── Workspace operations ───────────────────────────────────────────────────────
def list_workspace_members() -> dict:
    if not WORKSPACE: return {"error": "BASE44_WORKSPACE not set"}
    return _get(f"/workspaces/{WORKSPACE}/members")

def invite_member(email: str, role: str = "editor") -> dict:
    if not WORKSPACE: return {"error": "BASE44_WORKSPACE not set"}
    return _post(f"/workspaces/{WORKSPACE}/invites", {"email": email, "role": role})

# ── FractalMesh quickstart — create standard app scaffold ─────────────────────
def create_fractalmesh_app(app_name: str = "FractalMesh Portal") -> dict:
    """Bootstrap a standard FractalMesh-branded no-code app with data models."""
    app = create_app(app_name, "Sovereign AI Automation Portal — FractalMesh OMEGA Titan")
    if "id" not in app: return {"error": "App creation failed", "detail": app}
    app_id = app["id"]

    # Core data models
    entities = [
        ("Leads",      [{"name":"email","type":"string","required":True},
                        {"name":"source","type":"string"},{"name":"score","type":"number"},
                        {"name":"status","type":"string"},{"name":"created_at","type":"date"}]),
        ("Revenue",    [{"name":"amount","type":"number","required":True},
                        {"name":"channel","type":"string"},{"name":"currency","type":"string"},
                        {"name":"ts","type":"date"}]),
        ("Strategies", [{"name":"title","type":"string","required":True},
                        {"name":"category","type":"string"},{"name":"difficulty","type":"string"},
                        {"name":"active","type":"boolean"}]),
    ]
    created_entities = []
    for ename, fields in entities:
        e = create_entity(app_id, ename, fields)
        created_entities.append({"name": ename, "id": e.get("id", "error")})

    # Core pages
    pages = [("Dashboard","/"),("Leads","/leads"),("Revenue","/revenue"),("Strategies","/strategies")]
    created_pages = []
    for pname, ppath in pages:
        p = create_page(app_id, pname, ppath)
        created_pages.append({"name": pname, "id": p.get("id","error")})

    return {
        "app_id":    app_id,
        "app_name":  app.get("name"),
        "entities":  created_entities,
        "pages":     created_pages,
        "status":    "scaffolded",
    }

# ── HTTP Handler ───────────────────────────────────────────────────────────────
class Base44Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _r(self, code, body):
        p = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type","application/json")
        self.send_header("Content-Length", str(len(p))); self.end_headers(); self.wfile.write(p)

    def do_GET(self):
        qs  = urllib.parse.parse_qs(self.path.split("?",1)[-1] if "?" in self.path else "")
        ep  = self.path.split("?")[0]
        app = qs.get("app_id",[""])[0]

        if ep == "/health":
            self._r(200, {"status":"ok","configured":bool(API_KEY),"workspace":WORKSPACE or "unset",
                          "base_url":BASE_URL})
        elif ep == "/apps":
            self._r(200, list_apps())
        elif ep == "/apps/status" and app:
            self._r(200, app_status(app))
        elif ep == "/pages" and app:
            self._r(200, list_pages(app))
        elif ep == "/components" and app:
            pg = qs.get("page_id",[""])[0]
            self._r(200, list_components(app, pg))
        elif ep == "/entities" and app:
            self._r(200, list_entities(app))
        elif ep == "/records" and app:
            eid = qs.get("entity_id",[""])[0]
            lim = int(qs.get("limit",[50])[0])
            self._r(200, list_records(app, eid, limit=lim))
        elif ep == "/integrations" and app:
            self._r(200, list_integrations(app))
        elif ep == "/webhooks" and app:
            self._r(200, list_webhooks(app))
        elif ep == "/deployments" and app:
            self._r(200, list_deployments(app))
        elif ep == "/members":
            self._r(200, list_workspace_members())
        elif ep == "/ops":
            c = sqlite3.connect(DB, timeout=5)
            rows = c.execute("SELECT op,resource,resource_id,status,ts FROM base44_ops ORDER BY ts DESC LIMIT 50").fetchall()
            c.close()
            self._r(200, {"ops":[{"op":r[0],"resource":r[1],"resource_id":r[2],"status":r[3],"ts":r[4]} for r in rows]})
        else:
            self._r(404, {"error":"not_found","endpoints":[
                "GET /health","GET /apps","GET /apps/status?app_id=","GET /pages?app_id=",
                "GET /components?app_id=&page_id=","GET /entities?app_id=","GET /records?app_id=&entity_id=",
                "GET /integrations?app_id=","GET /webhooks?app_id=","GET /deployments?app_id=",
                "GET /members","GET /ops",
                "POST /apps","POST /apps/deploy","POST /apps/scaffold",
                "POST /pages","POST /components","POST /entities","POST /records",
                "POST /integrations","POST /webhooks","POST /invite",
                "PATCH /apps","PATCH /components","PATCH /entities","PATCH /records",
                "DELETE /apps","DELETE /components","DELETE /entities",
                "DELETE /records","DELETE /webhooks","POST /rollback",
            ]})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length",0))
            d = json.loads(self.rfile.read(n)) if n else {}
            ep = self.path.split("?")[0]
            app = d.get("app_id","")

            if ep == "/apps":
                self._r(200, create_app(d.get("name","FractalMesh App"),
                                        d.get("description",""), d.get("template","")))
            elif ep == "/apps/scaffold":
                self._r(200, create_fractalmesh_app(d.get("name","FractalMesh Portal")))
            elif ep == "/apps/deploy":
                self._r(200, deploy_app(app, d.get("env","production")))
            elif ep == "/apps/rollback":
                self._r(200, rollback_deployment(app, d.get("deployment_id","")))
            elif ep == "/apps/delete":
                self._r(200, delete_app(app))
            elif ep == "/apps/update":
                self._r(200, update_app(app, d.get("fields",{})))
            elif ep == "/pages":
                self._r(200, create_page(app, d.get("name",""), d.get("path",""), d.get("layout","default")))
            elif ep == "/components":
                self._r(200, add_component(app, d.get("page_id",""), d.get("type","text"), d.get("props",{})))
            elif ep == "/components/update":
                self._r(200, update_component(app, d.get("component_id",""), d.get("props",{})))
            elif ep == "/components/delete":
                self._r(200, remove_component(app, d.get("component_id","")))
            elif ep == "/entities":
                self._r(200, create_entity(app, d.get("name",""), d.get("fields",[])))
            elif ep == "/entities/update":
                self._r(200, update_entity(app, d.get("entity_id",""), d.get("fields",[])))
            elif ep == "/entities/delete":
                self._r(200, delete_entity(app, d.get("entity_id","")))
            elif ep == "/records":
                self._r(200, create_record(app, d.get("entity_id",""), d.get("data",{})))
            elif ep == "/records/update":
                self._r(200, update_record(app, d.get("entity_id",""), d.get("record_id",""), d.get("data",{})))
            elif ep == "/records/delete":
                self._r(200, delete_record(app, d.get("entity_id",""), d.get("record_id","")))
            elif ep == "/integrations":
                self._r(200, add_integration(app, d.get("provider",""), d.get("config",{})))
            elif ep == "/webhooks":
                self._r(200, create_webhook(app, d.get("event",""), d.get("url",""), d.get("secret","")))
            elif ep == "/webhooks/delete":
                self._r(200, delete_webhook(app, d.get("webhook_id","")))
            elif ep == "/invite":
                self._r(200, invite_member(d.get("email",""), d.get("role","editor")))
            else:
                self._r(404, {"error":"unknown_endpoint"})
        except json.JSONDecodeError:
            self._r(400, {"error":"invalid_json"})
        except Exception as e:
            self._r(500, {"error": str(e)})

_running = True
def _shutdown(*_): global _running; _running = False
signal.signal(signal.SIGTERM, _shutdown); signal.signal(signal.SIGINT, _shutdown)

def main():
    _db_init()
    server = HTTPServer(("0.0.0.0", PORT), Base44Handler)
    log.info("base44 agent on port %d | workspace=%s | configured=%s", PORT, WORKSPACE or "unset", bool(API_KEY))
    try:
        while _running: server.handle_request()
    finally:
        server.server_close()

if __name__ == "__main__": main()
