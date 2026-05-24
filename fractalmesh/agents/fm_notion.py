#!/usr/bin/env python3
"""
fm_notion.py — Notion API Integration Agent (Port 7802)
Pages, databases, content calendar, revenue sync, strategy sync.
Samuel James Hiotis | ABN 56 628 117 363
"""
import os, json, time, signal, sqlite3, logging, urllib.request, urllib.error, urllib.parse
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _l in _vault.read_text().splitlines():
        if "=" in _l and not _l.startswith("#"):
            _k, _, _v = _l.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

PORT          = int(os.getenv("NOTION_PORT", "7802"))
ROOT          = Path(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")))
DB            = ROOT / "database" / "sovereign.db"
LOG           = ROOT / "logs" / "notion.log"
NOTION_TOKEN  = os.getenv("NOTION_TOKEN","")
NOTION_API    = "https://api.notion.com/v1"
NOTION_VER    = "2022-06-28"

# DB IDs from vault
DB_STRATEGIES = os.getenv("NOTION_DB_STRATEGIES","")
DB_REVENUE    = os.getenv("NOTION_DB_REVENUE","")
DB_CONTENT    = os.getenv("NOTION_DB_CONTENT","")
DB_LEADS      = os.getenv("NOTION_DB_LEADS","")
DB_TASKS      = os.getenv("NOTION_DB_TASKS","")

for p in (ROOT, LOG.parent, DB.parent): p.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [NOTION] %(message)s",
    handlers=[logging.FileHandler(LOG), logging.StreamHandler()])
log = logging.getLogger("notion")

def _db_init():
    c = sqlite3.connect(DB, timeout=10); c.execute("PRAGMA journal_mode=WAL")
    c.execute("""CREATE TABLE IF NOT EXISTS notion_sync (
        id INTEGER PRIMARY KEY, object_type TEXT, object_id TEXT, title TEXT,
        status TEXT, ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    c.commit(); c.close()

def _notion(method, path, body=None):
    if not NOTION_TOKEN: return {"error":"NOTION_TOKEN not configured"}
    data = json.dumps(body).encode() if body else None
    req  = urllib.request.Request(f"{NOTION_API}/{path.lstrip('/')}", data=data, method=method,
        headers={"Authorization":f"Bearer {NOTION_TOKEN}", "Notion-Version":NOTION_VER,
                 "Content-Type":"application/json", "Accept":"application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r: return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error":f"http_{e.code}","detail":e.read().decode()[:300]}
    except Exception as e: return {"error":str(e)}

# ── Notion data builders ──────────────────────────────────────────────────────

def _rt(text: str) -> list:
    return [{"type":"text","text":{"content":str(text)[:2000]}}]

def _prop(type_: str, value) -> dict:
    if type_ == "title":       return {"title": _rt(value)}
    if type_ == "rich_text":   return {"rich_text": _rt(value)}
    if type_ == "number":      return {"number": float(value) if value is not None else None}
    if type_ == "select":      return {"select":{"name":str(value)}}
    if type_ == "multi_select":return {"multi_select":[{"name":v} for v in (value if isinstance(value,list) else [value])]}
    if type_ == "checkbox":    return {"checkbox": bool(value)}
    if type_ == "url":         return {"url": str(value)}
    if type_ == "date":        return {"date":{"start":str(value)}}
    return {"rich_text": _rt(str(value))}

def _md_to_blocks(md: str) -> list:
    blocks = []; lines = md.split("\n")
    for line in lines[:100]:  # Notion limit
        line = line.rstrip()
        if line.startswith("# "):
            blocks.append({"type":"heading_1","heading_1":{"rich_text":_rt(line[2:])}})
        elif line.startswith("## "):
            blocks.append({"type":"heading_2","heading_2":{"rich_text":_rt(line[3:])}})
        elif line.startswith("### "):
            blocks.append({"type":"heading_3","heading_3":{"rich_text":_rt(line[4:])}})
        elif line.startswith("- ") or line.startswith("* "):
            blocks.append({"type":"bulleted_list_item","bulleted_list_item":{"rich_text":_rt(line[2:])}})
        elif line.startswith("```"):
            pass  # skip code fences
        elif line.strip():
            blocks.append({"type":"paragraph","paragraph":{"rich_text":_rt(line)}})
    return blocks

# ── DB operations ─────────────────────────────────────────────────────────────

def _db_query(db_id: str, filter_: dict = None) -> list:
    body = {"page_size":50}
    if filter_: body["filter"] = filter_
    r = _notion("POST", f"databases/{db_id}/query", body)
    return r.get("results",[]) if isinstance(r, dict) else []

def _db_add(db_id: str, properties: dict) -> dict:
    return _notion("POST","pages",{"parent":{"database_id":db_id},"properties":properties})

def _page_create(parent_id: str, title: str, content_md: str = "") -> dict:
    body = {"parent":{"page_id":parent_id} if len(parent_id) == 36 else {"database_id":parent_id},
            "properties":{"title":_prop("title",title)}}
    if content_md: body["children"] = _md_to_blocks(content_md)
    return _notion("POST","pages",body)

# ── sync helpers ──────────────────────────────────────────────────────────────

def _sync_revenue() -> dict:
    if not DB_REVENUE: return {"error":"NOTION_DB_REVENUE not configured"}
    try:
        c  = sqlite3.connect(DB, timeout=5)
        rows = c.execute("SELECT channel_id,amount,currency,period,ts FROM revenue_snapshots ORDER BY ts DESC LIMIT 10").fetchall()
        c.close()
    except Exception as e: return {"error":str(e)}
    results = []
    for row in rows:
        r = _db_add(DB_REVENUE,{
            "Channel":  _prop("title",   row[0]),
            "Amount":   _prop("number",  row[1]),
            "Currency": _prop("select",  row[2]),
            "Period":   _prop("rich_text",row[3]),
            "Date":     _prop("date",    row[4][:10]),
        })
        results.append({"channel":row[0],"notion_id":r.get("id","")})
    return {"synced":len(results),"rows":results}

def _content_calendar() -> dict:
    if not DB_CONTENT: return {"error":"NOTION_DB_CONTENT not configured"}
    import datetime
    today = datetime.date.today()
    plan  = [
        {"day":0,"title":"Dev.to: FractalMesh build series — MCP router deep dive","type":"Technical","platform":"Dev.to"},
        {"day":2,"title":"LinkedIn: Sovereign AI on Android — 3-month update",    "type":"Story",    "platform":"LinkedIn"},
        {"day":4,"title":"Dev.to: OpenRouter cost tracking — real numbers",        "type":"Tutorial", "platform":"Dev.to"},
        {"day":5,"title":"Twitter/X: 500 zero-capital strategies thread",          "type":"Thread",   "platform":"Twitter"},
        {"day":7,"title":"Dev.to: Zero-capital AI business — week 12 revenue",     "type":"Report",   "platform":"Dev.to"},
    ]
    results = []
    for item in plan:
        pub_date = str(today + datetime.timedelta(days=item["day"]))
        r = _db_add(DB_CONTENT,{
            "Title":    _prop("title",     item["title"]),
            "Type":     _prop("select",    item["type"]),
            "Platform": _prop("select",    item["platform"]),
            "Publish":  _prop("date",      pub_date),
            "Status":   _prop("select",    "Planned"),
        })
        results.append({"title":item["title"],"date":pub_date,"id":r.get("id","")})
    return {"created":len(results),"calendar":results}

class NotionHandler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _r(self, code, body):
        p = json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",str(len(p))); self.end_headers(); self.wfile.write(p)

    def do_GET(self):
        qs = urllib.parse.parse_qs(self.path.split("?",1)[-1] if "?" in self.path else "")
        ep = self.path.split("?")[0]
        if ep == "/health":
            self._r(200,{"status":"ok","configured":bool(NOTION_TOKEN),
                          "databases":{"strategies":bool(DB_STRATEGIES),"revenue":bool(DB_REVENUE),
                                       "content":bool(DB_CONTENT),"leads":bool(DB_LEADS),"tasks":bool(DB_TASKS)}})
        elif ep == "/pages":
            self._r(200, _notion("GET","pages"))
        elif ep == "/search":
            q = qs.get("q",[""])[0]
            self._r(200, _notion("POST","search",{"query":q,"page_size":10}) if q else {"error":"q required"})
        elif ep == "/db":
            db_id = qs.get("id",[""])[0]
            self._r(200, {"results":_db_query(db_id)} if db_id else {"error":"id required"})
        else:
            self._r(404,{"error":"not_found"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length",0)); d = json.loads(self.rfile.read(n))
            ep = self.path.split("?")[0]
            if ep == "/page":
                self._r(200, _page_create(d.get("parent_id",""), d.get("title",""), d.get("content_markdown","")))
            elif ep == "/db_entry":
                db_id = d.get("db_id",""); props = d.get("properties",{})
                notion_props = {k:_prop(v.get("type","rich_text"),v.get("value","")) for k,v in props.items()}
                self._r(200, _db_add(db_id, notion_props))
            elif ep == "/update":
                pid = d.get("page_id",""); props = d.get("properties",{})
                self._r(200, _notion("PATCH",f"pages/{pid}",{"properties":props}))
            elif ep == "/sync_revenue":
                self._r(200, _sync_revenue())
            elif ep == "/content_calendar":
                self._r(200, _content_calendar())
            elif ep == "/sync_strategies":
                # Push sample strategies in batches
                if not DB_STRATEGIES: self._r(400,{"error":"NOTION_DB_STRATEGIES not configured"}); return
                strategies = [
                    {"id":1,"title":"Upwork Python automation gig","category":"Freelance","income":"$15-50/hr","difficulty":"Entry"},
                    {"id":2,"title":"Fiverr data scraping micro-gig","category":"Freelance","income":"$5-100/order","difficulty":"Entry"},
                    {"id":10,"title":"Dev.to paid technical article","category":"Content","income":"$50-300","difficulty":"Entry"},
                    {"id":81,"title":"Scale AI data labelling","category":"Data/AI","income":"$10-30/hr","difficulty":"Entry"},
                    {"id":301,"title":"Manus affiliate (30% recurring)","category":"Affiliate","income":"30% MRR","difficulty":"Entry"},
                ]
                results = []
                for s in strategies:
                    r = _db_add(DB_STRATEGIES,{
                        "Title":      _prop("title",    s["title"]),
                        "Category":   _prop("select",   s["category"]),
                        "Income":     _prop("rich_text",s["income"]),
                        "Difficulty": _prop("select",   s["difficulty"]),
                        "ID":         _prop("number",   s["id"]),
                    })
                    results.append({"id":s["id"],"notion_id":r.get("id","")})
                self._r(200,{"synced":len(results),"results":results})
            else:
                self._r(404,{"error":"unknown_path"})
        except json.JSONDecodeError: self._r(400,{"error":"invalid_json"})
        except Exception as e: self._r(500,{"error":str(e)})

_running = True
def _shutdown(*_): global _running; _running = False
signal.signal(signal.SIGTERM, _shutdown); signal.signal(signal.SIGINT, _shutdown)

def main():
    _db_init(); server = HTTPServer(("0.0.0.0", PORT), NotionHandler)
    log.info("Notion agent on port %d | token=%s", PORT, bool(NOTION_TOKEN))
    try:
        while _running: server.handle_request()
    finally: server.server_close()

if __name__ == "__main__": main()
