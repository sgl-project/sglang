#!/usr/bin/env python3
"""
fm_scraper_v2.py — Advanced Web Scraper + Google Dorks Engine v2 (Port 7807)
30 dorks across 8 categories, sitemap crawl, recursive scrape, lead discovery.
Samuel James Hiotis | ABN 56 628 117 363
"""
import os, json, time, signal, sqlite3, logging, re, urllib.request, urllib.error, urllib.parse
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _l in _vault.read_text().splitlines():
        if "=" in _l and not _l.startswith("#"):
            _k, _, _v = _l.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

PORT       = int(os.getenv("SCRAPER_PORT", "7807"))
ROOT       = Path(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")))
DB         = ROOT / "database" / "sovereign.db"
LOG        = ROOT / "logs" / "scraper_v2.log"
CSE_KEY    = os.getenv("GOOGLE_CSE_API_KEY", "")
CSE_ID     = os.getenv("GOOGLE_CSE_ID", "")
SCRAPER_KEY= os.getenv("SCRAPERAPI_KEY", "")  # optional proxy

for p in (ROOT, LOG.parent, DB.parent): p.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [SCRAPER] %(message)s",
    handlers=[logging.FileHandler(LOG), logging.StreamHandler()])
log = logging.getLogger("scraper_v2")

# ── 30 dorks across 8 categories ─────────────────────────────────────────────
DORKS = {
    "linkedin_leads": [
        'site:linkedin.com/in "automation engineer" "Albury" OR "Wagga Wagga" OR "Canberra"',
        'site:linkedin.com/in "supply chain manager" "NSW" "open to opportunities"',
        'site:linkedin.com/company "AI automation" "Australia" employees:1-50',
        'site:linkedin.com/in "python developer" "freelance" "Australia"',
    ],
    "github_research": [
        'site:github.com "sovereign AI" OR "edge AI" language:Python stars:>10',
        'site:github.com "MCP server" "agent" language:Python',
        'site:github.com "termux" "automation" "python"',
        'site:github.com "LangGraph" "LangChain" "fastapi" stars:>50',
    ],
    "competitors": [
        '"AI automation" "per month" "Australia" "small business" -site:amazon.com',
        'intitle:"AI agent" "no-code" "Australia" "$" site:producthunt.com',
        '"mesh" "agent swarm" "sovereign" site:substack.com OR site:medium.com',
    ],
    "freelance_gigs": [
        'site:upwork.com/jobs "Python automation" "urgent" "budget"',
        'site:fiverr.com "scraping" OR "automation" "python" "1 day delivery"',
        'site:freelancer.com.au "AI" "automation" "Australia" posted:today',
        'site:airtasker.com "developer" "python" "Albury" OR "Sydney"',
    ],
    "depin_opportunities": [
        '"helium hotspot" "monthly earnings" "Australia" 2025 OR 2026',
        '"WiGLE" "wardriving" "contribute" "points" "API"',
        '"Akash Network" "provider" "earnings" "GPU" 2025',
        '"Streamr" "data publisher" "earnings" "node"',
        '"Ocean Protocol" "data monetization" "publish" "earn"',
    ],
    "content_research": [
        'site:dev.to "LangChain" OR "LangGraph" intitle:"tutorial" OR "how to" 2025',
        'site:dev.to "termux" "python" "server" reactions:>10',
        'site:medium.com "sovereign AI" "self-hosted" "2025" claps:>100',
        'site:hackernews.com "MCP" "agent" "local" comments:>20',
    ],
    "affiliate_research": [
        '"affiliate program" "AI tools" "30%" OR "40%" OR "50%" recurring',
        '"API" "referral" "commission" "developer tools" 2025 "Australia"',
        'site:impact.com "AI" "SaaS" "commission" "monthly"',
    ],
    "osint_defensive": [
        '"fractalmesh" site:pastebin.com OR site:github.com',
        '"56628117363" -site:linkedin.com -site:fractalmesh.net',
        '"samuel hiotis" site:github.com OR site:dev.to OR site:linkedin.com',
    ],
}

def _db_init():
    c = sqlite3.connect(DB, timeout=10); c.execute("PRAGMA journal_mode=WAL")
    c.execute("""CREATE TABLE IF NOT EXISTS scrape_jobs (
        id INTEGER PRIMARY KEY, url TEXT, job_type TEXT, status TEXT, result_count INT, ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    c.execute("""CREATE TABLE IF NOT EXISTS scrape_results (
        id INTEGER PRIMARY KEY, job_id INT, url TEXT, title TEXT, snippet TEXT, content TEXT, ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    c.execute("""CREATE TABLE IF NOT EXISTS dork_results (
        id INTEGER PRIMARY KEY, category TEXT, dork TEXT, result_url TEXT, title TEXT, snippet TEXT, ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    c.commit(); c.close()

UA_LIST = [
    "Mozilla/5.0 (Linux; Android 14; SM-S918B) AppleWebKit/537.36 Chrome/120.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/121.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2) AppleWebKit/605.1 Safari/605.1",
]
_ua_idx = 0

def _ua() -> str:
    global _ua_idx; ua = UA_LIST[_ua_idx % len(UA_LIST)]; _ua_idx += 1; return ua

def _fetch_page(url: str) -> dict:
    if SCRAPER_KEY:
        url = f"http://api.scraperapi.com/?api_key={SCRAPER_KEY}&url={urllib.parse.quote(url)}"
    req = urllib.request.Request(url, headers={"User-Agent":_ua(),"Accept":"text/html,*/*"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            raw  = r.read().decode("utf-8","replace")
            title_m = re.search(r'<title[^>]*>(.*?)</title>', raw, re.I|re.S)
            title   = (title_m.group(1) if title_m else "").strip()[:200]
            text    = re.sub(r'<[^>]+>','',raw)
            text    = re.sub(r'\s+',' ',text).strip()[:5000]
            links   = re.findall(r'href=["\']([^"\']+)["\']', raw)
            return {"url":url,"title":title,"content":text,"links":links[:50],"status":"ok"}
    except Exception as e:
        return {"url":url,"error":str(e),"status":"error"}

def _google_cse(query: str, num: int = 10) -> list:
    if not CSE_KEY or not CSE_ID:
        return [{"title":"Google CSE not configured","snippet":"Set GOOGLE_CSE_API_KEY and GOOGLE_CSE_ID in vault"}]
    params = urllib.parse.urlencode({"key":CSE_KEY,"cx":CSE_ID,"q":query,"num":min(num,10)})
    req    = urllib.request.Request(f"https://www.googleapis.com/customsearch/v1?{params}")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data  = json.loads(r.read())
            return [{"title":i.get("title",""),"link":i.get("link",""),"snippet":i.get("snippet","")}
                    for i in data.get("items",[])]
    except Exception as e:
        return [{"error":str(e)}]

def _run_dork(category: str, dork_query: str) -> list:
    results = _google_cse(dork_query)
    c = sqlite3.connect(DB, timeout=10)
    for r in results:
        if "error" not in r:
            c.execute("INSERT INTO dork_results (category,dork,result_url,title,snippet) VALUES (?,?,?,?,?)",
                (category, dork_query[:200], r.get("link","")[:500], r.get("title","")[:300], r.get("snippet","")[:500]))
    c.commit(); c.close()
    log.info("dork cat=%s results=%d", category, len([r for r in results if "error" not in r]))
    return results

def _scrape_job(url: str, mode: str = "single", depth: int = 1) -> dict:
    c = sqlite3.connect(DB, timeout=5)
    c.execute("INSERT INTO scrape_jobs (url,job_type,status,result_count) VALUES (?,?,?,0)",(url,mode,"running"))
    job_id = c.lastrowid; c.commit(); c.close()

    results = []; visited = set()

    def _crawl(u, d):
        if u in visited or d < 0 or len(results) >= 50: return
        visited.add(u); page = _fetch_page(u)
        if page.get("status") == "ok":
            results.append({"url":u,"title":page["title"],"content":page["content"][:1000]})
            c2 = sqlite3.connect(DB, timeout=5)
            c2.execute("INSERT INTO scrape_results (job_id,url,title,snippet,content) VALUES (?,?,?,?,?)",
                (job_id,u,page["title"],page["content"][:300],page["content"][:2000]))
            c2.commit(); c2.close()
            if d > 0:
                base = urllib.parse.urlparse(u).netloc
                for link in page.get("links",[])[:10]:
                    if link.startswith("http") and urllib.parse.urlparse(link).netloc == base:
                        time.sleep(0.5); _crawl(link, d-1)

    if mode == "sitemap":
        sm_url = url.rstrip("/") + "/sitemap.xml"
        sm     = _fetch_page(sm_url)
        urls_in_sm = re.findall(r'<loc>(.*?)</loc>', sm.get("content",""))
        for u in urls_in_sm[:20]:
            _crawl(u, 0); time.sleep(0.3)
    else:
        _crawl(url, depth if mode == "crawl" else 0)

    c = sqlite3.connect(DB, timeout=5)
    c.execute("UPDATE scrape_jobs SET status='done',result_count=? WHERE id=?",(len(results),job_id))
    c.commit(); c.close()
    return {"job_id":job_id,"mode":mode,"pages_scraped":len(results),"results":results[:5]}

def _results_today(category: str = "", limit: int = 20) -> list:
    c = sqlite3.connect(DB, timeout=5)
    q = "SELECT category,dork,result_url,title,snippet,ts FROM dork_results WHERE ts>datetime('now','-24 hours')"
    args = []
    if category: q += " AND category=?"; args.append(category)
    q += f" ORDER BY ts DESC LIMIT {limit}"
    rows = c.execute(q, args).fetchall(); c.close()
    return [{"category":r[0],"dork":r[1],"url":r[2],"title":r[3],"snippet":r[4],"ts":r[5]} for r in rows]

class ScraperHandler(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def _r(self,code,body):
        p=json.dumps(body).encode(); self.send_response(code)
        self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",str(len(p))); self.end_headers(); self.wfile.write(p)

    def do_GET(self):
        import urllib.parse
        qs=urllib.parse.parse_qs(self.path.split("?",1)[-1] if "?" in self.path else "")
        ep=self.path.split("?")[0]
        if ep=="/health":
            c=sqlite3.connect(DB,timeout=5)
            nd=c.execute("SELECT COUNT(*) FROM dork_results WHERE ts>datetime('now','-24 hours')").fetchone()[0]
            c.close(); self._r(200,{"status":"ok","dorks":sum(len(v) for v in DORKS.values()),"results_today":nd,"cse_configured":bool(CSE_KEY)})
        elif ep=="/dorks":
            self._r(200,{cat:[{"query":d} for d in dorks] for cat,dorks in DORKS.items()})
        elif ep=="/results":
            cat=qs.get("category",[""])[0]; limit=int(qs.get("limit",[20])[0])
            self._r(200,{"results":_results_today(cat,limit)})
        elif ep=="/export":
            c=sqlite3.connect(DB,timeout=5)
            rows=c.execute("SELECT * FROM dork_results ORDER BY ts DESC LIMIT 500").fetchall()
            c.close(); self._r(200,{"count":len(rows),"results":rows})
        else:
            self._r(404,{"error":"not_found"})

    def do_POST(self):
        try:
            n=int(self.headers.get("Content-Length",0)); d=json.loads(self.rfile.read(n)) if n else {}
            ep=self.path.split("?")[0]
            if ep=="/scrape":
                url=d.get("url",""); mode=d.get("mode","single"); depth=int(d.get("depth",1))
                self._r(200,_scrape_job(url,mode,depth))
            elif ep=="/dork":
                key=d.get("dork_key",""); cat=key.split(".")[0] if "." in key else list(DORKS.keys())[0]
                idx=int(key.split(".")[-1]) if "." in key else 0
                dork_list=DORKS.get(cat,[])
                if not dork_list: self._r(404,{"error":"category not found","available":list(DORKS.keys())}); return
                dork=dork_list[idx % len(dork_list)]
                self._r(200,{"category":cat,"dork":dork,"results":_run_dork(cat,dork)})
            elif ep=="/dork/batch":
                cat=d.get("category",""); dorks=DORKS.get(cat,[])
                all_results=[]
                for dork in dorks:
                    all_results.extend(_run_dork(cat,dork)); time.sleep(1)
                self._r(200,{"category":cat,"dorks_run":len(dorks),"results":len(all_results)})
            elif ep=="/dork/all":
                total=0
                for cat,dorks in DORKS.items():
                    for dork in dorks:
                        _run_dork(cat,dork); total+=1; time.sleep(2)
                self._r(200,{"dorks_run":total,"categories":list(DORKS.keys())})
            else:
                self._r(404,{"error":"unknown_path"})
        except json.JSONDecodeError: self._r(400,{"error":"invalid_json"})
        except Exception as e: self._r(500,{"error":str(e)})

_running=True
def _shutdown(*_): global _running; _running=False
signal.signal(signal.SIGTERM,_shutdown); signal.signal(signal.SIGINT,_shutdown)

def main():
    _db_init(); server=HTTPServer(("0.0.0.0",PORT),ScraperHandler)
    total_dorks=sum(len(v) for v in DORKS.values())
    log.info("Scraper v2 on port %d | dorks=%d | cse=%s",PORT,total_dorks,bool(CSE_KEY))
    try:
        while _running: server.handle_request()
    finally: server.server_close()

if __name__=="__main__": main()
