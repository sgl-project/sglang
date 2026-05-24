#!/usr/bin/env python3
"""
fm_lighthouse.py — Lighthouse / PageSpeed Insights Auditing Agent (Port 7799)
Audits FractalMesh pages for performance, accessibility, SEO, best practices.
Samuel James Hiotis | ABN 56 628 117 363
"""
import os, json, time, signal, sqlite3, logging, urllib.request, urllib.parse, subprocess
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _l in _vault.read_text().splitlines():
        if "=" in _l and not _l.startswith("#"):
            _k, _, _v = _l.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

PORT      = int(os.getenv("LIGHTHOUSE_PORT", "7799"))
ROOT      = Path(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")))
DB        = ROOT / "database" / "sovereign.db"
LOG       = ROOT / "logs" / "lighthouse.log"
PSI_KEY   = os.getenv("GOOGLE_PSI_KEY", "")
PSI_URL   = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
SITE_URLS = json.loads(os.getenv("LIGHTHOUSE_URLS", "[]")) or [
    "https://fractalmesh.net",
    "https://fractalmesh.net/strategies.html",
    "https://fractalmesh.net/products.html",
    "https://fractalmesh.net/blueprint.html",
    "https://fractalmesh.net/dashboard.html",
]
AUDIT_INTERVAL = int(os.getenv("LIGHTHOUSE_INTERVAL", "21600"))  # 6 hr

for p in (ROOT, LOG.parent, DB.parent): p.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [LIGHTHOUSE] %(message)s",
    handlers=[logging.FileHandler(LOG), logging.StreamHandler()])
log = logging.getLogger("lighthouse")

def _db_init():
    c = sqlite3.connect(DB, timeout=10); c.execute("PRAGMA journal_mode=WAL")
    c.execute("""CREATE TABLE IF NOT EXISTS lighthouse_audits (
        id INTEGER PRIMARY KEY, url TEXT, strategy TEXT,
        performance REAL, accessibility REAL, seo REAL, best_practices REAL,
        lcp REAL, fid REAL, cls REAL, ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    c.commit(); c.close()

def _db_save(url, strategy, scores: dict, cwv: dict):
    try:
        c = sqlite3.connect(DB, timeout=5)
        c.execute("INSERT INTO lighthouse_audits (url,strategy,performance,accessibility,seo,best_practices,lcp,fid,cls) VALUES (?,?,?,?,?,?,?,?,?)",
            (url, strategy, scores.get("performance"), scores.get("accessibility"),
             scores.get("seo"), scores.get("best-practices"),
             cwv.get("lcp"), cwv.get("fid"), cwv.get("cls")))
        c.commit(); c.close()
    except Exception as e: log.warning("db: %s", e)

def _score_color(s):
    if s is None: return "gray"
    if s >= 0.9:  return "green"
    if s >= 0.5:  return "amber"
    return "red"

def _psi_audit(url: str, strategy: str = "mobile") -> dict:
    params = urllib.parse.urlencode({"url": url, "strategy": strategy,
        "category": ["performance","accessibility","best-practices","seo"],
        **({"key": PSI_KEY} if PSI_KEY else {})}, doseq=True)
    req = urllib.request.Request(f"{PSI_URL}?{params}", headers={"Accept":"application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r: return json.loads(r.read())
    except Exception as e: return {"error": str(e)}

def _local_lighthouse(url: str) -> dict:
    try:
        out = subprocess.getoutput(f"lighthouse '{url}' --output=json --output-path=- --quiet --chrome-flags='--headless' 2>/dev/null")
        return json.loads(out) if out.strip().startswith("{") else {"error": "lighthouse_not_available"}
    except Exception: return {"error": "lighthouse_not_installed"}

def _parse_psi(result: dict) -> dict:
    cats = result.get("lighthouseResult", {}).get("categories", {})
    audits = result.get("lighthouseResult", {}).get("audits", {})
    scores = {k: cats[k].get("score") for k in cats}
    cwv = {
        "lcp": audits.get("largest-contentful-paint",{}).get("numericValue"),
        "fid": audits.get("max-potential-fid",{}).get("numericValue"),
        "cls": audits.get("cumulative-layout-shift",{}).get("numericValue"),
        "fcp": audits.get("first-contentful-paint",{}).get("numericValue"),
        "si":  audits.get("speed-index",{}).get("numericValue"),
        "tti": audits.get("interactive",{}).get("numericValue"),
    }
    return scores, cwv

def _run_audit(url: str, strategy: str = "mobile") -> dict:
    t0     = time.time()
    result = _psi_audit(url, strategy)
    if "error" in result:
        result = _local_lighthouse(url)
    if "error" in result:
        return {"url": url, "strategy": strategy, "error": result["error"]}
    scores, cwv = _parse_psi(result)
    _db_save(url, strategy, scores, cwv)
    latency = (time.time()-t0)*1000
    log.info("audit url=%s strategy=%s perf=%.0f%% latency=%.0fms",
             url, strategy, (scores.get("performance") or 0)*100, latency)
    return {"url": url, "strategy": strategy, "scores": scores, "cwv": cwv, "latency_ms": round(latency,1)}

def _batch_audit() -> list:
    results = []
    for url in SITE_URLS:
        results.append(_run_audit(url, "mobile"))
        time.sleep(1)
    return results

def _latest_scores() -> list:
    try:
        c = sqlite3.connect(DB, timeout=5)
        rows = c.execute("""
            SELECT url, strategy, performance, accessibility, seo, best_practices, lcp, cls, ts
            FROM lighthouse_audits a WHERE ts = (SELECT MAX(ts) FROM lighthouse_audits b WHERE b.url=a.url AND b.strategy=a.strategy)
            ORDER BY url""").fetchall()
        c.close()
        return [{"url":r[0],"strategy":r[1],"performance":r[2],"accessibility":r[3],
                 "seo":r[4],"best_practices":r[5],"lcp_ms":r[6],"cls":r[7],"ts":r[8]} for r in rows]
    except Exception: return []

def _html_report() -> str:
    scores = _latest_scores()
    rows   = ""
    for s in scores:
        def badge(v): return f'<span style="color:{_score_color(v)}">{int((v or 0)*100)}</span>'
        rows += f"<tr><td>{s['url']}</td><td>{s['strategy']}</td><td>{badge(s['performance'])}</td><td>{badge(s['accessibility'])}</td><td>{badge(s['seo'])}</td><td>{badge(s['best_practices'])}</td><td>{s['ts'][:16]}</td></tr>"
    return f"""<!DOCTYPE html><html><head><title>Lighthouse Report</title>
<style>body{{background:#0a0a0f;color:#e2e8f0;font-family:monospace;padding:20px}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #333;padding:8px;text-align:left}}
th{{background:#111827}}tr:nth-child(even){{background:#0d1117}}</style></head><body>
<h2 style="color:#22d3ee">FractalMesh Lighthouse Report</h2>
<table><tr><th>URL</th><th>Strategy</th><th>Perf</th><th>A11y</th><th>SEO</th><th>BP</th><th>Updated</th></tr>
{rows}</table></body></html>"""

class LHHandler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _r(self, code, body, ctype="application/json"):
        p = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(p))); self.end_headers(); self.wfile.write(p)

    def do_GET(self):
        qs = urllib.parse.parse_qs(self.path.split("?",1)[-1] if "?" in self.path else "")
        ep = self.path.split("?")[0]
        if ep == "/health":
            self._r(200, {"status":"ok","psi_key":bool(PSI_KEY),"pages":len(SITE_URLS)})
        elif ep == "/scores":
            self._r(200, {"scores": _latest_scores()})
        elif ep == "/report":
            self._r(200, _html_report(), "text/html")
        elif ep == "/audit":
            url = qs.get("url",[""])[0]; strategy = qs.get("strategy",["mobile"])[0]
            self._r(200, _run_audit(url, strategy) if url else {"error":"url required"})
        elif ep == "/history":
            url = qs.get("url",[""])[0]
            try:
                c = sqlite3.connect(DB, timeout=5)
                rows = c.execute("SELECT * FROM lighthouse_audits WHERE url=? ORDER BY ts DESC LIMIT 30",(url,)).fetchall()
                c.close(); self._r(200, {"url":url,"count":len(rows),"history":rows})
            except Exception as e: self._r(500,{"error":str(e)})
        else:
            self._r(404, {"error":"not_found"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length",0)); d = json.loads(self.rfile.read(n))
            ep = self.path.split("?")[0]
            if ep == "/audit":
                self._r(200, _run_audit(d.get("url",""), d.get("strategy","mobile")))
            elif ep == "/batch":
                self._r(200, {"results": _batch_audit()})
            else:
                self._r(404, {"error":"unknown_path"})
        except json.JSONDecodeError: self._r(400, {"error":"invalid_json"})
        except Exception as e: self._r(500, {"error":str(e)})

_running = True; _last_audit = 0
def _shutdown(*_): global _running; _running = False
signal.signal(signal.SIGTERM, _shutdown); signal.signal(signal.SIGINT, _shutdown)

def main():
    _db_init(); server = HTTPServer(("0.0.0.0", PORT), LHHandler)
    log.info("Lighthouse agent on port %d | pages=%d | psi_key=%s", PORT, len(SITE_URLS), bool(PSI_KEY))
    global _last_audit
    try:
        while _running:
            server.handle_request()
            if time.time() - _last_audit >= AUDIT_INTERVAL:
                log.info("auto_batch_audit start"); _batch_audit(); _last_audit = time.time()
    finally: server.server_close()

if __name__ == "__main__": main()
