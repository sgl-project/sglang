#!/usr/bin/env python3
"""
fm_rss_hub.py — Multi-RSS Feed Aggregator & Publisher (Port 7805)
Fetches 20 feeds, stores items, generates digest, exports RSS.
Samuel James Hiotis | ABN 56 628 117 363
"""
import os, json, time, signal, sqlite3, logging, urllib.request, urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _l in _vault.read_text().splitlines():
        if "=" in _l and not _l.startswith("#"):
            _k, _, _v = _l.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

PORT     = int(os.getenv("RSS_PORT", "7805"))
ROOT     = Path(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")))
DB       = ROOT / "database" / "sovereign.db"
LOG      = ROOT / "logs" / "rss_hub.log"
OR_URL   = os.getenv("OPENROUTER_URL", "http://127.0.0.1:7791")
FETCH_IV = int(os.getenv("RSS_FETCH_INTERVAL", "3600"))  # 1 hr

for p in (ROOT, LOG.parent, DB.parent): p.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [RSS] %(message)s",
    handlers=[logging.FileHandler(LOG), logging.StreamHandler()])
log = logging.getLogger("rss_hub")

DEFAULT_FEEDS = [
    ("O'Reilly Radar",      "https://feeds.feedburner.com/oreilly/radar",        "tech"),
    ("The Verge",           "https://www.theverge.com/rss/index.xml",            "tech"),
    ("TechCrunch",          "https://techcrunch.com/feed/",                      "tech"),
    ("Ars Technica",        "https://feeds.arstechnica.com/arstechnica/index",   "tech"),
    ("Hacker News",         "https://news.ycombinator.com/rss",                  "tech"),
    ("Product Hunt",        "https://www.producthunt.com/feed",                  "startup"),
    ("CoinDesk",            "https://www.coindesk.com/arc/outboundfeeds/rss/",   "crypto"),
    ("CoinTelegraph",       "https://cointelegraph.com/rss",                     "crypto"),
    ("Decrypt",             "https://decrypt.co/feed",                           "crypto"),
    ("Dev.to",              "https://dev.to/feed",                               "dev"),
    ("InfoQ",               "https://www.infoq.com/feed/",                       "dev"),
    ("VentureBeat",         "https://venturebeat.com/feed/",                     "ai"),
    ("LangChain Blog",      "https://blog.langchain.dev/rss/",                   "ai"),
    ("HuggingFace Blog",    "https://huggingface.co/blog/feed.xml",              "ai"),
    ("OpenAI Blog",         "https://openai.com/blog/rss/",                      "ai"),
    ("Entrepreneur",        "https://feeds.feedburner.com/entrepreneur/latest",  "business"),
    ("Smart Company",       "https://www.smartcompany.com.au/feed/",             "business"),
    ("AFR",                 "https://www.afr.com/rss",                           "business"),
    ("TechGeek AU",         "https://techgeek.com.au/feed/",                     "au_tech"),
    ("FractalMesh",         "https://fractalmesh.net/feed.xml",                  "own"),
]

def _db_init():
    c = sqlite3.connect(DB, timeout=10); c.execute("PRAGMA journal_mode=WAL")
    c.execute("""CREATE TABLE IF NOT EXISTS rss_feeds (
        id INTEGER PRIMARY KEY, url TEXT UNIQUE, name TEXT, category TEXT, active INT DEFAULT 1, ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    c.execute("""CREATE TABLE IF NOT EXISTS rss_items (
        id INTEGER PRIMARY KEY, feed_id INT, guid TEXT UNIQUE, title TEXT, link TEXT,
        summary TEXT, published TEXT, processed INT DEFAULT 0, ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    # Seed default feeds
    conn = sqlite3.connect(DB, timeout=10)
    for name, url, cat in DEFAULT_FEEDS:
        conn.execute("INSERT OR IGNORE INTO rss_feeds (url,name,category) VALUES (?,?,?)",(url,name,cat))
    conn.commit(); conn.close()

def _feeds_list() -> list:
    c = sqlite3.connect(DB, timeout=5)
    rows = c.execute("SELECT id,url,name,category,active FROM rss_feeds WHERE active=1").fetchall()
    c.close(); return [{"id":r[0],"url":r[1],"name":r[2],"category":r[3]} for r in rows]

def _parse_feed(xml_text: str, feed_id: int) -> list:
    items = []
    try:
        root = ET.fromstring(xml_text)
        ns   = {"atom":"http://www.w3.org/2005/Atom"}
        # RSS 2.0
        for item in root.findall(".//item"):
            g    = (item.findtext("guid") or item.findtext("link") or "")[:500]
            t    = (item.findtext("title") or "")[:300]
            l    = (item.findtext("link")  or "")[:500]
            s    = (item.findtext("description") or "")[:1000]
            pub  = (item.findtext("pubDate") or item.findtext("dc:date","",{"dc":"http://purl.org/dc/elements/1.1/"}) or "")[:100]
            if g: items.append({"feed_id":feed_id,"guid":g,"title":t,"link":l,"summary":s,"published":pub})
        # Atom
        if not items:
            for entry in root.findall(".//atom:entry",ns):
                g = (entry.findtext("atom:id",namespaces=ns) or "")[:500]
                t = (entry.findtext("atom:title",namespaces=ns) or "")[:300]
                l = ""
                for link in entry.findall("atom:link",ns):
                    if link.get("rel","alternate") == "alternate": l = link.get("href","")
                s   = (entry.findtext("atom:summary",namespaces=ns) or entry.findtext("atom:content",namespaces=ns) or "")[:1000]
                pub = (entry.findtext("atom:published",namespaces=ns) or "")[:100]
                if g: items.append({"feed_id":feed_id,"guid":g,"title":t,"link":l,"summary":s,"published":pub})
    except Exception as e: log.warning("parse_feed: %s",e)
    return items

def _fetch_feed(url: str, feed_id: int) -> int:
    headers = {"User-Agent":"FractalMesh-RSSHub/2.1 (+https://fractalmesh.net)","Accept":"application/rss+xml,application/xml,text/xml"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            xml_text = r.read().decode("utf-8","replace")
    except Exception as e:
        log.warning("fetch %s: %s",url,e); return 0
    items = _parse_feed(xml_text, feed_id)
    saved = 0
    c = sqlite3.connect(DB, timeout=10)
    for item in items:
        try:
            c.execute("INSERT OR IGNORE INTO rss_items (feed_id,guid,title,link,summary,published) VALUES (?,?,?,?,?,?)",
                (item["feed_id"],item["guid"],item["title"],item["link"],item["summary"],item["published"]))
            if c.rowcount > 0: saved += 1
        except Exception: pass
    c.commit(); c.close(); return saved

def _fetch_all() -> dict:
    feeds = _feeds_list(); total = 0
    for f in feeds:
        n = _fetch_feed(f["url"], f["id"]); total += n
        log.info("feed=%s new=%d", f["name"], n)
        time.sleep(0.5)
    return {"fetched_feeds":len(feeds),"new_items":total}

def _digest(limit=10) -> list:
    c = sqlite3.connect(DB, timeout=5)
    rows = c.execute("""
        SELECT i.title, i.link, i.summary, f.name, f.category, i.published
        FROM rss_items i JOIN rss_feeds f ON i.feed_id=f.id
        WHERE i.ts > datetime('now','-24 hours')
        ORDER BY i.ts DESC LIMIT ?""",(limit,)).fetchall()
    c.close()
    return [{"title":r[0],"link":r[1],"summary":r[2][:200],"feed":r[3],"category":r[4],"published":r[5]} for r in rows]

def _export_rss() -> str:
    items = _digest(50)
    entries = "".join(f"""<item>
  <title><![CDATA[{i['title']}]]></title>
  <link>{i['link']}</link>
  <description><![CDATA[{i['summary']}]]></description>
  <source>{i['feed']}</source>
  <pubDate>{i['published']}</pubDate>
</item>""" for i in items)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <title>FractalMesh RSS Digest</title>
  <link>https://fractalmesh.net</link>
  <description>Curated AI/tech/crypto/business feed — Samuel James Hiotis | ABN 56628117363</description>
  {entries}
</channel></rss>"""

def _generate_digest_nl() -> str:
    items = _digest(10)
    if not items: return "No items in last 24h."
    bullet_list = "\n".join(f"- [{i['title']}]({i['link']}) — {i['feed']}" for i in items)
    prompt = f"Summarise this news digest into a 3-paragraph newsletter for tech entrepreneurs:\n\n{bullet_list}"
    payload = json.dumps({"task":"summarize","tier":"balanced","prompt":prompt,"max_tokens":600}).encode()
    req = urllib.request.Request(f"{OR_URL}/route",data=payload,headers={"Content-Type":"application/json"},method="POST")
    try:
        with urllib.request.urlopen(req,timeout=30) as r: return json.loads(r.read()).get("content","")
    except Exception as e: return f"[digest gen failed: {e}]\n\n{bullet_list}"

class RSSHandler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _r(self, code, body, ctype="application/json"):
        p = body.encode() if isinstance(body,str) else json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type",ctype)
        self.send_header("Content-Length",str(len(p))); self.end_headers(); self.wfile.write(p)

    def do_GET(self):
        import urllib.parse
        qs = urllib.parse.parse_qs(self.path.split("?",1)[-1] if "?" in self.path else "")
        ep = self.path.split("?")[0]
        if ep == "/health":
            c = sqlite3.connect(DB,timeout=5)
            fc = c.execute("SELECT COUNT(*) FROM rss_feeds WHERE active=1").fetchone()[0]
            ic = c.execute("SELECT COUNT(*) FROM rss_items WHERE ts>datetime('now','-24 hours')").fetchone()[0]
            c.close(); self._r(200,{"status":"ok","feeds":fc,"items_today":ic})
        elif ep == "/feeds":
            c = sqlite3.connect(DB,timeout=5)
            rows = c.execute("SELECT f.id,f.name,f.category,COUNT(i.id) as cnt FROM rss_feeds f LEFT JOIN rss_items i ON i.feed_id=f.id GROUP BY f.id").fetchall()
            c.close(); self._r(200,{"feeds":[{"id":r[0],"name":r[1],"category":r[2],"items":r[3]} for r in rows]})
        elif ep == "/items":
            cat = qs.get("category",[""])[0]; limit = int(qs.get("limit",[20])[0])
            c = sqlite3.connect(DB,timeout=5)
            q = "SELECT i.title,i.link,i.summary,f.name,i.published FROM rss_items i JOIN rss_feeds f ON i.feed_id=f.id"
            args = []
            if cat: q += " WHERE f.category=?"; args.append(cat)
            q += " ORDER BY i.ts DESC LIMIT ?"; args.append(limit)
            rows = c.execute(q,args).fetchall(); c.close()
            self._r(200,{"items":[{"title":r[0],"link":r[1],"summary":r[2][:200],"feed":r[3],"published":r[4]} for r in rows]})
        elif ep == "/digest":
            self._r(200,{"digest":_digest()})
        elif ep == "/export":
            self._r(200,_export_rss(),"application/rss+xml")
        else:
            self._r(404,{"error":"not_found"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length",0)); d = json.loads(self.rfile.read(n)) if n else {}
            ep = self.path.split("?")[0]
            if ep == "/fetch":
                self._r(200,_fetch_all())
            elif ep == "/add_feed":
                url = d.get("url",""); name = d.get("name",url); cat = d.get("category","general")
                c = sqlite3.connect(DB,timeout=5)
                c.execute("INSERT OR IGNORE INTO rss_feeds (url,name,category) VALUES (?,?,?)",(url,name,cat))
                c.commit(); c.close(); self._r(200,{"added":url})
            elif ep == "/generate_digest":
                self._r(200,{"newsletter":_generate_digest_nl()})
            elif ep == "/process_item":
                item_id = d.get("item_id")
                if item_id:
                    c = sqlite3.connect(DB,timeout=5)
                    c.execute("UPDATE rss_items SET processed=1 WHERE id=?",(item_id,)); c.commit(); c.close()
                self._r(200,{"processed":item_id})
            else:
                self._r(404,{"error":"unknown_path"})
        except json.JSONDecodeError: self._r(400,{"error":"invalid_json"})
        except Exception as e: self._r(500,{"error":str(e)})

_running = True; _last_fetch = 0
def _shutdown(*_): global _running; _running = False
signal.signal(signal.SIGTERM, _shutdown); signal.signal(signal.SIGINT, _shutdown)

def main():
    _db_init(); server = HTTPServer(("0.0.0.0", PORT), RSSHandler)
    log.info("RSS Hub on port %d | feeds=%d | interval=%ds", PORT, len(DEFAULT_FEEDS), FETCH_IV)
    global _last_fetch
    try:
        while _running:
            server.handle_request()
            if time.time() - _last_fetch >= FETCH_IV:
                r = _fetch_all(); log.info("auto_fetch: %s", r); _last_fetch = time.time()
    finally: server.server_close()

if __name__ == "__main__": main()
