"""
FractalMesh OSINT Spider Agent
Google Dorking + analytics-grade lead discovery; dry-run safe; logs to sovereign.db
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import re
import json
import time
import signal
import sqlite3
import hashlib
import urllib.request
import urllib.parse
from datetime import datetime

ROOT     = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB       = os.path.join(ROOT, "database", "sovereign.db")
INTERVAL = int(os.getenv("OSINT_SPIDER_INTERVAL", "7200"))
DRY_RUN  = os.getenv("ENABLE_OSINT_SPIDER", "false").lower() != "true"
CSE_KEY  = os.getenv("GOOGLE_CSE_API_KEY", "")
CSE_ID   = os.getenv("GOOGLE_CSE_ID", "")

# ── Dork template library ──────────────────────────────────────────────────────
DORK_LIBRARY = {
    "logistics_albury": [
        '"logistics manager" "Albury" site:linkedin.com/in',
        '"supply chain" "Albury NSW" site:linkedin.com/company',
        '"freight" "Albury" contact email',
    ],
    "compliance_infra": [
        '"NCC 2025" "compliance" "infrastructure" filetype:pdf',
        '"SOC 2" "automation" "sovereign" OR "self-hosted"',
        '"AFSL" "automation" "compliance" "small business" site:linkedin.com',
    ],
    "depin_australia": [
        '"helium" OR "XYO" "node operator" "Australia"',
        '"DePIN" "Australia" "deploy" filetype:pdf OR site:medium.com',
        '"Raspberry Pi" "earn" "crypto" "Australia" site:reddit.com',
    ],
    "saas_targets": [
        '"automation" "SaaS" "NSW" "small business" site:linkedin.com',
        '"no-code" OR "low-code" "NSW" "consulting" email',
        '"AI automation" "business" "Albury" OR "Wagga Wagga" OR "Wodonga"',
    ],
    "ready_to_buy": [
        '"looking for" "automation" "quote" OR "proposal" "infrastructure"',
        '"we need" "automation" "compliance" "ready to deploy"',
        '"hire" "AI" "infrastructure" "Albury" OR "NSW" site:seek.com.au',
    ],
    "analytics_signals": [
        'site:fractalmesh.net',
        '"fractalmesh" site:twitter.com OR site:x.com',
        '"fractalmesh" site:reddit.com OR site:hackernews.com',
    ],
}

_running = True


def _db_init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS osint_log (
        id INTEGER PRIMARY KEY, category TEXT, query_hash TEXT UNIQUE,
        raw_query TEXT, result_count INTEGER, top_urls TEXT,
        source TEXT, ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS leads (
        id INTEGER PRIMARY KEY, industry TEXT, region TEXT,
        query_hash TEXT UNIQUE, raw_query TEXT, result_count INTEGER,
        source TEXT, ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()


def _qhash(q: str) -> str:
    return hashlib.sha256(q.encode()).hexdigest()[:16]


def _cse_search(query: str) -> dict:
    if not CSE_KEY or not CSE_ID or DRY_RUN:
        return {"total": 0, "urls": []}
    try:
        params = urllib.parse.urlencode({
            "key": CSE_KEY, "cx": CSE_ID, "q": query, "num": 5})
        with urllib.request.urlopen(
                f"https://www.googleapis.com/customsearch/v1?{params}", timeout=10) as r:
            data    = json.loads(r.read())
            total   = int(data.get("searchInformation", {}).get("totalResults", 0))
            urls    = [i.get("link", "") for i in data.get("items", [])]
            return {"total": total, "urls": urls}
    except Exception as e:
        return {"total": -1, "urls": [], "error": str(e)}


def _upsert_osint(category, query, result):
    qh   = _qhash(query)
    urls = json.dumps(result.get("urls", [])[:5])
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("""INSERT INTO osint_log (category,query_hash,raw_query,result_count,top_urls,source)
        VALUES (?,?,?,?,?,?)
        ON CONFLICT(query_hash) DO UPDATE
        SET result_count=excluded.result_count, top_urls=excluded.top_urls, ts=CURRENT_TIMESTAMP""",
        (category, qh, query[:300], result.get("total", 0), urls, "osint_spider"))
    conn.execute("""INSERT INTO leads (industry,region,query_hash,raw_query,result_count,source)
        VALUES (?,?,?,?,?,?)
        ON CONFLICT(query_hash) DO UPDATE
        SET result_count=excluded.result_count, ts=CURRENT_TIMESTAMP""",
        (category, "albury_nsw", qh, query[:200], result.get("total", 0), "osint_spider"))
    conn.commit(); conn.close()


def _analytics_fingerprint(urls: list) -> dict:
    """Basic pattern analysis on result URLs."""
    domains    = {}
    tlds       = {}
    for url in urls:
        try:
            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc
            tld    = domain.rsplit(".", 1)[-1] if "." in domain else domain
            domains[domain] = domains.get(domain, 0) + 1
            tlds[tld]       = tlds.get(tld, 0) + 1
        except Exception:
            pass
    return {
        "unique_domains":  len(domains),
        "top_domains":     sorted(domains.items(), key=lambda x: x[1], reverse=True)[:5],
        "tld_breakdown":   tlds,
    }


def run_cycle():
    ts      = datetime.utcnow().isoformat()
    total_q = sum(len(v) for v in DORK_LIBRARY.values())
    print(f"[fm-osint-spider] {ts} | {len(DORK_LIBRARY)} categories | {total_q} dorks | dry={DRY_RUN}")

    all_urls = []
    for category, queries in DORK_LIBRARY.items():
        print(f"   ── {category}")
        for q in queries:
            result = _cse_search(q)
            all_urls.extend(result.get("urls", []))
            count_str = "dry" if DRY_RUN else str(result.get("total", 0))
            err       = result.get("error", "")
            print(f"      [{count_str:>8}] {q[:70]}" + (f" ERR:{err[:30]}" if err else ""))
            _upsert_osint(category, q, result)
            if not DRY_RUN:
                time.sleep(0.5)   # rate-limit CSE calls

    if all_urls:
        fp = _analytics_fingerprint(all_urls)
        print(f"   Analytics: {fp['unique_domains']} unique domains | TLDs: {fp['tld_breakdown']}")


def _sigterm(s, f):
    global _running
    _running = False


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    _db_init()
    print(f"[fm-osint-spider] Active | interval={INTERVAL}s | dry={DRY_RUN} | CSE={'set' if CSE_KEY else 'NOT SET'}")
    while _running:
        try:
            run_cycle()
        except Exception as e:
            print(f"[fm-osint-spider] ERR {e}")
        for _ in range(INTERVAL):
            if not _running: break
            time.sleep(1)
    print("[fm-osint-spider] Stopped.")
