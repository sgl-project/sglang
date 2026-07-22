"""
FractalMesh Dork Engine Agent
OSINT lead-discovery via structured search dorks; logs to sovereign.db leads table
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
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
INTERVAL = int(os.getenv("DORK_ENGINE_INTERVAL", "7200"))
DRY_RUN  = os.getenv("ENABLE_DORK_ENGINE", "false").lower() != "true"
CSE_KEY  = os.getenv("GOOGLE_CSE_API_KEY", "")
CSE_ID   = os.getenv("GOOGLE_CSE_ID", "")

DORKS = [
    {"industry": "logistics",    "region": "Albury NSW",      "query": 'site:linkedin.com/in "logistics manager" "Albury"'},
    {"industry": "logistics",    "region": "Wagga Wagga NSW",  "query": 'site:linkedin.com/in "supply chain" "Wagga Wagga"'},
    {"industry": "construction", "region": "Albury NSW",       "query": 'site:linkedin.com/company "construction" "Albury NSW"'},
    {"industry": "compliance",   "region": "Australia",        "query": '"NCC 2025" "compliance" "infrastructure" filetype:pdf'},
    {"industry": "depin",        "region": "Australia",        "query": '"helium" OR "XYO" "node operator" "Australia"'},
    {"industry": "saas",         "region": "NSW",              "query": '"automation" "SaaS" "small business" "NSW" site:linkedin.com'},
]

_running = True


def _db_init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS leads (
        id INTEGER PRIMARY KEY, industry TEXT, region TEXT,
        query_hash TEXT UNIQUE, raw_query TEXT, result_count INTEGER,
        source TEXT, ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()


def _qhash(q: str) -> str:
    return hashlib.sha256(q.encode()).hexdigest()[:16]


def _search(dork: dict) -> int:
    if not CSE_KEY or not CSE_ID or DRY_RUN:
        return 0
    try:
        params = urllib.parse.urlencode({"key": CSE_KEY, "cx": CSE_ID, "q": dork["query"], "num": 1})
        with urllib.request.urlopen(
                f"https://www.googleapis.com/customsearch/v1?{params}", timeout=10) as r:
            return int(json.loads(r.read()).get("searchInformation", {}).get("totalResults", 0))
    except Exception:
        return -1


def _upsert(dork: dict, count: int):
    qh   = _qhash(dork["query"])
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("""INSERT INTO leads (industry,region,query_hash,raw_query,result_count,source)
        VALUES (?,?,?,?,?,?) ON CONFLICT(query_hash) DO UPDATE
        SET result_count=excluded.result_count, ts=CURRENT_TIMESTAMP""",
        (dork["industry"], dork["region"], qh, dork["query"][:200], count, "dork_engine"))
    conn.commit(); conn.close()


def run_cycle():
    print(f"[fm-dork-engine] {datetime.utcnow().isoformat()} | {len(DORKS)} dorks | dry={DRY_RUN}")
    for dork in DORKS:
        count = _search(dork)
        print(f"   → [{dork['industry']:<12}] {dork['region']:<18} hits={'dry' if DRY_RUN else count}")
        _upsert(dork, count)


def _sigterm(s, f):
    global _running
    _running = False


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    _db_init()
    print(f"[fm-dork-engine] Active | interval={INTERVAL}s | dry={DRY_RUN}")
    while _running:
        try:
            run_cycle()
        except Exception as e:
            print(f"[fm-dork-engine] ERR {e}")
        for _ in range(INTERVAL):
            if not _running: break
            time.sleep(1)
    print("[fm-dork-engine] Stopped.")
