#!/usr/bin/env python3
"""
FractalMesh Dorking — OSINT Lead Discovery Engine
Samuel James Hiotis | ABN 56628117363 | Albury NSW
Integrates: WiGLE WiFi API, Google Custom Search (OSINT dorking),
            local business discovery, lead scoring → sovereign.db
Runs every 15 min. Zero-capital monetization via pipeline enrichment.
"""
import os, json, time, sqlite3, logging, urllib.request, urllib.parse
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [DORKING] %(message)s")
log = logging.getLogger("dorking")

ROOT  = os.environ.get("FRACTALMESH_HOME", str(Path.home() / "fmsaas"))
VAULT = os.path.join(ROOT, ".env")
DB    = os.path.join(ROOT, "db", "sovereign.db")

def load_env(key, default=""):
    for f in [VAULT, str(Path.home() / ".env"), str(Path.home() / ".secrets/fractal.env")]:
        try:
            for line in Path(f).read_text().splitlines():
                s = line.strip()
                if s.startswith(key + "=") and not s.startswith("#"):
                    val = s.split("=",1)[1].strip().strip('"').strip("'")
                    if val and not val.startswith("YOUR_"):
                        return val
        except Exception:
            pass
    return os.environ.get(key, default)

def get_db():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn

# ─── WiGLE API ────────────────────────────────────────────────────────────────

def wigle_search_area(lat=-36.08, lon=146.92, distance=5):
    """Search WiGLE for WiFi networks near Albury NSW — returns network count + density."""
    api_key = load_env("WIGLE_API_KEY")
    if not api_key:
        log.info("WiGLE: no API key — using public stats endpoint")
        try:
            with urllib.request.urlopen("https://api.wigle.net/api/v2/stats/general", timeout=6) as r:
                data = json.loads(r.read())
                return {
                    "networks_worldwide": data.get("statistics",{}).get("totalWiFi",0),
                    "source": "WiGLE public stats",
                    "lat": lat, "lon": lon,
                }
        except Exception as e:
            return {"error":str(e),"source":"WiGLE (no key)"}
    try:
        params = urllib.parse.urlencode({
            "latrange1": lat - distance*0.009,
            "latrange2": lat + distance*0.009,
            "longrange1": lon - distance*0.013,
            "longrange2": lon + distance*0.013,
            "resultsPerPage": 25,
        })
        req = urllib.request.Request(
            f"https://api.wigle.net/api/v2/network/search?{params}",
            headers={"Authorization": f"Basic {api_key}", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read())
        networks = data.get("results",[])
        business_networks = [
            n for n in networks
            if any(t in n.get("ssid","").lower() for t in
                   ["business","office","cafe","hotel","bank","hospital","school","council"])
        ]
        return {
            "total_networks": data.get("totalResults", len(networks)),
            "business_networks": len(business_networks),
            "ssid_samples": [n.get("ssid","") for n in business_networks[:5]],
            "lat": lat, "lon": lon, "radius_km": distance,
            "source": "WiGLE API v2",
        }
    except Exception as e:
        log.warning("WiGLE error: %s", e)
        return {"error":str(e),"source":"WiGLE API"}

# ─── Google Custom Search OSINT ───────────────────────────────────────────────

DORK_QUERIES = [
    "site:linkedin.com/in albury NSW director CTO CFO -jobs",
    '"albury" OR "wodonga" "director" OR "CEO" inurl:linkedin.com -jobs',
    'site:abn.business.gov.au "albury" "technology" OR "software"',
    '"albury wodonga" "request for proposal" OR "RFP" 2024 OR 2025',
    '"albury" "AI" OR "artificial intelligence" "procurement"',
]

def google_dork_search(query):
    """Google Custom Search API — requires GOOGLE_CSE_KEY + GOOGLE_CSE_ID."""
    key = load_env("GOOGLE_CSE_KEY") or load_env("GOOGLE_API_KEY")
    cse = load_env("GOOGLE_CSE_ID")
    if not key or not cse:
        log.info("Google CSE: no credentials — skipping dork")
        return []
    try:
        params = urllib.parse.urlencode({"key":key,"cx":cse,"q":query,"num":5})
        with urllib.request.urlopen(
            f"https://www.googleapis.com/customsearch/v1?{params}", timeout=8
        ) as r:
            items = json.loads(r.read()).get("items",[])
        return [{"title":i.get("title",""),"link":i.get("link",""),
                 "snippet":i.get("snippet","")} for i in items]
    except Exception as e:
        log.warning("Google CSE error: %s", e)
        return []

# ─── Local business targets ───────────────────────────────────────────────────

ALBURY_TARGETS = [
    # (company, contact, phone, score, context)
    ("Albury City Council",        "Procurement Team",   "02 6023 8111", 85,
     "Enterprise SaaS + AI governance platform"),
    ("Border Bank",                "Digital Innovation", "02 6041 2200", 82,
     "Open banking API, crypto custody, fintech"),
    ("Albury Wodonga Health",      "IT Department",      "02 6058 2222", 88,
     "Healthcare AI pipeline, patient data analytics"),
    ("Wodonga TAFE",               "Innovation Office",  "02 6055 6333", 72,
     "EdTech + RL curriculum, student analytics"),
    ("Hume Bank",                  "CTO Office",         "02 6058 1000", 75,
     "Core banking modernisation, open finance"),
    ("Regional Express Airlines",  "Ops Technology",     "02 6021 1300", 78,
     "Route optimisation, predictive maintenance"),
    ("Murray River Group",         "Operations",         "02 6025 0200", 71,
     "Agricultural IoT, logistics optimisation"),
    ("North East Water",           "Infrastructure",     "02 5748 1000", 68,
     "Water infrastructure monitoring, IoT sensors"),
    ("Mungabareena Aboriginal",    "Community Programs", "02 6041 1304", 65,
     "Community AI platform, cultural data sovereignty"),
    ("Albury Northside Private",   "Procurement",        "02 6055 0000", 74,
     "Private hospital analytics, patient flow AI"),
    ("Charles Sturt University",   "Research Office",    "02 6051 6000", 80,
     "Research AI infrastructure, student engagement"),
    ("NSW Trade & Investment",     "Regional Programs",  "02 6051 6600", 77,
     "Regional economic development, startup ecosystem"),
]

def seed_leads():
    """Seed discovered leads into sovereign.db (idempotent)."""
    conn = get_db()
    inserted = 0
    for company, contact, phone, score, context in ALBURY_TARGETS:
        try:
            conn.execute(
                """INSERT OR IGNORE INTO leads(company,contact,phone,score,context,status)
                   VALUES(?,?,?,?,?,'new')""",
                (company, contact, phone, score, context)
            )
            if conn.execute("SELECT changes()").fetchone()[0]:
                inserted += 1
        except Exception as e:
            log.warning("Lead seed error: %s", e)
    conn.commit(); conn.close()
    if inserted:
        log.info("Seeded %d new leads into sovereign.db", inserted)

def score_and_update_leads():
    """Re-score existing leads based on activity, recency, and WiGLE density."""
    conn      = get_db()
    wigle_info = wigle_search_area()
    biz_nets  = wigle_info.get("business_networks", 0)
    # Boost score for leads in high-density WiFi areas (proxy for commercial activity)
    boost = min(5, biz_nets // 5)
    rows  = conn.execute("SELECT id, score FROM leads WHERE score < 95").fetchall()
    for row in rows:
        new_score = min(95, row["score"] + boost)
        conn.execute("UPDATE leads SET score=? WHERE id=?", (new_score, row["id"]))
    conn.execute("INSERT INTO audit_log(event,detail) VALUES('DORKING_CYCLE',?)",
                 (f"WiGLE biz_nets={biz_nets} boost={boost} leads_updated={len(rows)}",))
    conn.commit(); conn.close()
    log.info("Lead scoring: %d leads updated | WiGLE density boost=%d", len(rows), boost)

def run_dork_cycle():
    """Full dorking cycle: seed → WiGLE → score."""
    seed_leads()
    score_and_update_leads()
    # Run 1 Google dork per cycle (rate-limit friendly)
    if load_env("GOOGLE_CSE_KEY"):
        import random
        results = google_dork_search(random.choice(DORK_QUERIES))
        if results:
            log.info("Google dork: %d results", len(results))

def main():
    log.info("fm-dorking started | DB=%s", DB)
    while True:
        try:
            run_dork_cycle()
        except Exception as e:
            log.error("Cycle error: %s", e)
        time.sleep(900)  # 15 minutes

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
