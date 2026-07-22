"""
FractalMesh AdMob Bridge Agent v2.0.0
Tracks Google AdMob ad revenue, RPM, fill rate, and impressions.
Correlates ad performance with product/campaign data.
Samuel James Hiotis | ABN 56 628 117 363 | Sole Trader
"""
import os
import json
import time
import signal
import sqlite3
import base64
import urllib.request
import urllib.parse
from datetime import datetime, timedelta

ROOT       = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB         = os.path.join(ROOT, "database", "sovereign.db")
INTERVAL   = int(os.getenv("ADMOB_INTERVAL", "3600"))
DRY_RUN    = os.getenv("ENABLE_ADMOB", "false").lower() != "true"

PUBLISHER_ID = os.getenv("ADMOB_PUBLISHER_ID", "")
API_KEY      = os.getenv("ADMOB_API_KEY", "")
# OAuth tokens (long-lived access token or service account JWT)
ACCESS_TOKEN = os.getenv("ADMOB_ACCESS_TOKEN", "")

PHI      = 1.6180339887
_running = True


def _db_init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS admob_revenue (
        id INTEGER PRIMARY KEY, date TEXT, ad_unit_id TEXT, ad_type TEXT,
        impressions INTEGER, clicks INTEGER, estimated_earnings_aud REAL,
        rpm REAL, fill_rate REAL, phi_score REAL,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS admob_units (
        id INTEGER PRIMARY KEY, unit_id TEXT UNIQUE, unit_name TEXT,
        app_id TEXT, format TEXT, status TEXT,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()


def _admob_request(path: str) -> dict:
    if not PUBLISHER_ID or not ACCESS_TOKEN or DRY_RUN:
        return {}
    url = f"https://admob.googleapis.com/v1/{path}"
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {ACCESS_TOKEN}",
                      "Content-Type":  "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"[fm-admob] API error: {e}")
        return {}


def _generate_report(date_str: str = None) -> list:
    """Generate AdMob mediation report for given date (YYYY-MM-DD)."""
    if DRY_RUN or not PUBLISHER_ID:
        # Simulate demo data for dry-run
        return [{
            "date":       date_str or datetime.utcnow().strftime("%Y-%m-%d"),
            "ad_unit_id": "demo_banner_01",
            "ad_type":    "BANNER",
            "impressions": 1240,
            "clicks":      18,
            "estimated_earnings_aud": 0.84,
            "rpm":         0.68,
            "fill_rate":   92.5,
        }]

    if not date_str:
        date_str = (datetime.utcnow() - timedelta(days=1)).strftime("%Y%m%d")
    else:
        date_str = date_str.replace("-", "")

    year  = int(date_str[:4])
    month = int(date_str[4:6])
    day   = int(date_str[6:8])

    body  = json.dumps({
        "reportSpec": {
            "dateRange":  {"startDate": {"year": year, "month": month, "day": day},
                           "endDate":   {"year": year, "month": month, "day": day}},
            "dimensions": ["AD_UNIT", "FORMAT"],
            "metrics":    ["IMPRESSIONS", "CLICKS", "ESTIMATED_EARNINGS", "IMPRESSION_RPM",
                           "MATCH_RATE"],
        }
    }).encode()

    req = urllib.request.Request(
        f"https://admob.googleapis.com/v1/accounts/{PUBLISHER_ID}/mediationReport:generate",
        data=body, method="POST",
        headers={"Authorization": f"Bearer {ACCESS_TOKEN}",
                 "Content-Type":  "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            lines   = r.read().decode().splitlines()
            rows    = [json.loads(l) for l in lines if l.strip().startswith("{")]
            results = []
            for row in rows:
                if "row" not in row:
                    continue
                r_data = row["row"]
                dim_v  = r_data.get("dimensionValues", {})
                met_v  = r_data.get("metricValues", {})
                results.append({
                    "date":       date_str,
                    "ad_unit_id": dim_v.get("AD_UNIT", {}).get("value", ""),
                    "ad_type":    dim_v.get("FORMAT",  {}).get("value", ""),
                    "impressions": int(met_v.get("IMPRESSIONS", {}).get("integerValue", 0)),
                    "clicks":      int(met_v.get("CLICKS", {}).get("integerValue", 0)),
                    "estimated_earnings_aud": float(
                        met_v.get("ESTIMATED_EARNINGS", {}).get("microsValue", 0)) / 1e6,
                    "rpm":        float(met_v.get("IMPRESSION_RPM", {}).get("doubleValue", 0.0)),
                    "fill_rate":  float(met_v.get("MATCH_RATE", {}).get("doubleValue", 0.0)) * 100,
                })
            return results
    except Exception as e:
        print(f"[fm-admob] report error: {e}")
        return []


def _upsert_rows(rows: list):
    conn = sqlite3.connect(DB, timeout=10)
    for row in rows:
        phi   = round(row.get("estimated_earnings_aud", 0.0) * PHI, 6)
        try:
            conn.execute("""INSERT INTO admob_revenue
                (date,ad_unit_id,ad_type,impressions,clicks,
                 estimated_earnings_aud,rpm,fill_rate,phi_score)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (row["date"], row["ad_unit_id"], row["ad_type"],
                 row["impressions"], row["clicks"],
                 row["estimated_earnings_aud"], row["rpm"],
                 row["fill_rate"], phi))
        except sqlite3.IntegrityError:
            pass
    conn.commit(); conn.close()


def run_cycle():
    ts   = datetime.utcnow().isoformat()
    date = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"[fm-admob] {ts} | dry={DRY_RUN} | publisher={'set' if PUBLISHER_ID else 'NOT SET'}")

    rows = _generate_report(date)
    _upsert_rows(rows)

    total_earn = sum(r.get("estimated_earnings_aud", 0.0) for r in rows)
    total_imp  = sum(r.get("impressions", 0) for r in rows)
    avg_rpm    = sum(r.get("rpm", 0.0) for r in rows) / max(len(rows), 1)
    phi_total  = round(total_earn * PHI, 4)

    print(f"   Date:        {date}")
    print(f"   Impressions: {total_imp:,}")
    print(f"   Earnings:    ${total_earn:.4f} AUD | φ-weighted: ${phi_total:.4f}")
    print(f"   Avg RPM:     ${avg_rpm:.4f}")
    for r in rows:
        print(f"   [{r['ad_type']}] {r['ad_unit_id'][:30]} | "
              f"imp={r['impressions']} clk={r['clicks']} "
              f"fill={r['fill_rate']:.1f}% ${r['estimated_earnings_aud']:.4f}")


def _sigterm(s, f):
    global _running
    _running = False


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT,  _sigterm)
    _db_init()
    print(f"[fm-admob] Active | interval={INTERVAL}s | dry={DRY_RUN} | "
          f"publisher={'set' if PUBLISHER_ID else 'NOT SET'}")
    while _running:
        try:
            run_cycle()
        except Exception as e:
            print(f"[fm-admob] ERR {e}")
        for _ in range(INTERVAL):
            if not _running: break
            time.sleep(1)
    print("[fm-admob] Stopped.")
