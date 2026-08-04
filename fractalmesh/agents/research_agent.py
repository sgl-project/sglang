"""
FractalMesh Research Agent v2.1.0
NOAA geomagnetic Kp index, CoinGecko crypto prices,
business analytics aggregation. Feeds sovereign.db.
Samuel James Hiotis | ABN 56 628 117 363 | Sole Trader
"""
import os
import json
import time
import signal
import sqlite3
import hashlib
import urllib.request
import urllib.parse
from datetime import datetime, timezone

ROOT     = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB       = os.path.join(ROOT, "database", "sovereign.db")
INTERVAL = int(os.getenv("RESEARCH_INTERVAL", "1800"))

PHI      = 1.6180339887
_running = True


def _db_init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS research_cache (
        id INTEGER PRIMARY KEY,
        source TEXT, endpoint TEXT,
        response_hash TEXT UNIQUE,
        data TEXT, ttl_seconds INTEGER,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS business_analytics (
        id INTEGER PRIMARY KEY,
        metric_name TEXT, metric_type TEXT,
        value REAL, unit TEXT, period TEXT,
        source TEXT, phi_score REAL,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()


def _log(msg: str, data=None):
    print(f"[research] {msg}")
    conn = sqlite3.connect(DB, timeout=10)
    try:
        conn.execute("""INSERT INTO pulse_log (source,event,priority) VALUES (?,?,?)""",
                     ("research_agent", msg[:200], 1.0))
        conn.commit()
    except Exception:
        pass
    finally:
        conn.close()


def _cache_set(source: str, endpoint: str, data: dict, ttl: int = 3600):
    h = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]
    conn = sqlite3.connect(DB, timeout=10)
    try:
        conn.execute("""INSERT INTO research_cache
            (source,endpoint,response_hash,data,ttl_seconds) VALUES (?,?,?,?,?)
            ON CONFLICT(response_hash) DO UPDATE SET ts=CURRENT_TIMESTAMP""",
            (source, endpoint, h, json.dumps(data)[:2000], ttl))
        conn.commit()
    except Exception:
        pass
    finally:
        conn.close()


def _fetch_url(url: str, params: dict = None, timeout: int = 15) -> dict:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "FractalMesh-Research/2.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# ── NOAA Geomagnetic Kp index ─────────────────────────────────────────────────

def _fetch_noaa_kp() -> dict:
    try:
        rows   = _fetch_url(
            "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json")
        kp     = float(rows[-1][1])
        regime = ("CALM" if kp < 3 else
                  "ACTIVE" if kp < 5 else
                  "STORM" if kp < 7 else "EXTREME")
        result = {"kp_index": kp, "regime": regime,
                  "ts": datetime.now(tz=timezone.utc).isoformat()}
        _cache_set("noaa", "kp_index", result, 1800)
        _log(f"NOAA Kp={kp} regime={regime}")
        return result
    except Exception as e:
        _log(f"NOAA Kp fetch failed: {e}")
        return {}


# ── CoinGecko crypto prices ───────────────────────────────────────────────────

def _fetch_crypto() -> dict:
    try:
        data   = _fetch_url(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": "bitcoin,ethereum,helium,xyo-network",
                    "vs_currencies": "aud,usd"})
        btc    = data.get("bitcoin", {}).get("aud", 0)
        eth    = data.get("ethereum", {}).get("aud", 0)
        _cache_set("coingecko", "prices", data, 60)
        _log(f"CoinGecko: BTC=A${btc:,.0f} ETH=A${eth:,.0f}")
        return data
    except Exception as e:
        _log(f"CoinGecko fetch failed: {e}")
        return {}


# ── Business analytics from sovereign.db ─────────────────────────────────────

def _compute_analytics():
    conn = sqlite3.connect(DB, timeout=10)

    def safe_q(sql, default=0):
        try:
            return conn.execute(sql).fetchone()[0] or default
        except Exception:
            return default

    total_rev     = float(safe_q("SELECT COALESCE(SUM(amount_aud),0) FROM revenue"))
    total_orders  = safe_q("SELECT COUNT(*) FROM orders WHERE status='paid'")
    active_prods  = safe_q("SELECT COUNT(*) FROM products WHERE active=1")
    total_leads   = safe_q("SELECT COUNT(*) FROM leads")
    ip_value      = float(safe_q(
        "SELECT COALESCE(SUM(value_estimate_aud),0) FROM ip_registry"))
    aff_programs  = safe_q(
        "SELECT COUNT(*) FROM affiliates WHERE status='active'")
    aff_clicks_7d = safe_q(
        "SELECT COUNT(*) FROM affiliate_clicks WHERE ts>datetime('now','-7 days')")
    methane_anom  = safe_q(
        "SELECT COUNT(*) FROM methane_readings WHERE is_anomaly=1")
    ais_alerts    = safe_q(
        "SELECT COUNT(*) FROM ais_alerts WHERE resolved=0")
    conn.close()

    conv = (total_orders / max(total_leads, 1)) * 100
    phi  = round(total_rev * PHI / 10000, 4)

    metrics = [
        ("total_revenue",     "revenue", total_rev,     "AUD",     "all_time",  "revenue"),
        ("total_orders",      "kpi",     total_orders,  "count",   "all_time",  "orders"),
        ("active_products",   "kpi",     active_prods,  "count",   "current",   "products"),
        ("lead_conversion",   "ratio",   conv,          "percent", "all_time",  "calculated"),
        ("ip_portfolio",      "kpi",     ip_value,      "AUD",     "current",   "ip_registry"),
        ("affiliate_programs","kpi",     aff_programs,  "count",   "current",   "affiliates"),
        ("affiliate_clicks7d","kpi",     aff_clicks_7d, "count",   "7_days",    "affiliate_clicks"),
        ("methane_anomalies", "kpi",     methane_anom,  "count",   "all_time",  "methane_readings"),
        ("ais_open_alerts",   "kpi",     ais_alerts,    "count",   "current",   "ais_alerts"),
    ]

    conn = sqlite3.connect(DB, timeout=10)
    for name, mtype, value, unit, period, source in metrics:
        phi_s = round(value * PHI / max(value, 1), 4) if value else 0.0
        try:
            conn.execute("""INSERT INTO business_analytics
                (metric_name,metric_type,value,unit,period,source,phi_score)
                VALUES (?,?,?,?,?,?,?)""",
                (name, mtype, float(value), unit, period, source, phi_s))
        except Exception:
            pass
    conn.commit(); conn.close()

    _log(f"Analytics: rev=A${total_rev:.2f} orders={total_orders} "
         f"products={active_prods} conv={conv:.1f}% IP=A${ip_value:,.0f} "
         f"aff={aff_programs} CH4_anom={methane_anom} AIS={ais_alerts}")


def run_cycle():
    ts = datetime.now(tz=timezone.utc).isoformat()
    print(f"[research] {ts}")
    _fetch_noaa_kp()
    _fetch_crypto()
    _compute_analytics()


def _sigterm(s, f):
    global _running
    _running = False


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT,  _sigterm)
    _db_init()
    _log("Research Integration Agent started — NOAA, CoinGecko, Analytics")
    while _running:
        try:
            run_cycle()
        except Exception as e:
            _log(f"Error: {e}")
        for _ in range(INTERVAL):
            if not _running: break
            time.sleep(1)
    print("[research] Stopped.")
