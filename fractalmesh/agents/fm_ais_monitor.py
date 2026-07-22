"""
FractalMesh AIS Monitor v2.0.0
Detects dark fleet behaviour: vessels going AIS-dark, location spoofing,
suspicious gap patterns. Sources: AISHub (free share tier), VesselFinder.
Outputs structured alerts to ais_alerts table for insurance/intel reports.
Samuel James Hiotis | ABN 56 628 117 363 | Sole Trader
"""
import os
import json
import math
import time
import signal
import sqlite3
import hashlib
import urllib.request
import urllib.parse
from datetime import datetime, timedelta

ROOT     = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB       = os.path.join(ROOT, "database", "sovereign.db")
INTERVAL = int(os.getenv("AIS_INTERVAL", "900"))   # 15 min

# AISHub — free API (reciprocal data share required to activate)
# Register at https://www.aishub.net/join-us
AISHUB_USER   = os.getenv("AISHUB_USER",   "")
AISHUB_FORMAT = "json"

# VesselFinder free tier
VESSELFINDER_KEY = os.getenv("VESSELFINDER_API_KEY", "")

# Dark event thresholds
DARK_GAP_HOURS   = float(os.getenv("AIS_DARK_GAP_HOURS",  "6.0"))   # gap = dark
SPOOF_SPEED_KNOT = float(os.getenv("AIS_SPOOF_SPEED_KNOT","50.0"))   # impossible speed
SPOOF_JUMP_KM    = float(os.getenv("AIS_SPOOF_JUMP_KM",   "500.0"))  # impossible jump

# Areas of interest (bounding boxes for monitoring)
AOI_ZONES = {
    "torres_strait":   {"lat": (-11.0, -9.0),  "lon": (141.0, 143.0)},
    "aus_northwest":   {"lat": (-22.0, -15.0), "lon": (113.0, 122.0)},
    "singapore_strait":{"lat": (1.0,   1.5),   "lon": (103.5, 104.5)},
    "malacca_strait":  {"lat": (1.0,   6.0),   "lon": (99.0,  103.5)},
    "persian_gulf":    {"lat": (24.0,  27.0),  "lon": (50.0,  57.0)},
    "black_sea":       {"lat": (41.0,  47.0),  "lon": (28.0,  42.0)},
}

PHI      = 1.6180339887
_running = True


def _db_init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS ais_tracking (
        id INTEGER PRIMARY KEY,
        mmsi TEXT,
        vessel_name TEXT,
        vessel_type TEXT,
        flag TEXT,
        lat REAL, lon REAL,
        speed_knots REAL,
        heading INTEGER,
        nav_status TEXT,
        source TEXT,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS ais_alerts (
        id INTEGER PRIMARY KEY,
        mmsi TEXT,
        vessel_name TEXT,
        alert_type TEXT,
        severity TEXT,
        zone TEXT,
        lat REAL, lon REAL,
        gap_hours REAL,
        detail TEXT,
        phi_score REAL,
        resolved INTEGER DEFAULT 0,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS ais_dark_events (
        id INTEGER PRIMARY KEY,
        mmsi TEXT UNIQUE,
        vessel_name TEXT,
        last_seen DATETIME,
        last_lat REAL, last_lon REAL,
        gap_hours REAL,
        zone TEXT,
        phi_score REAL,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()


# ── AISHub API ────────────────────────────────────────────────────────────────

def _aishub_query(bbox: dict) -> list:
    """
    Query AISHub API for vessels in bounding box.
    Requires AISHUB_USER (free — register and share your own AIS data).
    """
    if not AISHUB_USER:
        return []
    params = urllib.parse.urlencode({
        "username":  AISHUB_USER,
        "format":    "1",              # JSON
        "output":    "json",
        "compress":  "0",
        "latmin":    bbox["lat"][0],
        "latmax":    bbox["lat"][1],
        "lonmin":    bbox["lon"][0],
        "lonmax":    bbox["lon"][1],
    })
    try:
        req = urllib.request.Request(
            f"https://data.aishub.net/ws.php?{params}",
            headers={"User-Agent": "FractalMesh-AIS/2.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
            if isinstance(data, list) and len(data) > 1:
                return data[1]  # [0] is header, [1] is array of vessels
            return []
    except Exception as e:
        print(f"[fm-ais] AISHub error: {e}")
        return []


def _vesselfinder_query(mmsi: str) -> dict:
    if not VESSELFINDER_KEY:
        return {}
    params = urllib.parse.urlencode({
        "userkey": VESSELFINDER_KEY,
        "mmsi":    mmsi,
    })
    try:
        req = urllib.request.Request(
            f"https://api.vesselfinder.com/vessels?{params}",
            headers={"User-Agent": "FractalMesh-AIS/2.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())
    except Exception:
        return {}


# ── Spoof / dark detection ────────────────────────────────────────────────────

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R    = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a    = (math.sin(dlat / 2) ** 2 +
            math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
            math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def _check_spoofing(mmsi: str, lat: float, lon: float,
                    speed: float, ts_now: str) -> list:
    """Compare current position against last known position."""
    alerts = []
    conn   = sqlite3.connect(DB, timeout=10)
    conn.row_factory = sqlite3.Row
    prev   = conn.execute("""SELECT lat, lon, speed_knots, ts FROM ais_tracking
        WHERE mmsi=? ORDER BY ts DESC LIMIT 1""", (mmsi,)).fetchone()
    conn.close()

    if not prev:
        return alerts

    try:
        prev_ts = datetime.fromisoformat(prev["ts"])
        now_ts  = datetime.utcnow()
        dt_hr   = (now_ts - prev_ts).total_seconds() / 3600.0

        if dt_hr > 0:
            dist_km    = _haversine(prev["lat"], prev["lon"], lat, lon)
            max_speed  = dist_km / dt_hr * 0.539957  # km/h → knots
            if max_speed > SPOOF_SPEED_KNOT and dist_km > 50:
                alerts.append({
                    "type":    "position_spoof",
                    "severity":"HIGH",
                    "detail":  f"Impossible jump: {dist_km:.0f}km in {dt_hr:.1f}h "
                               f"= {max_speed:.0f}kn",
                    "gap":     dt_hr,
                })

        # Speed reported vs calculated
        if speed > SPOOF_SPEED_KNOT:
            alerts.append({
                "type":    "speed_anomaly",
                "severity":"MEDIUM",
                "detail":  f"Reported speed {speed:.1f}kn exceeds physical limit",
                "gap":     0.0,
            })

    except Exception:
        pass

    return alerts


def _detect_dark_events():
    """Find vessels that went AIS-dark (no transmission > threshold)."""
    cutoff = (datetime.utcnow() - timedelta(hours=DARK_GAP_HOURS)).isoformat()
    conn   = sqlite3.connect(DB, timeout=10)
    conn.row_factory = sqlite3.Row
    # Latest position per vessel
    darks  = conn.execute("""
        SELECT mmsi, vessel_name, MAX(ts) as last_seen, lat, lon
        FROM ais_tracking
        GROUP BY mmsi
        HAVING last_seen < ?""", (cutoff,)).fetchall()
    conn.close()

    events = []
    for d in darks:
        gap_hr   = (datetime.utcnow() -
                    datetime.fromisoformat(d["last_seen"])).total_seconds() / 3600.0
        phi      = round(min(gap_hr / DARK_GAP_HOURS, 10.0) * PHI, 4)

        # Find which zone the vessel was last seen in
        zone = "open_ocean"
        for zname, zbbox in AOI_ZONES.items():
            if (zbbox["lat"][0] <= (d["lat"] or 0) <= zbbox["lat"][1] and
                    zbbox["lon"][0] <= (d["lon"] or 0) <= zbbox["lon"][1]):
                zone = zname
                break

        conn = sqlite3.connect(DB, timeout=10)
        conn.execute("""INSERT INTO ais_dark_events
            (mmsi,vessel_name,last_seen,last_lat,last_lon,gap_hours,zone,phi_score)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(mmsi) DO UPDATE
            SET last_seen=excluded.last_seen, gap_hours=excluded.gap_hours,
                phi_score=excluded.phi_score, ts=CURRENT_TIMESTAMP""",
            (d["mmsi"], d["vessel_name"] or "UNKNOWN",
             d["last_seen"], d["lat"], d["lon"], round(gap_hr, 2), zone, phi))
        conn.commit(); conn.close()
        events.append({"mmsi": d["mmsi"], "gap_hr": round(gap_hr, 2), "zone": zone})

    return events


def _log_vessel(vessel: dict, source: str, zone: str):
    mmsi    = str(vessel.get("MMSI", vessel.get("mmsi", "")))
    name    = vessel.get("NAME", vessel.get("name", "UNKNOWN"))
    vtype   = str(vessel.get("TYPE", vessel.get("type", "")))
    flag    = vessel.get("FLAG", vessel.get("flag", ""))
    lat     = float(vessel.get("LATITUDE",  vessel.get("lat",   0.0)))
    lon     = float(vessel.get("LONGITUDE", vessel.get("lon",   0.0)))
    speed   = float(vessel.get("SPEED",     vessel.get("speed", 0.0))) / 10.0
    heading = int(vessel.get("HEADING",     vessel.get("heading", 0)))
    status  = str(vessel.get("NAVSTAT",     vessel.get("status", "")))

    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("""INSERT INTO ais_tracking
        (mmsi,vessel_name,vessel_type,flag,lat,lon,speed_knots,heading,nav_status,source)
        VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (mmsi, name, vtype, flag, lat, lon, speed, heading, status, source))
    conn.commit(); conn.close()

    # Check for spoofing
    alerts = _check_spoofing(mmsi, lat, lon, speed, datetime.utcnow().isoformat())
    for alert in alerts:
        phi   = round(PHI ** 3 if alert["severity"] == "HIGH" else PHI, 4)
        conn  = sqlite3.connect(DB, timeout=10)
        conn.execute("""INSERT INTO ais_alerts
            (mmsi,vessel_name,alert_type,severity,zone,lat,lon,gap_hours,detail,phi_score)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (mmsi, name, alert["type"], alert["severity"], zone,
             lat, lon, alert["gap"], alert["detail"], phi))
        conn.commit(); conn.close()
        print(f"   *** {alert['severity']} ALERT [{alert['type']}] MMSI:{mmsi} "
              f"{name} | {alert['detail']}")


def run_cycle():
    ts = datetime.utcnow().isoformat()
    print(f"[fm-ais-monitor] {ts} | AISHub={'set' if AISHUB_USER else 'NOT SET'}")

    total_vessels = 0
    for zone_name, bbox in AOI_ZONES.items():
        vessels = _aishub_query(bbox)
        for v in vessels:
            _log_vessel(v, "aishub", zone_name)
        total_vessels += len(vessels)
        if vessels:
            print(f"   {zone_name}: {len(vessels)} vessels")
        if not AISHUB_USER:
            break  # no point iterating zones without credentials

    # Run dark event detection on all tracked vessels
    dark_events = _detect_dark_events()
    for ev in dark_events:
        print(f"   DARK: MMSI={ev['mmsi']} gap={ev['gap_hr']:.1f}h zone={ev['zone']}")

    # Summary alert counts
    conn   = sqlite3.connect(DB, timeout=10)
    high   = conn.execute("SELECT COUNT(*) FROM ais_alerts WHERE severity='HIGH' "
                          "AND resolved=0").fetchone()[0]
    medium = conn.execute("SELECT COUNT(*) FROM ais_alerts WHERE severity='MEDIUM' "
                          "AND resolved=0").fetchone()[0]
    darks  = conn.execute("SELECT COUNT(*) FROM ais_dark_events WHERE "
                          f"gap_hours > {DARK_GAP_HOURS}").fetchone()[0]
    conn.close()

    print(f"   Vessels tracked: {total_vessels} | Dark events: {len(dark_events)} "
          f"| Open alerts: HIGH={high} MED={medium} | Total dark: {darks}")

    if not AISHUB_USER:
        print("   NOTE: Register at https://www.aishub.net/join-us "
              "and set AISHUB_USER in vault to enable live tracking")


def _sigterm(s, f):
    global _running
    _running = False


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT,  _sigterm)
    _db_init()
    print(f"[fm-ais-monitor] Active | interval={INTERVAL}s | "
          f"dark_threshold={DARK_GAP_HOURS}h | spoof_speed={SPOOF_SPEED_KNOT}kn | "
          f"zones={len(AOI_ZONES)}")
    while _running:
        try:
            run_cycle()
        except Exception as e:
            print(f"[fm-ais-monitor] ERR {e}")
        for _ in range(INTERVAL):
            if not _running: break
            time.sleep(1)
    print("[fm-ais-monitor] Stopped.")
