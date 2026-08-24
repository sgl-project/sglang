"""
FractalMesh Geo Validator & Geotagging Agent v2.0.0
Validates prospect coordinates, tags leads with lat/lon metadata,
cross-references Albury-Wodonga corridor for outreach prioritisation.
Samuel James Hiotis | ABN 56 628 117 363 | Sole Trader
"""
import os
import re
import json
import math
import time
import signal
import sqlite3
import hashlib
import urllib.request
import urllib.parse
from datetime import datetime

ROOT     = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB       = os.path.join(ROOT, "database", "sovereign.db")
INTERVAL = int(os.getenv("GEO_INTERVAL", "1800"))
DRY_RUN  = os.getenv("ENABLE_GEO_VALIDATOR", "false").lower() != "true"

# Albury-Wodonga sovereign node home
HOME_LAT  = float(os.getenv("HOME_LAT",  "-36.0737"))
HOME_LON  = float(os.getenv("HOME_LON",  "146.9135"))
RADIUS_KM = float(os.getenv("GEO_RADIUS_KM", "200"))   # NSW/VIC corridor

# Known Albury/Wodonga/Wagga postcodes
TARGET_POSTCODES = {"2640", "2641", "2642", "2643", "3690", "3691",
                    "2650", "2651", "2652", "2653"}

# Region hierarchy for φ-scoring
REGIONS = {
    "albury_wodonga": (HOME_LAT,   HOME_LON,   50,   2.0),
    "wagga_wagga":    (-35.1082,   147.3598,   80,   1.5),
    "nsw_murray":     (-35.5000,   146.8000,  150,   1.2),
    "victoria_ne":    (-36.5000,   146.5000,  150,   1.2),
    "nsw_statewide":  (-33.8688,   151.2093,  900,   1.0),
}

PHI     = 1.6180339887
_running = True


def _db_init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS geo_tags (
        id INTEGER PRIMARY KEY, record_id TEXT, record_type TEXT,
        postcode TEXT, lat REAL, lon REAL, region TEXT,
        dist_km REAL, phi_score REAL, verified INTEGER DEFAULT 0,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS geo_validation_log (
        id INTEGER PRIMARY KEY, source TEXT, total INTEGER,
        tagged INTEGER, invalid INTEGER,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R    = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a    = (math.sin(dlat / 2) ** 2 +
            math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
            math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def _region_for(lat: float, lon: float) -> tuple:
    best_region = "unknown"
    best_dist   = float("inf")
    for name, (rlat, rlon, rmax, _) in REGIONS.items():
        d = _haversine(lat, lon, rlat, rlon)
        if d < rmax and d < best_dist:
            best_dist   = d
            best_region = name
    return best_region, round(best_dist, 2)


def _phi_score(region: str, dist_km: float) -> float:
    weight = REGIONS.get(region, ("", "", 0, 1.0))[3] if region in REGIONS else 1.0
    decay  = 1.0 / (1.0 + dist_km / 100.0)
    return round(weight * decay * PHI, 4)


def _postcode_to_coords(postcode: str) -> tuple:
    """Approximate lat/lon from known postcode table."""
    table = {
        "2640": (-36.0737, 146.9135), "2641": (-36.0700, 146.9300),
        "3690": (-36.1216, 146.8935), "3691": (-36.1400, 146.8800),
        "2650": (-35.1082, 147.3598), "2651": (-35.1000, 147.3500),
        "0200": (-35.2809, 149.1300),   # ACT
    }
    return table.get(str(postcode), (None, None))


def _geocode_query(address: str) -> tuple:
    """Use nominatim free geocoding (rate-limited)."""
    if DRY_RUN:
        return (None, None)
    try:
        params = urllib.parse.urlencode({"q": address, "format": "json", "limit": 1})
        req = urllib.request.Request(
            f"https://nominatim.openstreetmap.org/search?{params}",
            headers={"User-Agent": "FractalMesh-GeoValidator/2.0 samuel@fractalmesh.net"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
            if data:
                return (float(data[0]["lat"]), float(data[0]["lon"]))
    except Exception:
        pass
    return (None, None)


def _tag_leads():
    conn = sqlite3.connect(DB, timeout=10)
    conn.row_factory = sqlite3.Row
    # Find leads without geo tags
    leads = conn.execute("""
        SELECT l.id, l.raw_query, l.industry, l.region
        FROM leads l
        WHERE NOT EXISTS (
            SELECT 1 FROM geo_tags g
            WHERE g.record_id = CAST(l.id AS TEXT) AND g.record_type='lead'
        )
        LIMIT 50""").fetchall()
    conn.close()

    tagged = 0
    for lead in leads:
        # Extract postcode from query
        match = re.search(r'\b(2[56][0-9]{2}|36[0-9]{2})\b', lead["raw_query"] or "")
        pc    = match.group(1) if match else None
        lat, lon = _postcode_to_coords(pc) if pc else (None, None)

        if lat is None:
            # Try geocoding industry + region
            query = f"{lead['industry']} {lead['region'] or 'Albury NSW'}"
            lat, lon = _geocode_query(query)
            if not DRY_RUN:
                time.sleep(1)  # Nominatim rate limit

        if lat is None:
            lat, lon = HOME_LAT, HOME_LON  # default to home node

        region, dist = _region_for(lat, lon)
        phi          = _phi_score(region, dist)
        verified     = 1 if pc else 0

        conn = sqlite3.connect(DB, timeout=10)
        conn.execute("""INSERT INTO geo_tags
            (record_id,record_type,postcode,lat,lon,region,dist_km,phi_score,verified)
            VALUES (?,?,?,?,?,?,?,?,?)""",
            (str(lead["id"]), "lead", pc or "", lat, lon, region, dist, phi, verified))
        conn.commit(); conn.close()
        tagged += 1

    return len(leads), tagged


def _validate_wigle():
    """Validate and geotag WiGLE telemetry records."""
    conn = sqlite3.connect(DB, timeout=10)
    try:
        rows = conn.execute("""
            SELECT bssid, lat, lon FROM wigle_telemetry_ledger
            WHERE lat IS NOT NULL AND lon IS NOT NULL LIMIT 100""").fetchall()
    except Exception:
        conn.close()
        return 0
    conn.close()

    tagged = 0
    for bssid, lat, lon in rows:
        region, dist = _region_for(float(lat), float(lon))
        phi          = _phi_score(region, dist)
        conn = sqlite3.connect(DB, timeout=10)
        conn.execute("""INSERT OR IGNORE INTO geo_tags
            (record_id,record_type,lat,lon,region,dist_km,phi_score,verified)
            VALUES (?,?,?,?,?,?,?,1)""",
            (bssid, "wigle", float(lat), float(lon), region, dist, phi))
        conn.commit(); conn.close()
        tagged += 1
    return tagged


def run_cycle():
    ts = datetime.utcnow().isoformat()
    print(f"[fm-geo-validator] {ts} | dry={DRY_RUN}")

    total, tagged  = _tag_leads()
    wigle_tagged   = _validate_wigle()
    home_dist      = _haversine(HOME_LAT, HOME_LON, HOME_LAT, HOME_LON)

    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("""INSERT INTO geo_validation_log (source,total,tagged,invalid)
        VALUES (?,?,?,?)""", ("leads", total, tagged, total - tagged))
    conn.commit(); conn.close()

    print(f"   Leads: {total} found | {tagged} tagged | WiGLE: {wigle_tagged} tagged")
    print(f"   Home node: lat={HOME_LAT} lon={HOME_LON} | radius={RADIUS_KM}km")


def _sigterm(s, f):
    global _running
    _running = False


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT,  _sigterm)
    _db_init()
    print(f"[fm-geo-validator] Active | interval={INTERVAL}s | dry={DRY_RUN} | "
          f"home=({HOME_LAT},{HOME_LON}) | radius={RADIUS_KM}km")
    while _running:
        try:
            run_cycle()
        except Exception as e:
            print(f"[fm-geo-validator] ERR {e}")
        for _ in range(INTERVAL):
            if not _running: break
            time.sleep(1)
    print("[fm-geo-validator] Stopped.")
