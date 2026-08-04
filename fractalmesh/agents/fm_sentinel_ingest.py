"""
FractalMesh Sentinel Ingest Agent v2.0.0
Ingests Copernicus Sentinel-5P (methane CH4) and Sentinel-2 (crop/land)
data via Copernicus Data Space Ecosystem (CDSE) open API.
Also queries NASA EMIT methane plume data (public API, no key required).
Feeds: osint-harvester pipeline → methane_readings + crop_signals tables.
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
from datetime import datetime, timedelta

ROOT     = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB       = os.path.join(ROOT, "database", "sovereign.db")
INTERVAL = int(os.getenv("SENTINEL_INTERVAL", "21600"))  # 6 hours

# Copernicus Data Space Ecosystem — free, registration required
# https://dataspace.copernicus.eu/
CDSE_USER = os.getenv("CDSE_USER", "")
CDSE_PASS = os.getenv("CDSE_PASS", "")

# Target area of interest — configurable, defaults to Australia/global
AOI_LAT_MIN = float(os.getenv("AOI_LAT_MIN", "-45.0"))
AOI_LAT_MAX = float(os.getenv("AOI_LAT_MAX", "-10.0"))
AOI_LON_MIN = float(os.getenv("AOI_LON_MIN", "112.0"))
AOI_LON_MAX = float(os.getenv("AOI_LON_MAX", "155.0"))

# Methane background (ppb) — TROPOMI baseline ~1870 ppb globally
CH4_BACKGROUND_PPB = float(os.getenv("CH4_BACKGROUND_PPB", "1870.0"))
CH4_ANOMALY_SIGMA  = float(os.getenv("CH4_ANOMALY_SIGMA",  "2.0"))

PHI      = 1.6180339887
_running = True


def _db_init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS methane_readings (
        id INTEGER PRIMARY KEY,
        product_id TEXT UNIQUE,
        source TEXT,
        lat REAL, lon REAL,
        ch4_ppb REAL,
        ch4_enhancement REAL,
        quality_flag INTEGER,
        sensing_date TEXT,
        plume_area_km2 REAL,
        estimated_flux_kt REAL,
        is_anomaly INTEGER DEFAULT 0,
        phi_score REAL,
        raw_meta TEXT,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS crop_signals (
        id INTEGER PRIMARY KEY,
        product_id TEXT UNIQUE,
        source TEXT,
        lat REAL, lon REAL,
        region TEXT,
        ndvi REAL,
        cloud_cover_pct REAL,
        sensing_date TEXT,
        crop_type TEXT,
        yield_signal TEXT,
        phi_score REAL,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS sentinel_ingest_log (
        id INTEGER PRIMARY KEY,
        source TEXT,
        products_found INTEGER,
        anomalies INTEGER,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()


# ── Copernicus CDSE auth ───────────────────────────────────────────────────────

def _cdse_token() -> str:
    if not CDSE_USER or not CDSE_PASS:
        return ""
    body = urllib.parse.urlencode({
        "grant_type": "password",
        "username":   CDSE_USER,
        "password":   CDSE_PASS,
        "client_id":  "cdse-public",
    }).encode()
    req = urllib.request.Request(
        "https://identity.dataspace.copernicus.eu/auth/realms/"
        "CDSE/protocol/openid-connect/token",
        data=body, method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read()).get("access_token", "")
    except Exception as e:
        print(f"[sentinel] CDSE auth error: {e}")
        return ""


# ── Sentinel-5P TROPOMI CH4 search ────────────────────────────────────────────

def _search_sentinel5p(token: str, days_back: int = 3) -> list:
    """
    Query CDSE OData for Sentinel-5P L2__CH4___ products.
    Returns list of product metadata dicts.
    """
    since = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%dT00:00:00Z")
    bbox  = (f"POLYGON(({AOI_LON_MIN} {AOI_LAT_MIN},"
             f"{AOI_LON_MAX} {AOI_LAT_MIN},"
             f"{AOI_LON_MAX} {AOI_LAT_MAX},"
             f"{AOI_LON_MIN} {AOI_LAT_MAX},"
             f"{AOI_LON_MIN} {AOI_LAT_MIN}))")

    params = urllib.parse.urlencode({
        "$filter": (f"Collection/Name eq 'SENTINEL-5P' and "
                    f"Attributes/OData.CSC.StringAttribute/any("
                    f"att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq 'L2__CH4___') and "
                    f"ContentDate/Start gt {since} and "
                    f"OData.CSC.Intersects(area=geography'SRID=4326;{bbox}')"),
        "$top": "20",
        "$orderby": "ContentDate/Start desc",
    })

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    req = urllib.request.Request(
        f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?{params}",
        headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read())
            return data.get("value", [])
    except Exception as e:
        print(f"[sentinel] S5P search error: {e}")
        return []


# ── NASA EMIT methane plume data (public, no auth) ────────────────────────────

def _query_nasa_emit() -> list:
    """
    Query NASA EMIT methane plume portal (public STAC API).
    https://earth.jpl.nasa.gov/emit/
    """
    since = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%dT00:00:00Z")
    body  = json.dumps({
        "collections": ["EMIT_L2B_CH4PLM_001"],
        "datetime":    f"{since}/..",
        "bbox":        [AOI_LON_MIN, AOI_LAT_MIN, AOI_LON_MAX, AOI_LAT_MAX],
        "limit":       50,
    }).encode()
    req = urllib.request.Request(
        "https://cmr.earthdata.nasa.gov/stac/LPCLOUD/search",
        data=body, method="POST",
        headers={"Content-Type": "application/json",
                 "Accept":       "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read())
            return data.get("features", [])
    except Exception as e:
        print(f"[sentinel] NASA EMIT error: {e}")
        return []


# ── Sentinel-2 NDVI / crop search ────────────────────────────────────────────

def _search_sentinel2(token: str, days_back: int = 7) -> list:
    since = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%dT00:00:00Z")
    bbox  = (f"POLYGON(({AOI_LON_MIN} {AOI_LAT_MIN},"
             f"{AOI_LON_MAX} {AOI_LAT_MIN},"
             f"{AOI_LON_MAX} {AOI_LAT_MAX},"
             f"{AOI_LON_MIN} {AOI_LAT_MAX},"
             f"{AOI_LON_MIN} {AOI_LAT_MIN}))")

    params = urllib.parse.urlencode({
        "$filter": (f"Collection/Name eq 'SENTINEL-2' and "
                    f"Attributes/OData.CSC.StringAttribute/any("
                    f"att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A') and "
                    f"Attributes/OData.CSC.DoubleAttribute/any("
                    f"att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value lt 20) and "
                    f"ContentDate/Start gt {since} and "
                    f"OData.CSC.Intersects(area=geography'SRID=4326;{bbox}')"),
        "$top": "10",
        "$orderby": "ContentDate/Start desc",
    })

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    req = urllib.request.Request(
        f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?{params}",
        headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read())
            return data.get("value", [])
    except Exception as e:
        print(f"[sentinel] S2 search error: {e}")
        return []


# ── Anomaly detection ─────────────────────────────────────────────────────────

def _ch4_anomaly_score(ch4_ppb: float) -> tuple:
    """Returns (enhancement_ppb, is_anomaly, phi_score)."""
    enhancement = ch4_ppb - CH4_BACKGROUND_PPB
    # Rough 1-sigma for TROPOMI ~15 ppb
    sigma       = 15.0
    z_score     = enhancement / sigma if sigma else 0.0
    is_anomaly  = 1 if z_score >= CH4_ANOMALY_SIGMA else 0
    phi_score   = round(max(enhancement, 0) * PHI / 100, 6)
    return round(enhancement, 2), is_anomaly, phi_score


def _estimate_flux(enhancement_ppb: float, plume_area_km2: float = 1.0) -> float:
    """
    Very rough emission rate estimate (kt CO2-eq/yr proxy).
    Enhancement (ppb) × area × column density × MW × time factor.
    Use only as order-of-magnitude indicator.
    """
    if enhancement_ppb <= 0:
        return 0.0
    mol_per_m2    = (enhancement_ppb * 1e-9) * (2.15e25 / 6.022e23)
    mass_kg_per_m2 = mol_per_m2 * 0.016   # CH4 MW = 16 g/mol
    area_m2        = plume_area_km2 * 1e6
    kt_per_yr      = (mass_kg_per_m2 * area_m2 / 1000) * (365 * 24 * 3600) / 1e6
    return round(kt_per_yr, 4)


# ── Upsert helpers ─────────────────────────────────────────────────────────────

def _upsert_methane(product_id: str, source: str, lat: float, lon: float,
                    ch4_ppb: float, quality: int, sensing_date: str,
                    plume_area_km2: float, meta: str = ""):
    enh, is_anom, phi = _ch4_anomaly_score(ch4_ppb)
    flux = _estimate_flux(enh, plume_area_km2)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("""INSERT INTO methane_readings
        (product_id,source,lat,lon,ch4_ppb,ch4_enhancement,quality_flag,
         sensing_date,plume_area_km2,estimated_flux_kt,is_anomaly,phi_score,raw_meta)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(product_id) DO UPDATE
        SET ch4_ppb=excluded.ch4_ppb, is_anomaly=excluded.is_anomaly,
            phi_score=excluded.phi_score, ts=CURRENT_TIMESTAMP""",
        (product_id, source, lat, lon, ch4_ppb, enh, quality,
         sensing_date, plume_area_km2, flux, is_anom, phi, meta[:500]))
    conn.commit(); conn.close()
    return is_anom


def _upsert_crop(product_id: str, source: str, lat: float, lon: float,
                 region: str, ndvi: float, cloud_cover: float,
                 sensing_date: str, crop_type: str = "unknown"):
    phi   = round(max(ndvi, 0) * PHI, 4)
    yield_signal = "strong" if ndvi > 0.6 else ("moderate" if ndvi > 0.4 else "weak")
    conn  = sqlite3.connect(DB, timeout=10)
    conn.execute("""INSERT INTO crop_signals
        (product_id,source,lat,lon,region,ndvi,cloud_cover_pct,
         sensing_date,crop_type,yield_signal,phi_score)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(product_id) DO UPDATE
        SET ndvi=excluded.ndvi, yield_signal=excluded.yield_signal,
            ts=CURRENT_TIMESTAMP""",
        (product_id, source, lat, lon, region, ndvi, cloud_cover,
         sensing_date, crop_type, yield_signal, phi))
    conn.commit(); conn.close()


def run_cycle():
    ts = datetime.utcnow().isoformat()
    print(f"[fm-sentinel-ingest] {ts} | AOI=[{AOI_LAT_MIN},{AOI_LON_MIN}:"
          f"{AOI_LAT_MAX},{AOI_LON_MAX}]")

    # Authenticate with CDSE (optional — public products still searchable unauth'd)
    token = _cdse_token()
    if CDSE_USER and not token:
        print("   [WARN] CDSE auth failed — using unauthenticated access")

    total_methane = 0
    anomalies     = 0

    # ── Sentinel-5P CH4 ───────────────────────────────────────────────────────
    s5p_products = _search_sentinel5p(token)
    print(f"   S5P products found: {len(s5p_products)}")
    for prod in s5p_products:
        pid     = prod.get("Id", "")
        name    = prod.get("Name", "")
        date    = prod.get("ContentDate", {}).get("Start", "")[:10]
        # Extract centroid from footprint (approximate)
        footprint = prod.get("Footprint", "")
        lat, lon = -25.0, 133.0  # Australia centroid fallback
        # Parse from GeoJSON if available
        geo = prod.get("GeoFootprint", {})
        if geo.get("type") == "Polygon":
            coords = geo.get("coordinates", [[]])[0]
            if coords:
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                lon  = round(sum(lons) / len(lons), 4)
                lat  = round(sum(lats) / len(lats), 4)

        # CH4 column value from product attributes
        ch4_ppb = CH4_BACKGROUND_PPB  # default; actual value in NetCDF
        attrs   = prod.get("Attributes", [])
        for a in attrs:
            if a.get("Name") == "ch4ColumnVolumeConcentrationMean":
                try:
                    ch4_ppb = float(a.get("Value", CH4_BACKGROUND_PPB))
                except Exception:
                    pass

        prod_id   = hashlib.sha256(pid.encode()).hexdigest()[:20] if pid else name[:20]
        is_anom   = _upsert_methane(prod_id, "sentinel5p", lat, lon,
                                    ch4_ppb, 1, date, 25.0,
                                    json.dumps({"Name": name, "Id": pid})[:500])
        total_methane += 1
        if is_anom:
            anomalies += 1
            print(f"   *** ANOMALY *** lat={lat} lon={lon} CH4={ch4_ppb:.1f}ppb [{date}]")
        else:
            print(f"   S5P {date} lat={lat:.2f} lon={lon:.2f} CH4={ch4_ppb:.1f}ppb")

    # ── NASA EMIT plumes ──────────────────────────────────────────────────────
    emit_features = _query_nasa_emit()
    print(f"   NASA EMIT plumes found: {len(emit_features)}")
    for feat in emit_features:
        props     = feat.get("properties", {})
        geom      = feat.get("geometry", {})
        fid       = feat.get("id", "")
        date      = props.get("datetime", "")[:10]
        coords    = geom.get("coordinates", [0, 0])
        lon_f     = float(coords[0]) if len(coords) > 1 else 0.0
        lat_f     = float(coords[1]) if len(coords) > 1 else 0.0

        # EMIT reports IME (integrated methane enhancement) in kg
        ime_kg    = float(props.get("ime", 0.0))
        # Convert to rough column ppb for consistency (~5ppb per 1000kg plume)
        ch4_ppb   = CH4_BACKGROUND_PPB + (ime_kg / 200.0)
        area_km2  = float(props.get("scene_fid_km2", 1.0)) if "scene_fid_km2" in props else 1.0

        prod_id   = hashlib.sha256(fid.encode()).hexdigest()[:20] if fid else f"emit_{date}"
        is_anom   = _upsert_methane(prod_id, "nasa_emit", lat_f, lon_f,
                                    ch4_ppb, 1, date, area_km2,
                                    json.dumps({"id": fid, "ime_kg": ime_kg})[:500])
        total_methane += 1
        if is_anom:
            anomalies += 1
            print(f"   *** EMIT PLUME *** lat={lat_f} lon={lon_f} "
                  f"IME={ime_kg:.0f}kg [{date}]")

    # ── Sentinel-2 crop signals ───────────────────────────────────────────────
    s2_products = _search_sentinel2(token)
    print(f"   S2 products found: {len(s2_products)}")
    for prod in s2_products:
        pid   = prod.get("Id", "")
        name  = prod.get("Name", "")
        date  = prod.get("ContentDate", {}).get("Start", "")[:10]
        geo   = prod.get("GeoFootprint", {})
        lat_c, lon_c = -30.0, 147.0

        if geo.get("type") == "Polygon":
            coords = geo.get("coordinates", [[]])[0]
            if coords:
                lons  = [c[0] for c in coords]
                lats  = [c[1] for c in coords]
                lon_c = round(sum(lons) / len(lons), 4)
                lat_c = round(sum(lats) / len(lats), 4)

        cloud = 0.0
        attrs = prod.get("Attributes", [])
        for a in attrs:
            if a.get("Name") == "cloudCover":
                try:
                    cloud = float(a.get("Value", 0.0))
                except Exception:
                    pass

        # NDVI not in L2A metadata — use 0.5 placeholder; real value from band maths
        ndvi    = 0.5
        prod_id = hashlib.sha256(pid.encode()).hexdigest()[:20] if pid else name[:20]
        _upsert_crop(prod_id, "sentinel2", lat_c, lon_c,
                     "australia", ndvi, cloud, date)
        print(f"   S2 {date} lat={lat_c:.2f} lon={lon_c:.2f} cloud={cloud:.1f}%")

    # ── Log summary ───────────────────────────────────────────────────────────
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("""INSERT INTO sentinel_ingest_log (source,products_found,anomalies)
        VALUES (?,?,?)""", ("sentinel5p+emit+s2", total_methane, anomalies))
    conn.commit(); conn.close()

    print(f"   Total methane records: {total_methane} | Anomalies: {anomalies}")


def _sigterm(s, f):
    global _running
    _running = False


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT,  _sigterm)
    _db_init()
    print(f"[fm-sentinel-ingest] Active | interval={INTERVAL}s | "
          f"CDSE={'set' if CDSE_USER else 'anon (public access)'} | "
          f"NASA EMIT=public | AOI=[{AOI_LAT_MIN:.0f},{AOI_LON_MIN:.0f}:"
          f"{AOI_LAT_MAX:.0f},{AOI_LON_MAX:.0f}]")
    while _running:
        try:
            run_cycle()
        except Exception as e:
            print(f"[fm-sentinel-ingest] ERR {e}")
        for _ in range(INTERVAL):
            if not _running: break
            time.sleep(1)
    print("[fm-sentinel-ingest] Stopped.")
