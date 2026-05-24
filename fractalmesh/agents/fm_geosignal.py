#!/usr/bin/env python3
"""
FractalMesh OMEGA Titan — GeoSignal / Location Intelligence Agent
Port: 7844
stdlib only: os, json, sqlite3, time, http.server, threading, math,
             urllib, pathlib, socket, hashlib, base64
"""

import os
import json
import sqlite3
import time
import threading
import math
import hashlib
import base64
import socket
import urllib.request
import urllib.parse
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

# ---------------------------------------------------------------------------
# Config / vault
# ---------------------------------------------------------------------------

ROOT = os.environ.get("FRACTALMESH_HOME", str(Path.home() / "fmsaas"))
DB_PATH = os.path.join(ROOT, "database", "sovereign.db")
VAULT_PATHS = [
    os.path.join(ROOT, ".env"),
    str(Path.home() / ".env"),
    str(Path.home() / ".secrets/fractal.env"),
]


def _load_env(key: str, default: str = "") -> str:
    for fpath in VAULT_PATHS:
        try:
            for line in Path(fpath).read_text().splitlines():
                s = line.strip()
                if s.startswith(key + "=") and not s.startswith("#"):
                    val = s.split("=", 1)[1].strip().strip('"').strip("'")
                    if val and not val.startswith("YOUR_"):
                        return val
        except Exception:
            pass
    return os.environ.get(key, default)


PORT = int(_load_env("GEOSIGNAL_PORT", "7844"))

# ---------------------------------------------------------------------------
# Database bootstrap
# ---------------------------------------------------------------------------

def _db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    return con


def _init_db():
    con = _db()
    con.executescript("""
        CREATE TABLE IF NOT EXISTS geo_observations (
            id         INTEGER PRIMARY KEY,
            source     TEXT,
            lat        REAL,
            lon        REAL,
            altitude   REAL,
            data       TEXT,
            created_at REAL
        );
        CREATE TABLE IF NOT EXISTS geo_targets (
            id          INTEGER PRIMARY KEY,
            name        TEXT UNIQUE,
            lat         REAL,
            lon         REAL,
            radius_km   REAL,
            description TEXT,
            active      INTEGER DEFAULT 1,
            created_at  REAL
        );
        CREATE TABLE IF NOT EXISTS iss_positions (
            id           INTEGER PRIMARY KEY,
            lat          REAL,
            lon          REAL,
            altitude_km  REAL,
            velocity_kms REAL,
            timestamp    REAL
        );
        CREATE TABLE IF NOT EXISTS apod_cache (
            id          INTEGER PRIMARY KEY,
            date        TEXT UNIQUE,
            title       TEXT,
            explanation TEXT,
            url         TEXT,
            media_type  TEXT,
            created_at  REAL
        );
    """)
    con.commit()
    # Seed Albury target
    try:
        con.execute(
            """INSERT OR IGNORE INTO geo_targets
               (name, lat, lon, radius_km, description, active, created_at)
               VALUES (?,?,?,?,?,1,?)""",
            ("Albury CBD", -36.0737, 146.9135, 10.0, "Primary AO", time.time()),
        )
        con.commit()
    except Exception:
        pass
    con.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _wigle_bounds(lat: float, lon: float, radius_km: float):
    """Approximate bounding box for WiGLE search."""
    deg_lat = radius_km / 111.0
    deg_lon = radius_km / (111.0 * math.cos(math.radians(lat)) + 1e-9)
    return (lat - deg_lat, lat + deg_lat, lon - deg_lon, lon + deg_lon)


def _wigle_search(lat: float, lon: float, radius_km: float) -> list:
    """Poll WiGLE for WiFi networks, anonymise BSSIDs."""
    api_name = _load_env("WIGLE_API_NAME", "")
    api_token = _load_env("WIGLE_API_TOKEN", "")
    if not api_name or not api_token:
        return []
    lat1, lat2, lon1, lon2 = _wigle_bounds(lat, lon, radius_km)
    params = urllib.parse.urlencode({
        "latrange1": lat1,
        "latrange2": lat2,
        "longrange1": lon1,
        "longrange2": lon2,
        "resultsPerPage": 100,
    })
    url = f"https://api.wigle.net/api/v2/network/search?{params}"
    creds = base64.b64encode(f"{api_name}:{api_token}".encode()).decode()
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Basic {creds}",
            "User-Agent": "FractalMesh-GeoSignal/1.0",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as r:
            data = json.loads(r.read())
        results = data.get("results", [])
        networks = []
        for net in results:
            bssid = net.get("netid", "")
            anon_bssid = hashlib.sha256(bssid.encode()).hexdigest()[:16]
            networks.append({
                "bssid_hash": anon_bssid,
                "ssid": net.get("ssid", ""),
                "lat": net.get("trilat", 0.0),
                "lon": net.get("trilong", 0.0),
                "signal": net.get("signal", None),
                "encryption": net.get("encryption", ""),
                "channel": net.get("channel", None),
            })
        return networks
    except Exception:
        return []


def _country_from_lat_lon(lat: float, lon: float) -> str:
    """Very rough continent/country guess from lat/lon."""
    if -45 < lat < -10 and 110 < lon < 155:
        return "Australia"
    if 24 < lat < 72 and -168 < lon < -52:
        return "North America"
    if -56 < lat < 15 and -82 < lon < -34:
        return "South America"
    if 35 < lat < 72 and -25 < lon < 60:
        return "Europe"
    if -35 < lat < 37 and -18 < lon < 52:
        return "Africa"
    if 5 < lat < 55 and 60 < lon < 150:
        return "Asia"
    if lat < -60:
        return "Antarctica"
    return "Ocean"


# ---------------------------------------------------------------------------
# Background: ISS tracker
# ---------------------------------------------------------------------------

_DB_LOCK = threading.Lock()


def _iss_tracker():
    while True:
        try:
            url = "https://api.wheretheiss.at/v1/satellites/25544"
            req = urllib.request.Request(
                url, headers={"User-Agent": "FractalMesh-GeoSignal/1.0"}
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                d = json.loads(r.read())
            lat = float(d.get("latitude", 0))
            lon = float(d.get("longitude", 0))
            alt = float(d.get("altitude", 0))
            vel = float(d.get("velocity", 0))
            ts = float(d.get("timestamp", time.time()))
            with _DB_LOCK:
                con = _db()
                con.execute(
                    "INSERT INTO iss_positions (lat,lon,altitude_km,velocity_kms,timestamp) VALUES (?,?,?,?,?)",
                    (lat, lon, alt, vel, ts),
                )
                # Keep only last 10000 rows
                con.execute(
                    "DELETE FROM iss_positions WHERE id NOT IN (SELECT id FROM iss_positions ORDER BY id DESC LIMIT 10000)"
                )
                con.commit()
                con.close()
            print(f"[GEOSIGNAL] ISS lat={lat:.2f} lon={lon:.2f} alt={alt:.0f}km")
        except Exception as exc:
            print(f"[GEOSIGNAL] ISS tracker error: {exc}")
        time.sleep(60)


# ---------------------------------------------------------------------------
# Background: WiFi scan
# ---------------------------------------------------------------------------

def _wifi_scan():
    while True:
        try:
            with _DB_LOCK:
                con = _db()
                targets = con.execute(
                    "SELECT id,name,lat,lon,radius_km FROM geo_targets WHERE active=1"
                ).fetchall()
                con.close()
            for tid, tname, tlat, tlon, tradius in targets:
                networks = _wigle_search(tlat, tlon, tradius)
                if networks:
                    with _DB_LOCK:
                        con = _db()
                        for net in networks:
                            con.execute(
                                "INSERT INTO geo_observations (source,lat,lon,altitude,data,created_at) VALUES (?,?,?,?,?,?)",
                                (
                                    "wigle",
                                    net["lat"],
                                    net["lon"],
                                    None,
                                    json.dumps(net),
                                    time.time(),
                                ),
                            )
                        con.commit()
                        con.close()
                    print(f"[GEOSIGNAL] WiFi scan target={tname} networks={len(networks)}")
        except Exception as exc:
            print(f"[GEOSIGNAL] WiFi scan error: {exc}")
        time.sleep(300)


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

def _json_response(handler, data, status=200):
    body = json.dumps(data, ensure_ascii=False).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def _read_body(handler) -> dict:
    length = int(handler.headers.get("Content-Length", 0))
    if length == 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _parse_qs(path: str) -> dict:
    if "?" in path:
        return dict(urllib.parse.parse_qsl(path.split("?", 1)[1]))
    return {}


class GeoSignalHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass  # suppress default access log noise

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        qs = _parse_qs(self.path)

        if path == "/health":
            _json_response(self, {
                "status": "ok",
                "service": "fm-geosignal",
                "port": PORT,
            })

        elif path == "/iss":
            self._handle_iss(qs)

        elif path == "/iss/history":
            self._handle_iss_history(qs)

        elif path == "/iss/next_pass":
            self._handle_iss_next_pass(qs, {})

        elif path == "/targets":
            self._handle_targets_list()

        elif path.startswith("/targets/") and path.endswith("/observations"):
            parts = path.split("/")
            try:
                tid = int(parts[2])
            except (IndexError, ValueError):
                _json_response(self, {"error": "invalid target id"}, 400)
                return
            self._handle_target_observations(tid)

        elif path == "/observations":
            self._handle_observations(qs)

        elif path == "/nasa/apod":
            self._handle_nasa_apod()

        elif path == "/nasa/apod/history":
            self._handle_nasa_apod_history()

        elif path == "/geo/nearby":
            self._handle_geo_nearby(qs)

        elif path == "/geo/heatmap":
            self._handle_geo_heatmap(qs)

        elif path == "/analytics":
            self._handle_analytics()

        else:
            _json_response(self, {"error": "not found"}, 404)

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        qs = _parse_qs(self.path)
        body = _read_body(self)

        if path == "/targets/add":
            self._handle_targets_add(body)

        elif path.startswith("/targets/") and path.endswith("/scan"):
            parts = path.split("/")
            try:
                tid = int(parts[2])
            except (IndexError, ValueError):
                _json_response(self, {"error": "invalid target id"}, 400)
                return
            self._handle_target_scan(tid)

        elif path == "/iss/next_pass":
            self._handle_iss_next_pass(qs, body)

        elif path == "/nasa/earth":
            self._handle_nasa_earth(body)

        else:
            _json_response(self, {"error": "not found"}, 404)

    # -----------------------------------------------------------------------
    # ISS
    # -----------------------------------------------------------------------

    def _handle_iss(self, qs):
        # Live fetch
        live = {}
        try:
            req = urllib.request.Request(
                "https://api.wheretheiss.at/v1/satellites/25544",
                headers={"User-Agent": "FractalMesh-GeoSignal/1.0"},
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                d = json.loads(r.read())
            live = {
                "lat": float(d.get("latitude", 0)),
                "lon": float(d.get("longitude", 0)),
                "altitude_km": float(d.get("altitude", 0)),
                "velocity_kms": float(d.get("velocity", 0)),
                "timestamp": float(d.get("timestamp", time.time())),
            }
            live["pass_over"] = _country_from_lat_lon(live["lat"], live["lon"])
        except Exception:
            # Fallback to last DB entry
            with _DB_LOCK:
                con = _db()
                row = con.execute(
                    "SELECT lat,lon,altitude_km,velocity_kms,timestamp FROM iss_positions ORDER BY id DESC LIMIT 1"
                ).fetchone()
                con.close()
            if row:
                live = {
                    "lat": row[0], "lon": row[1],
                    "altitude_km": row[2], "velocity_kms": row[3],
                    "timestamp": row[4],
                    "pass_over": _country_from_lat_lon(row[0], row[1]),
                }
            else:
                _json_response(self, {"error": "no ISS data available"}, 503)
                return
        _json_response(self, live)

    def _handle_iss_history(self, qs):
        limit = min(int(qs.get("limit", 100)), 1000)
        with _DB_LOCK:
            con = _db()
            rows = con.execute(
                "SELECT lat,lon,altitude_km,velocity_kms,timestamp FROM iss_positions ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            con.close()
        positions = [
            {"lat": r[0], "lon": r[1], "altitude_km": r[2], "velocity_kms": r[3], "timestamp": r[4]}
            for r in rows
        ]
        _json_response(self, {"positions": positions, "count": len(positions)})

    def _handle_iss_next_pass(self, qs, body):
        try:
            obs_lat = float(body.get("lat") or qs.get("lat", 0))
            obs_lon = float(body.get("lon") or qs.get("lon", 0))
        except (TypeError, ValueError):
            _json_response(self, {"error": "lat and lon required"}, 400)
            return

        with _DB_LOCK:
            con = _db()
            rows = con.execute(
                "SELECT lat,lon,altitude_km,timestamp FROM iss_positions ORDER BY id DESC LIMIT 1440"
            ).fetchall()
            con.close()

        if not rows:
            _json_response(self, {"error": "insufficient ISS history"}, 503)
            return

        best_dist = None
        best_ts = None
        for row in rows:
            dist = _haversine(obs_lat, obs_lon, row[0], row[1])
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_ts = row[3]

        # Estimate next pass: ISS orbits ~90 min; find soonest future window
        orbit_period_s = 5400
        now = time.time()
        elapsed = (now - best_ts) % orbit_period_s
        seconds_to_next = orbit_period_s - elapsed
        next_pass_ts = now + seconds_to_next
        next_pass_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(next_pass_ts))

        _json_response(self, {
            "observer_lat": obs_lat,
            "observer_lon": obs_lon,
            "next_pass_estimate": next_pass_str,
            "closest_approach_km": round(best_dist, 1) if best_dist else None,
        })

    # -----------------------------------------------------------------------
    # Targets
    # -----------------------------------------------------------------------

    def _handle_targets_list(self):
        with _DB_LOCK:
            con = _db()
            rows = con.execute(
                "SELECT id,name,lat,lon,radius_km,description,active,created_at FROM geo_targets ORDER BY id"
            ).fetchall()
            con.close()
        targets = [
            {
                "id": r[0], "name": r[1], "lat": r[2], "lon": r[3],
                "radius_km": r[4], "description": r[5],
                "active": bool(r[6]), "created_at": r[7],
            }
            for r in rows
        ]
        _json_response(self, {"targets": targets, "count": len(targets)})

    def _handle_targets_add(self, body):
        name = body.get("name", "").strip()
        try:
            lat = float(body["lat"])
            lon = float(body["lon"])
            radius_km = float(body.get("radius_km", 5.0))
        except (KeyError, TypeError, ValueError):
            _json_response(self, {"error": "lat, lon required"}, 400)
            return
        if not name:
            _json_response(self, {"error": "name required"}, 400)
            return
        description = body.get("description", "")
        with _DB_LOCK:
            con = _db()
            try:
                cur = con.execute(
                    "INSERT INTO geo_targets (name,lat,lon,radius_km,description,active,created_at) VALUES (?,?,?,?,?,1,?)",
                    (name, lat, lon, radius_km, description, time.time()),
                )
                con.commit()
                target_id = cur.lastrowid
            except sqlite3.IntegrityError:
                con.close()
                _json_response(self, {"error": f"target '{name}' already exists"}, 409)
                return
            con.close()
        _json_response(self, {"target_id": target_id}, 201)

    def _handle_target_scan(self, target_id):
        with _DB_LOCK:
            con = _db()
            row = con.execute(
                "SELECT name,lat,lon,radius_km FROM geo_targets WHERE id=?", (target_id,)
            ).fetchone()
            con.close()
        if not row:
            _json_response(self, {"error": "target not found"}, 404)
            return
        tname, tlat, tlon, tradius = row
        networks = _wigle_search(tlat, tlon, tradius)
        observations = []
        if networks:
            with _DB_LOCK:
                con = _db()
                for net in networks:
                    con.execute(
                        "INSERT INTO geo_observations (source,lat,lon,altitude,data,created_at) VALUES (?,?,?,?,?,?)",
                        ("wigle", net["lat"], net["lon"], None, json.dumps(net), time.time()),
                    )
                    observations.append(net)
                con.commit()
                con.close()
        _json_response(self, {
            "target": tname,
            "networks_found": len(networks),
            "observations": observations,
        })

    def _handle_target_observations(self, target_id):
        with _DB_LOCK:
            con = _db()
            trow = con.execute(
                "SELECT name,lat,lon,radius_km FROM geo_targets WHERE id=?", (target_id,)
            ).fetchone()
            if not trow:
                con.close()
                _json_response(self, {"error": "target not found"}, 404)
                return
            tname, tlat, tlon, tradius = trow
            rows = con.execute(
                "SELECT source,lat,lon,altitude,data,created_at FROM geo_observations ORDER BY created_at DESC LIMIT 5000"
            ).fetchall()
            con.close()

        obs = []
        for r in rows:
            if r[1] is not None and r[2] is not None:
                dist = _haversine(tlat, tlon, r[1], r[2])
                if dist <= tradius:
                    try:
                        data = json.loads(r[4]) if r[4] else {}
                    except Exception:
                        data = {}
                    obs.append({
                        "source": r[0], "lat": r[1], "lon": r[2],
                        "altitude": r[3], "data": data, "created_at": r[5],
                    })

        _json_response(self, {"target": tname, "observations": obs, "count": len(obs)})

    # -----------------------------------------------------------------------
    # Observations
    # -----------------------------------------------------------------------

    def _handle_observations(self, qs):
        source = qs.get("source", "")
        limit = min(int(qs.get("limit", 100)), 1000)
        with _DB_LOCK:
            con = _db()
            if source:
                rows = con.execute(
                    "SELECT id,source,lat,lon,altitude,data,created_at FROM geo_observations WHERE source=? ORDER BY id DESC LIMIT ?",
                    (source, limit),
                ).fetchall()
            else:
                rows = con.execute(
                    "SELECT id,source,lat,lon,altitude,data,created_at FROM geo_observations ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            con.close()
        result = []
        for r in rows:
            try:
                data = json.loads(r[5]) if r[5] else {}
            except Exception:
                data = {}
            result.append({
                "id": r[0], "source": r[1], "lat": r[2], "lon": r[3],
                "altitude": r[4], "data": data, "created_at": r[6],
            })
        _json_response(self, {"observations": result, "count": len(result)})

    # -----------------------------------------------------------------------
    # NASA
    # -----------------------------------------------------------------------

    def _handle_nasa_apod(self):
        nasa_key = _load_env("NASA_API_KEY", "DEMO_KEY")
        today = time.strftime("%Y-%m-%d")

        # Check cache first
        with _DB_LOCK:
            con = _db()
            cached = con.execute(
                "SELECT date,title,explanation,url,media_type FROM apod_cache WHERE date=?", (today,)
            ).fetchone()
            con.close()

        if cached:
            _json_response(self, {
                "date": cached[0], "title": cached[1],
                "explanation": cached[2], "url": cached[3],
                "media_type": cached[4], "cached": True,
            })
            return

        try:
            url = f"https://api.nasa.gov/planetary/apod?api_key={nasa_key}&count=1"
            req = urllib.request.Request(url, headers={"User-Agent": "FractalMesh-GeoSignal/1.0"})
            with urllib.request.urlopen(req, timeout=12) as r:
                items = json.loads(r.read())
            item = items[0] if isinstance(items, list) else items
            date = item.get("date", today)
            title = item.get("title", "")
            explanation = item.get("explanation", "")
            img_url = item.get("url", "")
            media_type = item.get("media_type", "image")

            with _DB_LOCK:
                con = _db()
                con.execute(
                    "INSERT OR REPLACE INTO apod_cache (date,title,explanation,url,media_type,created_at) VALUES (?,?,?,?,?,?)",
                    (date, title, explanation, img_url, media_type, time.time()),
                )
                con.commit()
                con.close()

            _json_response(self, {
                "date": date, "title": title,
                "explanation": explanation, "url": img_url,
                "media_type": media_type, "cached": False,
            })
        except Exception as exc:
            _json_response(self, {"error": f"NASA APOD fetch failed: {exc}"}, 502)

    def _handle_nasa_apod_history(self):
        with _DB_LOCK:
            con = _db()
            rows = con.execute(
                "SELECT date,title,explanation,url,media_type,created_at FROM apod_cache ORDER BY date DESC LIMIT 30"
            ).fetchall()
            con.close()
        entries = [
            {"date": r[0], "title": r[1], "explanation": r[2],
             "url": r[3], "media_type": r[4], "created_at": r[5]}
            for r in rows
        ]
        _json_response(self, {"apod_history": entries, "count": len(entries)})

    def _handle_nasa_earth(self, body):
        nasa_key = _load_env("NASA_API_KEY", "DEMO_KEY")
        try:
            lat = float(body["lat"])
            lon = float(body["lon"])
        except (KeyError, TypeError, ValueError):
            _json_response(self, {"error": "lat and lon required"}, 400)
            return
        date = body.get("date", time.strftime("%Y-%m-%d"))
        dim = float(body.get("dim", 0.1))
        params = urllib.parse.urlencode({
            "lat": lat, "lon": lon,
            "date": date, "dim": dim,
            "api_key": nasa_key,
        })
        url = f"https://api.nasa.gov/planetary/earth/imagery?{params}"
        _json_response(self, {"url": url, "date": date, "lat": lat, "lon": lon, "dim": dim})

    # -----------------------------------------------------------------------
    # Geo queries
    # -----------------------------------------------------------------------

    def _handle_geo_nearby(self, qs):
        try:
            lat = float(qs["lat"])
            lon = float(qs["lon"])
            radius_km = float(qs.get("radius_km", 10.0))
        except (KeyError, TypeError, ValueError):
            _json_response(self, {"error": "lat, lon required"}, 400)
            return
        source = qs.get("source", "")
        with _DB_LOCK:
            con = _db()
            if source:
                rows = con.execute(
                    "SELECT source,lat,lon,altitude,data,created_at FROM geo_observations WHERE source=? AND lat IS NOT NULL AND lon IS NOT NULL ORDER BY id DESC LIMIT 5000",
                    (source,),
                ).fetchall()
            else:
                rows = con.execute(
                    "SELECT source,lat,lon,altitude,data,created_at FROM geo_observations WHERE lat IS NOT NULL AND lon IS NOT NULL ORDER BY id DESC LIMIT 5000"
                ).fetchall()
            con.close()
        results = []
        for r in rows:
            dist = _haversine(lat, lon, r[1], r[2])
            if dist <= radius_km:
                try:
                    data = json.loads(r[4]) if r[4] else {}
                except Exception:
                    data = {}
                results.append({
                    "source": r[0], "lat": r[1], "lon": r[2],
                    "altitude": r[3], "data": data,
                    "created_at": r[5], "distance_km": round(dist, 3),
                })
        results.sort(key=lambda x: x["distance_km"])
        _json_response(self, {"nearby": results, "count": len(results),
                               "query_lat": lat, "query_lon": lon, "radius_km": radius_km})

    def _handle_geo_heatmap(self, qs):
        source = qs.get("source", "")
        limit = min(int(qs.get("limit", 500)), 5000)
        with _DB_LOCK:
            con = _db()
            if source:
                rows = con.execute(
                    "SELECT lat,lon FROM geo_observations WHERE source=? AND lat IS NOT NULL AND lon IS NOT NULL ORDER BY id DESC LIMIT ?",
                    (source, limit),
                ).fetchall()
            else:
                rows = con.execute(
                    "SELECT lat,lon FROM geo_observations WHERE lat IS NOT NULL AND lon IS NOT NULL ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            con.close()

        # Aggregate into ~0.01-degree grid cells
        grid = {}
        for lat, lon in rows:
            cell = (round(lat, 2), round(lon, 2))
            grid[cell] = grid.get(cell, 0) + 1
        heatmap = [{"lat": k[0], "lon": k[1], "count": v} for k, v in grid.items()]
        heatmap.sort(key=lambda x: -x["count"])
        _json_response(self, heatmap)

    # -----------------------------------------------------------------------
    # Analytics
    # -----------------------------------------------------------------------

    def _handle_analytics(self):
        with _DB_LOCK:
            con = _db()
            iss_count = con.execute("SELECT COUNT(*) FROM iss_positions").fetchone()[0]
            wifi_count = con.execute(
                "SELECT COUNT(*) FROM geo_observations WHERE source='wigle'"
            ).fetchone()[0]
            apod_count = con.execute("SELECT COUNT(*) FROM apod_cache").fetchone()[0]
            targets_active = con.execute(
                "SELECT COUNT(*) FROM geo_targets WHERE active=1"
            ).fetchone()[0]
            by_source = con.execute(
                "SELECT source, COUNT(*) FROM geo_observations GROUP BY source"
            ).fetchall()
            con.close()
        _json_response(self, {
            "iss_passes_logged": iss_count,
            "wifi_networks_observed": wifi_count,
            "apod_entries_cached": apod_count,
            "targets_active": targets_active,
            "observations_by_source": {r[0]: r[1] for r in by_source},
        })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"[GEOSIGNAL] Initialising database at {DB_PATH}")
    _init_db()

    print("[GEOSIGNAL] Starting ISS tracker thread (60s interval)")
    threading.Thread(target=_iss_tracker, daemon=True, name="iss-tracker").start()

    print("[GEOSIGNAL] Starting WiFi scan thread (300s interval)")
    threading.Thread(target=_wifi_scan, daemon=True, name="wifi-scan").start()

    server = HTTPServer(("0.0.0.0", PORT), GeoSignalHandler)
    print(f"[GEOSIGNAL] GeoSignal / Location Intelligence listening on :{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[GEOSIGNAL] Shutting down.")
        server.server_close()
