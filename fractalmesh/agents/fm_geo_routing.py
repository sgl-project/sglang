#!/usr/bin/env python3
"""
fm_geo_routing.py — FractalMesh OMEGA Titan Geo-Routing & Localisation Engine (Port 7891)
IP geolocation, currency conversion, content routing, and localisation.
Lookup visitor location, route to correct regional config, handle currency/timezone/language preferences.
Samuel James Hiotis | ABN 56 628 117 363
"""
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request as URLRequest
from urllib.request import urlopen

# ---------------------------------------------------------------------------
# Vault / env bootstrap — MUST be before any os.getenv calls
# ---------------------------------------------------------------------------
_ENV_FILE = Path.home() / ".secrets" / "fractal.env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

PORT          = int(os.getenv("GEO_ROUTING_PORT", "7891"))
IPSTACK_KEY   = os.getenv("IPSTACK_API_KEY", "")
ADMIN_SECRET  = os.getenv("ADMIN_SECRET", "")
ROOT          = Path.home() / "fmsaas"
DB            = ROOT / "database" / "sovereign.db"

for _p in (ROOT, DB.parent):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Common timezone → UTC offset mapping
# ---------------------------------------------------------------------------
TIMEZONE_OFFSETS: dict[str, str] = {
    "UTC":                    "+00:00",
    "Europe/London":          "+00:00",
    "Europe/Dublin":          "+00:00",
    "Europe/Lisbon":          "+00:00",
    "Europe/Paris":           "+01:00",
    "Europe/Berlin":          "+01:00",
    "Europe/Rome":            "+01:00",
    "Europe/Madrid":          "+01:00",
    "Europe/Amsterdam":       "+01:00",
    "Europe/Brussels":        "+01:00",
    "Europe/Vienna":          "+01:00",
    "Europe/Zurich":          "+01:00",
    "Europe/Warsaw":          "+01:00",
    "Europe/Prague":          "+01:00",
    "Europe/Stockholm":       "+01:00",
    "Europe/Copenhagen":      "+01:00",
    "Europe/Oslo":            "+01:00",
    "Europe/Helsinki":        "+02:00",
    "Europe/Kyiv":            "+02:00",
    "Europe/Bucharest":       "+02:00",
    "Europe/Sofia":           "+02:00",
    "Europe/Athens":          "+02:00",
    "Europe/Riga":            "+02:00",
    "Europe/Tallinn":         "+02:00",
    "Europe/Vilnius":         "+02:00",
    "Europe/Istanbul":        "+03:00",
    "Europe/Moscow":          "+03:00",
    "Asia/Dubai":             "+04:00",
    "Asia/Baku":              "+04:00",
    "Asia/Tbilisi":           "+04:00",
    "Asia/Yerevan":           "+04:00",
    "Asia/Kabul":             "+04:30",
    "Asia/Karachi":           "+05:00",
    "Asia/Tashkent":          "+05:00",
    "Asia/Colombo":           "+05:30",
    "Asia/Kolkata":           "+05:30",
    "Asia/Kathmandu":         "+05:45",
    "Asia/Dhaka":             "+06:00",
    "Asia/Almaty":            "+06:00",
    "Asia/Yangon":            "+06:30",
    "Asia/Bangkok":           "+07:00",
    "Asia/Ho_Chi_Minh":       "+07:00",
    "Asia/Jakarta":           "+07:00",
    "Asia/Shanghai":          "+08:00",
    "Asia/Hong_Kong":         "+08:00",
    "Asia/Taipei":            "+08:00",
    "Asia/Singapore":         "+08:00",
    "Asia/Kuala_Lumpur":      "+08:00",
    "Asia/Manila":            "+08:00",
    "Asia/Perth":             "+08:00",
    "Asia/Seoul":             "+09:00",
    "Asia/Tokyo":             "+09:00",
    "Australia/Adelaide":     "+09:30",
    "Australia/Darwin":       "+09:30",
    "Australia/Brisbane":     "+10:00",
    "Australia/Sydney":       "+10:00",
    "Australia/Melbourne":    "+10:00",
    "Australia/Hobart":       "+10:00",
    "Pacific/Port_Moresby":   "+10:00",
    "Australia/Lord_Howe":    "+10:30",
    "Pacific/Noumea":         "+11:00",
    "Pacific/Auckland":       "+12:00",
    "Pacific/Fiji":           "+12:00",
    "Pacific/Tongatapu":      "+13:00",
    "Pacific/Apia":           "+13:00",
    "America/New_York":       "-05:00",
    "America/Chicago":        "-06:00",
    "America/Denver":         "-07:00",
    "America/Los_Angeles":    "-08:00",
    "America/Anchorage":      "-09:00",
    "Pacific/Honolulu":       "-10:00",
    "America/Toronto":        "-05:00",
    "America/Vancouver":      "-08:00",
    "America/Winnipeg":       "-06:00",
    "America/Halifax":        "-04:00",
    "America/St_Johns":       "-03:30",
    "America/Sao_Paulo":      "-03:00",
    "America/Buenos_Aires":   "-03:00",
    "America/Santiago":       "-04:00",
    "America/Bogota":         "-05:00",
    "America/Lima":           "-05:00",
    "America/Caracas":        "-04:00",
    "America/Mexico_City":    "-06:00",
    "America/Monterrey":      "-06:00",
    "America/Tijuana":        "-08:00",
    "America/Havana":         "-05:00",
    "America/Jamaica":        "-05:00",
    "America/Santo_Domingo":  "-04:00",
    "America/Puerto_Rico":    "-04:00",
    "Africa/Cairo":           "+02:00",
    "Africa/Johannesburg":    "+02:00",
    "Africa/Nairobi":         "+03:00",
    "Africa/Lagos":           "+01:00",
    "Africa/Accra":           "+00:00",
    "Africa/Casablanca":      "+01:00",
    "Africa/Tunis":           "+01:00",
    "Africa/Algiers":         "+01:00",
    "Africa/Addis_Ababa":     "+03:00",
    "Africa/Khartoum":        "+03:00",
    "Africa/Dar_es_Salaam":   "+03:00",
    "Africa/Kampala":         "+03:00",
    "Africa/Luanda":          "+01:00",
    "Africa/Kinshasa":        "+01:00",
    "Africa/Harare":          "+02:00",
    "Africa/Lusaka":          "+02:00",
    "Africa/Maputo":          "+02:00",
    "Africa/Gaborone":        "+02:00",
    "Africa/Windhoek":        "+02:00",
}

# Default currency rates to seed at startup
DEFAULT_RATES = [
    ("AUD", "USD", 0.65),
    ("AUD", "EUR", 0.59),
    ("AUD", "GBP", 0.50),
    ("AUD", "JPY", 98.5),
    ("AUD", "CAD", 0.89),
    ("AUD", "NZD", 1.08),
    ("USD", "AUD", 1.54),
    ("EUR", "AUD", 1.69),
    ("GBP", "AUD", 2.00),
]

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------
def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(DB), timeout=15, check_same_thread=False)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA busy_timeout=10000")
    c.row_factory = sqlite3.Row
    return c


def _db_init():
    c = _conn()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS geo_cache (
            id           INTEGER PRIMARY KEY,
            ip_hash      TEXT UNIQUE,
            country_code TEXT,
            country_name TEXT,
            region       TEXT,
            city         TEXT,
            latitude     REAL,
            longitude    REAL,
            timezone     TEXT,
            currency     TEXT,
            language     TEXT,
            isp          TEXT,
            cached_at    REAL
        );

        CREATE TABLE IF NOT EXISTS routing_rules (
            id             INTEGER PRIMARY KEY,
            rule_id        TEXT UNIQUE,
            name           TEXT,
            country_codes  TEXT DEFAULT '[]',
            region_pattern TEXT,
            action         TEXT,
            destination    TEXT,
            config         TEXT DEFAULT '{}',
            priority       INTEGER DEFAULT 0,
            active         INTEGER DEFAULT 1,
            created_at     REAL
        );

        CREATE TABLE IF NOT EXISTS currency_rates (
            id            INTEGER PRIMARY KEY,
            from_currency TEXT,
            to_currency   TEXT,
            rate          REAL,
            updated_at    REAL,
            UNIQUE(from_currency, to_currency)
        );

        CREATE TABLE IF NOT EXISTS localisation (
            id           INTEGER PRIMARY KEY,
            locale_key   TEXT UNIQUE,
            locale       TEXT,
            namespace    TEXT,
            translations TEXT DEFAULT '{}',
            created_at   REAL,
            updated_at   REAL
        );

        CREATE TABLE IF NOT EXISTS access_log (
            id           INTEGER PRIMARY KEY,
            ip_hash      TEXT,
            country_code TEXT,
            path         TEXT,
            rule_matched TEXT,
            created_at   REAL
        );

        CREATE INDEX IF NOT EXISTS idx_geo_ip_hash   ON geo_cache(ip_hash);
        CREATE INDEX IF NOT EXISTS idx_geo_cached_at ON geo_cache(cached_at);
        CREATE INDEX IF NOT EXISTS idx_rules_active  ON routing_rules(active, priority);
        CREATE INDEX IF NOT EXISTS idx_log_created   ON access_log(created_at);
        CREATE INDEX IF NOT EXISTS idx_log_country   ON access_log(country_code);
    """)
    c.commit()

    # Seed default currency rates if the table is empty
    existing = c.execute("SELECT COUNT(*) FROM currency_rates").fetchone()[0]
    if existing == 0:
        now = time.time()
        c.executemany(
            "INSERT OR IGNORE INTO currency_rates(from_currency, to_currency, rate, updated_at) VALUES(?,?,?,?)",
            [(f, t, r, now) for f, t, r in DEFAULT_RATES],
        )
        c.commit()
    c.close()


# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------
def _ip_hash(ip: str) -> str:
    """Return first 16 hex chars of SHA-256 of the IP — never store raw IP."""
    return hashlib.sha256(ip.encode()).hexdigest()[:16]


def _check_admin(handler: "GeoHandler") -> bool:
    """Validate X-Admin-Secret header using constant-time comparison."""
    if not ADMIN_SECRET:
        return False
    provided = handler.headers.get("X-Admin-Secret", "")
    if not provided:
        return False
    try:
        return hmac.compare_digest(
            provided.encode("utf-8"),
            ADMIN_SECRET.encode("utf-8"),
        )
    except Exception:
        return False


def _send_json(handler: "GeoHandler", code: int, data: dict):
    body = json.dumps(data, ensure_ascii=False).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_body(handler: "GeoHandler") -> dict:
    length = int(handler.headers.get("Content-Length", "0"))
    if length == 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Geo lookup helpers
# ---------------------------------------------------------------------------
def _ipstack_lookup(ip: str) -> dict:
    """Call ipstack.com API. Returns dict or raises on failure."""
    url = f"https://api.ipstack.com/{ip}?access_key={IPSTACK_KEY}"
    req = URLRequest(url, headers={"User-Agent": "FractalMesh-GeoRouter/1.0"})
    with urlopen(req, timeout=8) as resp:
        data = json.loads(resp.read().decode())
    return data


def _parse_ipstack(data: dict) -> dict:
    """Normalise ipstack response to our internal geo dict."""
    lang_info = data.get("location", {}) or {}
    languages = lang_info.get("languages", []) or []
    language  = languages[0].get("code", "en") if languages else "en"
    return {
        "country_code": (data.get("country_code") or "XX").upper(),
        "country_name": data.get("country_name") or "Unknown",
        "region":       data.get("region_name") or "",
        "city":         data.get("city") or "",
        "latitude":     float(data.get("latitude") or 0.0),
        "longitude":    float(data.get("longitude") or 0.0),
        "timezone":     (data.get("time_zone") or {}).get("id", "UTC"),
        "currency":     (data.get("currency") or {}).get("code", "USD"),
        "language":     language,
        "isp":          (data.get("connection") or {}).get("isp", ""),
    }


def _geo_from_headers(headers) -> dict:
    """Best-effort geo data from CDN/CloudFlare headers when API is unavailable."""
    cc = (headers.get("CF-IPCountry") or
          headers.get("X-Country-Code") or
          headers.get("CloudFront-Viewer-Country") or
          "XX").upper()
    return {
        "country_code": cc,
        "country_name": "Unknown",
        "region":       "",
        "city":         "",
        "latitude":     0.0,
        "longitude":    0.0,
        "timezone":     "UTC",
        "currency":     "USD",
        "language":     "en",
        "isp":          "",
    }


_GEO_CACHE_TTL = 86400     # 24 h


def geo_lookup(ip: str, headers=None, db_conn=None) -> tuple[dict, bool]:
    """
    Return (geo_dict, cache_hit).
    Checks SQLite cache first; falls back to ipstack API; falls back to header hints.
    """
    close_after = db_conn is None
    c = db_conn or _conn()
    try:
        ip_h  = _ip_hash(ip)
        now   = time.time()
        cutoff = now - _GEO_CACHE_TTL

        row = c.execute(
            "SELECT * FROM geo_cache WHERE ip_hash=? AND cached_at>?",
            (ip_h, cutoff),
        ).fetchone()

        if row:
            geo = {
                "country_code": row["country_code"],
                "country_name": row["country_name"],
                "region":       row["region"],
                "city":         row["city"],
                "latitude":     row["latitude"],
                "longitude":    row["longitude"],
                "timezone":     row["timezone"],
                "currency":     row["currency"],
                "language":     row["language"],
                "isp":          row["isp"],
            }
            return geo, True

        # Cache miss — try API
        geo = None
        if IPSTACK_KEY:
            try:
                raw = _ipstack_lookup(ip)
                if raw.get("success") is False:
                    raise RuntimeError(raw.get("error", {}).get("info", "ipstack error"))
                geo = _parse_ipstack(raw)
            except Exception:
                geo = None

        if geo is None:
            geo = _geo_from_headers(headers or {})

        # Persist to cache
        c.execute("""
            INSERT INTO geo_cache
                (ip_hash, country_code, country_name, region, city,
                 latitude, longitude, timezone, currency, language, isp, cached_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(ip_hash) DO UPDATE SET
                country_code=excluded.country_code,
                country_name=excluded.country_name,
                region=excluded.region,
                city=excluded.city,
                latitude=excluded.latitude,
                longitude=excluded.longitude,
                timezone=excluded.timezone,
                currency=excluded.currency,
                language=excluded.language,
                isp=excluded.isp,
                cached_at=excluded.cached_at
        """, (
            ip_h,
            geo["country_code"], geo["country_name"],
            geo["region"],       geo["city"],
            geo["latitude"],     geo["longitude"],
            geo["timezone"],     geo["currency"],
            geo["language"],     geo["isp"],
            now,
        ))
        c.commit()
        return geo, False
    finally:
        if close_after:
            c.close()


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------
def match_routing_rule(geo: dict, path: str, c: sqlite3.Connection) -> dict | None:
    """Return the highest-priority active rule matching this geo+path, or None."""
    cc = geo.get("country_code", "XX")
    region = geo.get("region", "")
    rows = c.execute(
        "SELECT * FROM routing_rules WHERE active=1 ORDER BY priority DESC"
    ).fetchall()

    import re as _re
    for row in rows:
        try:
            codes = json.loads(row["country_codes"] or "[]")
        except Exception:
            codes = []
        pattern = row["region_pattern"] or ""

        country_match = (not codes) or (cc in codes)
        region_match  = (not pattern) or bool(_re.search(pattern, region, _re.IGNORECASE))

        if country_match and region_match:
            try:
                cfg = json.loads(row["config"] or "{}")
            except Exception:
                cfg = {}
            return {
                "rule_id":     row["rule_id"],
                "name":        row["name"],
                "action":      row["action"],
                "destination": row["destination"],
                "config":      cfg,
                "priority":    row["priority"],
            }
    return None


def _log_access(ip: str, country_code: str, path: str, rule_matched: str | None, c: sqlite3.Connection):
    c.execute(
        "INSERT INTO access_log(ip_hash, country_code, path, rule_matched, created_at) VALUES(?,?,?,?,?)",
        (_ip_hash(ip), country_code, path, rule_matched or "", time.time()),
    )
    c.commit()


# ---------------------------------------------------------------------------
# Currency helpers
# ---------------------------------------------------------------------------
def _get_rate(from_c: str, to_c: str, c: sqlite3.Connection) -> float | None:
    row = c.execute(
        "SELECT rate FROM currency_rates WHERE from_currency=? AND to_currency=?",
        (from_c.upper(), to_c.upper()),
    ).fetchone()
    return row["rate"] if row else None


# ---------------------------------------------------------------------------
# Background cleanup daemon
# ---------------------------------------------------------------------------
def _cleanup_daemon():
    while True:
        time.sleep(3600)
        try:
            c = _conn()
            now = time.time()
            geo_cutoff = now - 7 * 86400    # 7 days
            log_cutoff = now - 30 * 86400   # 30 days
            c.execute("DELETE FROM geo_cache  WHERE cached_at < ?", (geo_cutoff,))
            c.execute("DELETE FROM access_log WHERE created_at < ?", (log_cutoff,))
            c.commit()
            c.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------
class GeoHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):  # silence default access log
        pass

    # ------------------------------------------------------------------ GET --
    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        parts = [p for p in path.split("/") if p]

        # GET /health
        if path == "/health":
            _send_json(self, 200, {
                "status": "ok",
                "service": "fm-geo-routing",
                "port": PORT,
                "timestamp": time.time(),
            })
            return

        # GET /timezones
        if path == "/timezones":
            _send_json(self, 200, TIMEZONE_OFFSETS)
            return

        # GET /lookup  — caller's own IP
        if path == "/lookup":
            ip = (
                self.headers.get("X-Forwarded-For", "").split(",")[0].strip()
                or self.client_address[0]
            )
            c = _conn()
            try:
                geo, hit = geo_lookup(ip, self.headers, c)
                geo["cache_hit"] = hit
                _send_json(self, 200, geo)
            finally:
                c.close()
            return

        # GET /lookup/{ip}
        if len(parts) == 2 and parts[0] == "lookup":
            target_ip = parts[1]
            c = _conn()
            try:
                geo, hit = geo_lookup(target_ip, self.headers, c)
                geo["cache_hit"] = hit
                _send_json(self, 200, geo)
            finally:
                c.close()
            return

        # GET /currency/{from}/{to}
        if len(parts) == 3 and parts[0] == "currency":
            from_c, to_c = parts[1].upper(), parts[2].upper()
            c = _conn()
            try:
                rate = _get_rate(from_c, to_c, c)
                if rate is None:
                    _send_json(self, 404, {"error": f"No rate for {from_c}→{to_c}"})
                else:
                    _send_json(self, 200, {
                        "from": from_c,
                        "to":   to_c,
                        "rate": rate,
                    })
            finally:
                c.close()
            return

        # GET /rules  (admin)
        if path == "/rules":
            if not _check_admin(self):
                _send_json(self, 403, {"error": "Forbidden"})
                return
            c = _conn()
            try:
                rows = c.execute(
                    "SELECT * FROM routing_rules ORDER BY priority DESC"
                ).fetchall()
                result = []
                for r in rows:
                    try:
                        codes = json.loads(r["country_codes"] or "[]")
                    except Exception:
                        codes = []
                    try:
                        cfg = json.loads(r["config"] or "{}")
                    except Exception:
                        cfg = {}
                    result.append({
                        "id":             r["id"],
                        "rule_id":        r["rule_id"],
                        "name":           r["name"],
                        "country_codes":  codes,
                        "region_pattern": r["region_pattern"],
                        "action":         r["action"],
                        "destination":    r["destination"],
                        "config":         cfg,
                        "priority":       r["priority"],
                        "active":         bool(r["active"]),
                        "created_at":     r["created_at"],
                    })
                _send_json(self, 200, {"rules": result, "count": len(result)})
            finally:
                c.close()
            return

        # GET /locale/{locale}/{namespace}
        if len(parts) == 3 and parts[0] == "locale":
            locale, namespace = parts[1], parts[2]
            c = _conn()
            try:
                row = c.execute(
                    "SELECT * FROM localisation WHERE locale=? AND namespace=?",
                    (locale, namespace),
                ).fetchone()
                if not row:
                    _send_json(self, 404, {"error": "Locale not found"})
                else:
                    try:
                        trans = json.loads(row["translations"] or "{}")
                    except Exception:
                        trans = {}
                    _send_json(self, 200, {
                        "locale_key":   row["locale_key"],
                        "locale":       row["locale"],
                        "namespace":    row["namespace"],
                        "translations": trans,
                        "updated_at":   row["updated_at"],
                    })
            finally:
                c.close()
            return

        # GET /stats  (admin)
        if path == "/stats":
            if not _check_admin(self):
                _send_json(self, 403, {"error": "Forbidden"})
                return
            c = _conn()
            try:
                top_countries = c.execute("""
                    SELECT country_code, COUNT(*) as hits
                    FROM access_log
                    GROUP BY country_code
                    ORDER BY hits DESC
                    LIMIT 20
                """).fetchall()

                total_lookups = c.execute(
                    "SELECT COUNT(*) FROM geo_cache"
                ).fetchone()[0]

                # Cache-hit rate: how many lookups were served from non-expired cache
                # We approximate: entries still fresh / total entries
                now    = time.time()
                cutoff = now - _GEO_CACHE_TTL
                fresh  = c.execute(
                    "SELECT COUNT(*) FROM geo_cache WHERE cached_at > ?", (cutoff,)
                ).fetchone()[0]
                cache_hit_rate = round(fresh / total_lookups, 4) if total_lookups else 0.0

                total_log = c.execute("SELECT COUNT(*) FROM access_log").fetchone()[0]

                _send_json(self, 200, {
                    "total_geo_cached":    total_lookups,
                    "fresh_cache_entries": fresh,
                    "cache_hit_rate":      cache_hit_rate,
                    "total_access_logs":   total_log,
                    "top_countries": [
                        {"country_code": r["country_code"], "hits": r["hits"]}
                        for r in top_countries
                    ],
                })
            finally:
                c.close()
            return

        _send_json(self, 404, {"error": "Not found"})

    # ----------------------------------------------------------------- POST --
    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        parts = [p for p in path.split("/") if p]
        body  = _read_body(self)

        # POST /route
        if path == "/route":
            ip       = body.get("ip") or (
                self.headers.get("X-Forwarded-For", "").split(",")[0].strip()
                or self.client_address[0]
            )
            req_path = body.get("path", "/")
            c = _conn()
            try:
                geo, hit = geo_lookup(ip, self.headers, c)
                rule     = match_routing_rule(geo, req_path, c)
                _log_access(ip, geo["country_code"], req_path,
                            rule["rule_id"] if rule else None, c)
                _send_json(self, 200, {
                    "rule_matched": rule["rule_id"] if rule else None,
                    "action":       rule["action"]      if rule else None,
                    "destination":  rule["destination"] if rule else None,
                    "config":       rule["config"]      if rule else {},
                    "geo":          geo,
                    "cache_hit":    hit,
                })
            finally:
                c.close()
            return

        # POST /currency/convert
        if path == "/currency/convert":
            amount = body.get("amount")
            from_c = (body.get("from") or "").upper()
            to_c   = (body.get("to") or "").upper()
            if not all([amount, from_c, to_c]):
                _send_json(self, 400, {"error": "amount, from, to required"})
                return
            try:
                amount = float(amount)
            except (TypeError, ValueError):
                _send_json(self, 400, {"error": "amount must be numeric"})
                return
            c = _conn()
            try:
                rate = _get_rate(from_c, to_c, c)
                if rate is None:
                    _send_json(self, 404, {"error": f"No rate for {from_c}→{to_c}"})
                else:
                    converted = round(amount * rate, 6)
                    _send_json(self, 200, {
                        "from":      from_c,
                        "to":        to_c,
                        "amount":    amount,
                        "rate":      rate,
                        "converted": converted,
                    })
            finally:
                c.close()
            return

        # POST /currency/rates  (admin)
        if path == "/currency/rates":
            if not _check_admin(self):
                _send_json(self, 403, {"error": "Forbidden"})
                return
            rates = body.get("rates", {})
            if not isinstance(rates, dict) or not rates:
                _send_json(self, 400, {"error": "rates dict required"})
                return
            c = _conn()
            now = time.time()
            try:
                upserted = 0
                for key, rate_val in rates.items():
                    try:
                        # Accept "AUD_USD" or "AUD-USD" or "AUDUSD" (3+3)
                        if "_" in key:
                            from_c, to_c = key.split("_", 1)
                        elif "-" in key:
                            from_c, to_c = key.split("-", 1)
                        elif len(key) == 6:
                            from_c, to_c = key[:3], key[3:]
                        else:
                            continue
                        from_c = from_c.strip().upper()
                        to_c   = to_c.strip().upper()
                        rate_f = float(rate_val)
                        c.execute("""
                            INSERT INTO currency_rates(from_currency, to_currency, rate, updated_at)
                            VALUES(?,?,?,?)
                            ON CONFLICT(from_currency, to_currency) DO UPDATE SET
                                rate=excluded.rate, updated_at=excluded.updated_at
                        """, (from_c, to_c, rate_f, now))
                        upserted += 1
                    except Exception:
                        continue
                c.commit()
                _send_json(self, 200, {"upserted": upserted})
            finally:
                c.close()
            return

        # POST /rules  (admin)
        if path == "/rules":
            if not _check_admin(self):
                _send_json(self, 403, {"error": "Forbidden"})
                return
            name     = body.get("name")
            action   = body.get("action")
            dest     = body.get("destination")
            if not all([name, action, dest]):
                _send_json(self, 400, {"error": "name, action, destination required"})
                return
            if action not in ("redirect", "block", "localise", "custom"):
                _send_json(self, 400, {"error": "action must be redirect|block|localise|custom"})
                return
            rule_id  = secrets.token_hex(8)
            codes    = json.dumps(body.get("country_codes") or [])
            pattern  = body.get("region_pattern") or ""
            cfg      = json.dumps(body.get("config") or {})
            priority = int(body.get("priority", 0))
            now      = time.time()
            c = _conn()
            try:
                c.execute("""
                    INSERT INTO routing_rules
                        (rule_id, name, country_codes, region_pattern, action,
                         destination, config, priority, active, created_at)
                    VALUES(?,?,?,?,?,?,?,?,1,?)
                """, (rule_id, name, codes, pattern, action, dest, cfg, priority, now))
                c.commit()
                _send_json(self, 201, {
                    "rule_id":  rule_id,
                    "name":     name,
                    "action":   action,
                    "priority": priority,
                    "active":   True,
                })
            finally:
                c.close()
            return

        # POST /locale  (admin)
        if path == "/locale":
            if not _check_admin(self):
                _send_json(self, 403, {"error": "Forbidden"})
                return
            locale_key = body.get("locale_key")
            locale     = body.get("locale")
            namespace  = body.get("namespace")
            trans      = body.get("translations") or {}
            if not all([locale_key, locale, namespace]):
                _send_json(self, 400, {"error": "locale_key, locale, namespace required"})
                return
            now = time.time()
            c = _conn()
            try:
                c.execute("""
                    INSERT INTO localisation
                        (locale_key, locale, namespace, translations, created_at, updated_at)
                    VALUES(?,?,?,?,?,?)
                    ON CONFLICT(locale_key) DO UPDATE SET
                        locale=excluded.locale,
                        namespace=excluded.namespace,
                        translations=excluded.translations,
                        updated_at=excluded.updated_at
                """, (locale_key, locale, namespace, json.dumps(trans), now, now))
                c.commit()
                _send_json(self, 200, {
                    "locale_key": locale_key,
                    "locale":     locale,
                    "namespace":  namespace,
                    "updated_at": now,
                })
            finally:
                c.close()
            return

        _send_json(self, 404, {"error": "Not found"})

    # ------------------------------------------------------------------ PUT --
    def do_PUT(self):
        path  = self.path.split("?")[0].rstrip("/")
        parts = [p for p in path.split("/") if p]
        body  = _read_body(self)

        # PUT /rules/{rule_id}  (admin)
        if len(parts) == 2 and parts[0] == "rules":
            if not _check_admin(self):
                _send_json(self, 403, {"error": "Forbidden"})
                return
            rule_id = parts[1]
            c = _conn()
            try:
                row = c.execute(
                    "SELECT * FROM routing_rules WHERE rule_id=?", (rule_id,)
                ).fetchone()
                if not row:
                    _send_json(self, 404, {"error": "Rule not found"})
                    return

                # Build update from provided fields
                fields: list[str] = []
                values: list      = []

                if "name" in body:
                    fields.append("name=?"); values.append(body["name"])
                if "country_codes" in body:
                    fields.append("country_codes=?")
                    values.append(json.dumps(body["country_codes"]))
                if "region_pattern" in body:
                    fields.append("region_pattern=?"); values.append(body["region_pattern"])
                if "action" in body:
                    act = body["action"]
                    if act not in ("redirect", "block", "localise", "custom"):
                        _send_json(self, 400, {"error": "Invalid action"})
                        return
                    fields.append("action=?"); values.append(act)
                if "destination" in body:
                    fields.append("destination=?"); values.append(body["destination"])
                if "config" in body:
                    fields.append("config=?"); values.append(json.dumps(body["config"]))
                if "priority" in body:
                    fields.append("priority=?"); values.append(int(body["priority"]))
                if "active" in body:
                    fields.append("active=?"); values.append(1 if body["active"] else 0)

                if not fields:
                    _send_json(self, 400, {"error": "No updatable fields provided"})
                    return

                values.append(rule_id)
                c.execute(
                    f"UPDATE routing_rules SET {', '.join(fields)} WHERE rule_id=?",
                    values,
                )
                c.commit()
                _send_json(self, 200, {"rule_id": rule_id, "updated": True})
            finally:
                c.close()
            return

        _send_json(self, 404, {"error": "Not found"})

    # --------------------------------------------------------------- DELETE --
    def do_DELETE(self):
        path  = self.path.split("?")[0].rstrip("/")
        parts = [p for p in path.split("/") if p]

        # DELETE /rules/{rule_id}  (admin) — soft delete (deactivate)
        if len(parts) == 2 and parts[0] == "rules":
            if not _check_admin(self):
                _send_json(self, 403, {"error": "Forbidden"})
                return
            rule_id = parts[1]
            c = _conn()
            try:
                row = c.execute(
                    "SELECT id FROM routing_rules WHERE rule_id=?", (rule_id,)
                ).fetchone()
                if not row:
                    _send_json(self, 404, {"error": "Rule not found"})
                    return
                c.execute(
                    "UPDATE routing_rules SET active=0 WHERE rule_id=?", (rule_id,)
                )
                c.commit()
                _send_json(self, 200, {"rule_id": rule_id, "deactivated": True})
            finally:
                c.close()
            return

        _send_json(self, 404, {"error": "Not found"})


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main():
    _db_init()

    # Start background cleanup daemon
    t = threading.Thread(target=_cleanup_daemon, daemon=True)
    t.start()

    server = HTTPServer(("0.0.0.0", PORT), GeoHandler)
    print(f"[fm-geo-routing] Listening on port {PORT}  (pid {os.getpid()})")
    print(f"[fm-geo-routing] DB: {DB}")
    print(f"[fm-geo-routing] IPSTACK key: {'configured' if IPSTACK_KEY else 'NOT SET — header fallback only'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[fm-geo-routing] Shutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
