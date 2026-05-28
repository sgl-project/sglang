#!/usr/bin/env python3
"""
fm_data_pipeline.py — FractalMesh OMEGA Titan Data Pipeline & ETL Engine
Port: 7874

ETL (Extract, Transform, Load) pipeline engine.
Defines data sources, transformations, and destinations.
Runs pipelines on schedule or on demand.
Supports CSV imports, JSON API ingestion, cross-table aggregations, and data export.

Samuel James Hiotis | ABN 56 628 117 363
"""

import base64
import csv
import gzip
import hashlib
import hmac
import io
import json
import os
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ---------------------------------------------------------------------------
# Vault loading
# ---------------------------------------------------------------------------
_ENV_FILE = Path.home() / ".secrets" / "fractal.env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PORT                 = int(os.environ.get("DATA_PIPELINE_PORT", "7874"))
NEON_DB_URL          = os.environ.get("NEON_DB_URL", "")          # stored for reference; no psycopg2
SUPABASE_URL         = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
ADMIN_SECRET         = os.environ.get("ADMIN_SECRET", "")

ROOT        = Path.home() / "fmsaas"
DB_PATH     = ROOT / "database" / "sovereign.db"
EXPORTS_DIR = ROOT / "exports"
LOG_PATH    = ROOT / "logs" / "data_pipeline.log"

for _d in (DB_PATH.parent, EXPORTS_DIR, LOG_PATH.parent):
    _d.mkdir(parents=True, exist_ok=True)

START_TIME = time.time()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_log_lock = threading.Lock()

def _log(level: str, msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    line = f"{ts} [{level}] {msg}\n"
    with _log_lock:
        try:
            with open(LOG_PATH, "a") as fh:
                fh.write(line)
        except OSError:
            pass

def log_info(msg):  _log("INFO",  msg)
def log_warn(msg):  _log("WARN",  msg)
def log_error(msg): _log("ERROR", msg)

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=15, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_db()
    cur  = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS pipelines (
            id               INTEGER PRIMARY KEY,
            name             TEXT    UNIQUE NOT NULL,
            description      TEXT,
            source_type      TEXT    NOT NULL,
            source_config    TEXT    NOT NULL DEFAULT '{}',
            transform_config TEXT    NOT NULL DEFAULT '[]',
            destination_type TEXT    NOT NULL,
            destination_config TEXT  NOT NULL DEFAULT '{}',
            schedule         TEXT,
            enabled          INTEGER NOT NULL DEFAULT 1,
            last_run_at      REAL,
            next_run_at      REAL,
            run_count        INTEGER NOT NULL DEFAULT 0,
            created_at       REAL    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id                INTEGER PRIMARY KEY,
            pipeline_id       INTEGER,
            pipeline_name     TEXT    NOT NULL,
            status            TEXT    NOT NULL DEFAULT 'running',
            rows_extracted    INTEGER NOT NULL DEFAULT 0,
            rows_transformed  INTEGER NOT NULL DEFAULT 0,
            rows_loaded       INTEGER NOT NULL DEFAULT 0,
            rows_failed       INTEGER NOT NULL DEFAULT 0,
            error             TEXT,
            started_at        REAL,
            finished_at       REAL,
            duration_ms       REAL
        );

        CREATE TABLE IF NOT EXISTS data_sources (
            id           INTEGER PRIMARY KEY,
            name         TEXT    UNIQUE NOT NULL,
            source_type  TEXT    NOT NULL,
            config       TEXT    NOT NULL DEFAULT '{}',
            last_tested  REAL,
            status       TEXT    NOT NULL DEFAULT 'untested',
            created_at   REAL    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS exports (
            id         INTEGER PRIMARY KEY,
            name       TEXT    NOT NULL,
            query      TEXT    NOT NULL,
            format     TEXT    NOT NULL,
            output     TEXT,
            row_count  INTEGER,
            created_at REAL    NOT NULL
        );
    """)
    conn.commit()
    conn.close()
    log_info("Database initialised (WAL mode)")


def _seed_pipelines() -> None:
    """Insert default pipelines if the table is empty."""
    conn = get_db()
    count = conn.execute("SELECT COUNT(*) FROM pipelines").fetchone()[0]
    if count > 0:
        conn.close()
        return

    now = time.time()
    defaults = [
        {
            "name": "payments_daily_export",
            "description": "Daily export of payment totals grouped by date (last 30 days)",
            "source_type": "sqlite_query",
            "source_config": json.dumps({
                "query": (
                    "SELECT date(created_at,'unixepoch') as date, "
                    "COUNT(*) as count, SUM(amount)/100.0 as total "
                    "FROM payments WHERE status='succeeded' "
                    "GROUP BY date ORDER BY date DESC LIMIT 30"
                ),
                "params": []
            }),
            "transform_config": json.dumps([]),
            "destination_type": "json_file",
            "destination_config": json.dumps({}),
            "schedule": "daily",
        },
        {
            "name": "crm_contacts_export",
            "description": "Weekly export of CRM contacts added in the last 7 days",
            "source_type": "sqlite_query",
            "source_config": json.dumps({
                "query": (
                    "SELECT * FROM contacts "
                    "WHERE created_at >= (strftime('%s','now') - 604800) "
                    "ORDER BY created_at DESC"
                ),
                "params": []
            }),
            "transform_config": json.dumps([]),
            "destination_type": "csv_file",
            "destination_config": json.dumps({}),
            "schedule": "weekly",
        },
    ]

    for p in defaults:
        conn.execute(
            """INSERT OR IGNORE INTO pipelines
               (name, description, source_type, source_config,
                transform_config, destination_type, destination_config,
                schedule, enabled, created_at)
               VALUES (?,?,?,?,?,?,?,?,1,?)""",
            (
                p["name"], p["description"], p["source_type"],
                p["source_config"], p["transform_config"],
                p["destination_type"], p["destination_config"],
                p["schedule"], now
            )
        )
    conn.commit()
    conn.close()
    log_info("Seeded default pipelines")

# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------

def _extract_sqlite_query(config: dict) -> list:
    query  = config.get("query", "")
    params = config.get("params") or []
    if not query:
        raise ValueError("sqlite_query source requires 'query'")
    conn = get_db()
    try:
        cur = conn.execute(query, params)
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()
    return rows


def _extract_http_json(config: dict) -> list:
    url     = config.get("url", "")
    headers = config.get("headers") or {}
    jq_path = config.get("jq_path", "")
    if not url:
        raise ValueError("http_json source requires 'url'")

    req = urllib.request.Request(url)
    for k, v in headers.items():
        req.add_header(k, v)

    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
        encoding = resp.headers.get_content_charset("utf-8")
        data = json.loads(raw.decode(encoding))

    # navigate dot-notation path
    if jq_path:
        for part in jq_path.split("."):
            if part:
                if isinstance(data, dict):
                    data = data.get(part, [])
                else:
                    raise ValueError(f"Cannot navigate into non-dict at path segment '{part}'")

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"http_json: resolved value is not a list or dict (got {type(data).__name__})")


def _extract_csv_upload(config: dict) -> list:
    csv_b64   = config.get("csv_base64", "")
    delimiter = config.get("delimiter", ",")
    if not csv_b64:
        raise ValueError("csv_upload source requires 'csv_base64'")

    raw = base64.b64decode(csv_b64).decode("utf-8")
    reader = csv.DictReader(io.StringIO(raw), delimiter=delimiter)
    return [dict(row) for row in reader]


def _extract_supabase_table(config: dict) -> list:
    table = config.get("table", "")
    if not table:
        raise ValueError("supabase_table source requires 'table'")
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set for supabase_table source")

    url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/{urllib.parse.quote(table)}?select=*&limit=1000"
    req = urllib.request.Request(url)
    req.add_header("apikey", SUPABASE_SERVICE_KEY)
    req.add_header("Authorization", f"Bearer {SUPABASE_SERVICE_KEY}")
    req.add_header("Accept", "application/json")

    with urllib.request.urlopen(req, timeout=30) as resp:
        raw  = resp.read()
        data = json.loads(raw.decode("utf-8"))

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "message" in data:
        raise ValueError(f"Supabase error: {data['message']}")
    return [data] if isinstance(data, dict) else []


def extract(source_type: str, source_config: dict) -> list:
    dispatch = {
        "sqlite_query":   _extract_sqlite_query,
        "http_json":      _extract_http_json,
        "csv_upload":     _extract_csv_upload,
        "supabase_table": _extract_supabase_table,
    }
    fn = dispatch.get(source_type)
    if fn is None:
        raise ValueError(f"Unknown source_type: {source_type!r}")
    return fn(source_config)

# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

def _coerce(value, ref) -> object:
    """Try to coerce value to the same type as ref for numeric comparisons."""
    try:
        if isinstance(ref, float):
            return float(value)
        if isinstance(ref, int):
            return int(value)
    except (TypeError, ValueError):
        pass
    return value


def _apply_filter(rows: list, t: dict) -> list:
    field = t.get("field", "")
    op    = t.get("op", "eq")
    value = t.get("value")
    out   = []
    for row in rows:
        rv = row.get(field)
        cv = _coerce(rv, value) if not isinstance(value, str) else rv
        try:
            if op == "eq"       and cv == value:        out.append(row)
            elif op == "ne"     and cv != value:        out.append(row)
            elif op == "gt"     and float(cv or 0) > float(value):  out.append(row)
            elif op == "lt"     and float(cv or 0) < float(value):  out.append(row)
            elif op == "contains" and value in str(rv): out.append(row)
        except (TypeError, ValueError):
            pass
    return out


def _apply_map(rows: list, t: dict) -> list:
    field     = t.get("field", "")
    new_field = t.get("new_field", field)
    template  = t.get("template", "{value}")
    out = []
    for row in rows:
        r = dict(row)
        original = r.get(field, "")
        r[new_field] = template.replace("{value}", str(original))
        out.append(r)
    return out


def _apply_aggregate(rows: list, t: dict) -> list:
    group_by   = t.get("group_by") or []
    aggregates = t.get("aggregates") or []

    groups: dict = {}
    for row in rows:
        key = tuple(str(row.get(f, "")) for f in group_by)
        if key not in groups:
            groups[key] = []
        groups[key].append(row)

    result = []
    for key, group_rows in groups.items():
        rec: dict = {}
        for i, f in enumerate(group_by):
            rec[f] = key[i]
        for agg in aggregates:
            afield  = agg.get("field", "")
            func    = agg.get("func", "count").lower()
            out_key = agg.get("output", f"{func}_{afield}")
            values  = []
            for r in group_rows:
                try:
                    values.append(float(r.get(afield, 0) or 0))
                except (TypeError, ValueError):
                    pass
            if func == "count":
                rec[out_key] = len(group_rows)
            elif func == "sum":
                rec[out_key] = sum(values)
            elif func == "avg":
                rec[out_key] = sum(values) / len(values) if values else 0
            elif func == "min":
                rec[out_key] = min(values) if values else None
            elif func == "max":
                rec[out_key] = max(values) if values else None
            else:
                rec[out_key] = len(group_rows)
        result.append(rec)
    return result


def _apply_deduplicate(rows: list, t: dict) -> list:
    key_fields = t.get("key_fields") or []
    seen: set = set()
    out = []
    for row in rows:
        key = tuple(str(row.get(f, "")) for f in key_fields)
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def _apply_add_timestamp(rows: list, t: dict) -> list:
    field = t.get("field", "processed_at")
    ts    = time.time()
    return [{**row, field: ts} for row in rows]


def transform(rows: list, transform_config: list) -> list:
    current = list(rows)
    dispatch = {
        "filter":        _apply_filter,
        "map":           _apply_map,
        "aggregate":     _apply_aggregate,
        "deduplicate":   _apply_deduplicate,
        "add_timestamp": _apply_add_timestamp,
    }
    for step in transform_config:
        stype = step.get("type", "")
        fn = dispatch.get(stype)
        if fn is None:
            log_warn(f"Unknown transform type '{stype}', skipping")
            continue
        current = fn(current, step)
    return current

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def _load_sqlite_insert(rows: list, config: dict, pipeline_name: str) -> tuple:
    """Returns (rows_loaded, rows_failed)."""
    table            = config.get("table", pipeline_name.replace("-", "_"))
    create_if_missing = config.get("create_if_missing", True)
    upsert_key       = config.get("upsert_key", "")

    if not rows:
        return 0, 0

    conn = get_db()
    try:
        # Optionally create table from first-row keys
        if create_if_missing:
            cols_ddl = ", ".join(f'"{c}" TEXT' for c in rows[0].keys())
            conn.execute(
                f'CREATE TABLE IF NOT EXISTS "{table}" (id INTEGER PRIMARY KEY, {cols_ddl})'
            )
            conn.commit()

        loaded = failed = 0
        for row in rows:
            cols   = list(row.keys())
            vals   = [str(row[c]) if row[c] is not None else None for c in cols]
            ph     = ", ".join("?" * len(cols))
            col_str = ", ".join(f'"{c}"' for c in cols)

            if upsert_key and upsert_key in cols:
                sql = (
                    f'INSERT INTO "{table}" ({col_str}) VALUES ({ph}) '
                    f'ON CONFLICT("{upsert_key}") DO UPDATE SET '
                    + ", ".join(f'"{c}"=excluded."{c}"' for c in cols if c != upsert_key)
                )
            else:
                sql = f'INSERT INTO "{table}" ({col_str}) VALUES ({ph})'

            try:
                conn.execute(sql, vals)
                loaded += 1
            except sqlite3.Error as exc:
                log_warn(f"sqlite_insert row error: {exc}")
                failed += 1

        conn.commit()
        return loaded, failed
    finally:
        conn.close()


def _load_supabase_upsert(rows: list, config: dict) -> tuple:
    table = config.get("table", "")
    if not table:
        raise ValueError("supabase_upsert destination requires 'table'")
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")

    url  = f"{SUPABASE_URL.rstrip('/')}/rest/v1/{urllib.parse.quote(table)}"
    body = json.dumps(rows).encode("utf-8")
    req  = urllib.request.Request(url, data=body, method="POST")
    req.add_header("apikey", SUPABASE_SERVICE_KEY)
    req.add_header("Authorization", f"Bearer {SUPABASE_SERVICE_KEY}")
    req.add_header("Content-Type", "application/json")
    req.add_header("Prefer", "resolution=merge-duplicates")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            _ = resp.read()
        return len(rows), 0
    except urllib.error.HTTPError as exc:
        body_err = exc.read().decode("utf-8", errors="replace")
        raise ValueError(f"Supabase upsert failed {exc.code}: {body_err}")


def _load_json_file(rows: list, config: dict, pipeline_name: str) -> tuple:
    ts       = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    filename = f"{pipeline_name}_{ts}.json"
    out_path = EXPORTS_DIR / filename
    out_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    log_info(f"json_file export: {out_path} ({len(rows)} rows)")
    return len(rows), 0


def _load_csv_file(rows: list, config: dict, pipeline_name: str) -> tuple:
    if not rows:
        return 0, 0
    ts       = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    filename = f"{pipeline_name}_{ts}.csv"
    out_path = EXPORTS_DIR / filename
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    out_path.write_text(buf.getvalue(), encoding="utf-8")
    log_info(f"csv_file export: {out_path} ({len(rows)} rows)")
    return len(rows), 0


def load(rows: list, destination_type: str, destination_config: dict,
         pipeline_name: str) -> tuple:
    """Returns (rows_loaded, rows_failed)."""
    if destination_type == "sqlite_insert":
        return _load_sqlite_insert(rows, destination_config, pipeline_name)
    if destination_type == "supabase_upsert":
        return _load_supabase_upsert(rows, destination_config)
    if destination_type == "json_file":
        return _load_json_file(rows, destination_config, pipeline_name)
    if destination_type == "csv_file":
        return _load_csv_file(rows, destination_config, pipeline_name)
    raise ValueError(f"Unknown destination_type: {destination_type!r}")

# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

_run_lock = threading.Lock()


def run_pipeline(pipeline_name: str) -> dict:
    """Execute a named pipeline. Returns run summary dict."""
    started = time.time()
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM pipelines WHERE name=?", (pipeline_name,)
    ).fetchone()
    conn.close()

    if row is None:
        return {"error": f"Pipeline '{pipeline_name}' not found"}

    pipeline = dict(row)
    run_id   = None

    # Create run record
    conn = get_db()
    cur = conn.execute(
        """INSERT INTO pipeline_runs
           (pipeline_id, pipeline_name, status, started_at)
           VALUES (?,?,?,?)""",
        (pipeline["id"], pipeline_name, "running", started)
    )
    run_id = cur.lastrowid
    conn.commit()
    conn.close()

    log_info(f"Pipeline '{pipeline_name}' run #{run_id} started")

    rows_extracted = rows_transformed = rows_loaded = rows_failed = 0
    error_msg = None
    status    = "success"

    try:
        source_cfg    = json.loads(pipeline.get("source_config", "{}") or "{}")
        transform_cfg = json.loads(pipeline.get("transform_config", "[]") or "[]")
        dest_cfg      = json.loads(pipeline.get("destination_config", "{}") or "{}")

        # Extract
        extracted = extract(pipeline["source_type"], source_cfg)
        rows_extracted = len(extracted)

        # Transform
        transformed = transform(extracted, transform_cfg)
        rows_transformed = len(transformed)

        # Load
        rows_loaded, rows_failed = load(
            transformed, pipeline["destination_type"], dest_cfg, pipeline_name
        )

    except Exception as exc:
        error_msg = str(exc)
        status    = "failed"
        log_error(f"Pipeline '{pipeline_name}' run #{run_id} failed: {exc}")

    finished = time.time()
    duration = round((finished - started) * 1000, 2)

    conn = get_db()
    conn.execute(
        """UPDATE pipeline_runs SET status=?, rows_extracted=?,
           rows_transformed=?, rows_loaded=?, rows_failed=?,
           error=?, finished_at=?, duration_ms=?
           WHERE id=?""",
        (status, rows_extracted, rows_transformed, rows_loaded,
         rows_failed, error_msg, finished, duration, run_id)
    )
    conn.execute(
        """UPDATE pipelines SET last_run_at=?, run_count=run_count+1,
           next_run_at=? WHERE name=?""",
        (finished, _next_run_time(pipeline.get("schedule", "")), pipeline_name)
    )
    conn.commit()
    conn.close()

    summary = {
        "run_id": run_id,
        "pipeline": pipeline_name,
        "status": status,
        "rows_extracted": rows_extracted,
        "rows_transformed": rows_transformed,
        "rows_loaded": rows_loaded,
        "rows_failed": rows_failed,
        "duration_ms": duration,
        "error": error_msg,
    }
    log_info(f"Pipeline '{pipeline_name}' run #{run_id} {status} in {duration}ms")
    return summary


def _next_run_time(schedule: str) -> float:
    """Return unix timestamp for next scheduled run."""
    now = time.time()
    if not schedule:
        return 0
    s = schedule.lower().strip()
    if s == "hourly":
        return now + 3600
    if s == "daily":
        return now + 86400
    if s == "weekly":
        return now + 604800
    if s == "monthly":
        return now + 2592000
    # cron-like "every Xs": e.g. "every 300s", "every 10m"
    m = re.match(r"every\s+(\d+)([smhd]?)", s)
    if m:
        n = int(m.group(1))
        unit = m.group(2) or "s"
        multiplier = {"s": 1, "m": 60, "h": 3600, "d": 86400}.get(unit, 1)
        return now + n * multiplier
    return 0

# ---------------------------------------------------------------------------
# Background scheduler
# ---------------------------------------------------------------------------

def _scheduler_loop() -> None:
    log_info("Background scheduler started (interval=300s)")
    while True:
        time.sleep(300)
        try:
            now  = time.time()
            conn = get_db()
            due  = conn.execute(
                """SELECT name FROM pipelines
                   WHERE enabled=1 AND next_run_at > 0 AND next_run_at <= ?""",
                (now,)
            ).fetchall()
            conn.close()
            for r in due:
                pname = r["name"]
                log_info(f"Scheduler triggering pipeline: {pname}")
                threading.Thread(
                    target=run_pipeline, args=(pname,), daemon=True
                ).start()
        except Exception as exc:
            log_error(f"Scheduler error: {exc}")


def start_scheduler() -> None:
    t = threading.Thread(target=_scheduler_loop, daemon=True)
    t.start()

# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def _check_admin(handler: "PipelineHandler") -> bool:
    secret = handler.headers.get("X-Admin-Secret", "")
    if not ADMIN_SECRET or hmac.compare_digest(secret, ADMIN_SECRET):
        return True
    return False


def _json_response(handler: "PipelineHandler", code: int, data: dict) -> None:
    body = json.dumps(data, default=str).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_body(handler: "PipelineHandler") -> dict:
    length = int(handler.headers.get("Content-Length", "0"))
    if length == 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}

# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class PipelineHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):  # suppress default access log
        pass

    # -------------------------
    # Route dispatcher
    # -------------------------

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path   = parsed.path.rstrip("/")
        qs     = dict(urllib.parse.parse_qsl(parsed.query))

        if path == "/health":
            self._get_health()
        elif path == "/pipelines":
            self._get_pipelines()
        elif path.startswith("/pipelines/"):
            name = path[len("/pipelines/"):]
            self._get_pipeline(name)
        elif path == "/runs":
            self._get_runs(qs)
        elif path == "/sources":
            self._get_sources()
        elif path == "/exports":
            self._get_exports()
        else:
            _json_response(self, 404, {"error": "Not found"})

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path   = parsed.path.rstrip("/")

        if path == "/pipelines":
            self._post_pipeline()
        elif re.match(r"^/pipelines/[^/]+/run$", path):
            name = path.split("/")[2]
            self._post_run_pipeline(name)
        elif re.match(r"^/pipelines/[^/]+/enable$", path):
            name = path.split("/")[2]
            self._post_toggle_pipeline(name)
        elif path == "/sources":
            self._post_source()
        elif re.match(r"^/sources/[^/]+/test$", path):
            name = path.split("/")[2]
            self._post_test_source(name)
        elif path == "/export":
            self._post_export()
        else:
            _json_response(self, 404, {"error": "Not found"})

    # -------------------------
    # GET handlers
    # -------------------------

    def _get_health(self):
        conn  = get_db()
        total = conn.execute("SELECT COUNT(*) FROM pipelines").fetchone()[0]
        last  = conn.execute(
            "SELECT pipeline_name, status, finished_at FROM pipeline_runs "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        data = {
            "status":        "ok",
            "service":       "fm-data-pipeline",
            "port":          PORT,
            "uptime_s":      round(time.time() - START_TIME, 1),
            "pipeline_count": total,
            "last_run": dict(last) if last else None,
        }
        _json_response(self, 200, data)

    def _get_pipelines(self):
        conn = get_db()
        rows = conn.execute(
            """SELECT p.*,
               (SELECT status FROM pipeline_runs r
                WHERE r.pipeline_name=p.name ORDER BY r.id DESC LIMIT 1) AS last_status
               FROM pipelines p ORDER BY p.name"""
        ).fetchall()
        conn.close()
        _json_response(self, 200, {"pipelines": [dict(r) for r in rows]})

    def _get_pipeline(self, name: str):
        conn = get_db()
        row  = conn.execute("SELECT * FROM pipelines WHERE name=?", (name,)).fetchone()
        if row is None:
            conn.close()
            _json_response(self, 404, {"error": f"Pipeline '{name}' not found"})
            return
        recent = conn.execute(
            "SELECT * FROM pipeline_runs WHERE pipeline_name=? ORDER BY id DESC LIMIT 5",
            (name,)
        ).fetchall()
        conn.close()
        _json_response(self, 200, {
            "pipeline": dict(row),
            "recent_runs": [dict(r) for r in recent],
        })

    def _get_runs(self, qs: dict):
        filters = []
        params  = []
        if qs.get("pipeline_name"):
            filters.append("pipeline_name=?")
            params.append(qs["pipeline_name"])
        if qs.get("status"):
            filters.append("status=?")
            params.append(qs["status"])
        limit = min(int(qs.get("limit", "50")), 500)
        where = ("WHERE " + " AND ".join(filters)) if filters else ""
        conn  = get_db()
        rows  = conn.execute(
            f"SELECT * FROM pipeline_runs {where} ORDER BY id DESC LIMIT {limit}",
            params
        ).fetchall()
        conn.close()
        _json_response(self, 200, {"runs": [dict(r) for r in rows]})

    def _get_sources(self):
        conn = get_db()
        rows = conn.execute("SELECT * FROM data_sources ORDER BY name").fetchall()
        conn.close()
        _json_response(self, 200, {"sources": [dict(r) for r in rows]})

    def _get_exports(self):
        conn = get_db()
        rows = conn.execute(
            "SELECT * FROM exports ORDER BY id DESC LIMIT 100"
        ).fetchall()
        conn.close()
        _json_response(self, 200, {"exports": [dict(r) for r in rows]})

    # -------------------------
    # POST handlers
    # -------------------------

    def _post_pipeline(self):
        if not _check_admin(self):
            _json_response(self, 403, {"error": "Forbidden"})
            return
        body = _read_body(self)
        name = (body.get("name") or "").strip()
        if not name:
            _json_response(self, 400, {"error": "'name' is required"})
            return
        source_type  = body.get("source_type", "")
        dest_type    = body.get("destination_type", "")
        if not source_type or not dest_type:
            _json_response(self, 400, {"error": "'source_type' and 'destination_type' are required"})
            return

        now = time.time()
        schedule = body.get("schedule", "")
        try:
            conn = get_db()
            conn.execute(
                """INSERT INTO pipelines
                   (name, description, source_type, source_config,
                    transform_config, destination_type, destination_config,
                    schedule, enabled, next_run_at, created_at)
                   VALUES (?,?,?,?,?,?,?,?,1,?,?)""",
                (
                    name,
                    body.get("description", ""),
                    source_type,
                    json.dumps(body.get("source_config", {})),
                    json.dumps(body.get("transform_config", [])),
                    dest_type,
                    json.dumps(body.get("destination_config", {})),
                    schedule,
                    _next_run_time(schedule),
                    now,
                )
            )
            conn.commit()
            conn.close()
            _json_response(self, 201, {"ok": True, "name": name})
        except sqlite3.IntegrityError:
            _json_response(self, 409, {"error": f"Pipeline '{name}' already exists"})

    def _post_run_pipeline(self, name: str):
        if not _check_admin(self):
            _json_response(self, 403, {"error": "Forbidden"})
            return
        # Check pipeline exists
        conn = get_db()
        exists = conn.execute(
            "SELECT 1 FROM pipelines WHERE name=?", (name,)
        ).fetchone()
        conn.close()
        if not exists:
            _json_response(self, 404, {"error": f"Pipeline '{name}' not found"})
            return

        # Run async
        result_holder: dict = {}
        done_event = threading.Event()

        def _run():
            result_holder.update(run_pipeline(name))
            done_event.set()

        threading.Thread(target=_run, daemon=True).start()
        done_event.wait(timeout=120)  # wait up to 2 min
        if not result_holder:
            _json_response(self, 202, {"ok": True, "message": "Pipeline still running (timeout)"})
        else:
            code = 200 if result_holder.get("status") == "success" else 500
            _json_response(self, code, result_holder)

    def _post_toggle_pipeline(self, name: str):
        if not _check_admin(self):
            _json_response(self, 403, {"error": "Forbidden"})
            return
        conn = get_db()
        row  = conn.execute(
            "SELECT id, enabled FROM pipelines WHERE name=?", (name,)
        ).fetchone()
        if row is None:
            conn.close()
            _json_response(self, 404, {"error": f"Pipeline '{name}' not found"})
            return
        new_enabled = 0 if row["enabled"] else 1
        conn.execute(
            "UPDATE pipelines SET enabled=? WHERE name=?", (new_enabled, name)
        )
        conn.commit()
        conn.close()
        _json_response(self, 200, {"ok": True, "name": name, "enabled": bool(new_enabled)})

    def _post_source(self):
        if not _check_admin(self):
            _json_response(self, 403, {"error": "Forbidden"})
            return
        body = _read_body(self)
        name = (body.get("name") or "").strip()
        stype = body.get("source_type", "")
        if not name or not stype:
            _json_response(self, 400, {"error": "'name' and 'source_type' are required"})
            return
        now = time.time()
        try:
            conn = get_db()
            conn.execute(
                """INSERT INTO data_sources (name, source_type, config, status, created_at)
                   VALUES (?,?,?,?,?)""",
                (name, stype, json.dumps(body.get("config", {})), "untested", now)
            )
            conn.commit()
            conn.close()
            _json_response(self, 201, {"ok": True, "name": name})
        except sqlite3.IntegrityError:
            _json_response(self, 409, {"error": f"Source '{name}' already exists"})

    def _post_test_source(self, name: str):
        if not _check_admin(self):
            _json_response(self, 403, {"error": "Forbidden"})
            return
        conn = get_db()
        row  = conn.execute(
            "SELECT * FROM data_sources WHERE name=?", (name,)
        ).fetchone()
        conn.close()
        if row is None:
            _json_response(self, 404, {"error": f"Source '{name}' not found"})
            return

        cfg    = json.loads(row["config"] or "{}")
        stype  = row["source_type"]
        status = "ok"
        msg    = None
        now    = time.time()

        try:
            rows = extract(stype, cfg)
            msg  = f"Connection OK — {len(rows)} rows returned"
            log_info(f"Source '{name}' test passed: {msg}")
        except Exception as exc:
            status = "error"
            msg    = str(exc)
            log_warn(f"Source '{name}' test failed: {exc}")

        conn = get_db()
        conn.execute(
            "UPDATE data_sources SET last_tested=?, status=? WHERE name=?",
            (now, status, name)
        )
        conn.commit()
        conn.close()
        _json_response(self, 200, {"name": name, "status": status, "message": msg})

    def _post_export(self):
        if not _check_admin(self):
            _json_response(self, 403, {"error": "Forbidden"})
            return
        body   = _read_body(self)
        name   = (body.get("name") or "adhoc_export").strip()
        query  = (body.get("query") or "").strip()
        fmt    = (body.get("format") or "json").lower()

        if not query:
            _json_response(self, 400, {"error": "'query' is required"})
            return
        if not re.match(r"^SELECT\b", query, re.IGNORECASE):
            _json_response(self, 400, {"error": "Only SELECT queries are allowed"})
            return
        if fmt not in ("json", "csv"):
            _json_response(self, 400, {"error": "'format' must be 'json' or 'csv'"})
            return

        try:
            rows = _extract_sqlite_query({"query": query, "params": []})
        except sqlite3.Error as exc:
            _json_response(self, 400, {"error": f"Query error: {exc}"})
            return

        ts       = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        filename = f"{name}_{ts}.{fmt}"
        out_path = EXPORTS_DIR / filename
        out_text = ""

        if fmt == "json":
            out_text = json.dumps(rows, indent=2, default=str)
            out_path.write_text(out_text, encoding="utf-8")
        else:
            if rows:
                buf = io.StringIO()
                writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()),
                                        extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
                out_text = buf.getvalue()
            out_path.write_text(out_text, encoding="utf-8")

        now = time.time()
        conn = get_db()
        conn.execute(
            """INSERT INTO exports (name, query, format, output, row_count, created_at)
               VALUES (?,?,?,?,?,?)""",
            (name, query, fmt, str(out_path), len(rows), now)
        )
        conn.commit()
        conn.close()

        log_info(f"Ad-hoc export '{name}': {len(rows)} rows → {out_path}")
        _json_response(self, 200, {
            "ok":        True,
            "name":      name,
            "format":    fmt,
            "row_count": len(rows),
            "output":    str(out_path),
        })

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    init_db()
    _seed_pipelines()

    # Set next_run_at for seeded pipelines that have schedule but no next_run_at
    conn = get_db()
    pipelines = conn.execute(
        "SELECT name, schedule FROM pipelines WHERE next_run_at IS NULL OR next_run_at=0"
    ).fetchall()
    for p in pipelines:
        nrt = _next_run_time(p["schedule"] or "")
        if nrt:
            conn.execute(
                "UPDATE pipelines SET next_run_at=? WHERE name=?",
                (nrt, p["name"])
            )
    conn.commit()
    conn.close()

    start_scheduler()

    server = HTTPServer(("0.0.0.0", PORT), PipelineHandler)
    log_info(f"Data Pipeline & ETL Engine listening on port {PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log_info("Shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
