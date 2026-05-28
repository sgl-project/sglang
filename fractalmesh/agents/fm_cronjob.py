#!/usr/bin/env python3
"""
fm_cronjob.py — CronJob Manager Agent (Port 7831)
Scheduled intent dispatcher for FractalMesh OMEGA Titan.
Fires MCP intents on cron schedules; full run history in SQLite WAL.
Credentials sourced from ~/.secrets/fractal.env at runtime — never hardcoded.
Samuel James Hiotis | ABN 56 628 117 363
"""
import hashlib
import hmac
import json
import logging
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional

# ── vault ─────────────────────────────────────────────────────────────────────
_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _line in _vault.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

# ── config ────────────────────────────────────────────────────────────────────
PORT       = int(os.getenv("CRONJOB_PORT", "7831"))
MCP_PORT   = int(os.getenv("MCP_PORT", "7785"))
MCP_SECRET = os.getenv("MCP_SECRET", "fm_mcp_internal").encode()
ROOT       = Path(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")))
DB         = ROOT / "database" / "sovereign.db"
LOG        = ROOT / "logs" / "fm_cronjob.log"

ROOT.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)
DB.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [CRONJOB] %(message)s",
    handlers=[
        logging.FileHandler(LOG),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("fm_cronjob")

# ── scheduler state ───────────────────────────────────────────────────────────
_scheduler_running = False

# ── default jobs ─────────────────────────────────────────────────────────────
_DEFAULT_JOBS = [
    {
        "name": "hourly_coingecko",
        "intent": "coingecko_price",
        "kwargs": {"coins": ["bitcoin", "ethereum", "solana"]},
        "schedule": "0 * * * *",
    },
    {
        "name": "daily_lead_scan",
        "intent": "leadgen",
        "kwargs": {"op": "sequence_execute_due"},
        "schedule": "0 8 * * *",
    },
    {
        "name": "daily_osint_sweep",
        "intent": "osint_scan",
        "kwargs": {"scan_type": "domain", "target": "fractalmesh.net", "depth": "standard"},
        "schedule": "0 2 * * *",
    },
    {
        "name": "daily_nft_pricing",
        "intent": "nft_mint",
        "kwargs": {"op": "pricing_update", "strategy": "floor_plus_10pct"},
        "schedule": "30 6 * * *",
    },
    {
        "name": "weekly_bugcrowd",
        "intent": "bugcrowd_op",
        "kwargs": {"op": "submissions"},
        "schedule": "0 9 * * 1",
    },
    {
        "name": "hourly_mesh_status",
        "intent": "mesh_status",
        "kwargs": {},
        "schedule": "*/30 * * * *",
    },
    {
        "name": "daily_devto_publish",
        "intent": "devto_publish",
        "kwargs": {"auto": True},
        "schedule": "0 10 * * 2",
    },
    {
        "name": "daily_revenue_sync",
        "intent": "gumroad",
        "kwargs": {"op": "sync_sales"},
        "schedule": "0 0 * * *",
    },
]

# ── database ──────────────────────────────────────────────────────────────────
def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cron_jobs (
                id          INTEGER PRIMARY KEY,
                name        TEXT UNIQUE,
                intent      TEXT,
                kwargs      TEXT,
                schedule    TEXT,
                enabled     INTEGER DEFAULT 1,
                last_run    REAL,
                next_run    REAL,
                run_count   INTEGER DEFAULT 0,
                error_count INTEGER DEFAULT 0,
                last_error  TEXT,
                created_at  REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cron_runs (
                id          INTEGER PRIMARY KEY,
                job_id      INTEGER,
                started_at  REAL,
                finished_at REAL,
                status      TEXT,
                result      TEXT,
                error       TEXT
            )
        """)
        conn.commit()


# ── cron helpers ──────────────────────────────────────────────────────────────
def _match_field(field: str, value: int) -> bool:
    """Return True if cron field matches the given integer value."""
    if field == "*":
        return True
    if field.startswith("*/"):
        try:
            n = int(field[2:])
            return (value % n) == 0
        except ValueError:
            return False
    if "," in field:
        try:
            return value in [int(x) for x in field.split(",")]
        except ValueError:
            return False
    if "-" in field:
        parts = field.split("-", 1)
        try:
            lo, hi = int(parts[0]), int(parts[1])
            return lo <= value <= hi
        except ValueError:
            return False
    try:
        return value == int(field)
    except ValueError:
        return False


def _parse_cron(schedule_str: str, now_dt: datetime) -> bool:
    """Return True if the cron expression matches now_dt (minute precision)."""
    fields = schedule_str.strip().split()
    if len(fields) != 5:
        return False
    f_min, f_hour, f_dom, f_month, f_dow = fields
    return (
        _match_field(f_min,   now_dt.minute)
        and _match_field(f_hour,  now_dt.hour)
        and _match_field(f_dom,   now_dt.day)
        and _match_field(f_month, now_dt.month)
        and _match_field(f_dow,   now_dt.weekday() if _is_sun_zero(f_dow) else now_dt.isoweekday() % 7)
    )


def _is_sun_zero(dow_field: str) -> bool:
    """Detect if the day-of-week field uses 0=Sunday convention (standard cron)."""
    # We use isoweekday(): Monday=1 … Sunday=7, then % 7 → Sunday=0 … Saturday=6
    return True


def _next_run(schedule_str: str, from_dt: datetime) -> float:
    """Scan forward minute-by-minute (up to 1 year) to find next fire time.
    Returns Unix timestamp of next match."""
    # Advance to next whole minute
    import math
    base = from_dt.replace(second=0, microsecond=0)
    # Start 1 minute ahead so we don't re-fire the same minute
    check = datetime(
        base.year, base.month, base.day,
        base.hour, base.minute, 0,
        tzinfo=base.tzinfo
    )
    # Increment by 1 minute
    from datetime import timedelta
    delta = timedelta(minutes=1)
    check = check + delta

    for _ in range(525600):  # max 1 year of minutes
        if _parse_cron(schedule_str, check):
            return check.timestamp()
        check += delta

    # Fallback: 1 day ahead
    return (from_dt + timedelta(days=1)).timestamp()


# ── HMAC helper ───────────────────────────────────────────────────────────────
def _hmac_sig(body_bytes: bytes) -> str:
    return hmac.new(MCP_SECRET, body_bytes, hashlib.sha256).hexdigest()


# ── job firing ────────────────────────────────────────────────────────────────
def _fire_job(job: dict) -> None:
    job_id    = job["id"]
    intent    = job["intent"]
    try:
        kwargs = json.loads(job["kwargs"] or "{}")
    except Exception:
        kwargs = {}

    started_at = time.time()
    run_id     = None
    status     = "error"
    result_str = None
    error_str  = None

    # Insert pending run record
    with _db() as conn:
        cur = conn.execute(
            "INSERT INTO cron_runs (job_id, started_at, status) VALUES (?,?,?)",
            (job_id, started_at, "running"),
        )
        run_id = cur.lastrowid
        conn.commit()

    try:
        payload = json.dumps({"intent": intent, "kwargs": kwargs}).encode()
        sig     = _hmac_sig(payload)
        req     = urllib.request.Request(
            f"http://127.0.0.1:{MCP_PORT}/",
            data=payload,
            headers={
                "Content-Type":    "application/json",
                "X-MCP-Signature": sig,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result_str = resp.read().decode("utf-8", errors="replace")[:4096]
            status     = "success"
    except Exception as exc:
        error_str = str(exc)[:1024]
        status    = "error"
        log.warning("Job %s (%s) error: %s", job_id, intent, error_str)

    finished_at = time.time()
    now_dt      = datetime.now()

    with _db() as conn:
        # Update run record
        conn.execute(
            "UPDATE cron_runs SET finished_at=?, status=?, result=?, error=? WHERE id=?",
            (finished_at, status, result_str, error_str, run_id),
        )
        # Update job stats
        if status == "success":
            conn.execute(
                """UPDATE cron_jobs
                   SET last_run=?, next_run=?, run_count=run_count+1, last_error=NULL
                   WHERE id=?""",
                (started_at, _next_run(job["schedule"], now_dt), job_id),
            )
        else:
            conn.execute(
                """UPDATE cron_jobs
                   SET last_run=?, next_run=?, run_count=run_count+1,
                       error_count=error_count+1, last_error=?
                   WHERE id=?""",
                (started_at, _next_run(job["schedule"], now_dt), error_str, job_id),
            )
        conn.commit()


# ── scheduler thread ──────────────────────────────────────────────────────────
def _scheduler_loop() -> None:
    global _scheduler_running
    _scheduler_running = True
    log.info("Scheduler thread started.")
    while True:
        time.sleep(30)
        now_ts = time.time()
        try:
            with _db() as conn:
                rows = conn.execute(
                    "SELECT * FROM cron_jobs WHERE enabled=1",
                ).fetchall()
            for row in rows:
                job = dict(row)
                next_run = job.get("next_run") or 0
                if next_run <= now_ts:
                    t = threading.Thread(target=_fire_job, args=(job,), daemon=True)
                    t.start()
        except Exception as exc:
            log.error("Scheduler loop error: %s", exc)


def _start_scheduler() -> None:
    t = threading.Thread(target=_scheduler_loop, daemon=True, name="cron-scheduler")
    t.start()


# ── seed defaults ─────────────────────────────────────────────────────────────
def _seed_defaults() -> dict:
    created  = 0
    existing = 0
    now_dt   = datetime.now()
    now_ts   = time.time()

    with _db() as conn:
        for job in _DEFAULT_JOBS:
            row = conn.execute(
                "SELECT id FROM cron_jobs WHERE name=?", (job["name"],)
            ).fetchone()
            if row:
                existing += 1
                continue
            next_ts = _next_run(job["schedule"], now_dt)
            conn.execute(
                """INSERT INTO cron_jobs
                   (name, intent, kwargs, schedule, enabled, next_run, created_at)
                   VALUES (?,?,?,?,1,?,?)""",
                (
                    job["name"],
                    job["intent"],
                    json.dumps(job["kwargs"]),
                    job["schedule"],
                    next_ts,
                    now_ts,
                ),
            )
            created += 1
        conn.commit()

    return {"created": created, "existing": existing}


# ── HTTP helpers ──────────────────────────────────────────────────────────────
def _json_resp(handler: BaseHTTPRequestHandler, code: int, data) -> None:
    body = json.dumps(data).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_body(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", 0))
    if length == 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw.decode())
    except Exception:
        return {}


def _job_row_to_dict(row) -> dict:
    d = dict(row)
    try:
        d["kwargs"] = json.loads(d.get("kwargs") or "{}")
    except Exception:
        pass
    return d


# ── request handler ───────────────────────────────────────────────────────────
class CronJobHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # silence default access log
        log.debug(fmt, *args)

    # ── routing ───────────────────────────────────────────────────────────────
    def do_GET(self):
        p = self.path.split("?")[0].rstrip("/")
        if p == "/health":
            return self._health()
        if p == "/jobs":
            return self._list_jobs()
        if p == "/runs/recent":
            return self._recent_runs()
        if p == "/analytics":
            return self._analytics()
        # /jobs/{id}
        if p.startswith("/jobs/"):
            rest = p[len("/jobs/"):]
            if rest.isdigit():
                return self._get_job(int(rest))
            # /jobs/{id}/history
            parts = rest.split("/")
            if len(parts) == 2 and parts[0].isdigit() and parts[1] == "history":
                return self._job_history(int(parts[0]))
        _json_resp(self, 404, {"error": "not found"})

    def do_POST(self):
        p = self.path.split("?")[0].rstrip("/")
        if p == "/jobs/create":
            return self._create_job()
        if p == "/jobs/seed_defaults":
            return self._seed_defaults_endpoint()
        if p.startswith("/jobs/"):
            rest = p[len("/jobs/"):]
            parts = rest.split("/")
            if len(parts) == 2 and parts[0].isdigit() and parts[1] == "run_now":
                return self._run_now(int(parts[0]))
        _json_resp(self, 404, {"error": "not found"})

    def do_PUT(self):
        p = self.path.split("?")[0].rstrip("/")
        if p.startswith("/jobs/"):
            rest = p[len("/jobs/"):]
            if rest.isdigit():
                return self._update_job(int(rest))
        _json_resp(self, 404, {"error": "not found"})

    def do_DELETE(self):
        p = self.path.split("?")[0].rstrip("/")
        if p.startswith("/jobs/"):
            rest = p[len("/jobs/"):]
            if rest.isdigit():
                return self._delete_job(int(rest))
        _json_resp(self, 404, {"error": "not found"})

    # ── endpoint implementations ──────────────────────────────────────────────
    def _health(self):
        _json_resp(self, 200, {
            "status":    "ok",
            "service":   "fm-cronjob",
            "port":      PORT,
            "scheduler": "running" if _scheduler_running else "stopped",
        })

    def _list_jobs(self):
        with _db() as conn:
            rows = conn.execute(
                "SELECT * FROM cron_jobs ORDER BY name"
            ).fetchall()
        _json_resp(self, 200, [_job_row_to_dict(r) for r in rows])

    def _get_job(self, job_id: int):
        with _db() as conn:
            row = conn.execute(
                "SELECT * FROM cron_jobs WHERE id=?", (job_id,)
            ).fetchone()
        if not row:
            return _json_resp(self, 404, {"error": "job not found"})
        _json_resp(self, 200, _job_row_to_dict(row))

    def _create_job(self):
        body = _read_body(self)
        name     = body.get("name", "").strip()
        intent   = body.get("intent", "").strip()
        kwargs   = body.get("kwargs", {})
        schedule = body.get("schedule", "").strip()
        enabled  = 1 if body.get("enabled", True) else 0

        if not name or not intent or not schedule:
            return _json_resp(self, 400, {"error": "name, intent, schedule required"})
        if len(schedule.split()) != 5:
            return _json_resp(self, 400, {"error": "schedule must be 5-field cron expression"})

        now_dt  = datetime.now()
        now_ts  = time.time()
        next_ts = _next_run(schedule, now_dt)

        try:
            with _db() as conn:
                cur = conn.execute(
                    """INSERT INTO cron_jobs
                       (name, intent, kwargs, schedule, enabled, next_run, created_at)
                       VALUES (?,?,?,?,?,?,?)""",
                    (name, intent, json.dumps(kwargs), schedule, enabled, next_ts, now_ts),
                )
                job_id = cur.lastrowid
                conn.commit()
        except sqlite3.IntegrityError:
            return _json_resp(self, 409, {"error": f"job '{name}' already exists"})

        _json_resp(self, 201, {
            "job_id":   job_id,
            "name":     name,
            "next_run": next_ts,
        })

    def _update_job(self, job_id: int):
        with _db() as conn:
            row = conn.execute(
                "SELECT * FROM cron_jobs WHERE id=?", (job_id,)
            ).fetchone()
        if not row:
            return _json_resp(self, 404, {"error": "job not found"})

        body     = _read_body(self)
        updates  = {}
        now_dt   = datetime.now()

        if "schedule" in body:
            sched = body["schedule"].strip()
            if len(sched.split()) != 5:
                return _json_resp(self, 400, {"error": "schedule must be 5-field cron expression"})
            updates["schedule"] = sched
            updates["next_run"] = _next_run(sched, now_dt)
        if "enabled" in body:
            updates["enabled"] = 1 if body["enabled"] else 0
        if "kwargs" in body:
            updates["kwargs"] = json.dumps(body["kwargs"])
        if "intent" in body:
            updates["intent"] = body["intent"].strip()

        if not updates:
            return _json_resp(self, 400, {"error": "no updatable fields provided"})

        set_clause = ", ".join(f"{k}=?" for k in updates)
        values     = list(updates.values()) + [job_id]

        with _db() as conn:
            conn.execute(
                f"UPDATE cron_jobs SET {set_clause} WHERE id=?", values
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM cron_jobs WHERE id=?", (job_id,)
            ).fetchone()

        _json_resp(self, 200, _job_row_to_dict(row))

    def _delete_job(self, job_id: int):
        with _db() as conn:
            row = conn.execute(
                "SELECT id FROM cron_jobs WHERE id=?", (job_id,)
            ).fetchone()
            if not row:
                return _json_resp(self, 404, {"error": "job not found"})
            conn.execute("DELETE FROM cron_runs WHERE job_id=?", (job_id,))
            conn.execute("DELETE FROM cron_jobs WHERE id=?", (job_id,))
            conn.commit()
        _json_resp(self, 200, {"status": "deleted", "job_id": job_id})

    def _run_now(self, job_id: int):
        with _db() as conn:
            row = conn.execute(
                "SELECT * FROM cron_jobs WHERE id=?", (job_id,)
            ).fetchone()
        if not row:
            return _json_resp(self, 404, {"error": "job not found"})
        job = dict(row)
        t   = threading.Thread(target=_fire_job, args=(job,), daemon=True)
        t.start()
        _json_resp(self, 202, {"status": "triggered", "job_id": job_id})

    def _job_history(self, job_id: int):
        with _db() as conn:
            row = conn.execute(
                "SELECT id FROM cron_jobs WHERE id=?", (job_id,)
            ).fetchone()
            if not row:
                return _json_resp(self, 404, {"error": "job not found"})
            runs = conn.execute(
                """SELECT * FROM cron_runs
                   WHERE job_id=?
                   ORDER BY started_at DESC
                   LIMIT 50""",
                (job_id,),
            ).fetchall()
        _json_resp(self, 200, [dict(r) for r in runs])

    def _recent_runs(self):
        with _db() as conn:
            rows = conn.execute(
                """SELECT r.*, j.name AS job_name
                   FROM cron_runs r
                   LEFT JOIN cron_jobs j ON j.id = r.job_id
                   ORDER BY r.started_at DESC
                   LIMIT 100""",
            ).fetchall()
        _json_resp(self, 200, [dict(r) for r in rows])

    def _seed_defaults_endpoint(self):
        result = _seed_defaults()
        _json_resp(self, 200, result)

    def _analytics(self):
        with _db() as conn:
            total_jobs = conn.execute(
                "SELECT COUNT(*) FROM cron_jobs"
            ).fetchone()[0]
            enabled_jobs = conn.execute(
                "SELECT COUNT(*) FROM cron_jobs WHERE enabled=1"
            ).fetchone()[0]
            disabled_jobs = total_jobs - enabled_jobs

            today_start = datetime.now().replace(
                hour=0, minute=0, second=0, microsecond=0
            ).timestamp()
            runs_today = conn.execute(
                "SELECT COUNT(*) FROM cron_runs WHERE started_at >= ?",
                (today_start,),
            ).fetchone()[0]
            total_runs = conn.execute(
                "SELECT COUNT(*) FROM cron_runs"
            ).fetchone()[0]
            success_runs = conn.execute(
                "SELECT COUNT(*) FROM cron_runs WHERE status='success'"
            ).fetchone()[0]
            success_rate = round(
                (success_runs / total_runs * 100) if total_runs > 0 else 0.0, 2
            )

            top_intent_row = conn.execute(
                """SELECT intent, COUNT(*) AS cnt
                   FROM cron_jobs
                   GROUP BY intent
                   ORDER BY cnt DESC
                   LIMIT 1""",
            ).fetchone()
            most_active_intent = top_intent_row["intent"] if top_intent_row else None

        _json_resp(self, 200, {
            "total_jobs":          total_jobs,
            "enabled_jobs":        enabled_jobs,
            "disabled_jobs":       disabled_jobs,
            "runs_today":          runs_today,
            "total_runs":          total_runs,
            "success_rate_pct":    success_rate,
            "most_active_intent":  most_active_intent,
        })


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("Initialising database at %s", DB)
    _init_db()

    log.info("Seeding default jobs …")
    result = _seed_defaults()
    log.info("Seed result: created=%d existing=%d", result["created"], result["existing"])

    log.info("Starting scheduler thread …")
    _start_scheduler()

    server = HTTPServer(("0.0.0.0", PORT), CronJobHandler)
    log.info("fm-cronjob listening on port %d", PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
