#!/usr/bin/env python3
"""
fm_scheduler_pro.py — Advanced Job Scheduler for FractalMesh OMEGA Titan (Port 7863)
Cron-like job scheduler with persistence. Supports cron expressions, interval-based,
one-shot, and recurring jobs. Dispatches HTTP requests, shell commands (sandboxed),
or MCP intents. Tracks full execution history and failures.
Credentials sourced from ~/.secrets/fractal.env at runtime — never hardcoded.
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import json
import time
import hmac
import hashlib
import sqlite3
import math
import re
import threading
import subprocess
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ── vault ──────────────────────────────────────────────────────────────────────
_ENV_FILE = Path.home() / ".secrets" / "fractal.env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

# ── config ─────────────────────────────────────────────────────────────────────
PORT         = int(os.getenv("SCHEDULER_PRO_PORT", "7863"))
MCP_PORT     = int(os.getenv("MCP_PORT", "7785"))
MCP_SECRET   = os.getenv("MCP_SECRET", "fm_mcp_internal")
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")
ROOT         = Path(os.path.expanduser("~/fmsaas"))
DB_PATH      = ROOT / "database" / "sovereign.db"
LOG_PATH     = ROOT / "logs" / "fm_scheduler_pro.log"

ROOT.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Allowed command prefixes for sandboxed shell execution
ALLOWED_CMD_PREFIXES = ("python3 ", "pm2 ", "curl ")

START_TIME = time.time()

# ── logging ────────────────────────────────────────────────────────────────────
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SCHEDULER_PRO] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("scheduler_pro")

# ── database ───────────────────────────────────────────────────────────────────
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id               INTEGER PRIMARY KEY,
                name             TEXT UNIQUE NOT NULL,
                job_type         TEXT NOT NULL,
                schedule         TEXT NOT NULL,
                enabled          INTEGER DEFAULT 1,
                target_url       TEXT,
                target_method    TEXT DEFAULT 'POST',
                payload          TEXT,
                command          TEXT,
                mcp_intent       TEXT,
                timeout_seconds  INTEGER DEFAULT 30,
                max_retries      INTEGER DEFAULT 3,
                retry_count      INTEGER DEFAULT 0,
                last_run_at      REAL,
                next_run_at      REAL,
                run_count        INTEGER DEFAULT 0,
                success_count    INTEGER DEFAULT 0,
                fail_count       INTEGER DEFAULT 0,
                created_at       REAL,
                updated_at       REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS executions (
                id           INTEGER PRIMARY KEY,
                job_id       INTEGER,
                job_name     TEXT,
                status       TEXT,
                output       TEXT,
                error        TEXT,
                started_at   REAL,
                finished_at  REAL,
                duration_ms  REAL,
                retry_number INTEGER DEFAULT 0
            )
        """)
        conn.commit()


def seed_jobs() -> None:
    """Insert pre-seeded jobs if the jobs table is empty."""
    with get_db() as conn:
        count = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        if count > 0:
            return
        now = time.time()
        seeds = [
            {
                "name": "health_sweep",
                "job_type": "mcp_intent",
                "schedule": "cron:*/5 * * * *",
                "mcp_intent": "ping_all",
                "timeout_seconds": 30,
                "max_retries": 3,
            },
            {
                "name": "daily_backup",
                "job_type": "mcp_intent",
                "schedule": "cron:0 2 * * *",
                "mcp_intent": "memory_recall",
                "timeout_seconds": 120,
                "max_retries": 3,
            },
            {
                "name": "revenue_kpi",
                "job_type": "mcp_intent",
                "schedule": "cron:*/30 * * * *",
                "mcp_intent": "kpi_record",
                "timeout_seconds": 30,
                "max_retries": 3,
            },
        ]
        for s in seeds:
            next_run = _next_cron_run(s["schedule"].split(":", 1)[1])
            conn.execute(
                """INSERT OR IGNORE INTO jobs
                   (name, job_type, schedule, mcp_intent, timeout_seconds, max_retries,
                    next_run_at, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    s["name"], s["job_type"], s["schedule"],
                    s.get("mcp_intent"), s["timeout_seconds"], s["max_retries"],
                    next_run, now, now,
                ),
            )
        conn.commit()
        log.info("Pre-seeded 3 default jobs.")


# ── cron parser ────────────────────────────────────────────────────────────────
def _parse_field(field: str, lo: int, hi: int) -> list:
    """Parse a single cron field into a sorted list of valid integers.
    Supports: *, */n, a-b, comma-separated combinations, and plain numbers.
    """
    values = set()
    for part in field.split(","):
        part = part.strip()
        if part == "*":
            values.update(range(lo, hi + 1))
        elif part.startswith("*/"):
            step = int(part[2:])
            values.update(range(lo, hi + 1, step))
        elif "-" in part and "/" in part:
            range_part, step_part = part.split("/", 1)
            a_str, b_str = range_part.split("-", 1)
            a, b, step = int(a_str), int(b_str), int(step_part)
            values.update(range(a, b + 1, step))
        elif "-" in part:
            a_str, b_str = part.split("-", 1)
            a, b = int(a_str), int(b_str)
            values.update(range(a, b + 1))
        else:
            values.add(int(part))
    return sorted(v for v in values if lo <= v <= hi)


def _next_cron_run(expr: str) -> float:
    """Compute next firing time (UTC epoch) from a 5-field cron expression.
    Fields: minute hour day_of_month month day_of_week
    Iterates minute-by-minute from now+1 minute up to 366 days.
    """
    parts = expr.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Invalid cron expression (need 5 fields): {expr!r}")

    minutes_f, hours_f, doms_f, months_f, dows_f = parts

    minutes  = _parse_field(minutes_f, 0, 59)
    hours    = _parse_field(hours_f,   0, 23)
    doms     = _parse_field(doms_f,    1, 31)
    months   = _parse_field(months_f,  1, 12)
    dows     = _parse_field(dows_f,    0,  6)  # 0=Sunday

    import time as _time
    # Walk minute by minute from now+60s, up to 366 days
    now_epoch = _time.time()
    candidate = now_epoch + 60  # start one minute from now

    # Truncate to the start of that minute
    candidate = math.floor(candidate / 60) * 60

    limit = now_epoch + 366 * 24 * 3600

    while candidate <= limit:
        t = _time.gmtime(candidate)
        # tm_wday: Monday=0 … Sunday=6 in gmtime; cron uses Sunday=0
        cron_dow = (t.tm_wday + 1) % 7  # convert: Monday=1 … Sunday=0

        if (t.tm_mon  in months and
                t.tm_mday in doms    and
                (dows_f == "*" or cron_dow in dows) and
                t.tm_hour in hours   and
                t.tm_min  in minutes):
            return float(candidate)

        candidate += 60

    raise ValueError(f"Could not compute next run for cron expression: {expr!r}")


def _compute_next_run(schedule: str) -> float:
    """Compute next run epoch from schedule string."""
    now = time.time()
    if schedule.startswith("cron:"):
        expr = schedule[5:]
        return _next_cron_run(expr)
    elif schedule.startswith("interval:"):
        secs = float(schedule[9:])
        return now + secs
    elif schedule.startswith("at:"):
        iso = schedule[3:]
        import time as _time
        # Parse ISO timestamp (YYYY-MM-DDTHH:MM:SS or with Z/offset)
        iso_clean = iso.replace("Z", "+00:00")
        try:
            # Python 3.7+
            import datetime as _dt
            dt = _dt.datetime.fromisoformat(iso_clean)
            if dt.tzinfo is None:
                # assume UTC
                import calendar
                return calendar.timegm(dt.timetuple())
            else:
                return dt.timestamp()
        except Exception:
            raise ValueError(f"Cannot parse at: timestamp: {iso!r}")
    else:
        raise ValueError(f"Unknown schedule format: {schedule!r}")


# ── HMAC signing ───────────────────────────────────────────────────────────────
def _sign_payload(body: bytes) -> str:
    return hmac.new(MCP_SECRET.encode(), body, hashlib.sha256).hexdigest()


# ── job execution ──────────────────────────────────────────────────────────────
def _execute_http(job: sqlite3.Row) -> tuple:
    """Execute an HTTP job. Returns (status, output, error)."""
    url    = job["target_url"] or ""
    method = (job["target_method"] or "POST").upper()
    raw    = job["payload"] or "{}"
    timeout = job["timeout_seconds"] or 30

    try:
        body = json.dumps(json.loads(raw)).encode("utf-8") if raw else b""
    except Exception:
        body = raw.encode("utf-8") if isinstance(raw, str) else raw

    req = urllib.request.Request(url, data=body if method in ("POST", "PUT", "PATCH") else None, method=method)
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "FractalMesh-SchedulerPro/1.0")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            output = resp.read().decode("utf-8", errors="replace")[:4096]
            return "success", output, ""
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")[:1024]
        return "failure", "", f"HTTP {e.code}: {err_body}"
    except Exception as exc:
        return "failure", "", str(exc)


def _execute_command(job: sqlite3.Row) -> tuple:
    """Execute a sandboxed shell command. Returns (status, output, error)."""
    cmd_str = (job["command"] or "").strip()
    timeout = job["timeout_seconds"] or 30

    allowed = any(cmd_str.startswith(prefix) for prefix in ALLOWED_CMD_PREFIXES)
    if not allowed:
        return "failure", "", (
            f"Command rejected: must start with one of "
            f"{ALLOWED_CMD_PREFIXES}. Got: {cmd_str!r}"
        )

    # Split safely — no shell=True
    import shlex
    try:
        cmd_list = shlex.split(cmd_str)
    except Exception as exc:
        return "failure", "", f"Command parse error: {exc}"

    try:
        result = subprocess.run(
            cmd_list,
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = (result.stdout or "")[:4096]
        error  = (result.stderr or "")[:2048]
        status = "success" if result.returncode == 0 else "failure"
        return status, output, error
    except subprocess.TimeoutExpired:
        return "failure", "", f"Command timed out after {timeout}s"
    except Exception as exc:
        return "failure", "", str(exc)


def _execute_mcp_intent(job: sqlite3.Row) -> tuple:
    """POST a HMAC-signed MCP intent. Returns (status, output, error)."""
    intent  = job["mcp_intent"] or ""
    raw     = job["payload"] or "{}"
    timeout = job["timeout_seconds"] or 30

    try:
        extra = json.loads(raw) if raw else {}
    except Exception:
        extra = {}

    body_dict = {"intent": intent, "source": "scheduler_pro", **extra}
    body_bytes = json.dumps(body_dict).encode("utf-8")
    sig = _sign_payload(body_bytes)

    url = f"http://localhost:{MCP_PORT}/"
    req = urllib.request.Request(url, data=body_bytes, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("X-HMAC-Signature", sig)
    req.add_header("User-Agent", "FractalMesh-SchedulerPro/1.0")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            output = resp.read().decode("utf-8", errors="replace")[:4096]
            return "success", output, ""
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")[:1024]
        return "failure", "", f"HTTP {e.code}: {err_body}"
    except Exception as exc:
        return "failure", "", str(exc)


def _run_job(job_id: int, retry_number: int = 0) -> None:
    """Execute a job by ID and record the result. Called in a worker thread."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not row:
        return

    job_type   = row["job_type"]
    timeout    = row["timeout_seconds"] or 30
    max_retries = row["max_retries"] or 3

    started_at = time.time()

    if job_type == "http":
        status, output, error = _execute_http(row)
    elif job_type == "command":
        status, output, error = _execute_command(row)
    elif job_type == "mcp_intent":
        status, output, error = _execute_mcp_intent(row)
    else:
        status, output, error = "failure", "", f"Unknown job_type: {job_type!r}"

    finished_at  = time.time()
    duration_ms  = (finished_at - started_at) * 1000

    with get_db() as conn:
        conn.execute(
            """INSERT INTO executions
               (job_id, job_name, status, output, error, started_at, finished_at,
                duration_ms, retry_number)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (job_id, row["name"], status, output, error,
             started_at, finished_at, duration_ms, retry_number),
        )
        if status == "success":
            conn.execute(
                "UPDATE jobs SET success_count=success_count+1, retry_count=0 WHERE id=?",
                (job_id,),
            )
        else:
            conn.execute(
                "UPDATE jobs SET fail_count=fail_count+1 WHERE id=?",
                (job_id,),
            )
        conn.commit()

    if status == "failure" and retry_number < max_retries:
        log.warning(
            "Job %s failed (attempt %d/%d), retrying in 30s",
            row["name"], retry_number + 1, max_retries,
        )
        def _retry():
            time.sleep(30)
            _run_job(job_id, retry_number + 1)
        t = threading.Thread(target=_retry, daemon=True)
        t.start()
    else:
        log.info(
            "Job %s finished: %s (%.0f ms, attempt %d)",
            row["name"], status, duration_ms, retry_number,
        )


# ── background scheduler ───────────────────────────────────────────────────────
def _scheduler_loop() -> None:
    """Main scheduler thread: fires due jobs every 10 seconds."""
    log.info("Scheduler loop started.")
    while True:
        try:
            now = time.time()
            with get_db() as conn:
                due = conn.execute(
                    "SELECT * FROM jobs WHERE enabled=1 AND next_run_at IS NOT NULL AND next_run_at <= ?",
                    (now,),
                ).fetchall()

                for job in due:
                    job_id   = job["id"]
                    schedule = job["schedule"]

                    # Compute next_run_at before spawning to avoid double-fire
                    try:
                        if schedule.startswith("at:"):
                            # One-shot: disable after firing
                            next_run = None
                            conn.execute(
                                "UPDATE jobs SET enabled=0, last_run_at=?, run_count=run_count+1, "
                                "next_run_at=NULL, updated_at=? WHERE id=?",
                                (now, now, job_id),
                            )
                        elif schedule.startswith("interval:"):
                            secs = float(schedule[9:])
                            next_run = now + secs
                            conn.execute(
                                "UPDATE jobs SET last_run_at=?, run_count=run_count+1, "
                                "next_run_at=?, updated_at=? WHERE id=?",
                                (now, next_run, now, job_id),
                            )
                        else:
                            next_run = _compute_next_run(schedule)
                            conn.execute(
                                "UPDATE jobs SET last_run_at=?, run_count=run_count+1, "
                                "next_run_at=?, updated_at=? WHERE id=?",
                                (now, next_run, now, job_id),
                            )
                    except Exception as exc:
                        log.error("Failed to compute next_run for job %s: %s", job["name"], exc)
                        conn.execute(
                            "UPDATE jobs SET enabled=0, updated_at=? WHERE id=?",
                            (now, job_id),
                        )

                    conn.commit()

                    # Spawn worker thread
                    t = threading.Thread(target=_run_job, args=(job_id,), daemon=True)
                    t.start()
                    log.info("Dispatched job: %s", job["name"])

        except Exception as exc:
            log.error("Scheduler loop error: %s", exc)

        time.sleep(10)


def _start_scheduler() -> None:
    t = threading.Thread(target=_scheduler_loop, daemon=True)
    t.start()


# ── admin auth helper ──────────────────────────────────────────────────────────
def _check_admin(handler) -> bool:
    secret = handler.headers.get("X-Admin-Secret", "")
    if not ADMIN_SECRET:
        return True  # No secret configured — open
    return hmac.compare_digest(secret, ADMIN_SECRET)


# ── JSON helpers ───────────────────────────────────────────────────────────────
def _row_to_dict(row) -> dict:
    return dict(zip(row.keys(), tuple(row)))


def _send_json(handler, code: int, data) -> None:
    body = json.dumps(data, default=str).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_body(handler) -> dict:
    length = int(handler.headers.get("Content-Length", "0"))
    if length == 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return {}


# ── request handler ────────────────────────────────────────────────────────────
class SchedulerHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):  # silence default access log
        pass

    # ── routing ────────────────────────────────────────────────────────────────
    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        query = self._parse_query()

        if path == "/health":
            self._get_health()
        elif path == "/jobs":
            self._list_jobs(query)
        elif re.fullmatch(r"/jobs/[^/]+", path):
            name = path.split("/", 2)[2]
            self._get_job(name)
        elif path == "/executions":
            self._list_executions(query)
        elif path == "/next_runs":
            self._get_next_runs()
        else:
            _send_json(self, 404, {"error": "Not found"})

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")

        if path == "/jobs":
            self._create_job()
        elif re.fullmatch(r"/jobs/[^/]+/trigger", path):
            name = path.split("/")[2]
            self._trigger_job(name)
        elif re.fullmatch(r"/jobs/[^/]+/enable", path):
            name = path.split("/")[2]
            self._toggle_job(name)
        else:
            _send_json(self, 404, {"error": "Not found"})

    def do_PUT(self):
        path = self.path.split("?")[0].rstrip("/")
        if re.fullmatch(r"/jobs/[^/]+", path):
            name = path.split("/", 2)[2]
            self._update_job(name)
        else:
            _send_json(self, 404, {"error": "Not found"})

    def do_DELETE(self):
        path = self.path.split("?")[0].rstrip("/")
        if re.fullmatch(r"/jobs/[^/]+", path):
            name = path.split("/", 2)[2]
            self._delete_job(name)
        else:
            _send_json(self, 404, {"error": "Not found"})

    # ── query string ───────────────────────────────────────────────────────────
    def _parse_query(self) -> dict:
        q = {}
        if "?" in self.path:
            qs = self.path.split("?", 1)[1]
            for part in qs.split("&"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    q[k] = v
        return q

    # ── GET /health ────────────────────────────────────────────────────────────
    def _get_health(self):
        now = time.time()
        uptime = now - START_TIME
        with get_db() as conn:
            total      = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            enabled    = conn.execute("SELECT COUNT(*) FROM jobs WHERE enabled=1").fetchone()[0]
            disabled   = total - enabled
            recent_ok  = conn.execute(
                "SELECT COUNT(*) FROM executions WHERE status='success' AND started_at > ?",
                (now - 3600,),
            ).fetchone()[0]
            recent_err = conn.execute(
                "SELECT COUNT(*) FROM executions WHERE status='failure' AND started_at > ?",
                (now - 3600,),
            ).fetchone()[0]
            total_execs = conn.execute("SELECT COUNT(*) FROM executions").fetchone()[0]
        _send_json(self, 200, {
            "service": "fm_scheduler_pro",
            "status": "ok",
            "port": PORT,
            "uptime_seconds": round(uptime, 1),
            "jobs": {"total": total, "enabled": enabled, "disabled": disabled},
            "executions": {
                "total": total_execs,
                "last_hour_success": recent_ok,
                "last_hour_failure": recent_err,
            },
        })

    # ── GET /jobs ──────────────────────────────────────────────────────────────
    def _list_jobs(self, query: dict):
        sql = "SELECT * FROM jobs WHERE 1=1"
        params = []
        if "enabled" in query:
            sql += " AND enabled=?"
            params.append(int(query["enabled"]))
        if "job_type" in query:
            sql += " AND job_type=?"
            params.append(query["job_type"])
        sql += " ORDER BY name"
        with get_db() as conn:
            rows = conn.execute(sql, params).fetchall()
        _send_json(self, 200, {"jobs": [_row_to_dict(r) for r in rows]})

    # ── GET /jobs/{name} ──────────────────────────────────────────────────────
    def _get_job(self, name: str):
        with get_db() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE name=?", (name,)).fetchone()
            if not row:
                _send_json(self, 404, {"error": f"Job not found: {name}"})
                return
            job = _row_to_dict(row)
            execs = conn.execute(
                "SELECT * FROM executions WHERE job_name=? ORDER BY started_at DESC LIMIT 10",
                (name,),
            ).fetchall()
            job["recent_executions"] = [_row_to_dict(e) for e in execs]
        _send_json(self, 200, job)

    # ── GET /executions ────────────────────────────────────────────────────────
    def _list_executions(self, query: dict):
        sql = "SELECT * FROM executions WHERE 1=1"
        params = []
        if "job_name" in query:
            sql += " AND job_name=?"
            params.append(query["job_name"])
        if "status" in query:
            sql += " AND status=?"
            params.append(query["status"])
        sql += " ORDER BY started_at DESC LIMIT ?"
        limit = int(query.get("limit", "50"))
        params.append(min(limit, 500))
        with get_db() as conn:
            rows = conn.execute(sql, params).fetchall()
        _send_json(self, 200, {"executions": [_row_to_dict(r) for r in rows]})

    # ── GET /next_runs ────────────────────────────────────────────────────────
    def _get_next_runs(self):
        with get_db() as conn:
            rows = conn.execute(
                "SELECT name, schedule, next_run_at FROM jobs WHERE enabled=1 ORDER BY next_run_at",
            ).fetchall()
        result = []
        for r in rows:
            result.append({
                "name": r["name"],
                "schedule": r["schedule"],
                "next_run_at": r["next_run_at"],
                "next_run_human": (
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(r["next_run_at"]))
                    if r["next_run_at"] else None
                ),
            })
        _send_json(self, 200, {"next_runs": result})

    # ── POST /jobs ────────────────────────────────────────────────────────────
    def _create_job(self):
        if not _check_admin(self):
            _send_json(self, 403, {"error": "Forbidden"})
            return
        data = _read_body(self)
        required = ("name", "job_type", "schedule")
        for f in required:
            if not data.get(f):
                _send_json(self, 400, {"error": f"Missing required field: {f}"})
                return

        name      = data["name"].strip()
        job_type  = data["job_type"].strip()
        schedule  = data["schedule"].strip()

        if job_type not in ("http", "command", "mcp_intent"):
            _send_json(self, 400, {"error": f"Invalid job_type: {job_type!r}"})
            return

        try:
            next_run = _compute_next_run(schedule)
        except ValueError as exc:
            _send_json(self, 400, {"error": str(exc)})
            return

        now = time.time()
        try:
            with get_db() as conn:
                conn.execute(
                    """INSERT INTO jobs
                       (name, job_type, schedule, enabled, target_url, target_method,
                        payload, command, mcp_intent, timeout_seconds, max_retries,
                        next_run_at, created_at, updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        name,
                        job_type,
                        schedule,
                        int(data.get("enabled", 1)),
                        data.get("target_url"),
                        data.get("target_method", "POST"),
                        data.get("payload"),
                        data.get("command"),
                        data.get("mcp_intent"),
                        int(data.get("timeout_seconds", 30)),
                        int(data.get("max_retries", 3)),
                        next_run,
                        now,
                        now,
                    ),
                )
                conn.commit()
                row = conn.execute("SELECT * FROM jobs WHERE name=?", (name,)).fetchone()
        except sqlite3.IntegrityError:
            _send_json(self, 409, {"error": f"Job already exists: {name}"})
            return
        except Exception as exc:
            _send_json(self, 500, {"error": str(exc)})
            return

        log.info("Created job: %s (%s)", name, job_type)
        _send_json(self, 201, _row_to_dict(row))

    # ── POST /jobs/{name}/trigger ─────────────────────────────────────────────
    def _trigger_job(self, name: str):
        if not _check_admin(self):
            _send_json(self, 403, {"error": "Forbidden"})
            return
        with get_db() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE name=?", (name,)).fetchone()
        if not row:
            _send_json(self, 404, {"error": f"Job not found: {name}"})
            return

        job_id = row["id"]
        t = threading.Thread(target=_run_job, args=(job_id,), daemon=True)
        t.start()
        log.info("Manual trigger: %s", name)
        _send_json(self, 202, {"message": f"Job {name!r} triggered.", "job_id": job_id})

    # ── POST /jobs/{name}/enable ──────────────────────────────────────────────
    def _toggle_job(self, name: str):
        if not _check_admin(self):
            _send_json(self, 403, {"error": "Forbidden"})
            return
        data = _read_body(self)
        enabled_val = data.get("enabled")
        if enabled_val is None:
            _send_json(self, 400, {"error": "Body must include 'enabled': true/false or 1/0"})
            return

        enabled = 1 if enabled_val else 0
        now = time.time()
        with get_db() as conn:
            cur = conn.execute(
                "UPDATE jobs SET enabled=?, updated_at=? WHERE name=?",
                (enabled, now, name),
            )
            if cur.rowcount == 0:
                _send_json(self, 404, {"error": f"Job not found: {name}"})
                return
            # Recompute next_run if re-enabling
            if enabled:
                row = conn.execute("SELECT schedule FROM jobs WHERE name=?", (name,)).fetchone()
                if row:
                    try:
                        next_run = _compute_next_run(row["schedule"])
                        conn.execute(
                            "UPDATE jobs SET next_run_at=? WHERE name=?",
                            (next_run, name),
                        )
                    except Exception:
                        pass
            conn.commit()
            row = conn.execute("SELECT * FROM jobs WHERE name=?", (name,)).fetchone()
        log.info("Job %s %s", name, "enabled" if enabled else "disabled")
        _send_json(self, 200, _row_to_dict(row))

    # ── PUT /jobs/{name} ──────────────────────────────────────────────────────
    def _update_job(self, name: str):
        if not _check_admin(self):
            _send_json(self, 403, {"error": "Forbidden"})
            return
        data = _read_body(self)
        if not data:
            _send_json(self, 400, {"error": "Empty body"})
            return

        # Build dynamic update
        allowed_fields = {
            "job_type", "schedule", "enabled", "target_url", "target_method",
            "payload", "command", "mcp_intent", "timeout_seconds", "max_retries",
        }
        updates = {k: v for k, v in data.items() if k in allowed_fields}
        if not updates:
            _send_json(self, 400, {"error": "No updatable fields provided"})
            return

        now = time.time()
        updates["updated_at"] = now

        # If schedule changed, recompute next_run_at
        if "schedule" in updates:
            try:
                updates["next_run_at"] = _compute_next_run(updates["schedule"])
            except ValueError as exc:
                _send_json(self, 400, {"error": str(exc)})
                return

        set_clause = ", ".join(f"{k}=?" for k in updates)
        values = list(updates.values()) + [name]

        with get_db() as conn:
            cur = conn.execute(
                f"UPDATE jobs SET {set_clause} WHERE name=?",
                values,
            )
            if cur.rowcount == 0:
                _send_json(self, 404, {"error": f"Job not found: {name}"})
                return
            conn.commit()
            row = conn.execute("SELECT * FROM jobs WHERE name=?", (name,)).fetchone()

        log.info("Updated job: %s", name)
        _send_json(self, 200, _row_to_dict(row))

    # ── DELETE /jobs/{name} ───────────────────────────────────────────────────
    def _delete_job(self, name: str):
        if not _check_admin(self):
            _send_json(self, 403, {"error": "Forbidden"})
            return
        with get_db() as conn:
            row = conn.execute("SELECT id FROM jobs WHERE name=?", (name,)).fetchone()
            if not row:
                _send_json(self, 404, {"error": f"Job not found: {name}"})
                return
            job_id = row["id"]
            conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))
            conn.execute("DELETE FROM executions WHERE job_id=?", (job_id,))
            conn.commit()
        log.info("Deleted job: %s", name)
        _send_json(self, 200, {"message": f"Job {name!r} deleted."})


# ── entry point ────────────────────────────────────────────────────────────────
def main() -> None:
    init_db()
    seed_jobs()
    _start_scheduler()

    server = HTTPServer(("0.0.0.0", PORT), SchedulerHandler)
    log.info("FractalMesh SchedulerPro listening on port %d", PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("SchedulerPro shutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
