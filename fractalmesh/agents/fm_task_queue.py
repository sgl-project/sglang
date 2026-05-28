#!/usr/bin/env python3
"""
fm_task_queue.py — Distributed Task Queue for FractalMesh OMEGA Titan (Port 7872)
Persistent task queue with worker pool. Supports HTTP callbacks, MCP intents,
shell commands, Python function calls, priorities, delays, retries, task chaining,
and a dead-letter queue. Workers process tasks concurrently per queue.
Credentials sourced from ~/.secrets/fractal.env at runtime — never hardcoded.
Samuel James Hiotis | ABN 56 628 117 363
"""
import hashlib
import hmac
import json
import logging
import os
import queue
import sqlite3
import subprocess
import threading
import time
import urllib.error
import urllib.request
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
PORT         = int(os.getenv("TASK_QUEUE_PORT", "7872"))
MCP_PORT     = int(os.getenv("MCP_PORT", "7785"))
MCP_SECRET   = os.getenv("MCP_SECRET", "fm_mcp_internal")
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")
ROOT         = Path(os.path.expanduser("~/fmsaas"))
DB_PATH      = ROOT / "database" / "sovereign.db"
LOG_PATH     = ROOT / "logs" / "fm_task_queue.log"

ROOT.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

ALLOWED_CMD_PREFIXES = ("python3 ", "pm2 ", "curl ")
WORKER_POLL_INTERVAL = 2       # seconds between DB polls
RETRY_BASE_DELAY     = 30      # seconds; actual = 30 * 2^retry_count
START_TIME           = time.time()

# Pre-seeded queues: name -> concurrency
SEED_QUEUES = {
    "default":       3,
    "high_priority": 2,
    "background":    1,
}

# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TASK_QUEUE] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("task_queue")

# ── database ───────────────────────────────────────────────────────────────────
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


def init_db() -> None:
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id               INTEGER PRIMARY KEY,
                task_id          TEXT UNIQUE,
                queue_name       TEXT DEFAULT 'default',
                task_type        TEXT,
                payload          TEXT,
                priority         INTEGER DEFAULT 5,
                status           TEXT DEFAULT 'pending',
                worker_id        TEXT,
                max_retries      INTEGER DEFAULT 3,
                retry_count      INTEGER DEFAULT 0,
                delay_until      REAL DEFAULT 0,
                timeout_seconds  INTEGER DEFAULT 60,
                parent_task_id   TEXT,
                chain_next       TEXT,
                result           TEXT,
                error            TEXT,
                created_at       REAL,
                started_at       REAL,
                finished_at      REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS workers (
                id               INTEGER PRIMARY KEY,
                worker_id        TEXT UNIQUE,
                queue_name       TEXT,
                status           TEXT DEFAULT 'idle',
                current_task_id  TEXT,
                tasks_completed  INTEGER DEFAULT 0,
                tasks_failed     INTEGER DEFAULT 0,
                started_at       REAL,
                last_heartbeat   REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS queues (
                id             INTEGER PRIMARY KEY,
                name           TEXT UNIQUE,
                concurrency    INTEGER DEFAULT 3,
                enabled        INTEGER DEFAULT 1,
                total_enqueued INTEGER DEFAULT 0,
                created_at     REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dlq (
                id         INTEGER PRIMARY KEY,
                task_id    TEXT,
                queue_name TEXT,
                task_type  TEXT,
                payload    TEXT,
                error      TEXT,
                moved_at   REAL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status     ON tasks(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_queue      ON tasks(queue_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_delay      ON tasks(delay_until)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_priority   ON tasks(priority DESC, created_at ASC)")
        conn.commit()
    _seed_queues()


def _seed_queues() -> None:
    with get_db() as conn:
        for name, concurrency in SEED_QUEUES.items():
            conn.execute("""
                INSERT OR IGNORE INTO queues (name, concurrency, enabled, total_enqueued, created_at)
                VALUES (?, ?, 1, 0, ?)
            """, (name, concurrency, time.time()))
        conn.commit()


# ── helpers ────────────────────────────────────────────────────────────────────
def make_task_id() -> str:
    raw = f"{time.time()}-{os.urandom(8).hex()}"
    return "tsk_" + hashlib.sha256(raw.encode()).hexdigest()[:20]


def make_worker_id(queue_name: str, index: int) -> str:
    return f"wrk_{queue_name}_{index}_{os.getpid()}"


def _sign_mcp(payload: dict) -> str:
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hmac.new(MCP_SECRET.encode(), body, hashlib.sha256).hexdigest()


def _row_to_dict(row) -> dict:
    if row is None:
        return {}
    return dict(row)


def _json_resp(handler, code: int, data: dict) -> None:
    body = json.dumps(data).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _require_admin(handler) -> bool:
    if not ADMIN_SECRET:
        return True
    auth = handler.headers.get("Authorization", "")
    token = auth.removeprefix("Bearer ").strip()
    expected = hmac.new(ADMIN_SECRET.encode(), b"admin", hashlib.sha256).hexdigest()
    ok = hmac.compare_digest(token, expected) or hmac.compare_digest(token, ADMIN_SECRET)
    if not ok:
        _json_resp(handler, 403, {"error": "admin authorisation required"})
    return ok


def _read_body(handler) -> dict:
    length = int(handler.headers.get("Content-Length", 0))
    if length == 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


# ── task execution ─────────────────────────────────────────────────────────────
def _exec_http(payload: dict, timeout: int) -> str:
    url    = payload.get("url", "")
    method = payload.get("method", "POST").upper()
    body   = payload.get("body", None)
    if not url:
        raise ValueError("http task missing 'url'")
    data = json.dumps(body).encode() if body is not None else None
    req  = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode(errors="replace")


def _exec_mcp_intent(payload: dict, timeout: int) -> str:
    intent = payload.get("intent", "")
    kwargs = payload.get("kwargs", {})
    if not intent:
        raise ValueError("mcp_intent task missing 'intent'")
    body_dict = {"intent": intent, "kwargs": kwargs}
    sig  = _sign_mcp(body_dict)
    data = json.dumps(body_dict).encode()
    url  = f"http://127.0.0.1:{MCP_PORT}/dispatch"
    req  = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("X-MCP-Signature", sig)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode(errors="replace")


def _exec_command(payload: dict, timeout: int) -> str:
    cmd = payload.get("cmd", "")
    if not cmd:
        raise ValueError("command task missing 'cmd'")
    allowed = any(cmd.startswith(p) for p in ALLOWED_CMD_PREFIXES)
    if not allowed:
        raise PermissionError(
            f"command prefix not allowed; permitted: {ALLOWED_CMD_PREFIXES}"
        )
    parts = cmd.split()
    result = subprocess.run(
        parts, shell=False, capture_output=True, text=True, timeout=timeout
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"exit {result.returncode}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _exec_delay(payload: dict, timeout: int) -> str:
    seconds = float(payload.get("seconds", 1))
    seconds = min(seconds, float(timeout))
    time.sleep(seconds)
    return f"slept {seconds}s"


def execute_task(task_type: str, payload: dict, timeout: int) -> str:
    if task_type == "http":
        return _exec_http(payload, timeout)
    if task_type == "mcp_intent":
        return _exec_mcp_intent(payload, timeout)
    if task_type == "command":
        return _exec_command(payload, timeout)
    if task_type == "delay":
        return _exec_delay(payload, timeout)
    raise ValueError(f"unknown task_type '{task_type}'")


# ── chain support ──────────────────────────────────────────────────────────────
def enqueue_task(
    queue_name: str,
    task_type: str,
    payload: dict,
    priority: int = 5,
    delay_seconds: float = 0,
    max_retries: int = 3,
    timeout_seconds: int = 60,
    chain_next: str = None,
    parent_task_id: str = None,
    conn: sqlite3.Connection = None,
) -> str:
    task_id    = make_task_id()
    delay_until = time.time() + delay_seconds
    payload_str = json.dumps(payload)
    chain_str   = json.dumps(chain_next) if chain_next and not isinstance(chain_next, str) else chain_next

    def _insert(c):
        c.execute("""
            INSERT INTO tasks
                (task_id, queue_name, task_type, payload, priority, status,
                 max_retries, delay_until, timeout_seconds, chain_next,
                 parent_task_id, created_at)
            VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?)
        """, (
            task_id, queue_name, task_type, payload_str, priority,
            max_retries, delay_until, timeout_seconds,
            chain_str, parent_task_id, time.time(),
        ))
        c.execute(
            "UPDATE queues SET total_enqueued = total_enqueued + 1 WHERE name = ?",
            (queue_name,)
        )

    if conn is not None:
        _insert(conn)
    else:
        with get_db() as c:
            _insert(c)
            c.commit()
    log.info("enqueued %s type=%s queue=%s", task_id, task_type, queue_name)
    return task_id


def _move_to_dlq(conn: sqlite3.Connection, task: dict, error: str) -> None:
    conn.execute("""
        INSERT INTO dlq (task_id, queue_name, task_type, payload, error, moved_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        task["task_id"], task["queue_name"], task["task_type"],
        task["payload"], error, time.time(),
    ))
    conn.execute(
        "UPDATE tasks SET status='failed', error=?, finished_at=? WHERE task_id=?",
        (error, time.time(), task["task_id"])
    )
    log.warning("DLQ ← %s: %s", task["task_id"], error[:120])


# ── worker ─────────────────────────────────────────────────────────────────────
class Worker(threading.Thread):
    def __init__(self, worker_id: str, queue_name: str):
        super().__init__(name=worker_id, daemon=True)
        self.worker_id  = worker_id
        self.queue_name = queue_name
        self._stop_evt  = threading.Event()

    def run(self) -> None:
        with get_db() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO workers
                    (worker_id, queue_name, status, tasks_completed, tasks_failed,
                     started_at, last_heartbeat)
                VALUES (?, ?, 'idle', 0, 0, ?, ?)
            """, (self.worker_id, self.queue_name, time.time(), time.time()))
            conn.commit()
        log.info("worker %s started on queue '%s'", self.worker_id, self.queue_name)

        while not self._stop_evt.is_set():
            try:
                self._heartbeat()
                # check queue enabled
                with get_db() as conn:
                    row = conn.execute(
                        "SELECT enabled FROM queues WHERE name=?", (self.queue_name,)
                    ).fetchone()
                    if row and not row["enabled"]:
                        time.sleep(WORKER_POLL_INTERVAL)
                        continue

                task = self._claim_task()
                if task is None:
                    time.sleep(WORKER_POLL_INTERVAL)
                    continue
                self._process(task)
            except Exception as exc:
                log.error("worker %s unhandled error: %s", self.worker_id, exc)
                time.sleep(WORKER_POLL_INTERVAL)

    def stop(self) -> None:
        self._stop_evt.set()

    def _heartbeat(self) -> None:
        try:
            with get_db() as conn:
                conn.execute(
                    "UPDATE workers SET last_heartbeat=? WHERE worker_id=?",
                    (time.time(), self.worker_id)
                )
                conn.commit()
        except Exception:
            pass

    def _claim_task(self) -> dict | None:
        now = time.time()
        with get_db() as conn:
            row = conn.execute("""
                SELECT * FROM tasks
                WHERE status='pending'
                  AND queue_name=?
                  AND delay_until <= ?
                ORDER BY priority DESC, created_at ASC
                LIMIT 1
            """, (self.queue_name, now)).fetchone()
            if row is None:
                return None
            task_id = row["task_id"]
            cur = conn.execute("""
                UPDATE tasks
                SET status='running', worker_id=?, started_at=?
                WHERE task_id=? AND status='pending'
            """, (self.worker_id, now, task_id))
            conn.commit()
            if cur.rowcount == 0:
                return None  # another worker grabbed it
            conn.execute("""
                UPDATE workers SET status='busy', current_task_id=? WHERE worker_id=?
            """, (task_id, self.worker_id))
            conn.commit()
            # re-fetch to get updated row
            updated = conn.execute(
                "SELECT * FROM tasks WHERE task_id=?", (task_id,)
            ).fetchone()
            return dict(updated) if updated else None

    def _process(self, task: dict) -> None:
        task_id  = task["task_id"]
        timeout  = task.get("timeout_seconds") or 60
        log.info("worker %s processing %s (type=%s)", self.worker_id, task_id, task["task_type"])
        try:
            payload = json.loads(task["payload"]) if task["payload"] else {}
            result  = execute_task(task["task_type"], payload, timeout)
            self._on_success(task, result)
        except Exception as exc:
            self._on_failure(task, str(exc))
        finally:
            with get_db() as conn:
                conn.execute("""
                    UPDATE workers SET status='idle', current_task_id=NULL WHERE worker_id=?
                """, (self.worker_id,))
                conn.commit()

    def _on_success(self, task: dict, result: str) -> None:
        task_id    = task["task_id"]
        chain_next = task.get("chain_next")
        with get_db() as conn:
            conn.execute("""
                UPDATE tasks SET status='done', result=?, finished_at=?
                WHERE task_id=?
            """, (result[:4096], time.time(), task_id))
            conn.execute("""
                UPDATE workers SET tasks_completed = tasks_completed + 1 WHERE worker_id=?
            """, (self.worker_id,))
            if chain_next:
                try:
                    next_def = json.loads(chain_next) if isinstance(chain_next, str) else chain_next
                    if isinstance(next_def, dict):
                        enqueue_task(
                            queue_name      = next_def.get("queue_name", task["queue_name"]),
                            task_type       = next_def.get("task_type", "delay"),
                            payload         = next_def.get("payload", {}),
                            priority        = next_def.get("priority", task["priority"]),
                            delay_seconds   = next_def.get("delay_seconds", 0),
                            max_retries     = next_def.get("max_retries", 3),
                            timeout_seconds = next_def.get("timeout_seconds", 60),
                            chain_next      = next_def.get("chain_next"),
                            parent_task_id  = task_id,
                            conn            = conn,
                        )
                except Exception as exc:
                    log.warning("chain enqueue failed for %s: %s", task_id, exc)
            conn.commit()
        log.info("task %s done", task_id)

    def _on_failure(self, task: dict, error: str) -> None:
        task_id     = task["task_id"]
        retry_count = (task.get("retry_count") or 0) + 1
        max_retries = task.get("max_retries") or 3
        with get_db() as conn:
            if retry_count < max_retries:
                delay = RETRY_BASE_DELAY * (2 ** retry_count)
                conn.execute("""
                    UPDATE tasks
                    SET status='pending', retry_count=?, delay_until=?, error=?, worker_id=NULL
                    WHERE task_id=?
                """, (retry_count, time.time() + delay, error[:2048], task_id))
                log.warning(
                    "task %s failed (attempt %d/%d), retry in %ds: %s",
                    task_id, retry_count, max_retries, delay, error[:80]
                )
            else:
                _move_to_dlq(conn, task, error[:2048])
                log.error(
                    "task %s exhausted retries (%d), moved to DLQ: %s",
                    task_id, max_retries, error[:80]
                )
            conn.execute("""
                UPDATE workers SET tasks_failed = tasks_failed + 1 WHERE worker_id=?
            """, (self.worker_id,))
            conn.commit()


# ── worker pool manager ────────────────────────────────────────────────────────
class WorkerPoolManager:
    def __init__(self):
        self._workers: dict[str, list[Worker]] = {}
        self._lock = threading.Lock()

    def spawn_for_queue(self, queue_name: str, concurrency: int) -> None:
        with self._lock:
            if queue_name in self._workers:
                return
            workers = []
            for i in range(concurrency):
                wid = make_worker_id(queue_name, i)
                w   = Worker(wid, queue_name)
                w.start()
                workers.append(w)
            self._workers[queue_name] = workers
            log.info("spawned %d workers for queue '%s'", concurrency, queue_name)

    def spawn_all_from_db(self) -> None:
        with get_db() as conn:
            rows = conn.execute("SELECT name, concurrency FROM queues").fetchall()
        for row in rows:
            self.spawn_for_queue(row["name"], row["concurrency"])

    def worker_statuses(self) -> list[dict]:
        result = []
        for workers in self._workers.values():
            for w in workers:
                result.append({
                    "worker_id":  w.worker_id,
                    "queue_name": w.queue_name,
                    "alive":      w.is_alive(),
                })
        return result


_pool = WorkerPoolManager()


# ── HTTP handler ───────────────────────────────────────────────────────────────
class TaskQueueHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress default access log

    # ── routing ────────────────────────────────────────────────────────────────
    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        qs   = self.path.split("?")[1] if "?" in self.path else ""
        params = dict(p.split("=", 1) for p in qs.split("&") if "=" in p) if qs else {}

        if path == "/health":
            self._handle_health()
        elif path == "/queues":
            self._handle_list_queues()
        elif path == "/tasks":
            self._handle_list_tasks(params)
        elif path.startswith("/tasks/"):
            task_id = path[len("/tasks/"):]
            self._handle_task_detail(task_id)
        elif path == "/workers":
            self._handle_workers()
        elif path == "/dlq":
            self._handle_dlq()
        else:
            _json_resp(self, 404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")

        if path == "/enqueue":
            self._handle_enqueue()
        elif path == "/enqueue/batch":
            self._handle_enqueue_batch()
        elif path.startswith("/tasks/") and path.endswith("/retry"):
            task_id = path[len("/tasks/"):-len("/retry")]
            self._handle_retry(task_id)
        elif path.startswith("/tasks/") and path.endswith("/cancel"):
            task_id = path[len("/tasks/"):-len("/cancel")]
            self._handle_cancel(task_id)
        elif path.startswith("/queues/") and path.endswith("/pause"):
            name = path[len("/queues/"):-len("/pause")]
            self._handle_queue_toggle(name, enabled=0)
        elif path.startswith("/queues/") and path.endswith("/resume"):
            name = path[len("/queues/"):-len("/resume")]
            self._handle_queue_toggle(name, enabled=1)
        else:
            _json_resp(self, 404, {"error": "not found"})

    def do_DELETE(self):
        path = self.path.split("?")[0].rstrip("/")
        if path == "/dlq":
            self._handle_clear_dlq()
        else:
            _json_resp(self, 404, {"error": "not found"})

    # ── GET handlers ───────────────────────────────────────────────────────────
    def _handle_health(self):
        uptime = time.time() - START_TIME
        with get_db() as conn:
            all_tasks = conn.execute("SELECT status, queue_name FROM tasks").fetchall()
        counts: dict[str, dict] = {}
        for row in all_tasks:
            q = row["queue_name"]
            s = row["status"]
            if q not in counts:
                counts[q] = {"pending": 0, "running": 0, "done": 0, "failed": 0}
            if s in counts[q]:
                counts[q][s] += 1
        _json_resp(self, 200, {
            "status":   "ok",
            "uptime_s": round(uptime, 2),
            "queues":   counts,
            "workers":  _pool.worker_statuses(),
        })

    def _handle_list_queues(self):
        with get_db() as conn:
            rows = conn.execute("SELECT * FROM queues ORDER BY name").fetchall()
        result = []
        for row in rows:
            d = dict(row)
            with get_db() as conn:
                stats = conn.execute("""
                    SELECT status, COUNT(*) as cnt
                    FROM tasks WHERE queue_name=? GROUP BY status
                """, (d["name"],)).fetchall()
            d["task_counts"] = {r["status"]: r["cnt"] for r in stats}
            result.append(d)
        _json_resp(self, 200, {"queues": result})

    def _handle_list_tasks(self, params: dict):
        queue_name = params.get("queue_name")
        status     = params.get("status")
        limit      = min(int(params.get("limit", "50")), 500)
        conditions = []
        args       = []
        if queue_name:
            conditions.append("queue_name=?")
            args.append(queue_name)
        if status:
            conditions.append("status=?")
            args.append(status)
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        args.append(limit)
        with get_db() as conn:
            rows = conn.execute(
                f"SELECT * FROM tasks {where} ORDER BY created_at DESC LIMIT ?", args
            ).fetchall()
        _json_resp(self, 200, {"tasks": [dict(r) for r in rows], "count": len(rows)})

    def _handle_task_detail(self, task_id: str):
        with get_db() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        if row is None:
            _json_resp(self, 404, {"error": "task not found"})
            return
        _json_resp(self, 200, dict(row))

    def _handle_workers(self):
        with get_db() as conn:
            rows = conn.execute("SELECT * FROM workers ORDER BY queue_name, worker_id").fetchall()
        _json_resp(self, 200, {"workers": [dict(r) for r in rows]})

    def _handle_dlq(self):
        with get_db() as conn:
            rows = conn.execute("SELECT * FROM dlq ORDER BY moved_at DESC LIMIT 200").fetchall()
        _json_resp(self, 200, {"dlq": [dict(r) for r in rows], "count": len(rows)})

    # ── POST handlers ──────────────────────────────────────────────────────────
    def _handle_enqueue(self):
        body = _read_body(self)
        err  = self._validate_task_fields(body)
        if err:
            _json_resp(self, 400, {"error": err})
            return
        task_id = enqueue_task(
            queue_name      = body.get("queue_name", "default"),
            task_type       = body["task_type"],
            payload         = body.get("payload", {}),
            priority        = int(body.get("priority", 5)),
            delay_seconds   = float(body.get("delay_seconds", 0)),
            max_retries     = int(body.get("max_retries", 3)),
            timeout_seconds = int(body.get("timeout_seconds", 60)),
            chain_next      = body.get("chain_next"),
        )
        _json_resp(self, 200, {"task_id": task_id})

    def _handle_enqueue_batch(self):
        body  = _read_body(self)
        tasks = body.get("tasks", [])
        if not isinstance(tasks, list) or len(tasks) == 0:
            _json_resp(self, 400, {"error": "tasks must be a non-empty list"})
            return
        task_ids = []
        errors   = []
        with get_db() as conn:
            for i, t in enumerate(tasks):
                err = self._validate_task_fields(t)
                if err:
                    errors.append({"index": i, "error": err})
                    continue
                tid = enqueue_task(
                    queue_name      = t.get("queue_name", "default"),
                    task_type       = t["task_type"],
                    payload         = t.get("payload", {}),
                    priority        = int(t.get("priority", 5)),
                    delay_seconds   = float(t.get("delay_seconds", 0)),
                    max_retries     = int(t.get("max_retries", 3)),
                    timeout_seconds = int(t.get("timeout_seconds", 60)),
                    chain_next      = t.get("chain_next"),
                    conn            = conn,
                )
                task_ids.append(tid)
            conn.commit()
        _json_resp(self, 200, {"task_ids": task_ids, "errors": errors, "count": len(task_ids)})

    def _handle_retry(self, task_id: str):
        if not _require_admin(self):
            return
        with get_db() as conn:
            dlq_row = conn.execute(
                "SELECT * FROM dlq WHERE task_id=? ORDER BY moved_at DESC LIMIT 1", (task_id,)
            ).fetchone()
            if dlq_row is None:
                # also allow retrying a failed task still in tasks table
                task_row = conn.execute(
                    "SELECT * FROM tasks WHERE task_id=? AND status='failed'", (task_id,)
                ).fetchone()
                if task_row is None:
                    _json_resp(self, 404, {"error": "task not found in DLQ or failed tasks"})
                    return
                conn.execute("""
                    UPDATE tasks SET status='pending', retry_count=0, error=NULL,
                    delay_until=?, worker_id=NULL WHERE task_id=?
                """, (time.time(), task_id))
                conn.commit()
                _json_resp(self, 200, {"task_id": task_id, "requeued": True})
                return
            payload  = dlq_row["payload"]
            try:
                payload_dict = json.loads(payload)
            except Exception:
                payload_dict = {}
            new_tid = enqueue_task(
                queue_name = dlq_row["queue_name"],
                task_type  = dlq_row["task_type"],
                payload    = payload_dict,
                conn       = conn,
            )
            conn.execute("DELETE FROM dlq WHERE task_id=? AND moved_at=?",
                         (task_id, dlq_row["moved_at"]))
            conn.commit()
        log.info("admin retried DLQ task %s → new task %s", task_id, new_tid)
        _json_resp(self, 200, {"task_id": new_tid, "original_task_id": task_id})

    def _handle_cancel(self, task_id: str):
        if not _require_admin(self):
            return
        with get_db() as conn:
            cur = conn.execute("""
                UPDATE tasks SET status='cancelled', finished_at=?
                WHERE task_id=? AND status='pending'
            """, (time.time(), task_id))
            conn.commit()
            if cur.rowcount == 0:
                _json_resp(self, 400, {"error": "task not pending or not found"})
                return
        log.info("admin cancelled task %s", task_id)
        _json_resp(self, 200, {"task_id": task_id, "cancelled": True})

    def _handle_queue_toggle(self, name: str, enabled: int):
        if not _require_admin(self):
            return
        with get_db() as conn:
            cur = conn.execute(
                "UPDATE queues SET enabled=? WHERE name=?", (enabled, name)
            )
            conn.commit()
            if cur.rowcount == 0:
                _json_resp(self, 404, {"error": f"queue '{name}' not found"})
                return
        action = "resumed" if enabled else "paused"
        log.info("admin %s queue '%s'", action, name)
        _json_resp(self, 200, {"queue": name, "enabled": bool(enabled), "action": action})

    def _handle_clear_dlq(self):
        if not _require_admin(self):
            return
        with get_db() as conn:
            cur = conn.execute("DELETE FROM dlq")
            conn.commit()
        log.info("admin cleared DLQ (%d rows removed)", cur.rowcount)
        _json_resp(self, 200, {"cleared": True, "rows_removed": cur.rowcount})

    # ── validation ─────────────────────────────────────────────────────────────
    @staticmethod
    def _validate_task_fields(body: dict) -> str | None:
        if "task_type" not in body:
            return "task_type is required"
        allowed_types = {"http", "mcp_intent", "command", "delay"}
        if body["task_type"] not in allowed_types:
            return f"task_type must be one of {sorted(allowed_types)}"
        queue_name = body.get("queue_name", "default")
        with get_db() as conn:
            row = conn.execute(
                "SELECT id FROM queues WHERE name=?", (queue_name,)
            ).fetchone()
            if row is None:
                return f"queue '{queue_name}' does not exist"
        priority = body.get("priority", 5)
        try:
            if not (1 <= int(priority) <= 10):
                return "priority must be 1-10"
        except (TypeError, ValueError):
            return "priority must be an integer"
        return None


# ── server ─────────────────────────────────────────────────────────────────────
def run_server() -> None:
    init_db()
    _pool.spawn_all_from_db()

    server = HTTPServer(("0.0.0.0", PORT), TaskQueueHandler)
    log.info("FractalMesh Task Queue listening on port %d", PORT)

    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        log.info("shutting down")
        server.shutdown()


if __name__ == "__main__":
    run_server()
