#!/usr/bin/env python3
"""
fm_swarm.py — Swarm Batch Orchestration Agent (Port 7832)
FractalMesh OMEGA Titan — multi-strategy task batch dispatcher.
Credentials sourced from ~/.secrets/fractal.env at runtime — never hardcoded.
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import json
import time
import hmac
import hashlib
import sqlite3
import logging
import urllib.request
import urllib.error
import concurrent.futures
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Optional

# ── vault ─────────────────────────────────────────────────────────────────────
_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _line in _vault.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

# ── config ────────────────────────────────────────────────────────────────────
PORT         = int(os.getenv("SWARM_PORT", "7832"))
MCP_PORT     = int(os.getenv("MCP_PORT", "7785"))
MCP_SECRET   = os.getenv("MCP_SECRET", "fm_mcp_internal").encode()
MAX_WORKERS  = int(os.getenv("SWARM_MAX_CONCURRENT", "5"))
ROOT         = Path(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")))
DB           = ROOT / "database" / "sovereign.db"
LOG          = ROOT / "logs" / "fm_swarm.log"

ROOT.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)
DB.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [FM-SWARM] %(message)s",
    handlers=[
        logging.FileHandler(LOG),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("fm_swarm")

# ── global thread pool ────────────────────────────────────────────────────────
_pool = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)

# ── database ──────────────────────────────────────────────────────────────────
def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB), timeout=15, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS swarm_batches (
                id              INTEGER PRIMARY KEY,
                name            TEXT,
                description     TEXT,
                status          TEXT DEFAULT 'pending',
                total_tasks     INTEGER DEFAULT 0,
                completed_tasks INTEGER DEFAULT 0,
                failed_tasks    INTEGER DEFAULT 0,
                strategy        TEXT DEFAULT 'parallel',
                created_at      REAL,
                started_at      REAL,
                finished_at     REAL
            );
            CREATE TABLE IF NOT EXISTS swarm_tasks (
                id          INTEGER PRIMARY KEY,
                batch_id    INTEGER,
                seq         INTEGER,
                intent      TEXT,
                kwargs      TEXT,
                status      TEXT DEFAULT 'pending',
                depends_on  TEXT,
                result      TEXT,
                error       TEXT,
                started_at  REAL,
                finished_at REAL,
                retries     INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS swarm_results (
                id         INTEGER PRIMARY KEY,
                batch_id   INTEGER,
                task_id    INTEGER,
                output     TEXT,
                created_at REAL
            );
        """)
    log.info("DB tables ensured (WAL mode)")


# ── HMAC helper ───────────────────────────────────────────────────────────────
def _hmac_sig(body_bytes: bytes) -> str:
    return hmac.new(MCP_SECRET, body_bytes, hashlib.sha256).hexdigest()


# ── task dispatcher ───────────────────────────────────────────────────────────
def _dispatch_task(task_row: dict, prev_result: Optional[Any] = None) -> Any:
    """POST one task to MCP router. Returns parsed result or raises."""
    task_id  = task_row["id"]
    batch_id = task_row["batch_id"]
    intent   = task_row["intent"]
    kwargs   = json.loads(task_row["kwargs"] or "{}")
    strategy = task_row.get("strategy", "parallel")

    # pipeline: merge prev result dict into kwargs
    if strategy == "pipeline" and isinstance(prev_result, dict):
        kwargs.update(prev_result)

    payload = json.dumps({"intent": intent, "kwargs": kwargs}).encode()
    sig     = _hmac_sig(payload)
    url     = f"http://127.0.0.1:{MCP_PORT}/"

    now = time.time()
    with _db() as conn:
        conn.execute(
            "UPDATE swarm_tasks SET status='running', started_at=? WHERE id=?",
            (now, task_id),
        )

    result_data = None
    error_msg   = None
    new_status  = "completed"

    try:
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "X-MCP-Signature": sig,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode()
            try:
                result_data = json.loads(raw)
            except json.JSONDecodeError:
                result_data = {"raw": raw}
        log.info("Task %d (intent=%s) completed", task_id, intent)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode() if exc.fp else str(exc)
        error_msg  = f"HTTP {exc.code}: {body}"
        new_status = "failed"
        log.warning("Task %d failed: %s", task_id, error_msg)
    except Exception as exc:
        error_msg  = str(exc)
        new_status = "failed"
        log.warning("Task %d error: %s", task_id, error_msg)

    finished = time.time()

    with _db() as conn:
        conn.execute(
            """UPDATE swarm_tasks
               SET status=?, result=?, error=?, finished_at=?
               WHERE id=?""",
            (
                new_status,
                json.dumps(result_data) if result_data is not None else None,
                error_msg,
                finished,
                task_id,
            ),
        )
        if new_status == "completed":
            conn.execute(
                "UPDATE swarm_batches SET completed_tasks = completed_tasks + 1 WHERE id=?",
                (batch_id,),
            )
            conn.execute(
                "INSERT INTO swarm_results (batch_id, task_id, output, created_at) VALUES (?,?,?,?)",
                (batch_id, task_id, json.dumps(result_data), finished),
            )
        else:
            conn.execute(
                "UPDATE swarm_batches SET failed_tasks = failed_tasks + 1 WHERE id=?",
                (batch_id,),
            )

        # check if batch is fully done
        row = conn.execute(
            "SELECT total_tasks, completed_tasks, failed_tasks FROM swarm_batches WHERE id=?",
            (batch_id,),
        ).fetchone()
        if row and (row["completed_tasks"] + row["failed_tasks"]) >= row["total_tasks"]:
            batch_final = "completed" if row["failed_tasks"] == 0 else "failed"
            conn.execute(
                "UPDATE swarm_batches SET status=?, finished_at=? WHERE id=?",
                (batch_final, finished, batch_id),
            )
            log.info("Batch %d finished with status=%s", batch_id, batch_final)

    return result_data


# ── strategy runners ──────────────────────────────────────────────────────────
def _run_parallel(batch: dict, tasks: list) -> None:
    """Submit all tasks to pool concurrently; wait for all futures."""
    futures = []
    for task in tasks:
        f = _pool.submit(_dispatch_task, task, None)
        futures.append(f)
    concurrent.futures.wait(futures)


def _run_sequential(batch: dict, tasks: list) -> None:
    """Run tasks one by one in seq order."""
    sorted_tasks = sorted(tasks, key=lambda t: t["seq"])
    for task in sorted_tasks:
        _dispatch_task(task, None)


def _run_pipeline(batch: dict, tasks: list) -> None:
    """Run sequentially, passing each task's output into the next."""
    sorted_tasks = sorted(tasks, key=lambda t: t["seq"])
    prev_result  = None
    for task in sorted_tasks:
        prev_result = _dispatch_task(task, prev_result)


def _run_fan_out(batch: dict, tasks: list) -> None:
    """First task runs alone; remaining tasks fire in parallel."""
    sorted_tasks = sorted(tasks, key=lambda t: t["seq"])
    if not sorted_tasks:
        return
    seed  = sorted_tasks[0]
    seed_result = _dispatch_task(seed, None)
    rest  = sorted_tasks[1:]
    if rest:
        futures = [_pool.submit(_dispatch_task, t, seed_result) for t in rest]
        concurrent.futures.wait(futures)


def _run_fan_in(batch: dict, tasks: list) -> None:
    """All but last task run in parallel; final task runs after all complete."""
    sorted_tasks = sorted(tasks, key=lambda t: t["seq"])
    if not sorted_tasks:
        return
    *parallel_tasks, final_task = sorted_tasks
    if parallel_tasks:
        futures = [_pool.submit(_dispatch_task, t, None) for t in parallel_tasks]
        concurrent.futures.wait(futures)
    _dispatch_task(final_task, None)


def _execute_batch(batch_id: int) -> None:
    """Load batch + tasks from DB, run with appropriate strategy."""
    with _db() as conn:
        batch = conn.execute(
            "SELECT * FROM swarm_batches WHERE id=?", (batch_id,)
        ).fetchone()
        tasks = conn.execute(
            "SELECT *, ? AS strategy FROM swarm_tasks WHERE batch_id=? AND status='pending' ORDER BY seq",
            (batch["strategy"], batch_id),
        ).fetchall()

    if not tasks:
        log.warning("Batch %d has no pending tasks to execute", batch_id)
        return

    # convert rows to plain dicts (needed across thread boundaries)
    batch_dict = dict(batch)
    task_dicts = [dict(t) for t in tasks]

    strategy = batch_dict.get("strategy", "parallel")
    log.info("Batch %d starting strategy=%s tasks=%d", batch_id, strategy, len(task_dicts))

    if strategy == "sequential":
        _run_sequential(batch_dict, task_dicts)
    elif strategy == "pipeline":
        _run_pipeline(batch_dict, task_dicts)
    elif strategy == "fan_out":
        _run_fan_out(batch_dict, task_dicts)
    elif strategy == "fan_in":
        _run_fan_in(batch_dict, task_dicts)
    else:  # parallel (default)
        _run_parallel(batch_dict, task_dicts)


# ── preset definitions ────────────────────────────────────────────────────────
_PRESETS: dict[str, dict] = {
    "morning_sweep": {
        "name": "Morning Sweep",
        "description": "Daily revenue + crypto + lead pipeline sweep",
        "strategy": "sequential",
        "tasks": [
            {"seq": 1, "intent": "coingecko_price",  "kwargs": {"coins": ["bitcoin", "ethereum", "solana"]}},
            {"seq": 2, "intent": "mesh_status",       "kwargs": {"op": "health_check"}},
            {"seq": 3, "intent": "gumroad",           "kwargs": {"op": "sync_sales"}},
            {"seq": 4, "intent": "leadgen",           "kwargs": {"op": "sequence_execute_due"}},
            {"seq": 5, "intent": "sendgrid",          "kwargs": {"op": "stats"}},
        ],
    },
    "revenue_sync": {
        "name": "Revenue Sync",
        "description": "Parallel revenue aggregation across all platforms",
        "strategy": "parallel",
        "tasks": [
            {"seq": 1, "intent": "gumroad",   "kwargs": {"op": "sync_sales"}},
            {"seq": 2, "intent": "printful",  "kwargs": {"op": "sync_orders"}},
            {"seq": 3, "intent": "stripe",    "kwargs": {"op": "sync_charges"}},
            {"seq": 4, "intent": "coinbase",  "kwargs": {"op": "portfolio"}},
        ],
    },
    "lead_nurture": {
        "name": "Lead Nurture Pipeline",
        "description": "Crawl leads → bulk import → execute sequences",
        "strategy": "pipeline",
        "tasks": [
            {"seq": 1, "intent": "crawlbase",  "kwargs": {"op": "scrape_leads"}},
            {"seq": 2, "intent": "leadgen",    "kwargs": {"op": "bulk_import"}},
            {"seq": 3, "intent": "leadgen",    "kwargs": {"op": "sequence_execute_due"}},
        ],
    },
    "nft_refresh": {
        "name": "NFT Pricing Refresh",
        "description": "ETH price → dynamic NFT pricing → opensea portfolio sync",
        "strategy": "pipeline",
        "tasks": [
            {"seq": 1, "intent": "coingecko_price",  "kwargs": {"coins": ["ethereum"]}},
            {"seq": 2, "intent": "nft_engine",       "kwargs": {"op": "dynamic_pricing"}},
            {"seq": 3, "intent": "opensea",          "kwargs": {"op": "portfolio"}},
        ],
    },
    "osint_sweep": {
        "name": "OSINT Intelligence Sweep",
        "description": "Parallel person scans + domain recon, aggregated via fan_in",
        "strategy": "fan_in",
        "tasks": [
            {"seq": 1, "intent": "osint_spider", "kwargs": {"op": "person_scan", "target": "target_a"}},
            {"seq": 2, "intent": "osint_spider", "kwargs": {"op": "person_scan", "target": "target_b"}},
            {"seq": 3, "intent": "osint_spider", "kwargs": {"op": "person_scan", "target": "target_c"}},
            {"seq": 4, "intent": "osint_spider", "kwargs": {"op": "domain_scan"}},
        ],
    },
}


def _create_batch(name: str, description: str, strategy: str, tasks: list) -> int:
    """Insert batch + tasks into DB; return batch_id."""
    now = time.time()
    with _db() as conn:
        cur = conn.execute(
            """INSERT INTO swarm_batches (name, description, status, total_tasks,
               completed_tasks, failed_tasks, strategy, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (name, description, "pending", len(tasks), 0, 0, strategy, now),
        )
        batch_id = cur.lastrowid
        for t in tasks:
            conn.execute(
                """INSERT INTO swarm_tasks (batch_id, seq, intent, kwargs, status, depends_on)
                   VALUES (?,?,?,?,?,?)""",
                (
                    batch_id,
                    t.get("seq", 0),
                    t.get("intent", ""),
                    json.dumps(t.get("kwargs", {})),
                    "pending",
                    json.dumps(t.get("depends_on")) if t.get("depends_on") else None,
                ),
            )
    log.info("Created batch %d name=%s strategy=%s tasks=%d", batch_id, name, strategy, len(tasks))
    return batch_id


# ── HTTP handler ──────────────────────────────────────────────────────────────
class SwarmHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # suppress default access log
        log.debug(fmt, *args)

    def _send(self, code: int, data: Any) -> None:
        body = json.dumps(data, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Optional[dict]:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except Exception:
            return None

    def _path_parts(self) -> list:
        return [p for p in self.path.split("?")[0].split("/") if p]

    # ── routing ────────────────────────────────────────────────────────────────
    def do_GET(self):
        parts = self._path_parts()

        # GET /health
        if parts == ["health"]:
            return self._send(200, {
                "status": "ok",
                "service": "fm-swarm",
                "port": PORT,
                "max_concurrent": MAX_WORKERS,
            })

        # GET /batches
        if parts == ["batches"]:
            return self._handle_list_batches()

        # GET /analytics
        if parts == ["analytics"]:
            return self._handle_analytics()

        # GET /batch/{id}
        if len(parts) == 2 and parts[0] == "batch":
            return self._handle_get_batch(parts[1])

        # GET /batch/{id}/status
        if len(parts) == 3 and parts[0] == "batch" and parts[2] == "status":
            return self._handle_batch_status(parts[1])

        self._send(404, {"error": "not found"})

    def do_POST(self):
        parts = self._path_parts()

        # POST /batch/create
        if parts == ["batch", "create"]:
            return self._handle_create()

        # POST /batch/create_preset
        if parts == ["batch", "create_preset"]:
            return self._handle_create_preset()

        # POST /batch/{id}/start
        if len(parts) == 3 and parts[0] == "batch" and parts[2] == "start":
            return self._handle_start(parts[1])

        # POST /batch/{id}/retry_failed
        if len(parts) == 3 and parts[0] == "batch" and parts[2] == "retry_failed":
            return self._handle_retry_failed(parts[1])

        self._send(404, {"error": "not found"})

    def do_DELETE(self):
        parts = self._path_parts()

        # DELETE /batch/{id}
        if len(parts) == 2 and parts[0] == "batch":
            return self._handle_delete(parts[1])

        self._send(404, {"error": "not found"})

    # ── handlers ───────────────────────────────────────────────────────────────
    def _handle_create(self):
        body = self._read_json()
        if body is None:
            return self._send(400, {"error": "invalid JSON"})
        name        = body.get("name", "Unnamed Batch")
        description = body.get("description", "")
        strategy    = body.get("strategy", "parallel")
        tasks       = body.get("tasks", [])
        if not tasks:
            return self._send(400, {"error": "tasks list is required and must not be empty"})
        valid_strategies = {"parallel", "sequential", "pipeline", "fan_out", "fan_in"}
        if strategy not in valid_strategies:
            return self._send(400, {"error": f"strategy must be one of {sorted(valid_strategies)}"})
        batch_id = _create_batch(name, description, strategy, tasks)
        self._send(201, {"batch_id": batch_id, "task_count": len(tasks), "strategy": strategy})

    def _handle_create_preset(self):
        body = self._read_json()
        if body is None:
            return self._send(400, {"error": "invalid JSON"})
        preset_name = body.get("preset", "")
        preset = _PRESETS.get(preset_name)
        if not preset:
            return self._send(400, {"error": f"unknown preset; valid: {list(_PRESETS)}"})
        batch_id = _create_batch(
            preset["name"],
            preset["description"],
            preset["strategy"],
            preset["tasks"],
        )
        self._send(201, {
            "batch_id":   batch_id,
            "preset":     preset_name,
            "task_count": len(preset["tasks"]),
        })

    def _handle_start(self, raw_id: str):
        try:
            batch_id = int(raw_id)
        except ValueError:
            return self._send(400, {"error": "invalid batch id"})

        with _db() as conn:
            row = conn.execute(
                "SELECT * FROM swarm_batches WHERE id=?", (batch_id,)
            ).fetchone()
        if not row:
            return self._send(404, {"error": "batch not found"})
        if row["status"] not in ("pending", "failed"):
            return self._send(409, {"error": f"batch is {row['status']}; must be pending or failed to start"})

        now = time.time()
        with _db() as conn:
            conn.execute(
                "UPDATE swarm_batches SET status='running', started_at=? WHERE id=?",
                (now, batch_id),
            )
            total = conn.execute(
                "SELECT COUNT(*) AS c FROM swarm_tasks WHERE batch_id=? AND status='pending'",
                (batch_id,),
            ).fetchone()["c"]

        # dispatch async so the HTTP response returns immediately
        _pool.submit(_execute_batch, batch_id)

        self._send(202, {
            "batch_id":        batch_id,
            "status":          "running",
            "tasks_dispatched": total,
        })

    def _handle_get_batch(self, raw_id: str):
        try:
            batch_id = int(raw_id)
        except ValueError:
            return self._send(400, {"error": "invalid batch id"})

        with _db() as conn:
            batch = conn.execute(
                "SELECT * FROM swarm_batches WHERE id=?", (batch_id,)
            ).fetchone()
            if not batch:
                return self._send(404, {"error": "batch not found"})
            tasks = conn.execute(
                "SELECT * FROM swarm_tasks WHERE batch_id=? ORDER BY seq",
                (batch_id,),
            ).fetchall()

        task_list = []
        for t in tasks:
            d = dict(t)
            if d.get("kwargs"):
                try:
                    d["kwargs"] = json.loads(d["kwargs"])
                except Exception:
                    pass
            if d.get("result"):
                try:
                    d["result"] = json.loads(d["result"])
                except Exception:
                    pass
            if d.get("depends_on"):
                try:
                    d["depends_on"] = json.loads(d["depends_on"])
                except Exception:
                    pass
            task_list.append(d)

        resp = dict(batch)
        resp["tasks"] = task_list
        self._send(200, resp)

    def _handle_batch_status(self, raw_id: str):
        try:
            batch_id = int(raw_id)
        except ValueError:
            return self._send(400, {"error": "invalid batch id"})

        with _db() as conn:
            batch = conn.execute(
                "SELECT * FROM swarm_batches WHERE id=?", (batch_id,)
            ).fetchone()
        if not batch:
            return self._send(404, {"error": "batch not found"})

        pending = batch["total_tasks"] - batch["completed_tasks"] - batch["failed_tasks"]
        elapsed = None
        if batch["started_at"]:
            end = batch["finished_at"] or time.time()
            elapsed = round(end - batch["started_at"], 2)

        self._send(200, {
            "batch_id": batch_id,
            "status":   batch["status"],
            "progress": {
                "total":     batch["total_tasks"],
                "completed": batch["completed_tasks"],
                "failed":    batch["failed_tasks"],
                "pending":   max(0, pending),
            },
            "elapsed_seconds": elapsed,
        })

    def _handle_list_batches(self):
        with _db() as conn:
            rows = conn.execute(
                """SELECT id, name, status, total_tasks, completed_tasks,
                          failed_tasks, strategy, created_at, started_at, finished_at
                   FROM swarm_batches
                   ORDER BY id DESC"""
            ).fetchall()
        self._send(200, [dict(r) for r in rows])

    def _handle_delete(self, raw_id: str):
        try:
            batch_id = int(raw_id)
        except ValueError:
            return self._send(400, {"error": "invalid batch id"})

        with _db() as conn:
            row = conn.execute(
                "SELECT status FROM swarm_batches WHERE id=?", (batch_id,)
            ).fetchone()
            if not row:
                return self._send(404, {"error": "batch not found"})

            if row["status"] == "pending":
                conn.execute("DELETE FROM swarm_tasks WHERE batch_id=?", (batch_id,))
                conn.execute("DELETE FROM swarm_results WHERE batch_id=?", (batch_id,))
                conn.execute("DELETE FROM swarm_batches WHERE id=?", (batch_id,))
                log.info("Hard-deleted pending batch %d", batch_id)
                return self._send(200, {"batch_id": batch_id, "action": "deleted"})

            if row["status"] == "running":
                conn.execute(
                    "UPDATE swarm_tasks SET status='cancelled' WHERE batch_id=? AND status='pending'",
                    (batch_id,),
                )
                conn.execute(
                    "UPDATE swarm_batches SET status='cancelled' WHERE id=?",
                    (batch_id,),
                )
                log.info("Cancelled running batch %d", batch_id)
                return self._send(200, {"batch_id": batch_id, "action": "cancelled"})

        self._send(409, {"error": f"cannot delete batch with status={row['status']}"})

    def _handle_retry_failed(self, raw_id: str):
        try:
            batch_id = int(raw_id)
        except ValueError:
            return self._send(400, {"error": "invalid batch id"})

        with _db() as conn:
            row = conn.execute(
                "SELECT * FROM swarm_batches WHERE id=?", (batch_id,)
            ).fetchone()
            if not row:
                return self._send(404, {"error": "batch not found"})

            cur = conn.execute(
                """UPDATE swarm_tasks
                   SET status='pending', error=NULL, result=NULL,
                       started_at=NULL, finished_at=NULL
                   WHERE batch_id=? AND status='failed'""",
                (batch_id,),
            )
            retried = cur.rowcount
            conn.execute(
                """UPDATE swarm_batches
                   SET status='pending',
                       failed_tasks = failed_tasks - ?,
                       started_at=NULL, finished_at=NULL
                   WHERE id=?""",
                (retried, batch_id),
            )

        log.info("Batch %d: reset %d failed tasks to pending", batch_id, retried)
        self._send(200, {"batch_id": batch_id, "retried": retried})

    def _handle_analytics(self):
        with _db() as conn:
            totals = conn.execute(
                "SELECT COUNT(*) AS total, SUM(CASE WHEN status='running' THEN 1 ELSE 0 END) AS running FROM swarm_batches"
            ).fetchone()
            completed_count = conn.execute(
                "SELECT COUNT(*) AS c FROM swarm_batches WHERE status='completed'"
            ).fetchone()["c"]
            avg_dur = conn.execute(
                "SELECT AVG(finished_at - started_at) AS avg_d FROM swarm_batches WHERE finished_at IS NOT NULL AND started_at IS NOT NULL"
            ).fetchone()["avg_d"]
            task_totals = conn.execute(
                "SELECT COUNT(*) AS total FROM swarm_tasks"
            ).fetchone()["total"]
            task_by_status = conn.execute(
                "SELECT status, COUNT(*) AS cnt FROM swarm_tasks GROUP BY status"
            ).fetchall()

        total_b    = totals["total"] or 0
        running_b  = totals["running"] or 0
        success_r  = round(completed_count / total_b, 4) if total_b > 0 else 0.0

        self._send(200, {
            "total_batches":        total_b,
            "running":              running_b,
            "success_rate":         success_r,
            "avg_duration_seconds": round(avg_dur, 2) if avg_dur else None,
            "tasks_total":          task_totals,
            "tasks_by_status":      {r["status"]: r["cnt"] for r in task_by_status},
        })


# ── entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _init_db()
    server = HTTPServer(("0.0.0.0", PORT), SwarmHandler)
    log.info("FM-SWARM listening on port %d (max_concurrent=%d)", PORT, MAX_WORKERS)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down FM-SWARM")
        _pool.shutdown(wait=False)
        server.server_close()
