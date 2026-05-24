"""
fm_langsmith.py — LangSmith Observability + Tracing Agent (port 7803)
FractalMesh | Author: Samuel James Hiotis | ABN 56 628 117 363

Covers: LangSmith tracing, evaluation, feedback, dataset management (langwasp).
Vault keys: LANGCHAIN_API_KEY, LANGCHAIN_PROJECT
Base URL: https://api.smith.langchain.com
"""

import json
import logging
import math
import os
import signal
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [fm_langsmith] %(levelname)s %(message)s",
)
log = logging.getLogger("fm_langsmith")

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
VAULT_PATH = Path.home() / ".secrets" / "fractal.env"
DB_PATH = ROOT / "database" / "sovereign.db"
PORT = 7803
LANGSMITH_BASE = "https://api.smith.langchain.com"

# ---------------------------------------------------------------------------
# Vault loader
# ---------------------------------------------------------------------------
def load_vault() -> dict:
    """Load key=value pairs from ~/.secrets/fractal.env (ignores comments)."""
    vault: dict = {}
    if not VAULT_PATH.exists():
        log.warning("Vault not found at %s — running unconfigured", VAULT_PATH)
        return vault
    with VAULT_PATH.open() as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, _, v = line.partition("=")
                vault[k.strip()] = v.strip().strip('"').strip("'")
    log.info("Vault loaded (%d keys)", len(vault))
    return vault


VAULT: dict = load_vault()
LANGCHAIN_API_KEY: str = VAULT.get("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT: str = VAULT.get("LANGCHAIN_PROJECT", "fractalmesh-default")

# ---------------------------------------------------------------------------
# SQLite — WAL mode
# ---------------------------------------------------------------------------
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
_db_lock = threading.Lock()


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS langsmith_traces (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id      TEXT,
                name        TEXT,
                run_type    TEXT,
                latency_ms  REAL,
                total_tokens INT,
                status      TEXT,
                ts          TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS langsmith_feedback (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id   TEXT,
                score    REAL,
                comment  TEXT,
                ts       TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS langsmith_datasets (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                name     TEXT,
                examples TEXT,
                ts       TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.commit()
    log.info("DB initialised at %s", DB_PATH)


# ---------------------------------------------------------------------------
# LangSmith API helper
# ---------------------------------------------------------------------------
def _ls(method: str, path: str, body: dict | None = None) -> dict:
    """Call LangSmith REST API. Returns parsed JSON dict or error dict."""
    if not LANGCHAIN_API_KEY:
        return {"error": "LANGCHAIN_API_KEY not configured"}
    url = LANGSMITH_BASE.rstrip("/") + "/" + path.lstrip("/")
    data = json.dumps(body).encode() if body else None
    headers = {
        "X-API-Key": LANGCHAIN_API_KEY,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        body_txt = exc.read().decode(errors="replace")
        log.warning("LangSmith %s %s → HTTP %d: %s", method, path, exc.code, body_txt[:200])
        return {"error": f"HTTP {exc.code}", "detail": body_txt[:400]}
    except Exception as exc:
        log.error("LangSmith call failed: %s", exc)
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Business logic helpers
# ---------------------------------------------------------------------------
def _store_trace(run_id: str, name: str, run_type: str,
                 latency_ms: float, total_tokens: int, status: str) -> int:
    with _db_lock, get_db() as conn:
        cur = conn.execute(
            "INSERT INTO langsmith_traces (run_id, name, run_type, latency_ms, total_tokens, status) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, name, run_type, latency_ms, total_tokens, status),
        )
        conn.commit()
        return cur.lastrowid


def _local_runs(limit: int = 50) -> list:
    with _db_lock, get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM langsmith_traces ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def _runs_stats() -> dict:
    with _db_lock, get_db() as conn:
        rows = conn.execute("""
            SELECT
                name,
                COUNT(*) AS total,
                ROUND(AVG(latency_ms), 2) AS avg_latency_ms,
                ROUND(MIN(latency_ms), 2) AS min_latency_ms,
                ROUND(MAX(latency_ms), 2) AS max_latency_ms,
                SUM(total_tokens) AS total_tokens,
                ROUND(100.0 * SUM(CASE WHEN status='success' THEN 1 ELSE 0 END) / COUNT(*), 1) AS success_pct
            FROM langsmith_traces
            GROUP BY name
            ORDER BY total DESC
        """).fetchall()
    return [dict(r) for r in rows]


def _local_feedback(limit: int = 50) -> list:
    with _db_lock, get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM langsmith_feedback ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def _dashboard_summary() -> dict:
    with _db_lock, get_db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM langsmith_traces").fetchone()[0]
        today = conn.execute(
            "SELECT COUNT(*) FROM langsmith_traces WHERE DATE(ts)=DATE('now')"
        ).fetchone()[0]
        avg_lat = conn.execute(
            "SELECT ROUND(AVG(latency_ms),2) FROM langsmith_traces WHERE DATE(ts)=DATE('now')"
        ).fetchone()[0]
        success = conn.execute(
            "SELECT COUNT(*) FROM langsmith_traces WHERE status='success' AND DATE(ts)=DATE('now')"
        ).fetchone()[0]
        pipelines = conn.execute(
            "SELECT name, COUNT(*) AS cnt FROM langsmith_traces GROUP BY name ORDER BY cnt DESC LIMIT 10"
        ).fetchall()
        tok_today = conn.execute(
            "SELECT SUM(total_tokens) FROM langsmith_traces WHERE DATE(ts)=DATE('now')"
        ).fetchone()[0] or 0
    return {
        "total_traces": total,
        "traces_today": today,
        "avg_latency_ms_today": avg_lat,
        "success_today": success,
        "tokens_today": tok_today,
        "top_pipelines": [dict(r) for r in pipelines],
        "project": LANGCHAIN_PROJECT,
        "configured": bool(LANGCHAIN_API_KEY),
    }


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------
class LangSmithHandler(BaseHTTPRequestHandler):
    server_version = "FractalMesh/2.1"
    sys_version = ""

    def log_message(self, fmt, *args):  # suppress default access log
        log.debug(fmt, *args)

    # -- helpers ----------------------------------------------------------

    def _send(self, code: int, payload: object) -> None:
        body = json.dumps(payload, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def _qs(self) -> dict:
        parsed = urllib.parse.urlparse(self.path)
        return dict(urllib.parse.parse_qsl(parsed.query))

    def _path(self) -> str:
        return urllib.parse.urlparse(self.path).path

    # -- GET --------------------------------------------------------------

    def do_GET(self):
        p = self._path()

        if p == "/health":
            self._send(200, {
                "status": "ok",
                "project": LANGCHAIN_PROJECT,
                "configured": bool(LANGCHAIN_API_KEY),
                "port": PORT,
            })

        elif p == "/runs":
            if LANGCHAIN_API_KEY:
                qs = self._qs()
                project = qs.get("project", LANGCHAIN_PROJECT)
                api_path = f"/api/v1/runs?project_name={urllib.parse.quote(project)}&limit=50"
                result = _ls("GET", api_path)
                if "error" not in result:
                    self._send(200, result)
                    return
            # Fallback to local DB
            self._send(200, {"runs": _local_runs(), "source": "local_db"})

        elif p == "/runs/stats":
            self._send(200, {"stats": _runs_stats()})

        elif p == "/feedback":
            result = _local_feedback()
            self._send(200, {"feedback": result})

        elif p == "/dashboard":
            self._send(200, _dashboard_summary())

        else:
            self._send(404, {"error": "Not found", "path": p})

    # -- POST -------------------------------------------------------------

    def do_POST(self):
        p = self._path()
        body = self._read_json()

        if p == "/trace":
            name = body.get("name", "unnamed")
            inputs = body.get("inputs", {})
            outputs = body.get("outputs", {})
            latency_ms = float(body.get("latency_ms", 0))
            tokens = int(body.get("tokens", 0))
            run_type = body.get("run_type", "chain")
            status = body.get("status", "success")

            import uuid
            run_id = str(uuid.uuid4())

            # Push to LangSmith if configured
            ls_result = {}
            if LANGCHAIN_API_KEY:
                ls_result = _ls("POST", "/api/v1/runs", {
                    "id": run_id,
                    "name": name,
                    "run_type": run_type,
                    "inputs": inputs,
                    "outputs": outputs,
                    "extra": {"latency_ms": latency_ms, "total_tokens": tokens},
                    "project_name": LANGCHAIN_PROJECT,
                    "status": status,
                })

            # Always store locally
            row_id = _store_trace(run_id, name, run_type, latency_ms, tokens, status)
            self._send(200, {
                "ok": True,
                "run_id": run_id,
                "local_id": row_id,
                "langsmith": ls_result,
            })

        elif p == "/feedback":
            run_id = body.get("run_id", "")
            score = float(body.get("score", 0))
            comment = body.get("comment", "")

            ls_result = {}
            if LANGCHAIN_API_KEY and run_id:
                import uuid
                ls_result = _ls("POST", "/api/v1/feedback", {
                    "id": str(uuid.uuid4()),
                    "run_id": run_id,
                    "score": score,
                    "comment": comment,
                    "key": "user_score",
                })

            with _db_lock, get_db() as conn:
                conn.execute(
                    "INSERT INTO langsmith_feedback (run_id, score, comment) VALUES (?,?,?)",
                    (run_id, score, comment),
                )
                conn.commit()

            self._send(200, {"ok": True, "langsmith": ls_result})

        elif p == "/dataset":
            name = body.get("name", "")
            examples = body.get("examples", [])
            if not name:
                self._send(400, {"error": "name required"})
                return

            ls_result = {}
            if LANGCHAIN_API_KEY:
                ls_result = _ls("POST", "/api/v1/datasets", {
                    "name": name,
                    "description": f"Created by FractalMesh fm_langsmith agent",
                })
                if "id" in ls_result:
                    dataset_id = ls_result["id"]
                    for ex in examples:
                        _ls("POST", "/api/v1/examples", {
                            "dataset_id": dataset_id,
                            "inputs": {"input": ex.get("input", "")},
                            "outputs": {"output": ex.get("output", "")},
                        })

            with _db_lock, get_db() as conn:
                conn.execute(
                    "INSERT INTO langsmith_datasets (name, examples) VALUES (?,?)",
                    (name, json.dumps(examples)),
                )
                conn.commit()

            self._send(200, {"ok": True, "name": name, "examples": len(examples), "langsmith": ls_result})

        elif p == "/evaluate":
            dataset_id = body.get("dataset_id", "")
            pipeline = body.get("pipeline", "")
            if not dataset_id:
                self._send(400, {"error": "dataset_id required"})
                return
            ls_result = {}
            if LANGCHAIN_API_KEY:
                ls_result = _ls("POST", "/api/v1/runs", {
                    "name": f"eval_{pipeline}",
                    "run_type": "chain",
                    "inputs": {"dataset_id": dataset_id, "pipeline": pipeline},
                    "project_name": LANGCHAIN_PROJECT,
                })
            self._send(200, {"ok": True, "dataset_id": dataset_id, "pipeline": pipeline, "langsmith": ls_result})

        else:
            self._send(404, {"error": "Not found", "path": p})


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------
_server: HTTPServer | None = None


def _shutdown(signum, frame):
    log.info("Signal %d received — shutting down", signum)
    if _server:
        threading.Thread(target=_server.shutdown, daemon=True).start()


def main():
    global _server
    init_db()
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    _server = HTTPServer(("0.0.0.0", PORT), LangSmithHandler)
    log.info("fm_langsmith listening on port %d | project=%s | configured=%s",
             PORT, LANGCHAIN_PROJECT, bool(LANGCHAIN_API_KEY))
    try:
        _server.serve_forever()
    finally:
        log.info("fm_langsmith stopped")


if __name__ == "__main__":
    main()
