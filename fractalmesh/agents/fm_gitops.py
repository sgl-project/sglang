#!/usr/bin/env python3
"""
fm_gitops.py — GitOps / Deployment Automation Agent (Port 7847)
FractalMesh OMEGA Titan — pipeline orchestration, git operations, GitHub webhooks.
All credentials sourced from ~/.secrets/fractal.env at runtime.
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import json
import time
import hmac
import hashlib
import signal
import sqlite3
import subprocess
import threading
import logging
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

# ── vault ─────────────────────────────────────────────────────────────────────
_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _line in _vault.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

# ── config ────────────────────────────────────────────────────────────────────
PORT         = int(os.getenv("GITOPS_PORT", "7847"))
ROOT         = Path(os.path.expanduser(os.getenv("FRACTALMESH_HOME", "~/fmsaas")))
DB_PATH      = ROOT / "database" / "sovereign.db"
LOG_PATH     = ROOT / "logs" / "fm_gitops.log"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_ORG   = os.getenv("GITHUB_ORG", "")
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")
REPO_PATH    = Path(os.path.expanduser(os.getenv("GITOPS_REPO_PATH", "~/sglang")))

ROOT.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

DEFAULT_PIPELINE_STEPS = [
    {"name": "pull",         "cmd": "git -C {repo} pull origin {branch}"},
    {"name": "pm2_reload",   "cmd": "pm2 reload {agent_name}"},
    {"name": "syntax_check", "cmd": "python3 -m py_compile {file}"},
]
DEFAULT_PIPELINE_NAME  = "fractalmesh-auto-deploy"
DEFAULT_TRIGGER_BRANCH = "claude/deploy-fractalmesh-live-rkyHT"

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [FM-GITOPS] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
log = logging.getLogger("fm_gitops")

# ── database ──────────────────────────────────────────────────────────────────
def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=15, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    conn = _db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS deployments (
            id          INTEGER PRIMARY KEY,
            name        TEXT,
            branch      TEXT,
            commit_sha  TEXT,
            status      TEXT,
            trigger     TEXT,
            started_at  REAL,
            finished_at REAL,
            output      TEXT,
            error       TEXT
        );
        CREATE TABLE IF NOT EXISTS pipelines (
            id              INTEGER PRIMARY KEY,
            name            TEXT UNIQUE,
            steps           TEXT,
            trigger_branch  TEXT,
            enabled         INTEGER DEFAULT 1,
            last_run        REAL,
            run_count       INTEGER DEFAULT 0,
            created_at      REAL
        );
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id            INTEGER PRIMARY KEY,
            pipeline_id   INTEGER,
            deployment_id INTEGER,
            status        TEXT,
            started_at    REAL,
            finished_at   REAL,
            output        TEXT
        );
        CREATE TABLE IF NOT EXISTS git_events (
            id          INTEGER PRIMARY KEY,
            event_type  TEXT,
            branch      TEXT,
            commit_sha  TEXT,
            author      TEXT,
            message     TEXT,
            received_at REAL
        );
    """)
    conn.commit()
    conn.close()
    log.info("Database initialised at %s", DB_PATH)


def _seed_default_pipeline() -> None:
    conn = _db()
    existing = conn.execute(
        "SELECT id FROM pipelines WHERE name = ?", (DEFAULT_PIPELINE_NAME,)
    ).fetchone()
    if not existing:
        conn.execute(
            "INSERT INTO pipelines (name, steps, trigger_branch, enabled, created_at) "
            "VALUES (?, ?, ?, 1, ?)",
            (
                DEFAULT_PIPELINE_NAME,
                json.dumps(DEFAULT_PIPELINE_STEPS),
                DEFAULT_TRIGGER_BRANCH,
                time.time(),
            ),
        )
        conn.commit()
        log.info("Seeded default pipeline: %s", DEFAULT_PIPELINE_NAME)
    conn.close()

# ── helpers ───────────────────────────────────────────────────────────────────
def _run_cmd(cmd: str, timeout: int = 60, cwd: str = None):
    """Execute a shell command. Returns (stdout, stderr, returncode)."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except subprocess.TimeoutExpired:
        return "", f"Command timed out after {timeout}s", 1
    except Exception as exc:
        return "", str(exc), 1


def _check_auth(handler: "GitOpsHandler") -> bool:
    """Return True if X-Admin-Secret header matches ADMIN_SECRET."""
    provided = handler.headers.get("X-Admin-Secret", "")
    if not ADMIN_SECRET:
        return True  # no secret configured — open
    return hmac.compare_digest(provided, ADMIN_SECRET)


def _read_body(handler: "GitOpsHandler") -> dict:
    length = int(handler.headers.get("Content-Length", 0))
    if length == 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _send_json(handler: "GitOpsHandler", data: dict, code: int = 200) -> None:
    body = json.dumps(data).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _parse_path(path: str):
    """Split path and query string. Returns (clean_path, query_dict)."""
    if "?" in path:
        p, _, q = path.partition("?")
        params = {}
        for part in q.split("&"):
            if "=" in part:
                k, _, v = part.partition("=")
                params[k] = v
        return p.rstrip("/"), params
    return path.rstrip("/"), {}


def _split_pipeline_path(path: str):
    """
    Parse /pipelines/{id} and /pipelines/{id}/runs or /pipelines/{id}/run.
    Returns (pipeline_id_str, suffix) where suffix is '' | 'runs' | 'run'.
    """
    # strip leading /pipelines/
    rest = path[len("/pipelines/"):]
    parts = rest.split("/", 1)
    pid = parts[0]
    suffix = parts[1] if len(parts) > 1 else ""
    return pid, suffix

# ── deploy logic ──────────────────────────────────────────────────────────────
def _exec_deploy(deployment_id: int, branch: str) -> None:
    """Run default deploy pipeline in a background thread."""
    repo = str(REPO_PATH)
    steps = [
        f"git -C {repo} pull origin {branch}",
        "pm2 reload all",
    ]
    combined_out = []
    error_msg    = ""
    final_status = "success"

    for step_cmd in steps:
        log.info("[deploy:%d] running: %s", deployment_id, step_cmd)
        stdout, stderr, rc = _run_cmd(step_cmd, timeout=120)
        combined_out.append(f"$ {step_cmd}\n{stdout}")
        if stderr:
            combined_out.append(f"[stderr] {stderr}")
        if rc != 0:
            error_msg    = f"Step failed (rc={rc}): {step_cmd}\n{stderr}"
            final_status = "failed"
            break

    full_output = "\n".join(combined_out)
    conn = _db()
    conn.execute(
        "UPDATE deployments SET status=?, finished_at=?, output=?, error=? WHERE id=?",
        (final_status, time.time(), full_output, error_msg, deployment_id),
    )
    conn.commit()
    conn.close()
    log.info("[deploy:%d] finished with status=%s", deployment_id, final_status)


def _exec_pipeline(pipeline_id: int, branch: str, commit_sha: str) -> int:
    """
    Execute all pipeline steps sequentially in a background thread.
    Creates one deployment record and one pipeline_run record.
    Returns the pipeline_run id (or -1 on error).
    """
    conn = _db()
    row = conn.execute("SELECT * FROM pipelines WHERE id=?", (pipeline_id,)).fetchone()
    if not row:
        conn.close()
        return -1

    steps = json.loads(row["steps"])
    repo  = str(REPO_PATH)

    # deployment record
    dep_id = conn.execute(
        "INSERT INTO deployments (name, branch, commit_sha, status, trigger, started_at) "
        "VALUES (?, ?, ?, 'running', 'pipeline', ?)",
        (row["name"], branch, commit_sha, time.time()),
    ).lastrowid

    # pipeline_run record
    run_id = conn.execute(
        "INSERT INTO pipeline_runs (pipeline_id, deployment_id, status, started_at) "
        "VALUES (?, ?, 'running', ?)",
        (pipeline_id, dep_id, time.time()),
    ).lastrowid

    conn.execute(
        "UPDATE pipelines SET last_run=?, run_count=run_count+1 WHERE id=?",
        (time.time(), pipeline_id),
    )
    conn.commit()
    conn.close()

    combined_out = []
    error_msg    = ""
    final_status = "success"

    for step in steps:
        name    = step.get("name", "step")
        raw_cmd = step.get("cmd", "")
        cmd = raw_cmd.format(
            repo=repo,
            branch=branch,
            file=str(REPO_PATH / "fractalmesh" / "agents" / "fm_gitops.py"),
            agent_name=DEFAULT_PIPELINE_NAME,
        )
        log.info("[pipeline:%d run:%d] step=%s cmd=%s", pipeline_id, run_id, name, cmd)
        stdout, stderr, rc = _run_cmd(cmd, timeout=120)
        combined_out.append(f"[{name}] $ {cmd}\n{stdout}")
        if stderr:
            combined_out.append(f"[{name}][stderr] {stderr}")
        if rc != 0:
            error_msg    = f"Step '{name}' failed (rc={rc}): {stderr}"
            final_status = "failed"
            break

    full_output = "\n".join(combined_out)
    now = time.time()
    conn = _db()
    conn.execute(
        "UPDATE pipeline_runs SET status=?, finished_at=?, output=? WHERE id=?",
        (final_status, now, full_output, run_id),
    )
    conn.execute(
        "UPDATE deployments SET status=?, finished_at=?, output=?, error=? WHERE id=?",
        (final_status, now, full_output, error_msg, dep_id),
    )
    conn.commit()
    conn.close()
    log.info("[pipeline:%d run:%d] finished status=%s", pipeline_id, run_id, final_status)
    return run_id


def _trigger_matching_pipelines(branch: str, commit_sha: str) -> int:
    """Fire all enabled pipelines whose trigger_branch matches. Returns count."""
    conn = _db()
    rows = conn.execute(
        "SELECT id FROM pipelines WHERE trigger_branch=? AND enabled=1", (branch,)
    ).fetchall()
    conn.close()
    count = 0
    for row in rows:
        pid = row["id"]
        threading.Thread(
            target=_exec_pipeline, args=(pid, branch, commit_sha), daemon=True
        ).start()
        count += 1
    return count

# ── HTTP handler ──────────────────────────────────────────────────────────────
class GitOpsHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        log.info(fmt, *args)

    # ── routing ────────────────────────────────────────────────────────────────
    def do_GET(self):
        path, params = _parse_path(self.path)

        if path == "/health":
            return self._health()

        if path == "/deployments":
            return self._list_deployments()

        if path.startswith("/deployments/"):
            dep_id = path[len("/deployments/"):]
            return self._get_deployment(dep_id)

        if path == "/pipelines":
            return self._list_pipelines()

        if path.startswith("/pipelines/"):
            pid, suffix = _split_pipeline_path(path)
            if suffix == "runs":
                return self._pipeline_runs(pid)
            if suffix == "":
                return self._get_pipeline(pid)
            return _send_json(self, {"error": "not found"}, 404)

        if path == "/git/status":
            return self._git_status()

        if path == "/git/diff":
            return self._git_diff(params)

        if path == "/analytics":
            return self._analytics()

        _send_json(self, {"error": "not found"}, 404)

    def do_POST(self):
        path, _ = _parse_path(self.path)

        if path == "/deploy":
            return self._deploy()

        if path == "/pipelines/create":
            return self._create_pipeline()

        if path.startswith("/pipelines/"):
            pid, suffix = _split_pipeline_path(path)
            if suffix == "run":
                return self._run_pipeline(pid)
            return _send_json(self, {"error": "not found"}, 404)

        if path == "/webhooks/github":
            return self._github_webhook()

        if path == "/git/commit":
            return self._git_commit()

        if path == "/git/push":
            return self._git_push()

        _send_json(self, {"error": "not found"}, 404)

    def do_PUT(self):
        path, _ = _parse_path(self.path)
        if path.startswith("/pipelines/"):
            pid, suffix = _split_pipeline_path(path)
            if suffix == "":
                return self._update_pipeline(pid)
        _send_json(self, {"error": "not found"}, 404)

    # ── GET handlers ──────────────────────────────────────────────────────────
    def _health(self):
        _send_json(self, {"status": "ok", "service": "fm-gitops", "port": PORT})

    def _list_deployments(self):
        conn = _db()
        rows = conn.execute(
            "SELECT * FROM deployments ORDER BY id DESC LIMIT 20"
        ).fetchall()
        conn.close()
        _send_json(self, {"deployments": [dict(r) for r in rows]})

    def _get_deployment(self, dep_id: str):
        try:
            did = int(dep_id)
        except ValueError:
            return _send_json(self, {"error": "invalid id"}, 400)
        conn = _db()
        row = conn.execute("SELECT * FROM deployments WHERE id=?", (did,)).fetchone()
        conn.close()
        if not row:
            return _send_json(self, {"error": "not found"}, 404)
        _send_json(self, dict(row))

    def _list_pipelines(self):
        conn = _db()
        rows = conn.execute("SELECT * FROM pipelines ORDER BY id").fetchall()
        conn.close()
        result = []
        for r in rows:
            d = dict(r)
            try:
                d["steps"] = json.loads(d["steps"])
            except Exception:
                pass
            result.append(d)
        _send_json(self, {"pipelines": result})

    def _get_pipeline(self, pid: str):
        try:
            pid_int = int(pid)
        except ValueError:
            return _send_json(self, {"error": "invalid id"}, 400)
        conn = _db()
        row = conn.execute("SELECT * FROM pipelines WHERE id=?", (pid_int,)).fetchone()
        conn.close()
        if not row:
            return _send_json(self, {"error": "not found"}, 404)
        d = dict(row)
        try:
            d["steps"] = json.loads(d["steps"])
        except Exception:
            pass
        _send_json(self, d)

    def _pipeline_runs(self, pid: str):
        try:
            pid_int = int(pid)
        except ValueError:
            return _send_json(self, {"error": "invalid id"}, 400)
        conn = _db()
        rows = conn.execute(
            "SELECT * FROM pipeline_runs WHERE pipeline_id=? ORDER BY id DESC LIMIT 10",
            (pid_int,),
        ).fetchall()
        conn.close()
        _send_json(self, {"runs": [dict(r) for r in rows]})

    def _git_status(self):
        repo = str(REPO_PATH)
        status_out, _, _ = _run_cmd(f"git -C {repo} status --porcelain")
        log_out, _, _    = _run_cmd(f"git -C {repo} log --oneline -10")
        branch_out, _, _ = _run_cmd(f"git -C {repo} branch -a")
        _send_json(self, {
            "status_lines":   [line for line in status_out.splitlines() if line],
            "recent_commits": [line for line in log_out.splitlines() if line],
            "branches":       [line.strip() for line in branch_out.splitlines() if line],
        })

    def _git_diff(self, params: dict):
        branch = params.get("branch", "main")
        repo   = str(REPO_PATH)
        diff_out, _, _ = _run_cmd(
            f"git -C {repo} diff origin/{branch}...HEAD --stat"
        )
        lines = [line for line in diff_out.splitlines() if line]
        files_changed = 0
        for line in lines:
            if "changed" in line:
                try:
                    files_changed = int(line.strip().split()[0])
                except Exception:
                    pass
        _send_json(self, {"diff_stat": diff_out, "files_changed": files_changed})

    def _analytics(self):
        week_ago = time.time() - 7 * 86400
        conn = _db()
        success = conn.execute(
            "SELECT COUNT(*) FROM deployments WHERE status='success' AND started_at>=?",
            (week_ago,),
        ).fetchone()[0]
        failed = conn.execute(
            "SELECT COUNT(*) FROM deployments WHERE status='failed' AND started_at>=?",
            (week_ago,),
        ).fetchone()[0]
        avg_row = conn.execute(
            "SELECT AVG(finished_at - started_at) FROM deployments "
            "WHERE finished_at IS NOT NULL AND started_at>=?",
            (week_ago,),
        ).fetchone()[0]
        pipelines_enabled = conn.execute(
            "SELECT COUNT(*) FROM pipelines WHERE enabled=1"
        ).fetchone()[0]
        events_count = conn.execute("SELECT COUNT(*) FROM git_events").fetchone()[0]
        conn.close()
        _send_json(self, {
            "deployments_this_week": {"success": success, "failed": failed},
            "avg_deploy_time_seconds": round(avg_row or 0, 2),
            "pipelines_enabled": pipelines_enabled,
            "git_events_total": events_count,
        })

    # ── POST handlers ─────────────────────────────────────────────────────────
    def _deploy(self):
        body    = _read_body(self)
        name    = body.get("name", "fractalmesh-update")
        branch  = body.get("branch", "main")
        trigger = body.get("trigger", "manual")

        conn = _db()
        dep_id = conn.execute(
            "INSERT INTO deployments (name, branch, status, trigger, started_at) "
            "VALUES (?, ?, 'running', ?, ?)",
            (name, branch, trigger, time.time()),
        ).lastrowid
        conn.commit()
        conn.close()

        threading.Thread(
            target=_exec_deploy, args=(dep_id, branch), daemon=True
        ).start()
        _send_json(self, {"deployment_id": dep_id, "status": "running"})

    def _create_pipeline(self):
        if not _check_auth(self):
            return _send_json(self, {"error": "unauthorized"}, 401)
        body           = _read_body(self)
        name           = body.get("name")
        steps          = body.get("steps", [])
        trigger_branch = body.get("trigger_branch", "main")
        if not name:
            return _send_json(self, {"error": "name required"}, 400)
        try:
            conn = _db()
            pid = conn.execute(
                "INSERT INTO pipelines (name, steps, trigger_branch, enabled, created_at) "
                "VALUES (?, ?, ?, 1, ?)",
                (name, json.dumps(steps), trigger_branch, time.time()),
            ).lastrowid
            conn.commit()
            conn.close()
            _send_json(self, {"pipeline_id": pid}, 201)
        except sqlite3.IntegrityError:
            _send_json(self, {"error": "pipeline name already exists"}, 409)

    def _run_pipeline(self, pid: str):
        try:
            pid_int = int(pid)
        except ValueError:
            return _send_json(self, {"error": "invalid id"}, 400)

        body       = _read_body(self)
        branch     = body.get("branch", "main")
        commit_sha = body.get("commit_sha", "")

        conn = _db()
        row = conn.execute("SELECT id FROM pipelines WHERE id=?", (pid_int,)).fetchone()
        conn.close()
        if not row:
            return _send_json(self, {"error": "pipeline not found"}, 404)

        # Insert a placeholder run record so we can return run_id immediately.
        # _exec_pipeline will insert its own authoritative record; the placeholder
        # is marked 'queued' and updated to 'dispatched' once the thread starts.
        conn = _db()
        placeholder_id = conn.execute(
            "INSERT INTO pipeline_runs (pipeline_id, status, started_at) VALUES (?, 'queued', ?)",
            (pid_int, time.time()),
        ).lastrowid
        conn.commit()
        conn.close()

        def _run_thread():
            actual_run_id = _exec_pipeline(pid_int, branch, commit_sha)
            conn2 = _db()
            conn2.execute(
                "UPDATE pipeline_runs SET status='dispatched', deployment_id=? WHERE id=?",
                (actual_run_id, placeholder_id),
            )
            conn2.commit()
            conn2.close()

        threading.Thread(target=_run_thread, daemon=True).start()
        _send_json(self, {"run_id": placeholder_id, "status": "running"})

    def _github_webhook(self):
        event_type = self.headers.get("X-GitHub-Event", "unknown")
        body = _read_body(self)

        branch     = ""
        commit_sha = ""
        author     = ""
        message    = ""
        pipelines_triggered = 0

        if event_type == "push":
            ref        = body.get("ref", "")
            branch     = ref.replace("refs/heads/", "")
            commits    = body.get("commits", [])
            commit_sha = body.get("after", "")
            if commits:
                head    = commits[-1]
                author  = head.get("author", {}).get("name", "")
                message = head.get("message", "")
        elif event_type == "pull_request":
            pr         = body.get("pull_request", {})
            branch     = pr.get("head", {}).get("ref", "")
            commit_sha = pr.get("head", {}).get("sha", "")
            author     = pr.get("user", {}).get("login", "")
            message    = pr.get("title", "")

        conn = _db()
        conn.execute(
            "INSERT INTO git_events (event_type, branch, commit_sha, author, message, received_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (event_type, branch, commit_sha, author, message, time.time()),
        )
        conn.commit()
        conn.close()

        if event_type == "push" and branch:
            pipelines_triggered = _trigger_matching_pipelines(branch, commit_sha)
            log.info(
                "Push event branch=%s sha=%s — triggered %d pipelines",
                branch, commit_sha, pipelines_triggered,
            )

        _send_json(self, {
            "received": True,
            "event_type": event_type,
            "pipelines_triggered": pipelines_triggered,
        })

    def _git_commit(self):
        body    = _read_body(self)
        message = body.get("message", "Auto-commit from GitOps agent")
        files   = body.get("files", [])
        repo    = str(REPO_PATH)

        if files:
            files_arg = " ".join(f'"{f}"' for f in files)
            _, add_err, add_rc = _run_cmd(f"git -C {repo} add {files_arg}")
        else:
            _, add_err, add_rc = _run_cmd(f"git -C {repo} add -A")

        if add_rc != 0:
            return _send_json(self, {"committed": False, "reason": add_err}, 500)

        commit_out, commit_err, commit_rc = _run_cmd(
            f'git -C {repo} commit -m "{message}"'
        )
        if commit_rc != 0:
            reason = commit_err or commit_out
            if "nothing to commit" in reason.lower() or "nothing added" in reason.lower():
                return _send_json(self, {"committed": False, "reason": "nothing to commit"})
            return _send_json(self, {"committed": False, "reason": reason}, 500)

        sha_out, _, _ = _run_cmd(f"git -C {repo} rev-parse HEAD")
        _send_json(self, {"committed": True, "sha": sha_out})

    def _git_push(self):
        if not _check_auth(self):
            return _send_json(self, {"error": "unauthorized"}, 401)
        body   = _read_body(self)
        branch = body.get("branch", "main")
        repo   = str(REPO_PATH)
        _, push_err, push_rc = _run_cmd(
            f"git -C {repo} push origin {branch}", timeout=120
        )
        if push_rc != 0:
            return _send_json(self, {"pushed": False, "error": push_err}, 500)
        _send_json(self, {"pushed": True, "branch": branch})

    # ── PUT handlers ──────────────────────────────────────────────────────────
    def _update_pipeline(self, pid: str):
        if not _check_auth(self):
            return _send_json(self, {"error": "unauthorized"}, 401)
        try:
            pid_int = int(pid)
        except ValueError:
            return _send_json(self, {"error": "invalid id"}, 400)
        body = _read_body(self)

        conn = _db()
        row = conn.execute("SELECT * FROM pipelines WHERE id=?", (pid_int,)).fetchone()
        if not row:
            conn.close()
            return _send_json(self, {"error": "not found"}, 404)

        name           = body.get("name", row["name"])
        steps          = json.dumps(body.get("steps", json.loads(row["steps"])))
        trigger_branch = body.get("trigger_branch", row["trigger_branch"])
        enabled        = int(body.get("enabled", row["enabled"]))

        try:
            conn.execute(
                "UPDATE pipelines SET name=?, steps=?, trigger_branch=?, enabled=? WHERE id=?",
                (name, steps, trigger_branch, enabled, pid_int),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            conn.close()
            return _send_json(self, {"error": "name conflict"}, 409)
        conn.close()
        _send_json(self, {"updated": True, "pipeline_id": pid_int})

# ── server lifecycle ──────────────────────────────────────────────────────────
_server: HTTPServer = None


def _shutdown(signum, frame):
    log.info("Shutting down fm-gitops (signal %d)", signum)
    if _server:
        threading.Thread(target=_server.shutdown, daemon=True).start()


def main():
    global _server
    _init_db()
    _seed_default_pipeline()

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    _server = HTTPServer(("0.0.0.0", PORT), GitOpsHandler)
    log.info("fm-gitops listening on port %d | repo=%s", PORT, REPO_PATH)
    _server.serve_forever()


if __name__ == "__main__":
    main()
