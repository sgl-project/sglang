#!/usr/bin/env python3
"""
fm_docker.py — Docker Engine + Docker Hub API Agent (Port 7823)
Local Docker Engine via subprocess, Docker Hub via REST API.
All credentials sourced from ~/.secrets/fractal.env at runtime.
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import json
import time
import signal
import sqlite3
import logging
import subprocess
import urllib.request
import urllib.error
import urllib.parse
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
PORT             = int(os.getenv("DOCKER_PORT", "7823"))
ROOT             = Path(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")))
DB               = ROOT / "database" / "sovereign.db"
LOG              = ROOT / "logs" / "docker.log"
DOCKER_PAT       = os.getenv("DOCKER_PAT", "")
DOCKER_USERNAME  = os.getenv("DOCKER_USERNAME", "")
HUB_BASE         = "https://hub.docker.com/v2"

ROOT.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)
DB.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [DOCKER] %(message)s",
    handlers=[logging.FileHandler(LOG), logging.StreamHandler()],
)
log = logging.getLogger("docker")

# ── Docker Hub JWT cache ──────────────────────────────────────────────────────
_hub_token: Optional[str] = None
_hub_token_ts: float = 0.0
_HUB_TOKEN_TTL = 3000  # seconds (~50 min; Hub JWTs expire at 60 min)

# ── database ──────────────────────────────────────────────────────────────────

def _db_init():
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS docker_ops (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            op           TEXT,
            container_id TEXT,
            image        TEXT,
            status       TEXT,
            ts           DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def _log_op(op: str, container_id: str, image: str, status: str):
    try:
        conn = sqlite3.connect(DB, timeout=5)
        conn.execute(
            "INSERT INTO docker_ops (op, container_id, image, status) VALUES (?, ?, ?, ?)",
            (op, container_id, image, status[:2048]),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning("db_log error: %s", e)

def _recent_ops(limit: int = 50) -> list:
    try:
        conn = sqlite3.connect(DB, timeout=5)
        rows = conn.execute(
            "SELECT id, op, container_id, image, status, ts FROM docker_ops "
            "ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [
            {"id": r[0], "op": r[1], "container_id": r[2],
             "image": r[3], "status": r[4], "ts": r[5]}
            for r in rows
        ]
    except Exception:
        return []

# ── Docker CLI helpers (local engine) ────────────────────────────────────────

def _docker(*args: str, timeout: int = 30) -> dict:
    """Run a docker CLI command and return {stdout, stderr, returncode}."""
    cmd = ["docker"] + list(args)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return {
            "stdout":     proc.stdout.strip(),
            "stderr":     proc.stderr.strip(),
            "returncode": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "timeout", "returncode": -1}
    except FileNotFoundError:
        return {"stdout": "", "stderr": "docker not found", "returncode": -2}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "returncode": -3}

def _docker_json(*args: str, timeout: int = 30) -> list:
    """Run docker command that outputs one JSON object per line."""
    r = _docker(*args, timeout=timeout)
    if r["returncode"] != 0:
        return [{"error": r["stderr"] or "docker command failed"}]
    results = []
    for line in r["stdout"].splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError:
            results.append({"raw": line})
    return results

# ── Docker Hub API helpers ────────────────────────────────────────────────────

def _hub_login() -> Optional[str]:
    """POST /v2/users/login and cache the JWT. Returns token or None."""
    global _hub_token, _hub_token_ts
    now = time.time()
    if _hub_token and (now - _hub_token_ts) < _HUB_TOKEN_TTL:
        return _hub_token

    if not DOCKER_USERNAME or not DOCKER_PAT:
        log.warning("DOCKER_USERNAME or DOCKER_PAT not set")
        return None

    payload = json.dumps({"username": DOCKER_USERNAME, "password": DOCKER_PAT}).encode()
    req = urllib.request.Request(
        f"{HUB_BASE}/users/login",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode())
            token = data.get("token")
            if token:
                _hub_token    = token
                _hub_token_ts = now
                log.info("Docker Hub login successful")
            return token
    except Exception as e:
        log.error("hub login error: %s", e)
        return None

def _hub_req(method: str, path: str, body: Optional[dict] = None) -> dict:
    """Call Docker Hub REST API with cached JWT."""
    token = _hub_login()
    if not token:
        return {"error": "Docker Hub authentication failed"}

    url     = f"{HUB_BASE}{path}"
    payload = json.dumps(body).encode() if body else None
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept":        "application/json",
        "User-Agent":    "FractalMesh-Docker/1.0",
    }
    if payload:
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=payload, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            raw = r.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw.strip() else {"status": "ok", "http": r.status}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        log.warning("hub %s %s → %d: %s", method, path, e.code, detail[:120])
        return {"error": f"http_{e.code}", "detail": detail}
    except Exception as e:
        log.error("hub request error: %s", e)
        return {"error": str(e)}

# ── query string parser ───────────────────────────────────────────────────────

def _qs(path: str) -> dict:
    if "?" not in path:
        return {}
    return dict(urllib.parse.parse_qsl(path.split("?", 1)[1]))

# ── HTTP handler ──────────────────────────────────────────────────────────────

class DockerHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _respond(self, code: int, body: Any):
        payload = json.dumps(body, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def _base_path(self) -> str:
        return self.path.split("?")[0]

    # ── GET ───────────────────────────────────────────────────────────────────

    def do_GET(self):
        base = self._base_path()
        qs   = _qs(self.path)

        try:
            # ── Local Engine endpoints ────────────────────────────────────────

            if base == "/health":
                r = _docker("version", "--format", "json")
                ok = r["returncode"] == 0
                version_data = {}
                if ok:
                    try:
                        version_data = json.loads(r["stdout"])
                    except Exception:
                        version_data = {"raw": r["stdout"]}
                self._respond(200, {
                    "status":          "ok" if ok else "error",
                    "docker_available": ok,
                    "hub_user":         DOCKER_USERNAME or "not_set",
                    "version":          version_data,
                    "error":            r["stderr"] if not ok else None,
                })

            elif base == "/containers":
                containers = _docker_json(
                    "ps", "--format",
                    '{"ID":"{{.ID}}","Image":"{{.Image}}","Command":"{{.Command}}",'
                    '"Status":"{{.Status}}","Names":"{{.Names}}","Ports":"{{.Ports}}"}'
                )
                _log_op("list_containers", "", "", f"count={len(containers)}")
                self._respond(200, {"count": len(containers), "containers": containers})

            elif base == "/images":
                images = _docker_json(
                    "images", "--format",
                    '{"ID":"{{.ID}}","Repository":"{{.Repository}}","Tag":"{{.Tag}}",'
                    '"Size":"{{.Size}}","CreatedAt":"{{.CreatedAt}}"}'
                )
                _log_op("list_images", "", "", f"count={len(images)}")
                self._respond(200, {"count": len(images), "images": images})

            elif base == "/logs":
                cid  = qs.get("id", "")
                tail = qs.get("tail", "100")
                if not cid:
                    self._respond(400, {"error": "id param required"})
                    return
                r = _docker("logs", "--tail", str(tail), cid, timeout=15)
                _log_op("logs", cid, "", f"tail={tail}")
                self._respond(200, {
                    "container_id": cid,
                    "tail":         tail,
                    "stdout":       r["stdout"],
                    "stderr":       r["stderr"],
                    "returncode":   r["returncode"],
                })

            elif base == "/stats":
                cid = qs.get("id", "")
                if not cid:
                    self._respond(400, {"error": "id param required"})
                    return
                r = _docker("stats", cid, "--no-stream", "--format", "json", timeout=15)
                stats_data = {}
                if r["returncode"] == 0 and r["stdout"]:
                    try:
                        stats_data = json.loads(r["stdout"])
                    except Exception:
                        stats_data = {"raw": r["stdout"]}
                _log_op("stats", cid, "", "ok" if r["returncode"] == 0 else "error")
                self._respond(200 if r["returncode"] == 0 else 500, {
                    "container_id": cid,
                    "stats":        stats_data,
                    "error":        r["stderr"] if r["returncode"] != 0 else None,
                })

            # ── Docker Hub endpoints ──────────────────────────────────────────

            elif base == "/hub/repos":
                if not DOCKER_USERNAME:
                    self._respond(400, {"error": "DOCKER_USERNAME not configured"})
                    return
                result = _hub_req("GET", f"/repositories/{DOCKER_USERNAME}/?page_size=100")
                repos  = result.get("results", [])
                _log_op("hub_list_repos", "", "", f"count={len(repos)}")
                self._respond(200, {"count": len(repos), "repos": repos, "raw": result})

            elif base == "/hub/tags":
                repo = qs.get("repo", "")
                if not repo:
                    self._respond(400, {"error": "repo param required"})
                    return
                username = DOCKER_USERNAME or "_"
                result   = _hub_req("GET", f"/repositories/{username}/{repo}/tags/?page_size=100")
                tags     = result.get("results", [])
                _log_op("hub_list_tags", "", repo, f"count={len(tags)}")
                self._respond(200, {"repo": repo, "count": len(tags), "tags": tags})

            elif base == "/ops":
                rows = _recent_ops()
                self._respond(200, {"count": len(rows), "ops": rows})

            else:
                self._respond(404, {"error": "not_found", "path": base})

        except Exception as e:
            log.error("GET %s error: %s", base, e)
            self._respond(500, {"error": str(e)})

    # ── POST ──────────────────────────────────────────────────────────────────

    def do_POST(self):
        base = self._base_path()
        try:
            data = self._read_body()
        except json.JSONDecodeError:
            self._respond(400, {"error": "invalid_json"})
            return
        except Exception as e:
            self._respond(400, {"error": str(e)})
            return

        try:
            if base == "/run":
                self._handle_run(data)
            elif base == "/stop":
                self._handle_stop(data)
            elif base == "/remove":
                self._handle_remove(data)
            elif base == "/pull":
                self._handle_pull(data)
            elif base == "/build":
                self._handle_build(data)
            elif base == "/hub/login":
                self._handle_hub_login()
            else:
                self._respond(404, {"error": "unknown_path", "path": base})
        except Exception as e:
            log.error("POST %s error: %s", base, e)
            self._respond(500, {"error": str(e)})

    def _handle_run(self, data: dict):
        image = data.get("image", "")
        cmd   = data.get("cmd", "")
        name  = data.get("name", "")
        if not image:
            self._respond(400, {"error": "image required"})
            return
        args = ["run", "-d"]
        if name:
            args += ["--name", name]
        args.append(image)
        if cmd:
            args += cmd.split() if isinstance(cmd, str) else cmd
        r = _docker(*args, timeout=60)
        container_id = r["stdout"] if r["returncode"] == 0 else ""
        _log_op("run", container_id, image,
                "started" if r["returncode"] == 0 else r["stderr"])
        self._respond(
            200 if r["returncode"] == 0 else 500,
            {"container_id": container_id, "image": image,
             "error": r["stderr"] if r["returncode"] != 0 else None},
        )

    def _handle_stop(self, data: dict):
        cid = data.get("container_id") or data.get("id", "")
        if not cid:
            self._respond(400, {"error": "container_id required"})
            return
        r = _docker("stop", str(cid), timeout=30)
        _log_op("stop", str(cid), "", "ok" if r["returncode"] == 0 else r["stderr"])
        self._respond(
            200 if r["returncode"] == 0 else 500,
            {"container_id": cid, "stopped": r["returncode"] == 0,
             "error": r["stderr"] if r["returncode"] != 0 else None},
        )

    def _handle_remove(self, data: dict):
        cid = data.get("container_id") or data.get("id", "")
        if not cid:
            self._respond(400, {"error": "container_id required"})
            return
        r = _docker("rm", "-f", str(cid), timeout=15)
        _log_op("remove", str(cid), "", "ok" if r["returncode"] == 0 else r["stderr"])
        self._respond(
            200 if r["returncode"] == 0 else 500,
            {"container_id": cid, "removed": r["returncode"] == 0,
             "error": r["stderr"] if r["returncode"] != 0 else None},
        )

    def _handle_pull(self, data: dict):
        image = data.get("image", "")
        if not image:
            self._respond(400, {"error": "image required"})
            return
        r = _docker("pull", image, timeout=120)
        _log_op("pull", "", image, "ok" if r["returncode"] == 0 else r["stderr"])
        self._respond(
            200 if r["returncode"] == 0 else 500,
            {"image": image, "pulled": r["returncode"] == 0,
             "output": r["stdout"], "error": r["stderr"] if r["returncode"] != 0 else None},
        )

    def _handle_build(self, data: dict):
        tag  = data.get("tag", "")
        path = data.get("path", "")
        if not tag or not path:
            self._respond(400, {"error": "tag and path required"})
            return
        # Safety: path must be absolute and exist
        build_path = os.path.abspath(path)
        if not os.path.isdir(build_path):
            self._respond(400, {"error": f"path does not exist: {build_path}"})
            return
        r = _docker("build", "-t", tag, build_path, timeout=300)
        _log_op("build", "", tag, "ok" if r["returncode"] == 0 else r["stderr"])
        self._respond(
            200 if r["returncode"] == 0 else 500,
            {"tag": tag, "path": build_path, "built": r["returncode"] == 0,
             "output": r["stdout"][-2000:] if r["stdout"] else "",
             "error":  r["stderr"][-1000:] if r["returncode"] != 0 else None},
        )

    def _handle_hub_login(self):
        global _hub_token, _hub_token_ts
        # Force re-authentication
        _hub_token    = None
        _hub_token_ts = 0.0
        token = _hub_login()
        if token:
            self._respond(200, {"status": "ok", "message": "Docker Hub login successful"})
        else:
            self._respond(401, {"error": "Docker Hub login failed — check DOCKER_USERNAME and DOCKER_PAT"})

# ── shutdown ───────────────────────────────────────────────────────────────────

_running = True

def _shutdown(*_):
    global _running
    log.info("shutdown signal — exiting cleanly")
    _running = False

signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT,  _shutdown)

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    _db_init()
    server = HTTPServer(("0.0.0.0", PORT), DockerHandler)
    log.info("Docker agent listening on port %d", PORT)
    log.info("Hub user: %s | PAT: %s",
             DOCKER_USERNAME or "NOT SET",
             "configured" if DOCKER_PAT else "NOT SET")
    try:
        while _running:
            server.handle_request()
    finally:
        server.server_close()
        log.info("Docker agent stopped")

if __name__ == "__main__":
    main()
