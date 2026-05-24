#!/usr/bin/env python3
"""
fm_github_ops.py — GitHub REST API Integration Agent (Port 7794)
Full GitHub API operations: repos, issues, PRs, actions, webhooks, releases.
All credentials sourced from ~/.secrets/fractal.env at runtime.
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import json
import time
import signal
import sqlite3
import logging
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
PORT          = int(os.getenv("GITHUB_OPS_PORT", "7794"))
ROOT          = Path(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")))
DB            = ROOT / "database" / "sovereign.db"
LOG           = ROOT / "logs" / "github_ops.log"
GITHUB_TOKEN  = os.getenv("GITHUB_TOKEN", "")
GITHUB_ORG    = os.getenv("GITHUB_ORG", "")
GITHUB_PREFIX = os.getenv("GITHUB_REPO_PREFIX", "")
GH_API        = "https://api.github.com"

ROOT.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)
DB.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [GITHUB_OPS] %(message)s",
    handlers=[logging.FileHandler(LOG), logging.StreamHandler()],
)
log = logging.getLogger("github_ops")

# ── database ──────────────────────────────────────────────────────────────────

def _db_init():
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS github_events (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT,
            repo       TEXT,
            payload    TEXT,
            ts         DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def _log_event(event_type: str, repo: str, payload: Any):
    try:
        conn = sqlite3.connect(DB, timeout=5)
        conn.execute(
            "INSERT INTO github_events (event_type, repo, payload) VALUES (?, ?, ?)",
            (event_type, repo, json.dumps(payload)[:4096]),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning("db_log error: %s", e)

def _recent_events(limit: int = 50) -> list:
    try:
        conn = sqlite3.connect(DB, timeout=5)
        rows = conn.execute(
            "SELECT id, event_type, repo, payload, ts FROM github_events "
            "ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [
            {"id": r[0], "event_type": r[1], "repo": r[2],
             "payload": json.loads(r[3]) if r[3] else {}, "ts": r[4]}
            for r in rows
        ]
    except Exception:
        return []

# ── GitHub API helper ─────────────────────────────────────────────────────────

def _gh(method: str, path: str, body: Optional[dict] = None) -> dict:
    """Call GitHub API with Bearer token. Returns parsed JSON or error dict."""
    if not GITHUB_TOKEN:
        return {"error": "GITHUB_TOKEN not configured"}

    url     = f"{GH_API}{path}"
    payload = json.dumps(body).encode() if body else None
    headers = {
        "Authorization":        f"Bearer {GITHUB_TOKEN}",
        "Accept":               "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent":           "FractalMesh-GitHubOps/1.0",
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
        log.warning("gh %s %s → %d: %s", method, path, e.code, detail[:120])
        return {"error": f"http_{e.code}", "detail": detail}
    except Exception as e:
        log.error("gh request error: %s", e)
        return {"error": str(e)}

def _gh_paginate(path: str, per_page: int = 100) -> list:
    """Fetch all pages from a GitHub list endpoint."""
    results = []
    sep     = "&" if "?" in path else "?"
    page    = 1
    while True:
        data = _gh("GET", f"{path}{sep}per_page={per_page}&page={page}")
        if isinstance(data, list):
            results.extend(data)
            if len(data) < per_page:
                break
            page += 1
        else:
            # error or non-list response — return what we have plus error
            if "error" in data:
                results.append(data)
            break
    return results

# ── query string parser ───────────────────────────────────────────────────────

def _qs(path: str) -> dict:
    """Parse query string from request path."""
    if "?" not in path:
        return {}
    return dict(urllib.parse.parse_qsl(path.split("?", 1)[1]))

# ── HTTP handler ──────────────────────────────────────────────────────────────

class GitHubHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress default access log; we use our own

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
        raw = self.rfile.read(length)
        return json.loads(raw)

    def _base_path(self) -> str:
        return self.path.split("?")[0]

    # ── GET ──────────────────────────────────────────────────────────────────

    def do_GET(self):
        base = self._base_path()
        qs   = _qs(self.path)

        if base == "/health":
            configured = bool(GITHUB_TOKEN)
            # quick org check
            repos_count = 0
            if configured and GITHUB_ORG:
                r = _gh("GET", f"/orgs/{GITHUB_ORG}/repos?per_page=1")
                repos_count = -1 if "error" in r else 1
            self._respond(200, {
                "status":     "ok",
                "org":        GITHUB_ORG or "not_set",
                "prefix":     GITHUB_PREFIX or "",
                "repos":      repos_count,
                "configured": configured,
            })

        elif base == "/repos":
            # list repos for org or authenticated user
            if GITHUB_ORG:
                path = f"/orgs/{GITHUB_ORG}/repos?type=all"
            else:
                path = "/user/repos?affiliation=owner,collaborator&sort=updated"
            data = _gh_paginate(path)
            filtered = (
                [r for r in data if isinstance(r, dict)
                 and r.get("name", "").startswith(GITHUB_PREFIX)]
                if GITHUB_PREFIX else data
            )
            _log_event("list_repos", GITHUB_ORG or "user", {"count": len(filtered)})
            self._respond(200, {"repos": len(filtered), "data": filtered})

        elif base == "/issues":
            repo = qs.get("repo", "")
            if not repo:
                self._respond(400, {"error": "repo param required"})
                return
            owner = GITHUB_ORG or "me"
            data  = _gh("GET", f"/repos/{owner}/{repo}/issues?state=open&per_page=100")
            _log_event("list_issues", repo, {"count": len(data) if isinstance(data, list) else 0})
            self._respond(200, {"repo": repo, "issues": data})

        elif base == "/pulls":
            repo = qs.get("repo", "")
            if not repo:
                self._respond(400, {"error": "repo param required"})
                return
            owner = GITHUB_ORG or "me"
            data  = _gh("GET", f"/repos/{owner}/{repo}/pulls?state=open&per_page=100")
            _log_event("list_pulls", repo, {"count": len(data) if isinstance(data, list) else 0})
            self._respond(200, {"repo": repo, "pulls": data})

        elif base == "/actions":
            repo = qs.get("repo", "")
            if not repo:
                self._respond(400, {"error": "repo param required"})
                return
            owner = GITHUB_ORG or "me"
            data  = _gh("GET", f"/repos/{owner}/{repo}/actions/runs?per_page=25")
            _log_event("list_actions", repo, {"repo": repo})
            self._respond(200, {"repo": repo, "runs": data})

        elif base == "/events":
            events = _recent_events()
            self._respond(200, {"count": len(events), "events": events})

        else:
            self._respond(404, {"error": "not_found", "path": base})

    # ── POST ─────────────────────────────────────────────────────────────────

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
            if base == "/issue":
                self._handle_create_issue(data)
            elif base == "/pr":
                self._handle_create_pr(data)
            elif base == "/webhook":
                self._handle_webhook(data)
            elif base == "/dispatch":
                self._handle_dispatch(data)
            elif base == "/release":
                self._handle_release(data)
            else:
                self._respond(404, {"error": "unknown_path", "path": base})
        except Exception as e:
            log.error("POST %s error: %s", base, e)
            self._respond(500, {"error": str(e)})

    def _handle_create_issue(self, data: dict):
        repo   = data.get("repo", "")
        title  = data.get("title", "")
        body   = data.get("body", "")
        labels = data.get("labels", [])
        if not repo or not title:
            self._respond(400, {"error": "repo and title required"})
            return
        owner  = GITHUB_ORG or "me"
        result = _gh("POST", f"/repos/{owner}/{repo}/issues", {
            "title":  title,
            "body":   body,
            "labels": labels,
        })
        _log_event("create_issue", repo, {"title": title, "number": result.get("number")})
        self._respond(201 if "number" in result else 400, result)

    def _handle_create_pr(self, data: dict):
        repo  = data.get("repo", "")
        head  = data.get("head", "")
        base  = data.get("base", "main")
        title = data.get("title", "")
        body  = data.get("body", "")
        if not repo or not head or not title:
            self._respond(400, {"error": "repo, head, and title required"})
            return
        owner  = GITHUB_ORG or "me"
        result = _gh("POST", f"/repos/{owner}/{repo}/pulls", {
            "title": title,
            "head":  head,
            "base":  base,
            "body":  body,
        })
        _log_event("create_pr", repo, {"title": title, "number": result.get("number")})
        self._respond(201 if "number" in result else 400, result)

    def _handle_webhook(self, data: dict):
        event_type = self.headers.get("X-GitHub-Event", "unknown")
        repo_name  = data.get("repository", {}).get("name", "unknown")
        _log_event(f"webhook_{event_type}", repo_name, data)
        log.info("webhook event=%s repo=%s", event_type, repo_name)
        self._respond(200, {"status": "received", "event": event_type, "repo": repo_name})

    def _handle_dispatch(self, data: dict):
        repo        = data.get("repo", "")
        workflow_id = data.get("workflow_id", "")
        inputs      = data.get("inputs", {})
        ref         = data.get("ref", "main")
        if not repo or not workflow_id:
            self._respond(400, {"error": "repo and workflow_id required"})
            return
        owner  = GITHUB_ORG or "me"
        result = _gh("POST",
                     f"/repos/{owner}/{repo}/actions/workflows/{workflow_id}/dispatches",
                     {"ref": ref, "inputs": inputs})
        _log_event("dispatch", repo, {"workflow_id": workflow_id, "inputs": inputs})
        # GitHub returns 204 No Content on success → result will be {"status":"ok"}
        self._respond(200, {"status": "dispatched", "repo": repo,
                            "workflow": workflow_id, "result": result})

    def _handle_release(self, data: dict):
        repo      = data.get("repo", "")
        tag       = data.get("tag", "")
        name      = data.get("name", tag)
        body      = data.get("body", "")
        draft     = data.get("draft", False)
        prerelease= data.get("prerelease", False)
        if not repo or not tag:
            self._respond(400, {"error": "repo and tag required"})
            return
        owner  = GITHUB_ORG or "me"
        result = _gh("POST", f"/repos/{owner}/{repo}/releases", {
            "tag_name":   tag,
            "name":       name,
            "body":       body,
            "draft":      draft,
            "prerelease": prerelease,
        })
        _log_event("create_release", repo, {"tag": tag, "name": name})
        self._respond(201 if "id" in result else 400, result)

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
    server = HTTPServer(("0.0.0.0", PORT), GitHubHandler)
    log.info("GitHub Ops agent listening on port %d", PORT)
    log.info("ORG: %s | PREFIX: %s | Token: %s",
             GITHUB_ORG or "not_set",
             GITHUB_PREFIX or "none",
             "configured" if GITHUB_TOKEN else "NOT SET")
    try:
        while _running:
            server.handle_request()
    finally:
        server.server_close()
        log.info("GitHub Ops agent stopped")

if __name__ == "__main__":
    main()
