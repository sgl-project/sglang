#!/usr/bin/env python3
"""
fm_bugcrowd.py — Bugcrowd API v1 Integration Agent (Port 7825)
Bug bounty programs, submissions, leaderboard, rewards aggregation.
All credentials sourced from ~/.secrets/fractal.env at runtime.
Samuel James Hiotis | ABN 56 628 117 363
"""
import json
import logging
import os
import signal
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Optional

# ── vault ─────────────────────────────────────────────────────────────────────
_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _line in _vault.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

# ── config ────────────────────────────────────────────────────────────────────
PORT                = int(os.getenv("BUGCROWD_PORT", "7825"))
ROOT                = Path(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")))
DB                  = ROOT / "database" / "sovereign.db"
LOG                 = ROOT / "logs" / "bugcrowd.log"
BUGCROWD_API_TOKEN  = os.getenv("BUGCROWD_API_TOKEN", "")
BC_BASE             = "https://api.bugcrowd.com"
BC_CONTENT_TYPE     = "application/vnd.bugcrowd.v1+json"

ROOT.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)
DB.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [BUGCROWD] %(message)s",
    handlers=[logging.FileHandler(LOG), logging.StreamHandler()],
)
log = logging.getLogger("bugcrowd")

# ── database ──────────────────────────────────────────────────────────────────

def _db_init():
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bugcrowd_submissions (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            sub_id    TEXT,
            title     TEXT,
            severity  TEXT,
            status    TEXT,
            reward    TEXT,
            program   TEXT,
            ts        DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def _log_submission(sub_id: str, title: str, severity: str,
                    status: str, reward: str, program: str):
    try:
        conn = sqlite3.connect(DB, timeout=5)
        conn.execute(
            "INSERT INTO bugcrowd_submissions "
            "(sub_id, title, severity, status, reward, program) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (sub_id, title[:512], severity, status, str(reward), program[:256]),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning("db_log error: %s", e)

def _recent_submissions(limit: int = 50) -> list:
    try:
        conn = sqlite3.connect(DB, timeout=5)
        rows = conn.execute(
            "SELECT id, sub_id, title, severity, status, reward, program, ts "
            "FROM bugcrowd_submissions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [
            {"id": r[0], "sub_id": r[1], "title": r[2], "severity": r[3],
             "status": r[4], "reward": r[5], "program": r[6], "ts": r[7]}
            for r in rows
        ]
    except Exception:
        return []

# ── Bugcrowd API helper ───────────────────────────────────────────────────────

def _bc(method: str, path: str, body: Optional[dict] = None,
        params: Optional[dict] = None) -> dict:
    """Call Bugcrowd API v1 with Token authentication."""
    if not BUGCROWD_API_TOKEN:
        return {"error": "BUGCROWD_API_TOKEN not configured"}

    url = f"{BC_BASE}{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"

    payload = json.dumps(body).encode() if body else None
    headers = {
        "Authorization": f"Token token={BUGCROWD_API_TOKEN}",
        "Accept":        BC_CONTENT_TYPE,
        "User-Agent":    "FractalMesh-Bugcrowd/1.0",
    }
    if payload:
        headers["Content-Type"] = BC_CONTENT_TYPE

    req = urllib.request.Request(url, data=payload, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            raw = r.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw.strip() else {"status": "ok", "http": r.status}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        log.warning("bc %s %s → %d: %s", method, path, e.code, detail[:120])
        return {"error": f"http_{e.code}", "detail": detail}
    except Exception as e:
        log.error("bc request error: %s", e)
        return {"error": str(e)}

def _bc_paginate(path: str, extra_params: Optional[dict] = None,
                 page_size: int = 25) -> list:
    """Fetch all pages from a Bugcrowd list endpoint."""
    results = []
    page    = 1
    while True:
        params = {"page[number]": page, "page[size]": page_size}
        if extra_params:
            params.update(extra_params)
        data = _bc("GET", path, params=params)
        if "error" in data:
            results.append(data)
            break

        # Bugcrowd v1 wraps lists under a key matching the resource type
        # Try common keys: submissions, bounty_briefs, etc.
        items = None
        for key in ("submissions", "bounty_briefs", "researchers", "data"):
            if key in data and isinstance(data[key], list):
                items = data[key]
                break
        if items is None:
            # Fallback: return the whole response if we can't find a list
            results.append(data)
            break

        results.extend(items)
        # Check if there are more pages
        meta  = data.get("meta", {})
        total = meta.get("total_count", len(items))
        if len(results) >= total or len(items) < page_size:
            break
        page += 1
    return results

# ── query string parser ───────────────────────────────────────────────────────

def _qs(path: str) -> dict:
    if "?" not in path:
        return {}
    return dict(urllib.parse.parse_qsl(path.split("?", 1)[1]))

# ── Severity constants ────────────────────────────────────────────────────────

VALID_SEVERITIES = {"P1", "P2", "P3", "P4", "P5"}

# ── HTTP handler ──────────────────────────────────────────────────────────────

class BugcrowdHandler(BaseHTTPRequestHandler):
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
            if base == "/health":
                # Verify auth by fetching one bounty brief
                result = _bc("GET", "/bounty_briefs", params={"page[size]": 1})
                ok = "error" not in result
                self._respond(200, {
                    "status":      "ok" if ok else "error",
                    "configured":  bool(BUGCROWD_API_TOKEN),
                    "bc_response": result,
                })

            elif base == "/programs":
                result = _bc("GET", "/bounty_briefs", params={
                    "sort[field]":     "promoted",
                    "sort[direction]": "desc",
                    "page[size]":      50,
                })
                programs = result.get("bounty_briefs", []) if "error" not in result else []
                self._respond(200, {
                    "count":    len(programs),
                    "programs": programs,
                    "error":    result.get("error"),
                })

            elif base == "/program":
                program_id = qs.get("id", "")
                if not program_id:
                    self._respond(400, {"error": "id param required"})
                    return
                result = _bc("GET", f"/bounty_briefs/{program_id}")
                brief  = result.get("bounty_brief") or result
                self._respond(200 if "error" not in result else 404, brief)

            elif base == "/submissions":
                result      = _bc("GET", "/submissions", params={"page[size]": 50})
                submissions = result.get("submissions", []) if "error" not in result else []
                self._respond(200, {
                    "count":       len(submissions),
                    "submissions": submissions,
                    "error":       result.get("error"),
                })

            elif base == "/submission":
                sub_id = qs.get("id", "")
                if not sub_id:
                    self._respond(400, {"error": "id param required"})
                    return
                result = _bc("GET", f"/submissions/{sub_id}")
                sub    = result.get("submission") or result
                self._respond(200 if "error" not in result else 404, sub)

            elif base == "/rewards":
                self._handle_rewards()

            elif base == "/leaderboard":
                result      = _bc("GET", "/researchers/leaderboard")
                researchers = result.get("researchers", []) if "error" not in result else []
                self._respond(200, {
                    "count":       len(researchers),
                    "leaderboard": researchers,
                    "error":       result.get("error"),
                })

            elif base == "/categories":
                # Vulnerability categories used when submitting bugs
                categories = [
                    {"id": "cross_site_scripting",           "name": "Cross-Site Scripting (XSS)"},
                    {"id": "sql_injection",                  "name": "SQL Injection"},
                    {"id": "broken_authentication",          "name": "Broken Authentication"},
                    {"id": "sensitive_data_exposure",        "name": "Sensitive Data Exposure"},
                    {"id": "security_misconfiguration",      "name": "Security Misconfiguration"},
                    {"id": "insecure_direct_object_references", "name": "IDOR"},
                    {"id": "csrf",                           "name": "Cross-Site Request Forgery (CSRF)"},
                    {"id": "ssrf",                           "name": "Server-Side Request Forgery (SSRF)"},
                    {"id": "rce",                            "name": "Remote Code Execution (RCE)"},
                    {"id": "business_logic",                 "name": "Business Logic Flaw"},
                    {"id": "privilege_escalation",           "name": "Privilege Escalation"},
                    {"id": "information_disclosure",         "name": "Information Disclosure"},
                    {"id": "open_redirect",                  "name": "Open Redirect"},
                    {"id": "clickjacking",                   "name": "Clickjacking"},
                    {"id": "path_traversal",                 "name": "Path Traversal"},
                    {"id": "denial_of_service",              "name": "Denial of Service (DoS)"},
                    {"id": "cryptographic_issues",           "name": "Cryptographic Issues"},
                    {"id": "api_abuse",                      "name": "API Abuse"},
                    {"id": "other",                          "name": "Other"},
                ]
                self._respond(200, {"count": len(categories), "categories": categories})

            elif base == "/db_submissions":
                rows = _recent_submissions()
                self._respond(200, {"count": len(rows), "submissions": rows})

            else:
                self._respond(404, {"error": "not_found", "path": base})

        except Exception as e:
            log.error("GET %s error: %s", base, e)
            self._respond(500, {"error": str(e)})

    def _handle_rewards(self):
        """Aggregate total reward from resolved submissions."""
        result      = _bc("GET", "/submissions", params={
            "filter[state]": "resolved",
            "page[size]":    100,
        })
        submissions = result.get("submissions", []) if "error" not in result else []

        total_usd   = 0.0
        rewarded    = []
        for sub in submissions:
            # Reward may be nested under 'reward' or 'monetary_reward' depending on API version
            reward = (
                sub.get("monetary_reward")
                or sub.get("reward")
                or {}
            )
            amount = reward.get("amount") or reward.get("value") or 0
            try:
                amount = float(amount)
            except (TypeError, ValueError):
                amount = 0.0
            if amount > 0:
                total_usd += amount
                rewarded.append({
                    "id":       sub.get("uuid") or sub.get("id"),
                    "title":    sub.get("title", ""),
                    "severity": sub.get("severity", ""),
                    "program":  sub.get("program", {}).get("name", "") if isinstance(sub.get("program"), dict) else "",
                    "reward":   amount,
                })

        self._respond(200, {
            "total_usd":    round(total_usd, 2),
            "rewarded_count": len(rewarded),
            "total_resolved": len(submissions),
            "submissions":  rewarded,
            "error":        result.get("error"),
        })

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
            if base == "/submit":
                self._handle_submit(data)
            else:
                self._respond(404, {"error": "unknown_path", "path": base})
        except Exception as e:
            log.error("POST %s error: %s", base, e)
            self._respond(500, {"error": str(e)})

    def _handle_submit(self, data: dict):
        title        = data.get("title", "")
        description  = data.get("description", "")
        severity     = data.get("severity", "P3").upper()
        program_id   = data.get("bounty_brief_id") or data.get("program_id", "")
        vuln_refs    = data.get("vulnerability_references", [])
        custom_fields= data.get("custom_fields", {})

        # Validation
        if not title:
            self._respond(400, {"error": "title required"})
            return
        if not description:
            self._respond(400, {"error": "description required"})
            return
        if severity not in VALID_SEVERITIES:
            self._respond(400, {"error": f"severity must be one of {sorted(VALID_SEVERITIES)}"})
            return
        if not program_id:
            self._respond(400, {"error": "bounty_brief_id required"})
            return

        body = {
            "submission": {
                "title":                    title,
                "description":              description,
                "severity":                 severity,
                "bounty_brief_id":          str(program_id),
                "vulnerability_references": vuln_refs if isinstance(vuln_refs, list) else [],
                "custom_fields":            custom_fields if isinstance(custom_fields, dict) else {},
            }
        }

        result = _bc("POST", "/submissions", body=body)
        sub    = result.get("submission") or result

        if "error" not in result:
            sub_id  = str(sub.get("uuid") or sub.get("id") or "")
            status  = sub.get("state") or sub.get("status") or "new"
            program = str(program_id)
            _log_submission(sub_id, title, severity, status, "0", program)
            log.info("submission created: id=%s title=%s severity=%s program=%s",
                     sub_id, title, severity, program)
            self._respond(201, sub)
        else:
            log.warning("submission failed: %s", result.get("error"))
            self._respond(400, result)

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
    server = HTTPServer(("0.0.0.0", PORT), BugcrowdHandler)
    log.info("Bugcrowd agent listening on port %d", PORT)
    log.info("API token: %s", "configured" if BUGCROWD_API_TOKEN else "NOT SET")
    try:
        while _running:
            server.handle_request()
    finally:
        server.server_close()
        log.info("Bugcrowd agent stopped")

if __name__ == "__main__":
    main()
