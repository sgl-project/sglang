#!/usr/bin/env python3
"""
fm_project_manager.py — Project & Task Management Agent (Port 7899)
FractalMesh OMEGA Titan | Samuel James Hiotis | ABN 56 628 117 363
Credentials sourced from ~/.secrets/fractal.env — never hardcoded.
"""
import os
import json
import sqlite3
import time
import hashlib
import hmac
import secrets
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
import urllib.request
import urllib.error

# ── vault ─────────────────────────────────────────────────────────────────────
_ENV_FILE = Path.home() / ".secrets" / "fractal.env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

PORT         = int(os.getenv("PROJECT_MANAGER_PORT", "7899"))
SG_KEY       = os.getenv("SENDGRID_API_KEY", "")
SG_FROM      = os.getenv("SENDGRID_FROM_EMAIL", "noreply@fractalmesh.ai")
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")

ROOT = Path(os.getenv("FRACTALMESH_HOME", str(Path.home() / "fmsaas")))
DB   = ROOT / "database" / "sovereign.db"
ROOT.mkdir(parents=True, exist_ok=True)
DB.parent.mkdir(parents=True, exist_ok=True)

def _db():
    con = sqlite3.connect(str(DB), check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")
    con.row_factory = sqlite3.Row
    return con

def init_db():
    con = _db()
    con.executescript("""
        CREATE TABLE IF NOT EXISTS projects (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id   TEXT UNIQUE NOT NULL,
            name         TEXT NOT NULL,
            description  TEXT,
            status       TEXT NOT NULL DEFAULT 'active',
            priority     TEXT NOT NULL DEFAULT 'medium',
            owner_id     TEXT,
            team_members TEXT NOT NULL DEFAULT '[]',
            start_date   REAL,
            due_date     REAL,
            completed_at REAL,
            tags         TEXT NOT NULL DEFAULT '[]',
            created_at   REAL NOT NULL,
            updated_at   REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS pm_tasks (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id      TEXT UNIQUE NOT NULL,
            project_id   TEXT NOT NULL,
            title        TEXT NOT NULL,
            description  TEXT,
            status       TEXT NOT NULL DEFAULT 'todo',
            priority     TEXT NOT NULL DEFAULT 'medium',
            assignee_id  TEXT,
            reporter_id  TEXT,
            parent_id    TEXT,
            due_date     REAL,
            completed_at REAL,
            estimated_h  REAL DEFAULT 0,
            logged_h     REAL DEFAULT 0,
            tags         TEXT NOT NULL DEFAULT '[]',
            created_at   REAL NOT NULL,
            updated_at   REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS task_comments (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            comment_id   TEXT UNIQUE NOT NULL,
            task_id      TEXT NOT NULL,
            author_id    TEXT NOT NULL,
            content      TEXT NOT NULL,
            created_at   REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS time_entries (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_id     TEXT UNIQUE NOT NULL,
            task_id      TEXT NOT NULL,
            user_id      TEXT NOT NULL,
            hours        REAL NOT NULL,
            description  TEXT,
            date         REAL NOT NULL,
            created_at   REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS milestones (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            milestone_id TEXT UNIQUE NOT NULL,
            project_id   TEXT NOT NULL,
            title        TEXT NOT NULL,
            due_date     REAL NOT NULL,
            completed    INTEGER NOT NULL DEFAULT 0,
            completed_at REAL,
            created_at   REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_tasks_project  ON pm_tasks(project_id);
        CREATE INDEX IF NOT EXISTS idx_tasks_assignee ON pm_tasks(assignee_id);
        CREATE INDEX IF NOT EXISTS idx_tasks_status   ON pm_tasks(status);
    """)
    con.commit()
    con.close()

def _j(data, status=200):
    return status, json.dumps(data, default=str).encode()

def _err(msg, code=400):
    return _j({"error": msg}, code)

def _admin(h):
    v = h.get("X-Admin-Secret", "")
    return not ADMIN_SECRET or hmac.compare_digest(v, ADMIN_SECRET)

def _send_email(to, subject, body):
    if not SG_KEY:
        return
    payload = json.dumps({
        "personalizations": [{"to": [{"email": to}]}],
        "from": {"email": SG_FROM},
        "subject": subject,
        "content": [{"type": "text/html", "value": body}],
    }).encode()
    req = urllib.request.Request("https://api.sendgrid.com/v3/mail/send", data=payload)
    req.add_header("Authorization", f"Bearer {SG_KEY}")
    req.add_header("Content-Type", "application/json")
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass

def _overdue_daemon():
    while True:
        time.sleep(3600)
        try:
            con = _db()
            now = time.time()
            overdue = con.execute(
                "SELECT t.task_id, t.title, t.assignee_id, t.due_date, p.name as project_name "
                "FROM pm_tasks t JOIN projects p ON t.project_id=p.project_id "
                "WHERE t.status NOT IN ('done','cancelled') AND t.due_date < ? AND t.due_date IS NOT NULL",
                (now,)
            ).fetchall()
            for t in overdue:
                if t["assignee_id"] and "@" in t["assignee_id"]:
                    threading.Thread(target=_send_email, args=(
                        t["assignee_id"],
                        f"Overdue Task: {t['title']}",
                        f"<p>Task <strong>{t['title']}</strong> in project <strong>{t['project_name']}</strong> is overdue.</p>"
                    ), daemon=True).start()
            con.close()
        except Exception:
            pass

threading.Thread(target=_overdue_daemon, daemon=True).start()

class PMHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _read_body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n)) if n else {}

    def _send(self, code, body):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,PUT,PATCH,DELETE,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type,X-Admin-Secret,X-User-Id")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        p = parsed.path.strip("/").split("/")
        qs = parse_qs(parsed.query)
        try:
            code, body = self._get(p, qs)
        except Exception as e:
            code, body = _err(str(e), 500)
        self._send(code, body)

    def do_POST(self):
        p = self.path.strip("/").split("/")
        try:
            data = self._read_body()
            code, body = self._post(p, data)
        except Exception as e:
            code, body = _err(str(e), 500)
        self._send(code, body)

    def do_PUT(self):
        p = self.path.strip("/").split("/")
        try:
            data = self._read_body()
            code, body = self._put(p, data)
        except Exception as e:
            code, body = _err(str(e), 500)
        self._send(code, body)

    def _get(self, p, qs):
        con = _db()
        try:
            if p == ["health"]:
                return _j({"status": "ok", "port": PORT, "agent": "fm_project_manager"})

            if p == ["projects"]:
                status = qs.get("status", [None])[0]
                if status:
                    rows = con.execute("SELECT * FROM projects WHERE status=? ORDER BY name", (status,)).fetchall()
                else:
                    rows = con.execute("SELECT * FROM projects ORDER BY updated_at DESC").fetchall()
                return _j([dict(r) for r in rows])

            if len(p) == 2 and p[0] == "projects":
                row = con.execute("SELECT * FROM projects WHERE project_id=?", (p[1],)).fetchone()
                if not row:
                    return _err("Project not found", 404)
                proj = dict(row)
                tasks = con.execute(
                    "SELECT task_id,title,status,priority,assignee_id,due_date FROM pm_tasks WHERE project_id=? ORDER BY status",
                    (p[1],)
                ).fetchall()
                milestones = con.execute(
                    "SELECT * FROM milestones WHERE project_id=? ORDER BY due_date", (p[1],)
                ).fetchall()
                total = len(tasks)
                done = sum(1 for t in tasks if t["status"] == "done")
                proj["tasks"] = [dict(t) for t in tasks]
                proj["milestones"] = [dict(m) for m in milestones]
                proj["progress_pct"] = round((done / total * 100) if total > 0 else 0, 1)
                return _j(proj)

            if len(p) == 3 and p[0] == "projects" and p[2] == "tasks":
                status = qs.get("status", [None])[0]
                assignee = qs.get("assignee", [None])[0]
                q = "SELECT * FROM pm_tasks WHERE project_id=?"
                vals = [p[1]]
                if status:
                    q += " AND status=?"; vals.append(status)
                if assignee:
                    q += " AND assignee_id=?"; vals.append(assignee)
                q += " ORDER BY priority DESC, due_date"
                rows = con.execute(q, vals).fetchall()
                return _j([dict(r) for r in rows])

            if len(p) == 2 and p[0] == "tasks":
                row = con.execute("SELECT * FROM pm_tasks WHERE task_id=?", (p[1],)).fetchone()
                if not row:
                    return _err("Task not found", 404)
                task = dict(row)
                comments = con.execute(
                    "SELECT * FROM task_comments WHERE task_id=? ORDER BY created_at", (p[1],)
                ).fetchall()
                time_entries = con.execute(
                    "SELECT * FROM time_entries WHERE task_id=? ORDER BY date DESC", (p[1],)
                ).fetchall()
                task["comments"] = [dict(c) for c in comments]
                task["time_entries"] = [dict(e) for e in time_entries]
                return _j(task)

            if p == ["my-tasks"]:
                uid = self.headers.get("X-User-Id", "")
                if not uid:
                    return _err("X-User-Id header required")
                rows = con.execute(
                    "SELECT t.*, p.name as project_name FROM pm_tasks t "
                    "JOIN projects p ON t.project_id=p.project_id "
                    "WHERE t.assignee_id=? AND t.status NOT IN ('done','cancelled') ORDER BY t.due_date",
                    (uid,)
                ).fetchall()
                return _j([dict(r) for r in rows])

            if p == ["dashboard"]:
                now = time.time()
                total_projects = con.execute("SELECT COUNT(*) FROM projects WHERE status='active'").fetchone()[0]
                overdue_tasks = con.execute(
                    "SELECT COUNT(*) FROM pm_tasks WHERE status NOT IN ('done','cancelled') AND due_date < ?", (now,)
                ).fetchone()[0]
                tasks_by_status = con.execute(
                    "SELECT status, COUNT(*) as cnt FROM pm_tasks GROUP BY status"
                ).fetchall()
                return _j({
                    "active_projects": total_projects,
                    "overdue_tasks": overdue_tasks,
                    "tasks_by_status": {r["status"]: r["cnt"] for r in tasks_by_status},
                })

            return _err("Not found", 404)
        finally:
            con.close()

    def _post(self, p, data):
        con = _db()
        now = time.time()
        try:
            if p == ["projects"]:
                pid = "proj_" + secrets.token_hex(8)
                con.execute(
                    "INSERT INTO projects(project_id,name,description,status,priority,owner_id,team_members,"
                    "start_date,due_date,tags,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (pid, data.get("name",""), data.get("description"),
                     data.get("status","active"), data.get("priority","medium"),
                     data.get("owner_id"), json.dumps(data.get("team_members",[])),
                     data.get("start_date", now), data.get("due_date"),
                     json.dumps(data.get("tags",[])), now, now)
                )
                con.commit()
                return _j({"project_id": pid}, 201)

            if len(p) == 3 and p[0] == "projects" and p[2] == "tasks":
                tid = "task_" + secrets.token_hex(8)
                con.execute(
                    "INSERT INTO pm_tasks(task_id,project_id,title,description,status,priority,"
                    "assignee_id,reporter_id,parent_id,due_date,estimated_h,tags,created_at,updated_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (tid, p[1], data.get("title",""), data.get("description"),
                     data.get("status","todo"), data.get("priority","medium"),
                     data.get("assignee_id"), data.get("reporter_id"), data.get("parent_id"),
                     data.get("due_date"), data.get("estimated_h",0),
                     json.dumps(data.get("tags",[])), now, now)
                )
                con.commit()
                if data.get("assignee_id") and "@" in data.get("assignee_id",""):
                    threading.Thread(target=_send_email, args=(
                        data["assignee_id"], f"New Task Assigned: {data.get('title','')}",
                        f"<p>You have been assigned task: <strong>{data.get('title','')}</strong></p>"
                    ), daemon=True).start()
                return _j({"task_id": tid}, 201)

            if len(p) == 3 and p[0] == "tasks" and p[2] == "comments":
                cid = "cmt_" + secrets.token_hex(8)
                con.execute(
                    "INSERT INTO task_comments(comment_id,task_id,author_id,content,created_at) VALUES(?,?,?,?,?)",
                    (cid, p[1], data.get("author_id",""), data.get("content",""), now)
                )
                con.commit()
                return _j({"comment_id": cid}, 201)

            if len(p) == 3 and p[0] == "tasks" and p[2] == "time":
                eid = "te_" + secrets.token_hex(8)
                hours = data.get("hours", 0)
                con.execute(
                    "INSERT INTO time_entries(entry_id,task_id,user_id,hours,description,date,created_at) VALUES(?,?,?,?,?,?,?)",
                    (eid, p[1], data.get("user_id",""), hours, data.get("description"), data.get("date", now), now)
                )
                con.execute("UPDATE pm_tasks SET logged_h=logged_h+?, updated_at=? WHERE task_id=?", (hours, now, p[1]))
                con.commit()
                return _j({"entry_id": eid}, 201)

            if len(p) == 3 and p[0] == "projects" and p[2] == "milestones":
                mid = "ms_" + secrets.token_hex(8)
                con.execute(
                    "INSERT INTO milestones(milestone_id,project_id,title,due_date,created_at) VALUES(?,?,?,?,?)",
                    (mid, p[1], data.get("title",""), data.get("due_date", now), now)
                )
                con.commit()
                return _j({"milestone_id": mid}, 201)

            return _err("Not found", 404)
        finally:
            con.close()

    def _put(self, p, data):
        con = _db()
        now = time.time()
        try:
            if len(p) == 2 and p[0] == "tasks":
                allowed = ["title","description","status","priority","assignee_id","due_date","estimated_h","tags"]
                fields, vals = [], []
                for f in allowed:
                    if f in data:
                        fields.append(f"{f}=?")
                        vals.append(json.dumps(data[f]) if f == "tags" else data[f])
                if not fields:
                    return _err("No fields to update")
                if data.get("status") == "done":
                    fields.append("completed_at=?"); vals.append(now)
                vals.extend([now, p[1]])
                con.execute(f"UPDATE pm_tasks SET {','.join(fields)}, updated_at=? WHERE task_id=?", vals)
                con.commit()
                return _j({"task_id": p[1], "updated": True})

            if len(p) == 2 and p[0] == "projects":
                allowed = ["name","description","status","priority","due_date","team_members"]
                fields, vals = [], []
                for f in allowed:
                    if f in data:
                        fields.append(f"{f}=?")
                        vals.append(json.dumps(data[f]) if f == "team_members" else data[f])
                if not fields:
                    return _err("No fields to update")
                vals.extend([now, p[1]])
                con.execute(f"UPDATE projects SET {','.join(fields)}, updated_at=? WHERE project_id=?", vals)
                con.commit()
                return _j({"project_id": p[1], "updated": True})

            return _err("Not found", 404)
        finally:
            con.close()


def run():
    init_db()
    server = HTTPServer(("0.0.0.0", PORT), PMHandler)
    print(f"[fm_project_manager] listening on port {PORT}")
    server.serve_forever()


if __name__ == "__main__":
    run()
