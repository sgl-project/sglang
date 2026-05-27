#!/usr/bin/env python3
"""
fm_hr_manager.py — HR & Team Management Agent (Port 7897)
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

PORT         = int(os.getenv("HR_MANAGER_PORT", "7897"))
SG_KEY       = os.getenv("SENDGRID_API_KEY", "")
SG_FROM      = os.getenv("SENDGRID_FROM_EMAIL", "noreply@fractalmesh.ai")
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")
BIZ_NAME     = os.getenv("BUSINESS_NAME", "IronVision Nexus")

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
        CREATE TABLE IF NOT EXISTS employees (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            emp_id       TEXT UNIQUE NOT NULL,
            first_name   TEXT NOT NULL,
            last_name    TEXT NOT NULL,
            email        TEXT UNIQUE NOT NULL,
            phone        TEXT,
            department   TEXT,
            job_title    TEXT,
            employment_type TEXT DEFAULT 'full_time',
            status       TEXT DEFAULT 'active',
            start_date   REAL,
            end_date     REAL,
            salary       REAL DEFAULT 0,
            currency     TEXT DEFAULT 'AUD',
            manager_id   TEXT,
            avatar_url   TEXT,
            created_at   REAL NOT NULL,
            updated_at   REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS leave_requests (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            leave_id     TEXT UNIQUE NOT NULL,
            emp_id       TEXT NOT NULL,
            leave_type   TEXT NOT NULL,
            start_date   REAL NOT NULL,
            end_date     REAL NOT NULL,
            days         REAL NOT NULL,
            reason       TEXT,
            status       TEXT DEFAULT 'pending',
            approved_by  TEXT,
            created_at   REAL NOT NULL,
            updated_at   REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS performance_reviews (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            review_id    TEXT UNIQUE NOT NULL,
            emp_id       TEXT NOT NULL,
            reviewer_id  TEXT NOT NULL,
            period       TEXT NOT NULL,
            rating       INTEGER NOT NULL,
            strengths    TEXT,
            improvements TEXT,
            goals        TEXT,
            status       TEXT DEFAULT 'draft',
            created_at   REAL NOT NULL,
            updated_at   REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS time_logs (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            log_id       TEXT UNIQUE NOT NULL,
            emp_id       TEXT NOT NULL,
            clock_in     REAL NOT NULL,
            clock_out    REAL,
            hours        REAL DEFAULT 0,
            project      TEXT,
            notes        TEXT,
            created_at   REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS departments (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            dept_id      TEXT UNIQUE NOT NULL,
            name         TEXT UNIQUE NOT NULL,
            head_emp_id  TEXT,
            budget       REAL DEFAULT 0,
            created_at   REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_employees_dept   ON employees(department);
        CREATE INDEX IF NOT EXISTS idx_leave_emp        ON leave_requests(emp_id);
        CREATE INDEX IF NOT EXISTS idx_timelogs_emp     ON time_logs(emp_id);
    """)
    con.commit()
    con.close()

def _j(data, status=200):
    body = json.dumps(data, default=str).encode()
    return status, body

def _err(msg, code=400):
    return _j({"error": msg}, code)

def _admin(headers):
    h = headers.get("X-Admin-Secret", "")
    return not ADMIN_SECRET or hmac.compare_digest(h, ADMIN_SECRET)

def _send_email(to, subject, body):
    if not SG_KEY:
        return
    payload = json.dumps({
        "personalizations": [{"to": [{"email": to}]}],
        "from": {"email": SG_FROM, "name": BIZ_NAME},
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

def _workdays_between(start, end):
    days = 0
    cur = start
    while cur <= end:
        import datetime
        dt = datetime.datetime.fromtimestamp(cur)
        if dt.weekday() < 5:
            days += 1
        cur += 86400
    return days

class HRHandler(BaseHTTPRequestHandler):
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
        self.send_header("Access-Control-Allow-Headers", "Content-Type,X-Admin-Secret")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        parts = parsed.path.strip("/").split("/")
        qs = parse_qs(parsed.query)
        try:
            code, body = self._get(parts, qs)
        except Exception as e:
            code, body = _err(str(e), 500)
        self._send(code, body)

    def do_POST(self):
        parts = self.path.strip("/").split("/")
        try:
            data = self._read_body()
            code, body = self._post(parts, data)
        except Exception as e:
            code, body = _err(str(e), 500)
        self._send(code, body)

    def do_PUT(self):
        parts = self.path.strip("/").split("/")
        try:
            data = self._read_body()
            code, body = self._put(parts, data)
        except Exception as e:
            code, body = _err(str(e), 500)
        self._send(code, body)

    def _get(self, p, qs):
        con = _db()
        try:
            if p == ["health"]:
                return _j({"status": "ok", "port": PORT, "agent": "fm_hr_manager"})

            if p == ["employees"]:
                dept = qs.get("department", [None])[0]
                status = qs.get("status", ["active"])[0]
                if dept:
                    rows = con.execute(
                        "SELECT * FROM employees WHERE status=? AND department=? ORDER BY last_name",
                        (status, dept)
                    ).fetchall()
                else:
                    rows = con.execute(
                        "SELECT * FROM employees WHERE status=? ORDER BY last_name", (status,)
                    ).fetchall()
                return _j([dict(r) for r in rows])

            if len(p) == 2 and p[0] == "employees":
                row = con.execute("SELECT * FROM employees WHERE emp_id=?", (p[1],)).fetchone()
                if not row:
                    return _err("Employee not found", 404)
                emp = dict(row)
                emp["leave_balance"] = self._leave_balance(con, p[1])
                return _j(emp)

            if p == ["departments"]:
                rows = con.execute("SELECT * FROM departments ORDER BY name").fetchall()
                return _j([dict(r) for r in rows])

            if len(p) == 3 and p[0] == "employees" and p[2] == "leave":
                rows = con.execute(
                    "SELECT * FROM leave_requests WHERE emp_id=? ORDER BY created_at DESC", (p[1],)
                ).fetchall()
                return _j([dict(r) for r in rows])

            if len(p) == 3 and p[0] == "employees" and p[2] == "timelog":
                rows = con.execute(
                    "SELECT * FROM time_logs WHERE emp_id=? ORDER BY clock_in DESC LIMIT 50", (p[1],)
                ).fetchall()
                return _j([dict(r) for r in rows])

            if len(p) == 3 and p[0] == "employees" and p[2] == "reviews":
                rows = con.execute(
                    "SELECT * FROM performance_reviews WHERE emp_id=? ORDER BY created_at DESC", (p[1],)
                ).fetchall()
                return _j([dict(r) for r in rows])

            if p == ["leave", "pending"]:
                if not _admin(self.headers):
                    return _err("Unauthorized", 403)
                rows = con.execute(
                    "SELECT lr.*, e.first_name, e.last_name FROM leave_requests lr "
                    "JOIN employees e ON lr.emp_id=e.emp_id WHERE lr.status='pending' ORDER BY lr.created_at"
                ).fetchall()
                return _j([dict(r) for r in rows])

            if p == ["dashboard"]:
                if not _admin(self.headers):
                    return _err("Unauthorized", 403)
                total = con.execute("SELECT COUNT(*) FROM employees WHERE status='active'").fetchone()[0]
                by_dept = con.execute(
                    "SELECT department, COUNT(*) as cnt FROM employees WHERE status='active' GROUP BY department"
                ).fetchall()
                pending_leave = con.execute("SELECT COUNT(*) FROM leave_requests WHERE status='pending'").fetchone()[0]
                total_salary = con.execute("SELECT COALESCE(SUM(salary),0) FROM employees WHERE status='active'").fetchone()[0]
                return _j({
                    "active_employees": total,
                    "by_department": {r["department"]: r["cnt"] for r in by_dept},
                    "pending_leave_requests": pending_leave,
                    "total_annual_salary_aud": round(total_salary, 2),
                })

            return _err("Not found", 404)
        finally:
            con.close()

    def _leave_balance(self, con, emp_id):
        row = con.execute("SELECT start_date FROM employees WHERE emp_id=?", (emp_id,)).fetchone()
        if not row or not row["start_date"]:
            return {"annual": 20, "sick": 10, "used_annual": 0, "used_sick": 0}
        used = con.execute(
            "SELECT leave_type, COALESCE(SUM(days),0) as total FROM leave_requests "
            "WHERE emp_id=? AND status='approved' AND start_date >= ? GROUP BY leave_type",
            (emp_id, time.time() - 365 * 86400)
        ).fetchall()
        used_map = {r["leave_type"]: r["total"] for r in used}
        return {
            "annual": 20 - used_map.get("annual", 0),
            "sick": 10 - used_map.get("sick", 0),
            "used_annual": used_map.get("annual", 0),
            "used_sick": used_map.get("sick", 0),
        }

    def _post(self, p, data):
        con = _db()
        now = time.time()
        try:
            if p == ["employees"]:
                if not _admin(self.headers):
                    return _err("Unauthorized", 403)
                email = data.get("email", "")
                if not email:
                    return _err("email required")
                eid = "emp_" + secrets.token_hex(6)
                con.execute(
                    "INSERT INTO employees(emp_id,first_name,last_name,email,phone,department,job_title,"
                    "employment_type,status,start_date,salary,currency,manager_id,created_at,updated_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (eid, data.get("first_name",""), data.get("last_name",""), email,
                     data.get("phone"), data.get("department"), data.get("job_title"),
                     data.get("employment_type","full_time"), "active",
                     data.get("start_date", now), data.get("salary", 0),
                     data.get("currency","AUD"), data.get("manager_id"), now, now)
                )
                con.commit()
                threading.Thread(target=_send_email, args=(
                    email, f"Welcome to {BIZ_NAME}!",
                    f"<p>Hi {data.get('first_name','')}, welcome aboard at {BIZ_NAME}!</p>"
                ), daemon=True).start()
                return _j({"emp_id": eid, "email": email}, 201)

            if p == ["departments"]:
                if not _admin(self.headers):
                    return _err("Unauthorized", 403)
                did = "dept_" + secrets.token_hex(6)
                con.execute(
                    "INSERT INTO departments(dept_id,name,head_emp_id,budget,created_at) VALUES(?,?,?,?,?)",
                    (did, data.get("name",""), data.get("head_emp_id"), data.get("budget", 0), now)
                )
                con.commit()
                return _j({"dept_id": did}, 201)

            if len(p) == 3 and p[0] == "employees" and p[2] == "leave":
                lid = "leave_" + secrets.token_hex(6)
                start = data.get("start_date", now)
                end = data.get("end_date", now)
                days = _workdays_between(start, end)
                con.execute(
                    "INSERT INTO leave_requests(leave_id,emp_id,leave_type,start_date,end_date,days,reason,created_at,updated_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?)",
                    (lid, p[1], data.get("leave_type","annual"), start, end, days, data.get("reason"), now, now)
                )
                con.commit()
                return _j({"leave_id": lid, "days": days, "status": "pending"}, 201)

            if len(p) == 3 and p[0] == "leave" and p[2] in ("approve","reject"):
                if not _admin(self.headers):
                    return _err("Unauthorized", 403)
                status = "approved" if p[2] == "approve" else "rejected"
                con.execute(
                    "UPDATE leave_requests SET status=?, approved_by=?, updated_at=? WHERE leave_id=?",
                    (status, data.get("approved_by","admin"), now, p[1])
                )
                con.commit()
                return _j({"leave_id": p[1], "status": status})

            if len(p) == 3 and p[0] == "employees" and p[2] == "clockin":
                log_id = "tl_" + secrets.token_hex(6)
                con.execute(
                    "INSERT INTO time_logs(log_id,emp_id,clock_in,project,notes,created_at) VALUES(?,?,?,?,?,?)",
                    (log_id, p[1], now, data.get("project"), data.get("notes"), now)
                )
                con.commit()
                return _j({"log_id": log_id, "clock_in": now})

            if len(p) == 3 and p[0] == "employees" and p[2] == "clockout":
                row = con.execute(
                    "SELECT * FROM time_logs WHERE emp_id=? AND clock_out IS NULL ORDER BY clock_in DESC LIMIT 1",
                    (p[1],)
                ).fetchone()
                if not row:
                    return _err("No open clock-in found", 404)
                hours = (now - row["clock_in"]) / 3600
                con.execute(
                    "UPDATE time_logs SET clock_out=?, hours=? WHERE log_id=?",
                    (now, round(hours, 2), row["log_id"])
                )
                con.commit()
                return _j({"log_id": row["log_id"], "hours": round(hours, 2)})

            if len(p) == 3 and p[0] == "employees" and p[2] == "review":
                if not _admin(self.headers):
                    return _err("Unauthorized", 403)
                rid = "rev_" + secrets.token_hex(6)
                con.execute(
                    "INSERT INTO performance_reviews(review_id,emp_id,reviewer_id,period,rating,strengths,improvements,goals,status,created_at,updated_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (rid, p[1], data.get("reviewer_id","admin"), data.get("period",""),
                     data.get("rating", 3), data.get("strengths"), data.get("improvements"),
                     data.get("goals"), data.get("status","draft"), now, now)
                )
                con.commit()
                return _j({"review_id": rid}, 201)

            return _err("Not found", 404)
        finally:
            con.close()

    def _put(self, p, data):
        con = _db()
        now = time.time()
        try:
            if len(p) == 2 and p[0] == "employees":
                fields = []
                vals = []
                for f in ["first_name","last_name","phone","department","job_title",
                          "employment_type","status","salary","manager_id"]:
                    if f in data:
                        fields.append(f"{f}=?")
                        vals.append(data[f])
                if not fields:
                    return _err("No fields to update")
                vals.extend([now, p[1]])
                con.execute(
                    f"UPDATE employees SET {','.join(fields)}, updated_at=? WHERE emp_id=?", vals
                )
                con.commit()
                return _j({"emp_id": p[1], "updated": True})
            return _err("Not found", 404)
        finally:
            con.close()


def run():
    init_db()
    server = HTTPServer(("0.0.0.0", PORT), HRHandler)
    print(f"[fm_hr_manager] listening on port {PORT}")
    server.serve_forever()


if __name__ == "__main__":
    run()
