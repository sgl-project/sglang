#!/usr/bin/env python3
"""
FractalMesh OMEGA Titan — Online Course & Learning Platform
Port: 7880

Full-featured online course platform: create courses with modules and lessons,
sell them via Stripe, track student progress, issue completion certificates,
and gather reviews.

Author : Samuel James Hiotis | ABN 56 628 117 363
System : FractalMesh SaaS Platform
"""

# ---------------------------------------------------------------------------
# Vault loading — MUST run before any os.getenv calls
# ---------------------------------------------------------------------------
import os
from pathlib import Path

_ENV_FILE = Path.home() / ".secrets" / "fractal.env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

# ---------------------------------------------------------------------------
# stdlib imports
# ---------------------------------------------------------------------------
import base64
import hashlib
import hmac
import json
import secrets
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
AGENT_NAME = "fm_course_platform"
PORT = int(os.environ.get("COURSE_PLATFORM_PORT", "7880"))

STRIPE_SECRET_KEY  = os.environ.get("STRIPE_SECRET_KEY", "")
SENDGRID_API_KEY   = os.environ.get("SENDGRID_API_KEY", "")
SENDGRID_FROM_EMAIL = os.environ.get("SENDGRID_FROM_EMAIL", "noreply@fractalmesh.io")
ADMIN_SECRET       = os.environ.get("ADMIN_SECRET", "")

STRIPE_API_BASE   = "https://api.stripe.com/v1"
SENDGRID_API_BASE = "https://api.sendgrid.com/v3"

ROOT    = Path(os.environ.get("FRACTALMESH_HOME", Path.home() / "fmsaas"))
DB_PATH = ROOT / "database" / "sovereign.db"

START_TIME = time.time()

# ---------------------------------------------------------------------------
# Database bootstrap
# ---------------------------------------------------------------------------

def _db() -> sqlite3.Connection:
    con = sqlite3.connect(str(DB_PATH), timeout=15)
    con.execute("PRAGMA journal_mode=WAL")
    con.row_factory = sqlite3.Row
    return con


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = _db()
    con.executescript("""
        CREATE TABLE IF NOT EXISTS courses (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            slug             TEXT UNIQUE NOT NULL,
            title            TEXT NOT NULL,
            description      TEXT NOT NULL DEFAULT '',
            instructor       TEXT NOT NULL DEFAULT 'FractalMesh Team',
            price            REAL NOT NULL DEFAULT 0,
            currency         TEXT NOT NULL DEFAULT 'AUD',
            level            TEXT NOT NULL DEFAULT 'beginner',
            category         TEXT NOT NULL DEFAULT 'general',
            thumbnail_url    TEXT NOT NULL DEFAULT '',
            preview_video_url TEXT NOT NULL DEFAULT '',
            status           TEXT NOT NULL DEFAULT 'draft',
            duration_minutes INTEGER NOT NULL DEFAULT 0,
            student_count    INTEGER NOT NULL DEFAULT 0,
            rating           REAL NOT NULL DEFAULT 0,
            created_at       REAL NOT NULL,
            updated_at       REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS modules (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            course_id   INTEGER NOT NULL,
            title       TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            order_index INTEGER NOT NULL DEFAULT 0,
            created_at  REAL NOT NULL,
            FOREIGN KEY (course_id) REFERENCES courses(id)
        );

        CREATE TABLE IF NOT EXISTS lessons (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            module_id        INTEGER NOT NULL,
            title            TEXT NOT NULL,
            content          TEXT NOT NULL DEFAULT '',
            video_url        TEXT NOT NULL DEFAULT '',
            duration_minutes INTEGER NOT NULL DEFAULT 5,
            lesson_type      TEXT NOT NULL DEFAULT 'video',
            order_index      INTEGER NOT NULL DEFAULT 0,
            is_preview       INTEGER NOT NULL DEFAULT 0,
            created_at       REAL NOT NULL,
            FOREIGN KEY (module_id) REFERENCES modules(id)
        );

        CREATE TABLE IF NOT EXISTS enrollments (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            enrollment_id     TEXT UNIQUE NOT NULL,
            course_id         INTEGER NOT NULL,
            student_email     TEXT NOT NULL,
            student_name      TEXT NOT NULL DEFAULT '',
            stripe_payment_id TEXT NOT NULL DEFAULT '',
            amount_paid       REAL NOT NULL DEFAULT 0,
            status            TEXT NOT NULL DEFAULT 'active',
            enrolled_at       REAL NOT NULL,
            completed_at      REAL,
            certificate_id    TEXT NOT NULL DEFAULT '',
            FOREIGN KEY (course_id) REFERENCES courses(id)
        );

        CREATE TABLE IF NOT EXISTS progress (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            enrollment_id TEXT NOT NULL,
            lesson_id     INTEGER NOT NULL,
            completed     INTEGER NOT NULL DEFAULT 0,
            completed_at  REAL,
            time_spent    INTEGER NOT NULL DEFAULT 0,
            UNIQUE(enrollment_id, lesson_id),
            FOREIGN KEY (lesson_id) REFERENCES lessons(id)
        );

        CREATE TABLE IF NOT EXISTS certificates (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            certificate_id  TEXT UNIQUE NOT NULL,
            enrollment_id   TEXT NOT NULL,
            course_title    TEXT NOT NULL,
            student_name    TEXT NOT NULL,
            instructor      TEXT NOT NULL,
            completion_date TEXT NOT NULL,
            created_at      REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS reviews (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            course_id     INTEGER NOT NULL,
            student_email TEXT NOT NULL,
            rating        INTEGER NOT NULL,
            comment       TEXT NOT NULL DEFAULT '',
            created_at    REAL NOT NULL,
            UNIQUE(course_id, student_email),
            FOREIGN KEY (course_id) REFERENCES courses(id)
        );
    """)
    con.commit()
    con.close()
    print(f"[{AGENT_NAME}] DB initialised at {DB_PATH}", flush=True)


def seed_courses() -> None:
    """Seed starter courses if the courses table is empty."""
    con = _db()
    try:
        row = con.execute("SELECT COUNT(*) FROM courses").fetchone()
        if row[0] > 0:
            return

        now = time.time()

        # ------------------------------------------------------------------
        # Course 1: FractalMesh Developer Bootcamp
        # ------------------------------------------------------------------
        slug1 = "fractalmesh-developer-bootcamp"
        con.execute(
            """INSERT INTO courses
               (slug, title, description, instructor, price, currency, level,
                category, status, duration_minutes, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                slug1,
                "FractalMesh Developer Bootcamp",
                "Master the FractalMesh OMEGA Titan platform from scratch. "
                "Learn to build, deploy, and manage autonomous AI agents.",
                "FractalMesh Team",
                199.0, "AUD", "intermediate",
                "development", "published", 100, now, now,
            ),
        )
        cid1 = con.execute("SELECT id FROM courses WHERE slug=?", (slug1,)).fetchone()["id"]

        # Module 1
        con.execute(
            "INSERT INTO modules (course_id, title, description, order_index, created_at) VALUES (?,?,?,?,?)",
            (cid1, "Getting Started", "Set up your FractalMesh development environment.", 1, now),
        )
        mid1a = con.execute("SELECT id FROM modules WHERE course_id=? AND order_index=1", (cid1,)).fetchone()["id"]

        lessons_m1 = [
            ("Platform Overview", "Welcome to FractalMesh. In this lesson we walk through the overall architecture of OMEGA Titan.", "", 10, "video", 1, 1),
            ("Installation & Setup", "Step-by-step installation of all FractalMesh dependencies on Linux and macOS.", "", 15, "video", 2, 0),
            ("Your First Agent", "Build and run your very first autonomous FractalMesh agent end-to-end.", "", 20, "video", 3, 0),
        ]
        for title, content, video_url, dur, ltype, oidx, preview in lessons_m1:
            con.execute(
                """INSERT INTO lessons
                   (module_id, title, content, video_url, duration_minutes,
                    lesson_type, order_index, is_preview, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (mid1a, title, content, video_url, dur, ltype, oidx, preview, now),
            )

        # Module 2
        con.execute(
            "INSERT INTO modules (course_id, title, description, order_index, created_at) VALUES (?,?,?,?,?)",
            (cid1, "Core Agents", "Deep-dive into the most important FractalMesh agents.", 2, now),
        )
        mid1b = con.execute("SELECT id FROM modules WHERE course_id=? AND order_index=2", (cid1,)).fetchone()["id"]

        lessons_m2 = [
            ("MCP Router Deep-Dive", "Understand the MCP Router's routing logic, middleware hooks, and fault tolerance.", "", 30, "video", 1, 0),
            ("Database Layer", "Explore the sovereign SQLite WAL database pattern used across all agents.", "", 25, "video", 2, 0),
        ]
        for title, content, video_url, dur, ltype, oidx, preview in lessons_m2:
            con.execute(
                """INSERT INTO lessons
                   (module_id, title, content, video_url, duration_minutes,
                    lesson_type, order_index, is_preview, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (mid1b, title, content, video_url, dur, ltype, oidx, preview, now),
            )

        # Update duration
        con.execute(
            "UPDATE courses SET duration_minutes=100 WHERE id=?", (cid1,)
        )

        # ------------------------------------------------------------------
        # Course 2: AI Automation Fundamentals
        # ------------------------------------------------------------------
        slug2 = "ai-automation-fundamentals"
        con.execute(
            """INSERT INTO courses
               (slug, title, description, instructor, price, currency, level,
                category, status, duration_minutes, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                slug2,
                "AI Automation Fundamentals",
                "Learn the core principles of AI-driven automation. "
                "Perfect for beginners looking to leverage AI in their workflows.",
                "FractalMesh Team",
                99.0, "AUD", "beginner",
                "ai", "published", 25, now, now,
            ),
        )
        cid2 = con.execute("SELECT id FROM courses WHERE slug=?", (slug2,)).fetchone()["id"]

        con.execute(
            "INSERT INTO modules (course_id, title, description, order_index, created_at) VALUES (?,?,?,?,?)",
            (cid2, "Introduction", "Your first steps into AI Automation.", 1, now),
        )
        mid2a = con.execute("SELECT id FROM modules WHERE course_id=? AND order_index=1", (cid2,)).fetchone()["id"]

        lessons_intro = [
            ("What is AI Automation?", "Demystify AI automation: definitions, categories, and real-world examples.", "", 10, "video", 1, 1),
            ("Use Cases & Applications", "Survey the landscape of AI automation use cases across industries.", "", 15, "video", 2, 0),
        ]
        for title, content, video_url, dur, ltype, oidx, preview in lessons_intro:
            con.execute(
                """INSERT INTO lessons
                   (module_id, title, content, video_url, duration_minutes,
                    lesson_type, order_index, is_preview, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (mid2a, title, content, video_url, dur, ltype, oidx, preview, now),
            )

        con.commit()
        print(f"[{AGENT_NAME}] Seeded starter courses.", flush=True)
    finally:
        con.close()

# ---------------------------------------------------------------------------
# Stripe helpers
# ---------------------------------------------------------------------------

def _stripe_auth() -> str:
    creds = base64.b64encode(f"{STRIPE_SECRET_KEY}:".encode()).decode()
    return f"Basic {creds}"


def _stripe_post(path: str, data: dict) -> dict:
    url = STRIPE_API_BASE + path
    body = urllib.parse.urlencode(_build_flat(data)).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={
            "Authorization": _stripe_auth(),
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body_err = exc.read().decode(errors="replace")
        raise RuntimeError(f"Stripe POST {path} → {exc.code}: {body_err}") from exc


def _build_flat(d: dict, prefix: str = "") -> dict:
    """Flatten nested dict into Stripe-style dot/bracket notation for urlencode."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}[{k}]" if prefix else k
        if isinstance(v, dict):
            out.update(_build_flat(v, key))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    out.update(_build_flat(item, f"{key}[{i}]"))
                else:
                    out[f"{key}[{i}]"] = str(item)
        else:
            out[key] = str(v)
    return out


def stripe_create_payment_intent(amount_aud: float, description: str,
                                  payment_method_id: str,
                                  metadata: dict | None = None) -> dict:
    """Create and confirm a Stripe PaymentIntent in AUD cents."""
    data: dict = {
        "amount": int(round(amount_aud * 100)),
        "currency": "aud",
        "payment_method": payment_method_id,
        "confirm": "true",
        "description": description,
        "automatic_payment_methods[enabled]": "true",
        "automatic_payment_methods[allow_redirects]": "never",
    }
    if metadata:
        for mk, mv in metadata.items():
            data[f"metadata[{mk}]"] = str(mv)
    return _stripe_post("/payment_intents", data)

# ---------------------------------------------------------------------------
# SendGrid helpers
# ---------------------------------------------------------------------------

def _sendgrid_send(to_email: str, to_name: str, subject: str, html: str) -> None:
    if not SENDGRID_API_KEY:
        print(f"[{AGENT_NAME}] SendGrid key missing — skipping email to {to_email}", flush=True)
        return
    payload = json.dumps({
        "personalizations": [{"to": [{"email": to_email, "name": to_name}]}],
        "from": {"email": SENDGRID_FROM_EMAIL, "name": "FractalMesh Academy"},
        "subject": subject,
        "content": [{"type": "text/html", "value": html}],
    }).encode()
    req = urllib.request.Request(
        f"{SENDGRID_API_BASE}/mail/send",
        data=payload,
        headers={
            "Authorization": f"Bearer {SENDGRID_API_KEY}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20):
            pass
    except urllib.error.HTTPError as exc:
        print(f"[{AGENT_NAME}] SendGrid error {exc.code}: {exc.read().decode(errors='replace')}", flush=True)


def send_welcome_email(student_name: str, student_email: str,
                       course_title: str, enrollment_id: str) -> None:
    html = f"""
    <h2>Welcome to {course_title}, {student_name}!</h2>
    <p>You're now enrolled. Your enrollment ID is <strong>{enrollment_id}</strong>.</p>
    <p>Log in to the FractalMesh Academy portal and start learning today.</p>
    <p>— The FractalMesh Team</p>
    """
    _sendgrid_send(student_email, student_name,
                   f"Welcome to {course_title} — FractalMesh Academy", html)


def send_certificate_email(student_name: str, student_email: str,
                            course_title: str, certificate_id: str) -> None:
    html = f"""
    <h2>Congratulations, {student_name}!</h2>
    <p>You have successfully completed <strong>{course_title}</strong>.</p>
    <p>Your certificate ID is <strong>{certificate_id}</strong>.</p>
    <p>— The FractalMesh Team</p>
    """
    _sendgrid_send(student_email, student_name,
                   f"Your Certificate for {course_title} — FractalMesh Academy", html)

# ---------------------------------------------------------------------------
# Certificate generation
# ---------------------------------------------------------------------------

def _generate_certificate(enrollment_id: str, course_title: str,
                           student_name: str, instructor: str) -> str:
    """Create certificate, store it, return certificate_id."""
    raw = f"{enrollment_id}:{course_title}:{student_name}:{time.time()}"
    digest = hashlib.sha256(raw.encode()).hexdigest()
    cert_id = f"CERT-{digest[:10].upper()}"
    completion_date = time.strftime("%Y-%m-%d", time.gmtime())
    now = time.time()
    con = _db()
    try:
        con.execute(
            """INSERT OR IGNORE INTO certificates
               (certificate_id, enrollment_id, course_title, student_name,
                instructor, completion_date, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (cert_id, enrollment_id, course_title, student_name,
             instructor, completion_date, now),
        )
        con.execute(
            "UPDATE enrollments SET certificate_id=?, completed_at=? WHERE enrollment_id=?",
            (cert_id, now, enrollment_id),
        )
        con.commit()
    finally:
        con.close()
    return cert_id

# ---------------------------------------------------------------------------
# Background stats refresh
# ---------------------------------------------------------------------------

def _stats_refresh_loop() -> None:
    while True:
        time.sleep(3600)
        try:
            con = _db()
            # Update course ratings
            con.execute("""
                UPDATE courses SET rating = (
                    SELECT COALESCE(AVG(rating), 0) FROM reviews WHERE reviews.course_id = courses.id
                ), updated_at = ?
            """, (time.time(),))
            # Update student counts
            con.execute("""
                UPDATE courses SET student_count = (
                    SELECT COUNT(*) FROM enrollments
                    WHERE enrollments.course_id = courses.id
                    AND enrollments.status = 'active'
                ), updated_at = ?
            """, (time.time(),))
            con.commit()
            con.close()
            print(f"[{AGENT_NAME}] Stats refreshed.", flush=True)
        except Exception as exc:
            print(f"[{AGENT_NAME}] Stats refresh error: {exc}", flush=True)

# ---------------------------------------------------------------------------
# Admin auth helper
# ---------------------------------------------------------------------------

def _is_admin(handler: "CoursePlatformHandler") -> bool:
    auth = handler.headers.get("X-Admin-Secret", "")
    if not ADMIN_SECRET:
        return False
    return hmac.compare_digest(auth, ADMIN_SECRET)

# ---------------------------------------------------------------------------
# Request / response helpers
# ---------------------------------------------------------------------------

def _read_json(handler: "CoursePlatformHandler") -> dict:
    length = int(handler.headers.get("Content-Length", "0"))
    if length == 0:
        return {}
    return json.loads(handler.rfile.read(length))


def _send_json(handler: "CoursePlatformHandler", code: int, data: dict | list) -> None:
    body = json.dumps(data, indent=2).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def _err(handler: "CoursePlatformHandler", code: int, msg: str) -> None:
    _send_json(handler, code, {"error": msg})

# ---------------------------------------------------------------------------
# Course helpers
# ---------------------------------------------------------------------------

def _course_by_slug(slug: str, con: sqlite3.Connection) -> sqlite3.Row | None:
    return con.execute("SELECT * FROM courses WHERE slug=?", (slug,)).fetchone()


def _total_lessons(course_id: int, con: sqlite3.Connection) -> int:
    row = con.execute(
        """SELECT COUNT(*) FROM lessons l
           JOIN modules m ON m.id = l.module_id
           WHERE m.course_id=?""",
        (course_id,),
    ).fetchone()
    return row[0] if row else 0


def _completed_lessons(enrollment_id: str, course_id: int, con: sqlite3.Connection) -> int:
    row = con.execute(
        """SELECT COUNT(*) FROM progress p
           JOIN lessons l ON l.id = p.lesson_id
           JOIN modules m ON m.id = l.module_id
           WHERE p.enrollment_id=? AND m.course_id=? AND p.completed=1""",
        (enrollment_id, course_id),
    ).fetchone()
    return row[0] if row else 0


def _enrollment_progress(enrollment_id: str, con: sqlite3.Connection) -> dict:
    enr = con.execute(
        "SELECT * FROM enrollments WHERE enrollment_id=?", (enrollment_id,)
    ).fetchone()
    if not enr:
        return {}
    total = _total_lessons(enr["course_id"], con)
    done = _completed_lessons(enrollment_id, enr["course_id"], con)
    pct = round((done / total * 100) if total else 0, 1)
    prog_rows = con.execute(
        "SELECT lesson_id, completed, completed_at, time_spent FROM progress WHERE enrollment_id=?",
        (enrollment_id,),
    ).fetchall()
    return {
        "total_lessons": total,
        "completed_lessons": done,
        "completion_percentage": pct,
        "lesson_progress": [dict(r) for r in prog_rows],
    }

# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class CoursePlatformHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):  # suppress default access log
        pass

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Admin-Secret")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")
        qs = urllib.parse.parse_qs(parsed.query)

        # /health
        if path == "/health":
            self._handle_health()

        # /courses
        elif path == "/courses":
            self._handle_list_courses(qs)

        # /courses/{slug}
        elif path.startswith("/courses/") and path.count("/") == 2:
            slug = path.split("/")[2]
            if not slug:
                _err(self, 400, "Missing slug")
            else:
                self._handle_get_course(slug)

        # /courses/{slug}/reviews
        elif path.startswith("/courses/") and path.endswith("/reviews"):
            parts = path.split("/")
            if len(parts) == 4:
                self._handle_get_reviews(parts[2])
            else:
                _err(self, 404, "Not found")

        # /enrollments/{enrollment_id}
        elif path.startswith("/enrollments/") and path.count("/") == 2:
            eid = path.split("/")[2]
            self._handle_get_enrollment(eid)

        # /enrollments/{enrollment_id}/lessons/{lesson_id}
        elif path.startswith("/enrollments/") and "/lessons/" in path and not path.endswith("/complete"):
            parts = path.split("/")
            if len(parts) == 5 and parts[3] == "lessons":
                self._handle_get_lesson(parts[2], parts[4])
            else:
                _err(self, 404, "Not found")

        # /enrollments/{enrollment_id}/certificate
        elif path.startswith("/enrollments/") and path.endswith("/certificate"):
            parts = path.split("/")
            if len(parts) == 4:
                self._handle_get_certificate(parts[2])
            else:
                _err(self, 404, "Not found")

        # /students  (admin)
        elif path == "/students":
            if not _is_admin(self):
                _err(self, 403, "Forbidden")
            else:
                self._handle_list_students()

        # /analytics  (admin)
        elif path == "/analytics":
            if not _is_admin(self):
                _err(self, 403, "Forbidden")
            else:
                self._handle_analytics()

        else:
            _err(self, 404, "Not found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")

        # POST /courses  (admin)
        if path == "/courses":
            if not _is_admin(self):
                _err(self, 403, "Forbidden")
            else:
                self._handle_create_course()

        # POST /courses/{slug}/publish  (admin)
        elif path.startswith("/courses/") and path.endswith("/publish"):
            if not _is_admin(self):
                _err(self, 403, "Forbidden")
            else:
                slug = path.split("/")[2]
                self._handle_publish_course(slug)

        # POST /courses/{slug}/enroll/free
        elif path.startswith("/courses/") and path.endswith("/enroll/free"):
            slug = path.split("/")[2]
            self._handle_free_enroll(slug)

        # POST /courses/{slug}/enroll
        elif path.startswith("/courses/") and path.endswith("/enroll"):
            slug = path.split("/")[2]
            self._handle_enroll(slug)

        # POST /courses/{slug}/reviews
        elif path.startswith("/courses/") and path.endswith("/reviews"):
            parts = path.split("/")
            if len(parts) == 4:
                self._handle_post_review(parts[2])
            else:
                _err(self, 404, "Not found")

        # POST /enrollments/{enrollment_id}/lessons/{lesson_id}/complete
        elif path.startswith("/enrollments/") and path.endswith("/complete"):
            parts = path.split("/")
            if len(parts) == 6 and parts[3] == "lessons":
                self._handle_complete_lesson(parts[2], parts[4])
            else:
                _err(self, 404, "Not found")

        else:
            _err(self, 404, "Not found")

    # -----------------------------------------------------------------------
    # GET handlers
    # -----------------------------------------------------------------------

    def _handle_health(self):
        con = _db()
        try:
            course_count = con.execute("SELECT COUNT(*) FROM courses WHERE status='published'").fetchone()[0]
            student_count = con.execute("SELECT COUNT(DISTINCT student_email) FROM enrollments").fetchone()[0]
            enrollment_count = con.execute("SELECT COUNT(*) FROM enrollments WHERE status='active'").fetchone()[0]
        finally:
            con.close()
        _send_json(self, 200, {
            "status": "ok",
            "agent": AGENT_NAME,
            "port": PORT,
            "uptime_seconds": round(time.time() - START_TIME, 1),
            "published_courses": course_count,
            "unique_students": student_count,
            "active_enrollments": enrollment_count,
        })

    def _handle_list_courses(self, qs: dict):
        category = qs.get("category", [None])[0]
        level = qs.get("level", [None])[0]
        limit = int(qs.get("limit", ["50"])[0])

        con = _db()
        try:
            where_parts = ["c.status='published'"]
            params: list = []
            if category:
                where_parts.append("c.category=?")
                params.append(category)
            if level:
                where_parts.append("c.level=?")
                params.append(level)
            where = " AND ".join(where_parts)

            rows = con.execute(
                f"SELECT * FROM courses c WHERE {where} ORDER BY c.created_at DESC LIMIT ?",
                params + [limit],
            ).fetchall()

            courses = []
            for row in rows:
                c = dict(row)
                c["module_count"] = con.execute(
                    "SELECT COUNT(*) FROM modules WHERE course_id=?", (c["id"],)
                ).fetchone()[0]
                c["lesson_count"] = con.execute(
                    """SELECT COUNT(*) FROM lessons l
                       JOIN modules m ON m.id=l.module_id WHERE m.course_id=?""",
                    (c["id"],),
                ).fetchone()[0]
                courses.append(c)
        finally:
            con.close()
        _send_json(self, 200, {"courses": courses, "count": len(courses)})

    def _handle_get_course(self, slug: str):
        con = _db()
        try:
            course = _course_by_slug(slug, con)
            if not course:
                _err(self, 404, "Course not found")
                return
            c = dict(course)

            mods = con.execute(
                "SELECT * FROM modules WHERE course_id=? ORDER BY order_index", (c["id"],)
            ).fetchall()

            modules_out = []
            for mod in mods:
                m = dict(mod)
                lessons = con.execute(
                    "SELECT * FROM lessons WHERE module_id=? ORDER BY order_index", (mod["id"],)
                ).fetchall()
                lessons_out = []
                for les in lessons:
                    l = dict(les)
                    # Only return full content for preview lessons; hide otherwise
                    if not l["is_preview"]:
                        l["content"] = ""
                        l["video_url"] = ""
                    lessons_out.append(l)
                m["lessons"] = lessons_out
                modules_out.append(m)

            c["modules"] = modules_out
            c["module_count"] = len(modules_out)
            c["lesson_count"] = sum(len(m["lessons"]) for m in modules_out)
            c["review_count"] = con.execute(
                "SELECT COUNT(*) FROM reviews WHERE course_id=?", (c["id"],)
            ).fetchone()[0]
        finally:
            con.close()
        _send_json(self, 200, c)

    def _handle_get_reviews(self, slug: str):
        con = _db()
        try:
            course = _course_by_slug(slug, con)
            if not course:
                _err(self, 404, "Course not found")
                return
            rows = con.execute(
                "SELECT * FROM reviews WHERE course_id=? ORDER BY created_at DESC",
                (course["id"],),
            ).fetchall()
        finally:
            con.close()
        _send_json(self, 200, {"reviews": [dict(r) for r in rows], "count": len(rows)})

    def _handle_get_enrollment(self, enrollment_id: str):
        con = _db()
        try:
            enr = con.execute(
                "SELECT * FROM enrollments WHERE enrollment_id=?", (enrollment_id,)
            ).fetchone()
            if not enr:
                _err(self, 404, "Enrollment not found")
                return
            result = dict(enr)
            result["progress"] = _enrollment_progress(enrollment_id, con)
            course = con.execute("SELECT * FROM courses WHERE id=?", (enr["course_id"],)).fetchone()
            result["course"] = dict(course) if course else {}
        finally:
            con.close()
        _send_json(self, 200, result)

    def _handle_get_lesson(self, enrollment_id: str, lesson_id: str):
        con = _db()
        try:
            enr = con.execute(
                "SELECT * FROM enrollments WHERE enrollment_id=? AND status='active'",
                (enrollment_id,),
            ).fetchone()
            if not enr:
                _err(self, 403, "No active enrollment")
                return
            lesson = con.execute("SELECT * FROM lessons WHERE id=?", (lesson_id,)).fetchone()
            if not lesson:
                _err(self, 404, "Lesson not found")
                return
            # Verify lesson belongs to enrolled course
            mod = con.execute("SELECT * FROM modules WHERE id=?", (lesson["module_id"],)).fetchone()
            if not mod or mod["course_id"] != enr["course_id"]:
                _err(self, 403, "Lesson not in enrolled course")
                return
            prog = con.execute(
                "SELECT * FROM progress WHERE enrollment_id=? AND lesson_id=?",
                (enrollment_id, lesson_id),
            ).fetchone()
            result = dict(lesson)
            result["progress"] = dict(prog) if prog else {"completed": 0, "time_spent": 0}
        finally:
            con.close()
        _send_json(self, 200, result)

    def _handle_get_certificate(self, enrollment_id: str):
        con = _db()
        try:
            enr = con.execute(
                "SELECT * FROM enrollments WHERE enrollment_id=?", (enrollment_id,)
            ).fetchone()
            if not enr:
                _err(self, 404, "Enrollment not found")
                return

            total = _total_lessons(enr["course_id"], con)
            done = _completed_lessons(enrollment_id, enr["course_id"], con)

            if total == 0 or done < total:
                _err(self, 400, f"Course not complete ({done}/{total} lessons done)")
                return

            cert_id = enr["certificate_id"]
            if not cert_id:
                # Generate on-demand
                course = con.execute("SELECT * FROM courses WHERE id=?", (enr["course_id"],)).fetchone()
                cert_id = _generate_certificate(
                    enrollment_id,
                    course["title"],
                    enr["student_name"],
                    course["instructor"],
                )
                # Reload enrollment to reflect updated cert
                enr = con.execute(
                    "SELECT * FROM enrollments WHERE enrollment_id=?", (enrollment_id,)
                ).fetchone()

            cert = con.execute(
                "SELECT * FROM certificates WHERE certificate_id=?", (cert_id,)
            ).fetchone()
            if not cert:
                _err(self, 404, "Certificate record not found")
                return
            result = dict(cert)
            result["html_template"] = (
                f"<div class='certificate'>"
                f"<h1>Certificate of Completion</h1>"
                f"<h2>{result['course_title']}</h2>"
                f"<p>This certifies that <strong>{result['student_name']}</strong> "
                f"has successfully completed this course.</p>"
                f"<p>Instructor: {result['instructor']}</p>"
                f"<p>Date: {result['completion_date']}</p>"
                f"<p>Certificate ID: {result['certificate_id']}</p>"
                f"</div>"
            )
        finally:
            con.close()
        _send_json(self, 200, result)

    def _handle_list_students(self):
        con = _db()
        try:
            rows = con.execute(
                """SELECT e.student_email, e.student_name, COUNT(*) as course_count,
                          MIN(e.enrolled_at) as first_enrolled,
                          SUM(e.amount_paid) as total_spent
                   FROM enrollments e
                   WHERE e.status='active'
                   GROUP BY e.student_email
                   ORDER BY first_enrolled DESC"""
            ).fetchall()
        finally:
            con.close()
        _send_json(self, 200, {"students": [dict(r) for r in rows], "count": len(rows)})

    def _handle_analytics(self):
        con = _db()
        try:
            # Revenue by course
            revenue_rows = con.execute(
                """SELECT c.title, c.slug, SUM(e.amount_paid) as total_revenue,
                          COUNT(e.id) as enrollment_count
                   FROM enrollments e
                   JOIN courses c ON c.id = e.course_id
                   WHERE e.status='active'
                   GROUP BY c.id
                   ORDER BY total_revenue DESC"""
            ).fetchall()

            # Completion rates
            completion_rows = con.execute(
                """SELECT c.title, c.slug,
                          COUNT(DISTINCT e.enrollment_id) as enrolled,
                          COUNT(DISTINCT CASE WHEN e.completed_at IS NOT NULL THEN e.enrollment_id END) as completed
                   FROM courses c
                   LEFT JOIN enrollments e ON e.course_id = c.id AND e.status='active'
                   GROUP BY c.id"""
            ).fetchall()

            # Average ratings
            rating_rows = con.execute(
                """SELECT c.title, c.slug, ROUND(AVG(r.rating),2) as avg_rating, COUNT(r.id) as review_count
                   FROM courses c
                   LEFT JOIN reviews r ON r.course_id = c.id
                   GROUP BY c.id
                   HAVING review_count > 0
                   ORDER BY avg_rating DESC"""
            ).fetchall()

            # Popular lessons (most completions)
            popular_rows = con.execute(
                """SELECT l.title as lesson_title, c.title as course_title,
                          COUNT(p.id) as completion_count
                   FROM progress p
                   JOIN lessons l ON l.id = p.lesson_id
                   JOIN modules m ON m.id = l.module_id
                   JOIN courses c ON c.id = m.course_id
                   WHERE p.completed=1
                   GROUP BY p.lesson_id
                   ORDER BY completion_count DESC
                   LIMIT 10"""
            ).fetchall()

            total_revenue = con.execute(
                "SELECT COALESCE(SUM(amount_paid),0) FROM enrollments WHERE status='active'"
            ).fetchone()[0]
        finally:
            con.close()

        _send_json(self, 200, {
            "total_revenue_aud": round(total_revenue, 2),
            "revenue_by_course": [dict(r) for r in revenue_rows],
            "completion_rates": [
                {
                    "title": r["title"],
                    "slug": r["slug"],
                    "enrolled": r["enrolled"],
                    "completed": r["completed"],
                    "rate_pct": round((r["completed"] / r["enrolled"] * 100) if r["enrolled"] else 0, 1),
                }
                for r in completion_rows
            ],
            "ratings": [dict(r) for r in rating_rows],
            "popular_lessons": [dict(r) for r in popular_rows],
        })

    # -----------------------------------------------------------------------
    # POST handlers
    # -----------------------------------------------------------------------

    def _handle_enroll(self, slug: str):
        body = _read_json(self)
        student_email = body.get("student_email", "").strip()
        student_name  = body.get("student_name", "").strip()
        payment_method_id = body.get("payment_method_id", "").strip()

        if not student_email or not student_name:
            _err(self, 400, "student_email and student_name are required")
            return
        if not payment_method_id:
            _err(self, 400, "payment_method_id is required")
            return

        con = _db()
        try:
            course = _course_by_slug(slug, con)
            if not course:
                _err(self, 404, "Course not found")
                return
            if course["status"] != "published":
                _err(self, 400, "Course is not published")
                return
            if course["price"] == 0:
                _err(self, 400, "This is a free course. Use /enroll/free")
                return

            # Check duplicate enrollment
            existing = con.execute(
                "SELECT id FROM enrollments WHERE course_id=? AND student_email=? AND status='active'",
                (course["id"], student_email),
            ).fetchone()
            if existing:
                _err(self, 409, "Already enrolled")
                return

            # Stripe PaymentIntent
            try:
                pi = stripe_create_payment_intent(
                    course["price"],
                    f"Enrolment: {course['title']}",
                    payment_method_id,
                    {"course_slug": slug, "student_email": student_email},
                )
            except RuntimeError as exc:
                _err(self, 402, f"Payment failed: {exc}")
                return

            enrollment_id = f"ENR-{secrets.token_hex(8).upper()}"
            now = time.time()
            con.execute(
                """INSERT INTO enrollments
                   (enrollment_id, course_id, student_email, student_name,
                    stripe_payment_id, amount_paid, status, enrolled_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (enrollment_id, course["id"], student_email, student_name,
                 pi.get("id", ""), course["price"], "active", now),
            )
            con.execute(
                "UPDATE courses SET student_count=student_count+1, updated_at=? WHERE id=?",
                (now, course["id"]),
            )
            con.commit()
        finally:
            con.close()

        send_welcome_email(student_name, student_email, course["title"], enrollment_id)
        _send_json(self, 201, {
            "enrollment_id": enrollment_id,
            "client_secret": pi.get("client_secret", ""),
            "status": pi.get("status", ""),
            "message": f"Enrolled in {course['title']}",
        })

    def _handle_free_enroll(self, slug: str):
        body = _read_json(self)
        student_email = body.get("student_email", "").strip()
        student_name  = body.get("student_name", "").strip()

        if not student_email or not student_name:
            _err(self, 400, "student_email and student_name are required")
            return

        con = _db()
        try:
            course = _course_by_slug(slug, con)
            if not course:
                _err(self, 404, "Course not found")
                return
            if course["status"] != "published":
                _err(self, 400, "Course is not published")
                return
            if course["price"] > 0:
                _err(self, 400, "This course requires payment. Use /enroll")
                return

            existing = con.execute(
                "SELECT id FROM enrollments WHERE course_id=? AND student_email=? AND status='active'",
                (course["id"], student_email),
            ).fetchone()
            if existing:
                _err(self, 409, "Already enrolled")
                return

            enrollment_id = f"ENR-{secrets.token_hex(8).upper()}"
            now = time.time()
            con.execute(
                """INSERT INTO enrollments
                   (enrollment_id, course_id, student_email, student_name,
                    stripe_payment_id, amount_paid, status, enrolled_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (enrollment_id, course["id"], student_email, student_name,
                 "", 0.0, "active", now),
            )
            con.execute(
                "UPDATE courses SET student_count=student_count+1, updated_at=? WHERE id=?",
                (now, course["id"]),
            )
            con.commit()
        finally:
            con.close()

        send_welcome_email(student_name, student_email, course["title"], enrollment_id)
        _send_json(self, 201, {
            "enrollment_id": enrollment_id,
            "message": f"Enrolled in {course['title']} (free)",
        })

    def _handle_complete_lesson(self, enrollment_id: str, lesson_id: str):
        body = _read_json(self)
        time_spent = int(body.get("time_spent", 0))

        con = _db()
        try:
            enr = con.execute(
                "SELECT * FROM enrollments WHERE enrollment_id=? AND status='active'",
                (enrollment_id,),
            ).fetchone()
            if not enr:
                _err(self, 403, "No active enrollment")
                return

            lesson = con.execute("SELECT * FROM lessons WHERE id=?", (lesson_id,)).fetchone()
            if not lesson:
                _err(self, 404, "Lesson not found")
                return

            mod = con.execute("SELECT * FROM modules WHERE id=?", (lesson["module_id"],)).fetchone()
            if not mod or mod["course_id"] != enr["course_id"]:
                _err(self, 403, "Lesson not in enrolled course")
                return

            now = time.time()
            con.execute(
                """INSERT INTO progress (enrollment_id, lesson_id, completed, completed_at, time_spent)
                   VALUES (?,?,1,?,?)
                   ON CONFLICT(enrollment_id, lesson_id)
                   DO UPDATE SET completed=1, completed_at=?, time_spent=time_spent+?""",
                (enrollment_id, lesson_id, now, time_spent, now, time_spent),
            )
            con.commit()

            total = _total_lessons(enr["course_id"], con)
            done = _completed_lessons(enrollment_id, enr["course_id"], con)
            pct = round((done / total * 100) if total else 0, 1)

            cert_id = None
            if total > 0 and done >= total:
                course = con.execute(
                    "SELECT * FROM courses WHERE id=?", (enr["course_id"],)
                ).fetchone()
                cert_id = enr["certificate_id"] or _generate_certificate(
                    enrollment_id,
                    course["title"],
                    enr["student_name"],
                    course["instructor"],
                )
                if cert_id and not enr["certificate_id"]:
                    # Email the certificate
                    send_certificate_email(
                        enr["student_name"], enr["student_email"],
                        course["title"], cert_id,
                    )
        finally:
            con.close()

        resp: dict = {
            "enrollment_id": enrollment_id,
            "lesson_id": int(lesson_id),
            "completed": True,
            "completion_percentage": pct,
            "completed_lessons": done,
            "total_lessons": total,
        }
        if cert_id:
            resp["certificate_id"] = cert_id
            resp["message"] = "Congratulations! Course complete. Certificate issued."
        _send_json(self, 200, resp)

    def _handle_post_review(self, slug: str):
        body = _read_json(self)
        student_email = body.get("student_email", "").strip()
        rating = int(body.get("rating", 0))
        comment = body.get("comment", "").strip()

        if not student_email:
            _err(self, 400, "student_email is required")
            return
        if not (1 <= rating <= 5):
            _err(self, 400, "rating must be between 1 and 5")
            return

        con = _db()
        try:
            course = _course_by_slug(slug, con)
            if not course:
                _err(self, 404, "Course not found")
                return

            # Require completed enrollment
            enr = con.execute(
                """SELECT * FROM enrollments
                   WHERE course_id=? AND student_email=? AND status='active'
                   AND completed_at IS NOT NULL""",
                (course["id"], student_email),
            ).fetchone()
            if not enr:
                _err(self, 403, "Must complete the course before reviewing")
                return

            now = time.time()
            try:
                con.execute(
                    """INSERT INTO reviews (course_id, student_email, rating, comment, created_at)
                       VALUES (?,?,?,?,?)""",
                    (course["id"], student_email, rating, comment, now),
                )
            except sqlite3.IntegrityError:
                # Update existing review
                con.execute(
                    """UPDATE reviews SET rating=?, comment=?, created_at=?
                       WHERE course_id=? AND student_email=?""",
                    (rating, comment, now, course["id"], student_email),
                )
            # Refresh course rating
            avg = con.execute(
                "SELECT AVG(rating) FROM reviews WHERE course_id=?", (course["id"],)
            ).fetchone()[0] or 0
            con.execute(
                "UPDATE courses SET rating=?, updated_at=? WHERE id=?",
                (round(avg, 2), now, course["id"]),
            )
            con.commit()
        finally:
            con.close()
        _send_json(self, 201, {"message": "Review submitted", "rating": rating})

    def _handle_create_course(self):
        body = _read_json(self)
        required = ["title", "slug", "description", "price"]
        for field in required:
            if field not in body:
                _err(self, 400, f"Missing field: {field}")
                return

        now = time.time()
        slug = body["slug"].strip().lower()
        con = _db()
        try:
            existing = con.execute("SELECT id FROM courses WHERE slug=?", (slug,)).fetchone()
            if existing:
                _err(self, 409, f"Slug '{slug}' already exists")
                return

            con.execute(
                """INSERT INTO courses
                   (slug, title, description, instructor, price, currency, level,
                    category, thumbnail_url, preview_video_url, status,
                    duration_minutes, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    slug,
                    body["title"].strip(),
                    body["description"].strip(),
                    body.get("instructor", "FractalMesh Team"),
                    float(body["price"]),
                    body.get("currency", "AUD"),
                    body.get("level", "beginner"),
                    body.get("category", "general"),
                    body.get("thumbnail_url", ""),
                    body.get("preview_video_url", ""),
                    "draft",
                    int(body.get("duration_minutes", 0)),
                    now, now,
                ),
            )
            course_id = con.execute("SELECT id FROM courses WHERE slug=?", (slug,)).fetchone()["id"]

            # Optional nested modules/lessons
            for m_idx, mod in enumerate(body.get("modules", [])):
                con.execute(
                    "INSERT INTO modules (course_id, title, description, order_index, created_at) VALUES (?,?,?,?,?)",
                    (course_id, mod.get("title", f"Module {m_idx+1}"), mod.get("description", ""), m_idx + 1, now),
                )
                mod_id = con.execute(
                    "SELECT id FROM modules WHERE course_id=? AND order_index=?", (course_id, m_idx + 1)
                ).fetchone()["id"]
                for l_idx, les in enumerate(mod.get("lessons", [])):
                    con.execute(
                        """INSERT INTO lessons
                           (module_id, title, content, video_url, duration_minutes,
                            lesson_type, order_index, is_preview, created_at)
                           VALUES (?,?,?,?,?,?,?,?,?)""",
                        (
                            mod_id,
                            les.get("title", f"Lesson {l_idx+1}"),
                            les.get("content", ""),
                            les.get("video_url", ""),
                            int(les.get("duration_minutes", 5)),
                            les.get("lesson_type", "video"),
                            l_idx + 1,
                            1 if les.get("is_preview") else 0,
                            now,
                        ),
                    )
            con.commit()
        finally:
            con.close()

        _send_json(self, 201, {
            "message": "Course created",
            "course_id": course_id,
            "slug": slug,
            "status": "draft",
        })

    def _handle_publish_course(self, slug: str):
        con = _db()
        try:
            course = _course_by_slug(slug, con)
            if not course:
                _err(self, 404, "Course not found")
                return
            if course["status"] == "published":
                _send_json(self, 200, {"message": "Already published", "slug": slug})
                return
            con.execute(
                "UPDATE courses SET status='published', updated_at=? WHERE slug=?",
                (time.time(), slug),
            )
            con.commit()
        finally:
            con.close()
        _send_json(self, 200, {"message": "Course published", "slug": slug})

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    init_db()
    seed_courses()

    t = threading.Thread(target=_stats_refresh_loop, daemon=True)
    t.start()

    server = HTTPServer(("0.0.0.0", PORT), CoursePlatformHandler)
    print(f"[{AGENT_NAME}] Listening on port {PORT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"[{AGENT_NAME}] Shutting down.", flush=True)
        server.server_close()


if __name__ == "__main__":
    main()
