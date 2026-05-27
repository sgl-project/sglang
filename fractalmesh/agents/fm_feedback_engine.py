#!/usr/bin/env python3
"""
fm_feedback_engine.py — FractalMesh OMEGA Titan Customer Feedback & Survey Engine
Port: 7884

Customer feedback collection and analysis system. Create surveys, collect NPS
scores, analyse sentiment, and track customer satisfaction over time.

Samuel James Hiotis | ABN 56 628 117 363
"""

import os
import json
import sqlite3
import time
import hashlib
import hmac
import threading
import secrets
import statistics
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# ---------------------------------------------------------------------------
# Vault loading
# ---------------------------------------------------------------------------
_ENV_FILE = Path.home() / ".secrets" / "fractal.env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PORT              = int(os.environ.get("FEEDBACK_ENGINE_PORT", "7884"))
SENDGRID_API_KEY  = os.environ.get("SENDGRID_API_KEY", "")
SENDGRID_FROM     = os.environ.get("SENDGRID_FROM_EMAIL", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ADMIN_SECRET      = os.environ.get("ADMIN_SECRET", "")

ROOT     = Path.home() / "fmsaas"
DB_PATH  = ROOT / "database" / "sovereign.db"
LOG_PATH = ROOT / "logs" / "feedback_engine.log"

for _d in (DB_PATH.parent, LOG_PATH.parent):
    _d.mkdir(parents=True, exist_ok=True)

START_TIME = time.time()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_log_lock = threading.Lock()

def _log(level: str, msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    line = f"[{ts}] [{level}] {msg}\n"
    with _log_lock:
        try:
            with open(LOG_PATH, "a") as fh:
                fh.write(line)
        except OSError:
            pass

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=10, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn

def _init_db() -> None:
    conn = _get_conn()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS surveys (
                id              INTEGER PRIMARY KEY,
                survey_id       TEXT UNIQUE NOT NULL,
                title           TEXT NOT NULL,
                description     TEXT,
                survey_type     TEXT DEFAULT 'custom',
                status          TEXT DEFAULT 'draft',
                anonymous       INTEGER DEFAULT 1,
                response_count  INTEGER DEFAULT 0,
                avg_score       REAL,
                created_at      REAL,
                updated_at      REAL,
                closes_at       REAL
            );

            CREATE TABLE IF NOT EXISTS questions (
                id              INTEGER PRIMARY KEY,
                survey_id       TEXT NOT NULL,
                question_text   TEXT NOT NULL,
                question_type   TEXT NOT NULL,
                options         TEXT,
                required        INTEGER DEFAULT 1,
                order_index     INTEGER,
                created_at      REAL
            );

            CREATE TABLE IF NOT EXISTS responses (
                id               INTEGER PRIMARY KEY,
                survey_id        TEXT NOT NULL,
                respondent_email TEXT,
                respondent_name  TEXT,
                response_token   TEXT UNIQUE,
                submitted_at     REAL,
                ip_hash          TEXT,
                metadata         TEXT
            );

            CREATE TABLE IF NOT EXISTS answers (
                id              INTEGER PRIMARY KEY,
                response_id     INTEGER NOT NULL,
                question_id     INTEGER NOT NULL,
                answer_text     TEXT,
                answer_numeric  REAL,
                answer_options  TEXT
            );

            CREATE TABLE IF NOT EXISTS nps_scores (
                id              INTEGER PRIMARY KEY,
                score           INTEGER NOT NULL,
                comment         TEXT,
                customer_email  TEXT,
                context         TEXT,
                category        TEXT,
                sentiment       TEXT,
                sentiment_score REAL,
                created_at      REAL
            );

            CREATE TABLE IF NOT EXISTS feedback_items (
                id              INTEGER PRIMARY KEY,
                source          TEXT,
                content         TEXT NOT NULL,
                customer_email  TEXT,
                sentiment       TEXT,
                sentiment_score REAL,
                category        TEXT,
                priority        TEXT DEFAULT 'normal',
                status          TEXT DEFAULT 'open',
                resolved_at     REAL,
                created_at      REAL
            );
        """)
        conn.commit()
        _log("INFO", "Database initialised")
    finally:
        conn.close()

# ---------------------------------------------------------------------------
# Sentiment analysis
# ---------------------------------------------------------------------------
_POSITIVE_WORDS = {"great", "love", "excellent", "amazing", "perfect",
                   "helpful", "good", "fast"}
_NEGATIVE_WORDS = {"bad", "terrible", "awful", "slow", "broken",
                   "frustrated", "poor", "useless"}

def _analyse_sentiment(text: str) -> tuple[str, float]:
    """Return (label, score) where score is -1.0 to 1.0."""
    if not text or not text.strip():
        return "neutral", 0.0

    if ANTHROPIC_API_KEY:
        try:
            prompt = (
                f"Classify sentiment as positive/neutral/negative and score "
                f"-1.0 to 1.0: {text}"
            )
            payload = json.dumps({
                "model": "claude-haiku-4-5",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": prompt}],
            }).encode()
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=payload,
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read().decode())
            reply = body.get("content", [{}])[0].get("text", "").lower()
            # Extract score — look for a float pattern
            score = 0.0
            for token in reply.split():
                token = token.strip(".,;:()")
                try:
                    val = float(token)
                    if -1.0 <= val <= 1.0:
                        score = val
                        break
                except ValueError:
                    pass
            if "positive" in reply:
                label = "positive"
            elif "negative" in reply:
                label = "negative"
            else:
                label = "neutral"
            return label, round(score, 3)
        except Exception as exc:
            _log("WARN", f"Anthropic sentiment failed: {exc}; using fallback")

    # Fallback word-list scoring
    words = text.lower().split()
    pos = sum(1 for w in words if w in _POSITIVE_WORDS)
    neg = sum(1 for w in words if w in _NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return "neutral", 0.0
    score = round((pos - neg) / total, 3)
    if score > 0.1:
        label = "positive"
    elif score < -0.1:
        label = "negative"
    else:
        label = "neutral"
    return label, score

# ---------------------------------------------------------------------------
# NPS calculation
# ---------------------------------------------------------------------------
def _calc_nps(scores: list[int]) -> float:
    """Return NPS score (-100 to 100) rounded to 1 decimal."""
    if not scores:
        return 0.0
    n = len(scores)
    promoters  = sum(1 for s in scores if s >= 9)
    detractors = sum(1 for s in scores if s <= 6)
    return round(((promoters / n) - (detractors / n)) * 100, 1)

# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------
def _seed_surveys() -> None:
    conn = _get_conn()
    try:
        row = conn.execute("SELECT COUNT(*) as cnt FROM surveys").fetchone()
        if row["cnt"] > 0:
            return

        now = time.time()

        # Survey 1 — Product Satisfaction Survey
        sid1 = "survey_product_" + secrets.token_hex(6)
        conn.execute(
            """INSERT INTO surveys
               (survey_id, title, description, survey_type, status, anonymous,
                created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (sid1, "Product Satisfaction Survey",
             "Help us understand how satisfied you are with our product.",
             "custom", "active", 1, now, now),
        )
        q_data_1 = [
            ("How likely are you to recommend us? (0-10)", "nps",   None, 1, 1),
            ("What do you like most?",                     "text",  None, 1, 2),
            ("What could be improved?",                    "text",  None, 0, 3),
            ("Would you recommend us to others?",          "yes_no",None, 1, 4),
        ]
        for qtext, qtype, opts, req, idx in q_data_1:
            conn.execute(
                """INSERT INTO questions
                   (survey_id, question_text, question_type, options, required,
                    order_index, created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (sid1, qtext, qtype, json.dumps(opts) if opts else None,
                 req, idx, now),
            )

        # Survey 2 — Post-Purchase NPS
        sid2 = "survey_nps_" + secrets.token_hex(6)
        conn.execute(
            """INSERT INTO surveys
               (survey_id, title, description, survey_type, status, anonymous,
                created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (sid2, "Post-Purchase NPS",
             "Quick NPS survey after your recent purchase.",
             "nps", "active", 1, now, now),
        )
        q_data_2 = [
            ("How likely are you to recommend us to a friend? (0-10)", "nps",  None, 1, 1),
            ("Any additional comments?",                               "text", None, 0, 2),
        ]
        for qtext, qtype, opts, req, idx in q_data_2:
            conn.execute(
                """INSERT INTO questions
                   (survey_id, question_text, question_type, options, required,
                    order_index, created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (sid2, qtext, qtype, json.dumps(opts) if opts else None,
                 req, idx, now),
            )

        conn.commit()
        _log("INFO", "Seeded default surveys")
    finally:
        conn.close()

# ---------------------------------------------------------------------------
# Background maintenance thread
# ---------------------------------------------------------------------------
def _maintenance_loop() -> None:
    while True:
        try:
            _run_maintenance()
        except Exception as exc:
            _log("ERROR", f"Maintenance error: {exc}")
        time.sleep(3600)

def _run_maintenance() -> None:
    conn = _get_conn()
    try:
        now = time.time()

        # Close surveys past closes_at
        conn.execute(
            """UPDATE surveys SET status='closed', updated_at=?
               WHERE status='active' AND closes_at IS NOT NULL AND closes_at < ?""",
            (now, now),
        )

        # Sentiment analysis on unanalysed feedback_items
        rows = conn.execute(
            "SELECT id, content FROM feedback_items WHERE sentiment IS NULL"
        ).fetchall()
        for row in rows:
            label, score = _analyse_sentiment(row["content"] or "")
            conn.execute(
                "UPDATE feedback_items SET sentiment=?, sentiment_score=? WHERE id=?",
                (label, score, row["id"]),
            )

        # Recompute avg_score for active surveys
        surveys = conn.execute(
            "SELECT survey_id FROM surveys WHERE status='active'"
        ).fetchall()
        for sv in surveys:
            sid = sv["survey_id"]
            numeric_answers = conn.execute(
                """SELECT a.answer_numeric FROM answers a
                   JOIN responses r ON a.response_id = r.id
                   JOIN questions q ON a.question_id = q.id
                   WHERE r.survey_id=? AND q.question_type IN ('rating','nps')
                     AND a.answer_numeric IS NOT NULL""",
                (sid,),
            ).fetchall()
            if numeric_answers:
                vals = [x["answer_numeric"] for x in numeric_answers]
                avg = round(statistics.mean(vals), 2)
            else:
                avg = None
            rc = conn.execute(
                "SELECT COUNT(*) as cnt FROM responses WHERE survey_id=?", (sid,)
            ).fetchone()["cnt"]
            conn.execute(
                "UPDATE surveys SET avg_score=?, response_count=?, updated_at=? WHERE survey_id=?",
                (avg, rc, now, sid),
            )

        conn.commit()
        _log("INFO", "Maintenance run complete")
    finally:
        conn.close()

# ---------------------------------------------------------------------------
# Email helper
# ---------------------------------------------------------------------------
def _send_email(to_email: str, subject: str, body_html: str) -> bool:
    if not SENDGRID_API_KEY or not SENDGRID_FROM:
        _log("WARN", "SendGrid not configured, skipping email")
        return False
    payload = json.dumps({
        "personalizations": [{"to": [{"email": to_email}]}],
        "from": {"email": SENDGRID_FROM},
        "subject": subject,
        "content": [{"type": "text/html", "value": body_html}],
    }).encode()
    try:
        req = urllib.request.Request(
            "https://api.sendgrid.com/v3/mail/send",
            data=payload,
            headers={
                "Authorization": f"Bearer {SENDGRID_API_KEY}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status in (200, 202)
    except Exception as exc:
        _log("ERROR", f"SendGrid error: {exc}")
        return False

# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------
def _is_admin(headers) -> bool:
    auth = headers.get("X-Admin-Secret") or headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        auth = auth[7:]
    if not ADMIN_SECRET:
        return False
    return hmac.compare_digest(auth.strip(), ADMIN_SECRET)

# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------
class FeedbackHandler(BaseHTTPRequestHandler):
    server_version = "FractalMesh-Feedback/1.0"

    def log_message(self, fmt, *args):  # suppress default stdout logs
        _log("HTTP", fmt % args)

    # ---- helpers -----------------------------------------------------------
    def _send_json(self, data: dict | list, status: int = 200) -> None:
        body = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, msg: str, status: int = 400) -> None:
        self._send_json({"error": msg}, status)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode())

    def _ip_hash(self) -> str:
        addr = self.client_address[0] if self.client_address else "unknown"
        return hashlib.sha256(addr.encode()).hexdigest()[:16]

    def _parse_path(self):
        """Return (path_parts, query_dict)."""
        raw = self.path
        if "?" in raw:
            path_str, qs = raw.split("?", 1)
        else:
            path_str, qs = raw, ""
        parts = [p for p in path_str.strip("/").split("/") if p]
        query = {}
        for pair in qs.split("&"):
            if "=" in pair:
                k, v = pair.split("=", 1)
                query[k] = v
        return parts, query

    # ---- GET ---------------------------------------------------------------
    def do_GET(self):
        parts, query = self._parse_path()

        if not parts or parts[0] == "health":
            self._handle_health()
        elif parts[0] == "surveys":
            if len(parts) == 1:
                self._handle_list_surveys(query)
            elif len(parts) == 2:
                self._handle_get_survey(parts[1])
            elif len(parts) == 3 and parts[2] == "results":
                self._handle_survey_results(parts[1])
            elif len(parts) == 3 and parts[2] == "responses":
                self._handle_survey_responses(parts[1])
            else:
                self._send_error("Not found", 404)
        elif parts[0] == "nps":
            self._handle_nps_dashboard()
        elif parts[0] == "feedback":
            if len(parts) == 1:
                self._handle_list_feedback(query)
            elif len(parts) == 2:
                self._handle_get_feedback(parts[1])
            else:
                self._send_error("Not found", 404)
        elif parts[0] == "analytics":
            self._handle_analytics()
        else:
            self._send_error("Not found", 404)

    # ---- POST --------------------------------------------------------------
    def do_POST(self):
        parts, _ = self._parse_path()

        if not parts:
            self._send_error("Not found", 404)
        elif parts[0] == "surveys":
            if len(parts) == 1:
                self._handle_create_survey()
            elif len(parts) == 3 and parts[2] == "respond":
                self._handle_submit_response(parts[1])
            elif len(parts) == 3 and parts[2] == "send":
                self._handle_send_survey(parts[1])
            else:
                self._send_error("Not found", 404)
        elif parts[0] == "nps":
            self._handle_submit_nps()
        elif parts[0] == "feedback":
            self._handle_submit_feedback()
        else:
            self._send_error("Not found", 404)

    # ---- PUT ---------------------------------------------------------------
    def do_PUT(self):
        parts, _ = self._parse_path()
        if len(parts) == 3 and parts[0] == "feedback" and parts[2] == "resolve":
            self._handle_resolve_feedback(parts[1])
        else:
            self._send_error("Not found", 404)

    # =====================================================================
    # GET handlers
    # =====================================================================

    def _handle_health(self):
        conn = _get_conn()
        try:
            rc = conn.execute("SELECT COUNT(*) as cnt FROM responses").fetchone()["cnt"]
            nps_rows = conn.execute(
                "SELECT score FROM nps_scores ORDER BY created_at DESC LIMIT 100"
            ).fetchall()
            current_nps = _calc_nps([r["score"] for r in nps_rows])
            self._send_json({
                "status": "ok",
                "service": "feedback_engine",
                "port": PORT,
                "uptime_seconds": round(time.time() - START_TIME, 1),
                "response_count": rc,
                "current_nps": current_nps,
            })
        finally:
            conn.close()

    def _handle_list_surveys(self, query: dict):
        conn = _get_conn()
        try:
            clauses, params = [], []
            if query.get("status"):
                clauses.append("status=?")
                params.append(query["status"])
            if query.get("type"):
                clauses.append("survey_type=?")
                params.append(query["type"])
            where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            rows = conn.execute(
                f"SELECT * FROM surveys {where} ORDER BY created_at DESC", params
            ).fetchall()
            self._send_json([dict(r) for r in rows])
        finally:
            conn.close()

    def _handle_get_survey(self, survey_id: str):
        conn = _get_conn()
        try:
            sv = conn.execute(
                "SELECT * FROM surveys WHERE survey_id=?", (survey_id,)
            ).fetchone()
            if not sv:
                self._send_error("Survey not found", 404)
                return
            questions = conn.execute(
                "SELECT * FROM questions WHERE survey_id=? ORDER BY order_index",
                (survey_id,),
            ).fetchall()
            data = dict(sv)
            data["questions"] = [dict(q) for q in questions]
            # Never return responses for anonymous surveys
            self._send_json(data)
        finally:
            conn.close()

    def _handle_survey_results(self, survey_id: str):
        if not _is_admin(self.headers):
            self._send_error("Forbidden", 403)
            return
        conn = _get_conn()
        try:
            sv = conn.execute(
                "SELECT * FROM surveys WHERE survey_id=?", (survey_id,)
            ).fetchone()
            if not sv:
                self._send_error("Survey not found", 404)
                return
            questions = conn.execute(
                "SELECT * FROM questions WHERE survey_id=? ORDER BY order_index",
                (survey_id,),
            ).fetchall()
            results = []
            for q in questions:
                qid = q["id"]
                qtype = q["question_type"]
                agg: dict = {"question_id": qid, "question_text": q["question_text"],
                             "question_type": qtype}
                if qtype in ("rating", "nps"):
                    rows = conn.execute(
                        "SELECT answer_numeric FROM answers WHERE question_id=? AND answer_numeric IS NOT NULL",
                        (qid,),
                    ).fetchall()
                    vals = [r["answer_numeric"] for r in rows]
                    agg["count"] = len(vals)
                    agg["avg_score"] = round(statistics.mean(vals), 2) if vals else None
                    agg["min_score"] = min(vals) if vals else None
                    agg["max_score"] = max(vals) if vals else None
                    if qtype == "nps":
                        agg["nps"] = _calc_nps([int(v) for v in vals])
                elif qtype == "text":
                    rows = conn.execute(
                        "SELECT answer_text FROM answers WHERE question_id=? AND answer_text IS NOT NULL",
                        (qid,),
                    ).fetchall()
                    agg["responses"] = [r["answer_text"] for r in rows]
                    agg["count"] = len(agg["responses"])
                elif qtype in ("multiple_choice", "checkbox"):
                    rows = conn.execute(
                        "SELECT answer_options FROM answers WHERE question_id=? AND answer_options IS NOT NULL",
                        (qid,),
                    ).fetchall()
                    freq: dict[str, int] = {}
                    for r in rows:
                        try:
                            opts = json.loads(r["answer_options"])
                            if isinstance(opts, list):
                                for o in opts:
                                    freq[o] = freq.get(o, 0) + 1
                            else:
                                freq[str(opts)] = freq.get(str(opts), 0) + 1
                        except (json.JSONDecodeError, TypeError):
                            pass
                    agg["option_frequency"] = freq
                    agg["count"] = len(rows)
                elif qtype == "yes_no":
                    rows = conn.execute(
                        "SELECT answer_text FROM answers WHERE question_id=?", (qid,)
                    ).fetchall()
                    yes = sum(1 for r in rows if (r["answer_text"] or "").lower() in ("yes", "true", "1"))
                    no  = sum(1 for r in rows if (r["answer_text"] or "").lower() in ("no", "false", "0"))
                    agg["yes"] = yes
                    agg["no"]  = no
                    agg["count"] = len(rows)
                results.append(agg)
            self._send_json({"survey": dict(sv), "results": results})
        finally:
            conn.close()

    def _handle_survey_responses(self, survey_id: str):
        if not _is_admin(self.headers):
            self._send_error("Forbidden", 403)
            return
        conn = _get_conn()
        try:
            sv = conn.execute(
                "SELECT * FROM surveys WHERE survey_id=?", (survey_id,)
            ).fetchone()
            if not sv:
                self._send_error("Survey not found", 404)
                return
            resp_rows = conn.execute(
                "SELECT * FROM responses WHERE survey_id=? ORDER BY submitted_at DESC",
                (survey_id,),
            ).fetchall()
            out = []
            for resp in resp_rows:
                answers = conn.execute(
                    "SELECT * FROM answers WHERE response_id=?", (resp["id"],)
                ).fetchall()
                d = dict(resp)
                d["answers"] = [dict(a) for a in answers]
                out.append(d)
            self._send_json(out)
        finally:
            conn.close()

    def _handle_nps_dashboard(self):
        conn = _get_conn()
        try:
            all_scores = conn.execute(
                "SELECT score, created_at FROM nps_scores ORDER BY created_at DESC"
            ).fetchall()
            current_nps = _calc_nps([r["score"] for r in all_scores])

            # Trend — last 30 days in weekly buckets
            cutoff_30 = time.time() - 30 * 86400
            recent = [r for r in all_scores if r["created_at"] >= cutoff_30]
            recent_nps = _calc_nps([r["score"] for r in recent])

            # Category breakdown
            cats = conn.execute(
                "SELECT category, score FROM nps_scores WHERE category IS NOT NULL"
            ).fetchall()
            cat_map: dict[str, list[int]] = {}
            for r in cats:
                cat_map.setdefault(r["category"], []).append(r["score"])
            cat_breakdown = {k: _calc_nps(v) for k, v in cat_map.items()}

            total = len(all_scores)
            promoters  = sum(1 for r in all_scores if r["score"] >= 9)
            passives   = sum(1 for r in all_scores if 7 <= r["score"] <= 8)
            detractors = sum(1 for r in all_scores if r["score"] <= 6)

            self._send_json({
                "current_nps": current_nps,
                "last_30_days_nps": recent_nps,
                "total_responses": total,
                "promoters": promoters,
                "passives": passives,
                "detractors": detractors,
                "category_breakdown": cat_breakdown,
            })
        finally:
            conn.close()

    def _handle_list_feedback(self, query: dict):
        if not _is_admin(self.headers):
            self._send_error("Forbidden", 403)
            return
        conn = _get_conn()
        try:
            clauses, params = [], []
            if query.get("status"):
                clauses.append("status=?")
                params.append(query["status"])
            if query.get("sentiment"):
                clauses.append("sentiment=?")
                params.append(query["sentiment"])
            if query.get("priority"):
                clauses.append("priority=?")
                params.append(query["priority"])
            where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            limit = min(int(query.get("limit", "50")), 200)
            rows = conn.execute(
                f"SELECT * FROM feedback_items {where} ORDER BY created_at DESC LIMIT ?",
                params + [limit],
            ).fetchall()
            self._send_json([dict(r) for r in rows])
        finally:
            conn.close()

    def _handle_get_feedback(self, fid: str):
        if not _is_admin(self.headers):
            self._send_error("Forbidden", 403)
            return
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM feedback_items WHERE id=?", (fid,)
            ).fetchone()
            if not row:
                self._send_error("Feedback item not found", 404)
                return
            self._send_json(dict(row))
        finally:
            conn.close()

    def _handle_analytics(self):
        if not _is_admin(self.headers):
            self._send_error("Forbidden", 403)
            return
        conn = _get_conn()
        try:
            # NPS history — last 90 days in 7-day buckets
            cutoff = time.time() - 90 * 86400
            nps_all = conn.execute(
                "SELECT score, created_at FROM nps_scores WHERE created_at >= ? ORDER BY created_at",
                (cutoff,),
            ).fetchall()
            bucket_size = 7 * 86400
            nps_history: dict[str, float] = {}
            for r in nps_all:
                bucket_ts = int(r["created_at"] // bucket_size) * bucket_size
                label = time.strftime("%Y-%m-%d", time.gmtime(bucket_ts))
                nps_history.setdefault(label, [])
                nps_history[label].append(r["score"])
            nps_history_out = {k: _calc_nps(v) for k, v in nps_history.items()}

            # Sentiment distribution for feedback items
            sent_rows = conn.execute(
                "SELECT sentiment, COUNT(*) as cnt FROM feedback_items WHERE sentiment IS NOT NULL GROUP BY sentiment"
            ).fetchall()
            sentiment_dist = {r["sentiment"]: r["cnt"] for r in sent_rows}

            # Top issues by category
            cat_rows = conn.execute(
                """SELECT category, COUNT(*) as cnt FROM feedback_items
                   WHERE category IS NOT NULL GROUP BY category ORDER BY cnt DESC LIMIT 10"""
            ).fetchall()
            top_categories = [{"category": r["category"], "count": r["cnt"]} for r in cat_rows]

            # Satisfaction trend — avg_score per active survey
            sv_rows = conn.execute(
                "SELECT title, avg_score, response_count FROM surveys WHERE status='active' AND avg_score IS NOT NULL"
            ).fetchall()
            satisfaction_trend = [dict(r) for r in sv_rows]

            self._send_json({
                "nps_history": nps_history_out,
                "sentiment_distribution": sentiment_dist,
                "top_issues_by_category": top_categories,
                "satisfaction_trend": satisfaction_trend,
            })
        finally:
            conn.close()

    # =====================================================================
    # POST handlers
    # =====================================================================

    def _handle_create_survey(self):
        if not _is_admin(self.headers):
            self._send_error("Forbidden", 403)
            return
        body = self._read_body()
        title = (body.get("title") or "").strip()
        if not title:
            self._send_error("title is required")
            return
        questions_data = body.get("questions") or []
        now = time.time()
        survey_id = "survey_" + secrets.token_hex(8)
        closes_at = body.get("closes_at")
        conn = _get_conn()
        try:
            conn.execute(
                """INSERT INTO surveys
                   (survey_id, title, description, survey_type, status, anonymous,
                    created_at, updated_at, closes_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (survey_id, title, body.get("description", ""),
                 body.get("survey_type", "custom"),
                 body.get("status", "draft"),
                 int(body.get("anonymous", 1)),
                 now, now, closes_at),
            )
            for idx, q in enumerate(questions_data):
                qtext = (q.get("question_text") or "").strip()
                qtype = q.get("question_type", "text")
                if not qtext:
                    continue
                opts = q.get("options")
                conn.execute(
                    """INSERT INTO questions
                       (survey_id, question_text, question_type, options, required,
                        order_index, created_at)
                       VALUES (?,?,?,?,?,?,?)""",
                    (survey_id, qtext, qtype,
                     json.dumps(opts) if opts else None,
                     int(q.get("required", 1)),
                     q.get("order_index", idx + 1), now),
                )
            conn.commit()
            self._send_json({"survey_id": survey_id, "status": "created"}, 201)
        finally:
            conn.close()

    def _handle_submit_response(self, survey_id: str):
        conn = _get_conn()
        try:
            sv = conn.execute(
                "SELECT * FROM surveys WHERE survey_id=?", (survey_id,)
            ).fetchone()
            if not sv:
                self._send_error("Survey not found", 404)
                return
            if sv["status"] not in ("active",):
                self._send_error("Survey is not accepting responses", 400)
                return

            body   = self._read_body()
            now    = time.time()
            token  = secrets.token_urlsafe(24)
            email  = body.get("respondent_email") or None
            name   = body.get("respondent_name") or None
            meta   = json.dumps(body.get("metadata") or {})

            cur = conn.execute(
                """INSERT INTO responses
                   (survey_id, respondent_email, respondent_name, response_token,
                    submitted_at, ip_hash, metadata)
                   VALUES (?,?,?,?,?,?,?)""",
                (survey_id, email, name, token, now, self._ip_hash(), meta),
            )
            response_id = cur.lastrowid

            answers = body.get("answers") or []
            for ans in answers:
                qid     = ans.get("question_id")
                a_text  = ans.get("answer_text")
                a_num   = ans.get("answer_numeric")
                a_opts  = ans.get("answer_options")
                if qid is None:
                    continue
                conn.execute(
                    """INSERT INTO answers
                       (response_id, question_id, answer_text, answer_numeric, answer_options)
                       VALUES (?,?,?,?,?)""",
                    (response_id, qid, a_text, a_num,
                     json.dumps(a_opts) if a_opts else None),
                )

            # Update response count
            conn.execute(
                "UPDATE surveys SET response_count=response_count+1, updated_at=? WHERE survey_id=?",
                (now, survey_id),
            )
            conn.commit()
            self._send_json({"response_id": response_id, "token": token}, 201)
        finally:
            conn.close()

    def _handle_send_survey(self, survey_id: str):
        if not _is_admin(self.headers):
            self._send_error("Forbidden", 403)
            return
        conn = _get_conn()
        try:
            sv = conn.execute(
                "SELECT * FROM surveys WHERE survey_id=?", (survey_id,)
            ).fetchone()
            if not sv:
                self._send_error("Survey not found", 404)
                return
            body   = self._read_body()
            emails = body.get("emails") or []
            if not emails:
                self._send_error("emails list is required")
                return
            sent, failed = 0, 0
            for addr in emails:
                subject = f"We'd love your feedback: {sv['title']}"
                html    = (
                    f"<p>Hi there,</p>"
                    f"<p>Please take a moment to complete our survey: "
                    f"<strong>{sv['title']}</strong></p>"
                    f"<p>Survey ID: <code>{survey_id}</code></p>"
                    f"<p>Thank you!</p>"
                )
                if _send_email(addr, subject, html):
                    sent += 1
                else:
                    failed += 1
            self._send_json({"sent": sent, "failed": failed})
        finally:
            conn.close()

    def _handle_submit_nps(self):
        body = self._read_body()
        score_raw = body.get("score")
        if score_raw is None:
            self._send_error("score is required")
            return
        try:
            score = int(score_raw)
        except (ValueError, TypeError):
            self._send_error("score must be an integer 0-10")
            return
        if not 0 <= score <= 10:
            self._send_error("score must be 0-10")
            return

        comment = (body.get("comment") or "").strip()
        label, sent_score = _analyse_sentiment(comment) if comment else ("neutral", 0.0)
        now = time.time()
        conn = _get_conn()
        try:
            cur = conn.execute(
                """INSERT INTO nps_scores
                   (score, comment, customer_email, context, category,
                    sentiment, sentiment_score, created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (score, comment or None, body.get("customer_email"),
                 body.get("context"), body.get("category"),
                 label, sent_score, now),
            )
            conn.commit()
            self._send_json({"id": cur.lastrowid, "score": score, "sentiment": label}, 201)
        finally:
            conn.close()

    def _handle_submit_feedback(self):
        body = self._read_body()
        content = (body.get("content") or "").strip()
        if not content:
            self._send_error("content is required")
            return
        label, sent_score = _analyse_sentiment(content)
        now = time.time()
        conn = _get_conn()
        try:
            cur = conn.execute(
                """INSERT INTO feedback_items
                   (source, content, customer_email, sentiment, sentiment_score,
                    category, priority, status, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (body.get("source", "api"), content,
                 body.get("customer_email"),
                 label, sent_score, body.get("category"),
                 body.get("priority", "normal"), "open", now),
            )
            conn.commit()
            self._send_json({
                "id": cur.lastrowid,
                "sentiment": label,
                "sentiment_score": sent_score,
            }, 201)
        finally:
            conn.close()

    # =====================================================================
    # PUT handlers
    # =====================================================================

    def _handle_resolve_feedback(self, fid: str):
        if not _is_admin(self.headers):
            self._send_error("Forbidden", 403)
            return
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT id FROM feedback_items WHERE id=?", (fid,)
            ).fetchone()
            if not row:
                self._send_error("Feedback item not found", 404)
                return
            now = time.time()
            conn.execute(
                "UPDATE feedback_items SET status='resolved', resolved_at=? WHERE id=?",
                (now, fid),
            )
            conn.commit()
            self._send_json({"id": int(fid), "status": "resolved"})
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    _init_db()
    _seed_surveys()

    bg = threading.Thread(target=_maintenance_loop, daemon=True, name="maintenance")
    bg.start()

    server = HTTPServer(("0.0.0.0", PORT), FeedbackHandler)
    _log("INFO", f"Feedback Engine started on port {PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log("INFO", "Feedback Engine shutting down")
        server.server_close()


if __name__ == "__main__":
    main()
