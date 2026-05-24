#!/usr/bin/env python3
"""
FractalMesh OMEGA Titan — Content Generation Engine
Port: 7838
AI-powered content creation, scheduling, and publishing for FractalMesh.
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

# ---------------------------------------------------------------------------
# Vault loading
# ---------------------------------------------------------------------------
_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _line in _vault.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PORT               = int(os.getenv("CONTENT_ENGINE_PORT", "7838"))
ROOT               = Path(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")))
DB_PATH            = ROOT / "database" / "sovereign.db"
LOG_PATH           = ROOT / "logs" / "content_engine.log"
ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
DEVTO_API_KEY      = os.getenv("DEVTO_API_KEY", "")
DEVTO_PUBLISH_LIVE = os.getenv("DEVTO_PUBLISH_LIVE", "false")
SENDGRID_API_KEY   = os.getenv("SENDGRID_API_KEY", "")
SENDGRID_FROM      = os.getenv("SENDGRID_FROM_EMAIL", "")

ROOT.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [CONTENT] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("content_engine")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DDL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS content_pieces (
    id           INTEGER PRIMARY KEY,
    title        TEXT,
    content_type TEXT,
    status       TEXT DEFAULT 'draft',
    topic        TEXT,
    keywords     TEXT,
    body         TEXT,
    word_count   INTEGER,
    platform     TEXT,
    published_url TEXT,
    published_at REAL,
    created_at   REAL
);

CREATE TABLE IF NOT EXISTS content_templates (
    id           INTEGER PRIMARY KEY,
    name         TEXT UNIQUE,
    content_type TEXT,
    template     TEXT,
    variables    TEXT,
    created_at   REAL
);

CREATE TABLE IF NOT EXISTS content_calendar (
    id           INTEGER PRIMARY KEY,
    piece_id     INTEGER,
    scheduled_at REAL,
    platform     TEXT,
    status       TEXT DEFAULT 'scheduled',
    published_at REAL
);
"""

DEFAULT_TEMPLATES = [
    (
        "product_announcement",
        "social_post",
        "🚀 Exciting news! {product_name} is now live. {description} Try it at {url}",
        json.dumps(["product_name", "description", "url"]),
    ),
    (
        "lead_nurture_email",
        "email",
        (
            "Subject: {subject}\n\nHi {first_name},\n\n{opening_line}\n\n"
            "{body_content}\n\nReady to take the next step? {cta_text}: {cta_url}\n\n"
            "Best regards,\nThe FractalMesh Team\nhttps://fractalmesh.net"
        ),
        json.dumps(["subject", "first_name", "opening_line", "body_content", "cta_text", "cta_url"]),
    ),
    (
        "weekly_newsletter",
        "email",
        (
            "Subject: {subject}\n\nHello FractalMesh Community,\n\n"
            "{intro}\n\n## This Week's Highlights\n\n{highlights}\n\n"
            "## Featured Resource\n\n{featured_resource}\n\n"
            "{closing}\n\nUntil next week,\nSamuel | FractalMesh\nhttps://fractalmesh.net"
        ),
        json.dumps(["subject", "intro", "highlights", "featured_resource", "closing"]),
    ),
    (
        "twitter_thread_intro",
        "social_post",
        (
            "🧵 Thread: {thread_topic}\n\n"
            "{hook}\n\nHere's what you need to know 👇\n\n"
            "1/ {first_point}\n\n{hashtags}"
        ),
        json.dumps(["thread_topic", "hook", "first_point", "hashtags"]),
    ),
    (
        "linkedin_thought_leadership",
        "social_post",
        (
            "{opening_hook}\n\n"
            "Here's what I've learned working with Australian small businesses:\n\n"
            "{insight_1}\n\n{insight_2}\n\n{insight_3}\n\n"
            "The bottom line: {conclusion}\n\n"
            "What's your experience with {topic}? Drop a comment below.\n\n"
            "{hashtags}"
        ),
        json.dumps(["opening_hook", "insight_1", "insight_2", "insight_3", "conclusion", "topic", "hashtags"]),
    ),
]


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    with get_db() as conn:
        conn.executescript(DDL)
        now = time.time()
        for name, ctype, template, variables in DEFAULT_TEMPLATES:
            conn.execute(
                """INSERT OR IGNORE INTO content_templates
                   (name, content_type, template, variables, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (name, ctype, template, variables, now),
            )
        conn.commit()
    log.info("Database initialised at %s", DB_PATH)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _count_words(text: str) -> int:
    return len(text.split())


def _claude_generate(system_prompt: str, user_prompt: str, max_tokens: int = 2000) -> str:
    """Call Anthropic Messages API and return the assistant text."""
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    payload = json.dumps({
        "model": "claude-opus-4-5",
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
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
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data["content"][0]["text"]


def _devto_publish(title: str, body: str, tags: list, published: bool) -> tuple:
    """POST article to Dev.to. Returns (devto_id, url)."""
    if not DEVTO_API_KEY:
        raise RuntimeError("DEVTO_API_KEY not set")
    payload = json.dumps({
        "article": {
            "title": title,
            "body_markdown": body,
            "published": published,
            "tags": tags,
        }
    }).encode()
    req = urllib.request.Request(
        "https://dev.to/api/articles",
        data=payload,
        headers={
            "api-key": DEVTO_API_KEY,
            "content-type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data["id"], data.get("url", "")


def _extract_title(body: str) -> str:
    """Extract first markdown H1 heading from body text."""
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return "Untitled"


def _send_json(handler, status: int, payload: dict) -> None:
    body = json.dumps(payload).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_body(handler) -> dict:
    length = int(handler.headers.get("Content-Length", 0))
    raw = handler.rfile.read(length) if length else b"{}"
    return json.loads(raw)


# ---------------------------------------------------------------------------
# HTTP Request Handler
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a content marketer for FractalMesh, an AI automation platform for "
    "Australian small businesses. ABN: 56628117363. Write professional, SEO-optimised content."
)


class ContentEngineHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # silence default access log
        log.debug(fmt, *args)

    # ------------------------------------------------------------------ routing
    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        if path == "/health":
            return self._handle_health()
        if path == "/pieces":
            return self._handle_list_pieces()
        if path.startswith("/pieces/"):
            piece_id = path[len("/pieces/"):]
            return self._handle_get_piece(piece_id)
        if path == "/calendar":
            return self._handle_get_calendar()
        if path == "/templates":
            return self._handle_list_templates()
        if path == "/analytics":
            return self._handle_analytics()
        _send_json(self, 404, {"error": "not found"})

    def do_POST(self):
        path = self.path.rstrip("/")
        routes = {
            "/generate/article":      self._handle_generate_article,
            "/generate/social":       self._handle_generate_social,
            "/generate/email":        self._handle_generate_email,
            "/generate/batch":        self._handle_generate_batch,
            "/publish/devto":         self._handle_publish_devto,
            "/publish/newsletter":    self._handle_publish_newsletter,
            "/calendar/schedule":     self._handle_calendar_schedule,
            "/calendar/execute_due":  self._handle_calendar_execute_due,
            "/templates/create":      self._handle_template_create,
            "/templates/generate":    self._handle_template_generate,
        }
        handler = routes.get(path)
        if handler:
            return handler()
        _send_json(self, 404, {"error": "not found"})

    def do_PUT(self):
        path = self.path.rstrip("/")
        if path.startswith("/pieces/"):
            piece_id = path[len("/pieces/"):]
            return self._handle_update_piece(piece_id)
        _send_json(self, 404, {"error": "not found"})

    def do_DELETE(self):
        path = self.path.rstrip("/")
        if path.startswith("/pieces/"):
            piece_id = path[len("/pieces/"):]
            return self._handle_delete_piece(piece_id)
        _send_json(self, 404, {"error": "not found"})

    # ------------------------------------------------------------------ health
    def _handle_health(self):
        _send_json(self, 200, {"status": "ok", "service": "fm-content-engine", "port": 7838})

    # ------------------------------------------------------------------ generate article
    def _handle_generate_article(self):
        try:
            body = _read_body(self)
            topic        = body.get("topic", "AI automation")
            keywords     = body.get("keywords", [])
            word_count   = int(body.get("word_count", 800))
            style        = body.get("style", "informative")
            include_cta  = bool(body.get("include_cta", True))

            kw_str = ", ".join(keywords) if keywords else topic
            cta_str = "End with a call-to-action to visit https://fractalmesh.net." if include_cta else ""
            user_prompt = (
                f"Write a {word_count}-word {style} article about '{topic}'. "
                f"Include the keywords: {kw_str}. {cta_str} Format with markdown headers."
            )
            content = _claude_generate(SYSTEM_PROMPT, user_prompt, max_tokens=word_count * 2 + 500)
            title = _extract_title(content)
            wc = _count_words(content)
            now = time.time()
            with get_db() as conn:
                cur = conn.execute(
                    """INSERT INTO content_pieces
                       (title, content_type, status, topic, keywords, body, word_count, platform, created_at)
                       VALUES (?, 'article', 'draft', ?, ?, ?, ?, NULL, ?)""",
                    (title, topic, json.dumps(keywords), content, wc, now),
                )
                conn.commit()
                piece_id = cur.lastrowid
            _send_json(self, 200, {"piece_id": piece_id, "title": title, "word_count": wc, "status": "draft"})
        except Exception as exc:
            log.exception("generate/article failed")
            _send_json(self, 500, {"error": str(exc)})

    # ------------------------------------------------------------------ generate social
    def _handle_generate_social(self):
        try:
            body = _read_body(self)
            topic            = body.get("topic", "FractalMesh")
            platform         = body.get("platform", "twitter")
            tone             = body.get("tone", "professional")
            include_hashtags = bool(body.get("include_hashtags", True))

            platform_guides = {
                "twitter":  "Write a punchy Twitter/X post (strict maximum 280 characters).",
                "linkedin": "Write a professional LinkedIn post (150-300 words).",
                "facebook": "Write an engaging Facebook post (100-200 words).",
            }
            guide = platform_guides.get(platform, platform_guides["twitter"])
            hashtag_note = "Include relevant hashtags." if include_hashtags else "Do not include hashtags."
            user_prompt = (
                f"{guide} Tone: {tone}. Topic: {topic}. {hashtag_note} "
                "Return only the post text, no preamble."
            )
            content = _claude_generate(SYSTEM_PROMPT, user_prompt, max_tokens=600)
            content = content.strip()
            now = time.time()
            with get_db() as conn:
                cur = conn.execute(
                    """INSERT INTO content_pieces
                       (title, content_type, status, topic, keywords, body, word_count, platform, created_at)
                       VALUES (?, 'social_post', 'draft', ?, ?, ?, ?, ?, ?)""",
                    (f"{platform.title()} post: {topic}", topic, "[]", content, _count_words(content), platform, now),
                )
                conn.commit()
                piece_id = cur.lastrowid
            _send_json(self, 200, {
                "piece_id": piece_id,
                "content": content,
                "platform": platform,
                "char_count": len(content),
            })
        except Exception as exc:
            log.exception("generate/social failed")
            _send_json(self, 500, {"error": str(exc)})

    # ------------------------------------------------------------------ generate email
    def _handle_generate_email(self):
        try:
            body = _read_body(self)
            subject            = body.get("subject", "FractalMesh — AI Automation for Your Business")
            audience           = body.get("audience", "small business owners")
            goal               = body.get("goal", "sign_up")
            tone               = body.get("tone", "friendly")
            include_unsubscribe = bool(body.get("include_unsubscribe", True))

            unsub_note = (
                "End with a plain-text unsubscribe line: "
                "'To unsubscribe, reply UNSUBSCRIBE to this email.'"
                if include_unsubscribe else ""
            )
            user_prompt = (
                f"Write a {tone} marketing email for {audience}. "
                f"Subject: {subject}. Goal: {goal}. "
                "Provide a full HTML email body with inline styles, and also a plain-text version. "
                "Return a JSON object with keys 'html_body' and 'text_body'. "
                f"{unsub_note}"
            )
            raw = _claude_generate(SYSTEM_PROMPT, user_prompt, max_tokens=2500)
            # Attempt to parse JSON from the response
            try:
                # Find JSON block within possible markdown fences
                start = raw.find("{")
                end   = raw.rfind("}") + 1
                parsed = json.loads(raw[start:end])
                html_body = parsed.get("html_body", raw)
                text_body = parsed.get("text_body", raw)
            except (ValueError, json.JSONDecodeError):
                html_body = raw
                text_body = raw

            now = time.time()
            with get_db() as conn:
                cur = conn.execute(
                    """INSERT INTO content_pieces
                       (title, content_type, status, topic, keywords, body, word_count, platform, created_at)
                       VALUES (?, 'email', 'draft', ?, ?, ?, ?, 'email', ?)""",
                    (subject, audience, json.dumps([goal]), html_body, _count_words(text_body), now),
                )
                conn.commit()
                piece_id = cur.lastrowid
            _send_json(self, 200, {
                "piece_id": piece_id,
                "subject": subject,
                "html_body": html_body,
                "text_body": text_body,
            })
        except Exception as exc:
            log.exception("generate/email failed")
            _send_json(self, 500, {"error": str(exc)})

    # ------------------------------------------------------------------ generate batch
    def _handle_generate_batch(self):
        try:
            body = _read_body(self)
            topics                = body.get("topics", [])
            content_type          = body.get("content_type", "article")
            platform              = body.get("platform", "devto")
            schedule_interval_days = int(body.get("schedule_interval_days", 3))

            pieces = []
            calendar_entries = 0
            now = time.time()

            for i, topic in enumerate(topics):
                if content_type == "article":
                    user_prompt = (
                        f"Write an 800-word informative article about '{topic}' "
                        "for Australian small business owners. Format with markdown headers. "
                        "End with a call-to-action to visit https://fractalmesh.net."
                    )
                    content = _claude_generate(SYSTEM_PROMPT, user_prompt, max_tokens=2000)
                    title = _extract_title(content)
                else:
                    user_prompt = (
                        f"Write a professional LinkedIn post about '{topic}' "
                        "for Australian small business owners. 150-200 words. Include hashtags."
                    )
                    content = _claude_generate(SYSTEM_PROMPT, user_prompt, max_tokens=600)
                    title = f"{content_type.title()}: {topic}"

                wc = _count_words(content)
                with get_db() as conn:
                    cur = conn.execute(
                        """INSERT INTO content_pieces
                           (title, content_type, status, topic, keywords, body, word_count, platform, created_at)
                           VALUES (?, ?, 'draft', ?, ?, ?, ?, ?, ?)""",
                        (title, content_type, topic, "[]", content, wc, platform, now),
                    )
                    conn.commit()
                    piece_id = cur.lastrowid

                scheduled_at = now + i * schedule_interval_days * 86400
                with get_db() as conn:
                    conn.execute(
                        """INSERT INTO content_calendar
                           (piece_id, scheduled_at, platform, status)
                           VALUES (?, ?, ?, 'scheduled')""",
                        (piece_id, scheduled_at, platform),
                    )
                    conn.commit()
                calendar_entries += 1
                pieces.append({"piece_id": piece_id, "title": title, "word_count": wc, "topic": topic})

                if i < len(topics) - 1:
                    time.sleep(1)

            _send_json(self, 200, {"pieces": pieces, "calendar_entries": calendar_entries})
        except Exception as exc:
            log.exception("generate/batch failed")
            _send_json(self, 500, {"error": str(exc)})

    # ------------------------------------------------------------------ publish devto
    def _handle_publish_devto(self):
        try:
            body     = _read_body(self)
            piece_id = int(body.get("piece_id", 0))
            tags     = body.get("tags", ["ai", "automation", "productivity"])

            with get_db() as conn:
                row = conn.execute(
                    "SELECT * FROM content_pieces WHERE id=?", (piece_id,)
                ).fetchone()
            if not row:
                return _send_json(self, 404, {"error": "piece not found"})

            published = DEVTO_PUBLISH_LIVE.lower() == "true"
            devto_id, url = _devto_publish(row["title"], row["body"], tags, published)
            now = time.time()
            with get_db() as conn:
                conn.execute(
                    """UPDATE content_pieces
                       SET status='published', published_url=?, published_at=?
                       WHERE id=?""",
                    (url, now, piece_id),
                )
                conn.commit()
            _send_json(self, 200, {
                "piece_id": piece_id,
                "devto_id": devto_id,
                "url": url,
                "published": published,
            })
        except Exception as exc:
            log.exception("publish/devto failed")
            _send_json(self, 500, {"error": str(exc)})

    # ------------------------------------------------------------------ publish newsletter
    def _handle_publish_newsletter(self):
        try:
            body            = _read_body(self)
            piece_id        = int(body.get("piece_id", 0))
            list_id         = body.get("list_id", "")
            subject_override = body.get("subject_override", "")
            send_at         = body.get("send_at", None)

            with get_db() as conn:
                row = conn.execute(
                    "SELECT * FROM content_pieces WHERE id=?", (piece_id,)
                ).fetchone()
            if not row:
                return _send_json(self, 404, {"error": "piece not found"})

            if not SENDGRID_API_KEY:
                raise RuntimeError("SENDGRID_API_KEY not set")
            if not SENDGRID_FROM:
                raise RuntimeError("SENDGRID_FROM_EMAIL not set")

            subject = subject_override or row["title"] or "FractalMesh Newsletter"
            html_content = row["body"] or "<p>No content</p>"
            text_content = row["body"] or "No content"

            mail_payload: dict = {
                "from": {"email": SENDGRID_FROM},
                "subject": subject,
                "content": [
                    {"type": "text/plain", "value": text_content},
                    {"type": "text/html",  "value": html_content},
                ],
            }
            if list_id:
                mail_payload["personalizations"] = [
                    {"to": [{"email": SENDGRID_FROM}], "substitutions": {}}
                ]
                mail_payload["mail_settings"] = {"bypass_list_management": {"enable": False}}
            else:
                mail_payload["personalizations"] = [{"to": [{"email": SENDGRID_FROM}]}]

            if send_at:
                mail_payload["send_at"] = int(send_at)

            payload = json.dumps(mail_payload).encode()
            req = urllib.request.Request(
                "https://api.sendgrid.com/v3/mail/send",
                data=payload,
                headers={
                    "Authorization": f"Bearer {SENDGRID_API_KEY}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                msg_id = resp.headers.get("X-Message-Id", "")

            _send_json(self, 200, {"piece_id": piece_id, "sendgrid_message_id": msg_id})
        except Exception as exc:
            log.exception("publish/newsletter failed")
            _send_json(self, 500, {"error": str(exc)})

    # ------------------------------------------------------------------ list pieces
    def _handle_list_pieces(self):
        try:
            with get_db() as conn:
                rows = conn.execute(
                    """SELECT id, title, content_type, status, platform,
                              word_count, created_at
                       FROM content_pieces ORDER BY created_at DESC"""
                ).fetchall()
            _send_json(self, 200, [dict(r) for r in rows])
        except Exception as exc:
            _send_json(self, 500, {"error": str(exc)})

    # ------------------------------------------------------------------ get piece
    def _handle_get_piece(self, piece_id: str):
        try:
            with get_db() as conn:
                row = conn.execute(
                    "SELECT * FROM content_pieces WHERE id=?", (int(piece_id),)
                ).fetchone()
            if not row:
                return _send_json(self, 404, {"error": "piece not found"})
            _send_json(self, 200, dict(row))
        except ValueError:
            _send_json(self, 400, {"error": "invalid piece id"})
        except Exception as exc:
            _send_json(self, 500, {"error": str(exc)})

    # ------------------------------------------------------------------ update piece
    def _handle_update_piece(self, piece_id: str):
        try:
            body   = _read_body(self)
            fields = []
            values = []
            if "status" in body:
                fields.append("status=?")
                values.append(body["status"])
            if "body" in body:
                fields.append("body=?")
                values.append(body["body"])
                fields.append("word_count=?")
                values.append(_count_words(body["body"]))
            if "title" in body:
                fields.append("title=?")
                values.append(body["title"])
            if not fields:
                return _send_json(self, 400, {"error": "no updatable fields provided"})
            values.append(int(piece_id))
            with get_db() as conn:
                conn.execute(f"UPDATE content_pieces SET {', '.join(fields)} WHERE id=?", values)
                conn.commit()
            _send_json(self, 200, {"piece_id": int(piece_id), "updated": True})
        except ValueError:
            _send_json(self, 400, {"error": "invalid piece id"})
        except Exception as exc:
            _send_json(self, 500, {"error": str(exc)})

    # ------------------------------------------------------------------ delete piece
    def _handle_delete_piece(self, piece_id: str):
        try:
            with get_db() as conn:
                conn.execute("DELETE FROM content_pieces WHERE id=?", (int(piece_id),))
                conn.commit()
            _send_json(self, 200, {"piece_id": int(piece_id), "deleted": True})
        except ValueError:
            _send_json(self, 400, {"error": "invalid piece id"})
        except Exception as exc:
            _send_json(self, 500, {"error": str(exc)})

    # ------------------------------------------------------------------ calendar get
    def _handle_get_calendar(self):
        try:
            qs     = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            from_d = float(qs.get("from_date", [0])[0]) if "from_date" in qs else 0
            to_d   = float(qs.get("to_date", [9999999999])[0]) if "to_date" in qs else 9999999999
            # Accept ISO date strings like "2026-05-01"
            if isinstance(from_d, str) or (isinstance(from_d, float) and from_d < 1e9):
                from_d = 0
            with get_db() as conn:
                rows = conn.execute(
                    """SELECT cc.id, cc.piece_id, cc.scheduled_at, cc.platform,
                              cc.status, cc.published_at,
                              cp.title, cp.content_type
                       FROM content_calendar cc
                       LEFT JOIN content_pieces cp ON cp.id = cc.piece_id
                       WHERE cc.scheduled_at >= ? AND cc.scheduled_at <= ?
                       ORDER BY cc.scheduled_at""",
                    (from_d, to_d),
                ).fetchall()
            _send_json(self, 200, [dict(r) for r in rows])
        except Exception as exc:
            _send_json(self, 500, {"error": str(exc)})

    # ------------------------------------------------------------------ calendar schedule
    def _handle_calendar_schedule(self):
        try:
            body         = _read_body(self)
            piece_id     = int(body.get("piece_id", 0))
            scheduled_at = float(body.get("scheduled_at", time.time()))
            platform     = body.get("platform", "devto")
            with get_db() as conn:
                cur = conn.execute(
                    """INSERT INTO content_calendar (piece_id, scheduled_at, platform, status)
                       VALUES (?, ?, ?, 'scheduled')""",
                    (piece_id, scheduled_at, platform),
                )
                conn.commit()
                entry_id = cur.lastrowid
            _send_json(self, 200, {"entry_id": entry_id})
        except Exception as exc:
            _send_json(self, 500, {"error": str(exc)})

    # ------------------------------------------------------------------ calendar execute due
    def _handle_calendar_execute_due(self):
        published_count = 0
        failed_count    = 0
        now = time.time()
        try:
            with get_db() as conn:
                due = conn.execute(
                    """SELECT cc.id, cc.piece_id, cc.platform
                       FROM content_calendar cc
                       WHERE cc.scheduled_at <= ? AND cc.status='scheduled'""",
                    (now,),
                ).fetchall()

            for entry in due:
                entry_id = entry["id"]
                piece_id = entry["piece_id"]
                platform = entry["platform"]
                try:
                    if platform == "devto":
                        published = DEVTO_PUBLISH_LIVE.lower() == "true"
                        with get_db() as conn:
                            row = conn.execute(
                                "SELECT * FROM content_pieces WHERE id=?", (piece_id,)
                            ).fetchone()
                        if row:
                            devto_id, url = _devto_publish(
                                row["title"], row["body"], ["ai", "automation"], published
                            )
                            with get_db() as conn:
                                conn.execute(
                                    """UPDATE content_pieces
                                       SET status='published', published_url=?, published_at=?
                                       WHERE id=?""",
                                    (url, now, piece_id),
                                )
                                conn.commit()
                    with get_db() as conn:
                        conn.execute(
                            """UPDATE content_calendar
                               SET status='published', published_at=?
                               WHERE id=?""",
                            (now, entry_id),
                        )
                        conn.commit()
                    published_count += 1
                except Exception:
                    log.exception("calendar execute_due failed for entry %s", entry_id)
                    with get_db() as conn:
                        conn.execute(
                            "UPDATE content_calendar SET status='failed' WHERE id=?",
                            (entry_id,),
                        )
                        conn.commit()
                    failed_count += 1

            _send_json(self, 200, {"published": published_count, "failed": failed_count})
        except Exception as exc:
            _send_json(self, 500, {"error": str(exc)})

    # ------------------------------------------------------------------ templates list
    def _handle_list_templates(self):
        try:
            with get_db() as conn:
                rows = conn.execute(
                    "SELECT id, name, content_type, variables, created_at FROM content_templates ORDER BY id"
                ).fetchall()
            _send_json(self, 200, [dict(r) for r in rows])
        except Exception as exc:
            _send_json(self, 500, {"error": str(exc)})

    # ------------------------------------------------------------------ template create
    def _handle_template_create(self):
        try:
            body         = _read_body(self)
            name         = body.get("name", "")
            content_type = body.get("content_type", "social_post")
            template     = body.get("template", "")
            variables    = body.get("variables", [])
            if not name or not template:
                return _send_json(self, 400, {"error": "name and template are required"})
            now = time.time()
            with get_db() as conn:
                cur = conn.execute(
                    """INSERT INTO content_templates (name, content_type, template, variables, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (name, content_type, template, json.dumps(variables), now),
                )
                conn.commit()
                template_id = cur.lastrowid
            _send_json(self, 200, {"template_id": template_id, "name": name})
        except sqlite3.IntegrityError:
            _send_json(self, 409, {"error": "template name already exists"})
        except Exception as exc:
            _send_json(self, 500, {"error": str(exc)})

    # ------------------------------------------------------------------ template generate
    def _handle_template_generate(self):
        try:
            body        = _read_body(self)
            template_id = int(body.get("template_id", 0))
            variables   = body.get("variables", {})

            with get_db() as conn:
                row = conn.execute(
                    "SELECT * FROM content_templates WHERE id=?", (template_id,)
                ).fetchone()
            if not row:
                return _send_json(self, 404, {"error": "template not found"})

            content = row["template"]
            for key, val in variables.items():
                content = content.replace("{" + key + "}", str(val))

            now = time.time()
            with get_db() as conn:
                cur = conn.execute(
                    """INSERT INTO content_pieces
                       (title, content_type, status, topic, keywords, body, word_count, platform, created_at)
                       VALUES (?, ?, 'draft', ?, '[]', ?, ?, NULL, ?)""",
                    (
                        f"From template: {row['name']}",
                        row["content_type"],
                        row["name"],
                        content,
                        _count_words(content),
                        now,
                    ),
                )
                conn.commit()
                piece_id = cur.lastrowid

            _send_json(self, 200, {"piece_id": piece_id, "content": content})
        except ValueError:
            _send_json(self, 400, {"error": "invalid template_id"})
        except Exception as exc:
            _send_json(self, 500, {"error": str(exc)})

    # ------------------------------------------------------------------ analytics
    def _handle_analytics(self):
        try:
            now = time.time()
            day_start = now - (now % 86400)
            with get_db() as conn:
                by_status = conn.execute(
                    "SELECT status, COUNT(*) as count FROM content_pieces GROUP BY status"
                ).fetchall()
                by_type = conn.execute(
                    "SELECT content_type, COUNT(*) as count FROM content_pieces GROUP BY content_type"
                ).fetchall()
                by_platform = conn.execute(
                    "SELECT platform, COUNT(*) as count FROM content_pieces WHERE platform IS NOT NULL GROUP BY platform"
                ).fetchall()
                published_today = conn.execute(
                    "SELECT COUNT(*) as count FROM content_pieces WHERE published_at >= ?",
                    (day_start,),
                ).fetchone()["count"]
                total_published = conn.execute(
                    "SELECT COUNT(*) as count FROM content_pieces WHERE status='published'"
                ).fetchone()["count"]
                avg_wc_row = conn.execute(
                    "SELECT AVG(word_count) as avg FROM content_pieces WHERE word_count > 0"
                ).fetchone()
                avg_wc = round(avg_wc_row["avg"] or 0, 1)

            _send_json(self, 200, {
                "by_status":        [dict(r) for r in by_status],
                "by_type":          [dict(r) for r in by_type],
                "by_platform":      [dict(r) for r in by_platform],
                "published_today":  published_today,
                "total_published":  total_published,
                "avg_word_count":   avg_wc,
            })
        except Exception as exc:
            _send_json(self, 500, {"error": str(exc)})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    init_db()
    server = HTTPServer(("0.0.0.0", PORT), ContentEngineHandler)
    log.info("fm-content-engine starting on port %d", PORT)

    def _shutdown(sig, frame):
        log.info("Shutting down (signal %d)", sig)
        server.server_close()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    server.serve_forever()


if __name__ == "__main__":
    main()
