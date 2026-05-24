"""
FractalMesh OMEGA Titan — Affiliate Revenue Engine
Port: 7842
Samuel James Hiotis | ABN 56 628 117 363
"""

import os
import json
import sqlite3
import logging
import signal
import time
import hashlib
import urllib.request
import urllib.error
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from datetime import datetime, date

# ---------------------------------------------------------------------------
# Vault loading
# ---------------------------------------------------------------------------
_VAULT = Path.home() / ".secrets" / "fractal.env"
if _VAULT.exists():
    with open(_VAULT) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
BASE_DIR = Path.home() / "fmsaas"
DB_PATH  = BASE_DIR / "database" / "sovereign.db"
LOG_PATH = BASE_DIR / "logs" / "fm_affiliate_engine.log"

for _d in (DB_PATH.parent, LOG_PATH.parent):
    _d.mkdir(parents=True, exist_ok=True)

PORT               = int(os.environ.get("AFFILIATE_PORT", "7842"))
MANUS_REF_CODE     = os.environ.get("MANUS_REF_CODE", "")
BLOFIN_REF_CODE    = os.environ.get("BLOFIN_REF_CODE", "")
AFF_VULTR_URL      = os.environ.get("AFF_VULTR_URL", "")
AFF_DO_URL         = os.environ.get("AFF_DO_URL", "")
DEVTO_API_KEY      = os.environ.get("DEVTO_API_KEY", "")
ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
SENDGRID_API_KEY   = os.environ.get("SENDGRID_API_KEY", "")
SENDGRID_FROM      = os.environ.get("SENDGRID_FROM_EMAIL", "")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("fm_affiliate_engine")

# ---------------------------------------------------------------------------
# SQLite — WAL mode
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=15, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _init_db():
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS affiliate_programs (
            id              INTEGER PRIMARY KEY,
            name            TEXT UNIQUE,
            category        TEXT,
            ref_code        TEXT,
            link            TEXT,
            commission_pct  REAL DEFAULT 0,
            commission_flat REAL DEFAULT 0,
            cookie_days     INTEGER DEFAULT 30,
            status          TEXT DEFAULT 'active',
            clicks          INTEGER DEFAULT 0,
            conversions     INTEGER DEFAULT 0,
            revenue_usd     REAL DEFAULT 0,
            created_at      REAL
        );

        CREATE TABLE IF NOT EXISTS affiliate_clicks (
            id          INTEGER PRIMARY KEY,
            program_id  INTEGER,
            referrer    TEXT,
            ip_hash     TEXT,
            created_at  REAL
        );

        CREATE TABLE IF NOT EXISTS affiliate_content (
            id                INTEGER PRIMARY KEY,
            program_id        INTEGER,
            title             TEXT,
            platform          TEXT,
            content           TEXT,
            published_url     TEXT,
            clicks_generated  INTEGER DEFAULT 0,
            created_at        REAL
        );

        CREATE TABLE IF NOT EXISTS affiliate_campaigns (
            id              INTEGER PRIMARY KEY,
            name            TEXT,
            programs        TEXT,
            content_type    TEXT,
            status          TEXT,
            leads_targeted  INTEGER DEFAULT 0,
            clicks_total    INTEGER DEFAULT 0,
            created_at      REAL
        );
    """)
    conn.commit()
    conn.close()
    log.info("DB tables initialised")


# ---------------------------------------------------------------------------
# Default program seeds
# ---------------------------------------------------------------------------

def _build_default_programs():
    programs = []

    if MANUS_REF_CODE:
        programs.append({
            "name": "Manus AI",
            "category": "ai_tools",
            "ref_code": MANUS_REF_CODE,
            "link": f"https://manus.im/?aff={MANUS_REF_CODE}",
            "commission_pct": 30,
            "commission_flat": 0,
            "cookie_days": 30,
        })
    else:
        programs.append({
            "name": "Manus AI",
            "category": "ai_tools",
            "ref_code": "",
            "link": "https://manus.im/",
            "commission_pct": 30,
            "commission_flat": 0,
            "cookie_days": 30,
        })

    if BLOFIN_REF_CODE:
        programs.append({
            "name": "BloFin",
            "category": "crypto",
            "ref_code": BLOFIN_REF_CODE,
            "link": f"https://blofin.com/register?affiliate={BLOFIN_REF_CODE}",
            "commission_pct": 20,
            "commission_flat": 0,
            "cookie_days": 30,
        })
    else:
        programs.append({
            "name": "BloFin",
            "category": "crypto",
            "ref_code": "",
            "link": "https://blofin.com/register",
            "commission_pct": 20,
            "commission_flat": 0,
            "cookie_days": 30,
        })

    # Vultr — extract ref from URL or use the URL itself
    vultr_ref = ""
    if AFF_VULTR_URL:
        parsed = urllib.parse.urlparse(AFF_VULTR_URL)
        qs = urllib.parse.parse_qs(parsed.query)
        vultr_ref = qs.get("ref", [AFF_VULTR_URL])[0]
    programs.append({
        "name": "Vultr",
        "category": "hosting",
        "ref_code": vultr_ref,
        "link": AFF_VULTR_URL or "https://www.vultr.com/",
        "commission_pct": 0,
        "commission_flat": 35,
        "cookie_days": 30,
    })

    # DigitalOcean — extract ref from URL
    do_ref = ""
    if AFF_DO_URL:
        parsed = urllib.parse.urlparse(AFF_DO_URL)
        qs = urllib.parse.parse_qs(parsed.query)
        do_ref = qs.get("refcode", qs.get("ref", [AFF_DO_URL]))[0]
    programs.append({
        "name": "DigitalOcean",
        "category": "hosting",
        "ref_code": do_ref,
        "link": AFF_DO_URL or "https://www.digitalocean.com/",
        "commission_pct": 0,
        "commission_flat": 25,
        "cookie_days": 30,
    })

    programs.append({
        "name": "Anthropic Claude API",
        "category": "ai_api",
        "ref_code": "",
        "link": "https://console.anthropic.com/",
        "commission_pct": 0,
        "commission_flat": 0,
        "cookie_days": 0,
    })

    programs.append({
        "name": "OpenRouter",
        "category": "ai_api",
        "ref_code": "",
        "link": "https://openrouter.ai/",
        "commission_pct": 0,
        "commission_flat": 0,
        "cookie_days": 0,
    })

    programs.append({
        "name": "Together AI",
        "category": "ai_api",
        "ref_code": "",
        "link": "https://api.together.xyz/",
        "commission_pct": 0,
        "commission_flat": 0,
        "cookie_days": 0,
    })

    programs.append({
        "name": "Akash Network",
        "category": "compute",
        "ref_code": "",
        "link": "https://akash.network/",
        "commission_pct": 0,
        "commission_flat": 0,
        "cookie_days": 0,
    })

    return programs


def _seed_programs():
    programs = _build_default_programs()
    conn = _get_conn()
    seeded = 0
    existing = 0
    now = time.time()
    for p in programs:
        try:
            conn.execute(
                """INSERT OR IGNORE INTO affiliate_programs
                   (name, category, ref_code, link, commission_pct, commission_flat, cookie_days, created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (p["name"], p["category"], p["ref_code"], p["link"],
                 p["commission_pct"], p["commission_flat"], p["cookie_days"], now),
            )
            if conn.execute("SELECT changes()").fetchone()[0] > 0:
                seeded += 1
            else:
                existing += 1
        except Exception as exc:
            log.warning("Seed error for %s: %s", p["name"], exc)
            existing += 1
    conn.commit()
    conn.close()
    log.info("Programs seeded=%d existing=%d", seeded, existing)
    return seeded, existing


# ---------------------------------------------------------------------------
# External API helpers
# ---------------------------------------------------------------------------

def _claude_generate(system: str, user: str, max_tokens: int = 1500) -> str:
    if not ANTHROPIC_API_KEY:
        return f"[Claude API key not configured] Placeholder content for: {user[:80]}"
    payload = json.dumps({
        "model": "claude-opus-4-5",
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
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
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            return data["content"][0]["text"]
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        log.error("Claude API error %s: %s", exc.code, body)
        return f"[Claude API error {exc.code}]"
    except Exception as exc:
        log.error("Claude request failed: %s", exc)
        return f"[Claude request failed: {exc}]"


def _devto_publish(title: str, body: str, tags: list) -> tuple:
    """Returns (article_id, url) or raises."""
    if not DEVTO_API_KEY:
        raise ValueError("DEVTO_API_KEY not configured")
    payload = json.dumps({
        "article": {
            "title": title,
            "body_markdown": body,
            "published": True,
            "tags": tags[:4],
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
    return data["id"], data["url"]


def _build_affiliate_url(program: dict, utm_params: dict) -> str:
    base = program["link"] or ""
    if not base:
        return ""
    parsed = urllib.parse.urlparse(base)
    existing_qs = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    # Flatten existing params
    merged = {k: v[0] for k, v in existing_qs.items()}
    merged.update(utm_params)
    new_query = urllib.parse.urlencode(merged)
    rebuilt = urllib.parse.urlunparse((
        parsed.scheme, parsed.netloc, parsed.path,
        parsed.params, new_query, parsed.fragment,
    ))
    return rebuilt


# ---------------------------------------------------------------------------
# Content generation helper
# ---------------------------------------------------------------------------

def _generate_content(program: dict, content_type: str, target_keyword: str) -> dict:
    name = program["name"]
    link = program["link"]

    system_prompt = (
        "Write SEO content for affiliate marketing. "
        "Include natural affiliate link placement. "
        "Use clear headings, practical examples, and a compelling call to action."
    )

    if content_type == "review":
        user_prompt = (
            f"Write an honest review of {name} with pros/cons, use cases, and pricing. "
            f"Target keyword: {target_keyword}. Include a CTA at the end. "
            f"Add the affiliate sign-up link at the bottom: [Sign up here]({link})"
        )
        title = f"{name} Review: Pros, Cons & Pricing ({date.today().year})"
    elif content_type == "comparison":
        user_prompt = (
            f"Write a comparison of {name} vs its main competitors. "
            f"Target keyword: {target_keyword}. Include the affiliate link naturally. "
            f"Add: [Sign up here]({link})"
        )
        title = f"{name} vs Competitors: Full Comparison ({date.today().year})"
    elif content_type == "tutorial":
        user_prompt = (
            f"Write a step-by-step tutorial on how to use {name}. "
            f"Target keyword: {target_keyword}. Include the sign-up link. "
            f"Add: [Sign up here]({link})"
        )
        title = f"How to Use {name}: Step-by-Step Guide ({date.today().year})"
    else:
        user_prompt = (
            f"Write an SEO article about {name}. "
            f"Target keyword: {target_keyword}. "
            f"Add: [Sign up here]({link})"
        )
        title = f"{name}: Complete Guide ({date.today().year})"

    content = _claude_generate(system_prompt, user_prompt, max_tokens=1500)
    return {"title": title, "content": content}


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------

class AffiliateHandler(BaseHTTPRequestHandler):
    server_version = "FractalMesh-AffiliateEngine/1.0"

    def log_message(self, fmt, *args):
        log.info(fmt, *args)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")
        parts = [p for p in path.split("/") if p]

        if path == "/health":
            return self._json({"status": "ok", "service": "fm-affiliate-engine", "port": PORT})

        if path == "/programs":
            return self._get_programs()

        if len(parts) == 2 and parts[0] == "programs" and parts[1].isdigit():
            return self._get_program(int(parts[1]))

        if path == "/analytics":
            return self._get_analytics()

        if path == "/dashboard":
            return self._get_dashboard()

        self._json({"error": "not found"}, 404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")
        parts = [p for p in path.split("/") if p]
        body = self._read_body()

        if path == "/programs/seed":
            return self._post_seed()

        if len(parts) == 3 and parts[0] == "programs" and parts[1].isdigit() and parts[2] == "click":
            return self._post_click(int(parts[1]), body)

        if path == "/content/generate":
            return self._post_content_generate(body)

        if len(parts) == 3 and parts[0] == "content" and parts[1].isdigit() and parts[2] == "publish":
            return self._post_content_publish(int(parts[1]), body)

        if path == "/campaign/create":
            return self._post_campaign_create(body)

        if len(parts) == 3 and parts[0] == "campaign" and parts[1].isdigit() and parts[2] == "email_blast":
            return self._post_campaign_email_blast(int(parts[1]), body)

        if path == "/link/generate":
            return self._post_link_generate(body)

        self._json({"error": "not found"}, 404)

    def do_PUT(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")
        parts = [p for p in path.split("/") if p]
        body = self._read_body()

        if len(parts) == 2 and parts[0] == "programs" and parts[1].isdigit():
            return self._put_program(int(parts[1]), body)

        self._json({"error": "not found"}, 404)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length:
            raw = self.rfile.read(length)
            try:
                return json.loads(raw)
            except Exception:
                return {}
        return {}

    def _json(self, data: dict, status: int = 200):
        body = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _row_to_dict(self, row) -> dict:
        return dict(row) if row else {}

    # ------------------------------------------------------------------
    # GET /programs
    # ------------------------------------------------------------------

    def _get_programs(self):
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM affiliate_programs ORDER BY revenue_usd DESC, clicks DESC"
        ).fetchall()
        conn.close()
        self._json([self._row_to_dict(r) for r in rows])

    # ------------------------------------------------------------------
    # GET /programs/{id}
    # ------------------------------------------------------------------

    def _get_program(self, program_id: int):
        conn = _get_conn()
        row = conn.execute(
            "SELECT * FROM affiliate_programs WHERE id=?", (program_id,)
        ).fetchone()
        conn.close()
        if not row:
            return self._json({"error": "program not found"}, 404)
        self._json(self._row_to_dict(row))

    # ------------------------------------------------------------------
    # POST /programs/seed
    # ------------------------------------------------------------------

    def _post_seed(self):
        seeded, existing = _seed_programs()
        self._json({"seeded": seeded, "existing": existing})

    # ------------------------------------------------------------------
    # PUT /programs/{id}
    # ------------------------------------------------------------------

    def _put_program(self, program_id: int, body: dict):
        conn = _get_conn()
        row = conn.execute(
            "SELECT * FROM affiliate_programs WHERE id=?", (program_id,)
        ).fetchone()
        if not row:
            conn.close()
            return self._json({"error": "program not found"}, 404)

        updates = []
        params = []

        if "clicks" in body:
            updates.append("clicks = clicks + ?")
            params.append(int(body["clicks"]))
        if "conversions" in body:
            updates.append("conversions = conversions + ?")
            params.append(int(body["conversions"]))
        if "revenue_usd" in body:
            updates.append("revenue_usd = revenue_usd + ?")
            params.append(float(body["revenue_usd"]))

        if updates:
            params.append(program_id)
            conn.execute(
                f"UPDATE affiliate_programs SET {', '.join(updates)} WHERE id=?",
                params,
            )
            conn.commit()

        updated = conn.execute(
            "SELECT * FROM affiliate_programs WHERE id=?", (program_id,)
        ).fetchone()
        conn.close()
        self._json(self._row_to_dict(updated))

    # ------------------------------------------------------------------
    # POST /programs/{id}/click
    # ------------------------------------------------------------------

    def _post_click(self, program_id: int, body: dict):
        conn = _get_conn()
        row = conn.execute(
            "SELECT * FROM affiliate_programs WHERE id=?", (program_id,)
        ).fetchone()
        if not row:
            conn.close()
            return self._json({"error": "program not found"}, 404)

        referrer = body.get("referrer", "")
        ip_hash  = body.get("ip_hash", "")
        now      = time.time()

        conn.execute(
            "INSERT INTO affiliate_clicks (program_id, referrer, ip_hash, created_at) VALUES (?,?,?,?)",
            (program_id, referrer, ip_hash, now),
        )
        conn.execute(
            "UPDATE affiliate_programs SET clicks = clicks + 1 WHERE id=?",
            (program_id,),
        )
        conn.commit()

        updated = conn.execute(
            "SELECT * FROM affiliate_programs WHERE id=?", (program_id,)
        ).fetchone()
        conn.close()

        self._json({
            "redirect_url": dict(row)["link"],
            "program": self._row_to_dict(updated),
        })

    # ------------------------------------------------------------------
    # POST /content/generate
    # ------------------------------------------------------------------

    def _post_content_generate(self, body: dict):
        program_id   = body.get("program_id")
        content_type = body.get("content_type", "article")
        keyword      = body.get("target_keyword", "")

        if not program_id:
            return self._json({"error": "program_id required"}, 400)

        conn = _get_conn()
        row = conn.execute(
            "SELECT * FROM affiliate_programs WHERE id=?", (program_id,)
        ).fetchone()
        if not row:
            conn.close()
            return self._json({"error": "program not found"}, 404)

        program = self._row_to_dict(row)
        generated = _generate_content(program, content_type, keyword)
        title   = generated["title"]
        content = generated["content"]
        now     = time.time()

        cur = conn.execute(
            """INSERT INTO affiliate_content
               (program_id, title, platform, content, created_at)
               VALUES (?,?,?,?,?)""",
            (program_id, title, "draft", content, now),
        )
        content_id = cur.lastrowid
        conn.commit()
        conn.close()

        word_count = len(content.split())
        self._json({"content_id": content_id, "title": title, "word_count": word_count})

    # ------------------------------------------------------------------
    # POST /content/{id}/publish
    # ------------------------------------------------------------------

    def _post_content_publish(self, content_id: int, body: dict):
        platform = body.get("platform", "devto")
        tags     = body.get("tags", [])

        conn = _get_conn()
        row = conn.execute(
            "SELECT * FROM affiliate_content WHERE id=?", (content_id,)
        ).fetchone()
        if not row:
            conn.close()
            return self._json({"error": "content not found"}, 404)

        item = self._row_to_dict(row)

        if platform == "devto":
            try:
                article_id, url = _devto_publish(item["title"], item["content"], tags)
                conn.execute(
                    "UPDATE affiliate_content SET platform=?, published_url=? WHERE id=?",
                    ("devto", url, content_id),
                )
                conn.commit()
                conn.close()
                self._json({"published": True, "url": url, "article_id": article_id})
            except Exception as exc:
                conn.close()
                log.error("Dev.to publish failed: %s", exc)
                self._json({"published": False, "error": str(exc)}, 500)
        else:
            conn.close()
            self._json({"error": f"platform '{platform}' not supported"}, 400)

    # ------------------------------------------------------------------
    # POST /campaign/create
    # ------------------------------------------------------------------

    def _post_campaign_create(self, body: dict):
        name         = body.get("name", "Unnamed Campaign")
        program_ids  = body.get("programs", [])
        content_type = body.get("content_type", "review")

        if not program_ids:
            return self._json({"error": "programs list required"}, 400)

        conn = _get_conn()
        now = time.time()
        cur = conn.execute(
            """INSERT INTO affiliate_campaigns
               (name, programs, content_type, status, created_at)
               VALUES (?,?,?,?,?)""",
            (name, json.dumps(program_ids), content_type, "active", now),
        )
        campaign_id = cur.lastrowid
        conn.commit()

        content_ids = []
        for pid in program_ids:
            row = conn.execute(
                "SELECT * FROM affiliate_programs WHERE id=?", (pid,)
            ).fetchone()
            if not row:
                continue
            program  = self._row_to_dict(row)
            keyword  = f"{program['name']} {content_type}"
            generated = _generate_content(program, content_type, keyword)
            cur2 = conn.execute(
                """INSERT INTO affiliate_content
                   (program_id, title, platform, content, created_at)
                   VALUES (?,?,?,?,?)""",
                (pid, generated["title"], "draft", generated["content"], time.time()),
            )
            content_ids.append(cur2.lastrowid)
            conn.commit()

        conn.close()
        self._json({
            "campaign_id": campaign_id,
            "content_generated": len(content_ids),
            "content_ids": content_ids,
        })

    # ------------------------------------------------------------------
    # POST /campaign/{id}/email_blast
    # ------------------------------------------------------------------

    def _post_campaign_email_blast(self, campaign_id: int, body: dict):
        subject = body.get("subject", "FractalMesh Affiliate Picks")
        segment = body.get("segment", "all_leads")

        conn = _get_conn()

        # Fetch campaign
        camp_row = conn.execute(
            "SELECT * FROM affiliate_campaigns WHERE id=?", (campaign_id,)
        ).fetchone()
        if not camp_row:
            conn.close()
            return self._json({"error": "campaign not found"}, 404)

        camp = self._row_to_dict(camp_row)
        program_ids = json.loads(camp.get("programs", "[]"))

        # Fetch content for this campaign's programs
        content_ids = []
        email_body_parts = [f"<h2>{subject}</h2>"]
        for pid in program_ids:
            content_row = conn.execute(
                "SELECT * FROM affiliate_content WHERE program_id=? ORDER BY created_at DESC LIMIT 1",
                (pid,),
            ).fetchone()
            if content_row:
                item = self._row_to_dict(content_row)
                content_ids.append(item["id"])
                email_body_parts.append(f"<h3>{item['title']}</h3>")
                # Convert markdown to basic HTML for email
                preview = item["content"][:400].replace("\n", "<br>")
                email_body_parts.append(f"<p>{preview}...</p>")

        email_html = "\n".join(email_body_parts)

        # Fetch leads (best-effort — table may not exist)
        leads = []
        try:
            leads = conn.execute(
                "SELECT email FROM leads WHERE status='active' LIMIT 1000"
            ).fetchall()
        except Exception:
            pass

        sent = 0
        if not SENDGRID_API_KEY or not SENDGRID_FROM:
            conn.close()
            return self._json({
                "sent": 0,
                "content_ids": content_ids,
                "warning": "SendGrid not configured",
            })

        for lead in leads:
            to_email = lead[0] if isinstance(lead, (list, tuple)) else lead["email"]
            payload = json.dumps({
                "personalizations": [{"to": [{"email": to_email}]}],
                "from": {"email": SENDGRID_FROM},
                "subject": subject,
                "content": [{"type": "text/html", "value": email_html}],
            }).encode()
            req = urllib.request.Request(
                "https://api.sendgrid.com/v3/mail/send",
                data=payload,
                headers={
                    "Authorization": f"Bearer {SENDGRID_API_KEY}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=15):
                    sent += 1
            except Exception as exc:
                log.warning("SendGrid send to %s failed: %s", to_email, exc)

        # Update campaign leads_targeted
        conn.execute(
            "UPDATE affiliate_campaigns SET leads_targeted=? WHERE id=?",
            (sent, campaign_id),
        )
        conn.commit()
        conn.close()

        self._json({"sent": sent, "content_ids": content_ids})

    # ------------------------------------------------------------------
    # GET /analytics
    # ------------------------------------------------------------------

    def _get_analytics(self):
        conn = _get_conn()

        programs = conn.execute(
            "SELECT * FROM affiliate_programs ORDER BY revenue_usd DESC"
        ).fetchall()

        total_clicks      = 0
        total_conversions = 0
        total_revenue     = 0.0
        program_stats     = []

        for p in programs:
            d = self._row_to_dict(p)
            total_clicks      += d["clicks"]
            total_conversions += d["conversions"]
            # Revenue from recorded + estimated from commissions
            est = (d["conversions"] * d["commission_flat"]) + \
                  (d["conversions"] * d.get("commission_pct", 0) / 100 * 50)  # $50 avg sale estimate
            total_revenue += d["revenue_usd"]
            d["estimated_revenue"] = round(est, 2)
            program_stats.append(d)

        top_clicks      = sorted(program_stats, key=lambda x: x["clicks"], reverse=True)[:3]
        top_conversions = sorted(program_stats, key=lambda x: x["conversions"], reverse=True)[:3]
        top_revenue     = sorted(program_stats, key=lambda x: x["revenue_usd"], reverse=True)[:3]

        content_count = conn.execute(
            "SELECT COUNT(*) FROM affiliate_content WHERE published_url IS NOT NULL AND published_url != ''"
        ).fetchone()[0]

        conn.close()

        self._json({
            "total_programs": len(program_stats),
            "total_clicks": total_clicks,
            "total_conversions": total_conversions,
            "total_revenue_usd": round(total_revenue, 2),
            "content_published": content_count,
            "top_by_clicks": top_clicks,
            "top_by_conversions": top_conversions,
            "top_by_revenue": top_revenue,
            "all_programs": program_stats,
        })

    # ------------------------------------------------------------------
    # GET /dashboard
    # ------------------------------------------------------------------

    def _get_dashboard(self):
        conn = _get_conn()

        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()

        total_programs  = conn.execute("SELECT COUNT(*) FROM affiliate_programs").fetchone()[0]
        active_programs = conn.execute(
            "SELECT COUNT(*) FROM affiliate_programs WHERE status='active'"
        ).fetchone()[0]
        clicks_today    = conn.execute(
            "SELECT COUNT(*) FROM affiliate_clicks WHERE created_at >= ?", (today_start,)
        ).fetchone()[0]
        total_revenue   = conn.execute(
            "SELECT COALESCE(SUM(revenue_usd),0) FROM affiliate_programs"
        ).fetchone()[0]
        content_pieces  = conn.execute(
            "SELECT COUNT(*) FROM affiliate_content"
        ).fetchone()[0]

        best_row = conn.execute(
            "SELECT name, revenue_usd, clicks FROM affiliate_programs ORDER BY revenue_usd DESC, clicks DESC LIMIT 1"
        ).fetchone()
        best_performer = self._row_to_dict(best_row) if best_row else {}

        conn.close()

        self._json({
            "total_programs": total_programs,
            "active_programs": active_programs,
            "total_clicks_today": clicks_today,
            "total_revenue": round(total_revenue, 2),
            "best_performer": best_performer,
            "content_pieces": content_pieces,
        })

    # ------------------------------------------------------------------
    # POST /link/generate
    # ------------------------------------------------------------------

    def _post_link_generate(self, body: dict):
        program_id   = body.get("program_id")
        utm_source   = body.get("utm_source", "fractalmesh")
        utm_medium   = body.get("utm_medium", "web")
        utm_campaign = body.get("utm_campaign", "affiliate")

        if not program_id:
            return self._json({"error": "program_id required"}, 400)

        conn = _get_conn()
        row = conn.execute(
            "SELECT * FROM affiliate_programs WHERE id=?", (program_id,)
        ).fetchone()
        conn.close()

        if not row:
            return self._json({"error": "program not found"}, 404)

        program = self._row_to_dict(row)
        utm_params = {
            "utm_source":   utm_source,
            "utm_medium":   utm_medium,
            "utm_campaign": utm_campaign,
        }
        tracked_url  = _build_affiliate_url(program, utm_params)
        tracking_id  = hashlib.sha256(
            f"{program_id}-{utm_source}-{utm_campaign}-{time.time()}".encode()
        ).hexdigest()[:16]

        self._json({
            "tracked_url":  tracked_url,
            "program":      program,
            "tracking_id":  tracking_id,
        })


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

_server: HTTPServer | None = None


def _handle_signal(sig, frame):
    log.info("Signal %s received — shutting down", sig)
    if _server:
        _server.shutdown()


def main():
    global _server

    _init_db()
    _seed_programs()

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    _server = HTTPServer(("0.0.0.0", PORT), AffiliateHandler)
    log.info("fm_affiliate_engine listening on port %d", PORT)
    _server.serve_forever()


if __name__ == "__main__":
    main()
