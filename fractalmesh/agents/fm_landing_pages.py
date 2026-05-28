#!/usr/bin/env python3
"""
fm_landing_pages.py — Landing Page Builder & Funnel Manager (Port 7900)
FractalMesh OMEGA Titan | Samuel James Hiotis | ABN 56 628 117 363
Credentials sourced from ~/.secrets/fractal.env — never hardcoded.
"""
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# ── vault ─────────────────────────────────────────────────────────────────────
_ENV_FILE = Path.home() / ".secrets" / "fractal.env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

PORT         = int(os.getenv("LANDING_PAGES_PORT", "7900"))
SG_KEY       = os.getenv("SENDGRID_API_KEY", "")
SG_FROM      = os.getenv("SENDGRID_FROM_EMAIL", "noreply@fractalmesh.ai")
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")
BASE_URL     = os.getenv("BASE_URL", "https://fractalmesh.ai")

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
        CREATE TABLE IF NOT EXISTS landing_pages (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            page_id      TEXT UNIQUE NOT NULL,
            slug         TEXT UNIQUE NOT NULL,
            title        TEXT NOT NULL,
            headline     TEXT NOT NULL,
            subheadline  TEXT,
            body_html    TEXT NOT NULL DEFAULT '',
            cta_text     TEXT DEFAULT 'Get Started',
            cta_url      TEXT,
            template     TEXT DEFAULT 'default',
            status       TEXT DEFAULT 'draft',
            views        INTEGER DEFAULT 0,
            conversions  INTEGER DEFAULT 0,
            meta_title   TEXT,
            meta_desc    TEXT,
            og_image     TEXT,
            custom_css   TEXT,
            created_at   REAL NOT NULL,
            updated_at   REAL NOT NULL,
            published_at REAL
        );
        CREATE TABLE IF NOT EXISTS page_views (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            page_id      TEXT NOT NULL,
            ip_hash      TEXT,
            referrer     TEXT,
            user_agent   TEXT,
            created_at   REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS leads (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            lead_id      TEXT UNIQUE NOT NULL,
            page_id      TEXT NOT NULL,
            email        TEXT NOT NULL,
            name         TEXT,
            phone        TEXT,
            custom_data  TEXT DEFAULT '{}',
            ip_hash      TEXT,
            status       TEXT DEFAULT 'new',
            created_at   REAL NOT NULL,
            updated_at   REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS funnels (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            funnel_id    TEXT UNIQUE NOT NULL,
            name         TEXT NOT NULL,
            steps        TEXT NOT NULL DEFAULT '[]',
            active       INTEGER DEFAULT 1,
            created_at   REAL NOT NULL,
            updated_at   REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS funnel_steps (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            step_id      TEXT UNIQUE NOT NULL,
            funnel_id    TEXT NOT NULL,
            page_id      TEXT NOT NULL,
            step_order   INTEGER NOT NULL,
            step_name    TEXT NOT NULL,
            created_at   REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_leads_page    ON leads(page_id);
        CREATE INDEX IF NOT EXISTS idx_leads_email   ON leads(email);
        CREATE INDEX IF NOT EXISTS idx_views_page    ON page_views(page_id);
    """)
    con.commit()
    _seed_pages(con)
    con.close()

_DEFAULT_HTML = """<section style="max-width:800px;margin:0 auto;padding:40px 20px;font-family:sans-serif">
  <h1 style="font-size:2.5rem;color:#1a1a2e">{headline}</h1>
  <p style="font-size:1.2rem;color:#555;margin:20px 0">{subheadline}</p>
  {body}
  <div style="margin:40px 0">
    <a href="{cta_url}" style="background:#6366f1;color:#fff;padding:16px 32px;border-radius:8px;
       text-decoration:none;font-size:1.1rem;font-weight:bold">{cta_text}</a>
  </div>
  <form id="lead-form" style="background:#f8f9fa;padding:30px;border-radius:12px;margin-top:40px">
    <h3 style="margin:0 0 20px">Get in Touch</h3>
    <input name="email" type="email" placeholder="Your email" required
           style="width:100%;padding:12px;border:1px solid #ddd;border-radius:6px;margin-bottom:12px;box-sizing:border-box">
    <input name="name" placeholder="Your name"
           style="width:100%;padding:12px;border:1px solid #ddd;border-radius:6px;margin-bottom:12px;box-sizing:border-box">
    <button type="submit"
            style="background:#6366f1;color:#fff;padding:12px 24px;border:none;border-radius:6px;cursor:pointer;width:100%">
      Submit
    </button>
  </form>
</section>"""

def _seed_pages(con):
    if con.execute("SELECT COUNT(*) FROM landing_pages").fetchone()[0] > 0:
        return
    now = time.time()
    con.execute(
        "INSERT INTO landing_pages(page_id,slug,title,headline,subheadline,body_html,cta_text,cta_url,"
        "template,status,created_at,updated_at,published_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("pg_welcome", "welcome", "FractalMesh — Sovereign AI Automation",
         "The Most Powerful AI Automation Mesh",
         "127+ autonomous agents working together to scale your business",
         "<p>FractalMesh OMEGA Titan delivers enterprise-grade AI automation for Australian businesses.</p>",
         "Start Free Trial", f"{BASE_URL}/signup",
         "default", "published", now, now, now)
    )
    con.commit()

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

def _render_page(page):
    body_html = _DEFAULT_HTML.format(
        headline=page["headline"],
        subheadline=page["subheadline"] or "",
        body=page["body_html"] or "",
        cta_text=page["cta_text"] or "Get Started",
        cta_url=page["cta_url"] or "#",
    )
    css = page["custom_css"] or ""
    meta_title = page["meta_title"] or page["title"]
    meta_desc = page["meta_desc"] or page["subheadline"] or ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{meta_title}</title>
  <meta name="description" content="{meta_desc}">
  <style>*{{box-sizing:border-box}}body{{margin:0;background:#fff}}{css}</style>
</head>
<body>
{body_html}
<script>
document.getElementById('lead-form').addEventListener('submit', async function(e) {{
  e.preventDefault();
  const data = Object.fromEntries(new FormData(e.target));
  await fetch('/leads', {{method:'POST', headers:{{'Content-Type':'application/json'}},
    body: JSON.stringify({{...data, page_id: '{page["page_id"]}'}})}});
  e.target.innerHTML = '<p style="color:green;text-align:center">Thank you! We\\'ll be in touch.</p>';
}});
</script>
</body>
</html>"""

class LPHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _read_body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n)) if n else {}

    def _send(self, code, body, ct="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,PUT,DELETE,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type,X-Admin-Secret")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        p = parsed.path.strip("/").split("/")
        qs = parse_qs(parsed.query)
        try:
            code, body, ct = self._get(p, qs)
        except Exception as e:
            code, body, ct = 500, json.dumps({"error": str(e)}).encode(), "application/json"
        self._send(code, body, ct)

    def do_POST(self):
        p = self.path.strip("/").split("/")
        try:
            data = self._read_body()
            code, body = self._post(p, data)
        except Exception as e:
            code, body = _err(str(e), 500)
        self._send(code, body)

    def _get(self, p, qs):
        con = _db()
        try:
            def j(data, s=200):
                return s, json.dumps(data, default=str).encode(), "application/json"
            def e(msg, s=400):
                return j({"error": msg}, s)

            if p == ["health"]:
                return j({"status": "ok", "port": PORT, "agent": "fm_landing_pages"})

            if p == ["pages"]:
                if not _admin(self.headers):
                    return e("Unauthorized", 403)
                rows = con.execute("SELECT * FROM landing_pages ORDER BY updated_at DESC").fetchall()
                return j([dict(r) for r in rows])

            if len(p) == 2 and p[0] == "pages":
                row = con.execute("SELECT * FROM landing_pages WHERE page_id=?", (p[1],)).fetchone()
                if not row:
                    return e("Page not found", 404)
                return j(dict(row))

            if len(p) == 2 and p[0] == "pages" and False:  # placeholder
                pass

            # serve rendered HTML page by slug
            if len(p) == 2 and p[0] == "p":
                row = con.execute(
                    "SELECT * FROM landing_pages WHERE slug=? AND status='published'", (p[1],)
                ).fetchone()
                if not row:
                    html = b"<h1>Page Not Found</h1>"
                    return 404, html, "text/html"
                # record view
                ip = self.client_address[0]
                con.execute(
                    "INSERT INTO page_views(page_id,ip_hash,referrer,user_agent,created_at) VALUES(?,?,?,?,?)",
                    (row["page_id"], hashlib.sha256(ip.encode()).hexdigest()[:16],
                     self.headers.get("Referer",""), self.headers.get("User-Agent",""), time.time())
                )
                con.execute("UPDATE landing_pages SET views=views+1 WHERE page_id=?", (row["page_id"],))
                con.commit()
                html = _render_page(dict(row)).encode()
                return 200, html, "text/html"

            if p == ["leads"]:
                if not _admin(self.headers):
                    return e("Unauthorized", 403)
                page_id = qs.get("page_id", [None])[0]
                status = qs.get("status", [None])[0]
                q = "SELECT * FROM leads WHERE 1=1"
                vals = []
                if page_id:
                    q += " AND page_id=?"; vals.append(page_id)
                if status:
                    q += " AND status=?"; vals.append(status)
                q += " ORDER BY created_at DESC LIMIT 200"
                rows = con.execute(q, vals).fetchall()
                return j([dict(r) for r in rows])

            if p == ["analytics"]:
                if not _admin(self.headers):
                    return e("Unauthorized", 403)
                pages = con.execute(
                    "SELECT page_id, slug, title, views, conversions, "
                    "CASE WHEN views>0 THEN ROUND(CAST(conversions AS REAL)/views*100,1) ELSE 0 END as cvr "
                    "FROM landing_pages ORDER BY views DESC"
                ).fetchall()
                total_leads = con.execute("SELECT COUNT(*) FROM leads").fetchone()[0]
                return j({"pages": [dict(r) for r in pages], "total_leads": total_leads})

            if p == ["funnels"]:
                rows = con.execute("SELECT * FROM funnels WHERE active=1").fetchall()
                return j([dict(r) for r in rows])

            return e("Not found", 404)
        finally:
            con.close()

    def _post(self, p, data):
        con = _db()
        now = time.time()
        try:
            if p == ["pages"]:
                if not _admin(self.headers):
                    return _err("Unauthorized", 403)
                pid = "pg_" + secrets.token_hex(8)
                slug = data.get("slug", secrets.token_urlsafe(6).lower())
                con.execute(
                    "INSERT INTO landing_pages(page_id,slug,title,headline,subheadline,body_html,"
                    "cta_text,cta_url,template,status,meta_title,meta_desc,og_image,custom_css,created_at,updated_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (pid, slug, data.get("title",""), data.get("headline",""),
                     data.get("subheadline"), data.get("body_html",""),
                     data.get("cta_text","Get Started"), data.get("cta_url",""),
                     data.get("template","default"), data.get("status","draft"),
                     data.get("meta_title"), data.get("meta_desc"),
                     data.get("og_image"), data.get("custom_css"), now, now)
                )
                con.commit()
                return _j({"page_id": pid, "slug": slug, "url": f"{BASE_URL}/p/{slug}"}, 201)

            if len(p) == 3 and p[0] == "pages" and p[2] == "publish":
                if not _admin(self.headers):
                    return _err("Unauthorized", 403)
                con.execute(
                    "UPDATE landing_pages SET status='published', published_at=?, updated_at=? WHERE page_id=?",
                    (now, now, p[1])
                )
                con.commit()
                row = con.execute("SELECT slug FROM landing_pages WHERE page_id=?", (p[1],)).fetchone()
                return _j({"page_id": p[1], "status": "published",
                           "url": f"{BASE_URL}/p/{row['slug']}"})

            if p == ["leads"]:
                email = data.get("email", "")
                if not email:
                    return _err("email required")
                lid = "lead_" + secrets.token_hex(8)
                ip = self.client_address[0]
                con.execute(
                    "INSERT INTO leads(lead_id,page_id,email,name,phone,custom_data,ip_hash,created_at,updated_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?)",
                    (lid, data.get("page_id",""), email, data.get("name"),
                     data.get("phone"), json.dumps(data.get("custom_data",{})),
                     hashlib.sha256(ip.encode()).hexdigest()[:16], now, now)
                )
                con.execute(
                    "UPDATE landing_pages SET conversions=conversions+1 WHERE page_id=?",
                    (data.get("page_id",""),)
                )
                con.commit()
                threading.Thread(target=_send_email, args=(
                    email, "Welcome!",
                    f"<p>Hi {data.get('name','there')}, thanks for your interest!</p>"
                ), daemon=True).start()
                return _j({"lead_id": lid, "status": "captured"}, 201)

            if p == ["funnels"]:
                if not _admin(self.headers):
                    return _err("Unauthorized", 403)
                fid = "fn_" + secrets.token_hex(8)
                con.execute(
                    "INSERT INTO funnels(funnel_id,name,steps,created_at,updated_at) VALUES(?,?,?,?,?)",
                    (fid, data.get("name",""), json.dumps(data.get("steps",[])), now, now)
                )
                con.commit()
                return _j({"funnel_id": fid}, 201)

            return _err("Not found", 404)
        finally:
            con.close()


def run():
    init_db()
    server = HTTPServer(("0.0.0.0", PORT), LPHandler)
    print(f"[fm_landing_pages] listening on port {PORT}")
    server.serve_forever()


if __name__ == "__main__":
    run()
