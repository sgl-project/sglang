#!/usr/bin/env python3
"""
fm_canva.py — Canva Connect API Agent (Port 7789)
Auto-generates designs, exports assets, manages brand kit, and triggers
design-to-publish pipelines via Canva Connect API.
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
import urllib.parse
import urllib.error
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

# ── vault ─────────────────────────────────────────────────────────────────────
_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _line in _vault.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

# ── config ────────────────────────────────────────────────────────────────────
PORT          = int(os.getenv("CANVA_PORT", "7789"))
ROOT          = Path(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")))
DB            = ROOT / "database" / "sovereign.db"
LOG           = ROOT / "logs" / "canva.log"
CANVA_TOKEN   = os.getenv("CANVA_API_TOKEN", "")
CANVA_BASE    = "https://api.canva.com/rest/v1"
WWW_DIR       = ROOT / "www" / "assets" / "canva"
BRAND_ID      = os.getenv("CANVA_BRAND_ID", "")

ROOT.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)
DB.parent.mkdir(parents=True, exist_ok=True)
WWW_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [CANVA] %(message)s",
    handlers=[logging.FileHandler(LOG), logging.StreamHandler()],
)
log = logging.getLogger("canva")

# ── FractalMesh brand kit (fallback when no CANVA_API_TOKEN) ──────────────────
FM_BRAND = {
    "name":         "FractalMesh Omega Titan",
    "operator":     "Samuel James Hiotis",
    "abn":          "56 628 117 363",
    "primary":      "#22d3ee",
    "secondary":    "#7c3aed",
    "accent":       "#f59e0b",
    "bg":           "#0a0a0f",
    "surface":      "#111827",
    "text":         "#e2e8f0",
    "font_heading": "Space Grotesk",
    "font_body":    "Inter",
    "tagline":      "Sovereign, self-healing mesh systems.",
    "site":         "https://fractalmesh.net",
}

# ── template catalogue ────────────────────────────────────────────────────────
DESIGN_TEMPLATES = [
    {
        "id":       "social_post_square",
        "name":     "FractalMesh Social Post (1080×1080)",
        "format":   "instagram_square",
        "width":    1080, "height": 1080,
        "use_case": "social_media",
        "canva_template_id": os.getenv("CANVA_TPL_SOCIAL", ""),
    },
    {
        "id":       "blog_hero",
        "name":     "Dev.to Blog Hero (1600×840)",
        "format":   "blog_banner",
        "width":    1600, "height": 840,
        "use_case": "content",
        "canva_template_id": os.getenv("CANVA_TPL_BLOG", ""),
    },
    {
        "id":       "pitch_deck",
        "name":     "FractalMesh Pitch Deck",
        "format":   "presentation",
        "width":    1920, "height": 1080,
        "use_case": "consulting",
        "canva_template_id": os.getenv("CANVA_TPL_PITCH", ""),
    },
    {
        "id":       "product_card",
        "name":     "Product Card (800×600)",
        "format":   "card",
        "width":    800, "height": 600,
        "use_case": "marketplace",
        "canva_template_id": os.getenv("CANVA_TPL_PRODUCT", ""),
    },
]

# ── database ──────────────────────────────────────────────────────────────────

def _db_init():
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS canva_designs (
            id          INTEGER PRIMARY KEY,
            design_id   TEXT,
            template_id TEXT,
            title       TEXT,
            export_url  TEXT,
            local_path  TEXT,
            status      TEXT,
            ts          DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def _db_save(design_id: str, template_id: str, title: str,
             export_url: str, local_path: str, status: str):
    try:
        conn = sqlite3.connect(DB, timeout=5)
        conn.execute(
            "INSERT INTO canva_designs (design_id,template_id,title,export_url,local_path,status) "
            "VALUES (?,?,?,?,?,?)",
            (design_id, template_id, title, export_url, local_path, status),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning("db_save error: %s", e)

# ── Canva Connect API helpers ─────────────────────────────────────────────────

def _api(method: str, path: str, body: dict | None = None) -> dict:
    if not CANVA_TOKEN:
        raise ValueError("CANVA_API_TOKEN not configured")
    url  = f"{CANVA_BASE}/{path.lstrip('/')}"
    data = json.dumps(body).encode() if body else None
    req  = urllib.request.Request(
        url, data=data, method=method,
        headers={
            "Authorization":  f"Bearer {CANVA_TOKEN}",
            "Content-Type":   "application/json",
            "Accept":         "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())

def _create_design(template_id: str, title: str) -> dict:
    """Create a new design from a template."""
    payload = {"design_type": {"type": "preset", "name": "SocialMedia"}, "title": title}
    if template_id:
        payload["asset_id"] = template_id
    return _api("POST", "/designs", payload)

def _export_design(design_id: str, fmt: str = "png") -> dict:
    """Trigger an export job for a design."""
    return _api("POST", "/exports", {
        "design_id": design_id,
        "format":    {"type": fmt, "export_quality": "regular"},
    })

def _poll_export(export_id: str, max_wait: int = 30) -> str | None:
    """Poll export status and return download URL when ready."""
    for _ in range(max_wait):
        time.sleep(1)
        resp  = _api("GET", f"/exports/{export_id}")
        job   = resp.get("export_job", {})
        status = job.get("status")
        if status == "success":
            urls = job.get("urls", [])
            return urls[0] if urls else None
        if status == "failed":
            return None
    return None

def _download_asset(url: str, filename: str) -> str:
    """Download exported asset to www/assets/canva/."""
    dest = WWW_DIR / filename
    urllib.request.urlretrieve(url, dest)
    return str(dest)

# ── local design generation (no API key) ─────────────────────────────────────

def _generate_local_svg(template: dict, text: str = "") -> str:
    """Generate a branded SVG when Canva API is unavailable."""
    w, h = template["width"], template["height"]
    name = text or FM_BRAND["name"]
    svg  = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">
  <defs>
    <linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:{FM_BRAND['bg']}"/>
      <stop offset="100%" style="stop-color:{FM_BRAND['surface']}"/>
    </linearGradient>
  </defs>
  <rect width="{w}" height="{h}" fill="url(#g)"/>
  <rect x="0" y="{h-4}" width="{w}" height="4" fill="{FM_BRAND['primary']}"/>
  <text x="{w//2}" y="{h//2-30}" font-family="Space Grotesk,sans-serif"
        font-size="{min(w,h)//15}" fill="{FM_BRAND['primary']}"
        text-anchor="middle" font-weight="bold">{name}</text>
  <text x="{w//2}" y="{h//2+20}" font-family="Inter,sans-serif"
        font-size="{min(w,h)//30}" fill="{FM_BRAND['text']}"
        text-anchor="middle">{FM_BRAND['tagline']}</text>
  <text x="{w//2}" y="{h-20}" font-family="monospace"
        font-size="{min(w,h)//45}" fill="{FM_BRAND['primary']}88"
        text-anchor="middle">{FM_BRAND['site']} | ABN {FM_BRAND['abn']}</text>
</svg>"""
    filename  = f"{template['id']}_{int(time.time())}.svg"
    dest_path = WWW_DIR / filename
    dest_path.write_text(svg)
    return str(dest_path)

# ── orchestration ─────────────────────────────────────────────────────────────

def _create_and_export(template_id: str, title: str, text: str = "") -> dict:
    tpl = next((t for t in DESIGN_TEMPLATES if t["id"] == template_id), None)
    if not tpl:
        return {"error": f"unknown_template:{template_id}",
                "available": [t["id"] for t in DESIGN_TEMPLATES]}

    if not CANVA_TOKEN:
        local = _generate_local_svg(tpl, text)
        _db_save("local", template_id, title, "", local, "local_svg")
        log.info("local_svg template=%s path=%s", template_id, local)
        return {"status": "local_svg", "path": local, "note": "set CANVA_API_TOKEN for live designs"}

    try:
        design   = _create_design(tpl.get("canva_template_id", ""), title)
        did      = design.get("design", {}).get("id", "")
        export   = _export_design(did)
        eid      = export.get("export_job", {}).get("id", "")
        url      = _poll_export(eid)
        local    = _download_asset(url, f"{template_id}_{did}.png") if url else ""
        status   = "exported" if local else "export_pending"
        _db_save(did, template_id, title, url or "", local, status)
        log.info("canva_design id=%s status=%s", did, status)
        return {"status": status, "design_id": did, "export_url": url, "local": local}
    except Exception as e:
        log.error("canva_api error: %s", e)
        local = _generate_local_svg(tpl, text)
        _db_save("fallback", template_id, title, "", local, "fallback_svg")
        return {"status": "fallback_svg", "path": local, "error": str(e)}

def _batch_generate() -> list:
    results = []
    for tpl in DESIGN_TEMPLATES:
        r = _create_and_export(tpl["id"], f"FractalMesh {tpl['name']} {time.strftime('%Y-%m-%d')}")
        results.append(r)
    return results

# ── HTTP handler ───────────────────────────────────────────────────────────────

class CanvaHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _respond(self, code: int, body: Any):
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"status": "ok", "api_key": bool(CANVA_TOKEN),
                                "templates": len(DESIGN_TEMPLATES)})
        elif self.path == "/templates":
            self._respond(200, {"templates": DESIGN_TEMPLATES})
        elif self.path == "/brand":
            self._respond(200, FM_BRAND)
        elif self.path == "/batch":
            results = _batch_generate()
            self._respond(200, {"generated": len(results), "results": results})
        else:
            self._respond(404, {"error": "not_found"})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            data   = json.loads(self.rfile.read(length))
            template_id = data.get("template_id")
            title       = data.get("title", f"FractalMesh Design {time.strftime('%Y%m%d-%H%M%S')}")
            text        = data.get("text", "")

            if self.path == "/design":
                if not template_id:
                    self._respond(400, {"error": "template_id required",
                                        "available": [t["id"] for t in DESIGN_TEMPLATES]})
                    return
                result = _create_and_export(template_id, title, text)
                self._respond(200, result)

            elif self.path == "/batch":
                results = _batch_generate()
                self._respond(200, {"generated": len(results), "results": results})

            else:
                self._respond(404, {"error": "unknown_path"})

        except json.JSONDecodeError:
            self._respond(400, {"error": "invalid_json"})
        except Exception as e:
            log.error("handler_error: %s", e)
            self._respond(500, {"error": str(e)})

# ── main ───────────────────────────────────────────────────────────────────────

_running = True

def _shutdown(*_):
    global _running
    log.info("shutdown signal — exiting cleanly")
    _running = False

signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT,  _shutdown)

def main():
    _db_init()
    server = HTTPServer(("0.0.0.0", PORT), CanvaHandler)
    log.info("Canva agent listening on port %d", PORT)
    log.info("API key: %s | Templates: %d | Brand: FractalMesh Omega Titan",
             "configured" if CANVA_TOKEN else "fallback_mode", len(DESIGN_TEMPLATES))
    try:
        while _running:
            server.handle_request()
    finally:
        server.server_close()
        log.info("Canva agent stopped")

if __name__ == "__main__":
    main()
