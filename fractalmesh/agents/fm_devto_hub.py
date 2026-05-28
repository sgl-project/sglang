#!/usr/bin/env python3
"""
fm_devto_hub.py — Dev.to / Dev.community Full Integration Hub (Port 7792)
Article publishing, analytics, reaction tracking, comment management,
follower growth, series management, and OpenRouter-powered content generation.
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
from typing import Any

# ── vault ─────────────────────────────────────────────────────────────────────
_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _line in _vault.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

# ── config ────────────────────────────────────────────────────────────────────
PORT          = int(os.getenv("DEVTO_HUB_PORT", "7792"))
ROOT          = Path(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")))
DB            = ROOT / "database" / "sovereign.db"
LOG           = ROOT / "logs" / "devto_hub.log"
DEVTO_KEY     = os.getenv("DEVTO_API_KEY", "")
OR_URL        = os.getenv("OPENROUTER_URL", "http://127.0.0.1:7791")
DEVTO_API     = "https://dev.to/api"
OPERATOR      = "Samuel James Hiotis"
ABN           = "56 628 117 363"
SITE          = os.getenv("SITE_URL", "https://fractalmesh.net")
MANUS_REF     = os.getenv("MANUS_REF_CODE", "XDCMWO3VETC7FV")
DRY_RUN       = os.getenv("DEVTO_PUBLISH_LIVE", "false").lower() != "true"

ROOT.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)
DB.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [DEVTO] %(message)s",
    handlers=[logging.FileHandler(LOG), logging.StreamHandler()],
)
log = logging.getLogger("devto_hub")

# ── article series ─────────────────────────────────────────────────────────────
SERIES = {
    "fractalmesh_build": {
        "name":        "Building FractalMesh from an Android Phone",
        "description": "End-to-end journey of building a sovereign AI mesh on Termux/ARM64.",
        "tags":        ["ai", "android", "devops", "automation"],
        "articles": [
            "How I run 32 AI agents on a Samsung phone — no cloud required",
            "Termux as a production server: SQLite, Flask, and PM2 on Android",
            "Building an MCP intent router in 250 lines of Python",
            "Zero-capital monetization: my first $100 from sovereign AI",
            "Cloudflare tunnels + Termux: exposing local agents to the world",
        ],
    },
    "zero_capital_ai": {
        "name":        "Zero-Capital AI Business",
        "description": "Practical strategies for building revenue with AI tools and no startup capital.",
        "tags":        ["business", "ai", "freelance", "solopreneur"],
        "articles": [
            "500 ways to make money with AI and zero dollars",
            "Hugging Face free tier: run inference without an API bill",
            "OpenRouter model routing: getting GPT-4 quality at Mistral prices",
            "Dev.to as a revenue channel: affiliate + consulting funnel",
            "Canva + AI: design automation for solo operators",
        ],
    },
    "depin_sovereignty": {
        "name":        "DePIN & Digital Sovereignty",
        "description": "Decentralised infrastructure, RF intelligence, and data monetisation.",
        "tags":        ["web3", "depin", "infrastructure", "blockchain"],
        "articles": [
            "WiGLE wardriving: get paid to map WiFi networks",
            "Streamr data streams: publish sensor data, earn DATA tokens",
            "Akash Network: deploy containers without AWS",
            "Ocean Protocol datasets: monetise your data pipeline",
            "Helium mobile: run a hotspot, earn HNT",
        ],
    },
}

# ── affiliate link map ─────────────────────────────────────────────────────────
AFFILIATES = {
    "Manus":       f"https://manus.im/invitation/{MANUS_REF}",
    "Together AI": "https://api.together.ai",
    "Hugging Face":"https://huggingface.co",
    "OpenRouter":  "https://openrouter.ai",
    "Vultr":       os.getenv("AFF_VULTR_URL", "https://vultr.com"),
    "BloFin":      "https://blofin.com",
}

# ── database ──────────────────────────────────────────────────────────────────

def _db_init():
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS devto_articles (
            id          INTEGER PRIMARY KEY,
            devto_id    INTEGER,
            series      TEXT,
            title       TEXT,
            slug        TEXT,
            url         TEXT,
            tags        TEXT,
            status      TEXT,
            page_views  INTEGER DEFAULT 0,
            reactions   INTEGER DEFAULT 0,
            comments    INTEGER DEFAULT 0,
            published   INTEGER DEFAULT 0,
            ts          DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS devto_analytics (
            id          INTEGER PRIMARY KEY,
            devto_id    INTEGER,
            page_views  INTEGER,
            reactions   INTEGER,
            comments    INTEGER,
            snapshot_ts DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def _db_save_article(devto_id: int | None, series: str, title: str,
                     slug: str, url: str, tags: list, status: str):
    try:
        conn = sqlite3.connect(DB, timeout=5)
        conn.execute(
            "INSERT INTO devto_articles (devto_id,series,title,slug,url,tags,status,published) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (devto_id, series, title, slug, url, json.dumps(tags),
             status, 1 if status == "published" else 0),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning("db_save error: %s", e)

def _db_get_published() -> list:
    try:
        conn = sqlite3.connect(DB, timeout=5)
        rows = conn.execute(
            "SELECT devto_id,title,url,series,page_views,reactions,comments,ts "
            "FROM devto_articles WHERE published=1 ORDER BY ts DESC LIMIT 50"
        ).fetchall()
        conn.close()
        return [{"id": r[0], "title": r[1], "url": r[2], "series": r[3],
                 "views": r[4], "reactions": r[5], "comments": r[6], "ts": r[7]}
                for r in rows]
    except Exception:
        return []

# ── Dev.to API ────────────────────────────────────────────────────────────────

def _api(method: str, path: str, body: dict | None = None) -> dict:
    url  = f"{DEVTO_API}/{path.lstrip('/')}"
    data = json.dumps(body).encode() if body else None
    headers = {"Accept": "application/vnd.forem.api-v1+json"}
    if DEVTO_KEY:
        headers["api-key"] = DEVTO_KEY
    if data:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": f"http_{e.code}", "detail": e.read().decode("utf-8", errors="replace")[:300]}
    except Exception as e:
        return {"error": str(e)}

def _get_my_articles(page: int = 1, per_page: int = 30) -> list:
    result = _api("GET", f"articles/me?page={page}&per_page={per_page}")
    return result if isinstance(result, list) else []

def _get_article(article_id: int) -> dict:
    return _api("GET", f"articles/{article_id}")

def _get_trending(tag: str = "ai") -> list:
    result = _api("GET", f"articles?tag={tag}&top=1&per_page=10")
    return result if isinstance(result, list) else []

def _get_comments(article_id: int) -> list:
    result = _api("GET", f"comments?a_id={article_id}")
    return result if isinstance(result, list) else []

def _publish(title: str, body: str, tags: list,
             series: str = "", published: bool = True) -> dict:
    if DRY_RUN or not DEVTO_KEY:
        log.info("dry_run title=%s tags=%s", title, tags)
        return {"status": "dry_run", "title": title, "note": "set DEVTO_PUBLISH_LIVE=true to go live"}

    payload = {
        "article": {
            "title":      title,
            "body_markdown": body,
            "published":  published,
            "tags":       tags[:4],
        }
    }
    if series:
        payload["article"]["series"] = series

    result = _api("POST", "articles", payload)
    if "id" in result:
        _db_save_article(result["id"], series, title,
                         result.get("slug", ""), result.get("url", ""),
                         tags, "published" if published else "draft")
    return result

def _update_article(article_id: int, body_update: dict) -> dict:
    return _api("PUT", f"articles/{article_id}", {"article": body_update})

def _sync_analytics() -> list:
    """Pull view/reaction/comment stats for all published articles."""
    articles = _get_my_articles()
    updates  = []
    for a in articles:
        aid   = a.get("id")
        views = a.get("page_views_count", 0)
        reax  = a.get("positive_reactions_count", 0)
        comms = a.get("comments_count", 0)
        try:
            conn = sqlite3.connect(DB, timeout=5)
            conn.execute(
                "UPDATE devto_articles SET page_views=?,reactions=?,comments=?,updated_at=CURRENT_TIMESTAMP WHERE devto_id=?",
                (views, reax, comms, aid)
            )
            conn.execute(
                "INSERT INTO devto_analytics (devto_id,page_views,reactions,comments) VALUES (?,?,?,?)",
                (aid, views, reax, comms)
            )
            conn.commit()
            conn.close()
        except Exception:
            pass
        updates.append({"id": aid, "title": a.get("title"), "views": views,
                        "reactions": reax, "comments": comms})
    log.info("analytics_sync articles=%d", len(updates))
    return updates

# ── AI content generation via OpenRouter ─────────────────────────────────────

def _generate_article(title: str, tags: list, series: str) -> str:
    """Generate article body via local OpenRouter agent."""
    aff_block = "\n".join(f"- [{k}]({v})" for k, v in AFFILIATES.items())
    system    = (
        "You are a technical blogger writing for the Dev.to developer community. "
        "Write practical, first-person, code-rich content. No fluff. ~700 words. "
        "Use markdown with headers, code blocks, and bullet points."
    )
    prompt = (
        f"Write a Dev.to article titled: \"{title}\"\n\n"
        f"Series: {series}\n"
        f"Tags: {', '.join(tags)}\n\n"
        f"Naturally embed relevant affiliate links from:\n{aff_block}\n\n"
        f"End with author bio: **{OPERATOR}** | ABN {ABN} | [{SITE}]({SITE})"
    )
    try:
        payload = json.dumps({
            "task": "draft", "tier": "balanced",
            "prompt": prompt, "system": system,
            "max_tokens": 1200, "cache": True,
        }).encode()
        req = urllib.request.Request(
            f"{OR_URL}/route", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            resp = json.loads(r.read())
        return resp.get("content", "")
    except Exception as e:
        log.warning("openrouter_gen failed: %s — using stub", e)
        return (
            f"# {title}\n\n"
            f"*Generated stub — configure OpenRouter agent for AI content.*\n\n"
            f"This article covers: **{title}**\n\n"
            f"---\n*{OPERATOR} | ABN {ABN} | [{SITE}]({SITE})*"
        )

# ── HTTP handler ───────────────────────────────────────────────────────────────

class DevToHandler(BaseHTTPRequestHandler):
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
            self._respond(200, {"status": "ok", "api_key": bool(DEVTO_KEY),
                                "dry_run": DRY_RUN, "series": list(SERIES.keys())})
        elif self.path == "/articles":
            self._respond(200, {"articles": _db_get_published()})
        elif self.path == "/articles/live":
            self._respond(200, {"articles": _get_my_articles()})
        elif self.path == "/series":
            self._respond(200, SERIES)
        elif self.path == "/analytics":
            updates = _sync_analytics()
            self._respond(200, {"synced": len(updates), "articles": updates})
        elif self.path.startswith("/trending/"):
            tag = self.path.split("/")[-1] or "ai"
            self._respond(200, {"tag": tag, "articles": _get_trending(tag)})
        else:
            self._respond(404, {"error": "not_found"})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            data   = json.loads(self.rfile.read(length))

            if self.path == "/publish":
                title     = data.get("title", "")
                body      = data.get("body", "")
                tags      = data.get("tags", ["ai"])
                series    = data.get("series", "")
                published = data.get("published", True)
                if not title:
                    self._respond(400, {"error": "title required"})
                    return
                if not body:
                    body = _generate_article(title, tags, series)
                result = _publish(title, body, tags, series, published)
                self._respond(200, result)

            elif self.path == "/generate":
                series_key = data.get("series", "fractalmesh_build")
                s_config   = SERIES.get(series_key)
                if not s_config:
                    self._respond(404, {"error": "series_not_found",
                                        "available": list(SERIES.keys())})
                    return
                published_titles = {a["title"] for a in _db_get_published()}
                pending = [t for t in s_config["articles"] if t not in published_titles]
                if not pending:
                    self._respond(200, {"note": "all_articles_published", "series": series_key})
                    return
                title = pending[0]
                body  = _generate_article(title, s_config["tags"], s_config["name"])
                result = _publish(title, body, s_config["tags"], s_config["name"])
                self._respond(200, {"series": series_key, "title": title, "result": result})

            elif self.path == "/update":
                aid     = data.get("article_id")
                updates = {k: v for k, v in data.items() if k != "article_id"}
                if not aid:
                    self._respond(400, {"error": "article_id required"})
                    return
                result = _update_article(int(aid), updates)
                self._respond(200, result)

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
    server = HTTPServer(("0.0.0.0", PORT), DevToHandler)
    total_articles = sum(len(s["articles"]) for s in SERIES.values())
    log.info("Dev.to Hub listening on port %d", PORT)
    log.info("API key: %s | DryRun: %s | Series: %d | Planned articles: %d",
             "configured" if DEVTO_KEY else "none", DRY_RUN,
             len(SERIES), total_articles)
    try:
        while _running:
            server.handle_request()
    finally:
        server.server_close()
        log.info("Dev.to Hub stopped")

if __name__ == "__main__":
    main()
