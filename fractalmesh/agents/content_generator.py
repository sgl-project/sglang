"""
FractalMesh Content Generator v2.1.0
GPT-4o-mini article generation with affiliate link embedding.
Posts to dev.to; stores content in sovereign.db content_pieces table.
Samuel James Hiotis | ABN 56 628 117 363 | Sole Trader
"""
import os
import json
import time
import signal
import sqlite3
import urllib.request
import urllib.parse
from datetime import datetime, timezone

ROOT     = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB       = os.path.join(ROOT, "database", "sovereign.db")
INTERVAL = int(os.getenv("CONTENT_INTERVAL", "86400"))   # 24-hour default
DRY_RUN  = os.getenv("ENABLE_CONTENT", "false").lower() != "true"

OPENAI_KEY  = os.getenv("OPENAI_API_KEY", "")
DEVTO_KEY   = os.getenv("DEVTO_API_KEY", "")
OPERATOR    = "Samuel James Hiotis"
ABN         = "56 628 117 363"
SITE        = "https://fractalmesh.net"

PHI      = 1.6180339887
_running = True

TOPICS = [
    {
        "topic":      "How I run AI agents on an Android phone with Termux",
        "affiliates": ["Vultr", "Cursor AI"],
        "tags":       ["android", "ai", "termux", "automation"],
    },
    {
        "topic":      "Manus.im first look — autonomous AI agents, no code",
        "affiliates": ["Manus.im"],
        "tags":       ["ai", "agents", "productivity", "nocode"],
    },
    {
        "topic":      "Stop paying cloud tax — host AI on VPS for A$6/month",
        "affiliates": ["Vultr", "Hostinger", "DigitalOcean"],
        "tags":       ["vps", "selfhosted", "ai", "cloud"],
    },
    {
        "topic":      "Best tools for Australian solo operators building AI",
        "affiliates": ["Notion", "Gumroad", "Fiverr"],
        "tags":       ["australia", "solopreneur", "tools", "ai"],
    },
    {
        "topic":      "From construction to autonomous AI — 2 years on Android",
        "affiliates": ["Manus.im", "Vultr"],
        "tags":       ["story", "ai", "android", "journey"],
    },
]

# Affiliate ref URLs populated from DB at runtime; fallbacks here
AFF_FALLBACK = {
    "Manus.im":     "https://manus.im/invitation/XDCMWO3VETC7FV",
    "Vultr":        os.getenv("AFF_VULTR_URL",    "https://vultr.com"),
    "DigitalOcean": os.getenv("AFF_DO_URL",        "https://m.do.co/c/"),
    "Hostinger":    "https://hostinger.com",
    "Cursor AI":    "https://cursor.sh",
    "Notion":       "https://notion.so",
    "Perplexity AI":"https://perplexity.ai",
    "Fiverr":       "https://go.fiverr.com",
    "Gumroad":      "https://gumroad.com/affiliates",
    "Namecheap":    "https://namecheap.com",
}


def _aff_urls() -> dict:
    """Load ref_urls from DB, falling back to AFF_FALLBACK."""
    urls = dict(AFF_FALLBACK)
    try:
        conn = sqlite3.connect(DB, timeout=10)
        rows = conn.execute(
            "SELECT program, ref_url FROM affiliates WHERE status='active'").fetchall()
        conn.close()
        for prog, url in rows:
            if url:
                urls[prog] = url
    except Exception:
        pass
    return urls


def _gpt_article(topic: str, affiliates: list, aff_urls: dict) -> str:
    """Generate article via OpenAI chat completions."""
    if not OPENAI_KEY:
        return ""
    aff_list = "\n".join(
        f"  - {a}: {aff_urls.get(a, '#')}" for a in affiliates)
    system = (
        "You are a technical blogger writing for developers and solo operators. "
        "Write clear, practical, first-person content with concrete steps. "
        "Include affiliate links naturally where relevant. "
        "No fluff. Approx 600 words."
    )
    user = (
        f"Write a blog post titled: \"{topic}\"\n\n"
        f"Naturally embed these affiliate links in the body:\n{aff_list}\n\n"
        f"End with a brief author note: {OPERATOR} | ABN {ABN} | {SITE}"
    )
    payload = json.dumps({
        "model":      "gpt-4o-mini",
        "messages":   [{"role": "system", "content": system},
                       {"role": "user",   "content": user}],
        "max_tokens": 900,
        "temperature": 0.7,
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {OPENAI_KEY}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            resp = json.loads(r.read())
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[content-gen] GPT error: {e}")
        return ""


def _demo_article(topic: str, affiliates: list, aff_urls: dict) -> str:
    links = "  ".join(f"[{a}]({aff_urls.get(a,'#')})" for a in affiliates)
    return (
        f"# {topic}\n\n"
        f"*Demo article — set ENABLE_CONTENT=true and OPENAI_API_KEY to generate live.*\n\n"
        f"This post would cover: **{topic}**\n\n"
        f"Affiliate links: {links}\n\n"
        f"---\n*{OPERATOR} | ABN {ABN} | {SITE}*"
    )


def _post_devto(title: str, body: str, tags: list) -> str:
    """Publish article to dev.to. Returns canonical_url or error string."""
    if not DEVTO_KEY:
        return "no_devto_key"
    payload = json.dumps({
        "article": {
            "title":      title,
            "body_markdown": body,
            "published":  True,
            "tags":       tags[:4],
        }
    }).encode()
    req = urllib.request.Request(
        "https://dev.to/api/articles",
        data=payload,
        headers={
            "Content-Type":   "application/json",
            "api-key":        DEVTO_KEY,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            resp = json.loads(r.read())
        return resp.get("url", "published")
    except Exception as e:
        return f"err:{e}"


def _save_content(platform: str, title: str, body: str,
                  affiliates: list, url: str, status: str):
    aff_str = json.dumps(affiliates)
    conn    = sqlite3.connect(DB, timeout=10)
    try:
        conn.execute("""INSERT INTO content_pieces
            (platform,title,body,affiliates_embedded,url,status)
            VALUES (?,?,?,?,?,?)""",
            (platform, title[:200], body[:4000], aff_str, url[:500], status))
        conn.commit()
    except Exception as e:
        print(f"[content-gen] DB save error: {e}")
    finally:
        conn.close()


def run_cycle():
    ts      = datetime.now(tz=timezone.utc).isoformat()
    aff_urls = _aff_urls()
    print(f"[content-gen] {ts} | dry={DRY_RUN} | topics={len(TOPICS)}")

    # Pick next topic based on least-recently published
    conn  = sqlite3.connect(DB, timeout=10)
    done  = set()
    try:
        rows = conn.execute(
            "SELECT title FROM content_pieces WHERE platform='dev.to' "
            "ORDER BY ts DESC LIMIT 20").fetchall()
        done = {r[0][:50] for r in rows}
    except Exception:
        pass
    conn.close()

    topic_entry = next(
        (t for t in TOPICS if t["topic"][:50] not in done),
        TOPICS[0]
    )
    topic     = topic_entry["topic"]
    affiliates = topic_entry["affiliates"]
    tags      = topic_entry.get("tags", [])

    print(f"   Topic: {topic}")
    print(f"   Affiliates: {affiliates}")

    if DRY_RUN or not OPENAI_KEY:
        body   = _demo_article(topic, affiliates, aff_urls)
        status = "demo"
        url    = ""
    else:
        body = _gpt_article(topic, affiliates, aff_urls)
        if not body:
            print("   Article generation failed — skipping")
            return
        url    = _post_devto(topic, body, tags)
        status = "published" if url.startswith("https") else "staged"
        print(f"   Published: {url}")

    _save_content("dev.to", topic, body, affiliates, url, status)

    phi = round(len(affiliates) * PHI, 4)
    print(f"   Saved content_pieces | status={status} | φ={phi}")


def _sigterm(s, f):
    global _running
    _running = False


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT,  _sigterm)
    print(f"[content-gen] Active | interval={INTERVAL}s | dry={DRY_RUN} | "
          f"topics={len(TOPICS)}")
    while _running:
        try:
            run_cycle()
        except Exception as e:
            print(f"[content-gen] ERR {e}")
        for _ in range(INTERVAL):
            if not _running:
                break
            time.sleep(1)
    print("[content-gen] Stopped.")
