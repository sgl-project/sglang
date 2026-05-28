#!/usr/bin/env python3
"""
fm_knowledge_base.py — FractalMesh OMEGA Titan Knowledge Base & Documentation Engine (Port 7871)
Structured knowledge base with full-text search, versioning, collections, and AI summarisation.
Samuel James Hiotis | ABN 56 628 117 363
"""

import base64
import gzip
import hashlib
import hmac
import html
import json
import os
import re
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.parse import parse_qs, quote, urlparse
from urllib.request import Request, urlopen

# ---------------------------------------------------------------------------
# Vault / env loading
# ---------------------------------------------------------------------------
_ENV_FILE = Path.home() / ".secrets" / "fractal.env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

PORT          = int(os.getenv("KNOWLEDGE_BASE_PORT", "7871"))
MCP_PORT      = int(os.getenv("MCP_PORT", "7785"))
MCP_SECRET    = os.getenv("MCP_SECRET", "")
ADMIN_SECRET  = os.getenv("ADMIN_SECRET", "")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")

ROOT = Path(os.path.expanduser("~/fmsaas"))
DB   = ROOT / "database" / "sovereign.db"
LOG  = ROOT / "logs" / "knowledge_base.log"

for _p in (ROOT, DB.parent, LOG.parent):
    _p.mkdir(parents=True, exist_ok=True)

START_TIME = time.time()

# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------
STOPWORDS = {"the","a","an","is","in","of","to","and","or","for","with","on","at","by"}

# ---------------------------------------------------------------------------
# Database initialisation
# ---------------------------------------------------------------------------
def _db_conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(DB), timeout=15)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA foreign_keys=ON")
    c.row_factory = sqlite3.Row
    return c

def _db_init():
    c = _db_conn()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS articles (
            id           INTEGER PRIMARY KEY,
            slug         TEXT UNIQUE NOT NULL,
            title        TEXT NOT NULL,
            content      TEXT NOT NULL,
            summary      TEXT DEFAULT '',
            category     TEXT DEFAULT '',
            subcategory  TEXT DEFAULT '',
            tags         TEXT DEFAULT '',
            status       TEXT DEFAULT 'published',
            author       TEXT DEFAULT 'system',
            version      INTEGER DEFAULT 1,
            parent_id    INTEGER,
            view_count   INTEGER DEFAULT 0,
            helpful_yes  INTEGER DEFAULT 0,
            helpful_no   INTEGER DEFAULT 0,
            created_at   REAL,
            updated_at   REAL
        );

        CREATE TABLE IF NOT EXISTS article_versions (
            id             INTEGER PRIMARY KEY,
            article_id     INTEGER NOT NULL,
            version        INTEGER NOT NULL,
            title          TEXT NOT NULL,
            content        TEXT NOT NULL,
            changed_by     TEXT DEFAULT 'system',
            change_summary TEXT DEFAULT '',
            created_at     REAL
        );

        CREATE TABLE IF NOT EXISTS collections (
            id            INTEGER PRIMARY KEY,
            name          TEXT UNIQUE NOT NULL,
            slug          TEXT UNIQUE NOT NULL,
            description   TEXT DEFAULT '',
            article_count INTEGER DEFAULT 0,
            icon          TEXT DEFAULT '',
            order_index   INTEGER DEFAULT 0,
            created_at    REAL
        );

        CREATE TABLE IF NOT EXISTS collection_articles (
            collection_id INTEGER NOT NULL,
            article_id    INTEGER NOT NULL,
            order_index   INTEGER DEFAULT 0,
            PRIMARY KEY (collection_id, article_id)
        );

        CREATE TABLE IF NOT EXISTS search_index (
            id             INTEGER PRIMARY KEY,
            article_id     INTEGER UNIQUE NOT NULL,
            title_tokens   TEXT DEFAULT '',
            content_tokens TEXT DEFAULT '',
            updated_at     REAL
        );
    """)
    c.commit()
    c.close()

# ---------------------------------------------------------------------------
# Tokenisation & search helpers
# ---------------------------------------------------------------------------
def _tokenise(text: str) -> list:
    """Split on non-alphanumeric, lowercase, remove stopwords."""
    raw = re.split(r'[^a-zA-Z0-9]+', text.lower())
    return [t for t in raw if t and t not in STOPWORDS]

def _index_article(c: sqlite3.Connection, article_id: int, title: str, content: str):
    title_tokens   = " ".join(_tokenise(title))
    content_tokens = " ".join(_tokenise(content))
    now = time.time()
    c.execute("""
        INSERT INTO search_index (article_id, title_tokens, content_tokens, updated_at)
        VALUES (?,?,?,?)
        ON CONFLICT(article_id) DO UPDATE SET
            title_tokens=excluded.title_tokens,
            content_tokens=excluded.content_tokens,
            updated_at=excluded.updated_at
    """, (article_id, title_tokens, content_tokens, now))

def _search(query: str, limit: int = 20) -> list:
    """Score articles by token overlap, weighted by title matches and recency."""
    q_tokens = set(_tokenise(query))
    if not q_tokens:
        return []
    c = _db_conn()
    rows = c.execute("""
        SELECT si.article_id, si.title_tokens, si.content_tokens,
               a.title, a.content, a.slug, a.category, a.updated_at
        FROM search_index si
        JOIN articles a ON a.id = si.article_id
        WHERE a.status = 'published'
    """).fetchall()
    c.close()

    results = []
    now = time.time()
    for row in rows:
        t_tokens = set(row["title_tokens"].split())
        c_tokens = set(row["content_tokens"].split())
        t_score  = len(q_tokens & t_tokens) * 3   # title matches weighted x3
        c_score  = len(q_tokens & c_tokens)
        score    = t_score + c_score
        if score == 0:
            continue
        days_old      = (now - (row["updated_at"] or now)) / 86400
        recency       = 1.0 / (1 + days_old * 0.01)
        final_score   = score * recency
        # Excerpt: first 200 chars of content
        excerpt = row["content"][:200].strip()
        results.append({
            "slug":     row["slug"],
            "title":    row["title"],
            "category": row["category"],
            "score":    round(final_score, 4),
            "excerpt":  excerpt,
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:limit]

# ---------------------------------------------------------------------------
# Slug generation
# ---------------------------------------------------------------------------
def _slugify(title: str) -> str:
    s = title.lower().strip()
    s = re.sub(r'[^a-z0-9\s-]', '', s)
    s = re.sub(r'[\s-]+', '-', s)
    return s.strip('-')[:80]

def _unique_slug(c: sqlite3.Connection, base: str) -> str:
    slug = base
    n = 1
    while c.execute("SELECT 1 FROM articles WHERE slug=?", (slug,)).fetchone():
        slug = f"{base}-{n}"
        n += 1
    return slug

# ---------------------------------------------------------------------------
# Collection article count helper
# ---------------------------------------------------------------------------
def _refresh_collection_count(c: sqlite3.Connection, collection_id: int):
    cnt = c.execute(
        "SELECT COUNT(*) FROM collection_articles WHERE collection_id=?", (collection_id,)
    ).fetchone()[0]
    c.execute("UPDATE collections SET article_count=? WHERE id=?", (cnt, collection_id))

# ---------------------------------------------------------------------------
# AI summarisation (async)
# ---------------------------------------------------------------------------
def _ai_summarise(article_id: int, content: str):
    """Call Anthropic API to generate a 2-3 sentence summary; update DB if successful."""
    if not ANTHROPIC_KEY:
        return
    def _run():
        try:
            prompt = (
                "Summarise the following documentation article in exactly 2-3 clear, "
                "informative sentences suitable for a knowledge base preview. "
                "Return only the summary, no preamble.\n\n" + content[:4000]
            )
            payload = json.dumps({
                "model":      "claude-haiku-4-5",
                "max_tokens": 256,
                "messages":   [{"role": "user", "content": prompt}]
            }).encode()
            req = Request(
                "https://api.anthropic.com/v1/messages",
                data=payload,
                headers={
                    "Content-Type":      "application/json",
                    "x-api-key":         ANTHROPIC_KEY,
                    "anthropic-version": "2023-06-01",
                },
                method="POST",
            )
            with urlopen(req, timeout=30) as resp:
                data    = json.loads(resp.read())
                summary = data["content"][0]["text"].strip()
            conn = _db_conn()
            conn.execute("UPDATE articles SET summary=? WHERE id=?", (summary, article_id))
            conn.commit()
            conn.close()
        except Exception:
            pass
    t = threading.Thread(target=_run, daemon=True)
    t.start()

# ---------------------------------------------------------------------------
# Background index rebuild thread
# ---------------------------------------------------------------------------
_last_index_run = 0.0

def _index_worker():
    global _last_index_run
    while True:
        time.sleep(300)
        try:
            cutoff = _last_index_run
            _last_index_run = time.time()
            c = _db_conn()
            rows = c.execute(
                "SELECT id, title, content FROM articles WHERE updated_at > ?", (cutoff,)
            ).fetchall()
            for row in rows:
                _index_article(c, row["id"], row["title"], row["content"])
            c.commit()
            c.close()
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Pre-seeded content
# ---------------------------------------------------------------------------
SEED_DATA = {
    "collections": [
        {
            "name": "Getting Started",
            "slug": "getting-started",
            "description": "Everything you need to begin using FractalMesh OMEGA Titan.",
            "icon": "rocket",
            "order_index": 1,
            "articles": [
                {
                    "slug": "fractalmesh-overview",
                    "title": "FractalMesh Overview",
                    "category": "Getting Started",
                    "subcategory": "Introduction",
                    "tags": "overview,platform,introduction",
                    "content": """# FractalMesh OMEGA Titan — Platform Overview

FractalMesh OMEGA Titan is a next-generation autonomous revenue and orchestration platform designed to operate as a self-sustaining digital business infrastructure. Built on a modular agent architecture, the platform integrates over 120 specialised micro-agents that handle everything from payment processing and AI-powered content generation to blockchain interactions, affiliate marketing, and real-time analytics. Each agent is independently deployable, communicates over a lightweight MCP (Mesh Control Protocol) bus, and can be hot-swapped without downtime.

## Core Architecture

At its heart, FractalMesh is a distributed system governed by a Sovereign Database layer backed by SQLite with WAL-mode replication. The platform's hub-and-spoke design places the MCP Router at the centre, routing JSON-RPC calls between agents and external integrations. Agents expose standardised REST endpoints over dedicated ports and register their capabilities via the Pulse Bus — a publish-subscribe broadcast layer that enables real-time coordination without tight coupling.

## Revenue Engine

FractalMesh's Revenue Aggregator consolidates income streams across Stripe subscriptions, PayPal transactions, crypto rails (Coinbase, KuCoin), Gumroad digital product sales, affiliate commissions, and AdSense/AdMob advertising revenue. The Revenue Forecast agent applies time-series modelling to project monthly recurring revenue (MRR), churn rates, and expansion revenue, feeding data back into the Strategy Engine for autonomous goal adjustment.

## AI & Automation

The platform leverages multiple AI providers — Anthropic Claude, OpenRouter, and HuggingFace — through a unified AI-as-a-Service layer. Autonomous agents can generate blog posts, social media content, email campaigns, and technical documentation without human intervention. The RL Quad agent continuously optimises resource allocation using reinforcement learning, balancing compute costs against revenue-generating workloads.

## Security & Compliance

All sensitive credentials are stored in a vault file at `~/.secrets/fractal.env` and loaded via a secure bootstrap pattern that never commits secrets to version control. Rate limiting is enforced globally by the Rate Limiter agent, while the Security Monitor performs continuous anomaly detection, alerting via the Notifier agent when suspicious patterns are detected. GDPR compliance is maintained through data lifecycle policies enforced by the Sovereign Memory agent.""",
                },
                {
                    "slug": "quick-start-guide",
                    "title": "Quick Start Guide",
                    "category": "Getting Started",
                    "subcategory": "Setup",
                    "tags": "quickstart,setup,installation,configuration",
                    "content": """# Quick Start Guide

This guide walks you through deploying FractalMesh OMEGA Titan from a fresh Ubuntu 22.04+ server to a fully operational state in under 30 minutes.

## Prerequisites

You will need Python 3.10 or higher, Git, a domain name pointed at your server IP, and credentials for at least one payment provider (Stripe recommended). Ensure ports 7800-7900 are accessible internally and that port 443 is open for inbound HTTPS if you are fronting with Nginx or Caddy.

## Step 1 — Clone and Configure

```bash
git clone https://github.com/fractalmesh/omega-titan ~/fmsaas/repo
cd ~/fmsaas/repo
cp .env.example ~/.secrets/fractal.env
```

Edit `~/.secrets/fractal.env` and fill in your API keys. At minimum, set `ADMIN_SECRET`, `MCP_SECRET`, and `STRIPE_SECRET_KEY`. The platform will start with degraded functionality if optional keys (e.g. `ANTHROPIC_API_KEY`) are absent, but core features remain available.

## Step 2 — Initialise the Database

The Sovereign Database at `~/fmsaas/database/sovereign.db` is auto-created on first run. Each agent performs its own schema migration on startup, so no manual SQL is required. For production deployments, enable automated SQLite backups:

```bash
mkdir -p ~/fmsaas/backups
echo "0 2 * * * sqlite3 ~/fmsaas/database/sovereign.db '.backup ~/fmsaas/backups/sovereign-$(date +%Y%m%d).db'" | crontab -
```

## Step 3 — Start the Mesh

Launch the supervisor process, which reads `~/fmsaas/config/agents.json` and starts each enabled agent as a subprocess with automatic restart on failure:

```bash
python3 ~/fmsaas/repo/fractalmesh/system/supervisor.py --config ~/fmsaas/config/agents.json
```

The MCP Router (port 7785) must be the first agent to start. All other agents register with it at boot. You can verify the mesh is healthy by calling `GET http://localhost:7785/health`.

## Step 4 — Verify Installation

Access the Admin Dashboard at `http://localhost:7801` with your `ADMIN_SECRET`. The Mesh Integrator page shows agent registration status. Green dots indicate healthy, reachable agents; yellow indicates degraded mode; red indicates the agent is down.

## Next Steps

Read the Architecture Overview to understand how agents interoperate. Then configure your first revenue stream by visiting the Stripe Gateway setup page in the Admin Dashboard.""",
                },
                {
                    "slug": "architecture-overview",
                    "title": "Architecture Overview",
                    "category": "Getting Started",
                    "subcategory": "Architecture",
                    "tags": "architecture,design,agents,mcp,database",
                    "content": """# Architecture Overview

FractalMesh OMEGA Titan is structured around three fundamental layers: the Agent Layer, the Communication Layer, and the Persistence Layer. Understanding how these interact is essential for operating, extending, or debugging the platform.

## Agent Layer

Each agent is a self-contained Python process listening on a dedicated TCP port. Agents follow a strict contract: they expose a `GET /health` endpoint, accept `X-Admin-Secret` header authentication, and emit structured JSON responses. The Agent Registry (maintained by the MCP Router) maps agent names to their host, port, and capability manifest. New agents self-register at startup by POSTing their manifest to `POST /register` on the MCP Router.

Agents are categorised into five tiers by criticality:

- **Tier 0 (Core)**: MCP Router, Sovereign Memory, Rate Limiter — must always be running
- **Tier 1 (Revenue)**: Stripe Gateway, PayPal, Coinbase, Revenue Aggregator — critical for income
- **Tier 2 (Automation)**: Content Engine, Email Campaign, Social Manager — enhances revenue
- **Tier 3 (Analytics)**: Analytics Hub, Revenue Forecast, Strategy Engine — informs decisions
- **Tier 4 (Auxiliary)**: All other integrations — nice-to-have, gracefully degraded when absent

## Communication Layer — MCP Bus

The Mesh Control Protocol (MCP) is a lightweight JSON-RPC 2.0 dialect transported over HTTP/1.1. The MCP Router acts as a broker: agents publish events to named topics and subscribe to topics of interest. The Pulse Bus extends this with broadcast semantics, broadcasting heartbeats and alerts to all connected agents simultaneously.

Inter-agent calls use a standard envelope:

```json
{
  "jsonrpc": "2.0",
  "method": "agent.invoke",
  "params": {"target": "stripe_gateway", "action": "charge", "payload": {}},
  "id": "uuid-here"
}
```

Authentication between agents uses the shared `MCP_SECRET` header. Calls without a valid secret are rejected with HTTP 403.

## Persistence Layer — Sovereign Database

All agents share a single SQLite database at `~/fmsaas/database/sovereign.db` operating in WAL (Write-Ahead Logging) mode. WAL mode enables concurrent reads alongside writes, which is critical given the high read/write concurrency from 100+ agents. Each agent owns its own tables, prefixed with the agent name (e.g. `stripe_*`, `analytics_*`). Cross-agent queries are permitted read-only; agents must never write to another agent's tables directly — inter-agent data exchange goes through the MCP bus.

The database is backed up nightly and can be restored from any checkpoint. For high-availability deployments, Litestream replication to S3 is supported via the Cloud Storage agent configuration.""",
                },
            ],
        },
        {
            "name": "API Reference",
            "slug": "api-reference",
            "description": "Complete API reference for all FractalMesh endpoints and authentication.",
            "icon": "code",
            "order_index": 2,
            "articles": [
                {
                    "slug": "authentication",
                    "title": "Authentication",
                    "category": "API Reference",
                    "subcategory": "Security",
                    "tags": "authentication,api-key,security,admin-secret,mcp-secret",
                    "content": """# Authentication

FractalMesh OMEGA Titan uses a layered authentication model with distinct credentials for different access levels. Understanding which credential to use in which context prevents unauthorised access and ensures audit trails are accurate.

## Admin Secret (`ADMIN_SECRET`)

The Admin Secret is a high-privilege credential granting write access to all agent administration endpoints. It is passed via the `X-Admin-Secret` HTTP header. All mutating endpoints — `POST`, `PUT`, `PATCH`, `DELETE` — require a valid Admin Secret unless the agent explicitly documents otherwise. The Admin Secret is set in `~/.secrets/fractal.env` and must be a cryptographically random string of at least 32 characters.

```http
POST /articles HTTP/1.1
Host: localhost:7871
X-Admin-Secret: your-admin-secret-here
Content-Type: application/json

{"title": "New Article", "content": "...", "category": "Guides"}
```

If `ADMIN_SECRET` is not set (empty string), all admin endpoints are open — this is acceptable for local development but **must never** be used in production.

## MCP Secret (`MCP_SECRET`)

The MCP Secret authenticates inter-agent communication on the MCP bus. It is passed via the `X-MCP-Secret` header on calls routed through the MCP Router. Agents verify this secret before processing any inbound MCP call. The MCP Secret should be different from the Admin Secret to limit blast radius if either is compromised.

## Anthropic API Key (`ANTHROPIC_API_KEY`)

Used exclusively for AI-powered features such as article summarisation, content generation, and the AI Assistant agent. This key is never exposed via any API endpoint and is only used server-side. Rotate this key quarterly or immediately if a breach is suspected.

## Rate Limiting and Abuse Prevention

All external-facing endpoints are subject to rate limiting enforced by the Rate Limiter agent. Default limits are 100 requests per minute per IP for read endpoints and 20 requests per minute per IP for write endpoints. Exceeding limits returns HTTP 429 with a `Retry-After` header. The Security Monitor agent tracks failed authentication attempts and can auto-block IPs after 10 consecutive failures within a 5-minute window.

## Token Rotation

FractalMesh does not currently implement JWT or OAuth2 — it uses static shared secrets. A planned future enhancement (tracked in the roadmap) will add per-user API tokens with expiry, scopes, and revocation. Until then, rotate the Admin Secret and MCP Secret by updating `~/.secrets/fractal.env` and sending a `SIGHUP` to the supervisor process to reload credentials without downtime.""",
                },
                {
                    "slug": "rate-limits",
                    "title": "Rate Limits",
                    "category": "API Reference",
                    "subcategory": "Policies",
                    "tags": "rate-limits,throttling,429,quotas,api-usage",
                    "content": """# Rate Limits

FractalMesh OMEGA Titan applies rate limiting at two levels: per-IP limits enforced by the Rate Limiter agent, and per-route limits defined in each agent's configuration. Understanding these limits helps you build integrations that behave gracefully under load.

## Default Rate Limit Tiers

| Endpoint Category        | Requests / Minute | Burst Allowance |
|--------------------------|-------------------|-----------------|
| GET /health              | 300               | 50              |
| GET read endpoints       | 100               | 20              |
| POST/PUT write endpoints | 20                | 5               |
| DELETE endpoints         | 10                | 2               |
| AI-powered endpoints     | 5                 | 1               |
| Search endpoints         | 60                | 10              |

AI-powered endpoints (those that call Anthropic or OpenRouter) have tighter limits to control API costs. These limits apply per calling IP address.

## Rate Limit Headers

Responses from rate-limited endpoints include standard headers:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1717000060
Retry-After: 13
```

When the limit is exceeded, the server returns HTTP 429 with the `Retry-After` header indicating seconds until the rate limit window resets.

## Configuring Custom Limits

Rate limits are configured in `~/fmsaas/config/rate_limits.json`. Each entry maps an endpoint pattern (supporting glob wildcards) to a limit configuration:

```json
{
  "/search": {"rpm": 60, "burst": 10},
  "/articles/*/helpful": {"rpm": 30, "burst": 5}
}
```

Changes to `rate_limits.json` take effect within 60 seconds without requiring an agent restart. The Rate Limiter agent polls for config changes on a 60-second interval.

## Exemptions

Internal inter-agent calls authenticated with a valid `X-MCP-Secret` header are exempt from IP-based rate limiting. This allows high-frequency internal workflows (e.g. the Analytics Hub polling all agents for metrics) without triggering limits intended for external consumers.

If you are building an integration that legitimately requires higher limits, contact the platform administrator to configure an IP-level exemption in the Rate Limiter configuration file.""",
                },
            ],
        },
        {
            "name": "Monetization",
            "slug": "monetization",
            "description": "Revenue models, subscription plans, and payment integration guides.",
            "icon": "dollar-sign",
            "order_index": 3,
            "articles": [
                {
                    "slug": "subscription-plans",
                    "title": "Subscription Plans",
                    "category": "Monetization",
                    "subcategory": "Billing",
                    "tags": "subscriptions,pricing,plans,mrr,billing,stripe",
                    "content": """# Subscription Plans

FractalMesh OMEGA Titan ships with a pre-configured subscription billing system powered by Stripe. The platform supports multiple plan tiers, usage-based billing add-ons, free trials, and coupon codes out of the box.

## Default Plan Tiers

The platform seeds three default subscription plans at startup. These plans are stored in the `stripe_plans` table and mirrored to your Stripe account via the Stripe Gateway agent:

**Starter — $29/month**
Includes access to the core API with up to 10,000 API calls per month, 3 AI content generation requests per day, email support with 48-hour response time, and up to 2 team member seats. Ideal for individual developers and small projects.

**Growth — $99/month**
Includes 100,000 API calls per month, unlimited AI content generation, priority support with 8-hour response time, up to 10 team member seats, advanced analytics dashboard access, and webhook support for all events. Suitable for growing businesses that need automation at scale.

**Enterprise — $499/month**
Unlimited API calls, dedicated agent instances with guaranteed resource allocation, 24/7 priority support with 2-hour SLA, unlimited team seats, custom integration development support, SSO/SAML authentication, and a dedicated account manager. Designed for organisations requiring reliability guarantees and white-glove support.

## Usage-Based Billing

In addition to flat-rate plans, FractalMesh supports metered billing for high-volume API consumers. Usage is tracked by the Analytics Hub agent and synced to Stripe's usage records API at the end of each billing period. Configure metered pricing by setting `BILLING_MODE=metered` in your agent config and defining the per-unit price in Stripe.

## Free Trials and Coupons

All plans support a configurable free trial period (default 14 days). Trial accounts have full access to plan features with usage caps at 10% of plan limits. Coupon codes are managed via the Stripe Dashboard or the Admin API's `/billing/coupons` endpoint. Applied coupons are recorded in the `stripe_subscriptions` table for reporting purposes.

## Plan Changes and Proration

When a subscriber upgrades or downgrades their plan, FractalMesh handles proration automatically via Stripe. Upgrades take effect immediately with a prorated charge; downgrades take effect at the start of the next billing cycle. Cancellations preserve access until the end of the current paid period.""",
                },
                {
                    "slug": "payment-integration",
                    "title": "Payment Integration",
                    "category": "Monetization",
                    "subcategory": "Payments",
                    "tags": "payments,stripe,paypal,crypto,coinbase,webhooks,checkout",
                    "content": """# Payment Integration

FractalMesh OMEGA Titan supports multiple payment processors through dedicated gateway agents. Each gateway handles its own webhook verification, retry logic, and failure recovery, ensuring revenue is never lost due to transient network issues.

## Stripe Integration

The Stripe Gateway agent (port 7820) is the primary payment processor. It manages subscriptions, one-time charges, refunds, and disputes. To configure Stripe:

1. Set `STRIPE_SECRET_KEY` and `STRIPE_WEBHOOK_SECRET` in `~/.secrets/fractal.env`
2. Configure your Stripe webhook endpoint to point to `https://yourdomain.com/stripe/webhook`
3. Enable the following Stripe webhook events: `payment_intent.succeeded`, `payment_intent.payment_failed`, `customer.subscription.created`, `customer.subscription.deleted`, `invoice.paid`, `invoice.payment_failed`

The gateway validates webhook signatures using the `STRIPE_WEBHOOK_SECRET` to prevent replay attacks. All payment events are stored in `stripe_events` table and trigger downstream actions (e.g. provisioning access, sending confirmation emails) via the MCP bus.

## PayPal Integration

The PayPal agent (port 7823) handles PayPal Checkout, PayPal subscriptions, and instant payment notifications (IPN). Configure with `PAYPAL_CLIENT_ID` and `PAYPAL_CLIENT_SECRET`. The agent automatically handles OAuth2 token refresh, so you never need to manage access tokens manually.

## Crypto Payments via Coinbase Commerce

The Coinbase agent (port 7825) integrates with Coinbase Commerce for accepting Bitcoin, Ethereum, USDC, and other cryptocurrencies. Crypto payments require confirmation on-chain before access is provisioned — the agent monitors the Coinbase Commerce webhook and waits for the `charge:confirmed` event before crediting the account. Settlement to fiat is handled automatically by Coinbase if you configure the auto-conversion setting in your Coinbase Commerce dashboard.

## Revenue Reconciliation

The Revenue Aggregator agent runs a nightly reconciliation job that cross-checks Stripe, PayPal, and Coinbase records against the local database. Any discrepancies are flagged in the `revenue_discrepancies` table and trigger an admin alert. This catch-all ensures your MRR reporting is always accurate even if a webhook was missed or processed out of order.""",
                },
            ],
        },
    ]
}

def _seed_content():
    """Seed the database with initial collections and articles if empty."""
    c = _db_conn()
    count = c.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    if count > 0:
        c.close()
        return
    now = time.time()
    for coll_data in SEED_DATA["collections"]:
        c.execute("""
            INSERT OR IGNORE INTO collections (name, slug, description, icon, order_index, article_count, created_at)
            VALUES (?,?,?,?,?,0,?)
        """, (coll_data["name"], coll_data["slug"], coll_data["description"],
              coll_data["icon"], coll_data["order_index"], now))
        coll_id = c.execute("SELECT id FROM collections WHERE slug=?", (coll_data["slug"],)).fetchone()["id"]
        for idx, art in enumerate(coll_data["articles"]):
            c.execute("""
                INSERT OR IGNORE INTO articles
                    (slug, title, content, category, subcategory, tags, status, author, version, created_at, updated_at)
                VALUES (?,?,?,?,?,?,'published','system',1,?,?)
            """, (art["slug"], art["title"], art["content"], art["category"],
                  art["subcategory"], art["tags"], now, now))
            art_row = c.execute("SELECT id FROM articles WHERE slug=?", (art["slug"],)).fetchone()
            if art_row:
                art_id = art_row["id"]
                _index_article(c, art_id, art["title"], art["content"])
                c.execute("""
                    INSERT OR IGNORE INTO collection_articles (collection_id, article_id, order_index)
                    VALUES (?,?,?)
                """, (coll_id, art_id, idx))
        _refresh_collection_count(c, coll_id)
    c.commit()
    c.close()
    # Trigger async summaries for all seeded articles if AI key present
    if ANTHROPIC_KEY:
        conn = _db_conn()
        arts = conn.execute("SELECT id, content FROM articles").fetchall()
        conn.close()
        for a in arts:
            _ai_summarise(a["id"], a["content"])

# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------
def _is_admin(headers) -> bool:
    if not ADMIN_SECRET:
        return True
    return hmac.compare_digest(headers.get("X-Admin-Secret", ""), ADMIN_SECRET)

# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------
class KBHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass  # suppress default access log

    # ---- low-level helpers -------------------------------------------------

    def _send(self, code: int, data, content_type: str = "application/json"):
        body = json.dumps(data, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _ok(self, data):
        self._send(200, data)

    def _created(self, data):
        self._send(201, data)

    def _err(self, code: int, msg: str):
        self._send(code, {"error": msg})

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def _qs(self) -> dict:
        parsed = urlparse(self.path)
        return {k: v[0] if len(v) == 1 else v
                for k, v in parse_qs(parsed.query).items()}

    def _path_parts(self) -> list:
        return [p for p in urlparse(self.path).path.strip("/").split("/") if p]

    # ---- OPTIONS (CORS preflight) ------------------------------------------

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,PUT,DELETE,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type,X-Admin-Secret")
        self.send_header("Content-Length", "0")
        self.end_headers()

    # ---- GET dispatcher ----------------------------------------------------

    def do_GET(self):
        parts = self._path_parts()
        qs    = self._qs()

        if not parts or parts[0] == "health":
            return self._handle_health()
        if parts[0] == "articles":
            if len(parts) == 1:
                return self._list_articles(qs)
            if len(parts) == 2:
                return self._get_article(parts[1])
            if len(parts) == 3 and parts[2] == "versions":
                return self._article_versions(parts[1])
        if parts[0] == "collections":
            if len(parts) == 1:
                return self._list_collections()
            if len(parts) == 2:
                return self._get_collection(parts[1])
        if parts[0] == "search":
            return self._handle_search(qs)
        if parts[0] == "sitemap":
            return self._handle_sitemap()
        self._err(404, "Not found")

    # ---- POST dispatcher ---------------------------------------------------

    def do_POST(self):
        parts = self._path_parts()

        if parts[0] == "articles":
            if len(parts) == 1:
                return self._create_article()
            if len(parts) == 3 and parts[2] == "helpful":
                return self._helpful_vote(parts[1])
        if parts[0] == "collections" and len(parts) == 1:
            return self._create_collection()
        self._err(404, "Not found")

    # ---- PUT dispatcher ----------------------------------------------------

    def do_PUT(self):
        parts = self._path_parts()
        if parts[0] == "articles" and len(parts) == 2:
            return self._update_article(parts[1])
        self._err(404, "Not found")

    # ---- DELETE dispatcher -------------------------------------------------

    def do_DELETE(self):
        parts = self._path_parts()
        if parts[0] == "articles" and len(parts) == 2:
            return self._archive_article(parts[1])
        self._err(404, "Not found")

    # ======================================================================
    # Endpoint implementations
    # ======================================================================

    def _handle_health(self):
        c = _db_conn()
        art_count  = c.execute("SELECT COUNT(*) FROM articles WHERE status='published'").fetchone()[0]
        coll_count = c.execute("SELECT COUNT(*) FROM collections").fetchone()[0]
        c.close()
        self._ok({
            "status":     "ok",
            "port":       PORT,
            "uptime":     round(time.time() - START_TIME, 2),
            "articles":   art_count,
            "collections": coll_count,
            "ai_enabled": bool(ANTHROPIC_KEY),
        })

    # ---- articles list -----------------------------------------------------

    def _list_articles(self, qs: dict):
        category   = qs.get("category", "")
        status     = qs.get("status", "published")
        tags_filter = qs.get("tags", "")
        limit      = min(int(qs.get("limit", 20)), 100)
        offset     = int(qs.get("offset", 0))

        clauses = ["1=1"]
        params  = []
        if category:
            clauses.append("category=?")
            params.append(category)
        if status:
            clauses.append("status=?")
            params.append(status)
        if tags_filter:
            clauses.append("tags LIKE ?")
            params.append(f"%{tags_filter}%")

        where = " AND ".join(clauses)
        c = _db_conn()
        rows = c.execute(f"""
            SELECT id, slug, title, summary, category, subcategory, tags, status,
                   author, version, view_count, helpful_yes, helpful_no, created_at, updated_at
            FROM articles WHERE {where}
            ORDER BY updated_at DESC LIMIT ? OFFSET ?
        """, params + [limit, offset]).fetchall()
        total = c.execute(f"SELECT COUNT(*) FROM articles WHERE {where}", params).fetchone()[0]
        c.close()
        self._ok({
            "articles": [dict(r) for r in rows],
            "total":    total,
            "limit":    limit,
            "offset":   offset,
        })

    # ---- article detail ----------------------------------------------------

    def _get_article(self, slug: str):
        c = _db_conn()
        row = c.execute(
            "SELECT * FROM articles WHERE slug=? AND status != 'archived'", (slug,)
        ).fetchone()
        if not row:
            c.close()
            return self._err(404, "Article not found")
        # Increment view count
        c.execute("UPDATE articles SET view_count = view_count + 1 WHERE id=?", (row["id"],))
        c.commit()

        art = dict(row)
        # Find collection membership and prev/next
        ca = c.execute("""
            SELECT ca.collection_id, ca.order_index, co.slug as col_slug
            FROM collection_articles ca
            JOIN collections co ON co.id = ca.collection_id
            WHERE ca.article_id=?
            ORDER BY ca.order_index LIMIT 1
        """, (row["id"],)).fetchone()

        prev_art = next_art = None
        if ca:
            order = ca["order_index"]
            cid   = ca["collection_id"]
            prev_row = c.execute("""
                SELECT a.slug, a.title FROM collection_articles ca
                JOIN articles a ON a.id = ca.article_id
                WHERE ca.collection_id=? AND ca.order_index < ?
                ORDER BY ca.order_index DESC LIMIT 1
            """, (cid, order)).fetchone()
            next_row = c.execute("""
                SELECT a.slug, a.title FROM collection_articles ca
                JOIN articles a ON a.id = ca.article_id
                WHERE ca.collection_id=? AND ca.order_index > ?
                ORDER BY ca.order_index ASC LIMIT 1
            """, (cid, order)).fetchone()
            if prev_row:
                prev_art = {"slug": prev_row["slug"], "title": prev_row["title"]}
            if next_row:
                next_art = {"slug": next_row["slug"], "title": next_row["title"]}
            art["collection_slug"] = ca["col_slug"]

        c.close()
        art["prev"] = prev_art
        art["next"] = next_art
        self._ok(art)

    # ---- article versions --------------------------------------------------

    def _article_versions(self, slug: str):
        c = _db_conn()
        row = c.execute("SELECT id FROM articles WHERE slug=?", (slug,)).fetchone()
        if not row:
            c.close()
            return self._err(404, "Article not found")
        versions = c.execute("""
            SELECT id, version, title, changed_by, change_summary, created_at
            FROM article_versions WHERE article_id=?
            ORDER BY version DESC
        """, (row["id"],)).fetchall()
        c.close()
        self._ok({"slug": slug, "versions": [dict(v) for v in versions]})

    # ---- collections list --------------------------------------------------

    def _list_collections(self):
        c = _db_conn()
        rows = c.execute(
            "SELECT * FROM collections ORDER BY order_index ASC"
        ).fetchall()
        c.close()
        self._ok({"collections": [dict(r) for r in rows]})

    # ---- collection detail with articles -----------------------------------

    def _get_collection(self, slug: str):
        c = _db_conn()
        coll = c.execute("SELECT * FROM collections WHERE slug=?", (slug,)).fetchone()
        if not coll:
            c.close()
            return self._err(404, "Collection not found")
        arts = c.execute("""
            SELECT a.id, a.slug, a.title, a.summary, a.category, a.tags,
                   a.view_count, a.updated_at, ca.order_index
            FROM collection_articles ca
            JOIN articles a ON a.id = ca.article_id
            WHERE ca.collection_id=? AND a.status='published'
            ORDER BY ca.order_index ASC
        """, (coll["id"],)).fetchall()
        c.close()
        result = dict(coll)
        result["articles"] = [dict(a) for a in arts]
        self._ok(result)

    # ---- full-text search --------------------------------------------------

    def _handle_search(self, qs: dict):
        q      = qs.get("q", "").strip()
        limit  = min(int(qs.get("limit", 10)), 50)
        if not q:
            return self._err(400, "Missing query parameter 'q'")
        results = _search(q, limit)
        self._ok({"query": q, "count": len(results), "results": results})

    # ---- sitemap -----------------------------------------------------------

    def _handle_sitemap(self):
        c = _db_conn()
        rows = c.execute(
            "SELECT slug, updated_at FROM articles WHERE status='published' ORDER BY updated_at DESC"
        ).fetchall()
        c.close()
        self._ok({
            "articles": [{"slug": r["slug"], "updated_at": r["updated_at"]} for r in rows]
        })

    # ---- create article ----------------------------------------------------

    def _create_article(self):
        if not _is_admin(self.headers):
            return self._err(403, "Forbidden")
        body = self._body()
        title   = (body.get("title") or "").strip()
        content = (body.get("content") or "").strip()
        if not title or not content:
            return self._err(400, "title and content are required")

        category    = body.get("category", "")
        subcategory = body.get("subcategory", "")
        tags        = body.get("tags", "")
        author      = body.get("author", "api")
        collection_id = body.get("collection_id")
        now = time.time()

        c = _db_conn()
        base_slug = _slugify(title)
        slug      = _unique_slug(c, base_slug)
        c.execute("""
            INSERT INTO articles
                (slug, title, content, category, subcategory, tags, status, author, version, created_at, updated_at)
            VALUES (?,?,?,?,?,?,'published',?,1,?,?)
        """, (slug, title, content, category, subcategory, tags, author, now, now))
        art_id = c.lastrowid
        _index_article(c, art_id, title, content)

        if collection_id:
            max_idx = c.execute(
                "SELECT COALESCE(MAX(order_index),0) FROM collection_articles WHERE collection_id=?",
                (collection_id,)
            ).fetchone()[0]
            c.execute(
                "INSERT OR IGNORE INTO collection_articles (collection_id, article_id, order_index) VALUES (?,?,?)",
                (collection_id, art_id, max_idx + 1)
            )
            _refresh_collection_count(c, collection_id)

        c.commit()
        c.close()
        _ai_summarise(art_id, content)
        self._created({"slug": slug, "id": art_id, "message": "Article created"})

    # ---- update article ----------------------------------------------------

    def _update_article(self, slug: str):
        if not _is_admin(self.headers):
            return self._err(403, "Forbidden")
        body = self._body()
        c = _db_conn()
        row = c.execute("SELECT * FROM articles WHERE slug=?", (slug,)).fetchone()
        if not row:
            c.close()
            return self._err(404, "Article not found")

        # Save old version
        c.execute("""
            INSERT INTO article_versions
                (article_id, version, title, content, changed_by, change_summary, created_at)
            VALUES (?,?,?,?,?,?,?)
        """, (
            row["id"], row["version"], row["title"], row["content"],
            body.get("changed_by", "api"), body.get("change_summary", ""),
            time.time()
        ))

        new_title   = body.get("title", row["title"])
        new_content = body.get("content", row["content"])
        new_cat     = body.get("category", row["category"])
        new_subcat  = body.get("subcategory", row["subcategory"])
        new_tags    = body.get("tags", row["tags"])
        new_ver     = row["version"] + 1
        now         = time.time()

        c.execute("""
            UPDATE articles SET title=?, content=?, category=?, subcategory=?, tags=?,
                version=?, updated_at=? WHERE slug=?
        """, (new_title, new_content, new_cat, new_subcat, new_tags, new_ver, now, slug))
        _index_article(c, row["id"], new_title, new_content)
        c.commit()
        c.close()
        _ai_summarise(row["id"], new_content)
        self._ok({"slug": slug, "version": new_ver, "message": "Article updated"})

    # ---- helpful vote ------------------------------------------------------

    def _helpful_vote(self, slug: str):
        body = self._body()
        vote = body.get("vote", "")
        if vote not in ("yes", "no"):
            return self._err(400, "vote must be 'yes' or 'no'")
        c = _db_conn()
        if not c.execute("SELECT 1 FROM articles WHERE slug=?", (slug,)).fetchone():
            c.close()
            return self._err(404, "Article not found")
        col = "helpful_yes" if vote == "yes" else "helpful_no"
        c.execute(f"UPDATE articles SET {col} = {col} + 1 WHERE slug=?", (slug,))
        c.commit()
        c.close()
        self._ok({"slug": slug, "vote": vote, "message": "Vote recorded"})

    # ---- create collection -------------------------------------------------

    def _create_collection(self):
        if not _is_admin(self.headers):
            return self._err(403, "Forbidden")
        body = self._body()
        name = (body.get("name") or "").strip()
        if not name:
            return self._err(400, "name is required")
        desc        = body.get("description", "")
        icon        = body.get("icon", "")
        order_index = int(body.get("order_index", 0))
        slug        = _slugify(name)
        now         = time.time()
        c = _db_conn()
        try:
            c.execute("""
                INSERT INTO collections (name, slug, description, icon, order_index, article_count, created_at)
                VALUES (?,?,?,?,?,0,?)
            """, (name, slug, desc, icon, order_index, now))
            coll_id = c.lastrowid
            c.commit()
        except sqlite3.IntegrityError:
            c.close()
            return self._err(409, "Collection name or slug already exists")
        c.close()
        self._created({"id": coll_id, "slug": slug, "message": "Collection created"})

    # ---- archive article ---------------------------------------------------

    def _archive_article(self, slug: str):
        if not _is_admin(self.headers):
            return self._err(403, "Forbidden")
        c = _db_conn()
        if not c.execute("SELECT 1 FROM articles WHERE slug=?", (slug,)).fetchone():
            c.close()
            return self._err(404, "Article not found")
        c.execute("UPDATE articles SET status='archived', updated_at=? WHERE slug=?",
                  (time.time(), slug))
        c.commit()
        c.close()
        self._ok({"slug": slug, "status": "archived", "message": "Article archived"})


# ---------------------------------------------------------------------------
# Server bootstrap
# ---------------------------------------------------------------------------
def main():
    _db_init()
    _seed_content()

    # Start background index thread
    idx_thread = threading.Thread(target=_index_worker, daemon=True, name="kb-index")
    idx_thread.start()

    server = HTTPServer(("0.0.0.0", PORT), KBHandler)
    print(f"[KnowledgeBase] Listening on port {PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[KnowledgeBase] Shutting down")
        server.server_close()


if __name__ == "__main__":
    main()
