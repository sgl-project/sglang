#!/usr/bin/env python3
"""
fm_mcp_router.py — Master MCP Intent Router (Port 7785)
Unified intent multiplexer for cross-app integration.
Credentials sourced from ~/.secrets/fractal.env at runtime — never hardcoded.
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import json
import time
import hmac
import hashlib
import signal
import sqlite3
import logging
import subprocess
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
PORT     = int(os.getenv("MCP_PORT", "7785"))
SECRET   = os.getenv("MCP_SECRET", "fm_mcp_internal").encode()
ROOT     = Path(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")))
DB       = ROOT / "database" / "sovereign.db"
LOG      = ROOT / "logs" / "mcp_router.log"

ROOT.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)
DB.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MCP-ROUTER] %(message)s",
    handlers=[
        logging.FileHandler(LOG),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("mcp_router")

# ── LBA firewall — reject any payload exposing raw credentials ────────────────
_BANNED = [
    "sk_live_", "sk-ant-api", "ETH_PRIVATE_KEY", "PRIVATE_KEY=",
    "password=", "secret=", "[id number redacted]",
]

def _lba_check(payload: str) -> bool:
    low = payload.lower()
    return not any(b.lower() in low for b in _BANNED)

# ── database ──────────────────────────────────────────────────────────────────
def _db_init():
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mcp_log (
            id      INTEGER PRIMARY KEY,
            intent  TEXT,
            status  TEXT,
            latency REAL,
            ts      DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def _db_log(intent: str, status: str, latency: float):
    try:
        conn = sqlite3.connect(DB, timeout=5)
        conn.execute(
            "INSERT INTO mcp_log (intent, status, latency) VALUES (?,?,?)",
            (intent, status, latency),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning("db_log error: %s", e)

# ── HMAC verification (optional — set MCP_REQUIRE_SIG=1 to enforce) ───────────
def _verify_sig(sig: str, body: bytes) -> bool:
    if not os.getenv("MCP_REQUIRE_SIG"):
        return True
    expected = hmac.new(SECRET, body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, sig or "")

# ── intent handlers ────────────────────────────────────────────────────────────

def _handle_sync_calendar(args, kwargs) -> dict:
    label = args[0] if args else kwargs.get("label", "")
    ts    = args[1] if len(args) > 1 else kwargs.get("timestamp", "")
    return {"action": "calendar_sync", "label": label, "timestamp": ts, "queued": True}

def _handle_sync_reminder(args, kwargs) -> dict:
    content  = args[0] if args else kwargs.get("content", "")
    priority = kwargs.get("priority", "normal")
    return {"action": "reminder_set", "content": content, "priority": priority}

def _handle_sync_workspace(args, kwargs) -> dict:
    payload = args[0] if args else kwargs.get("payload", {})
    return {"action": "workspace_queued", "payload": payload, "status": "QUEUED"}

def _handle_device_pulse(args, kwargs) -> dict:
    try:
        out = subprocess.getoutput("termux-battery-status 2>/dev/null")
        battery = json.loads(out) if out.strip().startswith("{") else {}
    except Exception:
        battery = {}
    return {
        "node":    "arm64_local",
        "battery": battery,
        "uptime":  os.popen("uptime -p 2>/dev/null").read().strip() or "n/a",
        "ts":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

def _handle_send_message(args, kwargs) -> dict:
    recipient = args[0] if args else kwargs.get("recipient", "")
    body      = args[1] if len(args) > 1 else kwargs.get("body", "")
    return {"action": "sms_dispatch", "recipient": recipient, "body_len": len(body), "status": "QUEUED"}

def _handle_extract_research(args, kwargs) -> dict:
    url   = args[0] if args else kwargs.get("url", "")
    topic = args[1] if len(args) > 1 else kwargs.get("topic", "")
    return {"action": "research_initiated", "url": url, "topic": topic}

def _handle_apply_watermark(args, kwargs) -> dict:
    content = kwargs.get("content", args[0] if args else "")
    sig = hmac.new(SECRET, content.encode() if content else b"", hashlib.sha256).hexdigest()
    return {"action": "watermark_applied", "hmac": sig, "algorithm": "HMAC-SHA256"}

def _handle_mesh_status(args, kwargs) -> dict:
    return {
        "mesh":       "converged",
        "node":       os.uname().nodename,
        "agents":     82,
        "ports": {
            "mcp_router":        PORT,
            "web_terminal":      7777,
            "api_bridge":        7780,
            "strategy_engine":   7786,
            "revenue_aggregator":7787,
            "zapier_bridge":     7788,
            "canva":             7789,
            "huggingface":       7790,
            "openrouter":        7791,
            "devto_hub":         7792,
            "supabase":          7793,
            "github_ops":        7794,
            "firebase":          7795,
            "coolify":           7796,
            "paypal":            7797,
            "circle":            7798,
            "lighthouse":        7799,
            "opensea":           7800,
            "langchain":         7801,
            "notion":            7802,
            "langsmith":         7803,
            "admin_api":         7804,
            "rss_hub":           7805,
            "rag_pipeline":      7806,
            "scraper_v2":        7807,
            "minimax":           7808,
            "base44":            7809,
            "gumroad":           7810,
            "printful":          7811,
            "coinbase":          7812,
            "pionex":            7813,
            "kucoin":            7814,
            "elevenlabs":        7815,
            "twitter":           7816,
            "sendgrid":          7817,
            "alchemy":           7818,
            "moralis":           7819,
            "coingecko":         7820,
            "xyo":               7821,
            "producthunt":       7822,
            "docker":            7823,
            "crawlbase":         7824,
            "bugcrowd":          7825,
            "osintaas":          7826,
            "leadgen":           7827,
            "nft_engine":        7828,
            "data_api":          7829,
            "aiaas":             7830,
            "cronjob":           7831,
            "swarm":             7832,
            "admin_dashboard":   7833,
            "deep_scan":         7834,
            "metrics":           7835,
            "logic_bucket":      7836,
        },
        "abn":        os.getenv("ABN", "56628117363"),
        "compliance": ["ISO_27001", "APRA_CPS234"],
    }

def _handle_figma_sync(args, kwargs) -> dict:
    file_key = kwargs.get("file_key", os.getenv("FIGMA_FILE_KEY", ""))
    return {"action": "figma_sync_queued", "file_key": file_key or "default_tokens"}

def _handle_canva_design(args, kwargs) -> dict:
    template = kwargs.get("template_id", args[0] if args else "social_post_square")
    title    = kwargs.get("title", "FractalMesh Design")
    return {"action": "canva_design_queued", "template": template, "title": title}

def _handle_hf_infer(args, kwargs) -> dict:
    task   = kwargs.get("task", "text_generation")
    inputs = kwargs.get("inputs", args[0] if args else "")
    return {"action": "hf_infer_queued", "task": task, "input_len": len(str(inputs))}

def _handle_openrouter_route(args, kwargs) -> dict:
    prompt = kwargs.get("prompt", args[0] if args else "")
    task   = kwargs.get("task", "draft")
    tier   = kwargs.get("tier", "balanced")
    return {"action": "or_route_queued", "task": task, "tier": tier,
            "prompt_len": len(prompt)}

def _handle_devto_publish(args, kwargs) -> dict:
    title  = kwargs.get("title", args[0] if args else "")
    series = kwargs.get("series", "fractalmesh_build")
    return {"action": "devto_publish_queued", "title": title, "series": series}

def _handle_zapier_fire(args, kwargs) -> dict:
    zap     = kwargs.get("zap", args[0] if args else "")
    payload = kwargs.get("payload", {})
    return {"action": "zapier_fire_queued", "zap": zap, "payload_keys": list(payload.keys())}

def _handle_supabase_sync(args, kwargs) -> dict:
    table = kwargs.get("table", args[0] if args else "all")
    return {"action": "supabase_sync_queued", "table": table}

def _handle_generate_content(args, kwargs) -> dict:
    topic  = kwargs.get("topic", args[0] if args else "")
    series = kwargs.get("series", "fractalmesh_build")
    tier   = kwargs.get("tier", "balanced")
    return {"action": "content_gen_queued", "topic": topic, "series": series, "tier": tier}

def _handle_github_op(args, kwargs) -> dict:
    op   = kwargs.get("op", args[0] if args else "list_repos")
    repo = kwargs.get("repo", "")
    return {"action": "github_op_queued", "op": op, "repo": repo}

def _handle_notion_sync(args, kwargs) -> dict:
    target = kwargs.get("target", args[0] if args else "all")
    return {"action": "notion_sync_queued", "target": target}

def _handle_rag_query(args, kwargs) -> dict:
    query = kwargs.get("query", args[0] if args else "")
    top_k = kwargs.get("top_k", 5)
    return {"action": "rag_query_queued", "query": query[:200], "top_k": top_k}

def _handle_scrape(args, kwargs) -> dict:
    url  = kwargs.get("url", args[0] if args else "")
    mode = kwargs.get("mode", "single")
    return {"action": "scrape_queued", "url": url, "mode": mode}

def _handle_dork(args, kwargs) -> dict:
    category = kwargs.get("category", args[0] if args else "linkedin_leads")
    return {"action": "dork_queued", "category": category}

def _handle_mm_generate(args, kwargs) -> dict:
    prompt = kwargs.get("prompt", args[0] if args else "")
    model  = kwargs.get("model", "MiniMax-M2.7")
    return {"action": "minimax_generate_queued", "model": model, "prompt_len": len(prompt)}

def _handle_mm_tts(args, kwargs) -> dict:
    text  = kwargs.get("text", args[0] if args else "")
    voice = kwargs.get("voice", "English_expressive_narrator")
    return {"action": "minimax_tts_queued", "voice": voice, "text_len": len(text)}

def _handle_rss_fetch(args, kwargs) -> dict:
    category = kwargs.get("category", "")
    return {"action": "rss_fetch_queued", "category": category}

def _handle_lighthouse_audit(args, kwargs) -> dict:
    url      = kwargs.get("url", args[0] if args else "")
    strategy = kwargs.get("strategy", "mobile")
    return {"action": "lighthouse_audit_queued", "url": url, "strategy": strategy}

def _handle_langchain_run(args, kwargs) -> dict:
    pipeline = kwargs.get("pipeline", args[0] if args else "summarize")
    return {"action": "langchain_run_queued", "pipeline": pipeline}

def _handle_coolify_deploy(args, kwargs) -> dict:
    app_uuid = kwargs.get("app_uuid", args[0] if args else "")
    return {"action": "coolify_deploy_queued", "app_uuid": app_uuid}

def _handle_firebase_sync(args, kwargs) -> dict:
    collection = kwargs.get("collection", "fm_mesh_state")
    return {"action": "firebase_sync_queued", "collection": collection}

def _handle_admin_broadcast(args, kwargs) -> dict:
    intent = kwargs.get("intent", ""); sub_kwargs = kwargs.get("kwargs", {})
    return {"action": "admin_broadcast_queued", "intent": intent, "kwargs": sub_kwargs}

def _handle_base44_op(args, kwargs) -> dict:
    op      = kwargs.get("op", args[0] if args else "list_apps")
    app_id  = kwargs.get("app_id", "")
    payload = kwargs.get("payload", {})
    return {"action": "base44_op_queued", "op": op, "app_id": app_id, "payload_keys": list(payload.keys())}

def _handle_gumroad(args, kwargs) -> dict:
    op      = kwargs.get("op", args[0] if args else "products")
    return {"action": "gumroad_queued", "op": op}

def _handle_printful(args, kwargs) -> dict:
    op      = kwargs.get("op", args[0] if args else "products")
    return {"action": "printful_queued", "op": op}

def _handle_coinbase(args, kwargs) -> dict:
    op      = kwargs.get("op", args[0] if args else "portfolio")
    return {"action": "coinbase_queued", "op": op}

def _handle_pionex(args, kwargs) -> dict:
    op      = kwargs.get("op", args[0] if args else "account")
    symbol  = kwargs.get("symbol", "")
    return {"action": "pionex_queued", "op": op, "symbol": symbol}

def _handle_kucoin(args, kwargs) -> dict:
    op      = kwargs.get("op", args[0] if args else "accounts")
    symbol  = kwargs.get("symbol", "")
    return {"action": "kucoin_queued", "op": op, "symbol": symbol}

def _handle_elevenlabs(args, kwargs) -> dict:
    text    = kwargs.get("text", args[0] if args else "")
    voice   = kwargs.get("voice_id", "")
    return {"action": "elevenlabs_tts_queued", "text_len": len(text), "voice_id": voice}

def _handle_twitter(args, kwargs) -> dict:
    op      = kwargs.get("op", args[0] if args else "tweet")
    text    = kwargs.get("text", "")
    return {"action": "twitter_queued", "op": op, "text_len": len(text)}

def _handle_sendgrid(args, kwargs) -> dict:
    to      = kwargs.get("to", args[0] if args else "")
    subject = kwargs.get("subject", "")
    return {"action": "sendgrid_queued", "to": to, "subject": subject}

def _handle_alchemy(args, kwargs) -> dict:
    op      = kwargs.get("op", args[0] if args else "balance")
    address = kwargs.get("address", "")
    return {"action": "alchemy_queued", "op": op, "address": address[:10] + "..." if address else ""}

def _handle_moralis(args, kwargs) -> dict:
    op      = kwargs.get("op", args[0] if args else "nfts")
    chain   = kwargs.get("chain", "eth")
    return {"action": "moralis_queued", "op": op, "chain": chain}

def _handle_coingecko(args, kwargs) -> dict:
    op      = kwargs.get("op", args[0] if args else "market")
    coin_id = kwargs.get("id", "")
    return {"action": "coingecko_queued", "op": op, "coin_id": coin_id}

def _handle_xyo(args, kwargs) -> dict:
    op      = kwargs.get("op", args[0] if args else "balance")
    return {"action": "xyo_queued", "op": op}

def _handle_producthunt(args, kwargs) -> dict:
    op      = kwargs.get("op", args[0] if args else "trending")
    return {"action": "producthunt_queued", "op": op}

def _handle_docker(args, kwargs) -> dict:
    op      = kwargs.get("op", args[0] if args else "containers")
    image   = kwargs.get("image", "")
    return {"action": "docker_queued", "op": op, "image": image}

def _handle_crawlbase(args, kwargs) -> dict:
    url     = kwargs.get("url", args[0] if args else "")
    js      = kwargs.get("js", False)
    return {"action": "crawlbase_queued", "url": url, "js_render": js}

def _handle_bugcrowd(args, kwargs) -> dict:
    op      = kwargs.get("op", args[0] if args else "programs")
    return {"action": "bugcrowd_queued", "op": op}

def _handle_osintaas(args, kwargs) -> dict:
    scan_type = kwargs.get("scan_type", args[0] if args else "person")
    target    = kwargs.get("target", kwargs.get("domain", kwargs.get("username", "")))
    depth     = kwargs.get("depth", "standard")
    return {"action": "osint_scan_queued", "scan_type": scan_type, "target": target, "depth": depth}

def _handle_leadgen(args, kwargs) -> dict:
    op       = kwargs.get("op", args[0] if args else "campaign_run")
    industry = kwargs.get("industry", "")
    location = kwargs.get("location", "")
    return {"action": "leadgen_queued", "op": op, "industry": industry, "location": location}

def _handle_nft_engine(args, kwargs) -> dict:
    op         = kwargs.get("op", args[0] if args else "generate_image")
    prompt     = kwargs.get("prompt", "fractal mesh")
    collection = kwargs.get("collection_id", 0)
    return {"action": "nft_engine_queued", "op": op, "prompt": prompt, "collection_id": collection}

def _handle_data_api(args, kwargs) -> dict:
    dataset = kwargs.get("dataset", args[0] if args else "leads")
    filters = kwargs.get("filters", {})
    fmt     = kwargs.get("format", "json")
    return {"action": "data_api_queued", "dataset": dataset, "filters": filters, "format": fmt}

def _handle_aiaas(args, kwargs) -> dict:
    endpoint = kwargs.get("endpoint", args[0] if args else "chat")
    model    = kwargs.get("model", "claude-3-5-haiku")
    prompt   = kwargs.get("prompt", kwargs.get("content", ""))
    return {"action": "aiaas_queued", "endpoint": endpoint, "model": model, "prompt": prompt}

_INTENTS = {
    # ── core ──────────────────────────────────────────────────────────────────
    "sync_samsung_calendar":   _handle_sync_calendar,
    "sync_samsung_reminder":   _handle_sync_reminder,
    "sync_google_workspace":   _handle_sync_workspace,
    "device_utilities_pulse":  _handle_device_pulse,
    "send_samsung_message":    _handle_send_message,
    "extract_research":        _handle_extract_research,
    "apply_synthid_watermark": _handle_apply_watermark,
    "mesh_status":             _handle_mesh_status,
    # ── platform integrations ─────────────────────────────────────────────────
    "figma_sync":              _handle_figma_sync,
    "canva_design":            _handle_canva_design,
    "hf_infer":                _handle_hf_infer,
    "openrouter_route":        _handle_openrouter_route,
    "devto_publish":           _handle_devto_publish,
    "zapier_fire":             _handle_zapier_fire,
    "supabase_sync":           _handle_supabase_sync,
    # ── extended platform suite ───────────────────────────────────────────────
    "github_op":               _handle_github_op,
    "notion_sync":             _handle_notion_sync,
    "rag_query":               _handle_rag_query,
    "scrape":                  _handle_scrape,
    "dork":                    _handle_dork,
    "minimax_generate":        _handle_mm_generate,
    "minimax_tts":             _handle_mm_tts,
    "rss_fetch":               _handle_rss_fetch,
    "lighthouse_audit":        _handle_lighthouse_audit,
    "langchain_run":           _handle_langchain_run,
    "coolify_deploy":          _handle_coolify_deploy,
    "firebase_sync":           _handle_firebase_sync,
    "admin_broadcast":         _handle_admin_broadcast,
    "generate_content":        _handle_generate_content,
    # ── base44 no-code app builder ────────────────────────────────────────────
    "base44_op":               _handle_base44_op,
    # ── revenue / commerce ────────────────────────────────────────────────────
    "gumroad":                 _handle_gumroad,
    "printful":                _handle_printful,
    "coinbase":                _handle_coinbase,
    # ── trading ───────────────────────────────────────────────────────────────
    "pionex":                  _handle_pionex,
    "kucoin":                  _handle_kucoin,
    # ── content / outreach ────────────────────────────────────────────────────
    "elevenlabs_tts":          _handle_elevenlabs,
    "twitter_post":            _handle_twitter,
    "sendgrid_send":           _handle_sendgrid,
    # ── web3 / data ───────────────────────────────────────────────────────────
    "alchemy_query":           _handle_alchemy,
    "moralis_query":           _handle_moralis,
    "coingecko_price":         _handle_coingecko,
    "xyo_op":                  _handle_xyo,
    # ── platform ──────────────────────────────────────────────────────────────
    "producthunt_op":          _handle_producthunt,
    "docker_op":               _handle_docker,
    "crawlbase_scrape":        _handle_crawlbase,
    "bugcrowd_op":             _handle_bugcrowd,
    # ── monetisation pipeline ─────────────────────────────────────────────────
    "osint_scan":              _handle_osintaas,
    "leadgen":                 _handle_leadgen,
    "nft_mint":                _handle_nft_engine,
    "data_query":              _handle_data_api,
    "ai_infer":                _handle_aiaas,
    # ── automation / orchestration ────────────────────────────────────────────
    "cron_job":                lambda a, k: {"action": "cron_queued", "name": k.get("name", ""), "schedule": k.get("schedule", "")},
    "swarm_batch":             lambda a, k: {"action": "swarm_queued", "batch_name": k.get("name", ""), "strategy": k.get("strategy", "parallel")},
    "admin_query":             lambda a, k: {"action": "admin_queued", "query": k.get("query", "dashboard")},
    "deep_scan":               lambda a, k: {"action": "deep_scan_queued", "target": k.get("target", ""), "scan_type": k.get("scan_type", "domain")},
    "metrics_push":            lambda a, k: {"action": "metrics_push_queued", "name": k.get("name", ""), "value": k.get("value", 0)},
    "committee_ask":           lambda a, k: {"action": "committee_queued", "question": k.get("question", ""), "strategy": k.get("strategy", "consensus")},
}

# ── HTTP handler ───────────────────────────────────────────────────────────────

class MCPHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silence default access log

    def _respond(self, code: int, body: Any):
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path in ("/health", "/api/mesh/status"):
            self._respond(200, _handle_mesh_status([], {}))
        else:
            self._respond(404, {"error": "not_found"})

    def do_POST(self):
        t0 = time.time()
        try:
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            sig    = self.headers.get("X-MCP-Signature", "")

            if not _verify_sig(sig, body):
                self._respond(401, {"error": "invalid_signature"})
                return

            raw = body.decode("utf-8", errors="replace")
            if not _lba_check(raw):
                log.warning("LBA_BLOCKED len=%d", len(raw))
                self._respond(403, {"error": "lba_firewall_blocked"})
                return

            data   = json.loads(raw)
            intent = data.get("intent", "")
            args   = data.get("args", [])
            kwargs = data.get("kwargs", {})

            handler = _INTENTS.get(intent)
            if handler:
                result = handler(args, kwargs)
                status = "SUCCESS"
                code   = 200
            else:
                result = {"error": "unknown_intent", "available": list(_INTENTS)}
                status = "UNKNOWN"
                code   = 400

            latency = time.time() - t0
            _db_log(intent, status, latency)
            log.info("intent=%s status=%s latency=%.3fs", intent, status, latency)
            self._respond(code, {"status": status, "intent": intent, **result})

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
    server = HTTPServer(("0.0.0.0", PORT), MCPHandler)
    log.info("MCP Router listening on port %d", PORT)
    log.info("LBA firewall active — %d banned patterns", len(_BANNED))
    log.info("Available intents: %s", ", ".join(_INTENTS))
    try:
        while _running:
            server.handle_request()
    finally:
        server.server_close()
        log.info("MCP Router stopped")

if __name__ == "__main__":
    main()
