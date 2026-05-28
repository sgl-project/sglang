#!/usr/bin/env python3
"""
fm_admin_dashboard.py — Admin Dashboard Aggregator Agent (Port 7833)
FractalMesh OMEGA Titan — single-pane-of-glass over all 76 agents.
Credentials sourced from ~/.secrets/fractal.env at runtime — never hardcoded.
Samuel James Hiotis | ABN 56 628 117 363
"""
import hashlib
import hmac
import json
import logging
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ── vault ─────────────────────────────────────────────────────────────────────
_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _line in _vault.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

# ── config ────────────────────────────────────────────────────────────────────
PORT           = int(os.getenv("DASHBOARD_PORT", "7833"))
MCP_PORT       = int(os.getenv("MCP_PORT", "7785"))
MCP_SECRET     = os.getenv("MCP_SECRET", "fm_mcp_internal").encode()
ADMIN_SECRET   = os.getenv("ADMIN_SECRET", "")
ROOT           = Path(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")))
DB             = ROOT / "database" / "sovereign.db"
LOG_DIR        = ROOT / "logs"
LOG_FILE       = ROOT / "logs" / "fm_admin_dashboard.log"

ROOT.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
DB.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [FM-ADMIN-DASHBOARD] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("fm_admin_dashboard")

# ── known agents registry (76 agents) ────────────────────────────────────────
KNOWN_AGENTS = [
    {"name": "fm-mcp-router",         "port": 7785},
    {"name": "fm-strategy-engine",    "port": 7786},
    {"name": "fm-revenue-aggregator", "port": 7787},
    {"name": "fm-zapier-bridge",      "port": 7788},
    {"name": "fm-canva",              "port": 7789},
    {"name": "fm-huggingface",        "port": 7790},
    {"name": "fm-openrouter",         "port": 7791},
    {"name": "fm-devto-hub",          "port": 7792},
    {"name": "fm-supabase",           "port": 7793},
    {"name": "fm-github-ops",         "port": 7794},
    {"name": "fm-firebase",           "port": 7795},
    {"name": "fm-coolify",            "port": 7796},
    {"name": "fm-paypal",             "port": 7797},
    {"name": "fm-circle",             "port": 7798},
    {"name": "fm-lighthouse",         "port": 7799},
    {"name": "fm-opensea",            "port": 7800},
    {"name": "fm-langchain",          "port": 7801},
    {"name": "fm-notion",             "port": 7802},
    {"name": "fm-langsmith",          "port": 7803},
    {"name": "fm-admin-api",          "port": 7804},
    {"name": "fm-rss-hub",            "port": 7805},
    {"name": "fm-rag-pipeline",       "port": 7806},
    {"name": "fm-scraper-v2",         "port": 7807},
    {"name": "fm-minimax",            "port": 7808},
    {"name": "fm-base44",             "port": 7809},
    {"name": "fm-gumroad",            "port": 7810},
    {"name": "fm-printful",           "port": 7811},
    {"name": "fm-coinbase",           "port": 7812},
    {"name": "fm-pionex",             "port": 7813},
    {"name": "fm-kucoin",             "port": 7814},
    {"name": "fm-elevenlabs",         "port": 7815},
    {"name": "fm-twitter",            "port": 7816},
    {"name": "fm-sendgrid",           "port": 7817},
    {"name": "fm-alchemy",            "port": 7818},
    {"name": "fm-moralis",            "port": 7819},
    {"name": "fm-coingecko",          "port": 7820},
    {"name": "fm-xyo",                "port": 7821},
    {"name": "fm-producthunt",        "port": 7822},
    {"name": "fm-docker",             "port": 7823},
    {"name": "fm-crawlbase",          "port": 7824},
    {"name": "fm-bugcrowd",           "port": 7825},
    {"name": "fm-osintaas",           "port": 7826},
    {"name": "fm-leadgen",            "port": 7827},
    {"name": "fm-nft-engine",         "port": 7828},
    {"name": "fm-data-api",           "port": 7829},
    {"name": "fm-aiaas",              "port": 7830},
    {"name": "fm-cronjob",            "port": 7831},
    {"name": "fm-swarm",              "port": 7832},
    {"name": "fm-admin-dashboard",    "port": 7833},
    {"name": "fm-mesh-integrator",    "port": 8090},
    {"name": "fm-stripe-mon",         "port": 8091},
    {"name": "fm-gitops-runner",      "port": 8092},
    {"name": "fm-pulse-bus",          "port": 5060},
    {"name": "fm-notes-registrar",    "port": 5061},
    {"name": "fm-tunnel",             "port": 5062},
    {"name": "fm-geosignal",          "port": 5057},
    {"name": "fm-pod",                "port": 5058},
    {"name": "fm-domain",             "port": 5059},
    {"name": "fm-gateway",            "port": 8000},
    {"name": "fm-terminal-bridge",    "port": 5062},
    {"name": "fm-admob-bridge",       "port": 7840},
    {"name": "fm-advert",             "port": 7841},
    {"name": "fm-auto-advert",        "port": 7842},
    {"name": "fm-azr-rl",             "port": 7843},
    {"name": "fm-bounty",             "port": 7844},
    {"name": "fm-carbon-credits",     "port": 7845},
    {"name": "fm-contract-forge",     "port": 7846},
    {"name": "fm-delivery",           "port": 7847},
    {"name": "fm-device-bridge",      "port": 7848},
    {"name": "fm-dork-engine",        "port": 7849},
    {"name": "fm-dorking",            "port": 7850},
    {"name": "fm-drip-agent",         "port": 7851},
    {"name": "fm-email-listener",     "port": 7852},
    {"name": "fm-enochian-hash",      "port": 7853},
    {"name": "fm-figma",              "port": 7854},
    {"name": "fm-geo-validator",      "port": 7855},
    {"name": "fm-immortality",        "port": 7856},
]

# ── database helpers ───────────────────────────────────────────────────────────
def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB), timeout=15, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS dashboard_snapshots (
                id            INTEGER PRIMARY KEY,
                snapshot_type TEXT,
                data          TEXT,
                created_at    REAL
            );
            CREATE TABLE IF NOT EXISTS dashboard_alerts (
                id           INTEGER PRIMARY KEY,
                level        TEXT,
                source       TEXT,
                message      TEXT,
                acknowledged INTEGER DEFAULT 0,
                created_at   REAL
            );
        """)
    log.info("DB tables ensured.")


def _query_table(conn, table: str, columns: str, where: str = "", params: tuple = ()):
    """Safe SELECT with table existence guard. Returns list of Row or []."""
    try:
        sql = f"SELECT {columns} FROM {table}"
        if where:
            sql += f" WHERE {where}"
        return conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        return []


# ── HTTP helpers ───────────────────────────────────────────────────────────────
def _internal_get(url: str, timeout: int = 2) -> dict:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw)
    except Exception as exc:
        return {"error": str(exc)}


def _internal_post(url: str, body_dict: dict, timeout: int = 5) -> dict:
    try:
        payload = json.dumps(body_dict).encode()
        sig = hmac.new(MCP_SECRET, payload, hashlib.sha256).hexdigest()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type":    "application/json",
                "X-MCP-Signature": sig,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw)
    except Exception as exc:
        return {"error": str(exc)}


def _check_auth(handler) -> bool:
    auth = handler.headers.get("Authorization", "")
    if not ADMIN_SECRET:
        return False
    return auth == f"Bearer {ADMIN_SECRET}"


# ── aggregation helpers ────────────────────────────────────────────────────────
def _today_start() -> float:
    now = datetime.now(timezone.utc)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight.timestamp()


def _fetch_dashboard_data() -> dict:
    ts = time.time()
    today_start = _today_start()

    # 1. MCP router
    mesh = _internal_get(f"http://127.0.0.1:{MCP_PORT}/health", timeout=2)

    # 2. Revenue aggregator
    revenue_raw = _internal_get("http://127.0.0.1:7787/summary", timeout=2)

    # 3. Gumroad
    gumroad = _internal_get("http://127.0.0.1:7810/stats", timeout=2)

    # 4. Lead Gen
    leads_raw = _internal_get("http://127.0.0.1:7827/analytics", timeout=2)

    # 5. CoinGecko
    crypto = _internal_get("http://127.0.0.1:7820/prices?coins=bitcoin,ethereum", timeout=2)

    # 6. AIaaS — DB query
    aiaas_requests_today = 0
    aiaas_tokens_today = 0
    try:
        with _get_db() as conn:
            rows = _query_table(
                conn, "aiaas_requests", "COUNT(*), COALESCE(SUM(tokens_used),0)",
                "created_at > ?", (today_start,)
            )
            if rows:
                aiaas_requests_today = rows[0][0] or 0
                aiaas_tokens_today   = rows[0][1] or 0
    except Exception:
        pass

    # 7. Data API — DB query
    data_api_today = 0
    try:
        with _get_db() as conn:
            rows = _query_table(
                conn, "api_requests", "COUNT(*)",
                "created_at > ?", (today_start,)
            )
            if rows:
                data_api_today = rows[0][0] or 0
    except Exception:
        pass

    # 8. CronJob
    cron = _internal_get("http://127.0.0.1:7831/analytics", timeout=2)

    # 9. Swarm
    swarm = _internal_get("http://127.0.0.1:7832/analytics", timeout=2)

    # Assemble
    result = {
        "timestamp": int(ts),
        "mesh":      mesh,
        "revenue": {
            "gumroad":   gumroad,
            "total":     revenue_raw,
            "total_today": 0,
        },
        "leads":  leads_raw,
        "crypto": crypto,
        "aiaas": {
            "requests_today": aiaas_requests_today,
            "tokens_today":   aiaas_tokens_today,
        },
        "data_api": {
            "requests_today": data_api_today,
        },
        "automation": {
            "cron":  cron,
            "swarm": swarm,
        },
    }

    # Persist snapshot
    try:
        with _get_db() as conn:
            conn.execute(
                "INSERT INTO dashboard_snapshots (snapshot_type, data, created_at) VALUES (?,?,?)",
                ("full", json.dumps(result), ts),
            )
    except Exception as exc:
        log.warning("Snapshot save failed: %s", exc)

    return result


# ── request handler ───────────────────────────────────────────────────────────
class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "FractalMesh/1.0"

    def log_message(self, fmt, *args):
        log.info(fmt, *args)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _send_json(self, code: int, data):
        body = json.dumps(data, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _unauthorized(self):
        body = json.dumps({"error": "unauthorized"}).encode()
        self.send_response(401)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("WWW-Authenticate", 'X-Admin-Secret realm="FractalMesh"')
        self.end_headers()
        self.wfile.write(body)

    def _not_found(self):
        self._send_json(404, {"error": "not found"})

    def _bad_request(self, msg="bad request"):
        self._send_json(400, {"error": msg})

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode())
        except Exception:
            return {}

    def _parse_qs(self) -> dict:
        parsed = urllib.parse.urlparse(self.path)
        return dict(urllib.parse.parse_qsl(parsed.query))

    def _path_parts(self):
        return urllib.parse.urlparse(self.path).path.strip("/").split("/")

    # ── GET routing ──────────────────────────────────────────────────────────
    def do_GET(self):
        parts = self._path_parts()
        route = parts[0] if parts else ""

        # /health — no auth
        if route == "health" and len(parts) == 1:
            self._send_json(200, {
                "status":  "ok",
                "service": "fm-admin-dashboard",
                "port":    PORT,
            })
            return

        if not _check_auth(self):
            self._unauthorized()
            return

        if route == "dashboard" and len(parts) == 1:
            self._handle_dashboard()
        elif route == "revenue" and len(parts) == 1:
            self._handle_revenue()
        elif route == "leads" and len(parts) == 1:
            self._handle_leads()
        elif route == "agents" and len(parts) == 1:
            self._handle_agents()
        elif route == "logs" and len(parts) == 1:
            self._handle_logs()
        elif route == "alerts" and len(parts) == 1:
            self._handle_alerts_get()
        elif route == "snapshots" and len(parts) == 1:
            self._handle_snapshots_list()
        elif route == "snapshots" and len(parts) == 2:
            self._handle_snapshot_get(parts[1])
        elif route == "analytics" and len(parts) == 1:
            self._handle_analytics()
        else:
            self._not_found()

    # ── POST routing ─────────────────────────────────────────────────────────
    def do_POST(self):
        parts = self._path_parts()
        route = parts[0] if parts else ""

        if not _check_auth(self):
            self._unauthorized()
            return

        if route == "alerts" and len(parts) == 2 and parts[1] == "create":
            self._handle_alert_create()
        elif route == "alerts" and len(parts) == 3 and parts[2] == "acknowledge":
            self._handle_alert_ack(parts[1])
        elif route == "mesh" and len(parts) == 2 and parts[1] == "broadcast":
            self._handle_mesh_broadcast()
        else:
            self._not_found()

    # ── endpoint handlers ─────────────────────────────────────────────────────
    def _handle_dashboard(self):
        data = _fetch_dashboard_data()
        self._send_json(200, data)

    def _handle_revenue(self):
        today_start = _today_start()
        result_today = {
            "gumroad_sales":      0,
            "gumroad_revenue":    0.0,
            "printful_orders":    0,
            "coinbase_orders":    0,
            "coinbase_revenue":   0.0,
            "kucoin_orders":      0,
            "bugcrowd_resolved":  0,
        }
        result_alltime = {
            "gumroad_sales":      0,
            "gumroad_revenue":    0.0,
            "printful_orders":    0,
            "coinbase_orders":    0,
            "coinbase_revenue":   0.0,
            "kucoin_orders":      0,
            "bugcrowd_resolved":  0,
        }
        try:
            with _get_db() as conn:
                # gumroad_sales today
                rows = _query_table(conn, "gumroad_sales",
                                    "COUNT(*), COALESCE(SUM(price),0)",
                                    "created_at > ?", (today_start,))
                if rows:
                    result_today["gumroad_sales"]   = rows[0][0] or 0
                    result_today["gumroad_revenue"]  = round(float(rows[0][1] or 0), 2)
                rows = _query_table(conn, "gumroad_sales",
                                    "COUNT(*), COALESCE(SUM(price),0)")
                if rows:
                    result_alltime["gumroad_sales"]   = rows[0][0] or 0
                    result_alltime["gumroad_revenue"]  = round(float(rows[0][1] or 0), 2)

                # printful_orders today
                rows = _query_table(conn, "printful_orders", "COUNT(*)",
                                    "created_at > ?", (today_start,))
                if rows:
                    result_today["printful_orders"] = rows[0][0] or 0
                rows = _query_table(conn, "printful_orders", "COUNT(*)")
                if rows:
                    result_alltime["printful_orders"] = rows[0][0] or 0

                # coinbase_orders today
                rows = _query_table(conn, "coinbase_orders",
                                    "COUNT(*), COALESCE(SUM(total_value_usd),0)",
                                    "created_at > ?", (today_start,))
                if rows:
                    result_today["coinbase_orders"]  = rows[0][0] or 0
                    result_today["coinbase_revenue"] = round(float(rows[0][1] or 0), 2)
                rows = _query_table(conn, "coinbase_orders",
                                    "COUNT(*), COALESCE(SUM(total_value_usd),0)")
                if rows:
                    result_alltime["coinbase_orders"]  = rows[0][0] or 0
                    result_alltime["coinbase_revenue"] = round(float(rows[0][1] or 0), 2)

                # kucoin_orders today
                rows = _query_table(conn, "kucoin_orders", "COUNT(*)",
                                    "created_at > ?", (today_start,))
                if rows:
                    result_today["kucoin_orders"] = rows[0][0] or 0
                rows = _query_table(conn, "kucoin_orders", "COUNT(*)")
                if rows:
                    result_alltime["kucoin_orders"] = rows[0][0] or 0

                # bugcrowd resolved today
                rows = _query_table(conn, "bugcrowd_submissions", "COUNT(*)",
                                    "created_at > ? AND status='resolved'", (today_start,))
                if rows:
                    result_today["bugcrowd_resolved"] = rows[0][0] or 0
                rows = _query_table(conn, "bugcrowd_submissions", "COUNT(*)",
                                    "status='resolved'")
                if rows:
                    result_alltime["bugcrowd_resolved"] = rows[0][0] or 0

        except Exception as exc:
            log.warning("Revenue query error: %s", exc)

        self._send_json(200, {"today": result_today, "all_time": result_alltime})

    def _handle_leads(self):
        result = {
            "total": 0, "new": 0, "contacted": 0,
            "responded": 0, "converted": 0,
            "campaigns_total": 0, "campaigns_active": 0,
        }
        try:
            with _get_db() as conn:
                rows = _query_table(conn, "leads", "COUNT(*)")
                if rows:
                    result["total"] = rows[0][0] or 0

                for status_val, key in [
                    ("new",       "new"),
                    ("contacted", "contacted"),
                    ("responded", "responded"),
                    ("converted", "converted"),
                ]:
                    rows = _query_table(conn, "leads", "COUNT(*)",
                                        "status=?", (status_val,))
                    if rows:
                        result[key] = rows[0][0] or 0

                rows = _query_table(conn, "campaigns", "COUNT(*)")
                if rows:
                    result["campaigns_total"] = rows[0][0] or 0

                rows = _query_table(conn, "campaigns", "COUNT(*)",
                                    "status='active'")
                if rows:
                    result["campaigns_active"] = rows[0][0] or 0

        except Exception as exc:
            log.warning("Leads query error: %s", exc)

        self._send_json(200, result)

    def _handle_agents(self):
        results = []
        lock = threading.Lock()

        def check_agent(agent):
            port = agent["port"]
            url  = f"http://127.0.0.1:{port}/health"
            t0   = time.time()
            try:
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=1) as resp:
                    resp.read()
                ms = int((time.time() - t0) * 1000)
                entry = {
                    "name":        agent["name"],
                    "port":        port,
                    "status":      "up",
                    "response_ms": ms,
                }
            except Exception:
                entry = {
                    "name":        agent["name"],
                    "port":        port,
                    "status":      "down",
                    "response_ms": None,
                }
            with lock:
                results.append(entry)

        threads = [threading.Thread(target=check_agent, args=(a,), daemon=True)
                   for a in KNOWN_AGENTS]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2)

        results.sort(key=lambda x: x["port"])
        self._send_json(200, results)

    def _handle_logs(self):
        qs = self._parse_qs()
        agent = qs.get("agent", "")
        try:
            lines_n = int(qs.get("lines", "50"))
        except ValueError:
            lines_n = 50

        if not agent:
            self._bad_request("agent param required")
            return

        # sanitise agent name — strip non-alphanum/dash
        safe_agent = "".join(c for c in agent if c.isalnum() or c in "-_")
        if not safe_agent:
            self._bad_request("invalid agent name")
            return

        log_path = LOG_DIR / f"{safe_agent}-out.log"
        if not log_path.exists():
            log_path = LOG_DIR / f"{safe_agent}-error.log"

        if not log_path.exists():
            self._send_json(200, {"agent": safe_agent, "lines": [], "error": "log not found"})
            return

        try:
            all_lines = log_path.read_text(errors="replace").splitlines()
            tail = all_lines[-lines_n:] if lines_n > 0 else all_lines
            self._send_json(200, {"agent": safe_agent, "lines": tail})
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def _handle_alerts_get(self):
        qs = self._parse_qs()
        conditions = []
        params: list = []

        if "acknowledged" in qs:
            val = qs["acknowledged"].lower()
            conditions.append("acknowledged=?")
            params.append(0 if val in ("false", "0", "no") else 1)
        if "level" in qs:
            conditions.append("level=?")
            params.append(qs["level"])

        where = " AND ".join(conditions)
        try:
            with _get_db() as conn:
                rows = _query_table(conn, "dashboard_alerts",
                                    "id, level, source, message, acknowledged, created_at",
                                    where, tuple(params))
                data = [dict(r) for r in rows]
            self._send_json(200, data)
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def _handle_alert_create(self):
        body = self._read_body()
        level   = body.get("level", "info")
        source  = body.get("source", "")
        message = body.get("message", "")

        if not message:
            self._bad_request("message required")
            return

        if level not in ("info", "warn", "error"):
            level = "info"

        now = time.time()
        try:
            with _get_db() as conn:
                cur = conn.execute(
                    "INSERT INTO dashboard_alerts (level, source, message, created_at)"
                    " VALUES (?,?,?,?)",
                    (level, source, message, now),
                )
                alert_id = cur.lastrowid
            log.info("Alert created id=%d level=%s source=%s", alert_id, level, source)
            self._send_json(201, {"alert_id": alert_id})
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def _handle_alert_ack(self, alert_id_str: str):
        try:
            alert_id = int(alert_id_str)
        except ValueError:
            self._bad_request("invalid alert id")
            return
        try:
            with _get_db() as conn:
                conn.execute(
                    "UPDATE dashboard_alerts SET acknowledged=1 WHERE id=?",
                    (alert_id,),
                )
            self._send_json(200, {"alert_id": alert_id, "acknowledged": True})
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def _handle_snapshots_list(self):
        try:
            with _get_db() as conn:
                rows = conn.execute(
                    "SELECT id, snapshot_type, created_at, LENGTH(data) AS size"
                    " FROM dashboard_snapshots"
                    " ORDER BY id DESC LIMIT 20"
                ).fetchall()
                data = [{"id": r["id"], "snapshot_type": r["snapshot_type"],
                          "created_at": r["created_at"], "size_bytes": r["size"]}
                        for r in rows]
            self._send_json(200, data)
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def _handle_snapshot_get(self, snap_id_str: str):
        try:
            snap_id = int(snap_id_str)
        except ValueError:
            self._bad_request("invalid snapshot id")
            return
        try:
            with _get_db() as conn:
                row = conn.execute(
                    "SELECT id, snapshot_type, data, created_at"
                    " FROM dashboard_snapshots WHERE id=?",
                    (snap_id,)
                ).fetchone()
            if not row:
                self._not_found()
                return
            payload = {
                "id":            row["id"],
                "snapshot_type": row["snapshot_type"],
                "created_at":    row["created_at"],
                "data":          json.loads(row["data"]),
            }
            self._send_json(200, payload)
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def _handle_mesh_broadcast(self):
        body   = self._read_body()
        intent = body.get("intent", "mesh_status")
        kwargs = body.get("kwargs", {})

        mcp_url  = f"http://127.0.0.1:{MCP_PORT}/"
        response = _internal_post(mcp_url, {"intent": intent, "kwargs": kwargs}, timeout=5)
        self._send_json(200, response)

    def _handle_analytics(self):
        # Agents up/down
        total_agents  = len(KNOWN_AGENTS)
        up_count      = 0
        for agent in KNOWN_AGENTS:
            result = _internal_get(
                f"http://127.0.0.1:{agent['port']}/health", timeout=1
            )
            if "error" not in result:
                up_count += 1

        # Revenue today
        revenue_today = 0.0
        today_start   = _today_start()
        try:
            with _get_db() as conn:
                for table, col in [
                    ("gumroad_sales",   "price"),
                    ("coinbase_orders", "total_value_usd"),
                ]:
                    rows = _query_table(conn, table,
                                        f"COALESCE(SUM({col}),0)",
                                        "created_at > ?", (today_start,))
                    if rows:
                        revenue_today += float(rows[0][0] or 0)
        except Exception:
            pass

        # Leads in pipeline
        leads_pipeline = 0
        try:
            with _get_db() as conn:
                rows = _query_table(conn, "leads", "COUNT(*)",
                                    "status NOT IN ('converted','closed')")
                if rows:
                    leads_pipeline = rows[0][0] or 0
        except Exception:
            pass

        # Active batches (swarm)
        active_batches = 0
        try:
            with _get_db() as conn:
                rows = _query_table(conn, "swarm_batches", "COUNT(*)",
                                    "status='running'")
                if rows:
                    active_batches = rows[0][0] or 0
        except Exception:
            pass

        # Unacknowledged alerts
        unacked_alerts = 0
        try:
            with _get_db() as conn:
                rows = _query_table(conn, "dashboard_alerts", "COUNT(*)",
                                    "acknowledged=0")
                if rows:
                    unacked_alerts = rows[0][0] or 0
        except Exception:
            pass

        self._send_json(200, {
            "agents_total":        total_agents,
            "agents_up":           up_count,
            "agents_down":         total_agents - up_count,
            "uptime_ratio":        round(up_count / total_agents, 3) if total_agents else 0,
            "revenue_today_usd":   round(revenue_today, 2),
            "leads_in_pipeline":   leads_pipeline,
            "active_batches":      active_batches,
            "unacked_alerts":      unacked_alerts,
            "timestamp":           int(time.time()),
        })


# ── server startup ─────────────────────────────────────────────────────────────
def main():
    _init_db()
    server = HTTPServer(("0.0.0.0", PORT), DashboardHandler)
    log.info("fm-admin-dashboard listening on port %d", PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutdown requested.")
    finally:
        server.server_close()
        log.info("Server stopped.")


if __name__ == "__main__":
    main()
