#!/usr/bin/env python3
"""
fm_watchdog.py — FractalMesh Health Watchdog + Auto-Healer (Port 7852)
Monitors all 68 MCP-range agents (ports 7785-7852), auto-heals via PM2,
records uptime analytics and incident history in sovereign.db.
Samuel James Hiotis | ABN 56 628 117 363
"""

import os
import json
import sqlite3
import subprocess
import socket
import time
import threading
import signal
import logging
import hmac
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

# ---------------------------------------------------------------------------
# Vault / env bootstrap
# ---------------------------------------------------------------------------

_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _line in _vault.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

PORT           = int(os.getenv("WATCHDOG_PORT", "7852"))
CHECK_INTERVAL = int(os.getenv("WATCHDOG_CHECK_INTERVAL", "30"))
ADMIN_SECRET   = os.getenv("ADMIN_SECRET", "")
MCP_PORT       = int(os.getenv("MCP_PORT", "7785"))

ROOT = Path(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")))
DB   = ROOT / "database" / "sovereign.db"
LOG  = ROOT / "logs" / "fm-watchdog.log"

for _p in (ROOT, LOG.parent, DB.parent):
    _p.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [WATCHDOG] %(message)s",
    handlers=[logging.FileHandler(LOG), logging.StreamHandler()],
)
log = logging.getLogger("fm_watchdog")

# ---------------------------------------------------------------------------
# Agent registry  (name, port, pm2_name)  —  all 68 agents 7785 → 7852
# ---------------------------------------------------------------------------

AGENTS = [
    ("fm-mcp-router",          7785, "fm-mcp-router"),
    ("fm-strategy-engine",     7786, "fm-strategy-engine"),
    ("fm-revenue-aggregator",  7787, "fm-revenue-aggregator"),
    ("fm-zapier-bridge",       7788, "fm-zapier-bridge"),
    ("fm-canva",               7789, "fm-canva"),
    ("fm-huggingface",         7790, "fm-huggingface"),
    ("fm-openrouter",          7791, "fm-openrouter"),
    ("fm-devto-hub",           7792, "fm-devto-hub"),
    ("fm-supabase",            7793, "fm-supabase"),
    ("fm-github-ops",          7794, "fm-github-ops"),
    ("fm-firebase",            7795, "fm-firebase"),
    ("fm-coolify",             7796, "fm-coolify"),
    ("fm-paypal",              7797, "fm-paypal"),
    ("fm-circle",              7798, "fm-circle"),
    ("fm-lighthouse",          7799, "fm-lighthouse"),
    ("fm-opensea",             7800, "fm-opensea"),
    ("fm-langchain",           7801, "fm-langchain"),
    ("fm-notion",              7802, "fm-notion"),
    ("fm-langsmith",           7803, "fm-langsmith"),
    ("fm-admin-api",           7804, "fm-admin-api"),
    ("fm-rss-hub",             7805, "fm-rss-hub"),
    ("fm-rag-pipeline",        7806, "fm-rag-pipeline"),
    ("fm-scraper-v2",          7807, "fm-scraper-v2"),
    ("fm-minimax",             7808, "fm-minimax"),
    ("fm-base44",              7809, "fm-base44"),
    ("fm-gumroad",             7810, "fm-gumroad"),
    ("fm-printful",            7811, "fm-printful"),
    ("fm-coinbase",            7812, "fm-coinbase"),
    ("fm-pionex",              7813, "fm-pionex"),
    ("fm-kucoin",              7814, "fm-kucoin"),
    ("fm-elevenlabs",          7815, "fm-elevenlabs"),
    ("fm-twitter",             7816, "fm-twitter"),
    ("fm-sendgrid",            7817, "fm-sendgrid"),
    ("fm-alchemy",             7818, "fm-alchemy"),
    ("fm-moralis",             7819, "fm-moralis"),
    ("fm-coingecko",           7820, "fm-coingecko"),
    ("fm-xyo",                 7821, "fm-xyo"),
    ("fm-producthunt",         7822, "fm-producthunt"),
    ("fm-docker",              7823, "fm-docker"),
    ("fm-crawlbase",           7824, "fm-crawlbase"),
    ("fm-bugcrowd",            7825, "fm-bugcrowd"),
    ("fm-osintaas",            7826, "fm-osintaas"),
    ("fm-leadgen",             7827, "fm-leadgen"),
    ("fm-nft-engine",          7828, "fm-nft-engine"),
    ("fm-data-api",            7829, "fm-data-api"),
    ("fm-aiaas",               7830, "fm-aiaas"),
    ("fm-cronjob",             7831, "fm-cronjob"),
    ("fm-swarm",               7832, "fm-swarm"),
    ("fm-admin-dashboard",     7833, "fm-admin-dashboard"),
    ("fm-deep-scan",           7834, "fm-deep-scan"),
    ("fm-metrics",             7835, "fm-metrics"),
    ("fm-logic-bucket",        7836, "fm-logic-bucket"),
    ("fm-webhook-hub",         7837, "fm-webhook-hub"),
    ("fm-content-engine",      7838, "fm-content-engine"),
    ("fm-log-manager",         7839, "fm-log-manager"),
    ("fm-security-mon",        7840, "fm-security-mon"),
    ("fm-akash",               7841, "fm-akash"),
    ("fm-affiliate",           7842, "fm-affiliate"),
    ("fm-tunnel",              7843, "fm-tunnel"),
    ("fm-email-listener",      7844, "fm-email-listener"),
    ("fm-rate-limiter",        7845, "fm-rate-limiter"),
    ("fm-ab-testing",          7846, "fm-ab-testing"),
    ("fm-revenue-forecast",    7847, "fm-revenue-forecast"),
    ("fm-dork-engine",         7848, "fm-dork-engine"),
    ("fm-contract-forge",      7849, "fm-contract-forge"),
    ("fm-sovereign-ops",       7850, "fm-sovereign-ops"),
    ("fm-notifier",            7851, "fm-notifier"),
    ("fm-watchdog",            7852, "fm-watchdog"),
]

AGENT_COUNT = len(AGENTS)
# Build fast lookup maps
_AGENT_BY_NAME  = {name: (port, pm2) for name, port, pm2 in AGENTS}
_AGENT_BY_PORT  = {port: (name, pm2) for name, port, pm2 in AGENTS}

# In-memory consecutive-down counter per agent
_down_counts: dict[str, int] = {}
_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def _db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def _db_init() -> None:
    conn = _db_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS watchdog_checks (
            id          INTEGER PRIMARY KEY,
            agent_name  TEXT,
            port        INTEGER,
            status      TEXT,
            response_ms REAL,
            checked_at  REAL
        );
        CREATE TABLE IF NOT EXISTS watchdog_incidents (
            id            INTEGER PRIMARY KEY,
            agent_name    TEXT,
            port          INTEGER,
            incident_type TEXT,
            started_at    REAL,
            resolved_at   REAL,
            resolution    TEXT,
            auto_healed   INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS watchdog_actions (
            id         INTEGER PRIMARY KEY,
            agent_name TEXT,
            action     TEXT,
            result     TEXT,
            created_at REAL
        );
        CREATE INDEX IF NOT EXISTS idx_checks_agent
            ON watchdog_checks (agent_name, checked_at DESC);
        CREATE INDEX IF NOT EXISTS idx_incidents_agent
            ON watchdog_incidents (agent_name, started_at DESC);
    """)
    conn.commit()
    conn.close()
    log.info("Database initialised at %s", DB)

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _check_agent(name: str, port: int) -> tuple[str, float]:
    """Return (status, response_ms). status is 'up' or 'down'."""
    t0 = time.monotonic()
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            pass
        return "up", round((time.monotonic() - t0) * 1000, 3)
    except (ConnectionRefusedError, OSError, TimeoutError):
        return "down", round((time.monotonic() - t0) * 1000, 3)


def _record_check(name: str, port: int, status: str, response_ms: float) -> None:
    try:
        conn = _db_conn()
        conn.execute(
            "INSERT INTO watchdog_checks (agent_name, port, status, response_ms, checked_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (name, port, status, response_ms, time.time()),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        log.warning("record_check: %s", exc)


def _open_incident(name: str, port: int) -> int | None:
    """Open a new incident if none is open. Returns incident id."""
    try:
        conn = _db_conn()
        row = conn.execute(
            "SELECT id FROM watchdog_incidents "
            "WHERE agent_name=? AND resolved_at IS NULL ORDER BY id DESC LIMIT 1",
            (name,),
        ).fetchone()
        if row:
            conn.close()
            return int(row["id"])
        cur = conn.execute(
            "INSERT INTO watchdog_incidents "
            "(agent_name, port, incident_type, started_at) VALUES (?, ?, ?, ?)",
            (name, port, "port_down", time.time()),
        )
        iid = cur.lastrowid
        conn.commit()
        conn.close()
        return iid
    except Exception as exc:
        log.warning("open_incident: %s", exc)
        return None


def _close_incident(name: str, resolution: str, auto_healed: int = 0) -> None:
    try:
        conn = _db_conn()
        conn.execute(
            "UPDATE watchdog_incidents SET resolved_at=?, resolution=?, auto_healed=? "
            "WHERE agent_name=? AND resolved_at IS NULL",
            (time.time(), resolution, auto_healed, name),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        log.warning("close_incident: %s", exc)


def _log_action(name: str, action: str, result: str) -> None:
    try:
        conn = _db_conn()
        conn.execute(
            "INSERT INTO watchdog_actions (agent_name, action, result, created_at) "
            "VALUES (?, ?, ?, ?)",
            (name, action, result, time.time()),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        log.warning("log_action: %s", exc)


def _pm2_status(pm2_name: str) -> str:
    """Return PM2 process status string or 'unknown'."""
    try:
        result = subprocess.run(
            ["pm2", "show", pm2_name],
            capture_output=True, text=True, timeout=10,
        )
        output = result.stdout + result.stderr
        for line in output.splitlines():
            if "status" in line.lower():
                parts = line.split("│") if "│" in line else line.split("|")
                for part in parts:
                    part = part.strip()
                    if part in ("online", "stopped", "errored", "stopping",
                                "launching", "one-launch-status"):
                        return part
        if "online" in output:
            return "online"
        if "stopped" in output:
            return "stopped"
        if "errored" in output:
            return "errored"
        if "doesn't exist" in output or "not found" in output.lower():
            return "not_found"
        return "unknown"
    except Exception as exc:
        log.warning("pm2_status(%s): %s", pm2_name, exc)
        return "unknown"


def _pm2_restart(pm2_name: str) -> tuple[bool, str]:
    """Restart a PM2 process. Returns (success, output)."""
    try:
        result = subprocess.run(
            ["pm2", "restart", pm2_name],
            capture_output=True, text=True, timeout=30,
        )
        output = (result.stdout + result.stderr).strip()
        return result.returncode == 0, output
    except Exception as exc:
        return False, str(exc)


def _pm2_reload(pm2_name: str) -> tuple[bool, str]:
    """Reload a PM2 process (graceful). Returns (success, output)."""
    try:
        result = subprocess.run(
            ["pm2", "reload", pm2_name],
            capture_output=True, text=True, timeout=30,
        )
        output = (result.stdout + result.stderr).strip()
        return result.returncode == 0, output
    except Exception as exc:
        return False, str(exc)


def _run_heal(agent_name: str, pm2_name: str, port: int) -> dict:
    """
    Attempt to heal a downed agent via PM2.
    Logs action + updates incident. Returns result dict.
    """
    pm2_st = _pm2_status(pm2_name)
    log.info("heal(%s): pm2 status=%s", agent_name, pm2_st)

    if pm2_st == "not_found":
        result = f"PM2 has no process named '{pm2_name}'"
        _log_action(agent_name, f"heal_attempted pm2_status={pm2_st}", result)
        return {"action": "none", "pm2_status": pm2_st, "result": result}

    if pm2_st in ("stopped", "errored", "unknown"):
        success, output = _pm2_restart(pm2_name)
        action = "restart"
    else:
        # online but port not responding → graceful reload
        success, output = _pm2_reload(pm2_name)
        action = "reload"

    result_str = f"{'OK' if success else 'FAIL'}: {output[:300]}"
    _log_action(agent_name, f"{action} pm2_status={pm2_st}", result_str)

    if success:
        _close_incident(agent_name, f"auto-healed via pm2 {action}", auto_healed=1)
        log.info("heal(%s): %s succeeded", agent_name, action)
    else:
        log.warning("heal(%s): %s failed — %s", agent_name, action, output[:200])

    return {"action": action, "pm2_status": pm2_st, "success": success, "result": result_str}

# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

def _check_auth(handler: "WatchdogHandler") -> bool:
    if not ADMIN_SECRET:
        return True
    return hmac.compare_digest(handler.headers.get("X-Admin-Secret", ""), ADMIN_SECRET)

# ---------------------------------------------------------------------------
# Health-check daemon thread
# ---------------------------------------------------------------------------

def _health_check_loop() -> None:
    """Runs every CHECK_INTERVAL seconds, checking all agents."""
    log.info("Health-check thread started (interval=%ds)", CHECK_INTERVAL)
    while True:
        for name, port, pm2_name in AGENTS:
            status, response_ms = _check_agent(name, port)
            _record_check(name, port, status, response_ms)

            with _lock:
                if status == "up":
                    if _down_counts.get(name, 0) > 0:
                        _close_incident(name, "recovered naturally")
                    _down_counts[name] = 0
                else:
                    _down_counts[name] = _down_counts.get(name, 0) + 1
                    count = _down_counts[name]

                if status == "down" and count == 1:
                    _open_incident(name, port)
                    log.warning("Agent %s:%d is DOWN (consecutive=%d)", name, port, count)

                if status == "down" and count == 3:
                    log.warning("Auto-heal triggered for %s (3 consecutive failures)", name)
                    # Run heal outside lock in a thread to avoid blocking the loop
                    t = threading.Thread(
                        target=_run_heal, args=(name, pm2_name, port), daemon=True
                    )
                    t.start()

        time.sleep(CHECK_INTERVAL)

# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------

def _json_response(handler: "WatchdogHandler", code: int, data: dict | list) -> None:
    body = json.dumps(data, indent=2).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _parse_qs(query_string: str) -> dict[str, str]:
    params: dict[str, str] = {}
    if not query_string:
        return params
    for pair in query_string.split("&"):
        if "=" in pair:
            k, _, v = pair.partition("=")
            params[k] = v
    return params


class WatchdogHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # suppress default access log noise
        pass

    def _path_parts(self):
        path = self.path.split("?")[0].rstrip("/")
        return [p for p in path.split("/") if p]

    def _query(self):
        parts = self.path.split("?", 1)
        return _parse_qs(parts[1] if len(parts) > 1 else "")

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # GET dispatch
    # ------------------------------------------------------------------

    def do_GET(self):
        parts = self._path_parts()

        if parts == ["health"]:
            return self._get_health()
        if parts == ["status"]:
            return self._get_status()
        if len(parts) == 2 and parts[0] == "status":
            return self._get_status_agent(parts[1])
        if parts == ["incidents"]:
            return self._get_incidents()
        if len(parts) == 2 and parts[0] == "incidents":
            return self._get_incident_detail(parts[1])
        if parts == ["analytics"]:
            return self._get_analytics()
        if parts == ["uptime"]:
            return self._get_uptime()

        _json_response(self, 404, {"error": "not found"})

    # ------------------------------------------------------------------
    # POST dispatch
    # ------------------------------------------------------------------

    def do_POST(self):
        parts = self._path_parts()

        if parts == ["check_now"]:
            return self._post_check_now()
        if len(parts) == 2 and parts[0] == "heal":
            if not _check_auth(self):
                return _json_response(self, 401, {"error": "unauthorized"})
            if parts[1] == "all":
                return self._post_heal_all()
            return self._post_heal_agent(parts[1])

        _json_response(self, 404, {"error": "not found"})

    # ------------------------------------------------------------------
    # Endpoint implementations
    # ------------------------------------------------------------------

    def _get_health(self):
        _json_response(self, 200, {
            "status": "ok",
            "service": "fm-watchdog",
            "port": PORT,
            "monitoring": AGENT_COUNT,
        })

    def _get_status(self):
        try:
            conn = _db_conn()
            # Latest check per agent
            rows = conn.execute("""
                SELECT agent_name, port, status, response_ms, checked_at
                FROM watchdog_checks
                WHERE id IN (
                    SELECT MAX(id) FROM watchdog_checks GROUP BY agent_name
                )
                ORDER BY agent_name
            """).fetchall()
            conn.close()
        except Exception as exc:
            return _json_response(self, 500, {"error": str(exc)})

        agents = []
        up_count = 0
        for r in rows:
            s = r["status"]
            if s == "up":
                up_count += 1
            agents.append({
                "name": r["agent_name"],
                "port": r["port"],
                "status": s,
                "response_ms": r["response_ms"],
                "last_checked": r["checked_at"],
            })

        total = len(agents)
        _json_response(self, 200, {
            "total": total,
            "up": up_count,
            "down": total - up_count,
            "agents": agents,
        })

    def _get_status_agent(self, agent_name: str):
        if agent_name not in _AGENT_BY_NAME:
            return _json_response(self, 404, {"error": f"unknown agent: {agent_name}"})
        try:
            conn = _db_conn()
            rows = conn.execute("""
                SELECT agent_name, port, status, response_ms, checked_at
                FROM watchdog_checks WHERE agent_name=?
                ORDER BY checked_at DESC LIMIT 20
            """, (agent_name,)).fetchall()
            conn.close()
        except Exception as exc:
            return _json_response(self, 500, {"error": str(exc)})

        checks = [
            {
                "name": r["agent_name"],
                "port": r["port"],
                "status": r["status"],
                "response_ms": r["response_ms"],
                "checked_at": r["checked_at"],
            }
            for r in rows
        ]
        _json_response(self, 200, {"agent": agent_name, "checks": checks})

    def _get_incidents(self):
        q = self._query()
        agent_filter = q.get("agent", "")
        resolved_filter = q.get("resolved", "").lower()

        sql = "SELECT * FROM watchdog_incidents WHERE 1=1"
        params: list = []
        if agent_filter:
            sql += " AND agent_name=?"
            params.append(agent_filter)
        if resolved_filter == "false":
            sql += " AND resolved_at IS NULL"
        elif resolved_filter == "true":
            sql += " AND resolved_at IS NOT NULL"
        sql += " ORDER BY started_at DESC LIMIT 200"

        try:
            conn = _db_conn()
            rows = conn.execute(sql, params).fetchall()
            conn.close()
        except Exception as exc:
            return _json_response(self, 500, {"error": str(exc)})

        incidents = [dict(r) for r in rows]
        _json_response(self, 200, {"count": len(incidents), "incidents": incidents})

    def _get_incident_detail(self, incident_id: str):
        try:
            iid = int(incident_id)
        except ValueError:
            return _json_response(self, 400, {"error": "invalid incident id"})

        try:
            conn = _db_conn()
            row = conn.execute(
                "SELECT * FROM watchdog_incidents WHERE id=?", (iid,)
            ).fetchone()
            if not row:
                conn.close()
                return _json_response(self, 404, {"error": "incident not found"})

            actions = conn.execute(
                "SELECT * FROM watchdog_actions WHERE agent_name=? "
                "AND created_at >= ? ORDER BY created_at",
                (row["agent_name"], row["started_at"]),
            ).fetchall()
            conn.close()
        except Exception as exc:
            return _json_response(self, 500, {"error": str(exc)})

        data = dict(row)
        data["actions"] = [dict(a) for a in actions]
        _json_response(self, 200, data)

    def _get_analytics(self):
        now = time.time()
        day_ago = now - 86400
        try:
            conn = _db_conn()

            # Uptime % per agent (last 24h)
            rows_24h = conn.execute("""
                SELECT agent_name,
                       ROUND(100.0 * SUM(CASE WHEN status='up' THEN 1 ELSE 0 END) /
                             COUNT(*), 2) AS uptime_pct
                FROM watchdog_checks
                WHERE checked_at >= ?
                GROUP BY agent_name
            """, (day_ago,)).fetchall()

            # Mean time to recovery (auto-healed incidents)
            mttr_row = conn.execute("""
                SELECT ROUND(AVG(resolved_at - started_at), 2) AS mttr_s
                FROM watchdog_incidents
                WHERE auto_healed=1 AND resolved_at IS NOT NULL
            """).fetchone()

            # Most unstable agent (most incidents in last 24h)
            unstable_row = conn.execute("""
                SELECT agent_name, COUNT(*) AS cnt
                FROM watchdog_incidents
                WHERE started_at >= ?
                GROUP BY agent_name
                ORDER BY cnt DESC LIMIT 1
            """, (day_ago,)).fetchone()

            # Total incidents today
            today_start = now - (now % 86400)
            total_today = conn.execute(
                "SELECT COUNT(*) AS n FROM watchdog_incidents WHERE started_at >= ?",
                (today_start,),
            ).fetchone()["n"]

            conn.close()
        except Exception as exc:
            return _json_response(self, 500, {"error": str(exc)})

        uptime_map = {r["agent_name"]: r["uptime_pct"] for r in rows_24h}
        _json_response(self, 200, {
            "uptime_pct_24h": uptime_map,
            "mttr_seconds": mttr_row["mttr_s"] if mttr_row else None,
            "most_unstable_agent": dict(unstable_row) if unstable_row else None,
            "total_incidents_today": total_today,
        })

    def _get_uptime(self):
        now = time.time()
        day_ago  = now - 86400
        week_ago = now - 86400 * 7
        try:
            conn = _db_conn()
            rows_24h = conn.execute("""
                SELECT agent_name,
                       ROUND(100.0 * SUM(CASE WHEN status='up' THEN 1 ELSE 0 END) /
                             COUNT(*), 2) AS pct
                FROM watchdog_checks WHERE checked_at >= ?
                GROUP BY agent_name
            """, (day_ago,)).fetchall()

            rows_7d = conn.execute("""
                SELECT agent_name,
                       ROUND(100.0 * SUM(CASE WHEN status='up' THEN 1 ELSE 0 END) /
                             COUNT(*), 2) AS pct
                FROM watchdog_checks WHERE checked_at >= ?
                GROUP BY agent_name
            """, (week_ago,)).fetchall()
            conn.close()
        except Exception as exc:
            return _json_response(self, 500, {"error": str(exc)})

        pct_24h = {r["agent_name"]: r["pct"] for r in rows_24h}
        pct_7d  = {r["agent_name"]: r["pct"] for r in rows_7d}

        result = []
        for name, port, _ in AGENTS:
            result.append({
                "agent": name,
                "port": port,
                "uptime_pct_24h": pct_24h.get(name),
                "uptime_pct_7d":  pct_7d.get(name),
            })
        _json_response(self, 200, result)

    # ------------------------------------------------------------------
    # POST handlers
    # ------------------------------------------------------------------

    def _post_check_now(self):
        body = self._body()
        agent_name = body.get("agent", "")

        if agent_name:
            if agent_name not in _AGENT_BY_NAME:
                return _json_response(self, 404, {"error": f"unknown agent: {agent_name}"})
            port, pm2_name = _AGENT_BY_NAME[agent_name]
            status, response_ms = _check_agent(agent_name, port)
            _record_check(agent_name, port, status, response_ms)
            with _lock:
                if status == "up":
                    if _down_counts.get(agent_name, 0) > 0:
                        _close_incident(agent_name, "recovered (manual check)")
                    _down_counts[agent_name] = 0
                else:
                    _down_counts[agent_name] = _down_counts.get(agent_name, 0) + 1
            return _json_response(self, 200, {
                "agent": agent_name, "port": port,
                "status": status, "response_ms": response_ms,
            })

        # Check all — run in background thread, return immediately
        def _check_all_bg():
            results = []
            for name, port, pm2_name in AGENTS:
                status, response_ms = _check_agent(name, port)
                _record_check(name, port, status, response_ms)
                with _lock:
                    if status == "up":
                        if _down_counts.get(name, 0) > 0:
                            _close_incident(name, "recovered (manual check)")
                        _down_counts[name] = 0
                    else:
                        _down_counts[name] = _down_counts.get(name, 0) + 1
                results.append({"agent": name, "port": port, "status": status,
                                 "response_ms": response_ms})

        threading.Thread(target=_check_all_bg, daemon=True).start()
        _json_response(self, 202, {
            "status": "accepted",
            "message": f"Checking all {AGENT_COUNT} agents in background",
        })

    def _post_heal_agent(self, agent_name: str):
        if agent_name not in _AGENT_BY_NAME:
            return _json_response(self, 404, {"error": f"unknown agent: {agent_name}"})
        port, pm2_name = _AGENT_BY_NAME[agent_name]
        result = _run_heal(agent_name, pm2_name, port)
        _json_response(self, 200, result)

    def _post_heal_all(self):
        # Find all currently down agents from latest checks
        try:
            conn = _db_conn()
            rows = conn.execute("""
                SELECT agent_name, port FROM watchdog_checks
                WHERE id IN (
                    SELECT MAX(id) FROM watchdog_checks GROUP BY agent_name
                ) AND status='down'
            """).fetchall()
            conn.close()
        except Exception as exc:
            return _json_response(self, 500, {"error": str(exc)})

        healed = []
        for r in rows:
            name = r["agent_name"]
            port = r["port"]
            if name in _AGENT_BY_NAME:
                _, pm2_name = _AGENT_BY_NAME[name]
                res = _run_heal(name, pm2_name, port)
                healed.append({"agent": name, "port": port, **res})

        _json_response(self, 200, {"healed": len(healed), "agents": healed})

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _signal_handler(sig, frame):
    log.info("Signal %d received, shutting down", sig)
    raise SystemExit(0)


def main():
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    _db_init()

    checker = threading.Thread(target=_health_check_loop, daemon=True, name="health-checker")
    checker.start()

    server = HTTPServer(("0.0.0.0", PORT), WatchdogHandler)
    log.info("fm-watchdog listening on 0.0.0.0:%d  (monitoring %d agents)", PORT, AGENT_COUNT)
    try:
        server.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        log.info("fm-watchdog stopping")
        server.server_close()


if __name__ == "__main__":
    main()
