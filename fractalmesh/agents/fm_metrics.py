#!/usr/bin/env python3
"""
fm_metrics.py — Prometheus-compatible Metrics Exporter Agent (Port 7835)
FractalMesh OMEGA Titan — polls agent health, SQLite tables, system stats and
exposes them in Prometheus text format + JSON endpoints.
Credentials sourced from ~/.secrets/fractal.env at runtime — never hardcoded.
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import json
import time
import sqlite3
import logging
import threading
import urllib.request
import urllib.error
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional, Tuple

# ── vault ─────────────────────────────────────────────────────────────────────
_vault = Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _line in _vault.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

# ── config ────────────────────────────────────────────────────────────────────
PORT = int(os.getenv("METRICS_PORT", "7835"))
ROOT = Path(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")))
DB   = ROOT / "database" / "sovereign.db"
LOG  = ROOT / "logs" / "fm_metrics.log"

ROOT.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)
DB.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [FM-METRICS] %(message)s",
    handlers=[
        logging.FileHandler(LOG),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("fm_metrics")

# ── known agents (name → port) ────────────────────────────────────────────────
AGENTS: Dict[str, int] = {
    "fm-mcp-router":          7785,
    "fm-strategy-engine":     7786,
    "fm-revenue-aggregator":  7787,
    "fm-zapier-bridge":       7788,
    "fm-canva":               7789,
    "fm-huggingface":         7790,
    "fm-openrouter":          7791,
    "fm-devto-hub":           7792,
    "fm-supabase":            7793,
    "fm-github-ops":          7794,
    "fm-firebase":            7795,
    "fm-coolify":             7796,
    "fm-paypal":              7797,
    "fm-circle":              7798,
    "fm-lighthouse":          7799,
    "fm-opensea":             7800,
    "fm-langchain":           7801,
    "fm-notion":              7802,
    "fm-langsmith":           7803,
    "fm-admin-api":           7804,
    "fm-rss-hub":             7805,
    "fm-rag-pipeline":        7806,
    "fm-scraper":             7807,
    "fm-minimax":             7808,
    "fm-base44":              7809,
    "fm-gumroad":             7810,
    "fm-printful":            7811,
    "fm-coinbase":            7812,
    "fm-pionex":              7813,
    "fm-kucoin":              7814,
    "fm-elevenlabs":          7815,
    "fm-twitter":             7816,
    "fm-sendgrid":            7817,
    "fm-alchemy":             7818,
    "fm-moralis":             7819,
    "fm-coingecko":           7820,
    "fm-xyo":                 7821,
    "fm-producthunt":         7822,
    "fm-docker":              7823,
    "fm-crawlbase":           7824,
    "fm-bugcrowd":            7825,
    "fm-osintaas":            7826,
    "fm-leadgen":             7827,
    "fm-nft-engine":          7828,
    "fm-data-api":            7829,
    "fm-aiaas":               7830,
    "fm-cronjob":             7831,
    "fm-swarm":               7832,
    "fm-admin-dashboard":     7833,
}

# ── known SQLite tables to row-count ─────────────────────────────────────────
TRACKED_TABLES = [
    "leads", "campaigns", "sequences", "gumroad_sales", "printful_orders",
    "coinbase_orders", "kucoin_orders", "pionex_orders", "twitter_posts",
    "sendgrid_sends", "alchemy_events", "coingecko_queries", "osint_reports",
    "osint_leads", "deep_scans", "scan_findings", "swarm_batches",
    "swarm_tasks", "cron_jobs", "cron_runs", "api_requests", "aiaas_requests",
    "nft_tokens", "nft_listings", "nft_sales", "bugcrowd_submissions",
    "crawlbase_jobs",
]

# ── in-memory metrics store ───────────────────────────────────────────────────
_metrics_lock = threading.Lock()
_metrics: Dict[str, Dict[str, Any]] = {}

# Track when each agent was last seen up (for alert age calculation)
_agent_down_since: Dict[str, float] = {}

# ── helpers ───────────────────────────────────────────────────────────────────

def _prom_escape(s: str) -> str:
    """Escape a label value for Prometheus text format."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _set_metric(
    name: str,
    value: float,
    labels: Dict[str, str] = None,
    help_text: str = "",
    metric_type: str = "gauge",
) -> None:
    """Thread-safe update of a single metric in the in-memory store."""
    if labels is None:
        labels = {}
    with _metrics_lock:
        _metrics[name] = {
            "value":      value,
            "labels":     labels,
            "help":       help_text,
            "type":       metric_type,
            "updated_at": time.time(),
        }


def _format_prometheus() -> str:
    """Build the complete Prometheus text exposition from _metrics."""
    now_ms = int(time.time() * 1000)
    lines: list[str] = []
    with _metrics_lock:
        snapshot = dict(_metrics)

    # Group by base metric name (strip label suffix for HELP/TYPE blocks)
    seen_headers: set[str] = set()
    for metric_name, data in sorted(snapshot.items()):
        # Derive the base metric family name (everything before first "{")
        base = metric_name.split("{")[0]
        if base not in seen_headers:
            if data["help"]:
                lines.append(f"# HELP {base} {data['help']}")
            lines.append(f"# TYPE {base} {data['type']}")
            seen_headers.add(base)

        label_str = ""
        if data["labels"]:
            pairs = ",".join(
                f'{k}="{_prom_escape(str(v))}"'
                for k, v in sorted(data["labels"].items())
            )
            label_str = f"{{{pairs}}}"

        val = data["value"]
        val_str = f"{val:.6g}" if val != int(val) else str(int(val))
        lines.append(f"{base}{label_str} {val_str} {now_ms}")

    return "\n".join(lines) + "\n"


def _try_get(url: str, timeout: float = 1.0) -> Tuple[Optional[float], int]:
    """
    Perform a GET request.  Returns (response_time_ms, status_code).
    On connection error returns (None, 0).
    """
    t0 = time.monotonic()
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            elapsed_ms = (time.monotonic() - t0) * 1000
            return elapsed_ms, resp.status
    except Exception:
        return None, 0


def _read_proc_meminfo() -> Dict[str, int]:
    """Parse /proc/meminfo and return a dict of field → kB values."""
    result: Dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    try:
                        result[key] = int(parts[1])
                    except ValueError:
                        pass
    except OSError:
        pass
    return result


def _safe_count(conn: sqlite3.Connection, table: str, where: str = "") -> int:
    """Run a COUNT query; return 0 on OperationalError (table may not exist)."""
    try:
        sql = f"SELECT COUNT(*) FROM {table}"
        if where:
            sql += f" WHERE {where}"
        row = conn.execute(sql).fetchone()
        return int(row[0]) if row else 0
    except sqlite3.OperationalError:
        return 0


def _open_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB), timeout=15, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


# ── database init ─────────────────────────────────────────────────────────────

def _db_init() -> None:
    """Create required tables if they don't exist."""
    conn = _open_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS metrics_snapshots (
            id           INTEGER PRIMARY KEY,
            metric_name  TEXT,
            metric_value REAL,
            labels       TEXT,
            created_at   REAL
        );

        CREATE TABLE IF NOT EXISTS custom_metrics (
            id          INTEGER PRIMARY KEY,
            name        TEXT UNIQUE,
            value       REAL,
            labels      TEXT,
            help_text   TEXT,
            metric_type TEXT DEFAULT 'gauge',
            updated_at  REAL
        );
    """)
    conn.commit()
    conn.close()
    log.info("Database tables initialised.")


def _load_custom_metrics() -> None:
    """Load persisted custom metrics into _metrics at startup."""
    try:
        conn = _open_db()
        rows = conn.execute(
            "SELECT name, value, labels, help_text, metric_type, updated_at FROM custom_metrics"
        ).fetchall()
        conn.close()
        for row in rows:
            labels = {}
            try:
                labels = json.loads(row["labels"] or "{}")
            except (json.JSONDecodeError, TypeError):
                pass
            with _metrics_lock:
                _metrics[row["name"]] = {
                    "value":      float(row["value"] or 0),
                    "labels":     labels,
                    "help":       row["help_text"] or "",
                    "type":       row["metric_type"] or "gauge",
                    "updated_at": float(row["updated_at"] or time.time()),
                }
        log.info("Loaded %d custom metrics from DB.", len(rows))
    except Exception as exc:
        log.warning("Could not load custom metrics: %s", exc)


# ── background polling threads ────────────────────────────────────────────────

def _poll_agent_health() -> None:
    """Poll every 30 s; probe each known agent's /health endpoint."""
    log.info("Agent health poller started.")
    while True:
        for name, port in AGENTS.items():
            url = f"http://127.0.0.1:{port}/health"
            elapsed_ms, status = _try_get(url, timeout=1.0)
            up = 1 if (elapsed_ms is not None and status in (200, 204)) else 0
            labels = {"name": name, "port": str(port)}

            _set_metric(
                f"fm_agent_up__{name}",
                float(up),
                labels=labels,
                help_text="Agent health status (1=up, 0=down)",
                metric_type="gauge",
            )
            if elapsed_ms is not None:
                _set_metric(
                    f"fm_agent_response_ms__{name}",
                    elapsed_ms,
                    labels=labels,
                    help_text="Agent HTTP response time in milliseconds",
                    metric_type="gauge",
                )
            else:
                _set_metric(
                    f"fm_agent_response_ms__{name}",
                    0.0,
                    labels=labels,
                    help_text="Agent HTTP response time in milliseconds",
                    metric_type="gauge",
                )

            now = time.time()
            if up == 0:
                if name not in _agent_down_since:
                    _agent_down_since[name] = now
            else:
                _agent_down_since.pop(name, None)

        time.sleep(30)


def _poll_db_metrics() -> None:
    """Poll every 60 s; collect row counts and revenue figures from SQLite."""
    log.info("DB metrics poller started.")
    while True:
        try:
            conn = _open_db()

            # Table row counts
            for table in TRACKED_TABLES:
                count = _safe_count(conn, table)
                _set_metric(
                    f"fm_table_rows__{table}",
                    float(count),
                    labels={"table": table},
                    help_text="Number of rows in the named SQLite table",
                    metric_type="gauge",
                )

            # Revenue metrics
            try:
                row = conn.execute(
                    "SELECT COALESCE(SUM(price), 0) FROM gumroad_sales"
                ).fetchone()
                _set_metric(
                    "fm_gumroad_revenue_total",
                    float(row[0]) if row else 0.0,
                    help_text="Total Gumroad revenue (sum of price column)",
                    metric_type="counter",
                )
            except sqlite3.OperationalError:
                _set_metric("fm_gumroad_revenue_total", 0.0,
                            help_text="Total Gumroad revenue", metric_type="counter")

            _set_metric(
                "fm_gumroad_sales_total",
                float(_safe_count(conn, "gumroad_sales")),
                help_text="Total Gumroad sales count",
                metric_type="counter",
            )
            _set_metric(
                "fm_coinbase_orders_total",
                float(_safe_count(conn, "coinbase_orders")),
                help_text="Total Coinbase Commerce orders",
                metric_type="counter",
            )
            _set_metric(
                "fm_leads_total",
                float(_safe_count(conn, "leads")),
                help_text="Total leads in the database",
                metric_type="gauge",
            )
            _set_metric(
                "fm_leads_new",
                float(_safe_count(conn, "leads", "status='new'")),
                help_text="Leads with status=new",
                metric_type="gauge",
            )
            _set_metric(
                "fm_leads_converted",
                float(_safe_count(conn, "leads", "status='converted'")),
                help_text="Leads with status=converted",
                metric_type="gauge",
            )

            conn.close()
        except Exception as exc:
            log.warning("DB metrics poll error: %s", exc)

        time.sleep(60)


def _poll_system() -> None:
    """Poll every 10 s; collect CPU load, memory, disk via /proc and statvfs."""
    log.info("System metrics poller started.")
    while True:
        # CPU: 1-minute load average from /proc/loadavg
        try:
            with open("/proc/loadavg", "r") as fh:
                load1 = float(fh.read().split()[0])
        except OSError:
            load1 = 0.0
        _set_metric(
            "fm_cpu_percent",
            load1,
            help_text="System 1-minute load average from /proc/loadavg",
            metric_type="gauge",
        )

        # Memory: MemTotal - MemAvailable in bytes
        meminfo = _read_proc_meminfo()
        mem_total_kb = meminfo.get("MemTotal", 0)
        mem_avail_kb = meminfo.get("MemAvailable", 0)
        mem_used_bytes = (mem_total_kb - mem_avail_kb) * 1024
        _set_metric(
            "fm_memory_used_bytes",
            float(mem_used_bytes),
            help_text="Memory used in bytes (MemTotal - MemAvailable)",
            metric_type="gauge",
        )

        # Disk: used bytes on home filesystem
        try:
            sv = os.statvfs(os.path.expanduser("~"))
            disk_total = sv.f_blocks * sv.f_frsize
            disk_free  = sv.f_bfree  * sv.f_frsize
            disk_used  = disk_total - disk_free
        except OSError:
            disk_used = 0
        _set_metric(
            "fm_disk_used_bytes",
            float(disk_used),
            help_text="Disk space used in bytes on the home filesystem",
            metric_type="gauge",
        )

        time.sleep(10)


# ── HTTP handler ──────────────────────────────────────────────────────────────

class MetricsHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt: str, *args: Any) -> None:  # suppress default stdout
        log.debug("HTTP %s", fmt % args)

    # ── routing ──────────────────────────────────────────────────────────────

    def do_GET(self) -> None:
        path = self.path.split("?")[0].rstrip("/")

        if path == "/health":
            self._handle_health()
        elif path == "/metrics":
            self._handle_metrics_prom()
        elif path == "/metrics/json":
            self._handle_metrics_json()
        elif path.startswith("/metrics/") and len(path) > len("/metrics/"):
            metric_name = path[len("/metrics/"):]
            self._handle_metric_single(metric_name)
        elif path == "/summary":
            self._handle_summary()
        elif path == "/alerts":
            self._handle_alerts()
        else:
            self._send_json(404, {"error": "not found", "path": path})

    def do_POST(self) -> None:
        path = self.path.split("?")[0].rstrip("/")
        if path == "/metrics/push":
            self._handle_push()
        else:
            self._send_json(404, {"error": "not found"})

    # ── endpoint implementations ──────────────────────────────────────────────

    def _handle_health(self) -> None:
        self._send_json(200, {
            "status":            "ok",
            "service":           "fm-metrics",
            "port":              PORT,
            "agents_monitored":  len(AGENTS),
        })

    def _handle_metrics_prom(self) -> None:
        body = _format_prometheus().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_metrics_json(self) -> None:
        with _metrics_lock:
            snapshot = dict(_metrics)
        self._send_json(200, snapshot)

    def _handle_metric_single(self, name: str) -> None:
        with _metrics_lock:
            data = _metrics.get(name)
        if data is None:
            self._send_json(404, {"error": "metric not found", "name": name})
        else:
            self._send_json(200, {"name": name, **data})

    def _handle_push(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            payload = json.loads(raw)
        except Exception as exc:
            self._send_json(400, {"error": f"invalid JSON: {exc}"})
            return

        name = payload.get("name")
        if not name:
            self._send_json(400, {"error": "name is required"})
            return

        value      = float(payload.get("value", 0))
        labels     = payload.get("labels") or {}
        help_text  = payload.get("help", "")
        mtype      = payload.get("type", "gauge")
        now        = time.time()

        # Update in-memory store
        with _metrics_lock:
            _metrics[name] = {
                "value":      value,
                "labels":     labels,
                "help":       help_text,
                "type":       mtype,
                "updated_at": now,
            }

        # Upsert into custom_metrics table
        try:
            conn = _open_db()
            conn.execute(
                """
                INSERT INTO custom_metrics (name, value, labels, help_text, metric_type, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    value       = excluded.value,
                    labels      = excluded.labels,
                    help_text   = excluded.help_text,
                    metric_type = excluded.metric_type,
                    updated_at  = excluded.updated_at
                """,
                (name, value, json.dumps(labels), help_text, mtype, now),
            )
            conn.commit()
            conn.close()
        except Exception as exc:
            log.warning("Failed to persist custom metric '%s': %s", name, exc)

        self._send_json(200, {"name": name, "value": value, "updated_at": now})

    def _handle_summary(self) -> None:
        with _metrics_lock:
            snap = dict(_metrics)

        agents_up = sum(
            1 for k, v in snap.items()
            if k.startswith("fm_agent_up__") and v["value"] == 1
        )
        agents_total = len(AGENTS)

        def _val(key: str, default: float = 0.0) -> float:
            return float(snap.get(key, {}).get("value", default))

        summary = {
            "agents": {
                "up":    agents_up,
                "total": agents_total,
                "down":  agents_total - agents_up,
            },
            "revenue": {
                "gumroad_total":      _val("fm_gumroad_revenue_total"),
                "gumroad_sales":      _val("fm_gumroad_sales_total"),
                "coinbase_orders":    _val("fm_coinbase_orders_total"),
            },
            "leads": {
                "total":     _val("fm_leads_total"),
                "new":       _val("fm_leads_new"),
                "converted": _val("fm_leads_converted"),
            },
            "system": {
                "cpu_load_1m":        _val("fm_cpu_percent"),
                "memory_used_bytes":  _val("fm_memory_used_bytes"),
                "disk_used_bytes":    _val("fm_disk_used_bytes"),
            },
            "timestamp": time.time(),
        }
        self._send_json(200, summary)

    def _handle_alerts(self) -> None:
        alerts: list[dict] = []
        now = time.time()

        # Agents down for > 60 seconds
        for name, down_since in list(_agent_down_since.items()):
            secs_down = int(now - down_since)
            if secs_down > 60:
                alerts.append({
                    "level":   "warn",
                    "message": f"{name} has been down for {secs_down}s",
                    "agent":   name,
                    "seconds": secs_down,
                })

        # Tables that should have data but have 0 rows
        with _metrics_lock:
            snap = dict(_metrics)

        critical_tables = {
            "leads", "gumroad_sales", "cron_jobs", "swarm_batches",
            "api_requests", "campaigns",
        }
        for table in critical_tables:
            key = f"fm_table_rows__{table}"
            entry = snap.get(key)
            if entry is not None and entry["value"] == 0:
                alerts.append({
                    "level":   "warn",
                    "message": f"Table '{table}' has 0 rows — expected data",
                    "table":   table,
                })

        self._send_json(200, {"alerts": alerts, "count": len(alerts)})

    # ── response helper ───────────────────────────────────────────────────────

    def _send_json(self, status: int, data: Any) -> None:
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ── startup ───────────────────────────────────────────────────────────────────

def _start_background_threads() -> None:
    """Launch the three daemon polling threads."""
    threads = [
        threading.Thread(target=_poll_agent_health, name="agent-health-poller", daemon=True),
        threading.Thread(target=_poll_db_metrics,   name="db-metrics-poller",   daemon=True),
        threading.Thread(target=_poll_system,        name="system-poller",       daemon=True),
    ]
    for t in threads:
        t.start()
    log.info("Started %d background polling threads.", len(threads))


def main() -> None:
    log.info("FractalMesh Metrics Exporter starting on port %d …", PORT)
    _db_init()
    _load_custom_metrics()
    _start_background_threads()

    server = HTTPServer(("0.0.0.0", PORT), MetricsHandler)
    server.socket.setsockopt(1, 2, 1)  # SO_REUSEADDR via SOL_SOCKET / SO_REUSEADDR
    log.info("Metrics server listening on http://0.0.0.0:%d", PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Metrics exporter shutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
