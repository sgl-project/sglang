"""
FractalMesh Security Monitor v1.0.0
Real-time log scanning, system health monitoring, credential leak detection,
and port health checks for the FractalMesh OMEGA Titan platform.
Samuel James Hiotis | ABN 56 628 117 363 | Sole Trader
"""
import hmac
import json
import os
import pathlib
import re
import signal
import socket
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

# ---------------------------------------------------------------------------
# Configuration from environment (loaded from ~/.secrets/fractal.env by PM2)
# ---------------------------------------------------------------------------
HOME     = os.path.expanduser("~")
ROOT     = os.getenv("FRACTALMESH_HOME", os.path.join(HOME, "fmsaas"))
DB       = os.path.join(ROOT, "database", "sovereign.db")

PORT          = int(os.getenv("SECURITY_MONITOR_PORT", "7840"))
SCAN_INTERVAL = int(os.getenv("SECURITY_SCAN_INTERVAL", "120"))
ADMIN_SECRET  = os.getenv("ADMIN_SECRET", "")

# ---------------------------------------------------------------------------
# Alert patterns seeded as security_rules at startup
# ---------------------------------------------------------------------------
DEFAULT_RULES = [
    ("auth_failure",     r"authentication failure|invalid password|login failed",              "error"),
    ("unauthorized",     r"\b(401|403)\b.*unauthorized|unauthorized.*\b(401|403)\b",           "warn"),
    ("rate_limit",       r"\b429\b|rate.?limit|too many requests",                             "warn"),
    ("server_error",     r"\b500\b|internal server error|traceback.*error",                    "error"),
    ("oom_risk",         r"out of memory|cannot allocate|killed.*oom|signal 9",                "critical"),
    ("credential_leak",  r"sk-ant-api|sk_live_|sk-or-v1|ghp_|PRIVATE_KEY|password=",         "critical"),
    ("import_error",     r"ModuleNotFoundError|ImportError|No module named",                   "warn"),
    ("connection_error", r"ConnectionRefusedError|connection refused|ECONNREFUSED",            "warn"),
    ("disk_space",       r"no space left|disk full|ENOSPC",                                    "error"),
    ("port_conflict",    r"address already in use|EADDRINUSE|OSError.*98",                     "error"),
]

# ---------------------------------------------------------------------------
# FractalMesh port registry (7785-7840)
# ---------------------------------------------------------------------------
PORT_NAMES = {
    7785: "mcp_router",
    7786: "pulse_bus",
    7787: "enterprise_bus",
    7788: "admin_api",
    7789: "admin_dashboard",
    7790: "metrics",
    7791: "analytics",
    7792: "aiaas",
    7793: "data_api",
    7794: "rag_pipeline",
    7795: "scraper_v2",
    7796: "content_engine",
    7797: "drip_agent",
    7798: "leadgen",
    7799: "email_listener",
    7800: "sendgrid",
    7801: "stripe_gateway",
    7802: "stripe_mon",
    7803: "paypal",
    7804: "coinbase",
    7805: "kucoin",
    7806: "pionex",
    7807: "coingecko",
    7808: "moralis",
    7809: "opensea",
    7810: "nft_engine",
    7811: "smart_contracts",
    7812: "ipfs",
    7813: "xyo",
    7814: "firebase",
    7815: "supabase",
    7816: "notion",
    7817: "github_ops",
    7818: "gitops_runner",
    7819: "docker",
    7820: "coolify",
    7821: "tunnel",
    7822: "geo_validator",
    7823: "geosignal",
    7824: "wigle_oracle",
    7825: "osint_spider",
    7826: "osintaas",
    7827: "dork_engine",
    7828: "dorking",
    7829: "deep_scan",
    7830: "bounty",
    7831: "bugcrowd",
    7832: "ais_monitor",
    7833: "sentinel_ingest",
    7834: "oversight",
    7835: "healer",
    7836: "samsung_warden",
    7837: "device_bridge",
    7838: "sovereign_ops",
    7839: "salvage_crew",
    7840: "security_monitor",
}

_running = True
_port_state: dict[int, bool] = {}
_scan_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB, timeout=15, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def _db_init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = _db_connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS security_events (
            id            INTEGER PRIMARY KEY,
            level         TEXT,
            source        TEXT,
            pattern       TEXT,
            file          TEXT,
            detail        TEXT,
            acknowledged  INTEGER DEFAULT 0,
            created_at    REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS security_rules (
            id          INTEGER PRIMARY KEY,
            name        TEXT UNIQUE,
            pattern     TEXT,
            level       TEXT DEFAULT 'warn',
            enabled     INTEGER DEFAULT 1,
            match_count INTEGER DEFAULT 0,
            created_at  REAL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_events_level ON security_events(level)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_events_ack   ON security_events(acknowledged)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_events_src   ON security_events(source, pattern, created_at)")
    conn.commit()
    now = time.time()
    for name, pattern, level in DEFAULT_RULES:
        conn.execute(
            "INSERT OR IGNORE INTO security_rules (name, pattern, level, created_at) VALUES (?,?,?,?)",
            (name, pattern, level, now)
        )
    conn.commit()
    conn.close()


def _dedup_check(conn: sqlite3.Connection, source: str, pattern: str, window: int = 60) -> bool:
    """Return True if a duplicate event for (source, pattern) exists within the window."""
    cutoff = time.time() - window
    row = conn.execute(
        "SELECT 1 FROM security_events WHERE source=? AND pattern=? AND created_at>? LIMIT 1",
        (source, pattern, cutoff)
    ).fetchone()
    return row is not None


def _log_event(level: str, source: str, pattern: str, detail: str, file: str = "") -> int:
    """Insert a security event; returns new row id or -1 on dedup/error."""
    try:
        conn = _db_connect()
        if _dedup_check(conn, source, pattern):
            conn.close()
            return -1
        cur = conn.execute(
            "INSERT INTO security_events (level, source, pattern, file, detail, created_at) VALUES (?,?,?,?,?,?)",
            (level, source, pattern, file, detail, time.time())
        )
        conn.execute(
            "UPDATE security_rules SET match_count = match_count + 1 WHERE name=?",
            (pattern,)
        )
        conn.commit()
        row_id = cur.lastrowid
        conn.close()
        return row_id
    except Exception:
        return -1


# ---------------------------------------------------------------------------
# System metric helpers
# ---------------------------------------------------------------------------

def _read_proc_meminfo() -> dict:
    """Parse /proc/meminfo into a dict of key -> kB values."""
    result = {}
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    try:
                        result[key] = int(parts[1])
                    except ValueError:
                        result[key] = 0
    except Exception:
        pass
    return result


def _get_loadavg() -> tuple:
    """Return (load1, load5, load15) from /proc/loadavg."""
    try:
        with open("/proc/loadavg", "r") as f:
            parts = f.read().split()
        return float(parts[0]), float(parts[1]), float(parts[2])
    except Exception:
        return 0.0, 0.0, 0.0


def _check_port(port: int, timeout: float = 0.5) -> tuple:
    """Return (is_up: bool, response_ms: float)."""
    t0 = time.monotonic()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        result = s.connect_ex(("127.0.0.1", port))
        s.close()
        elapsed = (time.monotonic() - t0) * 1000
        return result == 0, round(elapsed, 2)
    except Exception:
        elapsed = (time.monotonic() - t0) * 1000
        return False, round(elapsed, 2)


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

def _check_auth(handler: "SecurityHandler") -> bool:
    """Validate X-Admin-Secret header against ADMIN_SECRET."""
    if not ADMIN_SECRET:
        return True
    provided = handler.headers.get("X-Admin-Secret", "")
    return hmac.compare_digest(provided, ADMIN_SECRET)


# ---------------------------------------------------------------------------
# Background scan
# ---------------------------------------------------------------------------

def _scan_logs(rules: list) -> int:
    """Scan PM2 error logs and fmsaas logs; return count of new events created."""
    created = 0
    log_dirs = [
        (pathlib.Path(HOME) / ".pm2" / "logs", "*-error-*.log"),
        (pathlib.Path(ROOT) / "logs",           "*.log"),
    ]
    for log_dir, glob_pat in log_dirs:
        if not log_dir.exists():
            continue
        for log_file in log_dir.glob(glob_pat):
            try:
                size = log_file.stat().st_size
                offset = max(0, size - 8192)
                with open(log_file, "rb") as f:
                    f.seek(offset)
                    content = f.read().decode("utf-8", errors="replace")
            except Exception:
                continue

            source = log_file.stem
            for name, pattern, level, enabled in rules:
                if not enabled:
                    continue
                try:
                    if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                        # Find matching lines for detail
                        matches = []
                        for line in content.splitlines():
                            if re.search(pattern, line, re.IGNORECASE):
                                matches.append(line.strip()[:200])
                                if len(matches) >= 3:
                                    break
                        detail = " | ".join(matches) if matches else "(pattern matched)"
                        row_id = _log_event(level, source, name, detail, str(log_file))
                        if row_id > 0:
                            created += 1
                except Exception:
                    continue
    return created


def _scan_system_health() -> int:
    """Check CPU, memory, disk via /proc and os.statvfs; return events created."""
    created = 0

    # CPU load
    load1, load5, load15 = _get_loadavg()
    if load1 > 4.0:
        level = "critical" if load1 > 8.0 else "warn"
        detail = f"1-min load average: {load1:.2f} (threshold 4.0)"
        rid = _log_event(level, "system:cpu", "high_load", detail)
        if rid > 0:
            created += 1

    # Memory
    mem = _read_proc_meminfo()
    mem_avail_kb = mem.get("MemAvailable", mem.get("MemFree", 0))
    mem_avail_mb = mem_avail_kb / 1024
    if mem_avail_mb < 100:
        detail = f"MemAvailable: {mem_avail_mb:.1f} MB (threshold 100 MB)"
        rid = _log_event("critical", "system:memory", "oom_risk", detail)
        if rid > 0:
            created += 1

    # Disk
    try:
        sv = os.statvfs(HOME)
        total_blocks = sv.f_blocks
        free_blocks  = sv.f_bfree
        if total_blocks > 0:
            used_ratio = (total_blocks - free_blocks) / total_blocks
            if used_ratio > 0.95:
                used_pct = used_ratio * 100
                detail = f"Disk used: {used_pct:.1f}% (threshold 95%)"
                rid = _log_event("error", "system:disk", "disk_space", detail)
                if rid > 0:
                    created += 1
    except Exception:
        pass

    return created


def _scan_port_health() -> int:
    """Check all FractalMesh ports; log if a previously-up port goes down."""
    global _port_state
    created = 0
    for port in range(7785, 7841):
        is_up, _ = _check_port(port)
        was_up = _port_state.get(port)
        if was_up is True and not is_up:
            name = PORT_NAMES.get(port, f"port_{port}")
            detail = f"Port {port} ({name}) was UP but is now DOWN"
            rid = _log_event("error", f"port:{port}", "port_down", detail)
            if rid > 0:
                created += 1
        _port_state[port] = is_up
    return created


def _scan_credentials() -> int:
    """Walk ~/sglang/fractalmesh/ for .py files; flag credential leaks."""
    created = 0
    repo_root = pathlib.Path(HOME) / "sglang" / "fractalmesh"
    if not repo_root.exists():
        return 0

    # Get the credential_leak pattern from DEFAULT_RULES
    cred_pattern = None
    for name, pattern, level in DEFAULT_RULES:
        if name == "credential_leak":
            cred_pattern = pattern
            break
    if not cred_pattern:
        return 0

    regex = re.compile(cred_pattern, re.IGNORECASE)
    for py_file in repo_root.rglob("*.py"):
        try:
            content = py_file.read_text(errors="replace")
        except Exception:
            continue
        matches = []
        for lineno, line in enumerate(content.splitlines(), 1):
            if regex.search(line):
                # Redact the matched value for safety
                matches.append(f"line {lineno}: {line.strip()[:120]}")
                if len(matches) >= 3:
                    break
        if matches:
            detail = " | ".join(matches)
            rid = _log_event("critical", "repo:scan", "credential_leak", detail, str(py_file))
            if rid > 0:
                created += 1
    return created


def _full_scan() -> tuple:
    """Run a complete scan cycle; return (events_created, system_ok)."""
    with _scan_lock:
        events_created = 0
        system_ok = True

        # Load enabled rules once
        try:
            conn = _db_connect()
            rows = conn.execute(
                "SELECT name, pattern, level, enabled FROM security_rules WHERE enabled=1"
            ).fetchall()
            conn.close()
            rules = [(r["name"], r["pattern"], r["level"], r["enabled"]) for r in rows]
        except Exception:
            rules = [(name, pat, level, 1) for name, pat, level in DEFAULT_RULES]

        events_created += _scan_logs(rules)
        events_created += _scan_system_health()
        events_created += _scan_port_health()
        events_created += _scan_credentials()

        # Determine system_ok by checking for unacknowledged critical events in last scan
        try:
            conn = _db_connect()
            cutoff = time.time() - SCAN_INTERVAL * 2
            row = conn.execute(
                "SELECT 1 FROM security_events WHERE level='critical' AND acknowledged=0 AND created_at>? LIMIT 1",
                (cutoff,)
            ).fetchone()
            conn.close()
            if row:
                system_ok = False
        except Exception:
            pass

        return events_created, system_ok


def _scan_thread():
    """Background daemon thread that runs full scan every SCAN_INTERVAL seconds."""
    global _running
    # Initial port state population (don't log on first scan)
    for port in range(7785, 7841):
        is_up, _ = _check_port(port)
        _port_state[port] = is_up

    while _running:
        try:
            _full_scan()
        except Exception:
            pass
        for _ in range(SCAN_INTERVAL * 10):
            if not _running:
                break
            time.sleep(0.1)


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class SecurityHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the Security Monitor API."""

    def log_message(self, format, *args):
        pass  # Suppress default access log spam

    def _send_json(self, data, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, message: str):
        self._send_json({"error": message}, status)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        query = self.path[len(path):]
        params = self._parse_query(query)

        if path == "/health":
            self._handle_health()
        elif path == "/events":
            self._handle_events(params)
        elif path == "/events/summary":
            self._handle_events_summary()
        elif path == "/rules":
            self._handle_rules_list()
        elif path == "/system":
            self._handle_system()
        elif path == "/ports":
            self._handle_ports()
        elif path == "/analytics":
            self._handle_analytics()
        else:
            self._send_error(404, "not found")

    def do_POST(self):
        path = self.path.rstrip("/")

        # POST /events/{id}/acknowledge
        m = re.match(r"^/events/(\d+)/acknowledge$", path)
        if m:
            self._handle_ack_event(int(m.group(1)))
            return

        if path == "/events/acknowledge_all":
            self._handle_ack_all()
        elif path == "/rules/create":
            self._handle_rule_create()
        elif path == "/scan/now":
            self._handle_scan_now()
        else:
            self._send_error(404, "not found")

    def do_PUT(self):
        path = self.path.rstrip("/")
        m = re.match(r"^/rules/(\d+)$", path)
        if m:
            self._handle_rule_update(int(m.group(1)))
        else:
            self._send_error(404, "not found")

    # ------------------------------------------------------------------
    # Endpoint implementations
    # ------------------------------------------------------------------

    def _handle_health(self):
        self._send_json({
            "status":  "ok",
            "service": "fm-security-monitor",
            "port":    PORT
        })

    def _parse_query(self, qs: str) -> dict:
        params = {}
        if not qs or qs == "?":
            return params
        qs = qs.lstrip("?")
        for part in qs.split("&"):
            if "=" in part:
                k, v = part.split("=", 1)
                params[k] = v
            elif part:
                params[part] = True
        return params

    def _handle_events(self, params: dict):
        level        = params.get("level")
        acknowledged = params.get("acknowledged")
        limit        = int(params.get("limit", 50))

        where_clauses = []
        args = []
        if level:
            where_clauses.append("level=?")
            args.append(level)
        if acknowledged is not None:
            ack_val = 0 if str(acknowledged).lower() in ("false", "0") else 1
            where_clauses.append("acknowledged=?")
            args.append(ack_val)

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        args.append(limit)

        try:
            conn = _db_connect()
            rows = conn.execute(
                f"SELECT id,level,source,pattern,file,detail,acknowledged,created_at "
                f"FROM security_events {where_sql} ORDER BY created_at DESC LIMIT ?",
                args
            ).fetchall()
            conn.close()
            self._send_json([dict(r) for r in rows])
        except Exception as e:
            self._send_error(500, str(e))

    def _handle_events_summary(self):
        try:
            conn = _db_connect()
            counts = {"critical": 0, "error": 0, "warn": 0, "info": 0}
            for row in conn.execute(
                "SELECT level, COUNT(*) as c FROM security_events GROUP BY level"
            ).fetchall():
                lvl = row["level"]
                if lvl in counts:
                    counts[lvl] = row["c"]

            unack_row = conn.execute(
                "SELECT COUNT(*) FROM security_events WHERE acknowledged=0"
            ).fetchone()
            unacknowledged = unack_row[0] if unack_row else 0

            latest_row = conn.execute(
                "SELECT MAX(created_at) FROM security_events"
            ).fetchone()
            latest = latest_row[0] if latest_row and latest_row[0] else None
            conn.close()

            self._send_json({
                "critical":      counts["critical"],
                "error":         counts["error"],
                "warn":          counts["warn"],
                "info":          counts["info"],
                "unacknowledged": unacknowledged,
                "latest":        latest
            })
        except Exception as e:
            self._send_error(500, str(e))

    def _handle_ack_event(self, event_id: int):
        if not _check_auth(self):
            self._send_error(401, "unauthorized")
            return
        try:
            conn = _db_connect()
            conn.execute(
                "UPDATE security_events SET acknowledged=1 WHERE id=?", (event_id,)
            )
            conn.commit()
            conn.close()
            self._send_json({"acknowledged": True})
        except Exception as e:
            self._send_error(500, str(e))

    def _handle_ack_all(self):
        if not _check_auth(self):
            self._send_error(401, "unauthorized")
            return
        body = self._read_body()
        level = body.get("level")
        try:
            conn = _db_connect()
            if level:
                cur = conn.execute(
                    "UPDATE security_events SET acknowledged=1 WHERE acknowledged=0 AND level=?",
                    (level,)
                )
            else:
                cur = conn.execute(
                    "UPDATE security_events SET acknowledged=1 WHERE acknowledged=0"
                )
            count = cur.rowcount
            conn.commit()
            conn.close()
            self._send_json({"count": count})
        except Exception as e:
            self._send_error(500, str(e))

    def _handle_rules_list(self):
        try:
            conn = _db_connect()
            rows = conn.execute(
                "SELECT id,name,pattern,level,enabled,match_count,created_at FROM security_rules ORDER BY id"
            ).fetchall()
            conn.close()
            self._send_json([dict(r) for r in rows])
        except Exception as e:
            self._send_error(500, str(e))

    def _handle_rule_create(self):
        if not _check_auth(self):
            self._send_error(401, "unauthorized")
            return
        body = self._read_body()
        name    = body.get("name", "").strip()
        pattern = body.get("pattern", "").strip()
        level   = body.get("level", "warn").strip()
        if not name or not pattern:
            self._send_error(400, "name and pattern are required")
            return
        if level not in ("critical", "error", "warn", "info"):
            self._send_error(400, "level must be critical|error|warn|info")
            return
        try:
            re.compile(pattern)
        except re.error as e:
            self._send_error(400, f"invalid regex: {e}")
            return
        try:
            conn = _db_connect()
            cur = conn.execute(
                "INSERT INTO security_rules (name, pattern, level, created_at) VALUES (?,?,?,?)",
                (name, pattern, level, time.time())
            )
            rule_id = cur.lastrowid
            conn.commit()
            conn.close()
            self._send_json({"id": rule_id, "name": name, "pattern": pattern, "level": level})
        except sqlite3.IntegrityError:
            self._send_error(409, "rule name already exists")
        except Exception as e:
            self._send_error(500, str(e))

    def _handle_rule_update(self, rule_id: int):
        if not _check_auth(self):
            self._send_error(401, "unauthorized")
            return
        body = self._read_body()
        updates = []
        args = []
        if "enabled" in body:
            updates.append("enabled=?")
            args.append(1 if body["enabled"] else 0)
        if "pattern" in body:
            try:
                re.compile(body["pattern"])
            except re.error as e:
                self._send_error(400, f"invalid regex: {e}")
                return
            updates.append("pattern=?")
            args.append(body["pattern"])
        if "level" in body:
            if body["level"] not in ("critical", "error", "warn", "info"):
                self._send_error(400, "level must be critical|error|warn|info")
                return
            updates.append("level=?")
            args.append(body["level"])
        if not updates:
            self._send_error(400, "no updatable fields provided")
            return
        args.append(rule_id)
        try:
            conn = _db_connect()
            conn.execute(f"UPDATE security_rules SET {', '.join(updates)} WHERE id=?", args)
            conn.commit()
            row = conn.execute(
                "SELECT id,name,pattern,level,enabled,match_count,created_at FROM security_rules WHERE id=?",
                (rule_id,)
            ).fetchone()
            conn.close()
            if row:
                self._send_json(dict(row))
            else:
                self._send_error(404, "rule not found")
        except Exception as e:
            self._send_error(500, str(e))

    def _handle_system(self):
        load1, load5, load15 = _get_loadavg()
        mem = _read_proc_meminfo()
        mem_total_kb = mem.get("MemTotal", 0)
        mem_avail_kb = mem.get("MemAvailable", mem.get("MemFree", 0))
        mem_used_kb  = mem_total_kb - mem_avail_kb
        mem_used_pct = round(mem_used_kb / mem_total_kb * 100, 1) if mem_total_kb > 0 else 0

        disk_info = {}
        try:
            sv = os.statvfs(HOME)
            total_bytes = sv.f_blocks * sv.f_frsize
            free_bytes  = sv.f_bfree  * sv.f_frsize
            used_bytes  = total_bytes - free_bytes
            disk_info = {
                "total_gb":  round(total_bytes / 1073741824, 2),
                "free_gb":   round(free_bytes  / 1073741824, 2),
                "used_pct":  round(used_bytes / total_bytes * 100, 1) if total_bytes > 0 else 0
            }
        except Exception:
            disk_info = {"total_gb": 0, "free_gb": 0, "used_pct": 0}

        ports_status = []
        for port in range(7785, 7841):
            is_up, ms = _check_port(port)
            ports_status.append({
                "port":        port,
                "name":        PORT_NAMES.get(port, f"port_{port}"),
                "status":      "up" if is_up else "down",
                "response_ms": ms
            })

        self._send_json({
            "cpu": {
                "load1":  load1,
                "load5":  load5,
                "load15": load15
            },
            "memory": {
                "total_mb":     round(mem_total_kb / 1024, 1),
                "available_mb": round(mem_avail_kb / 1024, 1),
                "used_pct":     mem_used_pct
            },
            "disk": disk_info,
            "ports": ports_status
        })

    def _handle_ports(self):
        results = []
        for port in range(7785, 7841):
            is_up, ms = _check_port(port)
            results.append({
                "port":        port,
                "name":        PORT_NAMES.get(port, f"port_{port}"),
                "status":      "up" if is_up else "down",
                "response_ms": ms
            })
        self._send_json(results)

    def _handle_scan_now(self):
        if not _check_auth(self):
            self._send_error(401, "unauthorized")
            return
        try:
            events_created, system_ok = _full_scan()
            self._send_json({"events_created": events_created, "system_ok": system_ok})
        except Exception as e:
            self._send_error(500, str(e))

    def _handle_analytics(self):
        cutoff = time.time() - 86400  # last 24h
        try:
            conn = _db_connect()

            # Events last 24h by level
            by_level = {"critical": 0, "error": 0, "warn": 0, "info": 0}
            for row in conn.execute(
                "SELECT level, COUNT(*) as c FROM security_events WHERE created_at>? GROUP BY level",
                (cutoff,)
            ).fetchall():
                lvl = row["level"]
                if lvl in by_level:
                    by_level[lvl] = row["c"]

            # Top 5 noisy sources
            top_sources = []
            for row in conn.execute(
                "SELECT source, COUNT(*) as c FROM security_events WHERE created_at>? "
                "GROUP BY source ORDER BY c DESC LIMIT 5",
                (cutoff,)
            ).fetchall():
                top_sources.append({"source": row["source"], "count": row["c"]})

            # Most frequent patterns
            top_patterns = []
            for row in conn.execute(
                "SELECT pattern, COUNT(*) as c FROM security_events WHERE created_at>? "
                "GROUP BY pattern ORDER BY c DESC LIMIT 10",
                (cutoff,)
            ).fetchall():
                top_patterns.append({"pattern": row["pattern"], "count": row["c"]})

            conn.close()
            self._send_json({
                "window_hours":  24,
                "by_level":      by_level,
                "top_sources":   top_sources,
                "top_patterns":  top_patterns
            })
        except Exception as e:
            self._send_error(500, str(e))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def _handle_signal(signum, frame):
    global _running
    _running = False


def main():
    global _running
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT,  _handle_signal)

    _db_init()

    t = threading.Thread(target=_scan_thread, daemon=True, name="security-scan")
    t.start()

    server = HTTPServer(("0.0.0.0", PORT), SecurityHandler)
    server.timeout = 1.0

    print(f"[fm-security-monitor] Listening on port {PORT}, scan interval {SCAN_INTERVAL}s", flush=True)

    while _running:
        try:
            server.handle_request()
        except Exception:
            pass

    server.server_close()
    print("[fm-security-monitor] Shutdown complete", flush=True)


if __name__ == "__main__":
    main()
