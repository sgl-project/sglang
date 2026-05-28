#!/usr/bin/env python3
"""
fm_log_manager.py — Log Manager + Compression Agent (Port 7839)
FractalMesh OMEGA Titan | Log rotation, compression, file-watcher, tail, purge.
All credentials sourced from ~/.secrets/fractal.env at runtime.
Samuel James Hiotis | ABN 56 628 117 363
"""
import gzip
import json
import os
import pathlib
import shutil
import signal
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

# ── vault ─────────────────────────────────────────────────────────────────────
_vault = pathlib.Path(os.path.expanduser("~/.secrets/fractal.env"))
if _vault.exists():
    for _line in _vault.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

# ── config ────────────────────────────────────────────────────────────────────
PORT              = int(os.getenv("LOG_MANAGER_PORT", "7839"))
LOG_MAX_BYTES     = int(os.getenv("LOG_MAX_BYTES", "10485760"))        # 10 MB
COMPRESS_INTERVAL = int(os.getenv("LOG_COMPRESS_INTERVAL", "3600"))   # 1 hour
GZ_MAX_AGE        = 604800                                             # 7 days

ROOT              = pathlib.Path(os.path.expanduser("~/fmsaas"))
DB                = ROOT / "database" / "sovereign.db"
FMSAAS_LOGS       = ROOT / "logs"
PM2_LOGS          = pathlib.Path(os.path.expanduser("~/.pm2/logs"))
AGENTS_DIR        = ROOT / "agents"
FILE_STATE_PATH   = FMSAAS_LOGS / "file_state.json"

ROOT.mkdir(parents=True, exist_ok=True)
FMSAAS_LOGS.mkdir(parents=True, exist_ok=True)
(ROOT / "database").mkdir(parents=True, exist_ok=True)
PM2_LOGS.mkdir(parents=True, exist_ok=True)
AGENTS_DIR.mkdir(parents=True, exist_ok=True)

# ── database ──────────────────────────────────────────────────────────────────
def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    return conn

def _init_db():
    with _get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS log_events (
                id          INTEGER PRIMARY KEY,
                agent       TEXT,
                log_file    TEXT,
                event_type  TEXT,
                detail      TEXT,
                bytes_before INTEGER,
                bytes_after  INTEGER,
                created_at   REAL
            )
        """)
        conn.commit()

def _log_event(agent: str, log_file: str, event_type: str, detail: str,
               bytes_before: int = 0, bytes_after: int = 0):
    try:
        with _get_db() as conn:
            conn.execute(
                "INSERT INTO log_events "
                "(agent, log_file, event_type, detail, bytes_before, bytes_after, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (agent, log_file, event_type, detail, bytes_before, bytes_after, time.time())
            )
            conn.commit()
    except Exception as exc:
        print(f"[LOG-MANAGER] db error: {exc}")

# ── helpers ───────────────────────────────────────────────────────────────────
def _dir_size(path: pathlib.Path) -> int:
    """Return total bytes of all files under path (recursive)."""
    total = 0
    if not path.exists():
        return 0
    for root, _dirs, files in os.walk(str(path)):
        for fname in files:
            try:
                total += os.path.getsize(os.path.join(root, fname))
            except OSError:
                pass
    return total

def _compress_file(path: pathlib.Path):
    """
    Compress path to {path}.{unix_timestamp}.gz using gzip.
    Truncates original to 0 bytes so PM2 keeps its file handle.
    Returns (gz_path, bytes_saved).
    """
    ts = int(time.time())
    gz_path = pathlib.Path(f"{path}.{ts}.gz")
    bytes_before = path.stat().st_size

    with open(str(path), "rb") as f_in:
        with gzip.open(str(gz_path), "wb", compresslevel=6) as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Truncate original — preserve the file descriptor for PM2
    with open(str(path), "wb"):
        pass

    bytes_after = gz_path.stat().st_size
    bytes_saved = bytes_before - bytes_after
    return gz_path, bytes_saved

def _tail_file(path: pathlib.Path, n: int = 100) -> list:
    """Return last n lines from path using binary seek from end."""
    if not path.exists():
        return []
    chunk = 8192
    lines = []
    try:
        with open(str(path), "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            if size == 0:
                return []
            pos = max(0, size - chunk)
            fh.seek(pos)
            data = fh.read()
            # If file is larger than chunk, keep reading backward until we have enough lines
            while data.count(b"\n") < n + 1 and pos > 0:
                pos = max(0, pos - chunk)
                fh.seek(pos)
                data = fh.read(size - pos)
            decoded = data.decode("utf-8", errors="replace")
            all_lines = decoded.splitlines()
            lines = all_lines[-n:] if len(all_lines) >= n else all_lines
    except OSError:
        pass
    return lines

def _resolve_log_dir(dir_param: str) -> pathlib.Path:
    if dir_param == "pm2":
        return PM2_LOGS
    return FMSAAS_LOGS

def _snapshot_agents() -> dict:
    """Return dict of agent .py files → {size, mtime}."""
    snapshot = {}
    if not AGENTS_DIR.exists():
        return snapshot
    for p in AGENTS_DIR.glob("*.py"):
        try:
            st = p.stat()
            snapshot[p.name] = {"size": st.st_size, "mtime": st.st_mtime}
        except OSError:
            pass
    return snapshot

def _scan_logs() -> tuple:
    """
    Main compression cycle.
    Scans fmsaas/logs and ~/.pm2/logs for .log files exceeding LOG_MAX_BYTES.
    Also purges .gz files older than 7 days.
    Returns (compressed_count, bytes_saved).
    """
    compressed_count = 0
    total_bytes_saved = 0
    now = time.time()

    scan_dirs = []
    if FMSAAS_LOGS.exists():
        scan_dirs.append(FMSAAS_LOGS)
    if PM2_LOGS.exists():
        scan_dirs.append(PM2_LOGS)

    for log_dir in scan_dirs:
        for p in log_dir.glob("*.log"):
            try:
                size = p.stat().st_size
                if size > LOG_MAX_BYTES:
                    gz_path, saved = _compress_file(p)
                    compressed_count += 1
                    total_bytes_saved += saved
                    _log_event(
                        agent="fm_log_manager",
                        log_file=str(p),
                        event_type="compress",
                        detail=f"compressed to {gz_path.name}",
                        bytes_before=size,
                        bytes_after=gz_path.stat().st_size,
                    )
                    print(f"[LOG-MANAGER] compressed {p.name} → {gz_path.name} "
                          f"({size:,} → {gz_path.stat().st_size:,} bytes, saved {saved:,})")
            except Exception as exc:
                print(f"[LOG-MANAGER] compress error {p}: {exc}")

        # Purge stale .gz files older than 7 days
        for gz in log_dir.glob("*.gz"):
            try:
                if os.path.getmtime(str(gz)) < now - GZ_MAX_AGE:
                    sz = gz.stat().st_size
                    gz.unlink()
                    _log_event(
                        agent="fm_log_manager",
                        log_file=str(gz),
                        event_type="purge",
                        detail="deleted gz older than 7 days",
                        bytes_before=sz,
                        bytes_after=0,
                    )
                    print(f"[LOG-MANAGER] purged old gz: {gz.name}")
            except Exception as exc:
                print(f"[LOG-MANAGER] purge error {gz}: {exc}")

    return compressed_count, total_bytes_saved

# ── background threads ────────────────────────────────────────────────────────
def _compression_thread():
    """Daemon thread: runs _scan_logs() every COMPRESS_INTERVAL seconds."""
    print(f"[LOG-MANAGER] compression thread started (interval={COMPRESS_INTERVAL}s, "
          f"max_bytes={LOG_MAX_BYTES:,})")
    while True:
        try:
            count, saved = _scan_logs()
            if count:
                print(f"[LOG-MANAGER] cycle done: {count} compressed, {saved:,} bytes saved")
        except Exception as exc:
            print(f"[LOG-MANAGER] compression cycle error: {exc}")
        time.sleep(COMPRESS_INTERVAL)

def _file_watcher_thread():
    """Daemon thread: watches agents dir for file changes every 60 seconds."""
    print("[LOG-MANAGER] file-watcher thread started")
    # Load persisted snapshot if available
    previous: dict = {}
    if FILE_STATE_PATH.exists():
        try:
            previous = json.loads(FILE_STATE_PATH.read_text())
        except Exception:
            previous = {}

    while True:
        try:
            current = _snapshot_agents()

            prev_keys = set(previous.keys())
            curr_keys = set(current.keys())

            for fname in curr_keys - prev_keys:
                detail = f"added: {fname} ({current[fname]['size']} bytes)"
                _log_event("fm_log_manager", fname, "file_change", detail)
                print(f"[LOG-MANAGER] file_change added: {fname}")

            for fname in prev_keys - curr_keys:
                detail = f"removed: {fname}"
                _log_event("fm_log_manager", fname, "file_change", detail)
                print(f"[LOG-MANAGER] file_change removed: {fname}")

            for fname in prev_keys & curr_keys:
                if (current[fname]["size"] != previous[fname]["size"] or
                        current[fname]["mtime"] != previous[fname]["mtime"]):
                    detail = (f"modified: {fname} "
                              f"size {previous[fname]['size']} → {current[fname]['size']}")
                    _log_event("fm_log_manager", fname, "file_change", detail)
                    print(f"[LOG-MANAGER] file_change modified: {fname}")

            previous = current
            # Persist snapshot
            FILE_STATE_PATH.write_text(json.dumps(current, indent=2))
        except Exception as exc:
            print(f"[LOG-MANAGER] file-watcher error: {exc}")
        time.sleep(60)

# ── HTTP handler ──────────────────────────────────────────────────────────────
class LogManagerHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # silence default access log
        pass

    def _send_json(self, data, status: int = 200):
        body = json.dumps(data, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        path = parsed.path.rstrip("/")

        if path == "/health":
            self._send_json({
                "status": "ok",
                "service": "fm-log-manager",
                "port": PORT,
                "compression": "active",
            })

        elif path == "/logs/list":
            dir_param = qs.get("dir", ["fmsaas"])[0]
            ext_param = qs.get("ext", ["log"])[0]
            log_dir = _resolve_log_dir(dir_param)
            result = []
            if log_dir.exists():
                pattern = f"*.{ext_param}"
                for p in sorted(log_dir.glob(pattern)):
                    try:
                        st = p.stat()
                        result.append({
                            "name": p.name,
                            "size_bytes": st.st_size,
                            "size_mb": round(st.st_size / 1048576, 3),
                            "mtime": st.st_mtime,
                            "compressed": p.suffix == ".gz",
                        })
                    except OSError:
                        pass
            self._send_json(result)

        elif path == "/logs/tail":
            fname = qs.get("file", [""])[0]
            lines_n = int(qs.get("lines", ["100"])[0])
            dir_param = qs.get("dir", ["fmsaas"])[0]
            log_dir = _resolve_log_dir(dir_param)
            if not fname:
                self._send_json({"error": "file param required"}, 400)
                return
            target = log_dir / pathlib.Path(fname).name
            if not target.exists():
                self._send_json({"error": "file not found"}, 404)
                return
            lines = _tail_file(target, lines_n)
            self._send_json({
                "file": str(target),
                "lines": lines,
                "total_lines": len(lines),
            })

        elif path == "/events":
            try:
                with _get_db() as conn:
                    rows = conn.execute(
                        "SELECT * FROM log_events ORDER BY created_at DESC LIMIT 100"
                    ).fetchall()
                    data = [dict(r) for r in rows]
            except Exception as exc:
                data = {"error": str(exc)}
            self._send_json(data)

        elif path == "/stats":
            fmsaas_size = _dir_size(FMSAAS_LOGS)
            pm2_size = _dir_size(PM2_LOGS)
            log_count = len(list(FMSAAS_LOGS.glob("*.log"))) + len(list(PM2_LOGS.glob("*.log")))
            gz_count = len(list(FMSAAS_LOGS.glob("*.gz"))) + len(list(PM2_LOGS.glob("*.gz")))

            largest_name = ""
            largest_size = 0
            for scan_dir in [FMSAAS_LOGS, PM2_LOGS]:
                if not scan_dir.exists():
                    continue
                for p in scan_dir.iterdir():
                    try:
                        sz = p.stat().st_size
                        if sz > largest_size:
                            largest_size = sz
                            largest_name = p.name
                    except OSError:
                        pass

            self._send_json({
                "fmsaas_logs_bytes": fmsaas_size,
                "fmsaas_logs_mb": round(fmsaas_size / 1048576, 3),
                "pm2_logs_bytes": pm2_size,
                "pm2_logs_mb": round(pm2_size / 1048576, 3),
                "log_file_count": log_count,
                "gz_file_count": gz_count,
                "largest_file": largest_name,
                "largest_file_bytes": largest_size,
            })

        elif path == "/files/snapshot":
            if FILE_STATE_PATH.exists():
                try:
                    data = json.loads(FILE_STATE_PATH.read_text())
                except Exception as exc:
                    data = {"error": str(exc)}
            else:
                data = {}
            self._send_json(data)

        elif path == "/files/changes":
            try:
                with _get_db() as conn:
                    rows = conn.execute(
                        "SELECT * FROM log_events WHERE event_type='file_change' "
                        "ORDER BY created_at DESC LIMIT 100"
                    ).fetchall()
                    data = [dict(r) for r in rows]
            except Exception as exc:
                data = {"error": str(exc)}
            self._send_json(data)

        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/logs/compress_now":
            body = self._read_body()
            fname = body.get("file", "")
            dir_param = body.get("dir", "fmsaas")
            log_dir = _resolve_log_dir(dir_param)

            if fname:
                # Compress single named file
                target = log_dir / pathlib.Path(fname).name
                if not target.exists():
                    self._send_json({"error": "file not found"}, 404)
                    return
                try:
                    size_before = target.stat().st_size
                    gz_path, saved = _compress_file(target)
                    _log_event(
                        agent="fm_log_manager",
                        log_file=str(target),
                        event_type="compress",
                        detail=f"on-demand compress → {gz_path.name}",
                        bytes_before=size_before,
                        bytes_after=gz_path.stat().st_size,
                    )
                    self._send_json({"compressed": 1, "bytes_saved": saved})
                except Exception as exc:
                    self._send_json({"error": str(exc)}, 500)
            else:
                # Full compression cycle
                try:
                    count, saved = _scan_logs()
                    self._send_json({"compressed": count, "bytes_saved": saved})
                except Exception as exc:
                    self._send_json({"error": str(exc)}, 500)

        elif path == "/logs/purge":
            body = self._read_body()
            older_than_days = int(body.get("older_than_days", 7))
            dir_param = body.get("dir", "fmsaas")
            ext = body.get("ext", "gz").lstrip(".")
            log_dir = _resolve_log_dir(dir_param)
            cutoff = time.time() - (older_than_days * 86400)

            deleted = 0
            bytes_freed = 0
            if log_dir.exists():
                for p in log_dir.glob(f"*.{ext}"):
                    try:
                        if os.path.getmtime(str(p)) < cutoff:
                            sz = p.stat().st_size
                            p.unlink()
                            deleted += 1
                            bytes_freed += sz
                            _log_event(
                                agent="fm_log_manager",
                                log_file=str(p),
                                event_type="purge",
                                detail=f"purged (older than {older_than_days}d)",
                                bytes_before=sz,
                                bytes_after=0,
                            )
                    except Exception as exc:
                        print(f"[LOG-MANAGER] purge error {p}: {exc}")

            self._send_json({"deleted": deleted, "bytes_freed": bytes_freed})

        else:
            self._send_json({"error": "not found"}, 404)

# ── startup ───────────────────────────────────────────────────────────────────
def _start_background_threads():
    ct = threading.Thread(target=_compression_thread, daemon=True, name="log-compressor")
    ct.start()
    wt = threading.Thread(target=_file_watcher_thread, daemon=True, name="file-watcher")
    wt.start()

def main():
    _init_db()
    _start_background_threads()

    server = HTTPServer(("0.0.0.0", PORT), LogManagerHandler)

    def _shutdown(signum, frame):
        print("\n[LOG-MANAGER] shutting down…")
        threading.Thread(target=server.shutdown).start()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    print(f"[LOG-MANAGER] started on port {PORT} | "
          f"max_bytes={LOG_MAX_BYTES:,} | interval={COMPRESS_INTERVAL}s")
    server.serve_forever()

if __name__ == "__main__":
    main()
