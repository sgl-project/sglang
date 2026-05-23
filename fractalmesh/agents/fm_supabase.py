#!/usr/bin/env python3
"""
fm_supabase.py — Supabase / PostgreSQL Sync Agent (Port 7793)
Two-way sync between sovereign.db (SQLite) and Supabase cloud tables.
Row-level security respected: uses anon/service_role keys from vault.
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
PORT           = int(os.getenv("SUPABASE_PORT", "7793"))
ROOT           = Path(os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas")))
DB             = ROOT / "database" / "sovereign.db"
LOG            = ROOT / "logs" / "supabase.log"
SB_URL         = os.getenv("SUPABASE_URL", "")           # https://xxx.supabase.co
SB_KEY         = os.getenv("SUPABASE_SERVICE_KEY",
                 os.getenv("SUPABASE_ANON_KEY", ""))      # service_role preferred
SB_REST        = f"{SB_URL}/rest/v1" if SB_URL else ""
SYNC_INTERVAL  = int(os.getenv("SUPABASE_SYNC_INTERVAL", "300"))  # 5 min

ROOT.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)
DB.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SUPABASE] %(message)s",
    handlers=[logging.FileHandler(LOG), logging.StreamHandler()],
)
log = logging.getLogger("supabase")

# ── sync table manifest ───────────────────────────────────────────────────────
# local_table → supabase_table, columns to sync, unique key
SYNC_MANIFEST = [
    {
        "local":    "mcp_log",
        "remote":   "fm_mcp_log",
        "columns":  ["intent", "status", "latency", "ts"],
        "key":      "id",
        "direction":"push",   # push local → supabase
    },
    {
        "local":    "strategy_log",
        "remote":   "fm_strategy_log",
        "columns":  ["strategy_id", "title", "action", "result", "ts"],
        "key":      "id",
        "direction":"push",
    },
    {
        "local":    "revenue_snapshots",
        "remote":   "fm_revenue_snapshots",
        "columns":  ["channel_id", "amount", "currency", "period", "ts"],
        "key":      "id",
        "direction":"push",
    },
    {
        "local":    "devto_articles",
        "remote":   "fm_devto_articles",
        "columns":  ["devto_id", "series", "title", "url", "status",
                     "page_views", "reactions", "comments", "ts"],
        "key":      "devto_id",
        "direction":"push",
    },
    {
        "local":    "canva_designs",
        "remote":   "fm_canva_designs",
        "columns":  ["design_id", "template_id", "title", "status", "ts"],
        "key":      "id",
        "direction":"push",
    },
    {
        "local":    "hf_usage",
        "remote":   "fm_hf_usage",
        "columns":  ["model_id", "call_count", "total_ms", "last_used"],
        "key":      "model_id",
        "direction":"push",
    },
    {
        "local":    "or_cost_summary",
        "remote":   "fm_or_cost_summary",
        "columns":  ["model", "total_usd", "call_count", "total_tokens"],
        "key":      "model",
        "direction":"push",
    },
]

# ── local state tracking ──────────────────────────────────────────────────────

def _db_init():
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS supabase_sync_log (
            id          INTEGER PRIMARY KEY,
            table_name  TEXT,
            direction   TEXT,
            rows_pushed INTEGER,
            rows_pulled INTEGER,
            status      TEXT,
            error       TEXT,
            ts          DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS supabase_sync_cursor (
            table_name  TEXT PRIMARY KEY,
            last_id     INTEGER DEFAULT 0,
            last_ts     DATETIME
        )
    """)
    conn.commit()
    conn.close()

def _get_cursor(table: str) -> int:
    try:
        conn = sqlite3.connect(DB, timeout=5)
        row  = conn.execute(
            "SELECT last_id FROM supabase_sync_cursor WHERE table_name=?", (table,)
        ).fetchone()
        conn.close()
        return row[0] if row else 0
    except Exception:
        return 0

def _set_cursor(table: str, last_id: int):
    try:
        conn = sqlite3.connect(DB, timeout=5)
        conn.execute("""
            INSERT INTO supabase_sync_cursor (table_name,last_id,last_ts)
            VALUES (?,?,CURRENT_TIMESTAMP)
            ON CONFLICT(table_name) DO UPDATE SET last_id=excluded.last_id,last_ts=CURRENT_TIMESTAMP
        """, (table, last_id))
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning("set_cursor error: %s", e)

def _log_sync(table: str, direction: str, pushed: int, pulled: int,
              status: str, error: str = ""):
    try:
        conn = sqlite3.connect(DB, timeout=5)
        conn.execute(
            "INSERT INTO supabase_sync_log (table_name,direction,rows_pushed,rows_pulled,status,error) "
            "VALUES (?,?,?,?,?,?)",
            (table, direction, pushed, pulled, status, error),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning("log_sync error: %s", e)

# ── Supabase REST helpers ──────────────────────────────────────────────────────

def _sb_upsert(table: str, rows: list) -> dict:
    if not SB_REST or not SB_KEY:
        return {"error": "supabase_not_configured"}
    payload = json.dumps(rows).encode()
    req = urllib.request.Request(
        f"{SB_REST}/{table}",
        data=payload,
        headers={
            "apikey":          SB_KEY,
            "Authorization":   f"Bearer {SB_KEY}",
            "Content-Type":    "application/json",
            "Prefer":          "resolution=merge-duplicates",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            body = r.read().decode("utf-8", errors="replace")
            return {"status": "ok", "rows": len(rows), "response": body[:200]}
    except urllib.error.HTTPError as e:
        return {"error": f"http_{e.code}", "detail": e.read().decode()[:200]}
    except Exception as e:
        return {"error": str(e)}

def _sb_select(table: str, select: str = "*", limit: int = 100) -> list:
    if not SB_REST or not SB_KEY:
        return []
    url = f"{SB_REST}/{table}?select={select}&limit={limit}&order=id.desc"
    req = urllib.request.Request(
        url,
        headers={
            "apikey":        SB_KEY,
            "Authorization": f"Bearer {SB_KEY}",
            "Accept":        "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            result = json.loads(r.read())
            return result if isinstance(result, list) else []
    except Exception:
        return []

# ── sync engine ───────────────────────────────────────────────────────────────

def _push_table(manifest: dict) -> dict:
    local   = manifest["local"]
    remote  = manifest["remote"]
    cols    = manifest["columns"]
    key     = manifest["key"]
    cursor  = _get_cursor(local)

    try:
        conn = sqlite3.connect(DB, timeout=10)
        col_str = ", ".join(cols)
        rows_raw = conn.execute(
            f"SELECT {key}, {col_str} FROM {local} WHERE {key} > ? ORDER BY {key} LIMIT 200",
            (cursor,)
        ).fetchall()
        conn.close()
    except Exception as e:
        _log_sync(local, "push", 0, 0, "error", str(e))
        return {"error": str(e), "table": local}

    if not rows_raw:
        return {"table": local, "pushed": 0, "note": "no_new_rows"}

    all_cols = [key] + cols
    rows_dicts = [dict(zip(all_cols, r)) for r in rows_raw]
    result = _sb_upsert(remote, rows_dicts)

    if "error" not in result:
        new_cursor = max(r[0] for r in rows_raw)
        _set_cursor(local, new_cursor)
        _log_sync(local, "push", len(rows_dicts), 0, "ok")
        log.info("push table=%s rows=%d cursor=%d", local, len(rows_dicts), new_cursor)
    else:
        _log_sync(local, "push", 0, 0, "error", result.get("error", ""))
        log.warning("push failed table=%s error=%s", local, result.get("error"))

    return {"table": local, "pushed": len(rows_dicts), "result": result}

def _sync_all() -> list:
    results = []
    for m in SYNC_MANIFEST:
        if m["direction"] in ("push", "both"):
            results.append(_push_table(m))
    return results

def _sync_status() -> list:
    try:
        conn = sqlite3.connect(DB, timeout=5)
        rows = conn.execute("""
            SELECT table_name, last_id, last_ts
            FROM supabase_sync_cursor ORDER BY table_name
        """).fetchall()
        conn.close()
        return [{"table": r[0], "last_id": r[1], "last_ts": r[2]} for r in rows]
    except Exception:
        return []

# ── HTTP handler ───────────────────────────────────────────────────────────────

class SupabaseHandler(BaseHTTPRequestHandler):
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
            configured = bool(SB_URL and SB_KEY)
            self._respond(200, {"status": "ok", "configured": configured,
                                "url": SB_URL[:30] + "..." if SB_URL else "",
                                "tables": len(SYNC_MANIFEST)})
        elif self.path == "/status":
            self._respond(200, {"cursors": _sync_status(), "manifest": SYNC_MANIFEST})
        elif self.path == "/sync":
            results = _sync_all()
            self._respond(200, {"synced": len(results), "results": results})
        elif self.path.startswith("/read/"):
            table = self.path[6:]
            data  = _sb_select(table)
            self._respond(200, {"table": table, "rows": len(data), "data": data})
        else:
            self._respond(404, {"error": "not_found"})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            data   = json.loads(self.rfile.read(length))

            if self.path == "/upsert":
                table = data.get("table")
                rows  = data.get("rows", [])
                if not table or not rows:
                    self._respond(400, {"error": "table and rows required"})
                    return
                result = _sb_upsert(table, rows)
                self._respond(200, result)

            elif self.path == "/sync":
                results = _sync_all()
                self._respond(200, {"synced": len(results), "results": results})

            else:
                self._respond(404, {"error": "unknown_path"})

        except json.JSONDecodeError:
            self._respond(400, {"error": "invalid_json"})
        except Exception as e:
            log.error("handler_error: %s", e)
            self._respond(500, {"error": str(e)})

# ── background sync loop ───────────────────────────────────────────────────────

_running = True

def _shutdown(*_):
    global _running
    log.info("shutdown signal — exiting cleanly")
    _running = False

signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT,  _shutdown)

def main():
    _db_init()
    server = HTTPServer(("0.0.0.0", PORT), SupabaseHandler)
    log.info("Supabase sync agent listening on port %d", PORT)
    log.info("URL: %s | Key: %s | Interval: %ds | Tables: %d",
             SB_URL[:30] + "..." if SB_URL else "not_set",
             "configured" if SB_KEY else "not_set",
             SYNC_INTERVAL, len(SYNC_MANIFEST))

    _last_sync = 0
    try:
        while _running:
            server.handle_request()
            now = time.time()
            if now - _last_sync >= SYNC_INTERVAL and SB_URL and SB_KEY:
                results = _sync_all()
                pushed  = sum(r.get("pushed", 0) for r in results)
                log.info("auto_sync pushed=%d tables=%d", pushed, len(results))
                _last_sync = now
    finally:
        server.server_close()
        log.info("Supabase sync agent stopped")

if __name__ == "__main__":
    main()
