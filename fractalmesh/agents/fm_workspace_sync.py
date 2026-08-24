"""
FractalMesh Workspace Sync Agent
Syncs working files between FRACTALMESH_HOME and git repo; logs drift to sovereign.db
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import time
import signal
import sqlite3
import hashlib
import subprocess
from datetime import datetime

ROOT      = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
REPO_ROOT = os.getenv("REPO_ROOT", os.path.expanduser("~/sglang/fractalmesh"))
DB        = os.path.join(ROOT, "database", "sovereign.db")
INTERVAL  = int(os.getenv("WORKSPACE_SYNC_INTERVAL", "300"))
DRY_RUN   = os.getenv("ENABLE_WORKSPACE_SYNC", "false").lower() != "true"

SYNC_DIRS = ["agents", "modules", "api", "scripts", "tunnels"]

_running = True


def _db_init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS workspace_sync_log (
        id INTEGER PRIMARY KEY, path TEXT, action TEXT, sha256 TEXT,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def _log(path, action, sha):
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("INSERT INTO workspace_sync_log (path,action,sha256) VALUES (?,?,?)",
                 (path[:300], action, sha[:64]))
    conn.commit(); conn.close()


def _git_status() -> list:
    try:
        r = subprocess.run(
            ["git", "-C", REPO_ROOT, "status", "--porcelain"],
            capture_output=True, text=True, timeout=15)
        return [l.strip() for l in r.stdout.splitlines() if l.strip()]
    except Exception:
        return []


def run_cycle():
    ts      = datetime.utcnow().isoformat()
    changed = _git_status()
    print(f"[fm-workspace-sync] {ts} | repo={REPO_ROOT} | changed={len(changed)} | dry={DRY_RUN}")
    for entry in changed[:20]:
        flag, path = entry[:2].strip(), entry[3:]
        sha        = _sha256(os.path.join(REPO_ROOT, path)) if os.path.exists(os.path.join(REPO_ROOT, path)) else ""
        action     = {"M": "modified", "A": "added", "D": "deleted", "?": "untracked"}.get(flag, flag)
        print(f"   → [{flag}] {path:<50} {sha[:12] if sha else 'n/a'}")
        _log(path, action, sha)
    if not changed:
        print(f"   → clean — no workspace drift")


def _sigterm(s, f):
    global _running
    _running = False


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    _db_init()
    print(f"[fm-workspace-sync] Active | interval={INTERVAL}s | dry={DRY_RUN}")
    while _running:
        try:
            run_cycle()
        except Exception as e:
            print(f"[fm-workspace-sync] ERR {e}")
        for _ in range(INTERVAL):
            if not _running: break
            time.sleep(1)
    print("[fm-workspace-sync] Stopped.")
