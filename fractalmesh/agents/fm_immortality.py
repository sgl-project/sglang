"""
FractalMesh Immortality Agent
Hardware immortality protocol: backs up sovereign.db, verifies checksums, rotates snapshots
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import time
import signal
import shutil
import hashlib
import sqlite3
from datetime import datetime

ROOT     = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB       = os.path.join(ROOT, "database", "sovereign.db")
BACKUP   = os.path.join(ROOT, "backups")
INTERVAL = int(os.getenv("IMMORTALITY_INTERVAL", "3600"))
KEEP     = int(os.getenv("IMMORTALITY_KEEP_SNAPSHOTS", "24"))

_running = True


def _db_init():
    os.makedirs(BACKUP, exist_ok=True)
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS immortality_log (
        id INTEGER PRIMARY KEY, snapshot TEXT, sha256 TEXT, size_bytes INTEGER,
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


def _snapshot() -> tuple:
    if not os.path.exists(DB):
        return None, 0, ""
    ts_str   = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    snap_path = os.path.join(BACKUP, f"sovereign_{ts_str}.db")
    try:
        shutil.copy2(DB, snap_path)
        sha  = _sha256(snap_path)
        size = os.path.getsize(snap_path)
        return snap_path, size, sha
    except Exception as e:
        print(f"   [immortality] snapshot failed: {e}")
        return None, 0, ""


def _rotate():
    snaps = sorted(
        [f for f in os.listdir(BACKUP) if f.startswith("sovereign_") and f.endswith(".db")]
    )
    while len(snaps) > KEEP:
        old = os.path.join(BACKUP, snaps.pop(0))
        try:
            os.remove(old)
            print(f"   [immortality] rotated {old}")
        except Exception:
            pass


def _log(snap, sha, size):
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("INSERT INTO immortality_log (snapshot,sha256,size_bytes) VALUES (?,?,?)",
                 (snap or "FAILED", sha, size))
    conn.commit(); conn.close()


def run_cycle():
    snap, size, sha = _snapshot()
    _rotate()
    ts = datetime.utcnow().isoformat()
    if snap:
        print(f"[fm-immortality] {ts} | snap={os.path.basename(snap)} | {size//1024}KB | sha={sha[:12]}…")
    else:
        print(f"[fm-immortality] {ts} | snapshot FAILED or DB missing")
    if sha:
        _log(snap, sha, size)


def _sigterm(s, f):
    global _running
    _running = False


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    _db_init()
    print(f"[fm-immortality] Active | interval={INTERVAL}s | keep={KEEP} snapshots")
    while _running:
        try:
            run_cycle()
        except Exception as e:
            print(f"[fm-immortality] ERR {e}")
        for _ in range(INTERVAL):
            if not _running: break
            time.sleep(1)
    print("[fm-immortality] Stopped.")
