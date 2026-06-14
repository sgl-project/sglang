"""
FractalMesh IPFS Agent
Pins sovereign documents to IPFS; logs CIDs to sovereign.db
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import json
import time
import signal
import sqlite3
import subprocess
import urllib.request
import urllib.parse
from datetime import datetime

ROOT     = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB       = os.path.join(ROOT, "database", "sovereign.db")
DIST     = os.path.join(ROOT, "dist")
INTERVAL = int(os.getenv("IPFS_INTERVAL", "3600"))
DRY_RUN  = os.getenv("ENABLE_IPFS_PIN", "false").lower() != "true"

# Supports local IPFS daemon (preferred) or Pinata API
IPFS_API      = os.getenv("IPFS_API_URL", "http://127.0.0.1:5001")
PINATA_JWT    = os.getenv("PINATA_JWT", "")

PIN_TARGETS = [
    "ai_node_blueprint.md",
    "obrien_compliance_v1.md",
    "obrien_msa_final.md",
    "general_msa_template.md",
]

_running = True


def _db_init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS ipfs_log (
        id INTEGER PRIMARY KEY, filename TEXT, cid TEXT UNIQUE, size_bytes INTEGER,
        backend TEXT, ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()


def _pin_local(filepath: str) -> str:
    try:
        r = subprocess.run(
            ["ipfs", "add", "--pin", "--quieter", filepath],
            capture_output=True, text=True, timeout=30)
        if r.returncode == 0:
            return r.stdout.strip()
        return f"err:{r.stderr.strip()[:60]}"
    except FileNotFoundError:
        return "ipfs_not_installed"
    except Exception as e:
        return f"err:{e}"


def _pin_pinata(filepath: str) -> str:
    if not PINATA_JWT:
        return "no_jwt"
    try:
        with open(filepath, "rb") as f:
            data = f.read()
        filename = os.path.basename(filepath)
        boundary = "----FractalMeshBoundary"
        body     = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
                    f"filename=\"{filename}\"\r\nContent-Type: text/markdown\r\n\r\n").encode() + \
                   data + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(
            "https://api.pinata.cloud/pinning/pinFileToIPFS",
            data=body,
            headers={"Authorization": f"Bearer {PINATA_JWT}",
                     "Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["IpfsHash"]
    except Exception as e:
        return f"err:{e}"


def _log(filename, cid, size, backend):
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("""INSERT INTO ipfs_log (filename,cid,size_bytes,backend) VALUES (?,?,?,?)
        ON CONFLICT(cid) DO UPDATE SET ts=CURRENT_TIMESTAMP""",
        (filename, cid, size, backend))
    conn.commit(); conn.close()


def run_cycle():
    ts = datetime.utcnow().isoformat()
    print(f"[fm-ipfs] {ts} | dry={DRY_RUN}")
    for fname in PIN_TARGETS:
        path = os.path.join(DIST, fname)
        if not os.path.exists(path):
            print(f"   → {fname:<30} missing"); continue
        size = os.path.getsize(path)
        if DRY_RUN:
            cid, backend = "dry_run", "none"
        elif PINATA_JWT:
            cid, backend = _pin_pinata(path), "pinata"
        else:
            cid, backend = _pin_local(path), "local"
        print(f"   → {fname:<30} {cid[:20] if len(cid)>20 else cid} [{backend}]")
        if not DRY_RUN:
            _log(fname, cid, size, backend)


def _sigterm(s, f):
    global _running
    _running = False


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    _db_init()
    print(f"[fm-ipfs] Active | interval={INTERVAL}s | dry={DRY_RUN}")
    while _running:
        try:
            run_cycle()
        except Exception as e:
            print(f"[fm-ipfs] ERR {e}")
        for _ in range(INTERVAL):
            if not _running: break
            time.sleep(1)
    print("[fm-ipfs] Stopped.")
