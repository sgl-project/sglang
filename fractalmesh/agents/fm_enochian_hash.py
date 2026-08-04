"""
FractalMesh Enochian Hash Agent
φ-layered multi-round SHA3-256 hashing for sovereign payload fingerprinting
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import sys
import math
import hmac
import time
import signal
import hashlib
import sqlite3
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT     = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB       = os.path.join(ROOT, "database", "sovereign.db")
INTERVAL = int(os.getenv("ENOCHIAN_INTERVAL", "300"))
PHI      = 1.6180339887
ROUNDS   = int(os.getenv("ENOCHIAN_ROUNDS", "7"))

_running = True


def enochian_hash(payload: bytes, rounds: int = ROUNDS) -> str:
    h = hashlib.sha3_256(payload).digest()
    for i in range(rounds):
        phi_seed = hashlib.sha256(str(PHI * (i + 1)).encode()).digest()
        h        = hashlib.sha3_256(h + phi_seed).digest()
    return h.hex()


def hmac_enochian(payload: bytes) -> str:
    secret = os.getenv("BUS_SECRET", "fallback").encode()
    e_hash = enochian_hash(payload).encode()
    return hmac.new(secret, e_hash, hashlib.sha256).hexdigest()


def phi_rotate(value: float, steps: int = 3) -> float:
    for _ in range(steps):
        value = (value * PHI) % (2 * math.pi)
    return round(value, 8)


def _db_init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS enochian_log (
        id INTEGER PRIMARY KEY, sample_id TEXT, enochian TEXT,
        hmac_sig TEXT, phi_rotation REAL, ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()


def _log(sample_id, e_hash, sig, phi_r):
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("INSERT INTO enochian_log (sample_id,enochian,hmac_sig,phi_rotation) VALUES (?,?,?,?)",
                 (sample_id, e_hash, sig, phi_r))
    conn.commit(); conn.close()


def run_cycle():
    ts      = datetime.utcnow().isoformat()
    samples = [("heartbeat", ts.encode()), ("abn", b"56628117363"),
               ("bus_secret", os.getenv("BUS_SECRET", "fallback").encode())]
    print(f"[fm-enochian-hash] {ts} | rounds={ROUNDS}")
    for sid, payload in samples:
        e_hash = enochian_hash(payload)
        sig    = hmac_enochian(payload)
        phi_r  = phi_rotate(float(int(e_hash[:8], 16)))
        print(f"   → {sid:<12} {e_hash[:16]}… φ={phi_r}")
        _log(sid, e_hash, sig, phi_r)


def _sigterm(s, f):
    global _running
    _running = False


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    _db_init()
    print(f"[fm-enochian-hash] Active | interval={INTERVAL}s | rounds={ROUNDS}")
    while _running:
        try:
            run_cycle()
        except Exception as e:
            print(f"[fm-enochian-hash] ERR {e}")
        for _ in range(INTERVAL):
            if not _running: break
            time.sleep(1)
    print("[fm-enochian-hash] Stopped.")
