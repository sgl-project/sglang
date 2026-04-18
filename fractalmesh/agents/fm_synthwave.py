"""
FractalMesh Synthwave Agent
Generates φ-harmonic content metadata and schedules media asset production
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import math
import time
import signal
import sqlite3
from datetime import datetime

ROOT     = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB       = os.path.join(ROOT, "database", "sovereign.db")
INTERVAL = int(os.getenv("SYNTHWAVE_INTERVAL", "3600"))
PHI      = 1.6180339887

# φ-harmonic frequency table for content generation
FREQ_TABLE = [
    {"note": "C4",  "hz": 261.63, "phi_mult": PHI ** 0},
    {"note": "E4",  "hz": 329.63, "phi_mult": PHI ** 1},
    {"note": "G4",  "hz": 392.00, "phi_mult": PHI ** 2},
    {"note": "B4",  "hz": 493.88, "phi_mult": PHI ** 3},
    {"note": "D5",  "hz": 587.33, "phi_mult": PHI ** 4},
]

CONTENT_QUEUE = [
    {"id": "ep01", "title": "Sovereign Node Deploy",   "genre": "synthwave", "bpm": 128},
    {"id": "ep02", "title": "HMAC Pulse",              "genre": "ambient",   "bpm": 90},
    {"id": "ep03", "title": "φ-Harmonic Resonance",   "genre": "darkwave",  "bpm": 110},
    {"id": "ep04", "title": "Edge Node Transmission",  "genre": "cyberpunk", "bpm": 140},
]

_running = True
_cycle   = 0


def _db_init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS synthwave_log (
        id INTEGER PRIMARY KEY, content_id TEXT, title TEXT, genre TEXT,
        bpm INTEGER, phi_freq REAL, ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()


def _phi_freq(bpm: int) -> float:
    base = (bpm / 60.0)  # Hz equivalent of BPM
    return round(base * PHI * math.sin(base * PHI), 4)


def _log(c: dict, phi_freq: float):
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("INSERT INTO synthwave_log (content_id,title,genre,bpm,phi_freq) VALUES (?,?,?,?,?)",
                 (c["id"], c["title"], c["genre"], c["bpm"], phi_freq))
    conn.commit(); conn.close()


def run_cycle():
    global _cycle
    _cycle += 1
    ts = datetime.utcnow().isoformat()
    print(f"[fm-synthwave] Cycle {_cycle} | {ts}")
    for note in FREQ_TABLE:
        phi_hz = round(note["hz"] * note["phi_mult"], 2)
        print(f"   ♪ {note['note']:<4} {note['hz']:>7.2f}Hz × φ^{FREQ_TABLE.index(note)} = {phi_hz:>9.2f}Hz")
    idx = (_cycle - 1) % len(CONTENT_QUEUE)
    c   = CONTENT_QUEUE[idx]
    pf  = _phi_freq(c["bpm"])
    print(f"   → Queued: [{c['id']}] \"{c['title']}\" | {c['genre']} {c['bpm']}bpm | φ-freq={pf}")
    _log(c, pf)


def _sigterm(s, f):
    global _running
    _running = False


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    _db_init()
    print(f"[fm-synthwave] Active | interval={INTERVAL}s | tracks={len(CONTENT_QUEUE)}")
    while _running:
        try:
            run_cycle()
        except Exception as e:
            print(f"[fm-synthwave] ERR {e}")
        for _ in range(INTERVAL):
            if not _running: break
            time.sleep(1)
    print("[fm-synthwave] Stopped.")
