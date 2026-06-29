"""
FractalMesh RL Quad Agent
4-quadrant reinforcement learning: revenue, compliance, infra, reputation axes
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import math
import time
import random
import signal
import sqlite3
from datetime import datetime

ROOT     = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB       = os.path.join(ROOT, "database", "sovereign.db")
INTERVAL = int(os.getenv("RL_QUAD_INTERVAL", "180"))
PHI      = 1.6180339887
ALPHA    = float(os.getenv("RL_QUAD_ALPHA", "0.05"))
GAMMA    = float(os.getenv("RL_QUAD_GAMMA", "0.98"))

QUADS = [
    {"id": "revenue",    "label": "Revenue Generation",       "weight": PHI ** 1},
    {"id": "compliance", "label": "Compliance & Audit",       "weight": PHI ** 2},
    {"id": "infra",      "label": "Infrastructure Stability", "weight": PHI ** 3},
    {"id": "reputation", "label": "Reputation & Authority",   "weight": PHI ** 0},
]
ACTIONS = ["invest", "optimise", "hold", "rebalance"]

_q       = {q["id"]: {a: 0.0 for a in ACTIONS} for q in QUADS}
_ep      = 0
_running = True


def _db_init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS rl_quad_log (
        id INTEGER PRIMARY KEY, episode INTEGER, quad_id TEXT,
        action TEXT, reward REAL, q_value REAL, phi_weight REAL,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()


def _reward(quad: dict, action: str) -> float:
    base  = math.log(quad["weight"] + 1) * PHI
    noise = random.gauss(0, 0.03)
    bonus = 0.15 if action == "invest" and quad["id"] == "revenue" else 0.0
    return round(base + noise + bonus, 4)


def _choose(quad_id: str) -> str:
    if random.random() < 0.1:
        return random.choice(ACTIONS)
    return max(_q[quad_id], key=_q[quad_id].get)


def _update(quad_id, action, reward) -> float:
    best = max(_q[quad_id].values())
    _q[quad_id][action] += ALPHA * (reward + GAMMA * best - _q[quad_id][action])
    return round(_q[quad_id][action], 4)


def _log(ep, quad_id, action, reward, q, phi_w):
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("INSERT INTO rl_quad_log (episode,quad_id,action,reward,q_value,phi_weight) VALUES (?,?,?,?,?,?)",
                 (ep, quad_id, action, reward, q, phi_w))
    conn.commit(); conn.close()


def run_cycle():
    global _ep
    _ep += 1
    print(f"[fm-rl-quad] Episode {_ep} | {datetime.utcnow().isoformat()}")
    for quad in QUADS:
        action = _choose(quad["id"])
        reward = _reward(quad, action)
        q_val  = _update(quad["id"], action, reward)
        print(f"   → {quad['id']:<12} {action:<12} R={reward:+.4f}  Q={q_val:.4f}  φw={quad['weight']:.4f}")
        _log(_ep, quad["id"], action, reward, q_val, quad["weight"])


def _sigterm(s, f):
    global _running
    _running = False


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    _db_init()
    print(f"[fm-rl-quad] Active | interval={INTERVAL}s | quads={len(QUADS)}")
    while _running:
        try:
            run_cycle()
        except Exception as e:
            print(f"[fm-rl-quad] ERR {e}")
        for _ in range(INTERVAL):
            if not _running: break
            time.sleep(1)
    print("[fm-rl-quad] Stopped.")
