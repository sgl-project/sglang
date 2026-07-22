"""
FractalMesh AZR Reinforcement Learning Agent
φ-weighted Q-learning over mesh node performance signals
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
INTERVAL = int(os.getenv("AZR_RL_INTERVAL", "120"))
ALPHA    = float(os.getenv("AZR_ALPHA", "0.1"))
GAMMA    = float(os.getenv("AZR_GAMMA", "0.95"))
PHI      = 1.6180339887

NODES    = ["fm-bus", "fm-pulse-bus", "fm-gitops-runner", "fm-integrator",
            "fm-stripe-mon", "fm-harmonic", "fm-warden", "fm-oversight",
            "fm-healer", "fm-immortality", "fm-sovereign-ops"]
ACTIONS  = ["boost", "throttle", "restart", "hold"]

_q: dict  = {}
_ep       = 0
_running  = True


def _db_init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS rl_episodes (
        id INTEGER PRIMARY KEY, episode INTEGER, node TEXT, action TEXT,
        reward REAL, q_value REAL, ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()


def _phi_reward(node: str, action: str) -> float:
    idx    = NODES.index(node) if node in NODES else 0
    base   = math.sin(idx * PHI) * PHI
    return round(base + random.gauss(0, 0.05) + (0.2 if action == "boost" else -0.1), 4)


def _choose(state: str) -> str:
    if state not in _q or random.random() < 0.15:
        return random.choice(ACTIONS)
    return max(_q[state], key=_q[state].get)


def _update(state: str, action: str, reward: float) -> float:
    for s in (state,):
        _q.setdefault(s, {a: 0.0 for a in ACTIONS})
    best = max(_q[state].values())
    _q[state][action] += ALPHA * (reward + GAMMA * best - _q[state][action])
    return round(_q[state][action], 4)


def _log(ep, node, action, reward, q):
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("INSERT INTO rl_episodes (episode,node,action,reward,q_value) VALUES (?,?,?,?,?)",
                 (ep, node, action, reward, q))
    conn.commit(); conn.close()


def run_episode():
    global _ep
    _ep += 1
    print(f"[fm-azr-rl] Episode {_ep} | {datetime.utcnow().isoformat()}")
    for node in NODES:
        state  = f"{node}:active"
        action = _choose(state)
        reward = _phi_reward(node, action)
        q_val  = _update(state, action, reward)
        print(f"   → {node:<24} {action:<10} R={reward:+.4f}  Q={q_val:.4f}")
        _log(_ep, node, action, reward, q_val)


def _sigterm(s, f):
    global _running
    _running = False


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    _db_init()
    print(f"[fm-azr-rl] Active | interval={INTERVAL}s | α={ALPHA} γ={GAMMA}")
    while _running:
        try:
            run_episode()
        except Exception as e:
            print(f"[fm-azr-rl] ERR {e}")
        for _ in range(INTERVAL):
            if not _running: break
            time.sleep(1)
    print("[fm-azr-rl] Stopped.")
