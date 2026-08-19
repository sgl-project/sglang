"""
FractalMesh Enterprise Bus v2.0.0
Thread-safe message bus with circuit breakers, HMAC signing, monetization tracking.
Samuel James Hiotis | ABN 56 628 117 363 | Sole Trader
"""
import os
import json
import time
import hmac
import signal
import sqlite3
import hashlib
import threading
import queue
from datetime import datetime
from collections import defaultdict, deque

ROOT     = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB       = os.path.join(ROOT, "database", "sovereign.db")
INTERVAL = int(os.getenv("ENTERPRISE_BUS_INTERVAL", "60"))
SECRET   = os.getenv("BUS_SECRET", "changeme").encode()

# Circuit-breaker thresholds
CB_ERROR_RATE    = float(os.getenv("CB_ERROR_RATE",    "0.05"))   # 5%
CB_WINDOW        = int(os.getenv("CB_WINDOW",          "60"))     # seconds
CB_HALF_OPEN_TTL = int(os.getenv("CB_HALF_OPEN_TTL",  "30"))     # seconds

PHI = 1.6180339887

_running    = True
_bus_lock   = threading.Lock()
_msg_queue  = queue.Queue(maxsize=1000)


# ── Circuit Breaker ────────────────────────────────────────────────────────────

class CircuitBreaker:
    CLOSED    = "closed"
    OPEN      = "open"
    HALF_OPEN = "half_open"

    def __init__(self, name: str, threshold: float = CB_ERROR_RATE, window: int = CB_WINDOW):
        self.name       = name
        self.threshold  = threshold
        self.window     = window
        self.state      = self.CLOSED
        self.events     = deque()   # (timestamp, success: bool)
        self.opened_at  = 0.0
        self._lock      = threading.Lock()

    def record(self, success: bool):
        now = time.time()
        with self._lock:
            self.events.append((now, success))
            # evict old events
            while self.events and self.events[0][0] < now - self.window:
                self.events.popleft()
            self._evaluate(now)

    def _evaluate(self, now: float):
        if not self.events:
            return
        total  = len(self.events)
        errors = sum(1 for _, ok in self.events if not ok)
        rate   = errors / total if total else 0.0

        if self.state == self.CLOSED and rate > self.threshold:
            self.state     = self.OPEN
            self.opened_at = now
        elif self.state == self.OPEN:
            if now - self.opened_at > CB_HALF_OPEN_TTL:
                self.state = self.HALF_OPEN
        elif self.state == self.HALF_OPEN:
            # One clean success closes; one failure re-opens
            last_ok = self.events[-1][1]
            self.state = self.CLOSED if last_ok else self.OPEN
            if not last_ok:
                self.opened_at = now

    @property
    def open(self) -> bool:
        return self.state == self.OPEN

    def status(self) -> dict:
        with self._lock:
            total  = len(self.events)
            errors = sum(1 for _, ok in self.events if not ok)
            return {
                "name":       self.name,
                "state":      self.state,
                "total":      total,
                "errors":     errors,
                "error_rate": round(errors / total, 4) if total else 0.0,
            }


# ── HMAC signing ──────────────────────────────────────────────────────────────

def _sign(payload: dict) -> str:
    body = json.dumps(payload, sort_keys=True).encode()
    return hmac.new(SECRET, body, hashlib.sha256).hexdigest()


def _verify(payload: dict, sig: str) -> bool:
    return hmac.compare_digest(_sign(payload), sig)


# ── DB ────────────────────────────────────────────────────────────────────────

def _db_init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS enterprise_bus_log (
        id INTEGER PRIMARY KEY, channel TEXT, event TEXT, payload TEXT,
        sig TEXT, cb_state TEXT, phi_score REAL,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS circuit_breaker_log (
        id INTEGER PRIMARY KEY, cb_name TEXT, state TEXT, error_rate REAL,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS monetization_events (
        id INTEGER PRIMARY KEY, source TEXT, event_type TEXT, amount_aud REAL,
        idempotency_key TEXT UNIQUE, channel TEXT, phi_score REAL,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()


def _log_event(channel: str, event: str, payload: dict, sig: str, cb_state: str, phi: float):
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("""INSERT INTO enterprise_bus_log
        (channel,event,payload,sig,cb_state,phi_score) VALUES (?,?,?,?,?,?)""",
        (channel, event, json.dumps(payload)[:500], sig[:16], cb_state, phi))
    conn.commit(); conn.close()


def _log_monetization(source: str, event_type: str, amount: float,
                       ikey: str, channel: str, phi: float):
    conn = sqlite3.connect(DB, timeout=10)
    try:
        conn.execute("""INSERT INTO monetization_events
            (source,event_type,amount_aud,idempotency_key,channel,phi_score)
            VALUES (?,?,?,?,?,?)""",
            (source, event_type, amount, ikey, channel, phi))
        conn.commit()
    except sqlite3.IntegrityError:
        pass  # idempotent
    finally:
        conn.close()


# ── Message Bus ───────────────────────────────────────────────────────────────

_circuit_breakers: dict[str, CircuitBreaker] = defaultdict(
    lambda: CircuitBreaker("default"))
_subscribers: dict[str, list] = defaultdict(list)


def publish(channel: str, event: str, payload: dict, amount_aud: float = 0.0) -> bool:
    cb = _circuit_breakers[channel]
    if cb.open:
        print(f"[bus] CIRCUIT OPEN: {channel} — dropping {event}")
        return False

    ts       = datetime.utcnow().isoformat()
    msg      = {"channel": channel, "event": event, "ts": ts, **payload}
    sig      = _sign(msg)
    layer    = list(_circuit_breakers.keys()).index(channel) if channel in _circuit_breakers else 0
    phi_score = round(1.0 * (PHI ** layer), 6)

    try:
        _msg_queue.put_nowait((channel, event, msg, sig, phi_score))
        cb.record(True)
        if amount_aud > 0:
            ikey = hashlib.sha256(f"{channel}{event}{ts}".encode()).hexdigest()[:24]
            _log_monetization(channel, event, amount_aud, ikey, channel, phi_score)
        return True
    except queue.Full:
        cb.record(False)
        return False


def subscribe(channel: str, handler):
    with _bus_lock:
        _subscribers[channel].append(handler)


def _dispatch_loop():
    while _running or not _msg_queue.empty():
        try:
            channel, event, msg, sig, phi = _msg_queue.get(timeout=1)
            handlers = _subscribers.get(channel, [])
            for h in handlers:
                try:
                    h(channel, event, msg)
                except Exception as e:
                    print(f"[bus] handler error on {channel}: {e}")
                    _circuit_breakers[channel].record(False)
            _log_event(channel, event, msg, sig,
                       _circuit_breakers[channel].state, phi)
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[bus] dispatch error: {e}")


# ── Health / Status ───────────────────────────────────────────────────────────

def bus_status() -> dict:
    return {
        "queue_size":  _msg_queue.qsize(),
        "channels":    len(_circuit_breakers),
        "subscribers": {ch: len(handlers) for ch, handlers in _subscribers.items()},
        "breakers":    [cb.status() for cb in _circuit_breakers.values()],
    }


def run_cycle():
    ts = datetime.utcnow().isoformat()
    # Emit a heartbeat to every registered channel
    for ch in list(_circuit_breakers.keys()):
        publish(ch, "heartbeat", {"source": "enterprise_bus", "ts": ts})

    # Log CB snapshot
    conn = sqlite3.connect(DB, timeout=10)
    for cb in _circuit_breakers.values():
        st = cb.status()
        conn.execute("""INSERT INTO circuit_breaker_log (cb_name,state,error_rate)
            VALUES (?,?,?)""", (st["name"], st["state"], st["error_rate"]))
    conn.commit(); conn.close()

    status = bus_status()
    print(f"[fm-enterprise-bus] {ts} | q={status['queue_size']} "
          f"| channels={status['channels']} | "
          + " ".join(f"{cb['name']}:{cb['state']}" for cb in status["breakers"]))


def _sigterm(s, f):
    global _running
    _running = False


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT,  _sigterm)
    _db_init()

    # Register default channels
    for ch in ["revenue", "compliance", "infra", "intelligence", "outreach",
               "health", "security", "tokenomics"]:
        _circuit_breakers[ch] = CircuitBreaker(ch)

    # Start dispatcher thread
    t = threading.Thread(target=_dispatch_loop, daemon=True)
    t.start()

    print(f"[fm-enterprise-bus] Active | channels={len(_circuit_breakers)} | "
          f"cb_threshold={CB_ERROR_RATE} | window={CB_WINDOW}s")

    while _running:
        try:
            run_cycle()
        except Exception as e:
            print(f"[fm-enterprise-bus] ERR {e}")
        for _ in range(INTERVAL):
            if not _running: break
            time.sleep(1)

    t.join(timeout=5)
    print("[fm-enterprise-bus] Stopped.")
