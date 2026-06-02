"""
fm_revenue_convergence.py — Revenue Convergence Monitor (port 7867)
Tracks multi-stream revenue vs hourly/daily targets, computes velocity,
sends Telegram alerts on milestones and swarm debt ratio spikes.
Credentials from ~/.secrets/fractal.env via os.environ — never hardcoded.
Samuel James Hiotis | ABN 56 628 117 363
"""

import hmac
import json
import logging
import os
import sqlite3
import subprocess
import threading
import time
from datetime import datetime, timedelta, timezone

import requests
from flask import Flask, jsonify, request
from waitress import serve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fm-revenue-convergence")

DB = os.path.expanduser(os.getenv("SOVEREIGN_DB", "~/fmsaas/data/sovereign.db"))
PORT = int(os.getenv("REVENUE_CONVERGENCE_PORT", "7867"))
DAILY_TARGET_AUD = float(os.getenv("DAILY_TARGET_AUD", "1070"))

# ── Credentials (from vault, never hardcoded) ─────────────────────────────────
TELEGRAM_BOT = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")
STRIPE_LINK = os.getenv("STRIPE_PAYMENT_LINK", "")
DEVTO_KEY = os.getenv("DEVTO_API_KEY", "")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")
CLOUDFLARE_TOKEN = os.getenv("CLOUDFLARE_TOKEN", "")
CLOUDFLARE_ACCT = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")

app = Flask(__name__)


# ── Auth ──────────────────────────────────────────────────────────────────────


def _check_auth(req: request) -> bool:
    if not ADMIN_SECRET:
        return True
    provided = req.headers.get("X-Admin-Secret", "")
    if not provided:
        return False
    return hmac.compare_digest(provided, ADMIN_SECRET)


# ── DB helpers ────────────────────────────────────────────────────────────────


def _db():
    conn = sqlite3.connect(DB, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_schema():
    try:
        c = _db()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS revenue_streams (
                id INTEGER PRIMARY KEY,
                stream_name TEXT UNIQUE,
                stream_type TEXT,
                target_aud REAL DEFAULT 0,
                hourly_target REAL DEFAULT 0,
                daily_target REAL DEFAULT 0,
                status TEXT DEFAULT 'active'
            );
            CREATE TABLE IF NOT EXISTS revenue_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stream_id INTEGER,
                event_type TEXT,
                value REAL DEFAULT 0,
                currency TEXT DEFAULT 'AUD',
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS convergence_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_24h REAL DEFAULT 0,
                total_1h REAL DEFAULT 0,
                velocity REAL DEFAULT 0,
                technical_debt_ratio REAL DEFAULT 0,
                swarm_health INTEGER DEFAULT 0,
                projection_24h REAL DEFAULT 0,
                gap_to_target REAL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS hourly_targets (
                hour INTEGER PRIMARY KEY,
                target_aud REAL DEFAULT 0
            );
            INSERT OR IGNORE INTO hourly_targets (hour, target_aud) VALUES
            (0,25),(1,15),(2,10),(3,8),(4,8),(5,10),(6,20),(7,35),
            (8,50),(9,65),(10,75),(11,80),(12,85),(13,80),(14,75),
            (15,70),(16,65),(17,60),(18,55),(19,50),(20,45),(21,40),
            (22,35),(23,30);
            INSERT OR IGNORE INTO revenue_streams
                (id,stream_name,stream_type,target_aud,hourly_target,daily_target)
            VALUES
            (1,'stripe','payment_gateway',150,6.25,150),
            (2,'devto','content_monetization',30,1.25,30),
            (3,'openrouter','api_usage',200,8.33,200),
            (4,'cloudflare_workers','saas_microservice',100,4.17,100),
            (5,'consulting','direct_service',500,20.83,500);
        """)
        c.commit()
        c.close()
    except Exception as e:
        logger.warning("Schema init: %s", e)


def _log_event(stream_id: int, event_type: str, value: float, meta=""):
    try:
        c = _db()
        meta_str = json.dumps(meta) if isinstance(meta, dict) else str(meta)
        c.execute(
            "INSERT INTO revenue_events (stream_id,event_type,value,metadata)"
            " VALUES (?,?,?,?)",
            (stream_id, event_type, value, meta_str),
        )
        c.commit()
        c.close()
    except Exception as e:
        logger.warning("log_event: %s", e)


# ── Telegram ──────────────────────────────────────────────────────────────────


def _telegram(msg: str):
    if not TELEGRAM_BOT or not TELEGRAM_CHAT:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT, "text": msg, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        logger.debug("telegram: %s", e)


# ── Stream checks ─────────────────────────────────────────────────────────────


def _check_devto() -> dict:
    if not DEVTO_KEY:
        return {"stream": "devto", "status": "no_key"}
    try:
        r = requests.get(
            "https://dev.to/api/articles/me?per_page=10",
            headers={"api-key": DEVTO_KEY},
            timeout=15,
        )
        articles = r.json() if r.ok else []
        total_views = sum(a.get("page_views_count", 0) for a in articles)
        return {
            "stream": "devto",
            "articles": len(articles),
            "total_views": total_views,
            "estimated_aud": round(total_views * 0.005, 2),
            "status": "ok",
        }
    except Exception as e:
        return {"stream": "devto", "status": f"error:{e}"}


def _check_openrouter() -> dict:
    if not OPENROUTER_KEY:
        return {"stream": "openrouter", "status": "no_key"}
    try:
        r = requests.get(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
            timeout=10,
        )
        if r.ok:
            usage = r.json().get("data", {}).get("usage", 0)
            return {
                "stream": "openrouter",
                "usage_units": usage,
                "estimated_aud": round(usage * 0.002, 2),
                "status": "ok",
            }
        return {"stream": "openrouter", "status": f"http_{r.status_code}"}
    except Exception as e:
        return {"stream": "openrouter", "status": f"error:{e}"}


def _check_cloudflare() -> dict:
    if not CLOUDFLARE_TOKEN or not CLOUDFLARE_ACCT:
        return {"stream": "cloudflare_workers", "status": "no_key"}
    try:
        r = requests.get(
            f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCT}/workers/services",
            headers={"Authorization": f"Bearer {CLOUDFLARE_TOKEN}"},
            timeout=10,
        )
        if r.ok:
            services = r.json().get("result", [])
            return {
                "stream": "cloudflare_workers",
                "worker_count": len(services),
                "estimated_aud": round(len(services) * 2.0, 2),
                "status": "ok",
            }
        return {"stream": "cloudflare_workers", "status": f"http_{r.status_code}"}
    except Exception as e:
        return {"stream": "cloudflare_workers", "status": f"error:{e}"}


# ── Swarm health ──────────────────────────────────────────────────────────────


def _swarm_health() -> dict:
    try:
        result = subprocess.run(
            ["pm2", "jlist"], capture_output=True, text=True, timeout=10
        )
        procs = json.loads(result.stdout) if result.returncode == 0 else []
        fm = [p for p in procs if p.get("name", "").startswith("fm-")]
        online = [p for p in fm if p.get("pm2_env", {}).get("status") == "online"]
        fail = len(fm) - len(online)
        return {
            "total": len(fm),
            "online": len(online),
            "fail": fail,
            "debt_ratio": round(fail / len(fm), 2) if fm else 0.0,
        }
    except Exception as e:
        return {"total": 0, "online": 0, "fail": 0, "debt_ratio": 1.0, "error": str(e)}


# ── Convergence computation ───────────────────────────────────────────────────


def _compute() -> dict:
    c = _db()
    now = datetime.now(tz=timezone.utc)
    hour = now.hour
    from_24h = (now - timedelta(hours=24)).isoformat()
    from_1h = (now - timedelta(hours=1)).isoformat()

    total_24h = c.execute(
        "SELECT COALESCE(SUM(value),0) FROM revenue_events WHERE created_at > ?",
        (from_24h,),
    ).fetchone()[0]
    total_1h = c.execute(
        "SELECT COALESCE(SUM(value),0) FROM revenue_events WHERE created_at > ?",
        (from_1h,),
    ).fetchone()[0]
    hourly_target_row = c.execute(
        "SELECT target_aud FROM hourly_targets WHERE hour=?", (hour,)
    ).fetchone()
    hourly_target = hourly_target_row[0] if hourly_target_row else 50.0

    total_24h = round(float(total_24h), 2)
    total_1h = round(float(total_1h), 2)
    velocity = round(total_1h / hourly_target, 2) if hourly_target else 0.0
    swarm = _swarm_health()
    projection = round(total_24h + velocity * (24 - hour) * hourly_target, 2)
    gap = round(DAILY_TARGET_AUD - projection, 2)

    c.execute(
        "INSERT INTO convergence_snapshots "
        "(total_24h,total_1h,velocity,technical_debt_ratio,swarm_health,projection_24h,gap_to_target)"
        " VALUES (?,?,?,?,?,?,?)",
        (
            total_24h,
            total_1h,
            velocity,
            swarm["debt_ratio"],
            swarm["online"],
            projection,
            gap,
        ),
    )
    c.commit()
    c.close()

    return {
        "timestamp": now.isoformat(),
        "total_24h_aud": total_24h,
        "total_1h_aud": total_1h,
        "hourly_velocity": velocity,
        "hourly_target_aud": hourly_target,
        "daily_target_aud": DAILY_TARGET_AUD,
        "projection_24h_aud": projection,
        "gap_to_target_aud": gap,
        "swarm": swarm,
        "streams": {
            "devto": _check_devto(),
            "openrouter": _check_openrouter(),
            "cloudflare": _check_cloudflare(),
        },
    }


# ── Background worker ─────────────────────────────────────────────────────────


def _worker():
    while True:
        try:
            result = _compute()
            logger.info(
                "24h=A$%.2f  velocity=%.2f  projection=A$%.2f  gap=A$%.2f",
                result["total_24h_aud"],
                result["hourly_velocity"],
                result["projection_24h_aud"],
                result["gap_to_target_aud"],
            )
            if 100.0 <= result["total_24h_aud"] < 105.0:
                _telegram(
                    f"🎯 <b>MILESTONE</b>\n24h revenue crossed <b>$100 AUD</b>\n"
                    f"Velocity: {result['hourly_velocity']}x\n"
                    f"Projection: ${result['projection_24h_aud']} AUD"
                )
            if result["gap_to_target_aud"] < 0:
                _telegram(
                    f"✅ <b>TARGET BREACHED</b>\n"
                    f"24h projection ${result['projection_24h_aud']} AUD exceeds daily target.\n"
                    f"Swarm: {result['swarm']['online']}/{result['swarm']['total']} online"
                )
            if result["swarm"]["debt_ratio"] > 0.5:
                _telegram(
                    f"⚠️ <b>SWARM DEBT RATIO {result['swarm']['debt_ratio']}</b>\n"
                    f"{result['swarm']['fail']} agents failing"
                )
        except Exception as e:
            logger.error("worker: %s", e)
        time.sleep(900)  # 15-min cycle


# ── Flask routes ──────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    return jsonify(
        {"status": "online", "agent": "fm-revenue-convergence", "port": PORT}
    )


@app.get("/status")
def status():
    if not _check_auth(request):
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify(_compute())


@app.post("/webhook/stripe")
def stripe_webhook():
    data = request.get_json(force=True, silent=True) or {}
    amount_cents = data.get("data", {}).get("object", {}).get("amount", 0)
    if amount_cents > 0:
        amount_aud = amount_cents / 100
        _log_event(1, "stripe_conversion", amount_aud, data)
        _telegram(f"💰 <b>STRIPE</b>  +${amount_aud:.2f} AUD\nLink: {STRIPE_LINK}")
    return jsonify({"received": True})


@app.get("/snapshots")
def snapshots():
    if not _check_auth(request):
        return jsonify({"error": "Unauthorized"}), 401
    c = _db()
    rows = c.execute(
        "SELECT snapshot_time,total_24h,total_1h,velocity,projection_24h,gap_to_target"
        " FROM convergence_snapshots ORDER BY snapshot_time DESC LIMIT 96"
    ).fetchall()
    c.close()
    return jsonify(
        {
            "snapshots": [
                {
                    "time": r[0],
                    "total_24h": r[1],
                    "total_1h": r[2],
                    "velocity": r[3],
                    "projection": r[4],
                    "gap": r[5],
                }
                for r in rows
            ]
        }
    )


@app.get("/streams")
def streams():
    if not _check_auth(request):
        return jsonify({"error": "Unauthorized"}), 401
    c = _db()
    rows = c.execute("SELECT * FROM revenue_streams").fetchall()
    c.close()
    return jsonify(
        {
            "streams": [
                {
                    "id": r[0],
                    "name": r[1],
                    "type": r[2],
                    "target_aud": r[3],
                    "hourly_aud": r[4],
                    "daily_aud": r[5],
                    "status": r[6],
                }
                for r in rows
            ]
        }
    )


if __name__ == "__main__":
    _ensure_schema()
    threading.Thread(target=_worker, daemon=True).start()
    logger.info("[fm-revenue-convergence] LIVE on :%d", PORT)
    serve(app, host="0.0.0.0", port=PORT, threads=4)
