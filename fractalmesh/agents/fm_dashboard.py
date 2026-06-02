"""
fm_dashboard.py — FractalMesh React/Flask API Gateway (port 8080)
Lightweight entry-point: health aggregate + agent-status JSON API.
Distinct from gateway.py (port 8000, AI proxy) and fm_omni_nexus.py (port 8095, FastAPI).
Samuel James Hiotis | ABN 56 628 117 363
"""

import json
import logging
import os
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS
from waitress import serve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fm-dashboard")

PORT = int(os.getenv("DASHBOARD_PORT", "8080"))
ROOT = os.getenv("FRACTALMESH_HOME", str(Path.home() / "fmsaas"))

app = Flask(__name__)
CORS(app)

# ── Agent port registry ───────────────────────────────────────────────────────
_AGENTS = {
    "fm-pod": 5058,
    "fm-geosignal": 5057,
    "fm-analytics": 5060,
    "fm-notes": 5061,
    "fm-tunnel": 5062,
    "unified-gateway": 8000,
    "fm-dashboard": 8080,
    "fm-omni-nexus": 8095,
    "fm-revenue-convergence": 7910,
    "fm-mistral": 7911,
}


def _probe(port: int, timeout: int = 2) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout):
            return True
    except Exception:
        return False


def _pm2_list() -> list:
    try:
        out = subprocess.check_output(
            ["pm2", "jlist"], timeout=10, stderr=subprocess.DEVNULL
        )
        return json.loads(out)
    except Exception:
        return []


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "agent": "fm-dashboard",
            "port": PORT,
            "ts": datetime.now(tz=timezone.utc).isoformat(),
        }
    )


@app.get("/api/status")
def status():
    procs = _pm2_list()
    pm2_map = {p.get("name"): p.get("pm2_env", {}).get("status") for p in procs}
    agents = []
    for name, port in _AGENTS.items():
        agents.append(
            {
                "name": name,
                "port": port,
                "pm2": pm2_map.get(name, "unknown"),
                "http": _probe(port),
            }
        )
    total = len(procs)
    online = sum(1 for p in procs if p.get("pm2_env", {}).get("status") == "online")
    return jsonify(
        {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "swarm": {"total": total, "online": online, "offline": total - online},
            "agents": agents,
        }
    )


@app.get("/api/agents")
def agents_list():
    procs = _pm2_list()
    return jsonify(
        {
            "count": len(procs),
            "agents": [
                {
                    "name": p.get("name"),
                    "status": p.get("pm2_env", {}).get("status"),
                    "restarts": p.get("pm2_env", {}).get("restart_time", 0),
                    "uptime": p.get("pm2_env", {}).get("pm_uptime"),
                    "cpu": p.get("monit", {}).get("cpu", 0),
                    "mem_mb": round(p.get("monit", {}).get("memory", 0) / 1_048_576, 1),
                }
                for p in procs
            ],
            "ts": datetime.now(tz=timezone.utc).isoformat(),
        }
    )


@app.get("/")
def index():
    return (
        "<html><head><title>FractalMesh Dashboard</title>"
        "<meta http-equiv='refresh' content='30'>"
        "<style>body{background:#0a0a0a;color:#00ff88;font-family:monospace;padding:24px}"
        "a{color:#00ffcc}</style></head><body>"
        "<h2>⬡ FractalMesh Dashboard — port 8080</h2>"
        "<p><a href='/api/status'>/api/status</a> &nbsp; "
        "<a href='/api/agents'>/api/agents</a> &nbsp; "
        "<a href='/health'>/health</a></p>"
        f"<p style='color:#666'>Samuel James Hiotis | ABN 56 628 117 363</p>"
        "</body></html>"
    )


if __name__ == "__main__":
    logger.info("[fm-dashboard] Listening on :%d", PORT)
    serve(app, host="0.0.0.0", port=PORT, threads=2)
