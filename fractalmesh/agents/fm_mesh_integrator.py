"""
FractalMesh Mesh Integrator
External webhook bridge — validates token, routes directives to internal bus
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import sys
import sqlite3
import json
import requests
from flask import Flask, request, jsonify

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT   = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB     = os.path.join(ROOT, "database", "sovereign.db")
PORT   = int(os.getenv("INTEGRATOR_PORT", "8090"))
TOKEN  = os.getenv("EXTERNAL_WEBHOOK_TOKEN", "")
BUS    = "http://127.0.0.1:5060"

app = Flask(__name__)


def _log(source: str, event: str):
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS pulse_log "
        "(id INTEGER PRIMARY KEY, source TEXT, event TEXT, priority REAL, "
        "ts DATETIME DEFAULT CURRENT_TIMESTAMP)"
    )
    conn.execute(
        "INSERT INTO pulse_log (source, event, priority) VALUES (?,?,?)",
        (source, event, 1.0)
    )
    conn.commit()
    conn.close()


def _forward_to_bus(agent: str, event: str):
    try:
        requests.post(
            f"{BUS}/",
            json={"agent": agent, "event": event, "priority": 1.0},
            timeout=3
        )
    except Exception:
        pass


@app.route("/webhook/llm_direct", methods=["POST"])
def llm_bridge():
    auth = request.headers.get("Authorization", "")
    if TOKEN and auth != f"Bearer {TOKEN}":
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    directive = data.get("directive", "")
    _log("fm-integrator", f"llm_direct:{directive[:80]}")
    _forward_to_bus("fm-integrator", directive[:80])
    print(f"[INTEGRATOR] Directive received: {directive[:80]}")
    return jsonify({"status": "ok", "directive": directive}), 200


@app.route("/health")
def health():
    return jsonify({"status": "online", "service": "fm-mesh-integrator"})


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "database"), exist_ok=True)
    print(f"[fm-mesh-integrator] Mesh integration layer active on :{PORT}")
    app.run(host="0.0.0.0", port=PORT)
