"""
FractalMesh GitOps Runner
Webhook receiver for GitHub push events — validates token, logs to DB
Samuel James Hiotis | ABN 56 628 117 363
"""
import os
import sys
import sqlite3
import json
import hmac
import hashlib
from flask import Flask, request, jsonify

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT  = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
DB    = os.path.join(ROOT, "database", "sovereign.db")
PORT  = int(os.getenv("GITOPS_PORT", "8092"))
TOKEN = os.getenv("MAKE_MCP_TOKEN", "")

app = Flask(__name__)


def _log(event: str):
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS pulse_log "
        "(id INTEGER PRIMARY KEY, source TEXT, event TEXT, priority REAL, "
        "ts DATETIME DEFAULT CURRENT_TIMESTAMP)"
    )
    conn.execute(
        "INSERT INTO pulse_log (source, event, priority) VALUES (?,?,?)",
        ("fm-gitops-runner", event, 1.0)
    )
    conn.commit()
    conn.close()


@app.route("/webhook/gitops", methods=["POST"])
def gitops():
    auth = request.headers.get("Authorization", "")
    if TOKEN and auth != f"Bearer {TOKEN}":
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    ref  = data.get("ref", "unknown")
    repo = data.get("repository", {}).get("full_name", "unknown")
    _log(f"push:{repo}:{ref}")
    print(f"[GITOPS] Push event — {repo} @ {ref}")
    return jsonify({"status": "ok", "repo": repo, "ref": ref}), 200


@app.route("/health")
def health():
    return jsonify({"status": "online", "service": "fm-gitops-runner"})


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "database"), exist_ok=True)
    print(f"[fm-gitops-runner] GitOps webhook active on :{PORT}")
    app.run(host="127.0.0.1", port=PORT)
