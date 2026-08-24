#!/usr/bin/env python3
"""
health_app.py — Flask health & status API for FractalMesh stack
Listens on APP_PORT (default 5057)
Samuel James Hiotis | ABN 56628117363
"""
import os, subprocess
from flask import Flask, jsonify
from datetime import datetime, timezone

app = Flask("fractalmesh_health")


def safe_pm2_list() -> str:
    try:
        return subprocess.check_output(["pm2", "jlist"], text=True, timeout=5)
    except Exception:
        return "[]"


@app.get("/health")
def health():
    return jsonify({
        "ok":      True,
        "service": "fractalmesh-health",
        "ts":      datetime.now(timezone.utc).isoformat(),
        "env":     os.getenv("APP_ENV", "production")
    })


@app.get("/status")
def status():
    return jsonify({
        "ok":      True,
        "service": "fractalmesh-health",
        "ts":      datetime.now(timezone.utc).isoformat(),
        "pm2":     safe_pm2_list()
    })


if __name__ == "__main__":
    port = int(os.getenv("APP_PORT", "5057"))
    app.run(host="0.0.0.0", port=port, debug=False)
