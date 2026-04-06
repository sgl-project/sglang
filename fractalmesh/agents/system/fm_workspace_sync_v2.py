#!/usr/bin/env python3
"""
fm_workspace_sync_v2.py — Workspace sync heartbeat (stub, v2)
FractalMesh Omega Titan | Samuel James Hiotis | ABN 56628117363
"""
import json, time, signal, sys
from datetime import datetime, timezone

running = True

def stop(*_a):
    global running
    running = False

signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)

print("[fm-workspace-sync] online", flush=True)
while running:
    print(json.dumps({
        "agent": "fm-workspace-sync",
        "status": "online",
        "message": "Workspace sync heartbeat",
        "ts": datetime.now(timezone.utc).isoformat()
    }), flush=True)
    time.sleep(120)
sys.exit(0)
