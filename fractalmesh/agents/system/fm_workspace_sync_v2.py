#!/usr/bin/env python3
"""
fm_workspace_sync_v2.py — Workspace sync v2 with drift detection
Compares FRACTALMESH_HOME vs repo, reports file count delta, auto-creates missing dirs
FractalMesh Omega Titan | Samuel James Hiotis | ABN 56628117363
"""
import json
import time
import signal
import sys
import os
import hashlib
import subprocess
from datetime import datetime, timezone

ROOT      = os.getenv("FRACTALMESH_HOME", os.path.expanduser("~/fmsaas"))
REPO_ROOT = os.getenv("REPO_ROOT", os.path.expanduser("~/sglang/fractalmesh"))
INTERVAL  = int(os.getenv("WORKSPACE_SYNC_V2_INTERVAL", "120"))
DIRS      = ["database", "dist", "backups", "logs"]

running   = True


def _stop(*_a):
    global running
    running = False


signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT, _stop)


def _ensure_dirs() -> list:
    created = []
    for d in DIRS:
        path = os.path.join(ROOT, d)
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
            created.append(d)
    return created


def _file_count(directory: str) -> int:
    try:
        return sum(len(files) for _, _, files in os.walk(directory))
    except Exception:
        return -1


def _git_status() -> int:
    try:
        r = subprocess.run(["git", "-C", REPO_ROOT, "status", "--porcelain"],
                           capture_output=True, text=True, timeout=10)
        return len([l for l in r.stdout.splitlines() if l.strip()])
    except Exception:
        return -1


print("[fm-workspace-sync] online", flush=True)
while running:
    created       = _ensure_dirs()
    home_files    = _file_count(ROOT)
    repo_files    = _file_count(REPO_ROOT)
    changed_files = _git_status()

    print(json.dumps({
        "agent":         "fm-workspace-sync",
        "status":        "online",
        "home_files":    home_files,
        "repo_files":    repo_files,
        "git_changed":   changed_files,
        "dirs_created":  created,
        "ts":            datetime.now(timezone.utc).isoformat(),
    }), flush=True)
    time.sleep(INTERVAL)

sys.exit(0)
