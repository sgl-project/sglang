#!/usr/bin/env python3
"""
git_oracle.py — Git repository state watcher
Reports branch, last commit, dirty files, and ahead/behind vs origin
FractalMesh Omega Titan | Samuel James Hiotis | ABN 56628117363
"""
import json
import time
import signal
import sys
import os
import subprocess
from datetime import datetime, timezone

REPO_ROOT = os.getenv("REPO_ROOT", os.path.expanduser("~/sglang"))
running   = True


def _stop(*_a):
    global running
    running = False


signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT, _stop)


def _git(args: list, cwd: str = REPO_ROOT) -> str:
    try:
        r = subprocess.run(["git", "-C", cwd] + args,
                           capture_output=True, text=True, timeout=15)
        return r.stdout.strip()
    except Exception:
        return ""


def _repo_state() -> dict:
    branch   = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    commit   = _git(["log", "-1", "--format=%H %s"])[:80]
    dirty    = _git(["status", "--porcelain"])
    ahead    = _git(["rev-list", "--count", "@{u}..HEAD"])
    behind   = _git(["rev-list", "--count", "HEAD..@{u}"])
    untracked = sum(1 for l in dirty.splitlines() if l.startswith("??"))
    modified  = sum(1 for l in dirty.splitlines() if not l.startswith("??"))
    return {
        "branch":         branch or "unknown",
        "last_commit":    commit or "none",
        "untracked_files": untracked,
        "modified_files":  modified,
        "ahead":           int(ahead) if ahead.isdigit() else -1,
        "behind":          int(behind) if behind.isdigit() else -1,
        "clean":           dirty == "",
    }


print("[git-oracle] online", flush=True)
while running:
    state = _repo_state()
    print(json.dumps({
        "agent":  "git-oracle",
        "status": "online",
        "repo":   REPO_ROOT,
        **state,
        "ts":     datetime.now(timezone.utc).isoformat(),
    }), flush=True)
    time.sleep(90)

sys.exit(0)
