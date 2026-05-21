"""Lightweight VLM profiling — gated by VLM_PROFILE=1 env var."""

import json
import os
import time

ENABLED = os.environ.get("VLM_PROFILE", "") == "1"

_file = None


def _ensure_file():
    global _file
    if _file is None:
        _file = open("/tmp/vlm_profile.jsonl", "a")


def log_stage(stage: str, **kwargs):
    if not ENABLED:
        return
    _ensure_file()
    record = {"ts": time.monotonic(), "stage": stage, **kwargs}
    _file.write(json.dumps(record) + "\n")
    _file.flush()
