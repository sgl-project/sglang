"""Persist OpAuto winners / demotions under the JIT cache dir."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def default_state_path() -> Path:
    from sglang.srt.environ import envs

    override = envs.SGLANG_OPAUTO_STATE.get()
    if override:
        return Path(os.path.expanduser(override))
    cache = envs.SGLANG_JIT_CACHE_DIR.get() or "~/.cache/sglang/jit"
    return Path(os.path.expanduser(cache)) / "opauto.json"


def load_state_file(path: Optional[Path] = None) -> dict[str, Any]:
    p = path or default_state_path()
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("OpAuto: failed to load %s: %s", p, e)
        return {}


def save_state_file(data: dict[str, Any], path: Optional[Path] = None) -> Path:
    p = path or default_state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(p)
    return p
