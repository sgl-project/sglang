"""Run provenance capture: everything needed to reproduce a baseline."""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from asc_bench import npu

_ENV_KEY_RE = re.compile(r"(ASCEND|HCCL|SGLANG|PYTORCH_NPU|TASK_QUEUE|LD_LIBRARY_PATH)")


def _py_version(python: str, statement: str) -> str | None:
    try:
        proc = subprocess.run(
            [python, "-c", statement],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    out = proc.stdout.strip()
    return out or None if proc.returncode == 0 else None


def _pkg_version(python: str, package: str) -> str | None:
    return _py_version(
        python, f"import importlib.metadata as m; print(m.version('{package}'))"
    )


def _git_sha() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() or None if proc.returncode == 0 else None


def _cann_version() -> str | None:
    root = Path("/usr/local/Ascend/ascend-toolkit/latest")
    for candidate in (
        root / "version.cfg",
        root / "aarch64-linux/ascend_toolkit_install.info",
    ):
        try:
            text = candidate.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        match = re.search(r"version[=:=]\s*([\w.\-]+)", text)
        if match:
            return match.group(1)
    return None


def capture(python: str | None = None) -> dict[str, Any]:
    """Collect versions, git state, NPU state, and Ascend-relevant env."""
    python = python or sys.executable
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "sglang_version": _pkg_version(python, "sglang"),
        "torch_version": _pkg_version(python, "torch"),
        "torch_npu_version": _pkg_version(python, "torch_npu"),
        "git_sha": _git_sha(),
        "cann_version": _cann_version(),
        "npu_driver_version": npu.driver_version(),
        "npu_topology": npu.topology_summary(),
        "env": {
            key: value
            for key, value in sorted(os.environ.items())
            if _ENV_KEY_RE.search(key)
        },
    }
