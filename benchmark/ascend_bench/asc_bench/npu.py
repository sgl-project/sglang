"""The only NPU-aware module: query node state via ``npu-smi``.

Every function returns ``None`` when the tool is missing or its output
cannot be parsed, so callers must degrade gracefully instead of crashing.
This module is the single relocation seam if the toolkit ever moves to a
different hardware-specific repository.
"""

from __future__ import annotations

import re
import subprocess

NPU_SMI = "npu-smi"

_MEM_TAIL_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s*$")
# Chip rows in `npu-smi info` carry a PCI bus id (e.g. 0000:C1:00.0); the
# NPU summary rows above them only carry hugepage counters.
_BUS_ID_RE = re.compile(r"^\d{4}:[0-9a-fA-F]{2}:\d{2}\.\d$")


def _run(cmd: list[str], timeout: float = 30.0) -> str | None:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def parse_hbm_used_mb(text: str) -> dict[int, int] | None:
    """Parse ``npu-smi info`` output into ``{device_index: used_mb}``.

    Only chip rows (identified by their bus id) are considered; the memory
    cell reads ``<aicore%> <used> / <total>``.  A real ``npu-smi info``
    sample from the target host should replace
    ``tests/fixtures/npu_smi_sample.txt`` and pin this parser.
    """
    used: dict[int, int] = {}
    for line in text.splitlines():
        if "|" not in line:
            continue
        cells = [cell.strip() for cell in line.split("|")]
        if not any(_BUS_ID_RE.match(cell) for cell in cells):
            continue
        first = cells[1].split() if len(cells) > 1 else []
        device = int(first[0]) if first and first[0].isdigit() else None
        if device is None:
            continue
        for cell in cells:
            match = _MEM_TAIL_RE.search(cell)
            if match:
                used.setdefault(device, int(match.group(1)))
                break
    return used or None


def hbm_used_mb() -> dict[int, int] | None:
    """Current HBM usage per device in MB, or None if unavailable."""
    text = _run([NPU_SMI, "info"])
    if text is None:
        return None
    return parse_hbm_used_mb(text)


def driver_version() -> str | None:
    """Driver/version banner from the npu-smi header, or None."""
    text = _run([NPU_SMI, "info"])
    if text is None:
        return None
    match = re.search(r"Version\s*:\s*(\S+)", text)
    return match.group(1) if match else None


def topology_summary(max_lines: int = 40) -> str | None:
    """Raw (truncated) npu-smi table for provenance records, or None."""
    text = _run([NPU_SMI, "info"])
    if text is None:
        return None
    return "\n".join(text.splitlines()[:max_lines])
