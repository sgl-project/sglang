"""Implementation home for kernels produced by Kernel Design Agents.

Runtime code should keep importing kernels through :mod:`sglang.kernels.ops`.
This package owns generated implementations and provenance metadata, while the
operator facades own registration, dispatch, and stable public exports.
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def _cuda_source(name: str) -> str:
    """Return an absolute path to a KDA-owned JIT CUDA source file."""
    return str(_ROOT / "csrc" / name)


__all__: list[str] = []
