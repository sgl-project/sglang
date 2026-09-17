"""MiniCPM-SALA kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sglang.kernels.ops.minicpm_sala.get_block_table import get_block_table


def __getattr__(name: str) -> Any:
    if name == "get_block_table":
        from sglang.kernels.ops.minicpm_sala.get_block_table import get_block_table

        globals()[name] = get_block_table
        return get_block_table
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = ["get_block_table"]
