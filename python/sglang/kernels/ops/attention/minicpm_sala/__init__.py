"""MiniCPM-SALA kernels."""

# Bound at import: a lazy lookup would return the same-named submodule whenever that
# submodule is imported first.
from sglang.kernels.ops.attention.minicpm_sala.get_block_table import get_block_table

__all__ = ["get_block_table"]
