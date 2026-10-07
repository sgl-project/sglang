"""Backend-neutral Metal commit for K/V deltas produced by MLX graphs."""

from sglang.kernels.ops.kvcache._deferred_kv_commit_metal_jit import (
    commit_deferred_kv,
    verify_deferred_kv_commit,
)

__all__ = ["commit_deferred_kv", "verify_deferred_kv_commit"]
