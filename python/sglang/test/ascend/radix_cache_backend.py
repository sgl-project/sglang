"""Test-only plugin for the Ascend --radix-cache-backend regression.

The test exposes ``register`` through a temporary ``sglang.srt.plugins``
entry point so each scheduler process registers the factory independently.
This backend reuses the real unified cache and records completed operations;
it does not replace KV tensors, attention kernels, or cache matching logic.
"""

import json
import os
from dataclasses import replace
from pathlib import Path

from sglang.srt.mem_cache.registry import register_radix_cache_backend
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

BACKEND_NAME = "ascend_test_radix"
EVENT_DIR_ENV = "SGLANG_TEST_RADIX_EVENT_DIR"


def _record(event, **fields):
    path = Path(os.environ[EVENT_DIR_ENV]) / f"{os.getpid()}.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": event, "pid": os.getpid(), **fields}) + "\n")


class ObservedNpuRadixCache(UnifiedRadixCache):
    def reset(self):
        super().reset()
        _record("reset")

    def match_prefix(self, params):
        result = super().match_prefix(params)
        if len(result.device_indices):
            _record(
                "hit",
                tokens=len(result.device_indices),
                device=str(result.device_indices.device),
            )
        return result

    def insert(self, params):
        result = super().insert(params)
        _record("insert", tokens=len(params.key))
        return result


def _factory(ctx):
    # This fixture deliberately targets Llama's full-attention KV cache.
    # Hybrid models need their SWA/Mamba components configured separately.
    if ctx.is_hybrid_swa or ctx.is_hybrid_ssm or ctx.enable_hierarchical_cache:
        raise ValueError("ascend_test_radix requires full attention without HiCache")
    allocator = ctx.params.token_to_kv_pool_allocator
    if str(allocator.device).split(":")[0] != "npu":
        raise ValueError(f"ascend_test_radix requires NPU, got {allocator.device}")
    cache = ObservedNpuRadixCache(
        replace(ctx.params, tree_components=(ComponentType.FULL,))
    )
    _record("factory", device=str(allocator.device), tp_rank=ctx.tp_rank)
    return cache


def register():
    register_radix_cache_backend(BACKEND_NAME, _factory)
    # Plugin loading can precede logging initialization in worker processes.
    _record("register", backend=BACKEND_NAME)
