# SPDX-License-Identifier: Apache-2.0
"""The opt-in codec preserves CPU/default behavior without importing Triton."""

import builtins

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.mamba_checkpoint_pool import Int8CheckpointStore
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("enabled", [False, True])
def test_cpu_fallback_without_cuda_kernel_import(monkeypatch, enabled):
    original = builtins.__import__

    def import_guard(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sglang.kernels.ops.kvcache.int8_mamba_checkpoint" or (
            name == "sglang.kernels.ops.kvcache" and "int8_mamba_checkpoint" in fromlist
        ):
            raise AssertionError("CPU fallback must not import the CUDA codec")
        return original(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_guard)
    with envs.SGLANG_EXPERIMENTAL_FUSED_INT8_CHECKPOINT.override(enabled):
        store = Int8CheckpointStore(
            num_layers=2,
            num_slots=4,
            num_heads=2,
            head_v_dim=8,
            head_k_dim=8,
            device="cpu",
        )
    active = torch.randn(2, 5, 2, 8, 8, dtype=torch.bfloat16)
    src = torch.tensor([1, 3])
    dst = torch.tensor([2, 1])
    q, scale = store.quantize(active[:, src])
    store.store_from_pool(active, src, dst)
    torch.testing.assert_close(store.qdata[:, dst], q, rtol=0, atol=0)
    torch.testing.assert_close(store.scale[:, dst], scale, rtol=0, atol=0)
    output = torch.full_like(active, 5)
    store.copy_to_pool(output, dst, src)
    torch.testing.assert_close(
        output[:, src], store.dequantize(q, scale, output.dtype), rtol=0, atol=0
    )
    assert store._fused_store_calls == 0
    assert store._fused_restore_calls == 0


if __name__ == "__main__":
    import sys

    args = [arg for arg in sys.argv[1:] if arg != "-f"]
    raise SystemExit(pytest.main([__file__, "-v", *args]))
