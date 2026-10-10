from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
)
from sglang.srt.environ import envs
from sglang.srt.mem_cache import mamba_slot_fused
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _make_pool(envelope_layout, dtype=torch.bfloat16):
    shape = Mamba2StateShape.create(
        tp_world_size=1,
        intermediate_size=8,
        n_groups=1,
        num_heads=2,
        head_dim=4,
        state_size=4,
        conv_kernel=4,
    )
    shape = replace(shape, conv=[(3, 64), (3, 128), (3, 1152)])
    layers = [0, 1, 2]
    pool = MambaPool(
        size=15,
        spec_state_size=0,
        cache_params=Mamba2CacheParams(
            shape=shape,
            dtype=Mamba2StateDType(conv=dtype, temporal=torch.float32),
            layers=layers,
        ),
        mamba_layer_ids=layers,
        envelope_layout=envelope_layout,
        device="cuda",
    )
    for state in [*pool.mamba_cache.conv, pool.mamba_cache.temporal]:
        state.normal_()
    return pool


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize(
    "envelope_layout", [False, True], ids=["layer-first", "page-first"]
)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("operation", ["clear", "copy"])
def test_pool_slot_ops_use_fused_kernel(envelope_layout, index_dtype, operation):
    pool = _make_pool(envelope_layout)
    states = [*pool.mamba_cache.conv, pool.mamba_cache.temporal]
    expected = [state.clone() for state in states]
    src = torch.tensor([1, 7, 5], device="cuda", dtype=index_dtype)
    dst = torch.tensor([14, 2, 10], device="cuda", dtype=index_dtype)
    if envelope_layout:
        # Use the pool's actual page-major allocation, including shared temporal
        # storage. Comparing every state also catches writes to untouched slots.
        assert all(not c.is_contiguous() for c in pool.mamba_cache.conv)
        assert all(c.stride(1) > c.stride(0) for c in pool.mamba_cache.conv)

    with (
        envs.SGLANG_DISABLE_FUSED_MAMBA_SLOT_OPS.override(False),
        patch.object(
            mamba_slot_fused,
            "fused_clear_conv_slots",
            wraps=mamba_slot_fused.fused_clear_conv_slots,
        ) as clear,
        patch.object(
            mamba_slot_fused,
            "fused_copy_conv_slots",
            wraps=mamba_slot_fused.fused_copy_conv_slots,
        ) as copy,
    ):
        if operation == "clear":
            pool.clear_slots(dst)
            clear.assert_called_once()
            copy.assert_not_called()
            for state in expected:
                state[:, dst] = 0
        else:
            pool.copy_from(src, dst)
            copy.assert_called_once()
            clear.assert_not_called()
            for state in expected:
                state[:, dst] = state[:, src]

    for actual, reference in zip(states, expected, strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize("reason", ["feature_stride", "dtype", "disabled"])
def test_pool_slot_ops_keep_required_fallbacks(reason):
    # Defer descriptor caching until after the test changes the layout.
    with patch.object(MambaPool, "_warmup_fused_copy_slot_kernel"):
        pool = _make_pool(True, torch.float16 if reason == "dtype" else torch.bfloat16)
    if reason == "feature_stride":
        pool.mamba_cache.conv[0] = pool.mamba_cache.conv[0].transpose(-1, -2)
    states = [*pool.mamba_cache.conv, pool.mamba_cache.temporal]
    expected = [state.clone() for state in states]
    src = torch.tensor([1, 7], device="cuda")
    dst = torch.tensor([2, 10], device="cuda")
    with (
        envs.SGLANG_DISABLE_FUSED_MAMBA_SLOT_OPS.override(reason == "disabled"),
        patch.object(mamba_slot_fused, "fused_clear_conv_slots") as clear,
        patch.object(mamba_slot_fused, "fused_copy_conv_slots") as copy,
    ):
        pool.copy_from(src, dst)
        pool.clear_slots(src)
        copy.assert_not_called()
        clear.assert_not_called()
    for actual, reference in zip(states, expected, strict=True):
        reference[:, dst] = reference[:, src]
        reference[:, src] = 0
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
