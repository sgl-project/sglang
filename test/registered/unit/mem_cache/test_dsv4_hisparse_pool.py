"""HiSparseC4DevicePool construction: FP8 works, MXFP4 is rejected.

Regression: the DSV4 pool factory passes ``use_mxfp4`` to every pool class,
but HiSparseC4DevicePool's signature never gained the parameter — enabling
HiSparse with the default FP8 KV cache crashed with
``unexpected keyword argument 'use_mxfp4'``, and its positional
``super().__init__`` shifted ``start_layer`` into the parent's
``use_mxfp4`` slot.
"""

from __future__ import annotations

import pytest
import torch

from sglang.srt.mem_cache.deepseek_v4_memory_pool import HiSparseC4DevicePool
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_SIZE = 256
_PAGE_SIZE = 256
_LAYER_NUM = 4
_NOPE = 448
_ROPE = 64


def _make_pool(use_mxfp4: bool) -> HiSparseC4DevicePool:
    return HiSparseC4DevicePool(
        size=_SIZE,
        page_size=_PAGE_SIZE,
        dtype=torch.float8_e4m3fn,
        qk_nope_head_dim=_NOPE,
        qk_rope_head_dim=_ROPE,
        layer_num=_LAYER_NUM,
        device="cuda",
        enable_memory_saver=False,
        use_mxfp4=use_mxfp4,
    )


def test_hisparse_pool_fp8_init():
    """FP8 (use_mxfp4=False) constructs without error."""
    pool = _make_pool(use_mxfp4=False)
    assert pool.dsv4_kv_cache_store_mxfp4 is False
    assert len(pool.kv_buffer) == _LAYER_NUM


def test_hisparse_pool_rejects_mxfp4():
    """MXFP4 + HiSparse is rejected at construction, not silently mis-routed."""
    with pytest.raises(ValueError, match="not supported with HiSparse"):
        _make_pool(use_mxfp4=True)


def test_hisparse_pool_positional_super_no_shift():
    """Positional super().__init__ args must not land in the parent's
    use_mxfp4 slot: start_layer/end_layer are preserved as layer ranges."""
    pool = HiSparseC4DevicePool(
        _SIZE,
        _PAGE_SIZE,
        torch.float8_e4m3fn,
        _NOPE,
        _ROPE,
        _LAYER_NUM,
        "cuda",
        False,
        False,  # use_mxfp4 (positional, as the factory would not call it)
        1,
        3,
    )
    # With the pre-fix signature this call would raise (unexpected keyword)
    # or, when called via the factory, shift start_layer into use_mxfp4.
    assert pool.dsv4_kv_cache_store_mxfp4 is False
