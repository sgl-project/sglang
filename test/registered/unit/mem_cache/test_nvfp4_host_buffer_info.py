"""Persistent NVFP4 scale descriptors must not silently drop a native layout."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.device_pool_info import KVScaleLayout
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def pool_fixture(recipe, *, linear, native):
    pool = object.__new__(MHATokenToKVPool)
    pool.quant_method = SimpleNamespace(name=recipe)
    pool.k_scale_buffer = (
        [torch.zeros(64, 2, 4, dtype=torch.uint8) for _ in range(2)] if linear else None
    )
    pool.v_scale_buffer = (
        [torch.ones_like(buffer) for buffer in pool.k_scale_buffer] if linear else None
    )
    pool.native_k_scale_buffer = (
        [torch.full((4, 2, 16, 4), 2, dtype=torch.uint8) for _ in range(2)]
        if native
        else None
    )
    pool.native_v_scale_buffer = (
        [torch.full_like(buffer, 3) for buffer in pool.native_k_scale_buffer]
        if native
        else None
    )
    return pool


@pytest.mark.parametrize("native", [False, True])
def test_nvfp4_descriptor_retains_every_persistent_scale_allocation(native):
    pool = pool_fixture("nvfp4", linear=True, native=native)
    scales = pool._mha_scale_buffers()
    assert scales.layout == KVScaleLayout.NVFP4
    for exported, original in (
        (scales.k, pool.k_scale_buffer),
        (scales.v, pool.v_scale_buffer),
        (scales.native_k, pool.native_k_scale_buffer or ()),
        (scales.native_v, pool.native_v_scale_buffer or ()),
    ):
        assert len(exported) == len(original)
        assert all(actual is expected for actual, expected in zip(exported, original))


def test_nvfp4_native_only_cannot_silently_export_as_plain_uint8():
    pool = pool_fixture("nvfp4", linear=False, native=True)
    with pytest.raises(NotImplementedError, match="(?i)(linear|native)"):
        pool._mha_scale_buffers()


def test_unquantized_pool_without_scales_preserves_none():
    pool = pool_fixture("unquantized", linear=False, native=False)
    assert pool._mha_scale_buffers() is None


def test_fp4_mx_keeps_existing_token_layout_without_nvfp4_dispatch():
    pool = pool_fixture("fp4_mx_block16", linear=True, native=False)
    scales = pool._mha_scale_buffers()
    assert scales.layout == KVScaleLayout.TOKEN
    assert scales.native_k == scales.native_v == ()
    assert all(
        actual is expected for actual, expected in zip(scales.k, pool.k_scale_buffer)
    )
