"""cast_store_kv_kernel (fp8 MHA pool fed bf16/fp16 K/V) vs the eager div + cast + scatter."""

import sys

import pytest
import torch

from sglang.srt.mem_cache.memory_pool import (
    _cast_store_scale_arg,
    _cast_store_scale_mode,
    cast_store_kv_kernel,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-small")

DEVICE = "cuda"
# the pool takes the kernel for e4m3 only: an e5m2 eager cast overflows to inf,
# Triton's saturates
FP8_DTYPES = [torch.float8_e4m3fn]
SRC_DTYPES = [torch.bfloat16, torch.float16]


def _scale(kind, value, src_dtype):
    """The kinds of k/v scale ``x.div_(scale)`` treats differently."""
    if kind == "none":
        return None
    if kind == "float":
        return value
    if kind == "cpu0d":
        return torch.tensor(value)
    if kind == "cuda0d":  # what a checkpoint's KV scale is
        return torch.tensor(value, device=DEVICE)
    assert kind == "cuda0d_src"
    return torch.tensor(value, device=DEVICE, dtype=src_dtype)


def _eager(k, v, k_scale, v_scale, loc, cache_dtype, slots):
    width_k, width_v = k[0].numel(), v[0].numel()
    k_ref = torch.full((slots, width_k), 0x55, dtype=torch.uint8, device=DEVICE)
    v_ref = torch.full((slots, width_v), 0x55, dtype=torch.uint8, device=DEVICE)
    ks, vs = k.clone(), v.clone()
    if k_scale is not None:
        ks.div_(k_scale)
    if v_scale is not None:
        vs.div_(v_scale)
    keep = loc != 0
    k_ref.view(cache_dtype)[loc[keep].long()] = ks[keep].to(cache_dtype).flatten(1)
    v_ref.view(cache_dtype)[loc[keep].long()] = vs[keep].to(cache_dtype).flatten(1)
    return k_ref, v_ref


def _fused(k, v, k_scale, v_scale, loc, cache_dtype, slots):
    n, heads, dk = k.shape
    dv = v.shape[-1]
    k_new = torch.full((slots, heads * dk), 0x55, dtype=torch.uint8, device=DEVICE)
    v_new = torch.full((slots, heads * dv), 0x55, dtype=torch.uint8, device=DEVICE)
    k_mode, v_mode = _cast_store_scale_mode(k_scale), _cast_store_scale_mode(v_scale)
    assert k_mode is not None and v_mode is not None
    cast_store_kv_kernel[(n, heads)](
        k,
        v,
        k_new.view(cache_dtype),
        v_new.view(cache_dtype),
        loc,
        _cast_store_scale_arg(k_scale, k_mode),
        _cast_store_scale_arg(v_scale, v_mode),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        k_new.stride(0),
        v_new.stride(0),
        slots,
        H=heads,
        DK=dk,
        DV=dv,
        K_SCALE_MODE=k_mode,
        V_SCALE_MODE=v_mode,
        num_warps=4,
    )
    torch.cuda.synchronize()
    return k_new, v_new


def _assert_same(got, ref, what):
    diff = (got != ref).nonzero()
    assert diff.numel() == 0, (
        f"{what}: {diff.shape[0]} bytes differ, first at {diff[0].tolist()}: "
        f"fused {got[tuple(diff[0])].item():#x} eager {ref[tuple(diff[0])].item():#x}"
    )


@pytest.mark.parametrize("cache_dtype", FP8_DTYPES)
@pytest.mark.parametrize("src_dtype", SRC_DTYPES)
@pytest.mark.parametrize("heads,head_dim", [(4, 256), (1, 512), (8, 128)])
@pytest.mark.parametrize(
    "scale_kind", ["none", "float", "cpu0d", "cuda0d", "cuda0d_src"]
)
@pytest.mark.parametrize("loc_dtype", [torch.int32, torch.int64])
def test_cast_store_matches_eager(
    cache_dtype, src_dtype, heads, head_dim, scale_kind, loc_dtype
):
    torch.manual_seed(0)
    n, slots = 777, 64 * 20
    k = torch.randn(n, heads, head_dim, device=DEVICE, dtype=src_dtype) * 3
    v = torch.randn(n, heads, head_dim, device=DEVICE, dtype=src_dtype) * 3
    # strided source rows, as when K/V are views into a fused qkv projection
    k = torch.cat([k, k], dim=-1)[..., :head_dim]
    k_scale = _scale(scale_kind, 0.37, src_dtype)
    v_scale = _scale(scale_kind, 0.21, src_dtype)
    loc = (torch.randperm(slots - 1, device=DEVICE)[:n] + 1).to(loc_dtype)
    loc[0] = 0  # reserved padding slot: must stay untouched

    k_ref, v_ref = _eager(k, v, k_scale, v_scale, loc, cache_dtype, slots)
    k_new, v_new = _fused(k, v, k_scale, v_scale, loc, cache_dtype, slots)
    _assert_same(k_new, k_ref, "k")
    _assert_same(v_new, v_ref, "v")


@pytest.mark.parametrize("src_dtype", SRC_DTYPES)
@pytest.mark.parametrize("k_kind,v_kind", [("cuda0d", "none"), ("float", "cuda0d")])
def test_cast_store_mixed_scale_kinds(src_dtype, k_kind, v_kind):
    torch.manual_seed(1)
    n, slots = 65, 256
    k = torch.randn(n, 2, 128, device=DEVICE, dtype=src_dtype) * 3
    v = torch.randn(n, 2, 128, device=DEVICE, dtype=src_dtype) * 3
    k_scale, v_scale = _scale(k_kind, 0.37, src_dtype), _scale(v_kind, 0.21, src_dtype)
    loc = torch.arange(1, n + 1, device=DEVICE)
    args = (k, v, k_scale, v_scale, loc, torch.float8_e4m3fn, slots)
    for got, ref, what in zip(_fused(*args), _eager(*args), "kv"):
        _assert_same(got, ref, what)


def test_cast_store_rounds_a_0d_tensor_scale_like_eager():
    """BUG REGRESSION. ``x.div_(scale)`` with a 0-dim CUDA scale rounds the scale
    to x's dtype first. Dividing by the fp32 scale stored 3.5 for this key where
    eager stores 3.75."""
    k = torch.full((1, 1, 128), 1.34375, device=DEVICE, dtype=torch.bfloat16)
    scale = torch.tensor(0.37, device=DEVICE)
    loc = torch.tensor([1], device=DEVICE)
    k_new, _ = _fused(k, k.clone(), scale, scale, loc, torch.float8_e4m3fn, 4)
    stored = k_new.view(torch.float8_e4m3fn)[1].float()
    assert torch.all(stored == 3.75), stored[:4]


@pytest.mark.parametrize("cache_dtype", FP8_DTYPES)
@pytest.mark.parametrize("src_dtype", SRC_DTYPES)
def test_cast_store_out_of_range_and_special_values(cache_dtype, src_dtype):
    """Overflow, infinities, NaN and subnormals convert as the eager cast does."""
    limit = torch.finfo(cache_dtype).max
    edge = [0.0, -0.0, 1e-3, 2e-3, 0.0137, 0.0195, limit, limit * 1.03, limit * 1.07]
    edge += [limit * 1.1, limit * 2, 6e4, float("inf"), float("nan")]
    values = torch.tensor(edge + [-x for x in edge], device=DEVICE, dtype=src_dtype)
    n = values.numel()
    k = values[:, None, None].expand(n, 1, 128).contiguous()
    loc = torch.arange(1, n + 1, device=DEVICE)
    args = (k, k.clone(), None, None, loc, cache_dtype, n + 1)
    for got, ref, what in zip(_fused(*args), _eager(*args), "kv"):
        _assert_same(got, ref, what)


def test_a_1d_scale_keeps_the_eager_path():
    # a 1-D scale promotes the eager division to fp32: neither kernel mode
    assert _cast_store_scale_mode(torch.tensor([0.37], device=DEVICE)) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
