"""cast_store_kv_kernel (fp8 MHA pool fed bf16/fp16 K/V) vs the eager div + cast + scatter."""

import sys

import pytest
import torch

from sglang.srt.mem_cache.memory_pool import cast_store_kv_kernel
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-small")

DEVICE = "cuda"


@pytest.mark.parametrize("cache_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("src_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("heads,head_dim", [(4, 256), (1, 512), (8, 128)])
@pytest.mark.parametrize("scaled", [False, True])
@pytest.mark.parametrize("loc_dtype", [torch.int32, torch.int64])
def test_cast_store_matches_eager(
    cache_dtype, src_dtype, heads, head_dim, scaled, loc_dtype
):
    torch.manual_seed(0)
    n, slots = 777, 64 * 20
    k = torch.randn(n, heads, head_dim, device=DEVICE, dtype=src_dtype) * 3
    v = torch.randn(n, heads, head_dim, device=DEVICE, dtype=src_dtype) * 3
    # strided source rows, as when K/V are views into a fused qkv projection
    k = torch.cat([k, k], dim=-1)[..., :head_dim]
    k_scale = torch.tensor([0.37], device=DEVICE) if scaled else None
    v_scale = torch.tensor([0.21], device=DEVICE) if scaled else None
    loc = (torch.randperm(slots - 1, device=DEVICE)[:n] + 1).to(loc_dtype)
    loc[0] = 0  # reserved padding slot: must stay untouched

    fill = torch.full((slots, heads * head_dim), 0x55, dtype=torch.uint8, device=DEVICE)
    k_ref, v_ref = fill.clone().view(cache_dtype), fill.clone().view(cache_dtype)
    k_new, v_new = fill.clone().view(cache_dtype), fill.clone().view(cache_dtype)

    ks, vs = k.clone(), v.clone()
    if scaled:
        ks.div_(k_scale)
        vs.div_(v_scale)
    keep = loc != 0
    k_ref[loc[keep].long()] = ks[keep].to(cache_dtype).flatten(1)
    v_ref[loc[keep].long()] = vs[keep].to(cache_dtype).flatten(1)

    one = torch.ones(1, dtype=torch.float32, device=DEVICE)
    cast_store_kv_kernel[(n, heads)](
        k,
        v,
        k_new,
        v_new,
        loc,
        k_scale if scaled else one,
        v_scale if scaled else one,
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        k_new.stride(0),
        v_new.stride(0),
        slots,
        H=heads,
        DK=head_dim,
        DV=head_dim,
        HAS_SCALE=scaled,
        num_warps=4,
    )
    torch.cuda.synchronize()
    assert torch.equal(k_new.view(torch.uint8), k_ref.view(torch.uint8))
    assert torch.equal(v_new.view(torch.uint8), v_ref.view(torch.uint8))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
