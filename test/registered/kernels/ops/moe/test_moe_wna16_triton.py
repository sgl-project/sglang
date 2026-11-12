import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import triton.language as tl

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-small")


def _launch_w4a16(b: torch.Tensor, scales, group_size: int) -> torch.Tensor:
    """One token routed to the last expert of b ([E, N, K // 2] uint8)."""
    from sglang.kernels.ops.moe import fused_moe_triton_kernels as kernels

    num_experts, n, half_k = b.shape
    expert = num_experts - 1
    scale = torch.tensor(scales, device="cuda", dtype=torch.bfloat16)
    c = torch.empty((1, n), device="cuda", dtype=torch.bfloat16)
    kernels.invoke_fused_moe_kernel(
        torch.ones((1, 2 * half_k), device="cuda", dtype=torch.bfloat16),
        b,
        None,
        c,
        None,
        scale.expand(num_experts, n, -1).contiguous(),
        None,
        torch.ones((1, 1), device="cuda", dtype=torch.float32),
        torch.full((1, 1), expert, device="cuda", dtype=torch.int32),
        torch.tensor([0] + [1] * 15, device="cuda", dtype=torch.int32),
        torch.tensor([expert], device="cuda", dtype=torch.int32),
        torch.tensor([16], device="cuda", dtype=torch.int32),
        False,
        1,
        {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": n, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1},
        compute_type=tl.bfloat16,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=True,
        per_channel_quant=False,
        block_shape=[0, group_size],
    )
    torch.cuda.synchronize()
    return c


def _cu_check(result):
    err = result[0] if isinstance(result, tuple) else result
    assert int(err) == 0, err
    return result[1] if isinstance(result, tuple) and len(result) == 2 else None


@contextmanager
def _tail_guarded_uint8(numel: int):
    import cuda.bindings.driver as cu

    # The caching allocator leaves slack after every tensor, so an over-read
    # only faults when the tensor ends a VMM mapping, as large expert weights do.
    prop = cu.CUmemAllocationProp()
    prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = torch.cuda.current_device()
    gran = _cu_check(
        cu.cuMemGetAllocationGranularity(
            prop, cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
        )
    )
    size = -(-numel // gran) * gran
    # One extra granule is reserved but never mapped.
    va = _cu_check(cu.cuMemAddressReserve(size + gran, 0, 0, 0))
    handle = _cu_check(cu.cuMemCreate(size, prop, 0))
    _cu_check(cu.cuMemMap(va, size, 0, handle, 0))
    access = cu.CUmemAccessDesc()
    access.location = prop.location
    access.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    _cu_check(cu.cuMemSetAccess(va, size, [access], 1))
    interface = {
        "shape": (numel,),
        "typestr": "|u1",
        "data": (int(va) + size - numel, False),
        "version": 3,
    }
    try:
        yield torch.as_tensor(
            SimpleNamespace(__cuda_array_interface__=interface), device="cuda"
        )
    finally:
        torch.cuda.synchronize()
        _cu_check(cu.cuMemUnmap(va, size))
        _cu_check(cu.cuMemRelease(handle))
        _cu_check(cu.cuMemAddressFree(va, size + gran))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_packed_k_tail_stays_inside_weight_tensor():
    """K not a multiple of BLOCK_SIZE_K must not read past the packed int4 rows.

    The unmasked tail load overran the last expert by 16 bytes per row at K=160.
    """
    num_experts, n, k = 2, 64, 160
    with _tail_guarded_uint8(num_experts * n * k // 2) as flat:
        # 0x99 holds two int4 values of 9; symmetric W4A16 subtracts 8, leaving 1.
        b = flat.view(num_experts, n, k // 2).fill_(0x99)
        c = _launch_w4a16(b, scales=[1.0, 2.0], group_size=128)
    # Groups cover K [0, 128) with scale 1 and [128, 160) with scale 2.
    torch.testing.assert_close(c, torch.full_like(c, 128 * 1 + 32 * 2), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_w4a16_rejects_incomplete_scale_grid():
    """A scale tensor without K's tail group must fail before launch, not read stale scales."""
    b = torch.full((1, 64, 80), 0x99, device="cuda", dtype=torch.uint8)
    with pytest.raises(ValueError, match="scale groups"):
        _launch_w4a16(b, scales=[1.0], group_size=128)


if __name__ == "__main__":
    args = ["-x" if arg == "-f" else arg for arg in sys.argv[1:]]
    sys.exit(pytest.main([__file__, *args]))
