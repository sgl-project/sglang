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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("path", ["fused", "runner_core"])
@pytest.mark.parametrize("inplace", [False, True])
def test_w2_block_shape_reaches_both_runner_paths(path, inplace):
    """Both Triton runner entry points must use w2_block_shape for the down GEMM."""
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.triton import (
        TritonMoeQuantInfo,
        TritonRunnerCore,
        fused_experts_none_to_triton,
        pre_permute_standard_to_triton,
    )
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.moe.utils import MoeRunnerBackend
    from sglang.srt.runtime_context import get_context, get_flags
    from sglang.test.layer_ut_utils import init_single_process_dist

    init_single_process_dist(master_port=29641)
    hidden, intermediate = 128, 160
    x = torch.full((1, hidden), 1 / hidden, dtype=torch.float16, device="cuda")
    w13 = torch.full(
        (1, 2 * intermediate, hidden // 2), 0x99, dtype=torch.uint8, device="cuda"
    )
    w2 = torch.full(
        (1, hidden, intermediate // 2), 0x99, dtype=torch.uint8, device="cuda"
    )
    w13_scale = torch.ones((1, 2 * intermediate, 1), dtype=x.dtype, device=x.device)
    w2_scale = (
        torch.arange(1, 6, dtype=x.dtype, device=x.device)
        .view(1, 1, 5)
        .expand(1, hidden, 5)
        .contiguous()
    )
    info = TritonMoeQuantInfo(
        w13_weight=w13,
        w2_weight=w2,
        use_int4_w4a16=True,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        block_shape=[0, 128],
        w2_block_shape=[0, 32],
    )
    config = MoeRunnerConfig(
        num_experts=1,
        num_local_experts=1,
        hidden_size=hidden,
        intermediate_size_per_partition=intermediate,
        top_k=1,
        activation="silu",
        is_gated=True,
        inplace=inplace,
    )
    dispatch = SimpleNamespace(
        hidden_states=x,
        hidden_states_pre_quant=None,
        topk_output=StandardTopKOutput(
            topk_weights=torch.ones((1, 1), dtype=torch.float32, device="cuda"),
            topk_ids=torch.zeros((1, 1), dtype=torch.int32, device="cuda"),
            router_logits=torch.zeros((1, 1), dtype=torch.float32, device="cuda"),
        ),
    )
    with (
        get_context().override_server_args(model_path="dummy"),
        get_flags().moe.override(runner_backend=MoeRunnerBackend.TRITON),
    ):
        if path == "fused":
            result = fused_experts_none_to_triton(dispatch, info, config).hidden_states
        else:
            state = {}
            runner_input = pre_permute_standard_to_triton(dispatch, info, config, state)
            result = (
                TritonRunnerCore(config).run(runner_input, info, state).hidden_states
            )

    # GEMM1 yields gate=up=1; GEMM2 uses five 32-element groups with scales 1..5.
    activation = torch.nn.functional.silu(torch.tensor(1.0)).to(x.dtype).float()
    expected = torch.full_like(result, float(activation * (32 * sum(range(1, 6)))))
    torch.testing.assert_close(result, expected, rtol=1e-3, atol=0.5)
    assert (result.data_ptr() == x.data_ptr()) == inplace


if __name__ == "__main__":
    args = ["-x" if arg == "-f" else arg for arg in sys.argv[1:]]
    sys.exit(pytest.main([__file__, *args]))
