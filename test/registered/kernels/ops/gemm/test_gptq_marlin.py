import sys

import pytest
import torch
from sgl_kernel.scalar_type import scalar_types

import sglang.kernels.ops.gemm.gptq_marlin as gptq_marlin_ops
from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.ops.gemm.gptq_marlin import gptq_marlin_gemm
from sglang.srt.layers.quantization.marlin_utils import (
    marlin_make_workspace,
)
from sglang.srt.layers.quantization.marlin_utils_fp8 import (
    apply_fp8_marlin_linear,
    prepare_fp8_layer_for_marlin,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_marlin_utils import (
    awq_marlin_quantize,
    marlin_quantize,
)

register_cuda_ci(est_time=18, stage="base-b-kernel-unit", runner_config="1-gpu-small")

MNK_FACTORS = [
    (1, 1, 1),
    (1, 4, 8),
    (13, 17, 67),
    (257, 13, 11),
]


@pytest.mark.parametrize("k_chunk", [128])
@pytest.mark.parametrize("n_chunk", [64, 256])
@pytest.mark.parametrize("quant_type", [scalar_types.uint4, scalar_types.uint4b8])
@pytest.mark.parametrize("group_size", [-1, 128])
@pytest.mark.parametrize("mnk_factors", MNK_FACTORS)
@pytest.mark.parametrize("act_order", [False, True])
def test_gptq_marlin_gemm(
    k_chunk,
    n_chunk,
    quant_type,
    group_size,
    mnk_factors,
    act_order,
):
    m_factor, n_factor, k_factor = mnk_factors
    has_zp = quant_type in [scalar_types.uint4, scalar_types.uint8]

    size_m = m_factor
    size_k = k_chunk * k_factor
    size_n = n_chunk * n_factor

    if act_order:
        if group_size == -1:
            return
        if group_size == size_k:
            return
        if has_zp:
            return

    if size_k % group_size != 0:
        return

    a_input = torch.randn((size_m, size_k), dtype=torch.float16, device="cuda")
    b_weight = torch.randn((size_k, size_n), dtype=torch.float16, device="cuda")

    if has_zp:
        w_ref, marlin_q_w, marlin_s, marlin_zp = awq_marlin_quantize(
            b_weight, quant_type, group_size
        )
        g_idx = None
        sort_indices = None
        marlin_s2 = None
    else:
        w_ref, marlin_q_w, marlin_s, g_idx, sort_indices, _ = marlin_quantize(
            b_weight, quant_type, group_size, act_order
        )
        marlin_zp = None
        marlin_s2 = None

    workspace = marlin_make_workspace(w_ref.device)

    output = gptq_marlin_gemm(
        a_input,
        None,
        marlin_q_w,
        marlin_s,
        marlin_s2,
        marlin_zp,
        g_idx,
        sort_indices,
        workspace,
        quant_type,
        a_input.shape[0],
        b_weight.shape[1],
        a_input.shape[1],
        is_k_full=True,
        use_atomic_add=False,
        use_fp32_reduce=False,
        is_zp_float=False,
    )

    output_ref = torch.matmul(a_input, w_ref)
    torch.cuda.synchronize()

    # JIT kernel should produce approximately correct results vs torch.matmul
    max_diff = torch.mean(torch.abs(output - output_ref)) / torch.mean(
        torch.abs(output_ref)
    )
    assert max_diff < 0.04


@cache_once
def _lagged_gptq_marlin_module(dtype: torch.dtype):
    args = make_cpp_args(dtype)
    return load_jit(
        "gptq_marlin_lagged_warps",
        *args,
        cuda_files=["gemm/marlin/gptq_marlin.cuh"],
        cuda_wrappers=[("gptq_marlin_gemm", f"gptq_marlin_gemm<{args}>")],
        extra_cuda_cflags=["-DSGLANG_MARLIN_TEST_LAG_WARPS_NS=20000"],
    )


def _fp8_block_gemm(size_m, size_n, size_k):
    """FP8 128x128-block GEMM through the serving path; the exact product is known."""
    q = torch.randint(-4, 5, (size_n, size_k), device="cuda").float()
    # Power-of-two block scales keep the product exact; each K group's scale
    # differs from the scale two groups later.
    exponents = torch.arange(size_k // 128, device="cuda") % 4 - 2
    block_scale = torch.exp2(exponents.float()).expand(size_n // 128, -1).contiguous()
    w_ref = q * block_scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    a = torch.randint(-3, 4, (size_m, size_k), device="cuda").to(torch.float16)
    expected = (a.double() @ w_ref.double().T).to(torch.float16)

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(q.to(torch.float8_e4m3fn), requires_grad=False)
    layer.weight_scale_inv = torch.nn.Parameter(block_scale, requires_grad=False)
    layer.output_size_per_partition = size_n
    layer.input_size_per_partition = size_k
    layer.orig_dtype = torch.float16
    layer.weight_block_size = [128, 128]
    prepare_fp8_layer_for_marlin(layer, size_k_first=False)

    def run():
        return apply_fp8_marlin_linear(
            input=a,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            workspace=layer.workspace,
            size_n=size_n,
            size_k=size_k,
            bias=None,
        )

    return run, expected


def _awq_gemm(size_m, size_n, size_k):
    """AWQ uint4 GEMM with group-128 zero points; the in-step kernel gives the reference."""
    w = torch.randn((size_k, size_n), dtype=torch.float16, device="cuda")
    _, q_w, scales, zeros = awq_marlin_quantize(w, scalar_types.uint4, 128)
    a = torch.randn((size_m, size_k), dtype=torch.float16, device="cuda")
    workspace = marlin_make_workspace(a.device)

    def run():
        return gptq_marlin_gemm(
            a=a,
            c=None,
            b_q_weight=q_w,
            b_scales=scales,
            global_scale=None,
            b_zeros=zeros,
            g_idx=None,
            perm=None,
            workspace=workspace,
            b_q_type=scalar_types.uint4,
            size_m=size_m,
            size_n=size_n,
            size_k=size_k,
            use_fp32_reduce=True,
        )

    return run, run()


@pytest.mark.parametrize(
    "make_gemm", [_fp8_block_gemm, _awq_gemm], ids=["fp8_block", "awq"]
)
def test_group_data_survives_lagging_warps(monkeypatch, make_gemm):
    """A group-quantized GEMM gives the same result when a block's warps run out of step.

    With M > 16 a 128-wide group spans two 64-wide K tiles; the lagged build holds
    back every warp but the one that loads the group scales and zero points.
    """
    # 2048 tiles, so on any GPU a block's slices outlast the 4-stage pipeline.
    run, expected = make_gemm(size_m=256, size_n=4096, size_k=2048)
    monkeypatch.setattr(
        gptq_marlin_ops, "_jit_gptq_marlin_module", _lagged_gptq_marlin_module
    )
    assert torch.equal(run(), expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
