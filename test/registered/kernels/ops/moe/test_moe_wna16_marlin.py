import itertools
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from sgl_kernel.scalar_type import scalar_types

from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.srt.layers import zero_copy_context
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
from sglang.srt.layers.moe.utils import MoeRunnerBackend, initialize_moe_config
from sglang.srt.layers.quantization.marlin_utils_fp4 import (
    prepare_moe_nvfp4_layer_for_marlin,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
from sglang.srt.utils.common import is_sm80_supported, is_sm90_supported
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_marlin_utils import (
    awq_marlin_quantize,
    make_nvfp4_weight_and_ref,
    marlin_quantize,
)

register_cuda_ci(est_time=720, stage="base-b", runner_config="1-gpu-large")


def _has_aot_moe_wna16_marlin_gemm() -> bool:
    return hasattr(torch.ops.sgl_kernel, "moe_wna16_marlin_gemm") and hasattr(
        torch.ops.sgl_kernel.moe_wna16_marlin_gemm, "default"
    )


AOT_AVAILABLE = _has_aot_moe_wna16_marlin_gemm()


def stack_and_dev(tensors: list[torch.Tensor]):
    dev = tensors[0].device
    return torch.stack(tensors, dim=0).to(dev)


def _get_scalar_type(num_bits: int, has_zp: bool):
    if has_zp:
        assert num_bits == 4
        return scalar_types.uint4
    else:
        return scalar_types.uint4b8 if num_bits == 4 else scalar_types.uint8b128


def _setup_moe_weights(e, n, k, quant_type, group_size, act_order, dtype):
    """Set up quantized MoE weights for a single gate (e experts, output n, input k)."""
    has_zp = quant_type in [scalar_types.uint4, scalar_types.uint8]

    w = torch.randn((e, n, k), device="cuda", dtype=dtype) / 20

    w_ref_l = []
    qweight_l = []
    scales_l = []
    zeros_l = []
    g_idx_l = []
    sort_indices_l = []

    for i in range(e):
        if has_zp:
            w_ref, qweight, scales, zeros = awq_marlin_quantize(
                w[i].transpose(1, 0), quant_type, group_size
            )
            w_ref_l.append(w_ref.T)
            qweight_l.append(qweight)
            scales_l.append(scales)
            zeros_l.append(zeros)
        else:
            test_perm = torch.randperm(k)
            w_ref, qweight, scales, g_idx, sort_indices, _ = marlin_quantize(
                w[i].transpose(1, 0), quant_type, group_size, act_order, test_perm
            )
            w_ref_l.append(w_ref.T)
            qweight_l.append(qweight)
            scales_l.append(scales)
            g_idx_l.append(g_idx)
            sort_indices_l.append(sort_indices)

    w_ref = stack_and_dev(w_ref_l)
    qweight = stack_and_dev(qweight_l).contiguous()
    scales = stack_and_dev(scales_l)
    g_idx = stack_and_dev(g_idx_l) if g_idx_l else None
    sort_indices = stack_and_dev(sort_indices_l) if sort_indices_l else None
    zeros = stack_and_dev(zeros_l) if zeros_l else None

    return w_ref, qweight, scales, zeros, g_idx, sort_indices


def _run_single_gemm(
    fn,
    a,
    c,
    qweight,
    scales,
    zeros,
    g_idx,
    sort_indices,
    workspace,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    topk_weights,
    quant_type,
    block_size_m,
    topk,
    size_m,
    size_n,
    size_k,
    mul_topk_weights,
    is_k_full,
    use_atomic_add,
):
    return fn(
        a,
        c,
        qweight,
        None,  # b_bias
        scales,
        None,  # global_scale
        zeros,
        g_idx,
        sort_indices,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size=block_size_m,
        top_k=topk,
        mul_topk_weights=mul_topk_weights,
        is_ep=False,
        b_q_type=quant_type,
        size_m=size_m,
        size_n=size_n,
        size_k=size_k,
        is_k_full=is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=True,
        is_zp_float=False,
    )


def _run_single_gemm_aot(
    a,
    c,
    qweight,
    scales,
    zeros,
    g_idx,
    sort_indices,
    workspace,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    topk_weights,
    quant_type,
    block_size_m,
    topk,
    size_m,
    size_n,
    size_k,
    mul_topk_weights,
    is_k_full,
    use_atomic_add,
):
    return torch.ops.sgl_kernel.moe_wna16_marlin_gemm.default(
        a,
        c,
        qweight,
        None,  # b_bias
        scales,
        None,  # global_scale
        zeros,
        g_idx,
        sort_indices,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size=block_size_m,
        top_k=topk,
        mul_topk_weights=mul_topk_weights,
        is_ep=False,
        b_q_type_id=quant_type.id,
        size_m=size_m,
        size_n=size_n,
        size_k=size_k,
        is_k_full=is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=True,
        is_zp_float=False,
    )


def generate_test_cases():
    m_list = [1, 123]
    n_list = [128, 1024]
    k_list = [256]
    e_list = [4]
    topk_list = [2]
    dtype_list = [torch.float16, torch.bfloat16]
    group_size_list = [128]
    act_order_list = [False, True]
    quant_type_list = [scalar_types.uint4, scalar_types.uint4b8]

    all_combinations = itertools.product(
        m_list,
        n_list,
        k_list,
        e_list,
        topk_list,
        dtype_list,
        group_size_list,
        act_order_list,
        quant_type_list,
    )

    def is_valid(m, n, k, e, topk, dtype, group_size, act_order, quant_type):
        has_zp = quant_type in [scalar_types.uint4, scalar_types.uint8]
        if act_order:
            if group_size == -1 or group_size == k:
                return False
            if has_zp:
                return False
        if group_size > 0 and k % group_size != 0:
            return False
        return True

    return [case for case in all_combinations if is_valid(*case)]


TEST_CASES = generate_test_cases()


@pytest.mark.parametrize(
    "m,n,k,e,topk,dtype,group_size,act_order,quant_type",
    TEST_CASES,
    ids=[
        f"m{c[0]}_n{c[1]}_k{c[2]}_e{c[3]}_t{c[4]}_{c[5].__name__ if hasattr(c[5], '__name__') else str(c[5]).split('.')[-1]}_g{c[6]}_act{c[7]}_{c[8]}"
        for c in TEST_CASES
    ],
)
def test_moe_wna16_marlin_gemm(
    m, n, k, e, topk, dtype, group_size, act_order, quant_type
):
    if not AOT_AVAILABLE:
        pytest.skip("sgl_kernel moe_wna16_marlin_gemm AOT op not available")

    torch.manual_seed(0)

    has_zp = quant_type in [scalar_types.uint4, scalar_types.uint8]

    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10

    # Set up quantized weights for first gemm (gate_up: output 2*n, input k)
    w_ref1, qweight1, scales1, zeros1, g_idx1, sort_indices1 = _setup_moe_weights(
        e, 2 * n, k, quant_type, group_size, act_order, dtype
    )

    # Compute block_size_m
    for block_size_m in [8, 16, 32, 48, 64]:
        if m * topk / e / block_size_m < 0.9:
            break

    # Align tokens
    score = torch.randn((m, e), device="cuda", dtype=dtype)
    score_softmax = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(score_softmax, topk)

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_size_m, e
    )

    # Workspace
    sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    max_workspace_size = (max(2 * n, k) // 64) * (
        sorted_token_ids.size(0) // block_size_m
    )
    max_workspace_size = min(max_workspace_size, sms * 4)
    workspace = torch.zeros(
        max_workspace_size, dtype=torch.int, device="cuda", requires_grad=False
    )

    use_atomic_add = (
        dtype == torch.half or torch.cuda.get_device_capability("cuda")[0] >= 9
    )

    scalar_type = _get_scalar_type(4, has_zp)

    # --- Run JIT kernel ---
    c_jit = torch.empty((m * topk, 2 * n), dtype=dtype, device="cuda")
    c_jit = _run_single_gemm(
        moe_wna16_marlin_gemm,
        a,
        c_jit,
        qweight1,
        scales1,
        zeros1,
        g_idx1,
        sort_indices1,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        scalar_type,
        block_size_m,
        topk,
        m,
        2 * n,
        k,
        False,
        True,
        use_atomic_add,
    )

    torch.cuda.synchronize()

    # --- Check bitwise equality with AOT kernel ---
    c_aot = torch.empty((m * topk, 2 * n), dtype=dtype, device="cuda")
    c_aot = _run_single_gemm_aot(
        a,
        c_aot,
        qweight1,
        scales1,
        zeros1,
        g_idx1,
        sort_indices1,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        scalar_type,
        block_size_m,
        topk,
        m,
        2 * n,
        k,
        False,
        True,
        use_atomic_add,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(c_jit, c_aot, rtol=0, atol=0)


@pytest.mark.skipif(
    not (is_sm80_supported() or is_sm90_supported()),
    reason="Non-gated NVFP4 Marlin fallback test requires CUDA SM8X/SM9X",
)
def test_fused_marlin_moe_non_gated_relu2():
    torch.manual_seed(0)

    m = 17
    n = 128
    k = 256
    e = 4
    topk = 2
    dtype = torch.float16
    group_size = 128
    quant_type = scalar_types.uint4b8

    hidden_states = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w_ref1, qweight1, scales1, zeros1, g_idx1, sort_indices1 = _setup_moe_weights(
        e, n, k, quant_type, group_size, False, dtype
    )
    w_ref2, qweight2, scales2, zeros2, g_idx2, sort_indices2 = _setup_moe_weights(
        e, k, n, quant_type, group_size, False, dtype
    )

    router_logits = torch.randn((m, e), device="cuda", dtype=dtype)
    score_softmax = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(score_softmax, topk)

    output = fused_marlin_moe(
        hidden_states=hidden_states,
        w1=qweight1,
        w2=qweight2,
        w1_scale=scales1,
        w2_scale=scales2,
        gating_output=router_logits,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        g_idx1=g_idx1,
        g_idx2=g_idx2,
        sort_indices1=sort_indices1,
        sort_indices2=sort_indices2,
        w1_zeros=zeros1,
        w2_zeros=zeros2,
        num_bits=4,
        is_k_full=True,
        routed_scaling_factor=1.0,
        activation="relu2",
        is_gated=False,
    )

    output_ref = torch.zeros_like(hidden_states)
    for token_idx in range(m):
        for route_idx in range(topk):
            expert_id = topk_ids[token_idx, route_idx]
            intermediate = hidden_states[token_idx] @ w_ref1[expert_id].T
            intermediate = torch.square(torch.relu(intermediate))
            routed = intermediate @ w_ref2[expert_id].T
            output_ref[token_idx] += routed * topk_weights[token_idx, route_idx]

    torch.cuda.synchronize()
    torch.testing.assert_close(output, output_ref, rtol=0.04, atol=0.04)


@pytest.mark.parametrize("m", [123, 2304])
@pytest.mark.parametrize("has_bias", [False, True])
def test_fused_marlin_moe_large_non_ep_schedule(m, has_bias):
    torch.manual_seed(0)

    n = 1024
    k = 512
    e = 8
    topk = 2
    dtype = torch.bfloat16
    group_size = 128
    quant_type = scalar_types.uint4b8

    hidden_states = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w_ref1, qweight1, scales1, zeros1, g_idx1, sort_indices1 = _setup_moe_weights(
        e, n, k, quant_type, group_size, False, dtype
    )
    w_ref2, qweight2, scales2, zeros2, g_idx2, sort_indices2 = _setup_moe_weights(
        e, k, n, quant_type, group_size, False, dtype
    )
    w1_bias = (
        torch.randn((e, n), device="cuda", dtype=dtype) / 100 if has_bias else None
    )
    w2_bias = (
        torch.randn((e, k), device="cuda", dtype=dtype) / 100 if has_bias else None
    )

    router_logits = torch.randn((m, e), device="cuda", dtype=dtype)
    score_softmax = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(score_softmax, topk)

    output = fused_marlin_moe(
        hidden_states=hidden_states,
        w1=qweight1,
        w2=qweight2,
        w1_scale=scales1,
        w2_scale=scales2,
        gating_output=router_logits,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        g_idx1=g_idx1,
        g_idx2=g_idx2,
        sort_indices1=sort_indices1,
        sort_indices2=sort_indices2,
        w1_zeros=zeros1,
        w2_zeros=zeros2,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        num_bits=4,
        is_k_full=True,
        routed_scaling_factor=1.0,
        activation="relu2",
        is_gated=False,
    )

    output_ref = torch.zeros_like(hidden_states, dtype=torch.float32)
    for expert_id in range(e):
        token_indices, route_indices = torch.where(topk_ids == expert_id)
        intermediate = hidden_states[token_indices] @ w_ref1[expert_id].T
        if w1_bias is not None:
            intermediate += w1_bias[expert_id]
        intermediate = torch.square(torch.relu(intermediate))
        routed = intermediate @ w_ref2[expert_id].T
        if w2_bias is not None:
            routed += w2_bias[expert_id]
        output_ref.index_add_(
            0,
            token_indices,
            routed.float() * topk_weights[token_indices, route_indices, None],
        )

    torch.cuda.synchronize()
    # The existing BF16/4-bit bias path has a few large-batch outliers just
    # above 0.04 even without the compile-time specialization.
    torch.testing.assert_close(output, output_ref.to(dtype), rtol=0.04, atol=0.06)


@pytest.mark.skipif(
    not (is_sm80_supported() or is_sm90_supported()),
    reason="NVFP4 Marlin MoE padding test requires CUDA SM8X/SM9X",
)
def test_fused_marlin_moe_nvfp4_non_gated_padded_intermediate_launches():
    torch.manual_seed(0)

    m = 17
    intermediate_size = 192
    hidden_size = 256
    e = 4
    topk = 2
    dtype = torch.bfloat16
    nvfp4_group_size = 16

    layer = torch.nn.Module()
    layer.quant_config = SimpleNamespace(group_size=nvfp4_group_size)
    layer.moe_runner_config = SimpleNamespace(is_gated=False)
    layer.params_dtype = dtype
    layer.intermediate_size_per_partition = intermediate_size
    layer.w13_weight = torch.nn.Parameter(
        torch.randint(
            0,
            256,
            (e, intermediate_size, hidden_size // 2),
            device="cuda",
            dtype=torch.uint8,
        ),
        requires_grad=False,
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.randint(
            0,
            256,
            (e, hidden_size, intermediate_size // 2),
            device="cuda",
            dtype=torch.uint8,
        ),
        requires_grad=False,
    )
    layer.w13_weight_scale = torch.nn.Parameter(
        torch.rand(
            (e, intermediate_size, hidden_size // nvfp4_group_size),
            device="cuda",
            dtype=dtype,
        ),
        requires_grad=False,
    )
    layer.w2_weight_scale = torch.nn.Parameter(
        torch.rand(
            (e, hidden_size, intermediate_size // nvfp4_group_size),
            device="cuda",
            dtype=dtype,
        ),
        requires_grad=False,
    )
    layer.w13_weight_scale_2 = torch.nn.Parameter(
        torch.ones((e,), device="cuda", dtype=dtype), requires_grad=False
    )
    layer.w2_weight_scale_2 = torch.nn.Parameter(
        torch.ones((e,), device="cuda", dtype=dtype), requires_grad=False
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)

    assert layer.w13_weight.shape[1] * 16 == 256
    assert layer.w2_weight.shape[1] * 16 == 256

    hidden_states = torch.randn((m, hidden_size), device="cuda", dtype=dtype) / 10

    score = torch.randn((m, e), device="cuda", dtype=dtype)
    score_softmax = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(score_softmax, topk)

    out = fused_marlin_moe(
        hidden_states=hidden_states,
        w1=layer.w13_weight,
        w2=layer.w2_weight,
        w1_scale=layer.w13_weight_scale,
        w2_scale=layer.w2_weight_scale,
        gating_output=score,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        w1_global_scale=layer.w13_weight_scale_2,
        w2_global_scale=layer.w2_weight_scale_2,
        workspace=layer.workspace,
        num_bits=4,
        is_k_full=True,
        routed_scaling_factor=1.0,
        activation="relu2",
        is_gated=False,
    )

    torch.cuda.synchronize()
    assert out.shape == (m, hidden_size)


@pytest.mark.skipif(
    not (is_sm80_supported() or is_sm90_supported()),
    reason="NVFP4 Marlin MoE numeric test requires CUDA SM80, SM86, or SM90",
)
def test_fused_marlin_moe_nvfp4_non_gated_matches_dequant_reference():
    torch.manual_seed(0)

    m = 17
    intermediate_size = 192
    hidden_size = 256
    e = 4
    topk = 2
    dtype = torch.bfloat16
    group_size = 16
    routed_scaling_factor = 1.0

    w13_packed_l, w13_scales_l, w13_gscale_l, w13_ref_l = [], [], [], []
    w2_packed_l, w2_scales_l, w2_gscale_l, w2_ref_l = [], [], [], []
    for _ in range(e):
        packed, scales, gscale, ref = make_nvfp4_weight_and_ref(
            intermediate_size, hidden_size, dtype, group_size=group_size
        )
        w13_packed_l.append(packed)
        w13_scales_l.append(scales)
        w13_gscale_l.append(gscale)
        w13_ref_l.append(ref)

        packed, scales, gscale, ref = make_nvfp4_weight_and_ref(
            hidden_size, intermediate_size, dtype, group_size=group_size
        )
        w2_packed_l.append(packed)
        w2_scales_l.append(scales)
        w2_gscale_l.append(gscale)
        w2_ref_l.append(ref)

    layer = torch.nn.Module()
    layer.quant_config = SimpleNamespace(group_size=group_size)
    layer.moe_runner_config = SimpleNamespace(is_gated=False)
    layer.params_dtype = dtype
    layer.intermediate_size_per_partition = intermediate_size
    layer.w13_weight = torch.nn.Parameter(
        torch.stack(w13_packed_l), requires_grad=False
    )
    layer.w2_weight = torch.nn.Parameter(torch.stack(w2_packed_l), requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(
        torch.stack(w13_scales_l), requires_grad=False
    )
    layer.w2_weight_scale = torch.nn.Parameter(
        torch.stack(w2_scales_l), requires_grad=False
    )
    layer.w13_weight_scale_2 = torch.nn.Parameter(
        torch.stack(w13_gscale_l), requires_grad=False
    )
    layer.w2_weight_scale_2 = torch.nn.Parameter(
        torch.stack(w2_gscale_l), requires_grad=False
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)

    # Scale activations down so relu² doesn't blow up intermediate magnitudes;
    # this keeps output values small so tighter element-wise tolerance is realistic.
    hidden_states = torch.randn((m, hidden_size), device="cuda", dtype=dtype) / 20
    router_logits = torch.randn((m, e), device="cuda", dtype=dtype)
    score_softmax = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(score_softmax, topk)

    output = fused_marlin_moe(
        hidden_states=hidden_states,
        w1=layer.w13_weight,
        w2=layer.w2_weight,
        w1_scale=layer.w13_weight_scale,
        w2_scale=layer.w2_weight_scale,
        gating_output=router_logits,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        w1_global_scale=layer.w13_weight_scale_2,
        w2_global_scale=layer.w2_weight_scale_2,
        workspace=layer.workspace,
        num_bits=4,
        is_k_full=True,
        routed_scaling_factor=routed_scaling_factor,
        activation="relu2",
        is_gated=False,
    )

    w13_ref = torch.stack(w13_ref_l)
    w2_ref = torch.stack(w2_ref_l)
    output_ref = torch.zeros_like(hidden_states)
    for token_idx in range(m):
        for route_idx in range(topk):
            expert_id = topk_ids[token_idx, route_idx]
            intermediate = hidden_states[token_idx] @ w13_ref[expert_id].T
            intermediate = torch.square(torch.relu(intermediate))
            routed = intermediate @ w2_ref[expert_id].T
            output_ref[token_idx] += routed * topk_weights[token_idx, route_idx]
    output_ref *= routed_scaling_factor

    torch.cuda.synchronize()
    torch.testing.assert_close(output, output_ref, rtol=0.05, atol=0.25)


def _dequant_mxfp4_reference(packed, scales):
    fp4 = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device="cuda",
        dtype=torch.float64,
    )
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    return fp4[codes.long()] * scales.double().repeat_interleave(32, -1)


@pytest.mark.skipif(not is_sm90_supported(), reason="Hopper MXFP4 decode tile")
@pytest.mark.parametrize("m", [1, 4])
def test_mxfp4_tp8_gate_decode(m):
    from sglang.srt.layers.quantization.marlin_utils_fp4 import (
        _permute_moe_fp4_scales_for_marlin,
        _repack_moe_fp4_weight_for_marlin,
        mxfp4_marlin_process_scales,
    )

    torch.manual_seed(741 + m)
    e, n, k, topk = 8, 640, 5120, 6
    raw = torch.randint(0, 256, (e, n, k // 2), device="cuda", dtype=torch.uint8)
    scale = torch.randint(
        120, 126, (e, n, k // 32), device="cuda", dtype=torch.uint8
    ).view(torch.float8_e8m0fnu)
    packed = _repack_moe_fp4_weight_for_marlin(
        raw,
        num_experts=e,
        size_n=n,
        size_k=k,
        perm=torch.empty(0, device="cuda", dtype=torch.int32),
    )
    sf = _permute_moe_fp4_scales_for_marlin(
        scale.bfloat16(),
        num_experts=e,
        size_n=n,
        size_k=k,
        group_size=32,
        process_scales=lambda s: mxfp4_marlin_process_scales(
            s, input_dtype=torch.bfloat16
        ),
    )
    weights = _dequant_mxfp4_reference(raw, scale)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    ids = torch.rand(m, e, device="cuda").topk(topk, dim=1).indices.int()
    topk_weights = torch.full((m, topk), 1 / topk, device="cuda")
    sorted_ids, expert_ids, count = moe_align_block_size(ids, 8, e)
    workspace = torch.zeros(
        torch.cuda.get_device_properties(0).multi_processor_count * 4,
        device="cuda",
        dtype=torch.int32,
    )

    def run():
        return _run_single_gemm(
            moe_wna16_marlin_gemm,
            x,
            None,
            packed,
            sf,
            None,
            None,
            None,
            workspace,
            sorted_ids,
            expert_ids,
            count,
            topk_weights,
            scalar_types.float4_e2m1f,
            8,
            topk,
            m,
            n,
            k,
            False,
            True,
            False,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for _ in range(2):
        x.neg_()
        graph.replay()
        expected = torch.stack(
            [
                x[row].double() @ weights[expert].T
                for row in range(m)
                for expert in ids[row].tolist()
            ]
        )
        torch.testing.assert_close(
            actual.double(),
            expected,
            rtol=1 / 128,
            atol=expected.square().mean().sqrt().item() * 1e-4,
        )


@pytest.mark.skipif(not is_sm90_supported(), reason="Hopper MXFP4 weight loading")
@pytest.mark.parametrize("tp_rank", [0, 7])
def test_mxfp4_tp8_load_before_padding(tp_rank):
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.quantization.marlin_utils_fp4 import (
        prepare_moe_mxfp4_layer_for_marlin,
    )
    from sglang.srt.layers.quantization.mxfp4_marlin_moe import Mxfp4MarlinMoEMethod

    torch.manual_seed(325 + tp_rank)
    e, h, intermediate, tp, m = 2, 256, 288, 8, 4
    method = Mxfp4MarlinMoEMethod(None, "test")
    layer = torch.nn.Module()
    layer.hidden_size = h
    with torch.device("cuda"):
        method.create_weights(layer, e, h, intermediate, torch.bfloat16)
    assert layer.w13_weight.shape == (e, 2 * intermediate, h // 2)
    loader = SimpleNamespace(
        moe_tp_size=tp,
        moe_tp_rank=tp_rank,
        use_padded_loading=False,
        use_presharded_weights=False,
        use_triton_kernels=False,
        moe_runner_config=SimpleNamespace(is_gated=True),
        quant_method=method,
        quant_config=None,
    )
    references = {}
    for shard in ("w1", "w3", "w2"):
        down = shard == "w2"
        n, k = (h, intermediate * tp) if down else (intermediate * tp, h)
        raw = torch.randint(0, 256, (e, n, k // 2), device="cuda", dtype=torch.uint8)
        scale = torch.randint(
            119, 123, (e, n, k // 32), device="cuda", dtype=torch.uint8
        ).view(torch.float8_e8m0fnu)
        prefix = "w2" if down else "w13"
        load = FusedMoE._load_w2 if down else FusedMoE._load_w13
        axis = 1 if down else 0
        expected = []
        for data, suffix in [
            (raw.view(torch.int8), "weight"),
            (scale, "weight_scale_inv"),
        ]:
            parameter = getattr(layer, f"{prefix}_{suffix}")
            for expert in range(e):
                load(loader, parameter[expert], axis, shard, data[expert])
            width = data.shape[axis + 1] // tp
            selected = data.narrow(axis + 1, tp_rank * width, width)
            actual = (
                parameter
                if down
                else parameter[:, :intermediate]
                if shard == "w1"
                else parameter[:, intermediate:]
            )
            torch.testing.assert_close(actual.float(), selected.float(), rtol=0, atol=0)
            expected.append(selected)
        packed, sf = expected
        packed = packed.view(torch.uint8)
        references[shard] = _dequant_mxfp4_reference(packed, sf)

    prepare_moe_mxfp4_layer_for_marlin(layer)
    assert layer.w13_weight_scale.shape[-1] == 640
    assert layer.w2_weight_scale.shape[1] == 10
    x = torch.randn(m, h, device="cuda", dtype=torch.bfloat16)
    ids = (
        torch.tensor([[0, 1]], device="cuda", dtype=torch.int32)
        .expand(m, -1)
        .contiguous()
    )
    weights = torch.tensor([[0.25, 0.75]], device="cuda").expand(m, -1).contiguous()

    def run():
        return fused_marlin_moe(
            x,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            torch.empty(m, e, device="cuda"),
            weights,
            ids,
            workspace=layer.workspace,
            num_bits=4,
            clamp_limit=10.0,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for _ in range(2):
        x.normal_()
        graph.replay()
        expected = []
        for expert in range(e):
            gate = (x.double() @ references["w1"][expert].T).bfloat16().clamp(max=10)
            up = (x.double() @ references["w3"][expert].T).bfloat16().clamp(-10, 10)
            activated = torch.nn.functional.silu(gate) * up
            projected = activated.double() @ references["w2"][expert].T
            # Marlin rounds GEMM2 before multiplying its BF16 routing weight.
            expected.append(projected.bfloat16() * weights[:, expert, None].bfloat16())
        reference = torch.stack(expected).float().sum(0).bfloat16()
        torch.testing.assert_close(
            actual,
            reference,
            rtol=0.03,
            atol=reference.float().square().mean().sqrt().item() * 0.002,
        )


def _marlin_inactive_weights(experts, output_size, input_size, quant_type):
    tensors = _setup_moe_weights(
        min(experts, 8), output_size, input_size, quant_type, 128, False, torch.bfloat16
    )
    result = []
    for tensor in tensors:
        if tensor is None:
            result.append(None)
            continue
        tensor = tensor.repeat((experts // tensor.shape[0],) + (1,) * (tensor.ndim - 1))
        # A leading poisoned record makes negative expert indexing observable
        # without reading outside an allocation during the failing regression.
        storage = tensor.new_empty((experts + 1, *tensor.shape[1:]))
        storage[0].fill_(float("nan") if tensor.is_floating_point() else 0)
        storage[1:].copy_(tensor)
        result.append(storage[1:])
    return result


@pytest.mark.parametrize(
    "rows,experts,intermediate",
    [(1, 8, 128), (4, 128, 128), (16, 8, 256), (64, 128, 128)],
)
@pytest.mark.parametrize(
    "quant_type,num_bits",
    [(scalar_types.uint4b8, 4), (scalar_types.uint8b128, 8), (scalar_types.uint4, 4)],
)
def test_fused_marlin_moe_inactive_routes(
    rows, experts, intermediate, quant_type, num_bits
):
    """EP1 padded graph rows contribute zero through Marlin and its shared caches."""
    _check_marlin_inactive_routes(rows, experts, intermediate, quant_type, num_bits)


def test_stock_marlin_lora_inactive_routes():
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Stock Marlin LoRA requires SM90 or newer")
    _check_marlin_inactive_routes(4, 8, 128, scalar_types.uint4b8, 4, "stock_lora")


def test_factored_marlin_lora_inactive_routes():
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Experimental Marlin LoRA requires SM90 or newer")
    _check_marlin_inactive_routes(4, 8, 128, scalar_types.uint4b8, 4, "factored_lora")


def _check_marlin_inactive_routes(
    rows, experts, intermediate, quant_type, num_bits, runner_kind="fused"
):
    torch.manual_seed(43)
    hidden, topk = 256, 2
    with get_context().override_server_args(
        model_path="dummy",
        device="cuda",
        moe_runner_backend="auto",
        ep_dispatch_algorithm=None,
        enable_eplb=False,
        enable_waterfill=False,
    ):
        flags = get_flags().moe
        flag_values = {name: getattr(flags, name) for name in flags.__struct_fields__}
        flag_values["in_speculative_scope"] = False
        with (
            flags.override(**flag_values),
            get_parallel().override(
                tp_size=1,
                tp_rank=0,
                moe_ep_size=1,
                moe_ep_rank=0,
                moe_tp_size=1,
                moe_tp_rank=0,
                attn_tp_size=1,
                attn_tp_rank=0,
                attn_dp_size=1,
                attn_cp_size=1,
                tp_group=None,
            ),
        ):
            initialize_moe_config()
            x = torch.randn((rows, hidden), device="cuda", dtype=torch.bfloat16)
            logits = torch.randn((rows, experts), device="cuda", dtype=torch.float32)
            w1_ref, w1, s1, z1, g1, p1 = _marlin_inactive_weights(
                experts, 2 * intermediate, hidden, quant_type
            )
            w2_ref, w2, s2, z2, g2, p2 = _marlin_inactive_weights(
                experts, hidden, intermediate, quant_type
            )
            router = TopK(
                top_k=topk,
                layer_id=0,
                renormalize=True,
                output_format=TopKOutputFormat.STANDARD,
                allow_routed_experts_capture=False,
            )
            router.bind_runner_backend(MoeRunnerBackend.MARLIN)
            live = torch.tensor(rows, device="cuda", dtype=torch.int32)
            batch = ForwardBatch(
                forward_mode=ForwardMode.EXTEND,
                batch_size=1,
                input_ids=torch.arange(rows, device="cuda"),
                req_pool_indices=torch.zeros(1, device="cuda", dtype=torch.int64),
                seq_lens=torch.tensor([rows], device="cuda", dtype=torch.int32),
                out_cache_loc=torch.arange(rows, device="cuda"),
                seq_lens_sum=rows,
                num_token_non_padded=live,
            )
            output = torch.full_like(x, float("nan"))
            finite_logits = logits.clone()
            hook_weights = []
            if runner_kind != "fused":
                from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
                from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo
                from sglang.srt.layers.moe.token_dispatcher.standard import (
                    StandardDispatchOutput,
                )

                config = MoeRunnerConfig(
                    num_experts=experts,
                    num_local_experts=experts,
                    hidden_size=hidden,
                    intermediate_size_per_partition=intermediate,
                    top_k=topk,
                    params_dtype=x.dtype,
                    runner_backend=MoeRunnerBackend.MARLIN,
                    inplace=False,
                )
                quant_info = MarlinMoeQuantInfo(
                    w13_qweight=w1,
                    w2_qweight=w2,
                    w13_scales=s1,
                    w2_scales=s2,
                    w13_g_idx_sort_indices=p1,
                    w2_g_idx_sort_indices=p2,
                    weight_bits=num_bits,
                    w13_g_idx=g1,
                    w2_g_idx=g2,
                    w13_qzeros=z1,
                    w2_qzeros=z2,
                )
                if runner_kind == "stock_lora":
                    from sglang.srt.lora.lora_moe_runner_marlin import (
                        MarlinLoraRunnerCore,
                    )
                    from sglang.srt.lora.lora_moe_runners import LoRAHooks

                    core = MarlinLoraRunnerCore(config)

                    def after_down(activation, projected, weights, ids):
                        hook_weights[:] = [weights, ids]

                    hooks = LoRAHooks(after_down=after_down)
                else:
                    from sglang.srt.lora.marlin_lora_temp.moe_runner import (
                        _use_shared_outer_factorization,
                        fused_experts_experimental_sgl_marlin_lora,
                    )

                    # A zero adapter must preserve base output, including NaN-padded
                    # routing weights consumed by the factored rank-sum reduction.
                    rank = 8
                    lora_info = SimpleNamespace(
                        lora_use_virtual_experts=True,
                        max_lora_rank=rank,
                        has_active_lora=True,
                        experts_shared_outer_loras=True,
                        gate_up_lora_a_weights=x.new_zeros((1, 1, 2 * rank, hidden)),
                        gate_up_lora_b_weights=x.new_zeros(
                            (1, experts, 2 * intermediate, rank)
                        ),
                        down_lora_a_weights=x.new_zeros(
                            (1, experts, rank, intermediate)
                        ),
                        down_lora_b_weights=x.new_zeros((1, 1, hidden, rank)),
                        token_lora_mapping=torch.zeros(
                            rows, device="cuda", dtype=torch.int32
                        ),
                    )
                    assert _use_shared_outer_factorization(lora_info, rows, topk)

            def run():
                with forward_context(
                    ForwardContext(attn_backend=None), forward_batch=batch
                ):
                    routing = router(x, logits)
                    if runner_kind != "fused":
                        dispatch = StandardDispatchOutput(x, None, routing)
                        if runner_kind == "stock_lora":
                            return core.run_from_dispatch(
                                dispatch, quant_info, config, hooks
                            ).hidden_states
                        return fused_experts_experimental_sgl_marlin_lora(
                            dispatch, quant_info, config, lora_info
                        ).hidden_states
                    return fused_marlin_moe(
                        x,
                        w1,
                        w2,
                        s1,
                        s2,
                        logits,
                        routing.topk_weights,
                        routing.topk_ids,
                        g_idx1=g1,
                        g_idx2=g2,
                        sort_indices1=p1,
                        sort_indices2=p2,
                        w1_zeros=z1,
                        w2_zeros=z2,
                        num_bits=num_bits,
                    )

            weights, ids = torch.topk(torch.softmax(logits, dim=-1), topk)
            weights /= weights.sum(dim=-1, keepdim=True)
            gate_up = torch.einsum("mh,mtih->mti", x, w1_ref[ids])
            gate, up = gate_up.chunk(2, dim=-1)
            activated = torch.nn.functional.silu(gate) * up
            projected = torch.einsum("mti,mthi->mth", activated, w2_ref[ids])
            reference = (projected.float() * weights[:, :, None]).sum(1).to(x.dtype)
            with patch.object(zero_copy_context, "get_moe_output", return_value=output):
                for _ in range(3):
                    run()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = run()
                for count in (max(rows - 1, 0), rows, 0, min(rows, 2)):
                    logits.copy_(finite_logits)
                    logits[count:].fill_(float("nan"))
                    live.fill_(count)
                    actual.fill_(float("nan"))
                    graph.replay()
                    expected = reference.clone()
                    expected[count:].zero_()
                    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.02)
                    assert torch.isfinite(actual).all()
                    assert torch.count_nonzero(actual[count:]) == 0
                    if hook_weights:
                        weights, ids = hook_weights
                        assert torch.count_nonzero(weights[ids < 0]) == 0


def test_moe_wna16_marlin_inactive_output_ownership():
    """An allocated output zeros inactive rows; caller-owned rows stay untouched."""
    torch.manual_seed(57)
    m, e, k, n, topk, block_m = 1, 8, 256, 256, 2, 8
    _, qweight, scales, zeros, g_idx, permutation = _marlin_inactive_weights(
        e, n, k, scalar_types.uint4b8
    )
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    ids = torch.full((m, topk), -1, device="cuda", dtype=torch.int32)
    routing_weights = torch.ones((m, topk), device="cuda", dtype=torch.float32)
    sorted_ids, expert_ids, num_post = moe_align_block_size(ids, block_m, e)
    workspace = torch.zeros(
        torch.cuda.get_device_properties(x.device).multi_processor_count * 4,
        device="cuda",
        dtype=torch.int32,
    )
    caller_output = torch.full((m * topk, n), 3.0, device="cuda", dtype=x.dtype)
    original_empty = torch.empty

    def poisoned_empty(*args, **kwargs):
        tensor = original_empty(*args, **kwargs)
        if tensor.is_floating_point():
            tensor.fill_(float("nan"))
        return tensor

    for output in (None, caller_output):
        with patch.object(torch, "empty", side_effect=poisoned_empty):
            result = _run_single_gemm(
                moe_wna16_marlin_gemm,
                x,
                output,
                qweight,
                scales,
                zeros,
                g_idx,
                permutation,
                workspace,
                sorted_ids,
                expert_ids,
                num_post,
                routing_weights,
                scalar_types.uint4b8,
                block_m,
                topk,
                m,
                n,
                k,
                False,
                True,
                True,
            )
        expected = (
            torch.zeros_like(result) if output is None else torch.full_like(result, 3.0)
        )
        torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_marlin_lora_direct_inactive_routes():
    """Direct LoRA projections overwrite inactive routes with finite zeros."""
    from sglang.srt.lora.marlin_lora_temp.direct_decode import (
        direct_decode_down_shrink,
        direct_decode_gate_expand,
    )

    torch.manual_seed(51)
    rows, topk, experts, intermediate, rank = 3, 6, 8, 384, 32
    ids = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [-1, -1, -1, -1, -1, -1], [2, -1, 4, 5, -1, 7]],
        device="cuda",
        dtype=torch.int32,
    )
    slots = torch.zeros(rows, device="cuda", dtype=torch.int32)
    shared = torch.randn((rows, 2 * rank), device="cuda", dtype=torch.bfloat16)
    gate_b = (
        torch.randn(
            (1, experts, 2 * intermediate, rank), device="cuda", dtype=torch.bfloat16
        )
        * 0.05
    )
    down_a = (
        torch.randn(
            (1, experts, rank, intermediate), device="cuda", dtype=torch.bfloat16
        )
        * 0.05
    )
    activation = torch.randn(
        (rows * topk, intermediate), device="cuda", dtype=torch.bfloat16
    )
    activation.view(rows, topk, intermediate)[ids < 0] = float("nan")
    gate_output = torch.full(
        (rows, topk, 2 * intermediate),
        float("nan"),
        device="cuda",
        dtype=torch.bfloat16,
    )
    down_output = torch.full(
        (rows, topk, rank), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    direct_decode_gate_expand(shared, gate_b, ids, slots, gate_output)
    direct_decode_down_shrink(activation, down_a, ids, slots, down_output)
    expected_gate = torch.zeros_like(gate_output)
    expected_down = torch.zeros_like(down_output)
    for token in range(rows):
        for route in range(topk):
            expert = int(ids[token, route])
            if expert < 0:
                continue
            for half in range(2):
                start = half * intermediate
                expected_gate[token, route, start : start + intermediate] = (
                    gate_b[0, expert, start : start + intermediate].float()
                    @ shared[token, half * rank : (half + 1) * rank].float()
                ).to(gate_output.dtype)
            expected_down[token, route] = (
                down_a[0, expert].float() @ activation[token * topk + route].float()
            ).to(down_output.dtype)
    torch.testing.assert_close(gate_output, expected_gate, rtol=0.03, atol=0.01)
    torch.testing.assert_close(down_output, expected_down, rtol=0.03, atol=0.01)
    assert torch.count_nonzero(gate_output[ids < 0]) == 0
    assert torch.count_nonzero(down_output[ids < 0]) == 0


@pytest.mark.parametrize("fused_norm", [False, True])
def test_muse_marlin_inactive_routes(fused_norm):
    """Muse maps global routes once and masks inactive weights before either norm tail."""
    from sglang.srt.layers.muse_spark.muse_spark_fp4_moe import MuseSpark1xFp4Experts

    torch.manual_seed(67)
    rows, local_experts, hidden, intermediate, topk = 4, 2, 256, 256, 2
    dtype = torch.bfloat16
    layer = MuseSpark1xFp4Experts.__new__(MuseSpark1xFp4Experts)
    torch.nn.Module.__init__(layer)
    layer.quant_config = SimpleNamespace(group_size=16)
    layer.moe_runner_config = SimpleNamespace(is_gated=True)
    layer.params_dtype = dtype
    layer.intermediate_size_per_partition = intermediate
    layer.num_local_experts = local_experts
    layer.global_num_experts = 4
    layer._expert_map = torch.tensor([-1, -1, 0, 1], device="cuda", dtype=torch.int32)
    references = []
    for prefix, n, k in (
        ("w13", 2 * intermediate, hidden),
        ("w2", hidden, intermediate),
    ):
        packed, scales, global_scales, refs = zip(
            *[make_nvfp4_weight_and_ref(n, k, dtype) for _ in range(local_experts)]
        )
        for suffix, tensors in (
            ("weight", packed),
            ("weight_scale", scales),
            ("weight_scale_2", global_scales),
        ):
            setattr(
                layer,
                f"{prefix}_{suffix}",
                torch.nn.Parameter(torch.stack(tensors), requires_grad=False),
            )
        references.append(torch.stack(refs))
    prepare_moe_nvfp4_layer_for_marlin(layer)
    x = torch.randn((rows, hidden), device="cuda", dtype=dtype) / 20
    ids = torch.tensor(
        [[2, 3], [3, -1], [0, 2], [-1, -1]], device="cuda", dtype=torch.int32
    )
    local_ids = torch.tensor(
        [[0, 1], [1, -1], [-1, 0], [-1, -1]], device="cuda", dtype=torch.int32
    )
    weights = torch.tensor(
        [[0.25, 0.75], [0.5, 0.5], [0.25, 0.75], [0.5, 0.5]], device="cuda"
    )
    weights[local_ids < 0] = float("nan")
    norm = torch.nn.RMSNorm(hidden, eps=1e-6, device="cuda", dtype=dtype)
    layer._post_expert_norm = norm
    layer._post_expert_norm_weight = norm.weight.detach() if fused_norm else None
    layer._post_expert_norm_eps = norm.eps
    layer._post_expert_norm_weight_first = False

    expert_output, mapped = layer._marlin_expert_outputs(x, weights, ids)
    torch.testing.assert_close(mapped, local_ids, rtol=0, atol=0)
    assert torch.count_nonzero(expert_output[local_ids < 0]) == 0
    expected_experts = torch.zeros((rows, topk, hidden), device="cuda", dtype=dtype)
    w13_ref, w2_ref = references
    for row in range(rows):
        for route in range(topk):
            expert = int(local_ids[row, route])
            if expert < 0:
                continue
            gate_up = x[row] @ w13_ref[expert].T
            gate, up = gate_up.chunk(2)
            activated = torch.nn.functional.silu(gate) * up
            expected_experts[row, route] = activated @ w2_ref[expert].T
    torch.testing.assert_close(expert_output, expected_experts, rtol=0.05, atol=0.25)
    safe_weights = weights.masked_fill(local_ids < 0, 0.0)
    expected = (
        (norm(expected_experts).float() * safe_weights[..., None]).sum(1).to(dtype)
    )
    actual = layer._forward_marlin(x, weights, ids, norm)
    torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.25)
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[-1]) == 0
    assert torch.isnan(weights[local_ids < 0]).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
