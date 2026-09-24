"""Cross-backend correctness tests for SGLang CUTLASS and FlashInfer Humming."""

import pytest
import torch
from flashinfer.fused_moe import (
    cutlass_fused_moe,
    preprocess_moe_weights_for_sm90_mixed_gemm_humming,
)
from flashinfer.fused_moe.core import ActivationType
from utils import is_hopper

from sglang.srt.layers.moe.cutlass_mxfp4a8_fused_moe import (
    CutlassMxfp4A8FusedMoeRunner,
)
from sglang.srt.runtime_context import get_parallel


def _prepare_weight_views(num_experts, hidden_size, intermediate_size, device):
    """Build both backend layouts from one logical MXFP4 weight set."""
    w1 = torch.randint(
        0,
        256,
        (num_experts, 2 * intermediate_size, hidden_size // 2),
        dtype=torch.uint8,
        device=device,
    )
    w2 = torch.randint(
        0,
        256,
        (num_experts, hidden_size, intermediate_size // 2),
        dtype=torch.uint8,
        device=device,
    )
    # Keep each expert's E8M0 range below Humming's max_range=11 so no value
    # is clipped. Both backends therefore consume the same logical weights.
    w1_scale = torch.randint(
        124,
        131,
        (num_experts, 2 * intermediate_size, hidden_size // 32),
        dtype=torch.uint8,
        device=device,
    )
    w2_scale = torch.randint(
        124,
        131,
        (num_experts, hidden_size, intermediate_size // 32),
        dtype=torch.uint8,
        device=device,
    )

    # SGLang CUTLASS consumes [gate; up], while FlashInfer Humming consumes
    # [up; gate]. The payload and scale values are otherwise identical.
    w1_up_gate = torch.cat(
        (w1[:, intermediate_size:], w1[:, :intermediate_size]), dim=1
    ).contiguous()
    w1_scale_up_gate = torch.cat(
        (
            w1_scale[:, intermediate_size:],
            w1_scale[:, :intermediate_size],
        ),
        dim=1,
    ).contiguous()

    ours_w1, ours_s1, ours_r1 = (
        preprocess_moe_weights_for_sm90_mixed_gemm_humming(w1, w1_scale)
    )
    humming_w1, humming_s1, humming_r1 = (
        preprocess_moe_weights_for_sm90_mixed_gemm_humming(
            w1_up_gate, w1_scale_up_gate
        )
    )
    ours_w2, ours_s2, ours_r2 = (
        preprocess_moe_weights_for_sm90_mixed_gemm_humming(w2, w2_scale)
    )
    humming_w2, humming_s2, humming_r2 = (
        preprocess_moe_weights_for_sm90_mixed_gemm_humming(w2, w2_scale)
    )
    return (
        (ours_w1, ours_s1, ours_r1, ours_w2, ours_s2, ours_r2),
        (
            humming_w1,
            humming_s1,
            humming_r1,
            humming_w2,
            humming_s2,
            humming_r2,
        ),
    )


def _make_routing(num_tokens, num_experts, topk, device):
    rows = torch.arange(num_tokens, device=device)
    topk_ids = torch.stack(
        tuple((rows + offset) % num_experts for offset in range(topk)), dim=1
    ).to(torch.int32)
    topk_weights = torch.rand(
        (num_tokens, topk), dtype=torch.float32, device=device
    )
    topk_weights /= topk_weights.sum(dim=1, keepdim=True)
    return topk_ids, topk_weights


def _run_sglang_cutlass(
    x,
    topk_ids,
    topk_weights,
    weights,
    num_experts,
    intermediate_size,
):
    w1, s1, r1, w2, s2, r2 = weights
    hidden_size = x.shape[1]

    def strides(width):
        return torch.full(
            (num_experts, 3), width, dtype=torch.int64, device=x.device
        )

    a_strides1 = strides(hidden_size)
    c_strides1 = strides(2 * intermediate_size)
    a_strides2 = strides(intermediate_size)
    c_strides2 = strides(hidden_size)
    expert_offsets = torch.empty(
        num_experts + 1, dtype=torch.int32, device=x.device
    )
    problem_sizes1 = torch.empty(
        (num_experts, 3), dtype=torch.int32, device=x.device
    )
    problem_sizes2 = torch.empty_like(problem_sizes1)

    with get_parallel().override(moe_ep_size=1):
        return CutlassMxfp4A8FusedMoeRunner()(
            x,
            None,
            None,
            None,
            None,
            w1.view(torch.int8),
            w2.view(torch.int8),
            s1,
            s2,
            (r1 * 64.0).contiguous(),
            (r2 * 64.0).contiguous(),
            topk_weights,
            topk_ids,
            a_strides1,
            a_strides1,
            c_strides1,
            a_strides2,
            a_strides2,
            c_strides2,
            c_strides1,
            c_strides2,
            expert_offsets,
            problem_sizes1,
            problem_sizes2,
        )


def _run_flashinfer_humming(
    x,
    topk_ids,
    topk_weights,
    weights,
):
    w1, s1, r1, w2, s2, r2 = weights
    output = torch.empty_like(x)
    cutlass_fused_moe(
        input=x,
        token_selected_experts=topk_ids,
        token_final_scales=topk_weights,
        fc1_expert_weights=w1,
        fc2_expert_weights=w2,
        output_dtype=torch.bfloat16,
        quant_scales=[
            s1.view(torch.int32),
            (r1 * 64.0).contiguous(),
            torch.ones((), dtype=torch.float32, device=x.device),
            s2.view(torch.int32),
            (r2 * 64.0).contiguous(),
        ],
        use_w4_group_scaling=True,
        use_wfp4afp8_humming=True,
        tune_max_num_tokens=1 << (x.shape[0] - 1).bit_length(),
        activation_type=ActivationType.Swiglu,
        output=output,
        use_fused_finalize=True,
    )
    return output


@pytest.mark.skipif(
    not is_hopper(),
    reason="SGLang CUTLASS MXFP4A8 and FlashInfer Humming require SM90",
)
@pytest.mark.parametrize(
    "num_tokens",
    [
        pytest.param(1, id="decode"),
        pytest.param(16, id="small-prefill"),
        pytest.param(65, id="external-metadata"),
    ],
)
def test_cutlass_mxfp4a8_matches_flashinfer_humming(num_tokens):
    torch.manual_seed(20260924 + num_tokens)
    device = torch.device("cuda")
    num_experts = 8
    topk = 6  # DeepSeek-V4-Flash num_experts_per_tok
    hidden_size = 256
    intermediate_size = 256
    x = (
        torch.randn(
            num_tokens,
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.25
    ).contiguous()
    topk_ids, topk_weights = _make_routing(
        num_tokens, num_experts, topk, device
    )
    ours_weights, humming_weights = _prepare_weight_views(
        num_experts, hidden_size, intermediate_size, device
    )

    ours = _run_sglang_cutlass(
        x,
        topk_ids,
        topk_weights,
        ours_weights,
        num_experts,
        intermediate_size,
    )
    humming = _run_flashinfer_humming(
        x,
        topk_ids,
        topk_weights,
        humming_weights,
    )
    torch.cuda.synchronize()

    assert torch.isfinite(ours).all()
    assert torch.isfinite(humming).all()
    diff = ours.float() - humming.float()
    relative_l2 = diff.norm() / humming.float().norm().clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(
        ours.float().flatten(),
        humming.float().flatten(),
        dim=0,
    )
    assert relative_l2.item() < 0.025
    assert cosine.item() > 0.9995
