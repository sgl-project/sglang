# SPDX-License-Identifier: Apache-2.0
"""End-to-end correctness tests for standard-dispatch contiguous DeepGEMM."""

from contextlib import nullcontext

import pytest
import torch

import sglang.srt.layers.moe.moe_runner.deep_gemm as deep_gemm_runner
from sglang.kernels.ops.moe.ep_moe_kernels import (
    standard_contig_all_tokens_upper_bound,
)
from sglang.srt.environ import envs
from sglang.srt.layers.deep_gemm_wrapper import compile_utils
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.deep_gemm import (
    DeepGemmMoeQuantInfo,
    DeepGemmRunnerCore,
    _post_permute_deep_gemm_to_standard_contig,
    _post_permute_deep_gemm_to_standard_masked,
    _pre_permute_standard_to_deep_gemm_contig,
    _pre_permute_standard_to_deep_gemm_masked,
    _select_contiguous_gemm_options,
    _should_use_standard_contiguous_layout,
)
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60,
    stage="base-b-kernel-unit",
    runner_config="4-gpu-b200",
)

M3_NUM_ROUTED_EXPERTS = 128
M3_NUM_EXPERTS = 129
M3_HIDDEN_SIZE = 6144
M3_INTERMEDIATE_SIZE_PER_TP = 384


def test_standard_contiguous_layout_is_opt_in():
    assert envs.SGLANG_ENABLE_DEEPGEMM_PURE_TP_CONTIGUOUS_MOE.default is False

    with envs.SGLANG_ENABLE_DEEPGEMM_PURE_TP_CONTIGUOUS_MOE.override(False):
        assert not _should_use_standard_contiguous_layout(MoeRunnerConfig())


def test_standard_contiguous_layout_requires_pure_tp():
    with envs.SGLANG_ENABLE_DEEPGEMM_PURE_TP_CONTIGUOUS_MOE.override(True):
        assert _should_use_standard_contiguous_layout(MoeRunnerConfig(moe_ep_size=1))
        assert not _should_use_standard_contiguous_layout(
            MoeRunnerConfig(moe_ep_size=2)
        )


@pytest.mark.parametrize(
    "tokens,top_k,experts,shared,block_e,expected",
    [
        (1, 5, 129, 1, 128, 640),
        (32, 5, 129, 1, 128, 16512),
        (33, 5, 129, 1, 128, 16512),
        (128, 5, 129, 1, 128, 16896),
        (129, 5, 129, 1, 128, 17024),
        (16, 6, 130, 2, 128, 8448),
    ],
)
def test_standard_contig_shared_expert_upper_bound(
    tokens, top_k, experts, shared, block_e, expected
):
    assert (
        standard_contig_all_tokens_upper_bound(
            tokens * top_k,
            experts,
            block_e,
            top_k=top_k,
            num_fused_shared_experts=shared,
        )
        == expected
    )


def _make_packed_scale(
    num_experts: int,
    n: int,
    k: int,
    block_size: int,
) -> torch.Tensor:
    from deep_gemm.utils.layout import (
        get_mn_major_tma_aligned_packed_ue8m0_tensor,
    )

    scale = torch.ones(
        (num_experts, n, k // block_size),
        device="cuda",
        dtype=torch.float32,
    )
    return get_mn_major_tma_aligned_packed_ue8m0_tensor(scale)


@pytest.fixture(scope="module")
def m3_quant_info() -> DeepGemmMoeQuantInfo:
    torch.manual_seed(1)
    block_size = 32
    gateup_size = M3_INTERMEDIATE_SIZE_PER_TP * 2
    w13_weight = (
        torch.randn(
            (M3_NUM_EXPERTS, gateup_size, M3_HIDDEN_SIZE),
            device="cuda",
            dtype=torch.float32,
        )
        .mul_(0.125)
        .to(torch.float8_e4m3fn)
    )
    w2_weight = (
        torch.randn(
            (
                M3_NUM_EXPERTS,
                M3_HIDDEN_SIZE,
                M3_INTERMEDIATE_SIZE_PER_TP,
            ),
            device="cuda",
            dtype=torch.float32,
        )
        .mul_(0.125)
        .to(torch.float8_e4m3fn)
    )
    return DeepGemmMoeQuantInfo(
        w13_weight=w13_weight,
        w2_weight=w2_weight,
        use_fp8=True,
        w13_scale=_make_packed_scale(
            M3_NUM_EXPERTS,
            gateup_size,
            M3_HIDDEN_SIZE,
            block_size,
        ),
        w2_scale=_make_packed_scale(
            M3_NUM_EXPERTS,
            M3_HIDDEN_SIZE,
            M3_INTERMEDIATE_SIZE_PER_TP,
            block_size,
        ),
        block_shape=[1, block_size],
        use_mxfp8=True,
    )


@pytest.fixture(scope="module")
def m3_config() -> MoeRunnerConfig:
    return MoeRunnerConfig(
        num_experts=M3_NUM_EXPERTS,
        num_local_experts=M3_NUM_EXPERTS,
        hidden_size=M3_HIDDEN_SIZE,
        intermediate_size_per_partition=M3_INTERMEDIATE_SIZE_PER_TP,
        layer_id=0,
        top_k=5,
        num_fused_shared_experts=1,
        params_dtype=torch.bfloat16,
        moe_ep_size=1,
        activation="silu",
        is_gated=True,
        routed_scaling_factor=2.0,
        gemm1_alpha=1.702,
        gemm1_clamp_limit=7.0,
        gate_up_interleaved=False,
    )


def _make_dispatch(num_tokens: int) -> StandardDispatchOutput:
    torch.manual_seed(100 + num_tokens)
    hidden_states = torch.randn(
        (num_tokens, M3_HIDDEN_SIZE),
        device="cuda",
        dtype=torch.bfloat16,
    )
    routed_ids = (
        torch.rand((num_tokens, M3_NUM_ROUTED_EXPERTS), device="cuda")
        .topk(4, dim=-1)
        .indices.to(torch.int32)
    )
    shared_ids = torch.full(
        (num_tokens, 1),
        M3_NUM_ROUTED_EXPERTS,
        device="cuda",
        dtype=torch.int32,
    )
    topk_ids = torch.cat((routed_ids, shared_ids), dim=-1)

    routed_weights = torch.rand(
        (num_tokens, 4),
        device="cuda",
        dtype=torch.float32,
    )
    routed_weights /= routed_weights.sum(dim=-1, keepdim=True)
    shared_weights = torch.full(
        (num_tokens, 1),
        0.5,
        device="cuda",
        dtype=torch.float32,
    )
    topk_weights = torch.cat((routed_weights, shared_weights), dim=-1)
    return StandardDispatchOutput(
        hidden_states,
        None,
        StandardTopKOutput(topk_weights, topk_ids, None),
    )


def _run_standard_deep_gemm(
    dispatch_output: StandardDispatchOutput,
    quant_info: DeepGemmMoeQuantInfo,
    config: MoeRunnerConfig,
    *,
    contiguous: bool,
    expected_psum: bool = False,
    expected_gemm2_zero_padding: bool = True,
) -> torch.Tensor:
    running_state = {}
    pre_permute = (
        _pre_permute_standard_to_deep_gemm_contig
        if contiguous
        else _pre_permute_standard_to_deep_gemm_masked
    )
    post_permute = (
        _post_permute_deep_gemm_to_standard_contig
        if contiguous
        else _post_permute_deep_gemm_to_standard_masked
    )
    runner_input = pre_permute(
        dispatch_output,
        quant_info,
        config,
        running_state,
    )

    if contiguous:
        (
            grouped_layout,
            use_psum_layout,
            gemm1_zero_padding,
            gemm2_zero_padding,
        ) = _select_contiguous_gemm_options(
            runner_input,
            running_state,
            config,
            quant_info.w13_weight.size(1),
        )
        expected_layout = (
            runner_input.psum_layout if expected_psum else runner_input.m_indices
        )
        assert grouped_layout is expected_layout
        assert use_psum_layout is expected_psum
        assert gemm1_zero_padding is not expected_psum
        assert gemm2_zero_padding is expected_gemm2_zero_padding

    runner_output = DeepGemmRunnerCore(config).run(
        runner_input,
        quant_info,
        running_state,
    )
    return post_permute(
        runner_output,
        quant_info,
        config,
        running_state,
    ).hidden_states


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="MXFP8 DeepGEMM requires SM100+",
)
@pytest.mark.parametrize(
    "num_tokens,expected_psum,expected_gemm2_zero_padding",
    [
        pytest.param(1, False, True, id="baseline"),
        pytest.param(5, False, True, id="before-psum"),
        pytest.param(6, True, False, id="after-psum"),
        pytest.param(25, True, False, id="presence-bincount"),
        pytest.param(26, True, False, id="atomic-bincount"),
        pytest.param(8192, True, False, id="before-gemm2-full-tail"),
        pytest.param(8193, True, True, id="after-gemm2-full-tail"),
    ],
)
@torch.inference_mode()
def test_standard_contiguous_deep_gemm_matches_masked(
    monkeypatch,
    m3_quant_info,
    m3_config,
    num_tokens,
    expected_psum,
    expected_gemm2_zero_padding,
):
    monkeypatch.setattr(
        compile_utils,
        "_ENABLE_JIT_DEEPGEMM_PRECOMPILE",
        False,
    )
    monkeypatch.setattr(deep_gemm_runner, "get_tp_group", lambda: None)
    monkeypatch.setattr(
        deep_gemm_runner,
        "is_allocation_symmetric",
        lambda: False,
    )
    monkeypatch.setattr(
        deep_gemm_runner,
        "use_symmetric_memory",
        lambda *args, **kwargs: nullcontext(),
    )
    masked = _run_standard_deep_gemm(
        _make_dispatch(num_tokens),
        m3_quant_info,
        m3_config,
        contiguous=False,
    )
    contiguous = _run_standard_deep_gemm(
        _make_dispatch(num_tokens),
        m3_quant_info,
        m3_config,
        contiguous=True,
        expected_psum=expected_psum,
        expected_gemm2_zero_padding=expected_gemm2_zero_padding,
    )

    assert torch.isfinite(masked).all()
    assert torch.count_nonzero(masked).item() > 0
    torch.testing.assert_close(contiguous, masked, rtol=2e-2, atol=2e-2)
