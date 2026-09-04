import random
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import sglang.srt.layers.moe.moe_runner.deep_gemm as deep_gemm_runner
from sglang.kernels.ops.moe.ep_moe_kernels import (
    fill_gateup_input_triton_kernel,
    moe_ep_deepgemm_preprocess,
)
from sglang.kernels.ops.quantization.minimax_quant_ue8m0 import (
    per_token_quant_fp8_ue8m0,
    per_token_quant_fp8_ue8m0_scatter,
)
from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.deep_gemm import (
    DeepGemmMoeQuantInfo,
    DeepGemmRunnerCore,
    post_permute_deep_gemm_to_standard,
    pre_permute_standard_to_deep_gemm,
)
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.quantization.fp8_utils import (
    quant_weight_ue8m0,
    transform_scale_ue8m0,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

dev = "cuda"


@pytest.mark.parametrize("num_tokens", [1, 7, 64, 256])
@pytest.mark.parametrize("topk", [4, 5, 8])
@pytest.mark.parametrize("hidden,group", [(6144, 32), (2048, 32), (4096, 128)])
def test_quant_scatter_matches_quant_plus_fill(num_tokens, topk, hidden, group):
    arch_major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
    if arch_major <= 9:
        pytest.skip("UE8M0 fusion is Blackwell-only")

    E = 129  # 128 routed + 1 fused shared
    G4 = (hidden // group) // 4
    m_max = (num_tokens // 256 + 1) * 256
    torch.manual_seed(num_tokens * 91 + topk + hidden)
    random.seed(num_tokens)

    x = (torch.randn(num_tokens, hidden, device=dev, dtype=torch.bfloat16)) * 4.0

    tids = torch.empty(num_tokens, topk, dtype=torch.int32, device=dev)
    tids_cpu = torch.empty(num_tokens, topk, dtype=torch.int32)
    s2d = [0] * (num_tokens * topk)
    cur = [0] * E
    for t in range(num_tokens):
        for j, e in enumerate(random.sample(range(E), topk)):
            tids_cpu[t, j] = e
            s2d[t * topk + j] = e * m_max + cur[e]
            cur[e] += 1
    tids.copy_(tids_cpu)
    s2d = torch.tensor(s2d, dtype=torch.int32, device=dev)

    x_q, x_sf = per_token_quant_fp8_ue8m0(x, group)
    gi_ref = torch.zeros(E, m_max, hidden, device=dev, dtype=torch.float8_e4m3fn)
    gs_ref = torch.zeros(E, G4, m_max, device=dev, dtype=torch.int32)
    fill_gateup_input_triton_kernel[(num_tokens,)](
        x_q,
        x_sf,
        gi_ref,
        gs_ref,
        s2d,
        tids,
        topk,
        hidden,
        G4,
        m_max,
        x_sf.stride(0),
        x_sf.stride(1),
        BLOCK_SIZE=1024,
        SCALE_MN_MAJOR=True,
    )

    gi_new = torch.zeros(E, m_max, hidden, device=dev, dtype=torch.float8_e4m3fn)
    gs_new = torch.zeros(E, G4, m_max, device=dev, dtype=torch.int32)
    per_token_quant_fp8_ue8m0_scatter(x, gi_new, gs_new, s2d, tids, topk, m_max, group)

    for t in range(num_tokens):
        for j in range(topk):
            e = int(tids_cpu[t, j])
            m = int(s2d[t * topk + j]) % m_max
            assert torch.equal(
                gi_new[e, m].view(torch.uint8), gi_ref[e, m].view(torch.uint8)
            ), f"fp8 mismatch token={t} slot={j} expert={e}"
            assert torch.equal(gs_new[e, :, m], gs_ref[e, :, m]), (
                f"scale mismatch token={t} slot={j} expert={e}"
            )


def test_standard_deepgemm_preprocess_quantizes_with_ue8m0_scale():
    arch_major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
    if arch_major <= 9:
        pytest.skip("UE8M0 fusion is Blackwell-only")

    num_tokens, topk, hidden, group, num_experts = 7, 4, 2048, 128, 8
    torch.manual_seed(1234)
    x = torch.randn(num_tokens, hidden, device=dev, dtype=torch.bfloat16)
    topk_ids = torch.stack(
        [
            (torch.arange(topk, device=dev, dtype=torch.int32) + token) % num_experts
            for token in range(num_tokens)
        ]
    )

    _, _, src2dst, grouped_x, grouped_scale = moe_ep_deepgemm_preprocess(
        topk_ids=topk_ids,
        num_local_experts=num_experts,
        hidden_states=x,
        top_k=topk,
        block_shape=[group, group],
        output_dtype=torch.float8_e4m3fn,
        use_mxfp8=False,
    )
    direct_x, direct_scale = per_token_quant_fp8_ue8m0(x, group)

    assert grouped_scale.dtype == torch.int32
    for token in range(num_tokens):
        for slot in range(topk):
            dst = int(src2dst[token * topk + slot])
            expert, row = divmod(dst, grouped_x.shape[1])
            assert torch.equal(
                grouped_x[expert, row].view(torch.uint8),
                direct_x[token].view(torch.uint8),
            )
            assert torch.equal(
                grouped_scale[expert, row],
                direct_scale[token],
            )


@pytest.mark.parametrize(
    "num_assignments,num_experts,expected",
    [
        (14, 2, 256),
        (10, 512, 1280),
        (20, 512, 2560),
        (320, 512, 40960),
        (640, 512, 65664),
        (1280, 512, 66304),
    ],
)
def test_compact_all_tokens_uses_tight_routing_independent_bound(
    num_assignments, num_experts, expected
):
    assert (
        deep_gemm_runner._get_compact_all_tokens(num_assignments, num_experts)
        == expected
    )


def test_compact_eager_keeps_masked_layout_for_cuda_graph(monkeypatch):
    config = MoeRunnerConfig(
        num_experts=128,
        num_local_experts=16,
        hidden_size=2048,
        intermediate_size_per_partition=4096,
        top_k=4,
        activation="silu",
        is_gated=True,
        inplace=False,
    )
    monkeypatch.setattr(
        deep_gemm_runner.envs.SGLANG_OPT_DG_COMPACT_EAGER, "get", lambda: True
    )
    capture = SimpleNamespace(disable_dispose_tensor=False)
    monkeypatch.setattr(
        deep_gemm_runner, "get_flags", lambda: SimpleNamespace(capture=capture)
    )
    hidden_states = torch.empty((128, 2048), device="meta")
    quant_info = DeepGemmMoeQuantInfo(
        w13_weight=torch.empty((1, 4096, 1), dtype=torch.float8_e4m3fn),
        w2_weight=torch.empty((1, 2048, 1), dtype=torch.float8_e4m3fn),
        use_fp8=True,
        block_shape=[128, 128],
    )
    with envs.SGLANG_DEEPGEMM_STANDARD_LAYOUT.override("masked"):
        assert not deep_gemm_runner._should_use_masked_standard_layout(
            config, quant_info, hidden_states
        )

        capture.disable_dispose_tensor = True
        assert deep_gemm_runner._should_use_masked_standard_layout(
            config, quant_info, hidden_states
        )


def test_standard_layout_auto_memory_policy(monkeypatch):
    config = MoeRunnerConfig(
        num_experts=512,
        num_local_experts=512,
        hidden_size=4096,
        intermediate_size_per_partition=256,
        top_k=8,
    )
    quant_info = DeepGemmMoeQuantInfo(
        w13_weight=torch.empty((1, 512, 1), dtype=torch.float8_e4m3fn),
        w2_weight=torch.empty((1, 4096, 1), dtype=torch.float8_e4m3fn),
        use_fp8=True,
        block_shape=[128, 128],
    )
    monkeypatch.setattr(
        deep_gemm_runner,
        "_masked_standard_layout_memory_budget_bytes",
        int(42.5 * (1 << 30)),
    )

    with envs.SGLANG_DEEPGEMM_STANDARD_LAYOUT.override("auto"):
        for num_tokens, expected in ((8192, True), (16384, False)):
            hidden_states = torch.empty((num_tokens, 4096), device="meta")
            assert (
                deep_gemm_runner._should_use_masked_standard_layout(
                    config, quant_info, hidden_states
                )
                is expected
            )


@pytest.mark.parametrize("weight_dtype", ["fp8", "bf16"])
def test_standard_masked_runner_matches_compact_end_to_end(monkeypatch, weight_dtype):
    """Exercise both production grouped GEMMs through the standard path."""
    arch_major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
    if arch_major <= 9:
        pytest.skip("DeepGEMM UE8M0 is Blackwell-only")

    # This kernel test runs outside a model-parallel process. Bypass only the
    # symmetric-allocation context; all pre-permute, DeepGEMM, activation,
    # quantization, down-GEMM, and post-permute kernels remain real.
    monkeypatch.setattr(deep_gemm_runner, "get_tp_group", lambda: None)
    monkeypatch.setattr(
        deep_gemm_runner,
        "use_symmetric_memory",
        lambda *args, **kwargs: nullcontext(),
    )

    # UE8M0 packs four 128-wide scale groups into each int32. Use the smallest
    # legal K for both the gate/up and down GEMMs.
    num_tokens, hidden, intermediate, topk, num_local_experts = 7, 512, 512, 2, 2
    torch.manual_seed(20260730)
    hidden_states = torch.randn(num_tokens, hidden, device=dev, dtype=torch.bfloat16)
    topk_ids = torch.tensor(
        [
            [0, -1],
            [1, -1],
            [0, 1],
            [-1, 1],
            [0, -1],
            [1, 0],
            [-1, 1],
        ],
        device=dev,
        dtype=torch.int32,
    )
    topk_weights = torch.tensor(
        [
            [0.8, 0.2],
            [0.7, 0.3],
            [0.6, 0.4],
            [0.1, 0.9],
            [0.75, 0.25],
            [0.55, 0.45],
            [0.35, 0.65],
        ],
        device=dev,
        dtype=torch.float32,
    )

    weight_std = hidden**-0.5
    w13_bf16 = (
        torch.randn(
            num_local_experts,
            2 * intermediate,
            hidden,
            device=dev,
            dtype=torch.bfloat16,
        )
        * weight_std
    )
    w2_bf16 = (
        torch.randn(
            num_local_experts,
            hidden,
            intermediate,
            device=dev,
            dtype=torch.bfloat16,
        )
        * weight_std
    )
    if weight_dtype == "fp8":
        w13, w13_scale = quant_weight_ue8m0(w13_bf16, [128, 128])
        w2, w2_scale = quant_weight_ue8m0(w2_bf16, [128, 128])
        quant_info = DeepGemmMoeQuantInfo(
            w13_weight=w13,
            w2_weight=w2,
            use_fp8=True,
            w13_scale=transform_scale_ue8m0(w13_scale, mn=w13.shape[-2]),
            w2_scale=transform_scale_ue8m0(w2_scale, mn=w2.shape[-2]),
            block_shape=[128, 128],
        )
    else:
        quant_info = DeepGemmMoeQuantInfo(
            w13_weight=w13_bf16,
            w2_weight=w2_bf16,
            use_fp8=False,
        )

    dispatch_output = StandardDispatchOutput(
        hidden_states=hidden_states,
        hidden_states_scale=None,
        topk_output=(topk_weights, topk_ids, None),
    )

    def run_with_layout(layout):
        config = MoeRunnerConfig(
            num_experts=8,
            num_local_experts=num_local_experts,
            hidden_size=hidden,
            intermediate_size_per_partition=intermediate,
            top_k=topk,
            activation="silu",
            is_gated=True,
            inplace=False,
        )
        running_state = {}
        with envs.SGLANG_DEEPGEMM_STANDARD_LAYOUT.override(layout):
            runner_input = pre_permute_standard_to_deep_gemm(
                dispatch_output,
                quant_info,
                config,
                running_state,
            )
        runner_output = DeepGemmRunnerCore(config).run(
            runner_input,
            quant_info,
            running_state,
        )
        return (
            runner_input.use_masked_gemm,
            running_state.get("all_tokens"),
            runner_input.m_indices,
            post_permute_deep_gemm_to_standard(
                runner_output,
                quant_info,
                config,
                running_state,
            ).hidden_states,
        )

    compact_is_masked, compact_all_tokens, compact_m_indices, compact_output = (
        run_with_layout("compact")
    )
    masked_is_masked, masked_all_tokens, masked_m_indices, masked_output = (
        run_with_layout("masked")
    )
    torch.cuda.synchronize()

    assert not compact_is_masked
    assert masked_is_masked
    assert compact_all_tokens == 256
    assert masked_all_tokens is None
    assert masked_m_indices is None
    valid_assignments = topk_ids[topk_ids >= 0]
    assert torch.equal(
        torch.bincount(
            compact_m_indices[compact_m_indices >= 0],
            minlength=num_local_experts,
        ),
        torch.bincount(valid_assignments, minlength=num_local_experts),
    )
    assert (compact_m_indices == -1).sum() == compact_all_tokens - len(
        valid_assignments
    )
    torch.testing.assert_close(
        masked_output,
        compact_output,
        rtol=5e-2,
        atol=5e-2,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-x"]))
