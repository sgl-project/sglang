"""Correctness coverage for direct-output MegaMoE front-end staging."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.dsv4 import (
    mask_topk_ids,
    mega_moe_pad_route,
    mega_moe_pre_dispatch,
    mega_moe_stage_activation,
)
from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _allocate_outputs(
    padded_max: int, hidden: int, topk: int, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.zeros((padded_max, hidden), dtype=torch.float8_e4m3fn, device="cuda"),
        torch.zeros(
            (padded_max, hidden // group_size // 4),
            dtype=torch.int32,
            device="cuda",
        ),
        torch.full((padded_max, topk), -777, dtype=torch.int64, device="cuda"),
        torch.full((padded_max, topk), float("nan"), device="cuda"),
    )


@pytest.mark.parametrize(
    "num_tokens,padded_max,hidden,group_size,topk",
    [
        (0, 16, 2048, 32, 6),
        (1, 1, 2048, 32, 6),
        (7, 64, 4096, 64, 4),
        (32, 32, 7168, 128, 8),
    ],
)
@torch.inference_mode()
def test_activation_staging_matches_combined(
    num_tokens: int,
    padded_max: int,
    hidden: int,
    group_size: int,
    topk: int,
) -> None:
    torch.manual_seed(num_tokens * 1009 + hidden + group_size)
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.randint(
        0, 256, (num_tokens, topk), dtype=torch.int32, device="cuda"
    )
    topk_weights = torch.rand((num_tokens, topk), device="cuda")
    combined = _allocate_outputs(padded_max, hidden, topk, group_size)
    staged = _allocate_outputs(padded_max, hidden, topk, group_size)

    mega_moe_pre_dispatch(
        x,
        topk_ids,
        topk_weights,
        *combined,
        quant_group_size=group_size,
    )
    mega_moe_stage_activation(x, staged[0], staged[1], group_size)
    torch.cuda.synchronize()

    for combined_tensor, staged_tensor in zip(combined[:2], staged[:2]):
        assert torch.equal(combined_tensor, staged_tensor)


@torch.inference_mode()
def test_topk_out_variant_stages_directly() -> None:
    num_tokens, padded_max, num_experts, topk = 17, 64, 256, 6
    torch.manual_seed(29)
    scores = torch.randn((num_tokens, num_experts), device="cuda")
    bias = torch.randn((num_experts,), device="cuda")
    expected_weights, expected_ids = moe_fused_gate(
        scores, bias, topk, scoring_func="sqrtsoftplus"
    )
    buf_topk_idx = torch.full(
        (padded_max, topk), -777, dtype=torch.int64, device="cuda"
    )
    buf_topk_weights = torch.full(
        (padded_max, topk), float("nan"), dtype=torch.float32, device="cuda"
    )

    actual_weights, actual_ids = moe_fused_gate(
        scores,
        bias,
        topk,
        scoring_func="sqrtsoftplus",
        out_weights=buf_topk_weights[:num_tokens],
        out_indices=buf_topk_idx[:num_tokens],
    )
    mega_moe_pad_route(actual_ids, buf_topk_idx, buf_topk_weights)
    torch.cuda.synchronize()

    assert actual_weights.data_ptr() == buf_topk_weights.data_ptr()
    assert actual_ids.data_ptr() == buf_topk_idx.data_ptr()
    assert torch.equal(actual_weights, expected_weights)
    assert torch.equal(actual_ids, expected_ids.to(torch.int64))
    assert torch.all(buf_topk_idx[num_tokens:] == -1)
    assert torch.all(buf_topk_weights[num_tokens:] == 0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="the direct-output production path is limited to SM100+",
)
@torch.inference_mode()
def test_direct_out_multistream_cuda_graph() -> None:
    num_tokens, padded_max = 16, 64
    hidden, group_size, num_experts, topk = 4096, 32, 256, 6
    torch.manual_seed(17)
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    scores = torch.randn((num_tokens, num_experts), device="cuda")
    bias = torch.randn((num_experts,), device="cuda")
    reference = _allocate_outputs(padded_max, hidden, topk, group_size)
    captured = _allocate_outputs(padded_max, hidden, topk, group_size)
    side_stream = torch.cuda.Stream()

    def route(
        out_weights: torch.Tensor | None = None,
        out_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return moe_fused_gate(
            scores,
            bias,
            topk,
            scoring_func="sqrtsoftplus",
            out_weights=out_weights,
            out_indices=out_indices,
        )

    # Compile every kernel before capture and establish a combined reference.
    topk_weights, topk_ids = route()
    mega_moe_pre_dispatch(
        x,
        topk_ids,
        topk_weights,
        *reference,
        quant_group_size=group_size,
    )

    def run_direct_graph() -> None:
        current = torch.cuda.current_stream()
        side_stream.wait_stream(current)
        with torch.cuda.stream(side_stream):
            mega_moe_stage_activation(
                x, captured[0], captured[1], quant_group_size=group_size
            )
            mega_moe_pad_route(
                captured[2][:num_tokens],
                captured[2],
                captured[3],
                quant_group_size=group_size,
            )
        route(
            out_weights=captured[3][:num_tokens],
            out_indices=captured[2][:num_tokens],
        )
        current.wait_stream(side_stream)

    run_direct_graph()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_direct_graph()
    graph.replay()
    torch.cuda.synchronize()

    for reference_tensor, captured_tensor in zip(reference, captured):
        assert torch.equal(reference_tensor, captured_tensor)


@torch.inference_mode()
def test_direct_out_production_orchestration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang.srt.distributed import parallel_state
    from sglang.srt.layers.moe import mega_moe as mega_moe_module

    num_tokens, padded_max = 16, 64
    hidden, group_size, num_experts, topk = 4096, 32, 256, 6
    torch.manual_seed(41)
    hidden_states = torch.randn(
        (num_tokens, hidden), dtype=torch.bfloat16, device="cuda"
    )
    gate_weight = torch.randn(
        (num_experts, hidden), dtype=torch.bfloat16, device="cuda"
    )
    bias = torch.randn((num_experts,), device="cuda")
    buf_tensors = _allocate_outputs(padded_max, hidden, topk, group_size)
    buf = SimpleNamespace(
        x=buf_tensors[0],
        x_sf=buf_tensors[1],
        topk_idx=buf_tensors[2],
        topk_weights=buf_tensors[3],
    )

    class FakeTopK:
        calls = 0
        allow_fallback = False

        def __call__(self, _hidden_states, router_logits, **kwargs):
            self.calls += 1
            if not self.allow_fallback:
                raise AssertionError("the eligible direct-output path must bypass TopK")
            weights, ids = moe_fused_gate(
                router_logits,
                bias,
                topk,
                scoring_func="sqrtsoftplus",
            )
            mask_topk_ids(ids, kwargs["num_token_non_padded"])
            return SimpleNamespace(topk_weights=weights, topk_ids=ids)

    class FakeMoE:
        num_fused_shared_experts = 0
        is_hash = False
        layer_id = 0
        routed_scaling_factor = 1.0

        def gate(self, x: torch.Tensor, **_) -> torch.Tensor:
            return torch.mm(x, gate_weight.t(), out_dtype=torch.float32)

    fake_moe = FakeMoE()
    fake_moe.topk = FakeTopK()
    fake_moe.topk.topk_config = SimpleNamespace(
        top_k=topk,
        use_grouped_topk=False,
        renormalize=True,
        num_fused_shared_experts=0,
        custom_routing_function=None,
        scoring_func="sqrtsoftplus",
        correction_bias=bias,
        routed_scaling_factor=1.0,
        apply_routed_scaling_factor_on_output=False,
        allow_routed_experts_capture=True,
    )
    fake_moe.config = SimpleNamespace(
        hidden_size=hidden,
        num_experts_per_tok=topk,
        moe_intermediate_size=1024,
        swiglu_limit=None,
    )
    fake_moe.experts = SimpleNamespace(
        num_experts=num_experts,
        mega_l1_weights=None,
        mega_l2_weights=None,
        should_fuse_routed_scaling_factor_in_topk=True,
    )
    fake_deep_gemm = SimpleNamespace(
        fp8_fp4_mega_moe=lambda y, *_args, **_kwargs: y.zero_()
    )
    monkeypatch.setitem(sys.modules, "deep_gemm", fake_deep_gemm)
    monkeypatch.setenv("SGLANG_OPT_DEEPGEMM_MEGA_MOE_DIRECT_OUT", "1")
    monkeypatch.setenv("SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK", "1")
    monkeypatch.setattr(mega_moe_module, "get_is_capture_mode", lambda: True)
    monkeypatch.setattr(mega_moe_module, "_device_sm", 120)
    monkeypatch.setattr(mega_moe_module, "_MEGA_MOE_STAGING_STREAMS", {})
    monkeypatch.setattr(
        mega_moe_module,
        "_get_mega_moe_symm_buffer",
        lambda *_args, **_kwargs: buf,
    )
    monkeypatch.setattr(
        mega_moe_module.ExpertLocationDispatchInfo,
        "init_new",
        classmethod(lambda _cls, layer_id: None),
    )
    monkeypatch.setattr(
        parallel_state,
        "get_moe_ep_group",
        lambda: SimpleNamespace(device_group=object()),
    )

    def run(forward_batch=None) -> torch.Tensor:
        return mega_moe_module._run_mega_routed(
            fake_moe,
            hidden_states,
            forward_batch=forward_batch,
            input_ids_global=None,
            num_tokens=num_tokens,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = run()
    graph.replay()
    torch.cuda.synchronize()

    routed_ids = buf_tensors[2][:num_tokens].to(torch.int32)
    routed_weights = buf_tensors[3][:num_tokens].clone()
    reference = _allocate_outputs(padded_max, hidden, topk, group_size)
    mega_moe_pre_dispatch(
        hidden_states,
        routed_ids,
        routed_weights,
        *reference,
        quant_group_size=group_size,
    )
    torch.cuda.synchronize()

    assert torch.count_nonzero(graph_output) == 0
    for reference_tensor, staged_tensor in zip(reference, buf_tensors):
        assert torch.equal(reference_tensor, staged_tensor)

    # Padded-token post-processing is not a no-op. The direct path must fall
    # back to the regular TopK + combined pre-dispatch path even when enabled.
    fake_moe.topk.allow_fallback = True
    num_valid = torch.tensor(num_tokens - 3, dtype=torch.int32, device="cuda")
    run(SimpleNamespace(num_token_non_padded=num_valid))
    torch.cuda.synchronize()

    assert fake_moe.topk.calls == 1
    assert torch.all(buf.topk_idx[num_tokens - 3 :] == -1)
