"""Benchmark the H200 GLM-4.5 fused MoE against the SGLang Triton path."""

from __future__ import annotations

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=240, stage="base-b-kernel-benchmark", runner_config="8-gpu-h200"
)

_EXPERTS = 161
_HIDDEN = 5120
_GATE_UP = 384
_INTERMEDIATE = 192
_TOP_K = 9


def _build(tokens: int):
    torch.manual_seed(0x45F00D)
    hidden = torch.randn((tokens, _HIDDEN), dtype=torch.bfloat16, device="cuda")
    w1 = (
        torch.empty(
            (_EXPERTS, _GATE_UP, _HIDDEN),
            dtype=torch.bfloat16,
            device="cuda",
        )
        .uniform_(-1.0, 1.0)
        .to(torch.float8_e4m3fn)
    )
    w2 = (
        torch.empty(
            (_EXPERTS, _HIDDEN, _INTERMEDIATE),
            dtype=torch.bfloat16,
            device="cuda",
        )
        .uniform_(-1.0, 1.0)
        .to(torch.float8_e4m3fn)
    )
    routing = torch.rand((tokens, _EXPERTS - 1), device="cuda")
    routed_ids = routing.topk(_TOP_K - 1, dim=1, sorted=True).indices.to(torch.int32)
    shared_ids = torch.full((tokens, 1), _EXPERTS - 1, dtype=torch.int32, device="cuda")
    topk_ids = torch.cat((routed_ids, shared_ids), dim=1)
    routed_weights = torch.rand((tokens, _TOP_K - 1), device="cuda")
    routed_weights.div_(routed_weights.sum(dim=1, keepdim=True))
    topk_weights = torch.cat(
        (
            routed_weights,
            torch.full((tokens, 1), 0.4, dtype=torch.float32, device="cuda"),
        ),
        dim=1,
    )
    w1_scale = torch.rand((_EXPERTS, _GATE_UP, 1), device="cuda").mul_(0.02)
    w2_scale = torch.rand((_EXPERTS, _HIDDEN, 1), device="cuda").mul_(0.02)
    return hidden, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale


def _fused_experts(hidden, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale):
    return fused_moe.fused_experts_impl(
        hidden,
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace=True,
        activation="silu",
        is_gated=True,
        apply_router_weight_on_input=False,
        use_fp8_w8a8=True,
        per_channel_quant=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        no_combine=False,
        routed_scaling_factor=2.5,
        filter_expert=False,
        gate_up_interleaved=True,
    )


@marker.parametrize(
    "tokens", [1, 16, 64, 113, 119, 129, 1935, 3451, 7497], [1, 129, 7497]
)
@marker.benchmark("provider", ["triton", "cuda"])
def benchmark(tokens: int, provider: str):
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    inputs = _build(tokens)
    old_value = fused_moe._enable_glm45_fused_moe
    fused_moe._enable_glm45_fused_moe = provider == "cuda"
    try:
        return marker.do_bench(
            _fused_experts,
            input_args=inputs,
            use_cuda_graph=False,
            warmup_iters=3,
            replay_iters=50,
            graph_clone_args=None,
            memory_args=None,
        )
    finally:
        fused_moe._enable_glm45_fused_moe = old_value


if __name__ == "__main__":
    benchmark.run()
