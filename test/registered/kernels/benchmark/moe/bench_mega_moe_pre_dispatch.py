"""Benchmark the routed TopK -> MegaMoE pre-dispatch dependency chain."""

from __future__ import annotations

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.ops.attention.dsv4.utils import make_name
from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
from sglang.srt.layers.moe.topk import biased_topk_impl
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=40, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

HIDDEN = 4096
GROUP_SIZE = 32
NUM_EXPERTS = 256
TOPK = 6
PADDED_MAX = 256


@cache_once
def _pre_dispatch_module(use_pdl: bool):
    args = make_cpp_args(GROUP_SIZE, use_pdl)
    return load_jit(
        make_name(f"mega_moe_pre_dispatch_bench_pdl_{int(use_pdl)}"),
        *args,
        cuda_files=["deepseek_v4/mega_moe_pre_dispatch.cuh"],
        cuda_wrappers=[("run", f"MegaMoEPreDispatchKernel<{args}>::run")],
    )


def _run_chain(
    module,
    mode: str,
    x: torch.Tensor,
    scores: torch.Tensor,
    bias: torch.Tensor,
    buf_x: torch.Tensor,
    buf_x_sf: torch.Tensor,
    buf_topk_idx: torch.Tensor,
    buf_topk_weights: torch.Tensor,
) -> None:
    if mode == "production_default":
        topk_weights, topk_ids = biased_topk_impl(
            hidden_states=x,
            gating_output=scores,
            correction_bias=bias,
            topk=TOPK,
            renormalize=True,
            scoring_func="sqrtsoftplus",
        )
    else:
        topk_weights, topk_ids = moe_fused_gate(
            scores,
            bias,
            TOPK,
            scoring_func="sqrtsoftplus",
        )
    module.run(
        x,
        topk_ids,
        topk_weights,
        buf_x,
        buf_x_sf,
        buf_topk_idx,
        buf_topk_weights,
    )


@marker.parametrize("execution", ["eager", "cuda_graph"], ci_vals=["eager"])
@marker.parametrize(
    "num_tokens", [1, 2, 4, 8, 16, 32, 64, 128, 256], ci_vals=[1, 16, 128]
)
@marker.benchmark(
    "mode",
    ["production_default", "pdl_off", "late_wait_pdl"],
)
def benchmark(num_tokens: int, mode: str, execution: str):
    torch.manual_seed(num_tokens)
    x = torch.randn((num_tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
    scores = torch.randn((num_tokens, NUM_EXPERTS), device="cuda")
    bias = torch.randn((NUM_EXPERTS,), device="cuda")
    buf_x = torch.empty((PADDED_MAX, HIDDEN), dtype=torch.float8_e4m3fn, device="cuda")
    buf_x_sf = torch.empty(
        (PADDED_MAX, HIDDEN // GROUP_SIZE // 4), dtype=torch.int32, device="cuda"
    )
    buf_topk_idx = torch.empty((PADDED_MAX, TOPK), dtype=torch.int64, device="cuda")
    buf_topk_weights = torch.empty((PADDED_MAX, TOPK), device="cuda")
    module = _pre_dispatch_module(use_pdl=mode != "pdl_off")

    return marker.do_bench(
        _run_chain,
        input_args=(
            module,
            mode,
            x,
            scores,
            bias,
            buf_x,
            buf_x_sf,
            buf_topk_idx,
            buf_topk_weights,
        ),
        use_cuda_graph=execution == "cuda_graph",
        warmup_iters=100,
        replay_iters=1000,
        metrics=(0.5,),
        graph_clone_args=(2, 3, 4),
        disable_log_bandwidth=True,
    )


if __name__ == "__main__":
    benchmark.run()
