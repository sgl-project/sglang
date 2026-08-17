"""Benchmark direct-output MegaMoE front-end schedules.

The baseline runs RouterGEMM -> TopK -> combined pre-dispatch on one stream.
Direct-out variants stage activations on a side stream, either before or after
RouterGEMM, while TopK writes the final symmetric-buffer route fields.
"""

from __future__ import annotations

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.attention.dsv4 import (
    linear_bf16_fp32,
    mask_topk_ids,
    mega_moe_pad_route,
    mega_moe_pre_dispatch,
    mega_moe_stage_activation,
)
from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

HIDDEN = 4096
GROUP_SIZE = 32
NUM_EXPERTS = 256
TOPK = 6
PADDED_MAX = 256


def _topk_and_postprocess(
    scores: torch.Tensor,
    bias: torch.Tensor,
    valid_tokens: torch.Tensor,
    physical_map: torch.Tensor,
    postprocess: str,
    out_weights: torch.Tensor | None = None,
    out_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights, ids = moe_fused_gate(
        scores,
        bias,
        TOPK,
        scoring_func="sqrtsoftplus",
        out_weights=out_weights,
        out_indices=out_indices,
    )
    if postprocess == "static_remap":
        ids = physical_map[ids]
    if postprocess != "identity":
        mask_topk_ids(ids, valid_tokens)
    return weights, ids


def _run_chain(
    mode: str,
    postprocess: str,
    side_stream: torch.cuda.Stream,
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    valid_tokens: torch.Tensor,
    physical_map: torch.Tensor,
    buf_x: torch.Tensor,
    buf_x_sf: torch.Tensor,
    buf_topk_idx: torch.Tensor,
    buf_topk_weights: torch.Tensor,
) -> None:
    current = torch.cuda.current_stream()
    if mode == "direct_out_overlap":
        # Fork before RouterGEMM. The event proves x's producer is complete;
        # both branches only read x until activation staging writes its own
        # disjoint symmetric-buffer fields.
        side_stream.wait_stream(current)
        with torch.cuda.stream(side_stream):
            mega_moe_stage_activation(x, buf_x, buf_x_sf, quant_group_size=GROUP_SIZE)
            mega_moe_pad_route(
                buf_topk_idx[: x.shape[0]],
                buf_topk_idx,
                buf_topk_weights,
                quant_group_size=GROUP_SIZE,
            )

    scores = linear_bf16_fp32(x, gate_weight)
    if mode == "direct_out_topk_overlap":
        # This variant avoids competing with RouterGEMM. It inserts the fork
        # after GEMM and measures overlap with TopK plus its real postprocess.
        side_stream.wait_stream(current)
        with torch.cuda.stream(side_stream):
            mega_moe_stage_activation(x, buf_x, buf_x_sf, quant_group_size=GROUP_SIZE)
            mega_moe_pad_route(
                buf_topk_idx[: x.shape[0]],
                buf_topk_idx,
                buf_topk_weights,
                quant_group_size=GROUP_SIZE,
            )

    direct_out = mode in ("direct_out_overlap", "direct_out_topk_overlap")
    topk_weights, topk_ids = _topk_and_postprocess(
        scores,
        bias,
        valid_tokens,
        physical_map,
        postprocess,
        out_weights=buf_topk_weights[: x.shape[0]] if direct_out else None,
        out_indices=buf_topk_idx[: x.shape[0]] if direct_out else None,
    )

    if direct_out:
        current.wait_stream(side_stream)
        return

    mega_moe_pre_dispatch(
        x,
        topk_ids,
        topk_weights,
        buf_x,
        buf_x_sf,
        buf_topk_idx,
        buf_topk_weights,
        quant_group_size=GROUP_SIZE,
    )


@marker.parametrize("execution", ["eager", "cuda_graph"], ci_vals=["cuda_graph"])
@marker.parametrize(
    "postprocess",
    ["identity", "padded_mask", "static_remap"],
    ci_vals=["padded_mask"],
)
@marker.parametrize(
    "num_tokens", [1, 2, 4, 8, 16, 32, 64, 128, 256], ci_vals=[1, 16, 128]
)
@marker.benchmark(
    "mode",
    [
        "combined",
        "direct_out_overlap",
        "direct_out_topk_overlap",
    ],
)
def benchmark(num_tokens: int, postprocess: str, execution: str, mode: str):
    if mode.startswith("direct_out") and postprocess != "identity":
        marker.skip("direct out-variant is intentionally limited to identity routing")
    torch.manual_seed(num_tokens)
    x = torch.randn((num_tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
    gate_weight = torch.randn(
        (NUM_EXPERTS, HIDDEN), dtype=torch.bfloat16, device="cuda"
    )
    bias = torch.randn((NUM_EXPERTS,), device="cuda")
    valid_tokens = torch.tensor(
        num_tokens if postprocess == "identity" else max(num_tokens - 1, 0),
        dtype=torch.int32,
        device="cuda",
    )
    physical_map = torch.randperm(NUM_EXPERTS, device="cuda", dtype=torch.int32)
    buf_x = torch.empty((PADDED_MAX, HIDDEN), dtype=torch.float8_e4m3fn, device="cuda")
    buf_x_sf = torch.empty(
        (PADDED_MAX, HIDDEN // GROUP_SIZE // 4), dtype=torch.int32, device="cuda"
    )
    buf_topk_idx = torch.empty((PADDED_MAX, TOPK), dtype=torch.int64, device="cuda")
    buf_topk_weights = torch.empty((PADDED_MAX, TOPK), device="cuda")
    side_stream = torch.cuda.Stream()

    return marker.do_bench(
        _run_chain,
        input_args=(
            mode,
            postprocess,
            side_stream,
            x,
            gate_weight,
            bias,
            valid_tokens,
            physical_map,
            buf_x,
            buf_x_sf,
            buf_topk_idx,
            buf_topk_weights,
        ),
        use_cuda_graph=execution == "cuda_graph",
        warmup_iters=50,
        replay_iters=1000,
        metrics=(0.5,),
        graph_clone_args=(3, 4, 5, 6, 7),
        disable_log_bandwidth=True,
    )


if __name__ == "__main__":
    benchmark.run()
