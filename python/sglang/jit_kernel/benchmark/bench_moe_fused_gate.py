# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Benchmark: JIT moe_fused_gate vs. PyTorch reference (biased grouped top-k).

Usage::

    python -m sglang.jit_kernel.benchmark.bench_moe_fused_gate

The benchmark sweeps over a range of token sequence lengths and compares:
  * ``original`` – pure-PyTorch ``biased_grouped_topk`` reference.
  * ``jit``      – JIT-compiled ``moe_fused_gate`` CUDA kernel.
"""
import itertools

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import get_benchmark_range, run_benchmark
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="stage-b-kernel-benchmark-1-gpu-large")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# DeepSeek V3 / R1 default config
NUM_EXPERTS = 256
NUM_EXPERT_GROUP = 8
TOPK_GROUP = 4
TOPK = 8
ROUTED_SCALING_FACTOR = 2.5
DTYPE = torch.bfloat16
DEVICE = "cuda"

SEQ_LENGTHS = get_benchmark_range(
    full_range=[1000, 2000, 4000, 8000, 16000, 32000, 64000],
    ci_range=[4000],
)

LINE_VALS = ["original", "jit"]
LINE_NAMES = ["PyTorch Reference", "SGL JIT Kernel"]
STYLES = [("blue", "--"), ("red", "-")]

configs = list(itertools.product(SEQ_LENGTHS))


# ---------------------------------------------------------------------------
# Provider functions
# ---------------------------------------------------------------------------


def run_original(scores: torch.Tensor, bias: torch.Tensor) -> None:
    """Pure-PyTorch biased grouped top-k (reference implementation)."""
    num_token = scores.shape[0]
    num_experts = scores.shape[1]

    sigmoid_scores = scores.sigmoid()
    scores_for_choice = sigmoid_scores + bias.unsqueeze(0)

    group_scores = (
        scores_for_choice.view(num_token, NUM_EXPERT_GROUP, -1)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, NUM_EXPERT_GROUP, num_experts // NUM_EXPERT_GROUP)
        .reshape(num_token, -1)
    )
    tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
    topk_weights, topk_ids = torch.topk(tmp_scores, k=TOPK, dim=-1, sorted=False)
    topk_weights = sigmoid_scores.gather(1, topk_ids)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)


def run_jit(scores: torch.Tensor, bias: torch.Tensor) -> None:
    """JIT-compiled moe_fused_gate kernel."""
    from sglang.jit_kernel.moe_fused_gate import moe_fused_gate

    moe_fused_gate(
        input=scores,
        bias=bias,
        num_expert_group=NUM_EXPERT_GROUP,
        topk_group=TOPK_GROUP,
        topk=TOPK,
        num_fused_shared_experts=0,
        routed_scaling_factor=ROUTED_SCALING_FACTOR,
        apply_routed_scaling_factor_on_output=False,
    )


# ---------------------------------------------------------------------------
# Triton benchmark
# ---------------------------------------------------------------------------

FN_MAP = {"original": run_original, "jit": run_jit}


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_length"],
        x_vals=[[s] for s in SEQ_LENGTHS],
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="moe-fused-gate-performance",
        args={},
    )
)
def benchmark(seq_length: int, provider: str):
    scores = torch.randn((seq_length, NUM_EXPERTS), dtype=DTYPE, device=DEVICE)
    bias = torch.rand(NUM_EXPERTS, dtype=DTYPE, device=DEVICE)
    fn = lambda: FN_MAP[provider](scores.clone(), bias.clone())
    return run_benchmark(fn)


if __name__ == "__main__":
    print(
        f"Benchmarking moe_fused_gate  "
        f"(num_experts={NUM_EXPERTS}, num_expert_group={NUM_EXPERT_GROUP}, "
        f"topk_group={TOPK_GROUP}, topk={TOPK}, dtype={DTYPE})..."
    )
    benchmark.run(print_data=True)
