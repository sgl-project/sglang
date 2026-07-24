"""Benchmark: KDA ReplaySSM commit fold, per-layer loop vs layer-batched.

The commit fold replays each request's accepted window into the fp32 checkpoint.
It ran as a Python loop calling commit_kda_replayssm_spec once per KDA layer
(~69 tiny launches at bs=1). commit_kda_replayssm_spec_all_layers folds every
layer in one launch (layer packed into the grid's head axis).

do_bench with use_cuda_graph=False (NOT True): the commit runs EAGER in
production (post-verify, host-driven on the accept pattern -- it is not in the
decode CUDA graph, unlike the verify kernel). The win here is launch-count
(69 -> 1), which a cuda graph would amortize away and hide. So the eager naive
loop is the production-relevant metric. Both paths timed whole, not by
subtraction. Correctness: test/registered/kernels/test_kda_replayssm_fold_batched.py.

Usage: python bench_kda_fold_batched.py
"""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.attention.fla.kda_replayssm_spec_decode import (
    commit_kda_replayssm_spec,
    commit_kda_replayssm_spec_all_layers,
)

NUM_LAYERS = 69      # K3 KDA layers per TP8 rank
HV = H = 32          # K3 KDA heads per TP8 rank
K = V = 128
L = 16               # ring length
ACCEPT = 7           # dspark γ=7 committed prefix


def make_inputs(bs, device):
    torch.manual_seed(0)
    num_slots = bs + 1
    return dict(
        temporal=torch.randn(NUM_LAYERS, num_slots, HV, V, K, device=device, dtype=torch.float32),
        rawv=torch.randn(NUM_LAYERS, num_slots, HV, L, V, device=device, dtype=torch.bfloat16),
        rawk=torch.randn(NUM_LAYERS, num_slots, H, L, K, device=device, dtype=torch.bfloat16),
        gk=(-5.0) * torch.sigmoid(torch.randn(NUM_LAYERS, num_slots, HV, L, K, device=device, dtype=torch.float32)),
        beta=torch.sigmoid(torch.randn(NUM_LAYERS, num_slots, HV, L, device=device, dtype=torch.float32)),
        slots=torch.arange(1, bs + 1, device=device, dtype=torch.int32),
        accept=torch.full((bs,), ACCEPT, device=device, dtype=torch.int32),
    )


def run_per_layer(inp):
    for li in range(NUM_LAYERS):
        commit_kda_replayssm_spec(
            checkpoint_state=inp["temporal"][li], rawv_cache=inp["rawv"][li],
            rawk_cache=inp["rawk"][li], gk_cache=inp["gk"][li], beta_cache=inp["beta"][li],
            ssm_state_indices=inp["slots"], accept_lens=inp["accept"],
            max_cache_len=L, num_k_heads=H, null_block_id=-1,
        )


def run_batched(inp):
    commit_kda_replayssm_spec_all_layers(
        checkpoint_state=inp["temporal"], rawv_cache=inp["rawv"], rawk_cache=inp["rawk"],
        gk_cache=inp["gk"], beta_cache=inp["beta"],
        ssm_state_indices=inp["slots"], accept_lens=inp["accept"],
        max_cache_len=L, num_k_heads=H, null_block_id=-1,
    )


@marker.parametrize("bs", [1, 4, 16, 32, 64], [4])
@marker.benchmark("impl", ["per_layer", "batched"])
def benchmark(bs: int, impl: str):
    inp = make_inputs(bs, "cuda")
    fn = (lambda: run_per_layer(inp)) if impl == "per_layer" else (lambda: run_batched(inp))
    return marker.do_bench(fn, use_cuda_graph=False, disable_log_bandwidth=True)


if __name__ == "__main__":
    benchmark.run()
