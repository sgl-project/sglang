"""
Benchmark: reconstruct_indices_from_tree_mask JIT vs AOT (sgl_kernel)

Measures throughput (µs) across typical batch sizes and tree sizes.

Run:
    python python/sglang/jit_kernel/benchmark/bench_ngram_utils.py
"""

import itertools

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import get_benchmark_range, run_benchmark
from sglang.jit_kernel.ngram_utils import (
    reconstruct_indices_from_tree_mask as reconstruct_jit,
)

try:
    from sgl_kernel import reconstruct_indices_from_tree_mask as reconstruct_aot

    AOT_AVAILABLE = True
except ImportError:
    reconstruct_aot = None
    AOT_AVAILABLE = False

DEVICE = "cuda"

LINE_VALS = ["jit", "aot"] if AOT_AVAILABLE else ["jit"]
LINE_NAMES = ["JIT (new)", "AOT sgl_kernel"] if AOT_AVAILABLE else ["JIT (new)"]

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

BATCH_SIZE_RANGE = get_benchmark_range(
    full_range=[1, 4, 8, 16],
    ci_range=[4],
)

DRAFT_TOKEN_RANGE = get_benchmark_range(
    full_range=[8, 16, 32, 64],
    ci_range=[16],
)


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------


def make_inputs(bs, draft_token_num):
    tree_mask = torch.zeros(
        bs * draft_token_num * draft_token_num, dtype=torch.bool, device=DEVICE
    )
    # Linear-chain tree mask
    base = draft_token_num * draft_token_num
    for b in range(bs):
        for i in range(draft_token_num):
            for j in range(i):
                tree_mask[b * base + i * draft_token_num + j] = True

    verified_seq_len = torch.full((bs,), 128, dtype=torch.int64, device=DEVICE)
    positions = torch.zeros(bs * draft_token_num, dtype=torch.int64, device=DEVICE)
    retrive_index = torch.zeros(bs, draft_token_num, dtype=torch.int64, device=DEVICE)
    retrive_next_token = torch.full((bs, draft_token_num), -1, dtype=torch.int64, device=DEVICE)
    retrive_next_sibling = torch.full((bs, draft_token_num), -1, dtype=torch.int64, device=DEVICE)

    return dict(
        tree_mask=tree_mask,
        verified_seq_len=verified_seq_len,
        positions=positions,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
    )


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["bs", "draft_token_num"],
        x_vals=list(itertools.product(BATCH_SIZE_RANGE, DRAFT_TOKEN_RANGE)),
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=[("blue", "--"), ("orange", "-")][: len(LINE_VALS)],
        ylabel="us",
        plot_name="reconstruct-indices-from-tree-mask-performance",
        args={},
    )
)
def bench_reconstruct(bs: int, draft_token_num: int, provider: str):
    inputs = make_inputs(bs, draft_token_num)

    mutated_keys = {"positions", "retrive_index", "retrive_next_token", "retrive_next_sibling"}
    backups = {k: inputs[k].clone() for k in mutated_keys}

    if provider == "jit":

        def fn():
            for k in mutated_keys:
                inputs[k].copy_(backups[k])
            reconstruct_jit(**inputs, batch_size=bs, draft_token_num=draft_token_num)

    elif provider == "aot":

        def fn():
            for k in mutated_keys:
                inputs[k].copy_(backups[k])
            reconstruct_aot(**inputs, batch_size=bs, draft_token_num=draft_token_num)

    else:
        raise ValueError(f"Unknown provider: {provider}")

    return run_benchmark(fn)


# ---------------------------------------------------------------------------
# Quick correctness diff
# ---------------------------------------------------------------------------


def calculate_diff():
    if not AOT_AVAILABLE:
        print("sgl_kernel not available — skipping AOT diff check")
        return

    print("Correctness diff — reconstruct_indices_from_tree_mask (JIT vs AOT):")
    for bs, draft_token_num in [(1, 8), (2, 16), (4, 32)]:
        inp_jit = make_inputs(bs, draft_token_num)
        inp_aot = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inp_jit.items()}

        reconstruct_jit(**inp_jit, batch_size=bs, draft_token_num=draft_token_num)
        reconstruct_aot(**inp_aot, batch_size=bs, draft_token_num=draft_token_num)

        match_pos = torch.equal(inp_jit["positions"], inp_aot["positions"])
        match_idx = torch.equal(inp_jit["retrive_index"], inp_aot["retrive_index"])
        match_next = torch.equal(inp_jit["retrive_next_token"], inp_aot["retrive_next_token"])
        match_sib = torch.equal(inp_jit["retrive_next_sibling"], inp_aot["retrive_next_sibling"])
        status = "OK" if all([match_pos, match_idx, match_next, match_sib]) else "MISMATCH"
        print(f"  bs={bs:2d} draft_token_num={draft_token_num:2d}  [{status}]")


if __name__ == "__main__":
    calculate_diff()
    print()
    bench_reconstruct.run(print_data=True)
