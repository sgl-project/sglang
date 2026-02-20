"""
Benchmark: eagle_utils JIT vs AOT (sgl_kernel)

Measures throughput (µs) for:
  - build_tree_kernel_efficient
  - verify_tree_greedy

Run:
    python python/sglang/jit_kernel/benchmark/bench_eagle_utils.py
"""

import itertools

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import get_benchmark_range, run_benchmark
from sglang.jit_kernel.eagle_utils import (
    build_tree_kernel_efficient as build_tree_jit,
    verify_tree_greedy as verify_tree_greedy_jit,
)

try:
    from sgl_kernel import (
        build_tree_kernel_efficient as build_tree_aot,
        verify_tree_greedy as verify_tree_greedy_aot,
    )

    AOT_AVAILABLE = True
except ImportError:
    build_tree_aot = None
    verify_tree_greedy_aot = None
    AOT_AVAILABLE = False

# TreeMaskMode constants (must match eagle_utils.cuh)
FULL_MASK = 0
QLEN_ONLY = 1
QLEN_ONLY_BITPACKING = 2

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

# (topk, depth) → draft_token_num = topk * (depth - 1) + 1
TREE_CONFIGS = get_benchmark_range(
    full_range=[
        (4, 5),   # typical EAGLE: 17 draft tokens
        (8, 4),   # wider: 25 draft tokens
        (4, 8),   # deeper: 29 draft tokens
    ],
    ci_range=[(4, 5)],
)

# (num_draft_tokens, num_spec_step, vocab_size) for verify_tree_greedy
VERIFY_CONFIGS = get_benchmark_range(
    full_range=[
        (16, 8, 32000),
        (32, 8, 32000),
        (16, 8, 128000),
    ],
    ci_range=[(16, 8, 32000)],
)


# ---------------------------------------------------------------------------
# build_tree_kernel_efficient inputs
# ---------------------------------------------------------------------------


def make_build_tree_inputs(bs, topk, depth, tree_mask_mode):
    draft_token_num = topk * (depth - 1) + 1
    seq_len = 128

    parent_list_size = topk * (depth - 1) + 1
    parent_list = torch.zeros(bs, parent_list_size, dtype=torch.int64, device=DEVICE)
    for i in range(1, parent_list_size):
        parent_list[:, i] = i - 1

    selected_index = (
        torch.arange(draft_token_num - 1, dtype=torch.int64, device=DEVICE)
        .unsqueeze(0)
        .expand(bs, -1)
        .contiguous()
    )
    verified_seq_len = torch.full((bs,), seq_len, dtype=torch.int64, device=DEVICE)

    if tree_mask_mode == QLEN_ONLY_BITPACKING:
        num_bytes = 4 if draft_token_num > 16 else (2 if draft_token_num > 8 else 1)
        tree_mask = torch.zeros(bs * draft_token_num * num_bytes, dtype=torch.uint8, device=DEVICE)
    else:
        tree_mask = torch.zeros(bs * draft_token_num * draft_token_num, dtype=torch.bool, device=DEVICE)

    positions = torch.zeros(bs, draft_token_num, dtype=torch.int64, device=DEVICE)
    retrive_index = torch.zeros(bs, draft_token_num, dtype=torch.int64, device=DEVICE)
    retrive_next_token = torch.full((bs, draft_token_num), -1, dtype=torch.int64, device=DEVICE)
    retrive_next_sibling = torch.full((bs, draft_token_num), -1, dtype=torch.int64, device=DEVICE)

    return dict(
        parent_list=parent_list,
        selected_index=selected_index,
        verified_seq_len=verified_seq_len,
        tree_mask=tree_mask,
        positions=positions,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        topk=topk,
        depth=depth,
        draft_token_num=draft_token_num,
        tree_mask_mode=tree_mask_mode,
    )


# ---------------------------------------------------------------------------
# verify_tree_greedy inputs
# ---------------------------------------------------------------------------


def make_verify_inputs(bs, num_draft_tokens, num_spec_step, vocab_size):
    tot_draft = bs * num_draft_tokens

    predicts = torch.zeros(tot_draft, dtype=torch.int32, device=DEVICE)
    accept_index = torch.zeros(bs, num_spec_step, dtype=torch.int32, device=DEVICE)
    accept_token_num = torch.zeros(bs, dtype=torch.int32, device=DEVICE)

    retrive_index = torch.stack(
        [torch.arange(num_draft_tokens, dtype=torch.int64, device=DEVICE)] * bs
    )
    next_token = torch.arange(1, num_draft_tokens + 1, dtype=torch.int64, device=DEVICE)
    next_token[-1] = -1
    retrive_next_token = next_token.unsqueeze(0).expand(bs, -1).contiguous()
    retrive_next_sibling = torch.full((bs, num_draft_tokens), -1, dtype=torch.int64, device=DEVICE)
    candidates = (
        (torch.arange(num_draft_tokens, dtype=torch.int64, device=DEVICE) % vocab_size)
        .unsqueeze(0)
        .expand(bs, -1)
        .contiguous()
    )
    target_predict = candidates.clone()

    return dict(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        target_predict=target_predict,
    )


# ---------------------------------------------------------------------------
# build_tree_kernel_efficient benchmark
# ---------------------------------------------------------------------------


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["bs", "topk", "depth", "tree_mask_mode"],
        x_vals=[
            (bs, topk, depth, QLEN_ONLY)
            for bs, (topk, depth) in itertools.product(BATCH_SIZE_RANGE, TREE_CONFIGS)
        ],
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=[("blue", "--"), ("orange", "-")][: len(LINE_VALS)],
        ylabel="us",
        plot_name="build-tree-performance",
        args={},
    )
)
def bench_build_tree(bs: int, topk: int, depth: int, tree_mask_mode: int, provider: str):
    draft_token_num = topk * (depth - 1) + 1
    inputs = make_build_tree_inputs(bs, topk, depth, tree_mask_mode)

    mutated_keys = {"positions", "retrive_index", "retrive_next_token", "retrive_next_sibling", "tree_mask"}
    backups = {k: inputs[k].clone() for k in mutated_keys}

    if provider == "jit":

        def fn():
            for k in mutated_keys:
                inputs[k].copy_(backups[k])
            build_tree_jit(**inputs)

    elif provider == "aot":

        def fn():
            for k in mutated_keys:
                inputs[k].copy_(backups[k])
            build_tree_aot(**inputs)

    else:
        raise ValueError(f"Unknown provider: {provider}")

    return run_benchmark(fn)


# ---------------------------------------------------------------------------
# verify_tree_greedy benchmark
# ---------------------------------------------------------------------------


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["bs", "num_draft_tokens", "num_spec_step", "vocab_size"],
        x_vals=[
            (bs, ndt, nss, vs)
            for bs, (ndt, nss, vs) in itertools.product(BATCH_SIZE_RANGE, VERIFY_CONFIGS)
        ],
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=[("blue", "--"), ("orange", "-")][: len(LINE_VALS)],
        ylabel="us",
        plot_name="verify-tree-greedy-performance",
        args={},
    )
)
def bench_verify_tree_greedy(
    bs: int,
    num_draft_tokens: int,
    num_spec_step: int,
    vocab_size: int,
    provider: str,
):
    inputs = make_verify_inputs(bs, num_draft_tokens, num_spec_step, vocab_size)

    mutated_keys = {"predicts", "accept_index", "accept_token_num"}
    backups = {k: inputs[k].clone() for k in mutated_keys}

    if provider == "jit":

        def fn():
            for k in mutated_keys:
                inputs[k].copy_(backups[k])
            verify_tree_greedy_jit(**inputs)

    elif provider == "aot":

        def fn():
            for k in mutated_keys:
                inputs[k].copy_(backups[k])
            verify_tree_greedy_aot(**inputs)

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

    print("Correctness diff — verify_tree_greedy (JIT vs AOT):")
    for bs, num_draft_tokens, num_spec_step, vocab_size in [
        (1, 4, 4, 32),
        (2, 8, 5, 32000),
        (4, 16, 8, 128000),
    ]:
        inp_jit = make_verify_inputs(bs, num_draft_tokens, num_spec_step, vocab_size)
        inp_aot = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inp_jit.items()}

        verify_tree_greedy_jit(**inp_jit)
        verify_tree_greedy_aot(**inp_aot)

        match_predicts = torch.equal(inp_jit["predicts"], inp_aot["predicts"])
        match_accept_num = torch.equal(inp_jit["accept_token_num"], inp_aot["accept_token_num"])
        status = "OK" if (match_predicts and match_accept_num) else "MISMATCH"
        print(
            f"  bs={bs:2d} num_draft={num_draft_tokens:2d} num_spec={num_spec_step:2d} "
            f"vocab={vocab_size:6d}  [{status}]"
        )


if __name__ == "__main__":
    calculate_diff()
    print()
    bench_build_tree.run(print_data=True)
    print()
    bench_verify_tree_greedy.run(print_data=True)
