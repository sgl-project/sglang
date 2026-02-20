"""
Benchmark: tree_speculative_sampling_target_only JIT vs AOT (sgl_kernel)

Measures throughput (µs) for tree speculative sampling across typical
LLM configurations.

Run:
    python python/sglang/jit_kernel/benchmark/bench_speculative_sampling.py
"""

import itertools

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import get_benchmark_range, run_benchmark
from sglang.jit_kernel.speculative_sampling import (
    tree_speculative_sampling_target_only as tree_spec_sampling_jit,
)

try:
    from sgl_kernel import (
        tree_speculative_sampling_target_only as tree_spec_sampling_aot,
    )

    AOT_AVAILABLE = True
except ImportError:
    tree_spec_sampling_aot = None
    AOT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

BATCH_SIZE_RANGE = get_benchmark_range(
    full_range=[1, 4, 8, 16],
    ci_range=[4],
)

# (num_draft_tokens, num_spec_step, vocab_size)
MODEL_CONFIGS = get_benchmark_range(
    full_range=[
        (8, 5, 32000),  # typical LLaMA
        (16, 8, 32000),  # wider tree
        (8, 5, 128000),  # large vocabulary
    ],
    ci_range=[(8, 5, 32000)],
)

LINE_VALS = ["jit", "aot"] if AOT_AVAILABLE else ["jit"]
LINE_NAMES = ["JIT (new)", "AOT sgl_kernel"] if AOT_AVAILABLE else ["JIT (new)"]
STYLES = [(("blue", "--"), ("orange", "-"))] if AOT_AVAILABLE else [("blue", "--")]

DEVICE = "cuda"


def make_inputs(bs, num_draft_tokens, num_spec_step, vocab_size):
    tot_draft = bs * num_draft_tokens

    predicts = torch.zeros(tot_draft, dtype=torch.int32, device=DEVICE)
    accept_index = torch.zeros(bs, num_spec_step, dtype=torch.int32, device=DEVICE)
    accept_token_num = torch.zeros(bs, dtype=torch.int32, device=DEVICE)

    # Linear chain tree
    retrive_index = torch.stack(
        [torch.arange(num_draft_tokens, dtype=torch.int64, device=DEVICE)] * bs
    )
    next_token = torch.arange(1, num_draft_tokens + 1, dtype=torch.int64, device=DEVICE)
    next_token[-1] = -1
    retrive_next_token = next_token.unsqueeze(0).expand(bs, -1).contiguous()
    retrive_next_sibling = torch.full(
        (bs, num_draft_tokens), -1, dtype=torch.int64, device=DEVICE
    )
    candidates = (
        (torch.arange(num_draft_tokens, dtype=torch.int64, device=DEVICE) % vocab_size)
        .unsqueeze(0)
        .expand(bs, -1)
        .contiguous()
    )

    uniform_samples = torch.rand(
        bs, num_draft_tokens, dtype=torch.float32, device=DEVICE
    )
    uniform_samples_for_final_sampling = torch.rand(
        bs, dtype=torch.float32, device=DEVICE
    )

    raw = torch.rand(
        bs, num_draft_tokens, vocab_size, dtype=torch.float32, device=DEVICE
    )
    target_probs = raw / raw.sum(dim=-1, keepdim=True)
    draft_probs = target_probs.clone()

    return dict(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=uniform_samples,
        uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
        target_probs=target_probs,
        draft_probs=draft_probs,
    )


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["bs", "num_draft_tokens", "num_spec_step", "vocab_size"],
        x_vals=[
            (bs, ndt, nss, vs)
            for bs, (ndt, nss, vs) in itertools.product(BATCH_SIZE_RANGE, MODEL_CONFIGS)
        ],
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=[("blue", "--"), ("orange", "-")][: len(LINE_VALS)],
        ylabel="us",
        plot_name="tree-speculative-sampling-performance",
        args={},
    )
)
def bench_speculative_sampling(
    bs: int,
    num_draft_tokens: int,
    num_spec_step: int,
    vocab_size: int,
    provider: str,
):
    inputs = make_inputs(bs, num_draft_tokens, num_spec_step, vocab_size)
    common_kwargs = dict(
        threshold_single=0.9,
        threshold_acc=0.9,
        deterministic=True,
    )

    if provider == "jit":

        def fn():
            inp = {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }
            tree_spec_sampling_jit(**inp, **common_kwargs)

    elif provider == "aot":

        def fn():
            inp = {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }
            tree_spec_sampling_aot(**inp, **common_kwargs)

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

    print("Correctness diff (JIT vs AOT):")
    for bs, num_draft_tokens, num_spec_step, vocab_size in [
        (1, 4, 4, 32),
        (2, 8, 5, 32000),
        (4, 16, 8, 128000),
    ]:
        inputs_jit = make_inputs(bs, num_draft_tokens, num_spec_step, vocab_size)
        inputs_aot = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in inputs_jit.items()
        }

        tree_spec_sampling_jit(**inputs_jit, threshold_single=0.9, threshold_acc=0.9)
        tree_spec_sampling_aot(**inputs_aot, threshold_single=0.9, threshold_acc=0.9)

        match_predicts = torch.equal(inputs_jit["predicts"], inputs_aot["predicts"])
        match_accept_num = torch.equal(
            inputs_jit["accept_token_num"], inputs_aot["accept_token_num"]
        )
        status = "OK" if (match_predicts and match_accept_num) else "MISMATCH"
        print(
            f"  bs={bs:2d} num_draft={num_draft_tokens:2d} num_spec={num_spec_step:2d} "
            f"vocab={vocab_size:6d}  [{status}]"
        )


if __name__ == "__main__":
    calculate_diff()
    print()
    bench_speculative_sampling.run(print_data=True)
