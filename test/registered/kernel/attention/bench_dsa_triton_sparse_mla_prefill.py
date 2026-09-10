# SPDX-License-Identifier: Apache-2.0
"""Benchmark the fused Triton sparse-MLA prefill against the kernels the DSA
prefill can otherwise dispatch to.

Index fixture note: the selections are built to look like the indexer's, not
like ``randint``. Two properties matter and uniform sampling has neither:

* Positions within a token are distinct (top-k cannot pick a slot twice).
* Neighbouring tokens select mostly the same rows, because the indexer's scores
  move slowly in ``t``. On captured GLM-5.1 traces the union of 2 adjacent
  tokens' selections is only ~1.05x the size of one token's, and of 4 tokens
  ~1.15x.

The second property is the whole point of the ``union`` path, so benchmarking it
on uniform-random indices understates it by roughly 2x and would make the path
look useless.
"""

from __future__ import annotations

import torch
import triton
import triton.testing

from sglang.kernels.jit.benchmark.utils import run_benchmark_no_cudagraph
from sglang.kernels.ops.attention.dsa.triton_sparse_mla_prefill import (
    sparse_mla_prefill,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.utils import is_in_ci

try:
    from sgl_kernel.flash_mla import flash_mla_sparse_fwd
except ImportError:
    flash_mla_sparse_fwd = None

try:
    # Same form as the kernel under test, but reachable only on gfx950. Kept as
    # a provider so the comparison against the closest sibling is visible here
    # rather than having to be taken on trust; it is dtype-generic enough to run
    # in bf16 on CUDA for the purpose of timing.
    from sglang.kernels.ops.attention.dsa.triton_sparse_mla import (
        triton_sparse_mla_fwd,
    )
except ImportError:
    triton_sparse_mla_fwd = None


def _flashmla_runnable() -> bool:
    # The op refuses anything but SM90a / SM100f at call time, so gate on the
    # device rather than on the import succeeding.
    return flash_mla_sparse_fwd is not None and (
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] in (9, 10)
    )


HAS_FLASHMLA = _flashmla_runnable()

register_cuda_ci(
    est_time=90, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

IS_CI = is_in_ci()
D_QK, D_V, SM_SCALE = 576, 512, 0.0625
CHURN = 0.02  # share of a token's rows replaced when moving to the next token
UNION_TILE_ROWS = 32

if IS_CI:
    CASES = [(1024, 8, 512), (2048, 8, 2048)]
else:
    # (num_tokens, num_q_heads_after_tp, topk)
    CASES = [
        (2048, 8, 2048),
        (4096, 8, 2048),
        (8192, 8, 2048),
        (8192, 16, 2048),
    ]

LINE_VALS = ["triton_prefill", "triton_prefill_union"]
LINE_NAMES = ["Triton prefill", "Triton prefill (union)"]
STYLES = [("blue", "-"), ("green", "-")]
if triton_sparse_mla_fwd is not None:
    LINE_VALS.insert(0, "triton_sparse_mla_gfx950")
    LINE_NAMES.insert(0, "Triton sparse-MLA (gfx950 path)")
    STYLES.insert(0, ("red", ":"))
if HAS_FLASHMLA:
    LINE_VALS.insert(0, "flashmla_sparse")
    LINE_NAMES.insert(0, "FlashMLA sparse")
    STYLES.insert(0, ("orange", "--"))


def _make_inputs(num_tokens: int, num_heads: int, topk: int):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(7000 + num_tokens + num_heads)
    # The KV pool must be comfortably larger than topk, or there are not enough
    # unselected rows left to vary the set from token to token.
    s_kv = max(num_tokens, 2 * topk)
    q = torch.randn(
        (num_tokens, num_heads, D_QK),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    kv = torch.randn(
        (s_kv, D_QK), dtype=torch.bfloat16, device="cuda", generator=generator
    )

    # Random walk, not i.i.d. draws from a shared pool: the indexer's ranking
    # drifts slowly with t, so token t+1's selection is token t's with a few
    # rows swapped. CHURN is set so the walk reproduces the union sizes measured
    # on captured GLM-5.1 traces (union of 2 adjacent tokens ~1.05x one token's
    # set, union of 4 ~1.15x). Drawing each token independently instead gives a
    # far weaker overlap and understates the union path by roughly 2x.
    perm = torch.randperm(s_kv, device="cuda", generator=generator)
    current, spare = perm[:topk].clone(), perm[topk:].clone()
    n_swap = max(1, int(topk * CHURN))
    assert spare.numel() >= n_swap, "KV pool too small to vary the selection"
    indices = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")
    for t in range(num_tokens):
        indices[t] = current.to(torch.int32)
        if t < topk:
            # A token whose causal prefix is shorter than topk cannot fill the
            # row; the indexer pads with -1. Production prefill spends its first
            # topk tokens here, so leaving it out would flatter every kernel
            # that ignores the padding and penalise the ones that skip it.
            indices[t, t + 1 :] = -1
        out_slots = torch.randperm(topk, device="cuda", generator=generator)[:n_swap]
        in_slots = torch.randperm(spare.numel(), device="cuda", generator=generator)[
            :n_swap
        ]
        evicted = current[out_slots].clone()
        current[out_slots] = spare[in_slots]
        spare[in_slots] = evicted  # swap keeps both sets duplicate-free
    return q, kv, indices


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens", "num_heads", "topk"],
        x_vals=CASES,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="dsa-triton-sparse-mla-prefill-performance",
        args={},
    )
)
def bench_dsa_triton_sparse_mla_prefill(
    num_tokens: int, num_heads: int, topk: int, provider: str
):
    q, kv, indices = _make_inputs(num_tokens, num_heads, topk)

    if provider == "flashmla_sparse":
        if not HAS_FLASHMLA:
            raise RuntimeError("sgl_kernel.flash_mla.flash_mla_sparse_fwd unavailable")
        kv_3d = kv.unsqueeze(1)
        idx_3d = indices.unsqueeze(1)
        # The kernel requires num_heads % 64 on SM90 and % 128 on SM100+, so the
        # backend zero-pads the head dim and slices the result back. Do the same
        # here: that padding is the cost being measured, not an artefact of the
        # benchmark.
        required = 128 if torch.cuda.get_device_capability()[0] >= 10 else 64
        if num_heads % required:
            q_in = q.new_zeros((num_tokens, required, D_QK))
            q_in[:, :num_heads, :] = q
        else:
            q_in = q

        def fn():
            return flash_mla_sparse_fwd(q_in, kv_3d, idx_3d, SM_SCALE, D_V)[0][
                :, :num_heads, :
            ]

    elif provider == "triton_sparse_mla_gfx950":
        q_nope = q[:, :, :D_V].contiguous()
        q_rope = q[:, :, D_V:].contiguous()

        def fn():
            return triton_sparse_mla_fwd(q_nope, q_rope, kv, indices, SM_SCALE, D_V)

    elif provider == "triton_prefill":

        def fn():
            return sparse_mla_prefill(q, kv, indices, SM_SCALE, D_V)

    elif provider == "triton_prefill_union":
        # num_heads * group must fit the 32-row union tile, the same rule the
        # backend validator applies; take the largest group that does.
        group = next((g for g in (4, 2) if num_heads * g <= UNION_TILE_ROWS), 0)
        if group == 0:
            raise RuntimeError(f"no legal union group for num_heads={num_heads}")

        def fn():
            return sparse_mla_prefill(q, kv, indices, SM_SCALE, D_V, union=group)

    else:
        raise ValueError(f"Unknown provider: {provider}")

    return run_benchmark_no_cudagraph(fn)


if __name__ == "__main__":
    bench_dsa_triton_sparse_mla_prefill.run(print_data=True)
