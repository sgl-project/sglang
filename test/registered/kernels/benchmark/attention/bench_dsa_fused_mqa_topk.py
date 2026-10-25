"""Fused LiteTopK DSA indexer top-k vs the dense paths it replaces.

``dense`` / ``dense_paged`` run the indexer's prefill path: DeepGEMM FP8 MQA
logits over row chunks sized by the logits budget, then the RAGGED or the PAGED
top-k transform. ``fused`` / ``fused_paged`` are one ``fused_mqa_topk_prefill``
launch writing the same indices. One causal request with random FP8 inputs;
every call runs with a cold L2 (naive loop, L2 flushed before each call).
"""

from types import SimpleNamespace

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.attention.dsa.fused_mqa_topk import (
    fused_mqa_topk_prefill,
    is_fused_mqa_topk_supported,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
    DSATopKBackend,
    TopkTransformMethod,
)
from sglang.srt.layers.attention.dsa.utils import resolve_fp8_mqa_logits_fn
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.dsa_fused_mqa_topk import make_page_table, make_ragged_batch

register_cuda_ci(
    est_time=60, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

TOPK = 2048
# The static logits budget SGLang derives for --mem-fraction-static 0.85, the
# GLM-5 B200 cookbook setting; a serving process may get less.
MEM_FRACTION_STATIC = 0.85


def _logits_budget_bytes() -> int:
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    return int(
        total
        * (1.0 - MEM_FRACTION_STATIC)
        * envs.SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION.get()
    )


def _dense(q_fp8, k_fp8, k_scale, weights, starts, ends, page_table, budget_bytes):
    """The dense branch of Indexer._get_topk_ragged for one request."""
    rows, keys = q_fp8.shape[0], k_fp8.shape[0]
    step = min(rows, max(1, budget_bytes // (keys * 4)))
    out = torch.empty(rows, TOPK, dtype=torch.int32, device=q_fp8.device)
    for lo in range(0, rows, step):
        hi = min(lo + step, rows)
        logits = resolve_fp8_mqa_logits_fn()(
            q_fp8[lo:hi],
            (k_fp8, k_scale),
            weights[lo:hi],
            starts[lo:hi],
            ends[lo:hi],
            clean_logits=False,
        )
        common = dict(lengths=ends[lo:hi] - starts[lo:hi], row_starts=starts[lo:hi])
        if page_table is None:
            transform = dict(
                topk_transform_method=TopkTransformMethod.RAGGED,
                attn_metadata=None,
                topk_indices_offset=starts[lo:hi],
            )
        else:
            # Like the indexer, a row chunk gathers one page-table row per logits row.
            row_batch = torch.zeros(hi - lo, dtype=torch.int32, device=q_fp8.device)
            chunked = step < rows
            transform = dict(
                topk_transform_method=TopkTransformMethod.PAGED,
                attn_metadata=SimpleNamespace(
                    page_table_1=page_table, token_to_batch_idx=row_batch
                ),
                cu_seqlens_q_topk=(
                    torch.arange(hi - lo + 1, dtype=torch.int32, device=q_fp8.device)
                    if chunked
                    else torch.tensor(
                        [0, hi - lo], dtype=torch.int32, device=q_fp8.device
                    )
                ),
                batch_idx_list=row_batch if chunked else None,
            )
        out[lo:hi] = DSATopKBackend.SGL_KERNEL.topk_transform(
            logits, topk=TOPK, **common, **transform
        )
    return out


def _fused(q_fp8, k_fp8, k_scale, weights, starts, ends, page_table, budget_bytes):
    if page_table is None:
        transform = dict(out_offsets=starts)
    else:
        transform = dict(page_table=page_table, row_batch=torch.zeros_like(starts))
    return fused_mqa_topk_prefill(
        q_fp8, k_fp8, k_scale, weights, starts, ends, TOPK, **transform
    )


@marker.parametrize("keys", [32768, 65536, 131072, 262144, 524288, 1048576], [262144])
@marker.parametrize("rows", [2048, 8192, 16384], [8192])
@marker.benchmark("impl", ["dense", "fused", "dense_paged", "fused_paged"], unit="ms")
def benchmark(keys: int, rows: int, impl: str):
    if not is_fused_mqa_topk_supported():
        marker.skip("requires CUDA SM100 or SM103")
    batch = make_ragged_batch([keys], [rows], seed=0)
    args = [
        batch[n] for n in ("q_fp8", "k_fp8", "k_scale", "weights", "starts", "ends")
    ]
    page_table = make_page_table([keys], seed=0) if impl.endswith("_paged") else None
    args += [page_table, _logits_budget_bytes()]
    return marker.do_bench(
        _dense if impl.startswith("dense") else _fused,
        input_args=tuple(args),
        use_cuda_graph=False,
        warmup_iters=3,
        replay_iters=10,
        estimated_time_ms=None,
        disable_log_bandwidth=True,
    )


if __name__ == "__main__":
    benchmark.run()
