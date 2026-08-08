"""Benchmark Indexer._pad_heads_for_deep_gemm's architecture-aware head padding.

For each head count natively supported by the current architecture (see
Indexer._deep_gemm_supported_head_counts: H=32/64 on SM90, H=8/16/32/64 on
SM100/SM120), this compares calling the DeepGEMM kernel directly ("native")
against routing through the real Indexer._pad_heads_for_deep_gemm
("padded") -- the two should be ~identical, confirming the fix is a no-op
whenever padding isn't actually needed. Head counts without a native kernel
on the current architecture (H=8/16 on SM90) only get a "padded" line, and
should track the timing of the next supported count they get padded up to
(H=32), not H=64.

Covers both the decode path (next_n=1, deepgemm_paged_mqa_logits_split) and
the target-verify path (next_n>=2, deepgemm_paged_mqa_logits_native), since
the indexer applies the same padding decision on both.
"""

from __future__ import annotations

import torch
import triton

from sglang.benchmark.bench_utils import run_bench
from sglang.kernels.jit.benchmark.utils import get_benchmark_range
from sglang.kernels.ops.attention.dsa import (
    deepgemm_paged_mqa_logits_native,
    deepgemm_paged_mqa_logits_split,
)
from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer
from sglang.srt.layers.attention.dsa.utils import (
    fp8_mqa_logits_ceil_to_ue8m0,
    fp8_mqa_logits_make_fused_kv,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=20, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

try:
    import deep_gemm
except Exception:
    deep_gemm = None

BLOCK_KV = 64
HEAD_DIM = 128
HEAD_COUNTS = (8, 16, 32, 64)

shape_range = [
    (batch, seq_len_kv, next_n)
    for batch, seq_len_kv in get_benchmark_range(
        full_range=[(8, 2048), (32, 2048), (128, 8192)],
        ci_range=[(32, 2048)],
    )
    for next_n in get_benchmark_range(full_range=[1, 2], ci_range=[1, 2])
]

# One color per head count; solid line for "native", dashed for "padded".
# A head count only gets a "native" line if the current architecture
# actually has a native kernel for it (see docstring above).
_HEAD_COLORS = dict(zip(HEAD_COUNTS, ("green", "blue", "orange", "purple")))
_NATIVELY_SUPPORTED = set(Indexer._deep_gemm_supported_head_counts())

PROVIDER_CONFIGS = {}  # provider -> (num_heads, use_indexer_padding)
LINE_NAMES = []
STYLES = []
for num_heads in HEAD_COUNTS:
    variants = [("padded", True, "--")]
    if num_heads in _NATIVELY_SUPPORTED:
        variants.insert(0, ("native", False, "-"))
    for variant, use_padding, line_style in variants:
        PROVIDER_CONFIGS[f"h{num_heads}_{variant}"] = (num_heads, use_padding)
        LINE_NAMES.append(f"H={num_heads} {variant}")
        STYLES.append((_HEAD_COLORS[num_heads], line_style))
LINE_VALS = list(PROVIDER_CONFIGS)


def _make_case(batch: int, seq_len_kv: int, next_n: int, num_heads: int):
    """Mirrors _generate_test_data/_run_deepgemm_paged_mqa_logits in
    test_deepgemm_paged_mqa_logits.py: next_n>=2 (target-verify) uses the
    native wrapper with a per-position increasing ctx_lens_2d and a
    repeat_interleave'd block table; next_n==1 uses the split wrapper with a
    plain (B, 1) ctx_lens_2d.
    """
    if deep_gemm is None:
        raise RuntimeError("DeepGEMM is required for this benchmark.")

    blocks_per_seq = triton.cdiv(seq_len_kv, BLOCK_KV)
    padded_len = blocks_per_seq * BLOCK_KV
    num_blocks = batch * blocks_per_seq
    page_table = torch.arange(num_blocks, dtype=torch.int32, device="cuda").view(
        batch, blocks_per_seq
    )
    context_lens = torch.full((batch,), seq_len_kv, dtype=torch.int32, device="cuda")

    if next_n >= 2:
        offsets = torch.arange(1, next_n + 1, device="cuda", dtype=torch.int32)
        ctx_lens_2d = context_lens.unsqueeze(-1) - next_n + offsets
        page_table = page_table.repeat_interleave(next_n, dim=0)
    else:
        ctx_lens_2d = context_lens.unsqueeze(-1)
    schedule = deep_gemm.get_paged_mqa_logits_metadata(
        ctx_lens_2d, BLOCK_KV, deep_gemm.get_num_sms()
    )

    q = torch.randn(
        batch, next_n, num_heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    q_fp8 = q.to(torch.float8_e4m3fn).view(batch * next_n, num_heads, HEAD_DIM)
    weights = torch.randn(batch * next_n, num_heads, device="cuda", dtype=torch.float32)

    kv_bf16 = torch.randn(num_blocks, BLOCK_KV, HEAD_DIM, device="cuda")
    kv_amax = kv_bf16.abs().float().amax(dim=-1, keepdim=True).clamp(1e-4)
    kv_scale = fp8_mqa_logits_ceil_to_ue8m0(kv_amax / 448.0).squeeze(-1)
    kv_fp8 = (kv_bf16 / kv_scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    kv_fused = fp8_mqa_logits_make_fused_kv(kv_fp8, kv_scale, BLOCK_KV, HEAD_DIM)

    return {
        "padded_len": padded_len,
        "page_table": page_table,
        "ctx_lens_2d": ctx_lens_2d,
        "schedule": schedule,
        "q_fp8": q_fp8,
        "weights": weights,
        "kv_fused": kv_fused,
    }


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch", "seq_len_kv", "next_n"],
        x_vals=shape_range,
        x_log=False,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="dsa-indexer-head-padding-performance",
        args={},
    )
)
def benchmark(batch: int, seq_len_kv: int, next_n: int, provider: str):
    if provider not in PROVIDER_CONFIGS:
        raise ValueError(f"Unknown provider: {provider}")
    num_heads, use_indexer_padding = PROVIDER_CONFIGS[provider]

    case = _make_case(batch, seq_len_kv, next_n, num_heads=num_heads)
    q_fp8, weights = case["q_fp8"], case["weights"]
    if use_indexer_padding:
        q_fp8, weights, _ = Indexer._pad_heads_for_deep_gemm(q_fp8, weights)

    common_kwargs = dict(
        fp8_paged_mqa_logits_fn=deep_gemm.fp8_paged_mqa_logits,
        q_fp8=q_fp8,
        kv_cache_fp8=case["kv_fused"],
        weights=weights,
        ctx_lens_2d=case["ctx_lens_2d"],
        block_tables=case["page_table"],
        schedule_metadata=case["schedule"],
        max_seq_len=case["padded_len"],
        q_offset=batch * next_n,
    )
    if next_n == 1:
        fn = lambda: deepgemm_paged_mqa_logits_split(**common_kwargs)
    else:
        fn = lambda: deepgemm_paged_mqa_logits_native(
            **common_kwargs, B=batch, next_n=next_n
        )
    return tuple(t * 1000 for t in run_bench(fn, use_cuda_graph=False))


if __name__ == "__main__":
    if deep_gemm is None:
        print("[skip] DeepGEMM is unavailable.")
    else:
        benchmark.run(print_data=True)
