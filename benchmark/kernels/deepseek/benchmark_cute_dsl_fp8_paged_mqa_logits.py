# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Standalone benchmark for the CuTe DSL FP8 Paged MQA Logits kernel
(``torch.ops.sglang.cute_dsl_fp8_paged_mqa_logits``) versus DeepGEMM's
``fp8_paged_mqa_logits`` on Blackwell SM100.

Timing backend: FlashInfer ``bench_gpu_time`` with CUPTI activity tracing
(``enable_cupti=True``) and CUDA-graph capture (``use_cuda_graph=True``) — same
configuration SGLang uses at runtime, so the numbers reflect the in-server
hot-path (launch overhead amortized via the captured graph, kernel time
measured directly via CUPTI).

Workload generation mirrors the TensorRT-LLM benchmark from
``tests/unittest/_torch/attention/sparse/test_cute_dsl_fp8_paged_mqa_logits.py``
(``benchmark_fp8_paged_mqa_logits``), with input/output shapes matched to
SGLang's NSA decode indexer (DSV3.2-style, indexer ``n_heads=64``,
``head_dim=128``, page size 64).

Usage:
    python benchmark/kernels/deepseek/benchmark_cute_dsl_fp8_paged_mqa_logits.py
    python benchmark/kernels/deepseek/benchmark_cute_dsl_fp8_paged_mqa_logits.py \\
        --batch_size 1 8 32 64 128 \\
        --next_n 1 2 4 \\
        --context_len 4096 32768 131072

Requires: cupti-python>=13 (``pip install -U cupti-python``). Falls back to
CUDA-graph + CUDA-event timing if CUPTI is unavailable.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

# Force registration of torch.ops.sglang.cute_dsl_fp8_paged_mqa_logits.
import sglang.srt.layers.attention.nsa.cute_dsl_paged_mqa_logits  # noqa: F401


def _device_cap_major() -> int:
    if not torch.cuda.is_available():
        return -1
    return torch.cuda.get_device_capability()[0]


def _ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def _make_fused_kv(
    kv_fp8: torch.Tensor,
    kv_scales: torch.Tensor,
    block_kv: int,
    head_dim: int,
) -> torch.Tensor:
    """Pack [K bytes | scale bytes] per token, viewed as [B, page, 1, D+4] uint8."""
    num_phys_blocks = kv_fp8.shape[0]
    per_token_size = head_dim + 4
    block_bytes = block_kv * per_token_size
    scale_offset = block_kv * head_dim

    fused = torch.zeros(
        num_phys_blocks, block_bytes, dtype=torch.uint8, device=kv_fp8.device
    )
    for blk in range(num_phys_blocks):
        fused[blk, :scale_offset] = kv_fp8[blk].view(torch.uint8).reshape(-1)
        fused[blk, scale_offset:] = (
            kv_scales[blk].float().contiguous().view(torch.uint8).reshape(-1)
        )
    return fused.view(num_phys_blocks, block_kv, 1, per_token_size)


def _generate_bench_data(
    batch_size: int,
    context_len: int,
    next_n: int,
    num_heads: int = 64,
    head_dim: int = 128,
    block_kv: int = 64,
    varlen: bool = False,
    device: str = "cuda",
) -> dict:
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    num_blocks_per_seq = (context_len + block_kv - 1) // block_kv

    if varlen:
        lo = min(2048, context_len)
        context_lens = torch.randint(
            lo, context_len + 1, (batch_size,), dtype=torch.int32, device=device
        )
        total_blocks = ((context_lens + block_kv - 1) // block_kv).sum().item()
        block_table = torch.zeros(
            (batch_size, num_blocks_per_seq), dtype=torch.int32, device=device
        )
        cursor = 0
        for i in range(batch_size):
            n_blks = (context_lens[i].item() + block_kv - 1) // block_kv
            block_table[i, :n_blks] = torch.arange(
                cursor, cursor + n_blks, dtype=torch.int32, device=device
            )
            cursor += n_blks
    else:
        total_blocks = batch_size * num_blocks_per_seq
        context_lens = torch.full(
            (batch_size,), context_len, dtype=torch.int32, device=device
        )
        block_table = torch.arange(
            total_blocks, dtype=torch.int32, device=device
        ).reshape(batch_size, num_blocks_per_seq)

    q_bf16 = torch.randn(
        batch_size, next_n, num_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    q_fp8 = q_bf16.to(torch.float8_e4m3fn)
    weights = torch.randn(
        batch_size * next_n, num_heads, device=device, dtype=torch.float32
    )

    kv_bf16 = torch.randn(
        total_blocks, block_kv, head_dim, device=device, dtype=torch.bfloat16
    )
    kv_amax = kv_bf16.abs().float().amax(dim=-1, keepdim=True).clamp(1e-4)
    kv_scales = _ceil_to_ue8m0(kv_amax / 448.0).squeeze(-1)
    kv_fp8 = (kv_bf16 / kv_scales.unsqueeze(-1)).to(torch.float8_e4m3fn)

    kv_fused = _make_fused_kv(kv_fp8, kv_scales, block_kv, head_dim)

    return {
        "q_fp8": q_fp8,
        "kv_fp8": kv_fp8,
        "kv_scales": kv_scales,
        "kv_fused": kv_fused,
        "weights": weights,
        "context_lens": context_lens,
        "block_table": block_table,
        "max_model_len": context_len,
        "total_blocks": total_blocks,
    }


def _choose_atom_split(
    batch: int,
    ctx: int,
    next_n: int,
    num_sms: int = 148,
    split_kv_tokens: int = 256,
    tie: str = "max_na",
    kernel_atoms=(1, 2, 3, 4),
):
    """Pick (num_atoms, atom_size) decomposition of next_n minimizing wave count;
    tie-break configurable via ``tie``:
      - ``max_na``:  prefer LARGEST num_atoms = smallest atom = most SMs busy per
                     wave; pays HBM cost of num_atoms× KV re-reads.
      - ``max_atom``: prefer LARGEST atom = smallest num_atoms = least HBM cost.

    FP8 kernel natively supports ``atom ∈ kernel_atoms`` (default ``(1, 2, 3, 4)``).
    Returns ``(num_atoms, atom)``.
    """
    cands = []
    for atom in kernel_atoms:
        if next_n % atom == 0:
            na = next_n // atom
            ntask = batch * na * ((ctx + split_kv_tokens - 1) // split_kv_tokens)
            waves = (ntask + num_sms - 1) // num_sms
            cands.append((waves, na, atom))
    if tie == "max_na":
        cands.sort(key=lambda x: (x[0], -x[1]))
    elif tie == "max_atom":
        cands.sort(key=lambda x: (x[0], x[1]))
    else:
        raise ValueError(f"unknown tie={tie!r}; expected 'max_na' or 'max_atom'")
    _, na, atom = cands[0]
    return na, atom


def benchmark(
    batch_sizes: list[int],
    next_ns: list[int],
    context_lens: list[int],
    output_dtype: torch.dtype = torch.float32,
    varlen: bool = False,
    block_kv: int = 64,
    enable_cupti: bool = True,
    use_cuda_graph: bool = True,
):
    """Benchmark CuTe DSL vs DeepGEMM FP8 paged MQA logits.

    The DSL kernel uses 1 atom per batch (real next_n positions per atom);
    DeepGEMM on SM100 falls back to per-token expansion for next_n in {2, 4+},
    which is where the DSL kernel pays off. ``use_cuda_graph=True`` matches
    SGLang's runtime decode path (cuda graph capture is enabled by default).
    """
    import deep_gemm
    from flashinfer.testing.utils import bench_gpu_time

    num_heads = 64
    head_dim = 128
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count

    dtype_str = str(output_dtype).split(".")[-1]
    mode_str = "varlen" if varlen else "fix-len"
    backend = (
        "CUPTI" if enable_cupti else ("CUDA-Graph" if use_cuda_graph else "CUDA-Event")
    )
    print(
        f"output_dtype={dtype_str}  mode={mode_str}  block_kv={block_kv}  "
        f"timing={backend}{' + CUDA-Graph' if (use_cuda_graph and enable_cupti) else ''}"
    )
    hdr = (
        f"{'batch':>5s} {'ctx':>7s} {'next_n':>6s} {'nblk':>7s} {'ntask':>6s} | "
        f"{'maxAtom':>7s} {'DSL(us)':>9s} | "
        f"{'maxNa':>5s} {'DSL(us)':>9s} {'max_atom/max_na':>15s} | "
        f"{'DG-exp(us)':>11s} {'DG-nat(us)':>11s} {'exp/DSL_maxA':>13s} {'nat/DSL_maxA':>13s}"
    )
    print(hdr)
    print("  exp = DG-expanded (q=[B*next_n,1,H,D]) — what SGLang's indexer uses today")
    print(
        "  nat = DG-native   (q=[B,next_n,H,D])   — TRT-LLM's path; not reachable from SGLang"
    )
    print("  maxAtom = baseline picker (min waves, tie-break largest atom = least HBM)")
    print(
        "  maxNa   = experimental picker (min waves, tie-break largest num_atoms = more SMs busy)"
    )
    print("-" * len(hdr))

    # See `cute_dsl_paged_mqa_logits.py` for the SPLIT_KV=256 alignment note:
    # DG metadata wrapper computes SPLIT_KV = block_kv_arg * 4 with the
    # multiplier hardcoded to 4 on SM100, and both kernels expect SPLIT_KV=256
    # (DSL: compute_tile=128 × kNumMathWarpGroups=2; DG: hardcoded). Pass 64.
    DG_METADATA_BLOCK_KV = 64

    for next_n in next_ns:
        for context_len in context_lens:
            for batch_size in batch_sizes:
                nblk = batch_size * ((context_len + block_kv - 1) // block_kv)
                SPLIT_KV_TOKENS = 256
                # Pick both atom-split strategies for A/B comparison:
                #   max_atom (baseline):     min waves, tie-break max atom (least HBM)
                #   max_na (experimental):   min waves, tie-break max num_atoms (more SMs busy)
                na_base, atom_base = _choose_atom_split(
                    batch_size,
                    context_len,
                    next_n,
                    num_sms=num_sms,
                    split_kv_tokens=SPLIT_KV_TOKENS,
                    tie="max_atom",
                    kernel_atoms=(1, 2, 3, 4),
                )
                na_exp, atom_exp = _choose_atom_split(
                    batch_size,
                    context_len,
                    next_n,
                    num_sms=num_sms,
                    split_kv_tokens=SPLIT_KV_TOKENS,
                    tie="max_na",
                    kernel_atoms=(1, 2, 3, 4),
                )
                ntask = (
                    batch_size
                    * na_base
                    * ((context_len + SPLIT_KV_TOKENS - 1) // SPLIT_KV_TOKENS)
                )

                data = _generate_bench_data(
                    batch_size,
                    context_len,
                    next_n,
                    num_heads,
                    head_dim,
                    block_kv,
                    varlen=varlen,
                )

                # Reshape Q + repeat ctx/block_table per (na, atom). weights
                # [B*next_n, H] = [B*na*atom, H] needs no reshape because the
                # row layout is preserved under [B*na, atom, ...] view.
                def _split(na, atom, data=data, B=batch_size):
                    if na > 1:
                        return {
                            "q": data["q_fp8"].reshape(
                                B * na, atom, num_heads, head_dim
                            ),
                            "ctx_lens": data["context_lens"].repeat_interleave(na),
                            "block_table": data["block_table"].repeat_interleave(
                                na, dim=0
                            ),
                        }
                    return {
                        "q": data["q_fp8"],
                        "ctx_lens": data["context_lens"],
                        "block_table": data["block_table"],
                    }

                base_t = _split(na_base, atom_base)
                strats_diverge = (na_base, atom_base) != (na_exp, atom_exp)
                exp_t = _split(na_exp, atom_exp) if strats_diverge else base_t

                # DSL: schedule input shape (B*na, 1) — picker decides na.
                dsl_schedule_meta_base = deep_gemm.get_paged_mqa_logits_metadata(
                    base_t["ctx_lens"].unsqueeze(-1),
                    DG_METADATA_BLOCK_KV,
                    num_sms,
                )
                dsl_schedule_meta_exp = (
                    deep_gemm.get_paged_mqa_logits_metadata(
                        exp_t["ctx_lens"].unsqueeze(-1),
                        DG_METADATA_BLOCK_KV,
                        num_sms,
                    )
                    if strats_diverge
                    else dsl_schedule_meta_base
                )

                def _make_dsl(t, schedule_meta, data=data):
                    def _dsl(t=t, schedule_meta=schedule_meta, data=data):
                        torch.ops.sglang.cute_dsl_fp8_paged_mqa_logits(
                            t["q"],
                            data["kv_fused"],
                            data["weights"],
                            t["ctx_lens"],
                            t["block_table"],
                            schedule_meta,
                            data["max_model_len"],
                            epi_dtype=output_dtype,
                            acc_dtype=output_dtype,
                            output_dtype=output_dtype,
                        )

                    return _dsl

                _dsl_base = _make_dsl(base_t, dsl_schedule_meta_base)
                _dsl_exp = _make_dsl(exp_t, dsl_schedule_meta_exp)

                # DG-native: schedule input shape (B, next_n) — wrapper
                # derives num_next_n_atoms = next_n. q stays [B, next_n, H, D].
                # On SM100, DG handles next_n ∈ {1, 2, 4} natively and falls
                # back to per-token expansion for next_n ∈ {3, 5+} internally.
                dg_nat_ctx_2d = (
                    data["context_lens"].unsqueeze(-1).expand(-1, next_n).contiguous()
                )
                dg_nat_schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(
                    dg_nat_ctx_2d, DG_METADATA_BLOCK_KV, num_sms
                )

                def _dg_native():
                    deep_gemm.fp8_paged_mqa_logits(
                        data["q_fp8"],
                        data["kv_fused"],
                        data["weights"],
                        dg_nat_ctx_2d,
                        data["block_table"],
                        dg_nat_schedule_meta,
                        data["max_model_len"],
                        clean_logits=False,
                    )

                # DG-expanded: SGLang's actual indexer path. q is reshaped to
                # next_n=1 with B'=B*next_n batches; block_table and ctx_lens
                # are repeat_interleave'd to match. Schedule input is
                # (B', 1) -> num_next_n_atoms=1. This is `nsa_indexer.py`'s
                # unconditional q.unsqueeze(1) layout.
                B_exp = batch_size * next_n
                q_exp = data["q_fp8"].reshape(B_exp, 1, num_heads, head_dim)
                weights_exp = data["weights"]  # already [B*next_n, H]
                # ctx per token = ctx_len for all next_n positions of a batch
                # (same as SGLang's seqlens_expanded layout for target_verify).
                ctx_exp = (
                    data["context_lens"]
                    .unsqueeze(-1)
                    .expand(-1, next_n)
                    .reshape(B_exp, 1)
                    .contiguous()
                )
                bt_exp = (
                    data["block_table"].repeat_interleave(next_n, dim=0).contiguous()
                )
                dg_exp_schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(
                    ctx_exp, DG_METADATA_BLOCK_KV, num_sms
                )

                def _dg_expanded():
                    deep_gemm.fp8_paged_mqa_logits(
                        q_exp,
                        data["kv_fused"],
                        weights_exp,
                        ctx_exp,
                        bt_exp,
                        dg_exp_schedule_meta,
                        data["max_model_len"],
                        clean_logits=False,
                    )

                bench_kwargs = dict(
                    dry_run_iters=5,
                    repeat_iters=30,
                    enable_cupti=enable_cupti,
                    use_cuda_graph=use_cuda_graph,
                    cold_l2_cache=True,
                )

                # First DSL call(s) trigger JIT compile; do them outside timing.
                _dsl_base()
                if strats_diverge:
                    _dsl_exp()
                torch.cuda.synchronize()

                base_ms = np.median(bench_gpu_time(_dsl_base, **bench_kwargs))
                exp_ms = (
                    np.median(bench_gpu_time(_dsl_exp, **bench_kwargs))
                    if strats_diverge
                    else base_ms
                )
                dg_exp_ms = np.median(bench_gpu_time(_dg_expanded, **bench_kwargs))
                dg_nat_ms = np.median(bench_gpu_time(_dg_native, **bench_kwargs))

                strat_speedup = base_ms / exp_ms if exp_ms > 0 else float("nan")
                exp_ratio = dg_exp_ms / base_ms if base_ms > 0 else float("nan")
                nat_ratio = dg_nat_ms / base_ms if base_ms > 0 else float("nan")
                base_lab = f"{na_base}/{atom_base}"
                exp_lab = f"{na_exp}/{atom_exp}"
                print(
                    f"{batch_size:5d} {context_len:7d} {next_n:6d} {nblk:7d} {ntask:6d} | "
                    f"{base_lab:>7s} {base_ms * 1e3:9.2f} | "
                    f"{exp_lab:>5s} {exp_ms * 1e3:9.2f} {strat_speedup:14.3f}x | "
                    f"{dg_exp_ms * 1e3:11.2f} {dg_nat_ms * 1e3:11.2f} "
                    f"{exp_ratio:12.3f}x {nat_ratio:12.3f}x"
                )

                torch.cuda.empty_cache()
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark CuTe DSL FP8 paged MQA logits vs DeepGEMM "
        "(CUPTI + CUDA-graph timing via flashinfer.bench_gpu_time)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="+",
        default=[1, 8, 32, 64, 128],
        help="Batch sizes to sweep (default: 1 8 32 64 128).",
    )
    parser.add_argument(
        "--next_n",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="next_n values to sweep (default: 1 2 4 — covers decode and "
        "spec-decode target_verify with num_draft_tokens in {2, 4}).",
    )
    parser.add_argument(
        "--context_len",
        type=int,
        nargs="+",
        default=[4096, 32768, 131072],
        help="Context lengths to sweep (default: 4096 32768 131072).",
    )
    parser.add_argument(
        "--output_dtype",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="Output dtype (default: float32).",
    )
    parser.add_argument(
        "--varlen",
        action="store_true",
        help="Use variable-length per-seq context (mimics mixed-batch serving). "
        "Default is fix-length for clean comparisons.",
    )
    parser.add_argument(
        "--block_kv",
        type=int,
        default=64,
        choices=[32, 64, 128],
        help="Cache page size in tokens (default: 64 — matches SGLang NSA).",
    )
    parser.add_argument(
        "--no-cupti",
        action="store_true",
        help="Disable CUPTI; use CUDA-graph + CUDA-event timing instead.",
    )
    parser.add_argument(
        "--no-cuda-graph",
        action="store_true",
        help="Disable CUDA-graph capture; measure launch overhead too.",
    )
    args = parser.parse_args()

    if _device_cap_major() != 10:
        print(
            "Skipping: CuTe DSL FP8 Paged MQA Logits kernel only supports "
            "SM 100 family (Blackwell)."
        )
        sys.exit(0)

    dtype_map = {"float32": torch.float32, "float16": torch.float16}
    benchmark(
        batch_sizes=args.batch_size,
        next_ns=args.next_n,
        context_lens=args.context_len,
        output_dtype=dtype_map[args.output_dtype],
        varlen=args.varlen,
        block_kv=args.block_kv,
        enable_cupti=not args.no_cupti,
        use_cuda_graph=not args.no_cuda_graph,
    )


if __name__ == "__main__":
    main()
