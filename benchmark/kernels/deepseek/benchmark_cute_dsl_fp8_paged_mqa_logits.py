# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

import sglang.jit_kernel.dsa.cutedsl_paged_mqa_logits  # noqa: F401
from sglang.jit_kernel.dsa import pick_dsl_expand
from sglang.srt.layers.attention.dsa.utils import (
    fp8_mqa_logits_ceil_to_ue8m0,
    fp8_mqa_logits_make_fused_kv,
)
from sglang.srt.utils import is_sm100_supported


def _generate_bench_data(
    batch_size: int,
    context_len: int,
    next_n: int,
    num_heads: int = 32,
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
    kv_scales = fp8_mqa_logits_ceil_to_ue8m0(kv_amax / 448.0).squeeze(-1)
    kv_fp8 = (kv_bf16 / kv_scales.unsqueeze(-1)).to(torch.float8_e4m3fn)

    kv_fused = fp8_mqa_logits_make_fused_kv(kv_fp8, kv_scales, block_kv, head_dim)

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


def benchmark(
    batch_sizes: list[int],
    next_ns: list[int],
    context_lens: list[int],
    output_dtype: torch.dtype = torch.float32,
    varlen: bool = False,
    block_kv: int = 64,
    use_cuda_graph: bool = True,
):
    """Benchmark CuTe DSL FP8 paged MQA logits vs DeepGEMM.

    The "DSL" column uses the SAME picker SGLang runs at runtime
    (``pick_dsl_expand``), so the table reflects the shipped path. For
    next_n=6 / num_heads<=42 that picker selects the native single-launch
    expansion (factor=1, atom=next_n, weights-in-SMEM) once there is enough
    work to fill the SMs, and a wave-minimizing split otherwise.

    The "native" column force-runs the native expansion (factor=1,
    atom=next_n) whenever it fits TMEM (``next_n*num_heads <= 256``), so the
    native path is always visible even on shapes where the picker prefers a
    split. ``use_cuda_graph=True`` matches SGLang's runtime decode path.
    """
    import deep_gemm
    from flashinfer.testing.utils import bench_gpu_time

    num_heads = 32
    head_dim = 128
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count

    dtype_str = str(output_dtype).split(".")[-1]
    mode_str = "varlen" if varlen else "fix-len"
    backend = "CUPTI + CUDA-Graph" if use_cuda_graph else "CUPTI"
    print(
        f"output_dtype={dtype_str}  mode={mode_str}  block_kv={block_kv}  "
        f"num_heads={num_heads}  timing={backend}"
    )
    hdr = (
        f"{'batch':>5s} {'ctx':>7s} {'next_n':>6s} {'nblk':>7s} | "
        f"{'pick':>5s} {'DSL(us)':>9s} | "
        f"{'native':>6s} {'nat(us)':>9s} {'nat/DSL':>8s} | "
        f"{'DG-nat(us)':>11s} {'DG/DSL':>7s}"
    )
    print(hdr)
    print("  DSL    = production picker (pick_dsl_expand) — exactly what SGLang runs")
    print("  pick   = chosen factor/atom (1/6 = native single-launch; 2/3 = split)")
    print("  native = forced native expansion (factor=1, atom=next_n); '-' if N>256")
    print("  DG-nat = deep_gemm native (q=[B,next_n,H,D]) — TRT-LLM's path")
    print("  nat/DSL>1 => native beats the picker; DG/DSL>1 => DSL beats deep_gemm")
    print("-" * len(hdr))

    # See `cutedsl_paged_mqa_logits.py` for the SPLIT_KV=256 alignment note:
    # DG metadata wrapper computes SPLIT_KV = block_kv_arg * 4 with the
    # multiplier hardcoded to 4 on SM100, and both kernels expect SPLIT_KV=256
    # (DSL: compute_tile=128 × kNumMathWarpGroups=2; DG: hardcoded). Pass 64.
    DG_METADATA_BLOCK_KV = 64

    for next_n in next_ns:
        for context_len in context_lens:
            for batch_size in batch_sizes:
                nblk = batch_size * ((context_len + block_kv - 1) // block_kv)

                data = _generate_bench_data(
                    batch_size,
                    context_len,
                    next_n,
                    num_heads,
                    head_dim,
                    block_kv,
                    varlen=varlen,
                )

                # Reshape Q + repeat ctx/block_table per (factor, atom). weights
                # [B*next_n, H] = [B*factor*atom, H] needs no reshape because the
                # row layout is preserved under [B*factor, atom, ...] view.
                def _split(factor, atom, data=data, B=batch_size):
                    if factor > 1:
                        return {
                            "q": data["q_fp8"].reshape(
                                B * factor, atom, num_heads, head_dim
                            ),
                            "ctx_lens": data["context_lens"].repeat_interleave(factor),
                            "block_table": data["block_table"].repeat_interleave(
                                factor, dim=0
                            ),
                        }
                    return {
                        "q": data["q_fp8"],
                        "ctx_lens": data["context_lens"],
                        "block_table": data["block_table"],
                    }

                def _meta(t):
                    return deep_gemm.get_paged_mqa_logits_metadata(
                        t["ctx_lens"].unsqueeze(-1), DG_METADATA_BLOCK_KV, num_sms
                    )

                def _make_dsl(t, schedule_meta, data=data, epi_dtype=output_dtype):
                    def _dsl(
                        t=t,
                        schedule_meta=schedule_meta,
                        data=data,
                        epi_dtype=epi_dtype,
                    ):
                        torch.ops.sglang.cute_dsl_fp8_paged_mqa_logits(
                            t["q"],
                            data["kv_fused"],
                            data["weights"],
                            t["ctx_lens"],
                            t["block_table"],
                            schedule_meta,
                            data["max_model_len"],
                            epi_dtype=epi_dtype,
                            acc_dtype=output_dtype,
                            output_dtype=output_dtype,
                        )

                    return _dsl

                # Production pick — exactly what SGLang's runtime selects. The op
                # then auto-tunes the epilogue (incl. max_w_in_reg=8 for native).
                factor, atom = pick_dsl_expand(
                    next_n, batch_size, context_len, num_sms, num_heads=num_heads
                )
                prod_t = _split(factor, atom)
                _dsl_prod = _make_dsl(prod_t, _meta(prod_t))

                # Forced native expansion (factor=1, atom=next_n) when it fits TMEM.
                native_fits = next_n * num_heads <= 256
                if native_fits:
                    nat_t = _split(1, next_n)
                    _dsl_nat = _make_dsl(nat_t, _meta(nat_t))

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

                bench_kwargs = dict(
                    dry_run_iters=5,
                    repeat_iters=30,
                    enable_cupti=True,
                    use_cuda_graph=use_cuda_graph,
                    cold_l2_cache=True,
                )

                # First DSL call(s) trigger JIT compile; do them outside timing.
                _dsl_prod()
                if native_fits:
                    _dsl_nat()
                torch.cuda.synchronize()

                prod_ms = np.median(bench_gpu_time(_dsl_prod, **bench_kwargs))
                nat_ms = (
                    np.median(bench_gpu_time(_dsl_nat, **bench_kwargs))
                    if native_fits
                    else float("nan")
                )
                dg_nat_ms = np.median(bench_gpu_time(_dg_native, **bench_kwargs))

                pick_lab = f"{factor}/{atom}"
                nat_over_prod = (
                    prod_ms / nat_ms if native_fits and nat_ms > 0 else float("nan")
                )
                dg_over_prod = dg_nat_ms / prod_ms if prod_ms > 0 else float("nan")
                native_tag = f"1/{next_n}" if native_fits else "-"
                nat_us_cell = f"{nat_ms * 1e3:9.2f}" if native_fits else f"{'-':>9s}"
                nat_ratio_cell = (
                    f"{nat_over_prod:7.3f}x" if native_fits else f"{'-':>8s}"
                )
                print(
                    f"{batch_size:5d} {context_len:7d} {next_n:6d} {nblk:7d} | "
                    f"{pick_lab:>5s} {prod_ms * 1e3:9.2f} | "
                    f"{native_tag:>6s} {nat_us_cell} {nat_ratio_cell} | "
                    f"{dg_nat_ms * 1e3:11.2f} {dg_over_prod:6.3f}x"
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
        default=[1, 2, 4, 6, 8, 10, 12, 14, 16],
        help="Batch sizes to sweep (default: 1 2 4 6 8 10 12 14 16).",
    )
    parser.add_argument(
        "--next_n",
        type=int,
        nargs="+",
        default=[1, 2, 4, 6],
        help="next_n values to sweep (default: 1 2 4 6 — covers decode and "
        "spec-decode target_verify with num_draft_tokens in {2, 4, 6}).",
    )
    parser.add_argument(
        "--context_len",
        type=int,
        nargs="+",
        default=[4096, 10240, 32768, 81920, 131072],
        help="Context lengths to sweep (default: 4096 10240 32768 81920 131072).",
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
        "--no-cuda-graph",
        action="store_true",
        help="Disable CUDA-graph capture; measure launch overhead too.",
    )
    args = parser.parse_args()

    if not is_sm100_supported():
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
        use_cuda_graph=not args.no_cuda_graph,
    )


if __name__ == "__main__":
    main()
