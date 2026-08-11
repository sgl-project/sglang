"""Micro-benchmark: split-KV EAGLE-verify kernel vs extend_attention_fwd.

Times ``verify_splitkv_fwd`` against the baseline ``extend_attention_fwd`` on
the verify shape (a few draft-token queries over a long prefix KV) across
context lengths and head dims, and reports the per-kernel latency, the speedup,
and the achieved KV-read bandwidth. Model-independent (head_dim is just a shape
parameter).

NOTE: this benchmark targets AMD MI35x (gfx950). The verify kernel's block config
and its CDNA-only Triton launch hints (waves_per_eu, matrix_instr_nonkdim) are
tuned and validated only on gfx950, and the kernel is gated to gfx95 in production
-- so these numbers are meaningful only on MI35x. GPU + Triton required.

    python3 benchmark/kernels/verify_splitkv_triton/bench_verify_splitkv.py
"""

import argparse

import torch
import triton

from sglang.kernels.ops.attention.extend_attention import (
    extend_attention_fwd,
)
from sglang.kernels.ops.attention.verify_splitkv import verify_splitkv_fwd
from sglang.srt.utils import is_gfx95_supported


def build_inputs(prefix_len, l_ext, h_q, h_kv, head_dim, v_head_dim, dtype, device):
    """One verify-shaped sequence repeated to batch size 1 per call here; the
    kernels are timed at bs=1 to isolate the per-(seq,head) bandwidth story."""
    total_prefix = prefix_len
    k_buffer = torch.randn(total_prefix, h_kv, head_dim, dtype=dtype, device=device)
    v_buffer = torch.randn(total_prefix, h_kv, v_head_dim, dtype=dtype, device=device)
    kv_indptr = torch.tensor([0, total_prefix], dtype=torch.int32, device=device)
    # kv_indices is int64 in production (TritonAttnBackend allocates int64).
    kv_indices = torch.arange(total_prefix, dtype=torch.int64, device=device)
    q = torch.randn(l_ext, h_q, head_dim, dtype=dtype, device=device)
    k = torch.randn(l_ext, h_kv, head_dim, dtype=dtype, device=device)
    v = torch.randn(l_ext, h_kv, v_head_dim, dtype=dtype, device=device)
    qo_indptr = torch.tensor([0, l_ext], dtype=torch.int32, device=device)
    return q, k, v, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, l_ext


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l-ext", type=int, default=4, help="draft tokens per seq")
    ap.add_argument("--h-q", type=int, default=16)
    ap.add_argument("--h-kv", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=256)
    args = ap.parse_args()

    if not is_gfx95_supported():
        raise SystemExit(
            "This benchmark is for AMD MI35x (gfx950) only: the verify kernel's "
            "block config and CDNA launch hints are tuned/validated there, and the "
            "kernel is gated to gfx95 in production, so results on other hardware "
            "are not representative."
        )

    if not torch.cuda.is_available():
        raise SystemExit("GPU required")
    device, dtype = "cuda", torch.bfloat16
    hd, vhd = args.head_dim, args.head_dim
    sm_scale = 1.0 / (hd**0.5)
    kv_bytes_per_tok = 2 * args.h_kv * hd * torch.tensor([], dtype=dtype).element_size()

    print(
        f"verify-split-KV vs extend_attention_fwd  "
        f"(l_ext={args.l_ext}, H_Q={args.h_q}, H_KV={args.h_kv}, head_dim={hd}, bf16)\n"
    )
    print(
        f"{'ctx':>8} {'extend(ms)':>12} {'splitkv(ms)':>12} {'speedup':>9} {'splitkv GB/s':>13}"
    )
    for ctx in (1024, 2048, 4096, 8192, 16384):
        q, k, v, kb, vb, qo, kvp, kvi, mle = build_inputs(
            ctx, args.l_ext, args.h_q, args.h_kv, hd, vhd, dtype, device
        )
        o = torch.empty(q.shape[0], args.h_q, vhd, dtype=dtype, device=device)

        def run_extend():
            extend_attention_fwd(
                q,
                k,
                v,
                o,
                kb,
                vb,
                qo,
                kvp,
                kvi,
                None,
                True,
                None,
                mle,
                1.0,
                1.0,
                sm_scale=sm_scale,
            )

        def run_split():
            verify_splitkv_fwd(
                q,
                k,
                v,
                o,
                kb,
                vb,
                qo,
                kvp,
                kvi,
                None,
                True,
                None,
                mle,
                1.0,
                1.0,
                sm_scale=sm_scale,
            )

        # Ensure the split-KV path actually handled this shape before timing it
        # (verify_splitkv_fwd returns False + no-ops on unsupported cases).
        assert verify_splitkv_fwd(
            q,
            k,
            v,
            o,
            kb,
            vb,
            qo,
            kvp,
            kvi,
            None,
            True,
            None,
            mle,
            1.0,
            1.0,
            sm_scale=sm_scale,
        ), "verify_splitkv_fwd did not handle the verify shape"

        t_ext = triton.testing.do_bench(run_extend)
        t_spl = triton.testing.do_bench(run_split)
        kv_bytes = int(kv_bytes_per_tok) * ctx
        gbs = kv_bytes / (t_spl * 1e-3) / 1e9
        print(
            f"{ctx:>8} {t_ext:>12.3f} {t_spl:>12.3f} {t_ext / t_spl:>8.2f}x {gbs:>12.0f}"
        )


if __name__ == "__main__":
    main()
