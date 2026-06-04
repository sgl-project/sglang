"""Self-contained micro-benchmark + correctness check for the qkv_lora_b Triton kernel
(``_qkv_lora_b_kernel`` / ``qkv_lora_b_fwd``, the dense LoRA-B expand-add for the fused
QKV projection).

The kernel packs the three q/k/v sgemms into one launch: for each segment it computes
``scaling * (x_slice @ B_slice.T)`` and atomic-adds the result into the matching output
columns. ``x`` is the rank-packed LoRA-A output (slice i at columns ``[i*r, (i+1)*r)``);
``B`` is ``[num_lora, N_q + 2*N_kv, r]``.

Scope per request: single adapter, rank 16, Triton path only (the 1-adapter cuBLAS dense
fast path is deliberately NOT exercised here -- it is forced off by leaving
``uniform_weight_index`` unset).

Default shape is Qwen3.5 full-attn fused QKV at TP=4: q 4 heads*256 = 1024,
kv 1 head*256 = 256 (2 KV heads replicated across 4 ranks), so total output =
1024 + 2*256 = 1536, max_qkv_out_dim = 1024.

  python3 bench_qkv_lora_b.py --mode correctness          # sweeps bs x seg shapes
  python3 bench_qkv_lora_b.py --mode bench                 # decode bs sweep, amortized device time
  python3 bench_qkv_lora_b.py --mode bench --bs 64
  python3 bench_qkv_lora_b.py --mode profile --iters 2     # eager, for ncu/nsys
"""

from __future__ import annotations

import argparse

import torch
import triton
import triton.testing

from sglang.srt.lora.triton_ops.qkv_lora_b import qkv_lora_b_fwd
from sglang.srt.lora.utils import LoRABatchInfo


def make_batch_info(seg_lens, rank, scaling, device, use_cuda_graph=True):
    """Single-adapter (slot 0) decode/prefill batch info for the Triton path.

    ``uniform_weight_index`` is left None so qkv_lora_b_fwd never dispatches the cuBLAS
    dense fast path -- this bench only measures the Triton kernel.
    """
    bs = len(seg_lens)
    seg_lens_t = torch.tensor(seg_lens, dtype=torch.int32, device=device)
    seg_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    seg_indptr[1:] = torch.cumsum(seg_lens_t, dim=0)
    return LoRABatchInfo(
        use_cuda_graph=use_cuda_graph,
        bs=bs,
        num_segments=bs,
        seg_indptr=seg_indptr,
        weight_indices=torch.zeros(bs, dtype=torch.int32, device=device),
        lora_ranks=torch.tensor([rank], dtype=torch.int32, device=device),
        scalings=torch.tensor([scaling], dtype=torch.float32, device=device),
        max_len=int(max(seg_lens)),
        seg_lens=seg_lens_t,
        permutation=None,
    )


def make_inputs(s, n_q, n_kv, rank, scaling, dtype, device, n_slices=3):
    """qkv_lora_b inputs for a single rank-`rank` adapter.

    ``x`` is the rank-packed LoRA-A output [s, n_slices*rank]; ``qkv_lora_b`` is the
    single-adapter LoRA-B weight [1, n_q + 2*n_kv, rank]. Small magnitudes keep the
    bf16 atomic-add accumulation well-conditioned.
    """
    torch.manual_seed(0)
    total_out = n_q + 2 * n_kv
    x = torch.randn(s, n_slices * rank, device=device, dtype=dtype) * 0.1
    w = torch.randn(1, total_out, rank, device=device, dtype=dtype) * 0.1
    output_offset = torch.tensor(
        [0, n_q, n_q + n_kv, n_q + 2 * n_kv], dtype=torch.int32, device=device
    )
    max_qkv_out_dim = max(n_q, n_kv)
    return x, w, output_offset, max_qkv_out_dim


def run_qkv_b(x, w, batch_info, output_offset, max_qkv_out_dim, n_slices=3):
    # base_output=None -> kernel allocates a zeroed output and atomic-adds into it,
    # so each call is self-contained (no cross-call accumulation under the bench loop).
    return qkv_lora_b_fwd(
        x,
        w,
        batch_info,
        output_offset,
        max_qkv_out_dim,
        base_output=None,
        n_slices=n_slices,
        output_offset_cpu=None,
    )


def ref_qkv_b(x, w, output_offset, scaling, rank, n_slices=3):
    s = x.shape[0]
    total_out = w.shape[-2]
    wb = w[0].float()
    out = torch.zeros(s, total_out, device=x.device, dtype=torch.float32)
    offs = output_offset.tolist()
    for i in range(n_slices):
        lo, hi = offs[i], offs[i + 1]
        xi = x[:, i * rank : (i + 1) * rank].float()
        out[:, lo:hi] = scaling * (xi @ wb[lo:hi, :rank].t())
    return out


def bench_ms(fn, warmup=25, rep=100, inner=200):
    """Per-call milliseconds via the amortized-cudagraph trick.

    A single fn()-per-graph do_bench floors at ~8-10us for any tiny op (fixed per-replay
    launch/dispatch overhead). Capturing ``inner`` back-to-back fn() calls in ONE graph and
    dividing by ``inner`` drives that overhead to ~0 and exposes the true device time.
    (Same technique as bench_expand_add_down.py / bench_shrink_splitk.py.)
    """
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            for _ in range(inner):
                fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(inner):
            fn()
    torch.cuda.synchronize()
    ms = triton.testing.do_bench(g.replay, warmup=warmup, rep=rep) / inner
    torch.cuda.synchronize()
    return float(ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode", choices=["bench", "correctness", "profile"], default="bench"
    )
    ap.add_argument(
        "--bs", type=int, default=64, help="decode batch size (1 token/req)"
    )
    ap.add_argument("--n-q", type=int, default=1024, help="q output dim (per TP shard)")
    ap.add_argument(
        "--n-kv", type=int, default=256, help="kv output dim (per TP shard)"
    )
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--scaling", type=float, default=2.0)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--tol", type=float, default=5e-2)
    args = ap.parse_args()

    dev = "cuda"
    dtype = torch.bfloat16
    n_slices = 3

    if args.mode == "correctness":
        # Sweep decode (seg_len=1) and prefill (one long segment) shapes; the Triton
        # kernel tiles segments along BLOCK_S so multi-token segments exercise the
        # pid_s loop that decode (seg_len=1) never does. Reference accumulates in fp32,
        # kernel accumulates bf16 -> generous abs tol.
        cases = [
            ("decode bs=1", [1]),
            ("decode bs=8", [1] * 8),
            ("decode bs=64", [1] * 64),
            ("prefill s=128", [128]),
            ("mixed", [1, 1, 64, 200]),
        ]
        failures = 0
        for name, seg_lens in cases:
            s = sum(seg_lens)
            x, w, output_offset, max_qkv_out_dim = make_inputs(
                s, args.n_q, args.n_kv, args.rank, args.scaling, dtype, dev, n_slices
            )
            bi = make_batch_info(seg_lens, args.rank, args.scaling, dev)
            ref = ref_qkv_b(x, w, output_offset, args.scaling, args.rank, n_slices)
            out = run_qkv_b(x, w, bi, output_offset, max_qkv_out_dim, n_slices).float()
            err = float((out - ref).abs().max().item())
            rel = err / float(ref.abs().max().item() + 1e-9)
            ok = err <= args.tol
            failures += int(not ok)
            print(
                f"{'PASS' if ok else 'FAIL'} {name:<14s} s={s:<4d} "
                f"max_abs_err={err:.4e} rel={rel:.2e}"
            )
        if failures:
            raise SystemExit(1)
        return

    # bench / profile: decode batch (seg_len=1 per request)
    seg_lens = [1] * args.bs
    s = args.bs
    x, w, output_offset, max_qkv_out_dim = make_inputs(
        s, args.n_q, args.n_kv, args.rank, args.scaling, dtype, dev, n_slices
    )
    bi = make_batch_info(seg_lens, args.rank, args.scaling, dev)
    fn = lambda: run_qkv_b(x, w, bi, output_offset, max_qkv_out_dim, n_slices)

    if args.mode == "profile":
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        for _ in range(args.iters):
            fn()
        torch.cuda.synchronize()
        return

    ms = bench_ms(fn)
    print(
        f"BENCH qkv_lora_b bs={args.bs} r={args.rank} n_q={args.n_q} n_kv={args.n_kv} "
        f"total_out={args.n_q + 2 * args.n_kv}: {ms * 1000:.2f} us (amortized true device time)"
    )


if __name__ == "__main__":
    main()
