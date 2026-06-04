"""Self-contained micro-benchmark + correctness check for the qkv_lora_b Triton kernel
(``_qkv_lora_b_kernel`` / ``qkv_lora_b_fwd``, the multi-slice fused LoRA-B expand-add).

Shapes reproduce the measured e2e decode of Qwen3.5-35B-A3B-FP8 tp4/ep4 bs64 with a
single rank-16 adapter (see the 2026-06-04 SHAPECAP capture). The kernel has TWO
distinct callsites with different shapes:

  * ``--preset linear-attn`` (30/40 layers, ``in_proj_qkvz``): n_slices=4, per-rank
    slice dims [512, 512, 1024, 1024] -> total_out 3072, max_qkv_out_dim 1024,
    x [64, 64], B [1, 3072, 16]. e2e grid (64, 4, 1).
  * ``--preset full-attn`` (10/40 layers, ``qkv_proj`` with attn_output_gate): n_slices=3,
    per-rank slice dims [2048, 256, 256] -> total_out 2560, max_qkv_out_dim 2048,
    x [64, 48], B [1, 2560, 16]. e2e grid (128, 3, 1).

Decode batch_info is the merged single-adapter segment exactly as production builds it:
bs=1, seg_lens=[64], permutation=[64] (SORTED_BY_ADAPTER=True), adapter slot 0, rank 16,
scaling 2.0, use_cuda_graph=True. The cuBLAS fast path never dispatches here: it needs
output_offset_cpu (None at this callsite) and, since the lora-mq-a merge, the
SGLANG_OPT_LORA_CUBLAS(_QKV) env opt-in.

Benchmark methodology (decode kernels are ~10us, L2-resident if naively looped):
  * ``triton.testing.do_bench_cudagraph`` -- the whole rotation sweep is captured in one
    CUDA graph, so per-call host launch overhead is ~0.
  * N buffer groups (x / B / base_output) are rotated inside the graph so no tensor is
    re-read from L2 across consecutive calls. N is auto-sized so the total footprint is
    ``--l2-mult`` (default 4) times the L2 cache; override with ``--num-groups``.
    Reported time = graph time / N.

  python3 bench_qkv_lora_b.py --mode bench --preset all
  python3 bench_qkv_lora_b.py --mode correctness
  python3 bench_qkv_lora_b.py --mode profile --preset linear-attn --iters 4   # for ncu/nsys
"""

from __future__ import annotations

import argparse
import math

import torch
import triton
import triton.testing

from sglang.srt.lora.triton_ops.qkv_lora_b import qkv_lora_b_fwd
from sglang.srt.lora.utils import LoRABatchInfo

# Per-rank output slice dims measured e2e (qwen3.5-35b tp4, SHAPECAP 2026-06-04).
PRESETS = {
    # in_proj_qkvz of the 30 linear-attention (GatedDeltaNet) layers: q, k, v, z.
    "linear-attn": [512, 512, 1024, 1024],
    # qkv_proj of the 10 full-attention layers: q (gated, 2x), k, v.
    "full-attn": [2048, 256, 256],
}


def disable_pdl(modules) -> None:
    """Launch kernels without PDL (launch_pdl/gdc_wait). The default PDL launch lets
    back-to-back identical kernels in the bench graph overlap launch tails, reporting
    a faster per-call time than an e2e nsys duration, which includes the gdc_wait
    stall on a DIFFERENT (often slower) producer kernel. --no-pdl gives the
    standalone-execution number for comparing against e2e profile durations."""
    import sglang.srt.lora.triton_ops.kernel_utils as _ku

    def no_pdl():
        return False, {}

    _ku.get_pdl_launch_metadata = no_pdl
    for mod in modules:
        mod.get_pdl_launch_metadata = no_pdl
    globals()["get_pdl_launch_metadata"] = no_pdl


def make_merged_decode_batch_info(
    s: int, rank: int, scaling: float, device, shuffle_permutation: bool = False
) -> LoRABatchInfo:
    """The production single-adapter decode batch info: all ``s`` tokens merged into ONE
    segment (bs=1), with a token permutation (SORTED_BY_ADAPTER=True).

    ``use_cuda_graph=True`` like real decode; the cuBLAS fast path is off because
    output_offset_cpu=None here (and it additionally needs the SGLANG_OPT_LORA_CUBLAS
    env opt-in since the lora-mq-a merge).
    """
    permutation = torch.arange(s, dtype=torch.int32, device=device)
    if shuffle_permutation:
        permutation = permutation[torch.randperm(s, device=device)]
    return LoRABatchInfo(
        use_cuda_graph=True,
        bs=1,
        num_segments=1,
        seg_indptr=torch.tensor([0, s], dtype=torch.int64, device=device),
        weight_indices=torch.zeros(1, dtype=torch.int32, device=device),
        lora_ranks=torch.tensor([rank], dtype=torch.int64, device=device),
        scalings=torch.tensor([scaling], dtype=torch.float32, device=device),
        max_len=s,
        seg_lens=torch.tensor([s], dtype=torch.int64, device=device),
        permutation=permutation,
        single_adapter=(0, rank),
    )


def make_multiseg_batch_info(
    seg_lens: list[int], rank: int, scaling: float, device
) -> LoRABatchInfo:
    """Plain unsorted multi-segment batch info (SORTED_BY_ADAPTER=False), for
    correctness coverage of the non-permuted indexing path."""
    bs = len(seg_lens)
    seg_lens_t = torch.tensor(seg_lens, dtype=torch.int32, device=device)
    seg_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    seg_indptr[1:] = torch.cumsum(seg_lens_t, dim=0)
    return LoRABatchInfo(
        use_cuda_graph=True,
        bs=bs,
        num_segments=bs,
        seg_indptr=seg_indptr,
        weight_indices=torch.zeros(bs, dtype=torch.int32, device=device),
        lora_ranks=torch.tensor([rank], dtype=torch.int64, device=device),
        scalings=torch.tensor([scaling], dtype=torch.float32, device=device),
        max_len=int(max(seg_lens)),
        seg_lens=seg_lens_t,
        permutation=None,
    )


def make_inputs(s: int, slice_dims: list[int], rank: int, dtype, device, seed: int = 0):
    """x is the rank-packed LoRA-A output (slice i at columns [i*r, (i+1)*r));
    B is the single-adapter LoRA-B weight [1, sum(slice_dims), r]; base_output is a
    non-zero stand-in for the base GEMM output the kernel atomic-adds into."""
    gen = torch.Generator(device=device).manual_seed(seed)
    n_slices = len(slice_dims)
    total_out = sum(slice_dims)
    x = torch.randn(s, n_slices * rank, generator=gen, device=device, dtype=dtype) * 0.1
    w = torch.randn(1, total_out, rank, generator=gen, device=device, dtype=dtype) * 0.1
    base_output = (
        torch.randn(s, total_out, generator=gen, device=device, dtype=dtype) * 0.1
    )
    output_offset = torch.zeros(n_slices + 1, dtype=torch.int32, device=device)
    output_offset[1:] = torch.cumsum(
        torch.tensor(slice_dims, dtype=torch.int32, device=device), dim=0
    )
    return x, w, output_offset, base_output


def run_qkv_b(x, w, batch_info, output_offset, max_qkv_out_dim, base_output, n_slices):
    # base_output is passed in (the e2e callsite adds into the base GEMM output), so the
    # timed region contains no allocation/zeroing kernel.
    return qkv_lora_b_fwd(
        x,
        w,
        batch_info,
        output_offset,
        max_qkv_out_dim,
        base_output=base_output,
        n_slices=n_slices,
        output_offset_cpu=None,
    )


def ref_qkv_b(x, w, output_offset, scaling, rank, base_output):
    """fp32 reference of the expand-add: base + scaling * (x_slice @ B_slice^T)."""
    out = base_output.float().clone()
    offs = output_offset.tolist()
    wb = w[0].float()
    for i in range(len(offs) - 1):
        lo, hi = offs[i], offs[i + 1]
        xi = x[:, i * rank : (i + 1) * rank].float()
        out[:, lo:hi] += scaling * (xi @ wb[lo:hi, :rank].t())
    return out


def auto_num_groups(
    group_bytes: int, l2_mult: float, min_groups: int, max_groups: int
) -> int:
    """Enough buffer groups that the rotation footprint is ``l2_mult`` x L2, so no
    group survives in L2 until its next use. Err on the high side: an optimized kernel
    that reads less memory needs MORE groups for the same eviction guarantee."""
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    l2_bytes = getattr(props, "L2_cache_size", 128 * 1024 * 1024)
    need = math.ceil(l2_bytes * l2_mult / max(group_bytes, 1))
    n = max(min_groups, min(need, max_groups))
    if n * group_bytes < l2_mult * l2_bytes:
        print(
            f"WARNING: rotation footprint {n * group_bytes / 1e6:.0f} MB < "
            f"{l2_mult:.1f} x L2 ({l2_mult * l2_bytes / 1e6:.0f} MB); raise "
            f"--max-groups for full L2 eviction (small-shape kernel)"
        )
    return n


def bench_us_rotated(calls, rep_ms: int) -> float:
    """Capture all rotated calls in ONE CUDA graph via do_bench_cudagraph; per-call us =
    graph time / num_groups. Host launch overhead amortizes to ~0; rotation defeats L2.
    """

    def fn():
        for call in calls:
            call()

    fn()  # eager warmup: triton JIT compile outside graph capture
    torch.cuda.synchronize()
    ms = triton.testing.do_bench_cudagraph(fn, rep=rep_ms)
    return float(ms) * 1e3 / len(calls)


def run_correctness(args, dtype, device) -> None:
    cases = [
        ("decode-merged(e2e)", "merged", 64, False),
        ("decode-merged-shuffled", "merged", 64, True),
        ("decode-merged bs128", "merged", 128, False),
        ("multiseg [1,1,64,200]", "multiseg", [1, 1, 64, 200], False),
        ("prefill s=256", "multiseg", [256], False),
    ]
    failures = 0
    for preset, slice_dims in PRESETS.items():
        n_slices = len(slice_dims)
        for name, kind, arg, shuffle in cases:
            if kind == "merged":
                s = arg
                bi = make_merged_decode_batch_info(
                    s, args.rank, args.scaling, device, shuffle_permutation=shuffle
                )
            else:
                s = sum(arg)
                bi = make_multiseg_batch_info(arg, args.rank, args.scaling, device)
            x, w, output_offset, base = make_inputs(
                s, slice_dims, args.rank, dtype, device
            )
            ref = ref_qkv_b(x, w, output_offset, args.scaling, args.rank, base)
            out = run_qkv_b(
                x, w, bi, output_offset, max(slice_dims), base.clone(), n_slices
            ).float()
            err = float((out - ref).abs().max().item())
            rel = err / float(ref.abs().max().item() + 1e-9)
            ok = err <= args.tol
            failures += int(not ok)
            print(
                f"{'PASS' if ok else 'FAIL'} {preset:<12s} {name:<24s} s={s:<4d} "
                f"max_abs_err={err:.4e} rel={rel:.2e}"
            )
    if failures:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode", choices=["bench", "correctness", "profile"], default="bench"
    )
    ap.add_argument(
        "--preset", choices=[*PRESETS, "all"], default="all", help="qkv callsite shape"
    )
    ap.add_argument("--bs", type=int, default=64, help="decode batch size (1 tok/req)")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--scaling", type=float, default=2.0)
    ap.add_argument(
        "--num-groups",
        type=int,
        default=0,
        help="rotated buffer groups; 0 = auto-size to --l2-mult x L2",
    )
    ap.add_argument("--l2-mult", type=float, default=4.0)
    ap.add_argument("--min-groups", type=int, default=32)
    ap.add_argument("--max-groups", type=int, default=1024)
    ap.add_argument("--rep-ms", type=int, default=100)
    ap.add_argument("--iters", type=int, default=4, help="profile-mode eager sweeps")
    ap.add_argument("--tol", type=float, default=5e-2)
    ap.add_argument(
        "--no-pdl", action="store_true", help="disable PDL (see disable_pdl docstring)"
    )
    args = ap.parse_args()
    if args.no_pdl:
        import sglang.srt.lora.triton_ops.qkv_lora_b as _qkv

        disable_pdl([_qkv])

    device = "cuda"
    dtype = torch.bfloat16

    if args.mode == "correctness":
        run_correctness(args, dtype, device)
        return

    presets = list(PRESETS) if args.preset == "all" else [args.preset]
    s = args.bs
    bi = make_merged_decode_batch_info(s, args.rank, args.scaling, device)

    for preset in presets:
        slice_dims = PRESETS[preset]
        n_slices = len(slice_dims)
        total_out = sum(slice_dims)
        max_out = max(slice_dims)

        group_bytes = 2 * (
            s * n_slices * args.rank + total_out * args.rank + s * total_out
        )
        num_groups = args.num_groups or auto_num_groups(
            group_bytes, args.l2_mult, args.min_groups, args.max_groups
        )
        groups = [
            make_inputs(s, slice_dims, args.rank, dtype, device, seed=g)
            for g in range(num_groups)
        ]
        calls = [
            (
                lambda x=x, w=w, off=off, base=base: run_qkv_b(
                    x, w, bi, off, max_out, base, n_slices
                )
            )
            for x, w, off, base in groups
        ]
        # Mirror of qkv_lora_b_fwd's launch geometry (BLOCK_S=16, BLOCK_OUT=64; the
        # jybsuper#31 BLOCK_OUT=128 change was reverted after the testbed measured a
        # ~70% decode regression).
        grid = (
            triton.cdiv(s, 16) * triton.cdiv(max_out, 64),
            n_slices,
            1,
        )

        if args.mode == "profile":
            for _ in range(2):
                calls[0]()
            torch.cuda.synchronize()
            for _ in range(args.iters):
                for call in calls:
                    call()
            torch.cuda.synchronize()
            print(f"PROFILE {preset}: {args.iters} sweeps x {num_groups} groups done")
            continue

        us = bench_us_rotated(calls, args.rep_ms)
        print(
            f"BENCH qkv_lora_b preset={preset} s={s} r={args.rank} n_slices={n_slices} "
            f"total_out={total_out} max_out={max_out} grid={grid} "
            f"groups={num_groups} ({group_bytes * num_groups / 1e6:.0f} MB rotated): "
            f"{us:.2f} us"
        )


if __name__ == "__main__":
    main()
