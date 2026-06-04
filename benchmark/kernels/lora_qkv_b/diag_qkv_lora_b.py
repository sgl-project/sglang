"""Diagnosis harness for the qkv_lora_b 4x gap (see 2026-06-04-qkv-lora-b-roofline).

Reuses the validated bench_qkv_lora_b methodology (do_bench_cudagraph + L2 rotation,
base_output passed in so the timed region has no alloc/zero). Adds a single
parametrized kernel so we can sweep, in ONE harness:

  X3 (writeback mode), holding tile/warps at baseline:
    * atomic         -- baseline: tl.atomic_add (RMW + atomic semantics)
    * load_add_store -- gate_up-style RMW, no atomic (same traffic, no atomic cost)
    * store          -- overwrite only (no base read, no atomic): pure compute+write floor
    Decomposition: (atomic - load_add_store) = atomic-semantics cost;
                   (load_add_store - store)  = base-read (RMW read) cost.

  X2 (tile/occupancy), holding writeback=atomic (the production semantics):
    * BLOCK_S x BLOCK_OUT x num_warps grid; also reproduces the jybsuper#31
      BLOCK_OUT 64->128 regression on the qwen3.5 presets.

Only the 'atomic' mode is numerically correct (it equals the production kernel and
the fp32 ref); 'store'/'load_add_store' are SPEED HACKS (store overwrites instead of
adds; load_add_store double-counts under tile overlap -- here tiles are disjoint so it
is actually correct too, but we keep it labelled as a probe). Correctness of the
'atomic' mode vs the real kernel is asserted before any timing.

  python3 diag_qkv_lora_b.py --mode x3       # writeback decomposition
  python3 diag_qkv_lora_b.py --mode x2       # tile/warps sweep
  python3 diag_qkv_lora_b.py --mode verify   # assert parametrized atomic == production
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import triton
import triton.language as tl

from sglang.srt.lora.triton_ops.kernel_utils import (
    _resolve_token_positions,
)
from sglang.srt.lora.triton_ops.qkv_lora_b import qkv_lora_b_fwd

# Reuse the validated testbed helpers (same shapes / batch_info / rotation).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_qkv_lora_b import (  # noqa: E402
    PRESETS,
    auto_num_groups,
    bench_us_rotated,
    make_inputs,
    make_merged_decode_batch_info,
    ref_qkv_b,
)

WB_ATOMIC = 0
WB_LOAD_ADD_STORE = 1
WB_STORE = 2


@triton.jit
def _qkv_diag_kernel(
    x,
    weights,
    output,
    K,
    max_qkv_out_dim,
    x_stride_0,
    x_stride_1,
    w_stride_0,
    w_stride_1,
    w_stride_2,
    output_stride_0,
    output_stride_1,
    seg_lens,
    seg_indptr,
    weight_indices,
    lora_ranks,
    n_offs,
    sorted_token_ids,
    SORTED_BY_ADAPTER: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    scalings,
    WRITEBACK: tl.constexpr,
    ENABLE_PDL: tl.constexpr = False,
):
    """Copy of _qkv_lora_b_kernel with the writeback op selected by WRITEBACK and PDL
    optional (off by default for clean ncu; on to sweep the production launch path)."""
    batch_id = tl.program_id(axis=2)
    w_index = tl.load(weight_indices + batch_id)
    rank = tl.load(lora_ranks + w_index)
    if rank == 0:
        return
    qkv_id = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)
    seg_len = tl.load(seg_lens + batch_id)
    if seg_len == 0:
        return
    seg_start = tl.load(seg_indptr + batch_id)
    n_start = tl.load(n_offs + qkv_id)
    n_size = tl.load(n_offs + qkv_id + 1) - n_start
    scaling = tl.load(scalings + w_index)
    K = tl.minimum(K, rank)

    num_pid_n = tl.cdiv(max_qkv_out_dim, BLOCK_N)
    pid_s = pid // num_pid_n
    pid_n = pid % num_pid_n
    if pid_s * BLOCK_S >= seg_len:
        return

    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = tl.arange(0, BLOCK_K)

    s_physical = _resolve_token_positions(
        sorted_token_ids, seg_start, s_offset, seg_len, SORTED_BY_ADAPTER
    )
    x_ptrs = (
        x
        + (qkv_id * K) * x_stride_1
        + (s_physical[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1)
    )
    w_ptrs = (weights + w_index * w_stride_0 + n_start * w_stride_1) + (
        k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1
    )

    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    n_mask = n_offset[None, :] < n_size
    x_tile = tl.load(
        x_ptrs,
        mask=(s_offset[:, None] < seg_len) & (k_offset[None, :] < K),
        other=0.0,
    )
    w_tile = tl.load(
        w_ptrs,
        mask=(k_offset[:, None] < K) & n_mask,
        other=0.0,
    )
    partial_sum = tl.dot(x_tile.to(w_tile.dtype), w_tile)
    partial_sum *= scaling
    partial_sum = partial_sum.to(output.dtype.element_ty)

    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    output_ptr = (
        output
        + n_start * output_stride_1
        + (s_physical[:, None] * output_stride_0 + n_offset[None, :] * output_stride_1)
    )
    output_mask = (s_offset[:, None] < seg_len) & (n_offset[None, :] < n_size)
    if WRITEBACK == 0:  # WB_ATOMIC
        tl.atomic_add(output_ptr, partial_sum, mask=output_mask, sem="relaxed")
    elif WRITEBACK == 1:  # WB_LOAD_ADD_STORE
        partial_sum += tl.load(output_ptr, mask=output_mask, other=0.0)
        tl.store(output_ptr, partial_sum, mask=output_mask)
    else:  # WB_STORE
        tl.store(output_ptr, partial_sum, mask=output_mask)


def run_diag(
    x,
    w,
    batch_info,
    output_offset,
    max_out,
    base_output,
    n_slices,
    writeback: int,
    block_s: int,
    block_out: int,
    num_warps: int,
    enable_pdl: bool = False,
):
    s = x.shape[0]
    r = w.shape[-1]
    output_dim = w.shape[-2]
    grid = (
        triton.cdiv(batch_info.max_len, block_s) * triton.cdiv(max_out, block_out),
        n_slices,
        batch_info.bs,
    )
    pdl_kwargs = {"launch_pdl": True} if enable_pdl else {}
    _qkv_diag_kernel[grid](
        x,
        w,
        base_output,
        r,
        max_out,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        w.stride(2),
        base_output.stride(0),
        base_output.stride(1),
        batch_info.seg_lens,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        output_offset,
        batch_info.permutation,
        batch_info.permutation is not None,
        block_s,
        block_out,
        triton.next_power_of_2(r),
        batch_info.scalings,
        writeback,
        ENABLE_PDL=enable_pdl,
        num_warps=num_warps,
        **pdl_kwargs,
    )
    return base_output


def verify(args, device) -> None:
    """Hard guardrail (SystemExit(1) on mismatch) that the diag kernel is a faithful
    proxy of the production kernel, so the X3/X2 numbers below are trustworthy:

      * diag WB_ATOMIC          == production qkv_lora_b_fwd (atomic, default)  [bitwise]
      * diag WB_LOAD_ADD_STORE  == production qkv_lora_b_fwd (STORE=True)       [bitwise]

    The second check is the important one: the production store path is load+add+store,
    which corresponds to diag's WB_LOAD_ADD_STORE -- NOT diag's WB_STORE (pure overwrite,
    a speed-floor probe only). Both diag modes are also checked against the fp32 ref.
    """
    from sglang.srt.environ import envs

    dtype = torch.bfloat16
    failures = 0
    for preset, slice_dims in PRESETS.items():
        n_slices = len(slice_dims)
        s = args.bs
        bi = make_merged_decode_batch_info(s, args.rank, args.scaling, device)
        x, w, off, base = make_inputs(s, slice_dims, args.rank, dtype, device, seed=1)
        max_out = max(slice_dims)
        with envs.SGLANG_OPT_LORA_QKV_B_STORE.override(False):
            prod_atomic = qkv_lora_b_fwd(
                x, w, bi, off, max_out, base_output=base.clone(),
                n_slices=n_slices, output_offset_cpu=None,
            )
        with envs.SGLANG_OPT_LORA_QKV_B_STORE.override(True):
            prod_store = qkv_lora_b_fwd(
                x, w, bi, off, max_out, base_output=base.clone(),
                n_slices=n_slices, output_offset_cpu=None,
            )
        diag_atomic = run_diag(
            x, w, bi, off, max_out, base.clone(), n_slices, WB_ATOMIC, 16, 64, 4
        )
        diag_las = run_diag(
            x, w, bi, off, max_out, base.clone(), n_slices, WB_LOAD_ADD_STORE, 16, 64, 4
        )
        ref = ref_qkv_b(x, w, off, args.scaling, args.rank, base)
        bw_atomic = torch.equal(prod_atomic, diag_atomic)
        bw_store = torch.equal(prod_store, diag_las)
        err = float((diag_atomic.float() - ref).abs().max())
        las_err = float((diag_las.float() - ref).abs().max())
        ok = bw_atomic and bw_store and err < 5e-2 and las_err < 5e-2
        failures += int(not ok)
        print(
            f"{'OK  ' if ok else 'FAIL'} {preset:<12s} "
            f"diag_atomic==prod_atomic={bw_atomic} diag_las==prod_store={bw_store} "
            f"atomic_vs_ref={err:.3e} las_vs_ref={las_err:.3e}"
        )
    if failures:
        raise SystemExit(1)


def make_groups(preset, s, rank, dtype, device, l2_mult, max_groups):
    slice_dims = PRESETS[preset]
    total_out = sum(slice_dims)
    group_bytes = 2 * (s * len(slice_dims) * rank + total_out * rank + s * total_out)
    num_groups = auto_num_groups(group_bytes, l2_mult, 32, max_groups)
    groups = [
        make_inputs(s, slice_dims, rank, dtype, device, seed=g)
        for g in range(num_groups)
    ]
    return groups, num_groups, max(slice_dims), len(slice_dims)


def mode_x3(args, device) -> None:
    dtype = torch.bfloat16
    s = args.bs
    bi = make_merged_decode_batch_info(s, args.rank, args.scaling, device)
    print(f"=== X3 writeback decomposition (BLOCK_S=16 BLOCK_OUT=64 warps=4) ===")
    for preset in PRESETS:
        groups, ng, max_out, n_slices = make_groups(
            preset, s, args.rank, dtype, device, args.l2_mult, args.max_groups
        )
        row = {}
        for label, wb in [
            ("atomic", WB_ATOMIC),
            ("load_add_store", WB_LOAD_ADD_STORE),
            ("store", WB_STORE),
        ]:
            calls = [
                (
                    lambda x=x, w=w, off=off, base=base: run_diag(
                        x, w, bi, off, max_out, base, n_slices, wb, 16, 64, 4
                    )
                )
                for x, w, off, base in groups
            ]
            us = bench_us_rotated(calls, args.rep_ms)
            row[label] = us
        print(
            f"{preset:<12s} groups={ng} | atomic={row['atomic']:.2f} "
            f"load_add_store={row['load_add_store']:.2f} store={row['store']:.2f} | "
            f"atomic_cost={row['atomic']-row['load_add_store']:+.2f} "
            f"base_read_cost={row['load_add_store']-row['store']:+.2f} us"
        )


def mode_x2(args, device) -> None:
    dtype = torch.bfloat16
    s = args.bs
    bi = make_merged_decode_batch_info(s, args.rank, args.scaling, device)
    wb = {"atomic": 0, "load_add_store": 1, "store": 2}[args.wb]
    print(f"=== X2 tile/warps sweep (writeback={args.wb}) ===")
    block_s_list = [16, 32, 64]
    block_out_list = [16, 32, 64, 128, 256]
    warps_list = [2, 4, 8]
    for preset in PRESETS:
        groups, ng, max_out, n_slices = make_groups(
            preset, s, args.rank, dtype, device, args.l2_mult, args.max_groups
        )
        print(f"--- {preset} (groups={ng}, max_out={max_out}) ---")
        results = []
        for bs_ in block_s_list:
            for bo in block_out_list:
                for nw in warps_list:
                    grid0 = triton.cdiv(s, bs_) * triton.cdiv(max_out, bo)
                    calls = [
                        (
                            lambda x=x, w=w, off=off, base=base, bs_=bs_, bo=bo, nw=nw: run_diag(
                                x,
                                w,
                                bi,
                                off,
                                max_out,
                                base,
                                n_slices,
                                wb,
                                bs_,
                                bo,
                                nw,
                                args.pdl,
                            )
                        )
                        for x, w, off, base in groups
                    ]
                    try:
                        us = bench_us_rotated(calls, args.rep_ms)
                    except Exception as e:
                        us = float("inf")
                    results.append((us, bs_, bo, nw, grid0 * n_slices))
        results.sort(key=lambda t: t[0])
        for us, bs_, bo, nw, gprog in results[:8]:
            tag = " <- baseline" if (bs_, bo, nw) == (16, 64, 4) else ""
            print(
                f"  {us:6.2f}us  BLOCK_S={bs_:<3d} BLOCK_OUT={bo:<4d} warps={nw} grid_prog={gprog}{tag}"
            )
        base = next(r for r in results if (r[1], r[2], r[3]) == (16, 64, 4))
        print(
            f"  baseline(16,64,4)={base[0]:.2f}us  best={results[0][0]:.2f}us  speedup={base[0]/results[0][0]:.2f}x"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["x3", "x2", "verify"], default="x3")
    ap.add_argument(
        "--wb",
        choices=["atomic", "load_add_store", "store"],
        default="atomic",
        help="writeback mode for x2 sweep",
    )
    ap.add_argument("--pdl", action="store_true", help="enable PDL in x2 sweep")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--scaling", type=float, default=2.0)
    ap.add_argument("--l2-mult", type=float, default=4.0)
    ap.add_argument("--max-groups", type=int, default=2048)
    ap.add_argument("--rep-ms", type=int, default=100)
    args = ap.parse_args()
    device = "cuda"
    if args.mode == "verify":
        verify(args, device)
    elif args.mode == "x3":
        mode_x3(args, device)
    else:
        mode_x2(args, device)


if __name__ == "__main__":
    main()
