"""Sanity test that DETECTS the missing-upper-clamp bug in safe_rows.

The bug
-------
In `_block_sparse_mqa_persistent_kernel` (triton_kernels.py:844)::

    safe_rows = tl.maximum(k_rows, 0)

`safe_rows` is clamped from below (handles topk_block_ids == -1) but NOT
from above. When ``seq_kv % kv_block_size != 0``, the last topk block is
*partial*: rows ``[seq_kv, next_aligned - 1]`` are beyond the K tensor.

Two symptoms:

1. **CUDA illegal memory access** — when the OOB load crosses an allocator
   page boundary. Production hits this. Hard to reproduce reliably.
2. **Silent garbage** — when the OOB load lands in the same allocator block
   as K, no fault. ``pos_valid`` then masks the result to -inf, so the
   *output* is correct. Plain compute returns OK and the bug stays hidden.

Why the obvious test fails
--------------------------
A naive test ("set seq_kv = 65473, force last block in topk, run kernel")
returns OK because:
- The OOB load lands in the allocator's alignment padding (no fault).
- ``pos_valid`` masks the OOB output to -inf (output is correct).

This test design
----------------
We make the OOB read **observable in the output** by two tricks:

1. **View trick**: allocate ``k_poison_full`` of size ``seq_kv_aligned``
   with a distinctive POISON value in the tail. Take a narrow view
   ``k_narrow = k_poison_full[:seq_kv_unaligned]``. The kernel sees
   ``seq_kv = unaligned`` (from view's shape) but the pointer is shared
   with ``k_poison_full``, so OOB reads land on POISON values inside the
   same allocation (safe from CUDA's POV, observably wrong from ours).

2. **cu_seqlen_ke bypass**: set ``cu_seqlen_ke = seq_kv_aligned`` so
   ``pos_valid = (k_rows < ke_max)`` is True for OOB rows, letting their
   logits flow through to the output instead of being masked to -inf.

Reference: ``k_ref_full`` is the same shape but its tail is set to
``k_data[-1]`` (replicated) — i.e. the value the FIXED kernel would read
when it clamps ``safe_rows`` to ``seq_kv_unaligned - 1``.

Comparison
----------
At slot 0's OOB output columns ``[seq_kv % K, K)``::

  buggy out: logit from POISON tail (high values, large k_scale)
  ref out:   logit from k_data[-1]  (small/normal values)

If they DIFFER → bug present. If they MATCH → kernel clamped → fix works.

Sanity check: at non-OOB columns (slot 0 head + all other slots) the two
outputs must be identical, otherwise the test setup itself is broken.
"""
from __future__ import annotations

import os
import sys
import time
import traceback

import torch

from sglang.srt.layers.attention.nsa.hisa.triton_kernels import (
    block_sparse_mqa_triton,
)


DEVICE = torch.device("cuda")
DTYPE_FP8 = torch.float8_e4m3fn
HEADS = 64
DIM = 128

# POISON values: chosen large + distinctive so the dot product against any
# random q produces a logit that's clearly different from k_data[-1]·q.
# fp8_e4m3fn max is 448; 7.0 is well within range and big enough to dominate.
POISON_K = 7.0
POISON_SCALE = 100.0
ATOL = 1e-2  # output diff threshold for "different"


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def make_inputs(
    *,
    seq_q: int,
    seq_kv_unaligned: int,
    kv_block_size: int,
    block_topk: int,
    seed: int = 0,
):
    """Build (buggy_inputs, ref_inputs, bookkeeping) for one test config."""
    K = kv_block_size
    assert seq_kv_unaligned % K != 0, (
        f"seq_kv_unaligned={seq_kv_unaligned} must NOT be a multiple of K={K}"
    )
    seq_kv_aligned = ((seq_kv_unaligned + K - 1) // K) * K
    tail_size = seq_kv_aligned - seq_kv_unaligned
    last_block_idx = seq_kv_aligned // K - 1   # = num_kv_blocks - 1

    g = torch.Generator(device=DEVICE).manual_seed(seed)

    # Shared Q and W
    q_fp8 = torch.randn(seq_q, HEADS, DIM, dtype=torch.bfloat16,
                        device=DEVICE, generator=g).to(DTYPE_FP8)
    weights = torch.randn(seq_q, HEADS, dtype=torch.float32,
                          device=DEVICE, generator=g)

    # Shared k_data (first seq_kv_unaligned rows of both tensors)
    k_data_bf16 = torch.randn(seq_kv_unaligned, DIM, dtype=torch.bfloat16,
                              device=DEVICE, generator=g)
    k_data_fp8 = k_data_bf16.to(DTYPE_FP8)
    k_scale_data = (torch.rand(seq_kv_unaligned, dtype=torch.float32,
                               device=DEVICE, generator=g) + 0.5)

    # POISON tensor (full aligned size, tail = POISON values)
    k_poison_full = torch.empty(seq_kv_aligned, DIM, dtype=DTYPE_FP8, device=DEVICE)
    k_poison_full[:seq_kv_unaligned] = k_data_fp8
    k_poison_full[seq_kv_unaligned:] = (
        torch.full((tail_size, DIM), POISON_K, dtype=torch.float32, device=DEVICE)
        .to(DTYPE_FP8)
    )

    k_scale_poison = torch.empty(seq_kv_aligned, dtype=torch.float32, device=DEVICE)
    k_scale_poison[:seq_kv_unaligned] = k_scale_data
    k_scale_poison[seq_kv_unaligned:] = POISON_SCALE

    # REF tensor (full aligned, tail = k_data[-1] replicated = "what fixed
    # kernel reads after clamping safe_rows to seq_kv_unaligned - 1")
    k_ref_full = torch.empty(seq_kv_aligned, DIM, dtype=DTYPE_FP8, device=DEVICE)
    k_ref_full[:seq_kv_unaligned] = k_data_fp8
    k_ref_full[seq_kv_unaligned:] = k_data_fp8[-1:].expand(tail_size, DIM)

    k_scale_ref = torch.empty(seq_kv_aligned, dtype=torch.float32, device=DEVICE)
    k_scale_ref[:seq_kv_unaligned] = k_scale_data
    k_scale_ref[seq_kv_unaligned:] = k_scale_data[-1].item()

    # VIEWS for the buggy kernel — kernel sees shape [unaligned, ...] but
    # data pointer reaches into the POISON tail (silent OOB, no CUDA fault).
    k_narrow = k_poison_full[:seq_kv_unaligned]
    k_scale_narrow = k_scale_poison[:seq_kv_unaligned]

    # topk: slot 0 = last (partial) block; other slots = block 0 (safe)
    topk_block_index = torch.zeros(seq_q, block_topk, dtype=torch.int64, device=DEVICE)
    topk_block_index[:, 0] = last_block_idx

    # cu_seqlens: ke = aligned (bypass pos_valid masking for OOB rows)
    cu_ks = torch.zeros(seq_q, dtype=torch.int32, device=DEVICE)
    cu_ke = torch.full((seq_q,), seq_kv_aligned, dtype=torch.int32, device=DEVICE)

    buggy_call = dict(
        q_fp8=q_fp8, k_fp8=k_narrow, k_scale=k_scale_narrow,
        topk_block_index=topk_block_index, kv_block_size=K, weights=weights,
        cu_seqlen_ks=cu_ks, cu_seqlen_ke=cu_ke,
    )
    ref_call = dict(
        q_fp8=q_fp8, k_fp8=k_ref_full, k_scale=k_scale_ref,
        topk_block_index=topk_block_index, kv_block_size=K, weights=weights,
        cu_seqlen_ks=cu_ks, cu_seqlen_ke=cu_ke,
    )
    info = dict(
        K=K,
        oob_start=seq_kv_unaligned % K,
        tail_size=tail_size,
        seq_kv_unaligned=seq_kv_unaligned,
        seq_kv_aligned=seq_kv_aligned,
        last_block_idx=last_block_idx,
        block_topk=block_topk,
    )
    return buggy_call, ref_call, info


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_one(label: str, params: dict, seed: int = 0) -> tuple[str, str]:
    """Returns (status, message). Status ∈ {'BUG', 'OK', 'SANITY_FAIL', 'CRASH'}."""
    try:
        buggy_call, ref_call, info = make_inputs(seed=seed, **params)
        torch.cuda.synchronize()
        out_buggy = block_sparse_mqa_triton(**buggy_call)
        out_ref = block_sparse_mqa_triton(**ref_call)
        torch.cuda.synchronize()
    except Exception as e:
        return "CRASH", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    K = info["K"]
    oob_start = info["oob_start"]

    # OOB output cols (slot 0's tail): [oob_start, K)
    oob_buggy = out_buggy[:, oob_start:K]
    oob_ref = out_ref[:, oob_start:K]

    # Sanity: non-OOB cols MUST match (slot 0's head + all other slots)
    head_buggy = out_buggy[:, :oob_start]
    head_ref = out_ref[:, :oob_start]
    rest_buggy = out_buggy[:, K:]
    rest_ref = out_ref[:, K:]
    head_ok = torch.allclose(head_buggy, head_ref, atol=ATOL, equal_nan=True)
    rest_ok = torch.allclose(rest_buggy, rest_ref, atol=ATOL, equal_nan=True)

    if not (head_ok and rest_ok):
        head_diff = (head_buggy - head_ref).abs().max().item() if head_buggy.numel() else 0
        rest_diff = (rest_buggy - rest_ref).abs().max().item() if rest_buggy.numel() else 0
        return "SANITY_FAIL", (
            f"non-OOB cols differ — test setup broken. "
            f"head_diff={head_diff:.4f} rest_diff={rest_diff:.4f}"
        )

    # OOB cols: differ → bug present
    oob_diff = (oob_buggy - oob_ref).abs().max().item()
    if oob_diff > ATOL:
        return "BUG", (
            f"OOB max diff = {oob_diff:.4f} (>{ATOL}) — "
            f"buggy kernel read POISON instead of clamping to row "
            f"{info['seq_kv_unaligned'] - 1}"
        )
    return "OK", f"OOB max diff = {oob_diff:.6f} (<={ATOL}) — clamped correctly"


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def cases():
    cs = []

    # K=128 — production crash path. Various tail sizes.
    for offset in [1, 7, 63, 65, 97, 127]:
        cs.append((
            f"K=128 seq_kv=65536-{offset:<3} seq_q=64  btk=64",
            dict(seq_q=64, seq_kv_unaligned=65536 - offset,
                 kv_block_size=128, block_topk=64),
        ))

    # K=128 — different seq_q (grid shape coverage)
    for seq_q in [1, 8, 512, 4096]:
        cs.append((
            f"K=128 seq_kv=65473    seq_q={seq_q:<5} btk=64",
            dict(seq_q=seq_q, seq_kv_unaligned=65473,
                 kv_block_size=128, block_topk=64),
        ))

    # K=128 — small ctx
    for base in [4096, 16384, 32768]:
        cs.append((
            f"K=128 seq_kv={base-1:<5}  seq_q=64  btk=64 small_ctx",
            dict(seq_q=64, seq_kv_unaligned=base - 1,
                 kv_block_size=128, block_topk=64),
        ))

    # K=64
    for offset in [1, 33, 63]:
        cs.append((
            f"K=64  seq_kv=65536-{offset:<3} seq_q=64  btk=128",
            dict(seq_q=64, seq_kv_unaligned=65536 - offset,
                 kv_block_size=64, block_topk=128),
        ))

    # K=8 (production uses block_topk=512)
    for offset in [1, 4, 7]:
        cs.append((
            f"K=8   seq_kv=65536-{offset:<3} seq_q=64  btk=512",
            dict(seq_q=64, seq_kv_unaligned=65536 - offset,
                 kv_block_size=8, block_topk=512),
        ))

    return cs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {torch.cuda.get_device_name()}")
    print()
    print("Sanity test for `safe_rows` upper-bound clamp.")
    print("  Bug:  safe_rows = tl.maximum(k_rows, 0)        # only clamps low")
    print("  Fix:  safe_rows = tl.minimum(tl.maximum(k_rows, 0), seq_kv - 1)")
    print()
    print("Test exits 0 if kernel clamps correctly (OK on every case).")
    print("Test exits 1 if any case shows OOB reads (BUG / SANITY / CRASH).")
    print()

    all_cases = cases()
    seeds = [0, 1, 2]
    print(f"Total: {len(all_cases)} cases × {len(seeds)} seeds = "
          f"{len(all_cases) * len(seeds)} runs")
    print("=" * 90)

    counts = {"OK": 0, "BUG": 0, "SANITY_FAIL": 0, "CRASH": 0}
    failures = []
    t0 = time.time()
    for idx, (label, params) in enumerate(all_cases):
        for seed in seeds:
            sub_label = f"[{idx:2d}] seed={seed} {label}"
            status, msg = run_one(label, params, seed=seed)
            counts[status] += 1
            tag = {"OK": "OK   ", "BUG": "BUG  ",
                   "SANITY_FAIL": "SANITY", "CRASH": "CRASH"}[status]
            elapsed = time.time() - t0
            print(f"{tag} t={elapsed:5.1f}s  {sub_label}  | {msg.splitlines()[0]}",
                  flush=True)
            if status in ("CRASH", "SANITY_FAIL"):
                failures.append((sub_label, status, msg))
                # CRASH may leave GPU in bad state
                if status == "CRASH":
                    print("\nCUDA crash — stopping (GPU may be in error state).")
                    break
            elif status == "BUG":
                failures.append((sub_label, status, msg))
        if any(s == "CRASH" for _, s, _ in failures):
            break

    print()
    print("=" * 90)
    print(f"Summary: OK={counts['OK']}  BUG={counts['BUG']}  "
          f"SANITY_FAIL={counts['SANITY_FAIL']}  CRASH={counts['CRASH']}")
    print(f"Elapsed: {time.time() - t0:.1f}s")

    if counts["BUG"] == 0 and counts["SANITY_FAIL"] == 0 and counts["CRASH"] == 0:
        print("\nALL OK — kernel correctly clamps safe_rows to [0, seq_kv-1].")
        return 0

    if counts["BUG"] > 0:
        print(f"\nBUG DETECTED in {counts['BUG']} cases — kernel reads OOB (POISON).")
        print("Fix triton_kernels.py:844:")
        print("  safe_rows = tl.minimum(tl.maximum(k_rows, 0), seq_kv - 1)")
    if counts["SANITY_FAIL"] > 0:
        print(f"\nSANITY FAILURES in {counts['SANITY_FAIL']} cases — test setup is wrong.")
    if counts["CRASH"] > 0:
        print(f"\nCRASHES in {counts['CRASH']} cases — likely real CUDA OOB at page boundary.")

    print("\nDetails of first few failures:")
    for sub_label, status, msg in failures[:3]:
        print(f"\n--- [{status}] {sub_label} ---")
        print(msg)

    return 1


if __name__ == "__main__":
    sys.exit(main())
