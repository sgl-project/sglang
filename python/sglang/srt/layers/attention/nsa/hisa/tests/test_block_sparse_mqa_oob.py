"""Stress test for `_block_sparse_mqa_persistent_kernel` OOB hunt (K=128).

Production hit illegal memory access in this kernel under longbench prefill.
Hypothesis: K_CHUNKS=64 + small block_topk + specific (cu_seqlens_ks, ke,
topk_block_index) combinations expose a boundary mask bug. This script
replays many such combinations directly against the kernel — no model, no
server, no graph capture.

Run patterns
------------
1. **Plain run** (catches any direct CUDA error):

       python test_block_sparse_mqa_oob.py

2. **compute-sanitizer** for line-level OOB:

       compute-sanitizer --tool memcheck \\
           python test_block_sparse_mqa_oob.py 2>&1 | tee oob.log

   Then `grep -E "Invalid|out-of-range|memcheck" oob.log`.

The script's exit code is nonzero if any case throws.
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


# ---------------------------------------------------------------------------
# Inputs builder. Uses production HISA shapes:
#   - HEADS = 64, DIM = 128 (DSv3.2 standard)
#   - kv_block_size ∈ {8, 16, 32, 64, 128}
#   - block_topk = 8192 // kv_block_size  (production formula)
# ---------------------------------------------------------------------------

HEADS = 64
DIM = 128


def make_inputs(
    *,
    seq_q: int,                # number of query tokens (M, ragged)
    seq_kv: int,               # total K tokens (sum of ke-ks across rows)
    kv_block_size: int,
    block_topk: int,
    ks_pattern: str = "uniform",   # "uniform" | "skewed" | "edge"
    seed: int = 0,
):
    """Construct one set of (q, k, weights, ks, ke, topk_block_index)."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    # Q [seq_q, H, D] fp8 + weights [seq_q, H] f32
    q_bf16 = torch.randn(seq_q, HEADS, DIM, dtype=torch.bfloat16, device=DEVICE,
                         generator=g)
    q_fp8 = q_bf16.to(DTYPE_FP8)
    weights = torch.randn(seq_q, HEADS, dtype=torch.float32, device=DEVICE,
                          generator=g)

    # K [seq_kv, D] fp8 + scale [seq_kv] f32
    k_bf16 = torch.randn(seq_kv, DIM, dtype=torch.bfloat16, device=DEVICE,
                         generator=g)
    k_fp8 = k_bf16.to(DTYPE_FP8)
    k_scale = torch.rand(seq_kv, dtype=torch.float32, device=DEVICE, generator=g) + 0.5

    # Per-row (cu_seqlen_ks[m], cu_seqlen_ke[m]) — integers, one row per query.
    # ks_pattern controls the K-range distribution:
    #   uniform  — all rows cover [0, seq_kv)
    #   skewed   — rows cover increasingly long prefixes (extend-style)
    #   edge     — many rows have ke - ks much smaller than kv_block_size,
    #              forcing partial-block boundary cases
    if ks_pattern == "uniform":
        cu_seqlen_ks = torch.zeros(seq_q, dtype=torch.int32, device=DEVICE)
        cu_seqlen_ke = torch.full((seq_q,), seq_kv, dtype=torch.int32, device=DEVICE)
    elif ks_pattern == "skewed":
        # Each row's K range = [0, prefix_i) where prefix_i grows with i.
        ke = torch.linspace(max(kv_block_size, seq_kv // seq_q),
                            seq_kv, seq_q, device=DEVICE).round().to(torch.int32)
        cu_seqlen_ks = torch.zeros(seq_q, dtype=torch.int32, device=DEVICE)
        cu_seqlen_ke = ke
    elif ks_pattern == "edge":
        # Some rows have ke-ks just barely > 1 K-block, others much larger.
        spans = torch.randint(
            kv_block_size + 1, max(seq_kv, kv_block_size + 2),
            (seq_q,), dtype=torch.int32, device=DEVICE, generator=g,
        )
        cu_seqlen_ks = torch.zeros(seq_q, dtype=torch.int32, device=DEVICE)
        cu_seqlen_ke = torch.minimum(
            spans,
            torch.full_like(spans, seq_kv),
        )
    else:
        raise ValueError(f"unknown ks_pattern={ks_pattern}")

    # topk_block_index [seq_q, block_topk] i64, values in
    # [0, ceildiv(seq_kv, kv_block_size)).
    num_kv_blocks = (seq_kv + kv_block_size - 1) // kv_block_size
    topk_block_index = torch.randint(
        0, num_kv_blocks, (seq_q, block_topk),
        dtype=torch.int64, device=DEVICE, generator=g,
    )

    return {
        "q_fp8": q_fp8,
        "k_fp8": k_fp8,
        "k_scale": k_scale,
        "topk_block_index": topk_block_index,
        "kv_block_size": kv_block_size,
        "weights": weights,
        "cu_seqlen_ks": cu_seqlen_ks,
        "cu_seqlen_ke": cu_seqlen_ke,
    }


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def shape_cases():
    """Production-realistic shapes plus boundary stress.

    Returns list of dicts (one set of args per test case) labeled by name.
    """
    cases = []

    # Production HISA prefill shapes for K=128 (the path that crashed).
    # block_topk = 64 always (8192 // 128).
    BTK_128 = 64
    K = 128
    for seq_q in [1, 8, 64, 512, 1024, 4096, 8192]:
        for ctx in [4096, 16384, 65536, 131072]:
            if seq_q > ctx:
                continue
            for pattern in ("uniform", "skewed", "edge"):
                cases.append((
                    f"K=128 sq={seq_q:<5} ctx={ctx:<6} {pattern}",
                    dict(seq_q=seq_q, seq_kv=ctx, kv_block_size=K,
                         block_topk=BTK_128, ks_pattern=pattern),
                ))

    # Adversarial: very small block_topk (forces high K_CHUNKS masking).
    # Comment says K_CHUNKS=64 + block_topk≤256 is the regression zone.
    for btk in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        cases.append((
            f"K=128 sq=128   ctx=16K   BTK={btk:<3} adv",
            dict(seq_q=128, seq_kv=16384, kv_block_size=128,
                 block_topk=btk, ks_pattern="uniform"),
        ))

    # K=8 sanity (production already passes, used as control).
    for seq_q in [128, 1024]:
        for ctx in [16384, 65536]:
            cases.append((
                f"K=8   sq={seq_q:<5} ctx={ctx:<6} uniform",
                dict(seq_q=seq_q, seq_kv=ctx, kv_block_size=8,
                     block_topk=1024, ks_pattern="uniform"),
            ))

    return cases


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_one(label: str, kwargs: dict, seed: int = 0) -> tuple[bool, str]:
    """Build inputs and call kernel. Returns (ok, message)."""
    try:
        ins = make_inputs(seed=seed, **kwargs)
        # Force eager (sync) launch ordering inside this call only via
        # cuda.synchronize — keep CUDA_LAUNCH_BLOCKING for outer trace.
        torch.cuda.synchronize()
        logits = block_sparse_mqa_triton(
            q_fp8=ins["q_fp8"],
            k_fp8=ins["k_fp8"],
            k_scale=ins["k_scale"],
            topk_block_index=ins["topk_block_index"],
            kv_block_size=ins["kv_block_size"],
            weights=ins["weights"],
            cu_seqlen_ks=ins["cu_seqlen_ks"],
            cu_seqlen_ke=ins["cu_seqlen_ke"],
        )
        torch.cuda.synchronize()
        # Output sanity: shape + no NaN/inf.
        seq_q = ins["q_fp8"].shape[0]
        assert logits.shape == (seq_q, kwargs["block_topk"] * kwargs["kv_block_size"]), (
            f"bad output shape {logits.shape}"
        )
        # Don't check NaN — masked positions are -inf which is expected.
        return True, "ok"
    except Exception as e:
        tb = traceback.format_exc()
        return False, f"{type(e).__name__}: {e}\n{tb}"


def main():
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    print(f"CUDA_LAUNCH_BLOCKING={os.environ.get('CUDA_LAUNCH_BLOCKING')}")
    print(f"Device: {torch.cuda.get_device_name()}")
    print()

    cases = shape_cases()
    print(f"Total cases: {len(cases)}")
    print("=" * 80)

    failed = []
    seeds = [0, 1, 2]   # repeat each shape with 3 seeds to vary topk patterns
    t0 = time.time()
    for idx, (label, kwargs) in enumerate(cases):
        for seed in seeds:
            sub_label = f"[{idx:3d}] seed={seed} {label}"
            ok, msg = run_one(label, kwargs, seed=seed)
            status = "OK   " if ok else "FAIL "
            elapsed = time.time() - t0
            # Compact one-line per case; print failures verbose at the end.
            print(f"{status} t={elapsed:6.1f}s  {sub_label}", flush=True)
            if not ok:
                failed.append((sub_label, msg))
                # Bail fast: GPU may be in error state after illegal access.
                # Subsequent calls will return cascading errors.
                print(f"\nFirst failure — stopping cascade.")
                break
        if failed:
            break

    print()
    print("=" * 80)
    if not failed:
        print(f"ALL {len(cases) * len(seeds)} CASES PASSED in {time.time()-t0:.1f}s")
        return 0
    print(f"FAILED: {len(failed)} cases")
    for sub_label, msg in failed:
        print(f"\n=== {sub_label} ===\n{msg}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
