#!/usr/bin/env python3
"""Single-operator reproduction for the DSV4 compressor HIT-vs-MISS divergence.

Hypothesis under test
---------------------
The compressor keeps the partial compression window (the "carry") in a
per-call ping-pong workspace slot:

    vec1Info.c1v1DbIdx = (vec1Loop++ & (dbWorkspaceRatio - 1))

so the carry is keyed by a CALL-LOCAL loop counter, not by the absolute
compressed block index.  A suffix call (start_pos = 16384, 1139 tokens)
therefore assembles its windows differently from a single full call
(start_pos = 0, 17523 tokens), and the same absolute position can end up
with a different compressed value.

The probe evidence that motivated this test (swanew_10_20_3.txt):

  [C4POS-VERDICT] ... => phase_same            (RoPE is identical)
  [WSTAGE-VERDICT] layer=0 pos=17598 first_diff_stage=S4
  [WSTAGE-VERDICT] S0..S5: first_diff_stage MODE = S0   (layers > 0: propagation)

i.e. at layer 0 the per-token pipeline S0(pre-GEMM) .. S3(post-RoPE) is
identical and the first difference appears at S4 (the compressed result,
before the epilog).  Identical S3 inputs can only produce different S4
outputs if the WINDOW MEMBERSHIP / ORDER differs -> the carry.

Test design (no model, no scheduler, no KV pool)
------------------------------------------------
    A  full     : one call   (start_pos=0,     seqused=N)          -> out_A
    B1 first    : one call   (start_pos=0,     seqused=SPLIT)      -> state_cache updated in place
    B2 suffix   : one call   (start_pos=SPLIT, seqused=N-SPLIT)    -> out_B2
                   using the state_cache written by B1

    compare  out_A[SPLIT//cmp_ratio : N//cmp_ratio]
         vs  out_B2[0               : (N-SPLIT)//cmp_ratio]

Bit-exact (md5) equality is required.  A mismatch reproduces the defect at
operator level; equality kills the carry hypothesis and points at the
epilog/quantisation instead.

Caveats (must be confirmed on first run with --dry-run)
-------------------------------------------------------
* Tensor shapes below follow `compressor_def.h` (sgl-kernel-npu) and the
  golden harness in ops-transformer.  Verify them with --dry-run against
  the installed op before trusting a mismatch.
* `state_cache` is float32 and updated IN PLACE; B2 reuses B1's tensor.
* `rope_sin` / `rope_cos` must be gathered per call at ABSOLUTE positions
  (block boundary = k * cmp_ratio), never reused across calls.
* cache_mode=2 (explicit state_block_table), coff=2, rotary_mode=2 to match
  the runtime call in ascend_dsv4_backend.forward_compress.

Usage
-----
    # 1. shapes only, no op call
    python scripts/test_compressor_chunked_vs_full.py --dry-run

    # 2. the real comparison (device required)
    python scripts/test_compressor_chunked_vs_full.py \
        --seq-len 17523 --split 16384

    # 3. also probe an unaligned split (head partial window path)
    python scripts/test_compressor_chunked_vs_full.py \
        --seq-len 17523 --split 16385
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from typing import Tuple


def _md5(t) -> str:
    import numpy as np

    arr = t.detach().float().cpu().contiguous().numpy()
    return hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def _rope_tables(rope_head_dim: int, max_pos: int, device: str, theta: float = 10000.0):
    """Standard RoPE cos/sin table indexed by absolute position."""
    import torch

    half = rope_head_dim // 2
    inv = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    pos = torch.arange(max_pos, dtype=torch.float32).unsqueeze(1)
    ang = pos * inv.unsqueeze(0)
    return torch.cos(ang).to(device), torch.sin(ang).to(device)  # (max_pos, half)


def _gather_rope(cos, sin, positions_abs, device: str):
    """Gather the compact per-call table at absolute block-boundary positions."""
    import torch

    idx = torch.as_tensor(positions_abs, dtype=torch.int64, device=device)
    return cos.index_select(0, idx).float(), sin.index_select(0, idx).float()


def _build_state_block_table(
    n_blocks: int,
    history_blocks: int,
    block_size: int,
    device: str,
    batch: int = 1,
):
    """Per-position state slot table, identity-mapped over `n_blocks` blocks.

    Column j addresses the state row for logical position (j - history) * block_size.
    Identity mapping keeps block j -> block j, which is all this test needs:
    we only require that B1 and B2 see the SAME table for the same history.
    """
    import torch

    width = n_blocks + history_blocks
    cols = torch.arange(width, dtype=torch.int32)
    return cols.unsqueeze(0).repeat(batch, 1).contiguous().to(device)


def _call_compressor(
    op,
    x,
    wkv,
    wgate,
    state_cache,
    ape,
    norm_weight,
    rope_sin,
    rope_cos,
    state_block_table,
    cu_seqlens,
    seqused,
    start_pos,
    args,
):
    """Invoke the operator with the exact runtime signature."""
    if op == "npu":
        import torch

        return torch.ops.npu.compressor(
            x,
            wkv,
            wgate,
            state_cache,
            ape,
            norm_weight,
            rope_sin=rope_sin,
            rope_cos=rope_cos,
            rope_head_dim=args.rope_head_dim,
            cmp_ratio=args.cmp_ratio,
            state_block_table=state_block_table,
            cu_seqlens=cu_seqlens,
            seqused=seqused,
            start_pos=start_pos,
            coff=args.coff,
            norm_eps=args.norm_eps,
            rotary_mode=2,
            cache_mode=2,
        )
    if op == "cann":
        from cann_ops_transformer.ops import compressor as cann_compressor

        return cann_compressor(
            x,
            wkv,
            wgate,
            state_cache,
            ape,
            cmp_ratio=args.cmp_ratio,
            state_block_table=state_block_table,
            cu_seqlens=cu_seqlens,
            seqused=seqused,
            start_pos=start_pos,
            coff=args.coff,
            cache_mode=2,
        )
    raise ValueError(f"unknown --op {op!r} (expected 'npu' or 'cann')")


def _make_inputs(args):
    import torch

    dev = args.device
    torch.manual_seed(args.seed)

    n = args.seq_len
    hid = args.hidden_size
    hd = args.head_dim
    r = args.cmp_ratio

    x = torch.randn(n, hid, dtype=torch.bfloat16, device=dev)
    wkv = (torch.randn(hid, hd, dtype=torch.float32, device=dev) * 0.02).to(torch.bfloat16)
    wgate = (torch.randn(hid, hd, dtype=torch.float32, device=dev) * 0.02).to(torch.bfloat16)
    ape = torch.randn(r, hd, dtype=torch.float32, device=dev) * 0.01
    norm_weight = torch.ones(hd, dtype=torch.float32, device=dev)

    n_blocks = args.state_blocks
    state_cache = torch.zeros(n_blocks, args.block_size, 2 * hd, dtype=torch.float32, device=dev)

    return x, wkv, wgate, ape, norm_weight, state_cache


def _run_case(args, op, tag: str, split: int | None):
    """Run the full case (split=None) or the two-call chunked case."""
    import torch

    dev = args.device
    n = args.seq_len
    r = args.cmp_ratio
    x, wkv, wgate, ape, norm_weight, state_cache = _make_inputs(args)

    cos_t, sin_t = _rope_tables(args.rope_head_dim, n + r + 8, dev)

    table = _build_state_block_table(
        n_blocks=args.state_blocks,
        history_blocks=args.history_blocks,
        block_size=args.block_size,
        device=dev,
    )

    def _one_call(x_slice, start_pos, used):
        first_k = start_pos // r
        last_k = (start_pos + used) // r
        positions = [k * r for k in range(first_k, last_k)]
        rs, rc = _gather_rope(cos_t, sin_t, positions, dev)
        cu = torch.tensor([0, used], dtype=torch.int32, device=dev)
        sp = torch.tensor([start_pos], dtype=torch.int32, device=dev)
        su = torch.tensor([used], dtype=torch.int32, device=dev)
        out = _call_compressor(
            op, x_slice, wkv, wgate, state_cache, ape, norm_weight,
            rs, rc, table, cu, su, sp, args,
        )
        return out

    if split is None:
        out = _one_call(x[:n], 0, n)
        blocks = list(range(n // r))
        return {b: _md5(out[b]) for b in blocks}, out

    out1 = _one_call(x[:split], 0, split)
    out2 = _one_call(x[split:n], split, n - split)
    tail_blocks = list(range((n - split) // r))
    return {b: _md5(out2[b]) for b in tail_blocks}, out2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--op", default="npu", choices=["npu", "cann"])
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--seq-len", type=int, default=17523)
    ap.add_argument("--split", type=int, default=16384)
    ap.add_argument("--hidden-size", type=int, default=4096)
    ap.add_argument("--head-dim", type=int, default=512)
    ap.add_argument("--rope-head-dim", type=int, default=64)
    ap.add_argument("--cmp-ratio", type=int, default=4)
    ap.add_argument("--coff", type=int, default=2)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--state-blocks", type=int, default=512)
    ap.add_argument("--history-blocks", type=int, default=2)
    ap.add_argument("--norm-eps", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="print shapes only, do not call the op")
    args = ap.parse_args()

    n, r, split = args.seq_len, args.cmp_ratio, args.split
    if split % r != 0:
        print(f"[note] split={split} is NOT a multiple of cmp_ratio={r}: this exercises headHolderSeq != 0")
    if not (0 < split < n):
        print(f"[error] split must be in (0, {n})", file=sys.stderr)
        return 2

    if args.dry_run:
        print("=== shapes that must match the installed op ===")
        print(f"x                : ({n}, {args.hidden_size}) bf16")
        print(f"wkv / wgate      : ({args.hidden_size}, {args.head_dim}) bf16")
        print(f"state_cache      : ({args.state_blocks}, {args.block_size}, {2 * args.head_dim}) float32 (in-place)")
        print(f"ape              : ({r}, {args.head_dim}) float32")
        print(f"norm_weight      : ({args.head_dim},) float32")
        print(f"rope_sin/cos     : (ceil(used/{r}), {args.rope_head_dim // 2}) float32, gathered at ABS pos")
        print(f"state_block_table: (1, {args.state_blocks + args.history_blocks}) int32")
        print(f"cu_seqlens       : (2,) int32   seqused: (1,) int32   start_pos: (1,) int32")
        print(f"attrs            : rope_head_dim={args.rope_head_dim} cmp_ratio={r} coff={args.coff} "
              f"norm_eps={args.norm_eps} rotary_mode=2 cache_mode=2")
        return 0

    import torch  # noqa: F401  (device tensor creation happens in helpers)

    print("=== sanity: two identical full runs must match ===")
    full_a, _ = _run_case(args, args.op, "full-A", None)
    full_b, _ = _run_case(args, args.op, "full-B", None)
    if full_a != full_b:
        print("[FATAL] the operator is not deterministic across two identical calls; "
              "fix the harness before trusting any comparison")
        return 3
    print(f"[ok] full run is reproducible ({len(full_a)} blocks)")

    print(f"=== chunked run: split at {split} (start_pos={split}, used={n - split}) ===")
    chunked, _ = _run_case(args, args.op, "chunked", split)

    base = split // r
    never = n // r
    compared = 0
    diffs = []
    for i, blk in enumerate(range(base, min(never, base + (n - split) // r))):
        a = full_a.get(blk)
        b = chunked.get(i)
        if a is None or b is None:
            continue
        compared += 1
        if a != b:
            diffs.append((blk, i, a, b))

    print(f"compared blocks: {compared}   differing: {len(diffs)}   "
          f"first: {diffs[0] if diffs else 'none'}")
    print(f"VERDICT: {'CARRY-DEFECT-REPRODUCED' if diffs else 'SAME (carry hypothesis dead)'}")
    return 1 if diffs else 0


if __name__ == "__main__":
    raise SystemExit(main())
