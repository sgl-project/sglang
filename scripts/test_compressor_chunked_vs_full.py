#!/usr/bin/env python3
"""Single-operator reproduction for the DSV4 compressor HIT-vs-MISS divergence.

Hypothesis under test
---------------------
The compressor keeps the partial compression window (the "carry") in a
per-call ping-pong workspace slot:

    vec1Info.c1v1DbIdx = (vec1Loop++ & (dbWorkspaceRatio - 1))

keyed by a CALL-LOCAL loop counter, not by the absolute compressed block
index.  A suffix call (start_pos = split, N - split tokens) therefore
assembles its windows differently from one full call (start_pos = 0, N
tokens), and the same absolute position can end up with a different
compressed value.

Probe evidence that motivated this test (swanew_10_20_3.txt):

  [C4POS-VERDICT] ... => phase_same            (RoPE proven identical)
  [WSTAGE-VERDICT] layer=0 pos=17598 first_diff_stage=S4
  [WSTAGE-VERDICT] S0..S5: first_diff_stage MODE = S0   (layers > 0: propagation)

At layer 0 the per-token pipeline S0(pre-GEMM) .. S3(post-RoPE) is identical
and the first difference is S4 (the compressed result, before the epilog).
Identical S3 inputs can only give different S4 outputs if the WINDOW
MEMBERSHIP / ORDER differs -> the carry.

Test design (no model, no scheduler, no KV pool)
------------------------------------------------
    A  full    : one call  (start_pos=0,     seqused=N)        -> out_A
    B1 first   : one call  (start_pos=0,     seqused=SPLIT)    -> state_cache updated in place
    B2 suffix  : one call  (start_pos=SPLIT, seqused=N-SPLIT)  -> out_B2
                 with B1's state_cache and B2's own absolute state table

    compare out_A[SPLIT//r : N//r]  vs  out_B2[0 : (N-SPLIT)//r]   bit-exact

API details mirror sgl-kernel-npu tests/python/sgl_kernel_npu/test_compressor.py:
op is torch.ops.npu.compressor (requires `import sgl_kernel_npu`), rope_sin /
rope_cos are positional inputs 7 and 8, state_cache_stride_dim0 must be given,
and on A5 (arch35) cache_mode=2 is the request-bank ring layout, so the state
table comes from _build_explicit_state_loc_table (position // ring % banks).

Usage
-----
    python scripts/test_compressor_chunked_vs_full.py --dry-run
    python scripts/test_compressor_chunked_vs_full.py --seq-len 17523 --split 16384
    python scripts/test_compressor_chunked_vs_full.py --seq-len 17523 --split 16385
"""

from __future__ import annotations

import argparse
import hashlib
import sys

import torch

DEVICE_ID = 0


def _md5(t) -> str:
    import numpy as np

    arr = t.detach().float().cpu().contiguous().numpy()
    return hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def _build_explicit_state_loc_table(
    start_pos, capacities, block_size, coff, cmp_ratio, banks_per_batch=2
):
    """position -> state ring row (A5 request-bank layout, same as the runtime)."""
    history_size = coff * cmp_ratio
    max_capacity = max(max(capacities, default=0), 1)
    bank_ids = torch.arange(len(start_pos) * banks_per_batch, dtype=torch.int32).view(
        len(start_pos), banks_per_batch
    )
    dummy_bank = int(bank_ids.max().item()) + 1 if bank_ids.numel() else 0
    dummy_loc = dummy_bank * block_size
    table = torch.full(
        (len(start_pos), history_size + max_capacity), dummy_loc, dtype=torch.int32
    )
    for b, (batch_start, capacity) in enumerate(zip(start_pos, capacities)):
        for column in range(history_size + capacity):
            position = batch_start - history_size + column
            if position < 0:
                continue
            bank = int(bank_ids[b, (position // block_size) % banks_per_batch])
            table[b, column] = bank * block_size + position % block_size
    return table


def _make_inputs(args):
    """Inputs for one full sequence, modelled on the KN test helper."""
    gen = torch.Generator().manual_seed(args.seed)
    n, hid = args.seq_len, args.hidden_size
    ww = args.coff * args.head_dim

    x = (torch.randn(n, hid, generator=gen) * 0.02).to(torch.bfloat16)
    wkv = (torch.randn(ww, hid, generator=gen) * 0.02).to(torch.bfloat16)
    wgate = (torch.randn(ww, hid, generator=gen) * 0.02).to(torch.bfloat16)
    ape = torch.randn(args.cmp_ratio, ww, generator=gen).float() * 0.01
    norm_weight = torch.randn(args.head_dim, generator=gen).float() * 0.02 + 1.0

    rope_rows = min(n, n // args.cmp_ratio + 1)
    rope_sin = torch.randn(rope_rows, 64, generator=gen).float() * 0.01
    rope_cos = torch.ones(rope_rows, 64) + torch.randn(rope_rows, 64, generator=gen).float() * 0.01

    state_block_size = args.ring_size
    banks_per_batch = (n + state_block_size - 1) // state_block_size + 1
    state_cache = (
        torch.randn(banks_per_batch, state_block_size, 2 * ww, generator=gen).float() * 0.01
    )
    table_full = _build_explicit_state_loc_table(
        [0], [n], state_block_size, args.coff, args.cmp_ratio, banks_per_batch
    )
    return dict(
        x=x,
        wkv=wkv,
        wgate=wgate,
        ape=ape,
        norm_weight=norm_weight,
        rope_sin=rope_sin,
        rope_cos=rope_cos,
        state_cache=state_cache,
        table_full=table_full,
        state_block_size=state_block_size,
        banks_per_batch=banks_per_batch,
    )


def _call(args, inp, state_cache, table, x, rope_sin, rope_cos, start, used):
    cu = torch.tensor([0, used], dtype=torch.int32).npu()
    return torch.ops.npu.compressor(
        x.npu(),
        inp["wkv"].npu(),
        inp["wgate"].npu(),
        state_cache,
        inp["ape"].npu(),
        inp["norm_weight"].npu(),
        rope_sin.npu(),
        rope_cos.npu(),
        state_block_table=table.npu(),
        cu_seqlens=cu,
        seqused=torch.tensor([used], dtype=torch.int32).npu(),
        start_pos=torch.tensor([start], dtype=torch.int32).npu(),
        rope_head_dim=64,
        cmp_ratio=args.cmp_ratio,
        coff=args.coff,
        norm_eps=args.norm_eps,
        rotary_mode=2,
        cache_mode=2,
        state_cache_stride_dim0=0,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--seq-len", type=int, default=17523)
    ap.add_argument("--split", type=int, default=16384)
    ap.add_argument("--hidden-size", type=int, default=4096)
    ap.add_argument("--head-dim", type=int, default=512)
    ap.add_argument("--cmp-ratio", type=int, default=4)
    ap.add_argument("--coff", type=int, default=2)
    ap.add_argument("--ring-size", type=int, default=8)
    ap.add_argument("--norm-eps", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=20260813)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    n, r, split = args.seq_len, args.cmp_ratio, args.split
    if not (0 < split < n):
        print(f"[error] split must be in (0, {n})", file=sys.stderr)
        return 2
    if split % r != 0:
        print(f"[note] split={split} is not a multiple of cmp_ratio={r}: exercises headHolderSeq != 0")

    inp = _make_inputs(args)
    ww = args.coff * args.head_dim

    if args.dry_run:
        print("=== shapes (must match the installed op) ===")
        print(f"x                : {tuple(inp['x'].shape)} bf16 (TH layout)")
        print(f"wkv / wgate      : {tuple(inp['wkv'].shape)} bf16   (coff*head_dim, hidden)")
        print(f"state_cache      : {tuple(inp['state_cache'].shape)} float32 (in-place)")
        print(f"ape              : {tuple(inp['ape'].shape)} float32")
        print(f"norm_weight      : {tuple(inp['norm_weight'].shape)} float32")
        print(f"rope_sin/cos     : {tuple(inp['rope_sin'].shape)} float32 (rows = compressed rows)")
        print(f"state_block_table: {tuple(inp['table_full'].shape)} int32 (history+capacity)")
        print(f"ring_size={args.ring_size} banks={inp['banks_per_batch']} "
              f"cmp_ratio={r} coff={args.coff} ww={ww}")
        return 0

    # Importing sgl_kernel_npu is what registers torch.ops.npu.compressor.
    import sgl_kernel_npu  # noqa: F401
    import torch_npu

    torch_npu.npu.set_device(int(DEVICE_ID))

    print("=== A: one full call ===")
    state_a = inp["state_cache"].clone().npu()
    out_a = _call(args, inp, state_a, inp["table_full"], inp["x"],
                  inp["rope_sin"], inp["rope_cos"], 0, n)
    torch_npu.npu.synchronize()
    a_blocks = {b: _md5(out_a[b]) for b in range(out_a.shape[0])}

    print("=== A again (determinism sanity) ===")
    state_a2 = inp["state_cache"].clone().npu()
    out_a2 = _call(args, inp, state_a2, inp["table_full"], inp["x"],
                   inp["rope_sin"], inp["rope_cos"], 0, n)
    torch_npu.npu.synchronize()
    if {b: _md5(out_a2[b]) for b in range(out_a2.shape[0])} != a_blocks:
        print("[FATAL] two identical full calls differ; fix the harness first")
        return 3
    print(f"[ok] full call reproducible ({out_a.shape[0]} compressed rows)")

    print(f"=== B: split at {split} ===")
    first_k = split // r
    tail_k = (n - split) // r

    table_b1 = _build_explicit_state_loc_table(
        [0], [split], inp["state_block_size"], args.coff, r, inp["banks_per_batch"]
    )
    table_b2 = _build_explicit_state_loc_table(
        [split], [n - split], inp["state_block_size"], args.coff, r, inp["banks_per_batch"]
    )
    state_b = inp["state_cache"].clone().npu()
    _call(args, inp, state_b, table_b1, inp["x"][:split],
          inp["rope_sin"][:first_k], inp["rope_cos"][:first_k], 0, split)
    out_b2 = _call(args, inp, state_b, table_b2, inp["x"][split:],
                   inp["rope_sin"][first_k:first_k + tail_k],
                   inp["rope_cos"][first_k:first_k + tail_k], split, n - split)
    torch_npu.npu.synchronize()

    diffs = []
    for i in range(min(tail_k, out_b2.shape[0])):
        blk = first_k + i
        if a_blocks.get(blk) != _md5(out_b2[i]):
            diffs.append((blk, i, a_blocks.get(blk), _md5(out_b2[i])))

    print(f"compared={min(tail_k, out_b2.shape[0])} differing={len(diffs)} "
          f"first={diffs[0] if diffs else 'none'}")
    print("VERDICT:", "CARRY-DEFECT-REPRODUCED" if diffs else "SAME (carry hypothesis dead)")
    return 1 if diffs else 0


if __name__ == "__main__":
    raise SystemExit(main())
