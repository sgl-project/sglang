# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""K31HT_DSA_SELECT: per-rank candidates for the DSA DCP top-k merge from FlashInfer gvr_2.

Switch (read once at import, decode workers):
  unset / "0"  stock (default): dsa_dcp_sharded._local_candidates runs topk_transform_candidates_v2.
  "gvrm"       gvr_2 (vendored in k31ht_gvr2m, FlashInfer main 449a35734; the index-output paths unmodified) selects each row's
               top-k of its local keys (run_varlen, hint-free, next_n 1, per-row local lengths), then one
               Triton epilogue (score read from the logits row) writes the stock candidate format: (fp32 score bits, local * world + rank)
               per selected key, (0, -1) padded, slot order unspecified (as stock).
  "gvrmf"      as gvrm, but where the capture key routes to gvr_2's non-SPLIT streaming main family (the HT decode
               band) that kernel writes the candidate pairs itself (k31ht_gvr2m.run_varlen_cand: no epilogue launch);
               every other key keeps the gvrm path. Same selected set and the same pair values as gvrm by construction
               (same kernel body and plan; the score bits are the selected logit's).
               The fused form runs for calls with rows <= K31HT_DSA_SELECT_FUSED_MAX_ROWS (default 424), and for calls with
               rows <= K31HT_DSA_SELECT_FUSED_MAX_ROWS_P5B (default 1040; 0 = off) that run the warp-ballot P5 emit (host
               p5b_for: rows >= K31HT_DSA_SELECT_P5B_MIN_ROWS, 'auto' = SM count + 1); other calls keep the gvrm path.
  Both modes: gvr_2's main family uses its BLK 512 template up to K31HT_DSA_SELECT_BLK512_MAX_ROWS rows (default 424;
               read in k31ht_gvr2m.gvr2_topk_host at import), BLK 256 above.
Any other value raises at import.

Contract vs stock: the same value-exact set of each row's top-k (gvr_2 is tie-aware exact on finite
inputs). NOT tie-stable: when several keys share the score at the k-th place, stock keeps the smallest
positions, gvr_2 keeps an unspecified subset of them. The merge (dcp_topk_merge) consumes candidates as a
set, so its result stays a value-exact global top-k, but on such rows the selected keys can differ from
stock: treat the switch as numerics-changing.

gvr_2 compiles its launchers on the first eager call at a geometry (rows, logits row stride, max_seq_len)
and refuses to compile under CUDA graph capture; the eager warm-up runs before capture cover it.
"""

import hashlib
import os
import sys

import torch
import triton
import triton.language as tl

MODE = os.environ.get("K31HT_DSA_SELECT", "0")
if MODE not in ("", "0", "gvrm", "gvrmf"):
    raise ValueError(f"K31HT_DSA_SELECT must be unset, 0, gvrm or gvrmf, got {MODE!r}")
ENABLED = MODE in ("gvrm", "gvrmf")
# read at every call (tests and benchmarks flip it to compare the two forms in one process)
FUSED = MODE == "gvrmf"
# row cap of the fused form (read at every call, like FUSED); a non-integer or negative value raises at import
_MAXR = os.environ.get("K31HT_DSA_SELECT_FUSED_MAX_ROWS", "424")
if not _MAXR.isdigit():
    raise ValueError(f"K31HT_DSA_SELECT_FUSED_MAX_ROWS must be a non-negative integer, got {_MAXR!r}")
FUSED_MAX_ROWS = int(_MAXR)
# row cap of the fused form for calls that run the p5b compile (read at every call); 0 turns the raise off
_MAXR_P5B = os.environ.get("K31HT_DSA_SELECT_FUSED_MAX_ROWS_P5B", "1040")
if not _MAXR_P5B.isdigit():
    raise ValueError(f"K31HT_DSA_SELECT_FUSED_MAX_ROWS_P5B must be a non-negative integer, got {_MAXR_P5B!r}")
FUSED_MAX_ROWS_P5B = int(_MAXR_P5B)

# host-side engagement counters (calls that took the gvr_2 path; of them, calls written by the fused main family)
calls = 0
fused_calls = 0
# one 'K31HT_DSA_SELECT ENGAGE ...' line per process (stderr) at the first engaged call, for engagement checks in the logs
_engage_line = None

_BLOCK = 1024


@triton.jit
def _candidate_epilogue_kernel(
    idx_ptr, logits_ptr, logits_stride, out_ptr, rank, WORLD: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr
):
    # reads each selected key's score straight from the logits row (no separate values gather)
    row = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < K
    base = row.to(tl.int64) * K + offs
    i = tl.load(idx_ptr + base, mask=mask, other=-1)
    ok = i >= 0
    v = tl.load(logits_ptr + row.to(tl.int64) * logits_stride + i, mask=mask & ok, other=0.0)
    bits = tl.where(ok, v.to(tl.int32, bitcast=True), 0)
    pos = tl.where(ok, i * WORLD + rank, -1)
    # one 8-byte store per slot: low word = score bits, high word = global position (little endian = [.., 0], [.., 1])
    packed = (pos.to(tl.int64) << 32) | (bits.to(tl.int64) & 0xFFFFFFFF)
    tl.store(out_ptr + base, packed, mask=mask)


def _host():
    from .k31ht_gvr2m import gvr2_topk_host

    return gvr2_topk_host


def _fused_rows(rows: int) -> bool:
    """Row gate of the fused form: rows <= FUSED_MAX_ROWS, or rows <= FUSED_MAX_ROWS_P5B when the call runs the p5b compile."""
    if rows <= FUSED_MAX_ROWS:
        return True
    return rows <= FUSED_MAX_ROWS_P5B and bool(_host().p5b_for(rows, _host()._sm_count()))


def local_candidates(
    logits: torch.Tensor, local_lens: torch.Tensor, out: torch.Tensor, world: int, rank: int, topk: int
) -> torch.Tensor:
    """out[i] ([topk, 2] int32) = candidates of row i of ``logits`` over local positions 0 .. local_lens[i]-1.

    ``logits``: [rows, width] float32, inner stride 1, row stride a multiple of 4. ``out``: contiguous
    [rows, topk, 2] int32. Unlike the stock entry point, ``logits`` is not written."""
    global calls, fused_calls
    rows = out.shape[0]
    if logits.dtype != torch.float32 or logits.dim() != 2 or logits.stride(1) != 1:
        raise ValueError(f"K31HT_DSA_SELECT: logits must be 2-D float32 with inner stride 1, got {logits.dtype} {tuple(logits.stride())}")
    if logits.shape[0] != rows or not out.is_contiguous() or tuple(out.shape) != (rows, topk, 2) or out.dtype != torch.int32:
        raise ValueError("K31HT_DSA_SELECT: out must be contiguous [rows, topk, 2] int32")
    lens = local_lens
    if lens.dtype != torch.int32 or not lens.is_contiguous():
        lens = lens.to(torch.int32).contiguous()
    if FUSED and _fused_rows(rows) and _host().run_varlen_cand(
        logits, lens, out.view(rows, 2 * topk), topk, world, rank, max_seq_len=logits.shape[1]
    ):
        calls += 1
        fused_calls += 1
        if _engage_line is None:
            _log_engage(rows, logits.shape[1], world, rank, fused=True)
        return out
    idx = torch.empty((rows, topk), dtype=torch.int32, device=out.device)
    # indices only: gvr_2's values= output costs 5 extra torch launches (int64 cast, clamp, gather, copy, masked_fill)
    _host().run_varlen(logits, None, lens, idx, next_n=1, top_k=topk, max_seq_len=logits.shape[1])
    _candidate_epilogue_kernel[(rows, triton.cdiv(topk, _BLOCK))](
        idx, logits, logits.stride(0), out.view(torch.int64), rank, WORLD=world, K=topk, BLOCK=_BLOCK
    )
    calls += 1
    if _engage_line is None:
        _log_engage(rows, logits.shape[1], world, rank, fused=False)
    return out


def _log_engage(rows: int, width: int, world: int, rank: int, fused: bool) -> None:
    global _engage_line
    host = _host().__file__
    with open(host, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    _engage_line = (
        f"K31HT_DSA_SELECT ENGAGE pid={os.getpid()} mode={MODE} world={world} rank={rank} first_rows={rows} "
        f"first_width={width} first_fused={int(fused)} fused_max_rows={FUSED_MAX_ROWS} fused_max_rows_p5b={FUSED_MAX_ROWS_P5B} "
        f"blk512_max_rows={_host().BLK512_MAX_ROWS} blk512_u={_host().BLK512_U} p5b_min_rows={'auto' if _host().P5B_MIN_ROWS is None else _host().P5B_MIN_ROWS} gvr2m_host_md5={md5}"
    )
    print(_engage_line, file=sys.stderr, flush=True)
