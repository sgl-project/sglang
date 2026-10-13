# SPDX-License-Identifier: Apache-2.0
"""``SGLANG_MONO_DSV41_K2A=1`` inside SGLang's DSV4 HIP indexer (decode / verify rows): the K2i scorer replaces
``aiter_fp4_paged_mqa_logits``; the selection stays SGLang's (``topk_transform_paged_hip``, the candidate
blocks), except that a candidate consumer's top-k runs on K2i's compact candidate logits directly (no gather).
A step whose rows do not fit (more than MAX_ROWS, another plane page) takes SGLang's path unchanged."""

import os
import sys

import torch

from . import index_plan as ip

_MODE = []
_RUNNERS: dict = {}


def enabled() -> bool:
    if not _MODE:
        from ...spec import k2a_mode

        _MODE.append(k2a_mode())
    return _MODE[0] == 1


_STATS = {"full": [], "reindex": [], "logit_err": [], "calls": 0}


def _checking() -> bool:
    """``SGLANG_MONO_DSV41_K2A_CHECK=1`` (eager steps only): SGLang's path runs too and the selections are
    compared (per-row Jaccard, logits' max relative error), logged every 64 layers as ``[K2A_CHECK]``."""
    return (
        os.environ.get("SGLANG_MONO_DSV41_K2A_CHECK", "0") == "1"
        and not torch.cuda.is_current_stream_capturing()
    )


def _jaccard(a, b, bounds):
    out = []
    for x, y, n in zip(a.tolist(), b.tolist(), bounds.tolist()):
        if n <= ip.TOPK:
            continue
        x, y = {v for v in x if v >= 0}, {v for v in y if v >= 0}
        out.append(len(x & y) / max(len(x | y), 1))
    return out


def _report():
    _STATS["calls"] += 1
    if _STATS["calls"] % 64:
        return
    import numpy as np

    parts = []
    for kind in ("full", "reindex"):
        js = _STATS[kind]
        if js:
            parts.append(f"{kind} rows {len(js)} median {np.median(js):.4f} p1 {np.percentile(js, 1):.4f} "
                         f"min {min(js):.4f}")  # fmt: skip
    if _STATS["logit_err"]:
        parts.append(f"logits max rel err {max(_STATS['logit_err']):.2e}")
    print(
        f"[K2A_CHECK] layers {_STATS['calls']}: " + "; ".join(parts),
        file=sys.stderr,
        flush=True,
    )


def _check_full(logits, sg_logits, meta):
    from sglang.kernels.ops.attention.dsv4.candidate_blocks_hip import (
        topk_transform_paged_hip,
    )

    want = sg_logits()
    bounds = _bounds(meta)
    rows = logits.shape[0]
    for t, n in enumerate(bounds.tolist()):
        if n > ip.TOPK:
            a, b = logits[t, :n], want[t, :n]
            _STATS["logit_err"].append(
                float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))
            )
    sel = [
        torch.empty(rows, ip.TOPK, dtype=torch.int32, device=logits.device)
        for _ in range(4)
    ]
    topk_transform_paged_hip(
        logits, bounds, meta.page_table, sel[0], meta.compressed_page_size, sel[1]
    )
    topk_transform_paged_hip(
        want, bounds, meta.page_table, sel[2], meta.compressed_page_size, sel[3]
    )
    _STATS["full"] += _jaccard(sel[1], sel[3], bounds)
    _report()


def _runner(backend, device):
    from .runner import K2iRunner

    key = torch.device(device)
    if key not in _RUNNERS:
        _RUNNERS[key] = K2iRunner(key, backend.max_context_len)
    return _RUNNERS[key]


def _bounds(meta) -> torch.Tensor:
    lens = meta.compressed_seq_lens
    return (
        lens
        if lens.dtype == torch.int32 and lens.is_contiguous()
        else lens.to(torch.int32).contiguous()
    )


def fits(meta, rows: int) -> bool:
    if not enabled():
        return False
    from .runner import MAX_ROWS, PLANE_ROWS

    return (
        0 < rows <= MAX_ROWS
        and meta.compressed_page_size == PLANE_ROWS
        and meta.page_table.dtype == torch.int32
        and meta.page_table.shape[0] == rows
        and meta.page_table.stride(1) == 1
    )


def step_plans(backend, metadata_by_ratio: dict, forward_batch) -> dict:
    """Once a step (in the graph when captured): each scored ratio's FULL walk plan. -> ratio -> (rows, rows a
    request) of the ratios it planned (the step's ``k2i_planned``: their aiter decode schedule is not needed)."""
    done = {}
    for ratio, meta in metadata_by_ratio.items():
        rows = meta.page_table.shape[0]
        if not fits(meta, rows) or rows % forward_batch.batch_size:
            continue
        per_req = rows // forward_batch.batch_size
        r = _runner(backend, meta.page_table.device)
        owners, _ = r.owners(rows, per_req, meta.page_table.stride(0))
        r.plan(ratio, _bounds(meta), owners, rows)
        done[ratio] = (rows, per_req)
    return done


def _ready(backend, meta, ratio, rows):
    """(runner, owners, kmeta) when this step planned ``ratio`` for ``rows`` rows, else None."""
    planned = getattr(backend.forward_metadata, "k2i_planned", {}).get(ratio)
    if planned is None or planned[0] != rows:
        return None
    r = _runner(backend, meta.page_table.device)
    owners, kmeta = r.owners(rows, planned[1], meta.page_table.stride(0))
    return r, owners, kmeta


def full_logits(backend, layer, q_fp4, q_scale, weights, meta, ratio, sg_logits=None):
    """K2i's logits of a FULL layer ([rows, >= max bound] f32, valid below each row's compressed length), or None
    (this step was not planned for K2i: take SGLang's)."""
    rows = q_fp4.shape[0]
    got = _ready(backend, meta, ratio, rows) if fits(meta, rows) else None
    if got is None:
        return None
    r, owners, kmeta = got
    pool = backend.token_to_kv_pool
    logits = r.score(
        q_fp4=q_fp4.view(torch.uint8), q_scale=q_scale, weights=weights,
        k_payload=pool.get_index_k_fp4_payload_buffer(layer.layer_id).view(torch.uint8),
        k_scale=pool.get_index_k_fp4_scale_buffer(layer.layer_id), page_table=meta.page_table,
        bounds=_bounds(meta), plan=r.plans[ratio], owners=owners, kmeta=kmeta,
    )  # fmt: skip
    if sg_logits is not None and _checking():
        _check_full(logits, sg_logits, meta)
    return logits


def reindex_select(backend, layer, q_fp4, q_scale, weights, meta, ratio, candidates, page_indices, raw_indices,
                   sg_logits=None):  # fmt: skip
    """A candidate consumer's top-k inside ``candidates`` (SGLang ``CandidateBlocks``) from K2i's compact logits,
    written as ``topk_within_candidate_blocks_hip`` writes it. False: not planned for K2i (take SGLang's)."""
    from sglang.kernels.ops.attention.dsv4.candidate_blocks_hip import (
        _map_compact_selection_kernel,
        topk_transform_paged_hip,
    )

    rows = q_fp4.shape[0]
    got = _ready(backend, meta, ratio, rows) if fits(meta, rows) else None
    if (
        got is None
        or candidates.block_size != ip.BLOCK_ROWS
        or candidates.ids.shape[1] != ip.TOPK_BLOCKS
    ):
        return False
    r, owners, kmeta = got
    pool = backend.token_to_kv_pool
    bounds = _bounds(meta)
    compact, lens, ids = r.score_reindex(
        q_fp4=q_fp4.view(torch.uint8), q_scale=q_scale, weights=weights,
        k_payload=pool.get_index_k_fp4_payload_buffer(layer.layer_id).view(torch.uint8),
        k_scale=pool.get_index_k_fp4_scale_buffer(layer.layer_id), page_table=meta.page_table, bounds=bounds,
        owners=owners, kmeta=kmeta, cand_ids=candidates.ids.contiguous(),
    )  # fmt: skip
    from .runner import COMPACT

    pos = r.compact_pos[:rows]
    topk_transform_paged_hip(compact, lens, r.compact_table[:rows], pos, COMPACT, None)
    topk = page_indices.shape[1]
    assert topk == ip.TOPK and raw_indices.stride(0) == page_indices.stride(0)
    _map_compact_selection_kernel[(rows,)](
        pos, ids, bounds, meta.page_table, page_indices, raw_indices, ids.stride(0), meta.page_table.stride(0),
        page_indices.stride(0), meta.page_table.shape[1], BLOCK_SIZE=ip.BLOCK_ROWS, PAGE_SIZE=meta.compressed_page_size,
        TOPK=topk, BLOCK=topk, WRITE_RAW=True, SORT=False, PAD_KEY=torch.iinfo(torch.int32).max,
    )  # fmt: skip
    if sg_logits is not None and _checking():
        from sglang.kernels.ops.attention.dsv4.candidate_blocks_hip import (
            topk_within_candidate_blocks_hip,
        )

        want = [torch.empty_like(raw_indices[:rows]) for _ in range(2)]
        topk_within_candidate_blocks_hip(sg_logits(), bounds, candidates, page_table=meta.page_table,
                                         page_size=meta.compressed_page_size, page_indices=want[0],
                                         raw_indices=want[1])  # fmt: skip
        _STATS["reindex"] += _jaccard(raw_indices[:rows], want[1], bounds)
        _report()
    return True
