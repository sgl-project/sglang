# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1's indexer layers between K1 and K2 on monokernel steps (``SGLANG_TEST_MONO_DSV41_K2A=1``, the
default): SGLang's compressor, then SGLang's decode indexer (``low_ratio_index_topk_hip_decode``) with the K2i
scorer in place of ``aiter_fp4_paged_mqa_logits``; the selection stays SGLang's (``topk_transform_paged_hip``, the
candidate blocks), except that a candidate consumer's top-k runs on K2i's compact candidate logits directly (no
gather). A ratio whose rows do not fit (more than MAX_ROWS, another plane page) is scored by aiter as SGLang does."""

import sys

import torch

from sglang.kernels.ops.mono.deepseek_v41_gfx950.indexer import index_plan as ip

_MODE = []
_RUNNERS: dict = {}


def enabled() -> bool:
    if not _MODE:
        from .spec import k2a_mode

        _MODE.append(k2a_mode())
    return _MODE[0] == 1


_STATS = {"full": [], "reindex": [], "logit_err": [], "calls": 0}


# the SGLang names this module reads (pinned by test_mono_contract)
SGLANG_READS = (
    "sglang.srt.environ.envs",
    "sglang.srt.layers.attention.dsv4.low_ratio_backend_hip._indexer_inputs",
    "sglang.srt.layers.attention.dsv4.low_ratio_backend_hip.low_ratio_decode_rows_are_identity",
    "sglang.srt.layers.attention.dsv4.low_ratio_backend_hip.low_ratio_decode_rows_fit_candidate_span",
)


def _checking() -> bool:
    """``SGLANG_TEST_MONO_DSV41_K2A_CHECK=1`` (eager steps only): SGLang's path runs too and the selections are
    compared (per-row Jaccard, logits' max relative error), logged every 64 layers as ``[K2A_CHECK]``."""
    from sglang.srt.environ import envs

    return (
        envs.SGLANG_TEST_MONO_DSV41_K2A_CHECK.get()
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
    from sglang.kernels.ops.mono.deepseek_v41_gfx950.indexer.runner import K2iRunner

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
    from sglang.kernels.ops.mono.deepseek_v41_gfx950.indexer.runner import (
        MAX_ROWS,
        PLANE_ROWS,
    )

    return (
        0 < rows <= MAX_ROWS
        and meta.compressed_page_size == PLANE_ROWS
        and meta.page_table.dtype == torch.int32
        and meta.page_table.shape[0] == rows
        and meta.page_table.stride(1) == 1
    )


def skips_schedule(forward_batch, ratio, meta) -> bool:
    """SGLang's aiter decode schedule for ``ratio`` goes unused: K2i scores it this step (step_plans' test). The
    K2A check still runs SGLang's logits, so it keeps the schedule."""
    if _checking():
        return False
    rows = meta.page_table.shape[0]
    return fits(meta, rows) and rows % forward_batch.batch_size == 0


def step_plans(backend, forward_batch) -> dict:
    """Once a step (in the graph when captured): each scored ratio's FULL walk plan. -> ratio -> (rows, rows a
    request) of the ratios it planned."""
    from sglang.srt.layers.attention.dsv4.low_ratio_backend_hip import (
        low_ratio_decode_rows_are_identity,
    )

    metadata = backend.forward_metadata
    done = {}
    for ratio, meta in metadata.low_ratio_indexer_metadata_by_ratio().items():
        if low_ratio_decode_rows_are_identity(backend, forward_batch, ratio):
            continue  # identity rows are not scored
        rows = meta.page_table.shape[0]
        if not fits(meta, rows) or rows % forward_batch.batch_size:
            continue
        per_req = rows // forward_batch.batch_size
        r = _runner(backend, meta.page_table.device)
        owners, _ = r.owners(rows, per_req, meta.page_table.stride(0))
        r.plan(ratio, _bounds(meta), owners, rows)
        done[ratio] = (rows, per_req)
    return done


def _ready(backend, meta, ratio, rows, plans):
    """(runner, owners, kmeta) when this step planned ``ratio`` for ``rows`` rows, else None."""
    planned = plans.get(ratio)
    if planned is None or planned[0] != rows:
        return None
    r = _runner(backend, meta.page_table.device)
    owners, kmeta = r.owners(rows, planned[1], meta.page_table.stride(0))
    return r, owners, kmeta


def full_logits(
    backend, layer, q_fp4, q_scale, weights, meta, ratio, plans, sg_logits=None
):
    """K2i's logits of a FULL layer ([rows, >= max bound] f32, valid below each row's compressed length), or None
    (this step was not planned for K2i: take SGLang's)."""
    rows = q_fp4.shape[0]
    got = _ready(backend, meta, ratio, rows, plans) if fits(meta, rows) else None
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


def reindex_select(backend, layer, q_fp4, q_scale, weights, meta, ratio, plans, candidates, page_indices,
                   raw_indices, sg_logits=None):  # fmt: skip
    """A candidate consumer's top-k inside ``candidates`` (SGLang ``CandidateBlocks``) from K2i's compact logits,
    written as ``topk_within_candidate_blocks_hip`` writes it. False: not planned for K2i (take SGLang's)."""
    from sglang.kernels.ops.attention.dsv4.candidate_blocks_hip import (
        _map_compact_selection_kernel,
        topk_transform_paged_hip,
    )

    rows = q_fp4.shape[0]
    got = _ready(backend, meta, ratio, rows, plans) if fits(meta, rows) else None
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
    from sglang.kernels.ops.mono.deepseek_v41_gfx950.indexer.runner import COMPACT

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


def _aiter_logits(backend, layer, q_fp4, q_scale, weights, meta, ratio):
    from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
        LOW_RATIO_PAGE_TABLE_BUCKET,
        aiter_fp4_paged_mqa_logits,
    )

    pool = backend.token_to_kv_pool
    return aiter_fp4_paged_mqa_logits(
        q_fp4=q_fp4,
        q_scale=q_scale,
        k_payload=pool.get_index_k_fp4_payload_buffer(layer.layer_id),
        k_scale=pool.get_index_k_fp4_scale_buffer(layer.layer_id),
        weights=weights,
        page_table=meta.page_table,
        c4_seq_lens=meta.compressed_seq_lens,
        weight_scale=1.0,
        page_table_bucket=LOW_RATIO_PAGE_TABLE_BUCKET,
        is_decode=True,
        decode_workspace=backend.forward_metadata.fp4_low_ratio_decode_workspaces.get(
            ratio
        ),
    )


def index_topk_decode(backend, layer, x, q_lora, pos, forward_batch, plans) -> None:
    """SGLang's ``low_ratio_index_topk_hip_decode`` with K2i scoring the ratios ``plans`` covers: the page indices of
    the layer's ratio rewritten in place, as SGLang's writes them."""
    from sglang.kernels.ops.attention.dsv4 import topk_transform_paged
    from sglang.kernels.ops.attention.dsv4.candidate_blocks_hip import (
        CandidateBlocks,
        select_candidate_blocks_hip,
        topk_transform_paged_hip,
        topk_within_candidate_blocks_hip,
    )
    from sglang.srt.layers.attention.dsv4.low_ratio_backend_hip import (
        _indexer_inputs,
        low_ratio_decode_rows_are_identity,
        low_ratio_decode_rows_fit_candidate_span,
    )

    metadata = backend.forward_metadata
    core = metadata.core_metadata
    ratio = layer.compress_ratio
    indexer = layer.indexer
    meta = metadata.low_ratio_indexer_metadata(ratio)
    assert meta is not None, f"no decode indexer metadata for {ratio = }"
    core.drop_folded_sparse_indices(ratio)
    backend._drop_opus_topk_csr(ratio)
    page_indices = core.sparse_page_indices(ratio)
    raw_indices = core.sparse_raw_indices(ratio)
    if low_ratio_decode_rows_are_identity(backend, forward_batch, ratio):
        # every row takes the transform's sequential branch (len <= topk), which reads no scores
        scores = torch.empty(
            (page_indices.shape[0], 1), dtype=torch.float32, device=pos.device
        )
        topk_transform_paged(
            scores,
            meta.compressed_seq_lens,
            meta.page_table,
            page_indices,
            meta.compressed_page_size,
            raw_indices,
        )
        return
    two_level = indexer.is_candidate_source or indexer.uses_candidates
    if two_level and low_ratio_decode_rows_fit_candidate_span(backend, forward_batch):
        # every block is a candidate: the source publishes nothing, every layer runs the plain top-k
        if indexer.is_candidate_source:
            backend.candidate_masks = None
        two_level = False
    q_fp4, q_scale, weights = _indexer_inputs(layer, x, q_lora, pos)

    def sg_logits():
        return _aiter_logits(backend, layer, q_fp4, q_scale, weights, meta, ratio)

    if two_level and indexer.uses_candidates:
        if reindex_select(backend, layer, q_fp4, q_scale, weights, meta, ratio, plans, backend.candidate_masks,
                          page_indices, raw_indices, sg_logits):  # fmt: skip
            return
        logits = None
    else:
        logits = full_logits(
            backend, layer, q_fp4, q_scale, weights, meta, ratio, plans, sg_logits
        )
    if logits is None:
        logits = sg_logits()
    if two_level and indexer.uses_candidates:
        candidates = backend.candidate_masks
        assert (
            isinstance(candidates, CandidateBlocks)
            and candidates.ids.shape[0] == logits.shape[0]
            and candidates.block_size == indexer.candidate_block_size
        ), "candidate blocks missing for decode"
        topk_within_candidate_blocks_hip(
            logits, meta.compressed_seq_lens, candidates, page_table=meta.page_table,
            page_size=meta.compressed_page_size, page_indices=page_indices, raw_indices=raw_indices,
        )  # fmt: skip
        return
    if two_level and indexer.is_candidate_source:
        backend.candidate_masks = select_candidate_blocks_hip(
            logits, meta.compressed_seq_lens, topk_blocks=indexer.candidate_topk_blocks,
            block_size=indexer.candidate_block_size,
        )  # fmt: skip
    topk_transform_paged_hip(
        logits,
        meta.compressed_seq_lens,
        meta.page_table,
        page_indices,
        meta.compressed_page_size,
        raw_indices,
    )
