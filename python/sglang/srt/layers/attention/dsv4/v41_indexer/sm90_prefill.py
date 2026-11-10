"""Rank-local FP8 prefill scoring on Hopper."""

import torch

from sglang.kernels.ops.attention.dsv4.fp4_indexer import INDEX_K_SLOT_BYTES
from sglang.kernels.ops.attention.dsv4.sm90_deepgemm_prefill import (
    cast_index_queries_to_fp8,
    deepgemm_fp8_index_logits_prefill,
    gather_fp4_index_keys_as_fp8,
)
from sglang.srt.environ import envs

from .scoring import ChunkScores, RequestScores, _prefill_score_budget_bytes

_SCORE_BUDGET_BYTES = 64 << 20


def prefill_requests_fp8(inputs, pool):
    if envs.SGLANG_DSV41_SM90_PREFILL_INDEXER.get() != "deepgemm":
        raise ValueError("SGLANG_DSV41_SM90_PREFILL_INDEXER must be 'deepgemm'")
    counts, seq_lens = inputs.rows_per_request, inputs.seq_lens_cpu
    assert len(counts) == len(seq_lens)
    real_rows = sum(counts)
    assert real_rows <= inputs.positions.numel()
    inputs.reset_outputs()
    if not real_rows:
        return
    positions = inputs.positions[:real_rows]
    q = cast_index_queries_to_fp8(
        inputs.indexer.queries(inputs.q_lora[:real_rows], inputs.freqs_cis[positions])
    )
    weights = inputs.indexer.head_weights(inputs.x[:real_rows])
    lens = ((positions + 1) // inputs.compress_ratio).to(torch.int32)
    table = pool.get_index_k_with_scale_buffer(inputs.layer_id)
    page_size = table.shape[1] // INDEX_K_SLOT_BYTES
    row_end = key_end = 0
    for count, seq_len in zip(counts, seq_lens):
        width = seq_len // inputs.compress_ratio
        row_start, row_end = row_end, row_end + count
        key_start, key_end = key_end, key_end + width
        if not count or not width:
            continue
        slots = inputs.k_slots[key_start:key_end]
        request = RequestScores(
            lc=width,
            k=min(inputs.indexer.index_topk, width),
            columns=torch.arange(width, device=q.device),
            tok=torch.arange(row_start, row_end, device=q.device),
            lens=lens[row_start:row_end],
            slots=slots,
        )
        keys = gather_fp4_index_keys_as_fp8(slots, table, page_size)
        yield (
            request,
            _score_chunks(
                request, q[row_start:row_end], weights[row_start:row_end], keys
            ),
        )


def _score_chunks(request, q, weights, keys):
    stride = ((request.lc + 511) // 256) * 256
    alignment = 128 // q.shape[1]
    budget_bytes = _prefill_score_budget_bytes(_SCORE_BUDGET_BYTES)
    step = max(alignment, budget_bytes // (stride * 4 * alignment) * alignment)
    scales = torch.ones(request.lc, device=keys.device, dtype=torch.float32)
    for start in range(0, q.shape[0], step):
        rows = slice(start, start + step)
        lens = request.lens[rows]
        yield ChunkScores(
            rows=rows,
            tok=request.tok[rows],
            lens=lens,
            scores=deepgemm_fp8_index_logits_prefill(
                q[rows],
                weights[rows],
                keys,
                lens,
                key_scales=scales,
            ),
            use_ragged_topk=True,
        )
