"""DeepSeek-V4.1 attention pieces of the TOP program, shared by serving (every forward mode) and the trainer.

Serving's V4.1 attention picks kernels by forward mode and row count (prefill scores the indexer in torch chunks,
decode with the paged FP4 kernel; small-M GEMMs for wk and weights_proj; CUTLASS/cuBLAS FP32 projections for the
compressor; a gemv for wo_a at one token). Under TOP every row of every mode runs:

- BF16 projections (compressor wkv/wgate with FP32 output, indexer wk and weights_proj, grouped wo_a) on the
  batch-invariant persistent matmul;
- the compressor's torch pair pooling, FP32 RMSNorm and fused RoPE + FP4 quantization (the backend's torch path);
- the indexer as ``select``: serving's paged FP4 index logits for each query row against the request's compressed
  positions, the candidate source's block maxima ranked by a deterministic top-k (ties to the smaller block), a
  candidate mask for later index layers, and a deterministic top-k in ascending position order (ties to the smaller
  position); masked (-inf) picks are dropped;
- the sparse core of ``dsv4_attention`` (selected FP8 rows dequantized, flash_mla_sparse_fwd with the sink).
"""

import torch

_NEG_INF = float("-inf")


def project_fp32(x, weight):
    """x [T, K] BF16 @ weight [N, K]^T -> [T, N] FP32 on the batch-invariant matmul (the compressor projection)."""
    from sglang.srt.batch_invariant_ops.batch_invariant_ops import _mm_dtype_compat

    return _mm_dtype_compat(x.contiguous(), weight.t(), torch.float32)


def project_bf16(x, weight):
    """x [T, K] BF16 @ weight [N, K]^T -> [T, N] BF16 on the batch-invariant matmul."""
    from sglang.srt.batch_invariant_ops import matmul_persistent

    return matmul_persistent(x.contiguous(), weight.t())


def compressor_project(compressor, x):
    """(kv, score) FP32 projections of a V4.1 compressor; score is None at ratio 1."""
    if compressor.use_fused_gate:
        raise NotImplementedError(
            "[top] V4.1 compressor runs the separate wkv/wgate projections"
        )
    if compressor.compress_ratio == 1:
        return project_bf16(x, compressor.wkv.weight), None
    return project_fp32(x, compressor.wkv.weight), project_fp32(
        x, compressor.wgate.weight
    )


def wo_a(o, weight):
    """o [T, G, D] @ wo_a [G, R, D] -> [T, G, R] BF16 on the batch-invariant bmm."""
    from sglang.srt.batch_invariant_ops.batch_invariant_ops import bmm_batch_invariant

    out = torch.empty(
        (o.shape[0], weight.shape[0], weight.shape[1]), dtype=o.dtype, device=o.device
    )
    if o.shape[0]:
        out.transpose(0, 1).copy_(
            bmm_batch_invariant(o.transpose(0, 1).contiguous(), weight.transpose(1, 2))
        )
    return out


def _block_candidates(logits, lens, topk_blocks, block):
    """The candidate source's mask [rows, width]: the top ``topk_blocks`` blocks of ``block`` positions per row by
    their best logit (unreachable positions -inf), the block holding the row's newest position pinned in."""
    from sglang.srt.true_on_policy.dsa.ordered_topk import ordered_dsa_topk

    rows, width = logits.shape
    cols = torch.arange(width, device=logits.device)
    reach = cols[None, :] < lens[:, None]
    scores = torch.where(reach, logits, _NEG_INF)
    num_blocks = -(-width // block)
    scores = torch.nn.functional.pad(
        scores, (0, num_blocks * block - width), value=_NEG_INF
    )
    blocks = scores.view(rows, num_blocks, block).amax(dim=-1)
    last = ((lens - 1).clamp_min(0) // block).long()
    blocks.scatter_(1, last[:, None], float("inf"))
    blocks = torch.where(lens[:, None] > 0, blocks, _NEG_INF)
    picked = ordered_dsa_topk(
        blocks,
        torch.full((rows,), num_blocks, dtype=torch.int32, device=logits.device),
        min(topk_blocks, num_blocks),
    )
    valid = picked >= 0
    best = blocks.gather(1, picked.clamp_min(0).long())
    keep = torch.zeros((rows, num_blocks), dtype=torch.bool, device=logits.device)
    keep.scatter_(1, picked.clamp_min(0).long(), valid & (best > _NEG_INF))
    return keep.repeat_interleave(block, dim=1)[:, :width].contiguous()


def select(
    *,
    indexer,
    x,
    q_lora,
    freqs_cis,
    positions,
    req_rows,
    req_to_token,
    index_k_table,
    ratio,
    width,
    candidates=None,
    out_page_indices,
    out_raw_indices=None,
):
    """One index layer's top-k for every query row, written as request-local positions and index-K slots.

    ``width`` bounds every row's visible compressed positions (graph-stable under decode). Returns the candidate
    mask when ``indexer`` is the candidate source, else None.
    """
    from sglang.kernels.ops.attention.dsv4.fp4_indexer import fp4_index_logits_paged
    from sglang.srt.true_on_policy.dsa.ordered_topk import ordered_dsa_topk

    rows = positions.shape[0]
    topk = out_page_indices.shape[1]
    out_page_indices[:rows].fill_(-1)
    if out_raw_indices is not None:
        out_raw_indices[:rows].fill_(-1)
    if rows == 0 or width == 0:
        return None
    q = indexer.queries(q_lora, freqs_cis[positions])
    weights = indexer.head_weights(x)
    lens = ((positions + 1) // ratio).to(torch.int32)
    logits = fp4_index_logits_paged(
        q,
        weights,
        req_rows,
        req_to_token,
        lens,
        index_k_table,
        index_k_table.shape[1] // 68,
        width,
        ratio,
        candidates if indexer.uses_candidates else None,
    )
    published = None
    if indexer.is_candidate_source:
        published = _block_candidates(
            logits, lens, indexer.candidate_topk_blocks, indexer.candidate_block_size
        )
    idx = ordered_dsa_topk(logits, lens, min(indexer.index_topk, topk))
    picked = logits.gather(1, idx.clamp_min(0).long())
    idx = torch.where((idx >= 0) & (picked > _NEG_INF), idx, -1)
    slots = (
        req_to_token[req_rows.long()[:, None], (idx.clamp_min(0) * ratio).long()].long()
        // ratio
    )
    k = idx.shape[1]
    out_page_indices[:rows, :k] = torch.where(idx >= 0, slots, -1).to(torch.int32)
    if out_raw_indices is not None:
        out_raw_indices[:rows, :k] = idx
    return published
