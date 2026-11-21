"""DeepSeek-V4.1 signed index keys and candidate-provider integration on 910B."""

import logging
from functools import lru_cache

import torch

from sglang.srt.hardware_backend.npu.dsv4.dsv41_int8 import (
    decode_int8,
    encode_int8,
    metadata_bytes,
    physical_page_map,
)

logger = logging.getLogger(__name__)

INDEX_HEAD_DIM = 128
INDEX_METADATA_BYTES = metadata_bytes(INDEX_HEAD_DIM, INDEX_HEAD_DIM)


def quantize_rows(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the one-scale-per-head inputs consumed by the CANN INT8 indexer."""
    if rows.shape[-1] != INDEX_HEAD_DIM:
        raise ValueError("The DSv4.1 indexer expects 128 values per head")
    if not rows.numel():
        return rows.new_empty(rows.shape, dtype=torch.int8), rows.new_empty(
            rows.shape[:-1], dtype=torch.float16
        )
    import torch_npu

    values, scales = torch_npu.npu_dynamic_quant(
        rows.contiguous().reshape(-1, INDEX_HEAD_DIM), dst_type=torch.int8
    )
    return values.reshape_as(rows), scales.half().reshape(rows.shape[:-1])


def write_index_rows(key_buffer, metadata_buffer, locations, keys):
    values, metadata = encode_int8(keys, INDEX_HEAD_DIM)
    slots = locations.reshape(-1).long()
    valid = slots > 0
    # Padding writers all put zeros in the reserved page.
    key_buffer.view(-1, INDEX_HEAD_DIM).index_copy_(
        0,
        slots.clamp_min(0),
        torch.where(valid[:, None], values.reshape(-1, INDEX_HEAD_DIM), 0),
    )
    metadata_buffer.view(-1, INDEX_METADATA_BYTES).index_copy_(
        0,
        slots.clamp_min(0),
        torch.where(valid[:, None], metadata.reshape(-1, INDEX_METADATA_BYTES), 0),
    )


def prepare_indexer_operands(
    q,
    keys,
    weights,
    metadata,
    full_lens,
    host_lens,
    kwargs,
    *,
    capture,
    capacity,
    filtered,
):
    """Reconstruct the active pages. Only the coarse pruning pass quantizes Q."""
    if q.shape[0] == 0:
        q8, q_scale = quantize_rows(q)
        return q8, keys, weights.half(), q_scale, None
    physical, table = physical_page_map(
        kwargs["block_table"], keys.shape[1], kwargs["seqused_k"], fixed_shape=capture
    )
    active = decode_int8(
        keys.index_select(0, physical),
        metadata.index_select(0, physical),
        INDEX_HEAD_DIM,
    )
    kwargs["block_table"] = table
    lengths = host_lens if host_lens is not None else full_lens
    if (
        capacity is not None
        and not capture
        and not filtered
        and int(lengths.max()) // kwargs["cmp_ratio"] > capacity
    ):
        # The native ordinary operator accepts one scale per 128-value key.
        # The persistent format retains more precision for the subsequent rerank.
        q8, q_scale = quantize_rows(q)
        k8, k_scale = quantize_rows(active)
        return q8, k8, weights.half(), q_scale, k_scale.unsqueeze(-1)
    return q, active, weights.float(), None, None


@lru_cache(maxsize=1)
def _registered_ordinary_ops():
    # A 910B deployment can already have the ordinary integer operator.
    # Its paired CANN extension owns the dispatcher namespace for this worker.
    for namespace in (torch.ops.cann_ops_transformer, torch.ops.npu):
        operator = getattr(namespace, "quant_lightning_indexer", None)
        metadata = getattr(namespace, "quant_lightning_indexer_metadata", None)
        if operator is not None and metadata is not None:
            args = {arg.name for arg in operator.default._schema.arguments}
            if {"query_dequant_scale", "metadata", "quant_mode", "cmp_ratio"} <= args:
                return metadata, operator
    return None


def _ordinary_indices(ops, q, k, weights, q_scale, k_scale, topk, kw):
    # Give each query its exact visible C1/C2 prefix. This is also valid for
    # chunked prefill ending on the unmatched half of a ratio-two group.
    cu = kw["cu_seqlens_q"]
    row = torch.arange(q.shape[0], device=q.device, dtype=torch.int32)
    owner = torch.searchsorted(cu[1:].contiguous(), row, right=True).long()
    used = kw["seqused_k"].index_select(0, owner)
    residual = kw["cmp_residual_k"]
    tail = 0 if residual is None else residual.index_select(0, owner)
    ratio = kw["cmp_ratio"]
    visible = (
        (ratio * used + tail - cu[1:].index_select(0, owner) + row + 1).clamp_min(0)
        // ratio
    ).minimum(used)
    metadata, calculate = ops
    args = dict(
        cu_seqlens_q=torch.arange(q.shape[0] + 1, dtype=torch.int32, device=q.device),
        seqused_k=visible.clamp_min(1).int(),
        topk=topk,
        quant_mode=2,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=1,
    )
    spec = metadata(
        num_heads_q=q.shape[1], num_heads_k=1, head_dim=INDEX_HEAD_DIM, **args
    )
    result, _ = calculate(
        query=q,
        key=k,
        weights=weights,
        query_dequant_scale=q_scale,
        key_dequant_scale=k_scale.reshape(k.shape[:-1]),
        block_table=kw["block_table"].index_select(0, owner),
        metadata=spec,
        return_value=0,
        **args,
    )
    return torch.where(visible[:, None] > 0, result.reshape(q.shape[0], topk), -1)


def index_topk(
    q,
    k,
    weights,
    q_scale,
    k_scale,
    topk,
    *,
    candidate_topk_blocks=0,
    candidate_block_size=8,
    candidate_topk_index=None,
    **kwargs,
):
    """Use the CANN 9.1 ordinary INT8 op and sgl-kernel-npu's candidate modes."""
    empty = q.new_empty((q.shape[0], 1, 0), dtype=torch.int32)
    if not q.shape[0]:
        return q.new_empty((0, topk), dtype=torch.int32), q.new_empty(
            (0, 1, candidate_topk_blocks), dtype=torch.int32
        )
    if candidate_topk_blocks and candidate_topk_index is not None:
        raise ValueError("A caller must supply or consume candidate blocks separately")
    if q.dtype != torch.int8 or k.dtype != torch.int8:
        raise ValueError("The integer shortlist expects signed INT8 Q and K")
    ops = _registered_ordinary_ops() if candidate_topk_index is None else None
    native = (
        None
        if ops is None
        else _ordinary_indices(ops, q, k, weights, q_scale, k_scale, topk, kwargs)
    )
    if native is not None and not candidate_topk_blocks:
        logger.info_once(
            "DSv4.1 910B signed INT8 indexer used the ordinary CANN operator (ratio %d)",
            kwargs["cmp_ratio"],
        )
        return native, empty

    from sgl_kernel_npu.operators.indexer.quant_lightning_indexer import (
        quant_lightning_indexer,
    )

    if candidate_topk_blocks:
        mode = dict(candidate_mode=1, candidate_topk_blocks=candidate_topk_blocks)
    elif candidate_topk_index is not None:
        mode = dict(candidate_mode=2, candidate_topk_index=candidate_topk_index)
    else:
        mode = dict(candidate_mode=3)
    # CANN 9.1 provides the ordinary integer op. The two-level public eager
    # provider accepts INT8 Q/K and implements the A3 candidate contract on A2.
    result = quant_lightning_indexer(
        q,
        k,
        weights,
        q_scale,
        k_scale.reshape(k.shape[:-1]),
        topk,
        2,
        impl="torch_native",
        candidate_block_size=candidate_block_size,
        **mode,
        **kwargs,
    )
    if len(result) != 3:
        raise RuntimeError(
            "DSv4.1 candidate indexing requires sgl-kernel-npu's three-result INT8 candidate API (PR #1045)"
        )
    logger.info_once(
        "DSv4.1 910B signed INT8 indexer used sgl-kernel-npu torch_native mode %d (ratio %d)",
        mode["candidate_mode"],
        kwargs["cmp_ratio"],
    )
    indices = result[0].reshape(q.shape[0], topk) if native is None else native
    return indices, result[2] if candidate_topk_blocks else empty
