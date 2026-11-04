from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.compilation.compilation_config import register_split_op
from sglang.srt.layers.cp.interleave import is_interleave_extend
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    get_tc_piecewise_forward_context,
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.utils import is_cuda, is_gfx1250_supported, is_hip
from sglang.srt.utils.custom_op import register_custom_op

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_gfx1250_supported = is_gfx1250_supported()

GRAPH_WEIGHTS_PROJ_LORA_ERROR = (
    "DSA indexer weights_proj LoRA is incompatible with "
    "piecewise/breakable CUDA graph; remove the explicit "
    "prefill cuda-graph backend override or drop "
    "indexer.weights_proj from the LoRA target modules."
)


def _is_in_piecewise_or_breakable_cuda_graph() -> bool:
    return is_in_tc_piecewise_cuda_graph() or is_in_breakable_cuda_graph()


@register_custom_op(mutates_args=["topk_result"])
@register_split_op()
def pcg_dsa_indexer_prefill_split(
    layer_id: int,
    x: torch.Tensor,
    q_lora: torch.Tensor,
    positions: torch.Tensor,
    topk_result: torch.Tensor,
    x_s: Optional[torch.Tensor] = None,
    x_bf16: Optional[torch.Tensor] = None,
) -> None:
    # In-graph prefill indexer (also interleave CP under BCG): runs the indexer
    # (q/k proj, head gate, k-cache store, topk) as one eager split op. PCG calls
    # this as a split op; BCG uses the explicit eager wrapper below.
    #
    # Output contract (differs from the eager `forward` path): a split op returns
    # None, so results are delivered only by mutating `topk_result` in place. The
    # call site pre-allocates it at a static, padded shape and a downstream
    # captured graph reads it at a fixed address; eager code instead allocates
    # and returns a fresh, naturally-sized tensor each call.
    #
    # A quantized activation tuple (x, x_s[, x_bf16]) cannot cross the custom-op
    # schema, so the call site passes its parts separately.
    assert _is_cuda or _is_hip, (
        "Internal error: DSA graph dispatch is only supported on CUDA and ROCm"
    )
    if _is_hip and not _is_gfx1250_supported:
        from sglang.kernels.ops.quantization.act_quant_tilelang import act_quant
    else:
        from sglang.kernels.ops.quantization.act_quant_triton import act_quant
    if x_s is not None:
        x = (x, x_s) if x_bf16 is None else (x, x_s, x_bf16)

    forward_context = get_tc_piecewise_forward_context()
    forward_batch = forward_context.forward_batch
    indexer = forward_context.dsa_indexers[layer_id]
    metadata = get_attn_backend().get_indexer_metadata(layer_id, forward_batch)

    extend_num_tokens = forward_batch.extend_num_tokens
    cp = is_interleave_extend(forward_batch)
    query_tokens = extend_num_tokens
    logical_tokens = extend_num_tokens
    if cp:
        from sglang.srt.layers.cp.bcg.interleave_bcg import cp_interleave_indexer_rows

        query_tokens, logical_tokens = cp_interleave_indexer_rows(forward_batch)
        parts = x if isinstance(x, tuple) else (x,)
        if not 0 <= logical_tokens <= query_tokens or any(
            query_tokens > part.shape[0] for part in parts
        ):
            raise ValueError("Invalid CP indexer live rows")
        if query_tokens > q_lora.shape[0] or query_tokens > positions.shape[0]:
            raise ValueError("CP indexer input rows do not cover the live shard")
        x = (
            tuple(part[:query_tokens] for part in parts)
            if isinstance(x, tuple)
            else x[:query_tokens]
        )
        q_lora, positions = q_lora[:query_tokens], positions[:query_tokens]
        extend_num_tokens = forward_batch.attn_cp_metadata.total_seq_lens
        if extend_num_tokens > forward_batch.out_cache_loc.shape[0]:
            raise ValueError(
                "CP indexer cache locations do not cover live global tokens"
            )
        topk_result.fill_(-1)
    # Empty buffer encodes return_indices=False for graph dispatch.
    return_indices = topk_result.numel() != 0
    k_only = not return_indices or (
        indexer._should_skip_logits_computation(forward_batch)
        and not indexer.dsa_enable_prefill_cp
    )
    if k_only and not cp:
        indexer._forward_cuda_k_only(
            x,
            positions,
            forward_batch,
            layer_id,
            act_quant,
            metadata=metadata,
            return_indices=return_indices,
            num_tokens=extend_num_tokens,
            topk_result=topk_result,
        )
        return

    # Same fused Q/K writer as the eager ROCm path, sliced to the unpadded count.
    if not cp and indexer._aiter_fused_fp8_active(forward_batch):
        q_fp8, weights = indexer._aiter_fused_fp8_prepare_and_store(
            x,
            q_lora,
            positions,
            forward_batch,
            layer_id,
            num_tokens=extend_num_tokens,
        )
        indexer._get_topk_ragged(
            False,
            forward_batch,
            layer_id,
            q_fp8,
            weights,
            metadata,
            topk_result,
        )
        return

    # Fused path stores K (no-Hadamard) and computes q_fp8 + head gate in the
    # fused kernels, sliced to the unpadded count. Single stream: the split op is
    # captured, so the dual-stream overlap is disabled.
    if not cp and indexer.use_dsa_indexer_fusion:
        q_fp8, weights = indexer._fused_q_prepare_and_store(
            x,
            q_lora,
            positions,
            forward_batch,
            layer_id,
            act_quant,
            num_tokens=extend_num_tokens,
            enable_dual_stream=False,
        )
        indexer._get_topk_ragged(
            False,
            forward_batch,
            layer_id,
            q_fp8,
            weights,
            metadata,
            topk_result,
        )
        return

    query, key, weights_raw = indexer._get_q_k_bf16(
        q_lora,
        x,
        positions,
        enable_dual_stream=False,
        forward_batch=forward_batch,
    )
    if (
        _is_cuda
        and indexer.bf16_head_gate
        and indexer.skip_rope
        and indexer.head_dim == indexer.block_size == 128
    ):
        from sglang.kernels.ops.attention.dsa.head_gate import indexer_q_quant_head_gate

        gate = indexer._weights_proj_bf16_in_fp32_out(x)
        q_fp8, weights = indexer_q_quant_head_gate(
            query, gate, indexer.softmax_scale, indexer.scale_fmt
        )
    else:
        q_fp8, q_scale = act_quant(query, indexer.block_size, indexer.scale_fmt)
        weights = (
            indexer._scale_head_gates(weights_raw, q_scale)
            if indexer.use_dsa_indexer_fusion
            else indexer._get_logits_head_gate(x, q_scale)
        )
    # Store K cache + ragged top-k, sliced to the unpadded count and writing into
    # the static padded topk_result buffer (the graph contract). Mirrors the eager
    # path's store + _get_topk_ragged.
    indexer._store_index_k_cache(
        forward_batch=forward_batch,
        layer_id=layer_id,
        key=key[:extend_num_tokens],
        act_quant=act_quant,
        out_cache_loc=forward_batch.out_cache_loc[:extend_num_tokens],
    )
    if not return_indices:
        return
    indexer._get_topk_ragged(
        False,
        forward_batch,
        layer_id,
        q_fp8[:query_tokens],
        weights,
        metadata,
        topk_result,
    )
    if cp:
        topk_result[logical_tokens:].fill_(-1)


def _dsa_indexer_prefill_capture_stub(
    layer_id, x, q_lora, positions, topk_result, **kwargs
):
    topk_result.fill_(-1)


bcg_dsa_indexer_prefill_split = eager_on_graph(
    on_capture=_dsa_indexer_prefill_capture_stub
)(pcg_dsa_indexer_prefill_split)
