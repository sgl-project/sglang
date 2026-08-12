from __future__ import annotations

import torch

from sglang.srt.compilation.compilation_config import register_split_op
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
from sglang.srt.utils import is_cuda
from sglang.srt.utils.custom_op import register_custom_op

_is_cuda = is_cuda()

GRAPH_WEIGHTS_PROJ_LORA_ERROR = (
    "DSA indexer weights_proj LoRA is incompatible with "
    "piecewise/breakable CUDA graph; remove the explicit "
    "prefill cuda-graph backend override or drop "
    "indexer.weights_proj from the LoRA target modules."
)


def _is_in_piecewise_or_breakable_cuda_graph() -> bool:
    return is_in_tc_piecewise_cuda_graph() or is_in_breakable_cuda_graph()


if _is_cuda:

    def _scale_head_gate_graph_fake_impl(
        weights_raw: torch.Tensor,
        n_heads_inv_sqrt: float,
        softmax_scale: float,
        q_scale: torch.Tensor,
    ) -> torch.Tensor:
        return torch.empty(
            (weights_raw.shape[0], weights_raw.shape[1], q_scale.shape[-1]),
            dtype=torch.float32,
            device=weights_raw.device,
        )

    # In-graph (PCG/BCG) head gate for the fused path: weights_proj is folded
    # into wk_weights_proj, so weights_raw is precomputed and there is no GEMM.
    @register_custom_op(fake_impl=_scale_head_gate_graph_fake_impl)
    def scale_head_gate_graph(
        weights_raw: torch.Tensor,
        n_heads_inv_sqrt: float,
        softmax_scale: float,
        q_scale: torch.Tensor,
    ) -> torch.Tensor:
        weights = weights_raw * n_heads_inv_sqrt
        return weights.unsqueeze(-1) * q_scale * softmax_scale

    def _logits_head_gate_graph_fake_impl(
        x: torch.Tensor,
        weight: torch.Tensor,
        n_heads_inv_sqrt: float,
        softmax_scale: float,
        q_scale: torch.Tensor,
    ) -> torch.Tensor:
        return torch.empty(
            (x.shape[0], weight.shape[0], q_scale.shape[-1]),
            dtype=torch.float32,
            device=x.device,
        )

    # In-graph (PCG/BCG) head gate for the NON-prefill path
    @register_custom_op(fake_impl=_logits_head_gate_graph_fake_impl)
    def logits_head_gate_graph(
        x: torch.Tensor,
        weight: torch.Tensor,
        n_heads_inv_sqrt: float,
        softmax_scale: float,
        q_scale: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.mm(x, weight.t(), out_dtype=torch.float32)
        weights = out * n_heads_inv_sqrt
        weights = weights.unsqueeze(-1) * q_scale * softmax_scale
        return weights


@register_custom_op(mutates_args=["topk_result"])
@register_split_op()
def pcg_dsa_indexer_prefill_split(
    layer_id: int,
    x: torch.Tensor,
    q_lora: torch.Tensor,
    positions: torch.Tensor,
    topk_result: torch.Tensor,
) -> None:
    # Default in-graph indexer path for non-CP prefill: runs the whole indexer
    # (q/k proj, head gate, k-cache store, topk) as one eager split op. PCG calls
    # this as a split op; BCG uses the explicit eager wrapper below.
    #
    # Output contract (differs from the eager `forward` path): a split op returns
    # None, so results are delivered only by mutating `topk_result` in place. The
    # call site pre-allocates it at a static, padded shape and a downstream
    # captured graph reads it at a fixed address; eager code instead allocates
    # and returns a fresh, naturally-sized tensor each call.
    assert _is_cuda, "Internal error: DSA graph dispatch is only supported on CUDA"
    from sglang.kernels.ops.attention.dsa.triton_kernel import act_quant

    forward_context = get_tc_piecewise_forward_context()
    forward_batch = forward_context.forward_batch
    indexer = forward_context.dsa_indexers[layer_id]
    metadata = get_attn_backend().get_indexer_metadata(layer_id, forward_batch)

    extend_num_tokens = forward_batch.extend_num_tokens
    # Empty buffer encodes return_indices=False for graph dispatch.
    return_indices = topk_result.numel() != 0
    k_only = not return_indices or (
        indexer._should_skip_logits_computation(forward_batch)
        and not indexer.dsa_enable_prefill_cp
    )
    if k_only:
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

    # Fused path stores K (no-Hadamard) and computes q_fp8 + head gate in the
    # fused kernels, sliced to the unpadded count. Single stream: the split op is
    # captured, so the dual-stream overlap is disabled.
    if indexer.use_dsa_indexer_fusion:
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

    query, key, _ = indexer._get_q_k_bf16(
        q_lora,
        x,
        positions,
        enable_dual_stream=False,
        forward_batch=forward_batch,
    )
    q_fp8, q_scale = act_quant(query, indexer.block_size, indexer.scale_fmt)
    # Reuse the compiled head-gate util shared with the eager path.
    weights = indexer._get_logits_head_gate(x, q_scale)
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
    indexer._get_topk_ragged(
        False,
        forward_batch,
        layer_id,
        q_fp8[:extend_num_tokens],
        weights,
        metadata,
        topk_result,
    )


bcg_dsa_indexer_prefill_split = eager_on_graph(True)(pcg_dsa_indexer_prefill_split)
