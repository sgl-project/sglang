"""Fused top-k gating softmax with routed-pack output (JIT).

JIT port of sgl-kernel's AOT ``topk_softmax`` power-of-2 fast path
(``topkGatingSoftmax``) extended with a third output: the FlashInfer routed-MoE
packed format ``(topk_id << 16) | bf16_bits(topk_weight)`` computed in the
kernel epilogue after renormalization — bit-identical to running the standalone
``fused_pack_topk`` triton kernel on the post-processed topk_ids/topk_weights
(including the ``_mask_topk_ids_padded_region`` id=-1 sentinel for rows at or
beyond ``num_token_non_padded``). This removes the per-MoE-layer
``_pack_topk_kernel`` launch from the decode critical path.

Scope (callers must fall back to the AOT ``topk_softmax`` + separate pack
otherwise): power-of-2 ``num_experts`` in [1, 512]; no softcapping or
correction bias (the Qwen3-MoE softmax routing uses neither).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels._jit import cache_once, load_jit
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_topk_softmax_pack_module() -> Module:
    return load_jit(
        "topk_softmax_pack",
        cuda_files=["trtllm_lora_temp/topk_softmax_pack.cuh"],
        cuda_wrappers=[("topk_softmax_pack", "topk_softmax_pack")],
    )


@register_custom_op(mutates_args=["topk_weights", "topk_indices", "packed"])
def _jit_topk_softmax_pack_op(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    packed: torch.Tensor,
    gating_output: torch.Tensor,
    num_token_non_padded: Optional[torch.Tensor],
    renormalize: bool,
) -> None:
    module = _jit_topk_softmax_pack_module()
    module.topk_softmax_pack(
        topk_weights,
        topk_indices,
        packed,
        gating_output,
        num_token_non_padded,
        renormalize,
    )


def topk_softmax_pack(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    packed: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool,
    num_token_non_padded: Optional[torch.Tensor] = None,
) -> None:
    """Drop-in for the AOT ``topk_softmax`` that ALSO writes ``packed``.

    ``packed`` is int32 ``[num_tokens, topk]``, ``(id << 16) | bf16_bits(w)``
    with the final (renormalized) weights; rows >= ``num_token_non_padded``
    pack id = -1 (the padded-region sentinel). ``topk_weights``/``topk_indices``
    are written exactly like the AOT kernel (indices NOT masked here — the
    regular python post-process handles them).
    """
    assert gating_output.dim() == 2
    num_experts = gating_output.shape[-1]
    assert num_experts & (num_experts - 1) == 0 and num_experts <= 512, (
        "topk_softmax_pack supports power-of-2 num_experts in [1, 512] only; "
        "fall back to topk_softmax + fused_pack_topk"
    )
    if gating_output.shape[0] == 0:
        return
    _jit_topk_softmax_pack_op(
        topk_weights,
        topk_indices,
        packed,
        gating_output.contiguous(),
        num_token_non_padded,
        renormalize,
    )
