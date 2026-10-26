"""ROCm glue of LogitsProcessor: a gfx950 DSpark target verify leaves its logits
vocab-sharded and all-gathers one 64-byte row-argmax partial per row instead of the
[rows, vocab] logits; logits_processor binds it under _is_hip."""

from __future__ import annotations

import torch

from sglang.kernels.ops.speculative.dspark.dspark_gfx950 import tp_shard_row_argmax
from sglang.srt.runtime_context import LoRABatchLayout, get_forward, get_parallel


def can_verify_argmax_only(processor) -> bool:
    """Whether processor's logits reach the sampler through a plain TP gather alone, which
    is what deferred_verify_logits leaves out."""
    return bool(
        processor.do_tensor_parallel_all_gather
        and not processor.do_tensor_parallel_all_gather_dp_attn
        and not processor.use_attn_tp_group
        and not processor.use_tp_lm_head_all_to_all
        and processor.logit_scale is None
        and not processor.final_logit_softcapping
    )


def deferred_verify_logits(
    processor, hidden_states, lm_head, logits_metadata, hidden_states_to_store
):
    """This rank's [rows, vocab / tp] verify logits and the row argmax of the full ones."""
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput

    with get_forward().scoped(lora_batch_layout=LoRABatchLayout.DP_LOCAL):
        local = processor._compute_lm_head(hidden_states, lm_head)
    tp = get_parallel().tp_group
    shard = local.shape[-1]
    col_start = tp.rank_in_group * shard
    argmax = tp_shard_row_argmax(
        local,
        col_start=col_start,
        num_valid=max(0, min(shard, processor.vocab_size - col_start)),
        world_size=tp.world_size,
        all_gather_into_tensor=tp.all_gather_into_tensor,
    )
    return LogitsProcessorOutput(
        next_token_logits=None,
        hidden_states=hidden_states_to_store,
        mm_input_embeds=logits_metadata.mm_input_embeds,
        tp_local_logits=local,
        tp_logits_argmax=argmax,
    )


def gather_deferred_verify_logits(processor, local_logits) -> torch.Tensor:
    """The fp32 [rows, vocab] logits the TP gather would have returned, from the deferred
    verify's tp_local_logits rows."""
    logits = processor._logits_gatherer(local_logits.contiguous())
    if logits.shape[-1] > processor.vocab_size:
        logits = logits[:, : processor.vocab_size]
    return logits.float()
