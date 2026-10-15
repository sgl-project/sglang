"""Greedy (topk=1) draft steps without gathering the vocabulary: each rank packs
its shard argmax and that candidate's embedding row, one small all-gather replaces
the logits all-gather, and the winner's row becomes the next draft step's input
embedding (replacing the vocab-parallel lookup and its all-reduce). The selected
token equals the argmax over the gathered logits, ties to the lowest token id."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

from sglang.kernels.ops.speculative.draft_vocab_argmax import (
    META_ELEMS,
    PACK_BLOCKS,
    vocab_argmax_pack,
    vocab_argmax_select,
)
from sglang.srt.distributed.device_communicators import triton_symm_mem_ag
from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)


def _unsupported_reason(model, num_steps: int) -> Optional[str]:
    from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP

    if not isinstance(model, Qwen3_5ForCausalLMMTP):
        return f"draft model {type(model).__name__} has no input-embedding handoff"
    parallel = get_parallel()
    if parallel.tp_size <= 1:
        return "tp_size == 1"
    lp = model.logits_processor
    if not lp.do_tensor_parallel_all_gather or lp.do_tensor_parallel_all_gather_dp_attn:
        return "logits are not tp-gathered"
    if lp.use_attn_tp_group or lp.use_fp32_lm_head:
        return "attention-tp logits group or fp32 lm_head"
    if lp.logit_scale is not None or lp.final_logit_softcapping is not None:
        return "logit scaling or softcapping"
    embed, head = model.model.embed_tokens, model.lm_head
    if head is embed:
        return "tied embeddings"
    if should_apply_lm_head_quant_method(head, getattr(head, "quant_method", None)):
        return "quantized lm_head"
    for name, layer in (("embedding", embed), ("lm_head", head)):
        shard = layer.shard_indices
        if shard.num_org_vocab_padding != 0 or shard.num_added_elements != 0:
            return f"{name} shard has padding or added vocab"
        if layer.weight.dtype != torch.bfloat16:
            return f"{name} dtype {layer.weight.dtype}"
    if (
        embed.shard_indices.org_vocab_start_index
        != head.shard_indices.org_vocab_start_index
        or embed.weight.shape != head.weight.shape
    ):
        return "embedding and lm_head shards differ"
    if embed.weight.shape[0] % 8 != 0 or embed.weight.shape[1] % 8 != 0:
        return "shard shape not 16-byte tileable"
    return None


class DraftVocabExchange:
    """Owns the packed-row buffers, the all-gather and the per-step outputs."""

    def __init__(self, model, num_steps: int) -> None:
        self.model = model
        embed = model.model.embed_tokens
        self.embed_weight = embed.weight
        self.vocab_start = embed.shard_indices.org_vocab_start_index
        self.vocab_size = model.config.vocab_size
        self.world = get_parallel().tp_size
        hidden = embed.weight.shape[1]
        dev = embed.weight.device
        self.payload = torch.empty(1, hidden + META_ELEMS, dtype=torch.bfloat16, device=dev)
        self.part_val = torch.empty(PACK_BLOCKS, dtype=torch.float32, device=dev)
        self.part_idx = torch.empty(PACK_BLOCKS, dtype=torch.int32, device=dev)
        self.counter = torch.zeros(1, dtype=torch.int32, device=dev)
        self.next_embeds = torch.empty(1, hidden, dtype=torch.bfloat16, device=dev)
        self.topk_index = [torch.empty(1, 1, dtype=torch.int64, device=dev) for _ in range(num_steps)]
        self.topk_p = [torch.empty(1, 1, dtype=torch.float32, device=dev) for _ in range(num_steps)]
        self.no_tokens = torch.empty(0, dtype=torch.int64, device=dev)
        # Same guarded multimem all-gather as the logits gather; a cross-rank
        # all-reduce always sits between two draft steps' gathers.
        self.gatherer = triton_symm_mem_ag.MultimemAllGatherer(
            max_tokens=1, enabled=True, skip_entry_sync=True
        )

    @classmethod
    def maybe_create(cls, model, num_steps: int) -> Optional["DraftVocabExchange"]:
        reason = _unsupported_reason(model, num_steps)
        if reason is not None:
            logger.warning("Draft vocab exchange disabled (%s); using the stock path.", reason)
            return None
        logger.info("Draft vocab exchange active (tp=%d).", get_parallel().tp_size)
        return cls(model, num_steps)

    def begin_step(self, step: int) -> None:
        self.model.draft_input_embeds = self.next_embeds if step > 0 else None
        self.model.logits_processor.skip_vocab_gather = True

    def finish_step(
        self,
        local_logits: torch.Tensor,
        positions: torch.Tensor,
        draft_tokens: Optional[torch.Tensor],
        step: int,
        draft_column: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        vocab_argmax_pack(
            local_logits,
            self.embed_weight,
            self.payload,
            self.part_val,
            self.part_idx,
            self.counter,
            self.vocab_start,
            self.world,
        )
        gathered = self.gatherer(self.payload)
        topk_index, topk_p = self.topk_index[step], self.topk_p[step]
        vocab_argmax_select(
            gathered,
            topk_index,
            topk_p,
            draft_tokens if draft_tokens is not None else self.no_tokens,
            positions,
            self.next_embeds,
            draft_column,
            PACK_BLOCKS,
            self.world,
        )
        return topk_p, topk_index

    def end(self) -> None:
        self.model.draft_input_embeds = None
        self.model.logits_processor.skip_vocab_gather = False
