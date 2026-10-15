"""Greedy (topk=1) draft steps without gathering the vocabulary: each rank packs
its shard argmax and that candidate's embedding row, one small all-gather replaces
the logits all-gather, and the winner's row becomes the next draft step's input
embedding (replacing the vocab-parallel lookup and its all-reduce). The selected
token equals the argmax over the gathered logits, ties to the lowest token id;
NaN follows the argmax each path replaces (split argmax in draft steps,
torch.argmax after the draft extend)."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

from sglang.kernels.ops.speculative.draft_vocab_argmax import (
    META_ELEMS,
    NAN_KEY_SPLIT_ARGMAX,
    NAN_KEY_TORCH_ARGMAX,
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

    MAX_ROWS = 2

    def __init__(self, model, num_steps: int) -> None:
        self.model = model
        embed = model.model.embed_tokens
        self.embed_weight = embed.weight
        self.vocab_start = embed.shard_indices.org_vocab_start_index
        self.vocab_size = model.config.vocab_size
        self.local_vocab = embed.weight.shape[0]
        self.world = get_parallel().tp_size
        hidden = embed.weight.shape[1]
        dev = embed.weight.device
        rows = self.MAX_ROWS
        self.payload = torch.empty(rows, hidden + META_ELEMS, dtype=torch.bfloat16, device=dev)
        self.part_key = torch.empty(rows, PACK_BLOCKS, dtype=torch.int32, device=dev)
        self.part_idx = torch.empty(rows, PACK_BLOCKS, dtype=torch.int32, device=dev)
        self.counter = torch.zeros(rows, dtype=torch.int32, device=dev)
        self.next_embeds = torch.empty(rows, hidden, dtype=torch.bfloat16, device=dev)
        self.topk_index = [torch.empty(rows, 1, dtype=torch.int64, device=dev) for _ in range(num_steps)]
        self.topk_p = [torch.empty(rows, 1, dtype=torch.float32, device=dev) for _ in range(num_steps)]
        self.no_tokens = torch.empty(0, dtype=torch.int64, device=dev)
        self.no_positions = torch.empty(0, dtype=torch.int64, device=dev)
        self.num_rows = 1
        # Same guarded multimem all-gather as the logits gather; a cross-rank
        # all-reduce always sits between two draft steps' gathers.
        self.gatherer = triton_symm_mem_ag.MultimemAllGatherer(
            max_tokens=self.MAX_ROWS, enabled=True, skip_entry_sync=True
        )

    @classmethod
    def maybe_create(cls, model, num_steps: int) -> Optional["DraftVocabExchange"]:
        reason = _unsupported_reason(model, num_steps)
        if reason is not None:
            logger.warning("Draft vocab exchange disabled (%s); using the stock path.", reason)
            return None
        logger.info("Draft vocab exchange active (tp=%d).", get_parallel().tp_size)
        return cls(model, num_steps)

    def begin_step(self, step: int, num_rows: int) -> None:
        self.num_rows = num_rows
        self.model.draft_input_embeds = self.next_embeds[:num_rows] if step > 0 else None
        self.model.logits_processor.skip_vocab_gather_rows = self.MAX_ROWS

    def finish_step(
        self,
        local_logits: torch.Tensor,
        positions: torch.Tensor,
        draft_tokens: Optional[torch.Tensor],
        step: int,
        draft_column: int,
        mrope_positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        m = self.num_rows
        gathered = self._pack_and_gather(local_logits, m, NAN_KEY_SPLIT_ARGMAX)
        topk_index, topk_p = self.topk_index[step][:m], self.topk_p[step][:m]
        vocab_argmax_select(
            gathered,
            topk_index,
            topk_p,
            draft_tokens if draft_tokens is not None else self.no_tokens,
            positions,
            self.next_embeds[:m],
            draft_column,
            PACK_BLOCKS,
            self.world,
            mrope_positions,
        )
        return topk_p, topk_index

    def end(self) -> None:
        self.model.draft_input_embeds = None
        self.model.logits_processor.skip_vocab_gather_rows = 0

    def is_local(self, logits: torch.Tensor) -> bool:
        return logits.shape[-1] == self.local_vocab and 0 < logits.shape[0] <= self.MAX_ROWS

    def select_extend(self, local_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draft-extend argmax (torch.argmax semantics) from shard-local logits."""
        m = local_logits.shape[0]
        gathered = self._pack_and_gather(local_logits, m, NAN_KEY_TORCH_ARGMAX)
        topk_index = torch.empty((m, 1), dtype=torch.int64, device=local_logits.device)
        topk_p = torch.empty((m, 1), dtype=torch.float32, device=local_logits.device)
        vocab_argmax_select(
            gathered,
            topk_index,
            topk_p,
            self.no_tokens,
            self.no_positions,
            self.next_embeds[:m],
            0,
            PACK_BLOCKS,
            self.world,
        )
        return topk_p, topk_index

    def _pack_and_gather(self, local_logits: torch.Tensor, m: int, nan_key: int) -> torch.Tensor:
        vocab_argmax_pack(
            local_logits,
            self.embed_weight,
            self.payload[:m],
            self.part_key[:m],
            self.part_idx[:m],
            self.counter[:m],
            self.vocab_start,
            self.world,
            nan_key,
        )
        return self.gatherer(self.payload[:m])
