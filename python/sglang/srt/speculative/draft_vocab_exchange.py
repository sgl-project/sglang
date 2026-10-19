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

from sglang.kernels.ops.speculative.draft_peer_xchg import draft_argmax_xchg
from sglang.kernels.ops.speculative.draft_vocab_argmax import (
    META_ELEMS,
    NAN_KEY_SPLIT_ARGMAX,
    NAN_KEY_TORCH_ARGMAX,
    PACK_BLOCKS,
    vocab_argmax_pack,
    vocab_argmax_select,
)
from sglang.srt.distributed.device_communicators import triton_symm_mem_ag
from sglang.srt.environ import envs
from sglang.srt.layers.kdense_draft_xchg import make_argmax_slots
from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.draft_hot_vocab import DraftHotVocab, build_hot_head_shard

logger = logging.getLogger(__name__)


def _unsupported_reason(
    model, num_steps: int, hot: Optional[DraftHotVocab]
) -> Optional[str]:
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
    if hot is not None:
        # The hot head is plain bf16; only the embedding shard is checked.
        shard = embed.shard_indices
        if shard.num_org_vocab_padding != 0 or shard.num_added_elements != 0:
            return "embedding shard has padding or added vocab"
        if embed.weight.dtype != torch.bfloat16:
            return f"embedding dtype {embed.weight.dtype}"
        if head.weight.shape[0] % 8 != 0 or embed.weight.shape[1] % 8 != 0:
            return "hot shard shape not 16-byte tileable"
        return None
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

    MAX_ROWS = envs.SGLANG_Q35MF_DRAFT_VOCAB_EXCHANGE_MAX_ROWS.get()

    def __init__(
        self, model, num_steps: int, hot: Optional[DraftHotVocab] = None
    ) -> None:
        self.model = model
        self.num_steps = num_steps
        embed = model.model.embed_tokens
        parallel = get_parallel()
        self.world = parallel.tp_size
        self.vocab_size = model.config.vocab_size
        dev = embed.weight.device
        if hot is None:
            self.embed_weight = embed.weight
            self.vocab_start = embed.shard_indices.org_vocab_start_index
            self.token_map = torch.empty(0, dtype=torch.int64, device=dev)
        else:
            # Candidate rows are the embeddings of this rank's hot ids, in hot order.
            shard = embed.shard_indices
            self.embed_weight = build_hot_head_shard(
                weight=embed.weight.data,
                org_vocab_start=shard.org_vocab_start_index,
                org_vocab_end=shard.org_vocab_end_index,
                hot_ids=hot.token_ids.cpu(),
                tp_rank=parallel.tp_rank,
                tp_size=self.world,
                group=parallel.tp_group.device_group if self.world > 1 else None,
            )
            rows = self.embed_weight.shape[0]
            self.vocab_start = 0
            self.token_map = hot.token_ids[
                parallel.tp_rank * rows : (parallel.tp_rank + 1) * rows
            ].contiguous()
        self.local_vocab = self.embed_weight.shape[0]
        hidden = self.embed_weight.shape[1]
        # Draft steps of up to step_rows requests use the exchange; the draft extend keeps MAX_ROWS.
        step_rows = envs.SGLANG_Q35MF_DRAFT_EXCHANGE_STEP_ROWS.get()
        self.step_rows = (
            self.MAX_ROWS if step_rows is None else min(max(step_rows, 1), 8)
        )
        rows = max(self.MAX_ROWS, self.step_rows)
        self.payload = torch.empty(
            rows, hidden + META_ELEMS, dtype=torch.bfloat16, device=dev
        )
        self.part_key = torch.empty(rows, PACK_BLOCKS, dtype=torch.int32, device=dev)
        self.part_idx = torch.empty(rows, PACK_BLOCKS, dtype=torch.int32, device=dev)
        self.counter = torch.zeros(rows, dtype=torch.int32, device=dev)
        self.next_embeds = torch.empty(rows, hidden, dtype=torch.bfloat16, device=dev)
        self.topk_index = [
            torch.empty(rows, 1, dtype=torch.int64, device=dev)
            for _ in range(num_steps)
        ]
        self.topk_p = [
            torch.empty(rows, 1, dtype=torch.float32, device=dev)
            for _ in range(num_steps)
        ]
        self.no_tokens = torch.empty(0, dtype=torch.int64, device=dev)
        self.no_positions = torch.empty(0, dtype=torch.int64, device=dev)
        self.num_rows = 1
        # One-launch boundary (SGLANG_Q35MF_MKD_XCHG), attached after construction.
        self.mkd = None
        self.max_rows_seen = 0
        # Same guarded multimem all-gather as the logits gather; a cross-rank
        # all-reduce always sits between two draft steps' gathers.
        self.gatherer = triton_symm_mem_ag.MultimemAllGatherer(
            max_tokens=rows, enabled=True, skip_entry_sync=True
        )
        self.peer = (
            make_argmax_slots(rows, hidden)
            if envs.SGLANG_Q35MF_KDENSE_DRAFT_ARGMAX_XCHG.get() and self.world == 2
            else None
        )
        self.no_mrope = torch.empty(0, dtype=torch.int64, device=dev)

    @classmethod
    def maybe_create(
        cls, model, num_steps: int, hot: Optional[DraftHotVocab] = None
    ) -> Optional[DraftVocabExchange]:
        reason = _unsupported_reason(model, num_steps, hot)
        if reason is not None:
            logger.warning(
                "Draft vocab exchange disabled (%s); using the stock path.", reason
            )
            return None
        exchange = cls(model, num_steps, hot)
        logger.info(
            "Draft vocab exchange active (tp=%d%s, draft steps up to %d rows, draft extend up to %d rows).",
            get_parallel().tp_size,
            ", hot vocabulary" if hot is not None else "",
            exchange.step_rows,
            cls.MAX_ROWS,
        )
        return exchange

    def begin_step(self, step: int, num_rows: int) -> None:
        self.num_rows = num_rows
        if num_rows > self.max_rows_seen:
            self.max_rows_seen = num_rows
            logger.info(
                "Draft vocab exchange covers %d-row draft steps (limit %d).",
                num_rows,
                self.step_rows,
            )
        mkd = self.mkd
        if mkd is not None:
            mkd.armed = num_rows == 1
            mkd.step = step
            # The draft loop forwards steps 0 .. num_steps - 2.
            mkd.last_step = step == self.num_steps - 2
            if step == 0:
                # With extend_fc the draft caller guarantees fc_buf holds this step's fc.
                mkd.pending = mkd.extend_fc and num_rows == 1
        if mkd is not None and mkd.pending:
            # The model finishes the fc from the exchange; this row only carries the shape.
            self.model.draft_input_embeds = self.next_embeds[:num_rows]
        else:
            self.model.draft_input_embeds = (
                self.next_embeds[:num_rows] if step > 0 else None
            )
        self.model.logits_processor.skip_vocab_gather_rows = self.step_rows

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
        topk_index, topk_p = self.topk_index[step][:m], self.topk_p[step][:m]
        mkd = self.mkd
        if mkd is not None and mkd.armed:
            mkd.exchange(
                local_logits,
                topk_index,
                topk_p,
                draft_tokens,
                positions,
                draft_column,
                mrope_positions,
            )
            mkd.armed = False
            return topk_p, topk_index
        if self.peer is not None and self.peer.ready():
            self._exchange_select(
                local_logits,
                m=m,
                nan_key=NAN_KEY_SPLIT_ARGMAX,
                topk_index=topk_index,
                topk_p=topk_p,
                draft_tokens=draft_tokens
                if draft_tokens is not None
                else self.no_tokens,
                positions=positions,
                mrope_positions=mrope_positions
                if mrope_positions is not None
                else self.no_mrope,
                draft_column=draft_column,
            )
            return topk_p, topk_index
        gathered = self._pack_and_gather(local_logits, m, NAN_KEY_SPLIT_ARGMAX)
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
        return (
            logits.shape[-1] == self.local_vocab
            and 0 < logits.shape[0] <= self.MAX_ROWS
        )

    def select_extend(
        self, local_logits: torch.Tensor, hidden_states: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draft-extend argmax (torch.argmax semantics) from shard-local logits."""
        m = local_logits.shape[0]
        if self.mkd is not None and m == 1:
            return self.mkd.exchange_extend(local_logits, hidden_states)
        topk_index = torch.empty((m, 1), dtype=torch.int64, device=local_logits.device)
        topk_p = torch.empty((m, 1), dtype=torch.float32, device=local_logits.device)
        if self.peer is not None and self.peer.ready():
            self._exchange_select(
                local_logits,
                m=m,
                nan_key=NAN_KEY_TORCH_ARGMAX,
                topk_index=topk_index,
                topk_p=topk_p,
                draft_tokens=self.no_tokens,
                positions=self.no_positions,
                mrope_positions=self.no_mrope,
                draft_column=0,
            )
            return topk_p, topk_index
        gathered = self._pack_and_gather(local_logits, m, NAN_KEY_TORCH_ARGMAX)
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

    def _exchange_select(
        self,
        local_logits: torch.Tensor,
        *,
        m: int,
        nan_key: int,
        topk_index: torch.Tensor,
        topk_p: torch.Tensor,
        draft_tokens: torch.Tensor,
        positions: torch.Tensor,
        mrope_positions: torch.Tensor,
        draft_column: int,
    ) -> None:
        # Same pack and selection as _pack_and_gather + vocab_argmax_select, in one launch.
        draft_argmax_xchg(
            local_logits,
            self.embed_weight,
            self.part_key[:m],
            self.part_idx[:m],
            self.counter[:m],
            self.token_map,
            self.peer.slots,
            self.peer.peer_slots,
            self.peer.epoch,
            topk_index,
            topk_p,
            draft_tokens,
            positions,
            mrope_positions,
            self.next_embeds[:m],
            self.vocab_start,
            nan_key,
            self.peer.rank,
            draft_column,
        )

    def _pack_and_gather(
        self, local_logits: torch.Tensor, m: int, nan_key: int
    ) -> torch.Tensor:
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
            self.token_map,
        )
        return self.gatherer(self.payload[:m])
