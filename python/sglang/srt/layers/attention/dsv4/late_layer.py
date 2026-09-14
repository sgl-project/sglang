"""Token selection and DP collective layouts for decoder SWA bounded replay."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.attention.deepseek_v4_backend import LateLayerTail
    from sglang.srt.layers.dp_attention import DpPaddingMode
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def select_tail_rows(
    t: torch.Tensor, *, token_indices: torch.Tensor, contiguous_start: Optional[int]
) -> torch.Tensor:
    if contiguous_start is not None:
        # DP may have padded the full extend. Do not include that padding in the
        # tail: its attention metadata describes only the selected real rows.
        return t[contiguous_start : contiguous_start + token_indices.shape[0]]
    return t[token_indices]


def scatter_tail_rows(
    tail: LateLayerTail, rows: torch.Tensor, num_tokens: int
) -> torch.Tensor:
    # Only tail rows are read by the logits processor. Keep the original padded
    # shape without computing or initializing the skipped prompt rows.
    full = rows.new_empty((num_tokens, *rows.shape[1:]))
    n = tail.token_indices.shape[0]
    if tail.contiguous_start is not None:
        full[tail.contiguous_start : tail.contiguous_start + n].copy_(rows[:n])
    else:
        full[tail.token_indices] = rows[:n]
    return full


@dataclass
class LateLayerDPLayout:
    """A temporary collective layout; attention and logits retain their layouts.

    Every DP rank constructs this at the same layer boundary, including ranks
    doing decode or idle work. Only MoE and Engram inputs are padded, so tail
    attention metadata never describes communication-only padding rows.
    """

    local_rows: int
    dp_rank: int
    counts: list[int]
    counts_gpu: torch.Tensor
    padding_mode: DpPaddingMode
    num_token_non_padded: Optional[torch.Tensor]
    num_token_non_padded_cpu: Optional[int]

    @classmethod
    def prepare(cls, input_ids: torch.Tensor, batch: ForwardBatch) -> LateLayerDPLayout:
        from sglang.srt.distributed import get_tp_group
        from sglang.srt.layers.dp_attention import (
            get_attention_dp_rank,
            get_attention_dp_size,
            world_dp_gather_enabled,
        )
        from sglang.srt.runtime_context import get_parallel

        # A rank-local forward_mode check here would deadlock extend/idle or
        # extend/decode batches. The caller uses the DP-wide is_extend_in_batch.
        local_count = input_ids.new_tensor([input_ids.shape[0]], dtype=torch.int64)
        use_world = world_dp_gather_enabled()
        group = get_tp_group()
        world_size = (
            torch.distributed.get_world_size() if use_world else group.world_size
        )
        gathered = local_count.new_empty(world_size)
        if use_world:
            torch.distributed.all_gather_into_tensor(gathered, local_count)
        else:
            group.all_gather_into_tensor(gathered, local_count)
        attn_tp_size = get_parallel().attn_tp_size
        # Attention TP ranks own replicas of the same local token rows.
        counts = gathered.tolist()[::attn_tp_size]
        assert len(counts) == get_attention_dp_size()
        return cls.from_counts(
            counts,
            dp_rank=get_attention_dp_rank(),
            attn_tp_size=attn_tp_size,
            batch=batch,
            device=input_ids.device,
        )

    @classmethod
    def from_counts(
        cls,
        counts: list[int],
        *,
        dp_rank: int,
        attn_tp_size: int,
        batch: ForwardBatch,
        device: torch.device,
    ) -> LateLayerDPLayout:
        local_rows = counts[dp_rank]
        counts = [(n + attn_tp_size - 1) // attn_tp_size * attn_tp_size for n in counts]
        if batch.dp_padding_mode.is_max_len():
            counts = [max(counts)] * len(counts)
        non_padded = batch.num_token_non_padded
        non_padded_cpu = batch.num_token_non_padded_cpu
        return cls(
            local_rows=local_rows,
            dp_rank=dp_rank,
            counts=counts,
            counts_gpu=torch.tensor(counts, dtype=torch.int64, device=device),
            padding_mode=batch.dp_padding_mode,
            num_token_non_padded=(
                non_padded.clamp(max=local_rows) if non_padded is not None else None
            ),
            num_token_non_padded_cpu=(
                min(non_padded_cpu, local_rows) if non_padded_cpu is not None else None
            ),
        )

    def pad(self, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.shape[0] == self.local_rows
        padded_rows = self.counts[self.dp_rank]
        if padded_rows == self.local_rows:
            return tensor.contiguous()
        padded = tensor.new_zeros((padded_rows, *tensor.shape[1:]))
        padded[: self.local_rows].copy_(tensor)
        return padded

    def gather_input_ids(
        self, input_ids: torch.Tensor, batch: ForwardBatch, *, gather: bool
    ) -> torch.Tensor:
        local_ids = self.pad(input_ids)
        if not gather:
            return local_ids
        from sglang.srt.layers.dp_attention import dp_gather_replicate

        global_ids = input_ids.new_empty((sum(self.counts), 1))
        with self.activate(batch):
            # MAX_LEN gathering can zero non-leader attention-TP inputs in place.
            dp_gather_replicate(global_ids, local_ids[:, None].clone(), batch)
        return global_ids.squeeze(-1)

    @contextmanager
    def activate(self, batch: ForwardBatch):
        from sglang.srt.layers.dp_attention import dp_buffer_size_scope

        updates = dict(
            global_num_tokens_cpu=self.counts,
            global_num_tokens_gpu=self.counts_gpu,
            global_dp_buffer_len=sum(self.counts),
            dp_padding_mode=self.padding_mode,
            dp_local_start_pos=None,
            dp_local_num_tokens=None,
            num_token_non_padded=self.num_token_non_padded,
            num_token_non_padded_cpu=self.num_token_non_padded_cpu,
        )
        saved = {name: getattr(batch, name) for name in updates}
        try:
            for name, value in updates.items():
                setattr(batch, name, value)
            with dp_buffer_size_scope(
                sum(self.counts),
                self.counts[self.dp_rank],
                self.padding_mode.is_max_len(),
                self.counts,
                self.counts_gpu,
            ):
                yield
        finally:
            for name, value in saved.items():
                setattr(batch, name, value)
