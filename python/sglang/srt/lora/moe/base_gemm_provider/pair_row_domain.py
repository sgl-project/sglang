"""Rows kept in raw pair order behind a pair index sorted by expert."""

from __future__ import annotations

import msgspec
import torch

from sglang.srt.lora.moe.base_gemm_provider.base import (
    MoeBaseProvider,
    prepare_buffer,
)


class PairRowState(msgspec.Struct, kw_only=True):
    hidden_states: torch.Tensor  # [num_tokens, hidden] bf16, borrowed
    sorted_token_ids: torch.Tensor  # [em_ceiling] int32 route ids, pad = numel
    expert_ids: torch.Tensor  # [em_ceiling / block_m] int32 per-block expert
    num_tokens_post_padded: torch.Tensor  # [1] int32 device scalar
    pair_to_row: torch.Tensor  # [num_pairs] int32 pair index, or -1 when dropped
    num_pairs: int
    top_k: int


class PairRowDomainProvider(MoeBaseProvider):
    """The GEMMs gather rows through the sorted index and scatter into pair
    order; subclasses choose the row-block size and add their own state."""

    def __init__(self, quant_info, *, gate_up_slices: int, device: torch.device):
        self.quant_info = quant_info
        self._gate_up_slices = gate_up_slices
        # Finalize applies router weights; the launchers never read this one.
        self._unused_topk_weights = torch.zeros(
            (1, 1), dtype=torch.float32, device=device
        )

    def _sorted_rows(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        block_size: int,
        workspace,
    ) -> dict:
        """The PairRowState fields for pairs sorted by expert in row blocks."""
        from sglang.srt.lora.moe.base_gemm_provider.input_alignment import align_rows

        num_pairs = topk_ids.numel()
        device = hidden_states.device
        pair_to_row = prepare_buffer(
            workspace, "pair_to_row", (num_pairs,), dtype=torch.int32, device=device
        )
        # Finalize uses pair_to_row, with -1 for pairs without an expert.
        sorted_token_ids, expert_ids, num_tokens_post_padded = align_rows(
            topk_ids, block_size, self.num_local_experts, pair_to_row_out=pair_to_row
        )
        return dict(
            hidden_states=hidden_states,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            pair_to_row=pair_to_row,
            num_pairs=num_pairs,
            top_k=top_k,
        )

    def release_prepared_inputs(self, row_state: PairRowState) -> None:
        pass

    def gateup_out_shape(self, row_state: PairRowState) -> tuple[int, ...]:
        return (row_state.num_pairs, self.gate_up_slices * self.intermediate_size)

    def act_out_shape(self, row_state: PairRowState) -> tuple[int, ...]:
        return (row_state.num_pairs, self.intermediate_size)

    def down_out_shape(self, row_state: PairRowState) -> tuple[int, ...]:
        return (row_state.num_pairs, self.hidden_size)
