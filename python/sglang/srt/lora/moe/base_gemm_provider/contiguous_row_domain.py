"""Routed rows packed into aligned, contiguous expert segments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
import torch

from sglang.kernels.ops.lora.moe.dispatch_contiguous import (
    contiguous_m_pad_ceiling,
    dispatch_fill_rows_contiguous_bf16,
    dispatch_layout_contiguous,
)
from sglang.srt.lora.moe.base_gemm_provider.base import (
    MoeBaseProvider,
    RowBuffers,
    admit_weight_layout,
    expected_rows_per_expert,
    prepare_buffer,
)
from sglang.srt.lora.moe.quant_info import StandardLayoutQuantInfo

if TYPE_CHECKING:
    from sglang.srt.lora.workspace import LoraWorkspace


class ContiguousRowState(msgspec.Struct, kw_only=True):
    hidden_compact: torch.Tensor  # [m_pad_ceiling, hidden], the GEMM input dtype
    seg_counts: torch.Tensor  # [E_local] int32
    seg_offsets: torch.Tensor  # [E_local + 1] int32 first row of each segment
    pair_to_row: torch.Tensor  # [num_tokens * top_k] int32 compact rows
    m_pad_ceiling: int


class ContiguousRowDomainProvider(MoeBaseProvider):
    """CuTeDSL supplies tile hooks. _row_state_cls includes buffer and schedule fields."""

    _row_state_cls = ContiguousRowState
    # A fill that turns the layout's in-expert slots into rows itself.
    _fill_finalizes_map = False
    # Each token width divides 128; block-scaled segments start on 128-row scale atoms.
    M_ALIGNMENT = 128

    def __init__(self, quant_info: StandardLayoutQuantInfo):
        self.quant_info = quant_info
        self._gate_up_slices = admit_weight_layout(quant_info)

    def prepare(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        workspace: LoraWorkspace | None = None,
    ) -> ContiguousRowState:
        num_pairs = topk_ids.numel()
        num_experts = self.quant_info.num_local_experts
        m_pad_ceiling = contiguous_m_pad_ceiling(
            num_pairs, num_experts, self.M_ALIGNMENT
        )
        device = hidden_states.device
        prefix = f"contiguous:a{self.M_ALIGNMENT}"
        seg_counts = prepare_buffer(
            workspace,
            f"{prefix}:seg_counts",
            (num_experts,),
            dtype=torch.int32,
            device=device,
        )
        seg_offsets = prepare_buffer(
            workspace,
            f"{prefix}:seg_offsets",
            (num_experts + 1,),
            dtype=torch.int32,
            device=device,
        )
        pair_to_row = prepare_buffer(
            workspace,
            f"{prefix}:pair_to_row",
            (num_pairs,),
            dtype=torch.int32,
            device=device,
        )
        rows, buffers = self._input_buffers(
            workspace, prefix, (m_pad_ceiling,), hidden_states
        )
        dispatch_layout_contiguous(
            hidden_states,
            topk_ids,
            num_experts,
            top_k,
            self.M_ALIGNMENT,
            seg_counts_out=seg_counts,
            seg_offsets_out=seg_offsets,
            pair_to_row_out=pair_to_row,
            finalize=not self._fill_finalizes_map,
        )
        self._fill_contiguous(
            hidden_states, topk_ids, pair_to_row, seg_offsets, rows, buffers
        )
        # Distinct top-k experts bound each expert to one row per token.
        max_expert_rows = max(int(hidden_states.size(0)), 1)
        token_width = self._token_width_for(
            max_expert_rows, expected_rows_per_expert(num_pairs, num_experts)
        )
        return self._row_state_cls(
            hidden_compact=rows,
            seg_counts=seg_counts,
            seg_offsets=seg_offsets,
            pair_to_row=pair_to_row,
            m_pad_ceiling=m_pad_ceiling,
            **buffers,
            **self._schedules(
                seg_counts,
                token_width,
                f"cutedsl_contiguous:a{self.M_ALIGNMENT}",
                workspace,
                m_pad_ceiling=m_pad_ceiling,
                max_expert_rows=max_expert_rows,
                m_alignment=self.M_ALIGNMENT,
            ),
        )

    def _input_buffers(
        self,
        workspace: LoraWorkspace | None,
        prefix: str,
        lead: tuple[int, ...],
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, RowBuffers]:
        """Clear BF16 rows only on allocation. The fill writes the live rows."""
        # Padding outputs are ignored, so reused padding needs no reset.
        rows = prepare_buffer(
            workspace,
            f"{prefix}:hidden_compact",
            (*lead, hidden_states.size(1)),
            dtype=torch.bfloat16,
            device=hidden_states.device,
            zero_on_first_allocation=True,
        )
        return rows, {}

    def _fill_contiguous(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        pair_to_row: torch.Tensor,
        seg_offsets: torch.Tensor,
        rows: torch.Tensor,
        buffers: RowBuffers,
    ) -> None:
        dispatch_fill_rows_contiguous_bf16(
            hidden_states, topk_ids, pair_to_row, hidden_compact_out=rows
        )

    # The layout's view of rows for the GEMM launch and its compile stand-ins.
    @staticmethod
    def _rows_arg(tensor: torch.Tensor) -> torch.Tensor:
        # Flat rows occupy one expert slot in the swap_ab wrapper.
        return tensor.unsqueeze(0)

    @staticmethod
    def _input_rows(row_state: ContiguousRowState) -> torch.Tensor:
        return row_state.hidden_compact

    @staticmethod
    def _group_arg(row_state: ContiguousRowState) -> torch.Tensor:
        return row_state.seg_offsets

    def _dummy_layout(self, device) -> tuple[tuple, tuple, torch.Tensor]:
        experts = self.quant_info.num_local_experts
        return (
            (256,),
            (1, 256),
            torch.zeros(experts + 1, dtype=torch.int32, device=device),
        )

    def gateup_out_shape(self, row_state: ContiguousRowState) -> tuple[int, ...]:
        return (
            row_state.m_pad_ceiling,
            self.gate_up_slices * self.quant_info.intermediate_size,
        )

    def act_out_shape(self, row_state: ContiguousRowState) -> tuple[int, ...]:
        return (row_state.m_pad_ceiling, self.quant_info.intermediate_size)

    def down_out_shape(self, row_state: ContiguousRowState) -> tuple[int, ...]:
        return (row_state.m_pad_ceiling, self.quant_info.hidden_size)
