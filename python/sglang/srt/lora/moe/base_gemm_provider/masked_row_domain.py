"""Expert slabs [E_local, m_max, ...] with valid counts and pair mappings."""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
import torch

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


def masked_m_max(num_tokens: int, alignment: int = 8) -> int:
    """Fit all input rows in one expert, with a nonzero aligned slab extent."""
    return max(alignment, (num_tokens + alignment - 1) // alignment * alignment)


class MaskedRowState(msgspec.Struct, kw_only=True):
    """pair_to_row[t * top_k + k] = expert * m_max + offset when topk_ids[t, k] >= 0."""

    hidden_permuted: torch.Tensor  # [E_local, m_max, hidden], the GEMM input dtype
    masked_m: torch.Tensor  # [E_local] int32
    pair_to_row: torch.Tensor  # [num_tokens * top_k] int32
    m_max: int


class MaskedRowDomainProvider(MoeBaseProvider):
    """CuTeDSL supplies tile hooks. _row_state_cls includes buffer and schedule fields."""

    _row_state_cls = MaskedRowState

    def __init__(self, quant_info: StandardLayoutQuantInfo):
        self.quant_info = quant_info
        self._gate_up_slices = admit_weight_layout(quant_info)

    def prepare(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        workspace: LoraWorkspace | None = None,
    ) -> MaskedRowState:
        num_experts = self.quant_info.num_local_experts
        num_pairs = topk_ids.numel()
        expected_m = expected_rows_per_expert(num_pairs, num_experts)
        token_width, m_max = self._plan_rows(hidden_states.size(0), expected_m)
        device = hidden_states.device
        masked_m = prepare_buffer(
            workspace,
            "masked:masked_m",
            (num_experts,),
            dtype=torch.int32,
            device=device,
        )
        pair_to_row = prepare_buffer(
            workspace,
            "masked:pair_to_row",
            (num_pairs,),
            dtype=torch.int32,
            device=device,
        )
        rows, buffers = self._input_buffers(
            workspace, "masked", (num_experts, m_max), hidden_states
        )
        self._fill_masked(
            hidden_states, topk_ids, top_k, masked_m, pair_to_row, rows, buffers
        )
        return self._row_state_cls(
            hidden_permuted=rows,
            masked_m=masked_m,
            pair_to_row=pair_to_row,
            m_max=m_max,
            **buffers,
            **self._schedules(
                masked_m, token_width, "cutedsl_masked", workspace, m_max=m_max
            ),
        )

    def _input_buffers(
        self,
        workspace: LoraWorkspace | None,
        prefix: str,
        lead: tuple[int, ...],
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, RowBuffers]:
        """Allocate BF16 rows without clearing. The fill writes the live rows."""
        rows = prepare_buffer(
            workspace,
            f"{prefix}:hidden_permuted",
            (*lead, hidden_states.size(1)),
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )
        return rows, {}

    def _fill_masked(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        masked_m: torch.Tensor,
        pair_to_row: torch.Tensor,
        rows: torch.Tensor,
        buffers: RowBuffers,
    ) -> None:
        from sglang.kernels.ops.lora.moe.dispatch_masked import (
            dispatch_fill_masked_bf16,
        )

        dispatch_fill_masked_bf16(
            hidden_states,
            topk_ids,
            top_k,
            masked_m_out=masked_m,
            pair_to_row_out=pair_to_row,
            rows_out=rows,
        )

    # The layout's view of rows for the GEMM launch and its compile stand-ins.
    @staticmethod
    def _rows_arg(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    @staticmethod
    def _input_rows(row_state: MaskedRowState) -> torch.Tensor:
        return row_state.hidden_permuted

    @staticmethod
    def _group_arg(row_state: MaskedRowState) -> torch.Tensor:
        return row_state.masked_m

    def _dummy_layout(self, device) -> tuple[tuple, tuple, torch.Tensor]:
        experts = self.quant_info.num_local_experts
        return (
            (experts, 256),
            (experts, 256),
            torch.zeros(experts, dtype=torch.int32, device=device),
        )

    def gateup_out_shape(self, row_state: MaskedRowState) -> tuple[int, ...]:
        return (
            self.quant_info.num_local_experts,
            row_state.m_max,
            self.gate_up_slices * self.quant_info.intermediate_size,
        )

    def act_out_shape(self, row_state: MaskedRowState) -> tuple[int, ...]:
        return (
            self.quant_info.num_local_experts,
            row_state.m_max,
            self.quant_info.intermediate_size,
        )

    def down_out_shape(self, row_state: MaskedRowState) -> tuple[int, ...]:
        return (
            self.quant_info.num_local_experts,
            row_state.m_max,
            self.quant_info.hidden_size,
        )
