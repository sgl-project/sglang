"""Routed rows packed into aligned, contiguous expert segments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import msgspec
import torch

from sglang.srt.lora.moe.base_gemm_provider.base import (
    MoeBaseProvider,
    admit_bf16_weights,
)
from sglang.srt.lora.moe.kernels.activation_delta import (
    act_delta_contiguous,
)
from sglang.srt.lora.moe.kernels.dispatch_contiguous import (
    contiguous_m_pad_ceiling,
    dispatch_fill_rows_contiguous_bf16,
    dispatch_layout_contiguous,
)
from sglang.srt.lora.moe.kernels.fused_act import (
    fused_b_act_contiguous,
)
from sglang.srt.lora.moe.quant_info import MoeLoraBf16QuantInfo

if TYPE_CHECKING:
    from sglang.srt.lora.moe.route_view import RouteView
    from sglang.srt.lora.moe.workspace import MoeLoraWorkspace


class ContiguousRowState(msgspec.Struct, kw_only=True):
    hidden_compact: torch.Tensor  # [m_pad_ceiling, hidden] bf16
    seg_counts: torch.Tensor  # [E_local] int32
    seg_offsets: torch.Tensor  # [E_local + 1] int32 first row of each segment
    pair_to_row: torch.Tensor  # [num_tokens * top_k] int32 compact rows
    m_pad_ceiling: int
    retained_inputs: bool


class ContiguousRowDomainProvider(MoeBaseProvider):

    def __init__(self, quant_info: MoeLoraBf16QuantInfo, *, m_alignment: int):
        self.quant_info = quant_info
        if not isinstance(m_alignment, int) or m_alignment < 1:
            raise ValueError(f"m_alignment must be a positive int, got {m_alignment!r}")
        self._m_alignment = m_alignment
        self._gate_up_slices = admit_bf16_weights(quant_info)

    @property
    def m_alignment(self) -> int:
        return self._m_alignment

    def prepare(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        workspace: MoeLoraWorkspace | None = None,
    ) -> ContiguousRowState:
        num_pairs = topk_ids.numel()
        num_experts = self.quant_info.num_local_experts
        m_pad_ceiling = contiguous_m_pad_ceiling(
            num_pairs, num_experts, self._m_alignment
        )
        device = hidden_states.device
        if workspace is not None:
            prefix = f"contiguous:a{self._m_alignment}"
            seg_counts = workspace.tensor(
                f"{prefix}:seg_counts",
                (num_experts,),
                dtype=torch.int32,
                device=device,
            )
            seg_offsets = workspace.tensor(
                f"{prefix}:seg_offsets",
                (num_experts + 1,),
                dtype=torch.int32,
                device=device,
            )
            pair_to_row = workspace.tensor(
                f"{prefix}:pair_to_row",
                (num_pairs,),
                dtype=torch.int32,
                device=device,
            )
            # First-allocation zero only: stale rows are safe, since nothing
            # reads a padding row's output.
            hidden_compact = workspace.tensor(
                f"{prefix}:hidden_compact",
                (m_pad_ceiling, hidden_states.size(1)),
                dtype=torch.bfloat16,
                device=device,
                zero_on_first_allocation=True,
            )
        else:
            seg_counts = torch.empty(num_experts, dtype=torch.int32, device=device)
            seg_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
            pair_to_row = torch.empty(num_pairs, dtype=torch.int32, device=device)
            hidden_compact = torch.zeros(
                (m_pad_ceiling, hidden_states.size(1)),
                dtype=torch.bfloat16,
                device=device,
            )
        dispatch_layout_contiguous(
            hidden_states,
            topk_ids,
            num_experts,
            top_k,
            self._m_alignment,
            seg_counts_out=seg_counts,
            seg_offsets_out=seg_offsets,
            pair_to_row_out=pair_to_row,
        )
        if hidden_compact is not None:
            dispatch_fill_rows_contiguous_bf16(
                hidden_states, topk_ids, pair_to_row, hidden_compact_out=hidden_compact
            )
        return ContiguousRowState(
            hidden_compact=hidden_compact,
            seg_counts=seg_counts,
            seg_offsets=seg_offsets,
            pair_to_row=pair_to_row,
            m_pad_ceiling=m_pad_ceiling,
            retained_inputs=workspace is not None,
        )

    def release_prepared_inputs(self, row_state: ContiguousRowState) -> None:
        # A workspace tensor must keep its address for graph replay, so this
        # frees only an eagerly allocated buffer.
        if row_state.retained_inputs:
            return
        from sglang.srt.utils import dispose_tensor

        dispose_tensor(row_state.hidden_compact)

    def act_with_delta(
        self,
        row_state: ContiguousRowState,
        gateup_out: torch.Tensor,
        gate_up_delta: torch.Tensor | None,
        topk_ids: torch.Tensor,
        act_out: torch.Tensor,
        activation_lora_input: torch.Tensor,
        *,
        activation: str = "silu",
        consume_base_pdl: bool = False,
    ) -> None:
        act_delta_contiguous(
            gateup_out,
            gate_up_delta,
            act_out,
            activation_lora_input,
            row_state.pair_to_row,
            topk_ids,
            self.num_local_experts,
            gate_first=self.contract.gate_first,
            interleaved=self.contract.interleaved,
            activation=activation,
            consume_base_pdl=consume_base_pdl,
        )

    def fused_act(
        self,
        row_state: ContiguousRowState,
        *,
        activation: str,
        base_gateup: torch.Tensor,
        act_rows: torch.Tensor,
        act_pairs: torch.Tensor | None,
        routing: RouteView,
        config: Mapping[str, int],
        bridge_gateup: torch.Tensor | None = None,
        b_gate_up: torch.Tensor | None = None,
        bridge_top_k: int = 1,
        consume_base_pdl: bool = False,
    ) -> None:
        fused_b_act_contiguous(
            activation=activation,
            base_gateup=base_gateup,
            act_compact=act_rows,
            act_pairs=act_pairs,
            pair_to_row=row_state.pair_to_row,
            routing=routing,
            num_local_experts=self.num_local_experts,
            gate_first=self.contract.gate_first,
            interleaved=self.contract.interleaved,
            config=config,
            bridge_gateup=bridge_gateup,
            b_gate_up=b_gate_up,
            bridge_top_k=bridge_top_k,
            consume_base_pdl=consume_base_pdl,
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
