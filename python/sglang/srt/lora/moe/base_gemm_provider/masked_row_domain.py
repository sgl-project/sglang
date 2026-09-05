"""Expert slabs [E_local, m_max, ...] with valid counts and pair mappings."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import msgspec
import torch

from sglang.srt.lora.moe.base_gemm_provider.base import (
    MoeBaseProvider,
    admit_bf16_weights,
    expected_rows_per_expert,
    prepare_buffer,
)
from sglang.srt.lora.moe.quant_info import MoeLoraBf16QuantInfo

if TYPE_CHECKING:
    from sglang.srt.lora.moe.route_view import RouteView
    from sglang.srt.lora.moe.workspace import MoeLoraWorkspace


def masked_m_max(num_tokens: int) -> int:
    """Use the same slab bound as the upstream masked preprocess."""
    return (num_tokens // 256 + 1) * 256


class MaskedRowState(msgspec.Struct, kw_only=True):
    """``pair_to_row[t * top_k + k]`` is ``expert * m_max + offset``; a pair is
    valid only when ``topk_ids[t, k] >= 0``.
    """

    hidden_permuted: torch.Tensor  # [E_local, m_max, hidden], the GEMM input dtype
    masked_m: torch.Tensor  # [E_local] int32
    expected_m: int
    pair_to_row: torch.Tensor  # [num_tokens * top_k] int32
    m_max: int
    retained_inputs: bool


class MaskedRowDomainProvider(MoeBaseProvider):
    def __init__(self, quant_info: MoeLoraBf16QuantInfo):
        self.quant_info = quant_info
        self._gate_up_slices = admit_bf16_weights(quant_info)

        from sglang.srt.lora.moe.kernels.activation_delta import (
            act_delta_masked,
        )
        from sglang.srt.lora.moe.kernels.dispatch_masked import (
            dispatch_fill_masked_bf16,
        )

        self._preprocess = dispatch_fill_masked_bf16
        self._act_kernel = act_delta_masked

        from sglang.srt.lora.moe.kernels.fused_act import (
            fused_b_act_masked,
        )

        self._fused_act = fused_b_act_masked

    def prepare(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        workspace: MoeLoraWorkspace | None = None,
    ) -> MaskedRowState:
        num_experts = self.quant_info.num_local_experts
        num_pairs = topk_ids.numel()
        m_max = masked_m_max(hidden_states.size(0))
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
        hidden_permuted = prepare_buffer(
            workspace,
            "masked:hidden_permuted",
            (num_experts, m_max, hidden_states.size(1)),
            dtype=torch.bfloat16,
            device=device,
        )
        self._preprocess(
            hidden_states,
            topk_ids,
            top_k,
            masked_m_out=masked_m,
            pair_to_row_out=pair_to_row,
            rows_out=hidden_permuted,
        )
        return MaskedRowState(
            hidden_permuted=hidden_permuted,
            masked_m=masked_m,
            expected_m=expected_rows_per_expert(num_pairs, num_experts),
            pair_to_row=pair_to_row,
            m_max=m_max,
            retained_inputs=workspace is not None,
        )

    def release_prepared_inputs(self, row_state: MaskedRowState) -> None:
        # Workspace buffers retain their addresses for graph replay.
        if row_state.retained_inputs:
            return
        from sglang.srt.utils import dispose_tensor

        dispose_tensor(row_state.hidden_permuted)

    def act_with_delta(
        self,
        row_state: MaskedRowState,
        gateup_out: torch.Tensor,
        gate_up_delta: torch.Tensor | None,
        topk_ids: torch.Tensor,
        act_out: torch.Tensor,
        activation_lora_input: torch.Tensor,
        *,
        activation: str = "silu",
        consume_base_pdl: bool = False,
    ) -> None:
        self._act_kernel(
            gateup_out,
            gate_up_delta,
            act_out,
            activation_lora_input,
            row_state.pair_to_row,
            topk_ids,
            gate_first=self.contract.gate_first,
            interleaved=self.contract.interleaved,
            activation=activation,
            consume_base_pdl=consume_base_pdl,
        )

    def fused_act(
        self,
        row_state: MaskedRowState,
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
        self._fused_act(
            activation=activation,
            base_gateup=base_gateup,
            act_masked=act_rows,
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
