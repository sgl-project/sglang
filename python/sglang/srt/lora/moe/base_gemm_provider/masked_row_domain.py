"""The masked row domain: ``[E_local, m_max, ·]`` slabs bounded by
``masked_m``, with ``src2dst`` mapping each routed pair to its row. The
GEMM-engine subclasses add only ``gateup`` and ``down``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import msgspec
import torch

from sglang.srt.lora.moe.base_gemm_provider.base import (
    MoeBaseProvider,
    admit_bf16_weights,
)
from sglang.srt.lora.moe.quant_info import MoeLoraBf16QuantInfo

if TYPE_CHECKING:
    from sglang.srt.lora.moe.route_view import RouteView
    from sglang.srt.lora.moe.workspace import MoeLoraWorkspace


class MaskedRowState(msgspec.Struct, kw_only=True):
    """``src2dst[t * top_k + k]`` is ``expert * m_max + offset``; a pair is
    valid only when ``topk_ids[t, k] >= 0``.
    """

    hidden_permuted: torch.Tensor  # [E_local, m_max, hidden]
    masked_m: torch.Tensor  # [E_local] int32
    expected_m: int
    src2dst: torch.Tensor  # [num_tokens * top_k] int32
    m_max: int
    retained_inputs: bool


class MaskedRowDomainProvider(MoeBaseProvider):
    def __init__(self, quant_info: MoeLoraBf16QuantInfo):
        self.quant_info = quant_info
        self._gate_up_slices = admit_bf16_weights(quant_info)

        # Attach-time imports: no forward pass runs an import.
        from sglang.srt.lora.moe.kernels.activation_delta import (
            act_delta_masked,
        )
        from sglang.srt.lora.moe.kernels.dispatch import (
            dispatch_fill_masked,
        )

        self._preprocess = dispatch_fill_masked
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
        m_max = (hidden_states.size(0) // 256 + 1) * 256
        masked_m_out = None
        src2dst_out = None
        hidden_permuted_out = None
        if workspace is not None:
            masked_m_out = workspace.tensor(
                "masked:masked_m",
                (self.quant_info.num_local_experts,),
                dtype=torch.int32,
                device=hidden_states.device,
            )
            src2dst_out = workspace.tensor(
                "masked:src2dst",
                (topk_ids.numel(),),
                dtype=torch.int32,
                device=hidden_states.device,
            )
            hidden_permuted_out = workspace.tensor(
                "masked:hidden_permuted",
                (
                    self.quant_info.num_local_experts,
                    m_max,
                    hidden_states.size(1),
                ),
                dtype=torch.bfloat16,
                device=hidden_states.device,
            )
        masked_m, expected_m, src2dst, hidden_permuted = self._preprocess(
            topk_ids,
            self.quant_info.num_local_experts,
            hidden_states,
            top_k,
            masked_m_out=masked_m_out,
            src2dst_out=src2dst_out,
            gateup_input_out=hidden_permuted_out,
        )
        return MaskedRowState(
            hidden_permuted=hidden_permuted,
            masked_m=masked_m,
            expected_m=expected_m,
            src2dst=src2dst,
            m_max=hidden_permuted.shape[1],
            retained_inputs=workspace is not None,
        )

    def release_prepared_inputs(self, row_state: MaskedRowState) -> None:
        # A workspace tensor must keep its address for graph replay, so this
        # frees only an eagerly allocated buffer.
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
            row_state.src2dst,
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
            src2dst=row_state.src2dst,
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
