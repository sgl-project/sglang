"""The masked row layout that both BF16 base GEMM providers share.

The rows are ``[E_local, m_max, ·]``. ``masked_m`` holds the row count of each
expert. ``src2dst`` maps a routed pair to its row. The preprocess, the
activation, and the finalize are Triton kernels over this layout. They do not
depend on the GEMM engine, so this class runs them for every provider. The
DeepGEMM subclass and the CuTeDSL subclass add only ``gateup`` and ``down``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import msgspec
import torch

from sglang.srt.lora.moe.base_gemm_provider.base import (
    MoeBaseProvider,
)
from sglang.srt.lora.moe.quant_info import MoeLoraBf16QuantInfo

if TYPE_CHECKING:
    from sglang.srt.lora.moe.route_view import RouteView
    from sglang.srt.lora.moe.workspace import MoeLoraWorkspace


class MaskedRowState(msgspec.Struct, kw_only=True):
    """``src2dst[t * top_k + k]`` is ``expert * m_max + offset``. A pair is
    valid only when ``topk_ids[t, k] >= 0``. A provider that needs more
    per-forward state, such as a tile schedule, subclasses this.
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
        if quant_info.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
        expected_w2 = (
            quant_info.num_local_experts,
            quant_info.hidden_size,
            quant_info.intermediate_size,
        )
        if quant_info.w2_weight.shape != expected_w2:
            raise ValueError(
                f"w2_weight must be {expected_w2}, got "
                f"{tuple(quant_info.w2_weight.shape)}"
            )
        if (
            quant_info.w13_weight.ndim != 3
            or quant_info.w13_weight.shape[0] != quant_info.num_local_experts
            or quant_info.w13_weight.shape[2] != quant_info.hidden_size
        ):
            raise ValueError(
                "w13_weight must be [num_local_experts, slices*intermediate, hidden]"
            )
        gateup_width = quant_info.w13_weight.shape[1]
        if gateup_width % quant_info.intermediate_size:
            raise ValueError(
                "w13 output width must be an integer multiple of intermediate_size"
            )
        self._gate_up_slices = gateup_width // quant_info.intermediate_size
        if self._gate_up_slices not in (1, 2):
            raise ValueError(
                "masked BF16 provider supports one non-gated slice or two "
                f"gated gate/up slices, got {self._gate_up_slices}"
            )

        # This constructor runs once, when the LoRA attaches. These imports run
        # here so that no forward pass runs an import.
        from sglang.srt.lora.moe.base_gemm_provider.masked_activation import (
            act_delta_masked,
        )
        from sglang.srt.lora.moe.base_gemm_provider.masked_dispatch import (
            fused_masked_preprocess,
        )

        self._preprocess = fused_masked_preprocess
        self._act_kernel = act_delta_masked

        from sglang.srt.lora.moe.base_gemm_provider.masked_fused_act import (
            run_masked_fused_act,
        )

        self._fused_act = run_masked_fused_act

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
                "base:masked_m",
                (self.quant_info.num_local_experts,),
                dtype=torch.int32,
                device=hidden_states.device,
            )
            src2dst_out = workspace.tensor(
                "base:src2dst",
                (topk_ids.numel(),),
                dtype=torch.int32,
                device=hidden_states.device,
            )
            hidden_permuted_out = workspace.tensor(
                "base:hidden_permuted",
                (
                    self.quant_info.num_local_experts,
                    m_max,
                    hidden_states.size(1),
                ),
                dtype=torch.bfloat16,
                device=hidden_states.device,
            )
        masked_m, expected_m, src2dst, hidden_permuted, _scale = self._preprocess(
            topk_ids,
            self.quant_info.num_local_experts,
            hidden_states,
            top_k,
            None,
            output_dtype=torch.bfloat16,
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
        # The gate/up GEMM is the last reader of the permuted rows. This frees
        # them before the next stage allocates. A workspace tensor must keep
        # its address for CUDA-graph replay, so this never frees one.
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
        # ``act_delta_masked`` checks the activation name against the registry,
        # so this method does not check it.
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

    def run_fused_act(
        self,
        row_state: MaskedRowState,
        family: str,
        *,
        activation: str,
        base_gateup: torch.Tensor,
        act_masked: torch.Tensor,
        act_pairs: torch.Tensor | None,
        routing: RouteView,
        config: Mapping[str, int],
        bridge_gateup: torch.Tensor | None = None,
        b_gate_up: torch.Tensor | None = None,
        bridge_top_k: int = 1,
        consume_base_pdl: bool = False,
    ) -> None:
        self._fused_act(
            family,
            activation=activation,
            base_gateup=base_gateup,
            act_masked=act_masked,
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
