"""The masked-row-domain half of a BF16 MoE provider, GEMM-engine-agnostic.

S1 preprocess (`moe_ep_deepgemm_preprocess`), the S3 activation join
(`silu_mul_delta_masked`), and the S5 finalize (`post_reorder_deepgemm`) are
sglang Triton kernels over one physical layout — rows in ``[E_local, m_max,·]``
with ``masked_m`` counts and ``src2dst`` pair mapping — and carry nothing
specific to any GEMM engine. Both shipped providers (DeepGEMM and CuTeDSL)
consume exactly this layout and differ ONLY in how S2/S4 are executed, so the
domain lives here once and each engine subclass implements just ``gateup`` and
``down``.

Extracted per Yanbin's review (plan section 51): before this, the CuTeDSL
provider inherited from the DeepGEMM provider, which read as "CuTeDSL depends
on DeepGEMM" when the true relationship is "both specialize the masked row
domain".
"""

from __future__ import annotations

import msgspec
import torch

from sglang.srt.lora.sgl_lora.base_gemm_provider.base import MoeBaseProvider
from sglang.srt.lora.sgl_lora.quant_info import SglLoraBf16QuantInfo


class MaskedRowWorkspace(msgspec.Struct, kw_only=True):
    """Per-forward state of the masked row domain.

    Rows are ``[E_local, m_max, ·]`` with
    ``src2dst[t * top_k + k] = expert * m_max + offset``; validity is carried by
    ``topk_ids >= 0`` for the activation join and final reordering. Providers
    with extra per-forward state (e.g. tile schedules) subclass this.
    """

    hidden_permuted: torch.Tensor  # [E_local, m_max, hidden]
    masked_m: torch.Tensor  # [E_local] int32
    expected_m: int
    src2dst: torch.Tensor  # [num_tokens * top_k] int32
    m_max: int


class MaskedRowDomainProvider(MoeBaseProvider):
    """S1/S3/S5 plus geometry over the masked row domain; S2/S4 stay abstract."""

    def __init__(self, quant_info: SglLoraBf16QuantInfo):
        self.quant_info = quant_info

        # Bind callees once: this instance is constructed at LoRA-attach time
        # and lives for the layer's lifetime, so no per-forward imports.
        from sglang.kernels.ops.moe.ep_moe_kernels import (
            moe_ep_deepgemm_preprocess,
            post_reorder_deepgemm,
        )
        from sglang.srt.lora.sgl_lora.base_gemm_provider.masked_activation import (
            silu_mul_delta_masked,
        )

        self._preprocess = moe_ep_deepgemm_preprocess
        self._post_reorder = post_reorder_deepgemm
        self._act_kernel = silu_mul_delta_masked

    @property
    def num_local_experts(self) -> int:
        return self.quant_info.num_local_experts

    @property
    def intermediate_size(self) -> int:
        return self.quant_info.intermediate_size

    @property
    def hidden_size(self) -> int:
        return self.quant_info.hidden_size

    def prepare(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
    ) -> MaskedRowWorkspace:
        masked_m, expected_m, src2dst, hidden_permuted, _scale = self._preprocess(
            topk_ids,
            self.quant_info.num_local_experts,
            hidden_states,
            top_k,
            None,
            output_dtype=torch.bfloat16,
        )
        return MaskedRowWorkspace(
            hidden_permuted=hidden_permuted,
            masked_m=masked_m,
            expected_m=expected_m,
            src2dst=src2dst,
            m_max=hidden_permuted.shape[1],
        )

    def release_prepared_inputs(self, ws: MaskedRowWorkspace) -> None:
        # The permuted hidden rows are dead after the gate/up GEMM; free them
        # before the S3/S4 buffers are allocated.
        from sglang.srt.utils import dispose_tensor

        dispose_tensor(ws.hidden_permuted)

    def act_with_delta(
        self,
        ws: MaskedRowWorkspace,
        gateup_out: torch.Tensor,
        gate_up_delta: torch.Tensor | None,
        topk_ids: torch.Tensor,
        act_out: torch.Tensor,
        activation_lora_input: torch.Tensor,
    ) -> None:
        self._act_kernel(
            gateup_out,
            gate_up_delta,
            act_out,
            activation_lora_input,
            ws.src2dst,
            topk_ids,
            gate_first=self.contract.gate_first,
            interleaved=self.contract.interleaved,
        )

    def finalize(
        self,
        ws: MaskedRowWorkspace,
        down_out: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        routed_scaling_factor: float | None,
        output: torch.Tensor,
        *,
        pair_delta: torch.Tensor | None = None,
    ) -> None:
        num_tokens, hidden = output.shape
        self._post_reorder(
            down_out.view(-1, hidden),
            output,
            ws.src2dst,
            topk_ids,
            topk_weights,
            topk_ids.shape[1],
            num_tokens,
            hidden,
            routed_scaling_factor if routed_scaling_factor is not None else 1.0,
            pair_delta=pair_delta,
        )

    def gateup_out_shape(self, ws: MaskedRowWorkspace) -> tuple[int, ...]:
        return (
            self.quant_info.num_local_experts,
            ws.m_max,
            2 * self.quant_info.intermediate_size,
        )

    def act_out_shape(self, ws: MaskedRowWorkspace) -> tuple[int, ...]:
        return (
            self.quant_info.num_local_experts,
            ws.m_max,
            self.quant_info.intermediate_size,
        )

    def down_out_shape(self, ws: MaskedRowWorkspace) -> tuple[int, ...]:
        return (
            self.quant_info.num_local_experts,
            ws.m_max,
            self.quant_info.hidden_size,
        )
