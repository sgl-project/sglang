"""BF16 MoE provider backed by the DeepGEMM masked grouped GEMM.

Calls ``moe_ep_deepgemm_preprocess`` (S1), the raw
``grouped_gemm_nt_bf16_masked`` primitive (S2/S4), the LoRA-aware S3 activation
join, and the deterministic ``post_reorder_deepgemm`` finalize (S5) directly —
no stock ``MoeRunner`` core participates.
"""

from __future__ import annotations

import msgspec
import torch

from sglang.srt.lora.sgl_lora.base_gemm_provider.base import (
    MoeBaseProvider,
    MoeBaseProviderContract,
)
from sglang.srt.lora.sgl_lora.quant_info import SglLoraBf16QuantInfo


class DeepGemmMaskedWorkspace(msgspec.Struct, kw_only=True):
    """Per-forward state of the masked DeepGEMM row domain.

    Rows are ``[E_local, m_max, ·]`` with
    ``src2dst[t * top_k + k] = expert * m_max + offset``; validity is carried by
    ``topk_ids >= 0`` for the activation join and final reordering. This layout
    is specific to this provider — another provider returns its own type.
    """

    hidden_permuted: torch.Tensor  # [E_local, m_max, hidden]
    masked_m: torch.Tensor  # [E_local] int32
    expected_m: int
    src2dst: torch.Tensor  # [num_tokens * top_k] int32
    m_max: int


class DeepGemmBf16Provider(MoeBaseProvider):
    contract = MoeBaseProviderContract(
        key="deepgemm_bf16",
        gate_first=True,
        interleaved=False,
        gate_up_output_dtype=torch.bfloat16,
        lora_delta_dtype=torch.bfloat16,
        lora_activation_dtype=torch.bfloat16,
        supported_output_dtypes=(torch.bfloat16, torch.float32),
    )

    def __init__(self, quant_info: SglLoraBf16QuantInfo):
        self.quant_info = quant_info

        # Bind callees once: this instance is constructed at LoRA-attach time
        # and lives for the layer's lifetime, so no per-forward imports.
        from sglang.kernels.ops.moe.ep_moe_kernels import (
            moe_ep_deepgemm_preprocess,
            post_reorder_deepgemm,
        )
        from sglang.srt.layers import deep_gemm_wrapper
        from sglang.srt.lora.sgl_lora.base_gemm_provider.masked_activation import (
            silu_mul_delta_masked,
        )

        self._preprocess = moe_ep_deepgemm_preprocess
        self._post_reorder = post_reorder_deepgemm
        self._act_kernel = silu_mul_delta_masked
        self._grouped_gemm_bf16_masked = deep_gemm_wrapper.grouped_gemm_nt_bf16_masked

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
    ) -> DeepGemmMaskedWorkspace:
        masked_m, expected_m, src2dst, hidden_permuted, _scale = self._preprocess(
            topk_ids,
            self.quant_info.num_local_experts,
            hidden_states,
            top_k,
            None,
            output_dtype=torch.bfloat16,
        )
        return DeepGemmMaskedWorkspace(
            hidden_permuted=hidden_permuted,
            masked_m=masked_m,
            expected_m=expected_m,
            src2dst=src2dst,
            m_max=hidden_permuted.shape[1],
        )

    def gateup(self, ws: DeepGemmMaskedWorkspace, out: torch.Tensor) -> None:
        self._grouped_gemm_bf16_masked(
            ws.hidden_permuted,
            self.quant_info.w13_weight,
            out,
            ws.masked_m,
            ws.expected_m,
        )

    def release_prepared_inputs(self, ws: DeepGemmMaskedWorkspace) -> None:
        # The permuted hidden rows are dead after the gate/up GEMM; free them
        # before the S3/S4 buffers are allocated.
        from sglang.srt.utils import dispose_tensor

        dispose_tensor(ws.hidden_permuted)

    def act_with_delta(
        self,
        ws: DeepGemmMaskedWorkspace,
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

    def down(
        self, ws: DeepGemmMaskedWorkspace, act_out: torch.Tensor, out: torch.Tensor
    ) -> None:
        self._grouped_gemm_bf16_masked(
            act_out,
            self.quant_info.w2_weight,
            out,
            ws.masked_m,
            ws.expected_m,
        )

    def finalize(
        self,
        ws: DeepGemmMaskedWorkspace,
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

    def gateup_out_shape(self, ws: DeepGemmMaskedWorkspace) -> tuple[int, ...]:
        return (
            self.quant_info.num_local_experts,
            ws.m_max,
            2 * self.quant_info.intermediate_size,
        )

    def act_out_shape(self, ws: DeepGemmMaskedWorkspace) -> tuple[int, ...]:
        return (
            self.quant_info.num_local_experts,
            ws.m_max,
            self.quant_info.intermediate_size,
        )

    def down_out_shape(self, ws: DeepGemmMaskedWorkspace) -> tuple[int, ...]:
        return (
            self.quant_info.num_local_experts,
            ws.m_max,
            self.quant_info.hidden_size,
        )
