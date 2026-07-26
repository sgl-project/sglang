"""BF16 MoE provider backed by the DeepGEMM masked grouped GEMM.

The masked row domain (S1 preprocess, S3 activation join, S5 finalize,
geometry, workspace) lives in :mod:`masked_row_domain`; this class supplies
only the GEMM engine: the raw ``grouped_gemm_nt_bf16_masked`` primitive for
S2/S4. No stock ``MoeRunner`` core participates.
"""

from __future__ import annotations

import torch

from sglang.srt.lora.sgl_lora.base_gemm_provider.base import MoeBaseProviderContract
from sglang.srt.lora.sgl_lora.base_gemm_provider.masked_row_domain import (
    MaskedRowDomainProvider,
    MaskedRowWorkspace,
)
from sglang.srt.lora.sgl_lora.quant_info import SglLoraBf16QuantInfo

# The workspace was named for this provider before the masked-row-domain
# extraction; importers of the old name keep working.
DeepGemmMaskedWorkspace = MaskedRowWorkspace


class DeepGemmBf16Provider(MaskedRowDomainProvider):
    contract = MoeBaseProviderContract(
        key="deepgemm_bf16",
        gate_first=True,
        interleaved=False,
        gate_up_output_dtype=torch.bfloat16,
        lora_delta_dtype=torch.bfloat16,
        lora_activation_dtype=torch.bfloat16,
        supported_output_dtypes=(torch.bfloat16, torch.float32),
    )

    def __init__(
        self,
        quant_info: SglLoraBf16QuantInfo,
        *,
        expected_m_hint: int | None = None,
    ):
        """``expected_m_hint`` is a LAB hook: the provider bench sweeps it to
        reproduce the expected_m retune experiment (plan section 53.3) from a
        committed producer. Production callers never pass it — the kernel's
        own heuristic reads the workspace's measured ``expected_m``.
        """
        super().__init__(quant_info)
        from sglang.srt.layers import deep_gemm_wrapper

        self._grouped_gemm_bf16_masked = deep_gemm_wrapper.grouped_gemm_nt_bf16_masked
        self._expected_m_hint = expected_m_hint

    def _expected_m(self, ws: MaskedRowWorkspace) -> int:
        return (
            self._expected_m_hint
            if self._expected_m_hint is not None
            else ws.expected_m
        )

    def gateup(self, ws: MaskedRowWorkspace, out: torch.Tensor) -> None:
        self._grouped_gemm_bf16_masked(
            ws.hidden_permuted,
            self.quant_info.w13_weight,
            out,
            ws.masked_m,
            self._expected_m(ws),
        )

    def down(
        self, ws: MaskedRowWorkspace, act_out: torch.Tensor, out: torch.Tensor
    ) -> None:
        self._grouped_gemm_bf16_masked(
            act_out,
            self.quant_info.w2_weight,
            out,
            ws.masked_m,
            self._expected_m(ws),
        )
