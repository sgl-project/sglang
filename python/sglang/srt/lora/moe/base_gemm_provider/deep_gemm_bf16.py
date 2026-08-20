from __future__ import annotations

import torch

from sglang.srt.lora.moe.base_gemm_provider.base import MoeBaseProviderContract
from sglang.srt.lora.moe.base_gemm_provider.contiguous_row_domain import (
    ContiguousRowDomainProvider,
    ContiguousRowState,
)
from sglang.srt.lora.moe.base_gemm_provider.masked_row_domain import (
    MaskedRowDomainProvider,
    MaskedRowState,
)
from sglang.srt.lora.moe.quant_info import MoeLoraBf16QuantInfo


class DeepGemmBf16Provider(MaskedRowDomainProvider):
    contract = MoeBaseProviderContract(
        key="deepgemm_bf16",
        gate_first=True,
        interleaved=False,
        gate_up_output_dtype=torch.bfloat16,
        lora_delta_dtype=torch.bfloat16,
        lora_activation_dtype=torch.bfloat16,
    )

    def __init__(self, quant_info: MoeLoraBf16QuantInfo):
        super().__init__(quant_info)
        self._require_supported_geometry(quant_info)
        from sglang.srt.layers import deep_gemm_wrapper

        self._grouped_gemm_bf16_masked = deep_gemm_wrapper.grouped_gemm_nt_bf16_masked

        from sglang.srt.lora.moe.base_gemm_provider.gemm_config_store import (
            load_config_table,
        )

        # ``expected_m`` is the only config input DeepGEMM reads, and it picks
        # the internal config. ``masked_m`` still bounds the real work. A
        # bucket table that overrides the hint therefore changes speed only.
        self._config_table = load_config_table(
            self.contract.key,
            num_local_experts=quant_info.num_local_experts,
            n_gemm1=self.gate_up_slices * quant_info.intermediate_size,
            n_gemm2=quant_info.hidden_size,
            k=quant_info.hidden_size,
        )

    def _expected_m_hint(self, expected_m: int) -> int:
        if self._config_table is None:
            return expected_m
        return self._config_table.pick(expected_m).get("expected_m", expected_m)

    @staticmethod
    def _require_supported_geometry(quant_info: MoeLoraBf16QuantInfo) -> None:
        major, _minor = torch.cuda.get_device_capability(quant_info.w2_weight.device)
        if major >= 10:
            return
        offenders = {
            name: value
            for name, value in (
                ("hidden_size", quant_info.hidden_size),
                ("intermediate_size", quant_info.intermediate_size),
            )
            if value % 64 != 0
        }
        if offenders:
            detail = ", ".join(
                f"{name}={value}" for name, value in sorted(offenders.items())
            )
            raise ValueError(
                f"deepgemm_bf16 on SM{major}x requires every GEMM contraction "
                f"dimension to be a multiple of 64, but {detail}. gate/up "
                "contracts over hidden_size and down over intermediate_size, "
                "so both must qualify; SM100 has no such constraint"
            )

    def gateup(
        self,
        row_state: MaskedRowState,
        out: torch.Tensor,
    ) -> None:
        self._grouped_gemm_bf16_masked(
            row_state.hidden_permuted,
            self.quant_info.w13_weight,
            out,
            row_state.masked_m,
            self._expected_m_hint(row_state.expected_m),
        )

    def down(
        self,
        row_state: MaskedRowState,
        act_out: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        self._grouped_gemm_bf16_masked(
            act_out,
            self.quant_info.w2_weight,
            out,
            row_state.masked_m,
            self._expected_m_hint(row_state.expected_m),
        )


class DeepGemmBf16ContiguousProvider(ContiguousRowDomainProvider):
    """Route-major variant of :class:`DeepGemmBf16Provider`.

    The weights and the contract do not change. This provider calls
    ``m_grouped_bf16_gemm_nt_contiguous`` instead, and passes the domain's
    ``grouped_layout``. The wrapper keeps ``ensure_zero_padding=True``. The
    kernel therefore skips a row with ``grouped_layout`` entry ``-1``, and
    writes zeros there.
    """

    contract = MoeBaseProviderContract(
        key="deepgemm_bf16_contiguous",
        gate_first=True,
        interleaved=False,
        gate_up_output_dtype=torch.bfloat16,
        lora_delta_dtype=torch.bfloat16,
        lora_activation_dtype=torch.bfloat16,
    )

    def __init__(self, quant_info: MoeLoraBf16QuantInfo):
        # The contiguous kernel schedules and zero-pads by its own
        # m-alignment. Read that value from DeepGEMM. A hardcoded value breaks
        # the segment geometry after a change inside the kernel.
        import deep_gemm

        super().__init__(
            quant_info,
            m_alignment=int(deep_gemm.get_m_alignment_for_contiguous_layout()),
        )
        DeepGemmBf16Provider._require_supported_geometry(quant_info)
        from sglang.srt.layers import deep_gemm_wrapper

        self._grouped_gemm_bf16_contig = deep_gemm_wrapper.grouped_gemm_nt_bf16_contig

    def gateup(
        self,
        row_state: ContiguousRowState,
        out: torch.Tensor,
    ) -> None:
        self._grouped_gemm_bf16_contig(
            row_state.hidden_compact,
            self.quant_info.w13_weight,
            out,
            row_state.grouped_layout,
        )

    def down(
        self,
        row_state: ContiguousRowState,
        act_out: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        self._grouped_gemm_bf16_contig(
            act_out, self.quant_info.w2_weight, out, row_state.grouped_layout
        )
