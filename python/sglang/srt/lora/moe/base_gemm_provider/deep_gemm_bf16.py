"""BF16 MoE providers backed by the DeepGEMM grouped GEMMs.

The row domains (S1 preprocess, S3 activation join, S5 finalize, geometry,
workspace) live in :mod:`masked_row_domain` and :mod:`contiguous_row_domain`;
the classes here supply only the GEMM engine — the raw
``grouped_gemm_nt_bf16_masked`` / ``grouped_gemm_nt_bf16_contig`` primitives
for S2/S4. No stock ``MoeRunner`` core participates.
"""

from __future__ import annotations

import torch

from sglang.srt.lora.moe.base_gemm_provider.base import MoeBaseProviderContract
from sglang.srt.lora.moe.base_gemm_provider.contiguous_row_domain import (
    ContiguousRowDomainProvider,
    ContiguousRowWorkspace,
)
from sglang.srt.lora.moe.base_gemm_provider.masked_row_domain import (
    MaskedRowDomainProvider,
    MaskedRowWorkspace,
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
        supported_output_dtypes=(torch.bfloat16, torch.float32),
    )

    def __init__(self, quant_info: MoeLoraBf16QuantInfo):
        super().__init__(quant_info)
        self._require_supported_geometry(quant_info)
        from sglang.srt.layers import deep_gemm_wrapper

        self._grouped_gemm_bf16_masked = deep_gemm_wrapper.grouped_gemm_nt_bf16_masked

        from sglang.srt.lora.moe.base_gemm_provider.gemm_config_store import (
            load_config_table,
        )

        # ``expected_m`` is DeepGEMM's only config input (it drives the
        # internal get_best_config choice); a swept bucket table may override
        # the hint per M-bucket. ``masked_m`` still bounds the actual work,
        # so the override is purely a performance knob. None = passthrough.
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
        """Reject contraction dimensions unsupported by DeepGEMM on SM90.

        Hopper's BF16 kernel requires ``K % 64 == 0``. Gate/up contracts over
        ``hidden_size`` and down contracts over ``intermediate_size``, so both
        dimensions must qualify. The SM100 implementation has no such limit.
        """
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
        ws: MaskedRowWorkspace,
        out: torch.Tensor,
        *,
        produce_pdl: bool = False,
    ) -> None:
        if produce_pdl:
            raise NotImplementedError(
                "DeepGEMM does not expose a plan-local GEMM1 producer twin"
            )
        self._grouped_gemm_bf16_masked(
            ws.hidden_permuted,
            self.quant_info.w13_weight,
            out,
            ws.masked_m,
            self._expected_m_hint(ws.expected_m),
        )

    def down(
        self,
        ws: MaskedRowWorkspace,
        act_out: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        self._grouped_gemm_bf16_masked(
            act_out,
            self.quant_info.w2_weight,
            out,
            ws.masked_m,
            self._expected_m_hint(ws.expected_m),
        )


class DeepGemmBf16ContiguousProvider(ContiguousRowDomainProvider):
    """Route-major twin of :class:`DeepGemmBf16Provider`.

    Same resident ``[E, N, K]`` BF16 weights, same contract semantics; only
    the S2/S4 primitive changes to ``m_grouped_bf16_gemm_nt_contiguous``
    (through the ``grouped_gemm_nt_bf16_contig`` wrapper, the exact call
    convention of the upstream EP prefill path), driven by the domain's
    ``grouped_layout`` row-to-expert tensor.  The wrapper's defaults keep
    DeepGEMM's ``ensure_zero_padding=True``, so ``-1``-labeled ceiling rows
    are skipped work with zeroed output.
    """

    contract = MoeBaseProviderContract(
        key="deepgemm_bf16_contiguous",
        gate_first=True,
        interleaved=False,
        gate_up_output_dtype=torch.bfloat16,
        lora_delta_dtype=torch.bfloat16,
        lora_activation_dtype=torch.bfloat16,
        supported_output_dtypes=(torch.bfloat16, torch.float32),
    )

    def __init__(self, quant_info: MoeLoraBf16QuantInfo):
        # The contiguous m-block alignment is the engine's own contract
        # (typically 128); read it from DeepGEMM rather than hardcoding so a
        # kernel-side change cannot silently break segment geometry.  It is
        # the m-granule the contiguous kernel schedules and zero-pads by, so
        # no config-level override exists.
        import deep_gemm

        super().__init__(
            quant_info,
            m_alignment=int(deep_gemm.get_m_alignment_for_contiguous_layout()),
        )
        # The SM90 K % 64 constraint binds the BF16 kernel family, masked and
        # contiguous alike; SM100 has no such limit.
        DeepGemmBf16Provider._require_supported_geometry(quant_info)
        from sglang.srt.layers import deep_gemm_wrapper

        self._grouped_gemm_bf16_contig = deep_gemm_wrapper.grouped_gemm_nt_bf16_contig

    def gateup(
        self,
        ws: ContiguousRowWorkspace,
        out: torch.Tensor,
        *,
        produce_pdl: bool = False,
    ) -> None:
        if produce_pdl:
            raise NotImplementedError(
                "DeepGEMM does not expose a plan-local GEMM1 producer twin"
            )
        self._grouped_gemm_bf16_contig(
            ws.hidden_compact, self.quant_info.w13_weight, out, ws.grouped_layout
        )

    def down(
        self,
        ws: ContiguousRowWorkspace,
        act_out: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        self._grouped_gemm_bf16_contig(
            act_out, self.quant_info.w2_weight, out, ws.grouped_layout
        )
