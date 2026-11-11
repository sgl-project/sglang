"""CuTeDSL GEMMs for per-channel FP8 weights with per-token activation scales.

Weights keep their checkpoint values and one FP32 scale per output row;
tokens quantize to one scale per row during dispatch; GEMM outputs remain
BF16 for LoRA.
"""

from __future__ import annotations

import torch

from sglang.kernels.ops.lora.moe.fp8_quant import quantize_rows_per_token_fp8
from sglang.srt.lora.moe.base_gemm_provider.base import MoeBaseProviderContract
from sglang.srt.lora.moe.base_gemm_provider.contiguous_row_domain import (
    ContiguousRowDomainProvider,
    ContiguousRowState,
)
from sglang.srt.lora.moe.base_gemm_provider.cutedsl_fp8 import CuteDslFp8Mixin
from sglang.srt.lora.moe.base_gemm_provider.masked_row_domain import (
    MaskedRowDomainProvider,
    MaskedRowState,
)
from sglang.srt.lora.moe.quant_info import MoeLoraFp8ChannelQuantInfo

_CONTRACT = dict(
    quant_info_cls=MoeLoraFp8ChannelQuantInfo,
    gate_first=True,
    interleaved=False,
    gate_up_output_dtype=torch.bfloat16,
    lora_delta_dtype=torch.bfloat16,
    lora_activation_dtype=torch.bfloat16,
)


def _admit_tile_geometry(quant_info: MoeLoraFp8ChannelQuantInfo) -> None:
    """The kernel stages one weight-row scale per 128-row tile and tiles K by 128."""
    for name, value in (
        ("hidden_size", quant_info.hidden_size),
        ("intermediate_size", quant_info.intermediate_size),
    ):
        if value % 128:
            raise NotImplementedError(
                "per-channel FP8 MoE LoRA on CuTeDSL requires "
                f"{name} % 128 == 0, got {value}"
            )


class CuteDslFp8ChannelMixin(CuteDslFp8Mixin):
    """One scale per row, written by the dispatch fill and by the unfused
    activation: no separate quant pass unless the activation was fused."""

    _DTYPE_TAG = "fp8_channel"
    _PER_TOKEN = True
    _SCALE_BUFFERS = ("sf_tokens", "act_token_scale")
    # The per-token contiguous fill finishes the pair map on the way.
    _fill_finalizes_map = True

    def _bind_weights(self, quant_info: MoeLoraFp8ChannelQuantInfo) -> None:
        # [E, rows, 1] checkpoint scales -> [E, rows] for the kernel.
        self._sf_w13 = quant_info.w13_scale.flatten(1).contiguous()
        self._sf_w2 = quant_info.w2_scale.flatten(1).contiguous()


class CuteDslFp8ChannelMaskedRowState(MaskedRowState, kw_only=True):
    """MaskedRowState plus the buffers of CuteDslFp8Mixin._input_buffers() and
    the schedule fields (CuteDslTileMixin._schedules()). The activation
    wrappers assign ``act_quant_ready`` after every activation."""

    token_width: int
    sf_rows: torch.Tensor  # [E_local, m_max] fp32 token scales
    gemm1_schedule: torch.Tensor
    gemm1_tiles: torch.Tensor
    gemm2_schedule: torch.Tensor
    gemm2_tiles: torch.Tensor
    act_fp8: torch.Tensor  # [E_local, m_max, inter]
    act_scale: torch.Tensor  # [E_local, m_max]
    act_quant_ready: bool = False


class CuteDslFp8ChannelMaskedProvider(CuteDslFp8ChannelMixin, MaskedRowDomainProvider):
    contract = MoeBaseProviderContract(key="cutedsl_fp8_channel_masked", **_CONTRACT)

    _ROW_MODE = "masked"
    _row_state_cls = CuteDslFp8ChannelMaskedRowState

    def __init__(self, quant_info: MoeLoraFp8ChannelQuantInfo):
        super().__init__(quant_info)
        _admit_tile_geometry(quant_info)
        from sglang.kernels.ops.lora.moe.cutedsl.api import prepare_masked_fp8_rowwise
        from sglang.kernels.ops.lora.moe.cutedsl.schedule_builder import (
            build_dual_stage_schedules_masked,
            dual_stage_schedule_capacities_masked,
        )

        self._prepare_gemm = prepare_masked_fp8_rowwise
        self._build_schedules = build_dual_stage_schedules_masked
        self._schedule_capacities = dual_stage_schedule_capacities_masked
        self._init_tiles(quant_info)

    def _down_operands(
        self, row_state: CuteDslFp8ChannelMaskedRowState, act_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not row_state.act_quant_ready:
            # Fused-B activation stays BF16.
            quantize_rows_per_token_fp8(
                act_out,
                q_out=row_state.act_fp8,
                scale_out=row_state.act_scale,
                masked_m=row_state.masked_m,
            )
        return row_state.act_fp8, row_state.act_scale


class CuteDslFp8ChannelContiguousRowState(ContiguousRowState, kw_only=True):
    """ContiguousRowState plus the buffers of CuteDslFp8Mixin._input_buffers()
    and the schedule fields (CuteDslTileMixin._schedules()). The activation
    wrappers assign ``act_quant_ready`` after every activation."""

    token_width: int
    sf_rows: torch.Tensor  # [m_pad_ceiling] fp32 token scales
    gemm1_schedule: torch.Tensor
    gemm1_tiles: torch.Tensor
    gemm2_schedule: torch.Tensor
    gemm2_tiles: torch.Tensor
    act_fp8: torch.Tensor  # [m_pad_ceiling, inter]
    act_scale: torch.Tensor  # [m_pad_ceiling]
    act_quant_ready: bool = False


class CuteDslFp8ChannelContiguousProvider(
    CuteDslFp8ChannelMixin, ContiguousRowDomainProvider
):
    contract = MoeBaseProviderContract(
        key="cutedsl_fp8_channel_contiguous", **_CONTRACT
    )

    M_ALIGNMENT = 128

    _ROW_MODE = "contiguous"
    _row_state_cls = CuteDslFp8ChannelContiguousRowState

    def __init__(self, quant_info: MoeLoraFp8ChannelQuantInfo):
        super().__init__(quant_info, m_alignment=self.M_ALIGNMENT)
        _admit_tile_geometry(quant_info)
        from sglang.kernels.ops.lora.moe.cutedsl.api import (
            prepare_contiguous_fp8_rowwise,
        )
        from sglang.kernels.ops.lora.moe.cutedsl.schedule_builder import (
            build_dual_stage_schedules_contiguous,
            dual_stage_schedule_capacities_contiguous,
            validate_tile_geometry_contiguous,
        )

        self._prepare_gemm = prepare_contiguous_fp8_rowwise
        self._validate_tile_geometry = validate_tile_geometry_contiguous
        self._build_schedules = build_dual_stage_schedules_contiguous
        self._schedule_capacities = dual_stage_schedule_capacities_contiguous
        self._init_tiles(quant_info)

    def _admit_tile_width(self, token_width: int) -> None:
        self._validate_tile_geometry(token_width, self._m_alignment)

    def _down_operands(
        self, row_state: CuteDslFp8ChannelContiguousRowState, act_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not row_state.act_quant_ready:
            # Fused-B activation stays BF16; the segment total bounds the rows.
            quantize_rows_per_token_fp8(
                act_out,
                q_out=row_state.act_fp8,
                scale_out=row_state.act_scale,
                live_rows=row_state.seg_offsets[self.quant_info.num_local_experts :],
            )
        return row_state.act_fp8, row_state.act_scale
