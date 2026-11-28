"""CuTeDSL FP8 GEMMs with FP32 block or rowwise scales."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch

from sglang.kernels.ops.lora.moe.dispatch_contiguous import (
    dispatch_fill_rows_contiguous_fp8,
)
from sglang.kernels.ops.lora.moe.dispatch_masked import dispatch_fill_masked_fp8
from sglang.kernels.ops.lora.moe.fp8_quant import (
    quantize_pair_rows_fp8,
)
from sglang.kernels.ops.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8,
)
from sglang.srt.lora.moe.base_gemm_provider.base import (
    MoeBaseProviderContract,
    RowBuffers,
    prepare_buffer,
)
from sglang.srt.lora.moe.base_gemm_provider.contiguous_row_domain import (
    ContiguousRowDomainProvider,
    ContiguousRowState,
)
from sglang.srt.lora.moe.base_gemm_provider.cutedsl_common import CuteDslTileMixin
from sglang.srt.lora.moe.base_gemm_provider.masked_row_domain import (
    MaskedRowDomainProvider,
    MaskedRowState,
)
from sglang.srt.lora.moe.quant_info import (
    MoeLoraFp8QuantInfo,
)

if TYPE_CHECKING:
    from sglang.srt.lora.workspace import LoraWorkspace

_QUANT_GROUP = 128


class _CuteDslFp8Mixin(CuteDslTileMixin):
    """FP8 rows, FP32 scales per 128 columns (per row in the per-channel subclass)."""

    _DTYPE_TAG = "fp8"
    WIDE_EXPECTED_M_THRESHOLD = 32
    _PER_TOKEN = False
    # The contiguous fill finishes the pair map on the way.
    _fill_finalizes_map = True

    @staticmethod
    def _prepare_input_rows(workspace, shape, device):
        """FP8 gate/up rows at the front of the ``base:down`` slab: the fill writes
        them, gate/up reads them last, then down overwrites them on the same stream."""
        if workspace is None:
            return torch.empty(shape, dtype=torch.float8_e4m3fn, device=device)
        output = workspace.tensor(
            "base:down", shape, dtype=torch.bfloat16, device=device
        )
        return output.view(torch.float8_e4m3fn).flatten()[: output.numel()].view(shape)

    def _bind_weights(self, quant_info: MoeLoraFp8QuantInfo) -> None:
        self._sf_w13 = quant_info.w13_scale.contiguous()
        self._sf_w2 = quant_info.w2_scale.contiguous()

    def _row_scale_shape(self, lead: tuple, width: int) -> tuple:
        return lead if self._PER_TOKEN else (*lead, width // _QUANT_GROUP)

    def _prepare_dummy(self, stage: str, config):
        weight = self._stage_weight(stage)
        device = weight.device
        _, n, k = weight.shape
        rows, scales, groups = self._dummy_layout(device)
        return self._prepare_gemm(
            torch.zeros((*rows, k), dtype=torch.float8_e4m3fn, device=device),
            weight,
            torch.zeros(
                self._row_scale_shape(scales, k), dtype=torch.float32, device=device
            ),
            self._stage_scale(stage),
            torch.empty((*rows, n), dtype=torch.bfloat16, device=device),
            groups,
            config=config,
        )

    def _input_buffers(
        self,
        workspace: LoraWorkspace | None,
        prefix: str,
        lead: tuple[int, ...],
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, RowBuffers]:
        """The fill writes input scales. Activation or down writes the down input."""
        k, inter = hidden_states.size(1), self.quant_info.intermediate_size
        device = hidden_states.device

        def _buf(name, shape, dtype):
            return prepare_buffer(
                workspace, f"{prefix}:{name}", shape, dtype=dtype, device=device
            )

        rows = self._prepare_input_rows(workspace, (*lead, k), device)
        return rows, dict(
            sf_rows=_buf("sf_rows", self._row_scale_shape(lead, k), torch.float32),
            act_fp8=_buf("act_fp8", (*lead, inter), torch.float8_e4m3fn),
            act_scale=_buf(
                "act_scale", self._row_scale_shape(lead, inter), torch.float32
            ),
        )

    def _fill_masked(
        self, hidden_states, topk_ids, top_k, masked_m, pair_to_row, rows, buffers
    ) -> None:
        dispatch_fill_masked_fp8(
            hidden_states,
            topk_ids,
            top_k,
            masked_m_out=masked_m,
            pair_to_row_out=pair_to_row,
            rows_fp8_out=rows,
            scale_out=buffers["sf_rows"],
            per_token=self._PER_TOKEN,
        )

    def _fill_contiguous(
        self, hidden_states, topk_ids, pair_to_row, seg_offsets, rows, buffers
    ) -> None:
        dispatch_fill_rows_contiguous_fp8(
            hidden_states,
            topk_ids,
            pair_to_row,
            rows_fp8_out=rows,
            scale_out=buffers["sf_rows"],
            per_token=self._PER_TOKEN,
            seg_offsets=seg_offsets if self._fill_finalizes_map else None,
        )

    def _act_group(self) -> int:
        return self.quant_info.intermediate_size if self._PER_TOKEN else _QUANT_GROUP

    def _act_quant_target(
        self, row_state, *, fused_config: Mapping[str, int] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, int] | None:
        if fused_config is not None:
            return None
        return row_state.act_fp8, row_state.act_scale, self._act_group()

    def _launch(
        self,
        stage: str,
        a_fp8: torch.Tensor,
        sf_tokens: torch.Tensor,
        c: torch.Tensor,
        row_state,
        schedule: torch.Tensor,
        tiles: torch.Tensor,
    ) -> None:
        call = self._compiled[row_state.token_width][stage]
        dyn = self._as_dynamic_cute_tensor
        a_rows = self._rows_arg(a_fp8)
        scale_tail = () if self._PER_TOKEN else (-1,)
        scale_dim = 1 + len(scale_tail)
        call.compiled_fn(
            dyn(a_rows, leading_dim=2),
            call.b_arg,
            dyn(sf_tokens.view(*a_rows.shape[:2], *scale_tail), leading_dim=scale_dim),
            dyn(call.sf_weights, leading_dim=scale_dim),
            dyn(self._rows_arg(c), leading_dim=2),
            dyn(self._group_arg(row_state), leading_dim=0),
            dyn(schedule, leading_dim=0),
            dyn(tiles, leading_dim=0),
            self._stream(a_fp8.device),
        )

    def gateup(self, row_state, out: torch.Tensor) -> None:
        self._launch(
            "gemm1",
            self._input_rows(row_state),
            row_state.sf_rows,
            out,
            row_state,
            row_state.gemm1_schedule,
            row_state.gemm1_tiles,
        )

    def _down_operands(
        self, row_state, act_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fallback quantization leaves stale input rows only for ignored outputs."""
        if not row_state.act_quant_ready:
            quantize_pair_rows_fp8(
                act_out,
                q_out=row_state.act_fp8,
                scale_out=row_state.act_scale,
                pair_to_row=row_state.pair_to_row,
                group=self._act_group(),
            )
        return row_state.act_fp8, row_state.act_scale

    def down(self, row_state, act_out: torch.Tensor, out: torch.Tensor) -> None:
        act_fp8, sf_act = self._down_operands(row_state, act_out)
        self._launch(
            "gemm2",
            act_fp8,
            sf_act,
            out,
            row_state,
            row_state.gemm2_schedule,
            row_state.gemm2_tiles,
        )


class CuteDslFp8MaskedRowState(MaskedRowState, kw_only=True):
    token_width: int
    sf_rows: torch.Tensor  # fp32 [E, m_max, hidden // 128], or [E, m_max] per token
    gemm1_schedule: torch.Tensor
    gemm1_tiles: torch.Tensor
    gemm2_schedule: torch.Tensor
    gemm2_tiles: torch.Tensor
    act_fp8: torch.Tensor
    act_scale: torch.Tensor
    act_quant_ready: bool = False


class CuteDslFp8MaskedProvider(_CuteDslFp8Mixin, MaskedRowDomainProvider):
    contract = MoeBaseProviderContract(
        key="cutedsl_fp8_masked", quant_info_cls=MoeLoraFp8QuantInfo
    )

    _ROW_MODE = "masked"
    _row_state_cls = CuteDslFp8MaskedRowState

    def __init__(self, quant_info: MoeLoraFp8QuantInfo):
        super().__init__(quant_info)
        from sglang.kernels.ops.lora.moe.cutedsl.api import (
            prepare_masked_fp8,
            prepare_masked_fp8_rowwise,
        )

        self._prepare_gemm = (
            prepare_masked_fp8_rowwise if self._PER_TOKEN else prepare_masked_fp8
        )
        self._init_tiles(quant_info)

    @staticmethod
    def _down_operands(
        row_state: CuteDslFp8MaskedRowState, act_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if row_state.act_quant_ready:
            return row_state.act_fp8, row_state.act_scale
        return sglang_per_token_group_quant_fp8(
            act_out,
            _QUANT_GROUP,
            masked_m=row_state.masked_m,
            output_q=row_state.act_fp8,
            output_s=row_state.act_scale,
        )


class CuteDslFp8ContiguousRowState(ContiguousRowState, kw_only=True):
    token_width: int
    sf_rows: torch.Tensor  # fp32 [m_pad_ceiling, hidden // 128], or [m_pad_ceiling]
    gemm1_schedule: torch.Tensor
    gemm1_tiles: torch.Tensor
    gemm2_schedule: torch.Tensor
    gemm2_tiles: torch.Tensor
    act_fp8: torch.Tensor  # [m_pad_ceiling, inter]
    act_scale: torch.Tensor  # [m_pad_ceiling, inter // 128], or [m_pad_ceiling]
    act_quant_ready: bool = False


class CuteDslFp8ContiguousProvider(_CuteDslFp8Mixin, ContiguousRowDomainProvider):
    contract = MoeBaseProviderContract(
        key="cutedsl_fp8_contiguous", quant_info_cls=MoeLoraFp8QuantInfo
    )

    _ROW_MODE = "contiguous"
    _row_state_cls = CuteDslFp8ContiguousRowState

    def __init__(self, quant_info: MoeLoraFp8QuantInfo):
        super().__init__(quant_info)
        from sglang.kernels.ops.lora.moe.cutedsl.api import (
            prepare_contiguous_fp8,
            prepare_contiguous_fp8_rowwise,
        )

        self._prepare_gemm = (
            prepare_contiguous_fp8_rowwise
            if self._PER_TOKEN
            else prepare_contiguous_fp8
        )
        self._init_tiles(quant_info)
