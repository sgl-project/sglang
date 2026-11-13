"""CuTeDSL GEMMs using checkpoint FP8 values and FP32 block scales.

Activations use group-128 quantization; GEMM outputs remain BF16 for LoRA.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.ops.lora.moe.dispatch_contiguous import (
    dispatch_fill_rows_contiguous_fp8,
)
from sglang.kernels.ops.lora.moe.dispatch_masked import dispatch_fill_masked_fp8
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
from sglang.srt.lora.moe.quant_info import MoeLoraFp8QuantInfo

if TYPE_CHECKING:
    from sglang.srt.lora.workspace import LoraWorkspace

QUANT_GROUP = 128


class CuteDslFp8Mixin(CuteDslTileMixin):
    """FP8 rows with FP32 scales, one per 128 columns of a row; the per-channel
    subclass keeps one per row."""

    _DTYPE_TAG = "fp8"
    WIDE_EXPECTED_M_THRESHOLD = 32
    _PER_TOKEN = False
    # Workspace names of the input rows' and the activation's scales.
    _SCALE_BUFFERS = ("sf_rows", "act_scale")
    # Whether the row state owns down-input buffers an activation can fill.
    _OWNS_DOWN_INPUT = True

    @staticmethod
    def _prepare_input_rows(workspace, shape, device):
        """The FP8 input rows of gate/up, packed at the front of the
        ``base:down`` slab. Producer: the dispatch fill in prepare(). Last
        consumer: the gate/up GEMM. The down GEMM overwrites the storage with
        its BF16 output, on the same stream and after gate/up in every plan."""
        if workspace is None:
            return torch.empty(shape, dtype=torch.float8_e4m3fn, device=device)
        output = workspace.tensor(
            "base:down", shape, dtype=torch.bfloat16, device=device
        )
        return output.view(torch.float8_e4m3fn).flatten()[: output.numel()].view(shape)

    def _bind_weights(self, quant_info: MoeLoraFp8QuantInfo) -> None:
        self._sf_w13 = quant_info.w13_scale.contiguous()
        self._sf_w2 = quant_info.w2_scale.contiguous()

    def _stage_scale(self, stage: str) -> torch.Tensor:
        return self._sf_w13 if stage == "gemm1" else self._sf_w2

    def _row_scale_shape(self, lead: tuple, width: int) -> tuple:
        return lead if self._PER_TOKEN else (*lead, width // QUANT_GROUP)

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
        """The row domains' hook for FP8 rows. ``rows`` is FP8; with a
        workspace it is the front of the ``base:down`` slab (see
        _prepare_input_rows). ``buffers``:

          sf_rows             the rows' FP32 scales, one per 128 columns or one
                              per row (``_PER_TOKEN``); the dispatch fill writes
                              them with the rows
          act_fp8, act_scale  the down GEMM's input and its scales, only when
                              the row state owns them (``_OWNS_DOWN_INPUT``);
                              the unfused activation writes them. When the
                              activation left ``act_quant_ready`` false, down()
                              writes them itself for one scale per row, and
                              quantizes into tensors of its own for
                              128-column groups
        """
        k, inter = hidden_states.size(1), self.quant_info.intermediate_size
        device = hidden_states.device
        sf_name, act_scale_name = self._SCALE_BUFFERS

        def _buf(name, shape, dtype):
            return prepare_buffer(
                workspace, f"{prefix}:{name}", shape, dtype=dtype, device=device
            )

        rows = self._prepare_input_rows(workspace, (*lead, k), device)
        buffers = dict(
            sf_rows=_buf(sf_name, self._row_scale_shape(lead, k), torch.float32)
        )
        if self._OWNS_DOWN_INPUT:
            buffers.update(
                act_fp8=_buf("act_fp8", (*lead, inter), torch.float8_e4m3fn),
                act_scale=_buf(
                    act_scale_name, self._row_scale_shape(lead, inter), torch.float32
                ),
            )
        return rows, buffers

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

    def _act_quant_target(self, row_state, fused, config):
        # Only the unfused activation writes FP8; one scale per row when the
        # group is the row width.
        if fused or not self._OWNS_DOWN_INPUT:
            return None
        group = self.quant_info.intermediate_size if self._PER_TOKEN else QUANT_GROUP
        return row_state.act_fp8, row_state.act_scale, group

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
        # Runtime wrapping must match the compiled argument's MLIR type.
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
    """MaskedRowState plus the buffers of CuteDslFp8Mixin._input_buffers() and
    the schedule fields (CuteDslTileMixin._schedules()). The activation
    wrappers assign ``act_quant_ready`` after every activation."""

    token_width: int
    sf_rows: torch.Tensor  # [E_local, m_max, hidden // 128] fp32 group scales
    gemm1_schedule: torch.Tensor
    gemm1_tiles: torch.Tensor
    gemm2_schedule: torch.Tensor
    gemm2_tiles: torch.Tensor
    # Only materialized activation fills these; fused-B activation stays BF16.
    act_fp8: torch.Tensor
    act_scale: torch.Tensor
    act_quant_ready: bool = False


class CuteDslFp8MaskedProvider(CuteDslFp8Mixin, MaskedRowDomainProvider):
    contract = MoeBaseProviderContract(
        key="cutedsl_fp8_masked",
        quant_info_cls=MoeLoraFp8QuantInfo,
        gate_first=True,
        interleaved=False,
        gate_up_output_dtype=torch.bfloat16,
        lora_delta_dtype=torch.bfloat16,
        lora_activation_dtype=torch.bfloat16,
    )

    _ROW_MODE = "masked"
    _row_state_cls = CuteDslFp8MaskedRowState

    def __init__(self, quant_info: MoeLoraFp8QuantInfo):
        super().__init__(quant_info)
        from sglang.kernels.ops.lora.moe.cutedsl.api import prepare_masked_fp8
        from sglang.kernels.ops.lora.moe.cutedsl.schedule_builder import (
            build_dual_stage_schedules_masked,
            dual_stage_schedule_capacities_masked,
        )

        self._prepare_gemm = prepare_masked_fp8
        self._build_schedules = build_dual_stage_schedules_masked
        self._schedule_capacities = dual_stage_schedule_capacities_masked
        self._init_tiles(quant_info)

    @staticmethod
    def _down_operands(
        row_state: CuteDslFp8MaskedRowState, act_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if row_state.act_quant_ready:
            return row_state.act_fp8, row_state.act_scale
        # Fused-B activation needs separate quantization.
        return sglang_per_token_group_quant_fp8(
            act_out, QUANT_GROUP, masked_m=row_state.masked_m
        )


class CuteDslFp8ContiguousRowState(ContiguousRowState, kw_only=True):
    """ContiguousRowState plus the input-row scales of
    CuteDslFp8Mixin._input_buffers() and the schedule fields
    (CuteDslTileMixin._schedules()); it owns no down-input buffers."""

    token_width: int
    sf_rows: torch.Tensor  # [m_pad_ceiling, hidden // 128] fp32 group scales
    gemm1_schedule: torch.Tensor
    gemm1_tiles: torch.Tensor
    gemm2_schedule: torch.Tensor
    gemm2_tiles: torch.Tensor


class CuteDslFp8ContiguousProvider(CuteDslFp8Mixin, ContiguousRowDomainProvider):
    contract = MoeBaseProviderContract(
        key="cutedsl_fp8_contiguous",
        quant_info_cls=MoeLoraFp8QuantInfo,
        gate_first=True,
        interleaved=False,
        gate_up_output_dtype=torch.bfloat16,
        lora_delta_dtype=torch.bfloat16,
        lora_activation_dtype=torch.bfloat16,
    )

    M_ALIGNMENT = 128

    _ROW_MODE = "contiguous"
    _row_state_cls = CuteDslFp8ContiguousRowState
    # The stock group quantizer returns the down input in tensors of its own.
    _OWNS_DOWN_INPUT = False

    def __init__(self, quant_info: MoeLoraFp8QuantInfo):
        super().__init__(quant_info, m_alignment=self.M_ALIGNMENT)
        from sglang.kernels.ops.lora.moe.cutedsl.api import prepare_contiguous_fp8
        from sglang.kernels.ops.lora.moe.cutedsl.schedule_builder import (
            build_dual_stage_schedules_contiguous,
            dual_stage_schedule_capacities_contiguous,
            validate_tile_geometry_contiguous,
        )

        self._prepare_gemm = prepare_contiguous_fp8
        self._validate_tile_geometry = validate_tile_geometry_contiguous
        self._build_schedules = build_dual_stage_schedules_contiguous
        self._schedule_capacities = dual_stage_schedule_capacities_contiguous
        self._init_tiles(quant_info)

    def _admit_tile_width(self, token_width: int) -> None:
        self._validate_tile_geometry(token_width, self._m_alignment)

    @staticmethod
    def _down_operands(
        row_state: CuteDslFp8ContiguousRowState, act_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return sglang_per_token_group_quant_fp8(act_out, QUANT_GROUP)
