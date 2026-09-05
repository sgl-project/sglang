"""CuTeDSL GEMMs using checkpoint FP8 values and FP32 block scales.

Activations use group-128 quantization; GEMM outputs remain BF16 for LoRA.
"""

from __future__ import annotations

import torch

from sglang.kernels.ops.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8,
)
from sglang.srt.lora.moe.base_gemm_provider.base import (
    MoeBaseProviderContract,
    expected_rows_per_expert,
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
    masked_m_max,
)
from sglang.srt.lora.moe.kernels.dispatch_contiguous import (
    dispatch_fill_rows_contiguous_fp8,
)
from sglang.srt.lora.moe.kernels.dispatch_masked import dispatch_fill_masked_fp8
from sglang.srt.lora.moe.quant_info import MoeLoraFp8QuantInfo

QUANT_GROUP = 128


class _CuteDslFp8Mixin(CuteDslTileMixin):
    _DTYPE_TAG = "fp8"
    WIDE_EXPECTED_M_THRESHOLD = 32

    def _bind_weights(self, quant_info: MoeLoraFp8QuantInfo) -> None:
        self._sf_w13 = quant_info.w13_scale.contiguous()
        self._sf_w2 = quant_info.w2_scale.contiguous()

    def _stage_scale(self, stage: str) -> torch.Tensor:
        return self._sf_w13 if stage == "gemm1" else self._sf_w2

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
        call.compiled_fn(
            dyn(self._rows_arg(a_fp8), leading_dim=2),
            call.b_arg,
            dyn(self._sf_arg(sf_tokens, a_fp8.shape), leading_dim=2),
            dyn(call.sf_weights, leading_dim=2),
            dyn(self._rows_arg(c), leading_dim=2),
            dyn(self._group_arg(row_state), leading_dim=0),
            dyn(schedule, leading_dim=0),
            dyn(tiles, leading_dim=0),
            self._stream(a_fp8.device),
        )

    @staticmethod
    def _sf_arg(sf: torch.Tensor, rows_shape: tuple[int, ...]) -> torch.Tensor:
        if len(rows_shape) == 3:
            return sf.view(rows_shape[0], rows_shape[1], -1)
        return sf.view(rows_shape[0], -1).unsqueeze(0)

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


class CuteDslFp8MaskedProvider(_CuteDslFp8Mixin, MaskedRowDomainProvider):
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

    def __init__(self, quant_info: MoeLoraFp8QuantInfo):
        super().__init__(quant_info)
        from sglang.srt.lora.moe.kernels.cutedsl.api import prepare_masked_fp8
        from sglang.srt.lora.moe.kernels.cutedsl.schedule_builder import (
            build_dual_stage_schedules_masked,
            dual_stage_schedule_capacities_masked,
        )

        self._prepare_masked_gemm = prepare_masked_fp8
        self._build_schedules = build_dual_stage_schedules_masked
        self._schedule_capacities = dual_stage_schedule_capacities_masked
        self._init_tiles(quant_info)

    def _prepare_dummy(self, stage: str, config):
        weight, weight_sf = self._stage_weight(stage), self._stage_scale(stage)
        device = weight.device
        experts, n, k = weight.shape
        dummy_a = torch.zeros(
            (experts, 256, k), dtype=torch.float8_e4m3fn, device=device
        )
        dummy_sf = torch.zeros(
            (experts, 256, k // QUANT_GROUP), dtype=torch.float32, device=device
        )
        dummy_c = torch.empty((experts, 256, n), dtype=torch.bfloat16, device=device)
        dummy_masked = torch.zeros(experts, dtype=torch.int32, device=device)
        return self._prepare_masked_gemm(
            dummy_a,
            weight,
            dummy_sf,
            weight_sf,
            dummy_c,
            dummy_masked,
            config=config,
        )

    @staticmethod
    def _rows_arg(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    @staticmethod
    def _input_rows(row_state: CuteDslFp8MaskedRowState) -> torch.Tensor:
        return row_state.hidden_permuted

    @staticmethod
    def _group_arg(row_state: CuteDslFp8MaskedRowState) -> torch.Tensor:
        return row_state.masked_m

    def prepare(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        workspace=None,
    ) -> CuteDslFp8MaskedRowState:
        experts = self.quant_info.num_local_experts
        k = hidden_states.size(1)
        inter = self.quant_info.intermediate_size
        m_max = masked_m_max(hidden_states.size(0))
        device = hidden_states.device

        def _buf(name, shape, dtype):
            return prepare_buffer(workspace, name, shape, dtype=dtype, device=device)

        masked_m = _buf("masked:masked_m", (experts,), torch.int32)
        pair_to_row = _buf("masked:pair_to_row", (topk_ids.numel(),), torch.int32)
        rows_fp8 = _buf("masked:rows_fp8", (experts, m_max, k), torch.float8_e4m3fn)
        sf_rows = _buf(
            "masked:sf_rows", (experts, m_max, k // QUANT_GROUP), torch.float32
        )
        act_fp8 = _buf("masked:act_fp8", (experts, m_max, inter), torch.float8_e4m3fn)
        act_scale = _buf(
            "masked:act_scale", (experts, m_max, inter // QUANT_GROUP), torch.float32
        )
        dispatch_fill_masked_fp8(
            hidden_states,
            topk_ids,
            top_k,
            masked_m_out=masked_m,
            pair_to_row_out=pair_to_row,
            rows_fp8_out=rows_fp8,
            scale_out=sf_rows,
        )
        expected_m = expected_rows_per_expert(topk_ids.numel(), experts)
        token_width = self._token_width_for(m_max, expected_m)
        geometry = dict(
            token_width=token_width,
            n_gemm1=self.gate_up_slices * inter,
            n_gemm2=self.quant_info.hidden_size,
            output_width=self.OUTPUT_WIDTH,
            cluster_shape_mn=self.CLUSTER_SHAPE_MN,
            use_2cta_instrs=self.USE_2CTA_INSTRS,
        )
        schedule_outputs = {}
        if workspace is not None:
            schedule_outputs = self._schedule_buffers(
                workspace,
                f"cutedsl_masked:tw{token_width}",
                self._schedule_capacities(num_experts=experts, m_max=m_max, **geometry),
                device,
            )
        schedule1, tiles1, schedule2, tiles2 = self._build_schedules(
            masked_m, m_max=m_max, **geometry, **schedule_outputs
        )
        return CuteDslFp8MaskedRowState(
            hidden_permuted=rows_fp8,
            masked_m=masked_m,
            expected_m=expected_m,
            pair_to_row=pair_to_row,
            m_max=m_max,
            retained_inputs=workspace is not None,
            token_width=token_width,
            sf_rows=sf_rows,
            gemm1_schedule=schedule1,
            gemm1_tiles=tiles1,
            gemm2_schedule=schedule2,
            gemm2_tiles=tiles2,
            act_fp8=act_fp8,
            act_scale=act_scale,
        )

    def act_with_delta(
        self,
        row_state: CuteDslFp8MaskedRowState,
        gateup_out: torch.Tensor,
        gate_up_delta: torch.Tensor | None,
        topk_ids: torch.Tensor,
        act_out: torch.Tensor,
        activation_lora_input: torch.Tensor,
        *,
        activation: str = "silu",
    ) -> None:
        self._act_kernel(
            gateup_out,
            gate_up_delta,
            act_out,
            activation_lora_input,
            row_state.pair_to_row,
            topk_ids,
            gate_first=self.contract.gate_first,
            interleaved=self.contract.interleaved,
            activation=activation,
            act_quant=(row_state.act_fp8, row_state.act_scale, QUANT_GROUP),
        )
        row_state.act_quant_ready = True

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
    token_width: int
    sf_rows: torch.Tensor  # [m_pad_ceiling, hidden // 128] fp32 group scales
    gemm1_schedule: torch.Tensor
    gemm1_tiles: torch.Tensor
    gemm2_schedule: torch.Tensor
    gemm2_tiles: torch.Tensor


class CuteDslFp8ContiguousProvider(_CuteDslFp8Mixin, ContiguousRowDomainProvider):
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

    def __init__(self, quant_info: MoeLoraFp8QuantInfo):
        super().__init__(quant_info, m_alignment=self.M_ALIGNMENT)
        from sglang.srt.lora.moe.kernels.cutedsl.api import prepare_contiguous_fp8
        from sglang.srt.lora.moe.kernels.cutedsl.schedule_builder import (
            build_dual_stage_schedules_contiguous,
            dual_stage_schedule_capacities_contiguous,
            validate_tile_geometry_contiguous,
        )

        self._prepare_contiguous_gemm = prepare_contiguous_fp8
        self._validate_tile_geometry = validate_tile_geometry_contiguous
        self._build_schedules_contiguous = build_dual_stage_schedules_contiguous
        self._schedule_capacities = dual_stage_schedule_capacities_contiguous
        self._init_tiles(quant_info)

    def _admit_tile_width(self, token_width: int) -> None:
        self._validate_tile_geometry(token_width, self._m_alignment)

    def _prepare_dummy(self, stage: str, config):
        weight, weight_sf = self._stage_weight(stage), self._stage_scale(stage)
        device = weight.device
        num_experts, n, k = weight.shape
        dummy_a = torch.zeros((256, k), dtype=torch.float8_e4m3fn, device=device)
        dummy_sf = torch.zeros(
            (1, 256, k // QUANT_GROUP), dtype=torch.float32, device=device
        )
        dummy_c = torch.empty((256, n), dtype=torch.bfloat16, device=device)
        dummy_seg = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
        return self._prepare_contiguous_gemm(
            dummy_a,
            weight,
            dummy_sf,
            weight_sf,
            dummy_c,
            dummy_seg,
            config=config,
        )

    @staticmethod
    def _rows_arg(tensor: torch.Tensor) -> torch.Tensor:
        # Flat rows occupy one expert slot in the swap_ab wrapper.
        return tensor.unsqueeze(0)

    @staticmethod
    def _input_rows(row_state: CuteDslFp8ContiguousRowState) -> torch.Tensor:
        return row_state.hidden_compact

    @staticmethod
    def _group_arg(row_state: CuteDslFp8ContiguousRowState) -> torch.Tensor:
        return row_state.seg_offsets

    def prepare(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        workspace=None,
    ) -> CuteDslFp8ContiguousRowState:
        base = super().prepare(
            hidden_states, topk_ids, top_k, workspace, fill_rows=False
        )
        num_experts = self.quant_info.num_local_experts
        hidden = hidden_states.size(1)
        device = hidden_states.device
        prefix = f"contiguous:a{self._m_alignment}"
        rows_fp8 = prepare_buffer(
            workspace,
            f"{prefix}:rows_fp8",
            (base.m_pad_ceiling, hidden),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        sf_rows = prepare_buffer(
            workspace,
            f"{prefix}:sf_rows",
            (base.m_pad_ceiling, hidden // QUANT_GROUP),
            dtype=torch.float32,
            device=device,
        )
        dispatch_fill_rows_contiguous_fp8(
            hidden_states,
            topk_ids,
            base.pair_to_row,
            rows_fp8_out=rows_fp8,
            scale_out=sf_rows,
        )
        # Distinct top-k experts bound each expert to one row per token.
        max_expert_rows = max(int(hidden_states.size(0)), 1)
        expected_m = expected_rows_per_expert(topk_ids.numel(), num_experts)
        token_width = self._token_width_for(max_expert_rows, expected_m)
        geometry = dict(
            m_pad_ceiling=base.m_pad_ceiling,
            max_expert_rows=max_expert_rows,
            m_alignment=self._m_alignment,
            token_width=token_width,
            n_gemm1=self.gate_up_slices * self.quant_info.intermediate_size,
            n_gemm2=self.quant_info.hidden_size,
            output_width=self.OUTPUT_WIDTH,
            cluster_shape_mn=self.CLUSTER_SHAPE_MN,
            use_2cta_instrs=self.USE_2CTA_INSTRS,
        )
        schedule_outputs = {}
        if workspace is not None:
            schedule_outputs = self._schedule_buffers(
                workspace,
                f"cutedsl_contiguous:a{self._m_alignment}:tw{token_width}",
                self._schedule_capacities(num_experts=num_experts, **geometry),
                device,
            )
        schedule1, tiles1, schedule2, tiles2 = self._build_schedules_contiguous(
            base.seg_counts, **geometry, **schedule_outputs
        )
        return CuteDslFp8ContiguousRowState(
            hidden_compact=rows_fp8,
            seg_counts=base.seg_counts,
            seg_offsets=base.seg_offsets,
            pair_to_row=base.pair_to_row,
            m_pad_ceiling=base.m_pad_ceiling,
            retained_inputs=base.retained_inputs,
            token_width=token_width,
            sf_rows=sf_rows,
            gemm1_schedule=schedule1,
            gemm1_tiles=tiles1,
            gemm2_schedule=schedule2,
            gemm2_tiles=tiles2,
        )

    @staticmethod
    def _down_operands(
        row_state: CuteDslFp8ContiguousRowState, act_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return sglang_per_token_group_quant_fp8(act_out, QUANT_GROUP)
