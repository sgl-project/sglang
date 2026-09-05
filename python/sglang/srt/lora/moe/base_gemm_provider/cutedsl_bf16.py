from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.lora.moe.base_gemm_provider.base import (
    MoeBaseProviderContract,
    expected_rows_per_expert,
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
from sglang.srt.lora.moe.quant_info import MoeLoraBf16QuantInfo

if TYPE_CHECKING:
    from sglang.srt.lora.moe.workspace import MoeLoraWorkspace


class _CuteDslBf16Mixin(CuteDslTileMixin):
    _DTYPE_TAG = "bf16"
    WIDE_EXPECTED_M_THRESHOLD = 16

    def _launch(
        self,
        stage: str,
        a: torch.Tensor,
        c: torch.Tensor,
        row_state,
        schedule: torch.Tensor,
        tiles: torch.Tensor,
    ) -> None:
        call = self._compiled[row_state.token_width][stage]
        # Runtime wrapping must match the compiled argument's MLIR type.
        dyn = self._as_dynamic_cute_tensor
        call.compiled_fn(
            dyn(self._rows_arg(a), leading_dim=2),
            call.b_arg,
            dyn(self._rows_arg(c), leading_dim=2),
            dyn(self._group_arg(row_state), leading_dim=0),
            dyn(schedule, leading_dim=0),
            dyn(tiles, leading_dim=0),
            self._stream(a.device),
        )

    def gateup(self, row_state, out: torch.Tensor) -> None:
        self._launch(
            "gemm1",
            self._input_rows(row_state),
            out,
            row_state,
            row_state.gemm1_schedule,
            row_state.gemm1_tiles,
        )

    def down(self, row_state, act_out: torch.Tensor, out: torch.Tensor) -> None:
        self._launch(
            "gemm2",
            act_out,
            out,
            row_state,
            row_state.gemm2_schedule,
            row_state.gemm2_tiles,
        )


class CuteDslBf16MaskedRowState(MaskedRowState, kw_only=True):
    token_width: int
    gemm1_schedule: torch.Tensor
    gemm1_tiles: torch.Tensor
    gemm2_schedule: torch.Tensor
    gemm2_tiles: torch.Tensor


class CuteDslBf16MaskedProvider(_CuteDslBf16Mixin, MaskedRowDomainProvider):
    contract = MoeBaseProviderContract(
        key="cutedsl_bf16_masked",
        gate_first=True,
        interleaved=False,
        gate_up_output_dtype=torch.bfloat16,
        lora_delta_dtype=torch.bfloat16,
        lora_activation_dtype=torch.bfloat16,
    )

    _ROW_MODE = "masked"

    def __init__(self, quant_info: MoeLoraBf16QuantInfo):
        super().__init__(quant_info)
        from sglang.srt.lora.moe.kernels.cutedsl.api import prepare_masked_bf16
        from sglang.srt.lora.moe.kernels.cutedsl.schedule_builder import (
            build_dual_stage_schedules_masked,
            dual_stage_schedule_capacities_masked,
        )

        self._prepare_masked_gemm = prepare_masked_bf16
        self._build_schedules = build_dual_stage_schedules_masked
        self._schedule_capacities = dual_stage_schedule_capacities_masked
        self._init_tiles(quant_info)

    def _prepare_dummy(self, stage: str, config):
        weight = self._stage_weight(stage)
        device = weight.device
        experts, n, k = weight.shape
        # Compile the layout with a dynamic M extent.
        dummy_a = torch.zeros((experts, 256, k), dtype=torch.bfloat16, device=device)
        dummy_c = torch.empty((experts, 256, n), dtype=torch.bfloat16, device=device)
        dummy_masked = torch.zeros(experts, dtype=torch.int32, device=device)
        return self._prepare_masked_gemm(
            dummy_a,
            weight,
            dummy_c,
            dummy_masked,
            config=config,
        )

    @staticmethod
    def _rows_arg(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    @staticmethod
    def _input_rows(row_state: CuteDslBf16MaskedRowState) -> torch.Tensor:
        return row_state.hidden_permuted

    @staticmethod
    def _group_arg(row_state: CuteDslBf16MaskedRowState) -> torch.Tensor:
        return row_state.masked_m

    def prepare(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        workspace: MoeLoraWorkspace | None = None,
    ) -> CuteDslBf16MaskedRowState:
        base = super().prepare(hidden_states, topk_ids, top_k, workspace)
        token_width = self._token_width_for(base.m_max, base.expected_m)
        geometry = dict(
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
                f"cutedsl_masked:tw{token_width}",
                self._schedule_capacities(
                    num_experts=base.masked_m.numel(), m_max=base.m_max, **geometry
                ),
                hidden_states.device,
            )
        schedule1, tiles1, schedule2, tiles2 = self._build_schedules(
            base.masked_m, m_max=base.m_max, **geometry, **schedule_outputs
        )
        return CuteDslBf16MaskedRowState(
            hidden_permuted=base.hidden_permuted,
            masked_m=base.masked_m,
            expected_m=base.expected_m,
            pair_to_row=base.pair_to_row,
            m_max=base.m_max,
            retained_inputs=base.retained_inputs,
            token_width=token_width,
            gemm1_schedule=schedule1,
            gemm1_tiles=tiles1,
            gemm2_schedule=schedule2,
            gemm2_tiles=tiles2,
        )


class CuteDslBf16ContiguousRowState(ContiguousRowState, kw_only=True):
    token_width: int
    gemm1_schedule: torch.Tensor
    gemm1_tiles: torch.Tensor
    gemm2_schedule: torch.Tensor
    gemm2_tiles: torch.Tensor


class CuteDslBf16ContiguousProvider(_CuteDslBf16Mixin, ContiguousRowDomainProvider):

    contract = MoeBaseProviderContract(
        key="cutedsl_bf16_contiguous",
        gate_first=True,
        interleaved=False,
        gate_up_output_dtype=torch.bfloat16,
        lora_delta_dtype=torch.bfloat16,
        lora_activation_dtype=torch.bfloat16,
    )

    # Every compiled token width divides 128.
    M_ALIGNMENT = 128

    _ROW_MODE = "contiguous"

    def __init__(self, quant_info: MoeLoraBf16QuantInfo):
        super().__init__(quant_info, m_alignment=self.M_ALIGNMENT)

        from sglang.srt.lora.moe.kernels.cutedsl.api import prepare_contiguous_bf16
        from sglang.srt.lora.moe.kernels.cutedsl.schedule_builder import (
            build_dual_stage_schedules_contiguous,
            dual_stage_schedule_capacities_contiguous,
            validate_tile_geometry_contiguous,
        )

        self._prepare_contiguous_gemm = prepare_contiguous_bf16
        self._validate_tile_geometry = validate_tile_geometry_contiguous
        self._build_schedules_contiguous = build_dual_stage_schedules_contiguous
        self._schedule_capacities = dual_stage_schedule_capacities_contiguous

        # Hopper prefill uses wide tiles.
        device_capability = torch.cuda.get_device_capability(
            quant_info.w13_weight.device
        )
        self._init_tiles(quant_info, drop_narrow_tile=device_capability < (10, 0))

    def _admit_tile_width(self, token_width: int) -> None:
        self._validate_tile_geometry(token_width, self._m_alignment)

    def _prepare_dummy(self, stage: str, config):
        weight = self._stage_weight(stage)
        device = weight.device
        num_experts, n, k = weight.shape
        # Compile the layout with a dynamic M extent.
        dummy_a = torch.zeros((256, k), dtype=torch.bfloat16, device=device)
        dummy_c = torch.empty((256, n), dtype=torch.bfloat16, device=device)
        dummy_seg = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
        return self._prepare_contiguous_gemm(
            dummy_a,
            weight,
            dummy_c,
            dummy_seg,
            config=config,
        )

    @staticmethod
    def _rows_arg(tensor: torch.Tensor) -> torch.Tensor:
        # Flat rows occupy one expert slot in the swap_ab wrapper.
        return tensor.unsqueeze(0)

    @staticmethod
    def _input_rows(row_state: CuteDslBf16ContiguousRowState) -> torch.Tensor:
        return row_state.hidden_compact

    @staticmethod
    def _group_arg(row_state: CuteDslBf16ContiguousRowState) -> torch.Tensor:
        return row_state.seg_offsets

    def prepare(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        workspace: MoeLoraWorkspace | None = None,
    ) -> CuteDslBf16ContiguousRowState:
        base = super().prepare(hidden_states, topk_ids, top_k, workspace)
        num_experts = self.quant_info.num_local_experts
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
                hidden_states.device,
            )
        schedule1, tiles1, schedule2, tiles2 = self._build_schedules_contiguous(
            base.seg_counts, **geometry, **schedule_outputs
        )
        return CuteDslBf16ContiguousRowState(
            hidden_compact=base.hidden_compact,
            seg_counts=base.seg_counts,
            seg_offsets=base.seg_offsets,
            pair_to_row=base.pair_to_row,
            m_pad_ceiling=base.m_pad_ceiling,
            retained_inputs=base.retained_inputs,
            token_width=token_width,
            gemm1_schedule=schedule1,
            gemm1_tiles=tiles1,
            gemm2_schedule=schedule2,
            gemm2_tiles=tiles2,
        )
