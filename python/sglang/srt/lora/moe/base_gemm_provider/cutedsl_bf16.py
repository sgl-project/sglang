from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.ops.lora.moe.dispatch_masked_small import (
    small_masked_prepare,
    small_masked_prepare_applies,
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
from sglang.srt.lora.moe.quant_info import MoeLoraBf16QuantInfo

if TYPE_CHECKING:
    from sglang.srt.lora.workspace import LoraWorkspace


class _CuteDslBf16Mixin(CuteDslTileMixin):
    """BF16 rows are the row domains' default encoding: their fill and buffers."""

    _DTYPE_TAG = "bf16"
    WIDE_EXPECTED_M_THRESHOLD = 16

    def _plan_rows(self, num_tokens: int, expected_m: int) -> tuple[int, int]:
        """The masked row domain's hook for unscaled rows: m_max first, in the
        slab alignment, then the token width whose schedule fits it."""
        m_max = masked_m_max(num_tokens)
        return self._token_width_for(m_max, expected_m), m_max

    def _prepare_dummy(self, stage: str, config):
        weight = self._stage_weight(stage)
        device = weight.device
        _, n, k = weight.shape
        rows, _, groups = self._dummy_layout(device)
        # Compile the layout with a dynamic M extent.
        return self._prepare_gemm(
            torch.zeros((*rows, k), dtype=torch.bfloat16, device=device),
            weight,
            torch.empty((*rows, n), dtype=torch.bfloat16, device=device),
            groups,
            config=config,
        )

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

    def down_output(self, row_state, workspace: LoraWorkspace) -> torch.Tensor:
        """With a workspace the down output is the input-row buffer. Producer
        of the rows: the fill in prepare(). Last consumer: the gate/up GEMM.
        The down GEMM overwrites them; finalize reads the buffer as its output."""
        if row_state.input_buffer_reuse:
            return self._input_rows(row_state)
        return super().down_output(row_state, workspace)


class CuteDslBf16MaskedRowState(MaskedRowState, kw_only=True):
    """MaskedRowState plus the schedule fields: CuteDslTileMixin._schedules(),
    or the small prepare's own launch."""

    token_width: int
    gemm1_schedule: torch.Tensor
    gemm1_tiles: torch.Tensor
    gemm2_schedule: torch.Tensor
    gemm2_tiles: torch.Tensor


class CuteDslBf16MaskedProvider(_CuteDslBf16Mixin, MaskedRowDomainProvider):
    contract = MoeBaseProviderContract(
        key="cutedsl_bf16_masked",
        quant_info_cls=MoeLoraBf16QuantInfo,
        gate_first=True,
        interleaved=False,
        gate_up_output_dtype=torch.bfloat16,
        lora_delta_dtype=torch.bfloat16,
        lora_activation_dtype=torch.bfloat16,
    )

    _ROW_MODE = "masked"
    _row_state_cls = CuteDslBf16MaskedRowState

    def __init__(self, quant_info: MoeLoraBf16QuantInfo):
        super().__init__(quant_info)
        from sglang.kernels.ops.lora.moe.cutedsl.api import prepare_masked_bf16
        from sglang.kernels.ops.lora.moe.cutedsl.schedule_builder import (
            build_dual_stage_schedules_masked,
            dual_stage_schedule_capacities_masked,
        )

        self._prepare_gemm = prepare_masked_bf16
        self._build_schedules = build_dual_stage_schedules_masked
        self._schedule_capacities = dual_stage_schedule_capacities_masked
        self._init_tiles(quant_info)

    def prepare(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        workspace: LoraWorkspace | None = None,
    ) -> CuteDslBf16MaskedRowState:
        num_pairs = topk_ids.numel()
        if not small_masked_prepare_applies(hidden_states.size(0), num_pairs):
            return super().prepare(hidden_states, topk_ids, top_k, workspace)
        # Prepare rows, pair map and both schedules in one launch.
        num_experts = self.quant_info.num_local_experts
        expected_m = expected_rows_per_expert(num_pairs, num_experts)
        token_width, m_max = self._plan_rows(hidden_states.size(0), expected_m)
        geometry = self._stage_geometry(token_width)
        device = hidden_states.device
        masked_m = prepare_buffer(
            workspace,
            "masked:masked_m",
            (num_experts,),
            dtype=torch.int32,
            device=device,
        )
        pair_to_row = prepare_buffer(
            workspace,
            "masked:pair_to_row",
            (num_pairs,),
            dtype=torch.int32,
            device=device,
        )
        hidden_permuted = prepare_buffer(
            workspace,
            "masked:hidden_permuted",
            (num_experts, m_max, hidden_states.size(1)),
            dtype=torch.bfloat16,
            device=device,
        )
        outputs = self._small_prepare_outputs(
            workspace,
            f"cutedsl_masked:tw{token_width}",
            self._schedule_capacities(num_experts=num_experts, m_max=m_max, **geometry),
            num_experts,
            device,
        )
        small_masked_prepare(
            hidden_states,
            topk_ids,
            top_k,
            masked_m_out=masked_m,
            pair_to_row_out=pair_to_row,
            rows_out=hidden_permuted,
            token_width=token_width,
            out_clusters1=self._output_clusters(geometry["n_gemm1"]),
            out_clusters2=self._output_clusters(geometry["n_gemm2"]),
            **outputs,
        )
        return CuteDslBf16MaskedRowState(
            hidden_permuted=hidden_permuted,
            masked_m=masked_m,
            expected_m=expected_m,
            pair_to_row=pair_to_row,
            m_max=m_max,
            input_buffer_reuse=workspace is not None,
            token_width=token_width,
            gemm1_schedule=outputs["schedule1_out"],
            gemm1_tiles=outputs["tiles1_out"],
            gemm2_schedule=outputs["schedule2_out"],
            gemm2_tiles=outputs["tiles2_out"],
        )


class CuteDslBf16ContiguousRowState(ContiguousRowState, kw_only=True):
    """ContiguousRowState plus the schedule fields
    (CuteDslTileMixin._schedules())."""

    token_width: int
    gemm1_schedule: torch.Tensor
    gemm1_tiles: torch.Tensor
    gemm2_schedule: torch.Tensor
    gemm2_tiles: torch.Tensor


class CuteDslBf16ContiguousProvider(_CuteDslBf16Mixin, ContiguousRowDomainProvider):
    contract = MoeBaseProviderContract(
        key="cutedsl_bf16_contiguous",
        quant_info_cls=MoeLoraBf16QuantInfo,
        gate_first=True,
        interleaved=False,
        gate_up_output_dtype=torch.bfloat16,
        lora_delta_dtype=torch.bfloat16,
        lora_activation_dtype=torch.bfloat16,
    )

    # Every compiled token width divides 128.
    M_ALIGNMENT = 128

    _ROW_MODE = "contiguous"
    _row_state_cls = CuteDslBf16ContiguousRowState

    def __init__(self, quant_info: MoeLoraBf16QuantInfo):
        super().__init__(quant_info, m_alignment=self.M_ALIGNMENT)

        from sglang.kernels.ops.lora.moe.cutedsl.api import prepare_contiguous_bf16
        from sglang.kernels.ops.lora.moe.cutedsl.schedule_builder import (
            build_dual_stage_schedules_contiguous,
            dual_stage_schedule_capacities_contiguous,
            validate_tile_geometry_contiguous,
        )

        self._prepare_gemm = prepare_contiguous_bf16
        self._validate_tile_geometry = validate_tile_geometry_contiguous
        self._build_schedules = build_dual_stage_schedules_contiguous
        self._schedule_capacities = dual_stage_schedule_capacities_contiguous

        # Hopper prefill uses wide tiles.
        device_capability = torch.cuda.get_device_capability(
            quant_info.w13_weight.device
        )
        self._init_tiles(quant_info, drop_narrow_tile=device_capability < (10, 0))

    def _admit_tile_width(self, token_width: int) -> None:
        self._validate_tile_geometry(token_width, self._m_alignment)
