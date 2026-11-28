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
        """Unscaled rows: slab-aligned m_max first, then a token width that fits it."""
        m_max = masked_m_max(num_tokens)
        return self._token_width_for(m_max, expected_m), m_max

    def _prepare_dummy(self, stage: str, config):
        weight = self._stage_weight(stage)
        device = weight.device
        _, n, k = weight.shape
        rows, _, groups = self._dummy_layout(device)
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
        """Reuses the input-row buffer: the fill writes it, gate/up reads it last,
        then the down GEMM overwrites it with the output finalize reads."""
        return self._input_rows(row_state)


class CuteDslBf16MaskedRowState(MaskedRowState, kw_only=True):
    token_width: int
    gemm1_schedule: torch.Tensor
    gemm1_tiles: torch.Tensor
    gemm2_schedule: torch.Tensor
    gemm2_tiles: torch.Tensor


class CuteDslBf16MaskedProvider(_CuteDslBf16Mixin, MaskedRowDomainProvider):
    contract = MoeBaseProviderContract(
        key="cutedsl_bf16_masked", quant_info_cls=MoeLoraBf16QuantInfo
    )

    _ROW_MODE = "masked"
    _row_state_cls = CuteDslBf16MaskedRowState

    def __init__(self, quant_info: MoeLoraBf16QuantInfo):
        super().__init__(quant_info)
        from sglang.kernels.ops.lora.moe.cutedsl.api import prepare_masked_bf16

        self._prepare_gemm = prepare_masked_bf16
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
        hidden_permuted, _ = self._input_buffers(
            workspace, "masked", (num_experts, m_max), hidden_states
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
            out_clusters1=-(-geometry["n_gemm1"] // self.OUTPUT_WIDTH),
            out_clusters2=-(-geometry["n_gemm2"] // self.OUTPUT_WIDTH),
            **outputs,
        )
        return CuteDslBf16MaskedRowState(
            hidden_permuted=hidden_permuted,
            masked_m=masked_m,
            pair_to_row=pair_to_row,
            m_max=m_max,
            token_width=token_width,
            gemm1_schedule=outputs["schedule1_out"],
            gemm1_tiles=outputs["tiles1_out"],
            gemm2_schedule=outputs["schedule2_out"],
            gemm2_tiles=outputs["tiles2_out"],
        )

    def _small_prepare_outputs(
        self,
        workspace: LoraWorkspace | None,
        prefix: str,
        capacities: tuple[int, int],
        num_experts: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        import triton

        scratch_len = 2 * max(16, triton.next_power_of_2(num_experts))
        outputs = self._schedule_buffers(workspace, prefix, capacities, device)
        outputs["scratch"] = prepare_buffer(
            workspace,
            f"{prefix}:small_scratch",
            (scratch_len,),
            dtype=torch.int32,
            device=device,
        )
        return outputs


class CuteDslBf16ContiguousRowState(ContiguousRowState, kw_only=True):
    token_width: int
    gemm1_schedule: torch.Tensor
    gemm1_tiles: torch.Tensor
    gemm2_schedule: torch.Tensor
    gemm2_tiles: torch.Tensor


class CuteDslBf16ContiguousProvider(_CuteDslBf16Mixin, ContiguousRowDomainProvider):
    contract = MoeBaseProviderContract(
        key="cutedsl_bf16_contiguous", quant_info_cls=MoeLoraBf16QuantInfo
    )

    _ROW_MODE = "contiguous"
    _row_state_cls = CuteDslBf16ContiguousRowState

    def __init__(self, quant_info: MoeLoraBf16QuantInfo):
        super().__init__(quant_info)
        from sglang.kernels.ops.lora.moe.cutedsl.api import prepare_contiguous_bf16

        self._prepare_gemm = prepare_contiguous_bf16
        # Hopper prefill uses wide tiles.
        device_capability = torch.cuda.get_device_capability(
            quant_info.w13_weight.device
        )
        self._init_tiles(quant_info, drop_narrow_tile=device_capability < (10, 0))
