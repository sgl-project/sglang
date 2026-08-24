from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
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
from sglang.srt.lora.moe.kernels.dispatch import (
    contiguous_m_pad_ceiling,
)
from sglang.srt.lora.moe.quant_info import MoeLoraBf16QuantInfo

if TYPE_CHECKING:
    from sglang.srt.lora.moe.workspace import MoeLoraWorkspace

# Process-global compile cache: one compiled function serves every layer,
# because the weight is a runtime argument. It must hold compiled functions
# only -- a stored ``b_arg`` would pin one layer's weight tensor forever.
_COMPILE_CACHE: dict[tuple, object] = {}


class CuteDslStageCall(msgspec.Struct, frozen=True):
    compiled_fn: object
    b_arg: object


class _CuteDslTileMixin:
    """The attach flow and launch path both CuTeDSL providers share; a row
    mode adds ``_ROW_MODE``, ``_prepare_dummy``, and three argument hooks.
    """

    # The tile choice depends on the expected rows per expert. A choice that
    # used only the packing ceiling ran prefill on the decode tile.
    NARROW_TOKEN_WIDTH = 8
    NARROW_PERSISTENT_CLUSTERS = 128
    WIDE_TOKEN_WIDTH = 64
    WIDE_PERSISTENT_CLUSTERS = 128
    XWIDE_TOKEN_WIDTH = 128
    XWIDE_PERSISTENT_CLUSTERS = 128
    # The packed direct schedule works only with a 1x1 cluster and 1-CTA MMA.
    # The compiled config and the builder's guard both read these two
    # constants. They therefore cannot disagree.
    CLUSTER_SHAPE_MN = (1, 1)
    USE_2CTA_INSTRS = False
    # One CTA per SM and a 64-element K-tile (4 x k16 WGMMA steps). Class
    # constants so a sweep can subclass them; the shipped values are the
    # swept defaults.
    OCCUPANCY = 1
    MMA_INST_TILE_K = 4
    # Sweeps on GB300 chose these thresholds. The step from wide to xwide is
    # not monotonic, because the work rounds up to whole clusters.
    WIDE_EXPECTED_M_THRESHOLD = 16
    XWIDE_EXPECTED_M_THRESHOLD = 96
    OUTPUT_WIDTH = 128

    _ROW_MODE: str

    def _init_tiles(
        self, quant_info: MoeLoraBf16QuantInfo, *, drop_narrow_tile: bool
    ) -> None:
        from sglang.srt.lora.moe.kernels.cutedsl.api import (
            GroupedGemmConfig,
            as_dynamic_cute_tensor,
        )
        from sglang.srt.lora.moe.kernels.cutedsl.schedule_builder import (
            MAX_EXPERTS,
            MAX_TOKEN_CLUSTERS,
        )

        self._as_dynamic_cute_tensor = as_dynamic_cute_tensor
        # Fail at attach; the builder would only catch this at first forward.
        if quant_info.num_local_experts > MAX_EXPERTS:
            raise ValueError(
                f"{quant_info.num_local_experts} local experts exceed the "
                f"direct schedule's {MAX_EXPERTS}-expert packing; use the "
                "DeepGEMM provider for this geometry"
            )
        self._max_token_clusters = MAX_TOKEN_CLUSTERS

        device = quant_info.w13_weight.device
        device_properties = torch.cuda.get_device_properties(device)
        # Sweeps measured one cluster per SM only on the 152-SM GB300. Every
        # other device keeps the 128-cluster schedule.
        xwide_clusters = self.XWIDE_PERSISTENT_CLUSTERS
        if (
            torch.cuda.get_device_capability(device) >= (10, 0)
            and device_properties.multi_processor_count == 152
        ):
            xwide_clusters = 152
        tile_set = (
            (self.NARROW_TOKEN_WIDTH, self.NARROW_PERSISTENT_CLUSTERS),
            (self.WIDE_TOKEN_WIDTH, self.WIDE_PERSISTENT_CLUSTERS),
            (self.XWIDE_TOKEN_WIDTH, xwide_clusters),
        )
        if drop_narrow_tile:
            tile_set = tile_set[1:]

        from sglang.srt.lora.moe.base_gemm_provider.gemm_config_store import (
            cutedsl_version,
            load_config_table,
        )

        version = cutedsl_version()
        self._config_table = load_config_table(
            self.contract.key,
            num_local_experts=quant_info.num_local_experts,
            n_gemm1=self._gate_up_slices * quant_info.intermediate_size,
            n_gemm2=quant_info.hidden_size,
            k=quant_info.hidden_size,
            expected_versions={"cutedsl": version} if version else None,
        )
        if self._config_table is not None:
            for bucket_m, payload in self._config_table.buckets.items():
                if "token_width" not in payload:
                    raise ValueError(
                        f"{self.contract.key} config bucket {bucket_m} lacks "
                        "token_width"
                    )
            widths = dict(tile_set)
            widths.update(
                (tile.token_width, tile.persistent_clusters)
                for tile in self._config_table.tiles
            )
            tile_set = tuple(sorted(widths.items()))

        self._compiled: dict[int, dict[str, CuteDslStageCall]] = {}
        self._tile_configs: dict[int, object] = {}
        for token_width, persistent_clusters in tile_set:
            self._admit_tile_width(token_width)
            config = GroupedGemmConfig(
                mma_tiler_mn=(self.OUTPUT_WIDTH, token_width),
                cluster_shape_mn=self.CLUSTER_SHAPE_MN,
                use_2cta_instrs=self.USE_2CTA_INSTRS,
                occupancy=self.OCCUPANCY,
                mma_inst_tile_k=self.MMA_INST_TILE_K,
                persistent_clusters=persistent_clusters,
                swap_ab=True,
            )
            self._tile_configs[token_width] = config
            self._compiled[token_width] = {}
            self._compile_stage(token_width, "gemm1")
            self._compile_stage(token_width, "gemm2")
        torch.cuda.synchronize(device)

    def _admit_tile_width(self, token_width: int) -> None:
        """A row mode may reject a tile width. The contiguous mode checks its
        segment geometry here; the masked mode takes every width.
        """

    def _stage_weight(self, stage: str) -> torch.Tensor:
        if stage == "gemm1":
            return self.quant_info.w13_weight
        if stage == "gemm2":
            return self.quant_info.w2_weight
        raise ValueError(f"unknown CuTeDSL base stage {stage!r}")

    def _compile_stage(self, token_width: int, stage: str) -> None:
        if stage in self._compiled[token_width]:
            return
        weight = self._stage_weight(stage)
        config = self._tile_configs[token_width]
        device = weight.device
        # The row mode is an explicit key part: a masked build must never
        # serve a contiguous stage of the same geometry.
        key = (
            device.type,
            device.index,
            config,
            self._ROW_MODE,
            self.quant_info.num_local_experts,
            weight.shape[1],
            weight.shape[2],
            torch.bfloat16,
        )
        compiled_fn = _COMPILE_CACHE.get(key)
        if compiled_fn is None:
            prepared = self._prepare_dummy(weight, config)
            # This launch runs zero tiles. It loads the module before any
            # graph capture.
            prepared.launch()
            compiled_fn = prepared.compiled_fn
            _COMPILE_CACHE[key] = compiled_fn
        self._compiled[token_width][stage] = CuteDslStageCall(
            compiled_fn=compiled_fn,
            b_arg=self._as_dynamic_cute_tensor(weight, leading_dim=2),
        )

    def _token_width_for(self, m_max: int, expected_m: int) -> int:
        """The bucket table picks a width for speed only. The loop then widens
        that pick until the schedule can pack ``m_max`` rows. The second step
        is a correctness rule.
        """
        if self._config_table is not None:
            performance_width = self._config_table.pick(expected_m)["token_width"]
        elif expected_m >= self.XWIDE_EXPECTED_M_THRESHOLD:
            performance_width = self.XWIDE_TOKEN_WIDTH
        elif (
            expected_m >= self.WIDE_EXPECTED_M_THRESHOLD
            or self.NARROW_TOKEN_WIDTH not in self._compiled
        ):
            performance_width = self.WIDE_TOKEN_WIDTH
        else:
            performance_width = self.NARROW_TOKEN_WIDTH
        for width in sorted(self._compiled):
            if width >= performance_width and m_max <= width * self._max_token_clusters:
                return width
        widest = max(self._compiled)
        raise ValueError(
            f"m_max={m_max} exceeds the widest compiled tile's schedule "
            f"packing ({widest * self._max_token_clusters}); admission must "
            "fall back to DeepGEMM"
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
        import cuda.bindings.driver as cuda_driver

        call = self._compiled[row_state.token_width][stage]
        # ``_rows_arg`` must repeat the compile path's wrapping: a different
        # wrapping changes the argument's MLIR type, and nothing reports it.
        dyn = self._as_dynamic_cute_tensor
        stream = cuda_driver.CUstream(torch.cuda.current_stream(a.device).cuda_stream)
        call.compiled_fn(
            dyn(self._rows_arg(a), leading_dim=2),
            call.b_arg,
            dyn(self._rows_arg(c), leading_dim=2),
            dyn(self._group_arg(row_state), leading_dim=0),
            dyn(schedule, leading_dim=0),
            dyn(tiles, leading_dim=0),
            stream,
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


class CuteDslMaskedRowState(MaskedRowState, kw_only=True):
    token_width: int
    gemm1_schedule: torch.Tensor
    gemm1_tiles: torch.Tensor
    gemm2_schedule: torch.Tensor
    gemm2_tiles: torch.Tensor


class CuteDslBf16MaskedProvider(_CuteDslTileMixin, MaskedRowDomainProvider):
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
        from sglang.srt.lora.moe.kernels.cutedsl.api import prepare_masked
        from sglang.srt.lora.moe.kernels.cutedsl.schedule_builder import (
            build_dual_stage_schedules_masked,
            dual_stage_schedule_capacities_masked,
        )

        self._prepare_masked_gemm = prepare_masked
        self._build_schedules = build_dual_stage_schedules_masked
        self._schedule_capacities = dual_stage_schedule_capacities_masked
        # The narrow N=8 decode tile compiles on SM90 too (WGMMA takes any N
        # in 8..256 step 8); decode on the wide tile pays ~230 MB of padded
        # slab traffic on H200 q397.
        self._init_tiles(quant_info, drop_narrow_tile=False)

    def _prepare_dummy(self, weight: torch.Tensor, config):
        device = weight.device
        experts = self.quant_info.num_local_experts
        k = weight.shape[2]
        n = weight.shape[1]
        # The dummies fix the layout only; the m extent stays dynamic.
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
    def _input_rows(row_state: CuteDslMaskedRowState) -> torch.Tensor:
        return row_state.hidden_permuted

    @staticmethod
    def _group_arg(row_state: CuteDslMaskedRowState) -> torch.Tensor:
        return row_state.masked_m

    def prepare(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        workspace: MoeLoraWorkspace | None = None,
    ) -> CuteDslMaskedRowState:
        base = super().prepare(hidden_states, topk_ids, top_k, workspace)
        token_width = self._token_width_for(base.m_max, base.expected_m)
        schedule_outputs = {}
        if workspace is not None:
            capacity1, capacity2 = self._schedule_capacities(
                num_experts=base.masked_m.numel(),
                m_max=base.m_max,
                token_width=token_width,
                n_gemm1=self.gate_up_slices * self.quant_info.intermediate_size,
                n_gemm2=self.quant_info.hidden_size,
                output_width=self.OUTPUT_WIDTH,
                cluster_shape_mn=self.CLUSTER_SHAPE_MN,
                use_2cta_instrs=self.USE_2CTA_INSTRS,
            )
            prefix = f"cutedsl_masked:tw{token_width}"
            schedule_outputs = {
                "schedule1_out": workspace.tensor(
                    f"{prefix}:gemm1_schedule",
                    (capacity1,),
                    dtype=torch.int64,
                    device=hidden_states.device,
                ),
                "tiles1_out": workspace.tensor(
                    f"{prefix}:gemm1_tiles",
                    (1,),
                    dtype=torch.int32,
                    device=hidden_states.device,
                ),
                "schedule2_out": workspace.tensor(
                    f"{prefix}:gemm2_schedule",
                    (capacity2,),
                    dtype=torch.int64,
                    device=hidden_states.device,
                ),
                "tiles2_out": workspace.tensor(
                    f"{prefix}:gemm2_tiles",
                    (1,),
                    dtype=torch.int32,
                    device=hidden_states.device,
                ),
            }
        schedule1, tiles1, schedule2, tiles2 = self._build_schedules(
            base.masked_m,
            m_max=base.m_max,
            token_width=token_width,
            n_gemm1=self.gate_up_slices * self.quant_info.intermediate_size,
            n_gemm2=self.quant_info.hidden_size,
            output_width=self.OUTPUT_WIDTH,
            cluster_shape_mn=self.CLUSTER_SHAPE_MN,
            use_2cta_instrs=self.USE_2CTA_INSTRS,
            **schedule_outputs,
        )
        return CuteDslMaskedRowState(
            hidden_permuted=base.hidden_permuted,
            masked_m=base.masked_m,
            expected_m=base.expected_m,
            src2dst=base.src2dst,
            m_max=base.m_max,
            retained_inputs=base.retained_inputs,
            token_width=token_width,
            gemm1_schedule=schedule1,
            gemm1_tiles=tiles1,
            gemm2_schedule=schedule2,
            gemm2_tiles=tiles2,
        )


class CuteDslContiguousRowState(ContiguousRowState, kw_only=True):
    token_width: int
    gemm1_schedule: torch.Tensor
    gemm1_tiles: torch.Tensor
    gemm2_schedule: torch.Tensor
    gemm2_tiles: torch.Tensor


class CuteDslBf16ContiguousProvider(_CuteDslTileMixin, ContiguousRowDomainProvider):
    """Route-major variant of :class:`CuteDslBf16MaskedProvider`.

    A plan row selects this provider when ``base_gemm_rows`` is
    ``route_major``. Only the row domain changes.
    """

    contract = MoeBaseProviderContract(
        key="cutedsl_bf16_contiguous",
        gate_first=True,
        interleaved=False,
        gate_up_output_dtype=torch.bfloat16,
        lora_delta_dtype=torch.bfloat16,
        lora_activation_dtype=torch.bfloat16,
    )

    # 128 matches DeepGEMM's contiguous m-alignment, and every compiled
    # token width divides it.
    M_ALIGNMENT = 128

    _ROW_MODE = "contiguous"

    def __init__(self, quant_info: MoeLoraBf16QuantInfo):
        super().__init__(quant_info, m_alignment=self.M_ALIGNMENT)

        from sglang.srt.lora.moe.kernels.cutedsl.api import prepare_contiguous
        from sglang.srt.lora.moe.kernels.cutedsl.schedule_builder import (
            dual_stage_schedule_capacities_contiguous,
            dual_stage_schedule_pack_contiguous,
            validate_tile_geometry_contiguous,
        )

        self._prepare_contiguous_gemm = prepare_contiguous
        self._validate_tile_geometry = validate_tile_geometry_contiguous
        self._schedule_pack = dual_stage_schedule_pack_contiguous
        self._schedule_capacities = dual_stage_schedule_capacities_contiguous

        # Prefill picks the wide tiles, so Hopper skips the narrow one.
        device_capability = torch.cuda.get_device_capability(
            quant_info.w13_weight.device
        )
        self._init_tiles(quant_info, drop_narrow_tile=device_capability < (10, 0))

    def _admit_tile_width(self, token_width: int) -> None:
        self._validate_tile_geometry(token_width, self._m_alignment)

    def _prepare_dummy(self, weight: torch.Tensor, config):
        device = weight.device
        num_experts = self.quant_info.num_local_experts
        k = weight.shape[2]
        n = weight.shape[1]
        # The dummies fix the layout only; 256 rows is alignment-legal.
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
        # The single expert slot the swap_ab wrapper expects.
        return tensor.unsqueeze(0)

    @staticmethod
    def _input_rows(row_state: CuteDslContiguousRowState) -> torch.Tensor:
        return row_state.hidden_compact

    @staticmethod
    def _group_arg(row_state: CuteDslContiguousRowState) -> torch.Tensor:
        return row_state.seg_offsets

    def prepare(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        workspace: MoeLoraWorkspace | None = None,
    ) -> CuteDslContiguousRowState:
        num_pairs = topk_ids.numel()
        num_experts = self.quant_info.num_local_experts
        # The pack must exist before the dispatch launch that writes into it.
        m_pad_ceiling = contiguous_m_pad_ceiling(
            num_pairs, num_experts, self._m_alignment
        )
        # Top-k experts are distinct per token, so one expert takes at most
        # one pair per token: a host-value row bound, no device readback.
        max_expert_rows = max(int(hidden_states.size(0)), 1)
        expected_m = (num_pairs - 1) // num_experts + 1 if num_pairs else 1
        token_width = self._token_width_for(max_expert_rows, expected_m)
        n_gemm1 = self.gate_up_slices * self.quant_info.intermediate_size
        n_gemm2 = self.quant_info.hidden_size
        schedule_outputs = {}
        if workspace is not None:
            capacity1, capacity2 = self._schedule_capacities(
                num_experts=num_experts,
                m_pad_ceiling=m_pad_ceiling,
                max_expert_rows=max_expert_rows,
                m_alignment=self._m_alignment,
                token_width=token_width,
                n_gemm1=n_gemm1,
                n_gemm2=n_gemm2,
                output_width=self.OUTPUT_WIDTH,
                cluster_shape_mn=self.CLUSTER_SHAPE_MN,
                use_2cta_instrs=self.USE_2CTA_INSTRS,
            )
            prefix = f"cutedsl_contiguous:a{self._m_alignment}:tw{token_width}"
            schedule_outputs = {
                "schedule1_out": workspace.tensor(
                    f"{prefix}:gemm1_schedule",
                    (capacity1,),
                    dtype=torch.int64,
                    device=hidden_states.device,
                ),
                "tiles1_out": workspace.tensor(
                    f"{prefix}:gemm1_tiles",
                    (1,),
                    dtype=torch.int32,
                    device=hidden_states.device,
                ),
                "schedule2_out": workspace.tensor(
                    f"{prefix}:gemm2_schedule",
                    (capacity2,),
                    dtype=torch.int64,
                    device=hidden_states.device,
                ),
                "tiles2_out": workspace.tensor(
                    f"{prefix}:gemm2_tiles",
                    (1,),
                    dtype=torch.int32,
                    device=hidden_states.device,
                ),
            }
        pack = self._schedule_pack(
            num_experts=num_experts,
            m_pad_ceiling=m_pad_ceiling,
            max_expert_rows=max_expert_rows,
            m_alignment=self._m_alignment,
            token_width=token_width,
            n_gemm1=n_gemm1,
            n_gemm2=n_gemm2,
            output_width=self.OUTPUT_WIDTH,
            device=hidden_states.device,
            cluster_shape_mn=self.CLUSTER_SHAPE_MN,
            use_2cta_instrs=self.USE_2CTA_INSTRS,
            **schedule_outputs,
        )
        base = super().prepare(
            hidden_states, topk_ids, top_k, workspace, schedule_pack=pack
        )
        return CuteDslContiguousRowState(
            hidden_compact=base.hidden_compact,
            seg_counts=base.seg_counts,
            seg_offsets=base.seg_offsets,
            src2dst=base.src2dst,
            grouped_layout=base.grouped_layout,
            m_pad_ceiling=base.m_pad_ceiling,
            retained_inputs=base.retained_inputs,
            token_width=token_width,
            gemm1_schedule=pack.schedule1,
            gemm1_tiles=pack.tiles1,
            gemm2_schedule=pack.schedule2,
            gemm2_tiles=pack.tiles2,
        )
