from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
import torch

from sglang.srt.lora.moe.base_gemm_provider.base import MoeBaseProviderContract
from sglang.srt.lora.moe.base_gemm_provider.contiguous_row_domain import (
    ContiguousRowDomainProvider,
    ContiguousRowState,
    contiguous_m_pad_ceiling,
)
from sglang.srt.lora.moe.base_gemm_provider.masked_row_domain import (
    MaskedRowDomainProvider,
    MaskedRowState,
)
from sglang.srt.lora.moe.quant_info import MoeLoraBf16QuantInfo

if TYPE_CHECKING:
    from sglang.srt.lora.moe.workspace import MoeLoraWorkspace

# Process-global compile cache. ``cute.compile`` does not cache, so without
# this cache each provider instance compiles again at every server start. One
# compiled function serves every layer, because the weight is a runtime
# argument and not a Constexpr. The cache holds compiled functions only. It
# must never hold a ``b_arg``. A stored ``b_arg`` keeps a reference to one
# layer's weight tensor, and the process never frees that tensor.
_COMPILE_CACHE: dict[tuple, object] = {}


class CuteDslStageCall(msgspec.Struct, frozen=True):
    compiled_fn: object
    b_arg: object


class CuteDslMaskedRowState(MaskedRowState, kw_only=True):
    token_width: int
    gemm1_schedule: torch.Tensor
    gemm1_tiles: torch.Tensor
    gemm2_schedule: torch.Tensor
    gemm2_tiles: torch.Tensor


class CuteDslBf16Provider(MaskedRowDomainProvider):
    contract = MoeBaseProviderContract(
        key="cutedsl_bf16",
        gate_first=True,
        interleaved=False,
        gate_up_output_dtype=torch.bfloat16,
        lora_delta_dtype=torch.bfloat16,
        lora_activation_dtype=torch.bfloat16,
    )

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
    # Sweeps on GB300 chose these thresholds. The step from wide to xwide is
    # not monotonic, because the work rounds up to whole clusters.
    WIDE_EXPECTED_M_THRESHOLD = 16
    XWIDE_EXPECTED_M_THRESHOLD = 96
    OUTPUT_WIDTH = 128

    def __init__(self, quant_info: MoeLoraBf16QuantInfo):
        super().__init__(quant_info)
        from sglang.srt.lora.moe.base_gemm_provider.cutedsl_masked.api import (
            MaskedGroupedGemmConfig,
            as_dynamic_cute_tensor,
            prepare,
        )

        self._as_dynamic_cute_tensor = as_dynamic_cute_tensor
        self._prepare_masked_gemm = prepare

        from sglang.srt.lora.moe.base_gemm_provider.cutedsl_masked.schedule_builder import (
            MAX_EXPERTS,
            MAX_TOKEN_CLUSTERS,
            build_dual_stage_schedules,
            dual_stage_schedule_capacities,
        )

        # Fail here, at attach. The builder checks this limit only at the
        # first forward.
        if quant_info.num_local_experts > MAX_EXPERTS:
            raise ValueError(
                f"{quant_info.num_local_experts} local experts exceed the "
                f"direct schedule's {MAX_EXPERTS}-expert packing; use the "
                "DeepGEMM provider for this geometry"
            )
        self._build_schedules = build_dual_stage_schedules
        self._schedule_capacities = dual_stage_schedule_capacities
        self._max_token_clusters = MAX_TOKEN_CLUSTERS

        device = quant_info.w13_weight.device
        # BF16 WGMMA accepts every N from 8 to 256 in steps of 8. The narrow
        # N=8 tile is therefore legal on SM90, but this port does not build it.
        # Hopper compiles the wide and xwide tiles only, so decode there runs
        # on the 64-token tile.
        device_properties = torch.cuda.get_device_properties(device)
        device_capability = torch.cuda.get_device_capability(device)
        # Sweeps measured one cluster per SM only on the 152-SM GB300. Every
        # other device keeps the 128-cluster schedule.
        xwide_clusters = self.XWIDE_PERSISTENT_CLUSTERS
        if (
            device_capability >= (10, 0)
            and device_properties.multi_processor_count == 152
        ):
            xwide_clusters = 152
        tile_set = (
            (self.NARROW_TOKEN_WIDTH, self.NARROW_PERSISTENT_CLUSTERS),
            (self.WIDE_TOKEN_WIDTH, self.WIDE_PERSISTENT_CLUSTERS),
            (self.XWIDE_TOKEN_WIDTH, xwide_clusters),
        )
        if device_capability < (10, 0):
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
                        f"cutedsl_bf16 config bucket {bucket_m} lacks token_width"
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
            config = MaskedGroupedGemmConfig(
                mma_tiler_mn=(self.OUTPUT_WIDTH, token_width),
                cluster_shape_mn=self.CLUSTER_SHAPE_MN,
                use_2cta_instrs=self.USE_2CTA_INSTRS,
                occupancy=1,
                mma_inst_tile_k=4,
                persistent_clusters=persistent_clusters,
                swap_ab=True,
                direct_schedule=True,
            )
            self._tile_configs[token_width] = config
            self._compiled[token_width] = {}
            self._compile_stage(token_width, "gemm1")
            self._compile_stage(token_width, "gemm2")
        torch.cuda.synchronize(device)

    def _compile_stage(self, token_width: int, stage: str) -> None:
        call_key = stage
        if call_key in self._compiled[token_width]:
            return
        if stage == "gemm1":
            weight = self.quant_info.w13_weight
        elif stage == "gemm2":
            weight = self.quant_info.w2_weight
        else:
            raise ValueError(f"unknown CuTeDSL base stage {stage!r}")
        config = self._tile_configs[token_width]
        device = weight.device
        experts = self.quant_info.num_local_experts
        k = weight.shape[2]
        n = weight.shape[1]
        # The key holds ``config``, which fixes the geometry and the
        # ``produce_pdl`` flag. Two builds that differ in that flag therefore
        # get two cache entries.
        key = (
            device.type,
            device.index,
            config,
            experts,
            n,
            k,
            torch.bfloat16,
        )
        compiled_fn = _COMPILE_CACHE.get(key)
        if compiled_fn is None:
            # The dummy A and C tensors fix the layout only. The m extent
            # stays dynamic.
            dummy_a = torch.zeros(
                (experts, 256, k), dtype=torch.bfloat16, device=device
            )
            dummy_c = torch.empty(
                (experts, 256, n), dtype=torch.bfloat16, device=device
            )
            dummy_masked = torch.zeros(experts, dtype=torch.int32, device=device)
            prepared = self._prepare_masked_gemm(
                dummy_a,
                weight,
                dummy_c,
                dummy_masked,
                config=config,
            )
            # This launch runs zero tiles. It loads the module before any
            # graph capture.
            prepared.launch()
            compiled_fn = prepared.compiled_fn
            _COMPILE_CACHE[key] = compiled_fn
        self._compiled[token_width][call_key] = CuteDslStageCall(
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
            prefix = f"base:cutedsl:tw{token_width}"
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

    def _launch(
        self,
        stage: str,
        a: torch.Tensor,
        c: torch.Tensor,
        row_state: CuteDslMaskedRowState,
        schedule: torch.Tensor,
        tiles: torch.Tensor,
    ) -> None:
        import cuda.bindings.driver as cuda_driver
        from cutlass.cute.runtime import from_dlpack

        call = self._compiled[row_state.token_width][stage]

        def dyn(tensor: torch.Tensor, leading_dim: int):
            return from_dlpack(tensor, assumed_align=16).mark_layout_dynamic(
                leading_dim=leading_dim
            )

        stream = cuda_driver.CUstream(torch.cuda.current_stream(a.device).cuda_stream)
        call.compiled_fn(
            dyn(a, 2),
            call.b_arg,
            dyn(c, 2),
            dyn(row_state.masked_m, 0),
            dyn(schedule, 0),
            dyn(tiles, 0),
            stream,
        )

    def gateup(self, row_state: CuteDslMaskedRowState, out: torch.Tensor) -> None:
        self._launch(
            "gemm1",
            row_state.hidden_permuted,
            out,
            row_state,
            row_state.gemm1_schedule,
            row_state.gemm1_tiles,
        )

    def down(
        self,
        row_state: CuteDslMaskedRowState,
        act_out: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        self._launch(
            "gemm2",
            act_out,
            out,
            row_state,
            row_state.gemm2_schedule,
            row_state.gemm2_tiles,
        )


class CuteDslContiguousRowState(ContiguousRowState, kw_only=True):
    token_width: int
    gemm1_schedule: torch.Tensor
    gemm1_tiles: torch.Tensor
    gemm2_schedule: torch.Tensor
    gemm2_tiles: torch.Tensor


class CuteDslBf16ContiguousProvider(ContiguousRowDomainProvider):
    """Route-major variant of :class:`CuteDslBf16Provider`.

    A plan row selects this provider when ``base_gemm_rows`` is
    ``route_major``. Only the row domain changes.
    """

    contract = MoeBaseProviderContract(
        key="cutedsl_contiguous",
        gate_first=True,
        interleaved=False,
        gate_up_output_dtype=torch.bfloat16,
        lora_delta_dtype=torch.bfloat16,
        lora_activation_dtype=torch.bfloat16,
    )

    # The alignment stays at 128. It matches DeepGEMM's contiguous
    # m-alignment, and every compiled token width (8, 64, 128) divides it.
    # Decode with a smaller alignment ran slower than the masked path.
    M_ALIGNMENT = 128

    # These names alias the masked provider. A tuning change there also
    # changes this provider.
    NARROW_TOKEN_WIDTH = CuteDslBf16Provider.NARROW_TOKEN_WIDTH
    NARROW_PERSISTENT_CLUSTERS = CuteDslBf16Provider.NARROW_PERSISTENT_CLUSTERS
    WIDE_TOKEN_WIDTH = CuteDslBf16Provider.WIDE_TOKEN_WIDTH
    WIDE_PERSISTENT_CLUSTERS = CuteDslBf16Provider.WIDE_PERSISTENT_CLUSTERS
    XWIDE_TOKEN_WIDTH = CuteDslBf16Provider.XWIDE_TOKEN_WIDTH
    XWIDE_PERSISTENT_CLUSTERS = CuteDslBf16Provider.XWIDE_PERSISTENT_CLUSTERS
    CLUSTER_SHAPE_MN = CuteDslBf16Provider.CLUSTER_SHAPE_MN
    USE_2CTA_INSTRS = CuteDslBf16Provider.USE_2CTA_INSTRS
    WIDE_EXPECTED_M_THRESHOLD = CuteDslBf16Provider.WIDE_EXPECTED_M_THRESHOLD
    XWIDE_EXPECTED_M_THRESHOLD = CuteDslBf16Provider.XWIDE_EXPECTED_M_THRESHOLD
    OUTPUT_WIDTH = CuteDslBf16Provider.OUTPUT_WIDTH

    _token_width_for = CuteDslBf16Provider._token_width_for

    def __init__(self, quant_info: MoeLoraBf16QuantInfo):
        device = quant_info.w13_weight.device
        super().__init__(quant_info, m_alignment=self.M_ALIGNMENT)

        from sglang.srt.lora.moe.base_gemm_provider.cutedsl_masked.api import (
            MaskedGroupedGemmConfig,
            as_dynamic_cute_tensor,
            prepare_contiguous,
        )

        self._as_dynamic_cute_tensor = as_dynamic_cute_tensor
        self._prepare_contiguous_gemm = prepare_contiguous

        from sglang.srt.lora.moe.base_gemm_provider.cutedsl_masked.schedule_builder import (
            MAX_EXPERTS,
            MAX_TOKEN_CLUSTERS,
            contiguous_dual_stage_schedule_capacities,
            contiguous_dual_stage_schedule_pack,
            validate_contiguous_tile_geometry,
        )

        self._validate_tile_geometry = validate_contiguous_tile_geometry

        if quant_info.num_local_experts > MAX_EXPERTS:
            raise ValueError(
                f"{quant_info.num_local_experts} local experts exceed the "
                f"direct schedule's {MAX_EXPERTS}-expert packing; use the "
                "DeepGEMM provider for this geometry"
            )
        self._schedule_pack = contiguous_dual_stage_schedule_pack
        self._schedule_capacities = contiguous_dual_stage_schedule_capacities
        self._max_token_clusters = MAX_TOKEN_CLUSTERS

        # Sweeps measured one cluster per SM only on the 152-SM GB300. Every
        # other device keeps the 128-cluster schedule.
        device_properties = torch.cuda.get_device_properties(device)
        device_capability = torch.cuda.get_device_capability(device)
        xwide_clusters = self.XWIDE_PERSISTENT_CLUSTERS
        if (
            device_capability >= (10, 0)
            and device_properties.multi_processor_count == 152
        ):
            xwide_clusters = 152
        tile_set = (
            (self.NARROW_TOKEN_WIDTH, self.NARROW_PERSISTENT_CLUSTERS),
            (self.WIDE_TOKEN_WIDTH, self.WIDE_PERSISTENT_CLUSTERS),
            (self.XWIDE_TOKEN_WIDTH, xwide_clusters),
        )
        # The SM90 port accepts a tile N of 64, 128 or 256 only. Hopper
        # therefore drops the narrow 8-token tile.
        if device_capability < (10, 0):
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
                        f"cutedsl_contiguous config bucket {bucket_m} lacks "
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
            self._validate_tile_geometry(token_width, self._m_alignment)
            config = MaskedGroupedGemmConfig(
                mma_tiler_mn=(self.OUTPUT_WIDTH, token_width),
                cluster_shape_mn=self.CLUSTER_SHAPE_MN,
                use_2cta_instrs=self.USE_2CTA_INSTRS,
                occupancy=1,
                mma_inst_tile_k=4,
                persistent_clusters=persistent_clusters,
                swap_ab=True,
                direct_schedule=True,
                contiguous_segments=True,
            )
            self._tile_configs[token_width] = config
            self._compiled[token_width] = {}
            self._compile_stage(token_width, "gemm1")
            self._compile_stage(token_width, "gemm2")
        torch.cuda.synchronize(device)

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
        num_experts = self.quant_info.num_local_experts
        k = weight.shape[2]
        n = weight.shape[1]
        # ``config`` holds ``contiguous_segments``, so this entry never reuses
        # the masked provider's entry for the same geometry.
        key = (
            device.type,
            device.index,
            config,
            num_experts,
            n,
            k,
            torch.bfloat16,
        )
        compiled_fn = _COMPILE_CACHE.get(key)
        if compiled_fn is None:
            # The flat dummy tensors fix the layout only. The row count stays
            # dynamic. 256 rows is a multiple of the alignment, as the kernel
            # requires.
            dummy_a = torch.zeros((256, k), dtype=torch.bfloat16, device=device)
            dummy_c = torch.empty((256, n), dtype=torch.bfloat16, device=device)
            dummy_seg = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
            prepared = self._prepare_contiguous_gemm(
                dummy_a,
                weight,
                dummy_c,
                dummy_seg,
                config=config,
                m_alignment=self._m_alignment,
            )
            # This launch runs zero tiles. It loads the module before any
            # graph capture.
            prepared.launch()
            compiled_fn = prepared.compiled_fn
            _COMPILE_CACHE[key] = compiled_fn
        self._compiled[token_width][stage] = CuteDslStageCall(
            compiled_fn=compiled_fn,
            b_arg=self._as_dynamic_cute_tensor(weight, leading_dim=2),
        )

    def prepare(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        workspace: MoeLoraWorkspace | None = None,
    ) -> CuteDslContiguousRowState:
        num_pairs = topk_ids.numel()
        num_experts = self.quant_info.num_local_experts
        # The pack must run before the dispatch, because the dispatch kernel
        # also writes the schedules. Every input below is a host value, so the
        # pack and the dispatch agree on the row ceiling.
        m_pad_ceiling = contiguous_m_pad_ceiling(
            num_pairs, num_experts, self._m_alignment
        )
        # A token's top-k experts are distinct, so one expert takes at most one
        # pair per token. This bound is a host value, so the width escalation
        # and the schedule check need no device readback.
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
            # Two instances can share one layer workspace, and the capacities
            # depend on the alignment. The alignment tag in the name keeps
            # their schedule buffers apart.
            prefix = f"base:cutedsl_contig:a{self._m_alignment}:tw{token_width}"
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

    @staticmethod
    def _invoke(
        call: CuteDslStageCall,
        a: torch.Tensor,
        c: torch.Tensor,
        seg_offsets: torch.Tensor,
        schedule: torch.Tensor,
        tiles: torch.Tensor,
    ) -> None:
        import cuda.bindings.driver as cuda_driver
        from cutlass.cute.runtime import from_dlpack

        def dyn(tensor: torch.Tensor, leading_dim: int):
            return from_dlpack(tensor, assumed_align=16).mark_layout_dynamic(
                leading_dim=leading_dim
            )

        stream = cuda_driver.CUstream(torch.cuda.current_stream(a.device).cuda_stream)
        # The ``unsqueeze(0)`` repeats the wrapping that the compile path used
        # for its dummy tensors. The ``masked_m`` argument then holds
        # ``seg_offsets``.
        call.compiled_fn(
            dyn(a.unsqueeze(0), 2),
            call.b_arg,
            dyn(c.unsqueeze(0), 2),
            dyn(seg_offsets, 0),
            dyn(schedule, 0),
            dyn(tiles, 0),
            stream,
        )

    def _launch(
        self,
        stage: str,
        a: torch.Tensor,
        c: torch.Tensor,
        row_state: CuteDslContiguousRowState,
        schedule: torch.Tensor,
        tiles: torch.Tensor,
    ) -> None:
        call = self._compiled[row_state.token_width][stage]
        self._invoke(call, a, c, row_state.seg_offsets, schedule, tiles)

    def gateup(
        self,
        row_state: CuteDslContiguousRowState,
        out: torch.Tensor,
    ) -> None:
        self._launch(
            "gemm1",
            row_state.hidden_compact,
            out,
            row_state,
            row_state.gemm1_schedule,
            row_state.gemm1_tiles,
        )

    def down(
        self,
        row_state: CuteDslContiguousRowState,
        act_out: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        self._launch(
            "gemm2",
            act_out,
            out,
            row_state,
            row_state.gemm2_schedule,
            row_state.gemm2_tiles,
        )
