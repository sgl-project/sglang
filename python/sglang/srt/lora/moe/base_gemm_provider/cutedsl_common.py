"""Shared compilation, tile selection, and launch buffers for CuTeDSL."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypedDict

import msgspec
import torch

from sglang.srt.lora.moe.base_gemm_provider.base import prepare_buffer
from sglang.srt.lora.moe.base_gemm_provider.masked_row_domain import masked_m_max

if TYPE_CHECKING:
    from sglang.kernels.ops.lora.moe.cutedsl.api import GroupedGemmConfig
    from sglang.srt.lora.moe.base_gemm_provider.base import MoeBaseProviderContract
    from sglang.srt.lora.moe.base_gemm_provider.gemm_config_store import (
        GemmConfigTable,
    )
    from sglang.srt.lora.moe.quant_info import StandardLayoutQuantInfo
    from sglang.srt.lora.workspace import LoraWorkspace

logger = logging.getLogger(__name__)

# Cache code only: caching bound arguments would retain layer weights.
_COMPILE_CACHE: dict[tuple, object] = {}


class CuteDslStageCall(msgspec.Struct, frozen=True):
    compiled_fn: object
    b_arg: object
    sf_weights: torch.Tensor | None = None


class _GemmScheduleFields(TypedDict):
    """The row-state fields CuteDslTileMixin._schedules() returns; every
    CuTeDSL row state declares these five."""

    token_width: int  # the tile width the rows were planned and scheduled for
    gemm1_schedule: torch.Tensor  # int64 packed tile entries of gate/up
    gemm1_tiles: torch.Tensor  # int32 [1], the entries in use
    gemm2_schedule: torch.Tensor  # the same for down
    gemm2_tiles: torch.Tensor


class CuteDslTileMixin:
    """Tile selection, stage compilation and schedules shared by the CuTeDSL
    providers. A concrete provider is a format mixin (a subclass of this one)
    over a row domain, and supplies what this mixin reads:

    - the class attributes declared below;
    - bound in its ``__init__`` before ``_init_tiles()``: ``quant_info`` and
      ``_gate_up_slices`` (the row domain's ``__init__``), the stage binder
      ``_prepare_gemm`` and the row layout's ``_build_schedules`` and
      ``_schedule_capacities``; a contiguous provider also binds
      ``_validate_tile_geometry`` for its ``_admit_tile_width``;
    - ``_prepare_dummy(stage, config)`` from the format mixin: the stage bound
      to stand-in operands, which ``_compile_stage`` compiles once per key.

    ``_init_tiles()`` binds the rest: the compiled stages, the tile configs,
    the tuned config table and the launch helpers. The mixin implements the
    row domains' ``_schedules`` hook, the masked ``_plan_rows`` hook and the
    contiguous ``_token_width_for`` hook.
    """

    NARROW_TOKEN_WIDTH = 8
    NARROW_PERSISTENT_CLUSTERS = 128
    WIDE_TOKEN_WIDTH = 64
    WIDE_PERSISTENT_CLUSTERS = 128
    XWIDE_TOKEN_WIDTH = 128
    XWIDE_PERSISTENT_CLUSTERS = 128
    # The packed schedule supports only 1x1 clusters and 1-CTA MMA.
    CLUSTER_SHAPE_MN = (1, 1)
    USE_2CTA_INSTRS = False
    OCCUPANCY = 1
    MMA_INST_TILE_K = 4
    WIDE_EXPECTED_M_THRESHOLD: int
    XWIDE_EXPECTED_M_THRESHOLD = 96
    OUTPUT_WIDTH = 128
    TOKEN_WIDTHS = (NARROW_TOKEN_WIDTH, WIDE_TOKEN_WIDTH, XWIDE_TOKEN_WIDTH)

    # Supplied by the concrete provider: class attributes ...
    _ROW_MODE: str  # "masked" or "contiguous"; part of the compile key
    _DTYPE_TAG: str  # the weight encoding; part of the compile key
    contract: MoeBaseProviderContract  # its key names the tuned config table
    # ... and bound in its __init__ before _init_tiles().
    quant_info: StandardLayoutQuantInfo
    _gate_up_slices: int
    # cutedsl.api.prepare_<rows>_<encoding>: binds one stage to its operands and
    # returns the prepared launch (``launch()``, ``compiled_fn``).
    _prepare_gemm: Callable[..., Any]
    # schedule_builder.build_dual_stage_schedules_<rows>: (gemm1 schedule,
    # gemm1 tiles, gemm2 schedule, gemm2 tiles) from the rows per expert.
    _build_schedules: Callable[
        ..., tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]
    # schedule_builder.dual_stage_schedule_capacities_<rows>: the two
    # schedules' capacities in entries.
    _schedule_capacities: Callable[..., tuple[int, int]]

    # Bound by _init_tiles().
    _compiled: dict[int, dict[str, CuteDslStageCall]]  # token width -> stage
    _tile_configs: dict[int, GroupedGemmConfig]
    _config_table: GemmConfigTable | None

    def _init_tiles(
        self, quant_info: StandardLayoutQuantInfo, *, drop_narrow_tile: bool = False
    ) -> None:
        import cuda.bindings.driver as cuda_driver

        from sglang.kernels.ops.lora.moe.cutedsl.api import (
            GroupedGemmConfig,
            as_dynamic_cute_tensor,
        )
        from sglang.kernels.ops.lora.moe.cutedsl.schedule_builder import (
            MAX_EXPERTS,
            MAX_TOKEN_CLUSTERS,
        )
        from sglang.srt.lora.moe.base_gemm_provider.gemm_config_store import (
            cutedsl_version,
            load_config_table,
        )

        self._as_dynamic_cute_tensor = as_dynamic_cute_tensor
        self._cu_stream = cuda_driver.CUstream
        if quant_info.num_local_experts > MAX_EXPERTS:
            raise ValueError(
                f"{quant_info.num_local_experts} local experts exceed the "
                f"direct schedule's {MAX_EXPERTS}-expert packing"
            )
        self._max_token_clusters = MAX_TOKEN_CLUSTERS
        self._bind_weights(quant_info)

        device = quant_info.w13_weight.device
        # Keep the SM90 FP8 heuristic at 64-wide tiles; tables may override it.
        self._xwide_by_heuristic = not (
            self._DTYPE_TAG == "fp8"
            and torch.cuda.get_device_capability(device) < (10, 0)
        )
        # Use one persistent cluster per SM on 152-SM Blackwell devices.
        xwide_clusters = self.XWIDE_PERSISTENT_CLUSTERS
        if (
            torch.cuda.get_device_capability(device) >= (10, 0)
            and torch.cuda.get_device_properties(device).multi_processor_count == 152
        ):
            xwide_clusters = 152
        tile_set = (
            (self.NARROW_TOKEN_WIDTH, self.NARROW_PERSISTENT_CLUSTERS),
            (self.WIDE_TOKEN_WIDTH, self.WIDE_PERSISTENT_CLUSTERS),
            (self.XWIDE_TOKEN_WIDTH, xwide_clusters),
        )
        if drop_narrow_tile:
            tile_set = tile_set[1:]
        tile_set = tuple(tile for tile in tile_set if tile[0] in self.TOKEN_WIDTHS)

        version = cutedsl_version()
        self._config_table = load_config_table(
            self.contract.key,
            num_local_experts=quant_info.num_local_experts,
            n_gemm1=self._gate_up_slices * quant_info.intermediate_size,
            n_gemm2=quant_info.hidden_size,
            k=quant_info.hidden_size,
            installed_versions={"cutedsl": version} if version else None,
        )
        tile_set = self._merge_table_tiles(tile_set)

        self._compiled: dict[int, dict[str, CuteDslStageCall]] = {}
        self._tile_configs: dict[int, GroupedGemmConfig] = {}
        for token_width, persistent_clusters in tile_set:
            self._admit_tile_width(token_width)
            config = GroupedGemmConfig(
                mma_tiler_mn=(self.OUTPUT_WIDTH, token_width),
                cluster_shape_mn=self.CLUSTER_SHAPE_MN,
                use_2cta_instrs=self.USE_2CTA_INSTRS,
                occupancy=self.OCCUPANCY,
                mma_inst_tile_k=self.MMA_INST_TILE_K,
                persistent_clusters=persistent_clusters,
            )
            self._tile_configs[token_width] = config
            self._compiled[token_width] = {}
            self._compile_stage(token_width, "gemm1", config)
            self._compile_stage(token_width, "gemm2", config)
        torch.cuda.synchronize(device)

    def _merge_table_tiles(
        self, tile_set: tuple[tuple[int, int], ...]
    ) -> tuple[tuple[int, int], ...]:
        """Add the table's tiles; drop a table whose buckets pick uncompiled widths."""
        if self._config_table is None:
            return tile_set
        widths = dict(tile_set)
        widths.update(
            (tile.token_width, tile.persistent_clusters)
            for tile in self._config_table.tiles
        )
        undeclared = {
            payload["token_width"] for payload in self._config_table.buckets.values()
        } - set(widths)
        if undeclared:
            logger.warning(
                "Ignoring the %s GEMM config: buckets select token widths %s that "
                "no tile declares",
                self.contract.key,
                sorted(undeclared),
            )
            self._config_table = None
            return tile_set
        return tuple(sorted(widths.items()))

    def _bind_weights(self, quant_info: StandardLayoutQuantInfo) -> None:
        """Optional hook for dtype-specific weight preparation."""

    def _admit_tile_width(self, token_width: int) -> None:
        """Optional hook for row-layout constraints."""

    def _stage_weight(self, stage: str) -> torch.Tensor:
        if stage == "gemm1":
            return self.quant_info.w13_weight
        if stage == "gemm2":
            return self.quant_info.w2_weight
        raise ValueError(f"unknown CuTeDSL base stage {stage!r}")

    def _stage_scale(self, stage: str) -> torch.Tensor | None:
        return None

    def _stage_constexprs(self, stage: str) -> tuple:
        """Compile-time arguments beyond the operand geometry."""
        return ()

    def _compile_stage(
        self, token_width: int, stage: str, config, *, lora_k: int = 0
    ) -> None:
        """``lora_k`` > 0 compiles the stage with the rank-``lora_k`` epilogue
        LoRA (block-scaled providers only) under the call key ``<stage>:lora``."""
        weight = self._stage_weight(stage)
        device = weight.device
        key = (
            device.type,
            device.index,
            config,
            self._ROW_MODE,
            self._DTYPE_TAG,
            self.quant_info.num_local_experts,
            weight.shape[1],
            weight.shape[2],
            *self._stage_constexprs(stage),
            lora_k,
        )
        compiled_fn = _COMPILE_CACHE.get(key)
        if compiled_fn is None:
            prepared = self._prepare_dummy(
                stage, config, **({"lora_k": lora_k} if lora_k else {})
            )
            # Load the module before graph capture with a zero-tile launch.
            prepared.launch()
            compiled_fn = prepared.compiled_fn
            _COMPILE_CACHE[key] = compiled_fn
        call_key = f"{stage}:lora" if lora_k else stage
        self._compiled[token_width][call_key] = CuteDslStageCall(
            compiled_fn=compiled_fn,
            b_arg=self._as_dynamic_cute_tensor(weight, leading_dim=2),
            sf_weights=self._stage_scale(stage),
        )

    def _token_width_for(self, max_expert_rows: int, expected_m: int) -> int:
        """Widen the tuned choice if needed to fit the packed schedule."""
        if self._config_table is not None:
            performance_width = self._config_table.pick(expected_m)["token_width"]
        elif expected_m >= self.XWIDE_EXPECTED_M_THRESHOLD and self._xwide_by_heuristic:
            performance_width = self.XWIDE_TOKEN_WIDTH
        elif (
            expected_m >= self.WIDE_EXPECTED_M_THRESHOLD
            or self.NARROW_TOKEN_WIDTH not in self._compiled
        ):
            performance_width = self.WIDE_TOKEN_WIDTH
        else:
            performance_width = self.NARROW_TOKEN_WIDTH
        for width in sorted(self._compiled):
            if (
                width >= performance_width
                and max_expert_rows <= width * self._max_token_clusters
            ):
                return width
        widest = max(self._compiled)
        raise ValueError(
            f"max_expert_rows={max_expert_rows} exceeds the widest compiled "
            f"tile's schedule packing ({widest * self._max_token_clusters})"
        )

    def _stage_geometry(self, token_width: int, **rows) -> dict:
        return dict(
            token_width=token_width,
            n_gemm1=self.gate_up_slices * self.quant_info.intermediate_size,
            n_gemm2=self.quant_info.hidden_size,
            output_width=self.OUTPUT_WIDTH,
            cluster_shape_mn=self.CLUSTER_SHAPE_MN,
            use_2cta_instrs=self.USE_2CTA_INSTRS,
            **rows,
        )

    def _plan_rows(self, num_tokens: int, expected_m: int) -> tuple[int, int]:
        """The masked row domain's hook for a scaled encoding: the token width,
        then m_max in whole token tiles. Scale loads are unpredicated across
        the tile."""
        token_width = self._token_width_for(num_tokens, expected_m)
        return token_width, masked_m_max(num_tokens, token_width)

    def _schedules(
        self,
        counts: torch.Tensor,
        token_width: int,
        prefix: str,
        workspace: LoraWorkspace | None,
        **rows: int,
    ) -> _GemmScheduleFields:
        """The row domains' hook: both stages' packed schedules over the rows a
        row domain laid out (``rows`` is its geometry), as row-state fields.

        With a workspace the four outputs are its buffers
        ``<prefix>:tw<token_width>:gemm{1,2}_{schedule,tiles}``, sized by
        ``_schedule_capacities``; without one the builder allocates them.
        ``_build_schedules`` writes them on the device from ``counts``."""
        geometry = self._stage_geometry(token_width, **rows)
        outputs = {}
        if workspace is not None:
            outputs = self._schedule_buffers(
                workspace,
                f"{prefix}:tw{token_width}",
                self._schedule_capacities(
                    num_experts=self.quant_info.num_local_experts, **geometry
                ),
                counts.device,
            )
        schedule1, tiles1, schedule2, tiles2 = self._build_schedules(
            counts, **geometry, **outputs
        )
        return dict(
            token_width=token_width,
            gemm1_schedule=schedule1,
            gemm1_tiles=tiles1,
            gemm2_schedule=schedule2,
            gemm2_tiles=tiles2,
        )

    def _stream(self, device: torch.device):
        return self._cu_stream(torch.cuda.current_stream(device).cuda_stream)

    @staticmethod
    def _output_clusters(n: int) -> int:
        return (n + CuteDslTileMixin.OUTPUT_WIDTH - 1) // CuteDslTileMixin.OUTPUT_WIDTH

    def _small_prepare_outputs(
        self,
        workspace: LoraWorkspace | None,
        prefix: str,
        capacities: tuple[int, int],
        num_experts: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Schedule buffers plus the small prepare's stage-begin scratch."""
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

    @staticmethod
    def _schedule_buffers(
        workspace: LoraWorkspace | None,
        prefix: str,
        capacities: tuple[int, int],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        capacity1, capacity2 = capacities
        return {
            "schedule1_out": prepare_buffer(
                workspace,
                f"{prefix}:gemm1_schedule",
                (capacity1,),
                dtype=torch.int64,
                device=device,
            ),
            "tiles1_out": prepare_buffer(
                workspace,
                f"{prefix}:gemm1_tiles",
                (1,),
                dtype=torch.int32,
                device=device,
            ),
            "schedule2_out": prepare_buffer(
                workspace,
                f"{prefix}:gemm2_schedule",
                (capacity2,),
                dtype=torch.int64,
                device=device,
            ),
            "tiles2_out": prepare_buffer(
                workspace,
                f"{prefix}:gemm2_tiles",
                (1,),
                dtype=torch.int32,
                device=device,
            ),
        }
