"""Shared compilation, tile selection, and launch buffers for CuTeDSL."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import msgspec
import torch

if TYPE_CHECKING:
    from sglang.srt.lora.moe.quant_info import StandardLayoutQuantInfo
    from sglang.srt.lora.workspace import LoraWorkspace

logger = logging.getLogger(__name__)

# Cache code only: caching bound arguments would retain layer weights.
_COMPILE_CACHE: dict[tuple, object] = {}


class CuteDslStageCall(msgspec.Struct, frozen=True):
    compiled_fn: object
    b_arg: object
    sf_weights: torch.Tensor | None = None


class CuteDslTileMixin:
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

    _ROW_MODE: str
    _DTYPE_TAG: str

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
            self._compile_stage(token_width, "gemm1")
            self._compile_stage(token_width, "gemm2")
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

    def _compile_stage(self, token_width: int, stage: str) -> None:
        if stage in self._compiled[token_width]:
            return
        weight = self._stage_weight(stage)
        config = self._tile_configs[token_width]
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
        )
        compiled_fn = _COMPILE_CACHE.get(key)
        if compiled_fn is None:
            prepared = self._prepare_dummy(stage, config)
            # Load the module before graph capture with a zero-tile launch.
            prepared.launch()
            compiled_fn = prepared.compiled_fn
            _COMPILE_CACHE[key] = compiled_fn
        self._compiled[token_width][stage] = CuteDslStageCall(
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
        if workspace is None:
            capacity1, capacity2 = capacities
            return {
                "schedule1_out": torch.empty(
                    capacity1, dtype=torch.int64, device=device
                ),
                "tiles1_out": torch.empty(1, dtype=torch.int32, device=device),
                "schedule2_out": torch.empty(
                    capacity2, dtype=torch.int64, device=device
                ),
                "tiles2_out": torch.empty(1, dtype=torch.int32, device=device),
                "scratch": torch.empty(scratch_len, dtype=torch.int32, device=device),
            }
        outputs = self._schedule_buffers(workspace, prefix, capacities, device)
        outputs["scratch"] = workspace.tensor(
            f"{prefix}:small_scratch", (scratch_len,), dtype=torch.int32, device=device
        )
        return outputs

    @staticmethod
    def _schedule_buffers(
        workspace: LoraWorkspace,
        prefix: str,
        capacities: tuple[int, int],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        capacity1, capacity2 = capacities
        return {
            "schedule1_out": workspace.tensor(
                f"{prefix}:gemm1_schedule",
                (capacity1,),
                dtype=torch.int64,
                device=device,
            ),
            "tiles1_out": workspace.tensor(
                f"{prefix}:gemm1_tiles", (1,), dtype=torch.int32, device=device
            ),
            "schedule2_out": workspace.tensor(
                f"{prefix}:gemm2_schedule",
                (capacity2,),
                dtype=torch.int64,
                device=device,
            ),
            "tiles2_out": workspace.tensor(
                f"{prefix}:gemm2_tiles", (1,), dtype=torch.int32, device=device
            ),
        }
