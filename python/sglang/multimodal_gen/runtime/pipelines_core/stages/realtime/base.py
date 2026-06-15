# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import (
    scale_and_shift_latents,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE


@dataclass(frozen=True)
class RealtimeStageComponent:
    component_name: str
    precision_attr: str | None = None
    memory_intensive: bool = False
    keep_ready_after_warmup: bool = True


class RealtimeDiffusionStage(PipelineStage):
    """common contract for stateful realtime diffusion stages

    model-specific subclasses keep conditioning, latent scheduling, and model
    forward semantics
    """

    component_specs: tuple[RealtimeStageComponent, ...] = (
        RealtimeStageComponent("transformer", "dit_precision", memory_intensive=True),
        RealtimeStageComponent("vae", "vae_precision"),
    )

    def __init__(
        self,
        *,
        transformer: torch.nn.Module | None = None,
        vae: torch.nn.Module | None = None,
        model_path: str | None = None,
        default_height: int | None = None,
        default_width: int | None = None,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.vae = vae
        self.model_path = model_path
        self.default_height = default_height
        self.default_width = default_width

    @property
    def role_affinity(self):
        return RoleType.MONOLITHIC

    @property
    def parallelism_type(self) -> StageParallelismType:
        return StageParallelismType.REPLICATED

    def require_session(self, batch: Req, *, context: str | None = None):
        if batch.session is None:
            label = context or self.__class__.__name__
            raise ValueError(f"{label} requires a realtime session")
        return batch.session

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        stage_name = self._component_stage_name(stage_name)
        uses: list[ComponentUse] = []
        for spec in self.component_specs:
            target_dtype = None
            if spec.precision_attr is not None:
                precision = getattr(server_args.pipeline_config, spec.precision_attr)
                target_dtype = PRECISION_TO_TYPE[precision]
            uses.append(
                ComponentUse(
                    stage_name,
                    spec.component_name,
                    target_dtype=target_dtype,
                    memory_intensive=spec.memory_intensive,
                    keep_ready_after_warmup=spec.keep_ready_after_warmup,
                )
            )
        return uses

    def target_pixel_size(self, batch: Req) -> tuple[int, int]:
        height = batch.height if batch.height is not None else self.default_height
        width = batch.width if batch.width is not None else self.default_width
        if height is None or width is None:
            raise ValueError(
                f"{self.__class__.__name__} needs batch.height/batch.width or defaults"
            )
        return int(height), int(width)

    def _empty_output(self, batch: Req) -> OutputBatch:
        target_h, target_w = self.target_pixel_size(batch)
        output = torch.empty(
            (1, 3, 0, target_h, target_w),
            dtype=torch.float32,
            device=get_local_torch_device(),
        )
        return OutputBatch(output=output, metrics=batch.metrics)

    def scale_and_shift_latents(
        self,
        latents: torch.Tensor,
        server_args: ServerArgs,
    ) -> torch.Tensor:
        return scale_and_shift_latents(latents, server_args, self.vae)

    def ensure_causal_vae_conv_cache(self, state) -> dict:
        if state.conv_cache is None:
            state.conv_cache = self.vae.reset_decoder_cache()
        return state.conv_cache
