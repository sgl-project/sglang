# SPDX-License-Identifier: Apache-2.0
"""Auto-duration stage for LTX-2.5.

Runs between the text connectors and latent preparation, so it can rewrite
`batch.num_frames` before any latent shape is derived from it.
"""

import torch

from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.precision import resolve_precision

logger = init_logger(__name__)


class LTX2DurationStage(PipelineStage):
    """Predict `num_frames` from the caption when auto-duration is requested.

    Upstream expresses this by omitting `num_frames` on a pipeline that has a
    duration head. SGLang's sampling params always carry a frame count, so the
    request opts in explicitly via `auto_duration`.
    """

    def __init__(self, duration_head) -> None:
        super().__init__()
        self.duration_head = duration_head

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        if self.duration_head is None:
            return []
        dtype = resolve_precision(
            server_args, "duration_head", precision_attr="dit_precision"
        )
        return [
            ComponentUse(
                self._component_stage_name(stage_name),
                "duration_head",
                target_dtype=dtype,
            )
        ]

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if not batch.auto_duration:
            return batch

        if self.duration_head is None:
            raise ValueError(
                "auto_duration was requested but this checkpoint has no duration "
                "head. It ships from LTX-2.5 onward."
            )

        video_tokens = batch.prompt_embeds
        audio_tokens = batch.audio_prompt_embeds
        if isinstance(video_tokens, list):
            video_tokens = video_tokens[0]
        if isinstance(audio_tokens, list):
            audio_tokens = audio_tokens[0]

        # A CFG batch carries [negative, positive] with duplicated rows, so
        # predict from the first positive row only.
        with (
            self.use_declared_component(
                component_name="duration_head", module=self.duration_head
            ) as duration_head,
            torch.no_grad(),
        ):
            assert duration_head is not None
            num_frames = duration_head.predict_num_frames(
                video_tokens[:1],
                audio_tokens[:1],
                frame_rate=float(batch.fps),
                temporal_compression_ratio=int(
                    server_args.pipeline_config.vae_temporal_compression
                ),
                min_seconds=float(batch.auto_duration_min_seconds),
                max_seconds=float(batch.auto_duration_max_seconds),
            )

        logger.info(
            "Auto-duration: %d frames (requested %d) @ %.2f fps",
            num_frames,
            int(batch.num_frames),
            float(batch.fps),
        )
        batch.num_frames = int(num_frames)
        return batch
