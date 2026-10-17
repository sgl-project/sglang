# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 causal multiview rollout with the shared camera/LiDAR stages."""

from sglang.multimodal_gen.configs.pipeline_configs.cosmos3_sim_auto import (
    Cosmos3SimAutoConfig,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos3_sim_auto import (
    Cosmos3SimAutoTransformer,
)
from sglang.multimodal_gen.runtime.pipelines.cosmos3_multiview_pipeline import (
    Cosmos3MultiviewPipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3_multiview import (
    Cosmos3MultiviewDecodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3_sim_auto import (
    Cosmos3SimAutoRolloutStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class Cosmos3NanoSimAutoPipeline(Cosmos3MultiviewPipeline):
    """Per-camera WSM clips and captions to camera videos, one chunk at a time."""

    pipeline_name = "Cosmos3NanoSimAutoPipeline"

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        pipeline_config = server_args.pipeline_config
        if not isinstance(pipeline_config, Cosmos3SimAutoConfig):
            raise TypeError("Cosmos3NanoSimAutoPipeline requires Cosmos3SimAutoConfig")
        transformer = self.get_module("transformer")
        if not isinstance(transformer, Cosmos3SimAutoTransformer):
            raise TypeError(
                "Cosmos3NanoSimAutoPipeline requires Cosmos3SimAutoTransformer"
            )
        # the bidirectional layout is unused; the rollout builds its own replay mask
        self.add_multiview_input_stages(attention_backend="triton")
        self.add_stage(
            Cosmos3SimAutoRolloutStage(
                transformer=transformer,
                manifest=pipeline_config.sim_auto_manifest,
                attention_backend=self.attention_backend,
            )
        )
        self.add_stage(
            Cosmos3MultiviewDecodingStage(
                self.get_module("vae"),
                lidar_decoder=self.get_module("lidar_decoder"),
            )
        )


EntryClass = [Cosmos3NanoSimAutoPipeline]
