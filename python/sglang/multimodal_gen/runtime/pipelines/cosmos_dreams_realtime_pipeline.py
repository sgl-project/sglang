# SPDX-License-Identifier: Apache-2.0
"""Cosmos-Dreams realtime (tick) session pipeline.

Same modules as ``CosmosDreamsPipeline``; the stages keep the rollout state in
the realtime session so that each ``/v1/realtime_video`` request generates one
block of latent frames driven by the actions received since the last tick.
Requests without a session run the offline rollout unchanged.
"""

from sglang.multimodal_gen.configs.pipeline_configs.cosmos_dreams_realtime import (
    CosmosDreamsRealtimeConfig,
)
from sglang.multimodal_gen.configs.sample.cosmos_dreams import (
    CosmosDreamsSamplingParams,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos_dreams import (
    CosmosDreamsTransformer,
)
from sglang.multimodal_gen.runtime.pipelines.cosmos_dreams_pipeline import (
    CosmosDreamsPipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos_dreams import (
    CosmosDreamsImageStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos_dreams_realtime import (
    CosmosDreamsCausalDecodingStage,
    CosmosDreamsRealtimePrepareStage,
    CosmosDreamsRealtimeRolloutStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class CosmosDreamsRealtimePipeline(CosmosDreamsPipeline):
    """One autoregressive block per realtime tick, offline rollout otherwise."""

    pipeline_name = "CosmosDreamsRealtimePipeline"
    # Registered together so the explicit pipeline class refines the model-default
    # config and the realtime adapter registry resolves this pipeline.
    pipeline_config_cls = CosmosDreamsRealtimeConfig
    sampling_params_cls = CosmosDreamsSamplingParams

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        pipeline_config = server_args.pipeline_config
        if not isinstance(pipeline_config, CosmosDreamsRealtimeConfig):
            raise TypeError(
                "CosmosDreamsRealtimePipeline requires CosmosDreamsRealtimeConfig, got "
                f"{type(pipeline_config).__name__}; launch with "
                "--pipeline-class-name CosmosDreamsRealtimePipeline."
            )
        transformer = self.get_module("transformer")
        if not isinstance(transformer, CosmosDreamsTransformer):
            raise TypeError(
                "CosmosDreamsRealtimePipeline loaded the wrong transformer type: "
                f"{type(transformer).__name__}."
            )
        manifest = transformer.manifest
        vae = self.get_module("vae")
        self.add_stage(CosmosDreamsImageStage(canvas_tier=pipeline_config.canvas_tier))
        self.add_stage(
            CosmosDreamsRealtimePrepareStage(
                vae=vae,
                tokenizer=self.get_module("text_tokenizer"),
                manifest=manifest,
                max_pixels=pipeline_config.max_pixels,
            )
        )
        self.add_stage(
            CosmosDreamsRealtimeRolloutStage(
                transformer=transformer,
                scheduler=self.get_module("scheduler"),
                manifest=manifest,
            )
        )
        self.add_stage(
            CosmosDreamsCausalDecodingStage(vae, guardrails=False, sound_tokenizer=None)
        )
        logger.info(
            "Cosmos-Dreams realtime pipeline stages created (checkpoint %s, chunk_size=%d, window=%d frames)",
            manifest.checkpoint_id,
            manifest.chunk_size,
            manifest.window_frames,
        )


EntryClass = [CosmosDreamsRealtimePipeline]
