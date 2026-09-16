# SPDX-License-Identifier: Apache-2.0
"""Cosmos-Dreams (Cosmos3-Interactive) causal video pipeline.

Same Cosmos3 Omni weights, VAE, and tokenizer as ``Cosmos3Pipeline``, but the
transformer is loaded as ``CosmosDreamsTransformer`` and driven
autoregressively: chunked four-step SDE denoising conditioned on per-frame
actions, with clean K/V committed as history between chunks.
"""

from sglang.multimodal_gen.configs.pipeline_configs.cosmos_dreams import (
    CosmosDreamsConfig,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos_dreams import (
    CosmosDreamsTransformer,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3 import (
    Cosmos3DecodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos_dreams import (
    CosmosDreamsImageStage,
    CosmosDreamsPrepareStage,
    CosmosDreamsRolloutStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class CosmosDreamsPipeline(ComposedPipelineBase):
    """Image + prompt + camera/robot actions to video, one chunk at a time."""

    pipeline_name = "CosmosDreamsPipeline"
    is_video_pipeline = True

    _required_config_modules = [
        "text_tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        pipeline_config = server_args.pipeline_config
        if not isinstance(pipeline_config, CosmosDreamsConfig):
            raise TypeError(
                "CosmosDreamsPipeline requires CosmosDreamsConfig, got "
                f"{type(pipeline_config).__name__}; pass --model-id nvidia/Cosmos3-Nano-Sim-Bimanual "
                "or make the checkpoint's model_index.json name CosmosDreamsPipeline."
            )
        transformer = self.get_module("transformer")
        if not isinstance(transformer, CosmosDreamsTransformer):
            raise TypeError(
                "CosmosDreamsPipeline loaded the wrong transformer type: "
                f"{type(transformer).__name__}."
            )
        manifest = transformer.manifest
        vae = self.get_module("vae")

        self.add_stage(CosmosDreamsImageStage(canvas_tier=pipeline_config.canvas_tier))
        self.add_stage(
            CosmosDreamsPrepareStage(
                vae=vae,
                tokenizer=self.get_module("text_tokenizer"),
                manifest=manifest,
                max_pixels=pipeline_config.max_pixels,
            )
        )
        self.add_stage(
            CosmosDreamsRolloutStage(
                transformer=transformer,
                scheduler=self.get_module("scheduler"),
                manifest=manifest,
            )
        )
        self.add_stage(
            Cosmos3DecodingStage(vae, guardrails=False, sound_tokenizer=None)
        )
        logger.info(
            "Cosmos-Dreams pipeline stages created (checkpoint %s, chunk_size=%d, window=%d frames)",
            manifest.checkpoint_id,
            manifest.chunk_size,
            manifest.window_frames,
        )


EntryClass = [CosmosDreamsPipeline]
