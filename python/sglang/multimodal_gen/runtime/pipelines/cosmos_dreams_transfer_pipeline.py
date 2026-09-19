# SPDX-License-Identifier: Apache-2.0
"""Cosmos-Dreams-Transfer causal control-video pipeline.

Same weights layout, VAE, and tokenizer as ``CosmosDreamsPipeline``; the
control clip replaces action tokens as the conditioning signal.
"""

from sglang.multimodal_gen.configs.pipeline_configs.cosmos_dreams_transfer import (
    CosmosDreamsTransferConfig,
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
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos_dreams_transfer import (
    CosmosDreamsControlVideoStage,
    CosmosDreamsTransferPrepareStage,
    CosmosDreamsTransferRolloutStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class Cosmos3NanoSimTransferPipeline(ComposedPipelineBase):
    """Control video + prompt to video, one chunk at a time."""

    pipeline_name = "Cosmos3NanoSimTransferPipeline"
    is_video_pipeline = True

    _required_config_modules = [
        "text_tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        pipeline_config = server_args.pipeline_config
        if not isinstance(pipeline_config, CosmosDreamsTransferConfig):
            raise TypeError(
                "Cosmos3NanoSimTransferPipeline requires CosmosDreamsTransferConfig, got "
                f"{type(pipeline_config).__name__}; pass --model-id nvidia/Cosmos3-Nano-Sim-Depth "
                "or make the checkpoint's model_index.json name Cosmos3NanoSimTransferPipeline."
            )
        transformer = self.get_module("transformer")
        if not isinstance(transformer, CosmosDreamsTransformer):
            raise TypeError(
                "Cosmos3NanoSimTransferPipeline loaded the wrong transformer type: "
                f"{type(transformer).__name__}."
            )
        manifest = transformer.manifest
        vae = self.get_module("vae")

        self.add_stage(
            CosmosDreamsControlVideoStage(
                manifest=manifest,
                canvas_tier=pipeline_config.canvas_tier,
                max_pixels=pipeline_config.max_pixels,
            )
        )
        self.add_stage(
            CosmosDreamsTransferPrepareStage(
                vae=vae,
                tokenizer=self.get_module("text_tokenizer"),
                manifest=manifest,
            )
        )
        self.add_stage(
            CosmosDreamsTransferRolloutStage(
                transformer=transformer,
                scheduler=self.get_module("scheduler"),
                manifest=manifest,
            )
        )
        self.add_stage(
            Cosmos3DecodingStage(vae, guardrails=False, sound_tokenizer=None)
        )
        logger.info(
            "Cosmos-Dreams-Transfer pipeline stages created (checkpoint %s, hints %s, chunk_size=%d)",
            manifest.checkpoint_id,
            list(manifest.control_contract.hints),
            manifest.chunk_size,
        )


EntryClass = [Cosmos3NanoSimTransferPipeline]
