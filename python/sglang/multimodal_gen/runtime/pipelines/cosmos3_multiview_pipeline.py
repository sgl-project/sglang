# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 Multiview-AV pipeline.

Eleven fixed camera views are generated in one bidirectional denoising pass
from per-camera WSM (world-scenario map) control clips, an optional RGB
anchor per camera, and one caption. Same Nano weights, VAE, tokenizer, and
FlowUniPC schedule as ``Cosmos3Pipeline``; the difference is camera-major
packing, per-camera VAE calls, wrapped temporal positions, and the sparse
cross-camera attention installed by ``Cosmos3MultiviewTransformer``.
"""

from sglang.multimodal_gen.configs.pipeline_configs.cosmos3_multiview import (
    Cosmos3MultiviewConfig,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos3_multiview import (
    Cosmos3MultiviewTransformer,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3 import (
    Cosmos3DenoisingStage,
    Cosmos3TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3_multiview import (
    Cosmos3MultiviewDecodingStage,
    Cosmos3MultiviewInputStage,
    Cosmos3MultiviewLatentStage,
    Cosmos3MultiviewTokenizationStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class Cosmos3MultiviewPipeline(ComposedPipelineBase):
    """WSM control clips + caption to 11 RGB camera videos in one pass."""

    pipeline_name = "Cosmos3MultiviewPipeline"
    is_video_pipeline = True

    _required_config_modules = [
        "text_tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        """Stage order:

        1. Cosmos3MultiviewInputStage - per-camera media, camera-major packing
        2. Cosmos3MultiviewTokenizationStage - transfer prompt + WSM emphasis
        3. Cosmos3MultiviewLatentStage - per-camera VAE encode, anchors, layout
        4. Cosmos3TimestepPreparationStage - FlowUniPC timesteps (shift 10)
        5. Cosmos3DenoisingStage - sequential text CFG with the sparse mask
        6. Cosmos3MultiviewDecodingStage - per-camera VAE decode, camera-major
        """
        pipeline_config = server_args.pipeline_config
        if not isinstance(pipeline_config, Cosmos3MultiviewConfig):
            raise TypeError(
                "Cosmos3MultiviewPipeline requires Cosmos3MultiviewConfig, got "
                f"{type(pipeline_config).__name__}; make the checkpoint's "
                "model_index.json name Cosmos3MultiviewPipeline."
            )
        deployment = pipeline_config.multiview_deployment
        if deployment is None:
            raise ValueError(
                "Cosmos3MultiviewConfig did not resolve the checkpoint's multiview "
                "deployment config; model_path must be set before stage creation."
            )
        transformer = self.get_module("transformer")
        if not isinstance(transformer, Cosmos3MultiviewTransformer):
            raise TypeError(
                "Cosmos3MultiviewPipeline loaded the wrong transformer type: "
                f"{type(transformer).__name__}; transformer/config.json must set "
                "backbone_type='cosmos3_multiview'."
            )
        vae = self.get_module("vae")
        scheduler = self.get_module("scheduler")
        backend = pipeline_config.resolved_multiview_backend()

        self.add_stage(Cosmos3MultiviewInputStage(deployment))
        self.add_stage(
            Cosmos3MultiviewTokenizationStage(
                tokenizer=self.get_module("text_tokenizer")
            )
        )
        self.add_stage(
            Cosmos3MultiviewLatentStage(
                vae=vae,
                transformer=transformer,
                deployment=deployment,
                attention_backend=backend,
            )
        )
        self.add_stage(Cosmos3TimestepPreparationStage(scheduler))
        self.add_stage(
            Cosmos3DenoisingStage(
                transformer, scheduler, server_args=server_args, vae=vae
            )
        )
        self.add_stage(
            Cosmos3MultiviewDecodingStage(vae, guardrails=False, sound_tokenizer=None)
        )
        logger.info(
            "Cosmos3 multiview pipeline stages created (%d cameras, scope=%s, "
            "control_attends_sensor=%s, backend=%s)",
            deployment.num_views,
            deployment.attention_scope,
            deployment.control_attends_sensor,
            backend,
        )


EntryClass = [Cosmos3MultiviewPipeline]
