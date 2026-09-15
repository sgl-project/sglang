# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 Multiview-AV pipeline.

The rig's camera views are generated in one bidirectional denoising pass from
per-camera WSM (world-scenario map) control clips, an optional RGB anchor per
camera, and one caption per rig or per camera. Joint exports also take an
HD-map range-map control and denoise a LiDAR target alongside the cameras.
Same Nano weights, VAE, tokenizer, and FlowUniPC schedule as
``Cosmos3Pipeline``; the difference is camera-major packing, per-camera VAE
calls, wrapped temporal positions, and the sparse cross-camera attention
installed by ``Cosmos3MultiviewTransformer``.
"""

import importlib.util

from sglang.multimodal_gen.configs.pipeline_configs.cosmos3_multiview import (
    Cosmos3MultiviewConfig,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.models.dits.cosmos3_multiview import (
    Cosmos3MultiviewTransformer,
)
from sglang.multimodal_gen.runtime.models.vaes.cosmos3_lidar_decoder import (
    Cosmos3LidarDecoder,
)
from sglang.multimodal_gen.runtime.models.vaes.cosmos3_lidar_encoder import (
    Cosmos3LidarEncoder,
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
        6. Cosmos3MultiviewDecodingStage - per-camera VAE decode, camera-major,
           plus LiDAR range-map decode for joint requests
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
        lidar_encoder = None
        lidar_decoder = None
        if deployment.supports_lidar:
            if importlib.util.find_spec("natten") is None:
                # Weights load without it; the first joint request would fail in
                # the LiDAR encoder's neighborhood attention. Say so at startup.
                logger.warning(
                    "This checkpoint supports joint camera+LiDAR requests but the "
                    "natten package is not installed; camera-only requests work, LiDAR "
                    "requests will fail. Install it in the image: "
                    "pip install natten==0.21.7 -f https://whl.natten.org"
                )
            # The LiDAR range-map VAE is not a model_index component; it ships
            # as lidar_vae/ next to the transformer and stays in FP32.
            lidar_encoder = Cosmos3LidarEncoder.from_pretrained(
                server_args.model_path, deployment.lidar, get_local_torch_device()
            )
            lidar_decoder = Cosmos3LidarDecoder.from_pretrained(
                server_args.model_path, deployment.lidar, get_local_torch_device()
            )

        self.add_stage(Cosmos3MultiviewInputStage(deployment))
        self.add_stage(
            Cosmos3MultiviewTokenizationStage(
                tokenizer=self.get_module("text_tokenizer"), deployment=deployment
            )
        )
        self.add_stage(
            Cosmos3MultiviewLatentStage(
                vae=vae,
                transformer=transformer,
                deployment=deployment,
                attention_backend=backend,
                lidar_encoder=lidar_encoder,
            )
        )
        self.add_stage(Cosmos3TimestepPreparationStage(scheduler))
        self.add_stage(
            Cosmos3DenoisingStage(
                transformer, scheduler, server_args=server_args, vae=vae
            )
        )
        self.add_stage(
            Cosmos3MultiviewDecodingStage(
                vae, guardrails=False, sound_tokenizer=None, lidar_decoder=lidar_decoder
            )
        )
        logger.info(
            "Cosmos3 multiview pipeline stages created (%d cameras, scope=%s, "
            "control_attends_sensor=%s, backend=%s, schema=%s, per-camera captions=%s, "
            "lidar=%s)",
            deployment.num_views,
            deployment.attention_scope,
            deployment.control_attends_sensor,
            backend,
            deployment.schema_version,
            deployment.separate_view_text_tokenization,
            deployment.supports_lidar,
        )


EntryClass = [Cosmos3MultiviewPipeline]
