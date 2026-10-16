# SPDX-License-Identifier: Apache-2.0
"""Cosmos3-Nano-Sim-Auto pipeline.

Causal rolling multiview WSM transfer, optionally joint with LiDAR: the same
per-camera inputs and outputs as ``Cosmos3MultiviewPipeline``, generated chunk
by chunk by a distilled student with a KV ring of committed chunks. The
multiview input, tokenization, latent and decoding stages are reused; the
denoising stage is the Sim-Auto rollout.
"""

import importlib.util

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.configs.pipeline_configs.cosmos3_multiview import (
    COSMOS3_MULTIVIEW_AUTO_BACKEND,
)
from sglang.multimodal_gen.configs.pipeline_configs.cosmos3_sim_auto import (
    Cosmos3SimAutoConfig,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.models.dits.cosmos3_multiview_attention import (
    resolve_masked_backend,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos3_sim_auto import (
    Cosmos3SimAutoTransformer,
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
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3_multiview import (
    Cosmos3MultiviewDecodingStage,
    Cosmos3MultiviewInputStage,
    Cosmos3MultiviewLatentStage,
    Cosmos3MultiviewTokenizationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3_sim_auto import (
    Cosmos3SimAutoRolloutStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class Cosmos3NanoSimAutoPipeline(ComposedPipelineBase):
    """Per-camera WSM clips + captions (+ HD-map range maps) to camera videos, one chunk at a time."""

    pipeline_name = "Cosmos3NanoSimAutoPipeline"
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
        2. Cosmos3MultiviewTokenizationStage - per-camera captions with the AV system prompt
        3. Cosmos3MultiviewLatentStage - per-camera VAE encode, anchors, LiDAR encode
        4. Cosmos3SimAutoRolloutStage - chunk-causal fixed-step SDE with the KV ring
        5. Cosmos3MultiviewDecodingStage - per-camera VAE decode, LiDAR decode on request
        """
        pipeline_config = server_args.pipeline_config
        if not isinstance(pipeline_config, Cosmos3SimAutoConfig):
            raise TypeError(
                "Cosmos3NanoSimAutoPipeline requires Cosmos3SimAutoConfig, got "
                f"{type(pipeline_config).__name__}; make the checkpoint's model_index.json "
                "name Cosmos3NanoSimAutoPipeline."
            )
        manifest = pipeline_config.sim_auto_manifest
        deployment = pipeline_config.multiview_deployment
        if manifest is None or deployment is None:
            raise ValueError(
                "Cosmos3SimAutoConfig did not resolve the checkpoint's artifact; "
                "model_path must be set before stage creation."
            )
        transformer = self.get_module("transformer")
        if not isinstance(transformer, Cosmos3SimAutoTransformer):
            raise TypeError(
                "Cosmos3NanoSimAutoPipeline loaded the wrong transformer type: "
                f"{type(transformer).__name__}."
            )
        vae = self.get_module("vae")
        requested = pipeline_config.resolved_multiview_backend()
        backend = (
            resolve_masked_backend(get_local_torch_device())
            if requested == COSMOS3_MULTIVIEW_AUTO_BACKEND
            else requested
        )
        logger.info(
            "Cosmos3-Nano-Sim-Auto attention backend: %s (%s)",
            backend,
            "device default"
            if requested == COSMOS3_MULTIVIEW_AUTO_BACKEND
            else "pinned",
        )
        lidar_encoder = None
        lidar_decoder = None
        if deployment.supports_lidar:
            if envs.SGLANG_DIFFUSION_COSMOS3_LIDAR_USE_NATTEN:
                if importlib.util.find_spec("natten") is None:
                    raise ImportError(
                        "SGLANG_DIFFUSION_COSMOS3_LIDAR_USE_NATTEN=1 but the natten package is not "
                        "installed; unset it to use the built-in FlexAttention neighborhood attention."
                    )
                logger.info("Cosmos3 LiDAR neighborhood attention: natten na2d")
            else:
                logger.info(
                    "Cosmos3 LiDAR neighborhood attention: compiled FlexAttention (fp32)"
                )
            # The LiDAR VAE is not a model_index component; it ships as lidar_vae/
            # next to the transformer and stays in FP32.
            lidar_encoder = Cosmos3LidarEncoder.from_pretrained(
                self.model_path, deployment.lidar, get_local_torch_device()
            )
            lidar_decoder = Cosmos3LidarDecoder.from_pretrained(
                self.model_path, deployment.lidar, get_local_torch_device()
            )
        self.add_stage(Cosmos3MultiviewInputStage(deployment))
        self.add_stage(
            Cosmos3MultiviewTokenizationStage(
                tokenizer=self.get_module("text_tokenizer"), deployment=deployment
            )
        )
        # The latent stage's attention backend only names the bidirectional
        # layout it records; the rollout plans its own replay mask.
        self.add_stage(
            Cosmos3MultiviewLatentStage(
                vae=vae,
                transformer=transformer,
                deployment=deployment,
                lidar_encoder=lidar_encoder,
                attention_backend="triton",
            )
        )
        self.add_stage(
            Cosmos3SimAutoRolloutStage(
                transformer=transformer, manifest=manifest, attention_backend=backend
            )
        )
        self.add_stage(
            Cosmos3MultiviewDecodingStage(
                vae, guardrails=False, sound_tokenizer=None, lidar_decoder=lidar_decoder
            )
        )
        logger.info(
            "Cosmos3-Nano-Sim-Auto pipeline stages created (%s)", manifest.describe()
        )


EntryClass = [Cosmos3NanoSimAutoPipeline]
