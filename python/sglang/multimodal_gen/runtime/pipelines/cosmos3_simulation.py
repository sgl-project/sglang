# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 causal simulation and multiview pipelines."""

import importlib.util

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.configs.pipeline_configs.cosmos3_multiview import (
    COSMOS3_MULTIVIEW_AUTO_BACKEND,
    Cosmos3MultiviewConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.cosmos3_sim_auto import (
    Cosmos3SimAutoConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.cosmos_dreams import (
    CosmosDreamsConfig,
    CosmosDreamsRealtimeConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.cosmos_dreams_transfer import (
    CosmosDreamsTransferConfig,
)
from sglang.multimodal_gen.configs.sample.cosmos_dreams import (
    CosmosDreamsSamplingParams,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.models.dits.cosmos3_multiview import (
    Cosmos3MultiviewTransformer,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos3_sim_auto import (
    Cosmos3SimAutoTransformer,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos3_simulation.multiview_attention import (
    resolve_masked_backend,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos_dreams import (
    CosmosDreamsTransformer,
)
from sglang.multimodal_gen.runtime.models.vaes.cosmos3_lidar.decoder import (
    Cosmos3LidarDecoder,
)
from sglang.multimodal_gen.runtime.models.vaes.cosmos3_lidar.encoder import (
    Cosmos3LidarEncoder,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3 import (
    Cosmos3DecodingStage,
    Cosmos3DenoisingStage,
    Cosmos3TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3_simulation.dreams import (
    CosmosDreamsImageStage,
    CosmosDreamsPrepareStage,
    CosmosDreamsRolloutStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3_simulation.multiview import (
    Cosmos3MultiviewDecodingStage,
    Cosmos3MultiviewInputStage,
    Cosmos3MultiviewLatentStage,
    Cosmos3MultiviewTokenizationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3_simulation.realtime import (
    CosmosDreamsCausalDecodingStage,
    CosmosDreamsRealtimePrepareStage,
    CosmosDreamsRealtimeRolloutStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3_simulation.sim_auto import (
    Cosmos3SimAutoRolloutStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3_simulation.transfer import (
    CosmosDreamsControlVideoStage,
    CosmosDreamsTransferDecodingStage,
    CosmosDreamsTransferPrepareStage,
    CosmosDreamsTransferRolloutStage,
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

    def _causal_components(
        self, server_args: ServerArgs, config_type: type[CosmosDreamsConfig]
    ):
        pipeline_config = server_args.pipeline_config
        if not isinstance(pipeline_config, config_type):
            raise TypeError(
                f"{self.pipeline_name} requires {config_type.__name__}, got "
                f"{type(pipeline_config).__name__}."
            )
        transformer = self.get_module("transformer")
        if not isinstance(transformer, CosmosDreamsTransformer):
            raise TypeError(
                f"{self.pipeline_name} loaded the wrong transformer type: "
                f"{type(transformer).__name__}."
            )
        return (
            pipeline_config,
            transformer,
            transformer.manifest,
            self.get_module("vae"),
        )

    def _log_profile(self, manifest, profile) -> None:
        logger.info(
            "%s stages created (checkpoint %s, chunk_size=%d, denoise steps per chunk %s, %s)",
            self.pipeline_name,
            manifest.checkpoint_id,
            manifest.chunk_size,
            [len(schedule) for schedule in profile.frame_sigma_schedules],
            profile.describe(),
        )

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        pipeline_config, transformer, manifest, vae = self._causal_components(
            server_args, CosmosDreamsConfig
        )
        profile = pipeline_config.inference_profile(manifest)
        self.add_stage(CosmosDreamsImageStage(canvas_tier=pipeline_config.canvas_tier))
        self.add_stage(
            CosmosDreamsPrepareStage(
                vae=vae,
                tokenizer=self.get_module("text_tokenizer"),
                manifest=manifest,
                max_pixels=pipeline_config.max_pixels,
                profile=profile,
                model_path=server_args.model_path,
            )
        )
        self.add_stage(
            CosmosDreamsRolloutStage(
                transformer=transformer,
                scheduler=self.get_module("scheduler"),
                manifest=manifest,
                profile=profile,
                vae=vae,
                overlap_vae_decode=pipeline_config.overlap_vae_decode,
            )
        )
        self.add_stage(
            Cosmos3DecodingStage(vae, guardrails=False, sound_tokenizer=None)
        )
        self._log_profile(manifest, profile)


class Cosmos3NanoSimBimanualPipeline(CosmosDreamsPipeline):
    """Release ``_class_name`` of the Cosmos3-Nano-Sim-Bimanual export.

    The recipe-named ``CosmosDreamsPipeline`` stays registered for earlier
    exports; both resolve to the same native pipeline.
    """

    pipeline_name = "Cosmos3NanoSimBimanualPipeline"


class CosmosDreamsRealtimePipeline(CosmosDreamsPipeline):
    """One autoregressive block per realtime tick, offline rollout otherwise."""

    pipeline_name = "CosmosDreamsRealtimePipeline"
    # Registered together so the explicit pipeline class refines the model-default
    # config and the realtime adapter registry resolves this pipeline.
    pipeline_config_cls = CosmosDreamsRealtimeConfig
    sampling_params_cls = CosmosDreamsSamplingParams

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        pipeline_config, transformer, manifest, vae = self._causal_components(
            server_args, CosmosDreamsRealtimeConfig
        )
        self.add_stage(CosmosDreamsImageStage(canvas_tier=pipeline_config.canvas_tier))
        self.add_stage(
            CosmosDreamsRealtimePrepareStage(
                vae=vae,
                tokenizer=self.get_module("text_tokenizer"),
                manifest=manifest,
                max_pixels=pipeline_config.max_pixels,
            )
        )
        profile = pipeline_config.inference_profile(manifest)
        self.add_stage(
            CosmosDreamsRealtimeRolloutStage(
                transformer=transformer,
                scheduler=self.get_module("scheduler"),
                manifest=manifest,
                profile=profile,
            )
        )
        self.add_stage(
            CosmosDreamsCausalDecodingStage(vae, guardrails=False, sound_tokenizer=None)
        )
        self._log_profile(manifest, profile)


class Cosmos3NanoSimTransferPipeline(CosmosDreamsPipeline):
    """Control video + prompt to video, one chunk at a time."""

    pipeline_name = "Cosmos3NanoSimTransferPipeline"

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        pipeline_config, transformer, manifest, vae = self._causal_components(
            server_args, CosmosDreamsTransferConfig
        )
        self.add_stage(
            CosmosDreamsControlVideoStage(
                manifest=manifest,
                canvas_tier=pipeline_config.canvas_tier,
                max_pixels=pipeline_config.max_pixels,
                resize_mode=pipeline_config.transfer_resize_mode,
            )
        )
        self.add_stage(
            CosmosDreamsTransferPrepareStage(
                vae=vae,
                tokenizer=self.get_module("text_tokenizer"),
                manifest=manifest,
                emphasize_control_default=pipeline_config.emphasize_control_default,
            )
        )
        history_profile = pipeline_config.transfer_history_profile(manifest)
        self.add_stage(
            CosmosDreamsTransferRolloutStage(
                transformer=transformer,
                scheduler=self.get_module("scheduler"),
                manifest=manifest,
                history_profile=history_profile,
            )
        )
        self.add_stage(
            CosmosDreamsTransferDecodingStage(
                vae, guardrails=False, sound_tokenizer=None
            )
        )
        logger.info(
            "Cosmos-Dreams-Transfer pipeline stages created (checkpoint %s, hints %s, "
            "chunk_size=%d, control %s, %s history%s)",
            manifest.checkpoint_id,
            list(manifest.control_contract.hints),
            manifest.chunk_size,
            pipeline_config.transfer_resize_mode,
            history_profile.history_mode,
            (
                ""
                if history_profile.window_frames is None
                else f" of {history_profile.window_frames} latent frames, "
                f"{history_profile.sink_frames} pinned"
            ),
        )


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

    def initialize_pipeline(self, server_args: ServerArgs) -> None:
        deployment = server_args.pipeline_config.multiview_deployment
        if deployment is None:
            raise ValueError(
                "Cosmos3MultiviewConfig did not resolve the checkpoint's multiview "
                "deployment config; model_path must be set before stage creation."
            )
        requested = server_args.pipeline_config.resolved_multiview_backend()
        self.attention_backend = (
            resolve_masked_backend(get_local_torch_device())
            if requested == COSMOS3_MULTIVIEW_AUTO_BACKEND
            else requested
        )
        logger.info(
            "Cosmos3 multiview attention backend: %s (%s)",
            self.attention_backend,
            "device default"
            if requested == COSMOS3_MULTIVIEW_AUTO_BACKEND
            else "pinned",
        )
        if deployment.supports_lidar:
            if envs.SGLANG_DIFFUSION_COSMOS3_LIDAR_USE_NATTEN:
                if importlib.util.find_spec("natten") is None:
                    raise ImportError(
                        "SGLANG_DIFFUSION_COSMOS3_LIDAR_USE_NATTEN=1 but the natten "
                        "package is not installed; unset it to use the built-in "
                        "FlexAttention neighborhood attention, or install the wheel: "
                        "pip install natten==0.21.7 -f https://whl.natten.org"
                    )
                logger.info("Cosmos3 LiDAR neighborhood attention: natten na2d")
            else:
                logger.info(
                    "Cosmos3 LiDAR neighborhood attention: compiled FlexAttention "
                    "(fp32, IEEE); first LiDAR request compiles its three levels"
                )
            for name, model_cls in (
                ("lidar_encoder", Cosmos3LidarEncoder),
                ("lidar_decoder", Cosmos3LidarDecoder),
            ):
                device = (
                    "cpu"
                    if server_args.should_start_component_on_cpu(name)
                    else get_local_torch_device()
                )
                self.add_module(
                    name,
                    model_cls.from_pretrained(
                        self.model_path, deployment.lidar, device
                    ),
                )

    def add_multiview_input_stages(self, *, attention_backend: str) -> None:
        deployment = self.server_args.pipeline_config.multiview_deployment
        self.add_stage(Cosmos3MultiviewInputStage(deployment))
        self.add_stage(
            Cosmos3MultiviewTokenizationStage(
                tokenizer=self.get_module("text_tokenizer"), deployment=deployment
            )
        )
        self.add_stage(
            Cosmos3MultiviewLatentStage(
                vae=self.get_module("vae"),
                transformer=self.get_module("transformer"),
                deployment=deployment,
                lidar_encoder=self.get_module("lidar_encoder"),
                attention_backend=attention_backend,
            )
        )

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        if not isinstance(server_args.pipeline_config, Cosmos3MultiviewConfig):
            raise TypeError("Cosmos3MultiviewPipeline requires Cosmos3MultiviewConfig")
        transformer = self.get_module("transformer")
        if not isinstance(transformer, Cosmos3MultiviewTransformer):
            raise TypeError(
                "Cosmos3MultiviewPipeline requires Cosmos3MultiviewTransformer"
            )
        vae = self.get_module("vae")
        scheduler = self.get_module("scheduler")
        self.add_multiview_input_stages(attention_backend=self.attention_backend)
        self.add_stage(Cosmos3TimestepPreparationStage(scheduler))
        self.add_stage(
            Cosmos3DenoisingStage(
                transformer, scheduler, server_args=server_args, vae=vae
            )
        )
        self.add_stage(
            Cosmos3MultiviewDecodingStage(
                vae,
                guardrails=False,
                sound_tokenizer=None,
                lidar_decoder=self.get_module("lidar_decoder"),
            )
        )
        logger.info(
            "Cosmos3 multiview pipeline stages created (%d cameras, backend=%s, "
            "cross-view window=%.2f s, lidar=%s)",
            server_args.pipeline_config.multiview_deployment.num_views,
            self.attention_backend,
            server_args.pipeline_config.multiview_deployment.cross_view_past_window_seconds,
            server_args.pipeline_config.multiview_deployment.supports_lidar,
        )


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


EntryClass = [
    CosmosDreamsPipeline,
    Cosmos3NanoSimBimanualPipeline,
    CosmosDreamsRealtimePipeline,
    Cosmos3NanoSimTransferPipeline,
    Cosmos3MultiviewPipeline,
    Cosmos3NanoSimAutoPipeline,
]
