# SPDX-License-Identifier: Apache-2.0
"""Cosmos-Dreams pipeline configuration.

Cosmos-Dreams checkpoints are distilled Cosmos3 Omni checkpoints with a
``cosmos3_nano_sim_bimanual`` artifact in ``transformer/config.json``. The config reuses
``Cosmos3Config`` (Wan VAE, tokenizer, distilled sigmas) and swaps in the
causal transformer class.
"""

import math
from dataclasses import dataclass, replace
from typing import Any

from sglang.multimodal_gen.configs.models.dits.cosmos_dreams import (
    ACTION_CONDITIONING_MODE,
    HISTORY_MODE_FULL,
    CosmosDreamsInferenceProfile,
    CosmosDreamsManifest,
    load_cosmos_dreams_manifest,
    resolve_inference_profile,
)
from sglang.multimodal_gen.configs.pipeline_configs.cosmos3 import (
    Cosmos3Config,
    _transformer_config,
)
from sglang.multimodal_gen.configs.pipeline_configs.model_deployment_config import (
    ModelDeploymentConfig,
)

# Deployment default from the reference Cosmos-Dreams deployment: 720x1280.
COSMOS_DREAMS_MAX_PIXELS = 921_600
# Training rolled out whole 900-frame clips with an unbounded K/V cache; 901
# pixel frames (1 + 4 * 225) cover them at the 4x temporal compression.
COSMOS_DREAMS_HISTORY_MAX_FRAMES = 901


@dataclass
class CosmosDreamsConfig(Cosmos3Config):
    transformer_class_override: str | None = "CosmosDreamsTransformer"

    # Conditioning contract this pipeline drives; the Transfer subclass overrides it.
    conditioning_mode: str = ACTION_CONDITIONING_MODE

    # The Dreams stages own prompt formatting (training-time JSON caption); the
    # Cosmos3 system prompt and duration suffix were not used in training.
    use_duration_template: bool = False
    use_system_prompt: bool = False

    # Largest canvas admitted per request (height * width).
    max_pixels: int = COSMOS_DREAMS_MAX_PIXELS

    # Resolution tier whose trained canvases are used when a request leaves
    # height/width unset. The Sim-Bimanual checkpoint trained on the 480 tier only.
    canvas_tier: str = "480"

    # Latent frame contract of the rollout, refreshed from the transformer
    # manifest once the checkpoint is known; adjust_num_frames rounds to it.
    chunk_size: int = 4
    temporal_compression_factor: int = 4

    # Sigma schedule per chunk, picked by the chunk's first latent frame (the last entry
    # repeats). None applies the step42 budget the checkpoints were distilled with: every
    # artifact sigma for a chunk starting at latent frame 0, sigmas 0 and 2 afterwards. Override
    # with --pipeline-config-path JSON, e.g. [[1.0, 0.9375, 0.8333, 0.625]] for four steps everywhere.
    frame_sigma_schedules: list[list[float]] | None = None
    # "full" keeps the K/V of the first history_max_frames pixel frames like training over
    # whole clips (the action exports set no training window; their window_frames is an
    # exporter default); "sliding" evicts beyond the artifact's window_frames. The Transfer
    # config defaults to "auto" and follows the training config instead.
    history_mode: str = HISTORY_MODE_FULL
    history_max_frames: int = COSMOS_DREAMS_HISTORY_MAX_FRAMES

    def update_config_from_dict(self, args, prefix: str = "") -> None:
        super().update_config_from_dict(args, prefix)
        if self.model_path:
            manifest = self._validate_checkpoint(self.model_path)
            self.chunk_size = manifest.chunk_size
            self.temporal_compression_factor = manifest.temporal_compression_factor
            self._validate_history_settings(manifest)

    def _validate_history_settings(self, manifest: CosmosDreamsManifest) -> None:
        # Surface bad deployment settings in the launcher, before weights load.
        self.inference_profile(manifest)

    def inference_profile(
        self, manifest: CosmosDreamsManifest
    ) -> CosmosDreamsInferenceProfile:
        """Rollout schedule and history settings the stages run with."""
        return resolve_inference_profile(
            manifest,
            frame_sigma_schedules=self.frame_sigma_schedules,
            history_mode=self.history_mode,
            history_max_frames=self.history_max_frames,
        )

    def adjust_num_frames(self, num_frames: int, *, log_adjustment: bool = True) -> int:
        # Latent frame 0 is the conditioning image, so the smallest video is one
        # generated latent frame; Cosmos3 rounds 1..4 pixel frames to a T2I frame.
        adjusted = super().adjust_num_frames(num_frames, log_adjustment=log_adjustment)
        return max(adjusted, 1 + self.temporal_compression_factor)

    def _validate_checkpoint(self, model_path: str) -> CosmosDreamsManifest:
        """Fail in the launcher before 30 GB of weights are loaded."""
        manifest = load_cosmos_dreams_manifest(_transformer_config(model_path))
        if manifest.conditioning_mode != self.conditioning_mode:
            raise ValueError(
                f"Checkpoint {manifest.checkpoint_id} is conditioned on "
                f"{manifest.conditioning_mode!r} but {type(self).__name__} drives "
                f"{self.conditioning_mode!r}; use Cosmos3NanoSimBimanualPipeline for action "
                "checkpoints and Cosmos3NanoSimTransferPipeline for control-video checkpoints."
            )
        if self.distilled_sigmas is None:
            raise ValueError(
                "Cosmos-Dreams requires a distilled fixed-step scheduler "
                "(FlowMatchEulerDiscreteScheduler with fixed_step_sampler_config.t_list)."
            )
        if len(self.distilled_sigmas) != len(manifest.t_list) or any(
            not math.isclose(scheduler_sigma, manifest_sigma, rel_tol=0.0, abs_tol=1e-8)
            for scheduler_sigma, manifest_sigma in zip(
                self.distilled_sigmas, manifest.t_list, strict=True
            )
        ):
            raise ValueError(
                "Cosmos-Dreams scheduler and transformer manifests define different "
                f"fixed-step schedules: scheduler={self.distilled_sigmas}, "
                f"transformer={list(manifest.t_list)}."
            )
        return manifest

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        # Distilled checkpoints run one conditional branch per step, so CFG
        # parallel would only duplicate the rollout on a second GPU.
        return replace(
            super().get_model_deployment_config(),
            auto_enable_cfg_parallel=False,
            supports_cfg_parallel=False,
        )

    def validate_server_args(self, server_args: Any) -> None:
        super().validate_server_args(server_args)
        # The launcher hands GPUs left over after dp/tp/cfg to sequence
        # parallelism; the causal transformer only runs a chunk on one rank.
        if int(server_args.sp_degree or 1) > 1:
            raise ValueError(
                f"{type(self).__name__} does not support sequence parallelism "
                f"(resolved sp_degree={server_args.sp_degree} for "
                f"num_gpus={server_args.num_gpus}). Use --dp-size <num_gpus> for "
                "independent replicas or --tp-size <num_gpus> to shard the transformer."
            )

    def supports_action_endpoint(self) -> bool:
        # Forward dynamics only: actions condition the video, none are produced.
        return False

    def supports_dynamic_batching(self) -> bool:
        return False
