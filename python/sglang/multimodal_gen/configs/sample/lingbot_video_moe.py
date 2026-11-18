# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import (
    SamplingParams,
    resolve_sequence_shard,
)

DEFAULT_NEGATIVE_PROMPT = '{"universal_negative": {"visual_quality": ["low quality", "worst quality", "blurry", "pixelated", "jpeg artifacts", "low resolution", "unstable color", "color flicker", "underexposed", "overexposed", "invisible subject", "subject hidden in darkness"], "artistic_style": ["painting", "illustration", "drawing", "cartoon", "3d render", "cgi", "sketch", "digital art"], "composition_and_content": ["text", "watermark", "signature", "logo", "subtitles", "pillarboxed", "side bars", "portrait image in landscape frame"], "temporal_and_motion_stability": ["flickering", "jittery", "motion blur", "temporal inconsistency", "warping", "morphing", "incoherent motion", "unnatural movement", "static object with sudden jump", "frame-to-frame inconsistency"], "material_and_structure": ["plastic-like glass", "unrealistic texture", "deformed bottle", "liquid freezing improperly", "distorted reflections"]}}'


@dataclass
class LingBotVideoMoESamplingParams(SamplingParams):
    # prompt must be a structured-JSON caption; raw free-text is out-of-distribution.
    num_frames: int = 81
    height: int = 480
    width: int = 480
    fps: int = 16
    num_inference_steps: int = 40
    guidance_scale: float = 6.0
    flow_shift: float = 3.0
    negative_prompt: str | None = DEFAULT_NEGATIVE_PROMPT
    seed: int = 0

    def prepare_synthetic_warmup_request_for_queue(self, req, server_args) -> None:
        req.enable_sequence_shard = resolve_sequence_shard(
            server_args.pipeline_config, req.enable_sequence_shard
        )

    def _adjust_visual_fields(self, server_args, pipeline_config):
        super()._adjust_visual_fields(server_args, pipeline_config)
        sp_degree = server_args.sp_degree or 1
        if sp_degree <= 1:
            return
        if not self.enable_sequence_shard:
            raise ValueError(
                "LingBot-Video MoE requires sequence sharding when sp_degree is "
                "greater than 1, so leave enable_sequence_shard unset or true"
            )
        vae_config = pipeline_config.vae_config
        temporal = vae_config.arch_config.scale_factor_temporal
        spatial = vae_config.arch_config.scale_factor_spatial
        latent_frames = (
            (self.num_frames - 1) // temporal + 1
            if vae_config.use_temporal_scaling_frames
            else self.num_frames
        )
        pF, pH, pW = pipeline_config.dit_config.arch_config.patch_size
        n_video = (
            (latent_frames // pF)
            * (self.height // spatial // pH)
            * (self.width // spatial // pW)
        )
        if n_video % sp_degree != 0:
            raise ValueError(
                f"LingBot video token count ({n_video}) for {self.width}x"
                f"{self.height} with {self.num_frames} frames must be divisible "
                f"by sp_degree ({sp_degree})"
            )
