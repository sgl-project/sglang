# SPDX-License-Identifier: Apache-2.0
"""Sampling parameters for SANA-Video text-to-video generation."""

from __future__ import annotations

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.sample.easycache import EasyCacheParams
from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)


@dataclass
class SanaVideoSamplingParams(SamplingParams):
    """Defaults for the official SANA-Video 2B 480p Diffusers checkpoint."""

    data_type: DataType = DataType.VIDEO
    num_frames: int = 81
    guidance_scale: float = 6.0
    num_inference_steps: int = 50
    height: int | None = 480
    width: int | None = 832
    fps: int = 16
    negative_prompt: str = (
        "A chaotic sequence with misshapen, deformed limbs in heavy motion blur, "
        "sudden disappearance, jump cuts, jerky movements, rapid shot changes, "
        "frames out of sync, inconsistent character shapes, temporal artifacts, "
        "jitter, and ghosting effects, creating a disorienting visual experience."
    )

    # ``None`` inherits the deployment profile. The baseline profile resolves
    # it to False; SanaVideoOptimizedPipelineConfig resolves it to True.
    enable_easycache: bool | None = None
    easycache_params: EasyCacheParams = field(default_factory=EasyCacheParams)

    def __post_init__(self) -> None:
        if isinstance(self.easycache_params, dict):
            self.easycache_params = EasyCacheParams(**self.easycache_params)
        elif not isinstance(self.easycache_params, EasyCacheParams):
            raise TypeError(
                "easycache_params must be EasyCacheParams or a compatible dict, "
                f"got {type(self.easycache_params).__name__}"
            )
        super().__post_init__()

    def _adjust(self, server_args) -> None:
        super()._adjust(server_args)
        model_easycache_enabled = bool(
            server_args.pipeline_config.dit_config.enable_easycache
        )
        if self.enable_easycache is None:
            self.enable_easycache = model_easycache_enabled
        elif not isinstance(self.enable_easycache, bool):
            raise TypeError(
                f"enable_easycache must be bool or None, got {self.enable_easycache!r}"
            )
        if self.enable_easycache and not model_easycache_enabled:
            raise ValueError(
                "SANA-Video EasyCache must be enabled when the DiT is built. "
                "Set pipeline_config.dit_config.enable_easycache=True before "
                "creating the engine."
            )
        if self.enable_easycache and server_args.enable_cfg_parallel:
            raise ValueError(
                "SANA-Video EasyCache requires serial CFG because both CFG "
                "branches intentionally share one cached residual. Disable "
                "--enable-cfg-parallel or disable EasyCache."
            )
