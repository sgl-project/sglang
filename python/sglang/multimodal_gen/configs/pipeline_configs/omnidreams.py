# SPDX-License-Identifier: Apache-2.0
"""Pipeline config for NVIDIA OmniDreams.

Phase 0 wires the static structure (DiT config, VAE reuse, task type) and the
2-step flow-match sigma schedule. The denoising/decoding callbacks used at GPU
time are added in later phases.
"""

import json
from dataclasses import dataclass, field
from typing import Any

from sglang.multimodal_gen.configs.models.dits.omnidreams import OmniDreamsDiTConfig
from sglang.multimodal_gen.configs.models.omnidreams_components import (
    NativeAccelerationMode,
    OmniDreamsTextEncoderConfig,
    OmniDreamsVAEDecoderConfig,
    OmniDreamsVAEEncoderConfig,
    normalize_native_acceleration_mode,
)
from sglang.multimodal_gen.configs.models.vaes.wanvae import OmniDreamsVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)


def warp_flow_match_sigmas(
    denoising_timesteps: tuple[int, ...] = (1000, 450),
    flow_shift: float = 5.0,
    sigma_min: float = 0.0,
) -> list[float]:
    """OmniDreams 2-step flow-match sigma schedule.

    Each raw timestep ``t`` maps to ``s = t / 1000`` then is warped by
    ``shift*s / (1 + (shift-1)*s)``; ``sigma_min`` is appended as the final
    target. With the distilled defaults this yields ``[1.0, 0.8036, 0.0]``.
    """
    sigmas = [
        flow_shift * (t / 1000.0) / (1.0 + (flow_shift - 1.0) * (t / 1000.0))
        for t in denoising_timesteps
    ]
    sigmas.append(sigma_min)
    return sigmas


@dataclass
class OmniDreamsPipelineConfig(PipelineConfig):
    task_type: ModelTaskType = ModelTaskType.I2V
    # CFG disabled for the distilled checkpoint.
    should_use_guidance: bool = False
    # Native bf16 DiT; VAE in bf16. Measured vs fp32: ~1.5-1.9 GiB lower peak
    # VRAM, 20-27% faster decode, no quality regression (AR causal feature
    # cache stays numerically continuous across chunks; 13f rest-mean 53.4 vs
    # 52.6). Set "fp32" if you need bit-exact VAE numerics.
    dit_precision: str = "bf16"
    vae_precision: str = "bf16"
    # Flow-match warp shift (also drives warp_flow_match_sigmas).
    flow_shift: float | None = 5.0

    dit_config: OmniDreamsDiTConfig = field(default_factory=OmniDreamsDiTConfig)
    # A.5: OmniDreams uses a Cosmos-Predict2.5-based latent space; the
    # latents_mean/std defaults match Wan 2.1 (safe fallback). Override in
    # OmniDreamsVAEArchConfig once GPU validation confirms the correct values.
    vae_config: OmniDreamsVAEConfig = field(default_factory=OmniDreamsVAEConfig)

    # 2-step distilled flow-match schedule.
    denoising_timesteps: tuple[int, ...] = (1000, 450)
    sigma_min: float = 0.0

    # Capture the steady-state AR-rollout DiT calls into a CUDA graph and
    # replay (eliminates per-launch CPU overhead across the repeated
    # identical-shape calls). Numerically lossless. Default off until GPU
    # verification; env SGLANG_OMNIDREAMS_CUDA_GRAPH force-enables. The fill-phase
    # chunks (before the KV window is steady) always run eager.
    enable_cuda_graph: bool = False
    # Eager warmup iterations before graph capture (drains lazy allocs +
    # torch.compile autotune when --enable-torch-compile is also on).
    cuda_graph_warmup_iters: int = 2

    # ===== Unified nested Config architecture (replaces flat bool fields) =====
    text_encoder_config: OmniDreamsTextEncoderConfig | None = field(
        default_factory=OmniDreamsTextEncoderConfig
    )
    image_encoder_config: OmniDreamsVAEEncoderConfig | None = field(
        default_factory=lambda: OmniDreamsVAEEncoderConfig(impl="wanvae")
    )
    encoder_config: OmniDreamsVAEEncoderConfig = field(
        default_factory=lambda: OmniDreamsVAEEncoderConfig(impl="wanvae")
    )
    decoder_config: OmniDreamsVAEDecoderConfig = field(
        default_factory=lambda: OmniDreamsVAEDecoderConfig(impl="wanvae")
    )

    # DiT FP8 acceleration mode (kept as flat fields, not a nested Config).
    # Phase 1 honors ``disabled`` (eager bf16, default) and ``weight_only_fp8``
    # (dequantize pre-quantized FP8 → bf16, eager PyTorch forward).
    # Phase 2 adds ``fp8_compute``: swap the DiT linears to FP8-compute via
    # ``torch._scaled_mm`` (per-channel weight + per-token activation) and route
    # self-attn through the sage3 Blackwell kernel when
    # ``omnidreams_attn_backend="sage3"``. Falls back to eager bf16 on non-FP8
    # HW (CPU) or when ``sageattn3`` is unavailable.
    # ``auto``/``required`` are accepted as inert back-compat aliases mapped to
    # ``disabled``/``weight_only_fp8`` (see __post_init__).
    native_dit_acceleration: NativeAccelerationMode = "disabled"
    # Explicit path to pre-quantized FP8 DiT weights (.pt from the offline
    # exporter).  When None the DenoisingStage infers a default alongside the
    # raw checkpoint (omnidreams_fp8_dit.pt).
    native_dit_fp8_prepared_path: str | None = None
    # Phase 2: self-attention backend for the AR DiT ("sdpa" | "sage3"). Only
    # applies when ``native_dit_acceleration="fp8_compute"`` (sage3 is most
    # useful alongside FP8-compute). Cross-attn always uses sdpa. Overridden by
    # env ``SGLANG_OMNIDREAMS_ATTN_BACKEND``. Default "sdpa".
    omnidreams_attn_backend: str = "sdpa"

    def __post_init__(self):
        """Detect removed fields and guide migration to new Config structure."""
        self._rehydrate_component_configs()

        removed_fields = {
            "use_light_vae_encoder": "encoder_config.impl / image_encoder_config.impl",
            "light_vae_path": "encoder_config.checkpoint_path / image_encoder_config.checkpoint_path",
            "use_light_tae": "decoder_config.impl",
            "light_tae_path": "decoder_config.checkpoint_path",
            "use_fp8_dit": "native_dit_acceleration",
            "fp8_dit_attention_backend": "native_dit_acceleration",
            "use_light_vae_fp8": "encoder_config.native_acceleration",
            "light_vae_fp8_state_path": "encoder_config.native_acceleration",
        }
        for old, new in removed_fields.items():
            if hasattr(self, old):
                raise ValueError(
                    f"OmniDreamsPipelineConfig field '{old}' has been removed. "
                    f"Use '{new}' instead. See docs/omnidreams_config_migration.md"
                )

        # Normalize + validate the DiT acceleration mode. ``auto``/``required``
        # are accepted as inert back-compat aliases (mapped with a warning) so
        # shipped JSON configs keep loading.
        self.native_dit_acceleration = normalize_native_acceleration_mode(
            self.native_dit_acceleration
        )

    def denoising_sigmas(self) -> list[float]:
        return warp_flow_match_sigmas(
            self.denoising_timesteps,
            self.flow_shift if self.flow_shift is not None else 5.0,
            self.sigma_min,
        )

    # Fields whose JSON dicts must be intercepted BEFORE the base
    # ``update_pipeline_config`` loop so they aren't mistaken for ``ModelConfig``.
    # Single source of truth: name -> dataclass. Used both to intercept JSON
    # overrides (keys) and to rehydrate raw dicts back into dataclasses (values).
    _COMPONENT_CONFIG_FIELDS = {
        "text_encoder_config": OmniDreamsTextEncoderConfig,
        "image_encoder_config": OmniDreamsVAEEncoderConfig,
        "encoder_config": OmniDreamsVAEEncoderConfig,
        "decoder_config": OmniDreamsVAEDecoderConfig,
    }

    def _rehydrate_component_configs(self) -> None:
        """Convert dict-valued component configs back into their dataclasses.

        The base ``update_pipeline_config`` (used by ``--pipeline-config-path``
        JSON) only recurses into ``ModelConfig`` fields; the OmniDreams component
        configs are plain dataclasses, so a JSON override lands as a raw ``dict``
        and breaks ``.setup()``. Rehydrate them here (``__post_init__`` runs after
        the JSON merge), so a JSON like ``{"encoder_config": {"impl": "lightvae"}}``
        yields a real ``OmniDreamsVAEEncoderConfig``.
        """
        for name, cls in self._COMPONENT_CONFIG_FIELDS.items():
            value = getattr(self, name)
            if isinstance(value, dict):
                setattr(self, name, cls(**value))

    def load_from_json(self, file_path: str):
        """Load config from JSON, protecting component-config fields."""
        with open(file_path) as f:
            input_pipeline_dict = json.load(f)
        self._update_pipeline_config_with_component_protection(input_pipeline_dict)

    def update_pipeline_config(self, source_pipeline_dict: dict[str, Any]) -> None:
        """Override to protect component-config fields from the base-class
        ``ModelConfig`` recursion (these are plain dataclasses, not ModelConfigs)."""
        self._update_pipeline_config_with_component_protection(source_pipeline_dict)

    def _update_pipeline_config_with_component_protection(
        self, source_pipeline_dict: dict[str, Any]
    ) -> None:
        """Pop component-config dicts, delegate everything else to base, then rehydrate."""
        component_overrides: dict[str, Any] = {}
        for key in self._COMPONENT_CONFIG_FIELDS:
            if key in source_pipeline_dict:
                component_overrides[key] = source_pipeline_dict.pop(key)

        # Let base-class loop handle the remaining (e.g. dit_config, vae_config).
        super().update_pipeline_config(source_pipeline_dict)

        # Apply component-config overrides as raw dicts; _rehydrate picks them up.
        for key, value in component_overrides.items():
            setattr(self, key, value)

        self._rehydrate_component_configs()
