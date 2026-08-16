# SPDX-License-Identifier: Apache-2.0
import os
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.magi2 import (
    Magi2PreviewConfig,
    Magi2RefinerConfig,
)
from sglang.multimodal_gen.configs.models.encoders.magi2 import Magi2TextEncoderConfig
from sglang.multimodal_gen.configs.models.vaes.magi2 import (
    Magi2AudioVAEConfig,
    Magi2TurboVAEConfig,
    Magi2VideoVAEConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.model_deployment_config import (
    ModelDeploymentConfig,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@dataclass
class Magi2PipelineConfig(PipelineConfig):
    """MAGI-2-preview: joint audio-video generation from one MoE checkpoint.

    A 40-layer MoE preview DiT and a 30-layer dense refiner DiT share the root
    checkpoint; the refiner is only reachable at 1080p.
    """

    task_type: ModelTaskType = ModelTaskType.TI2V

    # ref_image_type is "original", so the generic TI2V resize/centre-crop must not run.
    skip_input_image_preprocess: bool = True

    # No model_index.json, so nothing loads through sglang's diffusers loaders.
    native_only_components = (
        "text_encoder",
        "tokenizer",
        "transformer",
        "transformer_2",
        "vae",
        "turbo_vae",
        "audio_vae",
    )

    dit_config: Magi2PreviewConfig = field(default_factory=Magi2PreviewConfig)
    refiner_dit_config: Magi2RefinerConfig = field(default_factory=Magi2RefinerConfig)
    dit_precision: str = "bf16"

    vae_config: Magi2VideoVAEConfig = field(default_factory=Magi2VideoVAEConfig)
    vae_precision: str = "fp32"
    turbo_vae_config: Magi2TurboVAEConfig = field(default_factory=Magi2TurboVAEConfig)
    audio_vae_config: Magi2AudioVAEConfig = field(default_factory=Magi2AudioVAEConfig)
    audio_vae_precision: str = "fp32"

    text_encoder_configs: tuple[Magi2TextEncoderConfig, ...] = field(
        default_factory=lambda: (Magi2TextEncoderConfig(),)
    )
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16",))
    text_encoder_extra_args: list[dict] = field(default_factory=lambda: [{}])

    should_use_guidance: bool = True
    flow_shift: float | None = 7.0
    refiner_flow_shift: float = 5.0

    enable_refiner: bool = True

    # Index into the ZeroSNR discretization used to renoise the upsampled preview.
    refiner_renoise_index: int = 220

    use_turbo_vae: bool = True

    output_audio_sample_rate: int | None = 44100
    output_audio_channels: int | None = 2

    # Defaults to world size, and must divide moe_num_heads (12), so ep=8 is rejected;
    # the reference instead pads 12 heads to 16 to run ep=8.
    ep_size: int | None = None

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        return ModelDeploymentConfig(
            supports_cfg_parallel=False,
            auto_enable_cfg_parallel=False,
            keep_resident_components=("vae", "turbo_vae", "audio_vae"),
        )

    def supports_disaggregation(self) -> bool:
        return False

    def _warn_about_allocator(self) -> None:
        """Warn at startup when the allocator will fail the 1080p decode."""
        if not self.enable_refiner:
            return
        conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if "expandable_segments:True" in conf:
            return
        logger.warning(
            "[magi2] PYTORCH_CUDA_ALLOC_CONF does not enable expandable_segments "
            "(currently %r). The two-stage 1080p tier requests a single ~9 GiB "
            "block during decode while both DiTs are resident, which typically "
            "fails on fragmentation after several minutes of denoising. Set "
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True before launching, "
            "or run the preview-only tier.",
            conf or "unset",
        )

    def validate_server_args(self, server_args) -> None:
        super().validate_server_args(server_args)
        self._warn_about_allocator()

        if server_args.enable_cfg_parallel:
            raise ValueError(
                "MAGI-2's denoise loop runs both CFG branches itself; "
                "--enable-cfg-parallel would double-apply the guidance"
            )

        if server_args.tp_size > 1:
            raise ValueError(
                f"MAGI-2 has no tensor-parallel layers; --tp-size "
                f"({server_args.tp_size}) would leave those ranks duplicating "
                "work instead of sharding the sequence (measured 1.7x slower at "
                "tp=2, 4.9x at tp=4). Put the whole degree in --ulysses-degree."
            )

        if server_args.ring_degree > 1:
            raise ValueError(
                "MAGI-2 shards its packed sequence over the full SP group but "
                "exchanges attention heads over the Ulysses group only; "
                f"--ring-degree ({server_args.ring_degree}) must be 1. Put the "
                "whole degree in --ulysses-degree."
            )

        num_gpus = server_args.num_gpus
        ep_size = self.ep_size or num_gpus
        if num_gpus % ep_size:
            raise ValueError(
                f"num_gpus ({num_gpus}) must be divisible by ep_size ({ep_size})"
            )

        moe_heads = self.dit_config.arch_config.moe_num_heads
        if moe_heads % ep_size:
            raise ValueError(
                f"MAGI-2 splits {moe_heads} MoE heads across expert-parallel "
                f"ranks; ep_size ({ep_size}) must divide {moe_heads}"
            )

        # The refiner's 8 KV heads are the tightest axis: with it enabled only 1, 2
        # and 4 divide every axis, where preview-only also admits 3, 6 and 12.
        preview = self.dit_config.arch_config
        axes = {
            "preview attention heads": preview.num_attention_heads,
            "preview MoE heads": preview.moe_num_heads,
        }
        if self.enable_refiner:
            refiner = self.refiner_dit_config.arch_config
            axes["refiner attention heads"] = refiner.num_attention_heads
            axes["refiner KV heads"] = refiner.num_query_groups

        for name, count in sorted(axes.items(), key=lambda item: item[1]):
            if count % num_gpus:
                raise ValueError(
                    f"MAGI-2 shards {name} ({count}) across ranks; num_gpus "
                    f"({num_gpus}) must divide it. Valid counts are "
                    f"{[n for n in range(1, max(axes.values()) + 1) if all(c % n == 0 for c in axes.values())]}"
                )
