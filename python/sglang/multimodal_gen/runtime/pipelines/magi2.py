# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from sglang.multimodal_gen.configs.pipeline_configs.magi2 import Magi2PipelineConfig
from sglang.multimodal_gen.configs.sample.magi2 import Magi2SamplingParams
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.loader.component_loaders.magi2_loader import (
    Magi2AudioVAELoader,
    Magi2PreviewDiTLoader,
    Magi2RefinerDiTLoader,
    Magi2TextEncoderLoader,
    Magi2TokenizerLoader,
    Magi2TurboVAELoader,
    Magi2VideoVAELoader,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency_strategies import (
    ComponentResidencyStrategy,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.magi2 import (
    Magi2DecodingStage,
    Magi2DenoisingStage,
    Magi2ImageEncodingStage,
    Magi2InputStage,
    Magi2LatentPreparationStage,
    Magi2PackingStage,
    Magi2StageHandoffStage,
    Magi2TextEncodingStage,
    Magi2TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model

# The checkpoint has no model_index.json and is not a diffusers tree: each
# component sits in its own directory, named apart from its module.
_COMPONENT_DIRS: dict[str, str] = {
    "transformer": "preview",
    "transformer_2": "refiner",
    "text_encoder": "text_encoder",
    "tokenizer": "text_encoder",
    "vae": "vae",
    "turbo_vae": "turbo_vae",
    "audio_vae": "stable-audio-open-1.0",
}


class Magi2Pipeline(ComposedPipelineBase):
    pipeline_name = "Magi2Pipeline"
    is_video_pipeline = True
    pipeline_config_cls = Magi2PipelineConfig
    sampling_params_cls = Magi2SamplingParams

    _required_config_modules = list(_COMPONENT_DIRS)
    component_loaders = {
        "transformer": Magi2PreviewDiTLoader,
        "transformer_2": Magi2RefinerDiTLoader,
        "text_encoder": Magi2TextEncoderLoader,
        "tokenizer": Magi2TokenizerLoader,
        "vae": Magi2VideoVAELoader,
        "turbo_vae": Magi2TurboVAELoader,
        "audio_vae": Magi2AudioVAELoader,
    }

    def validate_disagg_role(self, role: RoleType) -> None:
        # add_stage silently drops role_affinity mismatches, yielding an empty pipeline.
        if role != RoleType.MONOLITHIC:
            raise ValueError(
                "Magi2Pipeline only supports monolithic deployment; "
                f"disaggregation role {role.value!r} is not supported"
            )

    def _load_config(self) -> dict[str, Any]:
        self.model_path = maybe_download_model(
            self.model_path, revision=self.server_args.revision
        )
        config = self.server_args.pipeline_config
        return {
            "_class_name": self.pipeline_name,
            "_diffusers_version": "0.36.0",
            "transformer": ["diffusers", "Magi2PreviewDiT"],
            # null drops a disabled tier from the required modules.
            "transformer_2": (
                ["diffusers", "Magi2RefinerDiT"] if config.enable_refiner else None
            ),
            "text_encoder": ["transformers", "Qwen3_5TextModel"],
            "tokenizer": ["transformers", "AutoTokenizer"],
            "vae": ["diffusers", "AutoencoderKLWan"],
            "turbo_vae": (
                ["diffusers", "Magi2TurboVAE"] if config.use_turbo_vae else None
            ),
            "audio_vae": ["diffusers", "Magi2AudioVAE"],
        }

    def _resolve_component_path(self, server_args, module_name, load_module_name):
        return super()._resolve_component_path(
            server_args, module_name, _COMPONENT_DIRS[module_name]
        )

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        config = server_args.pipeline_config

        self.add_stage(Magi2InputStage())
        # Image before text: reversed, every image conditions on a zero vector.
        self.add_stage(
            Magi2ImageEncodingStage(
                vae=self.get_module("vae"),
                spatial_compression_ratio=(
                    config.vae_config.arch_config.spatial_compression_ratio
                ),
                latents_mean=config.vae_config.arch_config.latents_mean,
                latents_std=config.vae_config.arch_config.latents_std,
            )
        )
        # The no-op base strategy keeps the encoder where the loader put it: on the
        # CPU, which is where Magi2TextEncodingStage runs its forward.
        self.component_residency_strategies["text_encoder"] = (
            ComponentResidencyStrategy()
        )
        self.add_stage(
            Magi2TextEncodingStage(
                text_encoder=self.get_module("text_encoder"),
                tokenizer=self.get_module("tokenizer"),
                tokenizer_kwargs=(
                    config.text_encoder_configs[0].arch_config.tokenizer_kwargs
                ),
                skip_layer=config.text_encoder_configs[0].arch_config.skip_layer,
            )
        )
        self.add_stage(Magi2LatentPreparationStage())
        self.add_stage(Magi2TimestepPreparationStage())

        self.add_stage(
            Magi2PackingStage(grid_key="magi2_preview_grid", conditions_on_images=True),
            stage_name="magi2_packing_preview",
        )
        self.add_stage(
            Magi2DenoisingStage(
                transformer=self.get_module("transformer"), pipeline=self
            ),
            stage_name="magi2_denoising_preview",
        )

        if config.enable_refiner:
            self.add_stage(Magi2StageHandoffStage())
            self.add_stage(
                Magi2PackingStage(grid_key="magi2_refiner_grid", refiner_only=True),
                stage_name="magi2_packing_refiner",
            )
            self.add_stage(
                Magi2DenoisingStage(
                    transformer=self.get_module("transformer_2"),
                    pipeline=self,
                    guidance_key="refiner",
                    refiner_only=True,
                ),
                stage_name="magi2_denoising_refiner",
            )

        vae_arch = config.vae_config.arch_config
        self.add_stage(
            Magi2DecodingStage(
                video_vae=self.get_module("vae"),
                audio_vae=self.get_module("audio_vae"),
                turbo_vae=self.get_module("turbo_vae"),
                latents_mean=vae_arch.latents_mean,
                latents_std=vae_arch.latents_std,
            )
        )


EntryClass = Magi2Pipeline
