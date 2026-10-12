# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import numpy as np
import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.configs.models.dits.llada_image import (
    LLaDAImageDitConfig,
    editing_rope_rows,
)
from sglang.multimodal_gen.configs.models.vaes.flux import Flux2VAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    SpatialImagePipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.flux import _unpatchify_latents


def flux2_vae_bn_stats(vae, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(like)
    variance = vae.bn.running_var.view(1, -1, 1, 1)
    std = torch.sqrt(variance + vae.config.arch_config.batch_norm_eps).to(like)
    return mean, std


@dataclass
class LLaDAImagePipelineConfig(SpatialImagePipelineConfig):
    task_type: ModelTaskType = ModelTaskType.TI2I
    should_use_guidance: bool = False

    dit_config: LLaDAImageDitConfig = field(default_factory=LLaDAImageDitConfig)
    vae_config: Flux2VAEConfig = field(default_factory=Flux2VAEConfig)
    vae_precision: str = "bf16"
    vae_tiling: bool = False
    vae_sp: bool = False

    text_encoder_configs: tuple = ()
    text_encoder_precisions: tuple[str, ...] = ()
    preprocess_text_funcs: tuple = ()
    postprocess_text_funcs: tuple = ()

    latent_scale_factor: int = 16
    # Editing feeds the source to SigVQ at half size, and its patch is 16 pixels.
    editing_size_multiple: int = 32
    # The embedded text worker admits 8192 prefill tokens shared by the two
    # CFG sequences and each sequence appends 256 query tokens.
    max_request_text_tokens: int = 3584

    def prepare_sigmas(self, sigmas, num_inference_steps):
        if sigmas is not None:
            return sigmas
        schedule = np.linspace(0.001, 1.0, num_inference_steps + 1)[:-1]
        schedule = (1 - (1 - schedule**1.17) ** 0.8) ** 1.1
        return (1 - schedule).tolist()

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        del num_frames
        self.validate_output_size(
            batch.width, batch.height, editing=batch.condition_image is not None
        )
        return (
            batch_size,
            self.dit_config.num_channels_latents,
            batch.height // self.latent_scale_factor,
            batch.width // self.latent_scale_factor,
        )

    def prepare_calculated_size(self, image):
        del image
        return None

    def calculate_condition_image_size(self, image, width, height):
        del image, width, height
        return None

    def validate_server_args(self, server_args) -> None:
        super().validate_server_args(server_args)
        if any(
            value != 1
            for value in (
                server_args.num_gpus,
                server_args.sp_degree,
                server_args.tp_size,
                server_args.dp_size,
                server_args.cfg_parallel_degree,
            )
        ):
            raise ValueError("LLaDA-Image currently supports single-GPU serving")
        if server_args.quantization is not None:
            raise ValueError("LLaDA-Image currently supports BF16 checkpoints")
        if envs.SGLANG_CACHE_DIT_ENABLED:
            raise ValueError("LLaDA-Image does not support cache-dit")
        if server_args.text_encoder_cpu_offload:
            if server_args.is_arg_explicitly_set("text_encoder_cpu_offload"):
                raise ValueError("LLaDA-Image requires a resident text encoder")
            server_args.text_encoder_cpu_offload = False
        for attribute in ("layerwise_offload_components", "cpu_offload_components"):
            components = getattr(server_args, attribute)
            if components is not None and "text_encoder" in components:
                if server_args.is_arg_explicitly_set(attribute):
                    raise ValueError("LLaDA-Image requires a resident text encoder")
                setattr(
                    server_args,
                    attribute,
                    [name for name in components if name != "text_encoder"],
                )
        if server_args.explicit_residency_mode("text_encoder") not in (
            None,
            "resident",
        ):
            raise ValueError("LLaDA-Image requires a resident text encoder")

    def supports_disaggregation(self) -> bool:
        return False

    def validate_output_size(self, width: int, height: int, editing: bool) -> None:
        multiple = self.editing_size_multiple if editing else self.latent_scale_factor
        if width % multiple != 0 or height % multiple != 0:
            task = "editing" if editing else "generation"
            raise ValueError(
                f"LLaDA-Image {task} width and height must be divisible by "
                f"{multiple}, got {width}x{height}"
            )
        self._validate_spatial_rope_bounds(width, height)
        if editing:
            # Lower bound without the prompt, the DiT checks the exact length.
            image_tokens = (width // self.latent_scale_factor) * (
                height // self.latent_scale_factor
            )
            sigvq_tokens = (width // self.editing_size_multiple) * (
                height // self.editing_size_multiple
            )
            rows = editing_rope_rows(0, [image_tokens, image_tokens], sigvq_tokens)
            limit = self.dit_config.arch_config.axes_lens[0]
            if rows > limit:
                raise ValueError(
                    f"LLaDA-Image editing at {width}x{height} needs at least {rows} "
                    f"sequence positions, above the model limit of {limit}"
                )

    def _validate_spatial_rope_bounds(self, width: int, height: int) -> None:
        # The served DiT uses patch size 1.
        axes_lens = self.dit_config.arch_config.axes_lens
        max_height = axes_lens[1] * self.latent_scale_factor
        max_width = axes_lens[2] * self.latent_scale_factor
        if height > max_height or width > max_width:
            raise ValueError(
                f"LLaDA-Image output size {width}x{height} exceeds spatial RoPE "
                f"bounds of {max_width}x{max_height} pixels"
            )

    @staticmethod
    def _cond_kwargs(batch, device, dtype, negative: bool) -> dict:
        def to_device(values):
            if values is None:
                return None
            return [value.to(device=device, dtype=dtype) for value in values]

        image_embeds = batch.image_embeds
        if image_embeds:
            image_embeds = to_device(image_embeds)
            if negative:
                # The unconditional edit keeps the source latents without SigVQ.
                empty = image_embeds[0].new_zeros((0, image_embeds[0].shape[-1]))
                image_embeds = [empty] * batch.batch_size
        return {
            "encoder_hidden_states_image": image_embeds,
            "source_latents": to_device(batch.source_latents),
        }

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return self._cond_kwargs(batch, device, dtype, negative=False)

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return self._cond_kwargs(batch, device, dtype, negative=True)

    def get_decode_scale_and_shift(self, device, dtype, vae):
        del device, dtype, vae
        return 1.0, None

    def preprocess_decoding(self, latents, server_args=None, vae=None):
        vae_parameter = next(vae.parameters())
        latents = latents.to(device=vae_parameter.device, dtype=vae_parameter.dtype)
        mean, std = flux2_vae_bn_stats(vae, latents)
        return _unpatchify_latents(latents * std + mean)

    def post_denoising_loop(self, latents, batch):
        del batch
        return latents


def register():
    from sglang.multimodal_gen.configs.sample.llada_image import (
        LLaDAImageSamplingParams,
        LLaDAImageTurboSamplingParams,
    )
    from sglang.multimodal_gen.registry import register_configs

    # Base and Turbo checkpoints share every class name, so Turbo registers
    # first and the first matching detector wins.
    register_configs(
        sampling_param_cls=LLaDAImageTurboSamplingParams,
        pipeline_config_cls=LLaDAImagePipelineConfig,
        hf_model_paths=["inclusionAI/LLaDA-Image-Turbo"],
        model_detectors=[
            lambda hf_id: "llada-image" in hf_id.lower() and "turbo" in hf_id.lower()
        ],
    )
    register_configs(
        sampling_param_cls=LLaDAImageSamplingParams,
        pipeline_config_cls=LLaDAImagePipelineConfig,
        hf_model_paths=["inclusionAI/LLaDA-Image"],
        model_detectors=[
            lambda hf_id: (
                (
                    "lladaimagepipeline" in hf_id.lower()
                    or "llada-image" in hf_id.lower()
                )
                and "turbo" not in hf_id.lower()
            )
        ],
    )
