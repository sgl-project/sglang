# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import numpy as np
import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.configs.models.dits.llada_image import LLaDAImageDitConfig
from sglang.multimodal_gen.configs.models.vaes.flux import Flux2VAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    SpatialImagePipelineConfig,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import get_sp_world_size
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@dataclass
class LLaDAImagePipelineConfig(SpatialImagePipelineConfig):
    task_type: ModelTaskType = ModelTaskType.TI2I
    should_use_guidance: bool = False

    dit_config: LLaDAImageDitConfig = field(default_factory=LLaDAImageDitConfig)
    dit_precision: str = "bf16"
    vae_config: Flux2VAEConfig = field(default_factory=Flux2VAEConfig)
    vae_precision: str = "bf16"
    vae_tiling: bool = False
    vae_sp: bool = False

    text_encoder_configs: tuple = ()
    text_encoder_precisions: tuple[str, ...] = ()
    preprocess_text_funcs: tuple = ()
    postprocess_text_funcs: tuple = ()

    latent_scale_factor: int = 16
    text_encoder_mem_fraction_static: float | None = None
    max_request_pixel_area: int = 2048 * 2048
    # The embedded text worker admits 8192 prefill tokens shared by the two
    # CFG sequences and each sequence appends 256 query tokens.
    max_request_text_tokens: int = 3584
    max_request_total_pixel_area: int = 10 * 1024 * 1024

    def prepare_sigmas(self, sigmas, num_inference_steps):
        if sigmas is not None:
            return sigmas
        schedule = np.linspace(0.001, 1.0, num_inference_steps + 1)[:-1]
        schedule = (1 - (1 - schedule**1.17) ** 0.8) ** 1.1
        return (1 - schedule).tolist()

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        del num_frames
        if batch.height % self.latent_scale_factor != 0:
            raise ValueError("LLaDA-Image height must be divisible by 16")
        if batch.width % self.latent_scale_factor != 0:
            raise ValueError("LLaDA-Image width must be divisible by 16")
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

    def shard_latents_for_sp(self, batch, latents):
        sp_degree = get_sp_world_size()
        if latents.dim() == 4 and latents.shape[2] % sp_degree != 0:
            raise ValueError(
                f"LLaDA-Image latent height {latents.shape[2]} must be divisible "
                f"by SP degree {sp_degree}. Choose a compatible output height"
            )
        return super().shard_latents_for_sp(batch, latents)

    def validate_server_args(self, server_args) -> None:
        super().validate_server_args(server_args)
        if envs.SGLANG_CACHE_DIT_ENABLED:
            raise ValueError("LLaDA-Image does not support SGLANG_CACHE_DIT_ENABLED")
        # Replicated condition suffixes are currently de-duplicated only by the
        # Ulysses attention path.
        if server_args.ring_degree != 1:
            raise ValueError("LLaDA-Image sequence parallelism requires ring_degree=1")
        if server_args.ulysses_degree != server_args.sp_degree:
            raise ValueError(
                "LLaDA-Image sequence parallelism requires ulysses_degree == sp_degree"
            )
        if server_args.sp_degree not in (1, 2):
            raise ValueError("LLaDA-Image currently supports only SP degrees 1 and 2")
        if (
            server_args.tp_size != 1
            or server_args.dp_size != 1
            or server_args.cfg_parallel_degree != 1
        ):
            raise ValueError("LLaDA-Image requires diffusion parallelism TP=DP=CFG=1")
        if server_args.num_gpus != server_args.sp_degree:
            raise ValueError(
                "LLaDA-Image requires num_gpus == sp_degree so every GPU belongs "
                "to the sequence-parallel group"
            )
        if server_args.text_encoder_cpu_offload:
            if server_args.is_arg_explicitly_set("text_encoder_cpu_offload"):
                raise ValueError(
                    "LLaDA-Image runs its text encoder inside an embedded srt "
                    "worker that the component residency manager cannot "
                    "offload. Remove --text-encoder-cpu-offload"
                )
            # Auto-tuned default, corrected here because the tuner does not
            # know about the embedded worker.
            server_args.text_encoder_cpu_offload = False
            logger.info(
                "LLaDA-Image keeps the embedded text encoder resident and "
                "ignores the auto-tuned text_encoder_cpu_offload"
            )
        if server_args.residency_mode("text_encoder") != "resident":
            if (
                server_args.explicit_residency_mode("text_encoder") is not None
                or server_args.is_arg_explicitly_set("layerwise_offload_components")
                or server_args.is_arg_explicitly_set("cpu_offload_components")
            ):
                raise ValueError(
                    "LLaDA-Image requires its embedded text encoder to remain resident"
                )
            # Strip the encoder from auto-tuned offload selections the tuner
            # applies without knowing about the embedded worker.
            if server_args.layerwise_offload_components is not None:
                server_args.layerwise_offload_components = [
                    name
                    for name in server_args.layerwise_offload_components
                    if name != "text_encoder"
                ]
            if server_args.cpu_offload_components is not None:
                server_args.cpu_offload_components = [
                    name
                    for name in server_args.cpu_offload_components
                    if name != "text_encoder"
                ]
            logger.info(
                "LLaDA-Image keeps the embedded text encoder resident and "
                "removed it from auto-tuned offload selections"
            )
            if server_args.residency_mode("text_encoder") != "resident":
                raise ValueError(
                    "LLaDA-Image requires its embedded text encoder to remain resident"
                )

        area_override = server_args.llada_image_max_pixel_area
        if area_override is not None and area_override <= 0:
            raise ValueError("llada_image_max_pixel_area must be positive")
        total_area_override = server_args.llada_image_max_total_pixel_area
        if total_area_override is not None and total_area_override <= 0:
            raise ValueError("llada_image_max_total_pixel_area must be positive")
        tokens_override = server_args.llada_image_max_text_tokens
        if tokens_override is not None and not (
            0 < tokens_override <= self.max_request_text_tokens
        ):
            raise ValueError(
                "llada_image_max_text_tokens must be positive and at most "
                f"{self.max_request_text_tokens}, the embedded worker budget"
            )

    # The embedded srt worker and its KV pool are invisible to generic module
    # discovery, so partial sleep or update would misreport.
    def supports_memory_release(self) -> bool:
        return False

    def supports_hot_weight_updates(self) -> bool:
        return False

    def validate_num_outputs_per_prompt(
        self, num_outputs_per_prompt: int, server_args
    ) -> None:
        if server_args.sp_degree > 1 and num_outputs_per_prompt != 1:
            raise ValueError(
                "LLaDA-Image sequence parallelism supports only n=1. "
                "submit separate requests for multiple images"
            )

    def validate_edit_source_count(self, source_count: int, server_args) -> None:
        del server_args
        if source_count != 1:
            raise ValueError("LLaDA-Image editing requires exactly one source image")

    def validate_request_sampling_params(self, sampling_params, server_args) -> None:
        enable_cache_dit = sampling_params.enable_cache_dit
        if enable_cache_dit is not None and enable_cache_dit is not False:
            raise ValueError("LLaDA-Image does not support enable_cache_dit")
        if sampling_params.cache_dit_params is not None:
            raise ValueError("LLaDA-Image does not support cache_dit_params")

        attention_backend = sampling_params.attention_backend_override
        supported_attention_backends = {"FA", "TORCH_SDPA"}
        if attention_backend is not None and (
            not isinstance(attention_backend, str)
            or attention_backend.upper() not in supported_attention_backends
        ):
            raise ValueError(
                "LLaDA-Image attention_backend_override must be fa or torch_sdpa"
            )

        cfg_gate_step = sampling_params.cfg_gate_step
        if cfg_gate_step is not None and (
            isinstance(cfg_gate_step, bool)
            or not isinstance(cfg_gate_step, (int, float))
            or not 0.0 <= cfg_gate_step <= 1.0
            or not np.isfinite(cfg_gate_step)
        ):
            raise ValueError("LLaDA-Image cfg_gate_step must be between 0.0 and 1.0")

        width = int(sampling_params.width or 0)
        height = int(sampling_params.height or 0)
        if width <= 0 or height <= 0:
            raise ValueError("LLaDA-Image requires positive width and height")
        if width % self.latent_scale_factor or height % self.latent_scale_factor:
            raise ValueError(
                f"LLaDA-Image width and height must be divisible by "
                f"{self.latent_scale_factor}"
            )
        sp_height_multiple = self.latent_scale_factor * server_args.sp_degree
        if height % sp_height_multiple:
            raise ValueError(
                f"LLaDA-Image output height must be divisible by "
                f"{sp_height_multiple} at SP degree {server_args.sp_degree}"
            )
        area_cap = (
            server_args.llada_image_max_pixel_area
            if server_args.llada_image_max_pixel_area is not None
            else self.max_request_pixel_area
        )
        if width * height > area_cap:
            raise ValueError(
                f"LLaDA-Image output area {width}x{height} exceeds the "
                f"supported maximum of {area_cap} pixels"
            )
        total_area_cap = (
            server_args.llada_image_max_total_pixel_area
            if server_args.llada_image_max_total_pixel_area is not None
            else self.max_request_total_pixel_area
        )
        num_outputs = sampling_params.num_outputs_per_prompt
        if num_outputs is not None:
            total_area = width * height * max(1, int(num_outputs))
            if total_area > total_area_cap:
                raise ValueError(
                    f"LLaDA-Image total output area {total_area} pixels across "
                    f"{num_outputs} outputs exceeds the supported maximum of "
                    f"{total_area_cap} pixels per request"
                )
        tokens_cap = (
            server_args.llada_image_max_text_tokens
            if server_args.llada_image_max_text_tokens is not None
            else self.max_request_text_tokens
        )
        self._validate_max_sequence_length(
            sampling_params.max_sequence_length, tokens_cap
        )
        extra_kwargs = sampling_params.diffusers_kwargs
        if isinstance(extra_kwargs, dict) and "max_sequence_length" in extra_kwargs:
            # prepare_request later overrides the request value from this channel.
            self._validate_max_sequence_length(
                extra_kwargs["max_sequence_length"], tokens_cap
            )

    @staticmethod
    def _validate_max_sequence_length(max_sequence_length, tokens_cap: int) -> None:
        if max_sequence_length is None:
            return
        if (
            isinstance(max_sequence_length, bool)
            or not isinstance(max_sequence_length, int)
            or max_sequence_length <= 0
        ):
            raise ValueError("max_sequence_length must be a positive integer")
        if max_sequence_length > tokens_cap:
            raise ValueError(
                f"max_sequence_length {max_sequence_length} exceeds the "
                f"embedded text encoder budget of {tokens_cap} tokens"
            )

    @staticmethod
    def _prepare_condition_list(values, name: str, expected_size: int, device, dtype):
        if values is None:
            return None
        if len(values) != expected_size:
            raise ValueError(
                f"LLaDA-Image {name} has {len(values)} entries, "
                f"expected {expected_size}"
            )
        return [value.to(device=device, dtype=dtype) for value in values]

    def _prepare_source_latents(self, batch, device, dtype):
        source_latents = self._prepare_condition_list(
            batch.source_latents,
            "source_latents",
            batch.batch_size,
            device,
            dtype,
        )
        if source_latents and getattr(batch, "did_sp_shard_latents", False):
            source_latents = [
                self.shard_latents_for_sp(batch, latent)[0] for latent in source_latents
            ]
        return source_latents

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        del rotary_emb
        image_embeds = batch.image_embeds
        if image_embeds:
            image_embeds = self._prepare_condition_list(
                image_embeds,
                "image_embeds",
                batch.batch_size,
                device,
                dtype,
            )
        source_latents = self._prepare_source_latents(batch, device, dtype)
        return {
            "encoder_hidden_states_image": image_embeds,
            "source_latents": source_latents,
        }

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        del rotary_emb
        image_embeds = batch.image_embeds
        if image_embeds:
            image_embeds = self._prepare_condition_list(
                image_embeds,
                "image_embeds",
                batch.batch_size,
                device,
                dtype,
            )
            empty_image_embed = image_embeds[0].new_zeros(
                (0, image_embeds[0].shape[-1])
            )
            image_embeds = [empty_image_embed] * batch.batch_size
        source_latents = self._prepare_source_latents(batch, device, dtype)
        return {
            "encoder_hidden_states_image": image_embeds,
            "source_latents": source_latents,
        }

    def get_decode_scale_and_shift(self, device, dtype, vae):
        del device, dtype, vae
        return 1.0, None

    def preprocess_decoding(self, latents, server_args=None, vae=None):
        del server_args
        if vae is None or not hasattr(vae, "bn"):
            raise ValueError("LLaDA-Image decoding requires the Flux2 VAE BN state")
        vae_parameter = next(vae.parameters())
        latents = latents.to(device=vae_parameter.device, dtype=vae_parameter.dtype)
        vae_config = getattr(vae.config, "arch_config", vae.config)
        latent_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents)
        latent_std = torch.sqrt(
            vae.bn.running_var.view(1, -1, 1, 1) + vae_config.batch_norm_eps
        ).to(latents)
        latents = latents * latent_std + latent_mean
        batch_size, channels, height, width = latents.shape
        latents = latents.reshape(batch_size, channels // 4, 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        return latents.reshape(batch_size, channels // 4, height * 2, width * 2)

    def post_denoising_loop(self, latents, batch):
        del batch
        return latents
