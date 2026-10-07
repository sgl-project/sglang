# Copyright 2025 Alibaba Ovis-Image Team and The HuggingFace Team.
# SPDX-License-Identifier: Apache-2.0
"""Native Ovis-Image conditioning, packed latents, and deployment configuration."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.configs.models.dits.ovis_image import OvisImageConfig
from sglang.multimodal_gen.configs.models.encoders.qwen3 import Qwen3TextConfig
from sglang.multimodal_gen.configs.models.vaes.flux import FluxVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ImagePipelineConfig,
    ModelTaskType,
    TextConditioningOutput,
)
from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import _pack_latents
from sglang.multimodal_gen.runtime.distributed.cfg_policy import CFGPolicy

OVIS_IMAGE_SYSTEM_PROMPT = (
    "Describe the image by detailing the color, quantity, text, shape, size, "
    "texture, spatial relationships of the objects and background: "
)
OVIS_IMAGE_PROMPT_PREFIX_LENGTH = 28


def ovis_image_text_output(outputs, text_inputs) -> TextConditioningOutput:
    embeddings = outputs.last_hidden_state * text_inputs.attention_mask.unsqueeze(-1)
    embeddings = embeddings[:, OVIS_IMAGE_PROMPT_PREFIX_LENGTH:, :]
    # The reference DiT attends to fixed-length, zeroed encoder padding too.
    # Only padding introduced by distributed execution should be masked.
    mask = torch.ones(embeddings.shape[:2], dtype=torch.bool, device=embeddings.device)
    return TextConditioningOutput(
        embeddings, mask, [embeddings.shape[1]] * embeddings.shape[0]
    )


def _ovis_image_vae_config() -> FluxVAEConfig:
    config = FluxVAEConfig(use_temporal_scaling_frames=False)
    config.arch_config.temporal_compression_ratio = 1
    config.arch_config.scaling_factor = 0.3611
    config.arch_config.shift_factor = 0.1159
    config.arch_config.latent_channels = 16
    config.post_init()
    return config


def _ovis_image_text_encoder_config() -> Qwen3TextConfig:
    return Qwen3TextConfig(preserve_hf_numerics=True)


@dataclass
class OvisImagePipelineConfig(ImagePipelineConfig):
    task_type: ModelTaskType = ModelTaskType.T2I
    supported_task_types: ClassVar[tuple[ModelTaskType, ...]] = (ModelTaskType.T2I,)
    native_only_components: tuple[str, ...] = ("transformer", "text_encoder", "vae")
    cfg_policy: CFGPolicy = field(
        default_factory=lambda: CFGPolicy(parallel_uses_serial_arithmetic=True)
    )
    dit_config: OvisImageConfig = field(default_factory=OvisImageConfig)
    vae_config: FluxVAEConfig = field(default_factory=_ovis_image_vae_config)
    text_encoder_configs: tuple = field(
        default_factory=lambda: (_ovis_image_text_encoder_config(),)
    )
    text_encoder_precisions: tuple[str, ...] = ("bf16",)
    preprocess_text_funcs: tuple[Callable | None, ...] = (None,)
    postprocess_text_funcs: tuple[Callable, ...] = (ovis_image_text_output,)
    text_encoder_extra_args: list[dict] = field(
        default_factory=lambda: [{"max_length": 256}]
    )
    should_use_guidance: bool = False
    enable_autocast: bool = False
    vae_precision: str = "bf16"
    vae_tiling: bool = False
    vae_sp: bool = False

    def __post_init__(self):
        self.vae_config.use_parallel_decode = self.vae_sp

    def update_config_from_dict(self, args, prefix=""):
        super().update_config_from_dict(args, prefix)
        self.__post_init__()

    def validate_server_args(self, server_args) -> None:
        self.__post_init__()
        super().validate_server_args(server_args)
        tp_size = getattr(server_args, "tp_size", 1) or 1
        ulysses_degree = getattr(server_args, "ulysses_degree", 1) or 1
        dit_heads = self.dit_config.arch_config.num_attention_heads
        encoder_heads = self.text_encoder_configs[0].arch_config.num_attention_heads
        if dit_heads % (tp_size * ulysses_degree):
            raise ValueError("Ovis-Image TP × Ulysses must divide DiT attention heads")
        if encoder_heads % tp_size:
            raise ValueError("Ovis-Image TP must divide Qwen attention heads")
        model_path = str(server_args.model_path)
        if (
            server_args.comfyui_mode
            or Path(model_path).is_file()
            or model_path.lower().endswith((".safetensors", ".ckpt", ".gguf"))
        ):
            raise ValueError("Ovis-Image requires a Diffusers-format model directory")
        if (
            server_args.quantization
            or server_args.component_quantizations
            or server_args.nunchaku_config
            or server_args.kv_cache_quant_config.enabled
        ):
            raise ValueError("Ovis-Image native support does not include quantization")
        if server_args.lora_path:
            raise ValueError("Ovis-Image native support does not include LoRA")
        if server_args.enable_torch_compile:
            raise ValueError(
                "Ovis-Image does not support torch.compile or breakable CUDA graphs"
            )
        self.validate_breakable_cuda_graph(server_args)
        if (
            envs.SGLANG_CACHE_DIT_ENABLED
            or getattr(server_args, "cache_dit_config", None) is not None
        ):
            raise ValueError("Ovis-Image does not support Cache-DiT")

    def validate_breakable_cuda_graph(self, server_args) -> None:
        if server_args.enable_breakable_cuda_graph:
            raise ValueError("Ovis-Image does not support breakable CUDA graphs")

    def tokenize_prompt(self, prompt, tokenizer, tok_kwargs):
        length = tok_kwargs.get("max_length", 256)
        if (
            isinstance(length, bool)
            or not isinstance(length, int)
            or not 1 <= length <= 256
        ):
            raise ValueError("Ovis-Image max_sequence_length must be between 1 and 256")
        messages = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": OVIS_IMAGE_SYSTEM_PROMPT + text}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for text in prompt
        ]
        return tokenizer(
            messages,
            **{
                **tok_kwargs,
                "max_length": length + OVIS_IMAGE_PROMPT_PREFIX_LENGTH,
                "padding": "max_length",
                "truncation": True,
                "return_tensors": "pt",
                "add_special_tokens": False,
            },
        )

    def prepare_sigmas(self, sigmas, num_inference_steps):
        return self._prepare_sigmas(sigmas, num_inference_steps)

    def get_latent_dtype(self, prompt_dtype):
        return prompt_dtype

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        factor = self.vae_config.arch_config.vae_scale_factor * 2
        return (
            batch_size,
            self.dit_config.arch_config.in_channels // 4,
            2 * (batch.height // factor),
            2 * (batch.width // factor),
        )

    def maybe_pack_latents(self, latents, batch_size, batch):
        _, channels, height, width = latents.shape
        return _pack_latents(latents, batch_size, channels, height, width)

    def shard_latents_for_sp(self, batch, latents):
        # Shard inside the DiT, preserving replicated scheduler state.
        return latents, False

    def get_pos_prompt_embeds(self, batch):
        return batch.prompt_embeds[0]

    def get_neg_prompt_embeds(self, batch):
        return batch.negative_prompt_embeds[0]

    def expand_conditioning_to_sample_batch(self, batch):
        count = batch.num_outputs_per_prompt
        if count <= 1:
            return batch
        prefixes = [("prompt", "prompt_attention_mask")]
        if batch.do_classifier_free_guidance:
            prefixes.append(("negative_prompt", "negative_attention_mask"))
        for prefix, attention_mask_name in prefixes:
            for name in (
                f"{prefix}_embeds",
                f"{prefix}_embeds_mask",
                attention_mask_name,
            ):
                values = getattr(batch, name, None)
                if values is not None:
                    setattr(
                        batch,
                        name,
                        [
                            x.repeat_interleave(count, dim=0) if x is not None else None
                            for x in values
                        ],
                    )
            name = f"{prefix}_seq_lens"
            values = getattr(batch, name, None)
            if values is not None:
                setattr(
                    batch,
                    name,
                    [
                        [length for length in lengths for _ in range(count)]
                        for lengths in values
                    ],
                )
        return batch

    def _prepare_latent_image_ids(self, original_height, original_width, device):
        factor = self.vae_config.arch_config.vae_scale_factor * 2
        height, width = original_height // factor, original_width // factor
        ids = torch.zeros(height, width, 3, device=device)
        ids[..., 1] = torch.arange(height, device=device)[:, None]
        ids[..., 2] = torch.arange(width, device=device)[None, :]
        return ids.reshape(height * width, 3)

    def get_freqs_cis(
        self, prompt_embeds, width, height, device, rotary_emb, batch, txt_seq_lens
    ):
        seq_len = prompt_embeds.shape[1]
        if len(txt_seq_lens) != prompt_embeds.shape[0] or any(
            length != seq_len for length in txt_seq_lens
        ):
            raise ValueError("Ovis-Image requires fixed-length text conditioning")
        text_ids = torch.zeros(seq_len, 3, device=device)
        text_ids[:, 1:] = torch.arange(seq_len, device=device)[:, None]
        image_ids = self._prepare_latent_image_ids(height, width, device)
        return rotary_emb(
            torch.cat([text_ids, image_ids], dim=0).to(dtype=prompt_embeds.dtype)
        )

    def _prepare_cond_kwargs(self, batch, device, rotary_emb, *, negative):
        embeddings = (
            self.get_neg_prompt_embeds(batch)
            if negative
            else self.get_pos_prompt_embeds(batch)
        )
        lengths = self.require_text_seq_lens(
            batch, 0, negative=negative, expected_batch_size=embeddings.shape[0]
        )
        return {
            "freqs_cis": self.get_freqs_cis(
                embeddings,
                batch.width,
                batch.height,
                device,
                rotary_emb,
                batch,
                lengths,
            )
        }

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return self._prepare_cond_kwargs(batch, device, rotary_emb, negative=False)

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return self._prepare_cond_kwargs(batch, device, rotary_emb, negative=True)

    def post_denoising_loop(self, latents, batch):
        factor = self.vae_config.arch_config.vae_scale_factor * 2
        height, width = 2 * (batch.height // factor), 2 * (batch.width // factor)
        batch_size, _, channels = latents.shape
        latents = latents.reshape(
            batch_size, height // 2, width // 2, channels // 4, 2, 2
        )
        return latents.permute(0, 3, 1, 4, 2, 5).reshape(
            batch_size, channels // 4, height, width
        )


def register():
    from sglang.multimodal_gen.configs.sample.ovis_image import OvisImageSamplingParams
    from sglang.multimodal_gen.registry import register_configs

    register_configs(
        sampling_param_cls=OvisImageSamplingParams,
        pipeline_config_cls=OvisImagePipelineConfig,
        hf_model_paths=["ATH-MaaS/Ovis-Image-7B", "AIDC-AI/Ovis-Image-7B"],
        model_detectors=[
            lambda name: (
                name.lower() == "ovisimagepipeline" or "ovis-image-7b" in name.lower()
            )
        ],
    )
