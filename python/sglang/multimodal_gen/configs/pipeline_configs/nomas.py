# SPDX-License-Identifier: Apache-2.0
import math
from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.configs.models.dits.nomas import NomasConfig
from sglang.multimodal_gen.configs.models.encoders.qwen_image import (
    Qwen2_5VLConfig,
    QwenImageArchConfig,
)
from sglang.multimodal_gen.configs.models.vaes.nomas import NomasVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)

# Preserve upstream whitespace: this text is part of the conditioning contract.
NOMAS_SYSTEM_PROMPT = (
    "You are a helpful assistant. Describe the video by detailing the following aspects: "
    "        1. The main content and theme of the video. "
    "        2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. "
    "        3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. "
    "        4. background environment, light, style and atmosphere. "
    "        5. camera angles, movements, and transitions used in the video."
)


def nomas_postprocess_text(outputs, text_inputs):
    crop = int(text_inputs["nomas_crop_start"].item())
    return outputs.hidden_states[-3][:, crop:], text_inputs["attention_mask"][:, crop:]


@dataclass
class NomasPipelineConfig(PipelineConfig):
    task_type: ModelTaskType = ModelTaskType.T2V
    supports_auto_residency: bool = False
    native_only_components: tuple[str, ...] = ("transformer", "vae", "text_encoder")
    dit_config: NomasConfig = field(default_factory=NomasConfig)
    vae_config: NomasVAEConfig = field(default_factory=NomasVAEConfig)
    vae_precision: str = "bf16"
    vae_tiling: bool = False
    vae_sp: bool = False
    text_encoder_configs: tuple = field(
        default_factory=lambda: (
            Qwen2_5VLConfig(arch_config=QwenImageArchConfig(text_len=1280)),
        )
    )
    text_encoder_precisions: tuple[str, ...] = ("bf16",)
    preprocess_text_funcs: tuple = (None,)
    postprocess_text_funcs: tuple = (nomas_postprocess_text,)
    text_encoder_extra_args: list[dict] = field(
        default_factory=lambda: [
            {"padding": "max_length", "return_attention_mask": True}
        ]
    )
    flow_shift: float = 5.0
    should_use_guidance: bool = False
    segment_latent_frames: int = 10
    cache_noise_sigma: float = 0.206897
    max_cache_gib: float = 32.0
    cache_free_memory_reserve_gib: float = 8.0
    # Realtime sessions; names follow upstream inference.yaml / the demo worker.
    max_cache_segments: int | None = None  # FIFO by whole segment; None keeps all
    interactive_kv_cache_initial_tokens: int = 65536
    interactive_kv_cache_growth_factor: float = 1.5
    interactive_noise_max_seconds: float = 120.0
    interactive_noise_growth_seconds: float = 30.0

    def validate_server_args(self, server_args):
        tp_size = server_args.tp_size or 1
        sp_degree = server_args.sp_degree or 1
        if server_args.dp_size not in (None, 1):
            raise ValueError("Nomas does not support data parallelism")
        if (server_args.ring_degree or 1) != 1:
            raise ValueError(
                "Nomas supports Ulysses sequence parallelism only; set ring_degree=1"
            )
        if server_args.enable_cfg_parallel:
            raise ValueError("Nomas does not support CFG parallelism")
        heads = self.dit_config.num_attention_heads
        if heads % (tp_size * sp_degree):
            raise ValueError(
                f"Nomas attention heads ({heads}) must be divisible by tp_size * sp_degree"
            )
        if not math.isfinite(self.flow_shift) or self.flow_shift <= 0:
            raise ValueError("flow_shift must be positive and finite")
        if (
            not math.isfinite(self.max_cache_gib)
            or self.max_cache_gib <= 0
            or not math.isfinite(self.cache_free_memory_reserve_gib)
            or self.cache_free_memory_reserve_gib < 0
        ):
            raise ValueError("cache budget must be positive and reserve nonnegative")
        if self.segment_latent_frames < 1:
            raise ValueError("segment_latent_frames must be positive")
        if self.max_cache_segments is not None and self.max_cache_segments < 1:
            raise ValueError("max_cache_segments must be positive or None")
        if (
            self.interactive_kv_cache_initial_tokens < 1
            or self.interactive_kv_cache_growth_factor <= 1
        ):
            raise ValueError(
                "interactive KV cache initial tokens and growth must be positive"
            )
        if (
            self.interactive_noise_max_seconds <= 0
            or self.interactive_noise_growth_seconds <= 0
        ):
            raise ValueError("interactive noise bank sizes must be positive")
        if self.dit_precision != "bf16":
            raise ValueError("The Nomas pipeline currently requires BF16 DiT inference")

    def supports_dynamic_batching(self):
        return False

    def supports_disaggregation(self):
        return False

    def prepare_sigmas(self, sigmas, num_inference_steps):
        if sigmas is not None:
            raise ValueError("Custom sigma schedules are not supported by Nomas")
        # The common flow-match scheduler applies flow_shift and appends zero.
        return torch.linspace(1, 0, num_inference_steps + 1)[:-1].tolist()

    def tokenize_prompt(self, prompt, tokenizer, tok_kwargs):
        if tokenizer.padding_side != "right":
            raise ValueError("Nomas requires right-padded text")
        conversations = [
            [
                {"role": "system", "content": NOMAS_SYSTEM_PROMPT},
                {"role": "user", "content": text or " "},
            ]
            for text in prompt
        ]
        marker = tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
        ids = tokenizer.apply_chat_template(
            conversations[0],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        crop = next(
            (
                i + len(marker)
                for i in range(len(ids) - len(marker) + 1)
                if ids[i : i + len(marker)] == marker
            ),
            None,
        )
        if crop is None:
            raise ValueError("Tokenizer must provide the Qwen user chat marker")
        inputs = tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            truncation=True,
            max_length=int(tok_kwargs["max_length"]) + crop,
            padding="max_length",
            return_tensors="pt",
        )
        inputs["nomas_crop_start"] = torch.tensor(crop)
        return inputs

    def get_latent_dtype(self, prompt_dtype):
        return torch.bfloat16

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        if batch_size != 1 or batch.guidance_scale != 1:
            raise ValueError("Nomas supports batch size one and guidance_scale=1")
        if batch.height % 32 or batch.width % 32:
            raise ValueError("Nomas height and width must be multiples of 32")
        return (1, 32, num_frames, batch.height // 32, batch.width // 32)


def register():
    from sglang.multimodal_gen.configs.sample.nomas import NomasSamplingParams
    from sglang.multimodal_gen.registry import register_configs

    register_configs(
        NomasSamplingParams,
        NomasPipelineConfig,
        hf_model_paths=["tencent/Hy-Nomas"],
        model_detectors=[lambda name: name.lower() == "nomaspipeline"],
        pipeline_config_registry_entries={
            "NomasPipeline": (NomasPipelineConfig, NomasSamplingParams)
        },
    )
