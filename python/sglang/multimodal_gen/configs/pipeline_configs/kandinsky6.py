# SPDX-License-Identifier: Apache-2.0
"""Kandinsky 6 text-to-video/audio pipeline with optional image conditioning."""

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from sglang.multimodal_gen.configs.models import EncoderConfig
from sglang.multimodal_gen.configs.models.dits.kandinsky6 import (
    Kandinsky6VideoAudioConfig,
)
from sglang.multimodal_gen.configs.models.encoders import (
    BaseEncoderOutput,
    CLIPTextConfig,
)
from sglang.multimodal_gen.configs.models.encoders.kandinsky6_reason1 import (
    Reason1Config,
)
from sglang.multimodal_gen.configs.models.vaes.hunyuanvae import HunyuanVAEConfig
from sglang.multimodal_gen.configs.models.vaes.kandinsky6_audio import (
    Kandinsky6AudioVAEConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.kandinsky6_sr import (
    _is_kandinsky6_sr,
)
from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    qwen_image_postprocess_text,
)
from sglang.multimodal_gen.runtime.distributed.cfg_policy import CFGPolicy

# checkpoint-trained template: preserve its exact bytes, including misspellings
KANDINSKY6_PROMPT_TEMPLATE = "\n".join(
    [
        "<|im_start|>system\nYou are a promt engineer. Describe the video in detail.",  # codespell:ignore promt
        "Describe how the camera moves or shakes, describe the zoom and view angle, whether it follows the objects.",
        "Describe the location of the video, main characters or objects and their action.",
        "Describe the dynamism of the video and presented actions.",
        "Name the visual style of the video: whether it is a professional footage, user generated content, some kind of animation, video game or scren content.",  # codespell:ignore scren
        "Describe the visual effects, postprocessing and transitions if they are presented in the video.",
        "Pay attention to the order of key actions shown in the scene.<|im_end|>",
        "<|im_start|>user\n{}<|im_end|>",
    ]
)
# Reason1 template prefix length before user tokens
KANDINSKY6_PROMPT_TEMPLATE_ENCODE_START_IDX = 129


def kandinsky6_qwen_preprocess_text(prompt: str) -> str:
    if not prompt.strip():
        prompt = "."
    return KANDINSKY6_PROMPT_TEMPLATE.format(prompt)


def kandinsky6_qwen_postprocess_text(
    outputs: BaseEncoderOutput,
    text_inputs,
    return_attention_mask: bool = False,
):
    """Strip padding and the Reason1 template prefix, then repad embeddings and mask."""
    return qwen_image_postprocess_text(
        outputs,
        text_inputs,
        drop_idx=KANDINSKY6_PROMPT_TEMPLATE_ENCODE_START_IDX,
        return_attention_mask=return_attention_mask,
    )


def kandinsky6_clip_postprocess_text(
    outputs: BaseEncoderOutput, _text_inputs
) -> torch.Tensor:
    return outputs.pooler_output


@dataclass
class Kandinsky6TI2VAPipelineConfig(PipelineConfig):
    """Configuration for the Kandinsky6 TI2VA pipeline."""

    task_type: ModelTaskType = ModelTaskType.TI2V
    native_only_components: tuple[str, ...] = (
        "transformer",
        "text_encoder",
        "text_encoder_2",
        "vae",
        "audio_vae",
    )

    # image conditioning is handled by Kandinsky6ImageEncodingStage, not Wan TI2V preprocessing
    skip_input_image_preprocess: bool = True

    # a single joint DiT consumes video and audio latents
    dit_config: Kandinsky6VideoAudioConfig = field(
        default_factory=Kandinsky6VideoAudioConfig
    )
    dit_precision: str = "bf16"
    cfg_policy: CFGPolicy = field(
        default_factory=lambda: CFGPolicy(parallel_uses_serial_arithmetic=True)
    )

    # keep tiled decode: whole-clip sharding materializes a quadratic temporal mask
    vae_config: HunyuanVAEConfig = field(
        default_factory=lambda: HunyuanVAEConfig(parallel_decode_mode="tiled")
    )
    vae_precision: str = "bf16"
    vae_tiling: bool = True

    # audio_vae includes both the mel codec and vocoder
    audio_vae_config: Kandinsky6AudioVAEConfig = field(
        default_factory=Kandinsky6AudioVAEConfig
    )
    audio_vae_precision: str = "bf16"

    # Reason1 provides token embeddings; CLIP provides pooled text conditioning
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (Reason1Config(), CLIPTextConfig())
    )
    text_encoder_precisions: tuple[str, ...] = field(
        default_factory=lambda: ("bf16", "bf16")
    )
    preprocess_text_funcs: tuple[Callable[[str], str] | None, ...] = field(
        default_factory=lambda: (kandinsky6_qwen_preprocess_text, None)
    )
    postprocess_text_funcs: tuple[Callable[..., Any], ...] = field(
        default_factory=lambda: (
            kandinsky6_qwen_postprocess_text,
            kandinsky6_clip_postprocess_text,
        )
    )
    # Reason1: 129 template + 512 user tokens; CLIP: fixed 77-token context
    text_encoder_extra_args: list[dict] = field(
        default_factory=lambda: [
            dict(
                max_length=KANDINSKY6_PROMPT_TEMPLATE_ENCODE_START_IDX + 512,
                padding=True,
                truncation=True,
            ),
            dict(
                max_length=77,
                padding="max_length",
                truncation=True,
                add_special_tokens=True,
            ),
        ]
    )

    # Shift of both official checkpoints' schedulers (FlowMatchEulerDiscreteScheduler
    # for Pro-sft, PiflowScheduler for Pro-distill: scheduler_config.json "shift").
    flow_shift: float | None = 5.0

    # audio frames = ceil(pixel_frames / fps * sample_rate / downsample_factor)
    sample_fps: float = 24.0
    audio_sample_rate: int = 44100
    audio_downsample_factor: int = 1024

    def __post_init__(self) -> None:
        for name, values in (
            ("text encoders (Reason1 and CLIP)", self.text_encoder_configs),
            ("text encoder precisions", self.text_encoder_precisions),
            ("text encoder extra tokenizer arg dicts", self.text_encoder_extra_args),
        ):
            if len(values) != 2:
                raise ValueError(
                    f"Kandinsky6 pipeline requires exactly 2 {name}, but got {len(values)}."
                )

        # keep the video VAE encoder available for optional image conditioning
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True

    def supports_disaggregation(self) -> bool:
        # joint denoising currently runs monolithically
        return False

    def get_text_encoder_pooler_output(self, outputs, encoder_index):
        # only CLIP (index 1) returns pooler_output; Reason1 returns causal-LM output
        if encoder_index == 1:
            return outputs.pooler_output
        return None

    def get_pos_prompt_embeds(self, batch):
        return batch.prompt_embeds[0]

    def get_neg_prompt_embeds(self, batch):
        return batch.negative_prompt_embeds[0]

    def tokenize_prompt(self, prompt, tokenizer, tok_kwargs) -> dict:
        # Reason1 uses a multimodal processor: positional input would bind to images,
        # not text. The conditioning image is handled separately by the VAE.
        return tokenizer(text=prompt, **tok_kwargs)


def _is_kandinsky6_t2va_family(model_id: str) -> bool:
    normalized = model_id.lower().replace("-", "").replace("_", "")
    return "kandinsky6" in normalized and not _is_kandinsky6_sr(model_id)


def _is_kandinsky6_t2va_distilled(model_id: str) -> bool:
    short_name = model_id.lower().rstrip("/").split("/")[-1]
    return _is_kandinsky6_t2va_family(model_id) and "distill" in short_name


def _is_kandinsky6_t2va(model_id: str) -> bool:
    short_name = model_id.lower().rstrip("/").split("/")[-1]
    return _is_kandinsky6_t2va_family(model_id) and "distill" not in short_name


def register():
    from sglang.multimodal_gen.configs.sample.kandinsky6 import (
        Kandinsky6TI2VADistilledSamplingParams,
        Kandinsky6TI2VASamplingParams,
    )
    from sglang.multimodal_gen.registry import register_configs

    register_configs(
        sampling_param_cls=Kandinsky6TI2VADistilledSamplingParams,
        pipeline_config_cls=Kandinsky6TI2VAPipelineConfig,
        hf_model_paths=["kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers"],
        model_detectors=[_is_kandinsky6_t2va_distilled],
    )
    register_configs(
        sampling_param_cls=Kandinsky6TI2VASamplingParams,
        pipeline_config_cls=Kandinsky6TI2VAPipelineConfig,
        hf_model_paths=[
            "kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers",
            "kandinskylab/Kandinsky-6.0-Pro-sft-5s-Diffusers",
        ],
        model_detectors=[_is_kandinsky6_t2va],
    )


__all__ = [
    "KANDINSKY6_PROMPT_TEMPLATE",
    "KANDINSKY6_PROMPT_TEMPLATE_ENCODE_START_IDX",
    "Kandinsky6TI2VAPipelineConfig",
    "kandinsky6_clip_postprocess_text",
    "kandinsky6_qwen_postprocess_text",
    "kandinsky6_qwen_preprocess_text",
]
