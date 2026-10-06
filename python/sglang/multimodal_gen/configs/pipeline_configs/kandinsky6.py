# SPDX-License-Identifier: Apache-2.0
"""Kandinsky6 TI2VA (text[+image] -> video+audio) pipeline configuration.

One pipeline serves text-to-video+audio generation with an optional
conditioning image: pure text when none is supplied at generation time,
image+text (called "I2VA" in the diffusers reference) when one is -- mirroring
how the reference's ``Kandinsky6I2VAPipeline`` is itself just a thin subclass
of ``Kandinsky6T2VAPipeline`` sharing the same call path. This matches
``ModelTaskType.TI2V`` semantics (accepts an image, does not require one)
better than an ``I2V`` task type, which would force every request to supply
one.
"""

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

# Same Qwen2.5-VL "prompt engineer" system template and 129-token crop as
# Kandinsky5's own pipeline config (this codebase has no Kandinsky5 port to
# import it from, so it is kept as an independent, self-contained literal).
# Byte-identical, including its two misspelled words: the real checkpoints
# were trained with this exact system prompt, so correcting them would
# silently change the text conditioning the model saw during training.
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
# Tokenized length of everything in KANDINSKY6_PROMPT_TEMPLATE before the
# user's own prompt text (i.e. the fixed system-prompt + start-of-user-turn
# boilerplate). kandinsky6_qwen_postprocess_text crops exactly this many
# tokens off the front of Reason1's per-example hidden states.
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
    """Crop the KANDINSKY6_PROMPT_TEMPLATE system-prompt prefix off Reason1.

    Ports the diffusers reference's fixed-offset
    ``hidden_states[-1][:, KANDINSKY6_PROMPT_TEMPLATE_ENCODE_START_IDX:]``
    slice onto sglang's mask-based crop idiom (``qwen_image_postprocess_text``
    in ``configs/pipeline_configs/qwen_image.py``): it first strips padding
    per example via the tokenizer's attention mask (so the crop offset lands
    on the real token stream regardless of batch padding), then drops the
    template prefix, then re-pads the variable-length results and returns an
    embedding-aligned mask -- avoiding the fixed-offset version's failure
    mode of slicing into right-padding for a batch of unequal-length prompts.
    """
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

    # Kandinsky6ImageEncodingStage owns all conditioning-image preprocessing
    # (VAE-encode + tail-cond-frame append) itself. Without this, the generic
    # InputValidationStage's ModelTaskType.TI2V branch (hard-coded for
    # Wan2.2-5B TI2V) would run instead and crash on Kandinsky6's Hunyuan-
    # lineage VAE (no `scale_factor_spatial` attribute) before that stage
    # ever executes -- matching the MiniMaxH3/SanaWM/LTX-2 precedent.
    skip_input_image_preprocess: bool = True

    # Model configuration. Kandinsky6 uses a single joint video+audio DiT
    # (unlike MOVA's separate video/audio towers), so there is no second
    # `audio_dit_config` field here -- `hidden_states_audio` is just another
    # argument on the one `dit_config.arch_config`-described transformer.
    dit_config: Kandinsky6VideoAudioConfig = field(
        default_factory=Kandinsky6VideoAudioConfig
    )
    dit_precision: str = "bf16"
    cfg_policy: CFGPolicy = field(
        default_factory=lambda: CFGPolicy(parallel_uses_serial_arithmetic=True)
    )

    # Video VAE: Kandinsky6 reuses the Hunyuan-lineage VAE unmodified
    # (temporal_compression_ratio=4, matching the audio<->video alignment
    # constants below), same as its Kandinsky5 predecessor.
    # Keep tiled decoding on multiple GPUs: whole-clip spatial sharding would
    # materialize the Hunyuan VAE's quadratic temporal attention mask.
    vae_config: HunyuanVAEConfig = field(
        default_factory=lambda: HunyuanVAEConfig(parallel_decode_mode="tiled")
    )
    vae_precision: str = "bf16"
    vae_tiling: bool = True

    # Audio VAE: Kandinsky6AudioVAE bundles the mel-VAE decoder *and* the
    # BigVGAN-v2 vocoder in one checkpoint component (confirmed against a
    # real checkpoint's audio_vae/*.safetensors) -- unlike e.g. LTX-2's
    # separately loaded audio_vae/vocoder pair, there is no separate
    # "vocoder" pipeline module here.
    audio_vae_config: Kandinsky6AudioVAEConfig = field(
        default_factory=Kandinsky6AudioVAEConfig
    )
    audio_vae_precision: str = "bf16"

    # Text encoding stage: Reason1 (Qwen2.5-VL-based "prompt engineer",
    # token-level hidden_states[-1]) + CLIP (pooled embedding). Two
    # independent plain-text-only encoder towers -- the simple case the
    # generic TextEncodingStage already handles (same shape as Flux's
    # CLIP+T5 pair), so no custom text-encoding stage is needed.
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
    # Reason1 uses dynamic padding up to its own 641-token cap (129-token
    # template + 512-token user-prompt budget); CLIP uses OpenAI CLIP's
    # standard fixed 77-token context.
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

    # Audio<->video latent-length alignment. Kandinsky6LatentPreparationStage
    # derives the audio latent frame count from these, matching the diffusers
    # reference's Kandinsky6TI2VAPipeline exactly: audio_latent_frames =
    # ceil(((T_lat-1)*4+1) / sample_fps * audio_sample_rate /
    # audio_downsample_factor), where (T_lat-1)*4+1 is the causal video VAE's
    # pixel-frame count for T_lat latent frames (temporal_compression_ratio ==
    # 4, matching HunyuanVAEConfig above).
    sample_fps: float = 24.0
    audio_sample_rate: int = 44100
    audio_downsample_factor: int = 1024

    def __post_init__(self) -> None:
        if len(self.text_encoder_configs) != 2:
            raise ValueError(
                "Kandinsky6 pipeline requires exactly 2 text encoders "
                f"(Reason1 and CLIP), but got {len(self.text_encoder_configs)}."
            )
        if len(self.text_encoder_precisions) != 2:
            raise ValueError(
                "Kandinsky6 pipeline requires exactly 2 text encoder "
                f"precisions, but got {len(self.text_encoder_precisions)}."
            )
        if len(self.text_encoder_extra_args) != 2:
            raise ValueError(
                "Kandinsky6 pipeline requires exactly 2 text encoder extra "
                f"tokenizer arg dicts, but got {len(self.text_encoder_extra_args)}."
            )

        # The merged pipeline can receive an optional conditioning image on
        # any given call, so the video VAE encoder must always be available
        # (unlike Kandinsky5, which only flips this on for its dedicated I2V
        # config) -- Kandinsky6ImageEncodingStage VAE-encodes the
        # conditioning image through this same shared video VAE.
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True

    def supports_disaggregation(self) -> bool:
        # The joint video+audio denoising loop advances both modalities from
        # a single transformer forward per timestep (video via
        # scheduler.step, audio via a manual Euler update using that same
        # step's sigma delta) -- matching MiniMaxH3's simplest-v1 choice of
        # disabling disaggregated deployment rather than splitting that
        # coupled loop across separate encode/denoise/decode roles.
        return False

    def get_text_encoder_pooler_output(self, outputs, encoder_index):
        # Only the CLIP encoder (index 1) has a pooler_output. Reason1's
        # forward (Qwen2_5_VLForConditionalGeneration) returns transformers'
        # native Qwen2_5_VLCausalLMOutputWithPast, which has no
        # `pooler_output` field at all (unlike sglang's own BaseEncoderOutput
        # wrapper, which always carries one, defaulting to None) -- so
        # `outputs.pooler_output` raises AttributeError for encoder_index 0.
        # This hook is called once per encoder (text_encoding.py), so an
        # unconditional return (Flux's CLIP+T5 precedent, where *both*
        # encoders return BaseEncoderOutput) is not safe here. Kandinsky6's
        # actual architectural twin is Hunyuan (LLM encoder at index 0 +
        # CLIP at index 1), which guards the same way.
        if encoder_index == 1:
            return outputs.pooler_output
        return None

    def get_pos_prompt_embeds(self, batch):
        return batch.prompt_embeds[0]

    def get_neg_prompt_embeds(self, batch):
        return batch.negative_prompt_embeds[0]

    def tokenize_prompt(self, prompt, tokenizer, tok_kwargs) -> dict:
        # The Reason1 "tokenizer" component is the full Qwen2_5_VLProcessor
        # (a multimodal processor, not a plain text tokenizer) -- its
        # __call__ signature is (images=None, text=None, videos=None,
        # audio=None, **kwargs), so calling it positionally
        # (tokenizer(prompt, **tok_kwargs), the base PipelineConfig default)
        # feeds our prompt string into the `images` slot instead of `text`,
        # which then fails trying to interpret the prompt as an image
        # path/URL/base64 blob. Kandinsky6's own conditioning image (when
        # supplied) never goes through this processor at all -- it is
        # VAE-encoded directly into the video latents by
        # Kandinsky6ImageEncodingStage -- so `images` must always stay
        # unset here regardless of whether a conditioning image was passed
        # at the pipeline level. Passing `text=` explicitly is also valid
        # for the CLIP tokenizer (encoder_index 1), so this one override
        # is correct for both text encoders.
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
