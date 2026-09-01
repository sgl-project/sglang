import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.longcat_image import (
    LongCatImageDitConfig,
)
from sglang.multimodal_gen.configs.models.encoders.qwen_image import Qwen2_5VLConfig
from sglang.multimodal_gen.configs.models.vaes.longcat_image import (
    LongCatImageVAEConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ImagePipelineConfig,
    ModelTaskType,
    TextConditioningOutput,
)
from sglang.multimodal_gen.configs.pipeline_configs.model_deployment_config import (
    ModelDeploymentConfig,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_world_size,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Encode-side prompt length (governs tokenization truncation/padding and the
# image latent position-id start offset). Mirrors diffusers'
# LongCatImagePipeline.tokenizer_max_length.
TOKENIZER_MAX_LENGTH = 512

# Fixed chat-template wrappers prepended/appended around the encoded prompt.
# Mirrors diffusers LongCatImagePipeline._encode_prompt.
ENCODE_PREFIX_STR = (
    "<|im_start|>system\nAs an image captioning expert, generate a descriptive text prompt "
    "based on an image content, suitable for input to a text-to-image model.<|im_end|>\n"
    "<|im_start|>user\n"
)
ENCODE_SUFFIX_STR = "<|im_end|>\n<|im_start|>assistant\n"


def _split_quotation(prompt, quote_pairs=None):
    """Split prompt on quoted substrings, returning list of (text, is_quoted) tuples."""
    import re

    word_internal_quote_pattern = re.compile(r"[a-zA-Z]+'[a-zA-Z]+")
    matches = word_internal_quote_pattern.findall(prompt)
    mapping = []

    for i, word_src in enumerate(set(matches)):
        word_tgt = "longcat_$##$_longcat" * (i + 1)
        prompt = prompt.replace(word_src, word_tgt)
        mapping.append([word_src, word_tgt])

    if quote_pairs is None:
        quote_pairs = [
            ("'", "'"),
            ('"', '"'),
            ("‘", "’"),
            ("“", "”"),
        ]
    pattern = "|".join(
        [
            re.escape(q1) + r"[^" + re.escape(q1 + q2) + r"]*?" + re.escape(q2)
            for q1, q2 in quote_pairs
        ]
    )
    parts = re.split(f"({pattern})", prompt)

    result = []
    for part in parts:
        for word_src, word_tgt in mapping:
            part = part.replace(word_tgt, word_src)
        if re.match(pattern, part):
            if len(part):
                result.append((part, True))
        else:
            if len(part):
                result.append((part, False))
    return result


def _prepare_pos_ids(
    modality_id=0,
    token_type="text",
    start=(0, 0),
    num_token=None,
    height=None,
    width=None,
):
    if token_type == "text":
        assert num_token
        pos_ids = torch.zeros(num_token, 3)
        pos_ids[..., 0] = modality_id
        pos_ids[..., 1] = torch.arange(num_token) + start[0]
        pos_ids[..., 2] = torch.arange(num_token) + start[1]
    elif token_type == "image":
        assert height and width
        pos_ids = torch.zeros(height, width, 3)
        pos_ids[..., 0] = modality_id
        pos_ids[..., 1] = pos_ids[..., 1] + torch.arange(height)[:, None] + start[0]
        pos_ids[..., 2] = pos_ids[..., 2] + torch.arange(width)[None, :] + start[1]
        pos_ids = pos_ids.reshape(height * width, 3)
    else:
        raise KeyError(
            f'Unknown token_type {token_type}, only support "text" or "image".'
        )
    return pos_ids


def _tokenize_prompt_for_encode(prompt, tokenizer):
    """Quote-aware tokenization mirroring diffusers LongCatImagePipeline._encode_prompt.

    Quoted substrings are tokenized character-by-character; unquoted substrings
    are tokenized whole. Truncated/padded to TOKENIZER_MAX_LENGTH. Returns the
    padded (input_ids, attention_mask) for the prompt body (without prefix/suffix).
    """
    if isinstance(prompt, str):
        prompt = [prompt]

    batch_all_tokens = []
    for each_prompt in prompt:
        all_tokens = []
        for clean_prompt_sub, matched in _split_quotation(each_prompt):
            if matched:
                # Intentional: tokenize each character in quoted text individually,
                # mirroring diffusers LongCatImagePipeline._encode_prompt behavior.
                for sub_word in clean_prompt_sub:
                    tokens = tokenizer(sub_word, add_special_tokens=False)["input_ids"]
                    all_tokens.extend(tokens)
            else:
                tokens = tokenizer(clean_prompt_sub, add_special_tokens=False)[
                    "input_ids"
                ]
                all_tokens.extend(tokens)

        if len(all_tokens) > TOKENIZER_MAX_LENGTH:
            logger.warning(
                "Prompt truncated: max_sequence_length=%d, input_token_nums=%d",
                TOKENIZER_MAX_LENGTH,
                len(all_tokens),
            )
            all_tokens = all_tokens[:TOKENIZER_MAX_LENGTH]
        batch_all_tokens.append(all_tokens)

    text_tokens_and_mask = tokenizer.pad(
        {"input_ids": batch_all_tokens},
        max_length=TOKENIZER_MAX_LENGTH,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
    )
    return text_tokens_and_mask


def _unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    # VAE applies 8x compression, plus 2x packing factor
    h = 2 * (int(height) // (vae_scale_factor * 2))
    w = 2 * (int(width) // (vae_scale_factor * 2))

    latents = latents.view(batch_size, h // 2, w // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // (2 * 2), h, w)
    return latents


def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(
        batch_size, num_channels_latents, height // 2, 2, width // 2, 2
    )
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(
        batch_size, (height // 2) * (width // 2), num_channels_latents * 4
    )
    return latents


def _calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def longcat_postprocess_text(outputs, text_inputs, pipeline_config):
    """Slice hidden states to drop the encode prefix/suffix tokens.

    The tokenizer (via `LongCatImagePipelineConfig.tokenize_prompt`) wraps each
    prompt with a fixed system prefix and an assistant suffix. The DiT must
    receive only the 512-token prompt body, so we slice
    `hidden_states[-1][:, prefix_len:-suffix_len, :]`. A mask aligned to the
    sliced length is returned so `build_text_conditioning_mask`'s length check
    (raw_mask len vs embed seq len) does not raise on the unsliced
    prefix+suffix mask.

    LongCat feeds the full 512-token body (including padding) to the DiT, so the
    mask is all-ones and seq_lens are all 512 — preserving the pre-refactor
    behavior where the DiT attended over the entire padded prompt body.
    """
    prefix_len = pipeline_config._encode_prefix_len
    suffix_len = pipeline_config._encode_suffix_len

    hidden_states = outputs.hidden_states[-1]
    prompt_embeds = hidden_states[:, prefix_len:-suffix_len, :]

    seq_len = prompt_embeds.shape[1]
    batch_size = prompt_embeds.shape[0]
    prompt_embeds_mask = torch.ones(
        batch_size, seq_len, dtype=torch.bool, device=prompt_embeds.device
    )
    prompt_seq_lens = [seq_len] * batch_size
    return TextConditioningOutput(
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        prompt_seq_lens=prompt_seq_lens,
    )


@dataclass
class LongCatImagePipelineConfig(ImagePipelineConfig):
    """Configuration for the LongCat-Image T2I pipeline."""

    task_type: ModelTaskType = ModelTaskType.T2I

    vae_precision: str = "bf16"
    should_use_guidance: bool = True
    vae_tiling: bool = False
    vae_sp: bool = False
    enable_autocast: bool = False

    dit_config: DiTConfig = field(default_factory=LongCatImageDitConfig)
    vae_config: VAEConfig = field(default_factory=LongCatImageVAEConfig)

    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16",))
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (Qwen2_5VLConfig(),)
    )
    postprocess_text_funcs: tuple[Callable, ...] = field(
        default_factory=lambda: (longcat_postprocess_text,)
    )

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        return ModelDeploymentConfig(
            keep_resident_min_available_gb=70,
            keep_resident_components=("text_encoder", "vae"),
        )

    def expand_conditioning_to_sample_batch(self, batch):
        # Noise/reference latents are built at batch_size = prompts * num_outputs,
        # but text encoding stays per-prompt; repeat the embeds to match. No-op
        # for num_outputs == 1. Shared by T2I and Edit.
        from sglang.multimodal_gen.runtime.utils.condition_expansion import (
            PromptToSampleBatchExpander,
        )

        expander = PromptToSampleBatchExpander.from_batch(batch)
        if expander is None:
            return batch
        for field_name in (
            "prompt_embeds",
            "negative_prompt_embeds",
            "prompt_embeds_mask",
            "negative_prompt_embeds_mask",
            "prompt_seq_lens",
            "negative_prompt_seq_lens",
        ):
            expander.expand_field(batch, field_name)
        return batch

    # --- LatentPreparationStage hooks ---

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        vae_scale_factor = self.vae_config.get_vae_scale_factor()
        # LongCat packs 2x2 patches: effective spatial resolution after packing
        h = 2 * (int(batch.height) // (vae_scale_factor * 2))
        w = 2 * (int(batch.width) // (vae_scale_factor * 2))
        num_channels_latents = self.dit_config.arch_config.num_channels_latents
        # Unpacked shape — maybe_pack_latents will fold into tokens
        return (batch_size, num_channels_latents, h, w)

    def maybe_pack_latents(self, latents, batch_size, batch):
        num_channels_latents = self.dit_config.arch_config.num_channels_latents
        _, _, h, w = latents.shape
        return _pack_latents(latents, batch_size, num_channels_latents, h, w)

    def maybe_prepare_latent_ids(self, latents):
        # latents shape after packing: [B, (h//2)*(w//2), C*4]
        # We need h//2 and w//2 — derive from the unpacked shape stored on the config.
        # latents is still unpacked here (called before maybe_pack_latents in LatentPreparationStage)
        _, _, h, w = latents.shape
        return _prepare_pos_ids(
            modality_id=1,
            token_type="image",
            start=(TOKENIZER_MAX_LENGTH, TOKENIZER_MAX_LENGTH),
            height=h // 2,
            width=w // 2,
        )

    def get_latent_dtype(self, prompt_dtype: torch.dtype) -> torch.dtype:
        # Generate in float32 then cast to bfloat16, matching diffusers behavior.
        return torch.float32

    # --- TextEncodingStage hooks ---

    def _ensure_encode_prefix_suffix(self, tokenizer):
        """Lazily tokenize the fixed encode prefix/suffix (tokenizer unavailable
        at config construction time). Cached on the config instance."""
        if not hasattr(self, "_encode_prefix_ids"):
            self._encode_prefix_ids = tokenizer(
                ENCODE_PREFIX_STR, add_special_tokens=False
            )["input_ids"]
            self._encode_suffix_ids = tokenizer(
                ENCODE_SUFFIX_STR, add_special_tokens=False
            )["input_ids"]
            self._encode_prefix_len = len(self._encode_prefix_ids)
            self._encode_suffix_len = len(self._encode_suffix_ids)

    def tokenize_prompt(self, prompt, tokenizer, tok_kwargs):
        """Quote-aware tokenization + fixed prefix/suffix wrapping.

        Mirrors diffusers LongCatImagePipeline._encode_prompt: quoted substrings
        are tokenized character-by-character, the body is truncated/padded to
        TOKENIZER_MAX_LENGTH, then a fixed system prefix and assistant suffix are
        concatenated onto every sequence (with all-ones masks). Returns a
        BatchEncoding-like dict with `input_ids` / `attention_mask` of length
        `prefix_len + TOKENIZER_MAX_LENGTH + suffix_len`.
        """
        from transformers import BatchEncoding

        self._ensure_encode_prefix_suffix(tokenizer)

        body = _tokenize_prompt_for_encode(prompt, tokenizer)

        prefix_len = self._encode_prefix_len
        suffix_len = self._encode_suffix_len
        batch_size = body.input_ids.size(0)

        prefix_ids_t = (
            torch.tensor(self._encode_prefix_ids, dtype=body.input_ids.dtype)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        suffix_ids_t = (
            torch.tensor(self._encode_suffix_ids, dtype=body.input_ids.dtype)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        prefix_mask_t = torch.ones(
            batch_size, prefix_len, dtype=body.attention_mask.dtype
        )
        suffix_mask_t = torch.ones(
            batch_size, suffix_len, dtype=body.attention_mask.dtype
        )

        input_ids = torch.cat((prefix_ids_t, body.input_ids, suffix_ids_t), dim=-1)
        attention_mask = torch.cat(
            (prefix_mask_t, body.attention_mask, suffix_mask_t), dim=-1
        )
        # Return a BatchEncoding so TextEncodingStage can call `.to(device)` on
        # the result and index input_ids / attention_mask like a tokenizer output.
        return BatchEncoding(
            data={"input_ids": input_ids, "attention_mask": attention_mask}
        )

    # --- TimestepPreparationStage hook ---

    def prepare_sigmas(self, sigmas, num_inference_steps):
        if sigmas is None:
            sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
        return sigmas

    def get_pos_prompt_embeds(self, batch):
        return batch.prompt_embeds[0]

    def get_neg_prompt_embeds(self, batch):
        return batch.negative_prompt_embeds[0]

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        # txt_ids depend only on the prompt length (fixed at TOKENIZER_MAX_LENGTH);
        # img_ids are the latent position ids set by LatentPreparationStage.
        # image_rotary_emb is NOT precomputed — the DiT computes it on the fly from
        # txt_ids + img_ids (matching diffusers' transformer, which calls
        # self.pos_embed internally).
        num_token = batch.prompt_embeds[0].shape[1]
        return {
            "txt_ids": _prepare_pos_ids(
                modality_id=0, token_type="text", start=(0, 0), num_token=num_token
            ).to(device),
            "img_ids": batch.latent_ids,
        }

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        num_token = batch.negative_prompt_embeds[0].shape[1]
        return {
            "txt_ids": _prepare_pos_ids(
                modality_id=0, token_type="text", start=(0, 0), num_token=num_token
            ).to(device),
            "img_ids": batch.latent_ids,
        }

    def get_decode_scale_and_shift(self, device, dtype, vae):
        # scaling_factor/shift_factor live on the VAE runtime config (not the
        # arch config). AutoencoderKL always has them; access directly so a
        # missing field raises instead of silently decoding with 1.0/0.0.
        return vae.config.scaling_factor, vae.config.shift_factor

    def post_denoising_loop(self, latents, batch):
        vae_scale_factor = self.vae_config.get_vae_scale_factor()
        latents = _unpack_latents(latents, batch.height, batch.width, vae_scale_factor)
        # Add frames dimension for DecodingStage compatibility: [B, C, H, W] -> [B, C, 1, H, W]
        latents = latents.unsqueeze(2)
        return latents

    def preprocess_decoding(self, latents, server_args=None, vae=None):
        """Remove frames dimension before VAE decode: [B, C, 1, H, W] -> [B, C, H, W]."""
        if latents.dim() == 5 and latents.shape[2] == 1:
            latents = latents.squeeze(2)
        return latents

    def postprocess_cfg_noise(
        self,
        batch,
        noise_pred: torch.Tensor,
        noise_pred_cond: torch.Tensor,
    ) -> torch.Tensor:
        enable_cfg_renorm = getattr(batch, "enable_cfg_renorm", True)
        cfg_renorm_min = getattr(batch, "cfg_renorm_min", 0.0)
        if not enable_cfg_renorm:
            return noise_pred
        cond_norm = torch.norm(noise_pred_cond, dim=-1, keepdim=True)
        noise_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
        scale = (cond_norm / (noise_norm + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
        return noise_pred * scale


def _calculate_edit_dimensions(target_area, ratio):
    """Output size for LongCat-Image-Edit: fit `target_area`, ceil to /16.

    Copied from diffusers pipeline_longcat_image_edit.calculate_dimensions.
    Note this intentionally differs from sglang.multimodal_gen.utils
    calculate_dimensions (which rounds to /32).
    """
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = width if width % 16 == 0 else (width // 16 + 1) * 16
    height = height if height % 16 == 0 else (height // 16 + 1) * 16

    return int(width), int(height)


@dataclass
class LongCatImageEditPipelineConfig(LongCatImagePipelineConfig):
    """Configuration for the LongCat-Image-Edit I2I pipeline.

    Mirrors diffusers LongCatImageEditPipeline: the reference image is resized
    to the output resolution, VAE-encoded with argmax sampling, packed, and
    concatenated after the noisy latents along the sequence dim. RoPE ids use
    modality 1 for noisy tokens and modality 2 for reference tokens, both
    offset by the full text sequence length (VL image tokens + 512 body).
    """

    task_type: ModelTaskType = ModelTaskType.I2I
    # diffusers reference draws latent noise with a CPU generator
    generator_device: str = "cpu"

    # --- InputValidationStage hooks ---

    def calculate_condition_image_size(self, image, width, height):
        return _calculate_edit_dimensions(1024 * 1024, width / height)

    # --- LatentPreparationStage hooks ---

    def get_latent_dtype(self, prompt_dtype: torch.dtype) -> torch.dtype:
        # The edit reference draws noise directly in the prompt-embeds dtype
        # (bf16), unlike the T2I reference which draws in float32.
        return prompt_dtype

    def maybe_prepare_latent_ids(self, latents):
        # img_ids (noisy + reference) are built per-step in
        # prepare_*_cond_kwargs because the text start offset depends on the
        # VL image token count, which is unknown at latent preparation time.
        return None

    # --- ImageVAEEncodingStage hooks ---

    def preprocess_vae_encode(self, image, vae):
        # AutoencoderKL is a 2D image VAE; drop the frames dim added by
        # ImageVAEEncodingStage ([B, C, 1, H, W] -> [B, C, H, W]).
        if image.dim() == 5 and image.shape[2] == 1:
            image = image.squeeze(2)
        return image

    def postprocess_image_latent(self, latent_condition, batch):
        if latent_condition.dim() == 5 and latent_condition.shape[2] == 1:
            latent_condition = latent_condition.squeeze(2)
        batch_size = batch.batch_size
        if batch_size > latent_condition.shape[0]:
            if batch_size % latent_condition.shape[0] != 0:
                raise ValueError(
                    f"Cannot duplicate reference image of batch size "
                    f"{latent_condition.shape[0]} to {batch_size} prompts."
                )
            latent_condition = latent_condition.repeat(
                batch_size // latent_condition.shape[0], 1, 1, 1
            )
        _, num_channels_latents, height, width = latent_condition.shape
        return _pack_latents(
            latent_condition, batch_size, num_channels_latents, height, width
        )

    # --- Denoising hooks ---

    def shard_latents_for_sp(self, batch, latents):
        # (h/2)*(w/2) is odd at most ~1MP edit resolutions, so SP has to pad, and
        # the pads stay unmasked (USPAttention rejects a mask alongside the
        # replicated text prefix). Repeat the last token instead of the base
        # class's zeros, which would carry the RoPE of text token 0.
        if latents.dim() == 3 and model_parallel_is_initialized():
            sp_world_size = get_sp_world_size()
            remainder = latents.shape[1] % sp_world_size
            if remainder:
                pad = latents[:, -1:].expand(-1, sp_world_size - remainder, -1)
                latents = torch.cat([latents, pad], dim=1)
        return super().shard_latents_for_sp(batch, latents)

    def _maybe_shard_pos_ids_for_sp(self, batch, pos_ids):
        # RoPE ids must be sharded exactly like their latents, so reuse the same
        # helper. It reads the SP group, which only exists under model parallelism.
        if not model_parallel_is_initialized() or get_sp_world_size() == 1:
            return pos_ids
        sharded, _ = self.shard_latents_for_sp(batch, pos_ids.unsqueeze(0))
        return sharded.squeeze(0)

    def _edit_img_ids(self, batch, num_token, device):
        """Position ids for [noisy | reference] packed latent tokens.

        The reference image is resized to the output resolution, so both grids
        share the same shape; they differ only in modality id (1 vs 2).
        """
        vae_scale_factor = self.vae_config.get_vae_scale_factor()
        h = 2 * (int(batch.height) // (vae_scale_factor * 2))
        w = 2 * (int(batch.width) // (vae_scale_factor * 2))
        noisy_ids = _prepare_pos_ids(
            modality_id=1,
            token_type="image",
            start=(num_token, num_token),
            height=h // 2,
            width=w // 2,
        )
        ref_ids = _prepare_pos_ids(
            modality_id=2,
            token_type="image",
            start=(num_token, num_token),
            height=h // 2,
            width=w // 2,
        )
        noisy_ids = self._maybe_shard_pos_ids_for_sp(batch, noisy_ids)
        ref_ids = self._maybe_shard_pos_ids_for_sp(batch, ref_ids)
        return torch.cat([noisy_ids, ref_ids], dim=0).to(device)

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        num_token = batch.prompt_embeds[0].shape[1]
        return {
            "txt_ids": _prepare_pos_ids(
                modality_id=0, token_type="text", start=(0, 0), num_token=num_token
            ).to(device),
            "img_ids": self._edit_img_ids(batch, num_token, device),
        }

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        num_token = batch.negative_prompt_embeds[0].shape[1]
        return {
            "txt_ids": _prepare_pos_ids(
                modality_id=0, token_type="text", start=(0, 0), num_token=num_token
            ).to(device),
            "img_ids": self._edit_img_ids(batch, num_token, device),
        }

    def slice_noise_pred(self, noise, latents):
        # Drop predictions over the appended reference-image tokens.
        return noise[:, : latents.size(1)]

    def post_denoising_loop(self, latents, batch):
        # The SP gather leaves the noisy latents at their padded length; trim the
        # trailing pad tokens before unpacking to the (h/2)*(w/2) grid.
        if latents.dim() == 3:
            vae_scale_factor = self.vae_config.get_vae_scale_factor()
            h = 2 * (int(batch.height) // (vae_scale_factor * 2))
            w = 2 * (int(batch.width) // (vae_scale_factor * 2))
            expected = (h // 2) * (w // 2)
            if latents.shape[1] > expected:
                latents = latents[:, :expected, :]
        return super().post_denoising_loop(latents, batch)
