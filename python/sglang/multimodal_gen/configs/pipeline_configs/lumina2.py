# SPDX-License-Identifier: Apache-2.0
#
# Pipeline configuration for Lumina-Image-2.0 text-to-image generation.
#
# Lumina-2 produces 4D spatial latents (B, 16, H/8, W/8), so this inherits
# SpatialImagePipelineConfig rather than a packed-token config. Conditioning
# comes from Gemma-2 hidden_states[-2], not last_hidden_state.

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.configs.models import DiTConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.lumina2 import Lumina2Config
from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput
from sglang.multimodal_gen.configs.models.encoders.base import EncoderConfig
from sglang.multimodal_gen.configs.models.encoders.gemma2 import (
    Gemma2ArchConfig,
    Gemma2Config,
)
from sglang.multimodal_gen.configs.models.vaes.flux import FluxVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    SpatialImagePipelineConfig,
)
from sglang.multimodal_gen.runtime.utils.condition_expansion import (
    PromptToSampleBatchExpander,
)

# Prepended to the user prompt before Gemma-2 encoding, matching the
# diffusers Lumina2Pipeline.
LUMINA2_SYSTEM_PROMPT = (
    "You are an assistant designed to generate superior images with the superior "
    "degree of image-text alignment based on textual prompts or user prompts."
)
# "<Prompt Start>" is part of Lumina's trained prompt template.
LUMINA2_PROMPT_SEPARATOR = " <Prompt Start> "

# Gemma-2 caption length Lumina-2 was trained with.
LUMINA2_MAX_TEXT_LEN = 256


@dataclass
class Lumina2GemmaArchConfig(Gemma2ArchConfig):
    text_len: int = LUMINA2_MAX_TEXT_LEN


@dataclass
class Lumina2GemmaConfig(Gemma2Config):
    arch_config: Gemma2ArchConfig = field(default_factory=Lumina2GemmaArchConfig)


def lumina2_preprocess_text(prompt: str) -> str:
    """Prepend Lumina-2's system prompt to the positive caption.

    Diffusers applies this template to the conditional branch only; Lumina's
    ``negative_preprocess_text_funcs`` keeps negative prompts raw.
    """
    return LUMINA2_SYSTEM_PROMPT + LUMINA2_PROMPT_SEPARATOR + prompt


def lumina2_postprocess_text(outputs: BaseEncoderOutput, _text_inputs) -> torch.Tensor:
    # Second-to-last hidden state, padding intact: the DiT slices each caption to
    # its true length via the attention mask, and unpadding here breaks that
    # alignment. TextEncodingStage always requests output_hidden_states, so the
    # stack is present regardless of the encoder arch config.
    return outputs.hidden_states[-2]


@dataclass
class Lumina2PipelineConfig(SpatialImagePipelineConfig):
    task_type: ModelTaskType = ModelTaskType.T2I

    # Disables *embedded* guidance (a timestep-conditioned guidance token).
    # Standard two-branch CFG via guidance_scale is still active.
    should_use_guidance: bool = False
    enable_autocast: bool = False
    vae_tiling: bool = False
    # VAE sequence parallelism requires tiling, which Lumina disables.
    vae_sp: bool = False
    vae_precision: str = "bf16"

    dit_config: DiTConfig = field(default_factory=Lumina2Config)
    # Lumina-2 ships the FLUX.1-dev VAE verbatim (vae/config.json records
    # _name_or_path: black-forest-labs/FLUX.1-dev). Its scaling/shift factors
    # arrive from that config.json via vae_loader, so nothing to restate here.
    vae_config: VAEConfig = field(default_factory=FluxVAEConfig)

    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (Lumina2GemmaConfig(),)
    )
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16",))

    preprocess_text_funcs: tuple[Callable[[str], str] | None, ...] = field(
        default_factory=lambda: (lumina2_preprocess_text,)
    )
    negative_preprocess_text_funcs: tuple[Callable[[str], str] | None, ...] = field(
        default_factory=lambda: (None,)
    )
    postprocess_text_funcs: tuple[Callable, ...] = field(
        default_factory=lambda: (lumina2_postprocess_text,)
    )

    def tokenize_prompt(self, prompts: list[str], tokenizer, tok_kwargs) -> dict:
        # A per-request max_sequence_length is honored but capped: caption
        # positions index the DiT's axis-0 RoPE table, and the DiT parks every
        # image token at axis-0 position cap_len -- hence one below the row count.
        requested = tok_kwargs.pop("max_length", None) or LUMINA2_MAX_TEXT_LEN
        max_rope_caption_len = self.dit_config.arch_config.axes_lens[0] - 1
        effective_max_length = min(requested, max_rope_caption_len)
        # Caption extents come from encoder_attention_mask.sum(), so real tokens
        # must precede padding for RoPE positions to remain aligned.
        tokenizer.padding_side = "right"
        return tokenizer(
            prompts,
            padding="max_length",
            max_length=effective_max_length,
            truncation=True,
            return_tensors="pt",
        )

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        compression = self.vae_config.arch_config.spatial_compression_ratio
        height = batch.height // compression
        width = batch.width // compression
        num_channels = self.dit_config.arch_config.num_channels_latents
        return (batch_size, num_channels, height, width)

    def expand_conditioning_to_sample_batch(self, batch):
        # NOTE: this override is load-bearing. Req.batch_size scales latents by
        # num_outputs_per_prompt but leaves conditioning at one row per prompt,
        # and the DiT sizes its loops from encoder_attention_mask. The surplus
        # samples would come back zeroed, in a prediction with fewer rows than
        # the latents, which scheduler.step broadcasts rather than rejects.
        expander = PromptToSampleBatchExpander.from_batch(batch)
        if expander is None:
            return batch

        for field_name in (
            "prompt_embeds",
            "negative_prompt_embeds",
            "prompt_attention_mask",
            "negative_attention_mask",
        ):
            expander.expand_field(batch, field_name)
        return batch

    def get_pos_prompt_embeds(self, batch):
        return batch.prompt_embeds[0]

    def get_neg_prompt_embeds(self, batch):
        return batch.negative_prompt_embeds[0]

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {"encoder_attention_mask": _first_mask(batch.prompt_attention_mask)}

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {"encoder_attention_mask": _first_mask(batch.negative_attention_mask)}

    def postprocess_cfg_noise(
        self,
        batch,
        noise_pred: torch.Tensor,
        noise_pred_cond: torch.Tensor,
    ) -> torch.Tensor:
        """Lumina-2 renorm-CFG.

        NOTE: deliberately not routed through the `cfg_normalization` sampling
        param. That flag drives CFGPolicy._apply_cfg_normalization, a *global*
        max-norm clip (one scalar norm over the whole tensor, applied only when
        the guided norm exceeds the conditional one). Lumina rescales
        unconditionally and per position along dim=-1. Enabling both applies two
        different rescales; see configs/sample/lumina2.py.

        No CFG guard is needed: CFGPolicy only reaches this hook after combining
        two branches.
        """
        cond_norm = torch.norm(noise_pred_cond, dim=-1, keepdim=True)
        noise_norm = torch.norm(noise_pred, dim=-1, keepdim=True).clamp_min(1e-12)
        return noise_pred * (cond_norm / noise_norm)

    def shard_latents_for_sp(self, batch, latents):
        # NOTE: opts out of SpatialImagePipelineConfig's H'-dimension sharding.
        # The Lumina-2 DiT has no sequence-parallel handling: _patchify_and_rope
        # numbers image rows from whatever tensor it is given, so a rank holding
        # rows H'/2..H' gets the wrong absolute RoPE positions, and every rank
        # prepends its own full copy of the caption. Neither raises -- the image
        # is just wrong. Real SP support needs both fixed first.
        return latents, False

    def gather_latents_for_sp(self, latents, batch=None):
        return latents

    def prepare_sigmas(self, sigmas, num_inference_steps):
        return self._prepare_sigmas(sigmas, num_inference_steps)

    def post_denoising_loop(self, latents, batch):
        return latents


def _first_mask(mask):
    if isinstance(mask, (list, tuple)):
        return mask[0] if mask else None
    return mask
