# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Union, Optional
import html

import ftfy
import regex as re
import torch

from sglang.multimodal_gen.configs.models import DiTConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig
from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput, T5Config
from sglang.multimodal_gen.configs.models.vaes import WanVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig, ModelTaskType


@dataclass
class LongCatDiTArchConfig(DiTArchConfig):
    """Extended DiTArchConfig with LongCat-specific fields.

    NOTE: This is for Phase 1 wrapper compatibility. For native model (Phase 2),
    use LongCatVideoConfig from fastvideo.configs.models.dits.longcat instead.
    """
    # LongCat-specific architecture parameters
    adaln_tembed_dim: int = 512
    caption_channels: int = 4096
    depth: int = 48
    enable_bsa: bool = False
    enable_flashattn3: bool = False
    enable_flashattn2: bool = True
    enable_xformers: bool = False
    frequency_embedding_size: int = 256
    in_channels: int = 16
    mlp_ratio: int = 4
    num_heads: int = 32
    out_channels: int = 16
    num_channels_latents: int = 16
    text_tokens_zero_pad: bool = True
    patch_size: list[int] = field(default_factory=lambda: [1, 2, 2])
    cp_split_hw: Optional[list[int]] = None
    bsa_params: Optional[dict] = None


def longcat_preprocess_text(prompt: str) -> str:
    """Clean and preprocess text like original LongCat implementation.

    This function applies the same text cleaning pipeline as the original
    LongCat-Video implementation to ensure identical tokenization results.

    Steps:
    1. basic_clean: Fix unicode issues and unescape HTML entities
    2. whitespace_clean: Normalize whitespace to single spaces

    Args:
        prompt: Raw input text prompt

    Returns:
        Cleaned and normalized text prompt
    """
    # basic_clean: fix unicode and HTML entities
    text = ftfy.fix_text(prompt)
    text = html.unescape(html.unescape(text))
    text = text.strip()

    # whitespace_clean: normalize whitespace
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def umt5_postprocess_text(outputs: BaseEncoderOutput, _text_inputs) -> torch.Tensor:
    """
    Postprocess UMT5/T5 encoder outputs to fixed length 512 embeddings.
    """
    mask: torch.Tensor = outputs.attention_mask
    hidden_state: torch.Tensor = outputs.last_hidden_state
    seq_lens = mask.gt(0).sum(dim=1).long()
    assert torch.isnan(hidden_state).sum() == 0
    prompt_embeds = [u[:v] for u, v in zip(hidden_state, seq_lens, strict=True)]
    prompt_embeds_tensor: torch.Tensor = torch.stack([
        torch.cat([u, u.new_zeros(512 - u.size(0), u.size(1))])
        for u in prompt_embeds
    ],
        dim=0)
    return prompt_embeds_tensor


@dataclass
class LongCatT2V480PConfig(PipelineConfig):
    """Configuration for LongCat pipeline (480p) aligned to LongCat-Video modules.

    Components expected by loaders:
      - tokenizer: AutoTokenizer
      - text_encoder: UMT5EncoderModel
      - transformer: LongCatVideoTransformer3DModel (Phase 1 wrapper)
                  OR LongCatTransformer3DModel (Phase 2 native)
      - vae: AutoencoderKLWan (Wan VAE, 4x8 compression)
      - scheduler: FlowMatchEulerDiscreteScheduler
    """

    # DiT config with LongCat-specific arch_config
    # NOTE: For Phase 1 wrapper, uses LongCatDiTArchConfig
    # For Phase 2 native model, can use LongCatVideoConfig directly
    dit_config: DiTConfig = field(
        default_factory=lambda: DiTConfig(arch_config=LongCatDiTArchConfig()))
    task_type: ModelTaskType = ModelTaskType.T2V

    # VAE config: Wan VAE with encoder+decoder enabled
    vae_config: VAEConfig = field(default_factory=WanVAEConfig)
    vae_tiling: bool = False
    vae_sp: bool = False

    # Precision defaults
    dit_precision: str = "bf16"
    vae_precision: str = "bf16"
    text_encoder_precisions: tuple[str, ...] = field(
        default_factory=lambda: ("bf16",))

    # Text encoding (UMT5 uses T5-like config; postprocess to fixed 512)
    text_encoder_configs: tuple[T5Config, ...] = field(
        default_factory=lambda: (T5Config(),))
    preprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (longcat_preprocess_text,))
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor],
    ...] = field(default_factory=lambda:
    (umt5_postprocess_text,))

    # LongCat-specific runtime toggles (consumed by pipeline/stages)
    enable_kv_cache: bool = True
    offload_kv_cache: bool = False
    enable_bsa: bool = False
    use_distill: bool = False
    enhance_hf: bool = False
    # BSA runtime overrides (preferred over bsa_params if provided via CLI)
    bsa_sparsity: Optional[float] = None
    bsa_cdf_threshold: Optional[float] = None
    bsa_chunk_q: Optional[list[int]] = None
    bsa_chunk_k: Optional[list[int]] = None
    t_thresh: Optional[float] = None  # refine stage default controlled by sampling args

    # LongCat does not need flow_shift
    flow_shift: Optional[float] = None
    dmd_denoising_steps: Optional[list[int]] = None

    def __post_init__(self):
        # LongCat inference requires vae encoder and decoder
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True


@dataclass
class LongCatT2V704PConfig(LongCatT2V480PConfig):
    """Configuration for LongCat pipeline (704p) with BSA enabled by default.

    Uses the same resolution and BSA parameters as original LongCat refinement stage.
    BSA parameters configured in transformer config.json with chunk_3d_shape=[4,4,4]:
    - Input: 704×1280×96
    - VAE (8x): 88×160×96
    - Patch [1,2,2]: 44×80×96
    - chunk [4,4,4]: 96%4=0, 44%4=0, 80%4=0 ✅

    This configuration matches the original LongCat refinement stage parameters.
    """

    # Enable BSA by default for 704p
    enable_bsa: bool = True


ASPECT_RATIO_627 = {
    '0.26': ([320, 1216], 1),
    '0.31': ([352, 1120], 1),
    '0.38': ([384, 1024], 1),
    '0.43': ([416, 960], 1),
    '0.52': ([448, 864], 1),
    '0.58': ([480, 832], 1),
    '0.67': ([512, 768], 1),
    '0.74': ([544, 736], 1),
    '0.86': ([576, 672], 1),
    '0.95': ([608, 640], 1),
    '1.05': ([640, 608], 1),
    '1.17': ([672, 576], 1),
    '1.29': ([704, 544], 1),
    '1.35': ([736, 544], 1),
    '1.50': ([768, 512], 1),
    '1.67': ([800, 480], 1),
    '1.73': ([832, 480], 1),
    '2.00': ([896, 448], 1),
    '2.31': ([960, 416], 1),
    '2.58': ([992, 384], 1),
    '2.75': ([1056, 384], 1),
    '3.09': ([1088, 352], 1),
    '3.70': ([1184, 320], 1),
    '3.80': ([1216, 320], 1),
    '3.90': ([1248, 320], 1),
    '4.00': ([1280, 320], 1)
}

ASPECT_RATIO_627_F64 = {
    '0.26': ([320, 1216], 1),
    '0.38': ([384, 1024], 1),
    '0.50': ([448, 896], 1),
    '0.67': ([512, 768], 1),
    '0.82': ([576, 704], 1),
    '1.00': ([640, 640], 1),
    '1.22': ([704, 576], 1),
    '1.50': ([768, 512], 1),
    '1.86': ([832, 448], 1),
    '2.00': ([896, 448], 1),
    '2.50': ([960, 384], 1),
    '2.83': ([1088, 384], 1),
    '3.60': ([1152, 320], 1),
    '3.80': ([1216, 320], 1),
    '4.00': ([1280, 320], 1)
}

ASPECT_RATIO_627_F128 = {
    '0.25': ([256, 1024], 1),
    '0.38': ([384, 1024], 1),
    '0.43': ([384, 896], 1),
    '0.57': ([512, 896], 1),
    '0.67': ([512, 768], 1),
    '1.00': ([640, 640], 1),
    '1.50': ([768, 512], 1),
    '1.75': ([896, 512], 1),
    '2.33': ([896, 384], 1),
    '2.67': ([1024, 384], 1),
    '4.00': ([1024, 256], 1),
}

ASPECT_RATIO_627_F256 = {
    '0.25': ([256, 1024], 1),
    '0.33': ([256, 768], 1),
    '0.50': ([256, 512], 1),
    '0.67': ([512, 768], 1),
    '1.00': ([512, 512], 1),
    '1.50': ([768, 512], 1),
    '2.00': ([512, 256], 1),
    '3.00': ([768, 256], 1),
    '4.00': ([1024, 256], 1),
}

ASPECT_RATIO_960 = {
    '0.25': ([480, 1920], 1),
    '0.29': ([512, 1792], 1),
    '0.32': ([544, 1696], 1),
    '0.36': ([576, 1600], 1),
    '0.40': ([608, 1504], 1),
    '0.49': ([672, 1376], 1),
    '0.54': ([704, 1312], 1),
    '0.59': ([736, 1248], 1),
    '0.69': ([800, 1152], 1),
    '0.74': ([832, 1120], 1),
    '0.82': ([864, 1056], 1),
    '0.88': ([896, 1024], 1),
    '0.94': ([928, 992], 1),
    '1.00': ([960, 960], 1),
    '1.07': ([992, 928], 1),
    '1.14': ([1024, 896], 1),
    '1.22': ([1056, 864], 1),
    '1.31': ([1088, 832], 1),
    '1.35': ([1120, 832], 1),
    '1.44': ([1152, 800], 1),
    '1.70': ([1248, 736], 1),
    '2.00': ([1344, 672], 1),
    '2.05': ([1376, 672], 1),
    '2.47': ([1504, 608], 1),
    '2.53': ([1536, 608], 1),
    '2.83': ([1632, 576], 1),
    '3.06': ([1664, 544], 1),
    '3.12': ([1696, 544], 1),
    '3.62': ([1856, 512], 1),
    '3.93': ([1888, 480], 1),
    '4.00': ([1920, 480], 1)
}

ASPECT_RATIO_960_F64 = {
    '0.22': ([448, 2048], 1),
    '0.29': ([512, 1792], 1),
    '0.36': ([576, 1600], 1),
    '0.45': ([640, 1408], 1),
    '0.55': ([704, 1280], 1),
    '0.63': ([768, 1216], 1),
    '0.76': ([832, 1088], 1),
    '0.88': ([896, 1024], 1),
    '1.00': ([960, 960], 1),
    '1.14': ([1024, 896], 1),
    '1.31': ([1088, 832], 1),
    '1.50': ([1152, 768], 1),
    '1.58': ([1216, 768], 1),
    '1.82': ([1280, 704], 1),
    '1.91': ([1344, 704], 1),
    '2.20': ([1408, 640], 1),
    '2.30': ([1472, 640], 1),
    '2.67': ([1536, 576], 1),
    '2.89': ([1664, 576], 1),
    '3.62': ([1856, 512], 1),
    '3.75': ([1920, 512], 1)
}

ASPECT_RATIO_960_F128 = {
    '0.20': ([384, 1920], 1),
    '0.27': ([512, 1920], 1),
    '0.33': ([512, 1536], 1),
    '0.42': ([640, 1536], 1),
    '0.50': ([640, 1280], 1),
    '0.60': ([768, 1280], 1),
    '0.67': ([768, 1152], 1),
    '0.78': ([896, 1152], 1),
    '1.00': ([1024, 1024], 1),
    '1.29': ([1152, 896], 1),
    '1.50': ([1152, 768], 1),
    '1.67': ([1280, 768], 1),
    '2.00': ([1280, 640], 1),
    '2.40': ([1536, 640], 1),
    '3.00': ([1536, 512], 1),
    '3.75': ([1920, 512], 1),
    '5.00': ([1920, 384], 1),
}

ASPECT_RATIO_960_F256 = {
    '0.33': ([512, 1536], 1),
    '0.60': ([768, 1280], 1),
    '1.00': ([1024, 1024], 1),
    '1.67': ([1280, 768], 1),
    '3.00': ([1536, 512], 1),
}


def get_bucket_config(resolution, scale_factor_spatial):
    if resolution == '480p':
        if scale_factor_spatial == 16 or scale_factor_spatial == 32:
            return ASPECT_RATIO_627
        elif scale_factor_spatial == 64:
            return ASPECT_RATIO_627_F64
        elif scale_factor_spatial == 128:
            return ASPECT_RATIO_627_F128
        elif scale_factor_spatial == 256:
            return ASPECT_RATIO_627_F256
    elif resolution == '720p':
        if scale_factor_spatial == 16 or scale_factor_spatial == 32:
            return ASPECT_RATIO_960
        elif scale_factor_spatial == 64:
            return ASPECT_RATIO_960_F64
        elif scale_factor_spatial == 128:
            return ASPECT_RATIO_960_F128
        elif scale_factor_spatial == 256:
            return ASPECT_RATIO_960_F256

    raise ValueError(
        f"Unsupported resolution '{resolution}' or scale_factor_spatial '{scale_factor_spatial}'"
    )
