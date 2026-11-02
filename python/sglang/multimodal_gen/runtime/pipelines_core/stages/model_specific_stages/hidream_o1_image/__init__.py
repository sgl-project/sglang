"""Model-specific stages for the HiDream-O1-Image pipeline.

The three-stage set serves every checkpoint variant: before-denoising
(conditioning + initial noise), denoising (clean-image -> velocity loop with
CFG, co-batching, cache-dit), and decoding (pure unpatchify).
"""

from .generation import (
    HIDREAM_O1_BOI_TOKEN,
    HIDREAM_O1_MROPE_FIX_POINT,
    HIDREAM_O1_NOISE_SCALE,
    HIDREAM_O1_T_EPS,
    HIDREAM_O1_TMS_TOKEN,
    build_hidream_o1_attention_mask,
    build_hidream_o1_position_ids,
    patchify_pixels,
    unpatchify_pixels,
)
from .pipeline_stages import (
    HiDreamO1ImageBeforeDenoisingStage,
    HiDreamO1ImageDecodingStage,
    HiDreamO1ImageDenoisingStage,
    _resolve_single_prompt,
)

__all__ = [
    "HIDREAM_O1_BOI_TOKEN",
    "HIDREAM_O1_MROPE_FIX_POINT",
    "HIDREAM_O1_NOISE_SCALE",
    "HIDREAM_O1_TMS_TOKEN",
    "HIDREAM_O1_T_EPS",
    "HiDreamO1ImageBeforeDenoisingStage",
    "HiDreamO1ImageDecodingStage",
    "HiDreamO1ImageDenoisingStage",
    "_resolve_single_prompt",
    "build_hidream_o1_attention_mask",
    "build_hidream_o1_position_ids",
    "patchify_pixels",
    "unpatchify_pixels",
]
