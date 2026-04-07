from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from sglang.multimodal_gen.test.server.testcase_configs import DiffusionTestCase


class ComponentType(str, Enum):
    VAE = "vae"
    TRANSFORMER = "transformer"
    TEXT_ENCODER = "text_encoder"


@dataclass(frozen=True)
class ComponentSkip:
    reason: str


DEFAULT_TIMESTEP = 500.0
TIMESTEP_NORMALIZATION_FACTOR = 1000.0
I2V_IMAGE_DIM = 1280
I2V_TEXT_ENCODER_DIM = 5120

DEFAULT_TEXT_ENCODER_VOCAB_SIZE = 32000
TEXT_ENCODER_INPUT_SEED = 42
TEXT_ENCODER_TOKEN_MIN = 100
TEXT_ENCODER_TOKEN_MAX = 30000
TEXT_ENCODER_TOKEN_LENGTH = 32

# Default thresholds by component. Override per component/case if needed.
DEFAULT_THRESHOLDS = {
    ComponentType.VAE: 0.999,
    ComponentType.TRANSFORMER: 0.995,
    ComponentType.TEXT_ENCODER: 0.98,
}

# Optional per-case overrides: {case_id: {ComponentType: threshold}}
CASE_THRESHOLDS: Dict[str, Dict[ComponentType, float]] = {
    # Add overrides here when a specific model/component needs a different threshold.
    "flux_2_image_t2i": {ComponentType.TRANSFORMER: 0.99},
    "flux_2_image_t2i_layerwise_offload": {ComponentType.TRANSFORMER: 0.99},
    "flux_2_image_t2i_2_gpus": {ComponentType.TRANSFORMER: 0.99},
    "flux_2_klein_ti2i_2_gpus": {ComponentType.TRANSFORMER: 0.975},
    "flux_2_ti2i": {ComponentType.TRANSFORMER: 0.99},
    "flux_2_t2i_customized_vae_path": {ComponentType.TRANSFORMER: 0.99},
    "fast_hunyuan_video": {ComponentType.TRANSFORMER: 0.99},
    "fsdp-inference": {ComponentType.TRANSFORMER: 0.9935},
    "wan2_2_i2v_a14b_2gpu": {ComponentType.TRANSFORMER: 0.99},
    "wan2_2_t2v_a14b_2gpu": {ComponentType.TRANSFORMER: 0.99},
    "wan2_2_t2v_a14b_teacache_2gpu": {ComponentType.TRANSFORMER: 0.99},
    "wan2_2_t2v_a14b_lora_2gpu": {ComponentType.TRANSFORMER: 0.99},
    "zimage_image_t2i_2_gpus": {ComponentType.TRANSFORMER: 0.9935},
    "zimage_image_t2i_2_gpus_non_square": {ComponentType.TRANSFORMER: 0.9935},
}

# Active skip policy. Keep this limited to cases with current, concrete evidence
# of real divergence or unsupported reference loading in the harness.
SKIP_COMPONENTS: Dict[str, Dict[ComponentType, ComponentSkip]] = {
    "flux_image_t2i": {
        ComponentType.TEXT_ENCODER: ComponentSkip(
            "Text encoder diverges from HF baseline despite 100% matched weights (CosSim ~0.47)"
        )
    },
    "sana_image_t2i": {
        ComponentType.VAE: ComponentSkip(
            "HF AutoencoderDC checkpoint leaves required to_qkv_multiscale weights missing, so VAE transfer would compare against partially initialized reference weights"
        )
    },
    "mova_360p_1gpu": {
        ComponentType.TRANSFORMER: ComponentSkip(
            "HF reference transformer cannot be materialized from the video_dit repo layout"
        )
    },
    "qwen_image_t2i_cache_dit_enabled": {
        ComponentType.VAE: ComponentSkip(
            "Representative VAE accuracy is already covered by qwen_image_t2i for the same source component and topology"
        ),
        ComponentType.TRANSFORMER: ComponentSkip(
            "Representative transformer accuracy is already covered by qwen_image_t2i for the same source component and topology"
        ),
        ComponentType.TEXT_ENCODER: ComponentSkip(
            "Representative text encoder accuracy is already covered by qwen_image_t2i for the same source component and topology"
        ),
    },
    "flux_2_image_t2i_upscaling_4x": {
        ComponentType.VAE: ComponentSkip(
            "Representative VAE accuracy is already covered by flux_2_image_t2i for the same source component and topology"
        ),
        ComponentType.TRANSFORMER: ComponentSkip(
            "Representative transformer accuracy is already covered by flux_2_image_t2i for the same source component and topology"
        ),
        ComponentType.TEXT_ENCODER: ComponentSkip(
            "Representative text encoder accuracy is already covered by flux_2_image_t2i for the same source component and topology"
        ),
    },
    "layerwise_offload": {
        ComponentType.VAE: ComponentSkip(
            "Representative VAE accuracy is already covered by zimage_image_t2i for the same source component and topology"
        ),
        ComponentType.TRANSFORMER: ComponentSkip(
            "Representative transformer accuracy is already covered by zimage_image_t2i for the same source component and topology"
        ),
        ComponentType.TEXT_ENCODER: ComponentSkip(
            "Representative text encoder accuracy is already covered by zimage_image_t2i for the same source component and topology"
        ),
    },
    "zimage_image_t2i_fp8": {
        ComponentType.VAE: ComponentSkip(
            "Representative VAE accuracy is already covered by zimage_image_t2i for the same source component and topology"
        ),
        ComponentType.TEXT_ENCODER: ComponentSkip(
            "Representative text encoder accuracy is already covered by zimage_image_t2i for the same source component and topology"
        ),
    },
    "zimage_image_t2i_multi_lora": {
        ComponentType.VAE: ComponentSkip(
            "Representative VAE accuracy is already covered by zimage_image_t2i for the same source component and topology"
        ),
        ComponentType.TRANSFORMER: ComponentSkip(
            "Representative transformer accuracy is already covered by zimage_image_t2i for the same source component and topology"
        ),
        ComponentType.TEXT_ENCODER: ComponentSkip(
            "Representative text encoder accuracy is already covered by zimage_image_t2i for the same source component and topology"
        ),
    },
    "flux_2_ti2i": {
        ComponentType.VAE: ComponentSkip(
            "Representative VAE accuracy is already covered by flux_2_image_t2i for the same source component and topology"
        ),
        ComponentType.TRANSFORMER: ComponentSkip(
            "Representative transformer accuracy is already covered by flux_2_image_t2i for the same source component and topology"
        ),
        ComponentType.TEXT_ENCODER: ComponentSkip(
            "Representative text encoder accuracy is already covered by flux_2_image_t2i for the same source component and topology"
        ),
    },
    "flux_2_t2i_customized_vae_path": {
        ComponentType.VAE: ComponentSkip(
            "Customized VAE override points to FLUX.2 Tiny AutoEncoder, but the HF reference loader does not yet materialize a trustworthy matching VAE baseline"
        ),
        ComponentType.TRANSFORMER: ComponentSkip(
            "Representative transformer accuracy is already covered by flux_2_image_t2i for the same source component and topology"
        ),
        ComponentType.TEXT_ENCODER: ComponentSkip(
            "Representative text encoder accuracy is already covered by flux_2_image_t2i for the same source component and topology"
        ),
    },
    "wan2_1_t2v_1.3b_text_encoder_cpu_offload": {
        ComponentType.VAE: ComponentSkip(
            "Representative VAE accuracy is already covered by wan2_1_t2v_1.3b for the same source component and topology"
        ),
        ComponentType.TRANSFORMER: ComponentSkip(
            "Representative transformer accuracy is already covered by wan2_1_t2v_1.3b for the same source component and topology"
        ),
        ComponentType.TEXT_ENCODER: ComponentSkip(
            "Representative text encoder accuracy is already covered by wan2_1_t2v_1.3b for the same source component and topology"
        ),
    },
    "wan2_1_t2v_1.3b_teacache_enabled": {
        ComponentType.VAE: ComponentSkip(
            "Representative VAE accuracy is already covered by wan2_1_t2v_1.3b for the same source component and topology"
        ),
        ComponentType.TRANSFORMER: ComponentSkip(
            "Representative transformer accuracy is already covered by wan2_1_t2v_1.3b for the same source component and topology"
        ),
        ComponentType.TEXT_ENCODER: ComponentSkip(
            "Representative text encoder accuracy is already covered by wan2_1_t2v_1.3b for the same source component and topology"
        ),
    },
    "wan2_1_t2v_1.3b_frame_interp_2x": {
        ComponentType.VAE: ComponentSkip(
            "Representative VAE accuracy is already covered by wan2_1_t2v_1.3b for the same source component and topology"
        ),
        ComponentType.TRANSFORMER: ComponentSkip(
            "Representative transformer accuracy is already covered by wan2_1_t2v_1.3b for the same source component and topology"
        ),
        ComponentType.TEXT_ENCODER: ComponentSkip(
            "Representative text encoder accuracy is already covered by wan2_1_t2v_1.3b for the same source component and topology"
        ),
    },
    "wan2_1_t2v_1.3b_upscaling_4x": {
        ComponentType.VAE: ComponentSkip(
            "Representative VAE accuracy is already covered by wan2_1_t2v_1.3b for the same source component and topology"
        ),
        ComponentType.TRANSFORMER: ComponentSkip(
            "Representative transformer accuracy is already covered by wan2_1_t2v_1.3b for the same source component and topology"
        ),
        ComponentType.TEXT_ENCODER: ComponentSkip(
            "Representative text encoder accuracy is already covered by wan2_1_t2v_1.3b for the same source component and topology"
        ),
    },
    "wan2_1_t2v_1.3b_frame_interp_2x_upscaling_4x": {
        ComponentType.VAE: ComponentSkip(
            "Representative VAE accuracy is already covered by wan2_1_t2v_1.3b for the same source component and topology"
        ),
        ComponentType.TRANSFORMER: ComponentSkip(
            "Representative transformer accuracy is already covered by wan2_1_t2v_1.3b for the same source component and topology"
        ),
        ComponentType.TEXT_ENCODER: ComponentSkip(
            "Representative text encoder accuracy is already covered by wan2_1_t2v_1.3b for the same source component and topology"
        ),
    },
    "wan2_1_t2v_1_3b_lora_1gpu": {
        ComponentType.VAE: ComponentSkip(
            "Representative VAE accuracy is already covered by wan2_1_t2v_1.3b for the same source component and topology"
        ),
        ComponentType.TRANSFORMER: ComponentSkip(
            "Representative transformer accuracy is already covered by wan2_1_t2v_1.3b for the same source component and topology"
        ),
        ComponentType.TEXT_ENCODER: ComponentSkip(
            "Representative text encoder accuracy is already covered by wan2_1_t2v_1.3b for the same source component and topology"
        ),
    },
    "flux_2_ti2i_multi_image_cache_dit": {
        ComponentType.VAE: ComponentSkip(
            "Representative VAE accuracy is already covered by flux_2_image_t2i for the same source component and topology"
        ),
        ComponentType.TRANSFORMER: ComponentSkip(
            "Representative transformer accuracy is already covered by flux_2_image_t2i for the same source component and topology"
        ),
        ComponentType.TEXT_ENCODER: ComponentSkip(
            "Representative text encoder accuracy is already covered by flux_2_image_t2i for the same source component and topology"
        ),
    },
    "wan2_2_ti2v_5b": {
        ComponentType.TRANSFORMER: ComponentSkip(
            "SGLang transformer loader rejects new parameters in HF checkpoint"
        )
    },
    "fastwan2_2_ti2v_5b": {
        ComponentType.TRANSFORMER: ComponentSkip(
            "SGLang transformer loader rejects new parameters in HF checkpoint"
        )
    },
    "turbo_wan2_1_t2v_1.3b": {
        ComponentType.TRANSFORMER: ComponentSkip(
            "Weight transfer match ratio too low for reliable comparison"
        )
    },
    "wan2_1_i2v_14b_480P_2gpu": {
        ComponentType.TRANSFORMER: ComponentSkip(
            "Transformer diverges from Diffusers baseline in 2-GPU accuracy run (CosSim ~0.71) after full weight transfer and matching output shape"
        ),
        ComponentType.TEXT_ENCODER: ComponentSkip(
            "Text encoder diverges from HF baseline in 2-GPU SP-folded accuracy run (CosSim ~0.31) after 100% matched weight transfer"
        ),
    },
    "wan2_1_i2v_14b_lora_2gpu": {
        ComponentType.VAE: ComponentSkip(
            "Representative VAE accuracy is already covered by wan2_1_i2v_14b_720P_2gpu for the same source component and topology"
        ),
        ComponentType.TRANSFORMER: ComponentSkip(
            "Transformer diverges from Diffusers baseline in 2-GPU accuracy run (CosSim ~0.68) after full weight transfer and matching output shape"
        ),
        ComponentType.TEXT_ENCODER: ComponentSkip(
            "Text encoder diverges from HF baseline in 2-GPU SP-folded accuracy run (CosSim ~0.31) after 100% matched weight transfer"
        ),
    },
    "wan2_1_i2v_14b_720P_2gpu": {
        ComponentType.TRANSFORMER: ComponentSkip(
            "Transformer diverges from Diffusers baseline in 2-GPU accuracy run (CosSim ~0.68) after full weight transfer and matching output shape"
        ),
        ComponentType.TEXT_ENCODER: ComponentSkip(
            "Text encoder diverges from HF baseline in 2-GPU SP-folded accuracy run (CosSim ~0.31) after 100% matched weight transfer"
        ),
    },
    "wan2_2_i2v_a14b_2gpu": {
        ComponentType.TEXT_ENCODER: ComponentSkip(
            "Text encoder diverges from HF baseline in 2-GPU SP-folded accuracy run (CosSim ~0.31) after 100% matched weight transfer"
        )
    },
    "wan2_2_t2v_a14b_2gpu": {
        ComponentType.TEXT_ENCODER: ComponentSkip(
            "Text encoder diverges from HF baseline in 2-GPU SP-folded accuracy run (CosSim ~0.31) after 100% matched weight transfer"
        )
    },
    "wan2_2_t2v_a14b_teacache_2gpu": {
        ComponentType.VAE: ComponentSkip(
            "Representative VAE accuracy is already covered by wan2_2_t2v_a14b_2gpu for the same source component and topology"
        ),
        ComponentType.TRANSFORMER: ComponentSkip(
            "Representative transformer accuracy is already covered by wan2_2_t2v_a14b_2gpu for the same source component and topology"
        ),
        ComponentType.TEXT_ENCODER: ComponentSkip(
            "Text encoder diverges from HF baseline in 2-GPU SP-folded accuracy run (CosSim ~0.31) after 100% matched weight transfer"
        ),
    },
    "wan2_2_t2v_a14b_lora_2gpu": {
        ComponentType.VAE: ComponentSkip(
            "Representative VAE accuracy is already covered by wan2_2_t2v_a14b_2gpu for the same source component and topology"
        ),
        ComponentType.TRANSFORMER: ComponentSkip(
            "Representative transformer accuracy is already covered by wan2_2_t2v_a14b_2gpu for the same source component and topology"
        ),
        ComponentType.TEXT_ENCODER: ComponentSkip(
            "Text encoder diverges from HF baseline in 2-GPU SP-folded accuracy run (CosSim ~0.31) after 100% matched weight transfer"
        ),
    },
    "wan2_1_t2v_14b_2gpu": {
        ComponentType.TEXT_ENCODER: ComponentSkip(
            "Text encoder diverges from HF baseline in 2-GPU SP-folded accuracy run (CosSim ~0.31) after 100% matched weight transfer"
        )
    },
    "wan2_1_t2v_1.3b_cfg_parallel": {
        ComponentType.TEXT_ENCODER: ComponentSkip(
            "Text encoder diverges from HF baseline in 2-GPU accuracy run (CosSim ~0.31) after 100% matched weight transfer"
        )
    },
    "mova_360p_tp2": {
        ComponentType.TRANSFORMER: ComponentSkip(
            "HF reference transformer cannot be materialized from the MOVA video_dit repo layout"
        ),
        ComponentType.TEXT_ENCODER: ComponentSkip(
            "Text encoder diverges from HF baseline in 2-GPU accuracy run (CosSim ~0.31) after 100% matched weight transfer"
        ),
    },
    "mova_360p_ring1_uly2": {
        ComponentType.TRANSFORMER: ComponentSkip(
            "HF reference transformer cannot be materialized from the MOVA video_dit repo layout"
        ),
        ComponentType.TEXT_ENCODER: ComponentSkip(
            "Text encoder diverges from HF baseline in 2-GPU accuracy run (CosSim ~0.31) after 100% matched weight transfer"
        ),
    },
    "mova_360p_ring2_uly1": {
        ComponentType.TRANSFORMER: ComponentSkip(
            "HF reference transformer cannot be materialized from the MOVA video_dit repo layout"
        ),
        ComponentType.TEXT_ENCODER: ComponentSkip(
            "Text encoder diverges from HF baseline in 2-GPU accuracy run (CosSim ~0.31) after 100% matched weight transfer"
        ),
    },
    "flux_image_t2i_2_gpus": {
        ComponentType.TEXT_ENCODER: ComponentSkip(
            "Text encoder diverges from HF baseline in 2-GPU accuracy run (CosSim ~0.47) after 100% matched weight transfer"
        )
    },
    "zimage_image_t2i_2_gpus_non_square": {
        ComponentType.VAE: ComponentSkip(
            "Representative VAE accuracy is already covered by zimage_image_t2i_2_gpus for the same source component and topology"
        ),
        ComponentType.TRANSFORMER: ComponentSkip(
            "Representative transformer accuracy is already covered by zimage_image_t2i_2_gpus for the same source component and topology"
        ),
        ComponentType.TEXT_ENCODER: ComponentSkip(
            "Representative text encoder accuracy is already covered by zimage_image_t2i_2_gpus for the same source component and topology"
        ),
    },
    "flux_2_image_t2i_2_gpus": {
        ComponentType.TRANSFORMER: ComponentSkip(
            "2-GPU FLUX.2 transformer diverges strongly from Diffusers baseline (CosSim ~0.54) despite full weight transfer"
        )
    },
    "hunyuan3d_shape_gen": {
        ComponentType.VAE: ComponentSkip(
            "HF config cannot be parsed as valid JSON for component reference loading"
        ),
        ComponentType.TRANSFORMER: ComponentSkip(
            "HF config cannot be parsed as valid JSON for component reference loading"
        ),
        ComponentType.TEXT_ENCODER: ComponentSkip(
            "HF config cannot be parsed as valid JSON for component reference loading"
        ),
    },
}

# TODO: If a model needs extra compatibility logic, prefer adding a skip or an
# explicit override here instead of adding more ad-hoc hacks in the engine.


def get_threshold(case_id: str, component: ComponentType) -> float:
    overrides = CASE_THRESHOLDS.get(case_id, {})
    return overrides.get(component, DEFAULT_THRESHOLDS[component])


def get_skip_reason(case: DiffusionTestCase, component: ComponentType) -> Optional[str]:
    skip_entry = SKIP_COMPONENTS.get(case.id, {}).get(component)
    if skip_entry is None:
        return None
    return skip_entry.reason


def should_skip_component(case: DiffusionTestCase, component: ComponentType) -> bool:
    return get_skip_reason(case, component) is not None
