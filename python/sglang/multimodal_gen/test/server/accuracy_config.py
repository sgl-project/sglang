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
    "flux_2_ti2i": {ComponentType.TRANSFORMER: 0.99},
    "fast_hunyuan_video": {ComponentType.TRANSFORMER: 0.99},
}

# Optional per-case component skips: {case_id: {ComponentType: ComponentSkip}}
SKIP_COMPONENTS: Dict[str, Dict[ComponentType, ComponentSkip]] = {
    # Example:
    # "some_case_id": {ComponentType.TEXT_ENCODER: ComponentSkip("Diffusers baseline differs")},
    "flux_2_klein_image_t2i": {
        ComponentType.TRANSFORMER: ComponentSkip(
            "Diffusers transformer differs from SGLang baseline"
        ),
        ComponentType.TEXT_ENCODER: ComponentSkip(
            "Flux-2 klein text encoder weights do not align with SGLang baseline"
        ),
    },
    "qwen_image_layered_i2i": {
        ComponentType.VAE: ComponentSkip(
            "Diffusers VAE config mismatches checkpoint (conv_out shape mismatch)"
        )
    },
    "zimage_image_t2i": {
        ComponentType.TRANSFORMER: ComponentSkip(
            "SGLang ZImage transformer diverges from Diffusers baseline (CosSim ~0.61) despite matched weights and freqs"
        ),
        ComponentType.TEXT_ENCODER: ComponentSkip(
            "SGLang text encoder weights do not fully load from HF checkpoint"
        ),
    },
    "zimage_image_t2i_warmup": {
        ComponentType.TRANSFORMER: ComponentSkip(
            "SGLang ZImage transformer diverges from Diffusers baseline (CosSim ~0.61) despite matched weights and freqs"
        ),
        ComponentType.TEXT_ENCODER: ComponentSkip(
            "SGLang text encoder weights do not fully load from HF checkpoint"
        ),
    },
    "zimage_image_t2i_multi_lora": {
        ComponentType.TRANSFORMER: ComponentSkip(
            "SGLang ZImage transformer diverges from Diffusers baseline (CosSim ~0.61) despite matched weights and freqs"
        ),
        ComponentType.TEXT_ENCODER: ComponentSkip(
            "SGLang text encoder weights do not fully load from HF checkpoint"
        ),
    },
    "zimage_image_t2i_2_gpus": {
        ComponentType.TRANSFORMER: ComponentSkip(
            "SGLang ZImage transformer diverges from Diffusers baseline (CosSim ~0.61) despite matched weights and freqs"
        )
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
    "wan2_2_i2v_a14b_2gpu": {
        ComponentType.TRANSFORMER: ComponentSkip(
            "SGLang transformer loader rejects new parameters in HF checkpoint"
        )
    },
    "turbo_wan2_2_i2v_a14b_2gpu": {
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
            "SGLang WAN transformer diverges from Diffusers baseline (CosSim ~0.68-0.71) despite matched weights; optional norm_added_q params missing in SGLang model"
        )
    },
    "wan2_1_i2v_14b_lora_2gpu": {
        ComponentType.TRANSFORMER: ComponentSkip(
            "SGLang WAN transformer diverges from Diffusers baseline (CosSim ~0.68-0.71) despite matched weights; optional norm_added_q params missing in SGLang model"
        )
    },
    "wan2_1_i2v_14b_720P_2gpu": {
        ComponentType.TRANSFORMER: ComponentSkip(
            "SGLang WAN transformer diverges from Diffusers baseline (CosSim ~0.68-0.71) despite matched weights; optional norm_added_q params missing in SGLang model"
        )
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
