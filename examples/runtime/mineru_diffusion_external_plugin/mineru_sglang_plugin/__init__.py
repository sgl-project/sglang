"""MinerU-Diffusion plugin for SGLang external model loading."""

from .configuration_mineru_diffusion import MinerUDiffusionConfig, SDARConfig

try:
    from transformers import AutoConfig

    AutoConfig.register("mineru_diffusion", MinerUDiffusionConfig, exist_ok=True)
    AutoConfig.register("sdar", SDARConfig, exist_ok=True)
except Exception:
    pass


def _lazy_entry_class():
    from .modeling_mineru_diffusion import MinerUDiffusionForConditionalGeneration

    return MinerUDiffusionForConditionalGeneration


__all__ = [
    "MinerUDiffusionConfig",
    "SDARConfig",
    "_lazy_entry_class",
]
