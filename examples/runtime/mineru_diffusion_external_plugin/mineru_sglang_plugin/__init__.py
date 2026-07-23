"""MinerU-Diffusion plugin for SGLang external model loading."""

import logging

from .configuration_mineru_diffusion import MinerUDiffusionConfig, SDARConfig

logger = logging.getLogger(__name__)

try:
    from transformers import AutoConfig

    AutoConfig.register("mineru_diffusion", MinerUDiffusionConfig, exist_ok=True)
    AutoConfig.register("sdar", SDARConfig, exist_ok=True)
except ImportError:
    logger.warning("transformers is not available; AutoConfig registration is skipped.")
except Exception as exc:
    logger.warning("AutoConfig registration failed: %s", exc)


def _lazy_entry_class():
    from .modeling_mineru_diffusion import MinerUDiffusionForConditionalGeneration

    return MinerUDiffusionForConditionalGeneration


__all__ = [
    "MinerUDiffusionConfig",
    "SDARConfig",
    "_lazy_entry_class",
]
