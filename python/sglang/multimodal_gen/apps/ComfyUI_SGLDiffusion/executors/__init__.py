"""
ComfyUI SGLang Diffusion executors package.
Provides executor classes for different model types.
"""

from .adapter import ComfyUIModelAdapter, PackedForward, get_adapter_class
from .base import SGLDiffusionExecutor
from .flux import FluxAdapter, FluxExecutor
from .zimage import ZImageAdapter, ZImageExecutor

# Qwen adapters import ComfyUI (`comfy.ldm.common_dit`). Keep that optional so
# unit tests and SGLD-only paths can load Flux / Z-Image without ComfyUI.

__all__ = [
    "ComfyUIModelAdapter",
    "PackedForward",
    "SGLDiffusionExecutor",
    "FluxAdapter",
    "FluxExecutor",
    "ZImageAdapter",
    "ZImageExecutor",
    "QwenImageExecutor",
    "QwenImageEditExecutor",
    "get_adapter_class",
]


def __getattr__(name):
    if name in {"QwenImageExecutor", "QwenImageEditExecutor", "QwenImageAdapter", "QwenImageEditAdapter"}:
        from .qwen_image import (
            QwenImageAdapter,
            QwenImageEditAdapter,
            QwenImageEditExecutor,
            QwenImageExecutor,
        )

        mapping = {
            "QwenImageAdapter": QwenImageAdapter,
            "QwenImageEditAdapter": QwenImageEditAdapter,
            "QwenImageExecutor": QwenImageExecutor,
            "QwenImageEditExecutor": QwenImageEditExecutor,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
