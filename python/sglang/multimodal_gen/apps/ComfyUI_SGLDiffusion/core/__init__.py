"""
Core components for SGLang Diffusion ComfyUI integration.
Provides generator, model patcher, and server API client.
"""

from .generator import SGLDiffusionGenerator
from .server_api import SGLDiffusionServerAPI

__all__ = [
    "SGLDiffusionGenerator",
    "SGLDModelPatcher",
    "SGLDiffusionServerAPI",
]


def __getattr__(name):
    # ModelPatcher subclasses ComfyUI and is only needed when a graph loads a DiT.
    if name == "SGLDModelPatcher":
        from .model_patcher import SGLDModelPatcher

        return SGLDModelPatcher
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
