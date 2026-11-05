"""
ComfyUI SGLang Diffusion nodes package.
"""

import logging

logger = logging.getLogger(__name__)

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]


def __getattr__(name):
    # Spawned workers import worker_patches without loading ComfyUI dependencies.
    # ComfyUI requests these mappings when it registers the custom nodes.
    if name not in ("NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    except ImportError as exc:
        # Missing ComfyUI or SGLang deps; log why, or every node vanishes silently.
        logger.error(
            "ComfyUI_SGLDiffusion failed to register nodes: %r. If this is a "
            "real ComfyUI install, install the diffusion extras with "
            "'pip install sglang[diffusion]'.",
            exc,
        )
        NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = {}, {}
    globals().update(
        NODE_CLASS_MAPPINGS=NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS=NODE_DISPLAY_NAME_MAPPINGS,
    )
    return globals()[name]
