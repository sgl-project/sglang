"""
ComfyUI SGLang Diffusion nodes package.
"""

import logging

logger = logging.getLogger(__name__)

WEB_DIRECTORY = "./web"

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
except ImportError as exc:
    # Missing ComfyUI or SGLang deps; log why, or every node vanishes silently.
    logger.error(
        "ComfyUI_SGLDiffusion failed to register nodes: %r. If this is a "
        "real ComfyUI install, install the diffusion extras with "
        "'pip install sglang[diffusion]'.",
        exc,
    )
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
