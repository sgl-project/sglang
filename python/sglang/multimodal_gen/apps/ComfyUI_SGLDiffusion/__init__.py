"""
ComfyUI SGLang Diffusion nodes package.
"""

import logging

logger = logging.getLogger(__name__)

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
except ImportError as exc:
    # ComfyUI dependencies not available (e.g., in test environment). Log it:
    # a silent except here previously made every node disappear from ComfyUI
    # with no clue why whenever any import in the chain failed.
    logger.error(
        "ComfyUI_SGLDiffusion failed to register nodes: %r. If this is a "
        "real ComfyUI install, install the diffusion extras with "
        "'pip install sglang[diffusion]'.",
        exc,
    )
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
