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
    # Outside ComfyUI (e.g. the workflow converter CLI) there is nothing to register.
    # Inside it, a missing SGLang dep would make every node vanish silently: say why.
    outside_comfyui = (exc.name or "").split(".")[0] in ("comfy", "folder_paths")
    (logger.debug if outside_comfyui else logger.error)(
        "ComfyUI_SGLDiffusion failed to register nodes: %r. If this is a "
        "real ComfyUI install, install the diffusion extras with "
        "'pip install sglang[diffusion]'.",
        exc,
    )
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
