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
        # Outside ComfyUI (e.g. the workflow converter CLI) there is nothing to
        # register. Inside it, a missing SGLang dep would make every node vanish
        # silently: say why.
        outside_comfyui = (exc.name or "").split(".")[0] in ("comfy", "folder_paths")
        (logger.debug if outside_comfyui else logger.error)(
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
