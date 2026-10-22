"""
ComfyUI SGLang Diffusion nodes package.
"""

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]


def __getattr__(name):
    # Spawned workers import worker_patches without loading ComfyUI dependencies.
    # ComfyUI requests these mappings when it registers the custom nodes.
    if name not in ("NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    except ImportError:
        # Preserve imports in environments without ComfyUI dependencies.
        NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = {}, {}
    globals().update(
        NODE_CLASS_MAPPINGS=NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS=NODE_DISPLAY_NAME_MAPPINGS,
    )
    return globals()[name]
