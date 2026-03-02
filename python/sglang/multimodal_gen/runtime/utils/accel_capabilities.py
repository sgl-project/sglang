"""Feature probes for optional acceleration libraries used by diffusion runtime."""

from importlib.util import find_spec


def has_module(module_name: str) -> bool:
    try:
        return find_spec(module_name) is not None
    except (ModuleNotFoundError, ValueError):
        return False


def has_triton() -> bool:
    return has_module("triton")


def has_sgl_kernel() -> bool:
    return has_module("sgl_kernel")


def has_flash_attn_runtime() -> bool:
    return has_module("sgl_kernel.flash_attn")
