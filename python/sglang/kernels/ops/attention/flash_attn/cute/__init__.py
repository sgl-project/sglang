"""Flash Attention CUTE (CUDA Template Engine) implementation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fa4")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .interface import (
    clear_shear_bias_workspace,
    flash_attn_func,
    flash_attn_varlen_func,
)

__all__ = [
    "clear_shear_bias_workspace",
    "flash_attn_func",
    "flash_attn_varlen_func",
]
