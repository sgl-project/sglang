"""Flash Attention CUTE (CUDA Template Engine) implementation."""

__version__ = "0.1.0"

import cutlass.cute as cute

from .interface import (
    flash_attn_func,
    flash_attn_varlen_func,
)

from flash_attn.cute.cute_dsl_utils import cute_compile_patched

# Patch cute.compile to optionally dump SASS
cute.compile = cute_compile_patched


__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
]
