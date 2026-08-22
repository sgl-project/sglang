import os
import tempfile
from getpass import getuser
from pathlib import Path

from sglang.kernels.jit.cute_aot_cache import get_jit_cache as _get_jit_cache


def get_flash_attn_jit_cache(name: str):
    cache_dir = os.getenv("SGLANG_CUTE_AOT_CACHE_DIR") or None
    if (
        cache_dir is None
        and os.getenv("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "0") == "1"
    ):
        cache_dir = os.getenv("FLASH_ATTENTION_CUTE_DSL_CACHE_DIR") or None
        if cache_dir is None:
            cache_dir = (
                Path(tempfile.gettempdir())
                / getuser()
                / "flash_attention_cute_dsl_cache"
            )
    return _get_jit_cache(
        name,
        cache_dir=cache_dir,
        source_paths=(Path(__file__).resolve().parent,),
        enable_tvm_ffi=True,
    )
