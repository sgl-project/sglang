# SPDX-License-Identifier: Apache-2.0
"""Bind Step5 compiled functions to SGLang's persistent CuTe AOT cache."""

import functools
from pathlib import Path

from sglang.kernels.jit.cute_aot_cache import get_jit_cache

_STEP5_CUTE_ROOT = Path(__file__).resolve().parent
_SOURCE_PATHS = (
    str(_STEP5_CUTE_ROOT),
    str(_STEP5_CUTE_ROOT.parents[1] / "flash_attn" / "cute"),
)


def cached_compile_function(fn):
    cache = get_jit_cache(
        f"step5/{fn.__module__}.{fn.__name__}",
        source_paths=_SOURCE_PATHS,
        enable_tvm_ffi=True,
    )

    @functools.wraps(fn)
    def compile_or_get(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = fn(*args, **kwargs)
        return cache[key]

    return compile_or_get
