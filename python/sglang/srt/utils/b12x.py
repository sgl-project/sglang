# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Shared plumbing for the b12x backends (SM12x / GB10).

sglang runs b12x from more than one place -- the MXFP4 MoE runner and the
DSv4-Flash compressed-MLA attention branch -- and all of them need the same
three things: the pinned install hint, the generation guard, and the compile
cache pointed at sglang's cache tree. They live here so a pin move is one edit
and the guard cannot drift between backends.

The pin is the tree the official vLLM DGX Spark image ships (nightly-20260815,
``b4d6c7593``) plus the two upstream fixes sglang's staging needs -- see the
module docstring of :mod:`sglang.srt.layers.quantization.mxfp4_b12x_moe`.
"""

from __future__ import annotations

import os

from sglang.srt.environ import envs

B12X_PIN = "85d3681bb2c3749e4e94e121d5d829c7ec6f0b9f"

INSTALL_HINT = (
    "pip install --no-deps 'b12x @ https://github.com/local-inference-lab/"
    f"b12x/archive/{B12X_PIN}.tar.gz'"
)

# The op-registry entry points every b12x backend here is written against.
# A tree that has the module but not these names is a different generation.
_REQUIRED_FUSED_MOE_ATTRS = ("Caps", "plan", "plan_weights", "prepare_weights", "run")


def install_compile_cache_dir() -> None:
    """Route b12x's compile cache through sglang's cache tree.

    Same pattern as ``DG_JIT_CACHE_DIR`` in ``deep_gemm_wrapper``. b12x
    resolves ``B12X_COMPILE_CACHE_DIR`` at first compile, so any call before
    the first kernel launch is early enough; ``setdefault`` so an explicitly
    configured cache dir wins.

    The pre-op-registry name ``B12X_CUTE_COMPILE_CACHE_DIR`` must not be set:
    this generation ignores it, and unrecognized ``B12X_*`` variables are
    folded into the compile-cache key, which would fragment the cache.
    """
    os.environ.setdefault("B12X_COMPILE_CACHE_DIR", envs.SGLANG_B12X_CACHE_DIR.get())


def require_aligned_generation() -> None:
    """Fail unless the installed b12x is the pinned op-registry generation.

    The API moved twice (0.15.x ``b12x.integration.tp_moe`` -> PyPI 1.2.x same
    module, new kernels -> master op registry ``b12x.moe.fused_moe``), and the
    kernels changed with it, so a generation mismatch must fail here rather
    than surface as different numerics. The wheel version string cannot be
    used: it stayed 1.2.3 across the re-architecture. So the probe is
    structural -- the ``b12x.moe.fused_moe`` module has to exist, and it has
    to expose the op-registry entry points.
    """
    try:
        import b12x  # noqa: F401
    except ImportError as e:
        raise RuntimeError(f"b12x is not installed. {INSTALL_HINT}") from e
    try:
        from b12x.moe import fused_moe
    except ImportError as e:
        raise RuntimeError(
            "The installed b12x predates the op-registry API "
            f"(b12x.moe.fused_moe); this backend requires the pinned master "
            f"tree {B12X_PIN[:9]}. {INSTALL_HINT}"
        ) from e
    missing = [a for a in _REQUIRED_FUSED_MOE_ATTRS if not hasattr(fused_moe, a)]
    if missing:
        raise RuntimeError(
            f"b12x.moe.fused_moe is missing {', '.join(missing)}: this is not "
            f"the pinned op-registry generation {B12X_PIN[:9]}. {INSTALL_HINT}"
        )
