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
"""Kimi-K3 ROCm dispatch for the merged MoE front GEMM.

``KimiK3MoE`` builds a ``TunedFrontGemm`` once after merging its front weight,
and runs its own BF16 GEMM when there is none or the batch is outside the window.
"""

import functools
from typing import Callable, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.quantization import unquant
from sglang.srt.utils.log_utils import print_warning_once

# Merged MoE front [shared gate-up | router | latent down] at TP8; other TPs differ.
# Only this shape has tuned bf16 rows (ROCm/aiter#4834, gfx950 with 256 CUs only).
_TUNED_FRONT_SHAPE = (6016, 7168)
_TUNED_FRONT_TARGET = ("gfx950", 256)


class TunedFrontGemm:
    """AITER tuned BF16 GEMM for front batches within an inclusive token window."""

    def __init__(self, *, mm: Callable, min_tokens: int, max_tokens: int) -> None:
        self._mm = mm
        self._min_tokens = min_tokens
        self._max_tokens = max_tokens

    def covers(self, num_tokens: int) -> bool:
        return self._min_tokens <= num_tokens <= self._max_tokens

    def __call__(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return self._mm(inp=x, weights=weight, otype=x.dtype)


@functools.cache
def _first_untuned_tokens(
    *, get_tuned_config: Callable, n: int, k: int, min_tokens: int, max_tokens: int
):
    """First batch size in the window with no tuned bf16 row, or None."""
    for m in range(min_tokens, max_tokens + 1):
        config = get_tuned_config(
            M=m,
            N=n,
            K=k,
            bias=False,
            dtype=str(torch.bfloat16),
            otype=str(torch.bfloat16),
        )
        if config is None:
            return m
    return None


def build_tuned_front_gemm(*, weight: torch.Tensor) -> Optional[TunedFrontGemm]:
    if not (
        envs.SGLANG_USE_AITER.get() and envs.SGLANG_OPT_HIP_K3_TUNED_MOE_FRONT.get()
    ):
        return None
    min_tokens = envs.SGLANG_OPT_HIP_K3_TUNED_MOE_FRONT_MIN_TOKENS.get()
    max_tokens = envs.SGLANG_OPT_HIP_K3_TUNED_MOE_FRONT_MAX_TOKENS.get()
    if not 1 <= min_tokens <= max_tokens:
        raise ValueError(
            "SGLANG_OPT_HIP_K3_TUNED_MOE_FRONT_{MIN,MAX}_TOKENS must satisfy "
            f"1 <= MIN <= MAX; got {min_tokens}, {max_tokens}."
        )
    # Declines that need no AITER come first, so they warn even on an AITER without tgemm.
    if unquant.is_aiter_unquant_skipped():
        print_warning_once(
            "SGLANG_OPT_HIP_K3_TUNED_MOE_FRONT is ignored with deterministic "
            "inference: AITER picks the kernel and split-K by batch size."
        )
        return None
    shape = tuple(weight.shape)
    if weight.dtype != torch.bfloat16 or shape != _TUNED_FRONT_SHAPE:
        print_warning_once(
            "SGLANG_OPT_HIP_K3_TUNED_MOE_FRONT is set but only covers a bf16 "
            f"{_TUNED_FRONT_SHAPE} front (TP8); got {weight.dtype} {shape}, "
            "keeping the default GEMM."
        )
        return None
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx
    from aiter.tuned_gemm import tgemm

    try:
        from aiter.tuned_gemm import get_GEMM_A16W16_tuned_config
    except ImportError as e:
        raise ImportError(
            "SGLANG_OPT_HIP_K3_TUNED_MOE_FRONT=1 needs aiter.tuned_gemm."
            "get_GEMM_A16W16_tuned_config, which this AITER build lacks."
        ) from e

    target = (get_gfx(), get_cu_num())
    if target != _TUNED_FRONT_TARGET:
        print_warning_once(
            "SGLANG_OPT_HIP_K3_TUNED_MOE_FRONT is set but only covers the "
            f"front on {_TUNED_FRONT_TARGET}; got {target}, keeping the default GEMM."
        )
        return None
    # Without its tuned rows tgemm silently falls back to a default kernel.
    untuned = _first_untuned_tokens(
        get_tuned_config=get_GEMM_A16W16_tuned_config,
        n=shape[0],
        k=shape[1],
        min_tokens=min_tokens,
        max_tokens=max_tokens,
    )
    if untuned is not None:
        raise RuntimeError(
            "SGLANG_OPT_HIP_K3_TUNED_MOE_FRONT=1 but this AITER has no tuned bf16 GEMM "
            f"row for M={untuned}, N={shape[0]}, K={shape[1]} on {target}; it needs "
            "ROCm/aiter#4834 (rows for 1-128 and 192 tokens) and ROCm/aiter#6820 (rows "
            "for 144-176 tokens, which complete 129-192)."
        )
    return TunedFrontGemm(mm=tgemm.mm, min_tokens=min_tokens, max_tokens=max_tokens)
