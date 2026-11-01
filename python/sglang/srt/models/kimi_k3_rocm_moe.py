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

from typing import Callable, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.utils.common import print_warning_once

# Merged MoE front [shared gate-up | router | latent down] at TP8; other TPs differ.
# Only this shape has tuned bf16 rows (ROCm/aiter#6411, gfx950 only).
_TUNED_FRONT_SHAPE = (6016, 7168)
_TUNED_FRONT_ARCH = "gfx950"


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


def build_tuned_front_gemm(*, weight: torch.Tensor) -> Optional[TunedFrontGemm]:
    if not (
        envs.SGLANG_USE_AITER.get() and envs.SGLANG_OPT_HIP_K3_TUNED_MOE_FRONT.get()
    ):
        return None
    try:
        from aiter.jit.utils.chip_info import get_gfx
        from aiter.tuned_gemm import get_GEMM_A16W16_tuned_config, tgemm
    except ImportError as e:
        raise ImportError(
            "SGLANG_OPT_HIP_K3_TUNED_MOE_FRONT=1 needs aiter.tuned_gemm.tgemm and "
            "get_GEMM_A16W16_tuned_config, which this AITER build lacks."
        ) from e

    shape, gfx = tuple(weight.shape), get_gfx()
    if (
        weight.dtype != torch.bfloat16
        or shape != _TUNED_FRONT_SHAPE
        or gfx != _TUNED_FRONT_ARCH
    ):
        print_warning_once(
            "SGLANG_OPT_HIP_K3_TUNED_MOE_FRONT is set but only covers a bf16 "
            f"{_TUNED_FRONT_SHAPE} front (TP8) on {_TUNED_FRONT_ARCH}; got "
            f"{weight.dtype} {shape} on {gfx}, keeping the default GEMM."
        )
        return None
    # Without its tuned rows tgemm silently falls back to a default kernel.
    if (
        get_GEMM_A16W16_tuned_config(
            M=1,
            N=shape[0],
            K=shape[1],
            bias=False,
            dtype=str(torch.bfloat16),
            otype=str(torch.bfloat16),
        )
        is None
    ):
        raise RuntimeError(
            "SGLANG_OPT_HIP_K3_TUNED_MOE_FRONT=1 but this AITER has no tuned bf16 GEMM "
            f"row for N={shape[0]}, K={shape[1]} on {gfx}; it needs ROCm/aiter#6411."
        )
    return TunedFrontGemm(
        mm=tgemm.mm,
        min_tokens=envs.SGLANG_OPT_HIP_K3_TUNED_MOE_FRONT_MIN_TOKENS.get(),
        max_tokens=envs.SGLANG_OPT_HIP_K3_TUNED_MOE_FRONT_MAX_TOKENS.get(),
    )
