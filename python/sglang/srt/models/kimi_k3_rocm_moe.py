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

``kimi_k3.py`` reads the window behind ``_is_hip`` and runs its own BF16 GEMM
whenever ``tuned_front_gemm`` returns ``None``.
"""

from typing import Optional

import torch

from sglang.srt.environ import envs

# Merged MoE front [shared gate-up | router | latent down] at TP8; AITER's gfx950
# tuned BF16 GEMM table has rows for this shape.
_TUNED_FRONT_SHAPES = ((6016, 7168),)


def tuned_front_window() -> Optional[tuple[int, int]]:
    if not (
        envs.SGLANG_USE_AITER.get() and envs.SGLANG_OPT_HIP_K3_TUNED_MOE_FRONT.get()
    ):
        return None
    return (
        envs.SGLANG_OPT_HIP_K3_TUNED_MOE_FRONT_MIN_TOKENS.get(),
        envs.SGLANG_OPT_HIP_K3_TUNED_MOE_FRONT_MAX_TOKENS.get(),
    )


def tuned_front_gemm(
    *,
    x: torch.Tensor,
    weight: torch.Tensor,
    out_dtype: Optional[torch.dtype],
    window: tuple[int, int],
) -> Optional[torch.Tensor]:
    # tgemm writes x.dtype, so an fp32 front stays on the caller's GEMM.
    if out_dtype is not None or tuple(weight.shape) not in _TUNED_FRONT_SHAPES:
        return None
    if not window[0] <= x.shape[0] <= window[1]:
        return None
    from aiter.tuned_gemm import tgemm

    return tgemm.mm(inp=x, weights=weight, otype=x.dtype)
