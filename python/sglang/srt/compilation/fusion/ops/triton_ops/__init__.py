# Copyright 2023-2025 SGLang Team
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

from typing import Optional

import torch

from sglang.srt.utils import direct_register_custom_op

from .dual_gemm import dual_gemm, dual_gemm_fake


def dual_gemm_fwd(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: Optional[torch.Tensor] = None,
    w_scale: Optional[torch.Tensor] = None,
    o_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.ops.sglang.dual_gemm(x, w, x_scale, w_scale, o_scale)


def register_triton_fused_ops():
    direct_register_custom_op(
        op_name="dual_gemm",
        op_func=dual_gemm,
        mutates_args=[],
        fake_impl=dual_gemm_fake,
    )
