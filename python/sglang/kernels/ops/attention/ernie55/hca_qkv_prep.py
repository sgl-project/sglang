# Copyright 2026 Baidu ERNIE Team and The SGLang Team. All rights reserved.
#
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

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_QUANT_GROUP = 32


@cache_once
def _jit_module(page_size: int, use_pdl: bool) -> Module:
    args = make_cpp_args(page_size, use_pdl)
    return load_jit(
        "ernie55_hca_qkv_prep",
        *args,
        cuda_files=["ernie55/hca_qkv_prep.cuh"],
        cuda_wrappers=[("run", f"Ernie55HcaQkvPrepKernel<{args}>::run")],
    )


def hca_qkv_prep(
    qkv_a: torch.Tensor,
    q_weight: torch.Tensor,
    q_eps: float,
    kv_weight: torch.Tensor,
    kv_eps: float,
    freqs_real: torch.Tensor,
    positions: torch.Tensor,
    out_loc: torch.Tensor,
    kvcache: torch.Tensor,
    page_size: int,
    q_lora_rank: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    m = qkv_a.shape[0]
    q = torch.empty((m, q_lora_rank), dtype=torch.float8_e4m3fn, device=qkv_a.device)
    sf = torch.empty(
        (m, q_lora_rank // _QUANT_GROUP), dtype=torch.uint8, device=qkv_a.device
    )
    _jit_module(page_size, is_arch_support_pdl()).run(
        qkv_a,
        q_weight,
        q,
        sf,
        float(q_eps),
        kv_weight,
        freqs_real,
        positions,
        out_loc,
        kvcache,
        float(kv_eps),
        q_lora_rank,
    )
    return q, sf
