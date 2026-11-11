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
"""Multi-router top-k: R routers score the same experts and their sigmoids add.

``scores = sum_r sigmoid(logits[:, r*E:(r+1)*E])``; top-k of ``scores + bias``;
weights are ``scores`` at those ids, optionally renormalized, times
``per_expert_scale``. One program per token.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _sigmoid(x):
    # Bit-identical to torch.sigmoid on every bf16 input; tl.sigmoid differs on ~2%.
    return tl.math.div_rn(1.0, 1.0 + libdevice.exp(-x))


@triton.jit
def _multi_router_topk_kernel(
    logits_ptr,
    bias_ptr,
    scale_ptr,
    weights_ptr,
    ids_ptr,
    logits_stride,
    NUM_EXPERTS: tl.constexpr,
    NUM_ROUTERS: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_POW2: tl.constexpr,
    RENORMALIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    experts = tl.arange(0, NUM_EXPERTS)
    base = logits_ptr + row * logits_stride
    scores = tl.zeros([NUM_EXPERTS], dtype=tl.float32)
    for r in tl.static_range(NUM_ROUTERS):
        router_logits = tl.load(base + r * NUM_EXPERTS + experts).to(tl.float32)
        scores += _sigmoid(router_logits)
    biased = scores + tl.load(bias_ptr + experts).to(tl.float32)

    slots = tl.arange(0, TOPK_POW2)
    top_ids = tl.zeros([TOPK_POW2], dtype=tl.int32)
    top_scores = tl.zeros([TOPK_POW2], dtype=tl.float32)
    for k in tl.static_range(TOPK):
        best = tl.argmax(biased, axis=0).to(tl.int32)
        picked = experts == best
        top_ids = tl.where(slots == k, best, top_ids)
        top_scores = tl.where(
            slots == k, tl.sum(tl.where(picked, scores, 0.0)), top_scores
        )
        biased = tl.where(picked, float("-inf"), biased)

    if RENORMALIZE:
        denom = tl.sum(top_scores)
        top_scores = tl.math.div_rn(top_scores, tl.where(denom != 0, denom, 1.0))
    top_scores = top_scores * tl.load(scale_ptr + top_ids).to(tl.float32)
    tl.store(weights_ptr + row * TOPK + slots, top_scores, mask=slots < TOPK)
    tl.store(ids_ptr + row * TOPK + slots, top_ids, mask=slots < TOPK)


def multi_router_topk(
    logits: torch.Tensor,
    correction_bias: torch.Tensor,
    per_expert_scale: torch.Tensor,
    topk: int,
    renormalize: bool,
    *,
    num_routers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``logits`` is ``[T, num_routers * E]``, the routers' fused GEMM output with
    inner stride 1. Returns ``(weights fp32 [T, topk], ids int32 [T, topk])``.
    """
    num_tokens, width = logits.shape
    num_experts = width // num_routers
    assert (
        logits.stride(1) == 1
        and width == num_routers * num_experts
        and num_experts & (num_experts - 1) == 0
    ), (
        "the multi-router top-k needs row-major [T, num_routers * E] logits with a "
        f"power-of-two E, got {tuple(logits.shape)} (stride {logits.stride(1)}) "
        f"for {num_routers} routers"
    )
    weights = torch.empty(num_tokens, topk, dtype=torch.float32, device=logits.device)
    ids = torch.empty(num_tokens, topk, dtype=torch.int32, device=logits.device)
    if num_tokens == 0:
        return weights, ids
    _multi_router_topk_kernel[(num_tokens,)](
        logits,
        correction_bias,
        per_expert_scale,
        weights,
        ids,
        logits.stride(0),
        NUM_EXPERTS=num_experts,
        NUM_ROUTERS=num_routers,
        TOPK=topk,
        TOPK_POW2=triton.next_power_of_2(topk),
        RENORMALIZE=renormalize,
        # Measured (E=512, 2 routers, top-10): 4 warps up to ~512 tokens, 1 beyond.
        num_warps=4 if num_tokens <= 512 else 1,
    )
    return weights, ids
