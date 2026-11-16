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
"""VHA cross-head postmix fused into the wo_a fp8 activation quant.

``out[t, h] = sum_h' M[h, h'] * o[t, h']`` followed by
:func:`sglang_per_token_group_quant_fp8_dsv4_wo_a`, whose output layout this
reproduces: fp8 codes plus the packed UE8M0 scale words DeepGEMM's ``fp8_einsum``
reads. The mix is rounded to bf16 like the unfused matmul but never written to HBM.
Not bit-exact: the nh-term dot reduces in a different order than cuBLAS
(~1 bf16 ulp); with ``M = I`` it is bit-identical to the unfused pair.
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.fp8_group_major_quant_ue8m0 import (
    sglang_per_token_group_quant_fp8_dsv4_wo_a,
)

_GROUP = 128


@triton.jit
def _postmix_quant_kernel(
    x_ptr,  # [T, G, D] bf16, contiguous
    m_ptr,  # [NH, NH] bf16
    q_ptr,  # [T, G, D] fp8 out
    s_ptr,  # uint8 view of the [G, WORDS, aligned_tokens] int32 packed scales
    aligned_tokens,  # runtime: a constexpr would recompile per token count
    NH: tl.constexpr,
    HD: tl.constexpr,
    G: tl.constexpr,
    HPG: tl.constexpr,
    D: tl.constexpr,
    GROUP: tl.constexpr,
    QUARTERS: tl.constexpr,
    PER_PROG: tl.constexpr,
    WORDS: tl.constexpr,
):
    pid = tl.program_id(0)
    blocks: tl.constexpr = QUARTERS // PER_PROG
    # int64: t * G * D passes 2**31 at 64K tokens of a 64 x 512 head layout.
    t = (pid // blocks).to(tl.int64)
    blk = pid % blocks
    h = tl.arange(0, NH)
    hg = h // HPG
    hi = h % HPG
    col = tl.arange(0, GROUP)
    row = t * (G * D) + hg[:, None] * D + hi[:, None] * HD
    mix = tl.load(m_ptr + h[:, None] * NH + tl.arange(0, NH)[None, :])
    for j in tl.static_range(PER_PROG):
        q = blk * PER_PROG + j
        offs = row + q * GROUP + col[None, :]
        # fp32 accumulate, then bf16: what the unfused quant reads.
        y = tl.dot(mix, tl.load(x_ptr + offs)).to(tl.bfloat16).to(tl.float32)
        bits = (tl.maximum(tl.max(tl.abs(y), axis=1), 1e-10) * (1.0 / 448.0)).to(
            tl.uint32, bitcast=True
        )
        exp = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.uint32)
        inv_scale = ((254 - exp) << 23).to(tl.float32, bitcast=True)
        tl.store(
            q_ptr + offs,
            tl.minimum(y * inv_scale[:, None], 448.0).to(tl.float8e4nv),
        )
        # Byte k % 4 of word k // 4 for scale group k = head-in-group * QUARTERS + q.
        k = hi * QUARTERS + q
        tl.store(
            s_ptr + ((hg * WORDS + k // 4) * aligned_tokens + t) * 4 + (k % 4),
            exp.to(tl.uint8),
        )


def postmix_quant_supported(x: torch.Tensor, mix: torch.Tensor) -> bool:
    tokens, groups, hidden = x.shape
    num_heads = mix.shape[0]
    if tokens == 0 or num_heads % groups or (hidden * groups) % num_heads:
        return False
    head_dim = hidden * groups // num_heads
    return (
        x.is_contiguous()
        and x.dtype == mix.dtype == torch.bfloat16
        and mix.is_contiguous()
        and head_dim % _GROUP == 0
        and (hidden // _GROUP) % 4 == 0
        and num_heads >= 16
        # tl.arange needs a power-of-two head count.
        and num_heads & (num_heads - 1) == 0
    )


def postmix_fp8_wo_a_quant(
    x: torch.Tensor, mix: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``sglang_per_token_group_quant_fp8_dsv4_wo_a(postmix(x))``, same output layout.

    ``x`` is the ``[T, G, D]`` wo_a input, ``mix`` the ``[nh, nh]`` postmix matrix.
    Shapes the kernel does not cover mix with a matmul and quantize separately.
    """
    if not postmix_quant_supported(x, mix):
        tokens = x.shape[0]
        mixed = torch.matmul(mix, x.reshape(tokens, mix.shape[0], -1))
        return sglang_per_token_group_quant_fp8_dsv4_wo_a(
            mixed.reshape(x.shape).contiguous()
        )
    tokens, groups, hidden = x.shape
    num_heads = mix.shape[0]
    head_dim = hidden * groups // num_heads
    quarters = head_dim // _GROUP
    # Fewer quarters per program at small batches buys grid parallelism; at large
    # ones a program owns a whole token and loads the mix once.
    per_prog = 1 if tokens <= 64 else (2 if tokens <= 256 else quarters)
    while quarters % per_prog:
        per_prog -= 1
    words = hidden // _GROUP // 4
    aligned = (tokens + 3) // 4 * 4
    codes = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    storage = torch.empty((groups, words, aligned), device=x.device, dtype=torch.int32)
    _postmix_quant_kernel[(tokens * (quarters // per_prog),)](
        x,
        mix,
        codes,
        storage.view(torch.uint8),
        aligned,
        NH=num_heads,
        HD=head_dim,
        G=groups,
        HPG=num_heads // groups,
        D=hidden,
        GROUP=_GROUP,
        QUARTERS=quarters,
        PER_PROG=per_prog,
        WORDS=words,
        num_warps=4,
    )
    # MN-major packed words, as the logical [T, G, D/512] view fp8_einsum expects.
    return codes, storage.permute(0, 2, 1)[:, :tokens, :].transpose(0, 1)
