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
"""The "multimax" logit activation in one pointwise launch.

``x + t0*relu(r0-x) + t1*relu(x-r1) + t2*relu(r2-x)^2 + t3*relu(x-r3)^2`` with
every primitive rounded to bf16, so it is bit-identical to the eager spelling,
NaN and infinities included.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _round_bf16(value):
    # Round-to-nearest-even to bf16, kept in fp32 registers. The bit trick would
    # carry a NaN's mantissa into the sign bit, so NaN passes through unrounded.
    bits = value.to(tl.int32, bitcast=True)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & -65536
    return tl.where(value != value, value, bits.to(tl.float32, bitcast=True))


@triton.jit
def _relu(value):
    # F.relu keeps a NaN input, which tl.maximum alone would turn into 0.
    return tl.where(value != value, value, tl.maximum(value, 0))


@triton.jit
def _multimax_kernel(
    x_ptr, ranges_ptr, ts_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    # int64: 11K rows of a 200K vocab already pass 2**31 elements.
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    r0 = tl.load(ranges_ptr + 0).to(tl.float32)
    r1 = tl.load(ranges_ptr + 1).to(tl.float32)
    r2 = tl.load(ranges_ptr + 2).to(tl.float32)
    r3 = tl.load(ranges_ptr + 3).to(tl.float32)
    t0 = tl.load(ts_ptr + 0).to(tl.float32)
    t1 = tl.load(ts_ptr + 1).to(tl.float32)
    t2 = tl.load(ts_ptr + 2).to(tl.float32)
    t3 = tl.load(ts_ptr + 3).to(tl.float32)
    lower = _round_bf16(_relu(r0 - x))
    upper = _round_bf16(_relu(x - r1))
    lower_sq = _round_bf16(_relu(r2 - x))
    upper_sq = _round_bf16(_relu(x - r3))
    out = _round_bf16(x + _round_bf16(t0 * lower))
    out = _round_bf16(out + _round_bf16(t1 * upper))
    out = _round_bf16(out + _round_bf16(t2 * _round_bf16(lower_sq * lower_sq)))
    out = _round_bf16(out + _round_bf16(t3 * _round_bf16(upper_sq * upper_sq)))
    tl.store(out_ptr + offsets, out.to(out_ptr.dtype.element_ty), mask=mask)


def multimax_torch(
    x: torch.Tensor, ranges: torch.Tensor, ts: torch.Tensor
) -> torch.Tensor:
    out = x.clone()
    out += ts[0] * F.relu(ranges[0] - x)
    out += ts[1] * F.relu(x - ranges[1])
    out += ts[2] * F.relu(ranges[2] - x) ** 2
    out += ts[3] * F.relu(x - ranges[3]) ** 2
    return out


def multimax(x: torch.Tensor, ranges: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
    """``ranges`` / ``ts`` are 4-element bf16 tensors; a bf16 ``x`` runs the
    kernel, any other dtype the eager spelling."""
    if x.dtype != torch.bfloat16:
        return multimax_torch(x, ranges, ts)
    assert ranges.dtype == ts.dtype == torch.bfloat16, (
        f"multimax is bf16-only, got {ranges.dtype}, {ts.dtype}"
    )
    assert ranges.numel() == ts.numel() == 4
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return out
    # Arbitrary split between small (decode) and large (prefill) launches; not tuned.
    block = 1024 if n <= 1 << 17 else 2048
    _multimax_kernel[(triton.cdiv(n, block),)](
        x, ranges, ts, out, n, BLOCK_SIZE=block, num_warps=4 if block == 1024 else 8
    )
    return out
