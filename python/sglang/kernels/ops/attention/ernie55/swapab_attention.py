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


import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout

PARTIAL_BLOCK = 64
_COMBINE_BLOCK_D = 128


@triton.jit
def _combine(
    PART,
    MAX,
    SUM,
    SINK,
    OUT,
    NT: tl.constexpr,
    ST: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
):
    b, h, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    t, d = tl.arange(0, ST), tile * BD + tl.arange(0, BD)
    sink = tl.load(SINK + h)
    tl.extra.cuda.gdc_wait()
    den = tl.load(SUM + (b * NT + t) * H + h, t < NT, 0)
    mx = tl.load(MAX + (b * NT + t) * H + h, t < NT, 0)
    mx = tl.where(den > 0, mx, -float("inf"))
    m = tl.maximum(tl.max(mx, 0), sink)
    m = tl.where(tl.abs(m) == float("inf"), 0.0, m)
    factor = tl.exp(mx - m)
    denominator = tl.sum(den * factor, 0) + tl.exp(sink - m)
    vals = tl.load(
        PART + ((b * NT + t[:, None]) * H + h) * D + d[None, :],
        t[:, None] < NT,
        0,
    )
    out = tl.sum(vals * factor[:, None], 0) / denominator
    out = tl.where((denominator > 0) & (sink != float("inf")), out, 0.0)
    tl.store(OUT + (b * H + h) * D + d, out)
    tl.extra.cuda.gdc_launch_dependents()


def swapab_partial(
    q,
    kv,
    indices,
    lengths,
    scale,
    *,
    extra_kv,
    extra_indices,
    extra_lengths,
    kv_layout: KVLayout,
    freqs,
    positions,
    q_eps: float = 1e-6,
    tile_mode: int = 0,
    tokens_per_req: int = 1,
    compress_ratio: int = 0,
    out=None,
    late_sync=None,
):
    from sglang.kernels.ops.attention.ernie55.swapab_gluon_partial import (
        partial_gluon,
    )

    bs, num_heads, head_dim = q.shape
    nk, ne = indices.shape[-1], extra_indices.shape[-1]
    kv = kv.view(torch.uint8)
    extra_kv = extra_kv.view(torch.uint8)
    kt = triton.cdiv(nk, PARTIAL_BLOCK)
    nt = kt + triton.cdiv(ne, PARTIAL_BLOCK)
    if out is not None:
        partial, maximum, sums = out
    else:
        partial = torch.empty(
            (bs, nt, num_heads, head_dim), dtype=torch.float32, device=q.device
        )
        maximum = torch.empty((bs, nt, num_heads), dtype=torch.float32, device=q.device)
        sums = torch.empty_like(maximum)
    partial_gluon[(bs, 1) if tile_mode == 2 else (bs, nt)](
        q,
        kv,
        extra_kv,
        indices,
        extra_indices,
        lengths,
        extra_lengths,
        partial,
        maximum,
        sums,
        freqs,
        positions,
        late_sync if late_sync is not None else lengths,
        q,
        QS=q.stride(0),
        QH=q.stride(1),
        IS=indices.stride(0),
        EIS=extra_indices.stride(0),
        KS=kv.stride(0),
        NK=nk,
        NE=ne,
        NT=nt,
        KT=kt,
        BT=PARTIAL_BLOCK,
        H=num_heads,
        D=head_dim,
        SCALE=scale,
        KTOKENS=kv.shape[0] * kv.shape[1],
        ROPE=freqs.shape[-1],
        KV_TILE=kv_layout.tile_size,
        KP=kv.shape[1],
        EP=extra_kv.shape[1],
        ES=extra_kv.stride(0),
        ETOKENS=extra_kv.shape[0] * extra_kv.shape[1],
        QEPS=q_eps,
        KV_ROW=kv_layout.data_bytes,
        KV_SROW=kv_layout.scale_bytes,
        TILE_MODE=tile_mode,
        C_RATIO=compress_ratio,
        TOKENS_PER_REQ=tokens_per_req,
        DECOUPLE=late_sync is not None,
        num_warps=4,
        launch_pdl=is_arch_support_pdl(),
    )
    return partial, maximum, sums, nt


def swapab_partial_mla(
    q_nope,
    q_rope,
    kv,
    topk,
    topk_lengths,
    window,
    window_lengths,
    scale,
):
    from sglang.kernels.ops.attention.ernie55.swapab_gluon_partial import (
        partial_gluon,
    )

    bs, num_heads, head_dim = q_nope.shape
    rope_dim, row_bytes = q_rope.shape[-1], kv.shape[-1]
    kv = kv.view(torch.uint8).reshape(-1, row_bytes)
    nk, ne = topk.shape[-1], window.shape[-1]
    kt = triton.cdiv(nk, PARTIAL_BLOCK)
    nt = kt + triton.cdiv(ne, PARTIAL_BLOCK)
    partial = torch.empty(
        (bs, nt, num_heads, head_dim), dtype=torch.float32, device=kv.device
    )
    maximum = torch.empty((bs, nt, num_heads), dtype=torch.float32, device=kv.device)
    sums = torch.empty_like(maximum)
    partial_gluon[(bs, nt)](
        q_nope,
        kv,
        kv,
        topk,
        window,
        topk_lengths,
        window_lengths,
        partial,
        maximum,
        sums,
        q_nope,
        topk_lengths,
        topk_lengths,
        q_rope,
        QS=q_nope.stride(0),
        QH=q_nope.stride(1),
        IS=topk.stride(0),
        EIS=window.stride(0),
        KS=kv.stride(0),
        NK=nk,
        NE=ne,
        NT=nt,
        KT=kt,
        BT=PARTIAL_BLOCK,
        H=num_heads,
        D=head_dim,
        SCALE=scale,
        KTOKENS=kv.shape[0],
        ROPE=rope_dim,
        # Each row is fp8 nope, one fp32 scale per KV_TILE values, then bf16 rope.
        KV_TILE=head_dim * 4 // (row_bytes - head_dim - 2 * rope_dim),
        QRS=q_rope.stride(0),
        QRH=q_rope.stride(1),
        MLA=True,
        num_warps=4,
        launch_pdl=is_arch_support_pdl(),
    )
    return partial, maximum, sums, nt


def swapab_combine_local(partial, maximum, sums, sink, nt):
    bs, _, num_heads, head_dim = partial.shape
    out = torch.empty(
        (bs, num_heads, head_dim), dtype=torch.bfloat16, device=partial.device
    )
    _combine[(bs, num_heads, head_dim // _COMBINE_BLOCK_D)](
        partial,
        maximum,
        sums,
        sink,
        out,
        NT=nt,
        ST=triton.next_power_of_2(nt),
        H=num_heads,
        D=head_dim,
        BD=_COMBINE_BLOCK_D,
        num_warps=2,
        launch_pdl=is_arch_support_pdl(),
    )
    return out
