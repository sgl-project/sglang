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
"""Logits processing."""

import torch
import triton
import triton.language as tl

FMIX32_C1 = 0x85EBCA6B
FMIX32_C2 = 0xC2B2AE35
POS_C1 = 0x27D4EB2D
POS_C2 = 0x165667B1


@triton.jit
def _rotl32(x, r: tl.constexpr):
    return (x << r) | (x >> (32 - r))


@triton.jit
def _fmix32(x, C1: tl.constexpr, C2: tl.constexpr):
    c1 = tl.full((), C1, tl.uint32)
    c2 = tl.full((), C2, tl.uint32)
    x ^= x >> 16
    x = x * c1
    x ^= x >> 13
    x = x * c2
    x ^= x >> 16
    return x


@triton.jit
def hash_tiles32_kernel_blocked(
    in_ptr,
    out_ptr,
    n_u32,
    seed1,
    seed2,
    FM_C1: tl.constexpr,
    FM_C2: tl.constexpr,
    POS_A: tl.constexpr,
    POS_B: tl.constexpr,
    TILE: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_CG: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    base = pid * TILE

    s1 = tl.full((), seed1, tl.uint32)
    s2 = tl.full((), seed2, tl.uint32)
    posA = tl.full((), POS_A, tl.uint32)
    posB = tl.full((), POS_B, tl.uint32)

    h1 = tl.zeros((), dtype=tl.uint32)
    h2 = tl.zeros((), dtype=tl.uint32)

    for off in tl.static_range(0, TILE, BLOCK):
        idx = base + off + tl.arange(0, BLOCK)
        m = idx < n_u32

        if USE_CG:
            v = tl.load(in_ptr + idx, mask=m, other=0, cache_modifier=".cg")
        else:
            v = tl.load(in_ptr + idx, mask=m, other=0)
        v = v.to(tl.uint32)

        iu = idx.to(tl.uint32)
        p1 = (iu * posA + s1) ^ _rotl32(iu, 15)
        p2 = (iu * posB + s2) ^ _rotl32(iu, 13)

        k1 = _fmix32(v ^ p1, C1=FM_C1, C2=FM_C2)
        k2 = _fmix32(v ^ p2, C1=FM_C1, C2=FM_C2)

        zero32 = tl.zeros_like(k1)
        k1 = tl.where(m, k1, zero32)
        k2 = tl.where(m, k2, zero32)

        h1 += tl.sum(k1, axis=0).to(tl.uint32)
        h2 += tl.sum(k2, axis=0).to(tl.uint32)

    nbytes = tl.full((), n_u32 * 4, tl.uint32)
    h1 ^= nbytes
    h2 ^= nbytes
    h1 = _fmix32(h1, C1=FM_C1, C2=FM_C2)
    h2 = (
        _fmix32(h2, C1=FMIX32_C1, C2=FMIX32_C2)
        if False
        else _fmix32(h2, C1=FM_C1, C2=FM_C2)
    )

    out = (h1.to(tl.uint64) << 32) | h2.to(tl.uint64)
    tl.store(out_ptr + pid, out)


@triton.jit
def add_tree_reduce_u64_kernel(in_ptr, out_ptr, n_elems, CHUNK: tl.constexpr):
    pid = tl.program_id(axis=0)
    start = pid * CHUNK
    h = tl.zeros((), dtype=tl.uint64)
    for i in tl.static_range(0, CHUNK):
        idx = start + i
        m = idx < n_elems
        v = tl.load(in_ptr + idx, mask=m, other=0).to(tl.uint64)
        h += v
    tl.store(out_ptr + pid, h)


def _as_uint32_words(t: torch.Tensor) -> torch.Tensor:
    assert t.is_cuda, "Use .cuda() first"
    tb = t.contiguous().view(torch.uint8)
    nbytes = tb.numel()
    pad = (4 - (nbytes & 3)) & 3
    if pad:
        tb_p = torch.empty(nbytes + pad, dtype=torch.uint8, device=tb.device)
        tb_p[:nbytes].copy_(tb)
        tb_p[nbytes:].zero_()
        tb = tb_p
    return tb.view(torch.uint32)


def _final_splitmix64(x: int) -> int:
    mask = (1 << 64) - 1
    x &= mask
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & mask
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & mask
    x ^= x >> 31
    return x


@torch.inference_mode()
def gpu_tensor_hash(
    tensor: torch.Tensor,
    *,
    seed: int = 0x243F6A88,
    tile_words: int = 8192,
    block_words: int = 256,
    reduce_chunk: int = 1024,
    num_warps: int = 4,
    num_stages: int = 4,
    use_cg: bool = True,
) -> int:
    assert tensor.is_cuda, "Use .cuda() first"
    u32 = _as_uint32_words(tensor)
    n = u32.numel()
    if n == 0:
        return 0

    grid1 = (triton.cdiv(n, tile_words),)
    partials = torch.empty(grid1[0], dtype=torch.uint64, device=u32.device)
    hash_tiles32_kernel_blocked[grid1](
        u32,
        partials,
        n,
        seed1=seed & 0xFFFFFFFF,
        seed2=((seed * 0x9E3779B1) ^ 0xDEADBEEF) & 0xFFFFFFFF,
        FM_C1=FMIX32_C1,
        FM_C2=FMIX32_C2,
        POS_A=POS_C1,
        POS_B=POS_C2,
        TILE=tile_words,
        BLOCK=block_words,
        USE_CG=use_cg,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    cur = partials
    while cur.numel() > 1:
        n_elems = cur.numel()
        grid2 = (triton.cdiv(n_elems, reduce_chunk),)
        nxt = torch.empty(grid2[0], dtype=torch.uint64, device=cur.device)
        add_tree_reduce_u64_kernel[grid2](cur, nxt, n_elems, CHUNK=reduce_chunk)
        cur = nxt

    return _final_splitmix64(int(cur.item()))
