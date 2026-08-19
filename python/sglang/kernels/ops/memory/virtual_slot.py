# Copyright 2023-2026 SGLang Team
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
"""Virtual<->physical slot Triton kernels for the unified memory pool."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Fused take-physical-pages + bind for the alloc fast path. Invoked ONLY when
# `_hole_count == 0`; otherwise the slow path drains holes first (Invariant B,
# greedy hole reuse). Caller advances `watermark_physical` and checks overflow
# BEFORE launch, passing the PRE-extension watermark. Cuda-graph safe (no
# `.item()`, no tensor branching); runs on the scheduler thread.


@triton.jit
def alloc_bind_inplace_kernel(
    v_pages_ptr,  # in: [N] int64 — virtual page ids
    v2p_ptr,  # in/out: int64 — virtual_to_physical table
    p2v_ptr,  # in/out: int64 — physical_to_virtual table
    out_phys_ptr,  # out: [N] int64 — physical page ids
    N,  # runtime: number of pages to allocate
    start_phys,  # runtime: lowest physical page id in the new range
    BLOCK: tl.constexpr,
):
    """Fused: ascending arange + out_phys/v2p/p2v scatter.

    Caller pre-adjusts `start_phys` per direction so the range is always
    ascending (grow-up: start_wm; grow-down: start_wm - N + 1), making the
    v->p mapping byte-identical to the `torch.arange` slow path.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    phys = (start_phys + offs).to(tl.int64)
    v = tl.load(v_pages_ptr + offs, mask=mask, other=0).to(tl.int64)

    # Masked stores skip out-of-range lanes, and `other=0` keeps us off the
    # v2p[0]/p2v[0] padding-sink slot.
    tl.store(out_phys_ptr + offs, phys, mask=mask)
    tl.store(v2p_ptr + v, phys, mask=mask)
    tl.store(p2v_ptr + phys, v, mask=mask)


ALLOC_BIND_BLOCK = 128


def alloc_bind_inplace(
    v_pages: torch.Tensor,
    v2p: torch.Tensor,
    p2v: torch.Tensor,
    start_phys: int,
) -> torch.Tensor:
    """Allocate N ascending physical pages from `start_phys` and bind to `v_pages`.

    Caller must advance `watermark_physical` by N and verify overflow BEFORE
    calling; this launcher does neither.
    """
    N = int(v_pages.numel())
    if N == 0:
        return torch.empty(0, dtype=torch.int64, device=v_pages.device)
    if not v_pages.is_cuda:
        # Pure-torch CPU reference for the CUDA-only kernel.
        phys_pages = torch.arange(
            start_phys, start_phys + N, dtype=torch.int64, device=v_pages.device
        )
        v = v_pages.to(torch.int64)
        v2p[v] = phys_pages
        p2v[phys_pages] = v
        return phys_pages
    phys_pages = torch.empty(N, dtype=torch.int64, device=v_pages.device)
    grid = (triton.cdiv(N, ALLOC_BIND_BLOCK),)
    alloc_bind_inplace_kernel[grid](
        v_pages,
        v2p,
        p2v,
        phys_pages,
        N,
        start_phys,
        BLOCK=ALLOC_BIND_BLOCK,
    )
    return phys_pages
