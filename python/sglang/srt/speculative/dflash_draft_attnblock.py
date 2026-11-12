# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""DFlash MLA draft attention block (SGLANG_OPT_DFLASH_DRAFT_ATTNBLOCK, default off).

One clustered launch replaces the served P1 split-KV pair (kernels/ops/attention/dflash_draft_splitkv.py: partial + combine) and the
absorbed w_vc projection (WVC split-K or torch.bmm) of a DFlash MLA draft layer. DraftMlaQuery (q, k, v and the FP8 KV-row store) is
unchanged and runs before it. Kernel: kernels/jit/csrc/speculative/dflash_draft_attnblock.cuh.

Numerics are not bitwise to the served pair: QK and PV take F16 operands (served: q and p cast to FP8 for the rail dots), the
per-worker unnormalized partials are rounded to BF16 and projected through w_vc (FP32 accumulate) before the FP32 combine, and the output is
rounded to BF16 once.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

ROWS = 8  # draft block rows per request (block size 8): the only extend length the kernel covers
MAX_BS = 4  # requests per call in the first build
CLUSTER = 8
# cluster size per draft batch (requests), 6 heads per rank, rail <= 17 tiles + the own block, one CTA per SM: the fastest
# size in the c=1 / c=8-form launch chains whose direct-call time was also faster:
# bs 1 -> 19 (one slot per worker), 2 -> 10, 3 -> 7, 4 -> 6 (7 was faster in the chain but slower back-to-back).
CLUSTER_BY_BS = {1: 19, 2: 10, 3: 7, 4: 6}


def attnblock_enabled() -> bool:
    return os.environ.get("SGLANG_OPT_DFLASH_DRAFT_ATTNBLOCK", "0") not in ("", "0", "false", "False")


@cache_once
def _module(cluster: int) -> "Module":
    args = make_cpp_args(cluster)
    return load_jit(
        "dflash_draft_attnblock",
        *args,
        cuda_files=["speculative/dflash_draft_attnblock.cuh"],
        cuda_wrappers=[("run", f"::sglang::dflash_attnblock::AttnBlockKernel<{args}>::run")],
    )


def attnblock_covers(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, k_buffer: torch.Tensor, qo_indptr: torch.Tensor,
                     w_vc: Optional[torch.Tensor], max_len_extend: int, k_scale: float, v_scale: float) -> str:
    """'' when the fused block serves this call, else the reason it declines (the caller then runs the served P1 + w_vc)."""
    bs = qo_indptr.shape[0] - 1
    checks = (
        (q.is_cuda and q.dtype == torch.bfloat16 and q.dim() == 3 and q.shape[2] == 576 and q.stride(2) == 1 and q.stride(1) == 576, "q form"),
        (int(max_len_extend) == ROWS and q.shape[0] == bs * ROWS, "extend length"),
        (1 <= bs <= MAX_BS, f"bs {bs}"),
        (k.dtype == torch.bfloat16 and k.shape[-1] == 576 and k.stride(-1) == 1 and v.dtype == torch.bfloat16 and v.stride(-1) == 1, "k / v form"),
        (k_buffer.dtype == torch.float8_e4m3fn and k_buffer.shape[-1] == 576 and k_buffer.stride(-1) == 1, "pool form"),
        (float(k_scale) == 1.0 and float(v_scale) == 1.0, "kv scales"),
        (w_vc is not None and w_vc.dtype == torch.bfloat16 and tuple(w_vc.shape) == (q.shape[1], 512, 128)
         and w_vc.transpose(1, 2).is_contiguous(), "w_vc form"),
    )
    return "; ".join(why for ok, why in checks if not ok)


def dflash_draft_attnblock_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, k_buffer: torch.Tensor, qo_indptr: torch.Tensor,
                               kv_indptr: torch.Tensor, kv_indices: torch.Tensor, w_vc: torch.Tensor, out: torch.Tensor, sm_scale: float,
                               k_scale: float = 1.0, sliding_window_size: int = -1, cluster: Optional[int] = None,
                               max_kv: int = 1087) -> torch.Tensor:
    """out [rows, heads * 128] BF16 = (causal extend attention of q over the FP8 rail + the block's own k / v rows) @ w_vc per head.

    q [rows, heads, 576] BF16; k [rows, (1,) 576] / v [rows, (1,) 512] BF16 (the block's own rows, DraftMlaQuery's outputs); k_buffer the
    layer's FP8 [slots, 1, 576] pool; w_vc the absorbed [heads, 512, 128] BF16 view whose transpose is contiguous; launched with PDL."""
    bs = qo_indptr.shape[0] - 1
    if cluster is None:
        cluster = CLUSTER_BY_BS[bs]
    k2 = k.reshape(k.shape[0], -1)
    v2 = v.reshape(v.shape[0], -1)
    pool = k_buffer.view(torch.uint8).reshape(k_buffer.shape[0], -1)
    wvc_t = w_vc.transpose(1, 2)  # [heads, 128, 512] contiguous
    _module(int(cluster)).run(
        q, k2, v2, pool, qo_indptr, kv_indptr, kv_indices, wvc_t, out, q.stride(0), k2.stride(0), v2.stride(0), pool.stride(0), out.stride(0),
        float(sm_scale) * float(k_scale), float(sm_scale), int(sliding_window_size), int(bs), int(max_kv),
    )
    return out
