"""Top-k over the candidate slabs of the prefill DSA indexer (SM100).

One CTA per row selects the ``k`` largest of at most 12288 candidates (all
``>= thr[row]``) and writes their key indices, unordered, ``-1`` past
``min(k, count[row])``; which copies of the k-th value are taken is
unspecified, and a rank-k bin wider than 256 elements (rare) leaves the
part of the quota beyond them at ``-1``. fp32 and bf16 slabs run separate kernels with the same
interface:
``kernels/jit/csrc/dsa/litetopk_prefill/topk_candidates_{fp32,bf16}.cuh``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

TOPK_CANDIDATES_CAP = 12288
SUPPORTED_TOPK = (512, 1024, 2048)
_KERNELS = {
    torch.float32: ("fp32", "TopKCandidatesFP32Kernel"),
    torch.bfloat16: ("bf16", "TopKCandidatesBF16Kernel"),
}


@cache_once
def _jit_topk_candidates_module(topk: int, dtype: torch.dtype) -> Module:
    if topk not in SUPPORTED_TOPK:
        raise RuntimeError(f"topk must be one of {SUPPORTED_TOPK}, got {topk}")
    if dtype not in _KERNELS:
        raise RuntimeError(f"scores must be fp32 or bf16, got {dtype}")
    name, kernel = _KERNELS[dtype]
    args = make_cpp_args(topk)
    return load_jit(
        f"dsa_prefill_topk_candidates_{name}",
        *args,
        cuda_files=[f"dsa/litetopk_prefill/topk_candidates_{name}.cuh"],
        cuda_wrappers=[("run", f"{kernel}<{args}>::run")],
    )


def topk_candidates(
    scores: torch.Tensor,
    indices: torch.Tensor,
    count: torch.Tensor,
    thr: torch.Tensor,
    out: torch.Tensor,
    *,
    out_offsets: Optional[torch.Tensor] = None,
) -> None:
    """``scores`` fp32 or bf16 ``[rows, cap]`` and ``indices`` int32
    ``[rows, cap]`` (only the first ``min(count[row], cap)`` entries are read: an overflowing row is clipped), ``thr``
    fp32 ``[rows]`` (a per-row threshold <= every candidate, ``-inf`` keeps
    every key), ``out`` int32 ``[rows, k]``, ``k`` in {512, 1024, 2048};
    a bf16 slab's ``-inf`` row holds every key in key order and its
    ``indices`` are not read (the index of slot ``s`` is ``s``);
    ``out_offsets[row]`` is added to every emitted index. bf16 slabs need
    ``cap`` in [4096, 12288], a multiple of 8."""
    module = _jit_topk_candidates_module(out.shape[1], scores.dtype)
    module.run(scores, indices, count, thr, out_offsets, out)
