"""Runtime-topk variant of sgl_kernel.fast_topk_v2.

Loads ``csrc/fast_topk_runtime.cu`` via ``torch.utils.cpp_extension.load``
on first import; subsequent imports hit the cached ``.so`` in
``~/.cache/torch_extensions/``.

Constraints:
  * ``topk`` in ``(0, 2048]`` (SMEM-bound; upstream's tuned range).
    Production hisa block_topk is ``8192 // k_block_size`` (= 512/256/128/64
    for K=16/32/64/128), all within bound.
  * ``score.size(1) >= topk`` — caller clamps topk to row width.
  * **Caller is responsible for masking invalid positions to ``-inf``**
    in ``score``. The kernel selects from the full row; -inf entries lose
    the radix selection naturally. There is no per-row ``lengths`` arg.

Usage::

    from sglang.srt.layers.attention.nsa.hisa.fast_topk_runtime import (
        fast_topk_runtime,
    )

    indices = fast_topk_runtime(score_2d_f32, topk)
    # Returns [B, topk] int32 (unsorted, no -1 sentinels).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_HERE = Path(__file__).resolve().parent
_CSRC = _HERE / "csrc"


def _load_module():
    """JIT-compile the .cu via torch.utils.cpp_extension.load.

    First call: invokes nvcc, ~20-60s. The resulting .so is cached under
    ``$TORCH_EXTENSIONS_DIR`` (default ``~/.cache/torch_extensions``)
    keyed by source hash, so subsequent imports are near-instant.

    Returns the loaded module (side-effect: registers
    ``torch.ops.hisa_fast_topk.fast_topk_runtime``).
    """
    # is_python_module=False → load as a torch-op library (registers via
    # TORCH_LIBRARY/_IMPL static initializers when the .so is dlopen'd).
    # Returns None; the op surfaces under torch.ops.hisa_fast_topk.* .
    load(
        name="hisa_fast_topk_runtime",
        sources=[str(_CSRC / "fast_topk_runtime.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        is_python_module=False,
        verbose=bool(int(os.environ.get("HISA_FAST_TOPK_VERBOSE", "0"))),
    )
    return None


# Side-effect: load + register the torch op at import time.
_module = _load_module()


MAX_TOPK = 2048


def fast_topk_runtime(score: torch.Tensor, topk: int) -> torch.Tensor:
    """Top-k indices via 8-bit radix-select (runtime-topk version).

    Selects top-k from each full row of ``score``. Caller masks invalid
    positions to -inf so they lose the radix selection naturally — no
    per-row lengths or row_starts.

    Args:
        score: ``[B, L]`` f32, last-dim contiguous (``stride(1) == 1``).
            ``L >= topk`` required (caller clamps).
        topk: int in ``(0, topk_max]``, ``topk_max=2048``.

    Returns:
        ``[B, topk]`` i32 — top-``topk`` indices (unsorted). No ``-1``
        sentinels (since ``L >= topk``).
    """
    if not 0 < topk <= MAX_TOPK:
        raise ValueError(
            f"fast_topk_runtime: topk={topk} out of supported range (0, {MAX_TOPK}]"
        )
    if topk > score.size(1):
        raise ValueError(
            f"fast_topk_runtime: topk={topk} > score.size(1)={score.size(1)}; "
            "caller must clamp topk to row width."
        )
    indices = score.new_empty((score.size(0), topk), dtype=torch.int32)
    torch.ops.hisa_fast_topk.fast_topk_runtime(score, indices)
    return indices
