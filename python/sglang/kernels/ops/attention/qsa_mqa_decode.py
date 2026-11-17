"""QSA indexer decode scoring as a JIT CUDA kernel.

``qsa_mqa_decode_logits`` writes the same logits as the TileLang paged decode
kernels of ``srt/layers/attention/qsa/mqa.py``, bit for bit: every logit runs the
instructions TileLang emits for them, in the same order. Only the data movement
differs: the query rows that share a page-table row are staged once per CTA, and
each CTA strides over many 64-key groups of its request with double-buffered
cp.async gathers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_qsa_mqa_decode_module(rows_per_cta: int) -> Module:
    args = make_cpp_args(rows_per_cta, is_arch_support_pdl())
    return load_jit(
        "qsa_mqa_decode",
        *args,
        cuda_files=["attention/qsa_mqa_decode.cuh"],
        cuda_wrappers=[("run", f"QsaMqaDecode<{args}>::run")],
        # TileLang builds the reference kernels with fast math; the final
        # division must compile the same way.
        extra_cuda_cflags=["--use_fast_math"],
    )


def qsa_mqa_decode_logits(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
    logits: torch.Tensor,
    *,
    scale: float,
    min_scored_len: int,
    rows_per_cta: int,
    ctas_per_request: int,
) -> None:
    """Write the decode logits of ``q`` into ``logits``.

    Shapes: ``q`` [rows, 8, 128] bf16 (query heads zero-padded to 8), ``k_cache``
    [pages, 16, 1, 128] bf16, ``page_table`` [rows, max_pages] int32,
    ``context_lens`` [rows] int32, ``logits`` [rows, max_model_len] fp32.

    Each group of ``rows_per_cta`` consecutive rows reads the page-table row and
    length of its last row (verify rows of one request share the page prefix and
    have increasing lengths). Rows whose length is at most ``min_scored_len`` and
    positions at or past a row's length are not written. ``ctas_per_request``
    CTAs stride over each group's 64-key blocks; it never changes a logit.
    """
    _jit_qsa_mqa_decode_module(rows_per_cta).run(
        q,
        k_cache,
        page_table,
        context_lens,
        logits,
        float(scale),
        int(min_scored_len),
        int(ctas_per_request),
    )
