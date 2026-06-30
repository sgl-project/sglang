# Copyright 2025-2026 SGLang Team
"""Backend selection for the FP8 paged MQA logits used by the DSA indexer.

This is shared so the SM120 (Blackwell desktop / RTX PRO 6000) fallbacks live in
one place. On SM120 the ``deep_gemm`` paged-MQA-logits kernel is unavailable
(no tcgen05/TMEM), so the indexer must fall back to a TileLang kernel (preferred,
fast) or a pure-torch reference (slower, correctness-first).

All returned callables share the same signature::

    fn(q, kv, weights, seq_lens, page_table, deep_gemm_metadata, max_seq_len,
       clean_logits=...)

The TileLang/torch fns ignore ``deep_gemm_metadata``. Callers use the returned
backend name to decide:
  * ``seq_lens`` layout: ``[B]`` for "tilelang", ``[B, 1]`` for "torch"/"deep_gemm".
  * whether the deep_gemm schedule metadata / native next_n layout applies
    (only for "deep_gemm").
"""

from typing import Callable, Tuple

from sglang.srt.environ import envs
from sglang.srt.utils import is_sm120_supported


def select_fp8_paged_mqa_logits_fn() -> Tuple[Callable, str]:
    """Return ``(fn, backend_name)`` for the FP8 paged MQA logits.

    ``backend_name`` is one of ``{"tilelang", "torch", "deep_gemm"}``.

    Selection order mirrors the DeepSeek-V4 indexer: an explicit TileLang opt-in
    wins, then the torch fallback (SM120-specialized when on SM120), otherwise
    the default ``deep_gemm`` kernel.
    """
    if envs.SGLANG_OPT_USE_TILELANG_INDEXER.get():
        from sglang.srt.layers.attention.dsa.tilelang_kernel import (
            tilelang_fp8_paged_mqa_logits,
        )

        return tilelang_fp8_paged_mqa_logits, "tilelang"

    if envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get():
        from sglang.srt.layers.attention.dsv4.indexer import (
            fp8_paged_mqa_logits_torch,
            fp8_paged_mqa_logits_torch_sm120,
        )

        if is_sm120_supported():
            return fp8_paged_mqa_logits_torch_sm120, "torch"
        return fp8_paged_mqa_logits_torch, "torch"

    import deep_gemm

    return deep_gemm.fp8_paged_mqa_logits, "deep_gemm"
