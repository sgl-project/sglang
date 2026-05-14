"""Top-K selector backends for the native DS sparse-decode pipeline.

After the score kernel writes ``att_out_approx[bs, h_kv, max_ctx]`` (with
sink / recent / oob positions masked to ``-inf``), a selector backend
chooses ``top_k`` indices per ``(bs, h_kv)`` and lays them out as
physical KV-cache positions in ``selected_physical[bs, h_kv, top_k]``.
The orchestrator then appends sink + recent physical ids into the
remaining ``[top_k:]`` slots.

Backends (selected via ``DoubleSparsityRuntimeConfig.selector_backend``):

* ``torch`` — ``torch.topk`` returns logical indices, then a Triton kernel
  fuses the logical->physical gather with the sink/recent append. Default.
* ``flashinfer_topk_page_table`` — ``flashinfer.top_k_page_table_transform``
  emits top-k physical indices directly (fused topk + page-table lookup).
  A smaller follow-on Triton kernel only writes the sink + recent slots.
* ``sgl_fast_topk_transform`` — ``sgl_kernel.fast_topk_transform_fused``
  is the SGLang-native counterpart; same shape contract as the flashinfer
  backend.
* ``jit_fused_selector`` — placeholder for a future single-kernel
  selector (score + topk + physical translation + sink/recent in one
  pass). Not yet implemented.

Parity test: ``select(... torch ...)`` and ``select(... flashinfer ...)``
must produce the same SET of selected physical ids (per ``(bs, h_kv)``)
on small shapes. Ordering inside the top-k slot is not required to
match.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.layers.attention.triton_ops.double_sparsity_native_decode import (
    _append_sink_recent_physical,
    _build_selected_physical,
)

logger = logging.getLogger(__name__)

SUPPORTED_SELECTOR_BACKENDS = (
    "torch",
    "flashinfer_topk_page_table",
    "sgl_fast_topk_transform",
    "jit_fused_selector",
)


class _BaseSelector:
    """Selector backend interface.

    Each backend writes ``out[bs, h_kv, :total_selected]`` in-place:
      ``[..., :top_k]``                     = top-k physical ids
      ``[..., top_k : top_k + sink]``        = sink physical ids
      ``[..., top_k + sink : total]``        = recent physical ids
    """

    name: str = "<base>"

    def __init__(self) -> None:
        pass

    def select(
        self,
        *,
        att_out_approx: torch.Tensor,  # [bs, h_kv, max_ctx] fp32
        req_to_token_indexed: torch.Tensor,  # [bs, max_ctx] int32
        seq_lens: torch.Tensor,  # [bs] int64
        top_k: int,
        sink_tokens: int,
        recent_tokens: int,
        out: torch.Tensor,  # [bs, h_kv, total] int32
    ) -> None:
        raise NotImplementedError


class TorchTopKSelector(_BaseSelector):
    """The default backend. ``torch.topk`` on the scored att_out, then
    the existing fused ``_build_selected_physical`` Triton kernel maps
    logical -> physical and appends sink + recent in one launch."""

    name = "torch"

    def select(
        self,
        *,
        att_out_approx: torch.Tensor,
        req_to_token_indexed: torch.Tensor,
        seq_lens: torch.Tensor,
        top_k: int,
        sink_tokens: int,
        recent_tokens: int,
        out: torch.Tensor,
    ) -> None:
        topk_logical = torch.topk(att_out_approx, top_k, dim=-1, sorted=False).indices
        if topk_logical.dtype != torch.int32:
            topk_logical = topk_logical.to(torch.int32)
        _build_selected_physical(
            topk_logical=topk_logical,
            req_to_token_indexed=req_to_token_indexed,
            seq_lens=seq_lens,
            sink_tokens=sink_tokens,
            recent_tokens=recent_tokens,
            out=out,
        )


FLASHINFER_TOPK_MAX = 2048
"""Upper bound on ``top_k`` accepted by
``flashinfer.top_k_page_table_transform`` in flashinfer >= 0.6.11.
Requests above this raise CUDA ``operation not supported`` from inside
the kernel. Empirically the bound holds independent of ``max_ctx``."""


class FlashInferTopKPageTableSelector(_BaseSelector):
    """Top-k via ``flashinfer.top_k_page_table_transform`` (fused topk +
    page-table lookup) + a small sink/recent Triton kernel.

    Layout / arg mapping:
      * ``input``           -> ``att_out_approx`` flattened to ``[bs*h_kv, max_ctx]``.
      * ``src_page_table``  -> ``req_to_token_indexed``, shape ``[bs, max_ctx]`` int32.
      * ``lengths``         -> per-row history bound (``seq_lens - 1`` since
                              the score kernel already masked the current
                              decode position; repeat across h_kv heads).
      * ``row_to_batch``    -> identity (``[0, 1, 2, ...]``) when ``h_kv == 1``;
                              ``i // h_kv`` mapping otherwise.

    Output is ``[bs*h_kv, top_k]`` int32 of physical KV positions, which
    we view as ``[bs, h_kv, top_k]`` and copy into ``out[..., :top_k]``.
    Sink + recent slots are filled by ``_append_sink_recent_physical``.

    **Top-k ceiling**: ``top_k <= FLASHINFER_TOPK_MAX`` (2048). The kernel
    fails CUDA with ``operation not supported`` for larger ``k``;
    ``DoubleSparsityRuntimeConfig.validate()`` rejects mismatched
    configs at startup so the server never enters a captured graph
    that can't actually fire.
    """

    name = "flashinfer_topk_page_table"

    def __init__(self) -> None:
        super().__init__()
        try:
            import flashinfer  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "FlashInfer is required for "
                "--double-sparsity-selector-backend=flashinfer_topk_page_table"
            ) from e
        # Cached aux tensors (allocated lazily on first call to match
        # the per-shape contract). Stored on the algorithm side ideally,
        # but kept here in-selector for now — small allocations bounded
        # by max_running_requests * h_kv.
        self._lengths_cache: Optional[torch.Tensor] = None
        self._row_to_batch_cache: Optional[torch.Tensor] = None
        self._cached_bs: int = -1
        self._cached_h_kv: int = -1

    def _ensure_aux(self, bs: int, h_kv: int, seq_lens: torch.Tensor) -> tuple:
        """Materialize / refresh ``lengths`` and ``row_to_batch``.

        ``lengths`` changes per step (it's ``seq_lens - 1`` per row); allocate
        a stable shape-``[bs*h_kv]`` int32 buffer so the captured-graph
        pointer is stable and only the data refreshes.
        """
        n_rows = bs * h_kv
        if (
            self._lengths_cache is None
            or self._cached_bs != bs
            or self._cached_h_kv != h_kv
            or self._lengths_cache.device != seq_lens.device
        ):
            self._lengths_cache = torch.zeros(
                n_rows, dtype=torch.int32, device=seq_lens.device
            )
            if h_kv == 1:
                self._row_to_batch_cache = torch.arange(
                    bs, dtype=torch.int32, device=seq_lens.device
                )
            else:
                # row i -> batch (i // h_kv)
                self._row_to_batch_cache = (
                    torch.arange(n_rows, dtype=torch.int32, device=seq_lens.device)
                    // h_kv
                )
            self._cached_bs = bs
            self._cached_h_kv = h_kv

        # Refresh contents in-place (capture-safe).
        seq_minus_one = (seq_lens - 1).to(torch.int32)
        if h_kv == 1:
            self._lengths_cache.copy_(seq_minus_one)
        else:
            # Repeat each row's length across the h_kv heads:
            # lengths_cache[i] = seq_lens[i // h_kv] - 1
            self._lengths_cache.copy_(seq_minus_one.repeat_interleave(h_kv))
        return self._lengths_cache, self._row_to_batch_cache

    def select(
        self,
        *,
        att_out_approx: torch.Tensor,
        req_to_token_indexed: torch.Tensor,
        seq_lens: torch.Tensor,
        top_k: int,
        sink_tokens: int,
        recent_tokens: int,
        out: torch.Tensor,
    ) -> None:
        import flashinfer

        if top_k > FLASHINFER_TOPK_MAX:
            raise ValueError(
                f"top_k={top_k} exceeds FlashInfer's "
                f"top_k_page_table_transform ceiling ({FLASHINFER_TOPK_MAX}). "
                f"Lower --double-sparsity-token-budget or switch to "
                f"--double-sparsity-selector-backend=torch."
            )
        bs, h_kv, max_ctx = att_out_approx.shape
        # FlashInfer expects scores [num_rows, max_len]. h_kv collapses
        # into the row dim; row_to_batch then maps each row back to the
        # batch row in src_page_table.
        scores = att_out_approx.view(bs * h_kv, max_ctx)
        if req_to_token_indexed.dtype != torch.int32:
            raise ValueError(
                f"req_to_token_indexed must be int32, got {req_to_token_indexed.dtype}"
            )
        lengths, row_to_batch = self._ensure_aux(bs, h_kv, seq_lens)

        topk_phys_flat = flashinfer.top_k_page_table_transform(
            scores,
            req_to_token_indexed,
            lengths,
            top_k,
            row_to_batch=row_to_batch,
            dsa_graph_safe=True,
        )
        # Output is [bs*h_kv, top_k] int32 — copy into out[..., :top_k].
        out[..., :top_k].copy_(topk_phys_flat.view(bs, h_kv, top_k))

        # Sink + recent are appended into out[..., top_k:].
        _append_sink_recent_physical(
            req_to_token_indexed=req_to_token_indexed,
            seq_lens=seq_lens,
            out=out,
            top_k=top_k,
            sink_tokens=sink_tokens,
            recent_tokens=recent_tokens,
        )


class SglFastTopKTransformSelector(_BaseSelector):
    """SGLang-native ``sgl_kernel.fast_topk_transform_fused`` selector.

    The current ``sgl_kernel`` API (signature:
    ``(score, lengths, page_table_size_1, cu_seqlens_q, topk,
    row_starts=None)``) does not match the FlashInfer-style
    ``row_to_batch`` contract we use for the per-head-h_kv broadcast,
    and ``page_table_size_1`` is interpreted by the kernel as a
    ``(B, topk)`` output buffer rather than a page table. Wiring this
    correctly requires either a per-row score expansion (duplicate
    the same row across the h_kv heads inline, so ``B = bs * h_kv``
    and the page table is per-row) or a kernel-side ``row_to_batch``
    parameter that the installed sgl_kernel build doesn't expose yet.

    PLAN.md gates this backend behind a measurement step ("test
    tb=2048 first"); we wire it through the registry so the
    benchmark harness can request it once the kernel signature aligns
    with ours (or we land an adaptor that duplicates the score rows
    across heads).
    """

    name = "sgl_fast_topk_transform"

    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError(
            "sgl_fast_topk_transform selector is not yet wired: the installed "
            "sgl_kernel.fast_topk_transform_fused signature does not accept a "
            "row_to_batch parameter, which we need to broadcast the per-bs "
            "page table across h_kv heads. Use 'flashinfer_topk_page_table' "
            "until either the kernel exposes row_to_batch or we add a "
            "score-row-duplication adaptor on the algorithm side."
        )

    def select(
        self,
        *,
        att_out_approx: torch.Tensor,
        req_to_token_indexed: torch.Tensor,
        seq_lens: torch.Tensor,
        top_k: int,
        sink_tokens: int,
        recent_tokens: int,
        out: torch.Tensor,
    ) -> None:
        raise NotImplementedError


def make_selector(backend: str) -> _BaseSelector:
    """Construct the requested selector. Fails loud if the optional
    backend is unavailable (FlashInfer / sgl_kernel not installed)."""
    if backend == "torch":
        return TorchTopKSelector()
    if backend == "flashinfer_topk_page_table":
        return FlashInferTopKPageTableSelector()
    if backend == "sgl_fast_topk_transform":
        return SglFastTopKTransformSelector()
    if backend == "jit_fused_selector":
        raise NotImplementedError(
            "jit_fused_selector is gated behind the FlashInfer / SGL "
            "measurement step in PLAN.md — not implemented yet."
        )
    raise ValueError(
        f"unsupported selector_backend: {backend!r}; "
        f"choose from {SUPPORTED_SELECTOR_BACKENDS}"
    )


__all__ = [
    "SUPPORTED_SELECTOR_BACKENDS",
    "FlashInferTopKPageTableSelector",
    "SglFastTopKTransformSelector",
    "TorchTopKSelector",
    "make_selector",
]
