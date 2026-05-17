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
* ``ftka_raft_topk`` — experimental: substitutes ``torch.topk`` with
  ``ftka.cuda_ops.raft_topk`` (RAFT radix top-k from tsinghua-ideal/
  flash-topk-attention, pinned to commit
  ``d8803b29961c44d77a747636ad4282bd7a9094af``). Logical indices are then
  routed through the same ``_build_selected_physical`` Triton kernel as
  the torch backend. ``ftka`` is an optional dep; construction raises a
  clear ``RuntimeError`` when the package is not importable. **Not yet
  qualified for production** — used by the FTKA evaluation microbench
  to gather per-call timing and CUDA-graph capture status before any
  promotion decision.
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
    "ftka_raft_topk",
    "jit_fused_selector",
)

# tsinghua-ideal/flash-topk-attention commit hash this integration
# targets. Recorded so the FTKA microbench and any parity test can verify
# they're evaluating the expected build of the optional dep.
FTKA_TARGET_COMMIT = "d8803b29961c44d77a747636ad4282bd7a9094af"


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


# Known limitation, May 2026, flashinfer 0.6.11 + Triton (installed env):
#
# The `flashinfer.top_k_page_table_transform` Triton kernel **crashes
# with "Triton Error [CUDA]: an illegal memory access" inside SGLang's
# CUDA graph capture region** even after a pre-capture warmup sweep
# that triggers JIT compilation at every bs in the capture ladder
# (1, 2, 4, 8, 12, 16, 24, 32). The crash occurs at the first
# captured forward call, in `load_binary` — i.e. Triton attempting
# to load an already-compiled kernel handle that is bound to a
# different stream / context than the capture region uses. Bypassing
# `dsa_graph_safe=True` does not help (it's a kernel-internal flag
# for filtering top-k, not the cause of the load_binary failure).
#
# The selector microbench (no server, no capture) shows correct
# parity and ~1.10x to 1.30x speedup vs torch.topk at bs>=16 (see
# `benchmark/double_sparsity/repro_session/microbench_selector_backends.py`),
# so the backend is wired correctly but cannot yet be used under
# graph replay. Investigation deferred: a future session may need
# either a FlashInfer upstream fix, an opt-out from graph capture
# for the DS layers (large perf cost), or a swap to a different
# fused-topk + page-table-transform kernel.
#
# Until then, use ``selector_backend='torch'`` for production — see
# ``benchmark/double_sparsity/DESIGN.md`` for the gate-passing recipe
# (conc=16 / tb=2048 with retrieval-shaped calibration).


def _row_to_batch_buffer(bs: int, h_kv: int, device: torch.device) -> torch.Tensor:
    """Build the FlashInfer ``row_to_batch`` index for a fixed
    ``(bs, h_kv)``. Pure function — captured at selector construction
    so the per-step ``select()`` does no allocation."""
    if h_kv == 1:
        return torch.arange(bs, dtype=torch.int32, device=device)
    return torch.arange(bs * h_kv, dtype=torch.int32, device=device) // h_kv


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

    def __init__(self, *, max_bs: int, h_kv: int, device: torch.device) -> None:
        super().__init__()
        try:
            import flashinfer  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "FlashInfer is required for "
                "--double-sparsity-selector-backend=flashinfer_topk_page_table"
            ) from e
        # Pre-allocate aux tensors at construction time (outside any
        # CUDA-graph capture region). select() only refreshes contents
        # in-place; the captured device pointers stay stable.
        # Sized to the worst case `max_bs` so smaller-bs captured graphs
        # narrow into the same buffers.
        self._max_bs = max_bs
        self._h_kv = h_kv
        self._lengths_buf = torch.zeros(max_bs * h_kv, dtype=torch.int32, device=device)
        self._row_to_batch_buf = _row_to_batch_buffer(max_bs, h_kv, device)

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
        if bs > self._max_bs or h_kv != self._h_kv:
            raise ValueError(
                f"selector built for max_bs={self._max_bs}/h_kv={self._h_kv} "
                f"but called with bs={bs}/h_kv={h_kv}; constructor mismatch."
            )
        if req_to_token_indexed.dtype != torch.int32:
            raise ValueError(
                f"req_to_token_indexed must be int32, got {req_to_token_indexed.dtype}"
            )

        # Refresh the lengths buffer in-place. For h_kv==1 the per-row
        # length is just seq_lens-1; for h_kv>1 each batch repeats h_kv
        # times. We slice the preallocated buffer to the current bs;
        # no allocation either way.
        lengths = self._lengths_buf[: bs * h_kv]
        row_to_batch = self._row_to_batch_buf[: bs * h_kv]
        seq_minus_one = (seq_lens - 1).to(torch.int32)
        if h_kv == 1:
            lengths.copy_(seq_minus_one)
        else:
            # Inline the (i // h_kv) gather without allocating: write
            # h_kv copies of each row's length into adjacent slots.
            # `index_select` writes into our preallocated buffer.
            torch.index_select(
                seq_minus_one, 0, row_to_batch.to(torch.int64), out=lengths
            )

        # FlashInfer expects scores [num_rows, max_len].
        scores = att_out_approx.view(bs * h_kv, max_ctx)
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


class FtkaRaftTopKSelector(_BaseSelector):
    """Experimental: top-k via ``ftka.cuda_ops.raft_topk`` (RAFT radix
    top-k, BitsPerPass=8) + the existing ``_build_selected_physical``
    Triton kernel for the logical->physical gather and sink/recent
    append.

    Substitutes ONLY the top-k step. The score kernel and the
    sink/recent fill path are unchanged from the ``torch`` backend, so
    the produced ``selected_physical`` set must equal the torch backend
    modulo tie-breaking within the top-k slot. The parity test asserts
    set equality.

    Layout / arg mapping:
      * Score rows are viewed as ``[bs*h_kv, max_ctx]`` fp32, same as the
        FlashInfer backend.
      * Output ``values_buf`` / ``indices_buf`` are preallocated at
        construction so the per-step call does no device allocation
        (capture-friendly *if* the underlying ``raft_topk`` kernel is
        capture-safe; capture status is probed by the FTKA microbench
        and reported in its JSON output).
      * ``scratch_buf`` is an opaque int32 workspace passed to RAFT as
        ``buf``. We size it generously (``max_bs * h_kv * max_ctx`` int32
        cells), which covers the radix histograms + filtered-candidate
        scratch RAFT consumes internally; the buffer is opaque and we
        never read it back.

    Constraints:
      * ``max_top_k`` must be passed at construction time. The default
        ``make_selector`` signature uses ``max_top_k=None``; when this
        backend is requested without an explicit ``max_top_k``, the
        factory raises ``ValueError`` instead of constructing a
        crippled selector. Production code path: when the runtime
        builds the selector via ``_allocate_native_scratch``, it must
        pass ``max_top_k=token_budget``.
      * ``ftka`` must be importable. Failure is loud (``RuntimeError``).
    """

    name = "ftka_raft_topk"

    def __init__(
        self,
        *,
        max_bs: int,
        h_kv: int,
        max_top_k: int,
        max_ctx: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        try:
            from ftka.cuda_ops import raft_topk as _raft_topk  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "ftka is required for "
                "--double-sparsity-selector-backend=ftka_raft_topk. Install "
                "from the pinned commit, e.g.: "
                "`pip install git+https://github.com/tsinghua-ideal/"
                f"flash-topk-attention.git@{FTKA_TARGET_COMMIT}#egg=ftka` "
                f"(microbench targets commit {FTKA_TARGET_COMMIT})."
            ) from e
        self._raft_topk = _raft_topk
        self._max_bs = max_bs
        self._h_kv = h_kv
        self._max_top_k = max_top_k
        self._max_ctx = max_ctx
        num_rows = max_bs * h_kv
        self._values_buf = torch.empty(
            (num_rows, max_top_k), dtype=torch.float32, device=device
        )
        self._indices_buf = torch.empty(
            (num_rows, max_top_k), dtype=torch.int32, device=device
        )
        # Opaque RAFT workspace. Generous worst-case; bounded by max_ctx.
        self._scratch_buf = torch.empty(
            num_rows * max_ctx, dtype=torch.int32, device=device
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
        bs, h_kv, max_ctx = att_out_approx.shape
        if bs > self._max_bs or h_kv != self._h_kv or max_ctx > self._max_ctx:
            raise ValueError(
                f"FtkaRaftTopK selector built for max_bs={self._max_bs}/"
                f"h_kv={self._h_kv}/max_ctx={self._max_ctx} but called with "
                f"bs={bs}/h_kv={h_kv}/max_ctx={max_ctx}; constructor mismatch."
            )
        if top_k > self._max_top_k:
            raise ValueError(
                f"top_k={top_k} exceeds FtkaRaftTopK selector's preallocated "
                f"max_top_k={self._max_top_k}; rebuild the selector with a "
                f"larger max_top_k or lower the token_budget."
            )

        num_rows = bs * h_kv
        scores = att_out_approx.view(num_rows, max_ctx)
        values = self._values_buf[:num_rows, :top_k]
        indices = self._indices_buf[:num_rows, :top_k]
        self._raft_topk(scores, values, indices, self._scratch_buf, top_k)

        topk_logical = indices.view(bs, h_kv, top_k)
        _build_selected_physical(
            topk_logical=topk_logical,
            req_to_token_indexed=req_to_token_indexed,
            seq_lens=seq_lens,
            sink_tokens=sink_tokens,
            recent_tokens=recent_tokens,
            out=out,
        )


def make_selector(
    backend: str,
    *,
    max_bs: int = 1,
    h_kv: int = 1,
    device: torch.device = torch.device("cpu"),
    max_top_k: int | None = None,
    max_ctx: int | None = None,
) -> _BaseSelector:
    """Construct the requested selector.

    ``max_bs`` / ``h_kv`` / ``device`` are only consulted by backends
    that pre-allocate aux state at construction time (currently the
    FlashInfer and FTKA backends). ``max_top_k`` / ``max_ctx`` are
    additionally required for ``ftka_raft_topk`` since RAFT top-k
    needs its values/indices/workspace tensors sized up front to keep
    ``select()`` allocation-free. Defaults keep ``make_selector('torch')``
    backward-compatible.

    Fails loud:
      * unknown backend names -> ValueError
      * registered but not-yet-implemented backends
        (``sgl_fast_topk_transform``, ``jit_fused_selector``) ->
        NotImplementedError with the gating rationale
      * the optional dep for a registered backend is missing
        (FlashInfer, FTKA) -> RuntimeError from the backend ctor
      * ``ftka_raft_topk`` without explicit ``max_top_k`` / ``max_ctx``
        -> ValueError (no fallback default — the caller must size
        scratch deliberately)
    """
    if backend == "torch":
        return TorchTopKSelector()
    if backend == "flashinfer_topk_page_table":
        return FlashInferTopKPageTableSelector(max_bs=max_bs, h_kv=h_kv, device=device)
    if backend == "ftka_raft_topk":
        if max_top_k is None or max_ctx is None:
            raise ValueError(
                "selector_backend='ftka_raft_topk' requires explicit "
                "max_top_k and max_ctx kwargs to make_selector(); RAFT "
                "top-k needs its scratch tensors sized at construction "
                "to stay capture-safe."
            )
        return FtkaRaftTopKSelector(
            max_bs=max_bs,
            h_kv=h_kv,
            max_top_k=max_top_k,
            max_ctx=max_ctx,
            device=device,
        )
    if backend == "sgl_fast_topk_transform":
        # Registered but not wired: ``sgl_kernel.fast_topk_transform_fused``
        # in the installed build doesn't accept ``row_to_batch``, which
        # the per-h_kv broadcast needs. Either the upstream kernel
        # exposes it, or we land a score-row-duplication adaptor on the
        # algorithm side.
        raise NotImplementedError(
            "sgl_fast_topk_transform: installed sgl_kernel.fast_topk"
            "_transform_fused lacks the row_to_batch parameter required "
            "for the per-h_kv broadcast. Use 'flashinfer_topk_page_table' "
            "or 'torch' for now."
        )
    if backend == "jit_fused_selector":
        raise NotImplementedError(
            "jit_fused_selector: not implemented yet. Reserved for a "
            "future single-kernel score+topk+physical-translation+"
            "sink/recent pass."
        )
    raise ValueError(
        f"unsupported selector_backend: {backend!r}; "
        f"choose from {SUPPORTED_SELECTOR_BACKENDS}"
    )


__all__ = [
    "FLASHINFER_TOPK_MAX",
    "FTKA_TARGET_COMMIT",
    "FlashInferTopKPageTableSelector",
    "FtkaRaftTopKSelector",
    "SUPPORTED_SELECTOR_BACKENDS",
    "TorchTopKSelector",
    "make_selector",
]
