"""PPU MiniMax-M3 SAIL MSA backend hooks.

Uses ``@plugin_hook`` to inject the PPU SAIL MSA sparse-attention path
(fmha_sm100 OnlyScore indexer + sparse top-k selection + optional direct
paged-NHD attend, implemented in ``minimax_sparse_ops.msa_ppu``) into the
MiniMax-M3 sparse attention stack without modifying the community source
code in ``minimax_sparse_backend.py`` / ``minimax_sparse.py``.

Registered hooks:

1. ``MiniMaxSparseAttnBackend.__init__`` (AROUND) — Rejects HND KV caches,
   evaluates the SAIL MSA routing gate, installs the cross-process fmha_sm100
   JIT lock, allocates the shared decode/prefill state, runs the pre-capture
   warmup, enforces the fp8 index-cache consistency guard, and registers the
   backend (keyed by its ``req_to_token`` tensor) for the module-level
   prefill hook. On any gate or init failure the backend silently falls back
   to the Triton sparse path.

2. ``MiniMaxSparseAttnBackend.init_forward_metadata_out_graph`` (AROUND) —
   Resets the per-forward SAIL MSA state: the next decode rebuilds its plans
   (CUDA-graph-safe refresh), the eager prefill chunk buffers are freed, and
   the live ``ForwardBatch`` is stashed for hook 3. The metadata hooks run
   before every forward (AttentionBackend contract), so the stash is always
   current.

3. ``minimax_sparse_prefill`` (AROUND, module-level kernel) — Intercepts the
   core sparse prefill kernel only; the community ``forward_extend`` keeps
   handling KV/index-cache writes, DP-padding trim and re-pad, and
   ``cu_seqlens``/``prefix_lens`` construction. When the SAIL MSA gate holds
   and the batch is eager (not under CUDA-graph capture), the Triton kernel
   is replaced with ``msa_ppu_forward_extend`` (chunked indexer +
   whole-batch attend). Every other case — gate off, capture, or a missing
   forward-batch context — falls back to the original Triton kernel (warned
   once per process under capture) after rejecting an fp8 index cache,
   which only the SAIL MSA indexer can read.

4. ``MiniMaxSparseAttnBackend.forward_decode`` (AROUND) — Routes decodes
   through the SAIL MSA indexer + selection + attend (graph-safe) when the
   gate holds; falls back to the original Triton path otherwise, again
   after rejecting an fp8 index cache.
"""

import logging

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)

_ppu_prefill_capture_warned = False

_BACKEND = "sglang.srt.layers.attention.minimax_sparse_backend.MiniMaxSparseAttnBackend"
_PREFILL_KERNEL = (
    "sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse."
    "minimax_sparse_prefill"
)

_FP8_KV_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

# Backend registry: ``minimax_sparse_prefill`` is a module-level function
# without a ``self``, so the prefill hook locates the owning backend through
# its ``req_to_token`` tensor. Each backend owns one persistent tensor, and
# the registry keeps a strong reference to it so ``id()`` cannot be recycled
# while the backend lives.
_BACKEND_REGISTRY: dict[int, tuple[torch.Tensor, object]] = {}


def _find_backend(req_to_token: torch.Tensor):
    """Return the sparse backend owning ``req_to_token``, or None."""
    entry = _BACKEND_REGISTRY.get(id(req_to_token))
    if entry is not None and entry[0] is req_to_token:
        return entry[1]
    return None


def _reject_fp8_index_cache(idx_k_cache, path: str) -> None:
    """Reject an fp8 index-K cache before a Triton sparse kernel runs.

    An fp8 index-K cache (SGLANG_SAIL_MINIMAX_M3_MSA_INDEXER_FP8=1) is only
    readable by the SAIL MSA indexer kernel; the Triton kernels assume a
    16-bit cache and would silently misread the fp8 bytes.
    """
    if idx_k_cache is not None and idx_k_cache.dtype in _FP8_KV_DTYPES:
        raise NotImplementedError(
            f"fp8 index-K cache reached the Triton {path} path; only the SAIL "
            "MSA indexer (SGLANG_SAIL_MINIMAX_M3_MSA=1) supports fp8 index K. "
            "Unset SGLANG_SAIL_MINIMAX_M3_MSA_INDEXER_FP8 or fix the MSA gate."
        )


def _init_ppu_msa(backend, runner) -> None:
    """Attach the SAIL MSA (PPU fmha_sm100) state to a sparse backend."""
    if backend.kv_pool.main_pool.use_hnd:
        raise NotImplementedError(
            "MiniMax-M3 sparse attention does not support HND KV cache. "
            "Disable SGLANG_USE_HND_KVCACHE (or any explicit HND layout) "
            "when using this model."
        )

    # Registered for the module-level minimax_sparse_prefill hook (see
    # _find_backend). Always registered — even when the gate fails below — so
    # the Triton fallback path can still enforce the fp8 index-cache guard.
    _BACKEND_REGISTRY[id(backend.req_to_token)] = (
        backend.req_to_token,
        backend,
    )

    # Defaults; also the fallback state when the gate or init fails below.
    backend.use_msa_ppu = False
    backend._ppu_msa_attend = False
    backend._ppu_msa_dec = None
    backend._ppu_msa_state = None
    backend._ppu_msa_shared = None
    backend._ppu_msa_current_batch = None
    backend.num_kv_heads = backend.kv_pool.main_pool.head_num

    # PPU (SM80) MSA path: indexer + selection (+ optional direct NHD attend)
    # via the ppu_dev fmha_sm100 kernels. Independent of the B200 gate in the
    # backend (msa_available() requires SM100); the Triton path remains the
    # fallback.
    try:
        from sglang.srt.layers.attention.minimax_sparse_ops.msa_ppu import (
            compute_msa_ppu_gate,
            init_msa_ppu_state,
            install_msa_jit_cross_process_lock,
            msa_ppu_warmup,
        )

        ok, use_attend, ppu_reasons = compute_msa_ppu_gate(backend, runner)
        backend.use_msa_ppu = ok
        backend._ppu_msa_attend = ok and use_attend
        if ok:
            install_msa_jit_cross_process_lock()
            init_msa_ppu_state(backend, runner)
            msa_ppu_warmup(backend)
        else:
            logger.warning(
                "[MiniMaxSparse][SAIL MSA] disabled (%s); falling back to the "
                "Triton sparse path. Set SGLANG_SAIL_MINIMAX_M3_MSA=0 to "
                "silence.",
                "; ".join(ppu_reasons),
            )
    # This optional backend may fail through Python imports, JIT compilation,
    # or device runtime errors. Keep the catch broad so every such failure
    # preserves the documented Triton fallback; the traceback is retained.
    except Exception:
        backend.use_msa_ppu = False
        backend._ppu_msa_attend = False
        logger.exception(
            "[MiniMaxSparse][SAIL MSA] init failed; falling back to the Triton "
            "sparse path. Set SGLANG_SAIL_MINIMAX_M3_MSA=0 to disable the probe."
        )

    # An fp8 index-K cache (SGLANG_SAIL_MINIMAX_M3_MSA_INDEXER_FP8=1) is only
    # readable by the SAIL MSA indexer kernel; the Triton sparse path would
    # silently misread the fp8 bytes. Fail loudly instead of falling back.
    _index_pool = backend.kv_pool.index_k_pool or backend.kv_pool.index_kv_pool
    if (
        _index_pool is not None
        and _index_pool.dtype in _FP8_KV_DTYPES
        and not backend.use_msa_ppu
    ):
        raise RuntimeError(
            "SGLANG_SAIL_MINIMAX_M3_MSA_INDEXER_FP8=1 allocated an fp8 index-K "
            "cache, but the SAIL MSA path is disabled (see the reasons/traceback "
            "above). The Triton sparse path cannot read an fp8 index cache; fix "
            "the gate conditions or unset "
            "SGLANG_SAIL_MINIMAX_M3_MSA_INDEXER_FP8."
        )

    logger.info(
        "[MiniMaxSparse][SAIL MSA] ppu_msa=%s",
        (
            "msa(attend=direct_nhd)"
            if backend.use_msa_ppu and backend._ppu_msa_attend
            else "msa(indexer-only)"
            if backend.use_msa_ppu
            else "off"
        ),
    )


@plugin_hook(f"{_BACKEND}.__init__", type=HookType.AROUND)
def _ppu_msa_backend_init(original_fn, self, runner):
    """Run the community init, then attach the SAIL MSA state on PPU."""
    original_fn(self, runner)
    _init_ppu_msa(self, runner)


@plugin_hook(f"{_BACKEND}.init_forward_metadata_out_graph", type=HookType.AROUND)
def _ppu_msa_init_forward_metadata_out_graph(
    original_fn, self, forward_batch, in_capture=False
):
    """Reset the per-forward SAIL MSA state before metadata init."""
    dec_state = getattr(self, "_ppu_msa_dec", None)
    if dec_state is not None:
        # SAIL MSA: force the next decode forward to rebuild its plans
        # (eager steps must refresh lengths; capture reruns the recorded
        # build kernels on replay). The prefill state is per-forward;
        # dropping it also frees the chunk buffers.
        dec_state.need_build = True
    self._ppu_msa_state = None
    # Stash the live ForwardBatch for the minimax_sparse_prefill hook: the
    # SAIL MSA prefill planner needs req_pool_indices plus a stable
    # per-forward identity for plan reuse across the ~57 sparse layers.
    # The metadata hooks run before every forward (AttentionBackend
    # contract), so the stash is always current.
    self._ppu_msa_current_batch = forward_batch
    return original_fn(self, forward_batch, in_capture)


@plugin_hook(_PREFILL_KERNEL, type=HookType.AROUND)
def _ppu_msa_sparse_prefill(
    original_fn,
    q,
    k_cache,
    v_cache,
    sink,
    idx_q,
    idx_k_cache,
    idx_v_cache,
    idx_sink,
    req_to_token,
    slot_ids,
    cu_seqlens,
    seq_lens,
    prefix_lens,
    max_seqlen_q,
    max_seqlen_k,
    block_size_q,
    block_size_k,
    topk,
    init_blocks,
    local_blocks,
    sm_scale=None,
    idx_sm_scale=None,
    score_type="max",
    disable_index_value=False,
    use_msa=False,
    cu_seqblocks_q=None,
    max_seqblock_q=None,
    all_seqblock_q=None,
    seqlens_cpu=None,
    q_scale=None,
    k_scale=None,
    v_scale=None,
    idx_q_scale=None,
    idx_k_scale=None,
    idx_v_scale=None,
):
    """SAIL MSA prefill: chunked indexer + whole-batch attend, else Triton.

    Hooking the module-level kernel (instead of ``forward_extend``) lets the
    community path keep handling KV/index-cache writes, DP-padding trim and
    re-pad, and ``cu_seqlens``/``prefix_lens`` construction: by the time this
    kernel is called, ``q``/``idx_q`` are already trimmed, and the caller
    re-pads the returned output. When the SAIL MSA gate holds and the batch
    is eager, the Triton kernel is swapped for ``msa_ppu_forward_extend``;
    every other case falls back to the original Triton kernel.
    """

    def _triton_prefill():
        _reject_fp8_index_cache(idx_k_cache, "prefill")
        return original_fn(
            q,
            k_cache,
            v_cache,
            sink,
            idx_q,
            idx_k_cache,
            idx_v_cache,
            idx_sink,
            req_to_token,
            slot_ids,
            cu_seqlens,
            seq_lens,
            prefix_lens,
            max_seqlen_q,
            max_seqlen_k,
            block_size_q,
            block_size_k,
            topk,
            init_blocks,
            local_blocks,
            sm_scale=sm_scale,
            idx_sm_scale=idx_sm_scale,
            score_type=score_type,
            disable_index_value=disable_index_value,
            use_msa=use_msa,
            cu_seqblocks_q=cu_seqblocks_q,
            max_seqblock_q=max_seqblock_q,
            all_seqblock_q=all_seqblock_q,
            seqlens_cpu=seqlens_cpu,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            idx_q_scale=idx_q_scale,
            idx_k_scale=idx_k_scale,
            idx_v_scale=idx_v_scale,
        )

    backend = _find_backend(req_to_token)
    if backend is None or not getattr(backend, "use_msa_ppu", False):
        return _triton_prefill()

    # Under capture (defensive; prefill graphs should be gated off) fall
    # back to the Triton path, which is capture-safe, and warn once.
    if torch.cuda.is_current_stream_capturing():
        # Intentionally once per worker process: avoiding cross-process IPC
        # on this exceptional path keeps capture fallback independent by
        # rank.
        global _ppu_prefill_capture_warned
        if not _ppu_prefill_capture_warned:
            logger.warning(
                "[MiniMaxSparse][SAIL MSA] sparse prefill under CUDA-graph "
                "capture; routing this batch to the Triton path."
            )
            _ppu_prefill_capture_warned = True
        return _triton_prefill()

    # msa_ppu_forward_extend needs the live ForwardBatch (req_pool_indices
    # for the packed page table, plus a stable per-forward identity so the
    # ~57 sparse layers reuse one forward's plans). The stash set by the
    # metadata hook is always current; fall back defensively if missing.
    forward_batch = getattr(backend, "_ppu_msa_current_batch", None)
    if forward_batch is None:
        return _triton_prefill()

    from sglang.srt.layers.attention.minimax_sparse_ops.msa_ppu import (
        msa_ppu_forward_extend,
    )

    # cu_seqlens == [0] + cumsum(extend_seq_lens): recover the per-request
    # extend lens the MSA planner consumes.
    extend_seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    # The SAIL MSA path never produces idx_o (the gate requires every sparse
    # layer to disable index value); the community caller re-pads o.
    _, o = msa_ppu_forward_extend(
        backend,
        q,
        idx_q,
        idx_k_cache,
        k_cache,
        v_cache,
        forward_batch,
        cu_seqlens=cu_seqlens,
        seq_lens=seq_lens,
        prefix_lens=prefix_lens,
        extend_seq_lens=extend_seq_lens,
        extend_seq_lens_cpu=seqlens_cpu,
        max_seqlen_q=max_seqlen_q,
        layer_id=-1,  # unused by msa_ppu_forward_extend
    )
    return None, o


def _triton_decode(
    original_fn, backend, q, k, v, layer, forward_batch, save_kv_cache, **kwargs
):
    """Fall back to the community Triton decode path.

    An fp8 index-K cache is only readable by the SAIL MSA indexer, so reject
    it before the Triton kernel silently misreads the fp8 bytes.
    """
    _reject_fp8_index_cache(
        (
            backend.kv_pool.get_index_k_buffer(layer.layer_id)
            if layer.layer_id in backend.disable_value_layer_ids
            else backend.kv_pool.get_index_kv_buffer(layer.layer_id)[0]
        ),
        "decode",
    )
    return original_fn(backend, q, k, v, layer, forward_batch, save_kv_cache, **kwargs)


@plugin_hook(f"{_BACKEND}.forward_decode", type=HookType.AROUND)
def _ppu_msa_forward_decode(
    original_fn, self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs
):
    """SAIL MSA decode: indexer + selection + attend, else the Triton path."""
    if not getattr(self, "use_msa_ppu", False):
        return _triton_decode(
            original_fn, self, q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )
    idx_q = kwargs.get("idx_q")
    idx_k = kwargs.get("idx_k")
    if idx_q is None or idx_k is None:
        return _triton_decode(
            original_fn, self, q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )

    from sglang.srt.layers.attention.minimax_sparse_ops.msa_ppu import (
        msa_ppu_forward_decode,
    )

    disable_value = layer.layer_id in self.disable_value_layer_ids
    self.kv_pool.set_fused_kv_index_buffer(
        layer,
        forward_batch.out_cache_loc,
        k,
        v,
        idx_k,
        None if disable_value else kwargs.get("idx_v"),
    )
    k_cache, v_cache = self.kv_pool.get_kv_buffer(layer.layer_id)
    if disable_value:
        idx_k_cache = self.kv_pool.get_index_k_buffer(layer.layer_id)
    else:
        idx_k_cache, _ = self.kv_pool.get_index_kv_buffer(layer.layer_id)
    # The gate requires every sparse layer to disable index value, so
    # idx_k_cache above is the K-only buffer the MSA indexer consumes.
    _, o = msa_ppu_forward_decode(
        self,
        q,
        idx_q,
        idx_k_cache,
        k_cache,
        v_cache,
        forward_batch,
        layer_id=layer.layer_id,
    )
    return None, o
