"""DeepSeek V4 attention backend for the flashinfer trtllm-gen sparse MLA
kernel (``--dsv4-attn-backend trtllm``, SM100/SM103).

Subclasses :class:`DeepseekV4AttnBackend` (FlashMLA) and overrides only the
kernel dispatch: decode / target-verify / draft-extend and dense prefill go
through ``trtllm_batch_decode_sparse_mla_dsv4`` against the uniform 512-dim
FP8 KV pools. Metadata construction (including the trtllm combined sparse
tables, ``DSV4AttnMetadata.init_trtllm_sparse_buffers``) stays on the shared
metadata class so BCG capture/replay ``copy_``/``assign_fields`` semantics
are identical for both backends.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.deepseek_v4_backend import (
    SWA_WINDOW,
    DeepseekV4AttnBackend,
    DeepseekV4MultiStepBackend,
)
from sglang.srt.runtime_context import get_exec, get_parallel

if TYPE_CHECKING:
    from sglang.srt.layers.attention.deepseek_v4_backend import DSV4AttnMetadata
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

# trtllm decode workspace: one zero-initialized int8 buffer, allocated once
# through the RuntimeContext persistent-buffer lifecycle and shared by all
# backend instances (same pattern as trtllm_mla_backend's
# "trtllm_mla_zero_workspace").
_TRTLLM_GEN_WORKSPACE_SIZE_MB = 128


def _get_trtllm_workspace_buffer(device: torch.device) -> torch.Tensor:
    from sglang.srt.runtime_context import get_buffer

    return get_buffer(
        "trtllm_dsv4_zero_workspace",
        lambda: torch.zeros(
            _TRTLLM_GEN_WORKSPACE_SIZE_MB * 1024 * 1024,
            dtype=torch.int8,
            device=device,
        ),
    )


_trtllm_semaphore_installed = False


def _install_persistent_trtllm_semaphores() -> None:
    """WAR for flashinfer's trtllm-gen multi-CTA KV counter (semaphore)
    buffer handling. flashinfer sizes it batch_size (= requests) x heads and
    allocates it per call, but the kernel indexes semaphores as
    [batch, numCtasForAllHeads, maxNumCtasQ] with maxNumCtasQ == seqLenQ for
    the DSv4 sparse kernels (trtllm-gen TmemCorr.h counterOffset) -- i.e.
    under-allocated for any multi-token launch. Remove once flashinfer
    accepts a caller-provided persistent buffer with tile-aware sizing."""
    global _trtllm_semaphore_installed
    if _trtllm_semaphore_installed:
        return
    import flashinfer.mla._core as _fi_core

    _orig = _fi_core._get_trtllm_gen_multi_ctas_kv_counter_buffer
    # Single persistent counter buffer, allocated once OUTSIDE any graph
    # capture and shared by every launch -- the same design TRT-LLM uses
    # (AttentionOp::mMultiBlockSemaphores). The kernel self-resets counters
    # at the end of each launch and launches are stream-ordered, so sharing
    # is safe. This removes both failure modes of per-call allocation:
    # (a) under-sizing for VarSeq multi-token launches (kernel indexes
    # counters per q-tile, TmemCorr.h counterOffset formula), sized here for
    # 16384 query rows; (b) capture-pool lifetime bugs (a buffer allocated
    # inside capture whose reference dies is recycled by later captures,
    # letting replays scribble counters over other graphs' tensors).
    state: dict = {}

    def _patched(batch_size, num_qo_heads, sm_count, device):
        buf = state.get("buf")
        if buf is None or buf.device != device:
            assert not torch.cuda.is_current_stream_capturing(), (
                "persistent trtllm semaphore buffer must be created outside "
                "graph capture (first call is expected during eager warmup)"
            )
            buf = _orig(16384, num_qo_heads, sm_count, device)
            state["buf"] = buf
        return buf

    _fi_core._get_trtllm_gen_multi_ctas_kv_counter_buffer = _patched
    _fi_core._trtllm_semaphore_state = state
    _trtllm_semaphore_installed = True
    logger.info(
        "trtllm-gen multi-CTA semaphores: single persistent 16384-row buffer "
        "shared across launches (flashinfer sizing WAR)."
    )


class TrtllmSparseTablePool:
    """Persistent backing storage for the trtllm combined sparse tables.

    The trtllm-gen kernel consumes exact-address int32 tables; building them
    as fresh allocations every step made their addresses and lifetime a
    function of allocator/capture state, which correlated with historical
    layout-dependent faults and silent accuracy drift. This pool allocates
    each table role ONCE (constant column width, conservatively 64-row-aligned
    capacity, filled with its inert value) and hands out ``[:rows]`` views
    that are re-inerted and rewritten in place every step:

    - addresses are stable, so CUDA graphs capture permanent pointers and
      there is nothing for the capture pool to recycle;
    - every mapped byte is either the inert fill or a value the current step
      legitimately wrote, keeping any padding/guard region deterministic;
    - views keep row stride == column width, so downstream ``.contiguous()``
      calls never copy.

    Roles used from CUDA-graph capture must be preallocated via
    ``preallocate`` before the first capture; ``view`` asserts it never
    (re)allocates while a stream is capturing.
    """

    def __init__(self, int32_kwargs: dict):
        self._kwargs = int32_kwargs
        self._bufs: dict = {}
        self._fills: dict = {}

    @staticmethod
    def _pad_rows(rows: int) -> int:
        # Retain the historical 64-row guard conservatively. Current cubins
        # pass memcheck with exact-row allocations, so this rounding is not a
        # proven kernel requirement and can be removed after full graph tests.
        return (rows + 63) // 64 * 64

    def preallocate(self, role: str, rows: int, fill: int, width: int = 0) -> None:
        self._ensure(role, self._pad_rows(rows), fill, width)

    def _ensure(self, role: str, rows_pad: int, fill: int, width: int) -> torch.Tensor:
        buf = self._bufs.get(role)
        if buf is not None:
            # width == 0 with an existing 2-D buffer means "use the
            # preallocated width" (the caller writes a column subrange).
            if width:
                assert (
                    buf.dim() == 2 and buf.shape[1] == width
                ), f"table role {role!r} width changed: {buf.shape} vs {width=}"
            assert self._fills[role] == fill, f"table role {role!r} fill changed"
        if buf is None or buf.shape[0] < rows_pad:
            assert not torch.cuda.is_current_stream_capturing(), (
                f"trtllm table pool role {role!r} would (re)allocate during "
                "CUDA graph capture; preallocate it with enough rows first."
            )
            new_rows = max(rows_pad, 0 if buf is None else buf.shape[0])
            if buf is not None and buf.dim() == 2:
                width = buf.shape[1]
            shape = (new_rows,) if not width else (new_rows, width)
            buf = torch.full(shape, fill, **self._kwargs)
            self._bufs[role] = buf
        self._fills[role] = fill
        return self._bufs[role]

    def view(
        self,
        role: str,
        rows: int,
        fill: int,
        src: Optional[torch.Tensor] = None,
        width: int = 0,
    ) -> torch.Tensor:
        """A ``[:rows]`` view of the role's parent, re-inerted then filled
        with ``src`` (if given) up to ``rows``. The 64-row-aligned tail is
        re-inerted too, matching the previous per-step allocation semantics.
        """
        rows_pad = self._pad_rows(rows)
        buf = self._ensure(role, rows_pad, fill, width)
        padded = buf[:rows_pad]
        padded.fill_(fill)
        if src is not None:
            padded[:rows].copy_(src)
        return padded[:rows]


class DeepseekV4TrtllmAttnBackend(DeepseekV4AttnBackend):
    """DSV4 attention through the trtllm-gen sparse MLA kernel."""

    trtllm_attn: bool = True
    # trtllm-gen handles per-rank head counts natively (verified
    # bit-identical h=16 vs padded h=64); skip the TP head padding.
    pads_tp_q_heads: bool = False

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        speculative_step_id=0,
        topk=0,
        speculative_num_steps=0,
    ):
        _install_persistent_trtllm_semaphores()
        super().__init__(
            model_runner,
            skip_prefill=skip_prefill,
            speculative_step_id=speculative_step_id,
            topk=topk,
            speculative_num_steps=speculative_num_steps,
        )
        assert (
            self.token_to_kv_pool.uniform_fp8
        ), "the trtllm backend requires the uniform-FP8 DSv4 KV pool."
        assert not envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get(), (
            "--dsv4-attn-backend trtllm does not support "
            "SGLANG_OPT_USE_ONLINE_COMPRESS yet."
        )
        # The TRT sparse-table path has not been validated with CP's
        # round-robin token/table reindexing.
        assert get_parallel().attn_cp_size == 1, (
            "--dsv4-attn-backend trtllm does not support "
            "context parallelism (attn_cp_size > 1) yet."
        )
        self.trtllm_workspace_buffer = _get_trtllm_workspace_buffer(self.device)
        self.trtllm_graph_output_buffer: torch.Tensor | None = None
        self.trtllm_eager_output_buffer: torch.Tensor | None = None
        # Let the indexer's per-layer top-k write straight into the combined
        # table tail (decode/verify) -- only when the stride-capable topk_v2
        # kernel is the guaranteed writer (sgl-kernel backend, v2 enabled,
        # no indexer-capture side channel that reroutes to the v1 kernel).
        self.trtllm_topk_writes_table = (
            envs.SGLANG_OPT_USE_TOPK_V2.get()
            and model_runner.server_args.dsa_topk_backend == "sgl-kernel"
            and not getattr(
                model_runner.server_args, "enable_return_indexer_topk", False
            )
        )

        # Persistent combined-table storage (see TrtllmSparseTablePool). The
        # decode-side roles are consumed inside CUDA-graph capture, so they
        # are preallocated here at their maxima: rows = one row per query
        # token of the largest verify batch; c128 columns = the whole
        # context in 128-token pages (per-row lens bound what the kernel
        # actually reads, so unused columns are dead weight, not reads).
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            PAGE_INDEX_ALIGNED_SIZE,
            SWA_WINDOW,
        )

        def _align(x: int) -> int:
            a = PAGE_INDEX_ALIGNED_SIZE
            return (x + a - 1) // a * a

        self.trtllm_table_pool = TrtllmSparseTablePool(self.cuda_int32_kwargs)
        max_decode_rows = self.req_to_token.shape[0] * max(
            1, self.speculative_num_draft_tokens or 1
        )
        w4 = _align(self.c4_topk)
        # c128 pages for the longest representable sequence: bound by the
        # req_to_token row length (context + scheduler margin), which is what
        # graph capture uses as its max seq len, plus one alignment block for
        # the producer's own padding.
        w128 = (
            _align((self.MAX_SEQ_LEN_FOR_CAPTURE + 127) // 128)
            + PAGE_INDEX_ALIGNED_SIZE
        )
        pool = self.trtllm_table_pool
        pool.preallocate("d_swa_lens", max_decode_rows, fill=SWA_WINDOW)
        pool.preallocate("d_c4", max_decode_rows, fill=-1, width=SWA_WINDOW + w4)
        pool.preallocate("d_c4_lens", max_decode_rows, fill=SWA_WINDOW)
        pool.preallocate("d_c128", max_decode_rows, fill=-1, width=SWA_WINDOW + w128)
        pool.preallocate("d_c128_lens", max_decode_rows, fill=SWA_WINDOW)
        # Prefill c128 also uses a full-context-width parent (its per-chunk
        # width varies, and constant width keeps row stride == width so
        # nothing downstream re-copies). Rows grow on demand (prefill is
        # eager); other prefill roles carry their width at the call site.
        pool.preallocate("p_c128", 64, fill=-1, width=SWA_WINDOW + w128)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int) -> None:
        super().init_cuda_graph_state(max_bs, max_num_tokens)
        num_heads = (
            self.model_runner.model_config.num_attention_heads
            // get_parallel().attn_tp_size
        )
        self.trtllm_graph_output_buffer = torch.empty(
            (max_num_tokens, num_heads, 512),
            dtype=torch.bfloat16,
            device=self.device,
        )

    def _padded_output_buffer(
        self,
        *,
        num_rows: int,
        num_real_rows: int,
        num_heads: int,
    ) -> torch.Tensor:
        """Return reusable BF16 output storage with a freshly zeroed pad tail."""
        assert 0 <= num_real_rows < num_rows
        buffer = self.trtllm_graph_output_buffer
        if buffer is None or buffer.shape[0] < num_rows or buffer.shape[1] != num_heads:
            assert not torch.cuda.is_current_stream_capturing(), (
                "trtllm DSV4 padded output exceeded its preallocated CUDA-graph "
                "capacity"
            )
            buffer = self.trtllm_eager_output_buffer
            if (
                buffer is None
                or buffer.shape[0] < num_rows
                or buffer.shape[1] != num_heads
            ):
                buffer = torch.empty(
                    (num_rows, num_heads, 512),
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                self.trtllm_eager_output_buffer = buffer
        output = buffer[:num_rows]
        output[num_real_rows:].zero_()
        return output

    # Metadata prep runs in-graph (base-class default), same as FlashMLA.
    # This subclass used to prep on the host for spec + DP attention (an
    # early containment for a draft-acceptance drop later attributed to the
    # idle-rank dummy-extend bug); the host-side copy path itself proved to
    # carry a stale-buffer defect -- silent GSM8K 0.65 with collapsed
    # acceptance once upstream's fused-mHC default shifted graph-pool
    # layout -- while in-graph prep measures clean on the same recipe
    # (15/15 bursts 0.945-0.970, accept 0.96+, idle-rank window healthy).

    def _forward_trtllm(
        self,
        *,
        q: torch.Tensor,
        layer: RadixAttention,
        compress_ratio: Literal[0, 4, 128],
        core_attn_metadata: DSV4AttnMetadata,
        forward_batch: ForwardBatch,
        attn_sink: torch.Tensor,
        swa_page_indices: torch.Tensor,
        extra_indices: Optional[torch.Tensor],
        extra_topk_lengths: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # The uniform-FP8 pool is only readable by the trtllm
        # backend. Decode runs one row per request; target-verify and
        # draft-extend run one row per query token against the
        # per-token metadata rows (seq_lens_casual and the per-token
        # index tables built by the caller), exactly like the flashmla path.
        assert attn_sink is not None
        if (
            forward_batch.forward_mode.is_decode_or_idle()
            or forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):
            # Verify / draft-extend run a UNIFORM number of query tokens per
            # request (num_draft_tokens); drive the kernel in varlen mode
            # (cum_seq_lens_q) for those, matching how TRT-LLM invokes it.
            # Plain decode stays one row per request.
            q_len_uniform = 1
            if not forward_batch.forward_mode.is_decode_or_idle():
                q_len_uniform = self.speculative_num_draft_tokens or 1
            return self._forward_trtllm_decode(
                q=q,
                layer=layer,
                compress_ratio=compress_ratio,
                core_attn_metadata=core_attn_metadata,
                attn_sink=attn_sink,
                swa_page_indices=swa_page_indices,
                extra_indices=extra_indices,
                extra_topk_lengths=extra_topk_lengths,
                q_len_uniform=q_len_uniform,
            )
        assert forward_batch.forward_mode.is_extend_without_speculative(), (
            "uniform-FP8 pool cannot be read by the packed FlashMLA "
            f"kernels; unsupported forward mode "
            f"{forward_batch.forward_mode} under "
            "--dsv4-attn-backend trtllm"
        )
        return self._forward_trtllm_prefill(
            q=q,
            layer=layer,
            compress_ratio=compress_ratio,
            forward_batch=forward_batch,
            attn_sink=attn_sink,
            extra_indices=extra_indices,
        )

    def _get_trtllm_bmm_scales(self, layer: RadixAttention) -> Tuple[float, float]:
        """Fixed (bmm1, bmm2) = (softmax_scale, 1.0) as host floats.

        The kv dequant scale must be 1.0 because the uniform-FP8 store
        quantizes at 1.0. We use host floats because tensor scales corrupt split-KV
        reduction on flashinfer <0.6.13, and these are per-layer constants.
        """

        assert layer.k_scale_float is None or layer.k_scale_float == 1.0, (
            "--dsv4-attn-backend trtllm stores KV with a "
            "fixed per-tensor scale of 1.0; a non-unit checkpoint kv-cache "
            f"scale (k_scale_float={layer.k_scale_float}) is not supported yet."
        )
        return (self.softmax_scale, 1.0)

    def _trtllm_kv_cache_views(
        self, layer_id: int, compress_ratio: Literal[0, 4, 128]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """HND paged views ``[pages, 1, page_size, 512]`` (e4m3) of the
        uniform-FP8 SWA and compressed-tier pools.

        The kernel requires a compressed cache tensor even for SWA-only
        (``compress_ratio == 0``) layers; the SWA pool is passed there and
        the compressed region stays fully masked via ``sparse_topk_lens``.
        """

        token_to_kv_pool = self.token_to_kv_pool
        swa_buf = token_to_kv_pool.get_swa_key_buffer_radix(layer_id)
        swa_page_size = token_to_kv_pool.swa_kv_pool.page_size
        swa_kv_cache = swa_buf.view(swa_buf.shape[0], 1, swa_page_size, 512)
        if compress_ratio == 0:
            compressed_kv_cache = swa_kv_cache
        else:
            extra_buf = token_to_kv_pool.get_extra_key_buffer(layer_id)
            extra_page_size = token_to_kv_pool.get_extra_key_page_size(layer_id)
            compressed_kv_cache = extra_buf.view(
                extra_buf.shape[0], 1, extra_page_size, 512
            )
        return swa_kv_cache, compressed_kv_cache

    def _forward_trtllm_decode(
        self,
        *,
        q: torch.Tensor,
        layer: RadixAttention,
        compress_ratio: Literal[0, 4, 128],
        core_attn_metadata: DSV4AttnMetadata,
        attn_sink: torch.Tensor,
        swa_page_indices: torch.Tensor,
        extra_indices: Optional[torch.Tensor],
        extra_topk_lengths: Optional[torch.Tensor],
        q_len_uniform: int = 1,
    ) -> torch.Tensor:
        """Sparse MLA decode via ``trtllm_batch_decode_sparse_mla_dsv4``.

        The combined sparse table lives in preallocated metadata buffers.
        ``q_len_uniform > 1`` (target-verify / draft-extend: every request
        carries exactly that many query tokens) switches the call to varlen
        mode: ``cum_seq_lens_q`` describes the per-request token runs and
        ``seq_lens`` becomes per-request totals, while the per-token table
        rows and lens are consumed in flattened query-token order (kernel
        contract). Kernel-level A/B: bit-identical outputs, perf within
        0-5 percent of the one-row-per-request mode at verify shapes.
        """

        from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4

        bs, num_heads, head_dim = q.shape
        assert head_dim == 512

        # Pre-pad-planned metadata: draft-extend plans its metadata BEFORE
        # prepare_mlp_sync_batch DP-pads the batch (prepare_for_draft_extend
        # marks the plan non-replannable, see #27091), so under DP MAX_LEN
        # padding q can carry MORE rows than the metadata. The extra rows are
        # padding whose outputs are discarded downstream; run the kernel on
        # the metadata-covered rows only and zero-fill the tail (finite, so
        # nothing NaN-propagates) -- same recipe as the padded-prefill path.
        # FlashMLA tolerates this overhang implicitly. The trtllm API instead
        # derives its logical table extent from q.shape[0], hence the slice.
        n_meta_rows = core_attn_metadata.seq_lens_casual.shape[0]
        out_pad_tail = None
        if n_meta_rows < bs:
            out_pad_tail = self._padded_output_buffer(
                num_rows=bs,
                num_real_rows=n_meta_rows,
                num_heads=num_heads,
            )
            q = q[:n_meta_rows]
            swa_page_indices = swa_page_indices[:n_meta_rows]
            if extra_indices is not None:
                extra_indices = extra_indices[:n_meta_rows]
            if extra_topk_lengths is not None:
                extra_topk_lengths = extra_topk_lengths[:n_meta_rows]
            bs = n_meta_rows

        # Per-ratio tables: layer-invariant content (SWA columns, the whole
        # c128 table, the c0 lens) was written once per step by
        # init_trtllm_sparse_buffers; only the c4 tail + lens (indexer
        # top-k) are written here.
        assert swa_page_indices.shape == (bs, SWA_WINDOW)
        if compress_ratio == 0:
            # swa_page_indices is itself a valid combined table (capacity
            # 128, all-SWA); no fill needed. Use the metadata's stable [:n]
            # view of a role-owned parent, not the match_num_queries-processed
            # argument, whose allocation/lifetime is not capture-stable.
            sparse_indices = core_attn_metadata.swa_page_indices
            sparse_topk_lens = core_attn_metadata.trtllm_swa_lens
        elif compress_ratio == 128:
            sparse_indices = core_attn_metadata.trtllm_c128_indices
            sparse_topk_lens = core_attn_metadata.trtllm_c128_lens
        else:
            sparse_indices = core_attn_metadata.trtllm_c4_indices
            sparse_topk_lens = core_attn_metadata.trtllm_c4_lens
        assert sparse_indices is not None and sparse_topk_lens is not None, (
            "trtllm decode requires metadata built with "
            "init_trtllm_sparse_buffers (decode-mode DSV4AttnMetadata)"
        )
        if sparse_indices.shape[0] != bs:
            # Metadata may be built against a padded batch; slice to the live
            # rows (views -- the c4 fill below stays in place).
            assert sparse_indices.shape[0] > bs, f"{sparse_indices.shape=} {bs=}"
            sparse_indices = sparse_indices[:bs]
        if sparse_topk_lens.shape[0] != bs:
            assert sparse_topk_lens.shape[0] > bs, f"{sparse_topk_lens.shape=}"
            sparse_topk_lens = sparse_topk_lens[:bs]

        if compress_ratio == 4:
            assert extra_indices is not None and extra_topk_lengths is not None
            width = extra_indices.shape[-1]
            assert (
                SWA_WINDOW + width == sparse_indices.shape[1]
            ), f"{width=} {sparse_indices.shape=}"
            # Only the index tail is per-layer (each layer's indexer top-k);
            # the lens (metadata-level c4_sparse_topk_lengths + SWA_WINDOW)
            # were written once per step by init_trtllm_sparse_buffers. When
            # trtllm_topk_writes_table aliased c4_sparse_page_indices to this
            # very tail, the indexer already wrote it in place -- skip the
            # copy (the alias decision is capture-stable, so this host branch
            # is safe under CUDA graphs).
            tail = sparse_indices[:, SWA_WINDOW:]
            if extra_indices.data_ptr() != tail.data_ptr() or extra_indices.stride(
                0
            ) != tail.stride(0):
                tail.copy_(extra_indices)

        swa_kv_cache, compressed_kv_cache = self._trtllm_kv_cache_views(
            layer.layer_id, compress_ratio
        )

        # RoPE is already applied upstream; the fused q norm+rope kernel
        # usually stores e4m3 directly (see _compute_q_b), otherwise the
        # per-tensor-scale-1.0 FP8 quantization is a plain e4m3 cast.
        q_fp8 = q if q.dtype == torch.float8_e4m3fn else q.to(torch.float8_e4m3fn)

        bmm1_scale, bmm2_scale = self._get_trtllm_bmm_scales(layer)

        seq_lens = core_attn_metadata.seq_lens_casual
        if seq_lens.shape[0] != bs:
            assert seq_lens.shape[0] > bs, f"{seq_lens.shape=} {bs=}"
            seq_lens = seq_lens[:bs]
        assert attn_sink.dtype == torch.float32
        assert self.trtllm_workspace_buffer is not None

        # Uniform verify/draft metadata carries prebuilt per-request qmeta and
        # uses VarSeq. Compact ragged verify deliberately omits that qmeta and
        # falls back to the dense one-query-token-per-entry layout, whose
        # per-token seq_lens/tables do not require uniform request groups.
        seq_lens_req = core_attn_metadata.trtllm_seq_lens_req
        cum_seq_lens_q = core_attn_metadata.trtllm_cum_seq_lens_q
        varlen = q_len_uniform > 1 and seq_lens_req is not None
        if varlen:
            # Uniformity is guaranteed by construction, not assumed: both
            # verify and draft-extend v2 metadata builders (graph replay AND
            # eager, see init_forward_metadata_draft_extend call sites) use a
            # fixed num_tokens_per_req == speculative_num_draft_tokens via
            # expand_extend_with_same_length, so ragged rows cannot reach
            # this call. Assert rather than silently falling back to the
            # one-row-per-request mode.
            assert bs > 0 and bs % q_len_uniform == 0, (
                f"non-uniform multi-token batch reached the trtllm decode "
                f"path: {bs=} {q_len_uniform=}"
            )
            n_req = bs // q_len_uniform
            assert cum_seq_lens_q is not None
            assert seq_lens_req.shape == (n_req,), f"{seq_lens_req.shape=} {n_req=}"
            assert cum_seq_lens_q.shape == (
                n_req + 1,
            ), f"{cum_seq_lens_q.shape=} {n_req=}"
            out = trtllm_batch_decode_sparse_mla_dsv4(
                query=q_fp8,
                swa_kv_cache=swa_kv_cache,
                workspace_buffer=self.trtllm_workspace_buffer,
                sparse_indices=sparse_indices,
                compressed_kv_cache=compressed_kv_cache,
                sparse_topk_lens=sparse_topk_lens,
                seq_lens=seq_lens_req,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                sinks=attn_sink,
                kv_layout="HND",
                out=None if out_pad_tail is None else out_pad_tail[:bs],
                cum_seq_lens_q=cum_seq_lens_q,
                max_q_len=q_len_uniform,
            )
        else:
            out = trtllm_batch_decode_sparse_mla_dsv4(
                query=q_fp8.view(bs, 1, num_heads, 512),
                swa_kv_cache=swa_kv_cache,
                workspace_buffer=self.trtllm_workspace_buffer,
                sparse_indices=sparse_indices,
                compressed_kv_cache=compressed_kv_cache,
                sparse_topk_lens=sparse_topk_lens,
                seq_lens=seq_lens,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                sinks=attn_sink,
                kv_layout="HND",
                out=(
                    None
                    if out_pad_tail is None
                    else out_pad_tail[:bs].view(bs, 1, num_heads, 512)
                ),
            )
        if out_pad_tail is not None:
            return out_pad_tail
        return out.view(bs, num_heads, 512)

    def _forward_trtllm_prefill(
        self,
        *,
        q: torch.Tensor,
        layer: RadixAttention,
        compress_ratio: Literal[0, 4, 128],
        forward_batch: ForwardBatch,
        attn_sink: torch.Tensor,
        extra_indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Sparse MLA prefill in the dense, one-query-token-per-entry shape.

        Treating every query token as a batch entry makes each query length
        exactly one. This changes the kernel-visible shape from
        ``(request_batch, max_q_len)`` to ``(sum_q, 1)`` and avoids the empty
        rectangular work of a highly ragged VarSeq launch. B200 kernel and
        end-to-end measurements both favor this representation for generic
        mixed/chunked prefill.

        The sparse table has one row per query token: the token's own causal
        SWA window in columns ``[0:128)`` and its compressed tier after.
        ``seq_lens`` is therefore the per-token causal KV length from
        ``core.seq_lens_casual`` rather than one total per request.
        """

        from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4

        assert q.ndim == 3, f"{q.shape=}"
        num_qo_padded, num_heads, head_dim = q.shape
        assert head_dim == 512

        # Dense per-token query structure, from the same host-side extend lens that
        # produced this metadata (init_forward_metadata_prefill /
        # expand_prefill_casually).
        core = self.forward_metadata.core_attn_metadata
        if core.trtllm_prefill_qmeta is None:
            extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
            assert extend_seq_lens_cpu is not None and len(extend_seq_lens_cpu) > 0
            sum_q = sum(int(x) for x in extend_seq_lens_cpu)
            # q (and the per-token metadata rows, via match_num_queries) may
            # be padded past the real extend tokens; the pad rows sit at the
            # end.
            assert 0 < sum_q <= num_qo_padded, f"{sum_q=} {num_qo_padded=}"
            seq_lens_i32 = core.seq_lens_casual[:sum_q].to(torch.int32)
            assert seq_lens_i32.shape == (sum_q,), f"{seq_lens_i32.shape=}"
            core.trtllm_prefill_qmeta = (sum_q, seq_lens_i32)
        sum_q, seq_lens = core.trtllm_prefill_qmeta

        # The layer-invariant tables were built during metadata preparation,
        # before the indexer. In the stride-capable topk_v2 configuration,
        # c4_sparse_page_indices is already the c4 combined table's tail, so
        # the indexer writes directly to the kernel input.
        swa_indices = core.trtllm_prefill_swa_indices
        assert swa_indices is not None
        assert swa_indices.shape == (sum_q, SWA_WINDOW), f"{swa_indices.shape=}"
        if extra_indices is None:
            # SWA-only (compress_ratio == 0) layer. SWA_WINDOW satisfies the
            # kernel's capacity constraints (>= 128, % 4 == 0).
            sparse_indices = swa_indices
            sparse_topk_lens = core.trtllm_prefill_swa_lens
        elif compress_ratio == 128:
            assert core.trtllm_prefill_c128 is not None
            sparse_indices, sparse_topk_lens = core.trtllm_prefill_c128
        else:
            width = extra_indices.shape[-1]
            # c4 index tables are padded to multiples of 64 upstream
            # (_pad_last_dim), so the combined capacity satisfies % 4 == 0.
            assert width % 4 == 0, f"{width=}"
            sparse_indices = core.trtllm_prefill_c4_indices
            assert sparse_indices is not None
            assert sparse_indices.shape == (
                sum_q,
                SWA_WINDOW + width,
            ), f"{sparse_indices.shape=} {width=}"
            tail = sparse_indices[:, SWA_WINDOW:]
            extra_indices = extra_indices[:sum_q]
            if extra_indices.data_ptr() != tail.data_ptr() or extra_indices.stride(
                0
            ) != tail.stride(0):
                tail.copy_(extra_indices)
            sparse_topk_lens = core.trtllm_prefill_c4_lens

        assert sparse_topk_lens is not None

        # FP8 query: RoPE already applied upstream; the fused q norm+rope
        # kernel usually stores e4m3 directly (see _compute_q_b), otherwise
        # the per-tensor-scale-1.0 quantization is a plain e4m3 cast.
        q_fp8 = q[:sum_q]
        if q_fp8.dtype != torch.float8_e4m3fn:
            q_fp8 = q_fp8.to(torch.float8_e4m3fn)
        q_fp8 = q_fp8.view(sum_q, 1, num_heads, 512)

        swa_kv_cache, compressed_kv_cache = self._trtllm_kv_cache_views(
            layer.layer_id, compress_ratio
        )
        bmm1_scale, bmm2_scale = self._get_trtllm_bmm_scales(layer)
        assert attn_sink.dtype == torch.float32
        assert self.trtllm_workspace_buffer is not None

        out_padded = None
        out_arg = None
        if num_qo_padded != sum_q:
            # Padded prefill: run the kernel over the real tokens only and
            # zero the pad rows (their outputs are discarded downstream, but
            # keep them finite so nothing NaN-propagates).
            out_padded = self._padded_output_buffer(
                num_rows=num_qo_padded,
                num_real_rows=sum_q,
                num_heads=num_heads,
            )
            out_arg = out_padded[:sum_q].view(sum_q, 1, num_heads, 512)

        out = trtllm_batch_decode_sparse_mla_dsv4(
            query=q_fp8,
            swa_kv_cache=swa_kv_cache,
            workspace_buffer=self.trtllm_workspace_buffer,
            sparse_indices=sparse_indices,
            compressed_kv_cache=compressed_kv_cache,
            sparse_topk_lens=sparse_topk_lens,
            seq_lens=seq_lens,
            out=out_arg,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            sinks=attn_sink,
            kv_layout="HND",
        )
        return out_padded if out_padded is not None else out.view(sum_q, num_heads, 512)


class DeepseekV4TrtllmMultiStepBackend(
    DeepseekV4MultiStepBackend, DeepseekV4TrtllmAttnBackend
):
    """Multi-step draft wrapper whose per-step backends are trtllm."""

    def _make_step_backend(
        self, model_runner: ModelRunner, step_id: int
    ) -> DeepseekV4AttnBackend:
        return DeepseekV4TrtllmAttnBackend(
            model_runner,
            speculative_step_id=step_id,
            topk=self.topk,
            speculative_num_steps=self.speculative_num_steps,
        )


def is_dsv4_trtllm_attn_enabled() -> bool:
    return get_exec().kernel.dsv4_attn_backend == "trtllm"


def create_deepseek_v4_attn_backend(
    model_runner: ModelRunner, **kwargs
) -> DeepseekV4AttnBackend:
    """Construct the DSV4 backend matching --dsv4-attn-backend."""
    cls = (
        DeepseekV4TrtllmAttnBackend
        if is_dsv4_trtllm_attn_enabled()
        else DeepseekV4AttnBackend
    )
    return cls(model_runner, **kwargs)


def create_deepseek_v4_multistep_backend(
    model_runner: ModelRunner, topk: int, speculative_num_steps: int
) -> DeepseekV4MultiStepBackend:
    cls = (
        DeepseekV4TrtllmMultiStepBackend
        if is_dsv4_trtllm_attn_enabled()
        else DeepseekV4MultiStepBackend
    )
    return cls(model_runner, topk=topk, speculative_num_steps=speculative_num_steps)
