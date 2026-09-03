"""DeepSeek V4 attention backend for the flashinfer trtllm-gen sparse MLA
kernel (``--dsv4-attn-backend trtllm``, SM100/SM103).

Subclasses :class:`DeepseekV4AttnBackend` (FlashMLA) and overrides only the
kernel dispatch: decode / target-verify / draft-extend and varlen prefill go
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
from sglang.srt.runtime_context import (
    get_exec,
    get_parallel,
    get_schedule,
    get_spec,
    max_prefill_buffer_tokens,
)

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
# Query rows the persistent semaphore buffer is sized for. Every launch
# (decode rows = requests x draft tokens, varlen prefill rows = sum_q) is
# checked against it in _check_trtllm_query_rows.
_trtllm_semaphore_rows: int = 0


def _trtllm_query_row_capacity(model_runner: ModelRunner) -> int:
    """Conservative bound on query rows per trtllm-gen launch in this process.

    Prefill launches carry at most one chunk (``chunked_prefill_size``, or the
    PP dynamic-chunk ceiling); decode / target-verify launches carry at most
    ``max_running_requests x speculative_num_draft_tokens`` rows.
    """
    schedule = get_schedule()
    rows = max(
        schedule.max_prefill_tokens or 0,
        max_prefill_buffer_tokens(),
    )
    spec = get_spec()
    rows_per_req = (
        (spec.speculative_num_draft_tokens or 1)
        if spec.speculative_algorithm is not None
        else 1
    )
    rows = max(rows, (schedule.max_running_requests or 0) * rows_per_req)
    return max(rows, 1)


def _install_persistent_trtllm_semaphores(capacity_rows: int) -> None:
    """WAR for flashinfer's trtllm-gen multi-CTA KV counter (semaphore)
    buffer handling. flashinfer sizes it batch_size (= requests) x heads and
    allocates it per call, but the kernel indexes semaphores as
    [batch, numCtasForAllHeads, maxNumCtasQ] with maxNumCtasQ == seqLenQ for
    the DSv4 sparse kernels (trtllm-gen TmemCorr.h counterOffset) -- i.e.
    under-allocated for any multi-token launch. Remove once flashinfer
    accepts a caller-provided persistent buffer with tile-aware sizing."""
    global _trtllm_semaphore_installed, _trtllm_semaphore_rows
    _trtllm_semaphore_rows = max(_trtllm_semaphore_rows, capacity_rows)
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
    # the largest configured launch; (b) capture-pool lifetime bugs (a buffer
    # allocated inside capture whose reference dies is recycled by later
    # captures, letting replays scribble counters over other graphs' tensors).
    state: dict = {}

    def _patched(batch_size, num_qo_heads, sm_count, device):
        buf = state.get("buf")
        if buf is None or buf.device != device:
            assert not torch.cuda.is_current_stream_capturing(), (
                "persistent trtllm semaphore buffer must be created outside "
                "graph capture (first call is expected during eager warmup)"
            )
            buf = _orig(_trtllm_semaphore_rows, num_qo_heads, sm_count, device)
            state["buf"] = buf
        return buf

    _fi_core._get_trtllm_gen_multi_ctas_kv_counter_buffer = _patched
    _trtllm_semaphore_installed = True
    logger.info(
        "trtllm-gen multi-CTA semaphores: single persistent buffer sized for "
        "%d query rows, shared across launches (flashinfer sizing WAR).",
        _trtllm_semaphore_rows,
    )


def _check_trtllm_query_rows(num_rows: int) -> None:
    assert num_rows <= _trtllm_semaphore_rows, (
        f"trtllm-gen launch with {num_rows} query rows exceeds the persistent "
        f"semaphore capacity of {_trtllm_semaphore_rows} rows derived from "
        "--chunked-prefill-size / --max-prefill-tokens / --max-running-requests; "
        "lower --chunked-prefill-size (chunking must stay enabled) or raise the "
        "capacity."
    )


class DeepseekV4TrtllmAttnBackend(DeepseekV4AttnBackend):
    """DSV4 attention through the trtllm-gen sparse MLA kernel."""

    trtllm_attn: bool = True

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        speculative_step_id=0,
        topk=0,
        speculative_num_steps=0,
    ):
        _install_persistent_trtllm_semaphores(_trtllm_query_row_capacity(model_runner))
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
        # The varlen prefill packs query tokens contiguously per request
        # (cum_seq_lens_q); CP round-robin reindexing breaks that packing.
        assert get_parallel().attn_cp_size == 1, (
            "--dsv4-attn-backend trtllm does not support "
            "context parallelism (attn_cp_size > 1) yet."
        )
        self.trtllm_workspace_buffer = _get_trtllm_workspace_buffer(self.device)

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
            return self._forward_trtllm_decode(
                q=q,
                layer=layer,
                compress_ratio=compress_ratio,
                core_attn_metadata=core_attn_metadata,
                attn_sink=attn_sink,
                swa_page_indices=swa_page_indices,
                extra_indices=extra_indices,
                extra_topk_lengths=extra_topk_lengths,
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
            swa_page_indices=swa_page_indices,
            extra_indices=extra_indices,
            extra_topk_lengths=extra_topk_lengths,
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
    ) -> torch.Tensor:
        """Sparse MLA decode via ``trtllm_batch_decode_sparse_mla_dsv4``.

        The combined sparse table lives in preallocated metadata buffers.
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
        # FlashMLA tolerates this overhang implicitly; the trtllm tables are
        # exact-rows, hence the explicit slice.
        n_meta_rows = core_attn_metadata.seq_lens_casual.shape[0]
        out_pad_tail = None
        if n_meta_rows < bs:
            out_pad_tail = torch.zeros(
                (bs, num_heads, 512), dtype=torch.bfloat16, device=q.device
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
            # 128, all-SWA); no fill needed. Use the METADATA's table (a
            # [:n] view of a 64-row-aligned inert parent), not the
            # match_num_queries-processed argument: the arg is an exact-row
            # tensor and the VarSeq kernel reads rows to the 64-token tile
            # boundary (tile-overrun guard, see init_trtllm_sparse_buffers).
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
            sparse_indices[:, SWA_WINDOW:].copy_(extra_indices)
            # Lens include the fixed 128 SWA slots; SWA validity itself is
            # derived from seq_lens inside the kernel.
            sparse_topk_lens.copy_(extra_topk_lengths)
            sparse_topk_lens.add_(SWA_WINDOW)

        swa_kv_cache, compressed_kv_cache = self._trtllm_kv_cache_views(
            layer.layer_id, compress_ratio
        )

        # RoPE is already applied upstream; at per-tensor scale 1.0 the FP8
        # quantization is a plain e4m3 cast.
        q_fp8 = q.to(torch.float8_e4m3fn).view(bs, 1, num_heads, 512)

        bmm1_scale, bmm2_scale = self._get_trtllm_bmm_scales(layer)

        seq_lens = core_attn_metadata.seq_lens_casual
        if seq_lens.shape[0] != bs:
            assert seq_lens.shape[0] > bs, f"{seq_lens.shape=} {bs=}"
            seq_lens = seq_lens[:bs]
        assert attn_sink.dtype == torch.float32
        assert self.trtllm_workspace_buffer is not None
        _check_trtllm_query_rows(bs)

        out = trtllm_batch_decode_sparse_mla_dsv4(
            query=q_fp8,
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
        )
        if out_pad_tail is not None:
            out_pad_tail[:bs] = out.view(bs, num_heads, 512)
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
        swa_page_indices: torch.Tensor,
        extra_indices: Optional[torch.Tensor],
        extra_topk_lengths: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Sparse MLA varlen prefill: the decode kernel driven with
        multi-token queries (``cum_seq_lens_q``/``max_q_len``).

        The sparse table has one row per query token: the token's own causal
        SWA window in columns ``[0:128)`` and its compressed tier after.
        ``seq_lens`` must be the per-request TOTAL KV length including any
        cached prefix (chunked prefill / cache-hit extends); the kernel
        derives each token's causal SWA validity from it, so no masks are
        built here. Runs eagerly, so per-call allocations are fine.
        """

        from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4

        assert q.ndim == 3, f"{q.shape=}"
        num_qo_padded, num_heads, head_dim = q.shape
        assert head_dim == 512

        # Varlen query structure, from the same host-side extend lens that
        # produced this metadata (init_forward_metadata_prefill /
        # expand_prefill_casually).
        core = self.forward_metadata.core_attn_metadata
        if core.trtllm_prefill_qmeta is None:
            extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
            assert extend_seq_lens_cpu is not None and len(extend_seq_lens_cpu) > 0
            batch_size = len(extend_seq_lens_cpu)
            cum_lens = [0] * (batch_size + 1)
            for i, extend_len in enumerate(extend_seq_lens_cpu):
                cum_lens[i + 1] = cum_lens[i] + int(extend_len)
            sum_q = cum_lens[-1]
            max_q_len = max(int(x) for x in extend_seq_lens_cpu)
            # q (and the per-token metadata rows, via match_num_queries) may
            # be padded past the real extend tokens; the pad rows sit at the
            # end.
            assert 0 < sum_q <= num_qo_padded, f"{sum_q=} {num_qo_padded=}"
            # Per-request TOTAL KV length (cached prefix + extend tokens).
            seq_lens_i32 = forward_batch.seq_lens.to(torch.int32)
            assert seq_lens_i32.shape == (batch_size,), f"{seq_lens_i32.shape=}"
            core.trtllm_prefill_qmeta = (
                self._move_to_device(cum_lens),
                max_q_len,
                sum_q,
                seq_lens_i32,
            )
        cum_seq_lens_q, max_q_len, sum_q, seq_lens = core.trtllm_prefill_qmeta

        # Combined per-token sparse table (physical indices, -1 invalid).
        # Layer-invariant parts are cached per chunk on the metadata: the c0
        # lens, the c4 table's SWA half (per-layer: the indexer top-k tail +
        # lens), and the whole c128 table + lens (its page list is
        # metadata-level).
        # TILE OVERRUN GUARD (see init_trtllm_sparse_buffers): the VarSeq
        # kernel reads per-token table rows up to the 64-token tile boundary,
        # so allocate every per-token tensor as a 64-row-aligned inert parent
        # and hand the kernel [:sum_q] views.
        sum_q_pad = (sum_q + 63) // 64 * 64

        def _tile_padded_pf(fill, src=None, width=None):
            shape = (sum_q_pad,) if width is None else (sum_q_pad, width)
            buf = torch.full(shape, fill, **self.cuda_int32_kwargs)
            if src is not None:
                buf[:sum_q].copy_(src)
            return buf[:sum_q]

        swa_indices = _tile_padded_pf(-1, swa_page_indices[:sum_q], width=SWA_WINDOW)
        assert swa_indices.shape == (sum_q, SWA_WINDOW), f"{swa_indices.shape=}"
        if extra_indices is None:
            # SWA-only (compress_ratio == 0) layer. SWA_WINDOW satisfies the
            # kernel's capacity constraints (>= 128, % 4 == 0).
            sparse_indices = swa_indices
            if core.trtllm_prefill_swa_lens is None:
                core.trtllm_prefill_swa_lens = _tile_padded_pf(SWA_WINDOW)
            sparse_topk_lens = core.trtllm_prefill_swa_lens
        elif compress_ratio == 128:
            if core.trtllm_prefill_c128 is None:
                width = extra_indices.shape[-1]
                assert width % 4 == 0, f"{width=}"
                table = _tile_padded_pf(-1, width=SWA_WINDOW + width)
                table[:, :SWA_WINDOW].copy_(swa_indices)
                table[:, SWA_WINDOW:].copy_(extra_indices[:sum_q])
                assert extra_topk_lengths is not None
                lens = _tile_padded_pf(
                    SWA_WINDOW,
                    extra_topk_lengths[:sum_q].to(torch.int32) + SWA_WINDOW,
                )
                core.trtllm_prefill_c128 = (table, lens)
            sparse_indices, sparse_topk_lens = core.trtllm_prefill_c128
        else:
            assert extra_topk_lengths is not None
            width = extra_indices.shape[-1]
            # c4 index tables are padded to multiples of 64 upstream
            # (_pad_last_dim), so the combined capacity satisfies % 4 == 0.
            assert width % 4 == 0, f"{width=}"
            if core.trtllm_prefill_c4_indices is None:
                core.trtllm_prefill_c4_indices = _tile_padded_pf(
                    -1, width=SWA_WINDOW + width
                )
                core.trtllm_prefill_c4_indices[:, :SWA_WINDOW].copy_(swa_indices)
            sparse_indices = core.trtllm_prefill_c4_indices
            assert sparse_indices.shape == (
                sum_q,
                SWA_WINDOW + width,
            ), f"{sparse_indices.shape=} {width=}"
            sparse_indices[:, SWA_WINDOW:].copy_(extra_indices[:sum_q])
            # Total lens include the fixed 128 SWA slots; SWA validity itself
            # is derived from seq_lens/cum_seq_lens_q inside the kernel.
            sparse_topk_lens = _tile_padded_pf(
                SWA_WINDOW,
                extra_topk_lengths[:sum_q].to(torch.int32) + SWA_WINDOW,
            )

        # FP8 query: RoPE already applied upstream; per-tensor scale 1.0 makes
        # quantization a plain e4m3 cast (same recipe as the decode branch).
        q_fp8 = q[:sum_q].to(torch.float8_e4m3fn)

        swa_kv_cache, compressed_kv_cache = self._trtllm_kv_cache_views(
            layer.layer_id, compress_ratio
        )
        bmm1_scale, bmm2_scale = self._get_trtllm_bmm_scales(layer)
        assert attn_sink.dtype == torch.float32
        assert self.trtllm_workspace_buffer is not None
        _check_trtllm_query_rows(sum_q)

        out_padded = None
        out_arg = None
        if num_qo_padded != sum_q:
            # Padded prefill: run the kernel over the real tokens only and
            # zero the pad rows (their outputs are discarded downstream, but
            # keep them finite so nothing NaN-propagates).
            out_padded = torch.zeros(
                (num_qo_padded, num_heads, 512),
                dtype=torch.bfloat16,
                device=q.device,
            )
            out_arg = out_padded[:sum_q]

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
            cum_seq_lens_q=cum_seq_lens_q,
            max_q_len=max_q_len,
        )
        return out_padded if out_padded is not None else out


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
