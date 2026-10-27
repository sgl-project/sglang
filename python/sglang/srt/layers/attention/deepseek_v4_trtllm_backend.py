"""DeepSeek V4 trtllm-gen sparse MLA backend for SM100/SM103.

Overrides only the kernel dispatch of :class:`DeepseekV4AttnBackend`; metadata
construction (incl. the trtllm combined tables) stays on the shared class.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, Optional, Sequence, Tuple

import torch

from sglang.kernels.ops.attention.dsv4.trtllm_metadata import pack_sparse_tail
from sglang.srt.environ import envs
from sglang.srt.layers.attention.deepseek_v4_backend import (
    PAGE_INDEX_ALIGNED_SIZE,
    SWA_WINDOW,
    DeepseekV4AttnBackend,
    DeepseekV4MultiStepBackend,
)
from sglang.srt.layers.attention.deepseek_v4_mega_backend import (
    DeepseekV4MegaAttnBackend,
)
from sglang.srt.layers.attention.dsv4.metadata import copy_unless_aliased
from sglang.srt.layers.attention.trtllm_mla_backend import (
    make_persistent_multi_ctas_kv_counter_buffer,
)
from sglang.srt.runtime_context import (
    get_buffer,
    get_context,
    get_exec,
    get_parallel,
    get_schedule,
    get_spec,
    max_prefill_buffer_tokens,
    max_speculative_num_draft_tokens,
)
from sglang.srt.utils import ceil_align, ceil_div, print_info_once

try:
    from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4
except ImportError as exc:
    _flashinfer_import_error = exc
else:
    _flashinfer_import_error = None

if TYPE_CHECKING:
    from sglang.srt.layers.attention.deepseek_v4_backend import DSV4AttnMetadata
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

# Shared zero-initialized workspace managed by the persistent-buffer lifecycle.
_TRTLLM_GEN_WORKSPACE_SIZE_MB = 128


def _get_trtllm_workspace_buffer(device: torch.device) -> torch.Tensor:
    return get_buffer(
        "trtllm_dsv4_zero_workspace",
        lambda: torch.zeros(
            _TRTLLM_GEN_WORKSPACE_SIZE_MB * 1024 * 1024,
            dtype=torch.int8,
            device=device,
        ),
    )


def _trtllm_query_row_capacity(model_runner: ModelRunner) -> int:
    """Bound query rows across prefill chunks and speculative decode batches.

    The DSv4 hook rejects the backend when chunked prefill is disabled, so the
    prefill chunk bound is always finite here.
    """
    schedule = get_schedule()
    rows = max(
        schedule.max_prefill_tokens or 0,
        max_prefill_buffer_tokens(),
    )
    # Adaptive spec may widen the draft up to its launch-time max.
    rows_per_req = (
        (max_speculative_num_draft_tokens() or 1)
        if get_spec().speculative_algorithm is not None
        else 1
    )
    rows = max(rows, (schedule.max_running_requests or 0) * rows_per_req)
    return max(rows, 1)


def _get_trtllm_counter_buffer(
    model_runner: ModelRunner, device: torch.device
) -> torch.Tensor:
    """Return the zeroed multi-CTA KV counter buffer shared by all launches.

    Without it FlashInfer zeroes a fresh buffer before every launch. The kernel
    resets its counters at the end of each launch, and launches run in order on
    the forward stream, as the shared workspace already assumes. Decode passes
    one query row per FlashInfer request, so the size bound is in query rows;
    the model's head count bounds the per-rank query heads, floored at the
    64-head query pad.
    """
    if _flashinfer_import_error is not None:
        raise ImportError(
            "--dsv4-attn-backend trtllm requires FlashInfer sparse MLA support"
        ) from _flashinfer_import_error
    return get_buffer(
        "trtllm_dsv4_multi_ctas_kv_counter",
        lambda: make_persistent_multi_ctas_kv_counter_buffer(
            device,
            max(64, model_runner.model_config.num_attention_heads),
            max(
                _trtllm_query_row_capacity(model_runner),
                model_runner.req_to_token_pool.req_to_token.shape[0]
                * (max_speculative_num_draft_tokens() or 1),
            ),
        ),
    )


def _tail_refresh_layers(
    compress_ratios: Sequence[int], index_source_layer_ids: Sequence[int]
) -> Optional[frozenset]:
    """Layers that must refresh the shared indexed tail; None means every indexed layer.

    Layers without their own indexer reuse the last source's top-k, already in the table.
    """
    if not index_source_layer_ids:
        return None
    sources = frozenset(index_source_layer_ids)
    prev_ratio = None
    for layer_id, ratio in enumerate(compress_ratios):
        if ratio in (1, 2, 4):
            # A non-source layer must follow a source of its own ratio.
            assert ratio == prev_ratio or layer_id in sources, (
                f"indexed layer {layer_id} (ratio {ratio}) reuses top-k across ratios"
            )
            prev_ratio = ratio
    return sources


def _refresh_indexed_tail(
    core: DSV4AttnMetadata,
    compress_ratio: int,
    extra_indices: Optional[torch.Tensor],
    table: torch.Tensor,
    lens: torch.Tensor,
) -> None:
    """Write this layer's top-k into the shared indexed table and its lens."""
    rows = table.shape[0]
    extra_indices = extra_indices[:rows]
    width = extra_indices.shape[-1]
    assert table.shape == (rows, SWA_WINDOW + width), f"{table.shape=} {width=}"
    if compress_ratio == 4 and core._c4_topk_writes_table():
        # Lens were set per step; the indexer wrote the tail in place.
        copy_unless_aliased(table[:, SWA_WINDOW:], extra_indices)
        return
    pack_sparse_tail(
        extra_indices,
        core.sparse_topk_lengths(compress_ratio)[:rows],
        table,
        lens,
    )


class TrtllmSparseTablePool:
    """One persistent int32 parent per table role, handed out as a ``[:rows]``
    view (2-D when ``width`` is given) whose 64-row tile pad is re-inerted each
    step, so kernel-visible addresses never depend on allocator state. Capacity
    grows only outside CUDA-graph capture."""

    def __init__(self, int32_kwargs: dict):
        self._kwargs = int32_kwargs
        self._bufs: dict = {}

    def _parent(self, role: str, numel: int, fill: int) -> torch.Tensor:
        buf = self._bufs.get(role)
        if buf is None or buf.numel() < numel:
            assert not torch.cuda.is_current_stream_capturing(), (
                f"trtllm table pool role {role!r} would (re)allocate during "
                "CUDA graph capture; preallocate it with enough rows first."
            )
            buf = torch.full((numel,), fill, **self._kwargs)
            self._bufs[role] = buf
        return buf

    def preallocate(self, role: str, rows: int, fill: int, width: int = 0) -> None:
        self._parent(role, ceil_align(rows, 64) * max(width, 1), fill)

    def view(
        self,
        role: str,
        rows: int,
        fill: int,
        src: Optional[torch.Tensor] = None,
        width: int = 0,
        *,
        rows_written_by_caller: bool = False,
        constant: bool = False,
    ) -> torch.Tensor:
        """Rows [:rows] come from ``src``, from the caller (who then writes every
        column the kernel can read), or are re-inerted; rows [rows, rows_pad)
        are re-inerted too, except in a ``constant`` role, which nothing writes."""
        rows_pad = ceil_align(rows, 64)
        numel = rows_pad * max(width, 1)
        padded = self._parent(role, numel, fill)[:numel]
        if width:
            padded = padded.view(rows_pad, width)
        if src is not None:
            padded[rows:].fill_(fill)
            padded[:rows].copy_(src)
        elif rows_written_by_caller:
            padded[rows:].fill_(fill)
        elif not constant:
            padded.fill_(fill)
        return padded[:rows]


class DeepseekV4TrtllmAttnBackend(DeepseekV4AttnBackend):
    """DSV4 attention through the trtllm-gen sparse MLA kernel."""

    trtllm_attn: bool = True
    # FlashInfer kernel choice (--dsv4-trtllm-flashinfer-backend); set in __init__.
    trtllm_fi_backend: str = "trtllm-gen"

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        speculative_step_id=0,
        topk=0,
        speculative_num_steps=0,
    ):
        super().__init__(
            model_runner,
            skip_prefill=skip_prefill,
            speculative_step_id=speculative_step_id,
            topk=topk,
            speculative_num_steps=speculative_num_steps,
        )
        assert self.token_to_kv_pool.uniform_fp8, (
            "the trtllm backend requires the uniform-FP8 DSv4 KV pool."
        )
        assert not envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get(), (
            "--dsv4-attn-backend trtllm does not support "
            "SGLANG_OPT_USE_ONLINE_COMPRESS yet."
        )
        assert get_parallel().attn_cp_size == 1, (
            "--dsv4-attn-backend trtllm does not support "
            "context parallelism (attn_cp_size > 1) yet."
        )
        self.trtllm_workspace_buffer = _get_trtllm_workspace_buffer(self.device)
        self.trtllm_counter_buffer = _get_trtllm_counter_buffer(
            model_runner, self.device
        )
        self.trtllm_fi_backend = get_exec().kernel.dsv4_trtllm_flashinfer_backend
        if self.trtllm_fi_backend == "auto":
            from flashinfer.mla._core import _resolve_dsv4_sparse_mla_backend

            # This backend runs only on SM10x; FlashInfer raises for an SM10x
            # outside SM100/SM103/SM107 (SM12x's "auto" is another call surface).
            resolved = _resolve_dsv4_sparse_mla_backend(self.device, "auto")
            print_info_once(f"DSv4 trtllm attention: FlashInfer 'auto' -> {resolved!r}")
        self.trtllm_graph_output_buffer: torch.Tensor | None = None
        self.trtllm_eager_output_buffer: torch.Tensor | None = None
        # (buffer, rows, real rows) whose pad tail is already zero (kernel writes [:real]).
        self._padded_output_zeroed: Optional[tuple[int, int, int]] = None
        cfg = model_runner.model_config.hf_text_config
        self._tail_refresh_layers = _tail_refresh_layers(
            cfg.compress_ratios, cfg.index_source_layer_ids
        )
        # topk_v2 writes the c4 tail in place only when no ratio-1/2 layer shares the table.
        self.trtllm_topk_writes_table = (
            self.dsa_topk_backend.should_use_topk_v2()
            and not get_exec().features.enable_return_indexer_topk
            and not self.low_ratios
        )

        # Preallocate every role at its maximum so nothing is allocated while serving.
        self.trtllm_table_pool = TrtllmSparseTablePool(self.cuda_int32_kwargs)
        # Adaptive spec may widen the draft later; capture cannot grow the pool.
        max_decode_rows = self.req_to_token.shape[0] * (
            max_speculative_num_draft_tokens() or 1
        )
        max_prefill_rows = _trtllm_query_row_capacity(model_runner)
        w4 = ceil_align(self.index_topk, PAGE_INDEX_ALIGNED_SIZE)
        # Longest sequence's c128 pages plus the producer's alignment block.
        w128 = (
            ceil_align(
                ceil_div(self.MAX_SEQ_LEN_FOR_CAPTURE, 128), PAGE_INDEX_ALIGNED_SIZE
            )
            + PAGE_INDEX_ALIGNED_SIZE
        )
        pool = self.trtllm_table_pool
        pool.preallocate("d_swa_lens", max_decode_rows, fill=SWA_WINDOW)
        pool.preallocate("d_swa_lens_wide", max_decode_rows, fill=SWA_WINDOW)
        pool.preallocate("p_swa", max_prefill_rows, fill=-1, width=SWA_WINDOW)
        pool.preallocate("p_swa_lens", max_prefill_rows, fill=SWA_WINDOW)
        if self.has_c4 or self.low_ratios:
            pool.preallocate("d_c4", max_decode_rows, fill=-1, width=SWA_WINDOW + w4)
            pool.preallocate("d_c4_lens", max_decode_rows, fill=SWA_WINDOW)
            pool.preallocate("p_c4", max_prefill_rows, fill=-1, width=SWA_WINDOW + w4)
            pool.preallocate("p_c4_lens", max_prefill_rows, fill=SWA_WINDOW)
        if self.has_c128:
            pool.preallocate(
                "d_c128", max_decode_rows, fill=-1, width=SWA_WINDOW + w128
            )
            pool.preallocate("d_c128_lens", max_decode_rows, fill=SWA_WINDOW)
            pool.preallocate("p_c128_lens", max_prefill_rows, fill=SWA_WINDOW)

    def _build_late_layer_tail_metadata(self, *args, **kwargs):
        # The tail's p_* tables share the pool parents with the full metadata's;
        # building them now would overwrite tables that layers before the switch read.
        self.trtllm_defer_tables = True
        try:
            return super()._build_late_layer_tail_metadata(*args, **kwargs)
        finally:
            self.trtllm_defer_tables = False

    def enter_late_layer_tail(self, forward_batch: ForwardBatch) -> tuple:
        saved = super().enter_late_layer_tail(forward_batch)
        # The full metadata's tables are dead after the switch; reuse the parents.
        tail_core = self.forward_metadata.core_attn_metadata
        tail_core.init_trtllm_prefill_sparse_buffers(
            sum(self.forward_metadata.late_layer_tail.extend_seq_lens_cpu)
        )
        tail_core.trtllm_tail_needs_pack = True
        return saved

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
        """Return reusable BF16 output storage whose pad tail is zero."""
        assert 0 <= num_real_rows < num_rows

        def fits(b: Optional[torch.Tensor]) -> bool:
            return b is not None and b.shape[0] >= num_rows and b.shape[1] == num_heads

        buffer = self.trtllm_graph_output_buffer
        if not fits(buffer):
            assert not torch.cuda.is_current_stream_capturing(), (
                "trtllm DSV4 padded output exceeded its preallocated CUDA-graph "
                "capacity"
            )
            buffer = self.trtllm_eager_output_buffer
            if not fits(buffer):
                buffer = torch.empty(
                    (num_rows, num_heads, 512),
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                self.trtllm_eager_output_buffer = buffer
        output = buffer[:num_rows]
        key = (buffer.data_ptr(), num_rows, num_real_rows)
        if self._padded_output_zeroed != key:
            output[num_real_rows:].zero_()
            self._padded_output_zeroed = key
        return output

    def _forward_trtllm(
        self,
        *,
        q: torch.Tensor,
        layer: RadixAttention,
        compress_ratio: Literal[0, 1, 2, 4, 128],
        core_attn_metadata: DSV4AttnMetadata,
        forward_batch: ForwardBatch,
        attn_sink: torch.Tensor,
        extra_indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        assert attn_sink is not None
        if self.is_dsv41:
            # FlashMLA pads TP query heads to 64. TRT-LLM accepts the native
            # width; do not compute attention for the discarded padding.
            q = q[:, : layer.tp_q_head_num, :]
            attn_sink = attn_sink[: layer.tp_q_head_num]
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
                extra_indices=extra_indices,
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

    def _refreshes_tail(self, layer_id: int) -> bool:
        return (
            self._tail_refresh_layers is None or layer_id in self._tail_refresh_layers
        )

    def _get_trtllm_bmm_scales(self, layer: RadixAttention) -> Tuple[float, float]:
        """Return host scales; KV uses the store path's fixed unit scale.

        Tensor scales corrupt split-KV reduction on FlashInfer < 0.6.13.
        """

        assert layer.k_scale_float is None or layer.k_scale_float == 1.0, (
            "--dsv4-attn-backend trtllm stores KV with a "
            "fixed per-tensor scale of 1.0; a non-unit checkpoint kv-cache "
            f"scale (k_scale_float={layer.k_scale_float}) is not supported yet."
        )
        return (self.softmax_scale, 1.0)

    def _trtllm_kv_cache_views(
        self, layer_id: int, compress_ratio: Literal[0, 1, 2, 4, 128]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return HND views of the uniform-FP8 SWA and compressed pools.

        SWA-only layers pass the SWA pool as the required compressed tensor;
        ``sparse_topk_lens`` masks that region.
        """

        token_to_kv_pool = self.token_to_kv_pool
        swa_buf = token_to_kv_pool.get_swa_key_buffer_radix(layer_id)
        swa_page_size = token_to_kv_pool.get_swa_key_page_size()
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
        compress_ratio: Literal[0, 1, 2, 4, 128],
        core_attn_metadata: DSV4AttnMetadata,
        attn_sink: torch.Tensor,
        extra_indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Sparse MLA decode, one query token per row with per-token causal seq_lens."""

        bs, num_heads, head_dim = q.shape
        assert head_dim == 512

        # Draft-extend q may carry DP pad rows beyond the metadata: run on the
        # covered rows and zero the discarded tail, as for padded prefill.
        n_meta_rows = core_attn_metadata.seq_lens_casual.shape[0]
        out_pad_tail = None
        if n_meta_rows < bs:
            out_pad_tail = self._padded_output_buffer(
                num_rows=bs,
                num_real_rows=n_meta_rows,
                num_heads=num_heads,
            )
            q = q[:n_meta_rows]
            bs = n_meta_rows

        # Only indexed tails and lens vary by layer; other table data is prebuilt.
        if compress_ratio == 0:
            # swa_page_indices is itself a valid all-SWA combined table.
            sparse_indices = core_attn_metadata.swa_page_indices
            sparse_topk_lens = core_attn_metadata.trtllm_swa_lens
        elif compress_ratio == 128:
            sparse_indices = core_attn_metadata.trtllm_c128_indices
            sparse_topk_lens = core_attn_metadata.trtllm_c128_lens
        else:
            # Ratios 1, 2 and 4 share one indexed table.
            sparse_indices = core_attn_metadata.trtllm_c4_indices
            sparse_topk_lens = core_attn_metadata.trtllm_c4_lens
        assert sparse_indices is not None and sparse_topk_lens is not None, (
            "trtllm decode requires metadata built with "
            "init_trtllm_sparse_buffers (decode-mode DSV4AttnMetadata)"
        )
        if sparse_indices.shape[0] != bs:
            assert sparse_indices.shape[0] > bs, f"{sparse_indices.shape=} {bs=}"
            sparse_indices = sparse_indices[:bs]
        if sparse_topk_lens.shape[0] != bs:
            assert sparse_topk_lens.shape[0] > bs, f"{sparse_topk_lens.shape=}"
            sparse_topk_lens = sparse_topk_lens[:bs]

        refreshed = compress_ratio in (1, 2, 4) and self._refreshes_tail(layer.layer_id)
        if refreshed:
            _refresh_indexed_tail(
                core_attn_metadata,
                compress_ratio,
                extra_indices,
                sparse_indices,
                sparse_topk_lens,
            )

        swa_kv_cache, compressed_kv_cache = self._trtllm_kv_cache_views(
            layer.layer_id, compress_ratio
        )

        # No-op when the fused q kernel already stored e4m3 (scale 1.0 cast otherwise).
        q_fp8 = q.to(torch.float8_e4m3fn)

        bmm1_scale, bmm2_scale = self._get_trtllm_bmm_scales(layer)

        seq_lens = core_attn_metadata.seq_lens_casual
        if seq_lens.shape[0] != bs:
            assert seq_lens.shape[0] > bs, f"{seq_lens.shape=} {bs=}"
            seq_lens = seq_lens[:bs]
        if core_attn_metadata.swa_page_indices.shape[1] > SWA_WINDOW:
            # DSpark attends noncausally to its window plus the whole draft block;
            # dense rows bounded by the block's visible length, all from SWA storage.
            compressed_kv_cache = swa_kv_cache
            seq_lens = core_attn_metadata.swa_topk_lengths[:bs]
        assert attn_sink.dtype == torch.float32
        assert self.trtllm_workspace_buffer is not None

        out = trtllm_batch_decode_sparse_mla_dsv4(
            query=q_fp8.view(bs, 1, num_heads, 512),
            swa_kv_cache=swa_kv_cache,
            workspace_buffer=self.trtllm_workspace_buffer,
            multi_ctas_kv_counter_buffer=self.trtllm_counter_buffer,
            sparse_indices=sparse_indices,
            compressed_kv_cache=compressed_kv_cache,
            sparse_topk_lens=sparse_topk_lens,
            seq_lens=seq_lens,
            out=None if out_pad_tail is None else out_pad_tail[:bs, None],
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            sinks=attn_sink,
            kv_layout="HND",
            backend=self.trtllm_fi_backend,
            # The kernel loads sparse_topk_lens before griddepcontrol.wait, so
            # it must not overlap the tail refresh that just wrote them.
            enable_pdl=False if refreshed else None,
        )
        if out_pad_tail is not None:
            return out_pad_tail
        return out.view(bs, num_heads, 512)

    def _forward_trtllm_prefill(
        self,
        *,
        q: torch.Tensor,
        layer: RadixAttention,
        compress_ratio: Literal[0, 1, 2, 4, 128],
        forward_batch: ForwardBatch,
        attn_sink: torch.Tensor,
        extra_indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Per-token sparse MLA prefill; eager, including at BCG breaks."""

        assert q.ndim == 3, f"{q.shape=}"
        num_qo_padded, num_heads, head_dim = q.shape
        assert head_dim == 512

        # Dense per-token query layout from the host-side extend lens.
        core = self.forward_metadata.core_attn_metadata
        if core.trtllm_prefill_qmeta is None:
            tail = self.forward_metadata.late_layer_tail
            extend_seq_lens_cpu = (
                tail.extend_seq_lens_cpu
                if tail is not None
                else forward_batch.extend_seq_lens_cpu
            )
            assert extend_seq_lens_cpu is not None and len(extend_seq_lens_cpu) > 0
            sum_q = sum(int(x) for x in extend_seq_lens_cpu)
            # q may be padded past the real extend tokens; pad rows sit at the end.
            assert 0 < sum_q <= num_qo_padded, f"{sum_q=} {num_qo_padded=}"
            # SWA columns are bounded by causal lengths, not -1 sentinels; tail and
            # request-window rows use their floored SWA counts, since window KV below
            # the floor is unwritten.
            floored = tail is not None or core.request_window_layout is not None
            seq_lens_i32 = (core.swa_topk_lengths if floored else core.seq_lens_casual)[
                :sum_q
            ].to(torch.int32)
            assert seq_lens_i32.shape == (sum_q,), f"{seq_lens_i32.shape=}"
            core.trtllm_prefill_qmeta = (sum_q, seq_lens_i32)
        sum_q, seq_lens = core.trtllm_prefill_qmeta

        # Layer-invariant tables were built before the indexer ran.
        swa_indices = core.trtllm_prefill_swa_indices
        assert swa_indices.shape == (sum_q, SWA_WINDOW), f"{swa_indices.shape=}"
        refreshed = False
        if extra_indices is None:
            sparse_indices = swa_indices
            sparse_topk_lens = core.trtllm_prefill_swa_lens
        elif compress_ratio == 128:
            assert core.trtllm_prefill_c128 is not None
            sparse_indices, sparse_topk_lens = core.trtllm_prefill_c128
        else:
            # Ratios 1, 2 and 4 share one indexed table.
            sparse_indices = core.trtllm_prefill_c4_indices
            sparse_topk_lens = core.trtllm_prefill_c4_lens
            assert sparse_indices is not None and sparse_topk_lens is not None
            if self._refreshes_tail(layer.layer_id) or core.trtllm_tail_needs_pack:
                _refresh_indexed_tail(
                    core,
                    compress_ratio,
                    extra_indices,
                    sparse_indices,
                    sparse_topk_lens,
                )
                refreshed = True
                core.trtllm_tail_needs_pack = False

        assert sparse_topk_lens is not None

        # No-op when the fused q kernel already stored e4m3 (scale 1.0 cast otherwise).
        q_fp8 = q[:sum_q].to(torch.float8_e4m3fn).view(sum_q, 1, num_heads, 512)

        swa_kv_cache, compressed_kv_cache = self._trtllm_kv_cache_views(
            layer.layer_id, compress_ratio
        )
        bmm1_scale, bmm2_scale = self._get_trtllm_bmm_scales(layer)
        assert attn_sink.dtype == torch.float32
        assert self.trtllm_workspace_buffer is not None

        out_padded = None
        out_arg = None
        if num_qo_padded != sum_q:
            # Run over the real tokens only; keep the discarded pad rows finite.
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
            multi_ctas_kv_counter_buffer=self.trtllm_counter_buffer,
            sparse_indices=sparse_indices,
            compressed_kv_cache=compressed_kv_cache,
            sparse_topk_lens=sparse_topk_lens,
            seq_lens=seq_lens,
            out=out_arg,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            sinks=attn_sink,
            kv_layout="HND",
            backend=self.trtllm_fi_backend,
            enable_pdl=False if refreshed else None,  # as in decode
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
    # Model construction outside a server (unit tests) has no published exec config.
    if not get_context().is_config_namespace_published("exec"):
        return False
    return get_exec().kernel.dsv4_attn_backend == "trtllm"


def is_dsv4_mega_attn_enabled() -> bool:
    return get_exec().kernel.dsv4_attn_backend == "flashmla_mega"


def create_deepseek_v4_attn_backend(
    model_runner: ModelRunner, **kwargs
) -> DeepseekV4AttnBackend:
    """Construct the DSV4 backend matching --dsv4-attn-backend."""
    if is_dsv4_trtllm_attn_enabled():
        cls = DeepseekV4TrtllmAttnBackend
    elif is_dsv4_mega_attn_enabled():
        cls = DeepseekV4MegaAttnBackend
    else:
        cls = DeepseekV4AttnBackend
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
