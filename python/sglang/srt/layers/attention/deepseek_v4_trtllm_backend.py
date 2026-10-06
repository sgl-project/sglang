"""DeepSeek V4 trtllm-gen sparse MLA backend for SM100/SM103.

Overrides only the kernel dispatch of :class:`DeepseekV4AttnBackend`; metadata
construction (incl. the trtllm combined tables) stays on the shared class.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, Optional, Tuple

import torch

from sglang.kernels.ops.attention.dsv4.trtllm_metadata import pack_sparse_tail
from sglang.srt.environ import envs
from sglang.srt.layers.attention.deepseek_v4_backend import (
    PAGE_INDEX_ALIGNED_SIZE,
    SWA_WINDOW,
    DeepseekV4AttnBackend,
    DeepseekV4MultiStepBackend,
)
from sglang.srt.layers.attention.dsv4.metadata import copy_unless_aliased
from sglang.srt.runtime_context import (
    get_buffer,
    get_exec,
    get_parallel,
    get_schedule,
    get_spec,
    max_prefill_buffer_tokens,
)
from sglang.srt.utils import ceil_align, ceil_div

try:
    import flashinfer.mla._core as _fi_core
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


_trtllm_semaphore_installed = False
# Capacity is in query rows: requests x draft tokens for decode, sum_q for prefill.
_trtllm_semaphore_rows: int = 0


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
    spec = get_spec()
    rows_per_req = (
        (spec.speculative_num_draft_tokens or 1)
        if spec.speculative_algorithm is not None
        else 1
    )
    rows = max(rows, (schedule.max_running_requests or 0) * rows_per_req)
    return max(rows, 1)


def _install_persistent_trtllm_semaphores(capacity_rows: int) -> None:
    """FlashInfer sizes its counter buffer by request count while the kernel
    indexes it by query row; install one sized by rows (remove once FlashInfer
    accepts a caller-owned buffer)."""
    global _trtllm_semaphore_installed, _trtllm_semaphore_rows
    if _flashinfer_import_error is not None:
        raise ImportError(
            "--dsv4-attn-backend trtllm requires FlashInfer sparse MLA support"
        ) from _flashinfer_import_error
    _trtllm_semaphore_rows = max(_trtllm_semaphore_rows, capacity_rows)
    if _trtllm_semaphore_installed:
        return
    _orig = _fi_core._get_trtllm_gen_multi_ctas_kv_counter_buffer
    # Allocate once outside graph capture. Stream ordering and the kernel's
    # counter reset make one shared buffer safe across launches.
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
    # A plain exception, not assert: an over-capacity launch scribbles past
    # the semaphore buffer, so this must fire even under python -O.
    if num_rows > _trtllm_semaphore_rows:
        raise RuntimeError(
            f"trtllm-gen launch with {num_rows} query rows exceeds the persistent "
            f"semaphore capacity of {_trtllm_semaphore_rows} rows derived from "
            "--chunked-prefill-size / --max-prefill-tokens / "
            "--max-running-requests; lower --chunked-prefill-size."
        )


class TrtllmSparseTablePool:
    """One persistent int32 parent per table role, handed out as a ``[:rows]``
    view (2-D when ``width`` is given) whose 64-row tile pad is re-inerted each
    step, so kernel-visible addresses never depend on allocator state. Capacity
    grows only outside CUDA-graph capture."""

    def __init__(self, int32_kwargs: dict):
        self._kwargs = int32_kwargs
        self._bufs: dict = {}

    @staticmethod
    def _pad_rows(rows: int) -> int:
        return ceil_align(rows, 64)

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
        self._parent(role, self._pad_rows(rows) * max(width, 1), fill)

    def view(
        self,
        role: str,
        rows: int,
        fill: int,
        src: Optional[torch.Tensor] = None,
        width: int = 0,
        *,
        rows_written_by_caller: bool = False,
    ) -> torch.Tensor:
        """Rows [:rows] come from ``src``, from the caller (who then writes every
        column the kernel can read), or are re-inerted; rows [rows, rows_pad)
        are always re-inerted."""
        rows_pad = self._pad_rows(rows)
        numel = rows_pad * max(width, 1)
        padded = self._parent(role, numel, fill)[:numel]
        if width:
            padded = padded.view(rows_pad, width)
        if src is not None:
            padded[rows:].fill_(fill)
            padded[:rows].copy_(src)
        elif rows_written_by_caller:
            padded[rows:].fill_(fill)
        else:
            padded.fill_(fill)
        return padded[:rows]


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
        self.trtllm_graph_output_buffer: torch.Tensor | None = None
        self.trtllm_eager_output_buffer: torch.Tensor | None = None
        # (buffer, rows, real rows) whose pad tail is already zero (kernel writes [:real]).
        self._padded_output_zeroed: Optional[tuple[int, int, int]] = None
        # Indexer top-k writes into the table tail only when strided topk_v2 is the writer.
        self.trtllm_topk_writes_table = (
            self.dsa_topk_backend.should_use_topk_v2()
            and not get_exec().features.enable_return_indexer_topk
        )

        # Preallocate every role at its maximum so nothing is allocated while serving.
        self.trtllm_table_pool = TrtllmSparseTablePool(self.cuda_int32_kwargs)
        max_decode_rows = self.req_to_token.shape[0] * (
            self.speculative_num_draft_tokens or 1
        )
        max_prefill_rows = _trtllm_query_row_capacity(model_runner)
        w4 = ceil_align(self.index_topk, PAGE_INDEX_ALIGNED_SIZE)
        # c128 pages of the longest representable sequence, plus the producer's
        # own alignment block.
        w128 = (
            ceil_align(
                ceil_div(self.MAX_SEQ_LEN_FOR_CAPTURE, 128), PAGE_INDEX_ALIGNED_SIZE
            )
            + PAGE_INDEX_ALIGNED_SIZE
        )
        pool = self.trtllm_table_pool
        pool.preallocate("d_swa_lens", max_decode_rows, fill=SWA_WINDOW)
        pool.preallocate("p_swa", max_prefill_rows, fill=-1, width=SWA_WINDOW)
        pool.preallocate("p_swa_lens", max_prefill_rows, fill=SWA_WINDOW)
        if self.has_c4:
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
        compress_ratio: Literal[0, 1, 2, 4, 128],
        core_attn_metadata: DSV4AttnMetadata,
        attn_sink: torch.Tensor,
        extra_indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Sparse MLA decode. Uniform multi-token metadata (verify /
        draft-extend) switches the call to varlen mode with per-request
        ``seq_lens``; plain decode stays one row per request."""

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
        if extra_indices is not None:
            extra_indices = extra_indices[:bs]

        # Only the c4 tail and lens vary by layer; other table data is prebuilt.
        if compress_ratio == 0:
            # swa_page_indices is itself a valid all-SWA combined table.
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
            assert sparse_indices.shape[0] > bs, f"{sparse_indices.shape=} {bs=}"
            sparse_indices = sparse_indices[:bs]
        if sparse_topk_lens.shape[0] != bs:
            assert sparse_topk_lens.shape[0] > bs, f"{sparse_topk_lens.shape=}"
            sparse_topk_lens = sparse_topk_lens[:bs]

        if compress_ratio == 4:
            assert extra_indices is not None
            width = extra_indices.shape[-1]
            assert SWA_WINDOW + width == sparse_indices.shape[1], (
                f"{width=} {sparse_indices.shape=}"
            )
            # No-op when the indexer already wrote the tail in place.
            copy_unless_aliased(sparse_indices[:, SWA_WINDOW:], extra_indices)

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
        if swa_width > SWA_WINDOW:
            # DSpark attends noncausally to its context window plus the whole
            # draft block. The tail still indexes SWA storage, not compressed
            # KV. Use the complete block's visible length rather than each
            # token's causal length, including for short requests.
            compressed_kv_cache = swa_kv_cache
            seq_lens = core_attn_metadata.swa_topk_lengths[:bs]
        assert attn_sink.dtype == torch.float32
        assert self.trtllm_workspace_buffer is not None
        _check_trtllm_query_rows(bs)

        # Uniform verify/draft-extend metadata uses VarSeq; ragged verify uses dense per-token.
        seq_lens_req = core_attn_metadata.trtllm_seq_lens_req
        cum_seq_lens_q = core_attn_metadata.trtllm_cum_seq_lens_q
        common = dict(
            swa_kv_cache=swa_kv_cache,
            workspace_buffer=self.trtllm_workspace_buffer,
            sparse_indices=sparse_indices,
            compressed_kv_cache=compressed_kv_cache,
            sparse_topk_lens=sparse_topk_lens,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            sinks=attn_sink,
            kv_layout="HND",
        )
        out_arg = None if out_pad_tail is None else out_pad_tail[:bs]
        if seq_lens_req is not None:
            # Builders use a fixed num_tokens_per_req, so ragged rows cannot reach here.
            n_req = seq_lens_req.shape[0]
            assert n_req > 0 and bs % n_req == 0, (
                f"non-uniform multi-token batch reached the trtllm decode "
                f"path: {bs=} {n_req=}"
            )
            q_len = bs // n_req
            assert cum_seq_lens_q is not None
            assert cum_seq_lens_q.shape == (n_req + 1,), (
                f"{cum_seq_lens_q.shape=} {n_req=}"
            )
            out = trtllm_batch_decode_sparse_mla_dsv4(
                query=q_fp8,
                seq_lens=seq_lens_req,
                out=out_arg,
                cum_seq_lens_q=cum_seq_lens_q,
                max_q_len=q_len,
                **common,
            )
        else:
            out = trtllm_batch_decode_sparse_mla_dsv4(
                query=q_fp8.view(bs, 1, num_heads, 512),
                seq_lens=seq_lens,
                out=None if out_arg is None else out_arg.view(bs, 1, num_heads, 512),
                **common,
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
        """Sparse MLA prefill in the dense one-query-token-per-entry shape
        (``(sum_q, 1)``, per-token causal ``seq_lens``): faster than VarSeq on
        ragged chunks."""

        assert q.ndim == 3, f"{q.shape=}"
        num_qo_padded, num_heads, head_dim = q.shape
        assert head_dim == 512

        # Dense per-token query layout from the host-side extend lens.
        core = self.forward_metadata.core_attn_metadata
        if core.trtllm_prefill_qmeta is None:
            extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
            assert extend_seq_lens_cpu is not None and len(extend_seq_lens_cpu) > 0
            sum_q = sum(int(x) for x in extend_seq_lens_cpu)
            # q may be padded past the real extend tokens; pad rows sit at the end.
            assert 0 < sum_q <= num_qo_padded, f"{sum_q=} {num_qo_padded=}"
            seq_lens_i32 = core.seq_lens_casual[:sum_q].to(torch.int32)
            assert seq_lens_i32.shape == (sum_q,), f"{seq_lens_i32.shape=}"
            core.trtllm_prefill_qmeta = (sum_q, seq_lens_i32)
        sum_q, seq_lens = core.trtllm_prefill_qmeta

        # Layer-invariant tables were built before the indexer ran.
        swa_indices = core.trtllm_prefill_swa_indices
        assert swa_indices is not None
        assert swa_indices.shape == (sum_q, SWA_WINDOW), f"{swa_indices.shape=}"
        if extra_indices is None:
            sparse_indices = swa_indices
            sparse_topk_lens = core.trtllm_prefill_swa_lens
        elif compress_ratio == 128:
            assert core.trtllm_prefill_c128 is not None
            sparse_indices, sparse_topk_lens = core.trtllm_prefill_c128
        else:
            width = extra_indices.shape[-1]
            # _pad_last_dim keeps the combined c4 capacity divisible by four.
            assert width % 4 == 0, f"{width=}"
            sparse_indices = core.trtllm_prefill_c4_indices
            assert sparse_indices is not None
            assert sparse_indices.shape == (
                sum_q,
                SWA_WINDOW + width,
            ), f"{sparse_indices.shape=} {width=}"
            copy_unless_aliased(sparse_indices[:, SWA_WINDOW:], extra_indices[:sum_q])
            sparse_topk_lens = core.trtllm_prefill_c4_lens

        assert sparse_topk_lens is not None

        # No-op when the fused q kernel already stored e4m3 (scale 1.0 cast otherwise).
        q_fp8 = q[:sum_q].to(torch.float8_e4m3fn).view(sum_q, 1, num_heads, 512)

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
