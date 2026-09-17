"""DeepSeek V4 trtllm-gen sparse MLA backend for SM100/SM103.

Decode and varlen prefill use a uniform 512-dim FP8 KV cache. Shared metadata
construction preserves the base backend's CUDA-graph replay semantics.
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

# Shared zero-initialized workspace managed by the persistent-buffer lifecycle.
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
    """Install a persistent counter buffer sized by query rows.

    FlashInfer sizes this private buffer by request count, while the DSV4
    VarSeq kernel indexes it by query row. Remove this workaround once
    FlashInfer accepts a caller-owned buffer.
    """
    global _trtllm_semaphore_installed, _trtllm_semaphore_rows
    _trtllm_semaphore_rows = max(_trtllm_semaphore_rows, capacity_rows)
    if _trtllm_semaphore_installed:
        return
    import flashinfer.mla._core as _fi_core

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
        # CP round-robin reindexing breaks VarSeq's per-request query packing.
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
        self, layer_id: int, compress_ratio: Literal[0, 4, 128]
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
        compress_ratio: Literal[0, 4, 128],
        core_attn_metadata: DSV4AttnMetadata,
        attn_sink: torch.Tensor,
        swa_page_indices: torch.Tensor,
        extra_indices: Optional[torch.Tensor],
        extra_topk_lengths: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run sparse MLA decode with preallocated metadata tables."""

        from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4

        bs, num_heads, head_dim = q.shape
        assert head_dim == 512

        # Draft-extend metadata predates DP MAX_LEN padding (#27091). Run only
        # its covered rows and leave the discarded padding output finite.
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

        # Only the c4 tail and lens vary by layer; other table data is prebuilt.
        assert swa_page_indices.shape == (bs, SWA_WINDOW)
        if compress_ratio == 0:
            # Use the metadata view backed by 64-row-aligned storage because
            # the VarSeq kernel reads table rows to the tile boundary.
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
            assert extra_indices is not None and extra_topk_lengths is not None
            width = extra_indices.shape[-1]
            assert SWA_WINDOW + width == sparse_indices.shape[1], (
                f"{width=} {sparse_indices.shape=}"
            )
            sparse_indices[:, SWA_WINDOW:].copy_(extra_indices)
            # Lens include all 128 SWA slots; seq_lens controls their validity.
            sparse_topk_lens.copy_(extra_topk_lengths)
            sparse_topk_lens.add_(SWA_WINDOW)

        swa_kv_cache, compressed_kv_cache = self._trtllm_kv_cache_views(
            layer.layer_id, compress_ratio
        )

        # RoPE is already applied; the unit scale makes this a plain e4m3 cast.
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
        """Drive the decode kernel as varlen prefill with one table row per token.

        ``seq_lens`` includes cached prefixes; the kernel derives causal SWA
        validity from it. This path runs eagerly.
        """

        from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4

        assert q.ndim == 3, f"{q.shape=}"
        num_qo_padded, num_heads, head_dim = q.shape
        assert head_dim == 512

        # Build VarSeq metadata from the same extend lengths as the sparse tables.
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
            assert 0 < sum_q <= num_qo_padded, f"{sum_q=} {num_qo_padded=}"
            seq_lens_i32 = forward_batch.seq_lens.to(torch.int32)
            assert seq_lens_i32.shape == (batch_size,), f"{seq_lens_i32.shape=}"
            core.trtllm_prefill_qmeta = (
                self._move_to_device(cum_lens),
                max_q_len,
                sum_q,
                seq_lens_i32,
            )
        cum_seq_lens_q, max_q_len, sum_q, seq_lens = core.trtllm_prefill_qmeta

        # Cache layer-invariant table parts per chunk. The VarSeq kernel reads
        # rows to a 64-token boundary, so views need inert, tile-aligned parents.
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
            # _pad_last_dim keeps the combined c4 capacity divisible by four.
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
            # Lens include 128 SWA slots; VarSeq metadata controls validity.
            sparse_topk_lens = _tile_padded_pf(
                SWA_WINDOW,
                extra_topk_lengths[:sum_q].to(torch.int32) + SWA_WINDOW,
            )

        # RoPE is already applied; the unit scale makes this a plain e4m3 cast.
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
            # Run only real tokens and keep discarded padding rows finite.
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
