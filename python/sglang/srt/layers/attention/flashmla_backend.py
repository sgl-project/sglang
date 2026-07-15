"""
Support attention backend for FlashMLA.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Tuple, Union

import torch
import triton
from sgl_kernel.flash_mla import flash_mla_with_kvcache, get_mla_metadata

from sglang.kernels.ops.attention.pad import (
    pad_draft_extend_query as pad_draft_extend_query_triton,
)
from sglang.kernels.ops.attention.pad import (
    unpad_draft_extend_output as unpad_draft_extend_output_triton,
)
from sglang.kernels.ops.attention.utils import (
    create_flashmla_kv_indices_triton,
    get_num_kv_index_blocks_flashmla,
)
from sglang.kernels.ops.quantization.fp8_kernel import scaled_fp8_quant
from sglang.srt.layers.attention.flashinfer_mla_backend import FlashInferMLAAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

PAGE_SIZE = 64


@dataclass
class FlashMLADecodeMetadata:
    flashmla_metadata: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    num_splits: Optional[torch.Tensor] = None
    block_kv_indices: Optional[torch.Tensor] = None
    # K lens the kernel reads for draft-extend (window-aligned, device int32).
    # Decode/verify compute theirs inline from forward_batch.seq_lens.
    cache_seqlens: Optional[torch.Tensor] = None

    def __init__(
        self,
        flashmla_metadata: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        num_splits: Optional[torch.Tensor] = None,
        block_kv_indices: Optional[torch.Tensor] = None,
        cache_seqlens: Optional[torch.Tensor] = None,
    ):
        self.flashmla_metadata = flashmla_metadata
        self.num_splits = num_splits
        self.block_kv_indices = block_kv_indices
        self.cache_seqlens = cache_seqlens


class FlashMLABackend(FlashInferMLAAttnBackend):
    # Decode/verify/draft-extend metadata is built device-side and the
    # tree-mask scratch is preallocated, so no seq_lens_cpu / seq_lens_sum
    # D2H is needed. Prefill (EXTEND) goes through the FlashInferMLA parent,
    # whose batches always carry the CPU mirror from the scheduler.
    needs_cpu_seq_lens: bool = False

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        kv_last_page_len_buf: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            model_runner, skip_prefill, kv_indptr_buf, kv_last_page_len_buf
        )

        self.num_q_heads = (
            model_runner.model_config.num_attention_heads // get_parallel().attn_tp_size
        )
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.num_local_heads = (
            model_runner.model_config.num_attention_heads // get_parallel().attn_tp_size
        )
        self.forward_metadata: Union[FlashMLADecodeMetadata] = None
        self.kv_lora_rank = model_runner.model_config.kv_lora_rank
        self.qk_nope_head_dim = model_runner.model_config.qk_nope_head_dim
        self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
        self.v_head_dim = model_runner.model_config.v_head_dim
        self.scaling = model_runner.model_config.scaling
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.kv_cache_dim = self.kv_lora_rank + self.qk_rope_head_dim
        self.is_fp8_kvcache = self.data_type in {
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        }

        self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens

        self.cuda_graph_kv_indices = None
        self.cuda_graph_mla_metadata = None
        self.cuda_graph_num_splits = None
        self.cuda_graph_mla_metadata_view = None
        self.cuda_graph_num_splits_view = None
        # Static K-lens buffer bound by the draft-extend graph kernel.
        self.cuda_graph_draft_extend_cache_seqlens = None
        # Preallocated tree-mask scratch (see get_verify_buffers_to_fill_after_draft).
        self.cuda_graph_custom_mask = None

        # get dcp info
        self.dcp_world_size = get_parallel().attn_dcp_size
        self.dcp_rank = get_parallel().attn_dcp_rank

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        forward_mode = forward_batch.forward_mode
        if (
            forward_mode.is_decode_or_idle()
            or forward_mode.is_target_verify()
            or forward_mode.is_draft_extend_v2()
        ):
            self._apply_decode_target_verify_metadata(
                bs=forward_batch.batch_size,
                req_pool_indices=forward_batch.req_pool_indices,
                seq_lens=forward_batch.seq_lens,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
                forward_mode=forward_mode,
            )
        else:
            super().init_forward_metadata_out_graph(
                forward_batch, in_capture=in_capture
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        bs = forward_batch.batch_size
        # Host max only sizes the block table: CPU mirror when published,
        # else the static bound (kernel reads are capped by cache_seqlens).
        seq_lens_cpu = forward_batch.seq_lens_cpu
        eager_max_k = (
            seq_lens_cpu.max().item()
            if seq_lens_cpu is not None
            else self.max_context_len
        )
        if forward_batch.forward_mode.is_decode_or_idle():
            max_seqlen_pad = triton.cdiv(eager_max_k, PAGE_SIZE)
            block_kv_indices = torch.full(
                (bs, max_seqlen_pad),
                -1,
                dtype=torch.int32,
                device=forward_batch.seq_lens.device,
            )
            create_flashmla_kv_indices_triton[
                (bs, get_num_kv_index_blocks_flashmla(max_seqlen_pad, PAGE_SIZE))
            ](
                self.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                None,
                block_kv_indices,
                self.req_to_token.stride(0),
                max_seqlen_pad,
            )
            mla_metadata, num_splits = get_mla_metadata(
                forward_batch.seq_lens.to(torch.int32),
                self.num_q_heads,
                1,
                is_fp8_kvcache=self.is_fp8_kvcache,
            )
            self.forward_metadata = FlashMLADecodeMetadata(
                mla_metadata,
                num_splits,
                block_kv_indices,
            )
        elif forward_batch.forward_mode.is_target_verify():
            seq_lens = forward_batch.seq_lens + self.num_draft_tokens

            max_seqlen_pad = triton.cdiv(eager_max_k + self.num_draft_tokens, PAGE_SIZE)
            block_kv_indices = torch.full(
                (bs, max_seqlen_pad),
                -1,
                dtype=torch.int32,
                device=seq_lens.device,
            )
            create_flashmla_kv_indices_triton[
                (bs, get_num_kv_index_blocks_flashmla(max_seqlen_pad, PAGE_SIZE))
            ](
                self.req_to_token,
                forward_batch.req_pool_indices,
                seq_lens,
                None,
                block_kv_indices,
                self.req_to_token.stride(0),
                max_seqlen_pad,
            )
            mla_metadata, num_splits = get_mla_metadata(
                seq_lens.to(torch.int32),
                self.num_draft_tokens * self.num_q_heads,
                1,
                is_fp8_kvcache=self.is_fp8_kvcache,
            )
            self.forward_metadata = FlashMLADecodeMetadata(
                mla_metadata,
                num_splits,
                block_kv_indices,
            )
        elif forward_batch.forward_mode.is_draft_extend_v2():
            # Fixed-q draft-extend: pad every q window to num_draft_tokens and
            # window-align the K lens; padded q rows are discarded after unpad.
            window = self.num_draft_tokens
            cache_seqlens = (
                forward_batch.seq_lens - forward_batch.extend_seq_lens + window
            ).to(torch.int32)

            max_seqlen_pad = triton.cdiv(eager_max_k + window, PAGE_SIZE)
            block_kv_indices = torch.full(
                (bs, max_seqlen_pad),
                -1,
                dtype=torch.int32,
                device=forward_batch.seq_lens.device,
            )
            create_flashmla_kv_indices_triton[
                (bs, get_num_kv_index_blocks_flashmla(max_seqlen_pad, PAGE_SIZE))
            ](
                self.req_to_token,
                forward_batch.req_pool_indices,
                cache_seqlens,
                None,
                block_kv_indices,
                self.req_to_token.stride(0),
                max_seqlen_pad,
            )
            mla_metadata, num_splits = get_mla_metadata(
                cache_seqlens,
                window * self.num_q_heads,
                1,
                is_fp8_kvcache=self.is_fp8_kvcache,
            )
            self.forward_metadata = FlashMLADecodeMetadata(
                mla_metadata,
                num_splits,
                block_kv_indices,
                cache_seqlens,
            )
        else:
            super().init_forward_metadata(forward_batch)

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        block_kv_indices: Optional[torch.Tensor] = None,
    ):
        if block_kv_indices is None:
            self.cuda_graph_kv_indices = torch.full(
                (max_bs, (self.max_context_len + PAGE_SIZE) // PAGE_SIZE),
                1,
                dtype=torch.int32,
                device="cuda",
            )
        else:
            self.cuda_graph_kv_indices = block_kv_indices

        device_props = torch.cuda.get_device_properties(self.req_to_token.device)
        max_num_sm_parts = device_props.multi_processor_count

        self.cuda_graph_mla_metadata = torch.empty(
            (max_num_sm_parts, 8),
            dtype=torch.int32,
            device="cuda",
        )
        self.cuda_graph_num_splits = torch.empty(
            max_bs + 1,
            dtype=torch.int32,
            device="cuda",
        )

        self.cuda_graph_mla_metadata_view = None
        self.cuda_graph_num_splits_view = None

        if self.num_draft_tokens:
            self.cuda_graph_draft_extend_cache_seqlens = torch.ones(
                max_bs, dtype=torch.int32, device="cuda"
            )
            if not self.skip_prefill:
                # Worst-case FULL_MASK tree-mask scratch (bool); build_tree
                # writes it in-place so the GPU-only path needs no seq_lens_sum.
                self.cuda_graph_custom_mask = torch.zeros(
                    max_num_tokens * (self.max_context_len + self.num_draft_tokens),
                    dtype=torch.bool,
                    device="cuda",
                )

    def get_verify_buffers_to_fill_after_draft(self):
        # build_tree_kernel_efficient fills this in-place (no seq_lens_sum).
        return [self.cuda_graph_custom_mask, None]

    def _apply_decode_target_verify_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        forward_mode: ForwardMode,
    ):
        """Shared decode/target-verify/draft-extend capture+replay body.

        Public entry: :py:meth:`init_forward_metadata_out_graph` (which routes
        to this helper for the decode-family modes and falls back to the
        FlashInferMLA parent for prefill).
        """
        if True:
            seq_lens = seq_lens[:bs]
            seq_lens_cpu = seq_lens_cpu[:bs] if seq_lens_cpu is not None else None

            if forward_mode.is_target_verify():
                seq_lens = seq_lens + self.num_draft_tokens
                if seq_lens_cpu is not None:
                    seq_lens_cpu = seq_lens_cpu + self.num_draft_tokens
            # draft_extend_v2 graph batches arrive with the padded q window
            # already included in seq_lens; use them as-is.

            # Tight block-table slice when the CPU mirror is free; static
            # bound otherwise (no D2H; kernel reads are capped by cache_seqlens).
            if seq_lens_cpu is not None:
                max_seqlen_pad = triton.cdiv(seq_lens_cpu.max().item(), PAGE_SIZE)
            else:
                max_seqlen_pad = self.cuda_graph_kv_indices.shape[1]

            create_flashmla_kv_indices_triton[
                (
                    bs,
                    get_num_kv_index_blocks_flashmla(
                        self.cuda_graph_kv_indices.stride(0), PAGE_SIZE
                    ),
                )
            ](
                self.req_to_token,
                req_pool_indices[:bs],
                seq_lens,
                None,
                self.cuda_graph_kv_indices,
                self.req_to_token.stride(0),
                self.cuda_graph_kv_indices.stride(0),
            )

            q_head_mult = (
                self.num_draft_tokens
                if forward_mode.is_target_verify() or forward_mode.is_draft_extend_v2()
                else 1
            )
            mla_metadata, num_splits = get_mla_metadata(
                seq_lens.to(torch.int32),
                q_head_mult * self.num_q_heads,
                1,
                is_fp8_kvcache=self.is_fp8_kvcache,
            )

            actual_num_sm_parts = mla_metadata.shape[0]
            assert actual_num_sm_parts <= self.cuda_graph_mla_metadata.shape[0], (
                f"num_sm_parts {actual_num_sm_parts} exceeds preallocated max "
                f"{self.cuda_graph_mla_metadata.shape[0]}"
            )

            if (
                self.cuda_graph_mla_metadata_view is None
                or actual_num_sm_parts != self.cuda_graph_mla_metadata_view.shape[0]
            ):
                if self.cuda_graph_mla_metadata_view is not None:
                    logger.warning(
                        f"num_sm_parts mismatch in CUDA Graph replay: "
                        f"capture={self.cuda_graph_mla_metadata_view.shape[0]}, "
                        f"replay={actual_num_sm_parts}. "
                        f"This may indicate batch size changed between capture and replay."
                    )
                self.cuda_graph_mla_metadata_view = self.cuda_graph_mla_metadata[
                    :actual_num_sm_parts
                ]
            # num_splits has shape (bs+1,) — always update for the current bs.
            self.cuda_graph_num_splits_view = self.cuda_graph_num_splits[: bs + 1]

            self.cuda_graph_mla_metadata[:actual_num_sm_parts].copy_(mla_metadata)
            self.cuda_graph_num_splits[: bs + 1].copy_(num_splits)

            cache_seqlens = None
            if forward_mode.is_draft_extend_v2():
                # The graph kernel binds this static buffer; refresh per replay.
                self.cuda_graph_draft_extend_cache_seqlens[:bs].copy_(seq_lens)
                cache_seqlens = self.cuda_graph_draft_extend_cache_seqlens[:bs]

            self.forward_metadata = FlashMLADecodeMetadata(
                self.cuda_graph_mla_metadata_view,
                self.cuda_graph_num_splits_view,
                self.cuda_graph_kv_indices[:bs, :max_seqlen_pad],
                cache_seqlens,
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        cache_loc = forward_batch.out_cache_loc

        if k is not None:
            assert v is not None
            if save_kv_cache:
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    cache_loc,
                    k,
                    v,
                )
        bs = forward_batch.batch_size
        k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)

        reshape_q = q.view(bs, -1, layer.tp_q_head_num, layer.head_dim)
        if self.is_fp8_kvcache:
            assert (
                self.dcp_world_size == 1
            ), "FlashMLA does not support DCP for FP8 kv cache"
            if layer.k_scale is not None:
                q_scale = layer.k_scale
                descale_q = layer.k_scale.reshape(1)
                descale_k = layer.k_scale.reshape(1)
            else:
                q_scale = torch.ones((1,), dtype=torch.float32, device=reshape_q.device)
                descale_q = torch.ones(
                    (1,), dtype=torch.float32, device=reshape_q.device
                )
                descale_k = torch.ones(
                    (1,), dtype=torch.float32, device=reshape_q.device
                )

            q_shape = reshape_q.shape
            reshape_q_2d = reshape_q.reshape(-1, q_shape[-1])
            reshape_q_fp8_2d, _ = scaled_fp8_quant(reshape_q_2d, q_scale)
            reshape_q_fp8 = reshape_q_fp8_2d.reshape(q_shape)
            o, _ = flash_mla_with_kvcache(
                q=reshape_q_fp8,
                k_cache=k_cache.view(-1, PAGE_SIZE, 1, self.kv_cache_dim),
                block_table=self.forward_metadata.block_kv_indices[:bs],
                cache_seqlens=forward_batch.seq_lens.to(torch.int32),
                head_dim_v=self.kv_lora_rank,
                tile_scheduler_metadata=self.forward_metadata.flashmla_metadata,
                num_splits=self.forward_metadata.num_splits,
                softmax_scale=layer.scaling,
                causal=True,
                descale_q=descale_q,
                descale_k=descale_k,
            )

            return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)
        else:
            # todo: need check all causal True or False?
            o, lse = flash_mla_with_kvcache(
                q=reshape_q,
                k_cache=k_cache.view(-1, PAGE_SIZE, 1, self.kv_cache_dim),
                block_table=self.forward_metadata.block_kv_indices[:bs],
                cache_seqlens=forward_batch.seq_lens.to(torch.int32),
                head_dim_v=self.kv_lora_rank,
                tile_scheduler_metadata=self.forward_metadata.flashmla_metadata,
                num_splits=self.forward_metadata.num_splits,
                softmax_scale=layer.scaling,
                causal=True,
            )
            o = o.view(-1, layer.tp_q_head_num * layer.v_head_dim)
            # TODO uniform output for forward_decode and forward_extend to
            # return tuple instead of single output
            # decode context parallel needs lse to correct attn_output via online softmax
            if get_parallel().dcp_enabled:
                return o, lse
            return o

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        if forward_batch.forward_mode == ForwardMode.EXTEND:
            return super().forward_extend(q, k, v, layer, forward_batch, save_kv_cache)
        else:
            # target_verify / draft_extend_v2: fixed-q decode-style kernel.
            cache_loc = forward_batch.out_cache_loc

            if k is not None:
                assert v is not None
                if save_kv_cache:
                    self.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

            bs = forward_batch.batch_size
            k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)

            unpad_args = None
            if forward_batch.forward_mode.is_draft_extend_v2():
                window = self.num_draft_tokens
                q_3d = q.view(-1, layer.tp_q_head_num, layer.head_dim)
                total_tokens = q_3d.shape[0]
                if total_tokens == bs * window:
                    # Graph-shaped batch: q already padded to the window.
                    reshape_q = q_3d.view(bs, window, *q_3d.shape[1:])
                else:
                    # Ragged accept lens: pad q to the window; padded rows
                    # trail the real tokens causally and are dropped below.
                    seq_lens_q = forward_batch.extend_seq_lens
                    cu_seqlens_q = torch.nn.functional.pad(
                        torch.cumsum(seq_lens_q, dim=0, dtype=torch.int32), (1, 0)
                    )
                    padded_q = torch.zeros(
                        (bs, window, layer.tp_q_head_num, layer.head_dim),
                        dtype=q_3d.dtype,
                        device=q_3d.device,
                    )
                    reshape_q = pad_draft_extend_query_triton(
                        q_3d, padded_q, seq_lens_q, cu_seqlens_q
                    )
                    unpad_args = (cu_seqlens_q, seq_lens_q, total_tokens)
                cache_seqlens = self.forward_metadata.cache_seqlens
            else:
                reshape_q = q.view(bs, -1, layer.tp_q_head_num, layer.head_dim)
                cache_seqlens = (
                    forward_batch.seq_lens.to(torch.int32) + self.num_draft_tokens
                )
            if self.is_fp8_kvcache:
                if layer.k_scale is not None:
                    q_scale = layer.k_scale
                    descale_q = layer.k_scale.reshape(1)
                    descale_k = layer.k_scale.reshape(1)
                else:
                    q_scale = torch.ones(
                        (1,), dtype=torch.float32, device=reshape_q.device
                    )
                    descale_q = torch.ones(
                        (1,), dtype=torch.float32, device=reshape_q.device
                    )
                    descale_k = torch.ones(
                        (1,), dtype=torch.float32, device=reshape_q.device
                    )

                q_shape = reshape_q.shape
                reshape_q_2d = reshape_q.reshape(-1, q_shape[-1])
                reshape_q_fp8_2d, _ = scaled_fp8_quant(reshape_q_2d, q_scale)
                reshape_q_fp8 = reshape_q_fp8_2d.reshape(q_shape)
                o, _ = flash_mla_with_kvcache(
                    q=reshape_q_fp8,
                    k_cache=k_cache.view(-1, PAGE_SIZE, 1, self.kv_cache_dim),
                    block_table=self.forward_metadata.block_kv_indices[:bs],
                    cache_seqlens=cache_seqlens,
                    head_dim_v=self.kv_lora_rank,
                    tile_scheduler_metadata=self.forward_metadata.flashmla_metadata,
                    num_splits=self.forward_metadata.num_splits,
                    softmax_scale=layer.scaling,
                    causal=True,
                    descale_q=descale_q,
                    descale_k=descale_k,
                )
            else:
                o, _ = flash_mla_with_kvcache(
                    q=reshape_q,
                    k_cache=k_cache.view(-1, PAGE_SIZE, 1, self.kv_cache_dim),
                    block_table=self.forward_metadata.block_kv_indices[:bs],
                    cache_seqlens=cache_seqlens,
                    head_dim_v=self.kv_lora_rank,
                    tile_scheduler_metadata=self.forward_metadata.flashmla_metadata,
                    num_splits=self.forward_metadata.num_splits,
                    softmax_scale=layer.scaling,
                    causal=True,
                )
            if unpad_args is not None:
                cu_seqlens_q, seq_lens_q, total_tokens = unpad_args
                o = unpad_draft_extend_output_triton(
                    o, cu_seqlens_q, seq_lens_q, total_tokens
                )
            return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)


class FlashMLAMultiStepDraftBackend:
    # Read by decide_needs_cpu_seq_lens (getattr defaults missing flags to True).
    needs_cpu_seq_lens: bool = False

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        if topk > 1:
            raise ValueError(
                "Currently FlashMLA only supports topk=1 for speculative decoding"
            )
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        max_bs = model_runner.req_to_token_pool.size * self.topk
        self.kv_indptr = torch.zeros(
            (
                self.speculative_num_steps,
                max_bs + 1,
            ),
            dtype=torch.int32,
            device=model_runner.device,
        )

        self.attn_backends = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                FlashMLABackend(
                    model_runner,
                    skip_prefill=True,
                    kv_indptr_buf=self.kv_indptr[i],
                    kv_last_page_len_buf=None,
                )
            )

    def common_template(
        self,
        forward_batch: ForwardBatch,
        call_fn: Callable,
    ):
        assert forward_batch.spec_info is not None

        for i in range(self.speculative_num_steps - 1):
            call_fn(i, forward_batch)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        def call_fn(i, forward_batch):
            assert forward_batch.spec_info is not None
            self.attn_backends[i].init_forward_metadata(forward_batch)

        self.common_template(forward_batch, call_fn)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_cuda_graph_state(
                max_bs, max_num_tokens, block_kv_indices=None
            )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        from sglang.srt.model_executor.forward_batch_info import (
            ForwardMode,
            build_inner_fb_view,
        )

        inner_fb = build_inner_fb_view(
            forward_batch,
            bs=forward_batch.batch_size,
            forward_mode=ForwardMode.DECODE,
        )

        def call_fn(i, _forward_batch):
            self.attn_backends[i].init_forward_metadata_out_graph(
                inner_fb, in_capture=in_capture
            )

        self.common_template(forward_batch, call_fn)

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch) -> None:
        for attn_backend in self.attn_backends:
            attn_backend.init_forward_metadata_in_graph(forward_batch)
