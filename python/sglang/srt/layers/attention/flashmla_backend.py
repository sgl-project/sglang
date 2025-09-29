from __future__ import annotations

from sglang.srt.layers.attention.nsa.nsa_indexer import BaseIndexerMetadata
from sglang.srt.layers.attention.nsa.topk import fast_topk_impl

"""
Support attention backend for FlashMLA.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple, Union, override

import torch
import triton
from flash_mla import flash_mla_with_kvcache, get_mla_metadata

from sglang.srt.configs.model_config import get_nsa_index_topk, is_deepseek_nsa
from sglang.srt.layers.attention.flashinfer_mla_backend import FlashInferMLAAttnBackend
from sglang.srt.layers.attention.nsa.quant_k_cache import quantize_k_cache
from sglang.srt.layers.attention.nsa.utils import (
    NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8,
    NSA_FUSE_TOPK,
    NSA_KV_CACHE_STORE_FP8,
    compute_nsa_seqlens,
)
from sglang.srt.layers.attention.utils import create_flashmla_kv_indices_triton
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInfo


# FlashMLA only supports pagesize=64
PAGE_SIZE = 64

# FlashMLA FP8 issue: https://github.com/deepseek-ai/FlashMLA/issues/56


@dataclass
class FlashMLADecodeMetadata:
    flashmla_metadata: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    num_splits: Optional[torch.Tensor] = None
    block_kv_indices: Optional[torch.Tensor] = None

    # borrowed from nsa_backend.py, TODO cleanup
    cache_seqlens_int32: torch.Tensor = None
    real_page_table: torch.Tensor = None
    nsa_seqlens_expanded: torch.Tensor = None  # expanded, unclipped `seqlens`

    def __init__(
        self,
        flashmla_metadata: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        num_splits: Optional[torch.Tensor] = None,
        block_kv_indices: Optional[torch.Tensor] = None,
    ):
        self.flashmla_metadata = flashmla_metadata
        self.num_splits = num_splits
        self.block_kv_indices = block_kv_indices


@dataclass(frozen=True)
class FlashMLAIndexerMetadata(BaseIndexerMetadata):
    attn_metadata: FlashMLADecodeMetadata

    @override
    def get_seqlens_int32(self) -> torch.Tensor:
        return self.attn_metadata.cache_seqlens_int32

    @override
    def get_page_table_64(self) -> torch.Tensor:
        return self.attn_metadata.real_page_table

    @override
    def get_seqlens_expanded(self) -> torch.Tensor:
        return self.attn_metadata.nsa_seqlens_expanded

    @override
    def topk_transform(
        self,
        logits: torch.Tensor,
        topk: int,
    ) -> torch.Tensor:
        if not NSA_FUSE_TOPK:
            return fast_topk_impl(logits, self.get_seqlens_expanded(), topk)

        raise NotImplementedError


class FlashMLABackend(FlashInferMLAAttnBackend):
    """Flashmla attention kernels."""

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
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.num_local_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.forward_metadata: Union[FlashMLADecodeMetadata] = None
        self.kv_lora_rank = model_runner.model_config.kv_lora_rank
        self.qk_nope_head_dim = model_runner.model_config.qk_nope_head_dim
        self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
        self.v_head_dim = model_runner.model_config.v_head_dim
        self.scaling = model_runner.model_config.scaling
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.kv_cache_dim = model_runner.token_to_kv_pool.kv_cache_dim

        self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens

        self.use_nsa = is_deepseek_nsa(model_runner.model_config.hf_config)
        self.nsa_index_topk = (
            get_nsa_index_topk(model_runner.model_config.hf_config)
            if self.use_nsa
            else None
        )

        # TODO copied from nsa_backend.py, unify it
        self.real_page_size = model_runner.page_size

    # TODO copied from nsa_backend.py, unify it
    def _transform_table_1_to_real(self, page_table: torch.Tensor) -> torch.Tensor:
        page_size = self.real_page_size
        if page_size == 1:
            return page_table
        max_seqlen_k = page_table.shape[1]
        strided_indices = torch.arange(
            0, max_seqlen_k, page_size, device=page_table.device, dtype=torch.int32
        )
        return page_table[:, strided_indices] // page_size

    def init_forward_metadata(self, forward_batch: ForwardBatch):

        bs = forward_batch.batch_size
        if forward_batch.forward_mode.is_decode_or_idle():
            max_seqlen_pad = triton.cdiv(
                forward_batch.seq_lens_cpu.max().item(), PAGE_SIZE
            )
            block_kv_indices = torch.full(
                (bs, max_seqlen_pad),
                -1,
                dtype=torch.int32,
                device=forward_batch.seq_lens.device,
            )
            create_flashmla_kv_indices_triton[(bs,)](
                self.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                None,
                block_kv_indices,
                self.req_to_token.stride(0),
                max_seqlen_pad,
            )
            mla_metadata, num_splits = _get_mla_metadata_wrapped(
                cache_seqlens=forward_batch.seq_lens.to(torch.int32),
                seq_len_q=1,
                num_heads_q=self.num_q_heads,
                num_heads_k=1,
                nsa_index_topk=self.nsa_index_topk,
            )
            self.forward_metadata = FlashMLADecodeMetadata(
                mla_metadata,
                num_splits,
                block_kv_indices,
            )
            if self.use_nsa:
                # TODO copied from nsa_backend, should unify
                max_seqlen_k = int(forward_batch.seq_lens_cpu.max().item())
                cache_seqlens_int32 = forward_batch.seq_lens.to(torch.int32)
                page_table = forward_batch.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, :max_seqlen_k
                ]
                self.forward_metadata.cache_seqlens_int32 = cache_seqlens_int32
                self.forward_metadata.real_page_table = self._transform_table_1_to_real(
                    page_table
                )
                self.forward_metadata.nsa_seqlens_expanded = cache_seqlens_int32

        elif forward_batch.forward_mode.is_target_verify():
            seq_lens_cpu = forward_batch.seq_lens_cpu + self.num_draft_tokens
            seq_lens = forward_batch.seq_lens + self.num_draft_tokens

            max_seqlen_pad = triton.cdiv(seq_lens_cpu.max().item(), PAGE_SIZE)
            block_kv_indices = torch.full(
                (bs, max_seqlen_pad),
                -1,
                dtype=torch.int32,
                device=seq_lens.device,
            )
            create_flashmla_kv_indices_triton[(bs,)](
                self.req_to_token,
                forward_batch.req_pool_indices,
                seq_lens,
                None,
                block_kv_indices,
                self.req_to_token.stride(0),
                max_seqlen_pad,
            )
            mla_metadata, num_splits = _get_mla_metadata_wrapped(
                cache_seqlens=seq_lens.to(torch.int32),
                seq_len_q=self.num_draft_tokens,
                num_heads_q=self.num_q_heads,
                num_heads_k=1,
                nsa_index_topk=self.nsa_index_topk,
            )

            # Use FlashMLADecodeMetadata which has the attributes forward_extend expects
            self.forward_metadata = FlashMLADecodeMetadata(
                mla_metadata,
                num_splits,
                block_kv_indices,
            )
            if self.use_nsa:
                raise NotImplementedError
        else:
            super().init_forward_metadata(forward_batch)

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        block_kv_indices: Optional[torch.Tensor] = None,
    ):
        if block_kv_indices is None:
            cuda_graph_kv_indices = torch.full(
                (max_bs, (self.max_context_len + PAGE_SIZE) // PAGE_SIZE),
                1,
                dtype=torch.int32,
                device="cuda",
            )
        else:
            cuda_graph_kv_indices = block_kv_indices

        if self.num_draft_tokens:
            self.cuda_graph_mla_metadata, self.cuda_graph_num_splits = (
                _get_mla_metadata_wrapped(
                    cache_seqlens=torch.ones(
                        max_bs, dtype=torch.int32, device=cuda_graph_kv_indices.device
                    ),
                    seq_len_q=self.num_draft_tokens,
                    num_heads_q=self.num_q_heads,
                    num_heads_k=1,
                    nsa_index_topk=self.nsa_index_topk,
                )
            )
        else:
            self.cuda_graph_mla_metadata, self.cuda_graph_num_splits = (
                _get_mla_metadata_wrapped(
                    cache_seqlens=torch.ones(
                        max_bs, dtype=torch.int32, device=cuda_graph_kv_indices.device
                    ),
                    seq_len_q=1,
                    num_heads_q=self.num_q_heads,
                    num_heads_k=1,
                    nsa_index_topk=self.nsa_index_topk,
                )
            )
        self.cuda_graph_kv_indices = cuda_graph_kv_indices

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInfo],
    ):
        if forward_mode.is_decode_or_idle():
            max_seqlen_pad = triton.cdiv(seq_lens.max().item(), PAGE_SIZE)

            create_flashmla_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                None,
                self.cuda_graph_kv_indices,
                self.req_to_token.stride(0),
                self.cuda_graph_kv_indices.stride(0),
            )
            mla_metadata, num_splits = _get_mla_metadata_wrapped(
                cache_seqlens=seq_lens.to(torch.int32),
                seq_len_q=1,
                num_heads_q=self.num_q_heads,
                num_heads_k=1,
                nsa_index_topk=self.nsa_index_topk,
            )
            self.cuda_graph_mla_metadata.copy_(mla_metadata)
            self.cuda_graph_num_splits[: bs + 1].copy_(num_splits)
            self.forward_metadata = FlashMLADecodeMetadata(
                self.cuda_graph_mla_metadata,
                self.cuda_graph_num_splits[: bs + 1],
                self.cuda_graph_kv_indices[:bs, :max_seqlen_pad],
            )
            if self.use_nsa:
                raise NotImplementedError
        elif forward_mode.is_target_verify():
            seq_lens = seq_lens + self.num_draft_tokens
            max_seqlen_pad = triton.cdiv(seq_lens.max().item(), PAGE_SIZE)

            create_flashmla_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                None,
                self.cuda_graph_kv_indices,
                self.req_to_token.stride(0),
                self.cuda_graph_kv_indices.stride(0),
            )
            mla_metadata, num_splits = _get_mla_metadata_wrapped(
                cache_seqlens=seq_lens.to(torch.int32),
                seq_len_q=self.num_draft_tokens,
                num_heads_q=self.num_q_heads,
                num_heads_k=1,
                nsa_index_topk=self.nsa_index_topk,
            )
            self.cuda_graph_mla_metadata.copy_(mla_metadata)
            self.cuda_graph_num_splits[: bs + 1].copy_(num_splits)
            self.forward_metadata = FlashMLADecodeMetadata(
                self.cuda_graph_mla_metadata,
                self.cuda_graph_num_splits[: bs + 1],
                self.cuda_graph_kv_indices[:bs, :max_seqlen_pad],
            )
            if self.use_nsa:
                raise NotImplementedError
        else:
            super().init_forward_metadata_capture_cuda_graph(
                bs,
                num_tokens,
                req_pool_indices,
                seq_lens,
                encoder_lens,
                forward_mode,
                spec_info,
            )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInfo],
        seq_lens_cpu: Optional[torch.Tensor],
    ):

        if forward_mode.is_decode_or_idle():
            assert seq_lens_cpu is not None
            seq_lens = seq_lens[:bs]
            seq_lens_cpu = seq_lens_cpu[:bs]
            max_seqlen_pad = triton.cdiv(seq_lens_cpu.max().item(), PAGE_SIZE)
            create_flashmla_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices[:bs],
                seq_lens,
                None,
                self.cuda_graph_kv_indices,
                self.req_to_token.stride(0),
                self.cuda_graph_kv_indices.stride(0),
            )
            mla_metadata, num_splits = _get_mla_metadata_wrapped(
                cache_seqlens=seq_lens.to(torch.int32),
                seq_len_q=1,
                num_heads_q=self.num_q_heads,
                num_heads_k=1,
                nsa_index_topk=self.nsa_index_topk,
            )
            self.cuda_graph_mla_metadata.copy_(mla_metadata)
            self.cuda_graph_num_splits[: bs + 1].copy_(num_splits)
            self.forward_metadata.mla_metadata = self.cuda_graph_mla_metadata
            self.forward_metadata.num_splits = self.cuda_graph_num_splits[: bs + 1]
            self.forward_metadata.block_kv_indices = self.cuda_graph_kv_indices[
                :bs, :max_seqlen_pad
            ]
        elif forward_mode.is_target_verify():
            seq_lens = seq_lens[:bs] + self.num_draft_tokens
            seq_lens_cpu = seq_lens_cpu[:bs] + self.num_draft_tokens
            max_seqlen_pad = triton.cdiv(seq_lens_cpu.max().item(), PAGE_SIZE)
            create_flashmla_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices[:bs],
                seq_lens,
                None,
                self.cuda_graph_kv_indices,
                self.req_to_token.stride(0),
                self.cuda_graph_kv_indices.stride(0),
            )
            mla_metadata, num_splits = _get_mla_metadata_wrapped(
                cache_seqlens=seq_lens.to(torch.int32),
                seq_len_q=self.num_draft_tokens,
                num_heads_q=self.num_q_heads,
                num_heads_k=1,
                nsa_index_topk=self.nsa_index_topk,
            )
            self.cuda_graph_mla_metadata.copy_(mla_metadata)
            self.cuda_graph_num_splits[: bs + 1].copy_(num_splits)
            self.forward_metadata.mla_metadata = self.cuda_graph_mla_metadata
            self.forward_metadata.num_splits = self.cuda_graph_num_splits[: bs + 1]
            self.forward_metadata.block_kv_indices = self.cuda_graph_kv_indices[
                :bs, :max_seqlen_pad
            ]
        else:
            super().init_forward_metadata_replay_cuda_graph(
                bs,
                req_pool_indices,
                seq_lens,
                seq_lens_sum,
                encoder_lens,
                forward_mode,
                spec_info,
                seq_lens_cpu,
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
        topk_indices: Optional[torch.Tensor] = None,
    ):
        cache_loc = forward_batch.out_cache_loc

        if k is not None:
            assert v is not None
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer,
                    cache_loc,
                    k,
                    v,
                )
        bs = forward_batch.batch_size
        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        k_cache = k_cache.view(-1, PAGE_SIZE, 1, self.kv_cache_dim)

        reshape_q = q.view(bs, -1, layer.tp_q_head_num, layer.head_dim)
        if (not self.use_nsa) and self.data_type == torch.float8_e4m3fn:
            reshape_q_fp8 = reshape_q.to(torch.float8_e4m3fn)
            o, _ = flash_mla_with_kvcache(
                q=reshape_q_fp8,
                k_cache=k_cache,
                block_table=self.forward_metadata.block_kv_indices[:bs],
                cache_seqlens=forward_batch.seq_lens.to(torch.int32),
                head_dim_v=self.kv_lora_rank,  # TODO Retrieve from config.
                tile_scheduler_metadata=self.forward_metadata.flashmla_metadata,
                num_splits=self.forward_metadata.num_splits,
                softmax_scale=layer.scaling,
                causal=True,
                descale_q=torch.ones((1), dtype=torch.float32, device=reshape_q.device),
                descale_k=torch.ones((1), dtype=torch.float32, device=reshape_q.device),
            )

            return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)
        else:
            block_table = self.forward_metadata.block_kv_indices[:bs]
            cache_seqlens = forward_batch.seq_lens.to(torch.int32)

            extra_kwargs: Dict
            if self.use_nsa:
                assert topk_indices is not None
                extra_kwargs = dict(
                    indices=_compute_indices_in_kvcache(
                        block_table=block_table,
                        topk_indices=topk_indices.to(torch.int32),
                        page_size=self.page_size,
                    ),
                    # doc says it is not used, but if pass in None then error
                    block_table=block_table,
                    is_fp8_kvcache=NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8,
                )
                cache_seqlens = compute_nsa_seqlens(
                    cache_seqlens, nsa_index_topk=self.nsa_index_topk
                )
            else:
                extra_kwargs = dict(
                    block_table=block_table,
                    causal=True,
                )

            if (
                self.use_nsa
                and NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8
                and not NSA_KV_CACHE_STORE_FP8
            ):
                # inefficiently quantize the whole cache
                k_cache = quantize_k_cache(k_cache)

            # todo: need check all causal True or False?
            o, _ = flash_mla_with_kvcache(
                q=reshape_q,
                k_cache=k_cache,
                cache_seqlens=cache_seqlens,
                head_dim_v=self.kv_lora_rank,  # TODO Retrieve from config.
                tile_scheduler_metadata=self.forward_metadata.flashmla_metadata,
                num_splits=self.forward_metadata.num_splits,
                softmax_scale=layer.scaling,
                **extra_kwargs,
            )

            return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        if (
            forward_batch.forward_mode == ForwardMode.EXTEND
            or forward_batch.forward_mode == ForwardMode.DRAFT_EXTEND
        ):
            return super().forward_extend(q, k, v, layer, forward_batch, save_kv_cache)
        else:
            cache_loc = forward_batch.out_cache_loc

            if k is not None:
                assert v is not None
                if save_kv_cache:
                    forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

            bs = forward_batch.batch_size
            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)

            reshape_q = q.view(bs, -1, layer.tp_q_head_num, layer.head_dim)
            if self.data_type == torch.float8_e4m3fn:
                reshape_q_fp8 = reshape_q.to(torch.float8_e4m3fn)
                o, _ = flash_mla_with_kvcache(
                    q=reshape_q_fp8,
                    k_cache=k_cache.view(-1, PAGE_SIZE, 1, self.kv_cache_dim),
                    block_table=self.forward_metadata.block_kv_indices[:bs],
                    cache_seqlens=forward_batch.seq_lens.to(torch.int32)
                    + self.num_draft_tokens,
                    head_dim_v=self.kv_lora_rank,
                    tile_scheduler_metadata=self.forward_metadata.flashmla_metadata,
                    num_splits=self.forward_metadata.num_splits,
                    softmax_scale=layer.scaling,
                    causal=True,
                    descale_q=torch.ones(
                        (1), dtype=torch.float32, device=reshape_q.device
                    ),
                    descale_k=torch.ones(
                        (1), dtype=torch.float32, device=reshape_q.device
                    ),
                )
            else:
                o, _ = flash_mla_with_kvcache(
                    q=reshape_q,
                    k_cache=k_cache.view(-1, PAGE_SIZE, 1, self.kv_cache_dim),
                    block_table=self.forward_metadata.block_kv_indices[:bs],
                    cache_seqlens=forward_batch.seq_lens.to(torch.int32)
                    + self.num_draft_tokens,
                    head_dim_v=self.kv_lora_rank,
                    tile_scheduler_metadata=self.forward_metadata.flashmla_metadata,
                    num_splits=self.forward_metadata.num_splits,
                    softmax_scale=layer.scaling,
                    causal=True,
                )
            return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def get_indexer_metadata(
        self, layer_id: int, forward_batch: ForwardBatch
    ) -> FlashMLAIndexerMetadata:
        return FlashMLAIndexerMetadata(attn_metadata=self.forward_metadata)


# TODO: multi step kv indices optimization
class FlashMLAMultiStepDraftBackend:
    """
    Wrap multiple flashmla attention backends as one for multiple consecutive
    draft decoding steps.
    """

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
        for i in range(self.speculative_num_steps):
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
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_cuda_graph_state(
                max_bs, max_num_tokens, block_kv_indices=None
            )

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

        self.common_template(forward_batch, call_fn)

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_replay_cuda_graph(
                bs,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                seq_lens_sum=-1,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
            )

        self.common_template(forward_batch, call_fn)


def _get_mla_metadata_wrapped(
    *,
    cache_seqlens: torch.Tensor,
    seq_len_q: int,
    num_heads_q: int,
    num_heads_k: int,
    nsa_index_topk: Optional[int],
):
    if nsa_index_topk is not None:
        assert nsa_index_topk is not None
        return get_mla_metadata(
            cache_seqlens=cache_seqlens,
            # TODO doc says `num_q_tokens_per_q_seq * num_heads_q // num_heads_k`
            #      but the name looks like need seq_len_q?
            num_q_tokens_per_head_k=seq_len_q * num_heads_q // num_heads_k,
            num_heads_k=num_heads_k,
            num_heads_q=num_heads_q,
            is_fp8_kvcache=NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8,
            topk=nsa_index_topk,
        )
    else:
        assert nsa_index_topk is None
        return get_mla_metadata(
            cache_seqlens=cache_seqlens,
            num_heads_per_head_k=seq_len_q * num_heads_q // num_heads_k,
            num_heads_k=num_heads_k,
        )


# TODO speedup
def _compute_indices_in_kvcache(block_table, topk_indices, page_size):
    topk_indices_safe = topk_indices.masked_fill(topk_indices == -1, 0)

    idx0 = torch.arange(block_table.size(0), device=topk_indices_safe.device).unsqueeze(
        1
    )
    block_idx = block_table[idx0, topk_indices_safe // page_size]
    offset = topk_indices_safe % page_size
    indices_in_kvcache = block_idx * page_size + offset

    # the kernel requires invalid entry to be -1
    assert indices_in_kvcache.shape == topk_indices.shape
    indices_in_kvcache[topk_indices == -1] = -1

    # return: (batch_size, seqlen_q_ori, topk)
    indices_in_kvcache = indices_in_kvcache[:, None, :]
    return indices_in_kvcache
