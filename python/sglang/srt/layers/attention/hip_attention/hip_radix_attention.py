from __future__ import annotations

from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.distributed.parallel_state import _TP

"""
Support different attention backends.
Now there are two backends: FlashInfer and Triton.
FlashInfer is faster and Triton is easier to customize.
Each backend supports two operators: extend (i.e. prefill with cached prefix) and decode.
"""

import os
from enum import Enum, auto
from typing import TYPE_CHECKING, List, Optional
import logging

import torch
import triton
import triton.language as tl

from sglang.global_config import global_config
from sglang.srt.layers.attention import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.mem_cache.hip_offload_kv_pool_mha import MHATokenToHiPOffloadKVPool
from sglang.srt.utils import (
    get_bool_env_var,
    is_flashinfer_available,
    should_use_tensor_core,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.hip_model_runner import HiPModelRunner
    from sglang.srt.layers.attention.hip_attention.hip_config import HiPAttentionConfig

from hip.models.hip_attention.gen3.attention_extend import dual_stage_quadratic_hip_attention
from hip.models.hip_attention.gen3.attention_metadata import HiPAttentionArgs
from hip.models.hip_attention.gen3.uvm_gpu_cache import HiPOffloadCache

if is_flashinfer_available():
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
    )
    from flashinfer.cascade import merge_state


logger = logging.getLogger(__name__)


class WrapperDispatch(Enum):
    SLIDING_WINDOW = auto()
    CROSS_ATTENTION = auto()


class HiPRadixAttentionBackend(AttentionBackend):

    def __init__(self, model_runner: HiPModelRunner):
        super().__init__()

        self.hip_config: HiPAttentionConfig = model_runner.hip_attention_config

        self.decode_use_tensor_cores = should_use_tensor_core(
            kv_cache_dtype=model_runner.kv_cache_dtype,
            num_attention_heads=model_runner.model_config.num_attention_heads
            // model_runner.tp_size,
            num_kv_heads=model_runner.model_config.get_num_kv_heads(
                model_runner.tp_size
            ),
        )

        self.max_context_len = model_runner.model_config.context_len

        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        if model_runner.sliding_window_size is not None:
            self.num_wrappers = 2
            self.dispatch_reason = WrapperDispatch.SLIDING_WINDOW
        elif model_runner.model_config.is_encoder_decoder:
            self.num_wrappers = 2
            self.dispatch_reason = WrapperDispatch.CROSS_ATTENTION
        else:
            self.num_wrappers = 1
            self.dispatch_reason = None

        # Allocate buffers
        self.workspace_buffer = torch.empty(
            global_config.flashinfer_workspace_size,
            dtype=torch.uint8,
            device=model_runner.device,
        )
        max_bs = model_runner.req_to_token_pool.size
        self.kv_indptr = [
            torch.zeros((max_bs + 1,), dtype=torch.int32, device=model_runner.device)
            for _ in range(self.num_wrappers)
        ]
        self.kv_last_page_len = torch.ones(
            (max_bs,), dtype=torch.int32, device=model_runner.device
        )
        self.qo_indptr = [
            torch.zeros((max_bs + 1,), dtype=torch.int32, device=model_runner.device)
            for _ in range(self.num_wrappers)
        ]

        # Create wrappers
        # NOTE: we do not use ragged attention when there are multiple wrappers
        self.prefill_wrapper_ragged = (
            BatchPrefillWithRaggedKVCacheWrapper(self.workspace_buffer, "NHD")
            if self.num_wrappers == 1
            else None
        )

        # Two wrappers: one for sliding window attention and one for full attention.
        # Using two wrappers is unnecessary in the current PR, but are prepared for future PRs
        self.prefill_wrappers_paged = []
        self.decode_wrappers = []
        for _ in range(self.num_wrappers):
            self.prefill_wrappers_paged.append(
                BatchPrefillWithPagedKVCacheWrapper(self.workspace_buffer, "NHD")
            )
            self.decode_wrappers.append(
                BatchDecodeWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    "NHD",
                    use_tensor_cores=self.decode_use_tensor_cores,
                )
            )

        # Create indices updater
        self.indices_updater_decode = FlashInferIndicesUpdaterDecode(model_runner, self)
        self.indices_updater_prefill = FlashInferIndicesUpdaterPrefill(
            model_runner, self
        )

        # Other metadata
        self.forward_metadata = None
        self.cuda_graph_metadata = {}

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if forward_batch.forward_mode.is_decode():
            self.indices_updater_decode.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                decode_wrappers=None,
                encoder_lens=forward_batch.encoder_lens,
            )
            self.forward_metadata = (self.decode_wrappers,)
        else:
            prefix_lens = forward_batch.extend_prefix_lens

            # Some heuristics to check whether to use ragged forward
            if forward_batch.extend_num_tokens >= 4096 and self.num_wrappers == 1:
                use_ragged = True
                extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)
            else:
                use_ragged = False
                extend_no_prefix = False

            self.indices_updater_prefill.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                prefix_lens,
                use_ragged=use_ragged,
                encoder_lens=forward_batch.encoder_lens,
            )

            self.forward_metadata = (use_ragged, extend_no_prefix)

    def init_cuda_graph_state(self, max_bs: int):
        cuda_graph_kv_indices = torch.zeros(
            (max_bs * self.max_context_len,),
            dtype=torch.int32,
            device="cuda",
        )
        self.cuda_graph_kv_indices = [cuda_graph_kv_indices] + [
            cuda_graph_kv_indices.clone() for _ in range(self.num_wrappers - 1)
        ]

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: torch.Tensor = None,
    ):
        decode_wrappers = []
        for i in range(self.num_wrappers):
            decode_wrappers.append(
                BatchDecodeWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    "NHD",
                    use_cuda_graph=True,
                    use_tensor_cores=self.decode_use_tensor_cores,
                    paged_kv_indptr_buffer=self.kv_indptr[i][: bs + 1],
                    paged_kv_indices_buffer=self.cuda_graph_kv_indices[i],
                    paged_kv_last_page_len_buffer=self.kv_last_page_len[:bs],
                )
            )

        seq_lens_sum = seq_lens.sum().item()
        self.indices_updater_decode.update(
            req_pool_indices,
            seq_lens,
            seq_lens_sum,
            decode_wrappers=decode_wrappers,
            encoder_lens=encoder_lens,
        )
        self.cuda_graph_metadata[bs] = decode_wrappers
        self.forward_metadata = (decode_wrappers,)

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: torch.Tensor = None,
    ):
        self.indices_updater_decode.update(
            req_pool_indices[:bs],
            seq_lens[:bs],
            seq_lens_sum,
            decode_wrappers=self.cuda_graph_metadata[bs],
            encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
        )

    def get_cuda_graph_seq_len_fill_value(self):
        return 0

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        require_dense = (
            # layer.layer_id in self.hip_config.dense_layers or
            self.hip_config.prefill_always_dense or
            self.hip_config.force_dense or
            any(map(lambda x: x <= self.hip_config.prefill_dense_threshold, forward_batch.extend_prefix_lens_cpu))
        )

        logger.debug(f'HiP attention is used in prompting (layer {layer.layer_id})!', stacklevel=0)

        is_offload_cache = isinstance(forward_batch.token_to_kv_pool, MHATokenToHiPOffloadKVPool)

        if is_offload_cache:
            assert isinstance(forward_batch.token_to_kv_pool, MHATokenToHiPOffloadKVPool)
            if k is not None:
                assert v is not None
                if save_kv_cache:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, 
                        cache_loc, 
                        k, 
                        v, 
                        async_copy=True,
                    )
            offload_cache = k_cache = v_cache = None
            # offload_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        else:
            if k is not None:
                assert v is not None
                if save_kv_cache:
                    forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)
            k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            offload_cache = None

        q_reshaped = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)

        # Output tensor
        o = torch.empty_like(q_reshaped)

        start_len = 0
        decoding_reqs = []
        decoding_reqs_poistions = []
        for idx_batch, seq_len in enumerate(forward_batch.extend_seq_lens_cpu):
            if seq_len == 0:  # Skip empty sequences
                decoding_reqs.append(idx_batch)
                decoding_reqs_poistions.append(start_len)
            else:
                if is_offload_cache:
                    assert isinstance(forward_batch.token_to_kv_pool, MHATokenToHiPOffloadKVPool)
                    k, v = forward_batch.token_to_kv_pool.get_fetched_prefix_kv_buffer(
                        layer_id=layer.layer_id, 
                        batch_id=idx_batch,
                        cache_k=k[start_len:start_len+seq_len].unsqueeze(0),
                        cache_v=v[start_len:start_len+seq_len].unsqueeze(0),
                    )
                    k_cache = v_cache = None
                    # if layer.layer_id == 31:
                    # print(layer.layer_id, k[0,::8192,0,0])
                else:
                    k = v = None
                    # if layer.layer_id == 31:
                    # print(layer.layer_id, k_cache[1::8192,0,0])
                
                o_req, _ = self.forward_paged_hip(
                    query=q_reshaped[start_len:start_len+seq_len],
                    sm_scale=layer.scaling,
                    batch_size=1,

                    k_cache=k_cache,
                    v_cache=v_cache,
                    offload_cache=offload_cache,

                    positions=forward_batch.positions[start_len:start_len+seq_len],
                    seq_lens=forward_batch.seq_lens[idx_batch:idx_batch+1],
                    req_to_tokens=forward_batch.req_to_token_pool.req_to_token,
                    req_pool_indices=forward_batch.req_pool_indices[idx_batch:idx_batch+1],

                    layer=layer,
                    # is_dense=require_dense,
                    
                    k=k,
                    v=v,
                )
                # if layer.layer_id == 31:
                # print(o_req[::8192,0,0])
                o[start_len:start_len+seq_len] = o_req
            start_len += seq_len
        assert len(decoding_reqs) == 0

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def forward_decode(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        decode_wrapper = self.forward_metadata[0][self._get_wrapper_idx(layer)]
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        require_dense = (
            # layer.layer_id in self.hip_config.dense_layers or
            self.hip_config.decode_always_dense or
            self.hip_config.force_dense
        )

        logger.debug(f'HiP attention is used in decoding (layer {layer.layer_id})!', stacklevel=0)

        is_offload_cache = isinstance(forward_batch.token_to_kv_pool, MHATokenToHiPOffloadKVPool)

        if is_offload_cache:
            assert isinstance(forward_batch.token_to_kv_pool, MHATokenToHiPOffloadKVPool)
            if k is not None:
                assert v is not None
                if save_kv_cache:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, async_copy=False
                    )
            k_cache = v_cache = None
            offload_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        else:
            if k is not None:
                assert v is not None
                if save_kv_cache:
                    forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)
            k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            offload_cache = None

        metadata = None
        if forward_batch.hip_use_cached_mask:
            metadata = forward_batch.hip_metadata_cache_pool.get_hip_metadata_cache(
                layer.layer_id, q.shape[0], forward_batch.batch_size)

        o, metadata = self.forward_paged_hip(
            query=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            sm_scale=layer.scaling,
            batch_size=forward_batch.batch_size,
            
            k_cache=k_cache,
            v_cache=v_cache,
            offload_cache=offload_cache,

            positions=forward_batch.positions,
            seq_lens=forward_batch.seq_lens,
            req_to_tokens=forward_batch.req_to_token_pool.req_to_token,
            req_pool_indices=forward_batch.req_pool_indices,

            layer=layer,
            cached_metadata=metadata,
            # is_dense=require_dense,
        )

        forward_batch.hip_metadata_cache_pool.set_hip_metadata_cache(
            layer.layer_id, q.shape[0], forward_batch.batch_size, metadata)

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def forward_paged_hip(
        self,
        query: torch.Tensor,
        sm_scale: float,
        batch_size: int,
        
        k_cache: Optional[torch.Tensor],
        v_cache: Optional[torch.Tensor],
        offload_cache: Optional[HiPOffloadCache],

        positions: torch.Tensor,
        seq_lens: torch.Tensor,
        req_to_tokens: torch.Tensor,
        req_pool_indices: torch.Tensor,

        layer: RadixAttention,

        cached_metadata=None,
        # is_dense: bool = False,

        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, "HiPAttentionOutputMetadata"]:
        is_dense = layer.layer_id in self.hip_config.dense_layers
        
        if len(self.hip_config.layers) == 2:
            layer_config = self.hip_config.layers[0 if is_dense else 1]
        else:
            layer_config = self.hip_config.layers[layer.layer_id]

        N, num_heads, hidden_dims = query.shape
        dst_seq_len = N // batch_size

        query = query.view(batch_size, dst_seq_len, num_heads, hidden_dims)

        if k_cache is not None:
            N_PAGE, num_heads_kv, hidden_dims_kv = k_cache.shape
            assert v_cache.shape == k_cache.shape
            assert hidden_dims_kv == hidden_dims

            k_cache = k_cache.view(N_PAGE, 1, num_heads_kv, hidden_dims)
            v_cache = v_cache.view(N_PAGE, 1, num_heads_kv, hidden_dims)

        block_table = req_to_tokens.index_select(dim=0, index=req_pool_indices)

        BLOCK_TABLE_BSZ, MODEL_SEQ_LEN = block_table.shape
        assert batch_size == BLOCK_TABLE_BSZ

        # NOTE(heejun): the whole point to need to find gemma is large size of hidden size
        # FIXME: find better way to detect Gemma
        if k_cache is not None:
            hidden_size = k_cache.shape[-1]
        elif k is not None:
            hidden_size = k.shape[-1]
        elif offload_cache is not None:
            hidden_size = offload_cache.k_uvm.bank_cpu.shape[-1]
        else:
            raise Exception()
        is_gemma = hidden_size > 128

        args = HiPAttentionArgs(
            k_cache=k_cache.view(torch.uint8) if isinstance(k_cache, torch.Tensor) and k_cache.dtype == torch.float8_e5m2 else k_cache,
            v_cache=v_cache.view(torch.uint8) if isinstance(k_cache, torch.Tensor) and v_cache.dtype == torch.float8_e5m2 else v_cache,
            offload_cache=offload_cache,
            block_table=block_table,
            cache_seq_lens=seq_lens,
            position_ids=positions.view(batch_size, dst_seq_len),

            block_size_k=32 if is_gemma else 64,  # BLOCK_CHUNK

            sliding_window_size=layer_config.sliding_window_size,
            sink_token_size=layer_config.sink_token_size,

            using_extend=True,
            need_apply_rope=True,
            rope_cos=layer.rope_cos,
            rope_sin=layer.rope_sin,

            logit_softcap=layer.logit_cap if layer.logit_cap != 0.0 else None,

            second_stage_k=layer_config.second_stage_k,
            stages=layer_config.stages,
            model_context_length=layer.orig_context_len,
            extend_context_length=self.max_context_len,
            block_sparse_block_size_q=self.hip_config.block_sparse_block_size_q,
            scan_extend_backend=('relative' if self.hip_config.apply_v_dot
                                 else ('streaming' if is_dense else 'relative')),
            sa_extend_backend=layer_config.sa_extend_backend,
        )
        
        context, metadata = dual_stage_quadratic_hip_attention(
            (query * sm_scale).to(query.dtype),
            k, v,
            args=args,
            cached_metadata=cached_metadata,
        )
        context = context.to(query.dtype)
        
        # print('cached meta?', cached_metadata is not None)
        # print('mask', metadata.mask_cache_statistics.compute_statistics())
        # print('sa', metadata.sa_cache_statistics.compute_statistics())

        return context.view(N, num_heads, hidden_dims), metadata

    def _get_wrapper_idx(self, layer: RadixAttention):
        if self.num_wrappers == 1:
            return 0

        if self.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
            return layer.sliding_window_size == -1
        if self.dispatch_reason == WrapperDispatch.CROSS_ATTENTION:
            return layer.is_cross_attention

        raise ValueError(f"Unknown dispatch reason: {self.dispatch_reason}")


class FlashInferIndicesUpdaterDecode:
    def __init__(self, model_runner: ModelRunner, attn_backend: AttentionBackend):
        # Constants
        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // model_runner.tp_size
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            model_runner.tp_size
        )
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.sliding_window_size = model_runner.sliding_window_size

        self.attn_backend = attn_backend

        # Buffers and wrappers
        self.kv_indptr = attn_backend.kv_indptr
        self.kv_last_page_len = attn_backend.kv_last_page_len
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.decode_wrappers = attn_backend.decode_wrappers

        # Dispatch
        if self.attn_backend.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
            self.update = self.update_sliding_window
        elif self.attn_backend.dispatch_reason == WrapperDispatch.CROSS_ATTENTION:
            self.update = self.update_cross_attention
        else:
            assert self.attn_backend.num_wrappers == 1
            self.update = self.update_single_wrapper

    def update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        decode_wrappers: List,
        encoder_lens: torch.Tensor,
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def update_single_wrapper(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        decode_wrappers: List,
        encoder_lens: torch.Tensor,
    ):
        decode_wrappers = decode_wrappers or self.decode_wrappers
        self.call_begin_forward(
            decode_wrappers[0],
            req_pool_indices,
            seq_lens,
            seq_lens_sum,
            self.kv_indptr[0],
            None,
        )

    def update_sliding_window(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        decode_wrappers: List,
        encoder_lens: torch.Tensor,
    ):
        decode_wrappers = decode_wrappers or self.decode_wrappers

        for wrapper_id in range(2):
            if wrapper_id == 0:
                # Sliding window attention
                paged_kernel_lens_tmp = torch.minimum(  # TODO: replace this with clamp
                    seq_lens,
                    torch.tensor(self.sliding_window_size + 1),
                )
                paged_kernel_lens_sum_tmp = paged_kernel_lens_tmp.sum().item()
                kv_start_idx_tmp = seq_lens - paged_kernel_lens_tmp
            else:
                # Full attention
                paged_kernel_lens_tmp = seq_lens
                paged_kernel_lens_sum_tmp = seq_lens_sum
                kv_start_idx_tmp = None

            self.call_begin_forward(
                decode_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens_tmp,
                paged_kernel_lens_sum_tmp,
                self.kv_indptr[wrapper_id],
                kv_start_idx_tmp,
            )

    def update_cross_attention(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        decode_wrappers: List,
        encoder_lens: torch.Tensor,
    ):
        decode_wrappers = decode_wrappers or self.decode_wrappers

        for wrapper_id in range(2):
            if wrapper_id == 0:
                # Normal attention
                paged_kernel_lens = seq_lens
                kv_start_idx = encoder_lens
            else:
                # Cross attention
                paged_kernel_lens = encoder_lens
                kv_start_idx = torch.zeros_like(encoder_lens)
                seq_lens_sum = encoder_lens.sum().item()

            self.call_begin_forward(
                decode_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens,
                seq_lens_sum,
                self.kv_indptr[wrapper_id],
                kv_start_idx,
            )

    def call_begin_forward(
        self,
        wrapper,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        kv_indptr: torch.Tensor,
        kv_start_idx: torch.Tensor,
    ):
        bs = len(req_pool_indices)
        kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
        kv_indptr = kv_indptr[: bs + 1]
        kv_indices = torch.empty(
            paged_kernel_lens_sum, dtype=torch.int32, device="cuda"
        )

        create_flashinfer_kv_indices_triton[(bs,)](
            self.req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            kv_indptr,
            kv_start_idx,
            kv_indices,
            self.req_to_token.shape[1],
        )

        wrapper.end_forward()
        wrapper.begin_forward(
            kv_indptr,
            kv_indices,
            self.kv_last_page_len[:bs],
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            1,
            data_type=self.data_type,
            q_data_type=self.q_data_type,
        )


class FlashInferIndicesUpdaterPrefill:
    def __init__(self, model_runner: ModelRunner, attn_backend: AttentionBackend):
        # Constants
        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // model_runner.tp_size
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            model_runner.tp_size
        )
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.sliding_window_size = model_runner.sliding_window_size

        self.attn_backend = attn_backend

        # Buffers and wrappers
        self.kv_indptr = attn_backend.kv_indptr
        self.kv_last_page_len = attn_backend.kv_last_page_len
        self.qo_indptr = attn_backend.qo_indptr
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.wrapper_ragged = attn_backend.prefill_wrapper_ragged
        self.wrappers_paged = attn_backend.prefill_wrappers_paged

        # Dispatch
        if self.attn_backend.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
            self.update = self.update_sliding_window
        elif self.attn_backend.dispatch_reason == WrapperDispatch.CROSS_ATTENTION:
            self.update = self.update_cross_attention
        else:
            assert self.attn_backend.num_wrappers == 1
            self.update = self.update_single_wrapper

    def update(
        self,
        req_pool_indices: torch.Tnesor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        use_ragged: bool,
        encoder_lens: torch.Tensor,
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def update_single_wrapper(
        self,
        req_pool_indices: torch.Tnesor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        use_ragged: bool,
        encoder_lens: torch.Tensor,
    ):
        if use_ragged:
            paged_kernel_lens = prefix_lens
            paged_kernel_lens_sum = paged_kernel_lens.sum().item()
        else:
            paged_kernel_lens = seq_lens
            paged_kernel_lens_sum = seq_lens_sum

        self.call_begin_forward(
            self.wrapper_ragged,
            self.wrappers_paged[0],
            req_pool_indices,
            paged_kernel_lens,
            paged_kernel_lens_sum,
            seq_lens,
            prefix_lens,
            None,
            self.kv_indptr[0],
            self.qo_indptr[0],
            use_ragged,
        )

    def update_sliding_window(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        use_ragged: bool,
        encoder_lens: torch.Tensor,
    ):
        for wrapper_id in range(2):
            if wrapper_id == 0:
                # window attention use paged only
                paged_kernel_lens = torch.minimum(
                    seq_lens,
                    torch.tensor(self.sliding_window_size) + seq_lens - prefix_lens,
                )
                paged_kernel_lens_sum = paged_kernel_lens.sum().item()
            else:
                # full attention
                paged_kernel_lens = seq_lens
                paged_kernel_lens_sum = seq_lens_sum

            kv_start_idx = seq_lens - paged_kernel_lens

            self.call_begin_forward(
                self.wrapper_ragged,
                self.wrappers_paged[wrapper_id],
                req_pool_indices,
                paged_kernel_lens,
                paged_kernel_lens_sum,
                seq_lens,
                prefix_lens,
                kv_start_idx,
                self.kv_indptr[wrapper_id],
                self.qo_indptr[wrapper_id],
                use_ragged,
            )

    def update_cross_attention(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        use_ragged: bool,
        encoder_lens: torch.Tensor,
    ):
        for wrapper_id in range(2):
            if wrapper_id == 0:
                # normal attention
                paged_kernel_lens = seq_lens
                kv_start_idx = encoder_lens
                paged_kernel_lens_sum = seq_lens_sum
            else:
                # cross attention
                paged_kernel_lens = encoder_lens
                kv_start_idx = torch.zeros_like(encoder_lens)
                paged_kernel_lens_sum = paged_kernel_lens.sum().item()

            self.call_begin_forward(
                self.wrapper_ragged,
                self.wrappers_paged[wrapper_id],
                req_pool_indices,
                paged_kernel_lens,
                paged_kernel_lens_sum,
                seq_lens,
                prefix_lens,
                kv_start_idx,
                self.kv_indptr[wrapper_id],
                self.qo_indptr[wrapper_id],
                use_ragged,
            )

    def call_begin_forward(
        self,
        wrapper_ragged,
        wrapper_paged,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        seq_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
        kv_start_idx: torch.Tensor,
        kv_indptr: torch.Tensor,
        qo_indptr: torch.Tensor,
        use_ragged: bool,
    ):
        bs = len(req_pool_indices)
        kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
        kv_indptr = kv_indptr[: bs + 1]
        kv_indices = torch.empty(
            paged_kernel_lens_sum, dtype=torch.int32, device="cuda"
        )
        create_flashinfer_kv_indices_triton[(bs,)](
            self.req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            kv_indptr,
            kv_start_idx,
            kv_indices,
            self.req_to_token.shape[1],
        )

        qo_indptr[1 : bs + 1] = torch.cumsum(seq_lens - prefix_lens, dim=0)
        qo_indptr = qo_indptr[: bs + 1]

        # extend part
        if use_ragged:
            wrapper_ragged.end_forward()
            wrapper_ragged.begin_forward(
                qo_indptr,
                qo_indptr,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                q_data_type=self.q_data_type,
            )

        # cached part
        wrapper_paged.end_forward()
        wrapper_paged.begin_forward(
            qo_indptr,
            kv_indptr,
            kv_indices,
            self.kv_last_page_len[:bs],
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            1,
            q_data_type=self.q_data_type,
        )


@triton.jit
def create_flashinfer_kv_indices_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_indptr,
    kv_start_idx,
    kv_indices_ptr,
    req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)

    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    kv_indices_offset = tl.load(kv_indptr + pid)

    kv_start = 0
    kv_end = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
        kv_end = kv_start
    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < kv_end - kv_start
        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + kv_start
            + offset,
            mask=mask,
        )
        tl.store(kv_indices_ptr + kv_indices_offset + offset, data, mask=mask)
