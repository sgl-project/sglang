from __future__ import annotations

"""
HiP Attention Backend for SGLang
https://arxiv.org/pdf/2406.09827
"""

import logging
from typing import TYPE_CHECKING, Optional, Union

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.mem_cache.hip_offload_kv_pool_mha import MHATokenToHiPOffloadKVPool

if TYPE_CHECKING:
    from hip_attn.v1_2 import HiPAttentionConfig

    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput
    from sglang.srt.speculative.spec_info import SpecInfo

logger = logging.getLogger(__name__)


class HiPAttentionBackend(AttentionBackend):

    def __init__(self, model_runner: ModelRunner):
        super().__init__()

        from hip_attn.v1_2 import forward_paged_hip

        self.forward_paged_hip = forward_paged_hip

        self.hip_config: HiPAttentionConfig = (
            model_runner.server_args.hip_attention_config
        )
        self.is_kv_cache_offload_enabled = (
            model_runner.server_args.enable_hip_kv_cache_offload
        )

        self.max_context_len = model_runner.model_config.context_len

        self.tp_rank = model_runner.tp_rank

        self.attention_chunk_size = model_runner.attention_chunk_size

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        pass

    def init_cuda_graph_state(self, max_bs: int):
        pass

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
        pass

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        pass

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

        if not self.is_kv_cache_offload_enabled:
            if k is not None:
                assert v is not None
                if save_kv_cache:
                    forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

            k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )
            offload_cache = None
            k_chunk = k.reshape(-1, layer.tp_k_head_num, layer.head_dim)
            v_chunk = v.reshape(-1, layer.tp_v_head_num, layer.v_head_dim)
            offloading_metadata = None

        else:  # Offloading enabled
            assert isinstance(
                forward_batch.token_to_kv_pool, MHATokenToHiPOffloadKVPool
            )
            if k is not None:
                assert v is not None
                if save_kv_cache:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, async_copy=True, push_to_gpu_cache=False
                    )

            k_cache = v_cache = offload_cache = None
            k_chunk, v_chunk, offloading_metadata = (
                forward_batch.token_to_kv_pool.get_fetched_prefix_kv_buffer(
                    layer_id=layer.layer_id,
                    extend_seq_lens=forward_batch.extend_seq_lens,
                    extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
                    cache_k=k,
                    cache_v=v,
                )
            )

        q_reshaped = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)

        using_chunked_sw = False
        sw_size = layer.sliding_window_size
        if layer.use_irope:
            using_chunked_sw = True
            sw_size = self.attention_chunk_size

        o, _ = self.forward_paged_hip(
            query=q_reshaped,
            sm_scale=layer.scaling,
            batch_size=forward_batch.batch_size,
            k=k_chunk,
            v=v_chunk,
            k_cache=k_cache,
            v_cache=v_cache,
            offload_cache=offload_cache,
            positions=forward_batch.positions,
            seq_lens=forward_batch.seq_lens,
            req_to_tokens=forward_batch.req_to_token_pool.req_to_token,
            req_pool_indices=forward_batch.req_pool_indices,
            rope_cos=layer.rope_cos,
            rope_sin=layer.rope_sin,
            rope_range=layer.rope_range,
            rope_is_neox_style=layer.rope_is_neox_style,
            layer_id=layer.layer_id,
            logit_cap=layer.logit_cap,
            orig_context_len=layer.orig_context_len,
            max_context_len=self.max_context_len,
            extend_seq_lens=forward_batch.extend_seq_lens,
            extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            hip_config=self.hip_config,
            is_kv_cache_offload_enabled=self.is_kv_cache_offload_enabled,
            online_update_cache=(
                forward_batch.token_to_kv_pool.is_online_cache_update_enabled()
                if self.is_kv_cache_offload_enabled
                else None
            ),
            is_decode=False,
            offloading_metadata=offloading_metadata,
            sliding_window_size=sw_size,
            using_chunked_sliding_window=using_chunked_sw,
        )

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_decode(
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

        metadata = forward_batch.hip_metadata_cache_pool.get_hip_metadata_cache(
            layer.layer_id,
            q.shape[0],
            forward_batch.batch_size,
            forward_batch.hip_metadata_cached_stages,
        )

        if not self.is_kv_cache_offload_enabled:
            if k is not None:
                assert v is not None
                if save_kv_cache:
                    forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)
            k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )
            offload_cache = offloading_metadata = None

        else:  # Offloading enabled
            assert isinstance(
                forward_batch.token_to_kv_pool, MHATokenToHiPOffloadKVPool
            )
            if k is not None:
                assert v is not None
                if save_kv_cache:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, async_copy=False, push_to_gpu_cache=True
                    )

            k_cache = v_cache = None
            offload_cache, offloading_metadata = (
                forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            )

        q_reshaped = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)
        k_reshaped = k.reshape(-1, layer.tp_k_head_num, layer.head_dim)
        v_reshaped = v.reshape(-1, layer.tp_v_head_num, layer.v_head_dim)

        using_chunked_sw = False
        sw_size = layer.sliding_window_size
        if layer.use_irope:
            using_chunked_sw = True
            sw_size = self.attention_chunk_size

        o, metadata = self.forward_paged_hip(
            query=q_reshaped,
            sm_scale=layer.scaling,
            batch_size=forward_batch.batch_size,
            k=k_reshaped,
            v=v_reshaped,
            k_cache=k_cache,
            v_cache=v_cache,
            offload_cache=offload_cache,
            positions=forward_batch.positions,
            seq_lens=forward_batch.seq_lens,
            req_to_tokens=forward_batch.req_to_token_pool.req_to_token,
            req_pool_indices=forward_batch.req_pool_indices,
            rope_cos=layer.rope_cos,
            rope_sin=layer.rope_sin,
            rope_range=layer.rope_range,
            rope_is_neox_style=layer.rope_is_neox_style,
            layer_id=layer.layer_id,
            logit_cap=layer.logit_cap,
            orig_context_len=layer.orig_context_len,
            max_context_len=self.max_context_len,
            hip_config=self.hip_config,
            is_kv_cache_offload_enabled=self.is_kv_cache_offload_enabled,
            cached_metadata=metadata,
            online_update_cache=(
                forward_batch.token_to_kv_pool.is_online_cache_update_enabled()
                if self.is_kv_cache_offload_enabled
                else None
            ),
            is_decode=True,
            offloading_metadata=offloading_metadata,
            sliding_window_size=sw_size,
            using_chunked_sliding_window=using_chunked_sw,
        )

        if metadata is not None:
            forward_batch.hip_metadata_cache_pool.set_hip_metadata_cache(
                layer_id=layer.layer_id,
                size=q.shape[0],
                batch_size=forward_batch.batch_size,
                metadata=metadata,
            )

            if self.is_kv_cache_offload_enabled:
                offload_cache.handle_cache_miss(metadata)

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)
