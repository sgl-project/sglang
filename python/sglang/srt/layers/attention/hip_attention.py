from __future__ import annotations

import os

"""
HiP Attention Backend for SGLang
https://arxiv.org/pdf/2406.09827
"""

import logging
import time
from typing import TYPE_CHECKING, Optional, Union

import torch
import triton

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.mem_cache.hip_offload_kv_pool_mha import MHATokenToHiPOffloadKVPool

if TYPE_CHECKING:
    from hip_attn.v1_2 import HiPAttentionConfig
    from sglang.srt.speculative.spec_info import SpecInfo

    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput

logger = logging.getLogger(__name__)

try:
    from sglang.srt.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
        model_parallel_is_initialized,
        tensor_model_parallel_all_gather,
    )

    SGLANG_DIST_AVAILABLE = True
except:
    SGLANG_DIST_AVAILABLE = False


def get_local_rank():
    if SGLANG_DIST_AVAILABLE:
        return (
            get_tensor_model_parallel_rank() if model_parallel_is_initialized() else 0
        )
    else:
        return 0


from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.flashattention_backend import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
)


class HiPAttentionBackend(AttentionBackend):

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        speculative_step_id=0,
        topk=0,
        speculative_num_steps=0,
    ):
        super().__init__()

        from hip_attn.v1_2.paged_hip import PagedHiPStateful

        self.use_mla = model_runner.model_config.attention_arch == AttentionArch.MLA
        self.page_size = model_runner.page_size
        assert self.page_size == 1

        self.forward_paged_hip = PagedHiPStateful(
            max_batch_size=32,
            num_layers=model_runner.model_config.num_hidden_layers,
            num_heads=model_runner.model_config.num_attention_heads
            // model_runner.tp_size,
            head_dim=(
                model_runner.model_config.head_dim
                if not hasattr(model_runner.model_config, "v_head_dim")
                else model_runner.model_config.v_head_dim
            ),
            device=model_runner.device,
        )

        self.hip_config: HiPAttentionConfig = (
            model_runner.server_args.hip_attention_config
        )
        self.is_kv_cache_offload_enabled = (
            model_runner.server_args.enable_hip_kv_cache_offload
        )

        self.max_context_len = model_runner.model_config.context_len

        self.tp_rank = model_runner.tp_rank

        self.attention_chunk_size = model_runner.attention_chunk_size

        self.flashattention_backend = FlashAttentionBackend(
            model_runner=model_runner,
            skip_prefill=skip_prefill,
            speculative_step_id=speculative_step_id,
            topk=topk,
            speculative_num_steps=speculative_num_steps,
        )

        self._last_tick = time.time()

        self._block_table: torch.Tensor = None

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        _table = self.flashattention_backend.req_to_token.index_select(
            dim=0, index=forward_batch.req_pool_indices
        )

        if self._block_table is not None:
            self._block_table[: _table.shape[0]] = _table
        else:
            self._block_table = _table

        self.flashattention_backend.init_forward_metadata(forward_batch=forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self._block_table = torch.zeros(
            max_bs,
            (self.max_context_len + self.page_size - 1) // self.page_size + 4,
            dtype=torch.int32,
            device=self.flashattention_backend.device,
        )

        self.flashattention_backend.init_cuda_graph_state(
            max_bs=max_bs,
            max_num_tokens=max_num_tokens,
        )

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
        _table = self.flashattention_backend.req_to_token.index_select(
            dim=0, index=req_pool_indices
        )
        self._block_table[: _table.shape[0]] = _table

        self.flashattention_backend.init_forward_metadata_capture_cuda_graph(
            bs=bs,
            num_tokens=num_tokens,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            encoder_lens=encoder_lens,
            forward_mode=forward_mode,
            spec_info=spec_info,
        )

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
        out_cache_loc: torch.Tensor = None,
    ):
        _table = self.flashattention_backend.req_to_token.index_select(
            dim=0, index=req_pool_indices
        )
        self._block_table[: _table.shape[0]] = _table

        self.flashattention_backend.init_forward_metadata_replay_cuda_graph(
            bs=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_sum=seq_lens_sum,
            encoder_lens=encoder_lens,
            forward_mode=forward_mode,
            spec_info=spec_info,
            seq_lens_cpu=seq_lens_cpu,
            out_cache_loc=out_cache_loc,
        )

        # print(seq_lens)
        # cache_seqlens = seq_lens[:bs].to(torch.int32)
        # print(cache_seqlens.shape)
        # cu_seqlens_q = torch.arange(
        #     0,
        #     bs + 1,
        #     1,
        #     device=seq_lens.device,
        #     dtype=torch.int32
        # )
        # print(cu_seqlens_q.shape)
        # cu_seqlens_k = cu_seqlens_q.clone()
        # cu_seqlens_k[1:] = cache_seqlens.cumsum(-1)

        # fa3_cache_seqlens=self.flashattention_backend.forward_metadata.cache_seqlens_int32[:bs]
        # fa3_cu_seqlens_q=self.flashattention_backend.forward_metadata.cu_seqlens_q[:bs+1]
        # fa3_cu_seqlens_k=self.flashattention_backend.forward_metadata.cu_seqlens_k[:bs+1]

        # print(seq_lens[:bs], fa3_cache_seqlens, fa3_cu_seqlens_q, fa3_cu_seqlens_k)

        # assert torch.all(fa3_cache_seqlens == cache_seqlens)
        # assert torch.all(fa3_cu_seqlens_q == cu_seqlens_q)
        # assert torch.all(fa3_cu_seqlens_k == cu_seqlens_k)

    def get_cuda_graph_seq_len_fill_value(self):
        assert self.flashattention_backend.get_cuda_graph_seq_len_fill_value() == 1
        return max(1, self.max_context_len - 1)

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        sk: int = None,
    ):
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        using_chunked_sw = False
        sw_size = layer.sliding_window_size
        if layer.use_irope:
            using_chunked_sw = True
            sw_size = self.attention_chunk_size

        using_dense_prefill = os.getenv("HIP_DEBUG_USING_DENSE_PREFILL", "0") == "1"
        using_dense_prefill = using_dense_prefill and (
            layer.layer_id in self.hip_config.dense_layers
        )

        force_dense_decode = os.getenv("HIP_DEBUG_FORCE_DENSE_DECODE", "0") == "1"

        delta_attention_args = os.getenv("HIP_DELTA_ATTENTION_ARGS", "")
        delta_dense_decode = any(
            ["dense_decode" == key for key in delta_attention_args.split("-")]
        )

        is_decode = False
        need_dense_prefill = using_chunked_sw or using_dense_prefill
        need_dense_decode = using_chunked_sw or delta_dense_decode

        run_benchmark = (
            (not torch.cuda.is_current_stream_capturing())
            and os.getenv("HIP_DEBUG_BENCH", "0") == "1"
            and (get_local_rank() == 0)
        )

        if run_benchmark:
            start_event = torch.cuda.Event(True)
            end_event = torch.cuda.Event(True)
            start_event.record()

        if (need_dense_prefill and (not is_decode)) or False:
            return self.flashattention_backend.forward_extend(
                q=q,
                k=k,
                v=v,
                layer=layer,
                forward_batch=forward_batch,
                save_kv_cache=save_kv_cache,
                # For multi-head latent attention
                q_rope=q_rope,
                k_rope=k_rope,
            )
        else:
            if not self.is_kv_cache_offload_enabled:
                if k is not None:
                    assert v is not None
                    if save_kv_cache:
                        if not self.use_mla:
                            forward_batch.token_to_kv_pool.set_kv_buffer(
                                layer, cache_loc, k, v
                            )
                        else:
                            forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                                layer,
                                cache_loc,
                                k,
                                k_rope,
                            )

                if not self.use_mla:
                    k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                        layer.layer_id
                    )
                    k_chunk = k.reshape(-1, layer.tp_k_head_num, layer.head_dim)
                    v_chunk = v.reshape(-1, layer.tp_v_head_num, layer.v_head_dim)
                else:
                    kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(
                        layer.layer_id
                    )

                offload_cache = None
                offloading_metadata = None

            else:  # Offloading enabled
                assert not self.use_mla
                assert isinstance(
                    forward_batch.token_to_kv_pool, MHATokenToHiPOffloadKVPool
                )
                if k is not None:
                    assert v is not None
                    if save_kv_cache:
                        if not self.use_mla:
                            forward_batch.token_to_kv_pool.set_kv_buffer(
                                layer,
                                cache_loc,
                                k,
                                v,
                                async_copy=True,
                                push_to_gpu_cache=False,
                            )
                        else:
                            raise Exception()

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

            # use_cascade_attn = (
            #     forward_batch.forward_mode.is_target_verify() and self.topk > 1
            # )
            use_cascade_attn = False

            if not self.use_mla:
                q_reshaped = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)

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
                    block_table=self._block_table[: forward_batch.batch_size],
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
                    extend_prefix_lens_cpu=forward_batch.extend_prefix_lens_cpu,
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
                    sliding_window_sink=sk,
                    using_chunked_sliding_window=using_chunked_sw,
                    self_extend_scale=self.hip_config.self_extend_scale,
                )
            else:
                if (
                    # not global_server_args_dict["disable_chunked_prefix_cache"]
                    # and forward_batch.attn_attend_prefix_cache is not None
                    # and not forward_batch.forward_mode.is_target_verify()
                    # and not forward_batch.forward_mode.is_draft_extend()
                    not global_server_args_dict["disable_chunked_prefix_cache"]
                    # and forward_batch.attn_attend_prefix_cache is not None
                    and forward_batch.forward_mode.is_extend()
                    and not forward_batch.forward_mode.is_target_verify()
                    and not forward_batch.forward_mode.is_draft_extend()
                ):
                    # Do multi-head attention with chunked prefix cache

                    assert q.shape[0] == 1, f"{q.shape=}"
                    k_reshaped = k.reshape(1, -1, layer.tp_k_head_num, layer.head_dim)
                    v_reshaped = v.reshape(1, -1, layer.tp_v_head_num, layer.v_head_dim)

                    assert not use_cascade_attn

                    o, metadata = self.forward_paged_hip(
                        query=q,
                        sm_scale=layer.scaling,
                        batch_size=forward_batch.batch_size,
                        k=k_reshaped,
                        v=v_reshaped,
                        k_cache=None,
                        v_cache=None,
                        offload_cache=offload_cache,
                        positions=forward_batch.positions,
                        seq_lens=forward_batch.seq_lens,
                        req_to_tokens=forward_batch.req_to_token_pool.req_to_token,
                        req_pool_indices=forward_batch.req_pool_indices,
                        block_table=self._block_table[: forward_batch.batch_size],
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
                        extend_prefix_lens_cpu=forward_batch.extend_prefix_lens_cpu,
                        hip_config=self.hip_config,
                        is_kv_cache_offload_enabled=self.is_kv_cache_offload_enabled,
                        cached_metadata=None,
                        online_update_cache=(
                            forward_batch.token_to_kv_pool.is_online_cache_update_enabled()
                            if self.is_kv_cache_offload_enabled
                            else None
                        ),
                        is_decode=False,
                        offloading_metadata=offloading_metadata,
                        sliding_window_size=sw_size,
                        sliding_window_sink=sk,
                        using_chunked_sliding_window=using_chunked_sw,
                        self_extend_scale=self.hip_config.self_extend_scale,
                    )
                else:
                    # Do absorbed multi-latent attention

                    require_metadata_checkout = False
                    if forward_batch.forward_mode.is_target_verify():
                        # NOTE: this condition will be graph captured.
                        metadata = forward_batch.hip_metadata_cache_pool.get_hip_metadata_cache(
                            layer.layer_id,
                            q.shape[0],
                            forward_batch.batch_size,
                            forward_batch.hip_metadata_cached_stages,
                            block_size_q=self.hip_config.block_sparse_block_size_q,
                        )
                        require_metadata_checkout = True
                    else:
                        metadata = None

                    kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(
                        layer.layer_id
                    )
                    nope_dim = triton.next_power_of_2(kv_cache.shape[-1]) // 2
                    rope_dim = kv_cache.shape[-1] - nope_dim
                    # print(q.shape, kv_cache.shape, nope_dim, rope_dim)

                    kv_head = kv_cache.shape[-2]
                    q_head = q.shape[-2]

                    k_rope = kv_cache[..., nope_dim:]
                    c_kv = kv_cache[..., :nope_dim]
                    # k_rope_cache = k_rope.view(
                    #     -1,
                    #     self.page_size,
                    #     layer.tp_k_head_num,
                    #     layer.head_dim - layer.v_head_dim,
                    # )
                    c_kv_cache = c_kv.view(-1, self.page_size, kv_head, nope_dim)
                    if q_rope is not None:
                        q_nope = q.view(-1, q_head, nope_dim)
                        q_rope = q_rope.view(-1, q_head, rope_dim)
                    else:
                        q_all = q.contiguous().view(-1, q_head, nope_dim + rope_dim)
                        q_nope = q_all[:, :, :nope_dim]
                        q_rope = q_all[:, :, nope_dim:]

                    assert q_nope.shape[-1] == layer.rope_range[0]
                    assert (q_rope.shape[-1] + q_nope.shape[-1]) == layer.rope_range[1]
                    q_merged = torch.cat([q_nope, q_rope], dim=-1)
                    # TODO FIXME
                    # k_cache = torch.cat([c_kv_cache, k_rope_cache], dim=-1)
                    k_cache = kv_cache
                    v_cache = c_kv_cache

                    if sk is not None:
                        if forward_batch.forward_mode.is_draft_extend():
                            sw_size = 512
                            sw_sink = 128
                        else:
                            sw_sink = -1
                    else:
                        sw_sink = sk

                    # print(q_merged.shape, k_cache.shape, v_cache.shape, sw_sink, sw_size)

                    o, metadata = self.forward_paged_hip(
                        query=q_merged,
                        sm_scale=layer.scaling,
                        batch_size=forward_batch.batch_size,
                        k=None,
                        v=None,
                        k_cache=k_cache,
                        v_cache=v_cache,
                        offload_cache=offload_cache,
                        positions=forward_batch.positions,
                        seq_lens=forward_batch.seq_lens,
                        req_to_tokens=forward_batch.req_to_token_pool.req_to_token,
                        req_pool_indices=forward_batch.req_pool_indices,
                        block_table=self._block_table[: forward_batch.batch_size],
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
                        sliding_window_sink=sw_sink,
                        using_chunked_sliding_window=using_chunked_sw,
                        self_extend_scale=self.hip_config.self_extend_scale,
                    )

                    if require_metadata_checkout and (metadata is not None):
                        forward_batch.hip_metadata_cache_pool.set_hip_metadata_cache(
                            layer_id=layer.layer_id,
                            tdst=q.shape[0],
                            batch_size=forward_batch.batch_size,
                            metadata=metadata,
                            block_size_q=self.hip_config.block_sparse_block_size_q,
                            cached_stages=forward_batch.hip_metadata_cached_stages,
                        )

                        if self.is_kv_cache_offload_enabled:
                            offload_cache.handle_cache_miss(metadata)

        if run_benchmark:
            from hip_attn.v1_2.utils import capture

            end_event.record()
            end_event.synchronize()

            elapsed = start_event.elapsed_time(end_event)
            elapsed_layer = (time.time() - self._last_tick) * 1000
            self._last_tick = time.time()
            capture.report()
            print(
                f"[hip] layer {layer.layer_id} took {elapsed:.2f} ms (from last tick: {elapsed_layer:.2f} ms)"
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
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        sk: Optional[int] = None,
    ):
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        using_chunked_sw = False
        sw_size = layer.sliding_window_size
        if layer.use_irope:
            using_chunked_sw = True
            sw_size = self.attention_chunk_size

        using_dense_prefill = os.getenv("HIP_DEBUG_USING_DENSE_PREFILL", "0") == "1"
        using_dense_prefill = using_dense_prefill and (
            layer.layer_id in self.hip_config.dense_layers
        )

        force_dense_decode = os.getenv("HIP_DEBUG_FORCE_DENSE_DECODE", "0") == "1"

        delta_attention_args = os.getenv("HIP_DELTA_ATTENTION_ARGS", "")
        delta_dense_decode = any(
            ["dense_decode" == key for key in delta_attention_args.split("-")]
        )

        is_decode = False
        need_dense_prefill = using_chunked_sw or using_dense_prefill
        need_dense_decode = using_chunked_sw or delta_dense_decode or force_dense_decode

        if need_dense_decode and False:
            o = self.flashattention_backend.forward_decode(
                q=q,
                k=k,
                v=v,
                layer=layer,
                forward_batch=forward_batch,
                save_kv_cache=save_kv_cache,
                # For multi-head latent attention
                q_rope=q_rope,
                k_rope=k_rope,
            )
        else:
            if (forward_batch.hip_metadata_cache_pool is not None) and (
                not delta_dense_decode
            ):
                metadata = forward_batch.hip_metadata_cache_pool.get_hip_metadata_cache(
                    layer.layer_id,
                    q.shape[0],
                    forward_batch.batch_size,
                    forward_batch.hip_metadata_cached_stages,
                    block_size_q=self.hip_config.block_sparse_block_size_q,
                )
            else:
                metadata = None

            if not self.is_kv_cache_offload_enabled:
                if k is not None:
                    assert v is not None
                    if save_kv_cache:
                        if not self.use_mla:
                            forward_batch.token_to_kv_pool.set_kv_buffer(
                                layer, cache_loc, k, v
                            )
                        else:
                            forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                                layer,
                                cache_loc,
                                k,
                                k_rope,
                            )
                if not self.use_mla:
                    k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                        layer.layer_id
                    )
                else:
                    kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(
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
                        if not self.use_mla:
                            forward_batch.token_to_kv_pool.set_kv_buffer(
                                layer,
                                cache_loc,
                                k,
                                v,
                                async_copy=False,
                                push_to_gpu_cache=True,
                            )
                        else:
                            raise Exception()

                k_cache = v_cache = None
                offload_cache, offloading_metadata = (
                    forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
                )

            if layer.layer_id == 0:
                self.cache_seqlens = (
                    forward_batch.positions.view(forward_batch.batch_size, -1)[:, -1]
                    + 1
                ).to(torch.int32)
                self.cu_seqlens_q = torch.arange(
                    0,
                    forward_batch.batch_size + 1,
                    q.shape[0] // forward_batch.batch_size,
                    device=q.device,
                    dtype=torch.int32,
                )
                self.cu_seqlens_k = self.cu_seqlens_q.clone()
                self.cu_seqlens_k[1:] = self.cache_seqlens.cumsum(-1)

            if not self.use_mla:
                k_descale = v_descale = None
                if k_cache is not None:
                    if k_cache.dtype not in [
                        torch.float32,
                        torch.float16,
                        torch.bfloat16,
                    ]:
                        assert k_cache.dtype in (
                            torch.float8_e5m2,
                            torch.float8_e4m3fn,
                        ), k_cache.dtype
                        if layer.k_scale is not None:
                            descale_shape = (
                                forward_batch.batch_size,
                                layer.tp_k_head_num,
                            )
                            k_descale = layer.k_scale.expand(descale_shape)
                            v_descale = layer.v_scale.expand(descale_shape)
                            # q = q.to(k_cache.dtype)
                        # assert layer.k_scale is not None, "fp8 scale should be handled"

                q_reshaped = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)
                k_reshaped = k.reshape(-1, layer.tp_k_head_num, layer.head_dim)
                v_reshaped = v.reshape(-1, layer.tp_v_head_num, layer.v_head_dim)

                # fa3_cache_seqlens=self.flashattention_backend.forward_metadata.cache_seqlens_int32
                # fa3_cu_seqlens_q=self.flashattention_backend.forward_metadata.cu_seqlens_q
                # fa3_cu_seqlens_k=self.flashattention_backend.forward_metadata.cu_seqlens_k

                # assert torch.all(fa3_cache_seqlens == cache_seqlens)
                # assert torch.all(fa3_cu_seqlens_q == cu_seqlens_q)
                # assert torch.all(fa3_cu_seqlens_k == cu_seqlens_k)

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
                    block_table=self._block_table[: forward_batch.batch_size],
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
                    sliding_window_sink=sk,
                    using_chunked_sliding_window=using_chunked_sw,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    # cache_seqlens=self.flashattention_backend.forward_metadata.cache_seqlens_int32[:forward_batch.batch_size],
                    # cu_seqlens_q=self.flashattention_backend.forward_metadata.cu_seqlens_k[:forward_batch.batch_size + 1],
                    # cu_seqlens_k=self.flashattention_backend.forward_metadata.cu_seqlens_q[:forward_batch.batch_size + 1],
                    cache_seqlens=self.cache_seqlens,
                    cu_seqlens_q=self.cu_seqlens_q,
                    cu_seqlens_k=self.cu_seqlens_k,
                    self_extend_scale=self.hip_config.self_extend_scale,
                )
            else:
                if k_cache is not None:
                    if k_cache.dtype not in [
                        torch.float32,
                        torch.float16,
                        torch.bfloat16,
                    ]:
                        assert k_cache.dtype in (torch.float8_e5m2, torch.float8_e4m3fn)
                        assert layer.k_scale is not None, "fp8 scale should be handled"
                # print(q.shape, k.shape, q_rope.shape, k_rope.shape)
                # torch.Size([1, 16, 512]) torch.Size([1, 1, 512]) torch.Size([1, 16, 64]) torch.Size([1, 1, 64])

                k_rope = kv_cache[:, :, layer.v_head_dim :]
                c_kv = kv_cache[:, :, : layer.v_head_dim]
                k_rope_cache = k_rope.view(
                    -1,
                    self.page_size,
                    layer.tp_k_head_num,
                    layer.head_dim - layer.v_head_dim,
                )
                c_kv_cache = c_kv.view(
                    -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                )

                if q_rope is not None:
                    q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
                    q_rope = q_rope.view(
                        -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
                    )
                else:
                    q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
                    q_nope = q_all[:, :, : layer.v_head_dim]
                    q_rope = q_all[:, :, layer.v_head_dim :]
                max_seqlen_q = (
                    self.flashattention_backend.forward_metadata.max_seq_len_q
                )

                # print(q_rope.shape, k_rope_cache.shape, c_kv_cache.shape, q_nope.shape)
                # torch.Size([1, 16, 64]) torch.Size([320001, 1, 1, 64]) torch.Size([320001, 1, 1, 512]) torch.Size([1, 16, 512])

                assert q_nope.shape[-1] == layer.rope_range[0]
                assert (q_rope.shape[-1] + q_nope.shape[-1]) == layer.rope_range[1]
                q_merged = torch.cat([q_nope, q_rope], dim=-1)
                # TODO FIXME
                # k_cache = torch.cat([c_kv_cache, k_rope_cache], dim=-1)
                k_cache = kv_cache
                v_cache = c_kv_cache

                o, metadata = self.forward_paged_hip(
                    query=q_merged,
                    sm_scale=layer.scaling,
                    batch_size=forward_batch.batch_size,
                    k=None,
                    v=None,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    offload_cache=offload_cache,
                    positions=forward_batch.positions,
                    seq_lens=forward_batch.seq_lens,
                    req_to_tokens=forward_batch.req_to_token_pool.req_to_token,
                    req_pool_indices=forward_batch.req_pool_indices,
                    block_table=self._block_table[: forward_batch.batch_size],
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
                    sliding_window_sink=sk,
                    using_chunked_sliding_window=using_chunked_sw,
                    cache_seqlens=self.cache_seqlens,
                    cu_seqlens_q=self.cu_seqlens_q,
                    cu_seqlens_k=self.cu_seqlens_k,
                    self_extend_scale=self.hip_config.self_extend_scale,
                )

            if (
                (metadata is not None)
                and (forward_batch.hip_metadata_cache_pool is not None)
                and (not delta_dense_decode)
            ):
                forward_batch.hip_metadata_cache_pool.set_hip_metadata_cache(
                    layer_id=layer.layer_id,
                    tdst=q.shape[0],
                    batch_size=forward_batch.batch_size,
                    metadata=metadata,
                    block_size_q=self.hip_config.block_sparse_block_size_q,
                    cached_stages=forward_batch.hip_metadata_cached_stages,
                )

                if self.is_kv_cache_offload_enabled:
                    offload_cache.handle_cache_miss(metadata)

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)


class HiPAttentionMultiStepBackend:

    def __init__(
        self, model_runner: ModelRunner, topk: int, speculative_num_steps: int
    ):
        self.model_runner = model_runner
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.attn_backends = []
        for i in range(self.speculative_num_steps):
            self.attn_backends.append(
                HiPAttentionBackend(
                    model_runner,
                    speculative_step_id=i,
                    topk=self.topk,
                    speculative_num_steps=self.speculative_num_steps,
                )
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int):
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_cuda_graph_state(max_bs)

    def init_forward_metadata_capture_cuda_graph(
        self,
        forward_batch: ForwardBatch,
    ):
        assert forward_batch.spec_info is not None
        assert isinstance(forward_batch.spec_info, EagleDraftInput)

        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=forward_batch.encoder_lens,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        assert forward_batch.spec_info is not None
        assert isinstance(forward_batch.spec_info, EagleDraftInput)

        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata_replay_cuda_graph(
                bs,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                encoder_lens=forward_batch.encoder_lens,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
                out_cache_loc=forward_batch.out_cache_loc,
            )
