from __future__ import annotations

"""
Support different attention backends.
Now there are two backends: FlashInfer and Triton.
FlashInfer is faster and Triton is easier to customize.
Each backend supports two operators: extend (i.e. prefill with cached prefix) and decode.
"""

import logging
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention import AttentionBackend
from sglang.srt.mem_cache.hip_offload_kv_pool_mha import MHATokenToHiPOffloadKVPool

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from sglang.srt.speculative.spec_info import SpecInfo

logger = logging.getLogger(__name__)


class HiPRadixAttentionBackend(AttentionBackend):

    def __init__(self, model_runner: ModelRunner):
        super().__init__()

        self.hip_config: "HiPAttentionConfig" = model_runner.hip_attention_config
        self.is_offload_enabled = model_runner.server_args.enable_hip_offload

        self.max_context_len = model_runner.model_config.context_len

        from hip.models.hip_attention.gen3 import forward_paged_hip
        self.forward_paged_hip = forward_paged_hip

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
        spec_info: Optional[SpecInfo],
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

        if not self.is_offload_enabled:
            if k is not None:
                assert v is not None
                if save_kv_cache:
                    forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)
            k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            offload_cache = None

        else:  # Offloading enabled
            assert isinstance(forward_batch.token_to_kv_pool, MHATokenToHiPOffloadKVPool)
            if k is not None:
                assert v is not None
                if save_kv_cache:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, async_copy=True, push_to_gpu_cache=False
                    )
            k_cache = v_cache = None
            # offload_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            offload_cache = None

        q_reshaped = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)

        # Output tensor
        o = torch.empty_like(q_reshaped)

        start_len = 0
        decoding_reqs = []
        decoding_reqs_positions = []
        for idx_batch, seq_len in enumerate(forward_batch.extend_seq_lens_cpu):
            if seq_len == 0:  # Skip empty sequences
                decoding_reqs.append(idx_batch)
                decoding_reqs_positions.append(start_len)

            else:
                if not self.is_offload_enabled:
                    k_chunk = v_chunk = None

                else:  # Offloading enabled
                    assert isinstance(forward_batch.token_to_kv_pool, MHATokenToHiPOffloadKVPool)
                    require_validation = forward_batch.token_to_kv_pool.require_validation
                    if require_validation:
                        k_chunk, v_chunk, k_pages, v_pages = (
                            forward_batch.token_to_kv_pool.get_fetched_prefix_kv_buffer(
                                layer_id=layer.layer_id,
                                batch_id=idx_batch,
                                cache_k=k[start_len: start_len + seq_len].unsqueeze(0),
                                cache_v=v[start_len: start_len + seq_len].unsqueeze(0),
                            )
                        )
                    else:
                        k_chunk, v_chunk = (
                            forward_batch.token_to_kv_pool.get_fetched_prefix_kv_buffer(
                                layer_id=layer.layer_id,
                                batch_id=idx_batch,
                                cache_k=k[start_len: start_len + seq_len].unsqueeze(0),
                                cache_v=v[start_len: start_len + seq_len].unsqueeze(0),
                            )
                        )
                    offload_cache = k_cache = v_cache = None

                if not self.is_offload_enabled:
                    o_req, _ = self.forward_paged_hip(
                        query=q_reshaped[start_len:start_len + seq_len],
                        sm_scale=layer.scaling,
                        batch_size=1,
                        k_cache=k_cache,
                        v_cache=v_cache,
                        offload_cache=offload_cache,
                        positions=forward_batch.positions[start_len:start_len + seq_len],
                        seq_lens=forward_batch.seq_lens[idx_batch:idx_batch + 1],
                        req_to_tokens=forward_batch.req_to_token_pool.req_to_token,
                        req_pool_indices=forward_batch.req_pool_indices[idx_batch:idx_batch + 1],
                        rope_cos=layer.rope_cos,
                        rope_sin=layer.rope_sin,
                        layer_id=layer.layer_id,
                        logit_cap=layer.logit_cap,
                        orig_context_len=layer.orig_context_len,
                        max_context_len=self.max_context_len,
                        hip_config=self.hip_config,
                        k=k_chunk,
                        v=v_chunk,
                    )

                    o[start_len:start_len + seq_len] = o_req

                else:  # Offloading enabled
                    # BUG: this padding is neccesary to match non offload scenario. why?
                    pad_size = self.max_context_len
                    if k_chunk.shape[1] != pad_size:
                        k_chunk_padded = torch.zeros(
                            (
                                k_chunk.shape[0],
                                pad_size,
                                k_chunk.shape[2],
                                k_chunk.shape[3],
                            ),
                            dtype=k_chunk.dtype,
                            device=k_chunk.device,
                        )
                        k_chunk_padded[:, : k_chunk.shape[1]] = k_chunk
                        del k_chunk
                        v_chunk_padded = torch.zeros(
                            (
                                v_chunk.shape[0],
                                pad_size,
                                v_chunk.shape[2],
                                v_chunk.shape[3],
                            ),
                            dtype=v_chunk.dtype,
                            device=v_chunk.device,
                        )
                        v_chunk_padded[:, : v_chunk.shape[1]] = v_chunk
                        del v_chunk
                        k_chunk = k_chunk_padded
                        v_chunk = v_chunk_padded

                    o_req, _ = self.forward_paged_hip(
                        query=q_reshaped[start_len:start_len + seq_len],
                        sm_scale=layer.scaling,
                        batch_size=1,
                        k_cache=k_cache,
                        v_cache=v_cache,
                        offload_cache=offload_cache,
                        positions=forward_batch.positions[start_len:start_len + seq_len],
                        seq_lens=forward_batch.seq_lens[idx_batch:idx_batch + 1],
                        req_to_tokens=forward_batch.req_to_token_pool.req_to_token,
                        req_pool_indices=forward_batch.req_pool_indices[idx_batch:idx_batch + 1],
                        rope_cos=layer.rope_cos,
                        rope_sin=layer.rope_sin,
                        layer_id=layer.layer_id,
                        logit_cap=layer.logit_cap,
                        orig_context_len=layer.orig_context_len,
                        max_context_len=self.max_context_len,
                        hip_config=self.hip_config,
                        k=k_chunk,
                        v=v_chunk,
                        online_update_cache=forward_batch.token_to_kv_pool.online_update_cache,
                    )

                    if require_validation:
                        o_req_valid, _ = self.forward_paged_hip(
                            query=q_reshaped[start_len: start_len + seq_len],
                            sm_scale=layer.scaling,
                            batch_size=1,
                            k_cache=k_pages,
                            v_cache=v_pages,
                            offload_cache=None,
                            positions=forward_batch.positions[start_len:start_len + seq_len],
                            seq_lens=forward_batch.seq_lens[idx_batch:idx_batch + 1],
                            req_to_tokens=forward_batch.req_to_token_pool.req_to_token,
                            req_pool_indices=forward_batch.req_pool_indices[idx_batch:idx_batch + 1],
                            rope_cos=layer.rope_cos,
                            rope_sin=layer.rope_sin,
                            layer_id=layer.layer_id,
                            logit_cap=layer.logit_cap,
                            orig_context_len=layer.orig_context_len,
                            max_context_len=self.max_context_len,
                            hip_config=self.hip_config,
                        )

                        o_err = ((o_req - o_req_valid) ** 2).sum()
                        assert o_err < 1e-6, o_err

                    o[start_len:start_len + seq_len] = o_req

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
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        metadata = None
        if forward_batch.hip_metadata_cached_stage is not None:
            metadata = forward_batch.hip_metadata_cache_pool.get_hip_metadata_cache(
                layer.layer_id,
                q.shape[0],
                forward_batch.batch_size,
                forward_batch.hip_metadata_cached_stage,
            )

        require_validation = False
        if not self.is_offload_enabled:
            if k is not None:
                assert v is not None
                if save_kv_cache:
                    forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)
            k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            offload_cache = None
        else:
            assert isinstance(forward_batch.token_to_kv_pool, MHATokenToHiPOffloadKVPool)
            require_validation = forward_batch.token_to_kv_pool.require_validation

            if k is not None:
                assert v is not None
                if save_kv_cache:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        v,
                        async_copy=False,
                        push_to_gpu_cache=True,
                    )

            if not require_validation:
                k_cache = v_cache = None
                offload_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            else:
                offload_cache, k_cache, v_cache = (
                    forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
                )

        if not require_validation:
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
                rope_cos=layer.rope_cos,
                rope_sin=layer.rope_sin,
                layer_id=layer.layer_id,
                logit_cap=layer.logit_cap,
                orig_context_len=layer.orig_context_len,
                max_context_len=self.max_context_len,
                hip_config=self.hip_config,
                cached_metadata=metadata,
                online_update_cache=(
                    forward_batch.token_to_kv_pool.online_update_cache
                    if self.is_offload_enabled else None
                ),
            )
        else:

            def sse(a: torch.Tensor, b: torch.Tensor):
                assert a.dtype == b.dtype
                return ((a - b) ** 2).sum().item()

            err_k = sse(offload_cache.k_uvm.bank_gpu, k_cache)
            err_v = sse(offload_cache.v_uvm.bank_gpu, v_cache)

            o, metadata_new = self.forward_paged_hip(
                query=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                sm_scale=layer.scaling,
                batch_size=forward_batch.batch_size,
                k_cache=None,
                v_cache=None,
                offload_cache=offload_cache,
                # NOTE: to test uvm only
                # k_cache=offload_cache.k_uvm.bank_gpu,
                # v_cache=offload_cache.v_uvm.bank_gpu,
                # offload_cache=None,
                # NOTE: to test on gpu only
                # k_cache=k_cache,
                # v_cache=v_cache,
                # offload_cache=None,
                positions=forward_batch.positions,
                seq_lens=forward_batch.seq_lens,
                req_to_tokens=forward_batch.req_to_token_pool.req_to_token,
                req_pool_indices=forward_batch.req_pool_indices,
                rope_cos=layer.rope_cos,
                rope_sin=layer.rope_sin,
                layer_id=layer.layer_id,
                logit_cap=layer.logit_cap,
                orig_context_len=layer.orig_context_len,
                max_context_len=self.max_context_len,
                hip_config=self.hip_config,
                cached_metadata=metadata,
                online_update_cache=(
                    forward_batch.token_to_kv_pool.online_update_cache
                    if self.is_offload_enabled else None
                ),
            )

            o_valid, metadata_valid = self.forward_paged_hip(
                query=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                sm_scale=layer.scaling,
                batch_size=forward_batch.batch_size,
                k_cache=k_cache,
                v_cache=v_cache,
                offload_cache=None,
                positions=forward_batch.positions,
                seq_lens=forward_batch.seq_lens,
                req_to_tokens=forward_batch.req_to_token_pool.req_to_token,
                req_pool_indices=forward_batch.req_pool_indices,
                rope_cos=layer.rope_cos,
                rope_sin=layer.rope_sin,
                layer_id=layer.layer_id,
                logit_cap=layer.logit_cap,
                orig_context_len=layer.orig_context_len,
                max_context_len=self.max_context_len,
                hip_config=self.hip_config,
                cached_metadata=metadata,
                online_update_cache=(
                    forward_batch.token_to_kv_pool.online_update_cache
                    if self.is_offload_enabled else None
                ),
            )

            err_thresh = 1e-7

            o_sse = sse(o, o_valid)
            err_retry = -1
            err_uvm = None
            if o_sse >= err_thresh:
                indices_err = sse(metadata_new.indices, metadata_valid.indices)
                ks_err = sse(metadata_new.ks, metadata_valid.ks)
                ks_count_err = sse(metadata_new.ks_count, metadata_valid.ks_count)
                ks_start_end_err = sse(
                    metadata_new.ks_start_end, metadata_valid.ks_start_end
                )
                if (metadata_valid.stage_caches is not None) and (
                    len(metadata_valid.stage_caches) > 0
                ):
                    stage1_left_err = sse(
                        metadata_new.stage_caches[1].indices_left,
                        metadata_valid.stage_caches[1].indices_left,
                    )
                    stage1_right_err = sse(
                        metadata_new.stage_caches[1].indices_right,
                        metadata_valid.stage_caches[1].indices_right,
                    )
                    stage1_score_err = sse(
                        metadata_new.stage_caches[1].out_scores,
                        metadata_valid.stage_caches[1].out_scores,
                    )
                    stage2_left_err = sse(
                        metadata_new.stage_caches[2].indices_left,
                        metadata_valid.stage_caches[2].indices_left,
                    )
                    stage2_right_err = sse(
                        metadata_new.stage_caches[2].indices_right,
                        metadata_valid.stage_caches[2].indices_right,
                    )
                    stage2_score_err = sse(
                        metadata_new.stage_caches[2].out_scores,
                        metadata_valid.stage_caches[2].out_scores,
                    )
                else:
                    stage1_left_err = stage1_right_err = stage1_score_err = (
                        stage2_left_err
                    ) = stage2_right_err = stage2_score_err = None
                online_update = (
                    forward_batch.token_to_kv_pool.online_update_cache
                    if self.is_offload_enabled else None
                )

                o_uvm, metadata_uvm = self.forward_paged_hip(
                    query=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    sm_scale=layer.scaling,
                    batch_size=forward_batch.batch_size,
                    k_cache=offload_cache.k_uvm.bank_gpu,
                    v_cache=offload_cache.v_uvm.bank_gpu,
                    offload_cache=None,
                    positions=forward_batch.positions,
                    seq_lens=forward_batch.seq_lens,
                    req_to_tokens=forward_batch.req_to_token_pool.req_to_token,
                    req_pool_indices=forward_batch.req_pool_indices,
                    rope_cos=layer.rope_cos,
                    rope_sin=layer.rope_sin,
                    layer_id=layer.layer_id,
                    logit_cap=layer.logit_cap,
                    orig_context_len=layer.orig_context_len,
                    max_context_len=self.max_context_len,
                    hip_config=self.hip_config,
                    cached_metadata=metadata,
                    online_update_cache=(
                        forward_batch.token_to_kv_pool.online_update_cache
                        if self.is_offload_enabled else None
                    ),
                )

                offload_cache.sa_kv_cache.flush()
                offload_cache.mask_k_cache.flush()

                o_retry, metadata_retry = self.forward_paged_hip(
                    query=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    sm_scale=layer.scaling,
                    batch_size=forward_batch.batch_size,
                    k_cache=None,
                    v_cache=None,
                    offload_cache=offload_cache,
                    positions=forward_batch.positions,
                    seq_lens=forward_batch.seq_lens,
                    req_to_tokens=forward_batch.req_to_token_pool.req_to_token,
                    req_pool_indices=forward_batch.req_pool_indices,
                    rope_cos=layer.rope_cos,
                    rope_sin=layer.rope_sin,
                    layer_id=layer.layer_id,
                    logit_cap=layer.logit_cap,
                    orig_context_len=layer.orig_context_len,
                    max_context_len=self.max_context_len,
                    hip_config=self.hip_config,
                    cached_metadata=metadata,
                    online_update_cache=(
                        forward_batch.token_to_kv_pool.online_update_cache
                        if self.is_offload_enabled else None
                    ),
                )
                err_uvm = sse(o, o_uvm)
                err_retry = sse(o_valid, o_retry)

                print(o)
                print(o_valid)
                print(metadata_new.indices)
                print(metadata_valid.indices)

                assert (
                    o_sse < err_thresh
                ), f"""
sse={o_sse}
err_k (uvm_k <=> valid_k) = {err_k}
err_v (uvm_v <=> valid_v) ={err_v}
err_retry (o_valid <=> o_retry) = {err_retry}
err_uvm (o_first <=> o_uvm_retry) = {err_uvm}
indices_err={indices_err}
ks_err={ks_err}
ks_count_err={ks_count_err}
ks_start_end_err={ks_start_end_err}
stage1_left_err={stage1_left_err}
stage1_right_err={stage1_right_err}
stage1_score_err={stage1_score_err}
stage2_left_err={stage2_left_err}
stage2_right_err={stage2_right_err}
stage2_score_err={stage2_score_err}
online_update={online_update}
"""

            metadata = metadata_new

        forward_batch.hip_metadata_cache_pool.set_hip_metadata_cache(
            layer_id=layer.layer_id,
            size=q.shape[0],
            batch_size=forward_batch.batch_size,
            metadata=metadata,
        )

        if self.is_offload_enabled:
            offload_cache.handle_cache_miss(metadata)

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)
