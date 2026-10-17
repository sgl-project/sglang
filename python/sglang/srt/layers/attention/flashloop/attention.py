# SPDX-License-Identifier: MIT
# Copyright (c) 2026 FlashLoop contributors
"""Cached-mass attention inside SGLang's native paged execution path."""

import torch

from sglang.srt.layers.attention.flashloop.kernels import (
    correct_attention,
    dense_paged_attention,
    select_source,
)
from sglang.srt.layers.attention.flashloop.prefill import sparse_paged_prefill
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_context import get_attn_backend


class FlashLoopAttention(RadixAttention):
    def __init__(self, *args, fraction, **kwargs):
        super().__init__(*args, **kwargs)
        self.fraction = fraction
        self.loop_index = 0
        self.source = None
        self.prefill_state = None

    def forward(self, q, k, v, forward_batch, save_kv_cache=True, **kwargs):
        backend = get_attn_backend()
        if hasattr(backend.token_to_kv_pool, "views"):
            return self.quantized_forward(q, k, v, forward_batch, backend)
        if self.prefill_state is not None:
            indices, positions, starts, counts, max_count, previous_layer = (
                self.prefill_state
            )
            backend = get_attn_backend()
            pool = backend.token_to_kv_pool
            locations = forward_batch.out_cache_loc
            # Copy previous-loop KV for inactive tokens, then overwrite only
            # active rows. Storage remains BF16; physical compression is separate.
            pool.set_kv_buffer(
                self,
                locations,
                pool.get_key_buffer(previous_layer)[locations.long()],
                pool.get_value_buffer(previous_layer)[locations.long()],
            )
            query = q.reshape(-1, self.tp_q_head_num, self.head_dim)
            pool.set_kv_buffer(
                self, locations[indices], k.reshape_as(query), v.reshape_as(query)
            )
            output = sparse_paged_prefill(
                query,
                pool.get_key_buffer(self.layer_id),
                pool.get_value_buffer(self.layer_id),
                positions,
                backend.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                starts,
                counts,
                max_count,
            )
            return output.reshape(q.shape[0], -1)
        if not forward_batch.forward_mode.is_decode() or self.fraction == 1.0:
            return super().forward(q, k, v, forward_batch, save_kv_cache, **kwargs)
        backend = get_attn_backend()
        pool = backend.token_to_kv_pool
        query = q.reshape(-1, self.tp_q_head_num, self.head_dim)
        req_map, req_ids, lengths = (
            backend.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
        )
        if self.loop_index == 0:
            return super().forward(q, k, v, forward_batch, save_kv_cache, **kwargs)
        if self.loop_index == 1:
            if save_kv_cache:
                pool.set_kv_buffer(
                    self,
                    forward_batch.out_cache_loc,
                    k.reshape_as(query),
                    v.reshape_as(query),
                )
            self.source = select_source(
                query,
                pool.get_key_buffer(self.layer_id),
                pool.get_value_buffer(self.layer_id),
                req_map,
                req_ids,
                lengths,
                backend.max_context_len,
                self.fraction,
                return_output=True,
            )
            return self.source[-1].reshape(q.shape[0], -1)
        if self.source is None:
            raise RuntimeError("Late recurrence requires loop-2 attention state")
        if save_kv_cache:
            pool.set_kv_buffer(
                self,
                forward_batch.out_cache_loc,
                k.reshape_as(query),
                v.reshape_as(query),
            )
        output = correct_attention(
            query,
            pool.get_key_buffer(self.layer_id),
            pool.get_value_buffer(self.layer_id),
            req_map,
            req_ids,
            lengths,
            self.source,
        )
        return output.reshape(q.shape[0], -1)

    def quantized_forward(self, q, k, v, batch, backend):
        pool = backend.token_to_kv_pool
        storage, loop, keys, values = pool.views(self.layer_id)
        query = q.reshape(-1, self.tp_q_head_num, self.head_dim)
        k, v = k.reshape_as(query), v.reshape_as(query)
        req_map, req_ids, lengths = (
            backend.req_to_token,
            batch.req_pool_indices,
            batch.seq_lens,
        )
        if batch.forward_mode.is_extend():
            if any(batch.extend_prefix_lens_cpu):
                raise ValueError("INT4 prefill requires zero cached prefix")
            total = sum(batch.extend_seq_lens_cpu)
            row_map = torch.arange(total, device=q.device, dtype=torch.int32)
            if self.prefill_state is not None:
                indices, positions, starts, counts, max_count, _ = self.prefill_state
                row_map.fill_(-1)
                row_map[indices] = torch.arange(
                    indices.numel(), device=q.device, dtype=torch.int32
                )
            storage.write_prefill(
                loop,
                k,
                v,
                row_map,
                batch.extend_start_loc,
                req_map,
                req_ids,
                lengths,
                max(batch.extend_seq_lens_cpu),
            )
            if self.prefill_state is not None:
                output = sparse_paged_prefill(
                    query,
                    keys,
                    values,
                    positions,
                    req_map,
                    req_ids,
                    lengths,
                    starts,
                    counts,
                    max_count,
                )
                return output.reshape(q.shape[0], -1)
            return super().forward(q, k, v, batch, save_kv_cache=False)
        if not batch.forward_mode.is_decode():
            raise ValueError("Packed KV supports extend and decode only")
        storage.write_decode(loop, k, v, req_map, req_ids, lengths)
        if loop == 0 or self.fraction == 1.0:
            output = dense_paged_attention(
                query, keys, values, req_map, req_ids, lengths, backend.max_context_len
            )
        elif loop == 1:
            self.source = select_source(
                query,
                keys,
                values,
                req_map,
                req_ids,
                lengths,
                backend.max_context_len,
                self.fraction,
                return_output=True,
            )
            output = self.source[-1]
        else:
            if self.source is None:
                raise RuntimeError("Late recurrence requires loop-2 attention state")
            output = correct_attention(
                query, keys, values, req_map, req_ids, lengths, self.source
            )
        return output.reshape(q.shape[0], -1)
