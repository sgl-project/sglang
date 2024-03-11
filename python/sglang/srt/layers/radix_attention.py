import torch
from sglang.srt.layers.context_flashattention_nopad import context_attention_fwd
from sglang.srt.layers.extend_attention import extend_attention_fwd
from sglang.srt.layers.token_attention import token_attention_fwd
from sglang.srt.managers.router.model_runner import ForwardMode, InputMetadata
from torch import nn


class RadixAttention(nn.Module):
    def __init__(self, num_heads, head_dim, scaling, num_kv_heads, layer_id):
        super().__init__()
        self.tp_q_head_num = num_heads
        self.tp_k_head_num = num_kv_heads
        self.tp_v_head_num = num_kv_heads
        self.head_dim = head_dim
        self.layer_id = layer_id

        from sglang.srt.managers.router.model_runner import global_server_args_dict

        if global_server_args_dict.get("enable_flashinfer", False):
            self.prefill_forward = self.prefill_forward_flashinfer
            self.extend_forward = self.prefill_forward_flashinfer
            self.decode_forward = self.decode_forward_flashinfer
        else:
            self.prefill_forward = self.prefill_forward_triton
            self.extend_forward = self.extend_forward_triton
            self.decode_forward = self.decode_forward_triton

    def prefill_forward_triton(self, q, k, v, input_metadata: InputMetadata):
        o = torch.empty_like(q)

        context_attention_fwd(
            q.view(-1, self.tp_q_head_num, self.head_dim),
            k,
            v,
            o.view(-1, self.tp_q_head_num, self.head_dim),
            input_metadata.start_loc,
            input_metadata.seq_lens,
            input_metadata.max_seq_len,
        )
        self.store_kv_cache(k, v, input_metadata)

        return o

    def extend_forward_triton(self, q, k, v, input_metadata: InputMetadata):
        o = torch.empty_like(q)
        self.store_kv_cache(k, v, input_metadata)
        extend_attention_fwd(
            q.view(-1, self.tp_q_head_num, self.head_dim),
            k.contiguous(),
            v.contiguous(),
            o.view(-1, self.tp_q_head_num, self.head_dim),
            input_metadata.token_to_kv_pool.get_key_buffer(self.layer_id),
            input_metadata.token_to_kv_pool.get_value_buffer(self.layer_id),
            input_metadata.req_to_token_pool.req_to_token,
            input_metadata.req_pool_indices,
            input_metadata.start_loc,
            input_metadata.seq_lens,
            input_metadata.prefix_lens,
            input_metadata.extend_start_loc,
            input_metadata.extend_seq_lens,
            input_metadata.max_seq_len,
            input_metadata.max_extend_len,
        )

        return o

    def decode_forward_triton(self, q, k, v, input_metadata: InputMetadata):
        o = torch.empty_like(q)
        self.store_kv_cache(k, v, input_metadata)

        token_attention_fwd(
            q.view(-1, self.tp_q_head_num, self.head_dim),
            input_metadata.token_to_kv_pool.get_key_buffer(self.layer_id),
            input_metadata.token_to_kv_pool.get_value_buffer(self.layer_id),
            o.view(-1, self.tp_q_head_num, self.head_dim),
            input_metadata.req_to_token_pool.req_to_token,
            input_metadata.req_pool_indices,
            input_metadata.start_loc,
            input_metadata.seq_lens,
            input_metadata.max_seq_len,
            input_metadata.other_kv_index,
            input_metadata.total_num_tokens,
        )

        return o

    def prefill_forward_flashinfer(self, q, k, v, input_metadata: InputMetadata):
        self.store_kv_cache(k, v, input_metadata)

        o = input_metadata.prefill_wrapper.forward(
            q.contiguous().view(-1, self.tp_q_head_num, self.head_dim),
            input_metadata.token_to_kv_pool.kv_data[self.layer_id],
        )

        return o.view(-1, self.tp_q_head_num * self.head_dim)

    def decode_forward_flashinfer(self, q, k, v, input_metadata: InputMetadata):
        self.store_kv_cache(k, v, input_metadata)

        o = input_metadata.decode_wrapper.forward(
            q.contiguous().view(-1, self.tp_q_head_num, self.head_dim),
            input_metadata.token_to_kv_pool.kv_data[self.layer_id],
        )

        return o.view(-1, self.tp_q_head_num * self.head_dim)

    def forward(self, q, k, v, input_metadata: InputMetadata):
        k = k.view(-1, self.tp_k_head_num, self.head_dim)
        v = v.view(-1, self.tp_v_head_num, self.head_dim)

        if input_metadata.forward_mode == ForwardMode.PREFILL:
            return self.prefill_forward(q, k, v, input_metadata)
        elif input_metadata.forward_mode == ForwardMode.EXTEND:
            return self.extend_forward(q, k, v, input_metadata)
        elif input_metadata.forward_mode == ForwardMode.DECODE:
            return self.decode_forward(q, k, v, input_metadata)

    def store_kv_cache(self, cache_k, cache_v, input_metadata: InputMetadata):
        key_buffer = input_metadata.token_to_kv_pool.get_key_buffer(self.layer_id)
        value_buffer = input_metadata.token_to_kv_pool.get_value_buffer(self.layer_id)
        if input_metadata.out_cache_loc is not None:
            key_buffer[input_metadata.out_cache_loc] = cache_k
            value_buffer[input_metadata.out_cache_loc] = cache_v
        elif input_metadata.out_cache_cont_start is not None:
            key_buffer[
                input_metadata.out_cache_cont_start : input_metadata.out_cache_cont_end
            ] = cache_k
            value_buffer[
                input_metadata.out_cache_cont_start : input_metadata.out_cache_cont_end
            ] = cache_v
        else:
            raise RuntimeError()
