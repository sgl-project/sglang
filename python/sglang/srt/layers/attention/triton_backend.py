from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.attention import AttentionBackend
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


class TritonAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        # Lazy import to avoid the initialization of cuda context
        from sglang.srt.layers.attention.triton_ops.decode_attention import (
            decode_attention_fwd,
        )
        from sglang.srt.layers.attention.triton_ops.extend_attention import (
            extend_attention_fwd,
        )

        super().__init__()

        self.decode_attention_fwd = decode_attention_fwd
        self.extend_attention_fwd = extend_attention_fwd

        if model_runner.server_args.enable_dp_attention:
            self.num_head = model_runner.model_config.num_attention_heads
        else:
            self.num_head = (
                model_runner.model_config.num_attention_heads // model_runner.tp_size
            )

        if global_server_args_dict.get("triton_attention_reduce_in_fp32", False):
            self.reduce_dtype = torch.float32
        else:
            self.reduce_dtype = torch.float16

        self.forward_metadata = None

        self.cuda_graph_max_seq_len = model_runner.model_config.context_len

        self.device = model_runner.device

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init auxiliary variables for triton attention backend."""

        if forward_batch.forward_mode.is_decode():
            start_loc = torch.zeros_like(forward_batch.seq_lens, dtype=torch.int32)
            start_loc[1:] = torch.cumsum(forward_batch.seq_lens[:-1], dim=0)

            total_num_tokens = forward_batch.seq_lens_sum
            attn_logits = torch.empty(
                (self.num_head, total_num_tokens),
                dtype=self.reduce_dtype,
                device=self.device,
            )

            max_seq_len = torch.max(forward_batch.seq_lens).item()
            max_extend_len = None
        else:
            start_loc = attn_logits = max_seq_len = None
            max_extend_len = torch.max(forward_batch.extend_seq_lens).item()

        self.forward_metadata = start_loc, attn_logits, max_seq_len, max_extend_len

    def init_cuda_graph_state(self, max_bs: int):
        self.cuda_graph_max_total_num_tokens = max_bs * self.cuda_graph_max_seq_len

        self.cuda_graph_start_loc = torch.zeros(
            (max_bs,), dtype=torch.int32, device=self.device
        )
        self.cuda_graph_attn_logits = torch.empty(
            (
                self.num_head,
                self.cuda_graph_max_total_num_tokens,
            ),
            dtype=self.reduce_dtype,
            device="cuda",
        )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens=None,
    ):
        # NOTE: encoder_lens expected to be zeros or None
        self.forward_metadata = (
            self.cuda_graph_start_loc,
            self.cuda_graph_attn_logits,
            self.cuda_graph_max_seq_len,
            None,
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens=None,
    ):
        # NOTE: encoder_lens expected to be zeros or None
        self.cuda_graph_start_loc.zero_()
        self.cuda_graph_start_loc[1:bs] = torch.cumsum(seq_lens[: bs - 1], dim=0)

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def forward_extend(
        self, q, k, v, layer: RadixAttention, forward_batch: ForwardBatch
    ):
        # TODO: reuse the buffer across layers
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        forward_batch.token_to_kv_pool.set_kv_buffer(
            layer, forward_batch.out_cache_loc, k, v
        )

        start_loc, attn_logits, max_seq_len, max_extend_len = self.forward_metadata
        self.extend_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k.contiguous(),
            v.contiguous(),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.extend_seq_lens,
            forward_batch.extend_start_loc,
            max_extend_len,
            layer.scaling,
            layer.logit_cap,
        )
        return o

    def forward_decode(
        self, q, k, v, layer: RadixAttention, forward_batch: ForwardBatch
    ):
        # During torch.compile, there is a bug in rotary_emb that causes the
        # output value to have a 3D tensor shape. This reshapes the output correctly.
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        # TODO: reuse the buffer across layers
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        start_loc, attn_logits, max_seq_len, max_extend_len = self.forward_metadata

        forward_batch.token_to_kv_pool.set_kv_buffer(
            layer, forward_batch.out_cache_loc, k, v
        )

        self.decode_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            start_loc,
            forward_batch.seq_lens,
            attn_logits,
            max_seq_len,
            layer.scaling,
            layer.logit_cap,
        )
        return o
