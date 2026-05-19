from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import get_global_server_args

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


class DoubleSparseFA3Backend(FlashAttentionBackend):
    """Double Sparsity backend on top of FlashAttention 3.

    Inherits FA3 for the dense extend and dense-fallback decode paths.
    Keeps the Triton sparse-decode kernel for the top-K / heavy-token
    path — FA3 has no fused equivalent.
    """

    BLOCK_SEQ = 128

    def __init__(self, model_runner: ModelRunner):
        from sglang.srt.layers.attention.triton_ops.double_sparsity_attention import (
            flash_decode_sparse_attention_fwd,
        )

        super().__init__(model_runner)

        model_config = model_runner.model_config
        if model_config.head_dim != model_config.v_head_dim:
            raise ValueError(
                f"DoubleSparseFA3Backend requires v_head_dim == head_dim "
                f"(got {model_config.head_dim} vs {model_config.v_head_dim}). "
                f"Use --attention-backend triton."
            )
        if model_runner.server_args.kv_cache_dtype != "auto":
            raise ValueError(
                "FP8 KV cache is not supported with --enable-double-sparsity. "
                "Set --kv-cache-dtype auto."
            )

        self._flash_decode_sparse_attention_fwd = flash_decode_sparse_attention_fwd
        self.heavy_token_num = model_runner.server_args.ds_heavy_token_num
        self.sparse_decode_threshold = (
            model_runner.server_args.ds_sparse_decode_threshold
        )
        self.sorted_channels = model_runner.sorted_channels
        self.reduce_dtype = (
            torch.float32
            if get_global_server_args().triton_attention_reduce_in_fp32
            else torch.float16
        )

    @staticmethod
    def _reject_unsupported(sinks, q_rope, k_rope):
        if sinks is not None or q_rope is not None or k_rope is not None:
            raise NotImplementedError(
                "DoubleSparseFA3Backend does not support sinks / split RoPE"
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        super().init_forward_metadata(forward_batch)

        if not forward_batch.forward_mode.is_decode():
            return

        bsz = forward_batch.batch_size
        max_seq_len = forward_batch.seq_lens_cpu.max().item()
        min_seq_len = forward_batch.seq_lens_cpu.min().item()
        device = forward_batch.seq_lens.device

        self.ds_max_seq_len = max_seq_len
        self.ds_min_seq_len = min_seq_len
        self.ds_req_to_token = forward_batch.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices
        ]

        self.att_out_approx = torch.empty(
            [self.num_attention_heads, bsz, max_seq_len],
            dtype=self.reduce_dtype,
            device=device,
        )
        block_seq_num = (self.heavy_token_num + self.BLOCK_SEQ - 1) // self.BLOCK_SEQ
        self.mid_out = torch.empty(
            [bsz, self.num_attention_heads, block_seq_num, self.head_dim],
            dtype=torch.float32,
            device=device,
        )
        self.mid_o_logexpsum = torch.empty(
            [bsz, self.num_attention_heads, block_seq_num],
            dtype=torch.float32,
            device=device,
        )

    def _gather_label(self, x: torch.Tensor, layer_id: int) -> torch.Tensor:
        return torch.gather(
            x,
            2,
            self.sorted_channels[layer_id].unsqueeze(0).expand(x.shape[0], -1, -1),
        )

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
        q_rope=None,
        k_rope=None,
    ):
        self._reject_unsupported(sinks, q_rope, k_rope)
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer,
                forward_batch.out_cache_loc,
                k,
                v,
                self._gather_label(k, layer.layer_id),
            )
        return super().forward_extend(
            q, k, v, layer, forward_batch, save_kv_cache=False
        )

    def forward_decode(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
        q_rope=None,
        k_rope=None,
    ):
        self._reject_unsupported(sinks, q_rope, k_rope)

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer,
                forward_batch.out_cache_loc,
                k,
                v,
                self._gather_label(k, layer.layer_id),
            )

        if (
            self.ds_min_seq_len < self.heavy_token_num
            or self.ds_max_seq_len < self.sparse_decode_threshold
        ):
            return super().forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache=False
            )

        o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        q_3d = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        self._flash_decode_sparse_attention_fwd(
            q_3d,
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            self._gather_label(q_3d, layer.layer_id),
            forward_batch.token_to_kv_pool.get_label_buffer(layer.layer_id),
            self.ds_req_to_token,
            forward_batch.seq_lens,
            self.ds_max_seq_len,
            layer.scaling,
            layer.logit_cap,
            self.heavy_token_num,
            self.att_out_approx,
            self.mid_out,
            self.mid_o_logexpsum,
            self.BLOCK_SEQ,
        )
        return o
