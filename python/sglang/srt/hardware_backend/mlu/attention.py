"""BF16 paged GQA attention for Cambricon MLUs.

Uses torch_mlu_ops.flash_attention for ragged prefill (including cached prefixes and
mixed batches), and the cached single-query kernel for decode.
"""

from dataclasses import dataclass

import torch
import torch_mlu_ops

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardMode


@dataclass
class ForwardMetadata:
    # Physical KV page indices, shape [batch_size, max_num_blocks_per_seq].
    block_tables: torch.Tensor

    # Total KV lengths including cached prefixes, shape [batch_size].
    seq_lens: torch.Tensor

    # Cumulative query lengths starting at zero, shape [batch_size + 1].
    # Used for extend/mixed batches; None for decode.
    cu_seqlens_q: torch.Tensor | None = None

    # Cumulative KV lengths starting at zero, shape [batch_size + 1].
    # Includes cached prefixes; used for extend/mixed batches, None for decode.
    cu_seqlens_kv: torch.Tensor | None = None

    # Maximum query length in the batch; only consumed by extend/mixed attention.
    max_seq_len_q: int = 1

    # Maximum KV length including cached prefixes for extend/mixed attention.
    # Set to the context limit for decode, where this field is not consumed.
    max_seq_len_kv: int = 0


class MLUAttnBackend(AttentionBackend):
    def __init__(self, model_runner):
        self.device = model_runner.device
        self.page_size = model_runner.page_size
        self.max_context_len = model_runner.model_config.context_len
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.kv_cache_dtype = model_runner.kv_cache_dtype
        self.forward_metadata = None
        self._graph_metadata = {}

    @staticmethod
    def _cumulative(lengths):
        out = torch.zeros(lengths.numel() + 1, dtype=torch.int32, device=lengths.device)
        torch.cumsum(lengths, dim=0, out=out[1:])
        return out

    def init_forward_metadata_out_graph(self, forward_batch, in_capture=False):
        fb = forward_batch
        if fb.forward_mode.is_idle():
            return
        tables = (
            self.req_to_token[
                fb.req_pool_indices, : self.max_context_len : self.page_size
            ]
            // self.page_size
        )
        seq_lens = fb.seq_lens.to(torch.int32)
        if fb.forward_mode.is_decode():
            bs = fb.batch_size
            # Only captured shapes need persistent addresses. Eager batches
            # must not retain one full page table for every observed size.
            meta = self._graph_metadata.get(bs)
            if meta is None:
                meta = ForwardMetadata(
                    tables,
                    seq_lens.clone() if in_capture else seq_lens,
                    max_seq_len_kv=self.max_context_len,
                )
                if in_capture:
                    self._graph_metadata[bs] = meta
            else:
                meta.block_tables.copy_(tables)
                meta.seq_lens.copy_(seq_lens)
        elif fb.forward_mode in (ForwardMode.EXTEND, ForwardMode.MIXED):
            # Mixed scheduling already represents decode rows as one-token
            # extensions. Ragged causal flash attention preserves their request
            # boundaries without inferring them from the query length.
            meta = ForwardMetadata(
                tables,
                seq_lens,
                self._cumulative(fb.extend_seq_lens),
                self._cumulative(seq_lens),
                max(fb.extend_seq_lens_cpu, default=0),
                max(fb.seq_lens_cpu.tolist(), default=0),
            )
        else:
            raise ValueError(
                f"Unsupported MLU attention forward mode: {fb.forward_mode}"
            )
        self.forward_metadata = meta

    def init_cuda_graph_state(self, max_bs, max_num_tokens):
        self._graph_metadata = {}

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    @staticmethod
    def _validate_layer(layer):
        if layer.logit_cap != 0:
            raise ValueError("MLU attention does not support nonzero logit_cap.")
        if layer.is_cross_attention:
            raise ValueError("MLU attention does not support cross-attention.")
        if layer.attn_type != AttentionType.DECODER:
            raise ValueError(
                "MLU attention only supports causal decoder self-attention."
            )

    def forward_extend(
        self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs
    ):
        self._validate_layer(layer)
        meta = self.forward_metadata
        q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        if save_kv_cache:
            self.token_to_kv_pool.set_kv_buffer(
                layer,
                forward_batch.out_cache_loc,
                k.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
                v.view(-1, layer.tp_v_head_num, layer.v_head_dim),
            )
        window = layer.sliding_window_size
        out = torch_mlu_ops.flash_attention(
            q=q,
            k=self.token_to_kv_pool.get_key_buffer(layer.layer_id),
            v=self.token_to_kv_pool.get_value_buffer(layer.layer_id),
            out=None,
            cu_seq_lens_q=meta.cu_seqlens_q,
            cu_seq_lens_kv=meta.cu_seqlens_kv,
            alibi_slope=None,
            attn_bias=None,
            max_seq_len_q=meta.max_seq_len_q,
            max_seq_len_kv=meta.max_seq_len_kv,
            softmax_scale=layer.scaling,
            is_causal=True,
            compute_dtype=torch.float32,
            return_lse=False,
            block_tables=meta.block_tables,
            out_dtype=q.dtype,
            window_size_left=window if window is not None and window > 0 else -1,
            window_size_right=0 if window is not None and window > 0 else -1,
        )
        return out.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_decode(
        self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs
    ):
        self._validate_layer(layer)
        meta = self.forward_metadata
        q = q.view(-1, 1, layer.tp_q_head_num, layer.qk_head_dim)
        if save_kv_cache:
            self.token_to_kv_pool.set_kv_buffer(
                layer,
                forward_batch.out_cache_loc,
                k.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
                v.view(-1, layer.tp_v_head_num, layer.v_head_dim),
            )
        window = layer.sliding_window_size
        out = torch_mlu_ops.single_query_cached_kv_attn(
            q=q,
            k_cache=self.token_to_kv_pool.get_key_buffer(layer.layer_id),
            v_cache=self.token_to_kv_pool.get_value_buffer(layer.layer_id),
            out=torch.empty_like(q),
            block_tables=meta.block_tables,
            context_lens=meta.seq_lens,
            k_cache_quant_scale=None,
            v_cache_quant_scale=None,
            alibi_slopes=None,
            max_context_len=self.max_context_len,
            windows_size_left=window if window is not None and window > 0 else -1,
            windows_size_right=-1,
            softmax_scale=layer.scaling,
            kv_cache_quant_bit_size=-1,
            head_size_v=-1,
            compute_dtype=torch.float32,
            q_quant_scale=None,
            out_quant_scale=None,
        )
        return out.view(-1, layer.tp_q_head_num * layer.v_head_dim)
