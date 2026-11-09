"""BF16 paged GQA attention for Cambricon MLUs.

Uses ragged flash attention for extend and cached single-query attention for
decode. Mixed batches preserve the shared scheduler's prefill-then-decode order;
a trailing one-token prefill is also valid input to the single-query kernel.
"""

from dataclasses import dataclass

import torch
import torch_mlu_ops

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_mode import ForwardMode


@dataclass
class ForwardMetadata:
    """Per-forward attention state shared across decoder layers.

    Keep one metadata class as the backend grows: extend it with optional or
    defaulted fields for new execution modes, cache layouts, attention features
    and kernel scheduling needs, rather than adding feature-specific classes.
    Prepare batch-dependent state once and reuse it across layers.
    """

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

    mixed_prefill_reqs: int = -1
    mixed_prefill_tokens: int = 0


class MLUAttnBackend(AttentionBackend):
    def __init__(self, model_runner):
        self.device = model_runner.device
        self.page_size = model_runner.page_size
        self.max_context_len = model_runner.model_config.context_len
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.kv_cache_dtype = model_runner.kv_cache_dtype
        self.forward_metadata = None

    def init_forward_metadata(self, forward_batch):
        fb = forward_batch
        if fb.forward_mode.is_idle():
            return
        # RoPE casts positions inside capture; keep the runner's input buffer.
        tables = (
            self.req_to_token[
                fb.req_pool_indices, : self.max_context_len : self.page_size
            ]
            // self.page_size
        )
        seq_lens = fb.seq_lens.to(torch.int32)
        if fb.forward_mode.is_decode():
            meta = ForwardMetadata(
                tables, seq_lens, max_seq_len_kv=self.max_context_len
            )
        elif fb.forward_mode == ForwardMode.EXTEND or fb.forward_mode.is_mixed():
            meta = ForwardMetadata(
                block_tables=tables,
                seq_lens=seq_lens,
                cu_seqlens_q=torch.nn.functional.pad(
                    torch.cumsum(fb.extend_seq_lens, dim=0, dtype=torch.int32),
                    (1, 0),
                ),
                cu_seqlens_kv=torch.nn.functional.pad(
                    torch.cumsum(seq_lens, dim=0, dtype=torch.int32), (1, 0)
                ),
                max_seq_len_q=max(fb.extend_seq_lens_cpu, default=0),
                max_seq_len_kv=max(fb.seq_lens_cpu.tolist(), default=0),
            )
        else:
            raise NotImplementedError(
                f"Unsupported MLU attention forward mode: {fb.forward_mode}"
            )
        if fb.forward_mode.is_mixed():
            # Keep a contiguous prefix; trailing one-token prefills can also
            # use the single-query decode kernel.
            lengths = fb.extend_seq_lens_cpu
            split = len(lengths)
            while split and lengths[split - 1] == 1:
                split -= 1
            meta.mixed_prefill_reqs = split
            meta.mixed_prefill_tokens = sum(lengths[:split])
        self.forward_metadata = meta

    def forward_extend(
        self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs
    ):
        meta = self.forward_metadata
        q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        if save_kv_cache:
            self.token_to_kv_pool.set_kv_buffer(
                layer,
                forward_batch.out_cache_loc,
                k.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
                v.view(-1, layer.tp_v_head_num, layer.v_head_dim),
            )
        if meta.mixed_prefill_reqs >= 0:
            out = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
            split = meta.mixed_prefill_tokens
            if meta.mixed_prefill_reqs:
                out[:split] = self._forward_prefill(
                    q[:split],
                    layer,
                    meta,
                    meta.mixed_prefill_reqs,
                    sinks=kwargs.get("sinks"),
                )
            if meta.mixed_prefill_reqs < forward_batch.batch_size:
                out[split:] = self._forward_decode(
                    q[split:],
                    layer,
                    meta,
                    meta.mixed_prefill_reqs,
                    sinks=kwargs.get("sinks"),
                )
            return out
        return self._forward_prefill(q, layer, meta, sinks=kwargs.get("sinks"))

    def _forward_prefill(self, q, layer, meta, num_reqs=None, sinks=None):
        if sinks is not None:
            raise NotImplementedError("MLU attention does not support attention sinks.")
        if layer.logit_cap != 0:
            raise NotImplementedError(
                "MLU attention does not support nonzero logit_cap."
            )
        if layer.is_cross_attention:
            raise NotImplementedError("MLU attention does not support cross-attention.")
        if layer.attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "MLU attention only supports causal decoder self-attention."
            )
        window = layer.sliding_window_size
        out = torch_mlu_ops.flash_attention(
            q=q,
            k=self.token_to_kv_pool.get_key_buffer(layer.layer_id),
            v=self.token_to_kv_pool.get_value_buffer(layer.layer_id),
            out=None,
            cu_seq_lens_q=meta.cu_seqlens_q[: num_reqs + 1]
            if num_reqs is not None
            else meta.cu_seqlens_q,
            cu_seq_lens_kv=meta.cu_seqlens_kv[: num_reqs + 1]
            if num_reqs is not None
            else meta.cu_seqlens_kv,
            alibi_slope=None,
            attn_bias=None,
            max_seq_len_q=meta.max_seq_len_q,
            max_seq_len_kv=meta.max_seq_len_kv,
            softmax_scale=layer.scaling,
            is_causal=True,
            compute_dtype=torch.float32,
            return_lse=False,
            block_tables=meta.block_tables[:num_reqs],
            out_dtype=q.dtype,
            window_size_left=window if window is not None and window > 0 else -1,
            window_size_right=0 if window is not None and window > 0 else -1,
        )
        return out.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_decode(
        self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs
    ):
        meta = self.forward_metadata
        q = q.view(-1, 1, layer.tp_q_head_num, layer.qk_head_dim)
        if save_kv_cache:
            self.token_to_kv_pool.set_kv_buffer(
                layer,
                forward_batch.out_cache_loc,
                k.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
                v.view(-1, layer.tp_v_head_num, layer.v_head_dim),
            )
        return self._forward_decode(q, layer, meta, sinks=kwargs.get("sinks"))

    def _forward_decode(self, q, layer, meta, start_req=0, sinks=None):
        if sinks is not None:
            raise NotImplementedError("MLU attention does not support attention sinks.")
        if layer.logit_cap != 0:
            raise NotImplementedError(
                "MLU attention does not support nonzero logit_cap."
            )
        if layer.is_cross_attention:
            raise NotImplementedError("MLU attention does not support cross-attention.")
        if layer.attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "MLU attention only supports causal decoder self-attention."
            )
        q = q.view(-1, 1, layer.tp_q_head_num, layer.qk_head_dim)
        window = layer.sliding_window_size
        out = torch_mlu_ops.single_query_cached_kv_attn(
            q=q,
            k_cache=self.token_to_kv_pool.get_key_buffer(layer.layer_id),
            v_cache=self.token_to_kv_pool.get_value_buffer(layer.layer_id),
            out=q.new_empty((q.shape[0], 1, layer.tp_q_head_num, layer.v_head_dim)),
            block_tables=meta.block_tables[start_req:],
            context_lens=meta.seq_lens[start_req:],
            k_cache_quant_scale=None,
            v_cache_quant_scale=None,
            alibi_slopes=None,
            max_context_len=self.max_context_len,
            windows_size_left=window if window is not None and window > 0 else -1,
            windows_size_right=-1,
            softmax_scale=layer.scaling,
            kv_cache_quant_bit_size=-1,
            head_size_v=layer.v_head_dim,
            compute_dtype=torch.float32,
            q_quant_scale=None,
            out_quant_scale=None,
        )
        return out.view(-1, layer.tp_q_head_num * layer.v_head_dim)
