from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import torch_mlu_ops
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


@dataclass
class ForwardMetadata:
    """Metadata for attention forward pass."""

    # Attention tensors (required for EXTEND/DECODE, optional for MIXED)
    cu_seqlens_q: Optional[torch.Tensor] = None
    cu_seqlens_kv: Optional[torch.Tensor] = None
    max_seq_len_q: int = 0
    max_seq_len_kv: int = 0

    # KV cache indexing (common to all modes)
    block_tables: Optional[torch.Tensor] = None
    seq_lens: Optional[torch.Tensor] = None

    # MIXED mode only: request-count boundary separating prefill and decode chunks
    prefill_bs: int = 0
    decode_bs: int = 0
    mixed_num_prefill_tokens: int = 0
    mixed_expected_tokens: int = 0
    mixed_decode_seq_lens: Optional[torch.Tensor] = None
    mixed_decode_max_seq_len_kv: int = 0

    # EXTEND mode only: whether this is a pure prefill without cached prefix
    is_uncached_prefill_only: bool = True

    # Attention compute dtype
    compute_dtype: torch.dtype = torch.float32


class MLUAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.forward_metadata = None
        self.device = model_runner.device
        self.page_size = model_runner.page_size
        self.max_context_len = model_runner.model_config.context_len
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.token_to_kv_pool = model_runner.token_to_kv_pool

    @staticmethod
    def _check_supported_attention_layer(
        layer: RadixAttention,
        k_rope: Optional[torch.Tensor],
    ) -> None:
        if k_rope is not None or layer.qk_head_dim != layer.v_head_dim:
            raise RuntimeError(
                "MLU attention backend currently supports MHA/GQA models only; "
                "MLA models are not supported."
            )

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.graph_metadata = {
            "block_tables": torch.empty(
                (max_bs, (self.max_context_len + self.page_size - 1) // self.page_size),
                dtype=torch.int32,
                device=self.device,
            ),
            "seq_lens": torch.empty(
                (max_bs,),
                dtype=torch.int32,
                device=self.device,
            ),
        }

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        if forward_batch.forward_mode.is_decode_or_idle() and hasattr(
            self, "graph_metadata"
        ):
            self._init_cuda_graph_forward_metadata(forward_batch, in_capture)
        else:
            self._init_eager_forward_metadata(forward_batch)

    def _init_cuda_graph_forward_metadata(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        bs = forward_batch.batch_size
        if not in_capture and bs in self.graph_metadata:
            metadata = self.graph_metadata[bs]
        else:
            metadata = ForwardMetadata()
            metadata.block_tables = self.graph_metadata["block_tables"][:bs, :]
            metadata.cu_seqlens_q = torch.arange(
                bs + 1, dtype=torch.int32, device=forward_batch.seq_lens.device
            )
            self.graph_metadata[bs] = metadata

        metadata.seq_lens = self.graph_metadata["seq_lens"][:bs]
        metadata.seq_lens.copy_(forward_batch.seq_lens[:bs])
        metadata.max_seq_len_q = 1
        metadata.max_seq_len_kv = int(forward_batch.seq_lens_cpu[:bs].max().item())

        max_seq_pages = (metadata.max_seq_len_kv + self.page_size - 1) // self.page_size
        metadata.block_tables.fill_(0)
        if max_seq_pages > 0:
            metadata.block_tables[:bs, :max_seq_pages].copy_(
                self.req_to_token[
                    forward_batch.req_pool_indices[:bs], : metadata.max_seq_len_kv
                ][:, :: self.page_size]
                // self.page_size
            )

        self.forward_metadata = metadata
        self.graph_mode = True

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        self._init_eager_forward_metadata(forward_batch)

    def _init_eager_forward_metadata(self, forward_batch: ForwardBatch):
        """Init metadata based on forward mode."""
        metadata = ForwardMetadata()
        self.forward_metadata = metadata
        meta = metadata
        mode = forward_batch.forward_mode

        # Common: block_tables and seq_lens for all modes
        meta.block_tables = (
            self.req_to_token[forward_batch.req_pool_indices, : self.max_context_len][
                :, :: self.page_size
            ]
            // self.page_size
        )
        meta.seq_lens = forward_batch.seq_lens.to(dtype=torch.int32)

        if mode == ForwardMode.EXTEND:
            batch_size = forward_batch.batch_size

            meta.cu_seqlens_q = torch.zeros(
                batch_size + 1, dtype=torch.int32, device=self.device
            )
            torch.cumsum(
                forward_batch.extend_seq_lens,
                dim=0,
                out=meta.cu_seqlens_q[1:],
            )
            meta.max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)

            meta.is_uncached_prefill_only = (
                forward_batch.extend_prefix_lens.sum().item() == 0
            )

            if meta.is_uncached_prefill_only:
                meta.cu_seqlens_kv = meta.cu_seqlens_q.clone()
                meta.max_seq_len_kv = meta.max_seq_len_q
            else:
                seq_lens = forward_batch.seq_lens.to(dtype=torch.int32)
                meta.cu_seqlens_kv = torch.zeros(
                    batch_size + 1, dtype=torch.int32, device=self.device
                )
                torch.cumsum(seq_lens, dim=0, out=meta.cu_seqlens_kv[1:])
                meta.max_seq_len_kv = int(seq_lens.max().item())

        elif mode == ForwardMode.DECODE:
            batch_size = forward_batch.batch_size
            meta.cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=self.device
            )
            meta.cu_seqlens_kv = torch.zeros(
                batch_size + 1, dtype=torch.int32, device=self.device
            )
            meta.cu_seqlens_kv[1:] = forward_batch.seq_lens.to(dtype=torch.int32)
            torch.cumsum(meta.cu_seqlens_kv, dim=0, out=meta.cu_seqlens_kv)
            meta.max_seq_len_q = 1
            meta.max_seq_len_kv = int(forward_batch.seq_lens.max().item())

        elif mode == ForwardMode.MIXED:
            batch_size = forward_batch.batch_size
            running_bs = getattr(forward_batch, "mix_decode_bs", 0)
            if running_bs <= 0:
                mix_running_indices = getattr(
                    forward_batch, "mix_running_indices", None
                )
                if mix_running_indices is not None:
                    running_bs = int(mix_running_indices.numel())
            if running_bs <= 0:
                raise RuntimeError(
                    "MLU mixed attention requires a positive ForwardBatch.mix_decode_bs."
                )
            prefill_bs = batch_size - running_bs
            assert prefill_bs >= 0, (
                f"Invalid mixed batch boundary: batch_size={batch_size}, "
                f"running_bs={running_bs}."
            )

            meta.prefill_bs = prefill_bs
            meta.decode_bs = running_bs
            meta.mixed_num_prefill_tokens = sum(
                forward_batch.extend_seq_lens_cpu[:prefill_bs]
            )
            meta.mixed_expected_tokens = meta.mixed_num_prefill_tokens + running_bs

            if prefill_bs > 0:
                prefill_lens = forward_batch.extend_seq_lens[
                    :prefill_bs
                ].to(dtype=torch.int32)
                meta.cu_seqlens_q = torch.zeros(
                    prefill_bs + 1, dtype=torch.int32, device=self.device
                )
                torch.cumsum(prefill_lens, dim=0, out=meta.cu_seqlens_q[1:])

                prefill_seq_lens = meta.seq_lens[:prefill_bs]
                meta.cu_seqlens_kv = torch.zeros(
                    prefill_bs + 1, dtype=torch.int32, device=self.device
                )
                torch.cumsum(
                    prefill_seq_lens, dim=0, out=meta.cu_seqlens_kv[1:]
                )

                meta.max_seq_len_q = max(
                    forward_batch.extend_seq_lens_cpu[:prefill_bs]
                )
                meta.max_seq_len_kv = int(
                    forward_batch.seq_lens_cpu[:prefill_bs].max().item()
                )

            if running_bs > 0:
                meta.mixed_decode_seq_lens = meta.seq_lens[prefill_bs:batch_size]
                meta.mixed_decode_max_seq_len_kv = int(
                    forward_batch.seq_lens_cpu[prefill_bs:batch_size].max().item()
                )

        self.forward_metadata = metadata

    def get_cuda_graph_seq_len_fill_value(self):
        return 0

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """Entry point - dispatch to appropriate forward method based on mode."""
        mode = forward_batch.forward_mode
        self._check_supported_attention_layer(layer, kwargs.get("k_rope"))
        if mode.is_idle():
            return q.new_empty(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
        elif mode.is_decode():
            return self.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        elif mode.is_mixed():
            return self.forward_mixed(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        else:
            return self.forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        """Pure prefill/extend mode - handles first token generation for new requests."""
        meta = self.forward_metadata

        # Reshape tensors
        q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        k = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
        v = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)

        # Write to KV cache before attention
        if save_kv_cache:
            self.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        if meta.is_uncached_prefill_only:
            # Direct attention without prefix KV cache
            out = torch_mlu_ops.flash_attention(
                q=q,
                k=k,
                v=v,
                out=None,
                cu_seq_lens_q=meta.cu_seqlens_q,
                cu_seq_lens_kv=meta.cu_seqlens_kv,
                alibi_slope=None,
                attn_bias=None,
                max_seq_len_q=meta.max_seq_len_q,
                max_seq_len_kv=meta.max_seq_len_kv,
                softmax_scale=layer.scaling,
                is_causal=True,
                compute_dtype=meta.compute_dtype,
                return_lse=False,
                block_tables=None,
                out_dtype=torch.bfloat16,
            )
        else:
            # Need to use KV cache with proper sequence boundaries
            seq_lens = forward_batch.seq_lens.to(dtype=torch.int32)
            cu_seqlens_kv = torch.zeros(
                forward_batch.batch_size + 1, dtype=torch.int32, device=self.device
            )
            torch.cumsum(seq_lens, dim=0, out=cu_seqlens_kv[1:])

            k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
            v_cache = self.token_to_kv_pool.get_value_buffer(layer.layer_id)

            out = torch_mlu_ops.flash_attention(
                q=q,
                k=k_cache,
                v=v_cache,
                out=None,
                cu_seq_lens_q=meta.cu_seqlens_q,
                cu_seq_lens_kv=cu_seqlens_kv,
                alibi_slope=None,
                attn_bias=None,
                max_seq_len_q=meta.max_seq_len_q,
                max_seq_len_kv=meta.max_seq_len_kv,
                softmax_scale=layer.scaling,
                is_causal=True,
                compute_dtype=meta.compute_dtype,
                return_lse=False,
                block_tables=meta.block_tables,
                out_dtype=torch.bfloat16,
            )

        return out.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        """Pure decode mode - handles continuation token generation."""
        batch_size = forward_batch.batch_size
        meta = self.forward_metadata

        # Reshape tensors
        k = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
        v = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)
        q = q.view(batch_size, -1, layer.tp_q_head_num, layer.qk_head_dim)

        if save_kv_cache:
            self.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = self.token_to_kv_pool.get_value_buffer(layer.layer_id)

        out = torch_mlu_ops.single_query_cached_kv_attn(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            out=None,
            block_tables=self.forward_metadata.block_tables,
            context_lens=self.forward_metadata.seq_lens,
            k_cache_quant_scale=None,
            v_cache_quant_scale=None,
            alibi_slopes=None,
            max_contxt_len=self.max_context_len,
            windows_size_left=-1,
            windows_size_right=-1,
            softmax_scale=layer.scaling,
            head_size_v=-1,
            compute_dtype=meta.compute_dtype,
            q_quant_scale=None,
            out_quant_scale=None,
        )

        return out.view(batch_size, -1)

    def forward_mixed(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        """MIXED mode - chunked prefill + decode mixed.

        q tensor layout: [prefill_tokens, decode_tokens]
        """
        meta = self.forward_metadata
        prefill_bs, decode_bs = meta.prefill_bs, meta.decode_bs
        batch_size = forward_batch.batch_size
        if (
            prefill_bs < 0
            or decode_bs < 0
            or prefill_bs + decode_bs != batch_size
        ):
            raise RuntimeError(
                "Invalid MLU mixed attention batch boundary: "
                f"prefill_bs={prefill_bs}, decode_bs={decode_bs}, "
                f"batch_size={batch_size}."
            )

        expected_tokens = meta.mixed_expected_tokens
        if q.shape[0] != expected_tokens:
            raise RuntimeError(
                "Invalid MLU mixed attention token boundary: "
                f"expected_tokens={expected_tokens}, q_tokens={q.shape[0]}."
            )
        if k.shape[0] != expected_tokens or v.shape[0] != expected_tokens:
            raise RuntimeError(
                "Invalid MLU mixed attention KV token count: "
                f"expected_tokens={expected_tokens}, "
                f"k_tokens={k.shape[0]}, v_tokens={v.shape[0]}."
            )

        # Reshape k/v before writing to KV cache
        k = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
        v = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)

        if save_kv_cache:
            self.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = self.token_to_kv_pool.get_value_buffer(layer.layer_id)

        output = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))

        num_prefill_tokens = meta.mixed_num_prefill_tokens
        if prefill_bs > 0:
            prefill_q = q[:num_prefill_tokens].view(
                -1, layer.tp_q_head_num, layer.qk_head_dim
            )
            prefill_output = output[:num_prefill_tokens].view(
                -1, layer.tp_q_head_num, layer.v_head_dim
            )

            torch_mlu_ops.flash_attention(
                q=prefill_q,
                k=k_cache,
                v=v_cache,
                out=prefill_output,
                cu_seq_lens_q=meta.cu_seqlens_q,
                cu_seq_lens_kv=meta.cu_seqlens_kv,
                alibi_slope=None,
                attn_bias=None,
                max_seq_len_q=meta.max_seq_len_q,
                max_seq_len_kv=meta.max_seq_len_kv,
                softmax_scale=layer.scaling,
                is_causal=True,
                compute_dtype=meta.compute_dtype,
                return_lse=False,
                block_tables=meta.block_tables[:prefill_bs],
                out_dtype=torch.bfloat16,
            )

        if decode_bs > 0:
            decode_seq_lens = meta.mixed_decode_seq_lens
            decode_q = q[num_prefill_tokens : num_prefill_tokens + decode_bs].view(
                decode_bs, -1, layer.tp_q_head_num, layer.qk_head_dim
            )
            decode_output = output[
                num_prefill_tokens : num_prefill_tokens + decode_bs
            ].view(decode_bs, -1, layer.tp_q_head_num, layer.v_head_dim)

            torch_mlu_ops.single_query_cached_kv_attn(
                q=decode_q,
                k_cache=k_cache,
                v_cache=v_cache,
                out=decode_output,
                block_tables=meta.block_tables[prefill_bs:batch_size],
                context_lens=decode_seq_lens,
                k_cache_quant_scale=None,
                v_cache_quant_scale=None,
                alibi_slopes=None,
                max_contxt_len=meta.mixed_decode_max_seq_len_kv,
                windows_size_left=-1,
                windows_size_right=-1,
                softmax_scale=layer.scaling,
                head_size_v=-1,
                compute_dtype=meta.compute_dtype,
                q_quant_scale=None,
                out_quant_scale=None,
            )

        return output
