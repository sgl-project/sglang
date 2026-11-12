"""Eager BF16 MLA with dense MHA one-shot and chunked-prefix prefill."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch_mlu_ops

from sglang.srt.configs.model_config import is_deepseek_dsa
from sglang.srt.hardware_backend.mlu.attention.metadata import (
    ForwardMetadata,
    build_forward_metadata,
)
from sglang.srt.hardware_backend.mlu.attention.utils import validate_attention
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner


@dataclass
class MLAForwardMetadata(ForwardMetadata):
    # Physical token slots in packed prefill request order, shared across layers.
    prefill_kv_indices: torch.Tensor | None = None


class MLUMhaChunkKVRunner:
    """Run dense MHA phases using the backend's shared forward metadata."""

    def __init__(self, attn_backend: "MLUMLABackend") -> None:
        self.attn_backend = attn_backend

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        return_lse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        forward_metadata = self.attn_backend.forward_metadata
        attend_prefix_cache = bool(forward_batch.attn_attend_prefix_cache)
        if attend_prefix_cache:
            chunk_idx = forward_batch.prefix_chunk_idx
            cu_seqlens_kv = forward_batch.prefix_chunk_cu_seq_lens[chunk_idx]
            max_seq_len_kv = forward_batch.prefix_chunk_max_seq_lens[chunk_idx]
        elif forward_batch.mha_one_shot:
            cu_seqlens_kv, max_seq_len_kv = (
                forward_metadata.cu_seqlens_kv,
                forward_metadata.max_seq_len_kv,
            )
        else:
            cu_seqlens_kv, max_seq_len_kv = (
                forward_metadata.cu_seqlens_q,
                forward_metadata.max_seq_len_q,
            )
        return_lse = bool(
            return_lse or attend_prefix_cache or forward_batch.mha_return_lse
        )
        result = torch_mlu_ops.flash_attention(
            q=q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k=k.reshape(-1, layer.tp_k_head_num, layer.qk_head_dim),
            v=v.reshape(-1, layer.tp_v_head_num, layer.v_head_dim),
            out=None,
            cu_seq_lens_q=forward_metadata.cu_seqlens_q,
            cu_seq_lens_kv=cu_seqlens_kv,
            alibi_slope=None,
            attn_bias=None,
            max_seq_len_q=forward_metadata.max_seq_len_q,
            max_seq_len_kv=max_seq_len_kv,
            softmax_scale=layer.scaling,
            is_causal=not attend_prefix_cache,
            compute_dtype=torch.float16,
            return_lse=return_lse,
            block_tables=None,
            out_dtype=q.dtype,
        )
        if return_lse:
            output, lse = result
            # Convert LSE from the operator's [num_heads, num_tokens] layout
            # to [num_tokens, num_heads] for shared attention state merging.
            return output, lse.transpose(0, 1)
        return result


class MLUMLABackend(AttentionBackend):
    def __init__(self, model_runner: "ModelRunner") -> None:
        self.device = model_runner.device
        self.page_size = model_runner.page_size
        self.max_context_len = model_runner.model_config.context_len
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.kv_cache_dtype = model_runner.kv_cache_dtype
        self.forward_metadata = None
        if is_deepseek_dsa(model_runner.model_config.hf_config):
            raise NotImplementedError("MLU MLA does not support DSA yet.")
        if self.kv_cache_dtype != torch.bfloat16:
            raise NotImplementedError("MLU MLA requires BF16 KV cache.")
        if get_parallel().attn_dcp_size != 1 or get_parallel().attn_cp_size != 1:
            raise NotImplementedError("MLU MLA does not support context parallelism.")
        self.kv_lora_rank = model_runner.model_config.kv_lora_rank
        self.mha_chunk_kv_runner = MLUMhaChunkKVRunner(self)

    def init_forward_metadata(self, forward_batch: "ForwardBatch") -> None:
        if forward_batch.forward_mode.is_idle():
            return
        self.forward_metadata = build_forward_metadata(
            forward_batch,
            self.req_to_token,
            self.page_size,
            self.max_context_len,
            metadata_cls=MLAForwardMetadata,
        )
        if forward_batch.forward_mode.is_decode():
            return
        forward_metadata = self.forward_metadata
        if forward_metadata.is_uncached_prefill:
            return
        num_prefill_reqs = forward_metadata.mixed_prefill_reqs
        if num_prefill_reqs < 0:
            num_prefill_reqs = forward_batch.batch_size
        # Physical slots, in packed request order. Build once, reuse across layers.
        rows = self.req_to_token[
            forward_batch.req_pool_indices[:num_prefill_reqs],
            : forward_metadata.max_seq_len_kv,
        ]
        mask = (
            torch.arange(forward_metadata.max_seq_len_kv, device=self.device)[None, :]
            < forward_metadata.seq_lens[:num_prefill_reqs, None]
        )
        forward_metadata.prefill_kv_indices = rows[mask].long()

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        validate_attention(layer, kwargs.get("sinks"))
        return_lse = kwargs.get("return_lse", False)
        if save_kv_cache:
            self.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, None
            )
        q = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
        return self._forward_mla_decode(q, layer, return_lse=return_lse)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        validate_attention(layer, kwargs.get("sinks"))
        return_lse = kwargs.get("return_lse", False)
        # MHA expands the latent cache in the model and has already stored KV.
        is_latent = (
            layer.tp_k_head_num == 1
            and layer.qk_head_dim == self.token_to_kv_pool.kv_cache_dim
            and layer.v_head_dim == self.kv_lora_rank
        )
        if not is_latent:
            if save_kv_cache:
                raise ValueError(
                    "MLU MHA prefill expects latent KV stored by the model."
                )
            result = self.mha_chunk_kv_runner.forward(
                q, k, v, layer, forward_batch, return_lse=return_lse
            )
            if isinstance(result, tuple):
                return result
            return result.flatten(1)
        if save_kv_cache:
            self.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, None
            )
        q = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
        forward_metadata = self.forward_metadata
        num_prefill_reqs = forward_metadata.mixed_prefill_reqs
        if num_prefill_reqs == 0:
            return self._forward_mla_decode(q, layer, return_lse=return_lse)
        num_prefill_tokens = (
            forward_metadata.mixed_prefill_tokens
            if num_prefill_reqs >= 0
            else q.shape[0]
        )
        prefill = self._forward_mla_prefill(
            q[:num_prefill_tokens], k, layer, return_lse=return_lse
        )
        if num_prefill_tokens == q.shape[0]:
            return prefill
        decode = self._forward_mla_decode(
            q[num_prefill_tokens:], layer, num_prefill_reqs, return_lse=return_lse
        )
        if return_lse:
            return torch.cat((prefill[0], decode[0]), dim=0), torch.cat(
                (prefill[1], decode[1]), dim=0
            )
        return torch.cat((prefill, decode), dim=0)

    def _forward_mla_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        layer: "RadixAttention",
        return_lse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        forward_metadata = self.forward_metadata
        if forward_metadata.is_uncached_prefill:
            if k is None:
                raise NotImplementedError(
                    "MLU MLA prefill without a cached prefix requires explicit K; "
                    "reading prewritten KV cache without K is not supported yet."
                )
            prefill_k = k.reshape(-1, layer.tp_k_head_num, layer.qk_head_dim)[
                : q.shape[0]
            ]
        else:
            prefill_k = self.token_to_kv_pool.get_key_buffer(layer.layer_id)[
                forward_metadata.prefill_kv_indices
            ]
        result = torch_mlu_ops.flash_attention(
            q=q,
            k=prefill_k,
            v=prefill_k[..., : self.kv_lora_rank],
            out=None,
            cu_seq_lens_q=forward_metadata.cu_seqlens_q,
            cu_seq_lens_kv=forward_metadata.cu_seqlens_kv,
            alibi_slope=None,
            attn_bias=None,
            max_seq_len_q=forward_metadata.max_seq_len_q,
            max_seq_len_kv=forward_metadata.max_seq_len_kv,
            softmax_scale=layer.scaling,
            is_causal=True,
            compute_dtype=torch.float16,
            return_lse=return_lse,
            block_tables=None,
            out_dtype=q.dtype,
        )
        if return_lse:
            output, lse = result
            # Convert [num_heads, num_tokens] to [num_tokens, num_heads].
            return output.flatten(1), lse.transpose(0, 1)
        return result.flatten(1)

    def _forward_mla_decode(
        self,
        q: torch.Tensor,
        layer: "RadixAttention",
        start_req: int = 0,
        return_lse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        forward_metadata = self.forward_metadata
        cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id).view(
            -1, 1, self.page_size, layer.qk_head_dim
        )
        result = torch_mlu_ops.single_query_cached_kv_attn(
            q=q.unsqueeze(1),
            k_cache=cache,
            v_cache=None,
            out=None,
            block_tables=forward_metadata.block_tables[start_req:],
            context_lens=forward_metadata.seq_lens[start_req:],
            k_cache_quant_scale=None,
            v_cache_quant_scale=None,
            alibi_slopes=None,
            max_context_len=self.max_context_len,
            windows_size_left=-1,
            windows_size_right=-1,
            softmax_scale=layer.scaling,
            return_lse=return_lse,
            head_size_v=layer.v_head_dim,
            compute_dtype=torch.float16,
            q_quant_scale=None,
            out_quant_scale=None,
        )
        if return_lse:
            output, lse = result
            # Each request has one query: [batch_size, num_heads, 1] -> [batch_size, num_heads].
            return output.reshape(q.shape[0], -1), lse.squeeze(-1)
        return result.reshape(q.shape[0], -1)
