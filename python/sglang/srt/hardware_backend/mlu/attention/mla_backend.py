"""Eager BF16 MLA with dense MHA one-shot and chunked-prefix prefill."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch_mlu_ops

from sglang.srt.configs.model_config import is_deepseek_dsa
from sglang.srt.hardware_backend.mlu.attention.mha_backend import (
    ForwardMetadata,
    MLUMHABackend,
)
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

    def __init__(self, attn_backend: "MLUMLABackend"):
        self.attn_backend = attn_backend

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        meta = self.attn_backend.forward_metadata
        attend_prefix_cache = bool(forward_batch.attn_attend_prefix_cache)
        if attend_prefix_cache:
            chunk_idx = forward_batch.prefix_chunk_idx
            cu_seqlens_kv = forward_batch.prefix_chunk_cu_seq_lens[chunk_idx]
            max_seq_len_kv = forward_batch.prefix_chunk_max_seq_lens[chunk_idx]
        elif forward_batch.mha_one_shot:
            cu_seqlens_kv, max_seq_len_kv = meta.cu_seqlens_kv, meta.max_seq_len_kv
        else:
            cu_seqlens_kv, max_seq_len_kv = meta.cu_seqlens_q, meta.max_seq_len_q
        return_lse = bool(attend_prefix_cache or forward_batch.mha_return_lse)
        result = torch_mlu_ops.flash_attention(
            q=q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k=k.reshape(-1, layer.tp_k_head_num, layer.qk_head_dim),
            v=v.reshape(-1, layer.tp_v_head_num, layer.v_head_dim),
            out=None,
            cu_seq_lens_q=meta.cu_seqlens_q,
            cu_seq_lens_kv=cu_seqlens_kv,
            alibi_slope=None,
            attn_bias=None,
            max_seq_len_q=meta.max_seq_len_q,
            max_seq_len_kv=max_seq_len_kv,
            softmax_scale=layer.scaling,
            is_causal=not attend_prefix_cache,
            compute_dtype=torch.float32,
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


class MLUMLABackend(MLUMHABackend):
    def __init__(self, model_runner: "ModelRunner"):
        super().__init__(model_runner)
        if is_deepseek_dsa(model_runner.model_config.hf_config):
            raise NotImplementedError("MLU MLA does not support DSA yet.")
        if self.kv_cache_dtype != torch.bfloat16:
            raise NotImplementedError("MLU MLA requires BF16 KV cache.")
        if get_parallel().attn_dcp_size != 1 or get_parallel().attn_cp_size != 1:
            raise NotImplementedError("MLU MLA does not support context parallelism.")
        self.kv_lora_rank = model_runner.model_config.kv_lora_rank
        self.mha_chunk_kv_runner = MLUMhaChunkKVRunner(self)

    def init_forward_metadata(self, forward_batch: "ForwardBatch") -> None:
        super().init_forward_metadata(forward_batch)
        if forward_batch.forward_mode.is_idle():
            return
        self.forward_metadata = MLAForwardMetadata(**vars(self.forward_metadata))
        if forward_batch.forward_mode.is_decode():
            return
        meta = self.forward_metadata
        if meta.is_uncached_prefill:
            return
        num_prefill_reqs = meta.mixed_prefill_reqs
        if num_prefill_reqs < 0:
            num_prefill_reqs = forward_batch.batch_size
        # Physical slots, in packed request order. Build once, reuse across layers.
        rows = self.req_to_token[
            forward_batch.req_pool_indices[:num_prefill_reqs], : meta.max_seq_len_kv
        ]
        mask = (
            torch.arange(meta.max_seq_len_kv, device=self.device)[None, :]
            < meta.seq_lens[:num_prefill_reqs, None]
        )
        meta.prefill_kv_indices = rows[mask].long()

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        self._validate_attention(
            layer, kwargs.get("sinks"), kwargs.get("return_lse", False)
        )
        if save_kv_cache:
            self.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, None
            )
        q = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
        return self._forward_mla_decode(q, layer)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        self._validate_attention(
            layer, kwargs.get("sinks"), kwargs.get("return_lse", False)
        )
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
            result = self.mha_chunk_kv_runner.forward(q, k, v, layer, forward_batch)
            if isinstance(result, tuple):
                return result
            return result.flatten(1)
        if save_kv_cache:
            self.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, None
            )
        q = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
        meta = self.forward_metadata
        num_prefill_reqs = meta.mixed_prefill_reqs
        if num_prefill_reqs == 0:
            return self._forward_mla_decode(q, layer)
        num_prefill_tokens = (
            meta.mixed_prefill_tokens if num_prefill_reqs >= 0 else q.shape[0]
        )
        prefill = self._forward_mla_prefill(q[:num_prefill_tokens], k, layer)
        if num_prefill_tokens == q.shape[0]:
            return prefill
        return torch.cat(
            (
                prefill,
                self._forward_mla_decode(
                    q[num_prefill_tokens:], layer, num_prefill_reqs
                ),
            ),
            dim=0,
        )

    def _forward_mla_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        layer: "RadixAttention",
    ) -> torch.Tensor:
        meta = self.forward_metadata
        if meta.is_uncached_prefill:
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
                meta.prefill_kv_indices
            ]
        return torch_mlu_ops.flash_attention(
            q=q,
            k=prefill_k,
            v=prefill_k[..., : self.kv_lora_rank],
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
            block_tables=None,
            out_dtype=q.dtype,
        ).flatten(1)

    def _forward_mla_decode(
        self,
        q: torch.Tensor,
        layer: "RadixAttention",
        start_req: int = 0,
    ) -> torch.Tensor:
        meta = self.forward_metadata
        cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id).view(
            -1, 1, self.page_size, layer.qk_head_dim
        )
        result = torch_mlu_ops.single_query_cached_kv_attn(
            q=q.unsqueeze(1),
            k_cache=cache,
            v_cache=None,
            out=None,
            block_tables=meta.block_tables[start_req:],
            context_lens=meta.seq_lens[start_req:],
            k_cache_quant_scale=None,
            v_cache_quant_scale=None,
            alibi_slopes=None,
            max_context_len=self.max_context_len,
            windows_size_left=-1,
            windows_size_right=-1,
            softmax_scale=layer.scaling,
            head_size_v=layer.v_head_dim,
            compute_dtype=torch.float32,
            q_quant_scale=None,
            out_quant_scale=None,
        )
        return result.reshape(q.shape[0], -1)
