from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.nsa.dequant_k_cache import dequantize_k_cache_paged
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.layers.attention.utils import concat_and_cast_mha_k_triton
from sglang.srt.layers.communicator import get_attn_tp_context
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.deepseek_common.utils import (
    _is_cuda,
    _is_hip,
    _is_npu,
    _use_aiter_gfx95,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import BumpAllocator, next_power_of_2

if TYPE_CHECKING:
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA

if _is_cuda:
    from sgl_kernel import concat_mla_k, merge_state_v2

if _use_aiter_gfx95:
    from aiter.ops.triton.fused_fp8_quant import fused_rms_fp8_group_quant

    from sglang.srt.layers.quantization.rocm_mxfp4_utils import fused_rms_mxfp4_quant

# Configs for DeepSeek-V3:
# num_local_heads = 128
# qk_nope_head_dim = 128
# qk_rope_head_dim = 64
# qk_head_dim = qk_nope_head_dim + qk_rope_head_dim = 192
# v_head_dim = 128

# Configs for kv chunking strategy:
# sum_prefix_length:
#   Total number of tokens to be fetched from kv cache for current batch.
#   e.g: For batch with 2 sequences, seq_lens_kv = [1024, 2048], seq_lens_q = [512, 1024], then sum_prefix_length = (1024 - 512) + (2048 - 1024) = 1536
# sum_extended_length:
#   Total number of tokens in the extended part of the current batch. (=sum(seq_lens_q))
# chunked_prefix_cache_threshold:
#   The minimum sum_prefix_length to enable mha with kv chunking, 8192 by default (can be changed with SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD)
#   For batches with smaller sum_prefix_length > 0, MLA kernel with absorption will be used instead.
# max_kv_chunk_capacity:
#   The maximum number of tokens in each kv chunk, 128 * 1024 by default (can be get with forward_batch.get_max_chunk_capacity())

# The forward methods for MHA in DeepSeek models:
#
# 1. forward_normal: AttnForwardMethod.MHA
#    use multi-head attention with empty kv cache (the first batch of chunked prefill, prefix lens = 0)
#    q: [sum_extended_length, num_local_heads, qk_head_dim]
#    k: [sum_extended_length, num_local_heads, qk_head_dim]
#    v: [sum_extended_length, num_local_heads, v_head_dim]
#
# 2. forward_normal_one_shot: AttnForwardMethod.MHA_ONE_SHOT
#    use multi-head attention with short kv prefix length (chunked_prefix_cache_threshold <= sum_prefix_lens <= max_kv_chunk_capacity)
#    the kv latent vectors are fetched from memory pool, with combined kv_indices of prefix part and extended part
#    q: [batch_size, num_local_heads, qk_head_dim]
#    k: [sum_extended_length + sum_prefix_length, num_local_heads, qk_head_dim]
#    v: [sum_extended_length + sum_prefix_length, num_local_heads, v_head_dim]
#
# 3. forward_normal_chunked_kv: AttnForwardMethod.MHA_CHUNKED_KV
#    multiple phases of multi-head attention with chunked kv cache (sum_prefix_length > max_kv_chunk_capacity)
#    For the first phase, it will execute normal forward method, and returns output o_1 and lse_1,
#       q_1: [sum_extended_length, num_local_heads, qk_head_dim],
#       k_1: [sum_extended_length, num_local_heads, qk_head_dim],
#       v_1: [sum_extended_length, num_local_heads, qk_head_dim],
#       acc_o_1, acc_lse_1 = o_1, lse_1
#    For i in range(2, n), (n-1 is the number of prefix chunks), kv latent vectors are fetched from memory pool with prefix kv indices
#       q_i: [sum_extended_length, num_local_heads, qk_head_dim],
#       k_i: [chunk_size, num_local_heads, qk_head_dim],
#       v_i: [chunk_size, num_local_heads, v_head_dim],
#       acc_o_i, acc_lse_i = merge_state(acc_o_{i-1}, acc_lse_{i-1}, o_i, lse_i)
#       The final output is the accumulated output acc_o_n


class DeepseekMHAForwardMixin:

    def init_mha_forward(self: DeepseekV2AttentionMLA):
        self.disable_chunked_prefix_cache = (
            get_global_server_args().disable_chunked_prefix_cache
        )

        # TODO: Design a finer way to determine the threshold
        self.chunked_prefix_cache_threshold = (
            envs.SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD.get()
        )

    def forward_normal_prepare(
        self: DeepseekV2AttentionMLA,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        if self.q_lora_rank is not None:
            q, latent_cache = (
                get_attn_tp_context()
                .fetch_qkv_latent()
                .split(
                    [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                    dim=-1,
                )
            )

            # NSA Indexer: cache quantized keys, auto-skip topk for sequences <= nsa_index_topk

            if self.use_nsa:
                # NSA requires unquantized q_lora for the indexer. When q_b_proj is FP8
                # on gfx95, we can still use fused RMSNorm+FP8 quant, but MUST request
                # the unquantized output for q_lora; otherwise q_lora becomes the (fp8,scale)
                # tuple.
                if (
                    _use_aiter_gfx95
                    and self.q_b_proj.weight.dtype == torch.float8_e4m3fn
                ):
                    q_quanted, q_lora, _, _ = fused_rms_fp8_group_quant(
                        q,
                        self.q_a_layernorm.weight,
                        self.q_a_layernorm.variance_epsilon,
                        None,
                        None,
                        None,
                        group_size=128,
                        dtype_quant=torch.float8_e4m3fn,
                        res1=None,
                        output_unquantized_inp1=True,
                    )
                    q = self.q_b_proj(q_quanted)[0].view(
                        -1, self.num_local_heads, self.qk_head_dim
                    )
                else:
                    q_lora = self.q_a_layernorm(q)
                    q = self.q_b_proj(q_lora)[0].view(
                        -1, self.num_local_heads, self.qk_head_dim
                    )
                _ = self.indexer(
                    x=hidden_states,
                    q_lora=q_lora,
                    positions=positions,
                    forward_batch=forward_batch,
                    layer_id=self.layer_id,
                    return_indices=False,
                )
            elif _use_aiter_gfx95 and self.q_b_proj.weight.dtype == torch.uint8:
                # MXFP4: fused RMSNorm + quant
                q, _, _, _ = fused_rms_mxfp4_quant(
                    q,
                    self.q_a_layernorm.weight,
                    self.q_a_layernorm.variance_epsilon,
                    None,
                    None,
                    None,
                )
                q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
            elif _use_aiter_gfx95 and self.q_b_proj.weight.dtype == torch.float8_e4m3fn:

                q, _, _, _ = fused_rms_fp8_group_quant(
                    q,
                    self.q_a_layernorm.weight,
                    self.q_a_layernorm.variance_epsilon,
                    None,
                    None,
                    None,
                    group_size=128,
                    dtype_quant=torch.float8_e4m3fn,
                    res1=None,
                    output_unquantized_inp1=False,
                )
                q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
            else:
                q = self.q_a_layernorm(q)
                q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)

        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]

        _, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        kv_a, _ = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        latent_cache = latent_cache.unsqueeze(1)

        if _use_aiter_gfx95 and self.kv_b_proj.weight.dtype == torch.float8_e4m3fn:

            kv_a_quanted, kv_a, _, _ = fused_rms_fp8_group_quant(
                kv_a,
                self.kv_a_layernorm.weight,
                self.kv_a_layernorm.variance_epsilon,
                None,
                None,
                None,
                group_size=128,
                dtype_quant=torch.float8_e4m3fn,
                res1=None,
                output_unquantized_inp1=True,  # return unqaunt kv_a
            )

        else:
            kv_a = self.kv_a_layernorm(kv_a)

        k_pe = latent_cache[:, :, self.kv_lora_rank :]
        if self.rotary_emb is not None:
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        q[..., self.qk_nope_head_dim :] = q_pe

        self._set_mla_kv_buffer(latent_cache, kv_a, k_pe, forward_batch)
        if (
            forward_batch.mha_one_shot
            and sum(forward_batch.extend_prefix_lens_cpu) != 0
        ):
            if self.use_nsa and self.kv_cache_dtype == "fp8_e4m3":
                # FP8 path: dequantize NSA-specific FP8 format to BF16
                kv_a, k_pe = self._get_mla_kv_buffer_from_fp8_for_nsa(forward_batch)
            else:
                # BF16/FP16 path: directly fetch from cache
                kv_a, k_pe = self._get_mla_kv_buffer(
                    forward_batch.fetch_mha_one_shot_kv_indices(),
                    q.dtype,
                    forward_batch,
                )
        if _use_aiter_gfx95 and self.kv_b_proj.weight.dtype == torch.float8_e4m3fn:
            kv = self.kv_b_proj(
                kv_a_quanted,
            )[0]
        else:
            kv = self.kv_b_proj(kv_a)[0]
        kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope = kv[..., : self.qk_nope_head_dim]
        v = kv[..., self.qk_nope_head_dim :]

        k = self._concat_and_cast_mha_k(k_nope, k_pe, forward_batch)
        return q, k, v, forward_batch

    def forward_normal_core(
        self: DeepseekV2AttentionMLA,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        attn_output = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)
        attn_output = attn_output.reshape(-1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj(attn_output)
        return output

    def forward_normal_chunked_kv_prepare(
        self: DeepseekV2AttentionMLA,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        # In normal mha, the k and v tensors will become overly large when the prefix length is long.
        # To avoid this, we split the kv cache into chunks and process them one after another.
        # Since mha is compute friendly, the for loop induced here will not introduce significant overhead.
        # The top comments in https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/mla/common.py
        # will be helpful for understanding the purpose of this function.

        # First do normal mha forward to get output for extended part
        return self.forward_normal_prepare(
            positions, hidden_states, forward_batch, zero_allocator
        )

    def forward_normal_chunked_kv_core(
        self: DeepseekV2AttentionMLA,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        has_extend_prefix = forward_batch.extend_prefix_lens_cpu is not None and any(
            forward_batch.extend_prefix_lens_cpu
        )
        # Only initialize the info once
        if has_extend_prefix and forward_batch.num_prefix_chunks is None:
            forward_batch.prepare_chunked_prefix_cache_info(q.device)
            if hasattr(forward_batch.attn_backend, "init_mha_chunk_metadata"):
                forward_batch.attn_backend.init_mha_chunk_metadata(forward_batch)

        forward_batch.mha_return_lse = has_extend_prefix
        # Do mha for extended part without prefix
        forward_batch.set_attn_attend_prefix_cache(False)
        attn_output = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)

        # Do mha attention with chunked prefix cache if there are any sequence with prefix
        if has_extend_prefix:
            attn_output, lse = attn_output
            forward_batch.set_attn_attend_prefix_cache(True)
            attn_output = self._chunked_prefix_attn_mha(
                q=q,
                accum_output=attn_output,
                accum_lse=lse,
                forward_batch=forward_batch,
            )

        attn_output = attn_output.reshape(-1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj(attn_output)
        return output

    def forward_normal_one_shot_prepare(
        self: DeepseekV2AttentionMLA,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        forward_batch.mha_one_shot = True
        return self.forward_normal_prepare(
            positions, hidden_states, forward_batch, zero_allocator
        )

    def forward_normal_one_shot_core(
        self: DeepseekV2AttentionMLA,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        has_extend_prefix = any(forward_batch.extend_prefix_lens_cpu)
        # Only initialize the info once
        if has_extend_prefix and forward_batch.num_prefix_chunks is None:
            forward_batch.num_prefix_chunks = 0
            if hasattr(forward_batch.attn_backend, "init_mha_chunk_metadata"):
                forward_batch.attn_backend.init_mha_chunk_metadata(forward_batch)
        forward_batch.mha_return_lse = False
        # Do mha for extended part without prefix
        forward_batch.set_attn_attend_prefix_cache(False)
        return self.forward_normal_core(q, k, v, forward_batch)

    def _chunked_prefix_attn_mha(
        self: DeepseekV2AttentionMLA,
        q: torch.Tensor,
        accum_output: torch.Tensor,
        accum_lse: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:

        assert forward_batch.num_prefix_chunks is not None
        for i in range(forward_batch.num_prefix_chunks):
            forward_batch.set_prefix_chunk_idx(i)

            kv_indices = forward_batch.prefix_chunk_kv_indices[i]
            # Fetch latent cache from memory pool with precomputed chunked kv indices
            kv_a_normed, k_pe = self._get_mla_kv_buffer(
                kv_indices, q.dtype, forward_batch
            )
            kv = self.kv_b_proj(kv_a_normed)[0]
            kv = kv.view(
                -1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            v = kv[..., self.qk_nope_head_dim :]
            k_nope = kv[..., : self.qk_nope_head_dim]

            k = torch.empty(
                (
                    k_nope.shape[0],
                    self.num_local_heads,
                    self.qk_nope_head_dim + self.qk_rope_head_dim,
                ),
                dtype=v.dtype,
                device=v.device,
            )
            k[..., : self.qk_nope_head_dim] = k_nope
            k[..., self.qk_nope_head_dim :] = k_pe

            output, lse = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)
            tmp_output = torch.empty_like(accum_output)
            tmp_lse = torch.empty_like(accum_lse)
            merge_state_v2(output, lse, accum_output, accum_lse, tmp_output, tmp_lse)
            accum_output, accum_lse = tmp_output, tmp_lse
            del kv, k, v, output, lse, tmp_output, tmp_lse

        return accum_output

    def _set_mla_kv_buffer(
        self: DeepseekV2AttentionMLA,
        latent_cache: torch.Tensor,
        kv_a: torch.Tensor,
        k_pe: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        if _is_cuda or _use_aiter_gfx95:
            # Save latent cache
            forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                self.attn_mha, forward_batch.out_cache_loc, kv_a.unsqueeze(1), k_pe
            )
        elif _is_npu:
            # To reduce a time-costing split operation
            forward_batch.token_to_kv_pool.set_kv_buffer(
                self.attn_mha, forward_batch.out_cache_loc, kv_a.unsqueeze(1), k_pe
            )
        else:
            latent_cache[:, :, : self.kv_lora_rank] = kv_a.unsqueeze(1)
            latent_cache[:, :, self.kv_lora_rank :] = k_pe

            # Save latent cache
            forward_batch.token_to_kv_pool.set_kv_buffer(
                self.attn_mha, forward_batch.out_cache_loc, latent_cache, None
            )

    def _get_mla_kv_buffer(
        self: DeepseekV2AttentionMLA,
        kv_indices: torch.Tensor,
        dst_dtype: torch.dtype,
        forward_batch: ForwardBatch,
    ):
        if _is_cuda or _use_aiter_gfx95:
            kv_a, k_pe = forward_batch.token_to_kv_pool.get_mla_kv_buffer(
                self.attn_mha, kv_indices, dst_dtype
            )
            kv_a = kv_a.squeeze(1)
        else:
            latent_cache_buf = forward_batch.token_to_kv_pool.get_key_buffer(
                self.attn_mha.layer_id
            )
            latent_cache = latent_cache_buf[kv_indices].contiguous().to(dst_dtype)

            kv_a, k_pe = latent_cache.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            kv_a = kv_a.squeeze(1).contiguous()
        return kv_a, k_pe

    def _get_mla_kv_buffer_from_fp8_for_nsa(
        self: DeepseekV2AttentionMLA,
        forward_batch: ForwardBatch,
    ):
        """
        Dequantize FP8 KV cache to BF16 for MLA attention (NSA-specific format).

        Returns: (kv_a, k_pe) both in BF16
        """
        backend = forward_batch.attn_backend
        if isinstance(backend, TboAttnBackend):  # if enable tbo, get primary backend
            backend = backend.primary
        kv_indices = backend.forward_metadata.page_table_1_flattened
        assert (
            kv_indices is not None
        ), "page_table_1_flattened should have been generated for FP8 MHA path"

        kv_cache_fp8 = forward_batch.token_to_kv_pool.get_key_buffer(
            self.attn_mha.layer_id
        )

        kv_latent_bf16 = dequantize_k_cache_paged(kv_cache_fp8, kv_indices)

        kv_a = kv_latent_bf16[:, :, : self.kv_lora_rank].squeeze(1).contiguous()
        k_pe = kv_latent_bf16[:, :, self.kv_lora_rank :]

        return kv_a, k_pe

    def _concat_and_cast_mha_k(
        self: DeepseekV2AttentionMLA,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        # Temporary for DeepSeek V3/R1 only, but can generalize if needed
        k_shape = (k_nope.shape[0], self.num_local_heads, self.qk_head_dim)
        if (
            _is_cuda
            and (self.num_local_heads == 128)
            and (self.qk_nope_head_dim == 128)
            and (self.qk_rope_head_dim == 64)
        ):
            k = k_nope.new_empty(*k_shape)
            concat_mla_k(k=k, k_nope=k_nope, k_rope=k_pe)
        elif (
            _is_cuda
            and next_power_of_2(self.num_local_heads) == self.num_local_heads
            and next_power_of_2(self.qk_nope_head_dim) == self.qk_nope_head_dim
            and next_power_of_2(self.qk_rope_head_dim) == self.qk_rope_head_dim
        ):
            # fa3 mha support fp8 inputs
            if (
                self.current_attention_backend == "fa3"
                and self.kv_cache_dtype != "auto"
            ):
                attn_dtype = forward_batch.token_to_kv_pool.dtype
            else:
                attn_dtype = k_nope.dtype
            k = k_nope.new_empty(*k_shape, dtype=attn_dtype)
            concat_and_cast_mha_k_triton(k, k_nope, k_pe)
        elif _is_hip and self.current_attention_backend == "aiter":
            k = k_nope.new_empty(*k_shape)
            concat_and_cast_mha_k_triton(k, k_nope, k_pe)
        else:
            k = k_nope.new_empty(*k_shape)
            k[..., : self.qk_nope_head_dim] = k_nope
            k[..., self.qk_nope_head_dim :] = k_pe
        return k
