"""AMD/ROCm multi-head attention forward path for DeepSeek models.

`AttnForwardMethod.MHA_ROCM` and `AttnForwardMethod.MHA_ONE_SHOT_ROCM` route
here, which keeps every AITER/gfx95 kernel choice out of the shared
`forward_mha.py`. Only the prepare step is platform specific; the attention
cores in `DeepseekMHAForwardMixin` are reused as-is.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.ops.attention.utils import concat_and_cast_mha_k_triton
from sglang.srt.layers.communicator import get_attn_tp_context
from sglang.srt.layers.dcp import (
    all_gather_kv_cache_for_mha_extend,
    filter_dcp_local_kv_indices,
)
from sglang.srt.layers.quantization.fp8_utils import (
    materialize_bpreshuffle_fp8_scale_tuple,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mha import (
    forward_dsa_indexer_for_mha,
    resolve_attn_backend,
)
from sglang.srt.models.deepseek_common.utils import (
    _is_block_scale_fp8,
    _use_aiter_bpreshuffle_gfx95,
    _use_aiter_gfx95,
)
from sglang.srt.runtime_context import get_exec, get_parallel
from sglang.srt.utils import BumpAllocator, get_bool_env_var

if TYPE_CHECKING:
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA

_use_fp8_prefill_attn = (
    get_bool_env_var("SGLANG_AITER_FP8_PREFILL_ATTN", "True") and _use_aiter_gfx95
)

if _use_aiter_gfx95:
    from aiter.ops.triton.fused_fp8_quant import fused_rms_fp8_group_quant

    from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
    from sglang.srt.layers.quantization.rocm_mxfp4_utils import fused_rms_mxfp4_quant


class DeepseekMHARocmForwardMixin:

    def forward_normal_rocm_prepare(
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

            # DSA Indexer: cache quantized keys, auto-skip topk for sequences <= dsa_index_topk

            if self.use_dsa:
                # DSA requires unquantized q_lora for the indexer. When q_b_proj is FP8
                # on gfx95, we can still use fused RMSNorm+FP8 quant, but MUST request
                # the unquantized output for q_lora; otherwise q_lora becomes the (fp8,scale)
                # tuple.
                if _use_aiter_gfx95 and _is_block_scale_fp8(self.q_b_proj):
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
                        transpose_scale=False,
                    )
                    if _use_aiter_bpreshuffle_gfx95:
                        q_quanted = materialize_bpreshuffle_fp8_scale_tuple(q_quanted)
                    q = self.q_b_proj(q_quanted)[0].view(
                        -1, self.num_local_heads, self.qk_head_dim
                    )
                else:
                    q_lora = self.q_a_layernorm(q)
                    q = self.q_b_proj(q_lora)[0].view(
                        -1, self.num_local_heads, self.qk_head_dim
                    )
                if self.should_run_indexer():
                    forward_dsa_indexer_for_mha(
                        self.indexer,
                        hidden_states=hidden_states,
                        q_lora=q_lora,
                        positions=positions,
                        forward_batch=forward_batch,
                        layer_id=self.layer_id,
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
            elif _use_aiter_gfx95 and _is_block_scale_fp8(self.q_b_proj):
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
                    transpose_scale=False,
                )
                if _use_aiter_bpreshuffle_gfx95:
                    q = materialize_bpreshuffle_fp8_scale_tuple(q)
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

        if _use_aiter_gfx95 and _is_block_scale_fp8(self.kv_b_proj):
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
                transpose_scale=False,
            )
            if _use_aiter_bpreshuffle_gfx95:
                kv_a_quanted = materialize_bpreshuffle_fp8_scale_tuple(kv_a_quanted)
        else:
            kv_a = self.kv_a_layernorm(kv_a)

        k_pe = latent_cache[:, :, self.kv_lora_rank :]

        # Backend prefill hook: the backend owns the BF16->FP8 transition
        # (fused RoPE + quantize for Q/K, direct FP8 KV-cache write) and
        # returns FP8 tensors ready for its kernel. Backends without the
        # hook fall through to the BF16 path below.
        backend = resolve_attn_backend(forward_batch)
        if hasattr(backend, "prepare_prefill_qkv"):
            q_out, k_out, v_out = backend.prepare_prefill_qkv(
                q=q,
                q_pe=q_pe,
                kv_a=kv_a,
                k_pe=k_pe,
                positions=positions,
                layer=self,
                forward_batch=forward_batch,
            )
            return q_out, k_out, v_out, forward_batch

        if self.rotary_emb is not None:
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        q[..., self.qk_nope_head_dim :] = q_pe

        self._set_mla_kv_buffer_rocm(latent_cache, kv_a, k_pe, forward_batch)
        if (
            forward_batch.mha_one_shot
            and sum(forward_batch.extend_prefix_lens_cpu) != 0
        ):
            if (
                self.use_dsa
                and self.kv_cache_dtype == "fp8_e4m3"
                and (
                    not get_exec().kernel.dsa_decode_backend == "trtllm"
                    or not get_exec().kernel.dsa_prefill_backend == "trtllm"
                )
            ):
                # FP8 path: dequantize DSA-specific FP8 format to BF16
                kv_a, k_pe = self._get_mla_kv_buffer_from_fp8_for_dsa(forward_batch)
            else:
                # BF16/FP16 path: directly fetch from cache
                if get_parallel().dcp_enabled:
                    kv_a, k_pe = all_gather_kv_cache_for_mha_extend(
                        get_token_to_kv_pool(),
                        self.attn_mha,
                        forward_batch.attn_dcp_metadata.dcp_local_prefix_kv_indices,
                        forward_batch.seq_lens,
                        forward_batch.extend_prefix_lens,
                        forward_batch.extend_prefix_lens_cpu,
                        forward_batch.extend_seq_lens,
                        kv_a,
                        k_pe,
                    )
                else:
                    kv_a, k_pe = self._get_mla_kv_buffer_rocm(
                        forward_batch.fetch_mha_one_shot_kv_indices(),
                        q.dtype,
                        forward_batch,
                    )
        if _use_fp8_prefill_attn and self.kv_b_proj.weight.dtype == torch.uint8:
            # MXFP4 weights + FP8 prefill: fuse GEMM, nope/v split, and k_pe cat
            # into a single kernel (fused_gemm_afp4wfp4_split_cat) that writes k and v
            # directly in FP8, avoiding a separate elementwise cast
            k, v = self.kv_b_proj(
                (
                    kv_a,
                    k_pe.expand(-1, self.num_local_heads, -1),
                    self.qk_nope_head_dim,
                    self.v_head_dim,
                    fp8_dtype,
                )
            )[0]
        else:
            if _use_aiter_gfx95 and _is_block_scale_fp8(self.kv_b_proj):
                kv = self.kv_b_proj(kv_a_quanted)[0]
            else:
                kv = self.kv_b_proj(kv_a)[0]
            kv = kv.view(
                -1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            k_nope = kv[..., : self.qk_nope_head_dim]
            v = kv[..., self.qk_nope_head_dim :]

            k = self._concat_and_cast_mha_k_rocm(k_nope, k_pe)
        return q, k, v, forward_batch

    def forward_normal_one_shot_rocm_prepare(
        self: DeepseekV2AttentionMLA,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        forward_batch.mha_one_shot = True
        return self.forward_normal_rocm_prepare(
            positions, hidden_states, forward_batch, zero_allocator
        )

    def _concat_and_cast_mha_k_rocm(
        self: DeepseekV2AttentionMLA,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor | None,
    ):
        if self.qk_rope_head_dim == 0:
            assert k_pe is None or k_pe.shape[-1] == 0
            return k_nope.contiguous()

        k_shape = (k_nope.shape[0], self.num_local_heads, self.qk_head_dim)
        k = k_nope.new_empty(*k_shape)
        if self.current_attention_backend == "aiter":
            concat_and_cast_mha_k_triton(k, k_nope, k_pe)
        else:
            k[..., : self.qk_nope_head_dim] = k_nope
            k[..., self.qk_nope_head_dim :] = k_pe
        return k

    def _set_mla_kv_buffer_rocm(
        self: DeepseekV2AttentionMLA,
        latent_cache: torch.Tensor,
        kv_a: torch.Tensor,
        k_pe: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        if _use_aiter_gfx95:
            get_token_to_kv_pool().set_mla_kv_buffer(
                self.attn_mha, forward_batch.out_cache_loc, kv_a.unsqueeze(1), k_pe
            )
        else:
            latent_cache[:, :, : self.kv_lora_rank] = kv_a.unsqueeze(1)
            latent_cache[:, :, self.kv_lora_rank :] = k_pe.clone()
            get_token_to_kv_pool().set_kv_buffer(
                self.attn_mha, forward_batch.out_cache_loc, latent_cache, None
            )

    def _get_mla_kv_buffer_rocm(
        self: DeepseekV2AttentionMLA,
        kv_indices: torch.Tensor,
        dst_dtype: torch.dtype,
        forward_batch: ForwardBatch,
    ):
        if _use_aiter_gfx95:
            kv_indices = filter_dcp_local_kv_indices(kv_indices=kv_indices)
            kv_a, k_pe = get_token_to_kv_pool().get_mla_kv_buffer(
                self.attn_mha, kv_indices, dst_dtype
            )
            kv_a = kv_a.squeeze(1)
        else:
            latent_cache_buf = get_token_to_kv_pool().get_key_buffer(
                self.attn_mha.layer_id
            )
            latent_cache = latent_cache_buf[kv_indices].contiguous().to(dst_dtype)
            kv_a, k_pe = latent_cache.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            kv_a = kv_a.squeeze(1).contiguous()
        return kv_a, k_pe
