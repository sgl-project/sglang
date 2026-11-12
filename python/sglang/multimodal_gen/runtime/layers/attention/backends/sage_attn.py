# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0


from functools import lru_cache

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (  # FlashAttentionMetadata,
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    trailing_padding_used_len,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.environ import envs

logger = init_logger(__name__)


@torch.compiler.disable
@lru_cache(maxsize=4)
def _trtllm_gen_workspace(device_index: int) -> torch.Tensor:
    # The TRT-LLM Gen ragged FMHA API takes caller-owned workspace. Reuse one
    # buffer because diffusion invokes the same attention shape many times.
    # compiler.disable is required: Dynamo traces through lru_cache and would
    # otherwise replay this memset on every compiled denoise step.
    return torch.zeros(
        256 * 1024 * 1024, dtype=torch.uint8, device=f"cuda:{device_index}"
    )


class SageAttentionBackend(AttentionBackend):
    @classmethod
    def supports_ring_rotation(cls) -> bool:
        return True

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        # SageAttention 2 pads smaller head dimensions to 64 or 128 internally
        return list(range(1, 129))

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.SAGE_ATTN

    @staticmethod
    def get_impl_cls() -> type["SageAttentionImpl"]:
        return SageAttentionImpl


class SageAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout = extra_impl_args.get("dropout_p", 0.0)
        self.packed_trailing_padding = extra_impl_args.get(
            "packed_trailing_padding", False
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
        *,
        return_softmax_lse: bool = False,
    ) -> torch.Tensor:
        if envs.SGLANG_USE_SAGE_ATTENTION_IMPL.get().lower() in {
            "trtllm_gen",
            "trtllm-gen",
        }:
            if return_softmax_lse:
                raise NotImplementedError(
                    "TRT-LLM Gen Sage attention does not return the diffusion backend's LSE layout"
                )
            if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
                raise ValueError(
                    "TRT-LLM Gen Sage attention expects dense [batch, sequence, heads, dim] Q/K/V"
                )
            batch, q_len, num_heads, head_dim = query.shape
            kv_batch, kv_len, kv_heads, kv_dim = key.shape
            if value.shape != key.shape or kv_batch != batch or kv_heads != num_heads:
                raise ValueError(
                    "TRT-LLM Gen Sage attention requires matching dense Q/K/V batches and heads"
                )
            if head_dim != kv_dim or query.dtype not in (torch.float16, torch.bfloat16):
                raise ValueError(
                    "TRT-LLM Gen Sage attention requires matching FP16/BF16 head dimensions"
                )

            from flashinfer.prefill import (
                trtllm_ragged_attention_deepseek,
                trtllm_sage_attention_quantize,
            )

            query_thd = query.contiguous().view(batch * q_len, num_heads, head_dim)
            key_thd = key.contiguous().view(batch * kv_len, num_heads, head_dim)
            value_thd = value.contiguous().view(batch * kv_len, num_heads, head_dim)
            device_index = query.device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            from sglang.multimodal_gen.runtime.managers.forward_context import (
                get_forward_context,
            )

            (
                qo_indptr,
                kv_indptr,
                seq_lens,
            ) = get_forward_context().get_trtllm_gen_sequence_metadata(
                device_index, batch, q_len, kv_len
            )
            q_int8, k_int8, v_fp8, q_sf, k_sf, v_sf = trtllm_sage_attention_quantize(
                query_thd,
                key_thd,
                value_thd,
                q_block_size=1,
                k_block_size=16,
                qk_quant_dtype=torch.int8,
                cum_seq_lens_q=qo_indptr,
                cum_seq_lens_kv=kv_indptr,
            )[:6]
            output = trtllm_ragged_attention_deepseek(
                q_int8,
                k_int8,
                v_fp8,
                _trtllm_gen_workspace(device_index),
                seq_lens,
                q_len,
                kv_len,
                self.softmax_scale,
                1.0,
                -1.0,
                batch,
                -1,
                qo_indptr,
                kv_indptr,
                False,
                self.causal,
                False,
                sage_attn_sfs=(q_sf, k_sf, None, v_sf),
                num_elts_per_sage_attn_blk=(1, 16, 0, 1),
            )
            logger.info_once("Using FlashInfer TRT-LLM Gen Sage attention")
            return output.view(batch, q_len, num_heads, head_dim)

        # SageAttention's public dispatcher does not yet select its FP8-PV
        # kernel on sm100. The FP8 CUDA kernel is compiled for Blackwell and
        # supports the same NHD layout used by this backend.
        if query.is_cuda and torch.cuda.get_device_capability(query.device)[0] == 10:
            from sageattention import sageattn_qk_int8_pv_fp8_cuda

            output = sageattn_qk_int8_pv_fp8_cuda(
                query,
                key,
                value,
                tensor_layout="NHD",
                is_causal=self.causal,
                sm_scale=self.softmax_scale,
                return_lse=return_softmax_lse,
                pv_accum_dtype="fp32+fp16",
            )
        else:
            from sageattention import sageattn

            output = sageattn(
                query,
                key,
                value,
                # since input is (batch_size, seq_len, head_num, head_dim)
                tensor_layout="NHD",
                is_causal=self.causal,
                sm_scale=self.softmax_scale,
                return_lse=return_softmax_lse,
            )
        if return_softmax_lse:
            output, softmax_lse = output
            return output, softmax_lse
        return output

    def forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        bounds = (
            cu_seqlens_host
            if cu_seqlens_host is not None
            else tuple(int(x) for x in cu_seqlens.tolist())
        )
        return self._sage_packed(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            bounds=bounds,
            max_seqlen=max_seqlen,
        )

    def _sage_packed(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        bounds: tuple[int, ...],
        max_seqlen: int,
    ) -> torch.Tensor:
        # MiniMax-H3 packs one live document as bounds=(0, used, total):
        # [0, used) are real tokens; [used, total) is 64-aligned tail padding.
        used = (
            trailing_padding_used_len(
                total_tokens=query.shape[0],
                max_seqlen=max_seqlen,
                bounds=bounds,
            )
            if self.packed_trailing_padding
            else None
        )
        if used is not None:
            live_out = self.forward(
                query[:used].unsqueeze(0),
                key[:used].unsqueeze(0),
                value[:used].unsqueeze(0),
                None,
            )[0]
            if used == query.shape[0]:
                return live_out
            # Keep padded tail at zero so downstream masked rows stay inactive.
            output = torch.zeros_like(query)
            output[:used] = live_out
            return output

        output = torch.empty_like(query)
        for start, stop in zip(bounds[:-1], bounds[1:]):
            if start == stop:
                continue
            output[start:stop] = self.forward(
                query[start:stop].unsqueeze(0),
                key[start:stop].unsqueeze(0),
                value[start:stop].unsqueeze(0),
                None,
            )[0]
        return output
