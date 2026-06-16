# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

"""Attention backend for the tokenspeed-mla CuTe DSL kernels on Blackwell.

Subclasses :class:`TRTLLMMLABackend` and overrides the kernel dispatch plus
tokenspeed-specific prefill Q/K/V preparation. Metadata, KV-cache layout,
CUDA-graph plumbing, draft-extend padding, and chunked-prefix dispatch are
inherited unchanged from the parent.
"""

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.fp8_quantize import fp8_quantize
from sglang.jit_kernel.mla_kv_pack_quantize_fp8 import mla_kv_pack_quantize_fp8
from sglang.jit_kernel.utils import is_arch_support_pdl
from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
    TRTLLMMLAMultiStepDraftBackend,
)
from sglang.srt.utils import is_flashinfer_available, is_tokenspeed_mla_available

if is_flashinfer_available():
    import flashinfer.rope as _flashinfer_rope

if is_tokenspeed_mla_available():
    import tokenspeed_mla

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA

logger = logging.getLogger(__name__)


# Workspace upper bound for tokenspeed_mla_decode:
#   num_sms * num_heads * max_q_len * (kv_lora_rank + 1) * sizeof(float32)
# MAX_Q_LEN=8 covers EAGLE3 num_draft_tokens=4 plus headroom.
_TOKENSPEED_MAX_Q_LEN = 8

_g_tokenspeed_workspace: dict[torch.device, torch.Tensor] = {}


def _get_tokenspeed_workspace(
    device: torch.device, num_heads: int, kv_lora_rank: int
) -> torch.Tensor:
    needed = (
        tokenspeed_mla.get_num_sm(device)
        * num_heads
        * _TOKENSPEED_MAX_Q_LEN
        * (kv_lora_rank + 1)
        * 4
    )
    existing = _g_tokenspeed_workspace.get(device)
    if existing is None or existing.numel() < needed:
        _g_tokenspeed_workspace[device] = torch.empty(
            needed, dtype=torch.int8, device=device
        )
    return _g_tokenspeed_workspace[device]


# TODO(Qiaolin-Yu): Merge this attention backend into trtllm_mla_backend.py
# once the same CuteDSL kernels in flashinfer_trtllm are stable
# and there is no performance gap compared to this backend.
class TokenspeedMLABackend(TRTLLMMLABackend):
    """tokenspeed-mla CuTe DSL attention backend (Blackwell SM100)."""

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        q_indptr_decode_buf: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            model_runner,
            skip_prefill,
            kv_indptr_buf,
            q_indptr_decode_buf,
        )

        if self.data_type not in (torch.float8_e4m3fn, torch.bfloat16, torch.float16):
            raise ValueError(
                "tokenspeed_mla backend requires --kv-cache-dtype "
                "fp8_e4m3, bf16, bfloat16, or fp16, "
                f"got data_type={self.data_type}."
            )
        if self.page_size not in (32, 64):
            raise ValueError(
                "tokenspeed_mla backend requires page_size in {32, 64}, "
                f"got page_size={self.page_size}."
            )

        self._tokenspeed_workspace: Optional[torch.Tensor] = None
        if is_tokenspeed_mla_available():
            self._tokenspeed_workspace = _get_tokenspeed_workspace(
                self.device, self.num_q_heads, self.kv_lora_rank
            )

            # Pre-JIT the prefill kernel variants. Each cute.compile takes 1-2
            # min; without warm-up the first request trips the 300 s scheduler
            # watchdog.
            _compile_prefill_kernel = tokenspeed_mla.mla_prefill._compile_prefill_kernel
            _compiled_kernels = tokenspeed_mla.mla_prefill._compiled_kernels
            head_dim_qk = self.qk_nope_head_dim + self.qk_rope_head_dim
            enable_ex2_emulation = tokenspeed_mla.mla_prefill._enable_ex2_emulation()
            use_pdl = is_arch_support_pdl()
            for is_causal in (True, False):
                for return_lse in (True, False):
                    # Non-causal is only entered from the chunked-prefix
                    # branch, which always asks for the LSE.
                    if is_causal is False and return_lse is False:
                        continue
                    # Runtime feeds self.data_type q/k/v
                    config = (
                        self.data_type,
                        head_dim_qk,
                        self.v_head_dim,
                        is_causal,
                        return_lse,
                        use_pdl,
                        enable_ex2_emulation,
                    )
                    if config in _compiled_kernels:
                        continue
                    _compiled_kernels[config] = _compile_prefill_kernel(
                        self.data_type,
                        head_dim_qk,
                        self.v_head_dim,
                        is_causal,
                        return_lse,
                        use_pdl=use_pdl,
                        enable_ex2_emulation=enable_ex2_emulation,
                    )

    def _fused_rope_fp8_quantize(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        positions: torch.Tensor,
        is_neox: bool,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused RoPE + FP8 quantize that also packs nope+pe along the last
        dim, so FMHA consumes contig FP8 Q/K without an extra concat or cast.
        """
        num_heads = q_nope.shape[1]
        seq_len = q_nope.shape[0]
        q_fp8 = torch.empty(
            (seq_len, num_heads, qk_nope_head_dim + qk_rope_head_dim),
            dtype=torch.float8_e4m3fn,
            device=q_nope.device,
        )
        k_fp8 = torch.empty(
            (seq_len, num_heads, qk_nope_head_dim + qk_rope_head_dim),
            dtype=torch.float8_e4m3fn,
            device=k_nope.device,
        )
        if seq_len == 0:
            return q_fp8, k_fp8

        # Broadcast the shared latent k_pe across heads — RoPE is position-only
        # so per-head outputs are identical, and the cache write below reuses
        # head 0.
        if k_pe.dim() == 3 and k_pe.shape[1] == 1:
            k_pe_expanded = k_pe.expand(-1, num_heads, -1)
        else:
            k_pe_expanded = k_pe

        _flashinfer_rope.mla_rope_quantize_fp8(
            q_rope=q_pe,
            k_rope=k_pe_expanded,
            q_nope=q_nope,
            k_nope=k_nope,
            cos_sin_cache=cos_sin_cache,
            pos_ids=positions,
            is_neox=is_neox,
            quantize_dtype=torch.float8_e4m3fn,
            q_rope_out=q_fp8[..., qk_nope_head_dim:],
            k_rope_out=k_fp8[..., qk_nope_head_dim:],
            q_nope_out=q_fp8[..., :qk_nope_head_dim],
            k_nope_out=k_fp8[..., :qk_nope_head_dim],
            quant_scale_q=1.0,
            quant_scale_kv=1.0,
            enable_pdl=is_arch_support_pdl(),
        )
        return q_fp8, k_fp8

    def prepare_prefill_qkv(
        self,
        *,
        q: torch.Tensor,
        q_pe: torch.Tensor,
        kv_a: torch.Tensor,
        k_pe: torch.Tensor,
        positions: torch.Tensor,
        layer: DeepseekV2AttentionMLA,
        forward_batch: ForwardBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build Q/K/V for the FMHA kernel and write the MLA KV cache."""
        kv = layer.kv_b_proj(kv_a)[0]
        kv = kv.view(
            -1, layer.num_local_heads, layer.qk_nope_head_dim + layer.v_head_dim
        )
        k_nope = kv[..., : layer.qk_nope_head_dim]
        v = kv[..., layer.qk_nope_head_dim :]
        q_nope = q[..., : layer.qk_nope_head_dim]

        if self.data_type == torch.float8_e4m3fn:
            q_fp8, k_fp8 = self._fused_rope_fp8_quantize(
                q_nope=q_nope,
                q_pe=q_pe,
                k_nope=k_nope,
                k_pe=k_pe,
                cos_sin_cache=layer.rotary_emb.cos_sin_cache,
                positions=positions,
                is_neox=getattr(layer.rotary_emb, "is_neox_style", True),
                qk_nope_head_dim=layer.qk_nope_head_dim,
                qk_rope_head_dim=layer.qk_rope_head_dim,
            )
            v_fp8 = fp8_quantize(v, enable_pdl=is_arch_support_pdl())

            # k_pe is shared across heads (RoPE is position-only), so head 0
            # reproduces the original [tokens, 1, qk_rope] latent layout.
            kv_a_fp8 = fp8_quantize(kv_a, enable_pdl=is_arch_support_pdl())
            k_pe_fp8 = k_fp8[:, 0:1, layer.qk_nope_head_dim :]
            self.token_to_kv_pool.set_mla_kv_buffer(
                layer.attn_mha,
                forward_batch.out_cache_loc,
                kv_a_fp8.unsqueeze(1),
                k_pe_fp8,
            )
            return q_fp8, k_fp8, v_fp8

        if layer.rotary_emb is not None:
            q_pe, k_pe = layer.rotary_emb(positions, q_pe, k_pe)
        self.token_to_kv_pool.set_mla_kv_buffer(
            layer.attn_mha,
            forward_batch.out_cache_loc,
            kv_a.unsqueeze(1),
            k_pe,
        )
        q_out = torch.cat([q_nope, q_pe], dim=-1).to(self.data_type)
        k_out = torch.cat(
            [k_nope, k_pe.expand(-1, layer.num_local_heads, -1)], dim=-1
        ).to(self.data_type)
        return q_out, k_out, v.to(self.data_type).contiguous()

    def pack_prefix_chunk_kv(
        self,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack strided ``k_nope``+``k_pe`` and match the backend KV dtype."""
        if self.data_type != torch.float8_e4m3fn:
            if k_pe.dim() == 2:
                k_pe = k_pe.unsqueeze(1)
            return (
                torch.cat([k_nope, k_pe.expand(-1, k_nope.shape[1], -1)], dim=-1).to(
                    self.data_type
                ),
                v.to(self.data_type).contiguous(),
            )
        return mla_kv_pack_quantize_fp8(
            k_nope, k_pe, v, enable_pdl=is_arch_support_pdl()
        )

    def _run_decode_kernel(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        layer: RadixAttention,
    ) -> torch.Tensor:
        if self.data_type == torch.float8_e4m3fn:
            k_scale = getattr(layer, "k_scale_float", None)
            if k_scale is None:
                k_scale = 1.0
        else:
            k_scale = 1.0
        softmax_scale = float(layer.scaling) * float(k_scale)
        output_scale = float(k_scale)

        seq_lens_i32 = (
            seq_lens if seq_lens.dtype == torch.int32 else seq_lens.to(torch.int32)
        )
        return tokenspeed_mla.tokenspeed_mla_decode(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=self._tokenspeed_workspace,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens_i32,
            max_seq_len=int(max_seq_len),
            softmax_scale=softmax_scale,
            output_scale=output_scale,
            enable_pdl=is_arch_support_pdl(),
        )

    def _run_prefill_kernel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        batch_size: int,
        cum_seq_lens_q: torch.Tensor,
        max_q_len: int,
        seq_lens_kv: torch.Tensor,
        cum_seq_lens_kv: torch.Tensor,
        max_kv_len: int,
        is_causal: bool,
        return_lse: bool,
        out_buffer: torch.Tensor,
        o_sf_scale: float = 1.0,
    ):  # Q/K/V arrive already in FP8 via the model-side fused path
        # (prepare_prefill_qkv / pack_prefix_chunk_kv); no quantize here.
        return tokenspeed_mla.tokenspeed_mla_prefill(
            query=q,
            key=k,
            value=v,
            seq_lens=seq_lens_kv,
            cum_seq_lens=cum_seq_lens_kv,
            max_seq_len=int(max_kv_len),
            batch_size=int(batch_size),
            softmax_scale=float(layer.scaling),
            is_causal=is_causal,
            return_lse=return_lse,
            cum_seq_lens_q=cum_seq_lens_q,
            max_seq_len_q=int(max_q_len),
            enable_pdl=is_arch_support_pdl(),
        )


class TokenspeedMLAMultiStepDraftBackend(TRTLLMMLAMultiStepDraftBackend):
    """Multi-step draft backend for tokenspeed_mla used by EAGLE."""

    def __init__(
        self, model_runner: ModelRunner, topk: int, speculative_num_steps: int
    ):
        super().__init__(model_runner, topk, speculative_num_steps)
        # Parent populates self.attn_backends with TRT-LLM instances; replace
        # them with tokenspeed instances sharing the parent's index buffers.
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i] = TokenspeedMLABackend(
                model_runner,
                skip_prefill=True,
                kv_indptr_buf=self.kv_indptr[i],
                q_indptr_decode_buf=self.q_indptr_decode,
            )
