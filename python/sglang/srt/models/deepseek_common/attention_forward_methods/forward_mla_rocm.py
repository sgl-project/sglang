"""AMD/ROCm absorbed multi-latent attention forward path for DeepSeek models.

`AttnForwardMethod.MLA_ROCM` routes here, which keeps every AITER/gfx95 kernel
choice out of the shared `forward_mla.py` and lets non-AMD builds avoid
importing `aiter` altogether.

The BMM absorb steps stay module level to keep ROCm kernel selection localized.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.kernels.ops.quantization.fp8_kernel import (
    fp8_dtype,
    per_token_group_quant_mla_deep_gemm_masked_fp8,
)
from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.attention.dsa.utils import dsa_use_prefill_cp
from sglang.srt.layers.communicator import get_attn_tp_context
from sglang.srt.layers.cp.utils import is_cp_v2_active
from sglang.srt.layers.dcp import (
    all_gather_kv_cache_for_mla_extend,
    all_gather_q_for_mla_decode,
    cp_lse_ag_out_rs_mla,
    dcp_a2a_lse_reduce,
)
from sglang.srt.layers.logits_processor import get_in_autotune_dummy_run
from sglang.srt.layers.quantization.fp8_utils import (
    emit_transposed_bpreshuffle_scale,
    materialize_bpreshuffle_fp8_scale_tuple,
    view_aiter_fused_rms_transposed_fp8_scale_tuple,
)
from sglang.srt.layers.utils.cp_utils import mla_use_prefill_cp
from sglang.srt.lora.deepseek_mla_correction import (
    apply_q_correction as apply_kv_b_lora_q_correction,
)
from sglang.srt.lora.deepseek_mla_correction import (
    apply_v_correction as apply_kv_b_lora_v_correction,
)
from sglang.srt.lora.deepseek_mla_correction import (
    is_kv_b_lora_active,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla import (
    _select_local_dcp_heads_for_autotune,
    is_dcp_mla_decode_phase,
    is_mla_dcp_lse_base_on_e,
    should_defer_dsa_cp_kv_gather,
)
from sglang.srt.models.deepseek_common.utils import (
    FORWARD_ABSORB_CORE_ATTENTION_BACKENDS,
    _is_block_scale_fp8,
    _is_gfx95_supported,
    _use_aiter,
    _use_aiter_bpreshuffle_gfx95,
    _use_aiter_gfx95,
)
from sglang.srt.runtime_context import get_exec, get_parallel
from sglang.srt.state_capturer.indexer_topk import (
    maybe_capture_indexer_topk,
)
from sglang.srt.utils import BumpAllocator, get_bool_env_var

logger = logging.getLogger(__name__)
_SGLANG_EXPERIMENTAL_LORA_OPTI = envs.SGLANG_EXPERIMENTAL_LORA_OPTI.get()

if TYPE_CHECKING:
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA

if _use_aiter:
    # On gfx1250 the aiter `module_fused_qk_norm_rope_cache_quant_shuffle` kernel
    # fails to JIT-build (its `rope_common.h` / `ck_tile/vec_convert.h` are
    # incompatible with this image's composable_kernel), which crashes the very
    # first MLA forward. This path is a pure RMSNorm (quant_type=No), so under the
    # gfx1250 workaround flag (AITER_FORCE_A8W4) substitute a self-contained Triton
    # RMSNorm that never touches the aiter fp4 kernel build.
    if get_bool_env_var("AITER_FORCE_A8W4", "false"):
        if get_bool_env_var("SGLANG_QK_RMSNORM_TORCH", "false"):
            from sglang.srt.models.deepseek_common.attention_forward_methods.triton_qk_rmsnorm import (
                fused_qk_rmsnorm_torch as fused_qk_rmsnorm_bf16,
            )
        else:
            from sglang.srt.models.deepseek_common.attention_forward_methods.triton_qk_rmsnorm import (
                fused_qk_rmsnorm_triton as fused_qk_rmsnorm_bf16,
            )
    else:
        # aiter ROCm/aiter#2958 renamed the public `fused_qk_rmsnorm` in
        # `aiter.ops.fused_qk_norm_rope_cache_quant` to a private `_fused_qk_rmsnorm`
        # and introduced a unified entry point in `aiter.ops.fused_qk_rmsnorm_group_quant`
        # with a different (in-place, kwarg-only, no-return) signature. Probe for the
        # new symbol first so SGLang works with both pre- and post-#2958 aiter without
        # requiring the docker pin to be bumped atomically.
        try:
            from aiter.ops.enum import QuantType as _AiterQuantType
            from aiter.ops.fused_qk_rmsnorm_group_quant import (
                fused_qk_rmsnorm as _aiter_fused_qk_rmsnorm_unified,
            )

            def fused_qk_rmsnorm_bf16(q, q_weight, q_eps, k, k_weight, k_eps):
                q_out = torch.empty_like(q)
                k_out = torch.empty_like(k)
                _aiter_fused_qk_rmsnorm_unified(
                    q_out_quantized=q_out,
                    k_out=k_out,
                    q=q,
                    q_weight=q_weight,
                    q_epsilon=q_eps,
                    k=k,
                    k_weight=k_weight,
                    k_epsilon=k_eps,
                    quant_type=_AiterQuantType.No,
                )
                return q_out, k_out

        except ImportError:
            from aiter.ops.fused_qk_norm_rope_cache_quant import (
                fused_qk_rmsnorm as fused_qk_rmsnorm_bf16,
            )

    from aiter.ops.triton.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
        batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant,
    )

if _use_aiter_gfx95:
    from aiter.ops.triton.fused_fp8_quant import (
        fused_flatten_fp8_group_quant,
        fused_rms_fp8_group_quant,
    )

    from sglang.srt.layers.quantization.rocm_mxfp4_utils import (
        batched_gemm_afp4wfp4_pre_quant,
        fused_flatten_mxfp4_quant,
        fused_rms_mxfp4_quant,
    )
    from sglang.srt.layers.rocm_linear_utils import fused_qk_rope_cat_and_cache_mla


def rocm_absorb_q_bmm(
    attn: DeepseekV2AttentionMLA,
    q_nope: torch.Tensor,
    *,
    is_capture_mode: bool,
) -> torch.Tensor:
    """Absorb ``q_nope @ w_kc`` on HIP/AITER (pre-transpose layout)."""
    # TODO(haishaw): add bmm_fp8 to ROCm
    if _use_aiter_gfx95 and attn.w_kc.dtype == torch.uint8:
        x = q_nope.transpose(0, 1)
        q_nope_out = torch.empty(
            x.shape[0],
            x.shape[1],
            attn.w_kc.shape[2],
            device=x.device,
            dtype=torch.bfloat16,
        )
        batched_gemm_afp4wfp4_pre_quant(
            x,
            attn.w_kc.transpose(-2, -1),
            attn.w_scale_k.transpose(-2, -1),
            torch.bfloat16,
            q_nope_out,
        )
    else:
        if (_use_aiter_gfx95 and attn.w_kc.dtype == torch.float8_e4m3fn) or (
            is_capture_mode and attn.w_kc.dtype == torch.float8_e4m3fnuz
        ):
            # fp8 Triton kernel: always on gfx950,
            # cudagraph-only on gfx942 (hides launch overhead)
            q_nope_out = (
                batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
                    X=q_nope,
                    WQ=attn.w_kc.transpose(-1, -2),
                    w_scale=attn.w_scale,
                    group_size=128,
                    YQ=None,  # allocate (B, M, N)
                    transpose_bm=False,  # (B, M, N)
                    transpose_bm_in=True,  # (M, B, K)
                    dtype=torch.bfloat16,
                )
            )
        else:
            q_nope_out = torch.bmm(
                q_nope.to(torch.bfloat16).transpose(0, 1),
                attn.w_kc.to(torch.bfloat16) * attn.w_scale,
            )
    return q_nope_out


def rocm_absorb_v_bmm(
    attn: DeepseekV2AttentionMLA,
    attn_output: torch.Tensor,
) -> torch.Tensor:
    """Absorb ``attn_output @ w_vc`` (+ optional fused flatten quant) on HIP."""
    # TODO(haishaw): add bmm_fp8 to ROCm
    if _use_aiter_gfx95 and attn.w_vc.dtype == torch.uint8:
        x = attn_output.transpose(0, 1)
        B_heads, M_batch = x.shape[0], x.shape[1]
        N_vdim = attn.w_vc.shape[2]
        # Allocate in (batch, heads, dim) so the post-GEMM
        # transpose+flatten is a free view instead of a copy.
        _bmm_buf = torch.empty(
            M_batch,
            B_heads,
            N_vdim,
            device=x.device,
            dtype=torch.bfloat16,
        )
        attn_bmm_output = _bmm_buf.transpose(0, 1)
        batched_gemm_afp4wfp4_pre_quant(
            x,
            attn.w_vc.transpose(-2, -1),
            attn.w_scale_v.transpose(-2, -1),
            torch.bfloat16,
            attn_bmm_output,
        )
    else:
        _bmm_buf = None
        if _use_aiter_gfx95 and attn.w_kc.dtype == torch.float8_e4m3fn:
            # As in the mxfp4 path above, write (batch, heads, dim) so the
            # post-GEMM flatten is a free view instead of a copy.
            _bmm_buf = torch.empty(
                attn_output.shape[0],
                attn.num_local_heads,
                attn.w_vc.shape[-1],
                device=attn_output.device,
                dtype=torch.bfloat16,
            )
            batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
                X=attn_output,
                WQ=attn.w_vc.transpose(-1, -2),
                w_scale=attn.w_scale,
                group_size=128,
                YQ=_bmm_buf,
                transpose_bm=True,
                transpose_bm_in=True,
                dtype=torch.bfloat16,
            )
        else:
            attn_bmm_output = torch.bmm(
                attn_output.to(torch.bfloat16).transpose(0, 1),
                attn.w_vc.to(torch.bfloat16) * attn.w_scale,
            )

    if _bmm_buf is not None:
        # _bmm_buf is already (batch, heads, dim) contiguous
        if attn.o_proj.weight.dtype == torch.uint8:
            attn_bmm_output = fused_flatten_mxfp4_quant(_bmm_buf)
        elif _is_block_scale_fp8(attn.o_proj):
            # No-copy fp8 scale: emit the bpreshuffle scale already transposed and
            # reinterpret it with a stride swap, instead of relaying out a copy.
            # Falls back to the materialize (copy) path at M == 1 / non-gfx95.
            _emit_bpre = emit_transposed_bpreshuffle_scale(
                _bmm_buf.shape[0],
                on_bpreshuffle_gfx95=_use_aiter_bpreshuffle_gfx95,
            )
            attn_bmm_output = fused_flatten_fp8_group_quant(
                _bmm_buf,
                group_size=128,
                dtype_quant=torch.float8_e4m3fn,
                transpose_scale=_emit_bpre,
            )
            if _emit_bpre:
                attn_bmm_output = view_aiter_fused_rms_transposed_fp8_scale_tuple(
                    attn_bmm_output
                )
            elif _use_aiter_bpreshuffle_gfx95:
                attn_bmm_output = materialize_bpreshuffle_fp8_scale_tuple(
                    attn_bmm_output
                )
        else:
            attn_bmm_output = _bmm_buf.flatten(1, 2)
    elif attn.o_proj.weight.dtype == torch.uint8:
        attn_bmm_output = attn_bmm_output.transpose(0, 1)
        attn_bmm_output = fused_flatten_mxfp4_quant(attn_bmm_output)
    elif _is_block_scale_fp8(attn.o_proj):
        attn_bmm_output = attn_bmm_output.transpose(0, 1)
        # No-copy fp8 scale: emit the bpreshuffle scale already transposed and
        # reinterpret it with a stride swap, instead of relaying out a copy.
        # Falls back to the materialize (copy) path at M == 1 / non-gfx95.
        _emit_bpre = emit_transposed_bpreshuffle_scale(
            attn_bmm_output.shape[0],
            on_bpreshuffle_gfx95=_use_aiter_bpreshuffle_gfx95,
        )
        attn_bmm_output = fused_flatten_fp8_group_quant(
            attn_bmm_output,
            group_size=128,
            dtype_quant=torch.float8_e4m3fn,
            transpose_scale=_emit_bpre,
        )
        if _emit_bpre:
            attn_bmm_output = view_aiter_fused_rms_transposed_fp8_scale_tuple(
                attn_bmm_output
            )
        elif _use_aiter_bpreshuffle_gfx95:
            attn_bmm_output = materialize_bpreshuffle_fp8_scale_tuple(attn_bmm_output)
    else:
        attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)

    return attn_bmm_output


def _fused_rope_cat_and_cache(
    attn: DeepseekV2AttentionMLA,
    q_nope_out: torch.Tensor,
    q_pe: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    positions: torch.Tensor,
    out_cache_loc: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """RoPE + concat + KV-cache write via the AITER fused kernel on gfx95."""
    kv_cache_dtype = (
        fp8_dtype if attn.kv_cache_dtype == "fp8_e4m3" else q_nope_out.dtype
    )
    return fused_qk_rope_cat_and_cache_mla(
        q_nope_out,
        q_pe,
        k_nope,
        k_pe,
        get_token_to_kv_pool().get_key_buffer(attn.attn_mqa.layer_id),
        out_cache_loc,
        positions,
        attn.rotary_emb.cos_cache,
        attn.rotary_emb.sin_cache,
        attn.attn_mqa.k_scale,
        attn.rotary_emb.is_neox_style,
        q_out_dtype=kv_cache_dtype,
    )


class DeepseekMLARocmForwardMixin:

    def forward_absorb_rocm_prepare(
        self: DeepseekV2AttentionMLA,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
        llama_4_scaling: Optional[torch.Tensor] = None,
        prev_topk_indices: Optional[torch.Tensor] = None,
    ):
        from sglang.srt.model_executor.runner import get_is_capture_mode

        q_replicate_active = (
            get_parallel().dcp_replicate_q_proj
            and is_dcp_mla_decode_phase(forward_batch)
            and not self.use_deep_gemm_bmm
            and self.w_kc_qrep is not None
            and self.q_b_proj_qrep_weight is not None
        )
        q_lora = None
        topk_indices = None
        if self.q_lora_rank is not None:
            q, latent_cache = (
                get_attn_tp_context()
                .fetch_qkv_latent()
                .split(
                    [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                    dim=-1,
                )
            )
            k_nope = latent_cache[..., : self.kv_lora_rank]

            # overlap qk norm
            if self.alt_stream is not None and get_is_capture_mode():
                current_stream = torch.cuda.current_stream()
                self.alt_stream.wait_stream(current_stream)
                q = self.q_a_layernorm(q)
                with torch.cuda.stream(self.alt_stream):
                    k_nope = self.kv_a_layernorm(k_nope)
                current_stream.wait_stream(self.alt_stream)
            elif _use_aiter_gfx95 and self.q_b_proj.weight.dtype == torch.uint8:
                q, _, k_nope, *_ = fused_rms_mxfp4_quant(
                    q,
                    self.q_a_layernorm.weight,
                    self.q_a_layernorm.variance_epsilon,
                    k_nope,
                    self.kv_a_layernorm.weight,
                    self.kv_a_layernorm.variance_epsilon,
                )
            elif _use_aiter_gfx95 and _is_block_scale_fp8(self.q_b_proj):
                if self.use_dsa:
                    q_quanted, q_lora, k_nope, _ = fused_rms_fp8_group_quant(
                        q,
                        self.q_a_layernorm.weight,
                        self.q_a_layernorm.variance_epsilon,
                        k_nope,
                        self.kv_a_layernorm.weight,
                        self.kv_a_layernorm.variance_epsilon,
                        group_size=128,
                        dtype_quant=torch.float8_e4m3fn,
                        res1=None,
                        output_unquantized_inp1=True,
                        transpose_scale=False,
                    )
                    if _use_aiter_bpreshuffle_gfx95:
                        q_quanted = materialize_bpreshuffle_fp8_scale_tuple(q_quanted)
                    q = q_quanted
                else:
                    q, _, k_nope, _ = fused_rms_fp8_group_quant(
                        q,
                        self.q_a_layernorm.weight,
                        self.q_a_layernorm.variance_epsilon,
                        k_nope,
                        self.kv_a_layernorm.weight,
                        self.kv_a_layernorm.variance_epsilon,
                        group_size=128,
                        dtype_quant=torch.float8_e4m3fn,
                        res1=None,
                        output_unquantized_inp1=False,
                        transpose_scale=False,
                    )
                    if _use_aiter_bpreshuffle_gfx95:
                        q = materialize_bpreshuffle_fp8_scale_tuple(q)
            elif _use_aiter:
                q, k_nope = fused_qk_rmsnorm_bf16(
                    q,
                    self.q_a_layernorm.weight,
                    self.q_a_layernorm.variance_epsilon,
                    k_nope,
                    self.kv_a_layernorm.weight,
                    self.kv_a_layernorm.variance_epsilon,
                )
            else:
                q = self.q_a_layernorm(q)
                k_nope = self.kv_a_layernorm(k_nope)

            # q_lora needed by indexer
            if self.use_dsa:
                if q_lora is None:
                    q_lora = q

            # overlap q_b_proj and indexer during decode
            if (
                self.alt_stream is not None
                and get_is_capture_mode()
                and forward_batch.forward_mode.is_decode_or_idle()
                and q_lora is not None
                and not q_replicate_active
            ):
                current_stream = torch.cuda.current_stream()
                self.alt_stream.wait_stream(current_stream)
                with torch.cuda.stream(self.alt_stream):
                    k_nope = k_nope.unsqueeze(1)
                    q = self.q_b_proj_forward(q)
                if self.should_run_indexer(prev_topk_indices):
                    topk_indices = self.indexer(
                        x=hidden_states,
                        q_lora=q_lora,
                        positions=positions,
                        forward_batch=forward_batch,
                        layer_id=self.layer_id,
                    )
                else:
                    # skip_topk reuses prev layer's indices; mirror into this
                    # layer's slot so the captured buffer matches what's used.
                    topk_indices = maybe_capture_indexer_topk(
                        self.layer_id, prev_topk_indices
                    )
                current_stream.wait_stream(self.alt_stream)
            else:
                k_nope = k_nope.unsqueeze(1)
                if q_replicate_active:
                    q = torch.nn.functional.linear(q, self.q_b_proj_qrep_weight).view(
                        -1,
                        self.num_local_heads * get_parallel().attn_dcp_size,
                        self.qk_head_dim,
                    )
                else:
                    q = self.q_b_proj_forward(q)

                if q_lora is not None:
                    if self.should_run_indexer(prev_topk_indices):
                        topk_indices = self.indexer(
                            x=hidden_states,
                            q_lora=q_lora,
                            positions=positions,
                            forward_batch=forward_batch,
                            layer_id=self.layer_id,
                        )
                    else:
                        topk_indices = maybe_capture_indexer_topk(
                            self.layer_id, prev_topk_indices
                        )
        else:
            if q_replicate_active:
                q = torch.nn.functional.linear(
                    hidden_states, self.q_b_proj_qrep_weight
                ).view(
                    -1,
                    self.num_local_heads * get_parallel().attn_dcp_size,
                    self.qk_head_dim,
                )
            else:
                q = self.q_proj(hidden_states)[0].view(
                    -1, self.num_local_heads, self.qk_head_dim
                )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
            k_nope = latent_cache[..., : self.kv_lora_rank]
            k_nope = self.kv_a_layernorm(k_nope).unsqueeze(1)

        q_nope, q_pe, k_pe = self._split_q_nope_pe(q, latent_cache)

        if q_replicate_active:
            q_nope_out = (
                torch.bmm(q_nope.transpose(0, 1), self.w_kc_qrep)
                .transpose(0, 1)
                .contiguous()
            )
        else:
            _kvb_q = None
            if _SGLANG_EXPERIMENTAL_LORA_OPTI:
                # Fork the kv_b q-correction A-step onto the LoRA side stream to overlap the bmm.
                from sglang.srt.lora.trtllm_lora_temp.deepseek_mla_correction import (
                    kv_b_lora_q_prepare,
                )

                _kvb_q = kv_b_lora_q_prepare(self, q_nope)

            if self.use_deep_gemm_bmm:
                (
                    q_nope_val,
                    q_nope_scale,
                    masked_m,
                    expected_m,
                    aligned_m,
                ) = per_token_group_quant_mla_deep_gemm_masked_fp8(
                    q_nope.transpose(0, 1)
                )
                q_nope_out = q_nope.new_empty(
                    (self.num_local_heads, aligned_m, self.kv_lora_rank)
                )
                deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
                    (q_nope_val, q_nope_scale),
                    (self.w_kc, self.w_scale_k),
                    q_nope_out,
                    masked_m,
                    expected_m,
                )
                q_nope_out = q_nope_out[:, :expected_m, :]
            else:
                q_nope_out = rocm_absorb_q_bmm(
                    self, q_nope, is_capture_mode=get_is_capture_mode()
                )

            q_nope_out = q_nope_out.transpose(0, 1)
            if _SGLANG_EXPERIMENTAL_LORA_OPTI:
                from sglang.srt.lora.trtllm_lora_temp.deepseek_mla_correction import (
                    kv_b_lora_q_apply,
                )

                q_nope_out = kv_b_lora_q_apply(self, q_nope, q_nope_out, _kvb_q)
            elif is_kv_b_lora_active(self):
                q_nope_out = apply_kv_b_lora_q_correction(self, q_nope, q_nope_out)

        fuse_rope_for_trtllm_mla = self._fuse_rope_for_trtllm_mla(forward_batch)
        if (
            self.rotary_emb is not None
            and (not fuse_rope_for_trtllm_mla)
            and (not self._skip_rope_for_dsa_tilelang_fused())
            and (not self._skip_rope_for_aiter_fused_mla())
            and (
                not _use_aiter
                or not _is_gfx95_supported
                or self.use_dsa
                # Non-fused, non-specialized attention backends (e.g. Triton) run
                # the cat path in forward_absorb_core and need RoPE applied here;
                # only the aiter fused MLA path and the specialized MLA backends
                # defer RoPE to their own kernels.
                or (
                    self.current_attention_backend
                    not in FORWARD_ABSORB_CORE_ATTENTION_BACKENDS
                    and self.current_attention_backend != "aiter"
                )
            )
        ):
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

        dsa_prefill_cp = dsa_use_prefill_cp(forward_batch)
        mla_prefill_cp = mla_use_prefill_cp(forward_batch)
        defer_kv_gather_until_after_rope = should_defer_dsa_cp_kv_gather(
            dsa_prefill_cp=dsa_prefill_cp,
            fuse_rope_for_trtllm_mla=fuse_rope_for_trtllm_mla,
        )
        if dsa_prefill_cp and not defer_kv_gather_until_after_rope:
            from sglang.srt.layers.attention.dsa_backend import materialize_full_kv_cp

            k_nope, k_pe = materialize_full_kv_cp(
                self,
                forward_batch,
                latent_cache,
                k_nope,
                k_pe,
            )
        elif mla_prefill_cp and not is_cp_v2_active(forward_batch):
            # CP-v1 gathers the latent here; CP-v2 gathers it in the attention
            # backend via the strategy (materialize_full_mla_kv).
            k_nope, k_pe = self.rebuild_cp_kv_cache(
                latent_cache,
                forward_batch,
                k_nope,
                k_pe,
            )

        # all_gather q_pe, q_nope_out,take tp8 as an example， q_pe [B, H, ROPE_DIM], q_nope_out [B, H, NOPE_DIM] gathered to [B, H * dcp_world_size, ROPE_DIM] [B, H * dcp_world_size, NOPE_DIM] for decode batch, and all gather k_pe, k_nope for extend batch.
        if get_parallel().dcp_enabled:
            if is_dcp_mla_decode_phase(forward_batch):
                if not q_replicate_active:
                    q_nope_out, q_pe = all_gather_q_for_mla_decode(
                        q_nope_out=q_nope_out,
                        q_pe=q_pe,
                    )
            elif forward_batch.forward_mode.is_extend():
                # for extend, gather kv
                all_gather_kv_cache_for_mla_extend(
                    get_token_to_kv_pool(),
                    self.attn_mqa,
                    forward_batch.extend_prefix_lens_cpu,
                    forward_batch.attn_dcp_metadata.dcp_local_prefix_kv_indices,
                    forward_batch.attn_dcp_metadata.dcp_extend_prefix_lens_sum,
                    forward_batch.attn_dcp_metadata.dcp_kv_buffer,
                    self.kv_lora_rank,
                    k_nope,
                    k_pe,
                )
            else:
                logger.warning(
                    f"not supported forward_mode {forward_batch.forward_mode}"
                )

        return (
            q_pe,
            k_pe,
            q_nope_out,
            k_nope,
            forward_batch,
            zero_allocator,
            positions,
            topk_indices,
            llama_4_scaling,
        )

    def forward_absorb_rocm_core(
        self: DeepseekV2AttentionMLA,
        q_pe,
        k_pe,
        q_nope_out,
        k_nope,
        forward_batch,
        zero_allocator,
        positions,
        topk_indices,
        llama_4_scaling,
    ):
        save_kv_cache = True

        if self.current_attention_backend in FORWARD_ABSORB_CORE_ATTENTION_BACKENDS:
            if self._skip_rope_for_dsa_tilelang_fused() and self.rotary_emb is not None:
                q_cat, _, k_pe_fused, _ = _fused_rope_cat_and_cache(
                    self,
                    q_nope_out,
                    q_pe,
                    k_nope,
                    k_pe,
                    positions,
                    forward_batch.out_cache_loc,
                )
                save_kv_cache = False
                # On decode, pass q_cat directly to attn_mqa with q_rope=None so
                # dsa_backend.forward_decode reuses q_cat as a zero-copy view
                # (`q.contiguous().view(...)` fast-path) instead of running the
                # redundant `concat_mla_absorb_q_general(q_nope_fused, q_pe_fused)`
                # that would otherwise rebuild a tensor byte-identical to q_cat.
                # On ROCm tilelang decode, this eliminates the
                # `CatArrayBatchedCopy<OpaqueType<1u>, ...>` kernel that used to
                # fire once per layer per decode step (~2.6 us / layer saved).
                # Prefill keeps the split form because dsa_backend.forward_extend
                # asserts `q_rope is not None`.
                if forward_batch.forward_mode.is_decode_or_idle():
                    if llama_4_scaling is not None:
                        # llama_4_scaling applies only to the q_nope portion;
                        # mutate in place via the slice view of q_cat.
                        q_cat[..., : self.kv_lora_rank] *= llama_4_scaling
                    attn_output = self.attn_mqa(
                        q_cat,
                        None,
                        None,
                        forward_batch,
                        q_rope=None,
                        k_rope=k_pe_fused,
                        save_kv_cache=save_kv_cache,
                        **(
                            dict(topk_indices=topk_indices)
                            if topk_indices is not None
                            else {}
                        ),
                    )
                else:
                    q_nope_fused = q_cat[..., : self.kv_lora_rank]
                    q_pe_fused = q_cat[..., self.kv_lora_rank :]
                    if llama_4_scaling is not None:
                        q_nope_fused *= llama_4_scaling
                    attn_output = self.attn_mqa(
                        q_nope_fused,
                        None,
                        None,
                        forward_batch,
                        q_rope=q_pe_fused,
                        k_rope=k_pe_fused,
                        save_kv_cache=save_kv_cache,
                        **(
                            dict(topk_indices=topk_indices)
                            if topk_indices is not None
                            else {}
                        ),
                    )
            else:
                extra_args = {}
                if self._fuse_rope_for_trtllm_mla(forward_batch):
                    extra_args = {
                        "cos_sin_cache": self.rotary_emb.cos_sin_cache,
                        "is_neox": self.rotary_emb.is_neox_style,
                        "llama_4_scaling": llama_4_scaling,
                    }
                if is_dcp_mla_decode_phase(forward_batch):
                    # set return_lse=True to correct attn_output
                    attn_output, lse = self.attn_mqa_for_dcp_decode(
                        q_nope_out,
                        k_nope,
                        k_nope,
                        forward_batch,
                        q_rope=q_pe,
                        k_rope=k_pe,
                        **extra_args,
                        **(
                            dict(topk_indices=topk_indices)
                            if topk_indices is not None
                            else {}
                        ),
                    )
                else:
                    attn_output = self.attn_mqa(
                        q_nope_out,
                        k_nope,
                        k_nope,
                        forward_batch,
                        q_rope=q_pe,
                        k_rope=k_pe,
                        **extra_args,
                        **(
                            dict(topk_indices=topk_indices)
                            if topk_indices is not None
                            else {}
                        ),
                    )
        else:
            if self._skip_rope_for_aiter_fused_mla():
                q, _, _, k = _fused_rope_cat_and_cache(
                    self,
                    q_nope_out,
                    q_pe,
                    k_nope,
                    k_pe,
                    positions,
                    forward_batch.out_cache_loc,
                )
                save_kv_cache = False
            else:
                q = torch.cat([q_nope_out, q_pe], dim=-1)
                k = torch.cat([k_nope, k_pe], dim=-1)

            # Apply llama 4 scaling if provided
            if llama_4_scaling is not None:
                q *= llama_4_scaling

            attn_output = self.attn_mqa(
                q,
                k,
                k_nope,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **(dict(topk_indices=topk_indices) if topk_indices is not None else {}),
            )

        # correct attn_output with respect to lse from other ranks
        if is_dcp_mla_decode_phase(forward_batch):
            attn_output = attn_output.view(
                -1,
                self.num_local_heads * get_parallel().attn_dcp_size,
                self.kv_lora_rank,
            )
            if get_in_autotune_dummy_run():
                # The synthetic FlashInfer MoE autotune pass discards model
                # outputs. Avoid an unnecessary cross-node MNNVL exchange of
                # zero attention partials.
                attn_output = _select_local_dcp_heads_for_autotune(
                    attn_output, self.num_local_heads
                )
            else:
                dcp_comm_backend = get_parallel().dcp_comm_backend
                is_lse_base_on_e = is_mla_dcp_lse_base_on_e(
                    self.current_attention_backend
                )
                if dcp_comm_backend in ("a2a", "fi_a2a"):
                    # A2A exchange of head partials + LSE, then local Triton combine.
                    attn_output = dcp_a2a_lse_reduce(
                        attn_output.contiguous(),
                        lse.contiguous(),
                        get_parallel().dcp_group,
                        is_lse_base_on_e=is_lse_base_on_e,
                        comm_backend=dcp_comm_backend,
                    )
                else:
                    attn_output = cp_lse_ag_out_rs_mla(
                        attn_output,
                        lse,
                        get_parallel().dcp_group,
                        is_lse_base_on_e=is_lse_base_on_e,
                    )
                    attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        _kvb_v = None
        if _SGLANG_EXPERIMENTAL_LORA_OPTI:
            # Fork the kv_b v-correction A-step onto the LoRA side stream to overlap the bmm.
            from sglang.srt.lora.trtllm_lora_temp.deepseek_mla_correction import (
                kv_b_lora_v_prepare,
            )

            _kvb_v = kv_b_lora_v_prepare(self, attn_output)

        if self.use_deep_gemm_bmm:
            (
                attn_output_val,
                attn_output_scale,
                masked_m,
                expected_m,
                aligned_m,
            ) = per_token_group_quant_mla_deep_gemm_masked_fp8(
                attn_output.transpose(0, 1)
            )
            attn_bmm_output = attn_output.new_empty(
                (self.num_local_heads, aligned_m, self.v_head_dim)
            )
            deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
                (attn_output_val, attn_output_scale),
                (self.w_vc, self.w_scale_v),
                attn_bmm_output,
                masked_m,
                expected_m,
            )
            attn_bmm_output = (
                attn_bmm_output[:, :expected_m, :].transpose(0, 1).flatten(1, 2)
            )
        else:
            attn_bmm_output = rocm_absorb_v_bmm(self, attn_output)

        if _SGLANG_EXPERIMENTAL_LORA_OPTI:
            from sglang.srt.lora.trtllm_lora_temp.deepseek_mla_correction import (
                kv_b_lora_v_apply,
            )

            attn_bmm_output = kv_b_lora_v_apply(
                self, attn_output, attn_bmm_output, _kvb_v
            )
        elif is_kv_b_lora_active(self):
            attn_bmm_output = apply_kv_b_lora_v_correction(
                self, attn_output, attn_bmm_output
            )
        output, _ = self.o_proj(attn_bmm_output)

        if self.next_skip_topk is None:
            return output

        # Return topk_indices for the next layer when enabling index cache
        if not self.next_skip_topk:
            return output, None
        else:
            return output, topk_indices

    def _skip_rope_for_dsa_tilelang_fused(self: DeepseekV2AttentionMLA) -> bool:
        """
        Check if we should skip rope and use fused rope+cache path for TileLang DSA on gfx95.
        """
        return (
            _use_aiter_gfx95
            and self.current_attention_backend in ("dsa", "nsa")
            and (
                get_exec().kernel.dsa_decode_backend == "tilelang"
                or get_exec().kernel.dsa_prefill_backend == "tilelang"
            )
        )

    def _skip_rope_for_aiter_fused_mla(self: DeepseekV2AttentionMLA) -> bool:
        """
        Skip rope in prepare and let the fused kernel in forward_absorb_rocm_core handle it,
        when running aiter-backend MLA on gfx95 (i.e., the `else` branch in
        forward_absorb_rocm_core that calls fused_qk_rope_cat_and_cache_mla).

        A layer without a rotary_emb has nothing to fuse: that branch reads
        rotary_emb.cos_cache, so skipping the standalone rope there ends in
        AttributeError on None. Kimi-K3 has such layers.
        """
        return (
            _use_aiter_gfx95
            and self.current_attention_backend == "aiter"
            and self.rotary_emb is not None
        )
