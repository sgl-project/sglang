from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.quantization.fp8_kernel import (
    fp8_dtype,
    per_tensor_quant_mla_fp8,
    per_token_group_quant_mla_deep_gemm_masked_fp8,
)
from sglang.srt.compilation.compilation_config import register_split_op
from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.attention.dsa.utils import (
    dsa_use_prefill_cp,
    is_graph_dsa_split_op_surface,
)
from sglang.srt.layers.communicator import get_attn_tp_context
from sglang.srt.layers.cp.utils import is_cp_v2_active
from sglang.srt.layers.dcp import (
    all_gather_kv_cache_for_mla_extend,
    all_gather_q_for_mla_decode,
    cp_lse_ag_out_rs_mla,
)
from sglang.srt.layers.quantization.fp8_utils import (
    materialize_bpreshuffle_fp8_scale_tuple,
)
from sglang.srt.layers.radix_attention import unified_attention_with_output
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
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.models.deepseek_common.utils import (
    FORWARD_ABSORB_CORE_ATTENTION_BACKENDS,
    _is_cpu,
    _is_cublas_ge_129,
    _is_cuda,
    _is_gfx95_supported,
    _is_hip,
    _is_musa,
    _use_aiter,
    _use_aiter_bpreshuffle_gfx95,
    _use_aiter_gfx95,
)
from sglang.srt.runtime_context import get_parallel, get_server_args
from sglang.srt.state_capturer.indexer_topk import (
    maybe_capture_indexer_topk,
)
from sglang.srt.utils import BumpAllocator
from sglang.srt.utils.custom_op import register_custom_op

logger = logging.getLogger(__name__)
_SGLANG_EXPERIMENTAL_LORA_OPTI = envs.SGLANG_EXPERIMENTAL_LORA_OPTI.get()

if TYPE_CHECKING:
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA


@dataclass(frozen=True)
class MlaBmmFusionPlan:
    q_nope_t: torch.Tensor
    q_nope_out_buf: torch.Tensor
    q_nope_out_view: torch.Tensor
    attn_output_buf: torch.Tensor


if _is_cuda:
    from sgl_kernel import bmm_fp8 as _raw_bmm_fp8

    # TODO(yuwei): remove this wrapper after sgl-kernel registers its own fake/meta impl
    # Wrap bmm_fp8 as a custom op so torch.compile does not trace into
    # torch.cuda.current_blas_handle() (which returns a non-Tensor).
    @register_custom_op(mutates_args=["out"])
    def _bmm_fp8_op(
        A: torch.Tensor,
        B: torch.Tensor,
        out: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
    ) -> None:
        _raw_bmm_fp8(A, B, A_scale, B_scale, out.dtype, out)

    def bmm_fp8(A, B, A_scale, B_scale, dtype, out=None):
        if out is None:
            out = torch.empty(
                (A.shape[0], A.shape[1], B.shape[2]),
                device=A.device,
                dtype=dtype,
            )
        _bmm_fp8_op(A, B, out, A_scale, B_scale)
        return out


if _use_aiter:
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


def _should_defer_dsa_cp_kv_gather(
    *,
    dsa_prefill_cp: bool,
    fuse_rope_for_trtllm_mla: bool,
) -> bool:
    return dsa_prefill_cp and fuse_rope_for_trtllm_mla


class DeepseekMLAForwardMixin:
    def init_mla_forward(self: DeepseekV2AttentionMLA):
        self.flashinfer_mla_disable_ragged = (
            get_server_args().flashinfer_mla_disable_ragged
        )

    def should_run_indexer(
        self: DeepseekV2AttentionMLA,
        prev_topk_indices: Optional[torch.Tensor] = None,
    ) -> bool:
        """Whether this layer runs its own indexer vs reusing carried topk.

        skip_topk (shared) layers carry no indexer weights in the checkpoint,
        so they must reuse the carried topk and never run the indexer. Do NOT
        widen this to `or prev_topk_indices is None` (the upstream gate): that
        recomputes with an uninitialized indexer whenever cross-layer
        propagation is unavailable (e.g. the TBO op path drops topk_indices),
        reintroducing the >index_topk garbling. The is_nextn clause is the
        sole intentional fallback (the NextN layer has its own weights).

        Eager-MHA prefill calls this with no argument: it needs no topk for
        the current forward, but producer layers must still fill their indexer
        K cache for later MLA/decode; shared layers' cache is never read, so
        filling it is dead work.
        """
        return not self.skip_topk or (self.is_nextn and prev_topk_indices is None)

    def _can_fuse_bmm_into_attention(
        self: DeepseekV2AttentionMLA, forward_batch: ForwardBatch
    ) -> bool:
        # Shared activation surface with the DSA indexer graph dispatch
        # (in piecewise/breakable graph + non-speculative extend). Like the indexer
        # dispatch, this fusion is on by default on that surface.
        if not is_graph_dsa_split_op_surface(forward_batch):
            return False
        if not self.use_dsa:
            return False
        if self.use_deep_gemm_bmm or _is_hip:
            return False
        if is_kv_b_lora_active(self):
            return False
        # The isolated 1-kernel graph is the bf16 fallback BMM. The fp8 and
        # DeepGEMM branches already use different fused paths.
        if self.w_kc.dtype == torch.float8_e4m3fn:
            return False
        if self.current_attention_backend not in FORWARD_ABSORB_CORE_ATTENTION_BACKENDS:
            return False
        return True

    def _split_q_nope_pe(
        self: DeepseekV2AttentionMLA,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        k_pe = latent_cache[..., self.kv_lora_rank :].unsqueeze(1)
        return q_nope, q_pe, k_pe

    def _make_mla_bmm_fusion_plan(
        self: DeepseekV2AttentionMLA,
        q: torch.Tensor,
        q_nope: torch.Tensor,
    ) -> MlaBmmFusionPlan:
        q_nope_out_buf = q.new_empty(
            (
                self.num_local_heads,
                q.shape[0],
                self.kv_lora_rank,
            )
        )
        q_nope_out_view = q_nope_out_buf.transpose(0, 1)
        attn_output_buf = q.new_empty(
            (
                q.shape[0],
                self.num_local_heads * self.kv_lora_rank,
            )
        )
        return MlaBmmFusionPlan(
            q_nope_t=q_nope.transpose(0, 1),
            q_nope_out_buf=q_nope_out_buf,
            q_nope_out_view=q_nope_out_view,
            attn_output_buf=attn_output_buf,
        )

    def forward_absorb_prepare(
        self: DeepseekV2AttentionMLA,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
        llama_4_scaling: Optional[torch.Tensor] = None,
        prev_topk_indices: Optional[torch.Tensor] = None,
    ):
        from sglang.srt.model_executor.runner import get_is_capture_mode

        fuse_bmm_attention = (
            self.q_lora_rank is not None
            and self._can_fuse_bmm_into_attention(forward_batch)
        )
        q_lora = None
        topk_indices = None
        q_nope = None
        q_pe = None
        k_pe = None
        fusion_plan: Optional[MlaBmmFusionPlan] = None
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
            else:
                if _use_aiter_gfx95 and self.q_b_proj.weight.dtype == torch.uint8:
                    q, _, k_nope, *_ = fused_rms_mxfp4_quant(
                        q,
                        self.q_a_layernorm.weight,
                        self.q_a_layernorm.variance_epsilon,
                        k_nope,
                        self.kv_a_layernorm.weight,
                        self.kv_a_layernorm.variance_epsilon,
                    )
                else:
                    q_lora = None
                    if (
                        _use_aiter_gfx95
                        and self.q_b_proj.weight.dtype == torch.float8_e4m3fn
                    ):
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
                                q_quanted = materialize_bpreshuffle_fp8_scale_tuple(
                                    q_quanted
                                )
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
                q_lora is not None
                and self.alt_stream is not None
                and get_is_capture_mode()
                and forward_batch.forward_mode.is_decode_or_idle()
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
                q = self.q_b_proj_forward(q)

                # Hoist these above the DSA indexer split op so the indexer
                # and the composite bmm+attention split op are adjacent in FX.
                if fuse_bmm_attention:
                    q_nope, q_pe, k_pe = self._split_q_nope_pe(q, latent_cache)
                    fusion_plan = self._make_mla_bmm_fusion_plan(q, q_nope)

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
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
            k_nope = latent_cache[..., : self.kv_lora_rank]
            k_nope = self.kv_a_layernorm(k_nope).unsqueeze(1)

        if q_nope is None:
            q_nope, q_pe, k_pe = self._split_q_nope_pe(q, latent_cache)

        _kvb_q = None
        if fusion_plan is not None:
            # The composite split op fills q_nope_out_buf and attention reads
            # this transposed alias directly.
            q_nope_out = fusion_plan.q_nope_out_view
        else:
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
            elif _is_hip:
                # TODO(haishaw): add bmm_fp8 to ROCm
                if _use_aiter_gfx95 and self.w_kc.dtype == torch.uint8:
                    x = q_nope.transpose(0, 1)
                    q_nope_out = torch.empty(
                        x.shape[0],
                        x.shape[1],
                        self.w_kc.shape[2],
                        device=x.device,
                        dtype=torch.bfloat16,
                    )
                    batched_gemm_afp4wfp4_pre_quant(
                        x,
                        self.w_kc.transpose(-2, -1),
                        self.w_scale_k.transpose(-2, -1),
                        torch.bfloat16,
                        q_nope_out,
                    )
                else:
                    if (
                        _use_aiter_gfx95 and self.w_kc.dtype == torch.float8_e4m3fn
                    ) or (
                        get_is_capture_mode()
                        and self.w_kc.dtype == torch.float8_e4m3fnuz
                    ):
                        # fp8 Triton kernel: always on gfx950,
                        # cudagraph-only on gfx942 (hides launch overhead)
                        q_nope_out = batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
                            X=q_nope,
                            WQ=self.w_kc.transpose(-1, -2),
                            w_scale=self.w_scale,
                            group_size=128,
                            YQ=None,  # allocate (B, M, N)
                            transpose_bm=False,  # (B, M, N)
                            transpose_bm_in=True,  # (M, B, K)
                            dtype=torch.bfloat16,
                        )

                    else:
                        q_nope_out = torch.bmm(
                            q_nope.to(torch.bfloat16).transpose(0, 1),
                            self.w_kc.to(torch.bfloat16) * self.w_scale,
                        )

            elif self.w_kc.dtype == torch.float8_e4m3fn:
                if _is_cpu:
                    q_nope_out = torch.bmm(
                        q_nope.to(torch.bfloat16).transpose(0, 1),
                        self.w_kc.to(torch.bfloat16) * self.w_scale,
                    )
                else:
                    # fix bmm_fp8 error under cublas12.9 caused by bumpallocator, detail in pr#11612
                    q_nope_val, q_nope_scale = per_tensor_quant_mla_fp8(
                        q_nope.transpose(0, 1),
                        (
                            torch.zeros((1,), dtype=torch.float32, device=q_nope.device)
                            if _is_cublas_ge_129
                            else zero_allocator.allocate(1)
                        ),
                    )
                    q_nope_out = bmm_fp8(
                        q_nope_val,
                        self.w_kc,
                        q_nope_scale,
                        self.w_scale,
                        torch.bfloat16,
                    )
            else:
                q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)

            q_nope_out = q_nope_out.transpose(0, 1)
            if _SGLANG_EXPERIMENTAL_LORA_OPTI:
                from sglang.srt.lora.trtllm_lora_temp.deepseek_mla_correction import (
                    kv_b_lora_q_apply,
                )

                q_nope_out = kv_b_lora_q_apply(self, q_nope, q_nope_out, _kvb_q)
            elif is_kv_b_lora_active(self):
                q_nope_out = apply_kv_b_lora_q_correction(self, q_nope, q_nope_out)

        fuse_rope_for_trtllm_mla = self._fuse_rope_for_trtllm_mla(forward_batch)
        skip_rope_for_dsa_tilelang_fused = self._skip_rope_for_dsa_tilelang_fused()
        skip_rope_for_aiter_fused_mla = self._skip_rope_for_aiter_fused_mla()
        if (
            self.rotary_emb is not None
            and (not fuse_rope_for_trtllm_mla)
            and (not skip_rope_for_dsa_tilelang_fused)
            and (not skip_rope_for_aiter_fused_mla)
            and (
                not _use_aiter
                or not _is_gfx95_supported
                or self.use_dsa
                or self.current_attention_backend == "triton"
            )
        ):
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

        dsa_prefill_cp = dsa_use_prefill_cp(forward_batch)
        mla_prefill_cp = mla_use_prefill_cp(forward_batch)
        defer_kv_gather_until_after_rope = _should_defer_dsa_cp_kv_gather(
            dsa_prefill_cp=dsa_prefill_cp,
            fuse_rope_for_trtllm_mla=fuse_rope_for_trtllm_mla,
        )
        if (
            (dsa_prefill_cp or mla_prefill_cp)
            and not defer_kv_gather_until_after_rope
            and not is_cp_v2_active(forward_batch)
        ):
            # CP-v1 gathers the latent here; CP-v2 gathers it in the attention
            # backend via the strategy (materialize_full_mla_kv).
            k_nope, k_pe = self.rebuild_cp_kv_cache(
                latent_cache, forward_batch, k_nope, k_pe
            )

        # all_gather q_pe, q_nope_out,take tp8 as an example， q_pe [B, H, ROPE_DIM], q_nope_out [B, H, NOPE_DIM] gathered to [B, H * dcp_world_size, ROPE_DIM] [B, H * dcp_world_size, NOPE_DIM] for decode batch, and all gather k_pe, k_nope for extend batch.
        if get_parallel().dcp_enabled:
            if forward_batch.forward_mode.is_decode():
                # if forward_batch.forward_mode is decode, gather q
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
            fusion_plan,
        )

    def forward_absorb_core(
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
        fusion_plan: Optional[MlaBmmFusionPlan] = None,
    ):
        save_kv_cache = True

        if self.current_attention_backend in FORWARD_ABSORB_CORE_ATTENTION_BACKENDS:
            if self._skip_rope_for_dsa_tilelang_fused() and self.rotary_emb is not None:
                cos = self.rotary_emb.cos_cache
                sin = self.rotary_emb.sin_cache
                kv_cache_dtype = (
                    fp8_dtype if self.kv_cache_dtype == "fp8_e4m3" else q_nope_out.dtype
                )
                q_cat, _, k_pe_fused, _ = fused_qk_rope_cat_and_cache_mla(
                    q_nope_out,
                    q_pe,
                    k_nope,
                    k_pe,
                    get_token_to_kv_pool().get_key_buffer(self.attn_mqa.layer_id),
                    forward_batch.out_cache_loc,
                    positions,
                    cos,
                    sin,
                    self.attn_mqa.k_scale,
                    self.rotary_emb.is_neox_style,
                    q_out_dtype=kv_cache_dtype,
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
                if fusion_plan is not None:
                    bmm_attention_fn = (
                        bcg_mla_bmm_then_unified_attention
                        if is_in_breakable_cuda_graph()
                        else mla_bmm_then_unified_attention
                    )
                    bmm_attention_fn(
                        fusion_plan.q_nope_t,
                        self.w_kc,
                        fusion_plan.q_nope_out_buf,
                        q_nope_out,
                        k_nope,
                        fusion_plan.attn_output_buf,
                        save_kv_cache,
                        self.layer_id,
                        q_pe,
                        k_pe,
                        cos_sin_cache=extra_args.get("cos_sin_cache"),
                        is_neox=extra_args.get("is_neox"),
                        llama_4_scaling=extra_args.get("llama_4_scaling"),
                        topk_indices=topk_indices,
                    )
                    attn_output = fusion_plan.attn_output_buf
                elif (
                    forward_batch.forward_mode.is_decode()
                    and get_parallel().dcp_enabled
                ):
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
            if _use_aiter_gfx95 and self.current_attention_backend == "aiter":
                cos = self.rotary_emb.cos_cache
                sin = self.rotary_emb.sin_cache

                kv_cache_dtype = (
                    fp8_dtype if self.kv_cache_dtype == "fp8_e4m3" else q_nope_out.dtype
                )

                q, _, _, k = fused_qk_rope_cat_and_cache_mla(
                    q_nope_out,
                    q_pe,
                    k_nope,
                    k_pe,
                    get_token_to_kv_pool().get_key_buffer(self.attn_mqa.layer_id),
                    forward_batch.out_cache_loc,
                    positions,
                    cos,
                    sin,
                    self.attn_mqa.k_scale,
                    self.rotary_emb.is_neox_style,
                    q_out_dtype=kv_cache_dtype,
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
        if forward_batch.forward_mode.is_decode() and get_parallel().dcp_enabled:
            attn_output = attn_output.view(
                -1,
                self.num_local_heads * get_parallel().attn_dcp_size,
                self.kv_lora_rank,
            )
            attn_output = cp_lse_ag_out_rs_mla(
                attn_output, lse, get_parallel().dcp_group
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
        elif _is_hip:
            # TODO(haishaw): add bmm_fp8 to ROCm
            if _use_aiter_gfx95 and self.w_vc.dtype == torch.uint8:
                x = attn_output.transpose(0, 1)
                B_heads, M_batch = x.shape[0], x.shape[1]
                N_vdim = self.w_vc.shape[2]
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
                    self.w_vc.transpose(-2, -1),
                    self.w_scale_v.transpose(-2, -1),
                    torch.bfloat16,
                    attn_bmm_output,
                )
            else:
                _bmm_buf = None
                if _use_aiter_gfx95 and self.w_kc.dtype == torch.float8_e4m3fn:
                    attn_bmm_output = batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
                        X=attn_output,
                        WQ=self.w_vc.transpose(-1, -2),
                        w_scale=self.w_scale,
                        group_size=128,
                        YQ=None,
                        transpose_bm=False,
                        transpose_bm_in=True,
                        dtype=torch.bfloat16,
                    )
                else:
                    attn_bmm_output = torch.bmm(
                        attn_output.to(torch.bfloat16).transpose(0, 1),
                        self.w_vc.to(torch.bfloat16) * self.w_scale,
                    )

            if _bmm_buf is not None:
                # _bmm_buf is already (batch, heads, dim) contiguous
                if self.o_proj.weight.dtype == torch.uint8:
                    attn_bmm_output = fused_flatten_mxfp4_quant(_bmm_buf)
                elif self.o_proj.weight.dtype == torch.float8_e4m3fn:
                    attn_bmm_output = fused_flatten_fp8_group_quant(
                        _bmm_buf,
                        group_size=128,
                        dtype_quant=torch.float8_e4m3fn,
                        transpose_scale=False,
                    )
                    if _use_aiter_bpreshuffle_gfx95:
                        attn_bmm_output = materialize_bpreshuffle_fp8_scale_tuple(
                            attn_bmm_output
                        )
                else:
                    attn_bmm_output = _bmm_buf.flatten(1, 2)
            elif self.o_proj.weight.dtype == torch.uint8:
                attn_bmm_output = attn_bmm_output.transpose(0, 1)
                attn_bmm_output = fused_flatten_mxfp4_quant(attn_bmm_output)
            elif self.o_proj.weight.dtype == torch.float8_e4m3fn:
                attn_bmm_output = attn_bmm_output.transpose(0, 1)
                attn_bmm_output = fused_flatten_fp8_group_quant(
                    attn_bmm_output,
                    group_size=128,
                    dtype_quant=torch.float8_e4m3fn,
                    transpose_scale=False,
                )
                if _use_aiter_bpreshuffle_gfx95:
                    attn_bmm_output = materialize_bpreshuffle_fp8_scale_tuple(
                        attn_bmm_output
                    )
            else:
                attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)

        elif self.w_vc.dtype == torch.float8_e4m3fn:
            if _is_cpu:
                attn_bmm_output = torch.bmm(
                    attn_output.to(torch.bfloat16).transpose(0, 1),
                    self.w_vc.to(torch.bfloat16) * self.w_scale,
                )
                attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
            else:
                attn_output_val, attn_output_scale = per_tensor_quant_mla_fp8(
                    attn_output.transpose(0, 1),
                    (
                        torch.zeros(
                            (1,), dtype=torch.float32, device=attn_output.device
                        )
                        if _is_cublas_ge_129
                        else zero_allocator.allocate(1)
                    ),
                )
                attn_bmm_output = bmm_fp8(
                    attn_output_val,
                    self.w_vc,
                    attn_output_scale,
                    self.w_scale,
                    torch.bfloat16,
                )
                attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
        elif _is_musa:
            attn_bmm_output = torch.bmm(
                attn_output.to(torch.bfloat16).transpose(0, 1), self.w_vc
            )
            attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
        else:
            if is_in_tc_piecewise_cuda_graph():
                # torch dynamo requires out= op was called where output tensor was non-contiguous
                attn_bmm_output = (
                    torch.bmm(attn_output.transpose(0, 1), self.w_vc)
                    .transpose(0, 1)
                    .flatten(1, 2)
                )
            else:
                attn_bmm_output = torch.empty(
                    (attn_output.shape[0], self.num_local_heads * self.v_head_dim),
                    dtype=attn_output.dtype,
                    device=attn_output.device,
                )
                torch.bmm(
                    attn_output.transpose(0, 1),
                    self.w_vc,
                    out=attn_bmm_output.view(
                        -1, self.num_local_heads, self.v_head_dim
                    ).transpose(0, 1),
                )
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

    def _fuse_rope_for_trtllm_mla(
        self: DeepseekV2AttentionMLA, forward_batch: ForwardBatch
    ) -> bool:
        """
        Check if we should skip rope and do fused rope+quantize for TRTLLM MLA decode in fp8_e4m3 path.
        """
        if self.current_attention_backend in ("dsa", "nsa"):
            return (
                get_server_args().dsa_decode_backend == "trtllm"
                or get_server_args().dsa_prefill_backend == "trtllm"
            ) and get_attn_backend().kv_cache_dtype == torch.float8_e4m3fn

        return (
            self.current_attention_backend
            in ("trtllm_mla", "tokenspeed_mla", "cutedsl_mla")
            and (
                forward_batch.forward_mode.is_decode_or_idle()
                or forward_batch.forward_mode.is_target_verify()
            )
            and get_attn_backend().data_type == torch.float8_e4m3fn
        )

    def _skip_rope_for_dsa_tilelang_fused(self: DeepseekV2AttentionMLA) -> bool:
        """
        Check if we should skip rope and use fused rope+cache path for TileLang DSA on gfx95.
        """
        server_args = get_server_args()
        return (
            _use_aiter_gfx95
            and self.current_attention_backend in ("dsa", "nsa")
            and (
                server_args.dsa_decode_backend == "tilelang"
                or server_args.dsa_prefill_backend == "tilelang"
            )
        )

    def _skip_rope_for_aiter_fused_mla(self: DeepseekV2AttentionMLA) -> bool:
        """
        Skip rope in prepare and let the fused kernel in forward_absorb_core handle it,
        when running aiter-backend MLA on gfx95 (i.e., the `else` branch in forward_absorb_core
        that calls fused_qk_rope_cat_and_cache_mla).
        """
        return _use_aiter_gfx95 and self.current_attention_backend == "aiter"


# Fuses the absorb BMM (`q_nope @ w_kc`) with `unified_attention_with_output`
# into one eager split op under both PCG and BCG. Without this, the bf16
# fallback BMM is captured alone in its own single-kernel CUDA graph submodule,
# paying per-submodule host overhead with no fusion benefit.
#
# `q_nope_out_view` aliases `q_nope_out_buf` (transposed). The op writes
# `q_nope_out_buf` via `torch.bmm(..., out=...)` and then reads through
# `q_nope_out_view`, so the alias's storage is mutated too. Declare it in
# `mutates_args` to keep the schema honest.
@register_custom_op(
    mutates_args=["q_nope_out_buf", "q_nope_out_view", "attn_output_buf"]
)
@register_split_op()
def mla_bmm_then_unified_attention(
    q_nope_t: torch.Tensor,
    w_kc: torch.Tensor,
    q_nope_out_buf: torch.Tensor,
    q_nope_out_view: torch.Tensor,
    k_nope: torch.Tensor,
    attn_output_buf: torch.Tensor,
    save_kv_cache: bool,
    layer_id: int,
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    cos_sin_cache: Optional[torch.Tensor] = None,
    is_neox: Optional[bool] = None,
    llama_4_scaling: Optional[torch.Tensor] = None,
    topk_indices: Optional[torch.Tensor] = None,
) -> None:
    torch.bmm(q_nope_t, w_kc, out=q_nope_out_buf)
    unified_attention_with_output(
        q_nope_out_view,
        k_nope,
        k_nope,
        attn_output_buf,
        save_kv_cache,
        layer_id,
        q_rope=q_pe,
        k_rope=k_pe,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
        llama_4_scaling=llama_4_scaling,
        topk_indices=topk_indices,
    )


bcg_mla_bmm_then_unified_attention = eager_on_graph(True)(
    mla_bmm_then_unified_attention
)
