"""Packed NVFP4 routed experts for Muse Spark, on public SGLang + FlashInfer.

Muse applies an RMSNorm to each expert's output *before* the top-k weighted sum.
The fused TRT-LLM MoE kernel folds GEMM2 and that combine together, so the norm
has nowhere to go. We therefore ask the kernel for its un-combined output
(``do_finalize=False``) and run a CuTeDSL unpermute -> norm -> weighted-sum
implementation built on public CUDA Python and CUTLASS APIs.

Weight allocation and scale processing come from the public ModelOpt quant
method, with Muse's gated TRT-LLM alignment substituted through an instance
hook; the GEMMs come from public FlashInfer. The Torch finalize below remains
as the no-post-norm fallback and reference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from torch import nn

from sglang.kernels.ops.muse_spark.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum,
)
from sglang.kernels.ops.muse_spark.fused_post_expert_norm_weighted_sum_trtllm import (
    FusedPostExpertNormWeightedSumTRTLLM,
)
from sglang.kernels.ops.muse_spark.norm_fp4_quant import fused_norm_fp4_quant
from sglang.srt.layers.moe import MoeRunnerBackend, get_moe_runner_backend
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.muse_spark.muse_spark_nvfp4_moe import (
    MuseSparkNvFp4FusedMoEMethod,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.utils.common import get_device_capability, is_cuda

if TYPE_CHECKING:
    from sglang.kernels.ops.muse_spark.fi_trtllm_moe_finalize_with_rmsnorm_cutedsl import (
        FinalizePushWorkspace,
    )
    from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
        FlashinferCombineInput,
        FlashinferDispatchOutput,
    )

# 1 / (fp8_e4m3_max * 6)
_NVFP4_PER_TOKEN_SCALE_INV = 1.0 / (448.0 * 6.0)

_MARLIN_SHAPE_ALIGNMENT = 64
_MARLIN_BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)


def _resolve_moe_runner_backend():
    backend = get_moe_runner_backend()
    if not backend.is_auto():
        return backend
    if is_cuda() and (8, 0) <= get_device_capability() < (10, 0):
        return MoeRunnerBackend.MARLIN
    return MoeRunnerBackend.FLASHINFER_TRTLLM


class _DirectDispatcher:
    """Stand-in for a FusedMoE token dispatcher.

    ``process_weights_after_loading`` hands the activation scales to
    ``layer.dispatcher`` so an all-to-all backend can quantize before dispatch.
    We invoke the kernel directly on a single node, so there is nothing to
    configure -- the scales are read straight off the layer in ``forward``.
    """

    local_expert_mapping = None

    def set_quant_config(self, quant_config: dict) -> None:
        self.quant_config = quant_config


class MuseSpark1xFp4Experts(nn.Module):
    """Routed experts held as packed NVFP4 and executed by the TRT-LLM kernel.

    Weight *preparation* is delegated to the public ModelOpt method with Muse's
    gated aligner; only the forward is hand-written, because of the
    post-expert norm.
    """

    def __init__(
        self,
        *,
        num_experts: int,
        num_local_experts: int,
        local_expert_offset: int,
        hidden_size: int,
        intermediate_size: int,
        top_k: int,
        quant_config: QuantizationConfig,
        params_dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        # create_weights sizes the activation scales from ``num_experts``, so it
        # must be the LOCAL count; the kernel separately wants the global count.
        self.num_experts = num_local_experts
        self.global_num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.local_expert_offset = local_expert_offset
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = intermediate_size
        self.top_k = top_k
        self.params_dtype = params_dtype or torch.get_default_dtype()
        self.quant_config = quant_config
        # SwiGLU, and no routed_scaling here -- Muse folds its route_scale into
        # the topk weights before they reach us.
        self.moe_runner_config = MoeRunnerConfig(
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size,
            top_k=top_k,
            num_fused_shared_experts=0,
            params_dtype=self.params_dtype,
            routing_method_type=1,
            activation="silu",
            is_gated=True,
            routed_scaling_factor=1.0,
            inplace=False,
        )
        self.dispatcher = _DirectDispatcher()

        capability = get_device_capability() if is_cuda() else None
        self.moe_runner_backend = _resolve_moe_runner_backend()
        self.use_marlin = self.moe_runner_backend.is_marlin()
        self._sm_major = capability[0] if capability is not None else 0
        if not self.use_marlin and capability is not None and capability < (10, 0):
            raise ValueError(
                f"--moe-runner-backend {self.moe_runner_backend.value} needs a "
                f"Blackwell (SM100+) device for packed NVFP4 experts, but this "
                f"device reports SM{capability[0]}{capability[1]}. Launch with "
                "--moe-runner-backend marlin, or with auto, which selects marlin "
                "on SM80-SM90."
            )
        if self.use_marlin:
            assert hidden_size % _MARLIN_SHAPE_ALIGNMENT == 0, (
                f"NVFP4 Marlin MoE needs expert hidden_size divisible by "
                f"{_MARLIN_SHAPE_ALIGNMENT}, got {hidden_size}. Gated NVFP4 Marlin "
                "preparation asserts on the unpadded shape instead of padding."
            )
            assert intermediate_size % _MARLIN_SHAPE_ALIGNMENT == 0, (
                f"NVFP4 Marlin MoE needs moe_intermediate_size divisible by "
                f"{_MARLIN_SHAPE_ALIGNMENT}, got {intermediate_size}. Gated NVFP4 "
                "Marlin preparation asserts on the unpadded shape instead of "
                "padding."
            )

        # post_expert_norm is unknown until set_post_expert_norm; start with the
        # OSS default (standard, eps=1e-8) and rebuild when the real module arrives.
        self._post_expert_norm: Optional[nn.Module] = None
        self._post_expert_norm_eps = 1e-8
        self._post_expert_norm_weight_first = True
        self._fused_finalize_op = (
            None
            if self.use_marlin
            else FusedPostExpertNormWeightedSumTRTLLM(
                topk=top_k, hidden_dim=hidden_size, eps=1e-8, weight_first=False
            )
        )
        self.register_buffer("_post_expert_norm_weight", None, persistent=False)

        expert_map = None
        if num_local_experts != num_experts:
            start = local_expert_offset
            expert_map = torch.full((num_experts,), -1, dtype=torch.int32)
            expert_map[start : start + num_local_experts] = torch.arange(
                num_local_experts, dtype=torch.int32
            )
        self.register_buffer("_expert_map", expert_map, persistent=False)

        self.quant_method = MuseSparkNvFp4FusedMoEMethod(quant_config)
        if not self.use_marlin and not self.quant_method.enable_flashinfer_trtllm_moe:
            raise ValueError(
                "Packed NVFP4 experts need the FlashInfer TRT-LLM MoE backend, "
                "which prepares the shuffled FP4 weights this forward reads. "
                "Launch with --moe-runner-backend flashinfer_trtllm."
            )
        self.quant_method._moe_runner_backend = self.moe_runner_backend
        self.quant_method.create_weights(
            layer=self,
            num_experts=num_local_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size,
            params_dtype=self.params_dtype,
            extra_weight_attrs={},
        )
        if self.use_marlin:
            self.w13_blockscale_swizzled = None
            self.w2_blockscale_swizzled = None

    @torch.no_grad()
    def set_post_expert_norm(self, post_expert_norm: Optional[nn.Module]) -> None:
        """Cache the effective weight and norm form for the fused finalize.

        Which form applies is selected by ``use_ffn_learnable_scales`` in the
        model config (see ``models/muse_spark.py``): ``True`` selects
        ``MuseSpark1xRMSNormWithInputScale`` (weight-first,
        ``rms_norm(weight*x)``), ``False`` selects ``MuseSpark1xRMSNorm``
        (standard, ``rms_norm(x) * weight``). The released checkpoint sets
        ``True``, so deriving the form here is robustness rather than a
        behaviour change for it; a ``False`` export would previously have been
        mis-scaled by the hardcoded weight-first form.

        ``post_expert_norm`` is *not* registered as a submodule — it already
        belongs to the parent block, so registering it here would duplicate its
        weight in ``state_dict`` and create a reference cycle.
        """
        if post_expert_norm is None:
            # Bypass nn.Module.__setattr__ so None (and later the module) are
            # not treated as submodules.
            object.__setattr__(self, "_post_expert_norm", None)
            self._post_expert_norm_weight = None
            return
        # Lazy import to avoid circular import at module load.
        from sglang.srt.layers.muse_spark.muse_spark_layers import (
            MuseSpark1xRMSNormWithInputScale,
        )

        weight = post_expert_norm._get_effective_weight()
        if weight is None:
            weight = torch.ones(
                self.hidden_size,
                dtype=torch.bfloat16,
                device=self.gate_up_proj.device,
            )
        object.__setattr__(self, "_post_expert_norm", post_expert_norm)
        self._post_expert_norm_weight = weight.detach().to(torch.bfloat16)
        eps = float(getattr(post_expert_norm, "eps", 1e-8))
        weight_first = isinstance(post_expert_norm, MuseSpark1xRMSNormWithInputScale)
        self._post_expert_norm_eps = eps
        self._post_expert_norm_weight_first = weight_first
        # Rebuild the fused op with the correct eps/weight_first; the old
        # op (wrong form) is discarded so it cannot silently mis-scale.
        if self._fused_finalize_op is not None and (
            self._fused_finalize_op.eps != eps
            or self._fused_finalize_op.weight_first != weight_first
        ):
            self._fused_finalize_op = FusedPostExpertNormWeightedSumTRTLLM(
                topk=self.top_k,
                hidden_dim=self.hidden_size,
                eps=eps,
                weight_first=weight_first,
            )

    def _pre_norm_weight(self, pre_expert_norm: nn.Module) -> Optional[torch.Tensor]:
        weight = pre_expert_norm._get_effective_weight()
        cached = getattr(self, "_pre_norm_weight_cache", (None, None))
        if cached[0] is not weight:
            cached = (
                weight,
                None if weight is None else weight.to(torch.bfloat16).contiguous(),
            )
            self._pre_norm_weight_cache = cached
        return cached[1]

    def forward_dispatched(
        self,
        dispatch_output: FlashinferDispatchOutput,
        post_expert_norm: Optional[nn.Module],
    ) -> FlashinferCombineInput:
        """Run expert-owned SP rows and build the BF16 A2A combine payload."""
        if self.use_marlin:
            raise ValueError(
                "Muse Spark sequence-parallel prefill dispatches the routed "
                "experts through the TRT-LLM FP4 MoE runner, which needs a "
                f"Blackwell (SM100+) device; this device reports SM{self._sm_major}x "
                "and resolved the Marlin W4A16 backend. Launch without "
                "sequence-parallel prefill on SM80-SM90."
            )
        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
            FlashInferTrtllmDeferredFinalizeOutput,
            FlashInferTrtllmFp4MoeQuantInfo,
            flashinfer_trtllm_deferred_finalize_context,
            fused_experts_none_to_flashinfer_trtllm_fp4,
        )
        from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
            FlashinferCombineInput,
        )

        quant_info = FlashInferTrtllmFp4MoeQuantInfo(
            w13_weight=self.gate_up_proj,
            w2_weight=self.down_proj,
            w13_weight_scale=self.gate_up_proj_weight_scale,
            w2_weight_scale=self.down_proj_weight_scale,
            g1_scale_c=self.g1_scale_c,
            g1_alphas=self.g1_alphas,
            g2_alphas=self.g2_alphas,
            w13_input_scale_quant=self.w13_input_scale_quant,
            global_num_experts=self.global_num_experts,
            local_expert_offset=self.local_expert_offset,
            local_num_experts=self.num_local_experts,
            intermediate_size_per_partition=self.intermediate_size_per_partition,
            routing_method_type=1,
            use_per_token_activation=self.quant_config.use_per_token_activation,
            gemm1_alpha=getattr(self, "gemm1_alpha", None),
            gemm1_beta=getattr(self, "gemm1_beta", None),
            gemm1_clamp_limit=getattr(self, "gemm1_clamp_limit", None),
        )
        with flashinfer_trtllm_deferred_finalize_context():
            runner_output = fused_experts_none_to_flashinfer_trtllm_fp4(
                dispatch_output,
                quant_info,
                self.moe_runner_config,
                use_routed_topk=True,
            )
        deferred_output = runner_output.hidden_states
        if not isinstance(deferred_output, FlashInferTrtllmDeferredFinalizeOutput):
            raise TypeError(
                "Muse Spark SP prefill requires deferred TRTLLM FP4 output, got "
                f"{type(deferred_output).__name__}."
            )
        combine_payload = self._finalize_deferred_output(
            deferred_output,
            num_tokens=dispatch_output.hidden_states.shape[0],
            post_expert_norm=post_expert_norm,
            out_dtype=torch.bfloat16,
        )
        return FlashinferCombineInput(combine_payload)

    def _finalize_deferred_output(
        self,
        deferred_output,
        *,
        num_tokens: int,
        post_expert_norm: Optional[nn.Module],
        out_dtype: torch.dtype,
        ar_workspace: FinalizePushWorkspace | None = None,
    ) -> torch.Tensor:
        use_fused = (
            self._post_expert_norm is not None
            and self._post_expert_norm is post_expert_norm
            and self._post_expert_norm_weight is not None
        )
        if not use_fused and post_expert_norm is not None:
            self.set_post_expert_norm(post_expert_norm)
            use_fused = self._post_expert_norm_weight is not None

        if not use_fused or post_expert_norm is None:
            output = finalize_with_post_expert_norm(
                gemm2_out=deferred_output.gemm2_out,
                expert_weights=deferred_output.expert_weights,
                expanded_idx_to_permuted_idx=(
                    deferred_output.expanded_idx_to_permuted_idx
                ),
                num_tokens=num_tokens,
                hidden_size=self.hidden_size,
                top_k=deferred_output.top_k,
                post_expert_norm=post_expert_norm,
                out_dtype=out_dtype,
            )

            if ar_workspace is not None:
                torch.distributed.all_reduce(output, group=ar_workspace.group)
            return output

        output = torch.empty(
            num_tokens,
            self.hidden_size,
            dtype=torch.bfloat16,
            device=deferred_output.gemm2_out.device,
        )
        self._fused_finalize_op(
            gemm2_output=deferred_output.gemm2_out,
            expert_weights=deferred_output.expert_weights,
            expanded_idx_to_permuted_idx=(deferred_output.expanded_idx_to_permuted_idx),
            rms_norm_weight=self._post_expert_norm_weight,
            output=output,
            ar_workspace=ar_workspace,
        )
        return output.to(out_dtype)

    def _marlin_expert_outputs(
        self,
        hidden_states: torch.Tensor,
        topk_scores: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        from sgl_kernel.scalar_type import scalar_types

        from sglang.kernels.ops.activation.activation import silu_and_mul
        from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
        from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
        from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace

        hidden_states = hidden_states.contiguous()
        num_tokens, size_k = hidden_states.shape
        topk = topk_ids.shape[1]
        size_n = self.intermediate_size_per_partition
        gemm1_n = 2 * size_n
        expert_map = self._expert_map

        local_topk_ids = topk_ids.to(torch.int32)
        if expert_map is not None:
            local_topk_ids = expert_map[local_topk_ids]
        local_topk_ids = local_topk_ids.contiguous()
        topk_weights = topk_scores.to(torch.float32).contiguous()

        for block_size_m in _MARLIN_BLOCK_SIZE_M_CANDIDATES:
            if num_tokens * topk / self.num_local_experts / block_size_m < 0.9:
                break

        if num_tokens == 1 and topk <= 32 and expert_map is None:
            from sglang.kernels.ops.moe.moe_align_single_token import (
                moe_align_single_token,
            )

            sorted_token_ids, expert_ids, num_tokens_post_padded = (
                moe_align_single_token(local_topk_ids, block_size_m)
            )
        else:
            align_num_experts = (
                self.global_num_experts
                if expert_map is not None
                else self.num_local_experts
            )
            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                local_topk_ids, block_size_m, align_num_experts
            )

        workspace = marlin_make_workspace(hidden_states.device, max_blocks_per_sm=4)
        cache2 = torch.empty(
            (num_tokens * topk, size_n),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        cache13 = torch.empty(
            (num_tokens * topk * max(gemm1_n, size_k),),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        cache1 = cache13[: num_tokens * topk * gemm1_n].view(-1, gemm1_n)
        cache3 = cache13[: num_tokens * topk * size_k].view(-1, size_k)

        cache1 = moe_wna16_marlin_gemm(
            hidden_states,
            cache1,
            self.w13_weight,
            None,
            self.w13_weight_scale,
            self.w13_weight_scale_2,
            None,
            None,
            None,
            workspace,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_weights,
            moe_block_size=block_size_m,
            top_k=topk,
            mul_topk_weights=False,
            is_ep=expert_map is not None,
            b_q_type=scalar_types.float4_e2m1f,
            size_m=num_tokens,
            size_n=gemm1_n,
            size_k=size_k,
            is_k_full=True,
            use_atomic_add=False,
            use_fp32_reduce=True,
            is_zp_float=False,
        )
        silu_and_mul(
            cache1.view(-1, gemm1_n),
            cache2,
            expert_ids=local_topk_ids.view(-1),
            expert_step=1,
        )
        return moe_wna16_marlin_gemm(
            cache2,
            cache3,
            self.w2_weight,
            None,
            self.w2_weight_scale,
            self.w2_weight_scale_2,
            None,
            None,
            None,
            workspace,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_weights,
            moe_block_size=block_size_m,
            top_k=1,
            mul_topk_weights=False,
            is_ep=expert_map is not None,
            b_q_type=scalar_types.float4_e2m1f,
            size_m=num_tokens * topk,
            size_n=size_k,
            size_k=size_n,
            is_k_full=True,
            use_atomic_add=False,
            use_fp32_reduce=True,
            is_zp_float=False,
        ).view(num_tokens, topk, size_k), local_topk_ids

    def _forward_marlin(
        self,
        hidden_states: torch.Tensor,
        topk_scores: torch.Tensor,
        topk_ids: torch.Tensor,
        post_expert_norm: Optional[nn.Module],
        pre_expert_norm: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        if post_expert_norm is None:
            raise ValueError(
                "The NVFP4 Marlin fallback requires moe_use_post_expert_norm=True. "
                "Marlin carries one NVFP4 global scale per expert, so "
                "w13_weight_scale_2 is collapsed to its gate column and every "
                "expert output picks up a constant gain 1/r with "
                "r = w3_weight_scale_2 / w1_weight_scale_2. The post-expert "
                "RMSNorm cancels that gain exactly; without it the expert "
                "outputs are mis-scaled per expert. Run this checkpoint on "
                "SM100+ with --moe-runner-backend flashinfer_trtllm instead."
            )
        if pre_expert_norm is not None:
            hidden_states = pre_expert_norm(hidden_states)
        expert_outputs, local_topk_ids = self._marlin_expert_outputs(
            hidden_states, topk_scores, topk_ids
        )
        if self._post_expert_norm is not post_expert_norm:
            self.set_post_expert_norm(post_expert_norm)
        if self._post_expert_norm_weight is not None:
            return fused_post_expert_norm_weighted_sum(
                expert_outputs,
                topk_scores,
                norm_weight=self._post_expert_norm_weight,
                eps=self._post_expert_norm_eps,
                weight_first=self._post_expert_norm_weight_first,
                slot_expert_ids=local_topk_ids,
            )
        return finalize_identity_with_post_expert_norm(
            expert_outputs=expert_outputs,
            topk_scores=topk_scores,
            post_expert_norm=post_expert_norm,
            out_dtype=hidden_states.dtype,
        )

    def prepare_input(
        self,
        hidden_states: torch.Tensor,
        pre_expert_norm: Optional[nn.Module] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Normalize and FP4-quantize activations. Does not depend on routing."""
        seq_len, hidden_size = hidden_states.shape
        if pre_expert_norm is not None and self.quant_config.use_per_token_activation:
            hs_fp4, hs_scale_linear, per_token_scale, _ = fused_norm_fp4_quant(
                hidden_states,
                self._pre_norm_weight(pre_expert_norm),
                pre_expert_norm.eps,
                _NVFP4_PER_TOKEN_SCALE_INV,
            )
        elif self.quant_config.use_per_token_activation:
            # Calibration-free checkpoints ship activation scales of exactly 1.0,
            # so the scale has to be computed per token at runtime instead.
            from flashinfer import SfLayout, nvfp4_quantize

            global_scale_inv = torch.full(
                (1,),
                _NVFP4_PER_TOKEN_SCALE_INV,
                dtype=torch.float32,
                device=hidden_states.device,
            )
            hs_fp4_bytes, hs_sf_bytes, per_token_scale = nvfp4_quantize(
                hidden_states,
                global_scale_inv,
                sfLayout=SfLayout.layout_linear,
                per_token_activation=True,
                backend="cute-dsl",
            )
            seq_len, hidden_size = hidden_states.shape
            hs_fp4 = hs_fp4_bytes.reshape(seq_len, hidden_size // 2)
            hs_scale_linear = hs_sf_bytes.view(torch.float8_e4m3fn).reshape(
                seq_len, hidden_size // 16
            )
        else:
            if pre_expert_norm is not None:
                hidden_states = pre_expert_norm(hidden_states)
            # Match the known-good public vLLM activation path. SGLang's
            # helper selects FlashInfer's experimental CuTeDSL quantizer on
            # SM100, while vLLM's scaled_fp4_quant uses the stable CUDA
            # implementation with a linear scale-factor layout.
            from flashinfer import fp4_quantize

            per_token_scale = None
            hs_fp4_bytes, hs_sf_bytes = fp4_quantize(
                hidden_states,
                self.w13_input_scale_quant,
                sf_vec_size=16,
                sf_use_ue8m0=False,
                is_sf_swizzled_layout=False,
                backend="cuda",
            )
            seq_len, hidden_size = hidden_states.shape
            hs_fp4 = hs_fp4_bytes.reshape(seq_len, hidden_size // 2)
            hs_scale_linear = hs_sf_bytes.view(torch.float8_e4m3fn).reshape(
                seq_len, hidden_size // 16
            )
        hs_scale = hs_scale_linear.view(torch.float8_e4m3fn).reshape(
            *hs_scale_linear.shape[:-1], -1
        )
        return hs_fp4, hs_scale, per_token_scale

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_scores: torch.Tensor,
        topk_ids: torch.Tensor,
        post_expert_norm: Optional[nn.Module],
        pre_expert_norm: Optional[nn.Module] = None,
        packed_topk_ids: Optional[torch.Tensor] = None,
        prepared_input: Optional[
            tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
        ] = None,
        *,
        ar_workspace: FinalizePushWorkspace | None = None,
    ) -> torch.Tensor:
        if self.use_marlin:
            return self._forward_marlin(
                hidden_states,
                topk_scores,
                topk_ids,
                post_expert_norm,
                pre_expert_norm=pre_expert_norm,
            )

        from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe

        num_tokens = hidden_states.shape[0]
        if num_tokens == 0:
            return hidden_states.new_zeros((0, self.hidden_size))
        if prepared_input is None:
            prepared_input = self.prepare_input(hidden_states, pre_expert_norm)
        hs_fp4, hs_scale, per_token_scale = prepared_input
        # Preserve Muse's packed fast path; the upstream FlashInfer API also
        # accepts materialized IDs and FP32 routing weights directly.
        routing = packed_topk_ids
        if routing is None:
            routing = (
                topk_ids.to(torch.int32).contiguous(),
                topk_scores.to(torch.float32).contiguous(),
            )

        gemm2_out, expert_weights, expanded_idx_to_permuted_idx = (
            trtllm_fp4_block_scale_routed_moe(
                topk_ids=routing,
                routing_bias=None,
                hidden_states=hs_fp4,
                hidden_states_scale=hs_scale,
                gemm1_weights=self.gate_up_proj,
                gemm1_weights_scale=self.gate_up_proj_weight_scale.view(
                    torch.float8_e4m3fn
                ),
                gemm1_bias=None,
                gemm1_alpha=getattr(self, "gemm1_alpha", None),
                gemm1_beta=getattr(self, "gemm1_beta", None),
                gemm1_clamp_limit=getattr(self, "gemm1_clamp_limit", None),
                gemm2_weights=self.down_proj,
                gemm2_weights_scale=self.down_proj_weight_scale.view(
                    torch.float8_e4m3fn
                ),
                gemm2_bias=None,
                output1_scale_scalar=self.g1_scale_c,
                output1_scale_gate_scalar=self.g1_alphas,
                output2_scale_scalar=self.g2_alphas,
                per_token_scale=per_token_scale,
                num_experts=self.global_num_experts,
                top_k=topk_ids.shape[1],
                n_group=0,
                topk_group=0,
                intermediate_size=self.intermediate_size_per_partition,
                local_expert_offset=self.local_expert_offset,
                local_num_experts=self.num_local_experts,
                # Muse folds its route_scale into the top-k weights, so the
                # kernel applies none of its own -- but it wants 1.0, not None.
                routed_scaling_factor=1.0,
                routing_method_type=1,  # unused on this path, must be 1
                do_finalize=False,  # we need the per-expert outputs
                activation_type=3,  # SwiGLU
            )[:3]
        )

        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
            FlashInferTrtllmDeferredFinalizeOutput,
        )

        return self._finalize_deferred_output(
            FlashInferTrtllmDeferredFinalizeOutput(
                gemm2_out=gemm2_out,
                expert_weights=expert_weights,
                expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                top_k=topk_ids.shape[1],
            ),
            num_tokens=num_tokens,
            post_expert_norm=post_expert_norm,
            out_dtype=hidden_states.dtype,
            ar_workspace=ar_workspace,
        )


def finalize_with_post_expert_norm(
    *,
    gemm2_out: torch.Tensor,
    expert_weights: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    num_tokens: int,
    hidden_size: int,
    top_k: int,
    post_expert_norm: Optional[nn.Module],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Unpermute, apply the per-expert norm, then the top-k weighted sum.

    ``expanded_idx_to_permuted_idx`` maps each (token, slot) pair to its row in
    the permuted ``gemm2_out``; ``-1`` marks a padded or non-local slot, which
    must contribute nothing.
    """
    flat_map = expanded_idx_to_permuted_idx.reshape(-1)
    valid = flat_map >= 0

    gathered = gemm2_out.new_zeros((flat_map.numel(), hidden_size))
    if valid.any():
        gathered[valid] = gemm2_out[flat_map[valid]].to(gathered.dtype)

    # Norm is per expert output, i.e. before the weighted sum -- not after.
    if post_expert_norm is not None:
        gathered = post_expert_norm(gathered)

    gathered = gathered.view(num_tokens, top_k, hidden_size)
    weights = expert_weights.reshape(num_tokens, top_k).to(gathered.dtype)
    weights = weights * valid.view(num_tokens, top_k).to(gathered.dtype)
    return torch.einsum("tkh,tk->th", gathered, weights).to(out_dtype)


def finalize_identity_with_post_expert_norm(
    *,
    expert_outputs: torch.Tensor,
    topk_scores: torch.Tensor,
    post_expert_norm: nn.Module,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    num_tokens, top_k = expert_outputs.shape[0], expert_outputs.shape[1]
    normed = post_expert_norm(expert_outputs)
    weights = topk_scores.reshape(num_tokens, top_k).to(normed.dtype)
    return (normed * weights.unsqueeze(-1)).sum(dim=1).to(out_dtype)


__all__ = [
    "MuseSpark1xFp4Experts",
    "finalize_with_post_expert_norm",
    "finalize_identity_with_post_expert_norm",
]
