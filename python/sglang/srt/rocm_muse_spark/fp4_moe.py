"""Packed NVFP4 routed experts for Muse Spark, on public SGLang + FlashInfer.

Muse applies an RMSNorm to each expert's output *before* the top-k weighted sum.
The fused TRT-LLM MoE kernel folds GEMM2 and that combine together, so the norm
has nowhere to go. We therefore ask the kernel for its un-combined output
(``do_finalize=False``) and run a CuTeDSL unpermute -> norm -> weighted-sum
implementation built on public CUDA Python and CUTLASS APIs.

Weight allocation and scale processing come from the public ModelOpt quant
method, with a corrected TRT-LLM weight aligner substituted before
processing; the GEMMs come from public FlashInfer. The Torch finalize below
remains as the no-post-norm fallback and reference.
"""

from __future__ import annotations

import os

import torch
from torch import nn

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.rocm_muse_spark.fused_post_expert_norm_weighted_sum_trtllm import (
    FusedPostExpertNormWeightedSumTRTLLM,
)
from sglang.srt.rocm_muse_spark.nvfp4_moe import (
    MuseSparkNvFp4FusedMoEMethod,
)

# 1 / (fp8_e4m3_max * 6)
_NVFP4_PER_TOKEN_SCALE_INV = 1.0 / (448.0 * 6.0)


def _native_finalize_enabled() -> bool:
    return os.environ.get("MUSE_SPARK_NVFP4_NATIVE_FINALIZE", "0") == "1"


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

    Weight *preparation* is delegated to the public ModelOpt method with the
    corrected aligner; only the forward is hand-written, because of
    the post-expert norm.
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
        params_dtype: torch.dtype | None = None,
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
        self.moe_runner_config = MoeRunnerConfig(activation="silu", is_gated=True)
        self.dispatcher = _DirectDispatcher()
        # post_expert_norm is unknown until set_post_expert_norm; start with the
        # OSS default (standard, eps=1e-8) and rebuild when the real module arrives.
        self._post_expert_norm: nn.Module | None = None
        self._fused_finalize_op = FusedPostExpertNormWeightedSumTRTLLM(
            topk=top_k, hidden_dim=hidden_size, eps=1e-8, weight_first=False
        )
        self.register_buffer("_post_expert_norm_weight", None, persistent=False)

        self.quant_method = MuseSparkNvFp4FusedMoEMethod(quant_config)
        if not self.quant_method.enable_flashinfer_trtllm_moe:
            raise ValueError(
                "Packed NVFP4 experts need the FlashInfer TRT-LLM MoE backend, "
                "which prepares the shuffled FP4 weights this forward reads. "
                "Launch with --moe-runner-backend flashinfer_trtllm."
            )
        self.quant_method.create_weights(
            layer=self,
            num_experts=num_local_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size,
            params_dtype=self.params_dtype,
            extra_weight_attrs={},
        )

    @torch.no_grad()
    def set_post_expert_norm(self, post_expert_norm: nn.Module | None) -> None:
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
        from sglang.srt.rocm_muse_spark.layers import (
            MuseSpark1xRMSNormWithInputScale,
        )

        weight = post_expert_norm._get_effective_weight()
        if weight is None:
            weight = torch.ones(
                self.hidden_size,
                dtype=torch.bfloat16,
                device=self.gemm1_weights_fp4_shuffled.device,
            )
        object.__setattr__(self, "_post_expert_norm", post_expert_norm)
        self._post_expert_norm_weight = weight.detach().to(torch.bfloat16)
        eps = float(getattr(post_expert_norm, "eps", 1e-8))
        weight_first = isinstance(post_expert_norm, MuseSpark1xRMSNormWithInputScale)
        # Rebuild the fused op with the correct eps/weight_first; the old
        # op (wrong form) is discarded so it cannot silently mis-scale.
        if (
            self._fused_finalize_op.eps != eps
            or self._fused_finalize_op.weight_first != weight_first
        ):
            self._fused_finalize_op = FusedPostExpertNormWeightedSumTRTLLM(
                topk=self.top_k,
                hidden_dim=self.hidden_size,
                eps=eps,
                weight_first=weight_first,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_scores: torch.Tensor,
        topk_ids: torch.Tensor,
        post_expert_norm: nn.Module | None,
    ) -> torch.Tensor:
        from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe

        from sglang.kernels.ops.moe.pack_topk_ids import PackTopkIds

        num_tokens = hidden_states.shape[0]
        if self.quant_config.use_per_token_activation:
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
        # The kernel takes ids and weights packed into one tensor. Muse has
        # already materialized its own sigmoid top-k, so pack it directly rather
        # than going through the runner's TopKOutput plumbing.
        # The packer requires int32 ids and float32 weights; Muse's top-k comes
        # out of torch.topk as int64.
        packed_topk_ids = PackTopkIds.execute(
            topk_ids.to(torch.int32).contiguous(),
            topk_scores.to(torch.float32).contiguous(),
        )

        gemm2_out, expert_weights, expanded_idx_to_permuted_idx = (
            trtllm_fp4_block_scale_routed_moe(
                topk_ids=packed_topk_ids,
                routing_bias=None,
                hidden_states=hs_fp4,
                hidden_states_scale=hs_scale,
                gemm1_weights=self.gemm1_weights_fp4_shuffled,
                gemm1_weights_scale=self.gemm1_scales_fp4_shuffled.view(
                    torch.float8_e4m3fn
                ),
                gemm1_bias=None,
                gemm1_alpha=getattr(self, "gemm1_alpha", None),
                gemm1_beta=getattr(self, "gemm1_beta", None),
                gemm1_clamp_limit=getattr(self, "gemm1_clamp_limit", None),
                gemm2_weights=self.gemm2_weights_fp4_shuffled,
                gemm2_weights_scale=self.gemm2_scales_fp4_shuffled.view(
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

        # Correctness path: route through the real per-expert norm module so
        # the FP4 finalize reproduces BF16's MuseSpark1xNativeExperts math
        # exactly (standard rms_norm(x)*weight with eps=1e-8 for OSS). The
        # fused op is kept as an opt-in fast path only when it can express the
        # configured form; otherwise we fall back to the module path.
        use_fused = (
            self._post_expert_norm is not None
            and self._post_expert_norm is post_expert_norm
            and self._post_expert_norm_weight is not None
        )
        if not use_fused and post_expert_norm is not None:
            # _post_expert_norm is stale or missing — refresh from caller
            self.set_post_expert_norm(post_expert_norm)
            use_fused = self._post_expert_norm_weight is not None

        if not use_fused or post_expert_norm is None:
            return finalize_with_post_expert_norm(
                gemm2_out=gemm2_out,
                expert_weights=expert_weights,
                expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                num_tokens=num_tokens,
                hidden_size=self.hidden_size,
                top_k=topk_ids.shape[1],
                post_expert_norm=post_expert_norm,
                out_dtype=hidden_states.dtype,
            )

        output = torch.empty(
            num_tokens,
            self.hidden_size,
            dtype=torch.bfloat16,
            device=gemm2_out.device,
        )
        finalize = (
            self._fused_finalize_op.forward_native
            if _native_finalize_enabled()
            else self._fused_finalize_op.forward
        )
        finalize(
            gemm2_output=gemm2_out,
            expert_weights=expert_weights,
            expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
            rms_norm_weight=self._post_expert_norm_weight,
            output=output,
        )
        return output.to(hidden_states.dtype)


def finalize_with_post_expert_norm(
    *,
    gemm2_out: torch.Tensor,
    expert_weights: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    num_tokens: int,
    hidden_size: int,
    top_k: int,
    post_expert_norm: nn.Module | None,
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


__all__ = [
    "MuseSpark1xFp4Experts",
    "finalize_with_post_expert_norm",
]
