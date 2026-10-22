"""Ascend MoE runner backend with NPU‑specific ops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.moe import dispatch as ascend_dispatch
from sglang.srt.hardware_backend.npu.moe.activation import (
    AllGatherActivationWrapper,
    NPUGeluAndMul,
    NPUSitu,
    NPUSituMXFP8Quant,
    NPUSwiglu,
    NPUSwigluDeepEPKernel,
    NPUSwigluMxfp8Quant,
    NPUSwigluOAI,
    NPUSwigluQuant,
    NPUSwigluStepAndMul,
)
from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
    NPUMXFP8MoEMethod,
    NPUW4A8Int8MoEMethod,
    NPUW4A8MXFP4MoEMethod,
    NPUW8A8Int8MoEMethod,
)


def _uses_fused_gmm1(kernel, config: MoeRunnerConfig) -> bool:
    """Whether gmm1 runs matmul+swiglu+requant fused (no separate activation)."""
    # The fused gmm1 kernel (npu_grouped_matmul_swiglu_quant_v2) applies plain
    # SiLU with no clamp. Models whose activation is SiLU-with-clamp
    # (swiglu_limit, e.g. DSV4) must keep the unfused path so the clamp is
    # applied (NPUSwigluMxfp8Quant / NPUSwigluStepAndMul below).
    has_clamp = config.swiglu_limit is not None and config.swiglu_limit > 0

    if not isinstance(kernel, (NPUMXFP8MoEMethod, NPUW4A8MXFP4MoEMethod)):
        return False
    if has_clamp:
        # MXFP8 has no unfused gmm1 path at all, so a swiglu_limit
        # checkpoint cannot be served by it.
        if isinstance(kernel, NPUMXFP8MoEMethod):
            raise NotImplementedError(
                "NPUMXFP8MoEMethod has no unfused gmm1 path, and the fused "
                "gmm1 kernel applies SiLU without clamp — a swiglu_limit "
                "checkpoint cannot be served by this method."
            )
        return False  # W4A8MXFP4: fall back to unfused to preserve the clamp
    if isinstance(kernel, NPUMXFP8MoEMethod):
        return True
    return kernel.use_fused_gmm1


from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    MoeRunnerCore,
    RunnerInput,
    RunnerOutput,
    register_post_permute,
    register_pre_permute,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.ascend_ops import (
        AscendOpsCombineInput,
        AscendOpsDispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPLLCombineInput,
        DeepEPLLDispatchOutput,
        DeepEPNormalCombineInput,
        DeepEPNormalDispatchOutput,
    )

from sglang.srt.layers.moe.utils import (
    MoeRunnerBackend,
    get_moe_a2a_backend,
)


# ---------------------------------------------------------------------------
# Runner IO dataclasses
# ---------------------------------------------------------------------------
@dataclass
class AscendRunnerInput(RunnerInput):
    """Input bundle for the NPU runner."""

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]  # None for unquant
    expert_tokens: torch.Tensor
    group_list_type: int  # 0 or 1 (passed to NPU ops)

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.ASCEND


@dataclass
class AscendRunnerOutput(RunnerOutput):
    """Output bundle from the NPU runner."""

    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.ASCEND


# ---------------------------------------------------------------------------
# Main runner core
# ---------------------------------------------------------------------------
class AscendRunnerCore(MoeRunnerCore):
    runner_backend = MoeRunnerBackend.ASCEND

    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)

        kernel = config.layer.w2_kernel

        if _uses_fused_gmm1(kernel, config):
            # Fused methods (MXFP8; MXFP4 W4A8 via use_fused_gmm1) fold
            # gate/up + swiglu + requant into gmm1, so there is no separate
            # activation step — run() skips it. Left None on purpose so that
            # reaching for it fails loudly instead of silently applying an
            # unfused swiglu to already-requantised activations. This holds
            # for every transport: the no-A2A path gets its activation quant
            # fused into routing, DeepEP dispatches bf16 and gmm1 quantises it
            # itself.
            self.activation = None
        elif (
            isinstance(kernel, NPUW4A8MXFP4MoEMethod)
            and config.swiglu_limit is not None
            and config.swiglu_limit > 0
        ):
            self.activation = NPUSwigluMxfp8Quant(config.swiglu_limit)
        elif get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_ascend_ops():
            # DeepEP / Ascend Ops path: the dispatcher hands over tokens grouped
            # by expert, so use a unified kernel that decides quantisation.
            is_quant_kernel = isinstance(
                kernel, (NPUW4A8Int8MoEMethod, NPUW8A8Int8MoEMethod)
            )
            if config.activation == "situ":
                beta = config.gemm1_alpha if config.gemm1_alpha is not None else 4.0
                if (
                    isinstance(kernel, NPUW4A8MXFP4MoEMethod)
                    and envs.SGLANG_NPU_MOE_SITU_MXFP8_FUSED.get()
                ):
                    if config.gemm1_clamp_limit is None:
                        raise ValueError(
                            "fused SiTU MXFP8 quantization requires gemm1_clamp_limit"
                        )
                    self.activation = NPUSituMXFP8Quant(
                        beta=beta,
                        linear_beta=config.gemm1_clamp_limit,
                    )
                else:
                    self.activation = NPUSitu(
                        need_quant=is_quant_kernel,
                        beta=beta,
                        linear_beta=config.gemm1_clamp_limit,
                    )
            else:
                self.activation = NPUSwigluDeepEPKernel(
                    need_quant=is_quant_kernel,
                    alpha=config.gemm1_alpha,
                    limit=config.gemm1_clamp_limit,
                )
        else:
            # Non-DeepEP, no-A2A routing path
            # 1. Choose the base activation according to the quant method
            if isinstance(kernel, (NPUW4A8Int8MoEMethod, NPUW8A8Int8MoEMethod)):
                inner = NPUSwigluQuant()
            elif config.activation == "situ":
                # Grouped SiTU (Kimi-K3). need_quant=False: the MXFP4 / BF16
                # gmm2 requantizes the activations itself, so no quant is
                # fused here. Matches the DeepEP branch below.
                inner = NPUSitu(
                    need_quant=False,
                    beta=config.gemm1_alpha if config.gemm1_alpha is not None else 4.0,
                    linear_beta=config.gemm1_clamp_limit,
                )
            else:
                if config.activation == "npu_swiglu_oai":
                    # NPUSwigluOAI requires the runner config to pass
                    # gemm1_alpha and gemm1_clamp_limit to the triton kernel.
                    inner = NPUSwigluOAI(moe_runner_config=config)
                elif config.activation == "silu":
                    if config.gemm1_clamp_limit is not None:
                        inner = NPUSwigluStepAndMul(
                            clamp_limit=config.gemm1_clamp_limit
                        )
                    else:
                        inner = NPUSwiglu()
                else:
                    inner = NPUGeluAndMul()

            # 2. If the quant method (GGUF) needs TP all‑gather, wrap the activation
            if getattr(config, "use_tp_all_gather_activation", False):
                self.activation = AllGatherActivationWrapper(inner, dim=-1)
            else:
                self.activation = inner

    def run(
        self,
        runner_input: AscendRunnerInput,
        quant_info: AscendQuantInfo,
        running_state: dict,
        hooks: Optional[Any] = None,
    ) -> AscendRunnerOutput:
        """
        Execute the MoE layer using NPU‑specific grouped matmul ops.
        """
        x = runner_input.hidden_states
        original_dtype = torch.float16 if x.dtype == torch.float16 else torch.bfloat16
        expert_tokens = runner_input.expert_tokens
        group_list_type = runner_input.group_list_type

        w13_kernel = self.config.layer.w13_kernel

        if _uses_fused_gmm1(w13_kernel, self.config):
            # --- w13 projection + activation, fused into one kernel ---
            # The fused gmm1 returns activations already requantised for gmm2,
            # so there is no separate activation step to run.
            hidden_states, pertoken_scale = w13_kernel.apply_fused_gmm1_swiglu(
                quant_info,
                x,
                expert_tokens,
                pertoken_scale=runner_input.hidden_states_scale,
                group_list_type=group_list_type,
            )
        else:
            # --- w13 (gate & up) projection ---
            hidden_states = w13_kernel.apply(
                quant_info,
                x,
                expert_tokens,
                pertoken_scale=runner_input.hidden_states_scale,
                output_dtype=original_dtype,
                weight_prefix="w13",
                group_list_type=group_list_type,
            )

            # --- Activation ---
            # Grouped-row activations require dispatch metadata.
            if isinstance(
                self.activation,
                (
                    NPUSwigluDeepEPKernel,
                    NPUSitu,
                    NPUSituMXFP8Quant,
                    NPUSwigluMxfp8Quant,
                ),
            ):
                hidden_states, pertoken_scale = self.activation._apply_activation(
                    hidden_states,
                    group_list=expert_tokens,
                    group_list_type=group_list_type,
                )
            else:
                hidden_states, pertoken_scale = self.activation._apply_activation(
                    hidden_states
                )

        # --- w2 (down) projection ---
        hidden_states = self.config.layer.w2_kernel.apply(
            quant_info,
            hidden_states,
            expert_tokens,
            pertoken_scale=pertoken_scale,
            output_dtype=original_dtype,
            weight_prefix="w2",
            group_list_type=group_list_type,
        )
        return AscendRunnerOutput(hidden_states=hidden_states)


# ---------------------------------------------------------------------------
# QuantInfo
# ---------------------------------------------------------------------------
@dataclass
class AscendQuantInfo(MoeQuantInfo):
    """Quantization payload for Ascend."""

    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    w13_weight_scale: Optional[torch.Tensor] = None
    w2_weight_scale: Optional[torch.Tensor] = None
    w13_weight_offset: Optional[torch.Tensor] = None
    w2_weight_offset: Optional[torch.Tensor] = None
    w13_weight_bias: Optional[torch.Tensor] = None
    w2_weight_bias: Optional[torch.Tensor] = None
    w13_scale_bias: Optional[torch.Tensor] = None
    w2_scale_bias: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Pre/Post permute hooks
# ---------------------------------------------------------------------------


def _routing_layer(runner_config: MoeRunnerConfig):
    """The layer the Ascend routing state lives on.

    ``AscendRunnerCore`` reads ``config.layer`` for its kernels, so the permute
    hooks can rely on it too; fail loudly instead of silently routing nothing.
    """
    layer = getattr(runner_config, "layer", None)
    if layer is None:
        raise RuntimeError(
            "Ascend MoE routing requires runner_config.layer to be set."
        )
    return layer


@register_pre_permute("standard", "ascend")
def pre_permute_standard_to_ascend(
    dispatch_output: StandardDispatchOutput,
    quant_info: AscendQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> AscendRunnerInput:
    """Sort tokens by expert and (optionally) quantize them in the same op.

    The community dispatcher only carries the token batch plus the top-k
    output, so the expert permutation and the fused activation quantization
    belong to the runner's IO adaptation: token routing by expert is not a
    communication concern.

    ``npu_moe_init_routing_v2`` returns, besides the permuted rows, the row
    mapping and the per-expert token counts the grouped matmuls need; both are
    stashed on ``running_state`` for the post-permute, which replaces what the
    dispatch half of the old Ascend dispatcher used to own.
    """
    # Imported here, not at module scope: ``topk`` pulls in eplb/observability/
    # disaggregation, which imports ``configs.model_config`` while it is still
    # initializing (model_config -> quantization -> ... -> this module).
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    topk_output = dispatch_output.topk_output
    if not TopKOutputChecker.format_is_standard(topk_output):
        raise NotImplementedError(
            "Ascend MoE routing requires a standard top-k output, got "
            f"{type(topk_output).__name__}."
        )
    if dispatch_output.hidden_states_pre_quant is not None:
        raise NotImplementedError(
            "Ascend MoE routing quantizes inside npu_moe_init_routing_v2 and "
            "cannot consume a pre-quantized routed input."
        )

    layer = _routing_layer(runner_config)
    hidden_states = dispatch_output.hidden_states
    topk_weights = topk_output.topk_weights.to(hidden_states.dtype)
    topk_ids = topk_output.topk_ids.to(torch.int32)
    top_k = topk_weights.shape[-1]

    init, finalize, group_list_type = ascend_dispatch.get_routing(layer)

    (
        permuted_hidden_states,
        expanded_row_idx,
        expert_tokens,
        hidden_states_scale,
    ) = init._init_routing(
        hidden_states,
        topk_ids,
        runner_config.num_experts,
        top_k,
    )

    running_state["ascend_routing"] = {
        "finalize": finalize,
        "expanded_row_idx": expanded_row_idx,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
    }

    return AscendRunnerInput(
        hidden_states=permuted_hidden_states,
        hidden_states_scale=hidden_states_scale,
        expert_tokens=expert_tokens,
        group_list_type=group_list_type,
    )


@register_pre_permute("ascend_ops", "ascend")
def pre_permute_ascend_ops_to_ascend(
    dispatch_output: AscendOpsDispatchOutput,
    quant_info: AscendQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> AscendRunnerInput:
    return AscendRunnerInput(
        hidden_states=dispatch_output.hidden_states,
        hidden_states_scale=dispatch_output.hidden_states_scale,
        expert_tokens=dispatch_output.expert_tokens.to(torch.int64),
        group_list_type=dispatch_output.group_list_type,
    )


@register_pre_permute("deepep_normal", "ascend")
def pre_permute_deepep_normal_to_ascend(
    dispatch_output: DeepEPNormalDispatchOutput,
    quant_info: AscendQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> AscendRunnerInput:
    (
        hidden_states,
        hidden_states_scale,
        topk_ids,
        topk_weights,
        num_recv_tokens_per_expert,
    ) = dispatch_output
    group_list = torch.tensor(
        num_recv_tokens_per_expert,
        dtype=torch.int64,
        device=hidden_states.device,
    )
    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights

    return AscendRunnerInput(
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        expert_tokens=group_list,
        group_list_type=1,
    )


@register_pre_permute("deepep_ll", "ascend")
def pre_permute_deepep_ll_to_ascend(
    dispatch_output: DeepEPLLDispatchOutput,
    quant_info: AscendQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> AscendRunnerInput:
    (
        hidden_states,
        hidden_states_scale,
        topk_ids,
        topk_weights,
        group_list,
        _,
    ) = dispatch_output
    group_list = group_list.to(torch.int64)
    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights
    return AscendRunnerInput(
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        expert_tokens=group_list,
        group_list_type=1,
    )


@register_post_permute("ascend", "standard")
def post_permute_ascend_to_standard(
    runner_output: AscendRunnerOutput,
    quant_info: AscendQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> StandardCombineInput:
    """Scatter the expert outputs back and sum them with the router weights.

    ``npu_moe_finalize_routing`` turns the ``[T*top_k, H]`` grouped output back
    into ``[T, H]``, doing the top-k weighted sum on the way. This replaces the
    combine half of the old Ascend dispatcher; the community dispatcher's
    combine is a passthrough for the no-A2A path.
    """
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
    )

    state = running_state.pop("ascend_routing")
    final_hidden_states = state["finalize"]._finalize_routing(
        runner_output.hidden_states,
        topk_weights=state["topk_weights"],
        expanded_row_idx=state["expanded_row_idx"],
        topk_ids=state["topk_ids"],
    )

    return StandardCombineInput(hidden_states=final_hidden_states)


@register_post_permute("ascend", "ascend_ops")
def post_permute_ascend_to_ascend_ops(
    runner_output: AscendRunnerOutput,
    quant_info: AscendQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> AscendOpsCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.ascend_ops import (
        AscendOpsCombineInput,
    )

    # The combine path (MC2 op or finalize routing) weights and sums the top-k
    # routes itself, so only the expert output is passed on.
    return AscendOpsCombineInput(hidden_states=runner_output.hidden_states)


@register_post_permute("ascend", "deepep_normal")
def post_permute_ascend_to_deepep_normal(
    runner_output: AscendRunnerOutput,
    quant_info: AscendQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepEPNormalCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPNormalCombineInput

    return DeepEPNormalCombineInput(
        hidden_states=runner_output.hidden_states,
        topk_ids=running_state["topk_ids"],
        topk_weights=running_state["topk_weights"],
    )


@register_post_permute("ascend", "deepep_ll")
def post_permute_ascend_to_deepep_ll(
    runner_output: AscendRunnerOutput,
    quant_info: AscendQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepEPLLCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLCombineInput

    return DeepEPLLCombineInput(
        hidden_states=runner_output.hidden_states,
        topk_ids=running_state["topk_ids"],
        topk_weights=running_state["topk_weights"],
    )
