"""Ascend MoE runner backend with NPU‑specific ops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.environ import envs
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
from sglang.srt.hardware_backend.npu.moe.finalize_routing import NPUMoETokenUnpermute
from sglang.srt.hardware_backend.npu.moe.init_routing import (
    MXFP8_QUANT_MODE,
    NPUMoEInitRouting_v2,
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

    if config.activation == "situ":
        return False
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
    from sglang.srt.layers.moe.token_dispatcher.ascend_tp import (
        AscendTPCombineInput,
        AscendTPDispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPLLCombineInput,
        DeepEPLLDispatchOutput,
        DeepEPNormalCombineInput,
        DeepEPNormalDispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )

from sglang.srt.layers.moe.utils import (
    DispatcherOutputDtype,
    MoeRunnerBackend,
    get_ascend_dispatcher_output_dtype,
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
            # for both dispatchers: ascend_tp gets its activation quant fused
            # into routing, DeepEP dispatches bf16 and gmm1 quantises it
            # itself.
            self.activation = None
        elif (
            isinstance(kernel, NPUW4A8MXFP4MoEMethod)
            and config.swiglu_limit is not None
            and config.swiglu_limit > 0
        ):
            self.activation = NPUSwigluMxfp8Quant(config.swiglu_limit)
        elif get_moe_a2a_backend().is_deepep():
            # DeepEP path: use a unified kernel that decides quantisation
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
            # Non‑DeepEP (ascend_tp) path
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


@register_pre_permute("ascend_tp", "ascend")
def pre_permute_ascend_tp_to_ascend(
    dispatch_output: AscendTPDispatchOutput,
    quant_info: AscendQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> AscendRunnerInput:
    return AscendRunnerInput(
        hidden_states=dispatch_output.hidden_states,
        hidden_states_scale=dispatch_output.hidden_states_scale,
        expert_tokens=dispatch_output.expert_tokens,
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


@register_post_permute("ascend", "ascend_tp")
def post_permute_ascend_to_ascend_tp(
    runner_output: AscendRunnerOutput,
    quant_info: AscendQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> AscendTPCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.ascend_tp import AscendTPCombineInput

    return AscendTPCombineInput(hidden_states=runner_output.hidden_states)


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


@register_pre_permute("standard", "ascend")
def pre_permute_standard_to_ascend(
    dispatch_output: StandardDispatchOutput,
    quant_info: AscendQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> AscendRunnerInput:
    if (runner_config.num_fused_shared_experts or 0) > 0:
        raise NotImplementedError(
            "Fused shared experts are not supported by the NPU standard "
            "all-gather dispatcher: every rank sees all tokens after the "
            "all-gather, so the shared experts would be computed num_ranks "
            "times and over-counted by the combine reduce-scatter."
        )

    topk_output = dispatch_output.topk_output
    topk_ids = topk_output.topk_ids.to(torch.int32)
    running_state["topk_weights"] = topk_output.topk_weights

    hidden_states = dispatch_output.hidden_states
    hidden_states_scale = dispatch_output.hidden_states_scale

    dispatcher_dtype = get_ascend_dispatcher_output_dtype(
        runner_config.layer.dispatcher
    )

    if hidden_states_scale is not None:
        # Activations were pre-quantized to MXFP8 (fp8_e4m3 payload + e8m0
        # block scale) before the all-gather; route both through init_routing
        # with quant_mode=-1 (no re-quantization) and let it carry the block
        # scale along with the tokens.
        init = NPUMoEInitRouting_v2(quant_mode=-1)
        permuted_hidden_states, expanded_row_idx, expert_tokens, hidden_states_scale = (
            init._init_routing(
                hidden_states,
                topk_ids,
                runner_config.num_local_experts,
                topk_ids.shape[-1],
                scale=hidden_states_scale,
                x_dtype=torch.float8_e4m3fn,
            )
        )
    else:
        # hidden_states_scale is None: the dispatcher did not pre-quantize,
        # so dispatcher_dtype can only be BF16 or INT8 here (MXFP8 always
        # pre-quantizes in StandardDispatcher.dispatch and enters the if
        # branch above). INT8 quantizes on the fly inside init_routing
        # (quant_mode=1); BF16 passes through (quant_mode=-1).
        if dispatcher_dtype == DispatcherOutputDtype.INT8:
            init = NPUMoEInitRouting_v2(quant_mode=1)
        else:
            init = NPUMoEInitRouting_v2(quant_mode=-1)
        permuted_hidden_states, expanded_row_idx, expert_tokens, hidden_states_scale = (
            init._init_routing(
                hidden_states,
                topk_ids,
                runner_config.num_local_experts,
                topk_ids.shape[-1],
            )
        )

    running_state["expanded_row_idx"] = expanded_row_idx
    return AscendRunnerInput(
        hidden_states=permuted_hidden_states,
        hidden_states_scale=hidden_states_scale,
        expert_tokens=expert_tokens,
        group_list_type=1,
    )


@register_post_permute("ascend", "standard")
def post_permute_ascend_to_standard(
    runner_output: AscendRunnerOutput,
    quant_info: AscendQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    # Restore the global token order (compact layout) and apply routing
    # weights; the dispatcher's combine then reduce-scatters partial outputs.
    hidden_states = NPUMoETokenUnpermute()._finalize_routing(
        runner_output.hidden_states,
        topk_weights=running_state["topk_weights"].to(
            runner_output.hidden_states.dtype
        ),
        expanded_row_idx=running_state["expanded_row_idx"],
        topk_ids=None,
    )
    return StandardCombineInput(hidden_states=hidden_states)


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
