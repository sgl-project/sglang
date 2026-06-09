"""Torch_npu MoE runner backend with NPU‑specific ops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.hardware_backend.npu.moe.activation import (
    NPUGeluAndMul,
    NPUSwiglu,
    NPUSwigluOAI,
    NPUSwigluQuant,
    NPUSwigluStepAndMul,
    NPUSwigluDeepEPKernel,
)
from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (
    NPUW4A8Int8MoEMethod,
    NPUW8A8Int8MoEMethod,
)
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
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPLLCombineInput,
        DeepEPLLDispatchOutput,
        DeepEPNormalCombineInput,
        DeepEPNormalDispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.torch_npu import (
        TorchNpuDispatchOutput,
        TorchNpuCombineInput,
    )

from sglang.srt.layers.moe.utils import (  # fixed import
    MoeRunnerBackend,
    get_moe_a2a_backend,
)


# ---------------------------------------------------------------------------
# Runner IO dataclasses
# ---------------------------------------------------------------------------
@dataclass
class TorchNpuRunnerInput(RunnerInput):
    """Input bundle for the NPU runner."""

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]  # None for unquant
    expert_tokens: torch.Tensor
    group_list_type: int  # 0 or 1 (passed to NPU ops)

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TORCH_NPU


@dataclass
class TorchNpuRunnerOutput(RunnerOutput):
    """Output bundle from the NPU runner."""

    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TORCH_NPU


# ---------------------------------------------------------------------------
# Main runner core
# ---------------------------------------------------------------------------
class TorchNpuRunnerCore(MoeRunnerCore):
    runner_backend = MoeRunnerBackend.TORCH_NPU

    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)

        kernel = config.layer.w2_kernel

        if get_moe_a2a_backend().is_deepep():
            # DeepEP path: use a unified kernel that decides quantisation
            is_quant_kernel = isinstance(
                kernel, (NPUW4A8Int8MoEMethod, NPUW8A8Int8MoEMethod)
            )
            self.activation = NPUSwigluDeepEPKernel(need_quant=is_quant_kernel)
        else:
            # Non‑DeepEP (torch_npu) path
            if isinstance(kernel, (NPUW4A8Int8MoEMethod, NPUW8A8Int8MoEMethod)):
                self.activation = NPUSwigluQuant()
            else:
                if config.activation == "npu_swiglu_oai":
                    self.activation = NPUSwigluOAI(layer=config.layer)
                elif config.activation == "silu":
                    if config.gemm1_clamp_limit is not None:
                        self.activation = NPUSwigluStepAndMul(
                            clamp_limit=config.gemm1_clamp_limit
                        )
                    else:
                        self.activation = NPUSwiglu()
                else:
                    self.activation = NPUGeluAndMul()

    def run(
        self,
        runner_input: TorchNpuRunnerInput,
        quant_info: "TorchNpuQuantInfo",
        running_state: dict,
        hooks: Optional[Any] = None,
    ) -> TorchNpuRunnerOutput:
        """
        Execute the MoE layer using NPU‑specific grouped matmul ops.
        """
        x = runner_input.hidden_states
        expert_tokens = runner_input.expert_tokens
        group_list_type = runner_input.group_list_type

        # --- w13 (gate & up) projection ---
        hidden_states = self.config.layer.w13_kernel.apply(
            self.config.layer,
            x,
            expert_tokens,
            pertoken_scale=runner_input.hidden_states_scale,
            output_dtype=torch.bfloat16,
            weight_prefix="w13",
            group_list_type=group_list_type,
        )

        # --- Activation ---
        # The DeepEP kernel expects extra dispatch metadata
        if isinstance(self.activation, NPUSwigluDeepEPKernel):
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
            self.config.layer,
            hidden_states,
            expert_tokens,
            pertoken_scale=pertoken_scale,
            output_dtype=torch.bfloat16,
            weight_prefix="w2",
            group_list_type=group_list_type,
        )

        return TorchNpuRunnerOutput(hidden_states=hidden_states)


# ---------------------------------------------------------------------------
# QuantInfo
# ---------------------------------------------------------------------------
@dataclass
class TorchNpuQuantInfo(MoeQuantInfo):
    """Quantization payload for torch‑npu."""

    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    w13_scale: Optional[torch.Tensor] = None
    w2_scale: Optional[torch.Tensor] = None
    w13_offset: Optional[torch.Tensor] = None
    w2_offset: Optional[torch.Tensor] = None
    w13_weight_bias: Optional[torch.Tensor] = None
    w2_weight_bias: Optional[torch.Tensor] = None
    w13_scale_bias: Optional[torch.Tensor] = None
    w2_scale_bias: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Pre/Post permute hooks
# ---------------------------------------------------------------------------


@register_pre_permute("torch_npu", "torch_npu")
def pre_permute_torch_npu_to_torch_npu(
    dispatch_output: TorchNpuDispatchOutput,
    quant_info: TorchNpuQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> TorchNpuRunnerInput:
    return TorchNpuRunnerInput(
        hidden_states=dispatch_output.hidden_states,
        hidden_states_scale=dispatch_output.hidden_states_scale,
        expert_tokens=dispatch_output.expert_tokens,
        group_list_type=dispatch_output.group_list_type,
    )


@register_pre_permute("deepep_normal", "torch_npu")
def pre_permute_deepep_normal_to_torch_npu(
    dispatch_output: DeepEPNormalDispatchOutput,
    quant_info: TorchNpuQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> TorchNpuRunnerInput:
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

    return TorchNpuRunnerInput(
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        expert_tokens=group_list,
        group_list_type=1,
    )


@register_pre_permute("deepep_ll", "torch_npu")
def pre_permute_deepep_ll_to_torch_npu(
    dispatch_output: DeepEPLLDispatchOutput,
    quant_info: TorchNpuQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> TorchNpuRunnerInput:
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
    return TorchNpuRunnerInput(
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        expert_tokens=group_list,
        group_list_type=1,
    )


@register_post_permute("torch_npu", "torch_npu")
def post_permute_torch_npu_to_torch_npu(
    runner_output: TorchNpuRunnerOutput,
    quant_info: TorchNpuQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> TorchNpuCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.torch_npu import (
        TorchNpuCombineInput
    )
    return TorchNpuCombineInput(hidden_states=runner_output.hidden_states)


@register_post_permute("torch_npu", "deepep_normal")
def post_permute_torch_npu_to_deepep_normal(
    runner_output: TorchNpuRunnerOutput,
    quant_info: TorchNpuQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepEPNormalCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPNormalCombineInput
    )
    return DeepEPNormalCombineInput(
        hidden_states=runner_output.hidden_states,
        topk_ids=running_state["topk_ids"],
        topk_weights=running_state["topk_weights"],
    )


@register_post_permute("torch_npu", "deepep_ll")
def post_permute_torch_npu_to_deepep_ll(
    runner_output: TorchNpuRunnerOutput,
    quant_info: TorchNpuQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepEPLLCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPLLCombineInput
    )
    return DeepEPLLCombineInput(
        hidden_states=runner_output.hidden_states,
        topk_ids=running_state["topk_ids"],
        topk_weights=running_state["topk_weights"],
    )
