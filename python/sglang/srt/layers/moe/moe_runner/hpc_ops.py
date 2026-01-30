from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.distributed import (
    get_moe_expert_parallel_rank,
    get_moe_expert_parallel_world_size,
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
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )

try:
    import hpc
except ImportError:
    raise ImportError(
        "hpc_ops import failed, please install hpc_ops package from https://github.com/Tencent/hpc-ops"
    )


@dataclass
class HpcOpsRunnerInput(RunnerInput):
    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.HPC_OPS


@dataclass
class HpcOpsRunnerOutput(RunnerOutput):
    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.HPC_OPS


@dataclass
class HpcOpsMoeQuantInfo(MoeQuantInfo):
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    w13_scale: Optional[torch.Tensor] = None
    w2_scale: Optional[torch.Tensor] = None
    block_shape: Optional[List[int]] = None
    num_local_experts: Optional[int] = None


class HpcOpsRunnerCore(MoeRunnerCore):
    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)
        assert self.config.activation == "silu"
        assert self.config.is_gated

    def run(
        self,
        runner_input: HpcOpsRunnerInput,
        quant_info: HpcOpsMoeQuantInfo,
        running_state: dict,
    ) -> HpcOpsRunnerOutput:
        if quant_info.block_quant:
            hidden_states = self._run_per_block_quant(runner_input, quant_info)
        else:
            hidden_states = self._run_per_tensor_quant(runner_input, quant_info)

        return HpcOpsRunnerOutput(hidden_states=hidden_states)

    def _run_per_block_quant(
        self, runner_input: HpcOpsRunnerInput, quant_info: HpcOpsMoeQuantInfo
    ) -> torch.Tensor:
        x_q, x_scale = sglang_per_token_group_quant_fp8(runner_input.hidden_states, 128)
        rank_ep = get_moe_expert_parallel_rank()
        size_ep = get_moe_expert_parallel_world_size()

        # Extract variables from quant_info for consistency
        w13_weight = quant_info.w13_weight
        w2_weight = quant_info.w2_weight
        w13_scale = quant_info.w13_scale
        w2_scale = quant_info.w2_scale
        num_local_experts = quant_info.num_local_experts

        # Convert local expert IDs to global expert IDs if needed
        topk_ids = runner_input.topk_ids
        if size_ep > 1:
            topk_ids = torch.where(
                topk_ids >= 0,
                topk_ids + rank_ep * num_local_experts,
                topk_ids,
            )

        output = hpc.fuse_moe_blockwise_fp8(
            x_q,
            x_scale,
            w13_weight,
            w13_scale,
            w2_weight,
            w2_scale,
            topk_ids,
            runner_input.topk_weights,
            rank_ep,
            num_local_experts,
        )

        return output

    def _run_per_tensor_quant(
        self,
        runner_input: HpcOpsRunnerInput,
        quant_info: HpcOpsMoeQuantInfo,
        running_state: dict,
    ) -> HpcOpsRunnerOutput:
        raise NotImplementedError("Per-tensor quantization is not supported yet")


@register_pre_permute("standard", "hpc_ops")
def pre_permute_standard_to_hpc_ops(
    dispatch_output: StandardDispatchOutput,
    quant_info: HpcOpsMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> HpcOpsRunnerInput:

    return HpcOpsRunnerInput(
        hidden_states=dispatch_output.hidden_states,
        topk_weights=dispatch_output.topk_output.topk_weights,
        topk_ids=dispatch_output.topk_output.topk_ids,
    )


@register_post_permute("hpc_ops", "standard")
def post_permute_hpc_ops_to_standard(
    runner_output: HpcOpsRunnerOutput,
    quant_info: HpcOpsMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    return StandardCombineInput(hidden_states=runner_output.hidden_states)
