from __future__ import annotations

from typing import NamedTuple, Optional

import torch

from sglang.srt.hardware_backend.npu.moe.finalize_routing import (
    NPUFinalizeRouting,
    NPUMoETokenUnpermute,
)
from sglang.srt.hardware_backend.npu.moe.init_routing import (
    NPUMoEInitRouting_Quant,
    NPUMoEInitRouting_v1,
    NPUMoEInitRouting_v2,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInputFormat,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.moe.utils import (
    get_ascend_dispatcher_output_dtype,
    DispatcherOutputDtype,
)


class TorchNpuDispatchOutput(NamedTuple):
    """Dispatch output specific to the TorchNpu dispatcher."""

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_output: TopKOutput
    expanded_row_idx: torch.Tensor
    expert_tokens: torch.Tensor
    group_list_type: int

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.TORCH_NPU


class TorchNpuCombineInput(NamedTuple):
    """Combine input specific to the TorchNpu dispatcher."""

    hidden_states: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.TORCH_NPU


class TorchNpuDispatcher(BaseDispatcher):
    """
    NPU MoE dispatcher that selects init / finalize routing kernels
    depending on the inference phase (prefill vs decode).
    """

    def __init__(self, moe_runner_config: MoeRunnerConfig):
        super().__init__()
        self.num_experts = moe_runner_config.num_experts
        self._dispatch_output: Optional[TorchNpuDispatchOutput] = None

        # These will be populated by set_quant_config / set_ascend_dispatcher_output_dtype
        self.quant_config: Optional[dict] = None
        self.set_ascend_dispatcher_output_dtype()

    def set_quant_config(self, quant_config: dict) -> None:
        self.quant_config = quant_config
        self.set_ascend_dispatcher_output_dtype()

    def set_ascend_dispatcher_output_dtype(self) -> None:
        """
        Choose the init & finalize routing kernels for both prefill and decode,
        based on the quantisation config (bf16 / int8).
        """
        self.ascend_dispatcher_output_dtype = get_ascend_dispatcher_output_dtype(self)

        if self.ascend_dispatcher_output_dtype == DispatcherOutputDtype.BF16:
            # Prefill
            self.init_routing_prefill = NPUMoEInitRouting_v2()
            self.finalize_routing_prefill = NPUFinalizeRouting(drop_pad_mode=3)
            self.group_list_type_prefill = 1
            # Decode
            self.init_routing_decode = NPUMoEInitRouting_v2()
            self.finalize_routing_decode = NPUFinalizeRouting(drop_pad_mode=3)
            self.group_list_type_decode = 1

        elif self.ascend_dispatcher_output_dtype == DispatcherOutputDtype.INT8:
            # Prefill
            self.init_routing_prefill = NPUMoEInitRouting_Quant()
            self.finalize_routing_prefill = NPUMoETokenUnpermute()
            # Decode
            self.init_routing_decode = NPUMoEInitRouting_v2()  # example
            self.finalize_routing_decode = NPUFinalizeRouting()  # example
            self.group_list_type = 0

        else:
            raise ValueError(
                f"Unsupported ascend_dispatcher_output_dtype: {self.ascend_dispatcher_output_dtype}"
            )

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> TorchNpuDispatchOutput:
        """
        Permute tokens to expert‑first order and optionally quantise them.
        Automatically selects prefill or decode routing kernels.
        """
        # Determine inference phase
        if torch.npu.is_current_stream_capturing():
            phase = "decode"
            init = self.init_routing_decode
            finalize = self.finalize_routing_decode
            group_list_type = self.group_list_type_decode
        else:
            phase = "prefill"
            init = self.init_routing_prefill
            finalize = self.finalize_routing_prefill
            group_list_type = self.group_list_type_prefill

        # Store phase so combine() uses the matching finalize kernel
        self._dispatch_phase = phase

        # Perform routing
        (
            permuted_hidden_states,
            expanded_row_idx,
            expert_tokens,
            hidden_states_scale,
        ) = init._init_routing(
            hidden_states,
            topk_output.topk_ids,
            self.num_experts,
        )

        self._dispatch_output = TorchNpuDispatchOutput(
            hidden_states=permuted_hidden_states,
            hidden_states_scale=hidden_states_scale,
            topk_output=topk_output,
            expanded_row_idx=expanded_row_idx,
            expert_tokens=expert_tokens,
            group_list_type=group_list_type,
        )
        return self._dispatch_output

    def combine(self, combine_input: TorchNpuCombineInput) -> torch.Tensor:
        """
        Reverse the token permutation and apply gating weights.
        Uses the same finalize kernel that was selected during dispatch.
        """
        if self._dispatch_output is None:
            raise RuntimeError("combine() called before dispatch()")

        # Retrieve the finalize routing that matches the dispatch phase
        phase = getattr(self, "_dispatch_phase", "prefill")
        finalize = (
            self.finalize_routing_decode
            if phase == "decode"
            else self.finalize_routing_prefill
        )

        dispatch_out = self._dispatch_output
        final_hidden_states = finalize._finalize_routing(
            combine_input.hidden_states,
            topk_weights=dispatch_out.topk_output.topk_weights,
            expanded_row_idx=dispatch_out.expanded_row_idx,
            topk_ids=dispatch_out.topk_output.topk_ids,
        )
        self._dispatch_output = None
        return final_hidden_states
