"""Architecture for the Ascend Ops MoE token dispatcher.

This module intentionally contains the dispatcher contract and path selection
only.  The NPU communication and routing operators will be added behind the
four private path methods as they are implemented.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch

from sglang.srt.layers.dp_attention import get_is_extend_in_batch
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.moe.utils import AscendOpsConfig
from sglang.srt.runtime_context import get_parallel


class AscendOpsDispatchOutput(NamedTuple):
    """Metadata passed from Ascend Ops dispatch to the MoE runner."""

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    expanded_row_idx: torch.Tensor
    expert_tokens: torch.Tensor
    group_list_type: int
    prefill_mode: str
    is_shortcut: bool = False

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.ASCEND_OPS


class AscendOpsCombineInput(NamedTuple):
    """Expert output consumed by the matching Ascend Ops combine path."""

    hidden_states: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.ASCEND_OPS


class AscendOpsDispatcher(BaseDispatcher):
    """Select one of the four Ascend Ops dispatch/combine execution paths.

    The methods below form the integration boundary for the future NPU
    operators.  Keeping path selection here makes the runner independent of
    whether a batch is prefill or decode and keeps the output contract stable.
    """

    def __init__(
        self,
        moe_runner_config: MoeRunnerConfig,
        ascend_ops_config: Optional[AscendOpsConfig] = None,
    ) -> None:
        super().__init__()
        self.ascend_ops_config = ascend_ops_config or AscendOpsConfig()
        # Prefill communication is scoped to TP; decode routing is scoped to EP.
        self.tp_group = get_parallel().tp_group.device_group
        self.ep_group = get_parallel().moe_ep_group.device_group
        self.num_experts = moe_runner_config.num_experts
        self.top_k = moe_runner_config.top_k
        self.hidden_size = moe_runner_config.hidden_size
        self._dispatch_output: Optional[AscendOpsDispatchOutput] = None
        self._dispatch_path: Optional[str] = None
        self._dispatch_group = None

    def set_quant_config(self, quant_config: dict) -> None:
        """Store quantization settings for the future routing kernels."""

        self.quant_config = quant_config

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> DispatchOutput:
        topk_weights, topk_ids, _ = topk_output
        if get_is_extend_in_batch():
            if self.ascend_ops_config.prefill == "allgather":
                return self._dispatch_prefill_allgather(
                    hidden_states, topk_weights, topk_ids
                )
            return self._dispatch_prefill_alltoallv(
                hidden_states, topk_weights, topk_ids
            )

        if self.ascend_ops_config.decode == "low_latency":
            return self._dispatch_decode_low_latency(
                hidden_states, topk_weights, topk_ids
            )
        return self._dispatch_decode_fused_op(hidden_states, topk_weights, topk_ids)

    def combine(self, combine_input: CombineInput) -> torch.Tensor:
        if combine_input.format != CombineInputFormat.ASCEND_OPS:
            raise TypeError(
                "AscendOpsDispatcher.combine expects an AscendOpsCombineInput, "
                f"got {combine_input.format.value}"
            )
        if self._dispatch_output is None or self._dispatch_path is None:
            raise RuntimeError("Ascend Ops combine called before dispatch")

        path = self._dispatch_path
        if path == "prefill_allgather":
            return self._combine_prefill_allgather(combine_input)
        if path == "prefill_alltoallv":
            return self._combine_prefill_alltoallv(combine_input)
        if path == "decode_low_latency":
            return self._combine_decode_low_latency(combine_input)
        if path == "decode_fused_op":
            return self._combine_decode_fused_op(combine_input)
        raise RuntimeError(f"unknown Ascend Ops dispatch path: {path}")

    # Dispatch path boundaries -------------------------------------------------

    def _dispatch_prefill_allgather(self, hidden_states, topk_weights, topk_ids):
        self._dispatch_path = "prefill_allgather"
        self._dispatch_group = self.tp_group
        raise NotImplementedError("Ascend Ops prefill allgather dispatch is pending")

    def _dispatch_prefill_alltoallv(self, hidden_states, topk_weights, topk_ids):
        self._dispatch_path = "prefill_alltoallv"
        self._dispatch_group = self.tp_group
        raise NotImplementedError("Ascend Ops prefill alltoallv dispatch is pending")

    def _dispatch_decode_low_latency(self, hidden_states, topk_weights, topk_ids):
        self._dispatch_path = "decode_low_latency"
        self._dispatch_group = self.ep_group
        raise NotImplementedError("Ascend Ops decode low-latency dispatch is pending")

    def _dispatch_decode_fused_op(self, hidden_states, topk_weights, topk_ids):
        self._dispatch_path = "decode_fused_op"
        self._dispatch_group = self.ep_group
        raise NotImplementedError("Ascend Ops decode fused-op dispatch is pending")

    # Combine path boundaries --------------------------------------------------

    def _combine_prefill_allgather(self, combine_input):
        raise NotImplementedError("Ascend Ops prefill allgather combine is pending")

    def _combine_prefill_alltoallv(self, combine_input):
        raise NotImplementedError("Ascend Ops prefill alltoallv combine is pending")

    def _combine_decode_low_latency(self, combine_input):
        raise NotImplementedError("Ascend Ops decode low-latency combine is pending")

    def _combine_decode_fused_op(self, combine_input):
        raise NotImplementedError("Ascend Ops decode fused-op combine is pending")
