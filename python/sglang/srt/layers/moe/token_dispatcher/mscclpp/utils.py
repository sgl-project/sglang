"""MSCCL++ dispatch and combine data contracts."""

from __future__ import annotations

from typing import Optional

import msgspec
import torch

from sglang.srt.layers.moe.token_dispatcher.base import (
    CombineInputFormat,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput


class MSCCLPPDispatchOutputBase(msgspec.Struct, frozen=True):
    """Fields shared by all MSCCL++ dispatch layouts."""

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]

    @property
    def format(self) -> DispatchOutputFormat:
        raise NotImplementedError


class MSCCLPPCombineInputBase(msgspec.Struct, frozen=True):
    """Fields shared by all MSCCL++ combine layouts."""

    hidden_states: torch.Tensor
    apply_router_weights: bool

    @property
    def format(self) -> CombineInputFormat:
        raise NotImplementedError


class MSCCLPPLatencyDispatchOutput(MSCCLPPDispatchOutputBase, frozen=True):
    """Base for MSCCL++ low-latency physical layouts."""


class MSCCLPPExpertMajorLatencyDispatchOutput(
    MSCCLPPLatencyDispatchOutput, frozen=True
):
    """Padded expert-major output consumed by the Triton runner.

    * ``hidden_states``      -> ``[num_local_experts, slots_per_expert, hidden]``
    * ``masked_m``           -> valid counts per local expert
    """

    masked_m: torch.Tensor

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.MSCCLPP_LATENCY_EXPERT_MAJOR


class MSCCLPPRankMajorLatencyDispatchOutput(MSCCLPPLatencyDispatchOutput, frozen=True):
    """Fixed-capacity rank-major output consumed by a compatible MoE runner."""

    topk_output: StandardTopKOutput
    expert_output_buffer: torch.Tensor
    enable_direct_send: bool

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.MSCCLPP_LATENCY_RANK_MAJOR


class MSCCLPPLatencyCombineInput(MSCCLPPCombineInputBase, frozen=True):
    """Base for MSCCL++ low-latency combine layouts."""


class MSCCLPPExpertMajorLatencyCombineInput(MSCCLPPLatencyCombineInput, frozen=True):
    """Expert-major output consumed by handle-driven combine."""

    apply_router_weights: bool = True

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.MSCCLPP_LATENCY_EXPERT_MAJOR


class MSCCLPPRankMajorLatencyCombineInput(MSCCLPPLatencyCombineInput, frozen=True):
    """Rank-major registered output consumed by handle-driven combine."""

    apply_router_weights: bool = False

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.MSCCLPP_LATENCY_RANK_MAJOR
