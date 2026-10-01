"""MSCCL++ dispatch and combine data contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.layers.moe.token_dispatcher.base import (
    CombineInputFormat,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput


@dataclass(frozen=True)
class MSCCLPPDispatchOutputBase(ABC):
    """Fields shared by all MSCCL++ dispatch layouts."""

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]

    @property
    @abstractmethod
    def format(self) -> DispatchOutputFormat:
        pass


@dataclass(frozen=True)
class MSCCLPPCombineInputBase(ABC):
    """Fields shared by all MSCCL++ combine layouts."""

    hidden_states: torch.Tensor

    @property
    @abstractmethod
    def format(self) -> CombineInputFormat:
        pass


@dataclass(frozen=True)
class MSCCLPPLLDispatchOutput(MSCCLPPDispatchOutputBase, ABC):
    """Base for MSCCL++ low-latency physical layouts."""


@dataclass(frozen=True)
class MSCCLPPExpertMajorLLDispatchOutput(MSCCLPPLLDispatchOutput):
    """Padded expert-major output consumed by the Triton runner.

    * ``hidden_states``      -> ``[num_local_experts, slots_per_expert, hidden]``
    * ``masked_m``           -> valid counts per local expert
    """

    masked_m: torch.Tensor

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.MSCCLPP_LATENCY_EXPERT_MAJOR


@dataclass(frozen=True)
class MSCCLPPRankMajorLLDispatchOutput(MSCCLPPLLDispatchOutput):
    """Fixed-capacity rank-major output consumed by a compatible MoE runner."""

    topk_output: StandardTopKOutput
    expert_output_buffer: torch.Tensor
    enable_direct_send: bool

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.MSCCLPP_LATENCY_RANK_MAJOR


@dataclass(frozen=True)
class MSCCLPPLLCombineInput(MSCCLPPCombineInputBase, ABC):
    """Base for MSCCL++ low-latency combine layouts."""


@dataclass(frozen=True)
class MSCCLPPExpertMajorLLCombineInput(MSCCLPPLLCombineInput):
    """Expert-major output consumed by handle-driven combine."""

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.MSCCLPP_LATENCY_EXPERT_MAJOR


@dataclass(frozen=True)
class MSCCLPPRankMajorLLCombineInput(MSCCLPPLLCombineInput):
    """Rank-major registered output consumed by handle-driven combine."""

    apply_router_weights: bool = False

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.MSCCLPP_LATENCY_RANK_MAJOR
