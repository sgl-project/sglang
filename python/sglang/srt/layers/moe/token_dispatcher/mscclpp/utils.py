"""MSCCL++ dispatch and combine data contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import torch

from sglang.srt.layers.moe.token_dispatcher.base import (
    CombineInputFormat,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput


class MSCCLPPOutputLayout(str, Enum):
    RANK_MAJOR = "rank_major"
    EXPERT_MAJOR = "expert_major"
    TOKEN_MAJOR = "token_major"


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
class MSCCLPPDispatchOutput(MSCCLPPDispatchOutputBase):
    """MSCCL++ high-throughput token-major dispatch output.

    Fields map the public ``MoECommunicator.dispatch`` result:

    * ``hidden_states``      -> ``recv_x``                       [num_recv_tokens, hidden]
    * ``hidden_states_scale``-> ``recv_x_scales``               (optional, fp8 path)
    * ``topk_output``        -> dispatched routing with global expert ids
    * ``num_recv_tokens_per_expert`` -> per-local-expert recv counts

    ``local_expert_start`` lets local-expert runners derive their id space from
    the canonical global ids without storing a second routing tensor.
    """

    topk_output: StandardTopKOutput
    num_recv_tokens_per_expert: List[int]
    local_expert_start: int

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.MSCCLPP


@dataclass(frozen=True)
class MSCCLPPCombineInputBase(ABC):
    """Fields shared by all MSCCL++ combine layouts."""

    hidden_states: torch.Tensor

    @property
    @abstractmethod
    def format(self) -> CombineInputFormat:
        pass


@dataclass(frozen=True)
class MSCCLPPCombineInput(MSCCLPPCombineInputBase):
    """High-throughput expert output consumed by handle-driven combine."""

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.MSCCLPP


@dataclass(frozen=True)
class MSCCLPPLLDispatchOutput(MSCCLPPDispatchOutputBase, ABC):
    """Base for MSCCL++ low-latency physical layouts."""


@dataclass(frozen=True)
class MSCCLPPExpertMajorLLDispatchOutput(MSCCLPPLLDispatchOutput):
    """Padded expert-major output consumed by the Triton runner.

    * ``hidden_states``      -> ``[num_local_experts, slots_per_expert, hidden]``
    * ``masked_m``           -> valid counts per local expert
    * ``expected_m``         -> average tokens per expert (GEMM size hint)
    """

    masked_m: torch.Tensor
    expected_m: int

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.MSCCLPP_LL_EXPERT_MAJOR


@dataclass(frozen=True)
class MSCCLPPRankMajorLLDispatchOutput(MSCCLPPLLDispatchOutput):
    """Fixed-capacity rank-major output consumed by FlashInfer CUTLASS."""

    topk_output: StandardTopKOutput
    expert_output_buffer: torch.Tensor

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.MSCCLPP_LL_RANK_MAJOR


@dataclass(frozen=True)
class MSCCLPPLLCombineInput(MSCCLPPCombineInputBase, ABC):
    """Base for MSCCL++ low-latency combine layouts."""


@dataclass(frozen=True)
class MSCCLPPExpertMajorLLCombineInput(MSCCLPPLLCombineInput):
    """Expert-major output consumed by handle-driven combine."""

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.MSCCLPP_LL_EXPERT_MAJOR


@dataclass(frozen=True)
class MSCCLPPRankMajorLLCombineInput(MSCCLPPLLCombineInput):
    """Rank-major registered output consumed by handle-driven combine."""

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.MSCCLPP_LL_RANK_MAJOR
