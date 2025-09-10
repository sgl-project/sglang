import logging

import torch

from typing import TYPE_CHECKING, List, NamedTuple, Optional, Tuple, Union

from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    BaseDispatcherConfig,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import get_bool_env_var, is_hip

from enum import Enum, IntEnum, auto

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()

logger = logging.getLogger(__name__)

class MoRINormalOutput(NamedTuple):
    """MoRI normal dispatch output."""
    # NOTE: To-be coded when MoRI supports HT

    #hidden_states: torch.Tensor | Tuple[torch.Tensor, torch.Tensor]
    #topk_idx: torch.Tensor
    #topk_weights: torch.Tensor
    #num_recv_tokens_per_expert: List[int]

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.MORI_NORMAL


class MoRILLOutput(NamedTuple):
    """MoRI low latency dispatch output."""
    # TODO : do below

    #hidden_states_fp8: Tuple[torch.Tensor, torch.Tensor]
    #topk_idx: torch.Tensor
    #topk_weights: torch.Tensor
    #masked_m: torch.Tensor
    #expected_m: int

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.MORI_LL

assert isinstance(MoRINormalOutput, DispatchOutput)
assert isinstance(MoRILLOutput, DispatchOutput)

class MoRIDispatchMode(IntEnum):
    NORMAL = auto()
    LOW_LATENCY = auto()

class MoRIConfig(BaseDispatcherConfig):
    _instance = None

    def __init__(self):
        pass

class MoRIDispatcher(BaseDispatcher):
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool = False,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
        # deepep_mode: DeepEPMode = DeepEPMode.AUTO,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ):
        # TODO: should Implement MORI dispatcher, below is nonsense code for removal
        # NOTE: to implement TBO with MoRI, first use same function name and logic flow
        #       of DeepEPDispatcher (check _use_aiter or _use_hip in deepep.py)
        pass
