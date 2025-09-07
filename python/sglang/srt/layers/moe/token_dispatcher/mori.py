import logging

import torch

from sglang.srt.layers.moe.token_dispatcher.base_dispatcher import BaseDispatcher
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import get_bool_env_var, is_hip

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()

logger = logging.getLogger(__name__)


class MoRIEPDispatcher(BaseDispatcher):
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
