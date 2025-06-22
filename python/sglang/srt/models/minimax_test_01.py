import copy 
import math 
from collections.abc import Iterable 
from typing import Optional, Union 

import regex as re 
import torch 
import torch.distributed 
import torch.nn.functional as F 
from einops import rearrange
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank, 
    get_tensor_model_parallel_world_size
)
from sglang.srt.distributed.communication_op import tensor_model_parallel_all_reduce


def replace_weight_name(
    name: str, 
    key: Optional[str]=None, 
    to: Optional[str]=None,
    count: Optional[int]=None,
    #prefix is not used (may consider removing it)
    prefix: Optional[str]=None,
) -> str: 
    name = name.replace(key, to) if count is not None else \
        name.replace(key, to, count) 
    return name

def wegith_loader_with_alias(alias:str): 
    def wrapper(func:callable): 
        def inner_func(
            param: torch.Tensor, 
            loaded_weights: torch.Tensor, 
            *args, 
            prefix: Optional[str] = None, 
            **kwargs): 
            value = func(param, loaded_weights, *args, **kwargs)
            return value
        return inner_func
    return wrapper


class MiniMaxTest01RMSNormTP(nn.Module): 
    name = "MiniMaxTest01RMSNormTP"

    def __init__(self, hidden_size: int, eps: float=1e-6) -> None: 
        super().__init__() 
        self.tp_world = get_tensor_model_parallel_world_size() 
        self.tp_rank = get_tensor_model_parallel_rank() 
        self.weight = nn.Parameter(torch.ones(int(hidden_size/
                                                  self.tp_world)))
        self.weight.weight_loader = self.weight_loader
        self.variance_epsilon = eps 
        return 
    
    @staticmethod
    def weight_loader(
        param: nn.Parameter, 
        loaded_weight: torch.Tensor,
    ) -> None: 
        tp_world = get_tensor_model_parallel_world_size() 
        tp_rank = get_tensor_model_parallel_rank() 

        shard_size = loaded_weight.shape[0] // tp_world
        shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)
        param.data.copy_(loaded_weight[shard])
        return 
    
    def _forward(
        self, 
        x: torch.Tensor,
    ) -> torch.Tensor: 
        orig_dtype = x.dtype 
        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=1, keepdim=True, dtype=torch.float32)
        if self.tp_world > 1: 
            variance = tensor_model_parallel_all_reduce(variance) / self.tp_world 
        x = x * torch.rsqrt(variance + self.variance_epsilon)

        weight = self.weight 
        if x.size(-1) != weight.size(0):
            if self.weight.size(0) < x.size(-1): 
                repeat_count = (x.size(-1) + self.weight.size(0)) // x.size(-1)
                full_weight = self.weight.repeat(repeat_count)
                weight = full_weight[:x.size(-1)]
            else: 
                weight = self.weight[:x.size(-1)]

        x = x.to(orig_dtype) * weight 
        return x 

    def forward(
        self, 
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]: 
        assert residual is None, "RMSNorm does not support residual connections"
        return self._forward(x)
