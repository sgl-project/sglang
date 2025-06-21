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
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.linear import ( ColumnParallelLinear, 
                                      MergedColumnParallelLinear,
                                      RowParallelLinear, 
                                      ReplicatedLinear, 
                                      QKVParallelLinear)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE


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

def weight_loader_with_alias(alias:str): 
    def wrapper(func:callable): 
        def wrapper_weight_loader(
            param: torch.Tensor, 
            loaded_weights: torch.Tensor, 
            *args, 
            prefix: Optional[str] = None, 
            **kwargs): 
            value = func(param, loaded_weights, *args, **kwargs)
            return value
        return wrapper_weight_loader
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
        if residual is not None:
            raise ValueError("RMSNorm does not support residual connections")
        return self._forward(x)
    

class MiniMaxText01RotaryEmbedding(nn.Module): 
    name = "MiniMaxText01RotaryEmbedding"

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position: int, 
        base: float, 
        is_neox_style: bool,
        cache_dtype: torch.dtype,
    ) -> None: 
        super().__init__()
        self.head_size = head_size 
        self.rotary_dim = rotary_dim 
        self.max_position_embeddings = max_position 
        self.base = base 
        self.is_neox_style = is_neox_style 
        self.cache_dtype = cache_dtype 
        cache = self._compute_cos_sin_cache().to(cache_dtype)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_inv_freq(self, base: float) -> torch.Tensor: 
        """Compute inverse frequency for rotary embedding"""
        inv_freq = 1.0 / (base ** (torch.arange(
            0,self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))
        return inv_freq
    
    def _compute_cos_sin_cache(self) -> torch.Tensor: 
        """Compute the sin and cos cache"""
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin() 
        cache = torch.stack((cos, sin), dim=-1)
        return cache 
    
    def forward( 
        self, 
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: 
        from vllm import _custom_ops as ops 
        self.cos_sin_cache = self.cos_sin_cache.to(positions.device)
        query_cast = query.to(self.cache_dtype)
        key_cast = key.to(self.cache_dtype)
        ops.rotary_embedding(positions, query_cast, key_cast, self.head_size,
                             self.cos_sin_cache, self.is_neox_style)
        query = query_cast.to(query.dtype)
        key = key_cast.to(key.dtype)
        return query, key 
    

class MiniMaxText01MLP(nn.Module): 

    def __init__(
        self, 
        hidden_size: int, 
        intermediate_size: int, 
        quant_config: Optional[QuantizationConfig] = None,
        layer_idx: Optional[int] = None,
        prefix: str = "mlp",
    ) -> None: 
        super().__init__()
        self.layer_idx = layer_idx 

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, 
            [intermediate_size] * 2, 
            bias=False, 
            quant_config=quant_config, 
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size, 
            hidden_size, 
            bias=False, 
            quant_config=quant_config, 
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()
        return 
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x 
    

class MiniMaxText01MoE(nn.Module): 

    def __init__(
        self, 
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        layer_idx: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "moe",
    ) -> None: 
        super().__init__()
        self.layer_idx = layer_idx 
        self.tp_size = get_tensor_model_parallel_world_size() 
        self.num_total_experts = num_experts 
        self.top_k = top_k 
        self.hidden_size = hidden_size 
        self.intermediate_size = intermediate_size // self.tp_size 
        self.quant_config = quant_config 

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype 

        self.gate = ReplicatedLinear(
            self.hidden_size, 
            self.num_total_experts, 
            bias=False, 
            params_dtype=torch.float32, 
            quant_config=None, 
            prefix=f"{prefix}.gate",
        )
        self.gate.weight.weight_loader = MiniMaxText01MoE.gate_weight_loader

        self.experts = FusedMoE(
            num_experts=self.num_total_experts, 
            top_k=self.top_k, 
            hidden_size=self.hidden_size, 
            intermediate_size=self.intermediate_size * self.tp_size, 
            params_dtype=self.params_dtype, 
            reduce_results=True, 
            renormalize=True, 
            quant_config=self.quant_config, 
            tp_size=self.tp_size, 
            prefix=f"{prefix}.experts",
        )
        return 
    
    @staticmethod
    def gate_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None: 
        assert param.size() == loaded_weight.size(), f"Shape mismatch: {param.size()} vs {loaded_weight.size()}"
        param.data.copy_(loaded_weight.to(torch.float32))
        return 
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: 
        num_tokens, hidden_size = hidden_states.shape 
        hidden_states = hidden_states.view(-1, self.hidden_size)
        router_logits_fp32, _ = self.gate(hidden_states.to(torch.float32))
        final_hidden_states = self.experts(
            hidden_states, router_logits_fp32.to(hidden_states.dtype))
        final_hidden = final_hidden_states.view(num_tokens, hidden_size)
        return final_hidden 
