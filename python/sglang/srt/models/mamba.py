from typing import ClassVar, Iterable, Literal, Optional, Tuple

import torch
from torch import nn
from transformers import MambaConfig

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.linear import ColumnParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.normalization import RMSNorm
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding, ParallelLMHead
from sglang.srt.managers.schedule_batch import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader


class MambaCacheParams:
    def __init__(self, conv_state: torch.Tensor, ssm_state: torch.Tensor, state_indices_tensor: torch.Tensor):
        self.conv_state = conv_state
        self.ssm_state = ssm_state
        self.state_indices_tensor = state_indices_tensor

    def at_layer_idx(self, layer_idx):
        return MambaCacheParams(
            self.conv_state[layer_idx],
            self.ssm_state[layer_idx],
            self.state_indices_tensor
        )


class MambaMixer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ssm_state_size: int,
        conv_kernel_size: int,
        intermediate_size: int,
        time_step_rank: int,
        use_conv_bias: bool,
        use_bias: bool,
        activation="silu",
        prefix: str = "",
    ):
        super().__init__()
        self.time_step_rank = time_step_rank
        self.ssm_state_size = ssm_state_size
        self.activation = activation
        self.conv_kernel_size = conv_kernel_size
        self.intermediate_size = intermediate_size
        
        self.conv1d = ColumnParallelLinear(
            input_size=conv_kernel_size,
            output_size=intermediate_size,
            bias=use_conv_bias,
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)
        
        self.in_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=use_bias
        )
        
        self.x_proj = RowParallelLinear(
            intermediate_size,
            time_step_rank + ssm_state_size * 2,
            bias=False,
        )
        
        self.dt_proj = ColumnParallelLinear(
            time_step_rank,
            intermediate_size,
            bias=True,
            skip_bias_add=True
        )
        
        tp_size = get_tensor_model_parallel_world_size()
        self.A = nn.Parameter(torch.empty(intermediate_size // tp_size, ssm_state_size))
        self.D = nn.Parameter(torch.ones(intermediate_size // tp_size))
        
        self.out_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=use_bias,
        )

    def forward(self, hidden_states: torch.Tensor, mamba_cache_params: MambaCacheParams):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project input
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)
        
        # TODO: Replace with causal_conv1d kernel
        conv_weights = self.conv1d.weight.squeeze(1)
        x = x @ conv_weights.T
        
        # SSM operations
        x_proj = self.x_proj(x)
        delta_t_raw = x_proj[..., :self.time_step_rank]
        B = x_proj[..., self.time_step_rank:self.time_step_rank + self.ssm_state_size]
        C = x_proj[..., self.time_step_rank + self.ssm_state_size:]
        
        delta, _ = self.dt_proj(delta_t_raw)
        delta = torch.nn.functional.softplus(delta)
        
        # TODO: Replace with selective_scan kernel
        y = x * self.D
        
        # Apply activation
        if self.activation == "silu":
            z = torch.nn.functional.silu(z)
        
        # Output projection
        output = self.out_proj(y * z)
        
        return output


class MambaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: MambaConfig,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.mixer = MambaMixer(
            hidden_size=config.hidden_size,
            ssm_state_size=config.state_size,
            conv_kernel_size=config.conv_kernel,
            intermediate_size=config.intermediate_size,
            time_step_rank=config.time_step_rank,
            use_conv_bias=config.use_conv_bias,
            use_bias=config.use_bias,
            activation=config.hidden_act,
            prefix=f"{prefix}.mixer"
        )
        
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        mamba_cache_params: MambaCacheParams,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, residual = self.norm(hidden_states, residual)
        
        hidden_states = self.mixer(hidden_states, mamba_cache_params)
        return hidden_states, residual


class MambaModel(nn.Module):
    def __init__(
        self,
        config: MambaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        self.embeddings = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )
        
        self.layers = nn.ModuleList([
            MambaDecoderLayer(config, i, quant_config, prefix=f"{prefix}.layers.{i}")
            for i in range(config.num_hidden_layers)
        ])
        
        self.norm_f = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mamba_cache_params: Optional[MambaCacheParams] = None,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embeddings(input_ids)
        else:
            hidden_states = input_embeds
        
        residual = None
        
        for i, layer in enumerate(self.layers):
            layer_cache_params = None
            if mamba_cache_params is not None:
                layer_cache_params = mamba_cache_params.at_layer_idx(i)
            
            hidden_states, residual = layer(
                hidden_states,
                residual,
                layer_cache_params,
            )
        
        hidden_states, _ = self.norm_f(hidden_states, residual)
        return hidden_states


class MambaForCausalLM(nn.Module):
    has_inner_state: ClassVar[Literal[True]] = True
    is_attention_free: ClassVar[Literal[True]] = True
    
    def __init__(
        self,
        config: MambaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config=None,
        lora_config=None,
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        
        self.backbone = MambaModel(config, quant_config)
        self.unpadded_vocab_size = config.vocab_size
        
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            quant_config=quant_config,
        )
        
        self.logits_processor = LogitsProcessor(
            self.unpadded_vocab_size,
            config.vocab_size,
            scale=1.0,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
    ) -> torch.Tensor:
        # TODO: Get mamba cache params from forward_batch
        mamba_cache_params = getattr(forward_batch, 'mamba_cache_params', None)
        
        hidden_states = self.backbone(
            input_ids,
            positions,
            mamba_cache_params,
            input_embeds,
        )
        
        if get_embedding:
            return hidden_states
        
        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
        )

    def get_config(self):
        return self.config
    
    def get_num_kv_heads(self):
        return 0
    
    def get_kv_head_dim(self):
        return 0

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = MambaForCausalLM