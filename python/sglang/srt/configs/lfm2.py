from dataclasses import dataclass
from typing import Any, Optional
from transformers.configuration_utils import PretrainedConfig

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape


class Lfm2Config(PretrainedConfig):
    model_type: str = "lfm2"
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    vocab_size: int
    block_dim: int  
    block_ff_dim: int  
    block_multiple_of: int  
    block_auto_adjust_ff_dim: bool
    block_ffn_dim_multiplier: Optional[float]
    block_use_swiglu: bool
    block_norm_eps: float
    block_use_xavier_init: bool
    block_mlp_init_scale: float
    block_out_init_scale: float
    conv_L_cache: int  
    conv_bias: bool
    conv_dim: int  
    conv_dim_out: int  
    conv_use_xavier_init: bool
    full_attn_idxs: list[int]  
    use_pos_enc: bool
    rope_theta: float
    rope_scaling: Optional[dict] = None
    norm_eps: float
    initializer_range: float
    use_cache: bool
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    
    def __init__(
        self,
        hidden_size: int = 1536,
        num_hidden_layers: int = 16,
        num_attention_heads: int = 24,
        num_key_value_heads: int = 8,
        num_heads: int = 24,  
        max_position_embeddings: int = 128000,
        vocab_size: int = 65536,
        block_dim: int = 1536,
        block_ff_dim: int = 10240,
        block_multiple_of: int = 256,
        block_auto_adjust_ff_dim: bool = True,
        block_ffn_dim_multiplier: Optional[float] = 1.0,
        block_use_swiglu: bool = True,
        block_norm_eps: float = 1e-05,
        block_use_xavier_init: bool = True,
        block_mlp_init_scale: float = 1.0,
        block_out_init_scale: float = 1.0,
        conv_L_cache: int = 3,
        conv_bias: bool = False,
        conv_dim: int = 1536,
        conv_dim_out: int = 1536,
        conv_use_xavier_init: bool = True,
        full_attn_idxs: Optional[list[int]] = None,
        use_pos_enc: bool = True,
        rope_theta: float = 1000000.0,
        rope_scaling: Optional[dict] = None,
        norm_eps: float = 1e-05,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        bos_token_id: int = 1,
        eos_token_id: int = 7,
        pad_token_id: int = 0,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.vocab_size = vocab_size
        
        self.block_dim = block_dim
        self.block_ff_dim = block_ff_dim
        self.block_multiple_of = block_multiple_of
        self.block_auto_adjust_ff_dim = block_auto_adjust_ff_dim
        self.block_ffn_dim_multiplier = block_ffn_dim_multiplier
        self.block_use_swiglu = block_use_swiglu
        self.block_norm_eps = block_norm_eps
        self.block_use_xavier_init = block_use_xavier_init
        self.block_mlp_init_scale = block_mlp_init_scale
        self.block_out_init_scale = block_out_init_scale
        
        self.conv_L_cache = conv_L_cache
        self.conv_bias = conv_bias
        self.conv_dim = conv_dim
        self.conv_dim_out = conv_dim_out
        self.conv_use_xavier_init = conv_use_xavier_init
        
        self.full_attn_idxs = full_attn_idxs 
        self.use_pos_enc = use_pos_enc
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        
        self.norm_eps = norm_eps
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )
    
    @property
    def layer_types(self) -> list[str]:
        types = []
        for i in range(self.num_hidden_layers):
            if i in self.full_attn_idxs:
                types.append("full_attention")
            else:
                types.append("short_conv")
        return types
    
    @property
    def full_attention_layer_ids(self) -> list[int]:
        return [
            idx
            for idx, layer_type in enumerate(self.layer_types)
            if layer_type == "full_attention"
        ]
    
    @property
    def short_conv_layer_ids(self) -> list[int]:
        return [
            idx
            for idx, layer_type in enumerate(self.layer_types)
            if layer_type == "short_conv"
        ]
    
    @property
    def mamba2_cache_params(self) -> Mamba2CacheParams:
        from sglang.srt.layers.dp_attention import get_attention_tp_size
        
        shape = Mamba2StateShape.create(
            tp_world_size=get_attention_tp_size(),
            intermediate_size=self.conv_dim,  
            n_groups=1,
            num_heads=1,
            head_dim=self.conv_dim,
            state_size=0,  
            conv_kernel=self.conv_L_cache,
        )
        
        return Mamba2CacheParams(
            shape=shape,
            layers=self.short_conv_layer_ids,
        )
