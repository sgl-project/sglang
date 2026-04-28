from dataclasses import field
from typing import Any, Dict, List, Optional

from transformers import PretrainedConfig


class DeepSeekV4Config(PretrainedConfig):
    attention_bias: bool = False
    attention_dropout: float = 0.0
    bos_token_id: int = 0
    eos_token_id: int = 1
    pad_token_id: Optional[int] = None
    ep_size: int = 1
    first_k_dense_replace: int = 0
    hidden_act: str = "silu"
    hidden_size: int = 4096
    index_head_dim: int = 128
    index_n_heads: int = 64
    index_topk: int = 512
    initializer_range: float = 0.02
    intermediate_size: int = 2048
    kv_lora_rank: int = 512
    max_position_embeddings: int = 65536
    model_type: str = "deepseek_v4"
    moe_intermediate_size: int = 2048
    moe_layer_freq: int = 1
    n_group: int = 8
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    norm_topk_prob: bool = True

    num_attention_heads: int = 64
    num_experts_per_tok: int = 6
    num_hidden_layers: int = 43
    num_key_value_heads: int = 1

    q_lora_rank: int = 1024
    qk_nope_head_dim: int = 448
    qk_rope_head_dim: int = 64

    quantization_config: Optional[Dict[str, Any]] = None

    rms_norm_eps: float = 1e-6

    rope_scaling: Dict[str, float] = field(default_factory=dict)
    rope_theta: int = 10000

    routed_scaling_factor: float = 1.5
    scoring_func: str = "sqrtsoftplus"

    tie_word_embeddings: bool = False

    topk_group: int = 8
    topk_method: str = "noaux_tc"

    use_cache: bool = True
    v_head_dim: int = 512
    vocab_size: int = 129280
    o_lora_rank: int = 1024
    o_groups: int = 8
    window_size: int = 128

    compress_rope_theta: int = 40000
    compress_ratios: List[int] = field(default_factory=list)

    n_hash_layers: int = 3
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
