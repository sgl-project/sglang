"""TeleChat4 model configuration."""

import logging
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional

from transformers import PretrainedConfig

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class TeleChat4Config(PretrainedConfig):
    model_type: ClassVar[str] = "telechat4"

    architectures: List[str]
    attention_bias: bool = False
    attention_dropout: float = 0.0
    bos_token_id: int = 1
    eos_token_id: int = 2
    ep_size: int = 1
    first_k_dense_replace: int = 0
    hidden_act: str = "silu"
    hidden_size: int = 3584

    # DSA (Deep Sparse Attention) fields. Only present in DSA-enabled checkpoints;
    # None for non-DSA models. is_deepseek_dsa() checks index_topk is not None.
    index_head_dim: Optional[int] = None
    index_n_heads: Optional[int] = None
    index_topk: Optional[int] = None

    initializer_range: float = 0.02
    intermediate_size: int = 9216
    kv_lora_rank: int = 512
    max_position_embeddings: int = 262144
    moe_intermediate_size: int = 1024
    moe_layer_freq: int = 1
    n_group: int = 1
    n_routed_experts: int = 64
    n_shared_experts: int = 1
    norm_topk_prob: bool = True

    num_attention_heads: int = 32
    num_experts_per_tok: int = 4
    num_hidden_layers: int = 40
    num_key_value_heads: int = 32

    q_lora_rank: int = 768
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64

    rms_norm_eps: float = 1e-6

    rope_parameters: Optional[Dict] = None
    rope_theta: int = 10000

    routed_scaling_factor: float = 2.0
    scoring_func: str = "sigmoid"

    tie_word_embeddings: bool = False

    topk_group: int = 1
    topk_method: str = "noaux_tc"

    use_cache: bool = True
    v_head_dim: int = 128
    vocab_size: int = 131072

    num_residual_streams: int = 4
    mhc_sinkhorn_iterations: int = 20
    mhc_init_gating_factor: float = 0.01
    mhc_h_res_clamp_min: float = -30.0
    mhc_h_res_clamp_max: float = 30.0

    num_nextn_predict_layers: int = 0
