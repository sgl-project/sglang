from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.configs.qwen3_next import HybridLayerType, Qwen3NextConfig

logger = logging.get_logger(__name__)


class BailingMoeLinearConfig(Qwen3NextConfig):

    model_type = "bailing_moe_linear"

    def __init__(
        self,
        vocab_size=157184,
        hidden_size=2048,
        intermediate_size=5120,
        num_hidden_layers=20,
        num_attention_heads=16,
        num_key_value_heads=4,
        hidden_act="silu",
        use_qkv_bias=False,
        use_bias=False,
        rms_norm_eps=1e-06,
        tie_word_embeddings=False,
        embedding_dropout=0.0,
        attention_dropout=0.0,
        output_dropout=0.0,
        initializer_range=0.02,
        max_position_embeddings=32768,
        rope_theta=600000.0,
        use_cache=True,
        max_window_layers=20,
        rope_scaling=None,
        pad_token_id=156892,
        eos_token_id=156892,
        num_experts=256,
        num_shared_experts=1,
        num_experts_per_tok=8,
        n_group=8,
        topk_group=4,
        moe_intermediate_size=512,
        first_k_dense_replace=1,
        head_dim=128,
        output_router_logits=False,
        use_qk_norm=True,
        num_nextn_predict_layers=0,
        mtp_loss_scaling_factor=0,
        moe_router_enable_expert_bias=True,
        routed_scaling_factor=1.0,
        layer_group_size=1,
        group_norm_size=1,
        linear_silu=False,
        **kwargs,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.use_qkv_bias = use_qkv_bias
        self.use_bias = use_bias
        self.rms_norm_eps = rms_norm_eps
        self.embedding_dropout = embedding_dropout
        self.attention_dropout = attention_dropout
        self.output_dropout = output_dropout
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.mtp_loss_scaling_factor = mtp_loss_scaling_factor
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.use_cache = use_cache
        self.max_window_layers = max_window_layers
        self.head_dim = head_dim or self.hidden_size // self.num_attention_heads
        self.rope_scaling = rope_scaling
        self.use_qk_norm = use_qk_norm
        self.moe_router_enable_expert_bias = moe_router_enable_expert_bias
        self.routed_scaling_factor = routed_scaling_factor
        # MoE configs
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_group = n_group
        self.topk_group = topk_group
        self.moe_intermediate_size = moe_intermediate_size
        self.first_k_dense_replace = first_k_dense_replace
        self.output_router_logits = output_router_logits
        # Linear configs
        self.layer_group_size = layer_group_size
        self.group_norm_size = group_norm_size
        self.linear_silu = linear_silu
        PretrainedConfig.__init__(
            self,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def layers_block_type(self):
        layer_type_list = []
        for l in range(self.num_hidden_layers):
            if (l + 1) % self.layer_group_size == 0:
                layer_type_list.append(HybridLayerType.full_attention.value)
            else:
                layer_type_list.append(HybridLayerType.linear_attention.value)
        return layer_type_list

    @property
    def mamba2_cache_params(self) -> Mamba2CacheParams:
        from sglang.srt.layers.dp_attention import get_attention_tp_size

        shape = Mamba2StateShape.create(
            tp_world_size=get_attention_tp_size(),
            intermediate_size=0,  # lightning attention
            n_groups=0,  # lightning attention
            num_heads=self.num_attention_heads,
            head_dim=self.head_dim,
            state_size=self.head_dim,
            conv_kernel=1,
        )
        return Mamba2CacheParams(shape=shape, layers=self.linear_layer_ids)
