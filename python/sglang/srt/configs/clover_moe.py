from copy import deepcopy

from transformers import PretrainedConfig

from sglang.srt.configs.linear_attn_model_registry import (
    LinearAttnModelSpec,
    register_linear_attn_model,
)
from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateShape,
    mamba2_state_dtype,
)
from sglang.srt.configs.update_config import adjust_tp_num_heads_if_necessary
from sglang.srt.runtime_context import get_parallel, linear_attn_tp_size
from sglang.srt.utils import is_cpu


class CloverMoeConfig(PretrainedConfig):
    model_type = "clover_moe"

    def __init__(
        self,
        attention_bias=False,
        attention_dropout=0.0,
        attn_output_gate=True,
        block_size=32,
        bos_token_id=248044,
        dtype="bfloat16",
        eos_token_id=248044,
        full_attention_interval=4,
        head_dim=256,
        hidden_act="silu",
        hidden_size=2048,
        initializer_range=0.02,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_value_head_dim=128,
        mamba_ssm_dtype="float32",
        mask_token_id=151669,
        max_position_embeddings=262144,
        moe_block_coreset_enabled=True,
        moe_block_coreset_top_p=0.9,
        moe_intermediate_size=512,
        num_attention_heads=16,
        num_experts=256,
        num_experts_per_tok=8,
        num_hidden_layers=40,
        num_key_value_heads=2,
        output_router_logits=False,
        pad_token_id=None,
        partial_rotary_factor=0.25,
        rms_norm_eps=1e-06,
        rope_parameters=None,
        shared_expert_intermediate_size=512,
        tie_word_embeddings=False,
        use_cache=True,
        vocab_size=248320,
        num_shared_experts=1,
        attention_backend="sdpa",
        layer_types=None,
        norm_topk_prob=True,
        output_gate_type=None,
        **kwargs,
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            dtype=dtype,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.attn_output_gate = attn_output_gate
        self.block_size = block_size
        self.full_attention_interval = full_attention_interval
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_value_head_dim = linear_value_head_dim
        self.mamba_ssm_dtype = mamba_ssm_dtype
        self.mask_token_id = mask_token_id
        self.max_position_embeddings = max_position_embeddings
        self.moe_block_coreset_enabled = moe_block_coreset_enabled
        self.moe_block_coreset_top_p = moe_block_coreset_top_p
        self.moe_intermediate_size = moe_intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.output_router_logits = output_router_logits
        self.partial_rotary_factor = partial_rotary_factor
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.vocab_size = vocab_size
        self.num_shared_experts = num_shared_experts
        self.attention_backend = attention_backend
        self.norm_topk_prob = norm_topk_prob
        self.output_gate_type = output_gate_type
        if num_shared_experts not in (0, 1):
            raise ValueError("Clover supports zero or one shared expert")
        self.shared_expert_intermediate_size = (
            shared_expert_intermediate_size if num_shared_experts else 0
        )

        self.rope_parameters = deepcopy(
            rope_parameters
            if rope_parameters is not None
            else {
                "rope_type": "default",
                "rope_theta": 10000000,
                "partial_rotary_factor": partial_rotary_factor,
            }
        )
        self.layer_types = deepcopy(layer_types)
        if not 0 < moe_block_coreset_top_p <= 1:
            raise ValueError("Clover coreset top_p must be in (0, 1]")

    @property
    def layers_block_type(self):
        if self.layer_types is None:
            return [
                (
                    "attention"
                    if (i + 1) % self.full_attention_interval == 0
                    else "linear_attention"
                )
                for i in range(self.num_hidden_layers)
            ]
        return [
            "attention" if kind == "full_attention" else kind
            for kind in self.layer_types
        ]

    @property
    def linear_layer_ids(self):
        return [
            i
            for i, kind in enumerate(self.layers_block_type)
            if kind == "linear_attention"
        ]

    @property
    def full_attention_layer_ids(self):
        return [
            i for i, kind in enumerate(self.layers_block_type) if kind == "attention"
        ]

    @property
    def mamba2_cache_params(self) -> Mamba2CacheParams:
        if is_cpu():
            adjust_tp_num_heads_if_necessary(self, get_parallel().attn_tp_size, False)
        key_dim = self.linear_key_head_dim * self.linear_num_key_heads
        value_dim = self.linear_value_head_dim * self.linear_num_value_heads
        shape = Mamba2StateShape.create(
            tp_world_size=linear_attn_tp_size(),
            intermediate_size=value_dim,
            n_groups=self.linear_num_key_heads,
            num_heads=self.linear_num_value_heads,
            head_dim=self.linear_value_head_dim,
            state_size=self.linear_key_head_dim,
            conv_kernel=self.linear_conv_kernel_dim,
            conv_shard_groups=[key_dim, key_dim, value_dim],
        )
        return Mamba2CacheParams(
            shape=shape, layers=self.linear_layer_ids, dtype=mamba2_state_dtype(self)
        )


register_linear_attn_model(
    LinearAttnModelSpec(
        config_class=CloverMoeConfig,
        backend_class_name="sglang.srt.layers.attention.linear.clover_backend.CloverGDNAttnBackend",
        arch_names=["CloverMoeModelLM"],
        support_mamba_cache_extra_buffer=True,
    )
)
