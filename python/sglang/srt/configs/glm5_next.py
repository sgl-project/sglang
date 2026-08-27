from typing import List, Optional, Union

from transformers.configuration_utils import PretrainedConfig
from transformers.models.glm_ocr.configuration_glm_ocr import GlmOcrVisionConfig

from sglang.srt.configs.mamba_utils import KimiLinearCacheParams, KimiLinearStateShape
from sglang.srt.runtime_context import get_parallel

_GLM5_NEXT_TOP_LEVEL_CONFIG_KEYS = (
    "architectures",
    "vocab_size",
    "hidden_size",
    "head_dim",
    "intermediate_size",
    "moe_intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "hidden_act",
    "max_position_embeddings",
    "rms_norm_eps",
    "use_cache",
    "pad_token_id",
    "bos_token_id",
    "eos_token_id",
    "rope_theta",
    "rope_scaling",
    "rope_parameters",
    "partial_rotary_factor",
    "tie_word_embeddings",
    "attention_bias",
    "attention_dropout",
    "n_routed_experts",
    "num_experts_per_tok",
    "n_shared_experts",
    "n_group",
    "topk_group",
    "norm_topk_prob",
    "routed_scaling_factor",
    "scoring_func",
    "topk_method",
    "first_k_dense_replace",
    "moe_layer_freq",
    "q_lora_rank",
    "kv_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
    "swiglu_limit",
    "mhc",
    "hc_mult",
    "hc_sinkhorn_iters",
    "hc_eps",
    "num_nextn_predict_layers",
    "linear_attn_config",
    "linear_head_dim",
    "linear_num_heads",
    "linear_conv_kernel_dim",
    "linear_lower_bound",
    "gate_lower_bound",
    "index_head_dim",
    "index_topk",
    "index_kpool",
    "index_kpool_always_select_tail",
    "index_kpool_compress",
    "index_n_heads",
    "index_topk_freq",
    "index_topk_pattern",
    "index_skip_topk_offset",
    "index_share_for_mtp_iteration",
    "indexer_rope_interleave",
    "layer_types",
    "mlp_layer_types",
    "quantization_config",
)


class Glm5NextTextConfig(PretrainedConfig):
    model_type = "glm5_next_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 154880,
        hidden_size: int = 4096,
        head_dim: Optional[int] = None,
        intermediate_size: int = 12288,
        moe_intermediate_size: int = 2048,
        num_hidden_layers: int = 45,
        num_attention_heads: int = 64,
        num_key_value_heads: Optional[int] = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 1013760,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        pad_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        rope_theta: float = 800000.0,
        rope_scaling: Optional[dict] = None,
        rope_parameters: Optional[dict] = None,
        partial_rotary_factor: float = 1.0,
        tie_word_embeddings: bool = False,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        n_routed_experts: Optional[int] = 288,
        num_experts_per_tok: int = 7,
        n_shared_experts: int = 1,
        n_group: int = 1,
        topk_group: int = 1,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 2.5,
        scoring_func: str = "sigmoid",
        topk_method: str = "noaux_tc",
        first_k_dense_replace: int = 3,
        moe_layer_freq: int = 1,
        q_lora_rank: Optional[int] = 1536,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 256,
        qk_rope_head_dim: int = 0,
        v_head_dim: int = 256,
        swiglu_limit: Optional[float] = None,
        mhc: bool = False,
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        hc_eps: float = 1e-6,
        num_nextn_predict_layers: int = 1,
        linear_attn_config: Optional[dict] = None,
        linear_head_dim: int = 128,
        linear_num_heads: int = 64,
        linear_conv_kernel_dim: int = 4,
        linear_lower_bound: Optional[float] = None,
        gate_lower_bound: Optional[float] = None,
        index_head_dim: int | None = None,
        index_topk: int | None = None,
        index_n_heads: int | None = None,
        index_topk_freq: int = 1,
        index_topk_pattern: Optional[str] = None,
        index_skip_topk_offset: Optional[int] = None,
        **kwargs,
    ):
        if rope_scaling is None and rope_parameters is not None:
            rope_scaling = rope_parameters
        if rope_parameters is not None:
            rope_theta = rope_parameters.get("rope_theta", rope_theta)
            partial_rotary_factor = rope_parameters.get(
                "partial_rotary_factor", partial_rotary_factor
            )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.partial_rotary_factor = partial_rotary_factor
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.swiglu_limit = swiglu_limit
        self.mhc = mhc
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps

        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.linear_head_dim = linear_head_dim
        self.linear_num_heads = linear_num_heads
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_lower_bound = linear_lower_bound
        self.gate_lower_bound = (
            gate_lower_bound if gate_lower_bound is not None else linear_lower_bound
        )
        if linear_attn_config is None:
            layer_types = kwargs.get("layer_types")
            if layer_types is None:
                kda_layers = [
                    layer_idx
                    for layer_idx in range(num_hidden_layers)
                    if layer_idx % 4 != 3
                ]
            else:
                kda_layers = [
                    layer_idx
                    for layer_idx, layer_type in enumerate(layer_types)
                    if layer_type == "linear_attention"
                ]
            kda_layer_set = set(kda_layers)
            linear_attn_config = {
                "full_attn_layers": [
                    layer_idx
                    for layer_idx in range(num_hidden_layers)
                    if layer_idx not in kda_layer_set
                ],
                "head_dim": linear_head_dim,
                "kda_layers": kda_layers,
                "num_heads": linear_num_heads,
                "short_conv_kernel_size": linear_conv_kernel_dim,
                "gate_lower_bound": self.gate_lower_bound,
            }
        self.linear_attn_config = linear_attn_config
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.index_n_heads = index_n_heads
        self.index_topk_freq = index_topk_freq
        self.index_topk_pattern = index_topk_pattern
        self.index_skip_topk_offset = index_skip_topk_offset

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        if rope_parameters is not None or rope_scaling is not None:
            self.rope_parameters = rope_parameters or rope_scaling

    def is_kda_layer(self, layer_idx: int):
        return (
            self.linear_attn_config is not None
            and layer_idx in self.linear_attn_config["kda_layers"]
        )

    @property
    def linear_layer_ids(self):
        return [i for i in range(self.num_hidden_layers) if self.is_kda_layer(i)]

    @property
    def nextn_layer_ids(self):
        num_nextn_layers = self.num_nextn_predict_layers or 0
        return [self.num_hidden_layers + i for i in range(num_nextn_layers)]

    @property
    def full_attention_layer_ids(self):
        return [i for i in range(self.num_hidden_layers) if not self.is_kda_layer(i)]

    @property
    def mamba2_cache_params(self) -> KimiLinearCacheParams:
        from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp

        head_shard_size = (
            get_parallel().attn_cp_size
            if is_dsa_enable_prefill_cp()
            else get_parallel().attn_tp_size
        )

        shape = KimiLinearStateShape.create(
            tp_world_size=head_shard_size,
            num_heads=self.linear_attn_config["num_heads"],
            head_dim=self.linear_attn_config["head_dim"],
            conv_kernel_size=self.linear_attn_config["short_conv_kernel_size"],
        )

        return KimiLinearCacheParams(shape=shape, layers=self.linear_layer_ids)


class Glm5NextVisionConfig(GlmOcrVisionConfig):
    def __init__(
        self,
        swiglu_limit: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.swiglu_limit = swiglu_limit


class Glm5NextConfig(PretrainedConfig):
    model_type = "glm5_next"
    sub_configs = {
        "vision_config": Glm5NextVisionConfig,
        "text_config": Glm5NextTextConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id: int = 59280,
        video_token_id: int = 59281,
        image_start_token_id: int = 59256,
        image_end_token_id: int = 59257,
        video_start_token_id: int = 59258,
        video_end_token_id: int = 59259,
        **kwargs,
    ):
        top_level_text_config = {
            key: kwargs[key]
            for key in _GLM5_NEXT_TOP_LEVEL_CONFIG_KEYS
            if key in kwargs
        }

        if isinstance(text_config, dict):
            text_config = {**top_level_text_config, **text_config}
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"](**top_level_text_config)
        else:
            self.text_config = text_config

        if vision_config is None:
            self.vision_config = None
        else:
            if isinstance(vision_config, dict):
                vision_config = dict(vision_config)
            else:
                vision_config = vision_config.to_dict()
            self.vision_config = self.sub_configs["vision_config"](**vision_config)

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.video_start_token_id = video_start_token_id
        self.video_end_token_id = video_end_token_id

        if getattr(self.text_config, "quantization_config", None) is not None:
            self.quantization_config = self.text_config.quantization_config

        super().__init__(**kwargs)
        for key in _GLM5_NEXT_TOP_LEVEL_CONFIG_KEYS:
            if hasattr(self.text_config, key):
                setattr(self, key, getattr(self.text_config, key))
