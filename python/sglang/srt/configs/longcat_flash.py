from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

FLASH_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class LongcatFlashConfig(PretrainedConfig):
    model_type = "longcat_flash"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=131072,
        hidden_size=6144,
        intermediate_size=None,
        ffn_hidden_size=12288,
        expert_ffn_hidden_size=2048,
        num_layers=28,
        num_hidden_layers=None,
        num_attention_heads=64,
        ep_size=1,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=128,
        qk_nope_head_dim=128,
        v_head_dim=128,
        n_routed_experts=512,
        moe_topk=12,
        norm_topk_prob=False,
        max_position_embeddings=131072,
        rms_norm_eps=1e-05,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mla_scale_q_lora=True,
        mla_scale_kv_lora=True,
        torch_dtype="bfloat16",
        params_dtype="bfloat16",
        rounter_params_dtype="float32",
        router_bias=False,
        topk_method=None,
        routed_scaling_factor=6.0,
        zero_expert_num=256,
        zero_expert_type="identity",
        nextn_use_scmoe=False,
        num_nextn_predict_layers=1,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            torch_dtype=torch_dtype,
            params_dtype=params_dtype,
            rounter_params_dtype=rounter_params_dtype,
            topk_method=topk_method,
            router_bias=router_bias,
            nextn_use_scmoe=nextn_use_scmoe,
            num_nextn_predict_layers=num_nextn_predict_layers,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = (
            num_hidden_layers if num_hidden_layers is not None else num_layers
        )
        self.intermediate_size = (
            intermediate_size if intermediate_size is not None else ffn_hidden_size
        )
        self.moe_intermediate_size = expert_ffn_hidden_size
        self.num_attention_heads = num_attention_heads
        self.ep_size = ep_size
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.n_routed_experts = n_routed_experts
        self.moe_topk = moe_topk
        self.norm_topk_prob = norm_topk_prob
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mla_scale_q_lora = mla_scale_q_lora
        self.mla_scale_kv_lora = mla_scale_kv_lora
        self.zero_expert_num = zero_expert_num
        self.zero_expert_type = zero_expert_type
        self.routed_scaling_factor = routed_scaling_factor
        self.hidden_act = "silu"
