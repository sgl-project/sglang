from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

FLASH_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class LongcatFlashProConfig(PretrainedConfig):
    model_type = "longcat_flash_pro"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=163840,
        hidden_size=8192,
        intermediate_size=None,
        ffn_hidden_size=12288,
        expert_ffn_hidden_size=2048,
        num_layers=38,
        num_hidden_layers=None,
        num_attention_heads=64,
        ep_size=1,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        n_routed_experts=768,
        moe_topk=12,
        norm_topk_prob=False,
        max_position_embeddings=262144,
        rms_norm_eps=1e-05,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
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
        routed_scaling_factor=9.0,
        zero_expert_num=128,
        zero_expert_type="identity",
        nextn_use_scmoe=False,
        num_nextn_predict_layers=1,
        ngram_vocab_size_ratio=None,
        emb_neighbor_num=None,
        emb_split_num=None,
        attention_method="MLA",
        use_longcat_dsa=True,
        index_head_dim=128,
        index_init_tokens=16,
        index_local_tokens=1024,
        index_n_heads=32,
        index_topk=2048,
        index_k_norm_type="rms",
        cli_factor=2,
        index_topk_pattern=None,
        moe_impl="mix",
        moe_switch_token_num=1024,
        dsa_mtp_cli=True,
        oe_neighbor_num=5,
        oe_split_num=4,
        oe_vocab_size_ratio=100.567,
        use_oe_embedding=None,
        mtp_num_layers=3,
        mtp_replicate_modules=True,
        mtp_disable_over_tokenizer=True,
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
        self.attention_method = attention_method
        self.mla_scale_q_lora = mla_scale_q_lora
        self.mla_scale_kv_lora = mla_scale_kv_lora
        self.zero_expert_num = zero_expert_num
        self.zero_expert_type = zero_expert_type
        self.routed_scaling_factor = routed_scaling_factor
        self.use_longcat_dsa = use_longcat_dsa
        self.index_head_dim = index_head_dim
        self.index_init_tokens = index_init_tokens
        self.index_local_tokens = index_local_tokens
        self.index_n_heads = index_n_heads
        self.index_topk = index_topk
        self.index_k_norm_type = index_k_norm_type
        self.cli_factor = cli_factor
        self.index_topk_pattern = index_topk_pattern
        self.moe_impl = moe_impl
        self.moe_switch_token_num = moe_switch_token_num
        self.dsa_mtp_cli = dsa_mtp_cli
        self.oe_neighbor_num = oe_neighbor_num
        self.oe_split_num = oe_split_num
        self.oe_vocab_size_ratio = oe_vocab_size_ratio
        self.use_oe_embedding = (
            use_oe_embedding
            if use_oe_embedding is not None
            else (
                oe_vocab_size_ratio is not None
                and oe_neighbor_num is not None
                and oe_split_num is not None
            )
        )
        self.oe_branch_num = (
            (oe_neighbor_num - 1) * oe_split_num if self.use_oe_embedding else 0
        )
        self.oe_hidden_dim = (
            hidden_size // self.oe_branch_num if self.oe_branch_num > 0 else None
        )
        self.oe_emb_dim = self.oe_hidden_dim
        self.oe_vocab_base = (
            int(oe_vocab_size_ratio * vocab_size)
            if oe_vocab_size_ratio is not None
            else None
        )
        self.oe_scale = 1 + self.oe_branch_num if self.use_oe_embedding else 1
        self.mtp_num_layers = mtp_num_layers
        self.mtp_replicate_modules = mtp_replicate_modules
        self.mtp_disable_over_tokenizer = mtp_disable_over_tokenizer
        self.hidden_act = "silu"
        self.use_ngram_embedding = (
            ngram_vocab_size_ratio is not None or self.use_oe_embedding
        )
        if self.use_ngram_embedding:
            if ngram_vocab_size_ratio is not None:
                self.ngram_embedding_m = int(ngram_vocab_size_ratio * vocab_size)
                self.ngram_embedding_n = emb_neighbor_num
                self.ngram_embedding_k = emb_split_num
            else:
                self.ngram_embedding_m = self.oe_vocab_base
                self.ngram_embedding_n = oe_neighbor_num
                self.ngram_embedding_k = oe_split_num
        if self.index_topk_pattern is None and self.use_longcat_dsa:
            self.index_topk_pattern = [
                "C" if i % 2 == 0 else "S"
                for i in range(self.num_hidden_layers * 2)
            ]
