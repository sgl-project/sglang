import math

from transformers import PretrainedConfig


def _attn_tp_world_size(default: int = 1) -> int:
    from sglang.srt.runtime_context import get_parallel

    try:
        return get_parallel().attn_tp_size
    except AssertionError:
        return default


class Qwen4ExpTextConfig(PretrainedConfig):
    model_type = "qwen4_exp_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=248320,
        hidden_size=2048,
        num_hidden_layers=40,
        num_attention_heads=16,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_parameters=None,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=256,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        num_experts_per_tok=10,
        num_experts=512,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        layer_types=None,
        full_attention_interval=4,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        partial_rotary_factor=0.25,
        hc_count=4,
        hc_lowrank=320,
        ple_layer_ids=None,
        ple_embed_dim=None,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=20_000_000,
        make_ngram_vocab_size_divisible_by=128,
        seed=1234,
        split_ngram_parts=512,
        indexer_n_heads=None,
        indexer_kv_heads=None,
        indexer_head_dim=None,
        indexer_budget=None,
        indexer_compress_ratio=None,
        norm_topk_prob=True,
        output_gate_type=None,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.full_attention_interval = full_attention_interval
        self.partial_rotary_factor = partial_rotary_factor
        self.hc_count = hc_count
        self.hc_lowrank = hc_lowrank
        self.ple_layer_ids = [] if ple_layer_ids is None else sorted(set(ple_layer_ids))
        self.ple_embed_dim = hidden_size if ple_embed_dim is None else ple_embed_dim
        self.ple_conv_kernel_size = ple_conv_kernel_size
        self.ngram_size = ngram_size
        self.heads_per_ngram = heads_per_ngram
        self.ngram_vocab_size_base = ngram_vocab_size_base
        self.make_ngram_vocab_size_divisible_by = make_ngram_vocab_size_divisible_by
        self.seed = seed
        self.split_ngram_parts = split_ngram_parts
        self.indexer_n_heads = indexer_n_heads
        self.indexer_kv_heads = indexer_kv_heads
        self.indexer_head_dim = indexer_head_dim
        self.indexer_budget = indexer_budget
        self.indexer_compress_ratio = indexer_compress_ratio
        self.norm_topk_prob = norm_topk_prob
        self.output_gate_type = output_gate_type
        self.number_of_conv_states = 3 if self.ple_layer_ids else 1

        self.rope_parameters = rope_parameters or {
            "rope_type": "default",
            "rope_theta": rope_theta,
            "partial_rotary_factor": partial_rotary_factor,
        }
        if "rope_theta" not in self.rope_parameters:
            self.rope_parameters["rope_theta"] = rope_theta
        if "partial_rotary_factor" not in self.rope_parameters:
            self.rope_parameters["partial_rotary_factor"] = partial_rotary_factor
        self.rope_scaling = self.rope_parameters

        if layer_types is None:
            layer_types = [
                (
                    "linear_attention"
                    if (layer_idx + 1) % full_attention_interval
                    else "qwen_sparse_attention"
                )
                for layer_idx in range(num_hidden_layers)
            ]
        self.layer_types = [
            "qwen_sparse_attention" if layer_type == "full_attention" else layer_type
            for layer_type in layer_types
        ]
        self._validate_qwen4_exp()

    @property
    def layers_block_type(self):
        return [
            "attention" if layer_type == "qwen_sparse_attention" else layer_type
            for layer_type in self.layer_types
        ]

    @property
    def linear_layer_ids(self):
        return [
            idx
            for idx, layer_type in enumerate(self.layer_types)
            if layer_type == "linear_attention"
        ]

    @property
    def full_attention_layer_ids(self):
        return [
            idx
            for idx, layer_type in enumerate(self.layer_types)
            if layer_type == "qwen_sparse_attention"
        ]

    @property
    def mamba2_cache_params(self):
        from sglang.srt.configs.mamba_utils import (
            Mamba2CacheParams,
            Mamba2StateShape,
            mamba2_state_dtype,
        )

        key_dim = self.linear_key_head_dim * self.linear_num_key_heads
        value_dim = self.linear_value_head_dim * self.linear_num_value_heads
        shape = Mamba2StateShape.create(
            tp_world_size=_attn_tp_world_size(),
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

    def _validate_qwen4_exp(self):
        unsupported = sorted(
            set(self.layer_types) - {"linear_attention", "qwen_sparse_attention"}
        )
        if unsupported:
            raise ValueError(f"Unsupported Qwen4-Exp layer types: {unsupported}.")
        output_gate_type = self.output_gate_type or self.hidden_act
        if output_gate_type not in {"sigmoid", "silu"}:
            raise ValueError(
                f"Unsupported Qwen4-Exp output gate activation: {output_gate_type}."
            )
        if self.hc_count <= 1:
            raise ValueError(f"Qwen4-Exp requires hc_count > 1, got {self.hc_count}.")
        if self.num_experts <= 0:
            raise ValueError(f"num_experts must be > 0, got {self.num_experts}.")
        if not 0 < self.num_experts_per_tok <= self.num_experts:
            raise ValueError(
                "num_experts_per_tok must be in [1, num_experts], "
                f"got {self.num_experts_per_tok} and {self.num_experts}."
            )
        if self.linear_num_value_heads % self.linear_num_key_heads != 0:
            raise ValueError(
                "linear_num_value_heads must be divisible by linear_num_key_heads, "
                f"got {self.linear_num_value_heads} and {self.linear_num_key_heads}."
            )
        qsa_fields = (
            "indexer_n_heads",
            "indexer_kv_heads",
            "indexer_head_dim",
            "indexer_budget",
            "indexer_compress_ratio",
        )
        qsa_values = {name: getattr(self, name) for name in qsa_fields}
        if any(value is not None for value in qsa_values.values()):
            missing = [name for name, value in qsa_values.items() if value is None]
            if missing:
                raise ValueError(f"QSA config is missing required fields: {missing}.")
            if any(value <= 0 for value in qsa_values.values()):
                raise ValueError(f"QSA config values must be positive: {qsa_values}.")
            if self.indexer_kv_heads != 1:
                raise ValueError("Qwen4-Exp QSA requires indexer_kv_heads=1.")
            if self.indexer_budget % self.indexer_compress_ratio != 0:
                raise ValueError(
                    "indexer_budget must be divisible by indexer_compress_ratio."
                )
            rotary_dim = int(
                self.head_dim * self.rope_parameters.get("partial_rotary_factor", 1.0)
            )
            if rotary_dim > self.indexer_head_dim:
                raise ValueError(
                    "Qwen4-Exp attention RoPE dimensions must fit the QSA index "
                    f"head: rotary_dim={rotary_dim}, "
                    f"indexer_head_dim={self.indexer_head_dim}."
                )

        if self.ple_layer_ids:
            ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
            if (
                ngram_heads <= 0
                or self.ple_embed_dim <= 0
                or self.ple_embed_dim % ngram_heads != 0
            ):
                raise ValueError(
                    "ple_embed_dim must be divisible by the number of n-gram heads: "
                    f"{self.ple_embed_dim} % {ngram_heads} != 0."
                )
            invalid = [
                layer_id
                for layer_id in self.ple_layer_ids
                if layer_id < 1 or layer_id > self.num_hidden_layers
            ]
            if invalid:
                raise ValueError(
                    "ple_layer_ids must contain one-indexed ids in "
                    f"[1, {self.num_hidden_layers}], got {invalid}."
                )
            non_linear = [
                layer_id
                for layer_id in self.ple_layer_ids
                if self.layer_types[layer_id - 1] != "linear_attention"
            ]
            if non_linear:
                raise ValueError(
                    "Qwen4-Exp PLE is only supported on linear_attention layers, "
                    f"got PLE on layers {non_linear}."
                )
            if self.eos_token_id is None:
                raise ValueError("eos_token_id must be set when PLE is enabled.")


class Qwen4ExpVisionConfig(PretrainedConfig):
    model_type = "qwen4_exp_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=27,
        hidden_size=1152,
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=4304,
        num_heads=16,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=3584,
        num_position_embeddings=2304,
        initializer_range=0.02,
        **kwargs,
    ):
        if kwargs.get("model_type") == "qwen4_exp":
            kwargs["model_type"] = self.model_type
        super().__init__(**kwargs)
        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.initializer_range = initializer_range


class Qwen4ExpConfig(PretrainedConfig):
    model_type = "qwen4_exp"
    sub_configs = {
        "vision_config": Qwen4ExpVisionConfig,
        "text_config": Qwen4ExpTextConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=248056,
        video_token_id=248057,
        vision_start_token_id=248053,
        vision_end_token_id=248054,
        tie_word_embeddings=False,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            if vision_config.get("model_type") == "qwen4_exp":
                vision_config = dict(vision_config)
                vision_config["model_type"] = "qwen4_exp_vision"
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()
        else:
            self.text_config = text_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    def get_text_config(self, decoder=False):
        del decoder
        return self.text_config


def qwen4_exp_padded_ngram_vocab_size(config: Qwen4ExpTextConfig) -> int:
    total_vocab_size = 0
    ngram_heads = (config.ngram_size - 1) * config.heads_per_ngram
    for head_idx in range(ngram_heads):
        total_vocab_size += _find_nth_prime_after(
            config.ngram_vocab_size_base - 1, head_idx + 1
        )
    divisor = config.make_ngram_vocab_size_divisible_by
    return math.ceil(total_vocab_size / divisor) * divisor


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _find_nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


__all__ = ["Qwen4ExpConfig", "Qwen4ExpTextConfig", "Qwen4ExpVisionConfig"]
