from typing import Any, TypedDict

from transformers import PretrainedConfig

from sglang.srt.configs.inkling import InklingMMConfig, InklingModelConfig

# config.json architectures of the target model; looped checkpoints use the second.
INKLING2_ARCHS = (
    "Inkling2ForConditionalGeneration",
    "Inkling2LoopForConditionalGeneration",
)


class Inkling2LoopConfig(TypedDict):
    core_start: int
    core_end: int
    num_passes: int
    gate_rank: int
    # True when the checkpoint was trained to run any number of passes.
    is_varloop: bool


class Inkling2ModelConfig(InklingModelConfig):
    model_type = "inkling2_model"

    def __init__(
        self,
        *,
        model_max_length: int = 1048576,
        pre_norm_pre_affine: bool = False,
        final_norm_pre_affine: bool = False,
        use_value_norm_no_gamma: bool = True,
        qk_l2_clip_threshold: float = 22.627416997969522,
        rel_l2_clip_threshold: float = 8.0,
        n_layers_per_attn_res_block: int = 6,
        attn_res_normalize_values: bool = True,
        mlp_gate_softcap: float = 4.0,
        mlp_up_softcap: float = 25.0,
        loop_config: Inkling2LoopConfig | None = None,
        # What the checkpoint writes between <|message_model|> and <|content_invoke_tool_json|>;
        # the default of --inkling2-tool-call-author-name-mode.
        tool_call_author_name_mode: str = "tool_name",
        **kwargs: Any,
    ) -> None:
        fixed = dict(
            use_embed_norm=True,
            use_sconv=True,
            dense_mlp_idx=0,
            use_global_scale=False,
            norm_after_topk=True,
            gate_activation="sigmoid",
            n_shared_experts=0,
            inference_moe_w13_interleaved=True,
            num_nextn_predict_layers=0,
            tie_word_embeddings=False,
        )
        for name, value in fixed.items():
            if name in kwargs and kwargs[name] != value:
                raise ValueError(f"Inkling2 requires {name}={value!r}")
            kwargs[name] = value
        defaults = dict(
            hidden_size=4608,
            num_hidden_layers=48,
            num_attention_heads=48,
            num_key_value_heads=8,
            head_dim=128,
            intermediate_size=2304,
            n_routed_experts=384,
            num_experts_per_tok=8,
            route_scale=8.0,
            use_gate_bias=True,
            log_scaling_n_floor=128000,
            log_scaling_alpha=0.1,
            final_logit_softcapping=30.0,
            # Logit precision comes from the checkpoint's unembed output dtype.
            enable_lm_head_output_fp32=False,
        )
        super().__init__(**(defaults | kwargs))
        self.model_max_length = model_max_length
        self.max_position_embeddings = model_max_length
        self.pre_norm_pre_affine = pre_norm_pre_affine
        self.final_norm_pre_affine = final_norm_pre_affine
        self.use_value_norm_no_gamma = use_value_norm_no_gamma
        self.qk_l2_clip_threshold = qk_l2_clip_threshold
        self.rel_l2_clip_threshold = rel_l2_clip_threshold
        self.n_layers_per_attn_res_block = n_layers_per_attn_res_block
        self.attn_res_normalize_values = attn_res_normalize_values
        self.mlp_gate_softcap = mlp_gate_softcap
        self.mlp_up_softcap = mlp_up_softcap
        self.loop_config = loop_config
        if tool_call_author_name_mode not in ("tool_name", "independent", "drop_name"):
            raise ValueError(
                "tool_call_author_name_mode must be 'tool_name', 'independent' or "
                f"'drop_name', got {tool_call_author_name_mode!r}"
            )
        self.tool_call_author_name_mode = tool_call_author_name_mode
        if mlp_gate_softcap < 0 or mlp_up_softcap < 0:
            raise ValueError("MLP softcaps must be nonnegative")
        if not 0 < n_layers_per_attn_res_block <= self.num_hidden_layers:
            raise ValueError(
                "Attention residual block size must be within the layer count"
            )
        if len(set(self.local_layer_ids)) != len(self.local_layer_ids) or any(
            i < 0 or i >= self.num_hidden_layers for i in self.local_layer_ids
        ):
            raise ValueError("local_layer_ids must contain unique valid layer indices")
        if not 0 < self.num_experts_per_tok <= self.n_routed_experts:
            raise ValueError(
                "num_experts_per_tok must be within the routed expert count"
            )
        if qk_l2_clip_threshold <= 0 or rel_l2_clip_threshold <= 0:
            raise ValueError("L2 clipping thresholds must be positive")
        if loop_config is not None:
            self._validate_loop_config(loop_config)
        self.loop_share_cache = False
        self._set_cache_layout()

    def configure_loop(self, *, num_passes: int | None, share_cache: bool) -> None:
        """Override the pass count or share KV across passes (variable-loop only)."""
        if self.loop_config is None or not self.loop_config["is_varloop"]:
            raise ValueError(
                "--inkling2-loop-num-passes and --inkling2-loop-share-cache need a "
                "checkpoint with loop_config.is_varloop=true"
            )
        if num_passes is not None:
            self._check_num_passes(num_passes)
            self.loop_config = {**self.loop_config, "num_passes": num_passes}
        self.loop_share_cache = share_cache
        self._set_cache_layout()

    def _set_cache_layout(self) -> None:
        # Each extra core pass gets its own KV cache layers, appended after the
        # physical layers, unless the passes share them (see cache_layer_id).
        cache_layers = list(range(self.num_hidden_layers))
        if self.loop_config is not None and not self.loop_share_cache:
            cache_layers += list(self.loop_core_layers) * (
                self.loop_config["num_passes"] - 1
            )
        self.num_attention_layers = len(cache_layers)
        self.is_hybrid_swa = True
        self.hybrid_layer_pattern = [
            int(i in self.local_layer_ids) for i in cache_layers
        ]

    def _validate_loop_config(self, loop_config: Inkling2LoopConfig) -> None:
        if not self.attn_res_normalize_values:
            raise ValueError("Loop K/V messages require attn_res_normalize_values=True")
        int_fields = {"core_start", "core_end", "num_passes", "gate_rank"}
        if (
            set(loop_config) != int_fields | {"is_varloop"}
            or any(type(loop_config[name]) is not int for name in int_fields)
            or type(loop_config["is_varloop"]) is not bool
        ):
            raise ValueError(
                f"loop_config requires integer fields {sorted(int_fields)} and a "
                f"boolean is_varloop, got {loop_config}"
            )
        start, end = loop_config["core_start"], loop_config["core_end"]
        if not 0 <= start < end <= self.num_hidden_layers:
            raise ValueError(
                f"Loop core [{start}, {end}) must lie within "
                f"{self.num_hidden_layers} physical layers"
            )
        block = self.n_layers_per_attn_res_block
        if start % block or end % block:
            raise ValueError(
                f"Loop core [{start}, {end}) must align with attention residual "
                f"blocks of {block} layers"
            )
        if loop_config["gate_rank"] <= 0:
            raise ValueError(f"Looping requires gate_rank > 0, got {loop_config}")
        self._check_num_passes(loop_config["num_passes"])

    def _check_num_passes(self, num_passes: int) -> None:
        min_passes = 1 if self.loop_config["is_varloop"] else 2
        if type(num_passes) is not int or num_passes < min_passes:
            raise ValueError(
                f"This loop checkpoint needs at least {min_passes} passes, "
                f"got {num_passes!r}"
            )

    @property
    def loop_core_layers(self) -> range:
        loop = self.loop_config
        return range(0) if loop is None else range(loop["core_start"], loop["core_end"])

    @property
    def num_layer_visits(self) -> int:
        if self.loop_config is None:
            return self.num_hidden_layers
        extra_passes = self.loop_config["num_passes"] - 1
        return self.num_hidden_layers + extra_passes * len(self.loop_core_layers)

    def cache_layer_id(self, layer_id: int, pass_idx: int = 0) -> int:
        """KV cache layer of a layer visit; shared caches reuse the physical layer."""
        if self.loop_share_cache:
            return layer_id
        return self.conv_layer_id(layer_id, pass_idx)

    def conv_layer_id(self, layer_id: int, pass_idx: int = 0) -> int:
        """Conv-state layer of a layer visit; every visit keeps its own."""
        if pass_idx == 0:
            return layer_id
        core = self.loop_core_layers
        assert layer_id in core and 0 < pass_idx < self.loop_config["num_passes"]
        return (
            self.num_hidden_layers + (pass_idx - 1) * len(core) + layer_id - core.start
        )

    @property
    def conv_layer_ids(self) -> list[int]:
        return list(range(self.num_layer_visits))

    @property
    def full_attention_layer_ids(self) -> list[int]:
        return list(range(self.num_attention_layers))


class Inkling2MMConfig(InklingMMConfig):
    model_type = "inkling2"
    sub_configs = {"text_config": Inkling2ModelConfig}

    def __init__(
        self,
        *,
        text_config: dict[str, Any] | Inkling2ModelConfig | None = None,
        vision_config: dict[str, Any] | None = None,
        mtp_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.text_config = (
            text_config
            if isinstance(text_config, Inkling2ModelConfig)
            else Inkling2ModelConfig(**(text_config or {}))
        )
        # transformers builds a default instance (no text_config) to diff against.
        if text_config is not None and (self.text_config.loop_config is not None) != (
            self.model_type == "inkling2_loop"
        ):
            raise ValueError(
                "Looped Inkling2 checkpoints use model_type inkling2_loop with a "
                f"loop_config; got model_type {self.model_type!r}"
            )
        self.vision_config = vision_config
        self.mtp_config = mtp_config
        # The Hub config.json carries a quantization_config stub (modelopt for NVFP4,
        # fp8 for MXFP8) only so the Hub counts parameters; the precision lives in
        # hf_quant_config.json.
        stub = kwargs.get("quantization_config")
        if isinstance(stub, dict) and stub.get("quant_method") in ("modelopt", "fp8"):
            del kwargs["quantization_config"]
        PretrainedConfig.__init__(self, **kwargs)


class Inkling2LoopMMConfig(Inkling2MMConfig):
    model_type = "inkling2_loop"

    # transformers generates __init__ for config subclasses that define none,
    # which would skip building text_config.
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def configure_loop(self, *, num_passes: int | None, share_cache: bool) -> None:
        self.text_config.configure_loop(num_passes=num_passes, share_cache=share_cache)


def inkling2_config_from_dict(config: dict[str, Any]) -> Inkling2MMConfig:
    cls = (
        Inkling2LoopMMConfig
        if config.get("model_type") == "inkling2_loop"
        else Inkling2MMConfig
    )
    return cls(**config)


class Inkling2MTPConfig(PretrainedConfig):
    model_type = "inkling2_mtp"

    @classmethod
    def from_target(cls, config: Inkling2MMConfig) -> "Inkling2MTPConfig":
        if config.mtp_config is None:
            raise ValueError("Inkling2 speculative decoding requires mtp_config")
        text = config.text_config
        draft = dict(config.mtp_config)
        required = {
            "max_position_embeddings",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "q_bias",
            "o_bias",
            "intermediate_size",
            "rms_norm_eps",
            "rope_theta",
            "local_layer_ids",
            "sliding_window_size",
            "block_size",
            "markov_rank",
            "markov_head_type",
            "target_layer_ids",
        }
        missing = required - draft.keys()
        if missing:
            raise ValueError(f"Inkling2 mtp_config is missing {sorted(missing)}")
        if any(
            draft[key] != "bfloat16" for key in ("dtype", "torch_dtype") if key in draft
        ):
            raise ValueError("Inkling2 MTP requires bfloat16 precision")
        draft.setdefault("torch_dtype", "bfloat16")
        if draft["hidden_size"] != text.hidden_size:
            raise ValueError("Inkling2 MTP and target hidden sizes must match")
        if draft["markov_head_type"] != "rnn_hidden":
            raise ValueError("Inkling2 MTP requires markov_head_type='rnn_hidden'")
        if draft["markov_rank"] <= 0 or draft["block_size"] < 2:
            raise ValueError("Inkling2 MTP needs positive rank and block_size >= 2")
        num_layers = draft["num_hidden_layers"]
        local = draft["local_layer_ids"]
        if (
            num_layers <= 0
            or len(set(local)) != len(local)
            or any(i < 0 or i >= num_layers for i in local)
        ):
            raise ValueError("MTP local_layer_ids must be unique valid layer indices")
        features = draft["target_layer_ids"]
        num_visits = text.num_layer_visits
        if (
            len(features) < 2
            or features != sorted(set(features))
            or features[-1] != num_visits
            or any(i < 0 or i >= num_visits for i in features[:-1])
        ):
            raise ValueError(
                "MTP target_layer_ids must contain sorted unique target layer visits "
                f"followed by the final-norm sentinel {num_visits}, got {features}"
            )
        if draft["num_attention_heads"] % draft["num_key_value_heads"]:
            raise ValueError("MTP query heads must be divisible by KV heads")
        draft.update(
            architectures=["Inkling2ForConditionalGenerationMTP"],
            vocab_size=text.vocab_size,
            unpadded_vocab_size=text.vocab_size,
            padded_vocab_size=text.padded_vocab_size,
            target_num_hidden_layers=text.num_hidden_layers,
            final_logit_softcapping=text.final_logit_softcapping,
            hidden_act="silu",
            tie_word_embeddings=False,
            is_causal=False,
            sample_from_anchor=False,
            is_hybrid_swa=bool(local),
            hybrid_layer_pattern=[int(i in local) for i in range(num_layers)],
            layer_types=[
                "sliding_attention" if i in local else "full_attention"
                for i in range(num_layers)
            ],
            sliding_window=draft["sliding_window_size"],
            dspark_block_size=draft["block_size"] - 1,
            dspark_target_layer_ids=list(features),
            mask_token_id=0,
            dflash_config=dict(
                block_size=draft["block_size"],
                num_target_layers=num_visits,
                target_layer_ids=list(features[:-1]),
            ),
        )
        return cls(**draft)
