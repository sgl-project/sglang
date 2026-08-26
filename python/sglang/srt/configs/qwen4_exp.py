from transformers import PretrainedConfig

from sglang.srt.configs.qwen3_next import Qwen3NextConfig
from sglang.srt.configs.qwen3_vl import Qwen3VLVisionConfig


class Qwen4ExpVisionConfig(Qwen3VLVisionConfig):
    model_type = "qwen4_exp"
    base_config_key = "vision_config"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Qwen4ExpTextConfig(Qwen3NextConfig):
    model_type = "qwen4_exp_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        hc_count=4,
        hc_lowrank=320,
        ple_layer_ids=None,
        ple_embed_dim=None,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=20000000,
        make_ngram_vocab_size_divisible_by=128,
        ple_offload_embedding=False,
        ple_embedding_dtype=None,
        index_share_for_mtp_iteration=True,
        rope_parameters=None,
        layer_types=None,
        **kwargs,
    ):
        if hc_count <= 1:
            raise ValueError(f"Qwen4-Exp requires hc_count > 1, got {hc_count}.")
        # Qwen3.5/Qwen4-Exp checkpoints may provide RoPE settings under
        # rope_parameters. Normalize it before parent init so Qwen3Next shared
        # config logic sees the expected rope_scaling and rope_theta fields.
        if rope_parameters is not None:
            if kwargs.get("rope_scaling") is None:
                kwargs["rope_scaling"] = rope_parameters
            if kwargs.get("rope_theta") is None and "rope_theta" in rope_parameters:
                kwargs["rope_theta"] = rope_parameters["rope_theta"]
            if (
                kwargs.get("partial_rotary_factor") is None
                and "partial_rotary_factor" in rope_parameters
            ):
                kwargs["partial_rotary_factor"] = rope_parameters[
                    "partial_rotary_factor"
                ]
        super().__init__(**kwargs)
        if self.rope_scaling is None:
            self.rope_scaling = rope_parameters or {}
        self.rope_parameters = rope_parameters or self.rope_scaling
        self.hc_count = hc_count
        # ModelConfig sizes the speculative hidden width off `hc_mult`
        # (the DeepSeek-V4 mHC field); Qwen4-Exp spells it `hc_count`.
        self.hc_mult = hc_count
        self.hc_lowrank = hc_lowrank
        self.layer_types = layer_types
        self.ple_layer_ids = ple_layer_ids or []
        self.ple_embed_dim = ple_embed_dim or self.hidden_size
        self.ple_conv_kernel_size = ple_conv_kernel_size
        self.ngram_size = ngram_size
        self.heads_per_ngram = heads_per_ngram
        self.ngram_vocab_size_base = ngram_vocab_size_base
        self.make_ngram_vocab_size_divisible_by = make_ngram_vocab_size_divisible_by
        self.ple_offload_embedding = ple_offload_embedding
        # "float8_e4m3fn" keeps fp8 PLE tables fp8-resident; text_config-scoped.
        self.ple_embedding_dtype = ple_embedding_dtype
        # MTP draft decode steps reuse the draft-extend indexer selection
        # (GLM-5.2 IndexShare); default on for Qwen4-Exp, checkpoint config
        # or --json-model-override-args can disable it.
        self.index_share_for_mtp_iteration = index_share_for_mtp_iteration

    @property
    def layers_block_type(self):
        if self.layer_types is not None:
            return [
                "attention" if layer_type == "full_attention" else layer_type
                for layer_type in self.layer_types
            ]
        return super().layers_block_type

    @property
    def short_conv_layer_ids(self):
        if not self.ple_layer_ids:
            return []
        return sorted({int(layer_id) - 1 for layer_id in self.ple_layer_ids})

    @property
    def short_conv_state_shape(self):
        if not self.short_conv_layer_ids:
            return None
        ple_state_len = (self.ple_conv_kernel_size - 1) * self.ngram_size
        ple_channels = self.hidden_size * self.hc_count
        return ple_channels, ple_state_len

    @property
    def ngram_context_len(self):
        if not self.ple_layer_ids:
            return 0
        return max(int(self.ngram_size) - 1, 0)


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
        rope_parameters=None,
        **kwargs,
    ):
        # The nested text config is authoritative; old exports also copied this
        # value to the top level.
        if text_config is not None:
            kwargs.pop("split_ngram_parts", None)

        # Backward compatibility: older Qwen4-Exp checkpoints were text-only
        # and stored text attributes at the top level.
        text_kwargs = (
            dict(kwargs)
            if text_config is None
            and "hidden_size" in kwargs
            and "num_hidden_layers" in kwargs
            else {}
        )
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"](**text_kwargs)
        else:
            self.text_config = text_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.rope_parameters = rope_parameters or getattr(
            self.text_config, "rope_parameters", {}
        )
        super().__init__(**kwargs, tie_word_embeddings=tie_word_embeddings)
