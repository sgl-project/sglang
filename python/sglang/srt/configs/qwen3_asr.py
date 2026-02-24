"""

Copy from [configuration_qwen3_asr.py](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/core/transformers_backend/configuration_qwen3_asr.py)
and add some typing.

+ Qwen3ASRAudioEncoderConfig
+ Qwen3ASRConfig
+ Qwen3ASRTextConfig
+ Qwen3ASRThinkerConfig
+ Qwen3ASRConfig

"""

from transformers import (
    AutoTokenizer,
    PretrainedConfig,
    ProcessorMixin,
    WhisperFeatureExtractor,
)

from sglang.srt.multimodal.customized_mm_processor_utils import (
    register_customized_processor,
)


class Qwen3ASRHFProcessor(ProcessorMixin):
    """Minimal processor for Qwen3-ASR that combines WhisperFeatureExtractor with a tokenizer."""

    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        feature_extractor: WhisperFeatureExtractor,
        tokenizer: AutoTokenizer,
        **kwargs,
    ):
        super().__init__(
            feature_extractor=feature_extractor, tokenizer=tokenizer, **kwargs
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> "Qwen3ASRHFProcessor":  # type: ignore
        trust_remote_code = kwargs.pop("trust_remote_code", True)
        revision = kwargs.pop("revision", None)
        feature_extractor = WhisperFeatureExtractor.from_pretrained(
            pretrained_model_name_or_path, revision=revision, **kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            revision=revision,
            **kwargs,
        )
        return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)


class Qwen3ASRAudioEncoderConfig(PretrainedConfig):
    model_type = "qwen3_asr_audio_encoder"

    def __init__(
        self,
        num_mel_bins: int = 128,
        encoder_layers: int = 32,
        encoder_attention_heads: int = 20,
        encoder_ffn_dim: int = 5120,
        d_model: int = 1280,
        dropout: float = 0,
        attention_dropout: float = 0,
        activation_function: str = "gelu",
        activation_dropout: float = 0,
        scale_embedding: bool = False,
        initializer_range: float = 0.02,
        max_source_positions: int = 1500,
        n_window: int = 100,
        output_dim: int = 3584,
        n_window_infer: int = 400,
        conv_chunksize: int = 500,
        downsample_hidden_size: int = 480,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.num_hidden_layers = encoder_layers
        self.initializer_range = initializer_range
        self.scale_embedding = (
            scale_embedding  # scale factor will be sqrt(d_model) if True
        )
        self.max_source_positions = max_source_positions
        self.n_window = n_window
        self.output_dim = output_dim
        self.n_window_infer = n_window_infer
        self.conv_chunksize = conv_chunksize
        self.downsample_hidden_size = downsample_hidden_size


class Qwen3ASRTextConfig(PretrainedConfig):

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 4096,
        intermediate_size: int = 22016,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 32,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 128000,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 5000000.0,
        rope_scaling=None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class Qwen3ASRThinkerConfig(PretrainedConfig):
    model_type = "qwen3_asr_thinker"

    attribute_map = {}
    sub_configs = {
        "audio_config": Qwen3ASRAudioEncoderConfig,
        "text_config": Qwen3ASRTextConfig,
    }

    def __init__(
        self,
        audio_config=None,
        text_config=None,
        audio_token_id: int = 151676,
        audio_start_token_id: int = 151669,
        user_token_id: int = 872,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.user_token_id = user_token_id
        self.audio_start_token_id = audio_start_token_id
        self.initializer_range = initializer_range

        if isinstance(audio_config, dict):
            audio_config = Qwen3ASRAudioEncoderConfig(**audio_config)
        elif audio_config is None:
            audio_config = Qwen3ASRAudioEncoderConfig()
        self.audio_config = audio_config

        if isinstance(text_config, dict):
            text_config = Qwen3ASRTextConfig(**text_config)
        elif text_config is None:
            text_config = Qwen3ASRTextConfig()
        self.text_config = text_config
        self.audio_token_id = audio_token_id


@register_customized_processor(Qwen3ASRHFProcessor)  # type: ignore
class Qwen3ASRConfig(PretrainedConfig):
    model_type = "qwen3_asr"
    sub_configs = {
        "thinker_config": Qwen3ASRThinkerConfig,
    }

    def __init__(
        self,
        thinker_config=None,
        support_languages=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if thinker_config is None:
            thinker_config = {}

        self.thinker_config = Qwen3ASRThinkerConfig(**thinker_config)
        self.support_languages = support_languages

    def get_text_config(self, decoder: bool = False) -> PretrainedConfig:  # type: ignore
        return self.thinker_config.get_text_config()
