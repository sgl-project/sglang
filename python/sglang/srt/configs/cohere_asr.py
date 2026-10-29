from typing import Any, Dict, List, Optional

from transformers import PretrainedConfig

# Encoder frames per mel frame is 1/8: three stride-2 depthwise-striding convs.
_SUBSAMPLING_CONV_STRIDES = (2, 2, 2)
_SUBSAMPLING_CONV_KERNEL = 3
_SUBSAMPLING_CONV_PADDING = 1


def cohere_asr_encoder_seq_len(mel_frames: int) -> int:
    """Conformer output frame count for a mel spectrogram of ``mel_frames``.

    Integer mirror of the strided convolutions in ``pre_encode``. The decoder
    prompt is prefixed with exactly this many placeholder tokens so the
    scheduler reserves matching cross-attention KV slots, so this must stay in
    lockstep with ``CohereAsrConvSubsampling``.
    """
    length = mel_frames
    for stride in _SUBSAMPLING_CONV_STRIDES:
        length = (
            length + 2 * _SUBSAMPLING_CONV_PADDING - _SUBSAMPLING_CONV_KERNEL
        ) // stride + 1
    return max(length, 1)


class CohereAsrConfig(PretrainedConfig):
    """Config for CohereLabs cohere-transcribe (``cohere_asr``).

    The checkpoint keeps every shape inside nested ``encoder`` /
    ``transf_decoder`` / ``head`` dicts and exposes no flat ``hidden_size``,
    which the generic shape derivation in `ModelConfig` requires. Flatten the
    decoder shapes onto the top level and keep the nested dicts for the model
    to build from.
    """

    model_type = "cohere_asr"

    def __init__(
        self,
        vocab_size: int = 16384,
        encoder: Optional[Dict[str, Any]] = None,
        transf_decoder: Optional[Dict[str, Any]] = None,
        head: Optional[Dict[str, Any]] = None,
        preprocessor: Optional[Dict[str, Any]] = None,
        max_audio_clip_s: float = 35.0,
        sample_rate: int = 16000,
        supported_languages: Optional[List[str]] = None,
        **kwargs,
    ):
        self.encoder = encoder or {}
        self.transf_decoder = transf_decoder or {}
        self.head = head or {}
        self.preprocessor = preprocessor or {}
        self.max_audio_clip_s = max_audio_clip_s
        self.sample_rate = sample_rate
        self.supported_languages = list(supported_languages or [])

        # PretrainedConfig.to_diff_dict() instantiates the class with no
        # arguments, so every flat shape needs a default to fall back on.
        decoder = self.transf_decoder.get("config_dict", {})
        self.hidden_size = decoder.get("hidden_size", 1024)
        self.intermediate_size = decoder.get("inner_size", 4096)
        self.num_attention_heads = decoder.get("num_attention_heads", 8)
        self.num_key_value_heads = self.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.hidden_act = decoder.get("hidden_act", "relu")
        self.decoder_layers = decoder.get("num_layers", 8)
        self.num_hidden_layers = self.decoder_layers
        self.max_target_positions = decoder.get("max_sequence_length", 1024)
        self.encoder_layers = self.encoder.get("n_layers", 48)
        self.vocab_size = self.head.get("num_classes", vocab_size)

        # Encoder placeholders share each KV row with the decoder tokens, so the
        # served context spans both: 2048 covers the 438 encoder frames of a
        # max-length 35s clip plus the full 1024-token decoder window. The
        # checkpoint's own `max_seq_len` (1024) counts decoder positions only and
        # is first in CONTEXT_LENGTH_KEYS' search order, so drop it here rather
        # than let it shadow the combined length.
        kwargs.pop("max_seq_len", None)
        kwargs.setdefault("max_sequence_length", 2048)
        kwargs.setdefault("is_encoder_decoder", True)
        # The checkpoint declares no dtype; "auto" would resolve float32 and
        # then downcast to float16, but these are bf16-trained weights.
        kwargs.setdefault("dtype", "bfloat16")
        super().__init__(**kwargs)
