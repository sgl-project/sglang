from __future__ import annotations

from typing import Optional

from transformers.configuration_utils import PretrainedConfig

from sglang.srt.configs.gigachat35 import GigaChat35Config
from sglang.srt.configs.internvl import InternVisionConfig


class GigaChat35OmniAudioConfig(PretrainedConfig):
    """Conformer audio encoder of GigaChat3.5-Omni.

    Mel frames go through log2(subsampling_factor) stride-2 convolutions, then
    conformer blocks whose attention never crosses an `attention_chunk_size`
    block; RoPE positions restart in every block.

    Args:
        hidden_size: Width of the conformer blocks.
        num_hidden_layers: Number of conformer blocks.
        num_attention_heads: Attention heads per block.
        intermediate_size: Width of the feed-forward modules.
        conv_kernel_size: Kernel of the depthwise convolution.
        num_mel_bins: Mel bins per input frame.
        subsampling_factor: Time reduction of the stride-2 convolutions.
        subsampling_conv_channels: Channels of the stride-2 convolutions.
        attention_chunk_size: Frames per attention block.
        rope_theta: RoPE base.
        layer_norm_eps: LayerNorm epsilon.
        **kwargs: Passed to `PretrainedConfig`.
    """

    model_type = "gigachat3_5_omni_audio"

    def __init__(
        self,
        hidden_size: int = 1024,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        intermediate_size: int = 4096,
        conv_kernel_size: int = 5,
        num_mel_bins: int = 64,
        subsampling_factor: int = 4,
        subsampling_conv_channels: int = 1024,
        attention_chunk_size: int = 200,
        rope_theta: float = 5000.0,
        layer_norm_eps: float = 1e-5,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.conv_kernel_size = conv_kernel_size
        self.num_mel_bins = num_mel_bins
        self.subsampling_factor = subsampling_factor
        self.subsampling_conv_channels = subsampling_conv_channels
        self.attention_chunk_size = attention_chunk_size
        self.rope_theta = rope_theta
        self.layer_norm_eps = layer_norm_eps
        super().__init__(**kwargs)


class GigaChat35OmniConfig(PretrainedConfig):
    """GigaChat3.5-Omni: GigaChat3.5 LM with InternViT vision and a conformer audio encoder.

    A video keeps all tokens of its first frame and `video_tokens_per_frame`
    tokens per later frame on average.

    Args:
        text_config: GigaChat3.5 LM config.
        vision_config: InternViT config.
        audio_config: Conformer config.
        downsample_ratio: Pixel-shuffle ratio of the InternViT patch grid.
        audio_stack_factor: Conformer frames stacked into one audio token.
        image_token_id: Placeholder of image tokens.
        video_token_id: Placeholder of video frame tokens.
        audio_token_id: Placeholder of audio file tokens.
        video_audio_token_id: Placeholder of video soundtrack tokens.
        video_tokens_per_frame: Average tokens kept per later video frame; the
            checkpoint was trained with this budget.
        **kwargs: Passed to `PretrainedConfig`.
    """

    model_type = "gigachat3_5_omni"
    sub_configs = {
        "text_config": GigaChat35Config,
        "vision_config": InternVisionConfig,
        "audio_config": GigaChat35OmniAudioConfig,
    }

    def __init__(
        self,
        text_config: Optional[dict | GigaChat35Config] = None,
        vision_config: Optional[dict | InternVisionConfig] = None,
        audio_config: Optional[dict | GigaChat35OmniAudioConfig] = None,
        downsample_ratio: float = 0.5,
        audio_stack_factor: int = 4,
        image_token_id: int = 128004,
        video_token_id: int = 128005,
        audio_token_id: int = 128006,
        video_audio_token_id: int = 128007,
        video_tokens_per_frame: int = 77,
        **kwargs,
    ):
        if isinstance(text_config, dict):
            text_config = GigaChat35Config(**text_config)

        if isinstance(vision_config, dict):
            vision_config = InternVisionConfig(**vision_config)

        if isinstance(audio_config, dict):
            audio_config = GigaChat35OmniAudioConfig(**audio_config)

        self.text_config = text_config or GigaChat35Config()
        self.vision_config = vision_config or InternVisionConfig()
        self.audio_config = audio_config or GigaChat35OmniAudioConfig()
        self.downsample_ratio = downsample_ratio
        self.audio_stack_factor = audio_stack_factor
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.audio_token_id = audio_token_id
        self.video_audio_token_id = video_audio_token_id
        self.video_tokens_per_frame = video_tokens_per_frame
        super().__init__(**kwargs)
