import collections
import collections.abc
import logging
from collections.abc import Callable, Sequence
from typing import Iterable, List, Optional, Tuple, TypeAlias, cast

import torch
import torch.nn as nn
import torchaudio.functional as F
from transformers import PretrainedConfig

from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2 import Qwen2ForCausalLM
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)
_Tuple2: TypeAlias = int | tuple[int, int] | Sequence[int]


def _resolve_tuple2(x: _Tuple2) -> tuple[int, int]:
    if isinstance(x, collections.abc.Sequence):
        assert (
            len(x) == 2
        ), f"Expected a sequence of length 2, got {x} with length {len(x)}"
        return cast(tuple[int, int], tuple(x))
    return (x, x)


def calculate_mel_frames_dasheng(
    audio_length_samples: int,
    n_fft: int = 512,
    hop_size: int = 160,
    dasheng_subsampling: int = 4,
    center=True,
    model_subsampling: int = 5,
) -> int:
    """Calculate the number of Mel-spectrogram frames."""
    if center:
        audio_length_samples = audio_length_samples + n_fft

    return (
        int(1 + ((audio_length_samples - n_fft) / hop_size))
        // dasheng_subsampling
        // model_subsampling
    )


class AudioPatchEmbed(nn.Module):
    def __init__(
        self,
        input_size: _Tuple2 = 64,
        patch_size: _Tuple2 = 16,
        patch_stride: _Tuple2 = 16,
        in_chans: int = 1,
        embed_dim: int = 768,
        norm_layer: Callable | None = None,
        flatten: bool = False,
    ):
        super().__init__()
        self.input_size = _resolve_tuple2(input_size)
        self.patch_size = _resolve_tuple2(patch_size)
        self.patch_stride = _resolve_tuple2(patch_stride)
        self.grid_size = (
            self.input_size[0] // self.patch_stride[0],
            self.input_size[1] // self.patch_stride[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_stride,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        if self.flatten:
            x = torch.permute(torch.flatten(x, 2, 3), (0, 2, 1))
        x = self.norm(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class DashengMlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ColumnParallelLinear(
            input_size=in_features,
            output_size=hidden_features,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("fc1", prefix),
        )
        self.act = nn.GELU()
        self.fc2 = RowParallelLinear(
            input_size=hidden_features,
            output_size=out_features,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("fc2", prefix),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc1(x)
        x = self.act(x)
        x, _ = self.fc2(x)
        return x


class DashengAttention(nn.Module):
    """Audio encoder attention using VisionAttention for compatibility."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.embed_dim = dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.attn = VisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            use_qkv_parallel=True,
            proj_bias=True,
            qkv_bias=qkv_bias,
            qkv_backend="sdpa",
            softmax_in_single_precision=False,
            flatten_batch=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Args:
            x: [B, N, C] tensor
            mask: [B, N] boolean mask
        """
        attn_mask = None
        if mask is not None:
            attn_mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
            attn_mask = attn_mask.float()
            attn_mask = (1.0 - attn_mask) * -10000.0

        x = self.attn(x, attn_mask=attn_mask)
        return x


class DashengBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        init_values: float | None = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = DashengAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = DashengMlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x), mask))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DashengFrontend(nn.Module):
    """Audio frontend that converts waveforms to log mel-spectrograms."""

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.win_length = config.win_length
        self.center = config.center
        spectrogram_window = torch.hann_window(config.win_length)
        self.register_buffer(
            "spectrogram_window",
            spectrogram_window,
            persistent=False,
        )
        self.spectrogram_window: torch.Tensor
        melscale_fbanks = F.melscale_fbanks(
            n_freqs=config.n_fft // 2 + 1,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
            sample_rate=config.sample_rate,
        )
        self.register_buffer("melscale_fbanks", melscale_fbanks, persistent=False)
        self.melscale_fbanks: torch.Tensor

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveform to log mel-spectrogram.

        Args:
            waveform: [B, T] tensor of audio samples

        Returns:
            log_mel_spectrogram: [B, n_mels, time] tensor
        """
        spectrogram = F.spectrogram(
            waveform=waveform.to(torch.float32),
            pad=0,
            window=self.spectrogram_window,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            power=2,
            normalized=False,
            center=self.center,
        )
        mel_spectrogram = (spectrogram.mT @ self.melscale_fbanks.to(torch.float32)).mT
        log_mel_spectrogram = F.amplitude_to_DB(
            mel_spectrogram.unsqueeze(1),
            multiplier=10,
            amin=1e-10,
            db_multiplier=0,
            top_db=120,
        ).squeeze(1)
        return log_mel_spectrogram.to(waveform.dtype)


class DashengAudioTransformer(nn.Module):
    """Audio encoder transformer."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.target_length = config.target_length
        self.hop_length = config.hop_length
        self.front_end = DashengFrontend(config)
        self.init_bn = nn.BatchNorm2d(config.n_mels, momentum=0.01)
        self.patch_embed = AudioPatchEmbed(
            input_size=(config.n_mels, config.target_length),
            embed_dim=config.embed_dim,
            in_chans=config.input_channels,
            patch_size=config.patch_size,
            flatten=False,
            patch_stride=config.patch_stride,
        )
        self.time_pos_embed = nn.Parameter(
            torch.empty(1, config.embed_dim, 1, self.patch_embed.grid_size[1])
        )
        self.freq_pos_embed = nn.Parameter(
            torch.empty(1, config.embed_dim, self.patch_embed.grid_size[0], 1)
        )
        self.blocks = nn.ModuleList(
            DashengBlock(
                dim=config.embed_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias,
                init_values=config.init_values,
                quant_config=quant_config,
                prefix=add_prefix(f"blocks.{i}", prefix),
            )
            for i in range(config.depth)
        )
        self.norm = nn.LayerNorm(config.embed_dim, eps=1e-6)

    def forward_features(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t = x.shape[-1]
        x = x + self.time_pos_embed[:, :, :, :t]
        x = x + self.freq_pos_embed[:, :, :, :]
        x = torch.permute(torch.flatten(x, 2, 3), (0, 2, 1))
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        return x

    def _to_mask(self, lengths: torch.Tensor, max_length: int) -> torch.Tensor:
        batch_size = len(lengths)
        idx = torch.arange(max_length, device=lengths.device)
        idx = idx.repeat(batch_size).view(batch_size, max_length)
        mask = (idx < lengths.unsqueeze(-1)).bool()
        return mask

    def forward(
        self,
        x: torch.Tensor,
        x_length: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            x: [B, T] audio waveform tensor
            x_length: [B] tensor of audio lengths

        Returns:
            x: [B, seq_len, embed_dim] encoded features
            mask: [B, seq_len] mask tensor
        """
        x = self.front_end(x)
        x = x.to(self.time_pos_embed.dtype)
        target_length_in_patches = self.target_length // 4
        x = x.unsqueeze(1)
        x = torch.permute(x, (0, 2, 1, 3))
        x = self.init_bn(x)
        x = torch.permute(x, (0, 2, 1, 3))
        x = self.patch_embed(x)
        t = x.shape[-1]
        input_splits = x.split(target_length_in_patches, dim=-1)
        if x_length is not None:
            assert len(x_length) == len(
                x
            ), "batchsizes of input x and x_length need to be same"
            assert x_length.ndim == 1, "Lengths are of size (B,)"
            scaled_lengths = (x_length / (self.hop_length * 4)).long()
            mask = self._to_mask(max_length=t, lengths=scaled_lengths)
            split_masks = mask.split(target_length_in_patches, dim=-1)
        else:
            mask = None
            split_masks = [None] * len(input_splits)
        outputs = []
        for split_x, split_mask in zip(input_splits, split_masks):
            forward_kwargs = {}
            forward_kwargs["mask"] = split_mask
            split_x = self.forward_features(split_x, **forward_kwargs)
            outputs.append(split_x)
        x = torch.cat(outputs, dim=1)
        return x, mask


class AudioProjectorSubsample(nn.Module):
    """Audio projector with subsampling."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        downsample_rate=5,
        dtype: torch.dtype | None = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.k = downsample_rate
        self.fc1 = ColumnParallelLinear(
            input_size=in_dim * self.k,
            output_size=out_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("net.0", prefix),
        )
        self.act = nn.GELU()
        self.fc2 = RowParallelLinear(
            input_size=out_dim,
            output_size=out_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("net.2", prefix),
        )

    def forward(self, x, mask=None):
        batch_size, seq_len, dim = x.shape
        num_frames_to_discard = seq_len % self.k
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
            if mask is not None:
                mask = mask[:, :-num_frames_to_discard]
        if mask is None:
            mask = torch.ones(x.shape[:-1], dtype=torch.long, device=x.device)
        x = x.reshape(batch_size, -1, self.k * dim)
        x, _ = self.fc1(x)
        x = self.act(x)
        x, _ = self.fc2(x)
        mask = mask.reshape(batch_size, -1, self.k)
        mask = mask.any(dim=-1).long()
        return x, mask


class MiDashengLMModel(nn.Module):
    """MiDashengLM model for audio-language processing."""

    default_bitsandbytes_target_modules = [
        ".fc1.",
        ".fc2.",
        ".gate_up_proj.",
        ".down_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]

    bitsandbytes_stacked_params_mapping = {
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        if (
            hasattr(config.text_config, "rope_scaling")
            and config.text_config.rope_scaling
        ):
            if "mrope_section" in config.text_config.rope_scaling:

                new_rope_scaling = {
                    k: v
                    for k, v in config.text_config.rope_scaling.items()
                    if k != "mrope_section"
                }
                config.text_config.rope_scaling = (
                    new_rope_scaling if new_rope_scaling else None
                )
        self.audio_encoder = DashengAudioTransformer(
            config.audio_encoder_config,
            quant_config=quant_config,
            prefix=add_prefix("audio_encoder", prefix),
        )
        self.audio_projector = AudioProjectorSubsample(
            in_dim=config.audio_encoder_config.embed_dim,
            out_dim=config.text_config.hidden_size,
            downsample_rate=config.subsample_factor,
            quant_config=quant_config,
            prefix=add_prefix("audio_projector", prefix),
        )
        self.language_model = Qwen2ForCausalLM(
            config.text_config,
            quant_config=quant_config,
            prefix=add_prefix("decoder", prefix),
        )
        self.logits_processor = self.language_model.logits_processor
        self.quant_config = quant_config

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        """Pad input IDs with multimodal tokens."""
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """Process audio inputs and return embeddings.

        Args:
            items: List of multimodal data items containing audio features

        Returns:
            audio_embeddings: Concatenated audio embeddings
        """
        logger.debug("=" * 80)
        logger.debug(f"get_audio_feature called with {len(items)} items")
        logger.debug("=" * 80)
        for i, item in enumerate(items):
            logger.debug(f"Item {i} feature shape: {item.feature.shape}")
            logger.debug(
                f"Item {i} audio_length: {getattr(item, 'audio_length', 'NOT SET')}"
            )
            logger.debug(f"Item {i} pad_value: {getattr(item, 'pad_value', 'NOT SET')}")
            logger.debug(f"Item {i} hash: {getattr(item, 'hash', 'NOT SET')}")
        input_values = torch.cat([item.feature for item in items], dim=0)
        logger.debug(f"Concatenated input_values shape: {input_values.shape}")
        audio_lengths = []
        for item in items:
            if hasattr(item, "audio_length") and item.audio_length is not None:
                audio_lengths.append(item.audio_length)
            else:
                audio_lengths.append(item.feature.shape[-1])
        audio_length = torch.tensor(audio_lengths, device=input_values.device)
        logger.debug(f"audio_length: {audio_length}")
        encoder_out, encoder_atts = self.audio_encoder(input_values, audio_length)
        logger.debug(f"Encoder output shape: {encoder_out.shape}")
        audio_embeddings, _ = self.audio_projector(encoder_out, encoder_atts)
        audio_embeddings = audio_embeddings.to(input_values.dtype)
        logger.debug(f"Projector output shape: {audio_embeddings.shape}")
        batch_size, max_audio_tokens, embed_dim = audio_embeddings.shape
        logger.debug(f"Using all {max_audio_tokens} audio tokens from projector output")
        masked_audio_features = audio_embeddings.reshape(-1, embed_dim)
        logger.debug(f"Final output shape: {masked_audio_features.shape}")
        logger.debug(
            f"Stats: min={masked_audio_features.min().item():.4f}, max={masked_audio_features.max().item():.4f}"
        )
        logger.debug(
            f"Audio embeddings dtype: {masked_audio_features.dtype}, device: {masked_audio_features.device}"
        )
        logger.debug(
            f"First 5 values of first audio token: {masked_audio_features[0, :5].tolist()}"
        )
        logger.debug("=" * 80)
        return masked_audio_features

    def get_input_embeddings(self):
        return self.language_model.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs,
    ):
        """Run forward pass for MiDashengLM.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a batch.
            positions: Flattened (concatenated) position ids corresponding to a batch.
            forward_batch: Forward batch information including multimodal data.
        """
        if forward_batch.contains_mm_inputs():
            logger.debug("=" * 80)
            logger.debug(f"input_ids shape: {input_ids.shape}")
            logger.debug(f"input_ids first 20: {input_ids[:20].tolist()}")
            logger.debug(
                f"input_ids unique values count: {len(torch.unique(input_ids))}"
            )
            if forward_batch.mm_inputs and len(forward_batch.mm_inputs) > 0:
                mm_input = forward_batch.mm_inputs[0]
                if mm_input and len(mm_input.mm_items) > 0:
                    pad_value = mm_input.mm_items[0].pad_value
                    logger.debug(f"Expected pad_value: {pad_value}")
                    logger.debug(
                        f"Count of pad_value in input_ids: {(input_ids == pad_value).sum().item()}"
                    )
                    if hasattr(mm_input, "audio_token_id") and mm_input.audio_token_id:
                        logger.debug(f"audio_token_id: {mm_input.audio_token_id}")
                        logger.debug(
                            f"Count of audio_token_id in input_ids: {(input_ids == mm_input.audio_token_id).sum().item()}"
                        )
            logger.debug("=" * 80)

        return general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            positions=positions,
            data_embedding_funcs={Modality.AUDIO: self.get_audio_feature},
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load model weights."""
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        buffers_dict = dict(self.named_buffers())
        audio_encoder_loaded = []
        audio_projector_loaded = []
        skipped_weights = []
        decoder_weights = []
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue
            if name.startswith("decoder"):
                decoder_weights.append((name, loaded_weight))
                continue
            original_name = name
            if "audio_encoder.front_end" in name:
                if ".mel_scale.fb" in name:
                    name = name.replace(".mel_scale.fb", ".melscale_fbanks")
                elif ".spectrogram.window" in name:
                    name = name.replace(".spectrogram.window", ".spectrogram_window")
            if "audio_encoder" in name and ".attn.qkv." in name:
                name = name.replace(".attn.qkv.", ".attn.attn.qkv_proj.")
            if "audio_encoder" in name and ".attn.proj." in name:
                name = name.replace(".attn.proj.", ".attn.attn.proj.")
            if "audio_projector" in name:
                name = name.replace(".net.0.", ".fc1.")
                name = name.replace(".net.2.", ".fc2.")
            if (
                name.endswith(".bias")
                and name not in params_dict
                and name not in buffers_dict
            ):
                skipped_weights.append(f"{original_name} (bias not in params/buffers)")
                continue
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            elif name in buffers_dict:
                buffers_dict[name].copy_(loaded_weight)
            else:
                if "audio_projector" in original_name:
                    skipped_weights.append(f"{original_name} -> {name} (NOT IN MODEL)")
                else:
                    skipped_weights.append(f"{original_name} (not in model)")
                continue

            if "audio_encoder" in original_name:
                audio_encoder_loaded.append(original_name)
            elif "audio_projector" in original_name:
                audio_projector_loaded.append(original_name)
        if decoder_weights:
            logger.debug(
                f"Passing {len(decoder_weights)} decoder weights to language_model.load_weights()"
            )
            decoder_weights_stripped = [
                (name.replace("decoder.", "", 1), weight)
                for name, weight in decoder_weights
            ]
            self.language_model.load_weights(decoder_weights_stripped)
        logger.debug("=" * 80)
        logger.debug(f"Audio encoder weights loaded: {len(audio_encoder_loaded)}")
        logger.debug(f"Audio projector weights loaded: {len(audio_projector_loaded)}")
        logger.debug(
            f"Decoder weights passed to language_model: {len(decoder_weights)}"
        )
        logger.debug(f"Skipped weights: {len(skipped_weights)}")
        encoder_skipped = [s for s in skipped_weights if "audio_encoder" in s]
        projector_skipped = [s for s in skipped_weights if "audio_projector" in s]
        if projector_skipped:
            logger.debug("Skipped audio_projector weights:")
            for s in projector_skipped:
                logger.debug(f"  {s}")
        if encoder_skipped:
            logger.debug(f"Skipped audio_encoder weights: {len(encoder_skipped)}")
            non_bias_skipped = [s for s in encoder_skipped if "bias" not in s]
            if non_bias_skipped:
                logger.debug("  First 10 non-bias skipped:")
                for s in non_bias_skipped[:10]:
                    logger.debug(f"    {s}")
        logger.debug("=" * 80)

    def get_embed_and_head(self):
        return (
            self.language_model.model.embed_tokens.weight,
            self.language_model.lm_head.weight,
        )


EntryClass = [MiDashengLMModel]
