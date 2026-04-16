from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.vae import (
    DecoderOutput,
    DiagonalGaussianDistribution,
)
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from torch import nn

from sglang.multimodal_gen.configs.models.vaes.ltx_audio import LTXAudioVAEConfig
from sglang.multimodal_gen.runtime.models.vaes.common import ParallelTiledVAE

LATENT_DOWNSAMPLE_FACTOR = 4


class LTX2AudioCausalConv2d(nn.Module):
    """
    A causal 2D convolution that pads asymmetrically along the causal axis.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: int = 1,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        causality_axis: str = "height",
    ) -> None:
        super().__init__()

        self.causality_axis = causality_axis
        kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )
        dilation = (dilation, dilation) if isinstance(dilation, int) else dilation

        pad_h = (kernel_size[0] - 1) * dilation[0]
        pad_w = (kernel_size[1] - 1) * dilation[1]

        if self.causality_axis == "none":
            padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        elif self.causality_axis in {"width", "width-compatibility"}:
            padding = (pad_w, 0, pad_h // 2, pad_h - pad_h // 2)
        elif self.causality_axis == "height":
            padding = (pad_w // 2, pad_w - pad_w // 2, pad_h, 0)
        else:
            raise ValueError(f"Invalid causality_axis: {causality_axis}")

        self.padding = padding
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self.padding)
        return self.conv(x)


class LTX2AudioPixelNorm(nn.Module):
    """
    Per-pixel (per-location) RMS normalization layer.
    """

    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_sq = torch.mean(x**2, dim=self.dim, keepdim=True)
        rms = torch.sqrt(mean_sq + self.eps)
        return x / rms


class LTX2AudioAttnBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        norm_type: str = "group",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels

        if norm_type == "group":
            self.norm = nn.GroupNorm(
                num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
            )
        elif norm_type == "pixel":
            self.norm = LTX2AudioPixelNorm(dim=1, eps=1e-6)
        else:
            raise ValueError(f"Invalid normalization type: {norm_type}")
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        batch, channels, height, width = q.shape
        q = q.reshape(batch, channels, height * width).permute(0, 2, 1).contiguous()
        k = k.reshape(batch, channels, height * width).contiguous()
        attn = torch.bmm(q, k) * (int(channels) ** (-0.5))
        attn = torch.nn.functional.softmax(attn, dim=2)

        v = v.reshape(batch, channels, height * width)
        attn = attn.permute(0, 2, 1).contiguous()
        h_ = torch.bmm(v, attn).reshape(batch, channels, height, width)

        h_ = self.proj_out(h_)
        return x + h_


class LTX2AudioResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        norm_type: str = "group",
        causality_axis: str = "height",
    ) -> None:
        super().__init__()
        self.causality_axis = causality_axis

        if (
            self.causality_axis is not None
            and self.causality_axis != "none"
            and norm_type == "group"
        ):
            raise ValueError("Causal ResnetBlock with GroupNorm is not supported.")
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        if norm_type == "group":
            self.norm1 = nn.GroupNorm(
                num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
            )
        elif norm_type == "pixel":
            self.norm1 = LTX2AudioPixelNorm(dim=1, eps=1e-6)
        else:
            raise ValueError(f"Invalid normalization type: {norm_type}")
        self.non_linearity = nn.SiLU()
        if causality_axis is not None:
            self.conv1 = LTX2AudioCausalConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                causality_axis=causality_axis,
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1
            )
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        if norm_type == "group":
            self.norm2 = nn.GroupNorm(
                num_groups=32, num_channels=out_channels, eps=1e-6, affine=True
            )
        elif norm_type == "pixel":
            self.norm2 = LTX2AudioPixelNorm(dim=1, eps=1e-6)
        else:
            raise ValueError(f"Invalid normalization type: {norm_type}")
        self.dropout = nn.Dropout(dropout)
        if causality_axis is not None:
            self.conv2 = LTX2AudioCausalConv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                causality_axis=causality_axis,
            )
        else:
            self.conv2 = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1
            )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                if causality_axis is not None:
                    self.conv_shortcut = LTX2AudioCausalConv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        causality_axis=causality_axis,
                    )
                else:
                    self.conv_shortcut = nn.Conv2d(
                        in_channels, out_channels, kernel_size=3, stride=1, padding=1
                    )
            else:
                if causality_axis is not None:
                    self.nin_shortcut = LTX2AudioCausalConv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=1,
                        causality_axis=causality_axis,
                    )
                else:
                    self.nin_shortcut = nn.Conv2d(
                        in_channels, out_channels, kernel_size=1, stride=1, padding=0
                    )

    def forward(
        self, x: torch.Tensor, temb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h = self.norm1(x)
        h = self.non_linearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(self.non_linearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = self.non_linearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = (
                self.conv_shortcut(x)
                if self.use_conv_shortcut
                else self.nin_shortcut(x)
            )

        return x + h


class LTX2AudioDownsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        with_conv: bool,
        causality_axis: Optional[str] = "height",
    ) -> None:
        super().__init__()
        self.with_conv = with_conv
        self.causality_axis = causality_axis

        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            # Padding tuple is in the order: (left, right, top, bottom).
            if self.causality_axis == "none":
                pad = (0, 1, 0, 1)
            elif self.causality_axis == "width":
                pad = (2, 0, 0, 1)
            elif self.causality_axis == "height":
                pad = (0, 1, 2, 0)
            elif self.causality_axis == "width-compatibility":
                pad = (1, 0, 0, 1)
            else:
                raise ValueError(
                    f"Invalid `causality_axis` {self.causality_axis}; supported values are `none`, `width`, `height`,"
                    f" and `width-compatibility`."
                )

            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            # with_conv=False implies that causality_axis is "none"
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class LTX2AudioUpsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        with_conv: bool,
        causality_axis: Optional[str] = "height",
    ) -> None:
        super().__init__()
        self.with_conv = with_conv
        self.causality_axis = causality_axis
        if self.with_conv:
            if causality_axis is not None:
                self.conv = LTX2AudioCausalConv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    causality_axis=causality_axis,
                )
            else:
                self.conv = nn.Conv2d(
                    in_channels, in_channels, kernel_size=3, stride=1, padding=1
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
            if self.causality_axis is None or self.causality_axis == "none":
                pass
            elif self.causality_axis == "height":
                x = x[:, :, 1:, :]
            elif self.causality_axis == "width":
                x = x[:, :, :, 1:]
            elif self.causality_axis == "width-compatibility":
                pass
            else:
                raise ValueError(f"Invalid causality_axis: {self.causality_axis}")

        return x


class LTX2AudioAudioPatchifier:
    """
    Patchifier for spectrogram/audio latents.
    """

    def __init__(
        self,
        patch_size: int,
        sample_rate: int = 16000,
        hop_length: int = 160,
        audio_latent_downsample_factor: int = 4,
        is_causal: bool = True,
    ):
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.audio_latent_downsample_factor = audio_latent_downsample_factor
        self.is_causal = is_causal
        self._patch_size = (1, patch_size, patch_size)

    def patchify(self, audio_latents: torch.Tensor) -> torch.Tensor:
        batch, channels, time, freq = audio_latents.shape
        return audio_latents.permute(0, 2, 1, 3).reshape(batch, time, channels * freq)

    def unpatchify(
        self, audio_latents: torch.Tensor, channels: int, mel_bins: int
    ) -> torch.Tensor:
        batch, time, _ = audio_latents.shape
        return audio_latents.view(batch, time, channels, mel_bins).permute(0, 2, 1, 3)

    @property
    def patch_size(self) -> Tuple[int, int, int]:
        return self._patch_size


class LTX2AudioEncoder(nn.Module):
    def __init__(
        self,
        base_channels: int = 128,
        output_channels: int = 1,
        num_res_blocks: int = 2,
        attn_resolutions: Optional[Tuple[int, ...]] = None,
        in_channels: int = 2,
        resolution: int = 256,
        latent_channels: int = 8,
        ch_mult: Tuple[int, ...] = (1, 2, 4),
        norm_type: str = "group",
        causality_axis: Optional[str] = "width",
        dropout: float = 0.0,
        mid_block_add_attention: bool = False,
        sample_rate: int = 16000,
        mel_hop_length: int = 160,
        is_causal: bool = True,
        mel_bins: Optional[int] = 64,
        double_z: bool = True,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.mel_hop_length = mel_hop_length
        self.is_causal = is_causal
        self.mel_bins = mel_bins

        self.base_channels = base_channels
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_ch = output_channels
        self.give_pre_end = False
        self.tanh_out = False
        self.norm_type = norm_type
        self.latent_channels = latent_channels
        self.channel_multipliers = ch_mult
        self.attn_resolutions = attn_resolutions
        self.causality_axis = causality_axis

        base_block_channels = base_channels
        base_resolution = resolution
        self.z_shape = (1, latent_channels, base_resolution, base_resolution)

        if self.causality_axis is not None:
            self.conv_in = LTX2AudioCausalConv2d(
                in_channels,
                base_block_channels,
                kernel_size=3,
                stride=1,
                causality_axis=self.causality_axis,
            )
        else:
            self.conv_in = nn.Conv2d(
                in_channels, base_block_channels, kernel_size=3, stride=1, padding=1
            )

        self.down = nn.ModuleList()
        block_in = base_block_channels
        curr_res = self.resolution

        for level in range(self.num_resolutions):
            stage = nn.Module()
            stage.block = nn.ModuleList()
            stage.attn = nn.ModuleList()
            block_out = self.base_channels * self.channel_multipliers[level]

            for _ in range(self.num_res_blocks):
                stage.block.append(
                    LTX2AudioResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        norm_type=self.norm_type,
                        causality_axis=self.causality_axis,
                    )
                )
                block_in = block_out
                if self.attn_resolutions:
                    if curr_res in self.attn_resolutions:
                        stage.attn.append(
                            LTX2AudioAttnBlock(block_in, norm_type=self.norm_type)
                        )

            if level != self.num_resolutions - 1:
                stage.downsample = LTX2AudioDownsample(
                    block_in, True, causality_axis=self.causality_axis
                )
                curr_res = curr_res // 2

            self.down.append(stage)

        self.mid = nn.Module()
        self.mid.block_1 = LTX2AudioResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
        )
        if mid_block_add_attention:
            self.mid.attn_1 = LTX2AudioAttnBlock(block_in, norm_type=self.norm_type)
        else:
            self.mid.attn_1 = nn.Identity()
        self.mid.block_2 = LTX2AudioResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
        )

        final_block_channels = block_in
        z_channels = 2 * latent_channels if double_z else latent_channels
        if self.norm_type == "group":
            self.norm_out = nn.GroupNorm(
                num_groups=32, num_channels=final_block_channels, eps=1e-6, affine=True
            )
        elif self.norm_type == "pixel":
            self.norm_out = LTX2AudioPixelNorm(dim=1, eps=1e-6)
        else:
            raise ValueError(f"Invalid normalization type: {self.norm_type}")
        self.non_linearity = nn.SiLU()

        if self.causality_axis is not None:
            self.conv_out = LTX2AudioCausalConv2d(
                final_block_channels,
                z_channels,
                kernel_size=3,
                stride=1,
                causality_axis=self.causality_axis,
            )
        else:
            self.conv_out = nn.Conv2d(
                final_block_channels, z_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states expected shape: (batch_size, channels, time, num_mel_bins)
        hidden_states = self.conv_in(hidden_states)

        for level in range(self.num_resolutions):
            stage = self.down[level]
            for block_idx, block in enumerate(stage.block):
                hidden_states = block(hidden_states, temb=None)
                if stage.attn:
                    hidden_states = stage.attn[block_idx](hidden_states)

            if level != self.num_resolutions - 1 and hasattr(stage, "downsample"):
                hidden_states = stage.downsample(hidden_states)

        hidden_states = self.mid.block_1(hidden_states, temb=None)
        hidden_states = self.mid.attn_1(hidden_states)
        hidden_states = self.mid.block_2(hidden_states, temb=None)

        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.non_linearity(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        return hidden_states


class LTX2AudioDecoder(nn.Module):
    """
    Symmetric decoder that reconstructs audio spectrograms from latent features.

    The decoder mirrors the encoder structure with configurable channel multipliers, attention resolutions, and causal
    convolutions.
    """

    def __init__(
        self,
        base_channels: int = 128,
        output_channels: int = 1,
        num_res_blocks: int = 2,
        attn_resolutions: Optional[Tuple[int, ...]] = None,
        in_channels: int = 2,
        resolution: int = 256,
        latent_channels: int = 8,
        ch_mult: Tuple[int, ...] = (1, 2, 4),
        norm_type: str = "group",
        causality_axis: Optional[str] = "width",
        dropout: float = 0.0,
        mid_block_add_attention: bool = False,
        sample_rate: int = 16000,
        mel_hop_length: int = 160,
        is_causal: bool = True,
        mel_bins: Optional[int] = 64,
    ) -> None:
        super().__init__()

        self.sample_rate = sample_rate
        self.mel_hop_length = mel_hop_length
        self.is_causal = is_causal
        self.mel_bins = mel_bins
        self.patchifier = LTX2AudioAudioPatchifier(
            patch_size=1,
            audio_latent_downsample_factor=LATENT_DOWNSAMPLE_FACTOR,
            sample_rate=sample_rate,
            hop_length=mel_hop_length,
            is_causal=is_causal,
        )

        self.base_channels = base_channels
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_ch = output_channels
        self.give_pre_end = False
        self.tanh_out = False
        self.norm_type = norm_type
        self.latent_channels = latent_channels
        self.channel_multipliers = ch_mult
        self.attn_resolutions = attn_resolutions
        self.causality_axis = causality_axis

        base_block_channels = base_channels * self.channel_multipliers[-1]
        base_resolution = resolution // (2 ** (self.num_resolutions - 1))
        self.z_shape = (1, latent_channels, base_resolution, base_resolution)

        if self.causality_axis is not None:
            self.conv_in = LTX2AudioCausalConv2d(
                latent_channels,
                base_block_channels,
                kernel_size=3,
                stride=1,
                causality_axis=self.causality_axis,
            )
        else:
            self.conv_in = nn.Conv2d(
                latent_channels, base_block_channels, kernel_size=3, stride=1, padding=1
            )
        self.non_linearity = nn.SiLU()
        self.mid = nn.Module()
        self.mid.block_1 = LTX2AudioResnetBlock(
            in_channels=base_block_channels,
            out_channels=base_block_channels,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
        )
        if mid_block_add_attention:
            self.mid.attn_1 = LTX2AudioAttnBlock(
                base_block_channels, norm_type=self.norm_type
            )
        else:
            self.mid.attn_1 = nn.Identity()
        self.mid.block_2 = LTX2AudioResnetBlock(
            in_channels=base_block_channels,
            out_channels=base_block_channels,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
        )

        self.up = nn.ModuleList()
        block_in = base_block_channels
        curr_res = self.resolution // (2 ** (self.num_resolutions - 1))

        for level in reversed(range(self.num_resolutions)):
            stage = nn.Module()
            stage.block = nn.ModuleList()
            stage.attn = nn.ModuleList()
            block_out = self.base_channels * self.channel_multipliers[level]

            for _ in range(self.num_res_blocks + 1):
                stage.block.append(
                    LTX2AudioResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        norm_type=self.norm_type,
                        causality_axis=self.causality_axis,
                    )
                )
                block_in = block_out
                if self.attn_resolutions:
                    if curr_res in self.attn_resolutions:
                        stage.attn.append(
                            LTX2AudioAttnBlock(block_in, norm_type=self.norm_type)
                        )

            if level != 0:
                stage.upsample = LTX2AudioUpsample(
                    block_in, True, causality_axis=self.causality_axis
                )
                curr_res *= 2

            self.up.insert(0, stage)

        final_block_channels = block_in

        if self.norm_type == "group":
            self.norm_out = nn.GroupNorm(
                num_groups=32, num_channels=final_block_channels, eps=1e-6, affine=True
            )
        elif self.norm_type == "pixel":
            self.norm_out = LTX2AudioPixelNorm(dim=1, eps=1e-6)
        else:
            raise ValueError(f"Invalid normalization type: {self.norm_type}")

        if self.causality_axis is not None:
            self.conv_out = LTX2AudioCausalConv2d(
                final_block_channels,
                output_channels,
                kernel_size=3,
                stride=1,
                causality_axis=self.causality_axis,
            )
        else:
            self.conv_out = nn.Conv2d(
                final_block_channels,
                output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )

    def forward(
        self,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        _, _, frames, mel_bins = sample.shape

        target_frames = frames * LATENT_DOWNSAMPLE_FACTOR

        if self.causality_axis is not None:
            target_frames = max(target_frames - (LATENT_DOWNSAMPLE_FACTOR - 1), 1)

        target_channels = self.out_ch
        target_mel_bins = self.mel_bins if self.mel_bins is not None else mel_bins

        hidden_features = self.conv_in(sample)
        hidden_features = self.mid.block_1(hidden_features, temb=None)
        hidden_features = self.mid.attn_1(hidden_features)
        hidden_features = self.mid.block_2(hidden_features, temb=None)

        for level in reversed(range(self.num_resolutions)):
            stage = self.up[level]
            for block_idx, block in enumerate(stage.block):
                hidden_features = block(hidden_features, temb=None)
                if stage.attn:
                    hidden_features = stage.attn[block_idx](hidden_features)

            if level != 0 and hasattr(stage, "upsample"):
                hidden_features = stage.upsample(hidden_features)

        if self.give_pre_end:
            return hidden_features

        hidden = self.norm_out(hidden_features)
        hidden = self.non_linearity(hidden)
        decoded_output = self.conv_out(hidden)
        decoded_output = torch.tanh(decoded_output) if self.tanh_out else decoded_output

        _, _, current_time, current_freq = decoded_output.shape
        target_time = target_frames
        target_freq = target_mel_bins

        decoded_output = decoded_output[
            :,
            :target_channels,
            : min(current_time, target_time),
            : min(current_freq, target_freq),
        ]

        time_padding_needed = target_time - decoded_output.shape[2]
        freq_padding_needed = target_freq - decoded_output.shape[3]

        if time_padding_needed > 0 or freq_padding_needed > 0:
            padding = (
                0,
                max(freq_padding_needed, 0),
                0,
                max(time_padding_needed, 0),
            )
            decoded_output = F.pad(decoded_output, padding)

        decoded_output = decoded_output[:, :target_channels, :target_time, :target_freq]

        return decoded_output


class AutoencoderKLLTX2Audio(ParallelTiledVAE):
    r"""
    LTX2 audio VAE for encoding and decoding audio latent representations.
    """

    _supports_gradient_checkpointing = False

    def __init__(
        self,
        config: LTXAudioVAEConfig,
    ) -> None:
        super().__init__(config=config)

        causality_axis = config.arch_config.causality_axis
        attn_resolutions = config.arch_config.attn_resolutions
        base_channels = config.arch_config.base_channels
        output_channels = config.arch_config.output_channels
        ch_mult = config.arch_config.ch_mult
        num_res_blocks = config.arch_config.num_res_blocks
        in_channels = config.arch_config.in_channels
        resolution = config.arch_config.resolution
        latent_channels = config.arch_config.latent_channels
        norm_type = config.arch_config.norm_type
        dropout = config.arch_config.dropout
        mid_block_add_attention = config.arch_config.mid_block_add_attention
        sample_rate = config.arch_config.sample_rate
        mel_hop_length = config.arch_config.mel_hop_length
        is_causal = config.arch_config.is_causal
        mel_bins = config.arch_config.mel_bins
        double_z = config.arch_config.double_z

        supported_causality_axes = {"none", "width", "height", "width-compatibility"}
        if causality_axis not in supported_causality_axes:
            raise ValueError(
                f"{causality_axis=} is not valid. Supported values: {supported_causality_axes}"
            )

        attn_resolution_set = (
            set(attn_resolutions) if attn_resolutions else attn_resolutions
        )

        self.encoder = LTX2AudioEncoder(
            base_channels=base_channels,
            output_channels=output_channels,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolution_set,
            in_channels=in_channels,
            resolution=resolution,
            latent_channels=latent_channels,
            norm_type=norm_type,
            causality_axis=causality_axis,
            dropout=dropout,
            mid_block_add_attention=mid_block_add_attention,
            sample_rate=sample_rate,
            mel_hop_length=mel_hop_length,
            is_causal=is_causal,
            mel_bins=mel_bins,
            double_z=double_z,
        )

        self.decoder = LTX2AudioDecoder(
            base_channels=base_channels,
            output_channels=output_channels,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolution_set,
            in_channels=in_channels,
            resolution=resolution,
            latent_channels=latent_channels,
            norm_type=norm_type,
            causality_axis=causality_axis,
            dropout=dropout,
            mid_block_add_attention=mid_block_add_attention,
            sample_rate=sample_rate,
            mel_hop_length=mel_hop_length,
            is_causal=is_causal,
            mel_bins=mel_bins,
        )

        # Per-channel statistics for normalizing and denormalizing the latent representation. This statistics is computed over
        # the entire dataset and stored in model's checkpoint under AudioVAE state_dict
        latents_std = torch.zeros((base_channels,))
        latents_mean = torch.ones((base_channels,))
        self.register_buffer("latents_mean", latents_mean, persistent=True)
        self.register_buffer("latents_std", latents_std, persistent=True)

        # TODO: confirm whether the mel compression ratio below is correct
        self.mel_compression_ratio = LATENT_DOWNSAMPLE_FACTOR
        self.use_slicing = False

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def encode(self, x: torch.Tensor, return_dict: bool = True):
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x)
        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def decode(
        self, z: torch.Tensor, return_dict: bool = True
    ) -> Union[DecoderOutput, torch.Tensor]:
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice) for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z)

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, torch.Tensor]:
        posterior = self.encode(sample).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z)
        if not return_dict:
            return (dec.sample,)
        return dec


EntryClass = AutoencoderKLLTX2Audio
