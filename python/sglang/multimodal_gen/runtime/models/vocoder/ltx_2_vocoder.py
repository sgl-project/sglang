import math
from abc import ABC
from contextlib import nullcontext
from typing import Tuple

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.configs.models.vocoder.ltx_vocoder import LTXVocoderConfig

LRELU_SLOPE = 0.1


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def _sinc(x: torch.Tensor) -> torch.Tensor:
    return torch.where(
        x == 0,
        torch.tensor(1.0, device=x.device, dtype=x.dtype),
        torch.sin(math.pi * x) / math.pi / x,
    )


def kaiser_sinc_filter1d(
    cutoff: float, half_width: float, kernel_size: int
) -> torch.Tensor:
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    amplitude = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if amplitude > 50.0:
        beta = 0.1102 * (amplitude - 8.7)
    elif amplitude >= 21.0:
        beta = 0.5842 * (amplitude - 21) ** 0.4 + 0.07886 * (amplitude - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    time = (
        torch.arange(-half_size, half_size) + 0.5
        if even
        else torch.arange(kernel_size) - half_size
    )
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * _sinc(2 * cutoff * time)
        filter_ /= filter_.sum()
    return filter_.view(1, 1, kernel_size)


class LowPassFilter1d(nn.Module):
    def __init__(
        self,
        cutoff: float = 0.5,
        half_width: float = 0.6,
        stride: int = 1,
        padding: bool = True,
        padding_mode: str = "replicate",
        kernel_size: int = 12,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.register_buffer(
            "filter", kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, channels, _ = x.shape
        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        return F.conv1d(
            x,
            self.filter.expand(channels, -1, -1),
            stride=self.stride,
            groups=channels,
        )


class UpSample1d(nn.Module):
    def __init__(
        self,
        ratio: int = 2,
        kernel_size: int | None = None,
        persistent: bool = True,
        window_type: str = "kaiser",
    ):
        super().__init__()
        self.ratio = ratio
        self.stride = ratio

        if window_type == "hann":
            rolloff = 0.99
            lowpass_filter_width = 6
            width = math.ceil(lowpass_filter_width / rolloff)
            self.kernel_size = 2 * width * ratio + 1
            self.pad = width
            self.pad_left = 2 * width * ratio
            self.pad_right = self.kernel_size - ratio
            time_axis = (torch.arange(self.kernel_size) / ratio - width) * rolloff
            time_clamped = time_axis.clamp(
                -lowpass_filter_width, lowpass_filter_width
            )
            window = torch.cos(
                time_clamped * math.pi / lowpass_filter_width / 2
            ) ** 2
            sinc_filter = (
                torch.sinc(time_axis) * window * rolloff / ratio
            ).view(1, 1, -1)
        else:
            self.kernel_size = (
                int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
            )
            self.pad = self.kernel_size // ratio - 1
            self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
            self.pad_right = (
                self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
            )
            sinc_filter = kaiser_sinc_filter1d(
                cutoff=0.5 / ratio,
                half_width=0.6 / ratio,
                kernel_size=self.kernel_size,
            )

        self.register_buffer("filter", sinc_filter, persistent=persistent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, channels, _ = x.shape
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        filt = self.filter.to(dtype=x.dtype, device=x.device).expand(channels, -1, -1)
        x = self.ratio * F.conv_transpose1d(x, filt, stride=self.stride, groups=channels)
        return x[..., self.pad_left : -self.pad_right]


class DownSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: int | None = None):
        super().__init__()
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lowpass(x)


class Activation1d(nn.Module):
    def __init__(
        self,
        activation: nn.Module,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ):
        super().__init__()
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.act(x)
        return self.downsample(x)


class Snake(nn.Module):
    def __init__(
        self,
        in_features: int,
        alpha: float = 1.0,
        alpha_trainable: bool = True,
        alpha_logscale: bool = True,
    ):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        self.alpha = nn.Parameter(
            torch.zeros(in_features)
            if alpha_logscale
            else torch.ones(in_features) * alpha
        )
        self.alpha.requires_grad = alpha_trainable
        self.eps = 1e-9

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        return x + (1.0 / (alpha + self.eps)) * torch.sin(x * alpha).pow(2)


class SnakeBeta(nn.Module):
    def __init__(
        self,
        in_features: int,
        alpha: float = 1.0,
        alpha_trainable: bool = True,
        alpha_logscale: bool = True,
    ):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        self.alpha = nn.Parameter(
            torch.zeros(in_features)
            if alpha_logscale
            else torch.ones(in_features) * alpha
        )
        self.alpha.requires_grad = alpha_trainable
        self.beta = nn.Parameter(
            torch.zeros(in_features)
            if alpha_logscale
            else torch.ones(in_features) * alpha
        )
        self.beta.requires_grad = alpha_trainable
        self.eps = 1e-9

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        return x + (1.0 / (beta + self.eps)) * torch.sin(x * alpha).pow(2)


class ResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilations: Tuple[int, ...] = (1, 3, 5),
        leaky_relu_negative_slope: float = 0.1,
        padding_mode: str = "same",
    ):
        super().__init__()
        self.dilations = dilations
        self.negative_slope = leaky_relu_negative_slope

        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=padding_mode,
                )
                for dilation in dilations
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=stride,
                    dilation=1,
                    padding=padding_mode,
                )
                for _ in range(len(dilations))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, negative_slope=self.negative_slope)
            xt = conv1(xt)
            xt = F.leaky_relu(xt, negative_slope=self.negative_slope)
            xt = conv2(xt)
            x = x + xt
        return x


class AMPBlock1(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple[int, int, int] = (1, 3, 5),
        activation: str = "snake",
    ):
        super().__init__()
        act_cls = SnakeBeta if activation == "snakebeta" else Snake
        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[0],
                    padding=get_padding(kernel_size, dilation[0]),
                ),
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[1],
                    padding=get_padding(kernel_size, dilation[1]),
                ),
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[2],
                    padding=get_padding(kernel_size, dilation[2]),
                ),
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                ),
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                ),
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                ),
            ]
        )
        self.acts1 = nn.ModuleList(
            [Activation1d(act_cls(channels)) for _ in range(len(self.convs1))]
        )
        self.acts2 = nn.ModuleList(
            [Activation1d(act_cls(channels)) for _ in range(len(self.convs2))]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv1, conv2, act1, act2 in zip(
            self.convs1, self.convs2, self.acts1, self.acts2
        ):
            xt = act1(x)
            xt = conv1(xt)
            xt = act2(xt)
            xt = conv2(xt)
            x = x + xt
        return x


class LTX23MelSTFT(nn.Module):
    class STFTFn(nn.Module):
        def __init__(self, filter_length: int, hop_length: int, win_length: int):
            super().__init__()
            self.hop_length = hop_length
            self.win_length = win_length
            n_freqs = filter_length // 2 + 1
            self.register_buffer(
                "forward_basis", torch.zeros(n_freqs * 2, 1, filter_length)
            )
            self.register_buffer(
                "inverse_basis", torch.zeros(n_freqs * 2, 1, filter_length)
            )

        def forward(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            if y.dim() == 2:
                y = y.unsqueeze(1)
            left_pad = max(0, self.win_length - self.hop_length)
            y = F.pad(y, (left_pad, 0))
            spec = F.conv1d(y, self.forward_basis, stride=self.hop_length, padding=0)
            n_freqs = spec.shape[1] // 2
            real, imag = spec[:, :n_freqs], spec[:, n_freqs:]
            magnitude = torch.sqrt(real**2 + imag**2)
            phase = torch.atan2(imag.float(), real.float()).to(real.dtype)
            return magnitude, phase

    def __init__(
        self, filter_length: int, hop_length: int, win_length: int, n_mel_channels: int
    ):
        super().__init__()
        self.stft_fn = self.STFTFn(filter_length, hop_length, win_length)
        n_freqs = filter_length // 2 + 1
        self.register_buffer("mel_basis", torch.zeros(n_mel_channels, n_freqs))

    def mel_spectrogram(
        self, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        magnitude, phase = self.stft_fn(y)
        energy = torch.norm(magnitude, dim=1)
        mel = torch.matmul(self.mel_basis.to(magnitude.dtype), magnitude)
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        return log_mel, magnitude, phase, energy


class LTX23VocoderCore(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        resblock_kernel_sizes: list[int] | None = None,
        upsample_rates: list[int] | None = None,
        upsample_kernel_sizes: list[int] | None = None,
        resblock_dilation_sizes: list[list[int]] | None = None,
        upsample_initial_channel: int = 1024,
        resblock: str = "1",
        output_sampling_rate: int = 24000,
        activation: str = "snake",
        use_tanh_at_final: bool = True,
        apply_final_activation: bool = True,
        use_bias_at_final: bool = True,
    ):
        super().__init__()
        if resblock_kernel_sizes is None:
            resblock_kernel_sizes = [3, 7, 11]
        if upsample_rates is None:
            upsample_rates = [6, 5, 2, 2, 2]
        if upsample_kernel_sizes is None:
            upsample_kernel_sizes = [16, 15, 8, 4, 4]
        if resblock_dilation_sizes is None:
            resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

        self.output_sampling_rate = output_sampling_rate
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.use_tanh_at_final = use_tanh_at_final
        self.apply_final_activation = apply_final_activation
        self.is_amp = resblock == "AMP1"

        self.conv_pre = nn.Conv1d(
            in_channels=128,
            out_channels=upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3,
        )
        self.ups = nn.ModuleList(
            nn.ConvTranspose1d(
                upsample_initial_channel // (2**i),
                upsample_initial_channel // (2 ** (i + 1)),
                kernel_size,
                stride,
                padding=(kernel_size - stride) // 2,
            )
            for i, (stride, kernel_size) in enumerate(
                zip(upsample_rates, upsample_kernel_sizes, strict=True)
            )
        )

        final_channels = upsample_initial_channel // (2 ** len(upsample_rates))
        self.resblocks = nn.ModuleList()
        for i in range(len(upsample_rates)):
            channels = upsample_initial_channel // (2 ** (i + 1))
            for kernel_size, dilations in zip(
                resblock_kernel_sizes, resblock_dilation_sizes, strict=True
            ):
                if self.is_amp:
                    self.resblocks.append(
                        AMPBlock1(
                            channels,
                            kernel_size,
                            tuple(dilations),
                            activation=activation,
                        )
                    )
                else:
                    self.resblocks.append(
                        ResBlock(
                            channels,
                            kernel_size=kernel_size,
                            dilations=tuple(dilations),
                            leaky_relu_negative_slope=LRELU_SLOPE,
                            padding_mode=get_padding(kernel_size, 1),
                        )
                    )

        self.act_post = (
            Activation1d(SnakeBeta(final_channels))
            if self.is_amp
            else nn.LeakyReLU()
        )
        self.conv_post = nn.Conv1d(
            in_channels=final_channels,
            out_channels=2,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=use_bias_at_final,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(2, 3)
        if x.dim() == 4:
            assert x.shape[1] == 2, "Input must have 2 channels for stereo"
            x = einops.rearrange(x, "b s c t -> b (s c) t")

        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            if not self.is_amp:
                x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            start = i * self.num_kernels
            end = start + self.num_kernels
            block_outputs = torch.stack(
                [self.resblocks[idx](x) for idx in range(start, end)],
                dim=0,
            )
            x = block_outputs.mean(dim=0)

        x = self.act_post(x)
        x = self.conv_post(x)
        if self.apply_final_activation:
            x = torch.tanh(x) if self.use_tanh_at_final else torch.clamp(x, -1, 1)
        return x


class LTX2Vocoder(ABC, nn.Module):
    r"""
    LTX 2.0 vocoder for converting generated mel spectrograms back to audio waveforms.
    """

    def __init__(
        self,
        config: LTXVocoderConfig,
    ):
        super().__init__()
        self.config = config
        nested_vocoder_cfg = getattr(config.arch_config, "vocoder", None)
        if isinstance(nested_vocoder_cfg, dict) and "bwe" in nested_vocoder_cfg:
            vocoder_cfg = nested_vocoder_cfg.get("vocoder", {})
            bwe_cfg = nested_vocoder_cfg["bwe"]
            self.vocoder = LTX23VocoderCore(
                resblock_kernel_sizes=vocoder_cfg.get("resblock_kernel_sizes"),
                upsample_rates=vocoder_cfg.get("upsample_rates"),
                upsample_kernel_sizes=vocoder_cfg.get("upsample_kernel_sizes"),
                resblock_dilation_sizes=vocoder_cfg.get("resblock_dilation_sizes"),
                upsample_initial_channel=vocoder_cfg.get(
                    "upsample_initial_channel", 1024
                ),
                resblock=vocoder_cfg.get("resblock", "1"),
                output_sampling_rate=bwe_cfg["input_sampling_rate"],
                activation=vocoder_cfg.get("activation", "snake"),
                use_tanh_at_final=vocoder_cfg.get("use_tanh_at_final", True),
                apply_final_activation=vocoder_cfg.get("apply_final_activation", True),
                use_bias_at_final=vocoder_cfg.get("use_bias_at_final", True),
            )
            self.bwe_generator = LTX23VocoderCore(
                resblock_kernel_sizes=bwe_cfg.get("resblock_kernel_sizes"),
                upsample_rates=bwe_cfg.get("upsample_rates"),
                upsample_kernel_sizes=bwe_cfg.get("upsample_kernel_sizes"),
                resblock_dilation_sizes=bwe_cfg.get("resblock_dilation_sizes"),
                upsample_initial_channel=bwe_cfg.get("upsample_initial_channel", 1024),
                resblock=bwe_cfg.get("resblock", "1"),
                output_sampling_rate=bwe_cfg["output_sampling_rate"],
                activation=bwe_cfg.get("activation", "snake"),
                use_tanh_at_final=bwe_cfg.get("use_tanh_at_final", True),
                apply_final_activation=bwe_cfg.get("apply_final_activation", True),
                use_bias_at_final=bwe_cfg.get("use_bias_at_final", True),
            )
            self.mel_stft = LTX23MelSTFT(
                filter_length=bwe_cfg["n_fft"],
                hop_length=bwe_cfg["hop_length"],
                win_length=bwe_cfg.get("win_size", bwe_cfg["n_fft"]),
                n_mel_channels=bwe_cfg["num_mels"],
            )
            self.input_sampling_rate = bwe_cfg["input_sampling_rate"]
            self.output_sampling_rate = bwe_cfg["output_sampling_rate"]
            self.hop_length = bwe_cfg["hop_length"]
            with torch.device("cpu"):
                self.resampler = UpSample1d(
                    ratio=self.output_sampling_rate // self.input_sampling_rate,
                    persistent=False,
                    window_type="hann",
                )
            self.sample_rate = self.output_sampling_rate
            return

        self.sample_rate = (
            getattr(config.arch_config, "sample_rate", None)
            or getattr(config.arch_config, "sampling_rate", None)
            or getattr(config.arch_config, "audio_sample_rate", None)
            or getattr(config.arch_config, "output_sampling_rate", None)
        )

        in_channels = config.arch_config.in_channels
        hidden_channels = config.arch_config.hidden_channels
        out_channels = config.arch_config.out_channels
        upsample_kernel_sizes = config.arch_config.upsample_kernel_sizes
        upsample_factors = config.arch_config.upsample_factors
        resnet_kernel_sizes = config.arch_config.resnet_kernel_sizes
        resnet_dilations = config.arch_config.resnet_dilations
        leaky_relu_negative_slope = config.arch_config.leaky_relu_negative_slope

        self.num_upsample_layers = len(upsample_kernel_sizes)
        self.resnets_per_upsample = len(resnet_kernel_sizes)
        self.out_channels = out_channels
        self.total_upsample_factor = math.prod(upsample_factors)
        self.negative_slope = leaky_relu_negative_slope

        if self.num_upsample_layers != len(upsample_factors):
            raise ValueError(
                f"`upsample_kernel_sizes` and `upsample_factors` should be lists of the same length but are length"
                f" {self.num_upsample_layers} and {len(upsample_factors)}, respectively."
            )

        if self.resnets_per_upsample != len(resnet_dilations):
            raise ValueError(
                f"`resnet_kernel_sizes` and `resnet_dilations` should be lists of the same length but are length"
                f" {len(self.resnets_per_upsample)} and {len(resnet_dilations)}, respectively."
            )

        self.conv_in = nn.Conv1d(
            in_channels, hidden_channels, kernel_size=7, stride=1, padding=3
        )

        self.upsamplers = nn.ModuleList()
        self.resnets = nn.ModuleList()
        input_channels = hidden_channels
        for i, (stride, kernel_size) in enumerate(
            zip(upsample_factors, upsample_kernel_sizes)
        ):
            output_channels = input_channels // 2
            self.upsamplers.append(
                nn.ConvTranspose1d(
                    input_channels,  # hidden_channels // (2 ** i)
                    output_channels,  # hidden_channels // (2 ** (i + 1))
                    kernel_size,
                    stride=stride,
                    padding=(kernel_size - stride) // 2,
                )
            )

            for kernel_size, dilations in zip(resnet_kernel_sizes, resnet_dilations):
                self.resnets.append(
                    ResBlock(
                        output_channels,
                        kernel_size,
                        dilations=dilations,
                        leaky_relu_negative_slope=leaky_relu_negative_slope,
                    )
                )
            input_channels = output_channels

        self.conv_out = nn.Conv1d(output_channels, out_channels, 7, stride=1, padding=3)

    def _compute_ltx23_mel(self, audio: torch.Tensor) -> torch.Tensor:
        batch, channels, _ = audio.shape
        flat = audio.reshape(batch * channels, -1)
        mel, _, _, _ = self.mel_stft.mel_spectrogram(flat)
        return mel.reshape(batch, channels, mel.shape[1], mel.shape[2])

    def forward(
        self, hidden_states: torch.Tensor, time_last: bool = False
    ) -> torch.Tensor:
        r"""
        Forward pass of the vocoder.

        Args:
            hidden_states (`torch.Tensor`):
                Input Mel spectrogram tensor of shape `(batch_size, num_channels, time, num_mel_bins)` if `time_last`
                is `False` (the default) or shape `(batch_size, num_channels, num_mel_bins, time)` if `time_last` is
                `True`.
            time_last (`bool`, *optional*, defaults to `False`):
                Whether the last dimension of the input is the time/frame dimension or the Mel bins dimension.

        Returns:
            `torch.Tensor`:
                Audio waveform tensor of shape (batch_size, out_channels, audio_length)
        """
        if hasattr(self, "bwe_generator"):
            input_dtype = hidden_states.dtype
            autocast_ctx = (
                torch.autocast(
                    device_type=hidden_states.device.type, dtype=torch.float32
                )
                if hidden_states.device.type != "cpu"
                else nullcontext()
            )
            with autocast_ctx:
                waveform = self.vocoder(hidden_states.float())
                length_low_rate = waveform.shape[-1]
                output_length = (
                    length_low_rate * self.output_sampling_rate // self.input_sampling_rate
                )
                remainder = length_low_rate % self.hop_length
                if remainder != 0:
                    waveform = F.pad(waveform, (0, self.hop_length - remainder))
                mel = self._compute_ltx23_mel(waveform)
                residual = self.bwe_generator(mel.transpose(2, 3))
                skip = self.resampler(waveform)
                assert residual.shape == skip.shape
                waveform = torch.clamp(residual + skip, -1, 1)[..., :output_length]
            return waveform.to(input_dtype)

        # Ensure that the time/frame dimension is last
        if not time_last:
            hidden_states = hidden_states.transpose(2, 3)
        # Combine channels and frequency (mel bins) dimensions
        hidden_states = hidden_states.flatten(1, 2)

        hidden_states = self.conv_in(hidden_states)

        for i in range(self.num_upsample_layers):
            hidden_states = F.leaky_relu(
                hidden_states, negative_slope=self.negative_slope
            )
            hidden_states = self.upsamplers[i](hidden_states)

            # Run all resnets in parallel on hidden_states
            start = i * self.resnets_per_upsample
            end = (i + 1) * self.resnets_per_upsample
            resnet_outputs = torch.stack(
                [self.resnets[j](hidden_states) for j in range(start, end)], dim=0
            )

            hidden_states = torch.mean(resnet_outputs, dim=0)

        # NOTE: unlike the first leaky ReLU, this leaky ReLU is set to use the default F.leaky_relu negative slope of
        # 0.01 (whereas the others usually use a slope of 0.1). Not sure if this is intended
        hidden_states = F.leaky_relu(hidden_states, negative_slope=0.01)
        hidden_states = self.conv_out(hidden_states)
        hidden_states = torch.tanh(hidden_states)

        return hidden_states


EntryClass = LTX2Vocoder
