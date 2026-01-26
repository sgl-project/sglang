# Copied and adapted from: https://github.com/descriptinc/descript-audio-codec

# SPDX-License-Identifier: MIT

import math
from bisect import bisect_right
from typing import Union

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from sglang.multimodal_gen.configs.models.vaes.dac import DacVAEConfig
from sglang.multimodal_gen.runtime.models.vaes.common import (
    DiagonalGaussianDistribution,
)


# Scripting this brings model speed up 1.4x
@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


class VectorQuantize(nn.Module):
    """
    Implementation of VQ similar to Karpathy's repo:
    https://github.com/karpathy/deep-vector-quantization
    Additionally uses following tricks from Improved VQGAN
    (https://arxiv.org/pdf/2110.04627.pdf):
        1. Factorized codes: Perform nearest neighbor lookup in low-dimensional space
            for improved codebook usage
        2. l2-normalized codes: Converts euclidean distance to cosine similarity which
            improves training stability
    """

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.in_proj = nn.Conv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = nn.Conv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, z):
        """Quantize the input tensor using a fixed codebook and return the corresponding codebook vectors.

        Args:
            z (torch.Tensor): Input tensor with shape ``[B, D, T]``.

        Returns:
            tuple: A tuple containing:
                - z_q (torch.Tensor): Quantized continuous representation with shape ``[B, D, T]``.
                - commitment_loss (torch.Tensor): Commitment loss scalar to train encoder to predict
                  vectors closer to codebook entries.
                - codebook_loss (torch.Tensor): Codebook loss scalar to update the codebook.
                - indices (torch.Tensor): Codebook indices (quantized discrete representation) with shape ``[B, T]``.
                - z_e (torch.Tensor): Projected latents (continuous representation before quantization) with shape ``[B, D, T]``.
        """

        # Factorized codes (ViT-VQGAN) Project input into low-dimensional space
        z_e = self.in_proj(z)  # z_e : (B x D x T)
        z_q, indices = self.decode_latents(z_e)

        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])

        z_q = (
            z_e + (z_q - z_e).detach()
        )  # noop in forward pass, straight-through gradient estimator in backward pass

        z_q = self.out_proj(z_q)

        return z_q, commitment_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight  # codebook: (N x D)

        # L2 normalize encodings and codebook (ViT-VQGAN)
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # Compute euclidean distance with codebook
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)
        return z_q, indices


class ResidualVectorQuantize(nn.Module):
    """
    Introduced in SoundStream: An end2end neural audio codec
    https://arxiv.org/abs/2107.03312
    """

    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.0,
    ):
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]

        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        dim_offsets = [0]
        for dim in self.codebook_dim:
            dim_offsets.append(dim_offsets[-1] + dim)
        self._codebook_dim_offsets = tuple(dim_offsets)

        self.quantizers = nn.ModuleList(
            [
                VectorQuantize(input_dim, codebook_size, codebook_dim[i])
                for i in range(n_codebooks)
            ]
        )
        self.quantizer_dropout = quantizer_dropout

    def forward(self, z, n_quantizers: int = None):
        """Quantize the input tensor using a fixed set of codebooks and return the corresponding codebook vectors.

        Args:
            z (torch.Tensor): Input tensor with shape ``[B, D, T]``.
            n_quantizers (int, optional): Number of quantizers to use. If ``None``,
                all quantizers are used. When ``n_quantizers`` < ``self.n_codebooks``,
                quantizer dropout is applied. Note: if ``self.quantizer_dropout`` > 0
                and in training mode, this argument is ignored and a random number of
                quantizers is used.

        Returns:
            tuple: A tuple containing:
                - z_q (torch.Tensor): Quantized continuous representation with shape ``[B, D, T]``.
                - codes (torch.Tensor): Codebook indices for each codebook with shape ``[B, N, T]``
                  (quantized discrete representation of input).
                - latents (torch.Tensor): Projected latents with shape ``[B, N*D, T]``
                  (continuous representation before quantization).
                - commitment_loss (torch.Tensor): Commitment loss scalar to train encoder to predict
                  vectors closer to codebook entries.
                - codebook_loss (torch.Tensor): Codebook loss scalar to update the codebook.
        """
        z_q = 0
        residual = z
        commitment_loss = 0
        codebook_loss = 0

        codebook_indices = []
        latents = []

        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        quantizers = self.quantizers
        if self.training:
            batch_size = z.shape[0]
            device = z.device
            n_quantizers = torch.full(
                (batch_size,),
                self.n_codebooks + 1,
                device=device,
                dtype=torch.long,
            )
            if self.quantizer_dropout > 0:
                dropout = torch.randint(
                    1,
                    self.n_codebooks + 1,
                    (batch_size,),
                    device=device,
                )
                n_dropout = int(batch_size * self.quantizer_dropout)
                if n_dropout > 0:
                    n_quantizers[:n_dropout] = dropout[:n_dropout]

            for i, quantizer in enumerate(quantizers):
                z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(
                    residual
                )

                # Create mask to apply quantizer dropout
                mask = i < n_quantizers
                z_q = z_q + z_q_i * mask[:, None, None]
                residual = residual - z_q_i

                # Sum losses
                commitment_loss += (commitment_loss_i * mask).mean()
                codebook_loss += (codebook_loss_i * mask).mean()

                codebook_indices.append(indices_i)
                latents.append(z_e_i)
        else:
            for i, quantizer in enumerate(quantizers):
                if i >= n_quantizers:
                    break
                z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(
                    residual
                )
                z_q = z_q + z_q_i
                residual = residual - z_q_i

                commitment_loss += commitment_loss_i.mean()
                codebook_loss += codebook_loss_i.mean()

                codebook_indices.append(indices_i)
                latents.append(z_e_i)

        codes = torch.stack(codebook_indices, dim=1)
        latents = torch.cat(latents, dim=1)

        return z_q, codes, latents, commitment_loss, codebook_loss

    def from_codes(self, codes: torch.Tensor):
        """Reconstruct the continuous representation from quantized codes.

        Args:
            codes (torch.Tensor): Quantized discrete representation with shape ``[B, N, T]``.

        Returns:
            tuple: A tuple containing:
                - z_q (torch.Tensor): Quantized continuous representation with shape ``[B, D, T]``.
                - z_p (torch.Tensor): Concatenated latent space representation with shape ``[B, N*D, T]``.
                - codes (torch.Tensor): Original input codebook indices with shape ``[B, N, T]``.
        """
        z_q = 0.0
        z_p = []
        n_codebooks = codes.shape[1]
        for i in range(n_codebooks):
            z_p_i = self.quantizers[i].decode_code(codes[:, i, :])
            z_p.append(z_p_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i
        return z_q, torch.cat(z_p, dim=1), codes

    def from_latents(self, latents: torch.Tensor):
        """Reconstruct the continuous representation from unquantized latents.

        Args:
            latents (torch.Tensor): Continuous representation after projection with shape ``[B, N*D, T]``.

        Returns:
            tuple: A tuple containing:
                - z_q (torch.Tensor): Quantized representation of full-projected space with shape ``[B, D, T]``.
                - z_p (torch.Tensor): Quantized representation of latent space with shape ``[B, N*D, T]``.
                - codes (torch.Tensor): Codebook indices with shape ``[B, N, T]``.
        """
        z_q = 0
        z_p = []
        codes = []
        dims = self._codebook_dim_offsets
        n_codebooks = bisect_right(dims, latents.shape[1]) - 1
        for i in range(n_codebooks):
            j, k = dims[i], dims[i + 1]
            z_p_i, codes_i = self.quantizers[i].decode_latents(latents[:, j:k, :])
            z_p.append(z_p_i)
            codes.append(codes_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i

        return z_q, torch.cat(z_p, dim=1), torch.stack(codes, dim=1)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            nn.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            nn.Conv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            nn.Conv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
    ):
        super().__init__()
        # Create first convolution
        self.block = [nn.Conv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]

        # Create last convolution
        self.block += [
            Snake1d(d_model),
            nn.Conv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            nn.ConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
    ):
        super().__init__()

        # Add first conv layer
        layers = [nn.Conv1d(input_channel, channels, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride)]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            nn.Conv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DAC(nn.Module):
    def __init__(
        self,
        config: DacVAEConfig,
    ):
        super().__init__()

        self.continuous = config.continuous
        self.decoder_dim = config.decoder_dim
        self.decoder_rates = config.decoder_rates
        self.encoder_dim = config.encoder_dim
        self.encoder_rates = config.encoder_rates
        self.hop_length = math.prod(config.encoder_rates)
        self.sample_rate = config.sample_rate

        if config.latent_dim is None:
            latent_dim = config.encoder_dim * (2 ** len(config.encoder_rates))
        else:
            latent_dim = config.latent_dim

        self.latent_dim = latent_dim

        if config.load_encoder:
            self.encoder = Encoder(config.encoder_dim, config.encoder_rates, latent_dim)

        if not config.continuous:
            self.n_codebooks = config.n_codebooks
            self.codebook_size = config.codebook_size
            self.codebook_dim = config.codebook_dim
            self.quantizer = ResidualVectorQuantize(
                input_dim=latent_dim,
                n_codebooks=config.n_codebooks,
                codebook_size=config.codebook_size,
                codebook_dim=config.codebook_dim,
                quantizer_dropout=config.quantizer_dropout,
            )
        else:
            self.quant_conv = torch.nn.Conv1d(latent_dim, 2 * latent_dim, 1)
            self.post_quant_conv = torch.nn.Conv1d(latent_dim, latent_dim, 1)

        if config.load_decoder:
            self.decoder = Decoder(
                latent_dim,
                config.decoder_dim,
                config.decoder_rates,
            )

        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv1d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))

        return audio_data

    def encode(
        self,
        audio_data: torch.Tensor,
        n_quantizers: int = None,
    ):
        """Encode audio data into latent representations.

        This method processes audio through the encoder network and optionally applies
        vector quantization (in VQ mode) or projects to a Gaussian distribution (in
        continuous mode) to produce latent representations.

        Args:
            audio_data (torch.Tensor): Audio data to encode, with shape ``[B, 1, T]``.
            n_quantizers (int, optional): Number of quantizers to use. If ``None``,
                all quantizers are used. Only applicable in VQ mode (``continuous=False``).

        Returns:
            tuple: A tuple containing:
                - z (torch.Tensor): Encoded representation. In VQ mode, this is the
                  quantized continuous representation with shape ``[B, D, T]``. In
                  continuous mode, this is a ``DiagonalGaussianDistribution`` object.
                - codes (torch.Tensor or None): Codebook indices with shape ``[B, N, T]``
                  in VQ mode, ``None`` in continuous mode.
                - latents (torch.Tensor or None): Projected latents with shape ``[B, N*D, T]``
                  in VQ mode, ``None`` in continuous mode.
                - commitment_loss (torch.Tensor): Commitment loss scalar.
                - codebook_loss (torch.Tensor): Codebook loss scalar.

        Note:
            In continuous mode, the encoded representation is projected through a
            quantization convolution layer and wrapped in a ``DiagonalGaussianDistribution``
            for VAE training.
        """
        z = self.encoder(audio_data)  # [B x D x T]
        if not self.continuous:
            z, codes, latents, commitment_loss, codebook_loss = self.quantizer(
                z, n_quantizers
            )
        else:
            z = self.quant_conv(z)  # [B x 2D x T]
            z = DiagonalGaussianDistribution(z)
            codes, latents, commitment_loss, codebook_loss = None, None, 0, 0

        return z, codes, latents, commitment_loss, codebook_loss

    def decode(self, z: torch.Tensor):
        """Decode latent representations back to audio waveforms.

        This method takes latent representations (either quantized from VQ mode or sampled
        from the posterior in continuous mode) and reconstructs the corresponding audio
        through the decoder network.

        Args:
            z (torch.Tensor): Latent representation to decode, with shape ``[B, D, T]``.
                In VQ mode (``continuous=False``), this is the quantized continuous
                representation. In continuous mode (``continuous=True``), this is sampled
                from the posterior distribution.

        Returns:
            torch.Tensor: Decoded audio data with shape ``[B, 1, T']``. The output length
            T' is determined by the decoder's upsampling rates and may differ from the
            input temporal dimension T.

        Note:
            In continuous mode (``continuous=True``), the input is first passed through
            a post-quantization convolution layer before being fed to the decoder.
        """
        if not self.continuous:
            audio = self.decoder(z)
        else:
            z = self.post_quant_conv(z)
            audio = self.decoder(z)

        return audio

    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: int = None,
        n_quantizers: int = None,
    ):
        """Model forward pass.

        Args:
            audio_data (torch.Tensor): Audio to encode, shape [B, 1, T].
            sample_rate (int, optional): Sample rate in Hz. Defaults to
                ``self.sample_rate`` when ``None``.
            n_quantizers (int, optional): Number of quantizers to use. When ``None``,
                all quantizers are used. Only used in VQ mode (``continuous=False``).

        Returns:
            dict: A dictionary containing different keys depending on the mode:

            **VQ Mode (``continuous=False``):**
                - "audio" (torch.Tensor): Decoded audio, shape [B, 1, length].
                - "z" (torch.Tensor): Quantized continuous representation, shape [B, D, T].
                - "codes" (torch.Tensor): Codebook indices, shape [B, N, T].
                - "latents" (torch.Tensor): Projected latents, shape [B, N*D, T].
                - "vq/commitment_loss" (torch.Tensor): Commitment loss.
                - "vq/codebook_loss" (torch.Tensor): Codebook loss.

            **Continuous Mode (``continuous=True``):**
                - "audio" (torch.Tensor): Decoded audio, shape [B, 1, length].
                - "z" (torch.Tensor): Latent representation, shape [B, D, T].
                - "kl_loss" (torch.Tensor): KL divergence loss (for VAE training).
        """
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        if not self.continuous:
            z, codes, latents, commitment_loss, codebook_loss = self.encode(
                audio_data, n_quantizers
            )

            x = self.decode(z)
            return {
                "audio": x[..., :length],
                "z": z,
                "codes": codes,
                "latents": latents,
                "vq/commitment_loss": commitment_loss,
                "vq/codebook_loss": codebook_loss,
            }
        else:
            posterior, _, _, _, _ = self.encode(audio_data, n_quantizers)
            z = posterior.sample()
            x = self.decode(z)

            kl_loss = posterior.kl(dims=(1, 2))
            kl_loss = kl_loss.mean()

            return {
                "audio": x[..., :length],
                "z": z,
                "kl_loss": kl_loss,
            }


EntryClass = DAC
