# Modified for SGLang; see this directory's README.md for upstream source.

import math

import torch
import torch.nn as nn


def modulate(x, shift, scale=None):
    if shift is None:
        return x * (1 + scale)
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element. These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.to(self.mlp[0].weight.dtype))
        return t_emb


class ResBlock(nn.Module):
    def __init__(self, channels, mlp_ratio=1.0):
        super().__init__()
        self.channels = channels
        self.intermediate_size = int(channels * mlp_ratio)

        self.in_ln = nn.LayerNorm(self.channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.channels, self.intermediate_size),
            nn.SiLU(),
            nn.Linear(self.intermediate_size, self.channels),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FlowMatchingHead(nn.Module):
    def __init__(self, input_dim, out_dim, dim=1536, layers=12, mlp_ratio=1.0):
        super(FlowMatchingHead, self).__init__()
        self.net = SimpleMLPAdaLN(
            input_dim=input_dim,
            out_dim=out_dim,
            dim=dim,
            layers=layers,
            mlp_ratio=mlp_ratio,
        )

    @property
    def dtype(self):
        return self.net.input_proj.weight.dtype

    @property
    def device(self):
        return self.net.input_proj.weight.device

    def forward(self, x, t):
        x = self.net(x, t)
        return x


class SimpleMLPAdaLN(nn.Module):
    """Timestep-conditioned MLP that maps LLM hidden states to flow-matching targets."""

    def __init__(self, input_dim, out_dim, dim=1536, layers=12, mlp_ratio=1.0):
        super().__init__()
        self.time_embed = TimestepEmbedder(dim)
        self.input_proj = nn.Linear(input_dim, dim)
        self.res_blocks = nn.ModuleList(
            [ResBlock(dim, mlp_ratio) for _ in range(layers)]
        )
        self.final_layer = FinalLayer(dim, out_dim)

    def forward(self, x, t):
        # x: (N, input_dim), t: (N,)
        x = self.input_proj(x)
        y = self.time_embed(t)
        for block in self.res_blocks:
            x = block(x, y)
        return self.final_layer(x, y)


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """

    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            model_channels, elementwise_affine=False, eps=1e-6
        )
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class ConvDecoder(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=1024):
        super().__init__()
        # layer 1: H/32 -> H/16 (2x upscale)
        self.ps1 = nn.PixelShuffle(2)
        self.conv1 = nn.Conv2d(input_dim // 4, hidden_dim, kernel_size=3, padding=1)
        self.act1 = nn.GELU()

        # layer 2: H/16 -> H/8 (2x upscale)
        self.ps2 = nn.PixelShuffle(2)
        self.conv2 = nn.Conv2d(hidden_dim // 4, 192, kernel_size=3, padding=1)

        # layer 3: H/8 -> H (8x upscale)
        self.ps3 = nn.PixelShuffle(8)

    def forward(self, x):
        x = self.act1(self.conv1(self.ps1((x))))
        x = self.ps3(self.conv2(self.ps2((x))))
        return x
