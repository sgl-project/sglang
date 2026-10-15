# Modified for SGLang; see this directory's README.md for upstream source.

import logging
import math
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def modulate(x, shift, scale=None):
    if shift is None:
        return x * (1 + scale)
    return x * (1 + scale) + shift


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight


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


# class FinalLayer(nn.Module):

#     def __init__(self, model_channels, out_channels):
#         super().__init__()
#         self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
#         self.linear = nn.Linear(model_channels, out_channels, bias=True)
#         self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_channels, 2 * model_channels, bias=True))

#     def forward(self, x, c):
#         shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
#         x = modulate(self.norm_final(x), shift, scale)
#         x = self.linear(x)
#         return x

# class SimpleMLPAdaLN(nn.Module):

#     def __init__(self, input_dim, out_dim, dim=1536, layers=12, mlp_ratio=1.0):
#         super().__init__()
#         self.input_dim = input_dim
#         self.out_dim = out_dim
#         self.dim = dim
#         self.layers = layers
#         self.mlp_ratio = mlp_ratio

#         self.time_embed = TimestepEmbedder(dim)
#         self.input_proj = nn.Linear(input_dim, dim)

#         res_blocks = []
#         for _ in range(layers):
#             res_blocks.append(ResBlock(dim, mlp_ratio))
#         self.res_blocks = nn.ModuleList(res_blocks)

#         self.final_layer = FinalLayer(dim, out_dim)

#         self.grad_checkpointing = False

#         self.initialize_weights()

#     def initialize_weights(self):
#         def _basic_init(module):
#             if isinstance(module, nn.Linear):
#                 torch.nn.init.xavier_uniform_(module.weight)
#                 if module.bias is not None:
#                     nn.init.constant_(module.bias, 0)

#         self.apply(_basic_init)

#         # Initialize timestep embedding MLP
#         nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
#         nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

#         # Zero-out adaLN modulation layers
#         for block in self.res_blocks:
#             nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
#             nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

#         # Zero-out output layers
#         nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
#         nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
#         nn.init.constant_(self.final_layer.linear.weight, 0)
#         nn.init.constant_(self.final_layer.linear.bias, 0)

#     def forward(self, x, t):
#         """
#         x.shape = (bsz, input_dim)
#         t.shape = (bsz,)
#         """

#         x = self.input_proj(x)
#         t = self.time_embed(t)

#         y = t

#         for block in self.res_blocks:
#             if self.grad_checkpointing and self.training:
#                 x = checkpoint(block, x, y, use_reentrant=True)
#             else:
#                 x = block(x, y)

#         return self.final_layer(x, y)


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


def precompute_freqs_cis_2d(
    dim: int, height: int, width: int, theta: float = 10000.0, scale=16.0
):
    # assert  H * H == end
    # flat_patch_pos = torch.linspace(-1, 1, end) # N = end
    x_pos = torch.linspace(0, scale, width)
    y_pos = torch.linspace(0, scale, height)
    y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing="ij")
    y_pos = y_pos.reshape(-1)
    x_pos = x_pos.reshape(-1)
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim)
    )  # Hc/4
    x_freqs = torch.outer(x_pos, freqs).float()  # N Hc/4
    y_freqs = torch.outer(y_pos, freqs).float()  # N Hc/4
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    freqs_cis = torch.cat(
        [x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1
    )  # N,Hc/4,2
    freqs_cis = freqs_cis.reshape(height * width, -1)
    return freqs_cis


class NerfEmbedder(nn.Module):
    def __init__(self, in_channels, hidden_size_input, max_freqs):
        super().__init__()
        self.max_freqs = max_freqs
        self.hidden_size_input = hidden_size_input
        self.embedder = nn.Sequential(
            nn.Linear(in_channels + max_freqs**2, hidden_size_input, bias=True),
        )

    @lru_cache
    def fetch_pos(self, patch_size, device, dtype):
        pos = precompute_freqs_cis_2d(
            self.max_freqs**2 * 2, patch_size, patch_size
        ).real
        pos = pos[None, :, :].to(device=device, dtype=dtype)
        return pos

    def forward(self, inputs):
        B, P2, C = inputs.shape
        patch_size = int(P2**0.5)
        device = inputs.device
        dtype = inputs.dtype
        dct = self.fetch_pos(patch_size, device, dtype)
        dct = dct.repeat(B, 1, 1)
        inputs = torch.cat([inputs, dct], dim=-1)
        inputs = self.embedder(inputs)
        return inputs


class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        patch_size,
        grad_checkpointing=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing
        self.patch_size = patch_size

        self.cond_embed = nn.Linear(z_channels, patch_size**2 * model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(
                ResBlock(
                    model_channels,
                )
            )

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        x = self.input_proj(x)
        c = self.cond_embed(c)

        y = c.reshape(-1, self.patch_size**2, self.model_channels)

        for block in self.res_blocks:
            x = block(x, y)

        return self.final_layer(x)


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

    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        return x


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(
    embed_dim, grid_size, cls_token=False, extra_tokens=0, pe_interpolation=1.0
):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32) / pe_interpolation
    grid_w = np.arange(grid_size, dtype=np.float32) / pe_interpolation
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(
    model_path, pe_key: str = "gen_pos_embed", new_len: int = 4096
):
    state_dict = torch.load(model_path, map_location="cpu")

    pos_embed_1d = state_dict[pe_key]
    _, ori_len, embed_dim = pos_embed_1d.shape

    ori_size = int(ori_len**0.5)
    new_size = int(new_len**0.5)

    if ori_size != new_size:
        logger.info(
            "Position interpolate from %dx%d to %dx%d"
            % (ori_size, ori_size, new_size, new_size)
        )
        pos_embed_2d = pos_embed_1d.reshape(-1, ori_size, ori_size, embed_dim).permute(
            0, 3, 1, 2
        )
        pos_embed_2d = torch.nn.functional.interpolate(
            pos_embed_2d, size=(new_size, new_size), mode="bicubic", align_corners=False
        )
        pos_embed_1d = pos_embed_2d.permute(0, 2, 3, 1).flatten(1, 2)
        state_dict[pe_key] = pos_embed_1d

    torch.save(state_dict, model_path)


class PositionEmbedding(nn.Module):
    def __init__(self, max_num_patch_per_side, hidden_size):
        super().__init__()
        self.max_num_patch_per_side = max_num_patch_per_side
        self.hidden_size = hidden_size
        self.pos_embed = nn.Parameter(
            torch.zeros(max_num_patch_per_side**2, hidden_size), requires_grad=False
        )
        self._init_weights()

    def _init_weights(self):
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.hidden_size, self.max_num_patch_per_side
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

    def forward(self, position_ids):
        return self.pos_embed[position_ids]


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.block[2].weight)
        nn.init.zeros_(self.block[2].bias)

    def forward(self, x):
        return x + self.block(x)


class PostConvSmoother(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=64, num_blocks=3):
        super().__init__()
        self.in_proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(
            *[ResidualConvBlock(hidden_channels) for _ in range(num_blocks)]
        )
        self.out_proj = nn.Conv2d(hidden_channels, in_channels, kernel_size=1)

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        h = self.in_proj(x)
        h = self.blocks(h)
        return x + self.out_proj(h)


class ProgressiveConvDecoder(nn.Module):
    def __init__(self, hidden_dim=4096, out_channels=3):
        super().__init__()

        # self.proj = nn.Linear(hidden_dim, 1024)
        # self.act = nn.SiLU()

        self.up_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(hidden_dim, 512, kernel_size=3, padding=1),
                    nn.GroupNorm(32, 512),
                    nn.SiLU(),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(512, 256, kernel_size=3, padding=1),
                    nn.GroupNorm(32, 256),
                    nn.SiLU(),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(256, 64, kernel_size=3, padding=1),
                    nn.GroupNorm(32, 64),
                    nn.SiLU(),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(64, 32, kernel_size=3, padding=1),
                    nn.GroupNorm(16, 32),
                    nn.SiLU(),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(32, 16, kernel_size=3, padding=1),
                    nn.SiLU(),
                ),
            ]
        )

        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)

    def forward(self, x_2d):
        # B, C, H, W = x_2d.shape
        # x = x_2d.permute(0, 2, 3, 1).contiguous() # (B, H, W, C)
        # x = self.proj(x)
        # x = self.act(x)
        # x = x.permute(0, 3, 1, 2).contiguous()    # (B, 512, H, W)
        x = x_2d
        for block in self.up_blocks:
            x = block(x)

        out = self.out_conv(x)
        return out


class PatchDecoder_postps(nn.Module):
    def __init__(self):
        super().__init__()
        # layer 1: H/32 -> H/8 (4x upscale)

        self.conv1 = nn.Conv2d(4096, 4096, kernel_size=3, padding=1)
        self.ps1 = nn.PixelShuffle(4)
        self.act1 = nn.GELU()

        # layer 2: H/8 -> H (8x upscale)
        self.conv2 = nn.Conv2d(256, 192, kernel_size=3, padding=1)
        self.ps2 = nn.PixelShuffle(8)

    def forward(self, x):
        # x shape: [B, 4096, H/32, W/32]
        x = self.ps1(self.act1(self.conv1(x)))  # -> [B, 256, H/8, W/8]
        x = self.ps2(self.conv2(x))  # -> [B, 3, H, W]
        return x


class PatchDecoder_preps(nn.Module):
    def __init__(self):
        super().__init__()
        # layer 1: H/32 -> H/16 (2x upscale)
        self.ps1 = nn.PixelShuffle(2)
        self.conv1 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.act1 = nn.GELU()

        # layer 2: H/16 -> H/8 (2x upscale)
        self.ps2 = nn.PixelShuffle(2)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.act2 = nn.GELU()

        # layer 3: H/8 -> H (8x upscale)
        self.ps3 = nn.PixelShuffle(8)
        self.conv3 = nn.Conv2d(4, 3, kernel_size=3, padding=1)

    def forward(self, x):
        # x shape: [B, 4096, H/32, W/32]
        x = self.act1(self.conv1(self.ps1((x))))  # -> [B, 256, H/16, W/16]
        x = self.act2(self.conv2(self.ps2((x))))  # -> [B, 256, H/8, W/8]
        x = self.conv3(self.ps3((x)))  # -> [B, 3, H, W]
        return x


class PatchDecoder_preps1(nn.Module):
    def __init__(self):
        super().__init__()
        # layer 1: H/32 -> H/16 (2x upscale)
        self.ps1 = nn.PixelShuffle(2)
        self.conv1 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.act1 = nn.GELU()

        # layer 2: H/16 -> H/8 (2x upscale)
        self.ps2 = nn.PixelShuffle(2)
        self.conv2 = nn.Conv2d(256, 192, kernel_size=3, padding=1)

        # layer 3: H/8 -> H (8x upscale)
        self.ps3 = nn.PixelShuffle(8)

    def forward(self, x):
        # x shape: [B, 4096, H/32, W/32]
        x = self.act1(self.conv1(self.ps1((x))))  # -> [B, 256, H/16, W/16]
        x = self.ps3(self.conv2(self.ps2((x))))  # -> [B, 256, H/8, W/8]
        return x


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
