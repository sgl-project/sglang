# SPDX-License-Identifier: Apache-2.0
"""FLUX.3 image VAE — the FLUX.2 packed image autoencoder (``AutoEncoder``).

Ported from the BFL FLUX.3 reference (``f3.models.image_vae.autoencoder``) so
``ae.safetensors`` loads 1:1. Exposed to SGLang via :class:`Flux3ImageVAE`
(discovered through ``EntryClass``); ``encode`` maps ``(B,3,H,W)`` in ``[-1,1]``
to packed latents ``(B, 4*z_channels, H/16, W/16)``, ``decode`` inverts it.

Every ``GroupNorm -> swish`` chain runs SGLang's fused GroupNorm+SiLU kernel
(:func:`apply_group_norm_silu`): bit-close, not bit-exact, to the eager chain,
since the kernel reduces in fp32. The added ``nn.SiLU`` modules carry no
parameters, so the checkpoint still loads 1:1.
"""

import math
from dataclasses import dataclass, field

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from sglang.kernels.ops.diffusion import apply_group_norm_silu
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_decode_parallel_rank,
    get_decode_parallel_world_size,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import (
    torch_default_sdpa_backends,
)
from sglang.multimodal_gen.runtime.layers.parallel_conv import (
    SpatialParallelConv2d,
    chunk_height_by_sizes,
    gather_and_trim_height,
    gather_variable_height,
    split_for_parallel_decode,
)
from sglang.multimodal_gen.runtime.models.vaes.common import has_decode_parallel_world
from sglang.multimodal_gen.runtime.models.vaes.fast_path_gate import (
    VaeFastPathGate,
    register_vae_fast_path_gate,
)


@dataclass
class AutoEncoderParams:
    resolution: int = 256
    in_channels: int = 3
    ch: int = 128
    out_ch: int = 3
    ch_mult: list[int] = field(default_factory=lambda: [1, 2, 4, 4])
    num_res_blocks: int = 2
    z_channels: int = 32
    strided_attention: bool = True


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int, strided: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.strided = strided
        self.fast_path_gate: VaeFastPathGate | None = None

        self.norm = nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )

        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        if self.strided:

            def flat(tensor):
                return tensor.reshape(b, c, h * w).permute(0, 2, 1).unsqueeze(1)

            q, k, v = flat(q), flat(k), flat(v)
        else:
            q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
            k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
            v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        # Reference numerics: PyTorch's default SDPA backend choice.
        with torch_default_sdpa_backends():
            if (
                self.fast_path_gate is not None
                and self.fast_path_gate.enabled
                and q.is_cuda
                and q.dtype == torch.float32
                and q.shape[-2] > 16384
            ):
                # NCHW views force quadratic math SDPA. Dense head channels
                # allow the existing FP32 memory-efficient kernel, if supported.
                dense = tuple(tensor.contiguous() for tensor in (q, k, v))
                params = torch.backends.cuda.SDPAParams(*dense, None, 0.0, False, False)
                if torch.backends.cuda.can_use_efficient_attention(params):
                    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                        h_ = nn.functional.scaled_dot_product_attention(*dense)
                else:
                    del params, dense
                    h_ = nn.functional.scaled_dot_product_attention(q, k, v)
            else:
                h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = nn.GroupNorm(
            num_groups=32, num_channels=out_channels, eps=1e-6, affine=True
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )
        # apply_group_norm_silu's fused path needs an nn.SiLU (== swish).
        self.act = nn.SiLU()

    def _norm_act(self, x: Tensor, norm: nn.Module) -> Tensor:
        return apply_group_norm_silu(x, norm, self.act)

    def forward(self, x):
        h = x
        h = self.conv1(self._norm_act(h, self.norm1))
        h = self.conv2(self._norm_act(h, self.norm2))

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=2, padding=0
        )

    def forward(self, x: Tensor):
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x: Tensor):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
        strided_attention: bool = False,
    ):
        super().__init__()
        self.quant_conv = torch.nn.Conv2d(2 * z_channels, 2 * z_channels, 1)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.conv_in = nn.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in, strided_attention)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        self.norm_out = nn.GroupNorm(
            num_groups=32, num_channels=block_in, eps=1e-6, affine=True
        )
        self.conv_out = nn.Conv2d(
            block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1
        )
        self.act = nn.SiLU()

    def _norm_act(self, x: Tensor, norm: nn.Module) -> Tensor:
        return apply_group_norm_silu(x, norm, self.act)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = self.conv_out(self._norm_act(h, self.norm_out))
        h = self.quant_conv(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
        strided_attention: bool = False,
    ):
        super().__init__()
        self.post_quant_conv = torch.nn.Conv2d(z_channels, z_channels, 1)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)

        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        self.conv_in = nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in, strided_attention)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        self.norm_out = nn.GroupNorm(
            num_groups=32, num_channels=block_in, eps=1e-6, affine=True
        )
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)
        self.act = nn.SiLU()

    def _norm_act(self, x: Tensor, norm: nn.Module) -> Tensor:
        return apply_group_norm_silu(x, norm, self.act)

    def forward(self, z: Tensor) -> Tensor:
        z = self.post_quant_conv(z)

        upscale_dtype = next(self.up.parameters()).dtype

        h = self.conv_in(z)

        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        h = h.to(upscale_dtype)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.conv_out(self._norm_act(h, self.norm_out))
        return h


class AutoEncoder(nn.Module):
    def __init__(self, params: AutoEncoderParams = AutoEncoderParams()):
        super().__init__()
        self.params = params
        self.encoder = Encoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
            strided_attention=params.strided_attention,
        )
        self.decoder = Decoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            out_ch=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
            strided_attention=params.strided_attention,
        )

        self.bn_eps = 1e-4
        self.bn_momentum = 0.1
        self.ps = [2, 2]
        self.bn = torch.nn.BatchNorm2d(
            math.prod(self.ps) * params.z_channels,
            eps=self.bn_eps,
            momentum=self.bn_momentum,
            affine=False,
            track_running_stats=True,
        )
        gate = VaeFastPathGate()
        for module in self.modules():
            if isinstance(module, AttnBlock):
                module.fast_path_gate = gate
        register_vae_fast_path_gate(self, gate)

    def normalize(self, z: Tensor) -> Tensor:
        self.bn.eval()
        return self.bn(z)

    def inv_normalize(self, z: Tensor) -> Tensor:
        self.bn.eval()
        s = torch.sqrt(self.bn.running_var.view(1, -1, 1, 1) + self.bn_eps)
        m = self.bn.running_mean.view(1, -1, 1, 1)
        return z * s + m

    def encode(self, x: Tensor) -> Tensor:
        moments = self.encoder(x)
        mean = torch.chunk(moments, 2, dim=1)[0]

        z = rearrange(
            mean,
            "... c (i pi) (j pj)  -> ... (c pi pj) i j",
            pi=self.ps[0],
            pj=self.ps[1],
        )
        z = self.normalize(z)
        return z

    def decode(self, z: Tensor) -> Tensor:
        z = self.inv_normalize(z)
        z = rearrange(
            z,
            "... (c pi pj) i j -> ... c (i pi) (j pj)",
            pi=self.ps[0],
            pj=self.ps[1],
        )
        dec = self.decoder(z)
        return dec


from sglang.multimodal_gen.configs.models.vaes.flux3_image import (  # noqa: E402
    Flux3ImageVAEConfig,
)


class Flux3ImageVAE(AutoEncoder):
    """SGLang wrapper around the reference ``AutoEncoder``.

    Builds ``AutoEncoderParams`` from a ``Flux3ImageVAEConfig`` and delegates to
    ``AutoEncoder.__init__``, keeping the module tree (and ``state_dict`` keys)
    identical so ``ae.safetensors`` loads without renaming.

    When a decode-parallel group exists (world_size > 1, e.g. ``--vae-sp N``) the
    decoder is height-sharded (see :func:`enable_spatial_parallel_decoder`); on a
    single GPU this is a no-op. The encoder is never sharded.
    """

    _aliases = ["AutoEncoder", "Flux3ImageVAE"]

    def __init__(
        self,
        config: "Flux3ImageVAEConfig | None" = None,
        hf_config: dict | None = None,
        **kwargs,
    ):
        if config is not None:
            ac = config.arch_config
            params = AutoEncoderParams(
                resolution=ac.resolution,
                in_channels=ac.in_channels,
                ch=ac.ch,
                out_ch=ac.out_ch,
                ch_mult=list(ac.ch_mult),
                num_res_blocks=ac.num_res_blocks,
                z_channels=ac.z_channels,
                strided_attention=ac.strided_attention,
            )
        else:
            params = AutoEncoderParams()
        super().__init__(params)
        self.config = config
        self.hf_config = hf_config or {}
        # Runs before weight load; SpatialParallelConv2d keeps the param names.
        self._spatial_parallel = has_decode_parallel_world()
        self._sp_upsamples = (
            enable_spatial_parallel_decoder(self.decoder)
            if self._spatial_parallel
            else 0
        )

    def decode(self, z: Tensor) -> Tensor:
        if self._spatial_parallel:
            return _spatial_parallel_decode(self, z, self._sp_upsamples)
        return super().decode(z)


# --------------------------------------------------------------------------- #
# Spatial-parallel (height-sharded) decode.
#
# When a decode-parallel group exists (world_size > 1, e.g. `--vae-sp N`) the
# decoder is sharded over image height: each rank owns a height slice and its
# convs exchange halo rows (SpatialParallelConv2d), while the two *global* spatial
# ops -- GroupNorm (needs full-height statistics) and the mid self-attention
# (mixes all positions) -- gather the full height, run, and re-chunk. This mirrors
# diffusers_spatial.py but is expressed against our own fused `_norm_act` and the
# custom `AttnBlock`, using only the public `parallel_conv` primitives. When
# world_size == 1 nothing here runs -> the decoder is plain nn.Conv2d, bit-identical
# to the serial VAE. The encoder is never sharded (decode-only).


def _sp_norm_act(self, x: Tensor, norm: nn.Module) -> Tensor:
    """Spatial-parallel ``_norm_act``: gather the full height, run the fused
    GroupNorm+SiLU on the full tensor (correct global stats), then restore the
    exact original per-rank shards."""
    x, heights = gather_variable_height(x)
    x = apply_group_norm_silu(x.contiguous(), norm, self.act)
    return chunk_height_by_sizes(x, heights)


def _wrap_attn_forward(attn: AttnBlock) -> None:
    """Gather the full height around the whole ``AttnBlock`` (its GroupNorm needs
    global stats and SDPA mixes all positions), then re-chunk. Its q/k/v/proj_out
    are 1x1 convs (halo 0), so running them on the gathered tensor is a plain conv."""
    inner = attn.forward

    def forward(x: Tensor) -> Tensor:
        x, heights = gather_variable_height(x)
        x = inner(x.contiguous())
        return chunk_height_by_sizes(x, heights)

    attn.forward = forward


def _replace_conv2d_with_spatial_parallel(module: nn.Module) -> None:
    """Recursively swap every plain ``nn.Conv2d`` for the public
    :class:`SpatialParallelConv2d`, reusing the same weight/bias Parameters. Since
    ``SpatialParallelConv2d`` subclasses ``nn.Conv2d`` the ``state_dict`` keys are
    unchanged, so the checkpoint still loads 1:1."""
    for name, child in list(module.named_children()):
        if type(child) is nn.Conv2d:
            sp = SpatialParallelConv2d(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=child.bias is not None,
                padding_mode=child.padding_mode,
            )
            sp.weight = child.weight
            sp.bias = child.bias
            setattr(module, name, sp)
        else:
            _replace_conv2d_with_spatial_parallel(child)


def enable_spatial_parallel_decoder(decoder: nn.Module) -> int:
    """Install height-sharded decode on a native decoder in place, using only the
    public ``parallel_conv`` API. Returns the number of 2x spatial upsamples (used
    to size the entry chunk)."""
    _replace_conv2d_with_spatial_parallel(decoder)
    for m in decoder.modules():
        if isinstance(m, (ResnetBlock, Decoder)):
            m._norm_act = _sp_norm_act.__get__(m, type(m))
        elif isinstance(m, AttnBlock):
            _wrap_attn_forward(m)
    return sum(1 for m in decoder.modules() if isinstance(m, Upsample))


def _spatial_parallel_decode(
    vae: Flux3ImageVAE, z: Tensor, upsample_count: int
) -> Tensor:
    """Height-sharded ``AutoEncoder.decode``: de-normalize + 2x2 unshuffle on the
    full latent, split the decoder input by height, decode per rank, gather."""
    z = vae.inv_normalize(z)
    z = rearrange(
        z,
        "... (c pi pj) i j -> ... c (i pi) (j pj)",
        pi=vae.ps[0],
        pj=vae.ps[1],
    )
    z, expected_height = split_for_parallel_decode(
        z,
        upsample_count=upsample_count,
        world_size=get_decode_parallel_world_size(),
        rank=get_decode_parallel_rank(),
    )
    dec = vae.decoder(z)
    return gather_and_trim_height(dec, expected_height)


EntryClass = Flux3ImageVAE
