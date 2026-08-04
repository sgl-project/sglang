# SPDX-License-Identifier: Apache-2.0
"""CUDA fast paths for the FLUX.2 VAE decoder (AutoencoderKLFlux2, diffusers
``Decoder``) on image workloads.

All rewrites are mathematically exact re-associations of the original
operators (weight folding is done lazily in fp32 on first fast-path use and
written back to the model compute dtype). The wrappers are installed once at
VAE load and stay in place; each forward dispatches on a shared
request-scoped :class:`VaeFastPathGate`: requests with ``quality == "high"``
run the fast paths, the ``"lossless"`` default runs the original module path
bit-for-bit.

- channels_last: run the decoder in channels_last so cuDNN convolutions run
  natively in NHWC (removes the nchwToNhwc/nhwcToNchw transpose kernels
  around every conv). The parameter layout is swapped at decode entry to
  match the gate, so lossless decodes always run the NCHW baseline kernels
  bit-for-bit. The mid-block attention needs a layout-safe forward because
  diffusers' ``AttnProcessor2_0`` calls ``.view`` on the 4D activation,
  which is illegal for channels_last tensors.
- norm+SiLU: two-pass channels_last GroupNorm(+SiLU) Triton fusion (fp32
  statistics) for the ResnetBlock2D norm1/norm2 + SiLU chains and the
  decoder ``conv_norm_out``/``conv_act`` tail, which upcast to fp32 under
  autocast and dominate the decode profile.
- fused upsample: nearest-2x upsample + Conv2d(3x3, p1) ==
  ConvTranspose2d(k4, s2, p1) with a lazily-summed kernel. Removes the 4x
  upsampled intermediate materialization.
- attention V/proj fold: fold the attention output projection into the V
  projection of the single-head mid-block attention (softmax rows sum to 1,
  so ``A @ (V W_v^T + b_v) W_o^T + b_o == A @ (V W_v'^T + b')``).

Install is all-or-nothing and fail-closed: without Triton, or with any
attention block lacking the layout-safe rewrite, no wrapper is installed and
every request runs the unmodified decoder.
"""

from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

try:
    from sglang.kernels.ops.diffusion.triton.group_norm_silu_twopass import (
        group_norm_silu_4d,
        group_norm_silu_rows,
    )

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False


class VaeFastPathGate:
    """Mutable fast-path flag shared by every wrapper of one VAE.

    Published on the VAE as ``_sgl_vae_fast_path_gate``; ``DecodingStage``
    enables it for the duration of a decode when the request's ``quality``
    sampling param is ``"high"``.
    """

    __slots__ = ("enabled",)

    def __init__(self) -> None:
        self.enabled = False


GATE_ATTR = "_sgl_vae_fast_path_gate"


# ---------------------------------------------------------------------------
# Fuse A: two-pass GroupNorm(+SiLU) fusion (channels_last Triton kernel)
# ---------------------------------------------------------------------------


class FusedGroupNormSiLU(nn.Module):
    """GroupNorm + SiLU fused with the two-pass channels_last Triton kernel.

    fp32 statistics and affine/SiLU application, output in the input dtype.
    Falls back to the original module chain (norm + F.silu, bit-identical to
    the original norm + nn.SiLU pair) for unsupported inputs and whenever
    the fast-path gate is disabled.
    """

    def __init__(self, norm: nn.GroupNorm, gate: VaeFastPathGate) -> None:
        super().__init__()
        # Keep the original GroupNorm state-dict layout (``weight``/``bias``)
        # so checkpoint loading and component-accuracy weight transfer do not
        # see wrapper-specific ``norm.*`` parameter names.
        self.num_groups = norm.num_groups
        self.num_channels = norm.num_channels
        self.eps = norm.eps
        self.affine = norm.affine
        self.weight = norm.weight
        self.bias = norm.bias
        self._sgl_gate = gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._sgl_gate.enabled and x.dim() == 4:
            y = group_norm_silu_4d(
                x,
                self.weight,
                self.bias,
                self.num_groups,
                self.eps,
                apply_silu=True,
            )
            if y is not None:
                return y
        return F.silu(
            F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        )


def _install_norm_silu(decoder, resnet_cls, gate: VaeFastPathGate) -> int:
    def _fusable(norm) -> bool:
        return (
            type(norm) is nn.GroupNorm
            and norm.affine
            and norm.weight is not None
            and norm.bias is not None
        )

    count = 0
    for m in decoder.modules():
        if (
            type(m) is resnet_cls
            and m.time_emb_proj is None
            # "default"/"group" apply plain norm2 (+ SiLU); "scale_shift" and
            # "spatial" modify the activation between norm2 and SiLU.
            and m.time_embedding_norm in ("default", "group")
            and m.upsample is None
            and m.downsample is None
            and isinstance(m.nonlinearity, nn.SiLU)
            and _fusable(m.norm1)
            and _fusable(m.norm2)
        ):
            m.norm1 = FusedGroupNormSiLU(m.norm1, gate)
            m.norm2 = FusedGroupNormSiLU(m.norm2, gate)
            m.nonlinearity = nn.Identity()
            count += 2
    if (
        type(getattr(decoder, "conv_norm_out", None)) is nn.GroupNorm
        and isinstance(getattr(decoder, "conv_act", None), nn.SiLU)
        and _fusable(decoder.conv_norm_out)
    ):
        decoder.conv_norm_out = FusedGroupNormSiLU(decoder.conv_norm_out, gate)
        decoder.conv_act = nn.Identity()
        count += 1
    return count


# ---------------------------------------------------------------------------
# Fuse B: nearest-2x upsample + Conv2d(3x3, p1) == ConvTranspose2d(k4, s2, p1)
# ---------------------------------------------------------------------------

# Which 3x3 conv taps sum into each 4x4 transposed-conv tap (per spatial axis).
_UPSAMPLE_TAP_MAP = {0: (2,), 1: (1, 2), 2: (0, 1), 3: (0,)}


def _fold_upsample2x_conv2d_weight(conv: nn.Conv2d) -> torch.Tensor:
    """Sum the 3x3 conv taps into the equivalent ConvTranspose2d(k4) kernel."""
    w = conv.weight.detach().float()  # [Cout, Cin, 3, 3]
    cout, cin = w.shape[:2]
    wt = w.new_zeros(cin, cout, 4, 4)  # ConvTranspose2d layout
    for a in range(4):
        for b in range(4):
            acc = w.new_zeros(cout, cin)
            for i in _UPSAMPLE_TAP_MAP[a]:
                for j in _UPSAMPLE_TAP_MAP[b]:
                    acc += w[:, :, i, j]
            wt[:, :, a, b] = acc.t()
    wt = wt.to(conv.weight.dtype)
    if conv.weight.is_contiguous(memory_format=torch.channels_last):
        wt = wt.contiguous(memory_format=torch.channels_last)
    return wt.to(conv.weight.device)


class FusedUpsample2xConv2d(nn.Module):
    """ConvTranspose2d(k4, s2, p1) equivalent of diffusers Upsample2D
    (nearest-2x interpolate + Conv2d(3x3, p1)).

    nearest 2x upsampling is pure pixel replication (no arithmetic), so the
    fusion only re-associates the conv taps; the kernel is summed lazily in
    fp32 on first fast-path use and written back to the conv dtype. With the
    fast-path gate disabled the original Upsample2D runs bit-for-bit.
    """

    def __init__(self, upsample: nn.Module, gate: VaeFastPathGate) -> None:
        super().__init__()
        # Keep ``conv`` registered directly on the wrapper so parameter names
        # remain ``...upsamplers.N.conv.*``. The unregistered original module
        # is retained only to run its exact lossless forward implementation.
        object.__setattr__(self, "_orig", upsample)
        self.conv = upsample.conv
        self.channels = upsample.channels
        self._sgl_gate = gate
        self._fused_weight = None

    def forward(self, hidden_states, output_size=None, *args, **kwargs):
        if (
            not self._sgl_gate.enabled
            or output_size is not None
            or hidden_states.shape[1] != self.channels
        ):
            return self._orig(hidden_states, output_size=output_size)
        conv = self.conv
        w = self._fused_weight
        if w is None:
            w = _fold_upsample2x_conv2d_weight(conv)
            self._fused_weight = w
        return F.conv_transpose2d(hidden_states, w, conv.bias, stride=2, padding=1)


def _install_fused_upsample(decoder, upsample_cls, gate: VaeFastPathGate) -> int:
    count = 0
    for blk in decoder.up_blocks:
        upsamplers = getattr(blk, "upsamplers", None)
        if not upsamplers:
            continue
        for i, up in enumerate(upsamplers):
            if type(up) is not upsample_cls:
                continue
            conv = getattr(up, "conv", None)
            if (
                up.use_conv
                and not up.use_conv_transpose
                and up.interpolate
                and up.norm is None
                and up.name == "conv"
                and type(conv) is nn.Conv2d
                and conv.kernel_size == (3, 3)
                and conv.stride == (1, 1)
                and conv.padding == (1, 1)
            ):
                upsamplers[i] = FusedUpsample2xConv2d(up, gate)
                count += 1
    return count


# ---------------------------------------------------------------------------
# Fuse C: layout-safe single-head attention forward with the V/proj fold
# ---------------------------------------------------------------------------


def _fold_attn_vproj(m) -> tuple[torch.Tensor, torch.Tensor]:
    w_v = m.to_v.weight.detach().float()
    b_v = m.to_v.bias.detach().float()
    w_o = m.to_out[0].weight.detach().float()
    b_o = m.to_out[0].bias.detach().float()
    dtype = m.to_v.weight.dtype
    # Softmax rows sum to 1, so the folded bias broadcasts exactly like the
    # original output-projection bias.
    return (w_o @ w_v).to(dtype).contiguous(), (w_o @ b_v + b_o).to(dtype)


def _attn_fast_forward(
    self,
    hidden_states,
    encoder_hidden_states=None,
    attention_mask=None,
    temb=None,
    **cross_attention_kwargs,
):
    if (
        not self._sgl_gate.enabled
        or encoder_hidden_states is not None
        or attention_mask is not None
        or temb is not None
        or hidden_states.ndim != 4
    ):
        # Lossless requests (and anything unexpected) run the stock
        # diffusers path; the decoder layout is NCHW in that case.
        return type(self).forward(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            temb=temb,
            **cross_attention_kwargs,
        )

    residual = hidden_states
    batch_size, channels, height, width = hidden_states.shape
    # Free view: with the gate enabled the decoder runs in channels_last.
    hs = hidden_states.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)

    if self.group_norm is not None:
        gn = self.group_norm
        y = group_norm_silu_rows(
            hs, gn.weight, gn.bias, gn.num_groups, gn.eps, apply_silu=False
        )
        hs = y if y is not None else gn(hs.transpose(1, 2)).transpose(1, 2)

    query = self.to_q(hs)
    key = self.to_k(hs)
    folded = self._sgl_folded_v
    if folded is None:
        folded = _fold_attn_vproj(self)
        self._sgl_folded_v = folded
    value = F.linear(hs, folded[0], folded[1])

    out = F.scaled_dot_product_attention(
        query.unsqueeze(1), key.unsqueeze(1), value.unsqueeze(1)
    )
    out = out.squeeze(1).to(query.dtype)
    out = self.to_out[1](out)  # dropout (identity in eval)

    # The permuted view has channels_last strides, matching the residual.
    out = out.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
    if self.residual_connection:
        out = out + residual
    return out / self.rescale_output_factor


def _attn_fast_compatible(m, attn_cls, processor_cls) -> bool:
    return (
        type(m) is attn_cls
        and isinstance(m.processor, processor_cls)
        and m.heads == 1
        and m.scale_qk
        and m.spatial_norm is None
        and not m.norm_cross
        and getattr(m, "norm_q", None) is None
        and getattr(m, "norm_k", None) is None
        and getattr(m, "add_k_proj", None) is None
        and type(m.to_q) is nn.Linear
        and type(m.to_k) is nn.Linear
        and type(m.to_v) is nn.Linear
        and type(m.to_out[0]) is nn.Linear
        and isinstance(m.to_out[1], nn.Dropout)
        # The V/proj fold needs both biases present.
        and m.to_v.bias is not None
        and m.to_out[0].bias is not None
    )


# ---------------------------------------------------------------------------
# channels_last layout dispatch (swapped at decode entry to match the gate)
# ---------------------------------------------------------------------------


def _decoder_layout_forward(self, *args, **kwargs):
    want_cl = self._sgl_gate.enabled
    if want_cl != self._sgl_channels_last:
        # Layout swaps are pure permutations of the parameter memory (values
        # are bit-identical), so flipping back to contiguous restores the
        # baseline NCHW cuDNN kernel selection exactly.
        self.to(
            memory_format=(torch.channels_last if want_cl else torch.contiguous_format)
        )
        self._sgl_channels_last = want_cl
        logger.info(
            "FLUX.2 VAE: decoder switched to %s layout.",
            "channels_last (NHWC)" if want_cl else "contiguous (NCHW)",
        )
    return type(self).forward(self, *args, **kwargs)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def maybe_optimize_flux2_vae(vae: nn.Module) -> nn.Module:
    """Install the quality-gated CUDA FLUX.2 VAE decoder fast paths."""
    from diffusers.models.attention_processor import Attention, AttnProcessor2_0
    from diffusers.models.autoencoders.vae import Decoder
    from diffusers.models.resnet import ResnetBlock2D
    from diffusers.models.upsampling import Upsample2D

    from sglang.multimodal_gen.runtime.models.vaes.autoencoder_kl_flux2 import (
        AutoencoderKLFlux2,
    )

    if not isinstance(vae, AutoencoderKLFlux2) or type(vae.decoder) is not Decoder:
        return vae
    if getattr(vae, "_spatial_parallel_decode_enabled", False):
        logger.info(
            "FLUX.2 VAE: spatial-parallel decode enabled; "
            "skipping CUDA decoder fast paths."
        )
        return vae
    if not _HAS_TRITON:
        # aten GroupNorm is ~2x slower on NHWC tensors than on NCHW, so the
        # channels_last fast path is only a net win with the Triton
        # GroupNorm+SiLU fuse (measured: 97 -> 141 ms at 1024^2 with
        # channels_last alone vs 97 -> 29 ms with both).
        logger.warning(
            "FLUX.2 VAE: Triton unavailable; skipping CUDA decoder fast paths."
        )
        return vae

    decoder = vae.decoder
    attn_modules = [
        m
        for m in decoder.modules()
        if _attn_fast_compatible(m, Attention, AttnProcessor2_0)
    ]
    n_attn_total = sum(1 for m in decoder.modules() if isinstance(m, Attention))
    if len(attn_modules) != n_attn_total:
        # AttnProcessor2_0 `.view`s the 4D activation, which is illegal on
        # channels_last tensors; without a layout-safe rewrite for every
        # attention block the layout switch cannot be applied (fail closed).
        logger.warning(
            "FLUX.2 VAE: %d/%d attention blocks lack a layout-safe rewrite; "
            "skipping CUDA decoder fast paths.",
            n_attn_total - len(attn_modules),
            n_attn_total,
        )
        return vae

    gate = VaeFastPathGate()
    decoder._sgl_gate = gate
    decoder._sgl_channels_last = False
    decoder.forward = MethodType(_decoder_layout_forward, decoder)
    n_up = _install_fused_upsample(decoder, Upsample2D, gate)
    for m in attn_modules:
        m._sgl_gate = gate
        m._sgl_folded_v = None
        m.forward = MethodType(_attn_fast_forward, m)
    n_norm = _install_norm_silu(decoder, ResnetBlock2D, gate)
    setattr(vae, GATE_ATTR, gate)
    logger.info(
        "FLUX.2 VAE: installed quality-gated decoder fast paths "
        "(channels_last dispatch, %d fused upsamplers, %d fast attention "
        "blocks, %d GroupNorm+SiLU fusions).",
        n_up,
        len(attn_modules),
        n_norm,
    )
    return vae
