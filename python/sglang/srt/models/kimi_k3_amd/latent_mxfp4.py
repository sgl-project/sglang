"""Optional MXFP4 copies of Kimi-K3's replicated latent MoE projections
(--enable-kimi-k3-latent-mxfp4)."""

import torch
from torch import nn

from sglang.kernels.ops.gemm import latent_mxfp4_aiter_hip
from sglang.srt.environ import envs
from sglang.srt.utils import is_hip

# Batches below this keep the BF16 projections, so decode and verify stay exact and
# keep the merged front GEMM that also reads the down projection. The MXFP4 GEMM
# alone is 2-2.7x faster than F.linear from 256 tokens up on MI355X; 2048 is the
# threshold the GSM8K and serving A/B ran with.
_MIN_TOKENS = 2048
# The latent down/up projections the packed copies replace.
_DOWN_SHAPE = (3584, 7168)
_UP_SHAPE = (7168, 3584)


def build_latent_mxfp4_projections(
    *,
    latent_quantized: bool,
    moe_a2a: bool,
    deterministic: bool,
) -> "LatentMXFP4Projections":
    """The per-layer holder of the MXFP4 copies. Raises where the flag cannot apply,
    since the user asked for it explicitly."""
    if deterministic:
        # MXFP4 is picked by the batch token count, which breaks batch invariance.
        raise ValueError(
            "--enable-kimi-k3-latent-mxfp4 is incompatible with "
            "--enable-deterministic-inference"
        )
    if latent_quantized or moe_a2a:
        raise ValueError(
            "--enable-kimi-k3-latent-mxfp4 needs BF16 latent projections and no "
            "--moe-a2a-backend"
        )
    if not (
        is_hip() and envs.SGLANG_USE_AITER.get() and latent_mxfp4_aiter_hip.available()
    ):
        raise ValueError(
            "--enable-kimi-k3-latent-mxfp4 needs SGLANG_USE_AITER=1 and AITER's "
            "MXFP4 GEMM on gfx950"
        )
    return LatentMXFP4Projections()


class LatentMXFP4Projections(nn.Module):
    down_weight: torch.Tensor | None
    down_scale: torch.Tensor | None
    up_weight: torch.Tensor | None
    up_scale: torch.Tensor | None

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("down_weight", None, persistent=False)
        self.register_buffer("down_scale", None, persistent=False)
        self.register_buffer("up_weight", None, persistent=False)
        self.register_buffer("up_scale", None, persistent=False)

    def prepare(self, *, down_weight: torch.Tensor, up_weight: torch.Tensor) -> None:
        """Pack the loaded BF16 latent weights; raises on any other shape or dtype."""
        if (
            tuple(down_weight.shape) != _DOWN_SHAPE
            or tuple(up_weight.shape) != _UP_SHAPE
            or down_weight.dtype != torch.bfloat16
            or up_weight.dtype != torch.bfloat16
            or down_weight.device != up_weight.device
        ):
            raise ValueError(
                "--enable-kimi-k3-latent-mxfp4 needs BF16 latent projections of "
                f"{_DOWN_SHAPE} / {_UP_SHAPE} on one device, got "
                f"{down_weight.dtype} {tuple(down_weight.shape)} on "
                f"{down_weight.device} / {up_weight.dtype} "
                f"{tuple(up_weight.shape)} on {up_weight.device}"
            )
        down, down_scale = latent_mxfp4_aiter_hip.pack(weight=down_weight)
        up, up_scale = latent_mxfp4_aiter_hip.pack(weight=up_weight)
        if self.down_weight is None:
            self.down_weight, self.down_scale = down, down_scale
            self.up_weight, self.up_scale = up, up_scale
            return
        # A reload (e.g. a weight update) refills the copies in place: captured
        # CUDA graphs hold their addresses.
        for dst, src in (
            (self.down_weight, down),
            (self.down_scale, down_scale),
            (self.up_weight, up),
            (self.up_scale, up_scale),
        ):
            dst.copy_(src)

    def can_run(self, *, num_tokens: int) -> bool:
        return self.down_weight is not None and num_tokens >= _MIN_TOKENS

    def down(self, *, x: torch.Tensor) -> torch.Tensor:
        if self.down_weight is None or self.down_scale is None:
            raise RuntimeError("latent MXFP4 down projection has not been prepared")
        return latent_mxfp4_aiter_hip.run(
            x=x, weight=self.down_weight, scale=self.down_scale
        )

    def up(self, *, x: torch.Tensor) -> torch.Tensor:
        if self.up_weight is None or self.up_scale is None:
            raise RuntimeError("latent MXFP4 up projection has not been prepared")
        return latent_mxfp4_aiter_hip.run(
            x=x, weight=self.up_weight, scale=self.up_scale
        )
