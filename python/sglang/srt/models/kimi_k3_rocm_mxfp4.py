"""Optional MXFP4 copies of Kimi-K3's replicated latent MoE projections."""

from typing import Optional

import torch
from torch import nn

from sglang.kernels.ops.gemm import latent_mxfp4_aiter_hip
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_exec
from sglang.srt.utils import print_warning_once


def build_latent_mxfp4_projections() -> Optional["LatentMXFP4Projections"]:
    """Per-layer holder when the device can run the MXFP4 GEMMs, else None."""
    # MXFP4 is picked by the batch token count, which breaks batch invariance.
    if get_exec().deterministic.enable_deterministic_inference:
        print_warning_once(
            "SGLANG_OPT_HIP_K3_MOE_LATENT_MXFP4 is ignored with "
            "--enable-deterministic-inference; keeping the BF16 latent projections."
        )
        return None
    if not (envs.SGLANG_USE_AITER.get() and latent_mxfp4_aiter_hip.supported()):
        print_warning_once(
            "SGLANG_OPT_HIP_K3_MOE_LATENT_MXFP4 needs SGLANG_USE_AITER=1 and AITER "
            "on gfx950; keeping the BF16 latent projections."
        )
        return None
    return LatentMXFP4Projections(
        min_tokens=envs.SGLANG_OPT_HIP_K3_MOE_LATENT_MXFP4_MIN_TOKENS.get()
    )


class LatentMXFP4Projections(nn.Module):
    down_weight: torch.Tensor | None
    down_scale: torch.Tensor | None
    up_weight: torch.Tensor | None
    up_scale: torch.Tensor | None

    def __init__(self, *, min_tokens: int):
        super().__init__()
        if min_tokens < 1:
            raise ValueError(
                "SGLANG_OPT_HIP_K3_MOE_LATENT_MXFP4_MIN_TOKENS must be positive, "
                f"got {min_tokens}"
            )
        self.min_tokens = min_tokens
        self.register_buffer("down_weight", None, persistent=False)
        self.register_buffer("down_scale", None, persistent=False)
        self.register_buffer("up_weight", None, persistent=False)
        self.register_buffer("up_scale", None, persistent=False)

    def prepare(self, *, down_weight: torch.Tensor, up_weight: torch.Tensor) -> None:
        if (
            down_weight.shape != (3584, 7168)
            or up_weight.shape != (7168, 3584)
            or down_weight.dtype != torch.bfloat16
            or up_weight.dtype != torch.bfloat16
            or down_weight.device != up_weight.device
            or not latent_mxfp4_aiter_hip.supported(device=down_weight.device)
        ):
            self.down_weight = self.down_scale = self.up_weight = self.up_scale = None
            print_warning_once(
                "Kimi-K3 latent MXFP4 is unavailable for these weights/device; "
                "keeping the original latent projections"
            )
            return
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
        return self.down_weight is not None and num_tokens >= self.min_tokens

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
