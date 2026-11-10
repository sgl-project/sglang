"""Fold a dense NVFP4 shared expert into the SM70 GLM routed decode.

The MoE block opens `fold_shared_expert(pack)` around its routed call; the
Marlin runner takes the pack only when it runs the HMMA decode, and the block
computes the shared expert itself when nobody did.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

import torch

from sglang.kernels.ops.moe.sm70_glm_nvfp4_moe_decode import Sm70SharedExpertPack

logger = logging.getLogger(__name__)


class SharedFoldSlot:
    __slots__ = ("pack", "folded")

    def __init__(self, pack: Sm70SharedExpertPack) -> None:
        self.pack = pack
        self.folded = False


_SLOT: ContextVar[SharedFoldSlot | None] = ContextVar("sm70_shared_fold", default=None)


@contextmanager
def fold_shared_expert(pack: Sm70SharedExpertPack) -> Iterator[SharedFoldSlot]:
    slot = SharedFoldSlot(pack)
    token = _SLOT.set(slot)
    try:
        yield slot
    finally:
        _SLOT.reset(token)


def take_shared_expert() -> Sm70SharedExpertPack | None:
    """Claim the open shared expert; the caller must add it to its output."""
    slot = _SLOT.get()
    if slot is None or slot.folded:
        return None
    slot.folded = True
    return slot.pack


def build_shared_expert_pack(
    gate_up_proj: torch.nn.Module, down_proj: torch.nn.Module
) -> Sm70SharedExpertPack | None:
    """HMMA pack of a Marlin NVFP4 shared expert, or None when its decode GEMMs
    are not bitwise the split-K GEMV the fold uses. Not capturable: it syncs."""
    from sglang.kernels.ops.gemm.sm70_glm_nvfp4_gemv import (
        _load,
        prepack_glm_nvfp4_weight,
        sm70_glm_nvfp4_gemv_available,
    )
    from sglang.kernels.ops.moe.sm70_glm_nvfp4_moe_decode import (
        sm70_glm_nvfp4_moe_decode_available,
    )
    from sglang.srt.layers.quantization.modelopt_quant import (
        ModelOptFp4LinearMethod,
        ModelOptNvFp4A16LinearMethod,
    )

    nvfp4 = (ModelOptFp4LinearMethod, ModelOptNvFp4A16LinearMethod)
    if not (
        sm70_glm_nvfp4_moe_decode_available()
        and sm70_glm_nvfp4_gemv_available()
        and isinstance(gate_up_proj.quant_method, nvfp4)
        and isinstance(down_proj.quant_method, nvfp4)
        and gate_up_proj.weight.dtype == torch.int32
        and down_proj.weight.dtype == torch.int32
    ):
        return None
    ext = _load()
    if ext is None:
        return None
    parts = []
    for proj in (gate_up_proj, down_proj):
        packed = prepack_glm_nvfp4_weight(proj.weight)
        if packed is None:
            return None
        parts += [
            packed,
            proj.weight_scale.contiguous(),
            proj.weight_global_scale.reshape(-1)[:1].float().contiguous(),
        ]
    pack = Sm70SharedExpertPack(*parts)
    # The fold replaces the linears' own decode GEMMs; prove they are the same.
    gen = torch.Generator(device=gate_up_proj.weight.device).manual_seed(0)
    for proj, packed, scales, global_scale in (
        (gate_up_proj, pack.w13_packed, pack.w13_scales, pack.w13_global),
        (down_proj, pack.w2_packed, pack.w2_scales, pack.w2_global),
    ):
        x = torch.randn(
            (1, proj.input_size_per_partition),
            generator=gen,
            device=gen.device,
            dtype=torch.float16,
        )
        expected, _ = proj(x)
        got = torch.empty_like(expected)
        if got.shape[1] != scales.shape[1]:
            return None
        ext.gemv_hmma_splitk(x, packed, scales, global_scale, got, 0)
        if not torch.equal(got, expected):
            logger.warning(
                "SM70 shared-expert fold disabled: %s decode GEMM is not the HMMA GEMV",
                type(proj).__name__,
            )
            return None
    return pack
