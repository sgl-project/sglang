"""Cake (FlashInfer) backends for the ``sampling`` operator group.

Metadata-only at import time: ``KernelSpec`` targets point at the lazy
adapters in :mod:`sglang.kernels.cake_kernels.sampling`, which import
FlashInfer only when a kernel is actually called. The Blackwell softmax
registration (``sampling.softmax``) stays in the group ``__init__``; this
module adds the fused top-k-first sampler and its stage-1 slab kernel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, Union

from sglang.kernels.registry import register_kernel
from sglang.kernels.selector import get_kernel
from sglang.kernels.spec import (
    CapabilityRequirement,
    FormatSignature,
    KernelBackend,
    KernelSpec,
)

if TYPE_CHECKING:
    import torch

# flashinfer.jit.cake_sampling.SUPPORTED_MAJOR_VERSIONS = (9, 10, 11, 12).
_CAKE_SAMPLING_CAPS = frozenset(
    {CapabilityRequirement.cuda(min_sm=(9, 0), max_sm=(12, 1))}
)

register_kernel(
    KernelSpec(
        op="sampling.top_k_top_p_sampling_from_probs_top_k_first",
        backend=KernelBackend.FLASHINFER,
        target=(
            "sglang.kernels.cake_kernels.sampling:"
            "top_k_top_p_sampling_from_probs_top_k_first"
        ),
        capabilities=_CAKE_SAMPLING_CAPS,
        format_signature=FormatSignature(
            supported_dtypes=("float32",),
            description=(
                "FP32 contiguous probs [batch, vocab], top_k int or int32 [batch] "
                "(1..1024), top_p float or FP32 [batch] -> int32 [batch] samples; "
                "top-k applied FIRST then top-p (filter_apply_order=top_k_first), "
                "bitwise deterministic"
            ),
        ),
        description=(
            "Cake fused radix top-k -> sparse top-p -> sampling (top-k first) "
            "distributed by FlashInfer. Not a drop-in for joint top-k/top-p."
        ),
    )
)

register_kernel(
    KernelSpec(
        op="sampling.top_k_probs_to_slab",
        backend=KernelBackend.FLASHINFER,
        target="sglang.kernels.cake_kernels.sampling:top_k_probs_to_slab",
        capabilities=_CAKE_SAMPLING_CAPS,
        format_signature=FormatSignature(
            supported_dtypes=("float32",),
            description=(
                "FP32 contiguous probs [batch, vocab], top_k int or int32 [batch] "
                "(1..1024) -> (vals f32 [batch,1024], idx i32 [batch,1024], "
                "counts i32 [batch]); exact unsorted top-k support per row"
            ),
        ),
        description=(
            "Cake exact radix top-k into a [batch, 1024] slab (stage 1 of the "
            "fused sampler) distributed by FlashInfer."
        ),
    )
)


def cake_top_k_top_p_sampling_from_probs_top_k_first(
    probs: torch.Tensor,
    top_k: Union[int, torch.Tensor],
    top_p: Union[float, torch.Tensor],
    *,
    top_k_max: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    philox_seed: Optional[int] = None,
    philox_offset: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    renorm_out: Optional[torch.Tensor] = None,
    workspace: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    enable_pdl: bool = True,
) -> torch.Tensor:
    """Explicit Cake entry point; top-k is applied FIRST, then top-p.

    Callers gate on the adapter's
    ``supports_top_k_top_p_sampling_top_k_first``. The sampler call site keeps
    its joint top-k/top-p filtering; this op is an explicit opt-in.
    """
    return get_kernel(
        "sampling.top_k_top_p_sampling_from_probs_top_k_first",
        KernelBackend.FLASHINFER,
    )(
        probs,
        top_k,
        top_p,
        top_k_max=top_k_max,
        generator=generator,
        philox_seed=philox_seed,
        philox_offset=philox_offset,
        out=out,
        renorm_out=renorm_out,
        workspace=workspace,
        enable_pdl=enable_pdl,
    )


def cake_top_k_probs_to_slab(
    probs: torch.Tensor,
    top_k: Union[int, torch.Tensor],
    *,
    top_k_max: Optional[int] = None,
    out_vals: Optional[torch.Tensor] = None,
    out_idx: Optional[torch.Tensor] = None,
    out_count: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Explicit Cake entry point for the stage-1 exact top-k slab."""
    return get_kernel("sampling.top_k_probs_to_slab", KernelBackend.FLASHINFER)(
        probs,
        top_k,
        top_k_max=top_k_max,
        out_vals=out_vals,
        out_idx=out_idx,
        out_count=out_count,
    )


__all__ = [
    "cake_top_k_probs_to_slab",
    "cake_top_k_top_p_sampling_from_probs_top_k_first",
]
