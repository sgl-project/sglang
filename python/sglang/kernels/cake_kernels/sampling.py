"""Cake sampling kernels through FlashInfer's public API.

Two Cake entries live here:

* The Blackwell softmax (``flashinfer.sampling.softmax`` auto-routes to the
  ``cake_blackwell_softmax`` JIT module on SM103): ``supports_softmax`` /
  ``softmax``.
* The fused radix top-k -> sparse top-p -> sampling pipeline
  (``flashinfer.cake_sampling``, JIT module ``flashinfer.jit.cake_sampling``):
  ``supports_top_k_top_p_sampling_top_k_first`` /
  ``top_k_top_p_sampling_from_probs_top_k_first`` and the stage-1-only
  ``top_k_probs_to_slab``.

Contract of the fused sampler at FlashInfer ``46340689a5ab``: FP32 contiguous
``probs [batch, vocab]``; ``top_k`` int or int32 ``[batch]`` with
``1 <= k <= 1024`` (the slab); ``top_p`` float or FP32 ``[batch]`` in ``(0, 1]``;
``top_k_max`` avoids a device sync for tensor ``top_k``; int32 ``[batch]``
samples. Built for compute capability 9.x, 10.x, 11.x and 12.x. Semantics are
``filter_apply_order="top_k_first"`` (top-k is applied first, then top-p on the
top-k-renormalized mass) with strict bitwise determinism across variants, stream
launches and CUDA-graph replay. Requests the frozen kernels cannot serve (top-k
disabled, ``k > 1024``, vocab too large for the device) are dispatched by
FlashInfer itself to ``flashinfer.sampling.top_k_top_p_sampling_from_probs(
filter_apply_order="top_k_first", deterministic=True)``, so the top-k-first
semantics hold on every route.

**Not a drop-in replacement** for SGLang's joint top-k/top-p filtering
(``filter_apply_order="joint"``): the kept support differs whenever top-p
removes mass that top-k would have kept, or vice versa. Callers must opt in
explicitly; the sampler call site is not rewired by this module.
"""

from __future__ import annotations

from functools import lru_cache
from importlib.util import find_spec
from typing import TYPE_CHECKING, Optional, Tuple, Union

from sglang.kernels.cake_kernels._support import (
    device_capability,
    flashinfer_module_available,
)

if TYPE_CHECKING:
    import torch


@lru_cache(maxsize=None)
def _device_supported(device_index: int) -> bool:
    import torch

    return (
        torch.cuda.get_device_capability(device_index) == (10, 3)
        and find_spec("flashinfer.jit.cake_blackwell_softmax") is not None
    )


def supports_softmax(logits: torch.Tensor) -> bool:
    """The large-vocabulary domain qualified against the eager Torch caller."""
    import torch

    return (
        logits.is_cuda
        and torch.version.cuda is not None
        and logits.dtype == torch.float32
        and logits.ndim == 2
        and 1 <= logits.shape[0] <= 64
        and 128256 <= logits.shape[1] <= 262144
        and logits.shape[1] % 8 == 0
        and logits.is_contiguous()
        and _device_supported(logits.device.index)
    )


def softmax(logits: torch.Tensor) -> torch.Tensor:
    """Return softmax probabilities through FlashInfer's public API.

    FlashInfer owns the architecture-specific Cake dispatch and its fallback
    for rows outside the optimized shape domain. No sampling distribution or
    random-number-generator behavior is changed by this operation.
    """
    from flashinfer.sampling import softmax as flashinfer_softmax

    return flashinfer_softmax(logits)


# ---------------------------------------------------------------------------
# Fused top-k -> top-p -> sampling (top-k first)
# ---------------------------------------------------------------------------

FI_MODULE = "flashinfer.cake_sampling"
FI_JIT_MODULE = "flashinfer.jit.cake_sampling"
# ``flashinfer.jit.cake_sampling.SUPPORTED_MAJOR_VERSIONS``: Hopper and newer.
ARCH_MAJORS = (9, 10, 11, 12)
SLAB = 1024


def _device_in_majors(device_index: int) -> bool:
    return device_capability(device_index)[0] in ARCH_MAJORS


def _top_k_max(top_k, top_k_max: Optional[int]) -> Optional[int]:
    if isinstance(top_k, int):
        return top_k
    return None if top_k_max is None else int(top_k_max)


def supports_top_k_top_p_sampling_top_k_first(
    probs: torch.Tensor,
    top_k: Union[int, torch.Tensor],
    top_k_max: Optional[int] = None,
) -> bool:
    """Admission check mirroring the FlashInfer contract; never raises.

    Only host-visible facts are checked (no device sync): FlashInfer module
    presence, a CUDA device of a built architecture, FP32 contiguous 2-D
    ``probs`` and a scalar ``top_k`` (or ``top_k_max``) inside ``[1, 1024]``.
    Tensor ``top_k`` without ``top_k_max`` is admitted; FlashInfer then
    resolves the slab bound with one ``.item()`` sync. Requests outside the
    frozen kernels' domain still keep top-k-first semantics through
    FlashInfer's own deterministic fallback.
    """
    import torch

    kmax = _top_k_max(top_k, top_k_max)
    return (
        flashinfer_module_available(FI_MODULE, FI_JIT_MODULE)
        and probs.is_cuda
        and torch.version.cuda is not None
        and probs.dtype == torch.float32
        and probs.ndim == 2
        and probs.is_contiguous()
        and top_k is not None
        and (kmax is None or 1 <= kmax <= min(SLAB, probs.shape[1] - 1))
        and _device_in_majors(probs.device.index)
    )


def sampling_route(
    probs: torch.Tensor,
    top_k: Union[int, torch.Tensor],
    top_k_max: Optional[int] = None,
) -> str:
    """``"pipeline"`` or ``"fallback:<reason>"`` from FlashInfer (diagnostics).

    Imports FlashInfer; call only after ``supports_*`` admitted the request.
    """
    from flashinfer.cake_sampling import cake_sampling_route

    return cake_sampling_route(probs, top_k, top_k_max)


def top_k_top_p_sampling_from_probs_top_k_first(
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
    """Forward to FlashInfer; returns int32 ``[batch]`` sampled token ids.

    Top-k is applied FIRST, then top-p on the top-k-renormalized mass
    (``filter_apply_order="top_k_first"``). This is NOT the joint top-k/top-p
    filter SGLang's sampler uses by default; see the module docstring.

    ``generator`` (default CUDA generator when omitted) is advanced exactly like
    ``flashinfer.sampling.top_k_top_p_sampling_from_probs``; passing
    ``philox_seed`` + ``philox_offset`` together makes the draw an explicit
    function of the inputs without touching any generator. ``renorm_out``
    (FP32 ``[batch, 1024]``) and ``workspace`` (``(vals f32 [batch, 1024],
    idx i32 [batch, 1024], counts i32 [batch])``) are served by the pipeline
    route only.
    """
    from flashinfer.cake_sampling import (
        top_k_top_p_sampling_from_probs as cake_top_k_top_p_sampling_from_probs,
    )

    return cake_top_k_top_p_sampling_from_probs(
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


def top_k_probs_to_slab(
    probs: torch.Tensor,
    top_k: Union[int, torch.Tensor],
    *,
    top_k_max: Optional[int] = None,
    out_vals: Optional[torch.Tensor] = None,
    out_idx: Optional[torch.Tensor] = None,
    out_count: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward stage 1 alone: exact per-row top-k into a ``[batch, 1024]`` slab.

    Returns ``(values f32 [batch, 1024], indices i32 [batch, 1024], counts i32
    [batch])``; the first ``count`` entries of a row are the top-k support
    (``lexsort(-prob, index)``), unsorted, entries beyond ``count`` undefined.
    Unlike the sampler there is no fallback: FlashInfer raises ``ValueError``
    with the route reason when the frozen kernels cannot serve the request.
    """
    from flashinfer.cake_sampling import top_k_probs_to_slab as cake_top_k_probs_to_slab

    return cake_top_k_probs_to_slab(
        probs,
        top_k,
        top_k_max=top_k_max,
        out_vals=out_vals,
        out_idx=out_idx,
        out_count=out_count,
    )
