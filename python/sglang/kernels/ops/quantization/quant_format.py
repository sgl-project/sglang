"""What a quantized tensor looks like on the host: the FP8 format this build
targets, and the buffers a per-token-group quant writes into.

A leaf module -- ``torch`` and ``functools`` only. Everything here used to sit in
``fp8_kernel.py``, which imports ``per_token_group_quant`` at module level, so
``per_token_group_quant.py`` could only reach the allocation through a lazy
import to break the cycle. Being a leaf lets both sides import it normally, which
is also why ``is_hip()`` is inlined as ``torch.version.hip is not None`` (exactly
what ``sglang.srt.utils.is_hip`` does) and ``ceil_align`` as two lines.

The scale allocation is a single function because there used to be two that
disagreed: ``fp8_kernel``'s produced fp32 powers of two for row-major UE8M0,
while ``per_token_group_quant``'s ``_allocate_outputs`` produced int32-packed
exponent bytes for the same flags. Callers had no way to say which they meant --
hence ``pack_ue8m0``.

Three things vary independently:

* **rounding** -- ``scale_ue8m0`` rounds the quant multiplier to a power of two
  (which is what makes the scale expressible as a lone 8-bit exponent).
* **storage** -- ``pack_ue8m0`` writes that exponent byte, 4 per int32 (the
  actual UE8M0 data type), instead of an fp32. Only meaningful under
  ``scale_ue8m0``; without the pow-2 rounding there is no single byte to pack.
  Unpacked UE8M0 stores ``2^(e-127)`` as fp32 -- deep_gemm's ``ceil_to_ue8m0`` /
  ``fp8_einsum`` convention.
* **layout** -- ``column_major_scales`` / ``scale_tma_aligned`` place the buffer
  in memory and say nothing about what it holds.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

import torch

# --------------------------------------------------------------------------- #
# The FP8 format this build quantizes to.
#
# ROCm gfx94x (MI300) implements FP8 as e4m3**fnuz**, a different bias that tops
# out at 224 instead of 448. The device kernels already follow suit --
# ``kFP8E4M3Max`` in ``sgl_kernel/type.cuh`` is 224 under ``HIP_FP8_TYPE_FNUZ``
# -- so a host buffer that hardcodes e4m3fn would disagree with what the kernel
# writes. Anything allocating an FP8 quant output has to go through these.
#
# Re-exported from ``fp8_kernel`` for the existing callers.
# --------------------------------------------------------------------------- #


@lru_cache()
def is_fp8_fnuz() -> bool:
    if torch.version.hip is None:
        return False
    # only device 0 is checked, this assumes MI300 platforms are homogeneous
    return "gfx94" in torch.cuda.get_device_properties(0).gcnArchName


if is_fp8_fnuz():
    fp8_dtype = torch.float8_e4m3fnuz
    fp8_max = 224.0
else:
    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max
fp8_min = -fp8_max


# --------------------------------------------------------------------------- #
# Output buffers.
# --------------------------------------------------------------------------- #


def _ceil_align(x: int, y: int) -> int:
    return (x + y - 1) // y * y


def create_group_quant_scale(
    x_shape,
    device,
    group_size: int,
    column_major_scales: bool,
    scale_tma_aligned: bool,
    scale_ue8m0: bool,
    pack_ue8m0: bool = True,
) -> torch.Tensor:
    """Allocate the scale buffer for quantizing ``x_shape`` in groups of
    ``group_size`` along the last dim.

    ``pack_ue8m0`` is read only when ``scale_ue8m0`` is set. The supported
    combinations, where ``T`` is ``x_shape[-2]`` and ``ng`` is
    ``x_shape[-1] // group_size``:

      ue8m0  pack  col_major  tma   dtype    shape
      -----  ----  ---------  ----  -------  --------------------------------
      F      -     F          -     float32  [..., T, ng]        contiguous
      F      -     T          T     float32  [..., T, ng]        TMA-aligned view
      F      -     T          F     float32  [..., T, ng]        col-major view
      T      T     T          T     int32    [..., T, ceil(ng,4)/4] TMA view
      T      T     F          -     int32    [..., ceil(ng/4)]   contiguous
      T      F     F          -     float32  [..., ng]           contiguous (2^(e-127))
    """
    num_groups = x_shape[-1] // group_size

    if not scale_ue8m0:
        if not column_major_scales:
            return torch.empty(
                x_shape[:-1] + (num_groups,), device=device, dtype=torch.float32
            )
        if scale_tma_aligned:
            # aligned to 4 * sizeof(float); `...` so batched (e.g. masked
            # [E, T, H]) shapes slice the token axis, not dim 0.
            aligned_size = _ceil_align(x_shape[-2], 4)
            return torch.empty(
                x_shape[:-2] + (num_groups, aligned_size),
                device=device,
                dtype=torch.float32,
            ).transpose(-1, -2)[..., : x_shape[-2], :]
        return torch.empty(
            (num_groups,) + x_shape[:-1], device=device, dtype=torch.float32
        ).permute(-1, -2)

    if not pack_ue8m0:
        assert (
            not column_major_scales
        ), "unpacked UE8M0 (fp32 2^(e-127)) has no column-major layout"
        return torch.empty(
            x_shape[:-1] + (num_groups,), device=device, dtype=torch.float32
        )

    if column_major_scales:
        assert scale_tma_aligned, (
            "column_major_scales requires scale_tma_aligned=True "
            "when scale_ue8m0 is enabled"
        )
        *x_batch, x_q_mn, _ = x_shape
        # TODO(FIXME): Fix cuda kernel and recover here to empty.
        return torch.empty(
            (*x_batch, _ceil_align(num_groups, 4) // 4, _ceil_align(x_q_mn, 4)),
            device=device,
            dtype=torch.int32,
        ).transpose(-1, -2)[..., :x_q_mn, :]

    # Row-major packed UE8M0: an unaligned ``ng`` leaves a partially-used last
    # int32 that the kernel zero-pads.
    return torch.empty(
        x_shape[:-1] + ((num_groups + 3) // 4,), device=device, dtype=torch.int32
    )


def create_group_quant_outputs(
    x_shape,
    device,
    group_size: int,
    column_major_scales: bool,
    scale_tma_aligned: bool,
    scale_ue8m0: bool,
    pack_ue8m0: bool = True,
    out_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Allocate ``(codes, scales)`` for quantizing ``x_shape``.

    ``out_dtype`` defaults to ``fp8_dtype`` above, i.e. e4m3fnuz on ROCm gfx94x
    and e4m3fn elsewhere -- matching what the device kernels write, which follow
    the same split via ``kFP8E4M3Max``. Pass it explicitly for int8.
    """
    codes = torch.empty(x_shape, device=device, dtype=out_dtype or fp8_dtype)
    scales = create_group_quant_scale(
        x_shape=x_shape,
        device=device,
        group_size=group_size,
        column_major_scales=column_major_scales,
        scale_tma_aligned=scale_tma_aligned,
        scale_ue8m0=scale_ue8m0,
        pack_ue8m0=pack_ue8m0,
    )
    return codes, scales


__all__ = [
    "is_fp8_fnuz",
    "fp8_dtype",
    "fp8_max",
    "fp8_min",
    "create_group_quant_scale",
    "create_group_quant_outputs",
]
