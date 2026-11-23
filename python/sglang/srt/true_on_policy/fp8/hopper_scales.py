"""Stands in for the stock inverse_transform_scale_ue8m0, a load-time converter whose consistency check syncs the host;
TOP converts weight scales after every weight write, including under CUDA graph capture.

Exact UE8M0 scale conversion for Hopper (SM90) DeepGEMM.

Payloads and quantization decisions remain unchanged. SM90 DeepGEMM consumes
FP32 block scales instead of SM100's packed exponent bytes.
"""

import weakref

import torch
import triton
import triton.language as tl

from sglang.srt.true_on_policy.fp8 import (
    _REFRESHED,
    _WEIGHT_GENERATION,
    _outside_weight_pool,
)


@triton.jit
def _unpack_ue8m0_kernel(
    src,
    dst,
    rows,
    groups,
    inner_rows,
    src_outer_stride,
    src_row_stride,
    src_col_stride,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0)
    g = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = g < groups
    row = (r // inner_rows) * src_outer_stride + (r % inner_rows) * src_row_stride
    word = tl.load(src + row + (g // 4) * src_col_stride, mask=mask, other=0)
    exponent = (word >> ((g % 4) * 8)) & 255
    bits = exponent << 23
    bits = tl.where(exponent == 0, 0x00400000, bits)
    bits = tl.where(exponent == 255, 0x7FC00000, bits)
    tl.store(dst + r * groups + g, bits, mask=mask)


def unpack_ue8m0(scales, groups, out=None):
    """[..., ceil(groups/4)] packed int32 UE8M0 -> [..., groups] FP32 bits, in one launch.

    Up to three dims are read through their strides, so a strided view (expert scales sliced every 128 rows) is never
    copied first. ``out``: a contiguous FP32 [..., groups] tensor written in place.
    """
    if scales.dtype != torch.int32:
        raise TypeError("Expected packed int32 UE8M0 scales")
    if scales.shape[-1] * 4 < groups:
        raise ValueError("Packed UE8M0 scales hold fewer groups than requested")
    lead = scales.shape[:-1]
    if scales.ndim > 3:
        scales = scales.reshape(-1, *scales.shape[-2:])
    view = scales if scales.ndim == 3 else scales.reshape(1, -1, scales.shape[-1])
    rows = view.shape[0] * view.shape[1]
    if out is None:
        out = torch.empty((*lead, groups), device=scales.device, dtype=torch.float32)
    if out.shape != (*lead, groups) or out.dtype != torch.float32:
        raise ValueError("unpack_ue8m0 out must be FP32 [..., groups]")
    if not out.is_contiguous():
        raise ValueError("unpack_ue8m0 out must be contiguous")
    if rows and groups:
        block = min(triton.next_power_of_2(groups), 1024)
        _unpack_ue8m0_kernel[(rows, triton.cdiv(groups, block))](
            view,
            out.view(torch.int32),
            rows,
            groups,
            view.shape[1],
            view.stride(0),
            view.stride(1),
            view.stride(2),
            BLOCK=block,
        )
    return out


# Every live cached entry, so refresh_cached_scales can rewrite them in place.
_CACHED: list = []


def block_scales(scales, groups, *, cached=False):
    """FP32 block scales of expanded packed weight scales [..., n, words]: every 128th row unpacked.

    ``cached``: a buffer on ``scales``, rewritten after weight writes (fp8.weights_changed); serving's own scale
    parameters only.
    """
    if not cached:
        return unpack_ue8m0(scales[..., ::128, :], groups)
    entry = getattr(scales, "_top_block_scales", None)
    shape = (tuple(scales.shape), groups)
    if torch.cuda.is_current_stream_capturing():
        if entry is None or entry["shape"] != shape or _stale(entry, scales):
            return unpack_ue8m0(scales[..., ::128, :], groups)
        return entry["out"]
    if entry is None or entry["shape"] != shape:
        if entry is None:
            _CACHED.append(weakref.ref(scales))
        # Outside every memory-saver region: a paused region never unmaps it.
        with _outside_weight_pool():
            out = unpack_ue8m0(scales[..., ::128, :], groups)
        entry = dict(shape=shape, out=out)
        scales._top_block_scales = entry
        _mark_fresh(entry, scales)
    elif _stale(entry, scales):
        _refresh(entry, scales)
    return entry["out"]


def _stale(entry, scales):
    return entry["key"] != (_WEIGHT_GENERATION[0], scales.data_ptr())


def _mark_fresh(entry, scales):
    entry["key"] = (_WEIGHT_GENERATION[0], scales.data_ptr())


def _refresh(entry, scales):
    # In place: a captured graph keeps reading this buffer.
    unpack_ue8m0(scales[..., ::128, :], entry["shape"][1], out=entry["out"])
    _mark_fresh(entry, scales)


def refresh_cached_scales():
    """Rewrite every cached block-scale buffer from its live scales after a weight write.

    Called once a weight write is done (fp8.refresh_derived): a CUDA graph
    replay reads the buffers without running block_scales, so they must
    already hold the new weights' scales.
    """
    generation = _WEIGHT_GENERATION[0]
    live = []
    for ref in _CACHED:
        scales = ref()
        entry = None if scales is None else getattr(scales, "_top_block_scales", None)
        if entry is None:
            continue
        if entry["shape"] != (tuple(scales.shape), entry["shape"][1]):
            # Reshaped scales get a new buffer on their next eager call.
            del scales._top_block_scales
            continue
        if _stale(entry, scales):
            _refresh(entry, scales)
        live.append(ref)
    _CACHED[:] = live
    _REFRESHED[0] = generation


def weight_scales(pair, *, cached=False):
    payload, scales = pair
    if scales.shape[-2] != payload.shape[-2] or payload.shape[-2] % 128:
        raise ValueError("Expected expanded 128x128 weight-block scales")
    return payload, block_scales(scales, payload.shape[-1] // 128, cached=cached)
