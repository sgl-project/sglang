import torch
import triton
import triton.language as tl


@triton.jit
def _small_copy_kernel(
    sources,
    destinations,
    SIZES: tl.constexpr,
    COLS: tl.constexpr,
    SOURCE_STRIDES: tl.constexpr,
    DEST_STRIDES: tl.constexpr,
    BIT_WIDTHS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    field = tl.program_id(0)
    offsets = (tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    for i in tl.static_range(len(SIZES)):
        if field == i:
            src_offset = (
                offsets // COLS[i] * SOURCE_STRIDES[i][0]
                + offsets % COLS[i] * SOURCE_STRIDES[i][1]
            )
            dst_offset = (
                offsets // COLS[i] * DEST_STRIDES[i][0]
                + offsets % COLS[i] * DEST_STRIDES[i][1]
            )
            src = sources[i]
            dst = destinations[i]
            if BIT_WIDTHS[i] == 8:
                src = src.to(tl.pointer_type(tl.uint8))
                dst = dst.to(tl.pointer_type(tl.uint8))
            elif BIT_WIDTHS[i] == 16:
                src = src.to(tl.pointer_type(tl.uint16))
                dst = dst.to(tl.pointer_type(tl.uint16))
            elif BIT_WIDTHS[i] == 32:
                src = src.to(tl.pointer_type(tl.uint32))
                dst = dst.to(tl.pointer_type(tl.uint32))
            elif BIT_WIDTHS[i] == 64:
                src = src.to(tl.pointer_type(tl.uint64))
                dst = dst.to(tl.pointer_type(tl.uint64))
            values = tl.load(src + src_offset, offsets < SIZES[i], other=0)
            tl.store(dst + dst_offset, values, offsets < SIZES[i])


def try_small_copy(dsts, srcs):
    if len(dsts) < 2 or len(dsts) != len(srcs):
        return False
    supported = (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    )
    device = dsts[0].device
    if device.type != "cuda":
        return False
    sizes, cols, src_strides, dst_strides, bit_widths = [], [], [], [], []
    src_ranges, dst_ranges = [], []
    for dst, src in zip(dsts, srcs):
        if (
            dst.dtype not in supported
            or src.dtype not in supported
            or dst.device != device
            or src.device != device
            or dst.shape != src.shape
            or dst.ndim not in (1, 2)
            or dst.numel() > 8192
            or dst.requires_grad
            or src.requires_grad
            or src.is_neg()
            or dst.is_neg()
        ):
            return False
        same_dtype = dst.dtype == src.dtype
        if not same_dtype and not (
            dst.dtype in (torch.int32, torch.int64)
            and src.dtype in (torch.int32, torch.int64)
        ):
            return False
        n = dst.numel()
        c = max(dst.shape[-1], 1)
        ds = dst.stride() if dst.ndim == 2 else (0, dst.stride(0))
        ss = src.stride() if src.ndim == 2 else (0, src.stride(0))
        if ds[1] <= 0 or (dst.ndim == 2 and ds[0] < c * ds[1]):
            return False
        sizes.append(n)
        cols.append(c)
        src_strides.append(ss)
        dst_strides.append(ds)
        bit_widths.append(dst.element_size() * 8 if same_dtype else 0)
        for tensor, ranges in ((src, src_ranges), (dst, dst_ranges)):
            first = tensor.data_ptr()
            extent = (
                (sum((d - 1) * s for d, s in zip(tensor.shape, tensor.stride())) + 1)
                * tensor.element_size()
                if n
                else 0
            )
            ranges.append((first, first + extent))
    for i, (first, last) in enumerate(dst_ranges):
        for j, (other_first, other_last) in enumerate(src_ranges):
            if first < other_last and other_first < last:
                if (
                    i != j
                    or dsts[i].data_ptr() != srcs[i].data_ptr()
                    or dsts[i].stride() != srcs[i].stride()
                    or dsts[i].dtype != srcs[i].dtype
                ):
                    return False
        for other_first, other_last in dst_ranges[:i]:
            if first < other_last and other_first < last:
                return False
    largest = max(sizes)
    if largest:
        _small_copy_kernel[(len(dsts), triton.cdiv(largest, 256))](
            tuple(srcs),
            tuple(dsts),
            tuple(sizes),
            tuple(cols),
            tuple(src_strides),
            tuple(dst_strides),
            tuple(bit_widths),
            BLOCK=256,
        )
    return True
