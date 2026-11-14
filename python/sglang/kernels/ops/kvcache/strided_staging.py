import torch
import triton
import triton.language as tl


@triton.jit
def _gather_strided_staging_kernel(
    pointers,
    strides,
    pages,
    staging,
    per_layer_elems,
    head_offset,
    ELEMS_PER_TOKEN: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    layer = tl.program_id(0)
    ptr = tl.load(pointers + layer).to(staging.dtype)
    stride = tl.load(strides + layer).to(tl.int64)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < per_layer_elems
    token = offsets // ELEMS_PER_TOKEN
    elem = offsets % ELEMS_PER_TOKEN
    page = tl.load(pages + token // PAGE_SIZE, mask=mask, other=0)
    row = page.to(tl.int64) * PAGE_SIZE + token % PAGE_SIZE
    value = tl.load(ptr + row * stride + head_offset + elem, mask)
    tl.store(staging + layer.to(tl.int64) * per_layer_elems + offsets, value, mask)


@triton.jit
def _scatter_strided_staging_kernel(
    pointers,
    strides,
    pages,
    staging,
    head_offsets,
    per_layer_elems,
    ELEMS_PER_TOKEN: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NSLOTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    program = tl.program_id(0)
    writer, layer = program // NSLOTS, program % NSLOTS
    ptr = tl.load(pointers + layer).to(staging.dtype)
    stride = tl.load(strides + layer).to(tl.int64)
    head_offset = tl.load(head_offsets + writer).to(tl.int64)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < per_layer_elems
    token = offsets // ELEMS_PER_TOKEN
    elem = offsets % ELEMS_PER_TOKEN
    page = tl.load(pages + token // PAGE_SIZE, mask=mask, other=0)
    row = page.to(tl.int64) * PAGE_SIZE + token % PAGE_SIZE
    src = program.to(tl.int64) * per_layer_elems + offsets
    value = tl.load(staging + src, mask)
    dst = row * stride + head_offset + elem
    tl.store(ptr + dst, value, mask)


def _check_staging(staging, elems_per_token, page_size, per_layer_elems):
    if staging.device.type != "cuda" or torch.version.hip is not None:
        raise RuntimeError("Strided KV staging currently requires NVIDIA CUDA")
    if elems_per_token <= 0 or page_size <= 0 or per_layer_elems < 0:
        raise ValueError("Invalid strided staging geometry")
    if staging.dtype not in (torch.int8, torch.int16, torch.int32):
        raise ValueError("Strided staging copies integer views of KV storage")


def gather_strided_staging(
    pointers,
    strides,
    pages,
    staging,
    per_layer_elems,
    head_offset,
    *,
    elems_per_token,
    page_size,
):
    _check_staging(staging, elems_per_token, page_size, per_layer_elems)
    if per_layer_elems == 0:
        return
    _gather_strided_staging_kernel[
        (pointers.numel(), triton.cdiv(per_layer_elems, 1024))
    ](
        pointers,
        strides,
        pages,
        staging,
        per_layer_elems,
        head_offset,
        ELEMS_PER_TOKEN=elems_per_token,
        PAGE_SIZE=page_size,
        BLOCK=1024,
    )


def scatter_strided_staging(
    pointers,
    strides,
    pages,
    staging,
    head_offsets,
    per_layer_elems,
    *,
    elems_per_token,
    page_size,
    num_slots,
    num_writers,
):
    _check_staging(staging, elems_per_token, page_size, per_layer_elems)
    if per_layer_elems == 0:
        return
    if pointers.numel() != num_slots or head_offsets.numel() != num_writers:
        raise ValueError("Strided staging writer/slot counts do not match metadata")
    _scatter_strided_staging_kernel[
        (num_slots * num_writers, triton.cdiv(per_layer_elems, 1024))
    ](
        pointers,
        strides,
        pages,
        staging,
        head_offsets,
        per_layer_elems,
        ELEMS_PER_TOKEN=elems_per_token,
        PAGE_SIZE=page_size,
        NSLOTS=num_slots,
        BLOCK=1024,
    )
