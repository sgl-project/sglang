"""Pack a uniform-width verify tree mask into existing FlashInfer buffers."""

import triton
import triton.language as tl


@triton.jit
def _pack_verify_mask_kernel(
    mask,
    kv_indptr,
    packed_indptr,
    output,
    Q: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    byte = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    bit = tl.arange(0, 8)
    kv_start = tl.load(kv_indptr + row)
    kv_end = tl.load(kv_indptr + row + 1)
    count = (kv_end - kv_start) * Q
    offsets = byte[:, None] * 8 + bit[None, :]
    values = tl.load(mask + kv_start * Q + offsets, offsets < count, other=0)
    packed = tl.sum(values.to(tl.uint32) << bit[None, :], axis=1)
    start = tl.load(packed_indptr + row)
    tl.store(output + start + byte, packed.to(tl.uint8), byte < tl.cdiv(count, 8))


def pack_verify_mask_into(
    mask, kv_indptr, packed_indptr, output, *, batch_size, query_width, max_kv_len
):
    _pack_verify_mask_kernel[
        (batch_size, triton.cdiv(query_width * max_kv_len, 8 * 256))
    ](
        mask.contiguous().view(-1),
        kv_indptr,
        packed_indptr,
        output,
        Q=query_width,
        BLOCK=256,
    )
