"""Row-block <-> head-group re-layout around the DSA prefill shard-Q all-to-alls.

Both directions swap the first two dims of a [A, B, L] tensor whose last dim is
contiguous; torch's copy of the transposed view runs ~1.4 TB/s on these shapes
(fp8 rows are not vectorized), this one moves each L-run with wide loads.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _swap_leading_dims_kernel(src, dst, A, B, L, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    a = pid // B
    b = pid % B
    src_row = src + pid * L
    dst_row = dst + (b * A + a) * L
    offs = tl.arange(0, BLOCK)
    for start in range(0, L, BLOCK):
        mask = start + offs < L
        tl.store(
            dst_row + start + offs,
            tl.load(src_row + start + offs, mask=mask),
            mask=mask,
        )


def swap_leading_dims(src: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """out[b, a] = src[a, b] for contiguous src [A, B, ...] and out [B, A, ...]."""
    assert src.is_contiguous() and out.is_contiguous()
    a, b = src.shape[0], src.shape[1]
    assert out.shape[0] == b and out.shape[1] == a and out.shape[2:] == src.shape[2:]
    # Byte view: the copy is dtype-agnostic and fp8 rows get 16-byte accesses.
    src_bytes = src.view(torch.uint8).view(a * b, -1)
    out_bytes = out.view(torch.uint8).view(a * b, -1)
    row = src_bytes.shape[1]
    _swap_leading_dims_kernel[(a * b,)](
        src_bytes, out_bytes, a, b, row, BLOCK=4096, num_warps=8
    )
    return out
