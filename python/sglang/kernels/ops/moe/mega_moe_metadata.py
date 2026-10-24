"""Routing metadata for an already quantized MegaMoE activation buffer."""

import triton
import triton.language as tl


@triton.jit
def _copy_metadata(I, W, OI, OW, N: tl.constexpr, P: tl.constexpr, B: tl.constexpr):
    off = tl.program_id(0) * B + tl.arange(0, B)
    indices = tl.load(I + off, off < N, -1)
    weights = tl.load(W + off, off < N, 0)
    tl.store(OI + off, indices, off < P)
    tl.store(OW + off, weights, off < P)


def copy_mega_moe_metadata(indices, weights, out_indices, out_weights):
    assert indices.shape == weights.shape
    assert out_indices.shape == out_weights.shape
    assert indices.shape[1] == out_indices.shape[1]
    assert indices.shape[0] <= out_indices.shape[0]
    assert all(t.is_contiguous() for t in (indices, weights, out_indices, out_weights))
    _copy_metadata[(triton.cdiv(out_indices.numel(), 256),)](
        indices,
        weights,
        out_indices,
        out_weights,
        indices.numel(),
        out_indices.numel(),
        256,
    )
