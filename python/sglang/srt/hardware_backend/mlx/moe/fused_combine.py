"""Fused moe_combine: sum_k (y[b,k,h] * scores[b,k]) in one Metal kernel.

Why this exists
---------------
The reference combine on Apple Silicon runs as two compute kernels:

    weighted = y * scores[..., None]   # broadcast + multiply
    out      = weighted.sum(axis=-2)   # reduce over TOP_K

Both walk the full y tensor and a transient ``weighted`` materializes between
them. Fusing into one kernel saves one read of y, one write of the
intermediate, and one dispatch per MoE layer.

Design
------
Operation:
    out[b, h] = sum over k in [0, TOP_K) of y[b, k, h] * scores[b, k]

Threadgroup geometry, cribbed from MLX's ``col_reduce_small`` in
``mlx/backend/metal/kernels/reduction/reduce_col.h``, which is the canonical
small reduction dim pattern (TOP_K=8 fits this).

    Threads per TG     : 64
    N_READS per thread : 4 contiguous output columns
    Outputs per TG     : 256
    Reduction axis     : looped in registers, no shared memory, no barriers

Each thread owns 4 contiguous outputs in the flat [B*H] index space, loops
over TOP_K experts, and accumulates in fp32.

Eligibility
-----------
- y dtype in {fp16, bf16}
- scores dtype in {fp16, fp32}, typed independently of y (template TS) and
  promoted to fp32 for the accumulate. Real MoE routers emit fp32 routing
  scores, so the common production combo is fp16 y with fp32 scores.
- y rank 3, scores rank 2, shapes aligned as [B, TOP_K, H] / [B, TOP_K]
- H divisible by 256 so each TG aligns on a row boundary and no thread
  straddles the [b, k, *] -> [b, k+1, *] boundary

Outside this regime, fall back to ``(y * scores[..., None]).sum(axis=-2)``.
"""

from __future__ import annotations

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)

_THREADS_PER_TG = 64
_N_READS = 4
_OUTPUTS_PER_TG = _THREADS_PER_TG * _N_READS  # 256


_KERNEL_SOURCE = r"""
    // Body only. MLX auto-generates the signature from input_names /
    // output_names and the template params below.
    //
    // The N_READS=4 outputs a thread writes are guaranteed to stay within a
    // single batch's H range because eligibility enforces
    // H % OUTPUTS_PER_TG (256) == 0, which implies H % N_READS == 0.
    // column_base / H gives the correct b; no cross-batch guard needed.

    constexpr int N_READS = 4;
    constexpr uint H = HIDDEN;
    constexpr uint TOPK = TOP_K;

    uint tid = thread_position_in_grid.x;
    uint column_base = tid * N_READS;
    uint b = column_base / H;
    uint h_base = column_base - b * H;   // == column_base % H

    // Scores for this (b, *) row hoisted into registers. Kept in their own
    // dtype TS (independent of y's dtype T): MLX types the `scores` pointer
    // from the array dtype, so fp32 scores are read as fp32 here, with no
    // narrowing to T, then promoted to fp32 for the multiply below.
    TS s_thread[TOPK];
    for (uint k = 0; k < TOPK; k++) {
        s_thread[k] = scores[b * TOPK + k];
    }

    float acc[N_READS] = {0.0f, 0.0f, 0.0f, 0.0f};

    const device T* y_p = y
        + uint64_t(b) * uint64_t(TOPK) * uint64_t(H)
        + uint64_t(h_base);
    for (uint k = 0; k < TOPK; k++) {
        float sk = float(s_thread[k]);
        for (uint r = 0; r < N_READS; r++) {
            acc[r] += float(y_p[r]) * sk;
        }
        y_p += H;
    }

    device T* out_p = out + uint64_t(b) * uint64_t(H) + uint64_t(h_base);
    for (uint r = 0; r < N_READS; r++) {
        out_p[r] = T(acc[r]);
    }
"""


_kernel_cache: dict[tuple[mx.Dtype, mx.Dtype], object] = {}


def _dtype_tag(dtype: mx.Dtype) -> str:
    # Metal's host_name attribute rejects '.', so strip the mlx.core prefix.
    return str(dtype).replace("mlx.core.", "").replace(".", "_")


def _get_kernel(y_dtype: mx.Dtype, scores_dtype: mx.Dtype):
    """Return a compiled mx.fast.metal_kernel for the (y, scores) dtype pair.

    Keyed on the dtype pair because y and scores now carry independent Metal
    types (T and TS). MLX handles per-template specialization internally, so
    different TOP_K / HIDDEN values reuse this same wrapper object.
    """
    key = (y_dtype, scores_dtype)
    if key not in _kernel_cache:
        name = f"fused_moe_combine_y{_dtype_tag(y_dtype)}_s{_dtype_tag(scores_dtype)}"
        _kernel_cache[key] = mx.fast.metal_kernel(
            name=name,
            input_names=["y", "scores"],
            output_names=["out"],
            source=_KERNEL_SOURCE,
        )
    return _kernel_cache[key]


_Y_DTYPES = (mx.float16, mx.bfloat16)
_SCORES_DTYPES = (mx.float16, mx.float32)


def can_fuse(y: mx.array, scores: mx.array) -> bool:
    """Cheap structural check: does this combine match the fast-path regime?"""
    if y.ndim != 3 or scores.ndim != 2:
        return False
    B, TOPK, H = y.shape
    if scores.shape != (B, TOPK):
        return False
    if y.dtype not in _Y_DTYPES or scores.dtype not in _SCORES_DTYPES:
        return False
    if H % _OUTPUTS_PER_TG != 0:
        return False
    return True


def fused_combine(y: mx.array, scores: mx.array) -> mx.array:
    """Compute ``out[b, h] = sum_k y[b, k, h] * scores[b, k]`` in one kernel.

    Numerical contract: this computes the same reduction as the reference::

        (y * scores[..., None]).sum(axis=-2)

    but is NOT bit-identical to it. The reference forms the ``y * scores``
    product in the operands' own dtype (an fp16 product when both are fp16)
    and reduces in that dtype. The fused kernel promotes both operands to
    fp32, forms the product in fp32, accumulates in fp32, and narrows to
    ``y.dtype`` only on the final write. The fused path is therefore strictly
    closer to the fp32 ground truth; the two paths can differ in the last
    fp16/bf16 ULPs on a given layer depending on which one can_fuse selects.

    Falls back to the broadcast/sum reference when can_fuse(y, scores) is
    False.
    """
    if not can_fuse(y, scores):
        return (y * scores[..., None]).sum(axis=-2)

    B, TOPK, H = y.shape
    kernel = _get_kernel(y.dtype, scores.dtype)
    (out,) = kernel(
        inputs=[y, scores],
        template=[
            ("T", y.dtype),
            ("TS", scores.dtype),
            ("TOP_K", TOPK),
            ("HIDDEN", H),
        ],
        # grid is in *threads* (MLX convention): total threads = B * H / N_READS,
        # one TG = 64 threads handles 256 contiguous outputs.
        grid=(B * H // _N_READS, 1, 1),
        threadgroup=(_THREADS_PER_TG, 1, 1),
        output_shapes=[(B, H)],
        output_dtypes=[y.dtype],
    )
    return out
