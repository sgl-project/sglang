"""Fuse the MoE combine, sum_k y[b,k,h] * scores[b,k], into one Metal kernel.

The reference combine runs two kernels, a broadcast multiply and a reduce
over TOP_K, with a transient materialized between them. One kernel saves a
full read of y, the intermediate round trip, and one dispatch per MoE layer.

Geometry follows the single-row pass of MLX's ``col_reduce_small``
(``mlx/backend/metal/kernels/reduction/reduce_col.h``): 64 threads per
threadgroup, 4 contiguous outputs per thread in the flat [B*H] space, so
256 outputs per threadgroup; TOP_K is looped in registers and accumulated
in fp32. No threadgroup memory, no barriers.

Fused when:
- y is fp16 or bf16, rank >= 3, shaped [..., TOP_K, H]
- scores is fp16, bf16, or fp32 (independent template dtype TS), shaped
  like y's leading dims: scores.shape == y.shape[:-1]
- every dim nonzero
- H % 256 == 0, so threadgroups tile rows exactly

Leading dims are flattened into the kernel's row dim B before dispatch, so
the mlx-lm combine site's [batch, seq, TOP_K, H] passes through unchanged.

Otherwise: ``(y * scores[..., None]).sum(axis=-2).astype(y.dtype)``.
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

    // MLX types the `scores` pointer from the array dtype, not from T, so
    // fp32 scores are read as fp32 (no narrowing) and promoted to fp32 for
    // the multiply below.
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
    """Return a compiled mx.fast.metal_kernel for the (y, scores) dtype pair."""
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
_SCORES_DTYPES = (mx.float16, mx.bfloat16, mx.float32)


def can_fuse(y: mx.array, scores: mx.array) -> bool:
    """Cheap structural check: does this combine match the fast-path regime?"""
    if y.ndim < 3 or scores.ndim != y.ndim - 1:
        return False
    if tuple(y.shape[:-1]) != tuple(scores.shape):
        return False
    if any(d == 0 for d in y.shape):
        return False
    if y.dtype not in _Y_DTYPES or scores.dtype not in _SCORES_DTYPES:
        return False
    if y.shape[-1] % _OUTPUTS_PER_TG != 0:
        return False
    return True


def fused_combine(y: mx.array, scores: mx.array) -> mx.array:
    """Compute ``out[..., h] = sum_k y[..., k, h] * scores[..., k]`` in one kernel.

    Numerical contract: this computes the same reduction as the reference::

        (y * scores[..., None]).sum(axis=-2)

    but is NOT bit-identical to it. The reference forms the ``y * scores``
    product in the operands' own dtype (an fp16 product when both are fp16)
    and reduces in that dtype. The fused kernel promotes both operands to
    fp32, forms the product in fp32, accumulates in fp32, and narrows to
    ``y.dtype`` only on the final write. The fused path is therefore strictly
    closer to the fp32 ground truth; the two paths can differ in the last
    fp16/bf16 ULPs on a given layer depending on which one can_fuse selects.

    Falls back to the broadcast/sum reference, narrowed to ``y.dtype``, when
    can_fuse(y, scores) is False; both paths return ``y.dtype``.

    Leading dims (everything before TOP_K) are flattened into the kernel's
    row dim and restored on the output, so rank 3 [B, TOP_K, H] and the
    mlx-lm site's rank 4 [batch, seq, TOP_K, H] both dispatch.
    """
    if not can_fuse(y, scores):
        return (y * scores[..., None]).sum(axis=-2).astype(y.dtype)

    TOPK, H = y.shape[-2], y.shape[-1]
    lead = tuple(y.shape[:-2])
    y = y.reshape(-1, TOPK, H)
    scores = scores.reshape(-1, TOPK)
    B = y.shape[0]
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
    return out.reshape(*lead, H)
