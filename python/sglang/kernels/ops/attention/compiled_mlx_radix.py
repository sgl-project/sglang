"""Read-only radix attention with an optional uncommitted previous-token tail."""

from functools import lru_cache
from math import isfinite

import mlx.core as mx

_SOURCE = r"""
threadgroup float maxima[8];
threadgroup float sums[8];
#ifndef HAS_TAIL
#define HAS_TAIL 0
#endif
threadgroup float partial[8 * D];
const uint b = threadgroup_position_in_grid.z;
const uint h = threadgroup_position_in_grid.y;
const uint kh = h / (H / KH);
const uint warp = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const long length = lengths[b];
const long request = requests[b];
const ulong qb = (ulong(b) * H + h) * D;
if (request < 0 || request >= ROWS || length < 1 + HAS_TAIL || length > WIDTH) {
  if (warp == 0) {
    for (uint i = 0; i < D / 32; ++i) out[qb + lane + i * 32] = T(NAN);
  }
  return;
}
float acc[D / 32];
float query[D / 32];
for (uint i = 0; i < D / 32; ++i) {
  acc[i] = 0;
  query[i] = float(q[qb + lane + i * 32]) * SCALE;
}
float maximum = -INFINITY;
float sum = 0;
for (long token = warp; token < length; token += 8) {
  const bool current = token == length - 1;
  bool previous = false;
#if HAS_TAIL
  previous = token == length - 2;
#endif
  const long slot = (current || previous) ? long(b) : long(table[request * WIDTH + token]);
  if (slot < 0 || (!current && !previous && slot >= SLOTS)) {
    maximum = NAN;
    sum = NAN;
    break;
  }
  const ulong base = (ulong(slot) * KH + kh) * D;
  float dot = 0;
  for (uint i = 0; i < D / 32; ++i) {
    const ulong offset = base + lane + i * 32;
#if HAS_TAIL
    float key = float(previous ? tail_k[offset] : (current ? k[offset] : k_pool[offset]));
#else
    float key = float(current ? k[offset] : k_pool[offset]);
#endif
    dot += query[i] * key;
  }
  dot = simd_sum(dot);
  const float next_max = max(maximum, dot);
  const float old_scale = metal::fast::exp(maximum - next_max);
  const float weight = metal::fast::exp(dot - next_max);
  for (uint i = 0; i < D / 32; ++i) {
    const ulong offset = base + lane + i * 32;
#if HAS_TAIL
    float value = float(previous ? tail_v[offset] : (current ? v[offset] : v_pool[offset]));
#else
    float value = float(current ? v[offset] : v_pool[offset]);
#endif
    acc[i] = acc[i] * old_scale + weight * value;
  }
  sum = sum * old_scale + weight;
  maximum = next_max;
}
if (lane == 0) { maxima[warp] = maximum; sums[warp] = sum; }
for (uint i = 0; i < D / 32; ++i) partial[warp * D + lane + i * 32] = acc[i];
threadgroup_barrier(mem_flags::mem_threadgroup);
if (warp == 0) {
  float row_max = -INFINITY;
  for (uint w = 0; w < 8; ++w) row_max = max(row_max, maxima[w]);
  float denominator = 0;
  float factors[8];
  for (uint w = 0; w < 8; ++w) {
    factors[w] = sums[w] == 0 ? 0 : metal::fast::exp(maxima[w] - row_max);
    denominator += sums[w] * factors[w];
  }
  for (uint i = 0; i < D / 32; ++i) {
    float numerator = 0;
    for (uint w = 0; w < 8; ++w) numerator += partial[w * D + lane + i * 32] * factors[w];
    out[qb + lane + i * 32] = T(numerator / denominator);
  }
}
"""


@lru_cache(maxsize=128)
def _kernel(dim, heads, kv_heads, rows, width, slots, scale, tail):
    names = ["q", "k", "v", "k_pool", "v_pool", "table", "requests", "lengths"]
    if tail:
        names.extend(("tail_k", "tail_v"))
    return mx.fast.metal_kernel(
        name="sglang_compiled_radix_tail" if tail else "sglang_compiled_radix",
        input_names=names,
        output_names=["out"],
        header=(
            f"#define HAS_TAIL {int(tail)}\n"
            f"constant uint D={dim}, H={heads}, KH={kv_heads};\n"
            f"constant long ROWS={rows}, WIDTH={width}, SLOTS={slots};\n"
            f"constant float SCALE={scale:.17g};\n"
        ),
        source=_SOURCE,
        ensure_row_contiguous=True,
    )


def radix_decode(q, k, v, kp, vp, table, requests, lengths, scale, *, tails=None):
    if q.ndim != 3 or k.ndim != 3 or kp.ndim != 3 or table.ndim != 2:
        raise ValueError("Compiled radix requires 3D Q/K/V/pools and a 2D table")
    batch, heads, dim = q.shape
    if (
        batch < 1
        or heads < 1
        or dim not in (64, 128, 256)
        or k.shape[1] <= 0
        or heads % k.shape[1]
        or k.shape != (batch, k.shape[1], dim)
        or v.shape != k.shape
        or kp.shape[1:] != k.shape[1:]
        or vp.shape != kp.shape
        or requests.shape != (batch,)
        or lengths.shape != (batch,)
        or not isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("Unsupported compiled radix attention geometry")
    if (
        q.dtype not in (mx.float32, mx.float16, mx.bfloat16)
        or any(x.dtype != q.dtype for x in (k, v, kp, vp))
        or table.dtype != mx.int32
        or any(x.dtype not in (mx.int32, mx.int64) for x in (requests, lengths))
    ):
        raise ValueError("Unsupported compiled radix attention dtype")
    if tails is not None and (
        len(tails) != 2 or any(x.shape != k.shape or x.dtype != q.dtype for x in tails)
    ):
        raise ValueError("Pending K/V must match the current-token K/V")
    kernel = _kernel(
        dim,
        heads,
        k.shape[1],
        table.shape[0],
        table.shape[1],
        kp.shape[0],
        scale,
        tails is not None,
    )
    inputs = [q, k, v, kp, vp, table, requests, lengths, *(tails or ())]
    return kernel(
        inputs=inputs,
        template=[("T", q.dtype)],
        grid=(256, heads, batch),
        threadgroup=(256, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )[0]
