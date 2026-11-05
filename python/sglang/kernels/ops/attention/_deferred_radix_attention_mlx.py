"""Deferred-commit GQA radix attention for an MLX island.

Kernels are templated per :class:`DeferredAttentionSpec` (query/KV head
counts and head dim), so any uniform-GQA softmax model shares this module.

This module deliberately has an MLX-array contract.  A caller may borrow the
Torch-owned Radix KV pool through :class:`MlxTensorView`, run the complete MLX
island, and commit the newly produced K/V tensors only after the island has
finished.  The kernel therefore treats both pool tensors as read-only and
uses ``current_k``/``current_v`` for the final logical token.

The returned array is lazy and newly allocated.  This module does not call
``mx.eval`` and does not synchronize either the Torch or MLX runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx


_DECODE_NUM_THREADS = 1024
_SIMD_SIZE = 32
_NUM_SIMDGROUPS = _DECODE_NUM_THREADS // _SIMD_SIZE


@dataclass(frozen=True)
class DeferredAttentionSpec:
    num_q_heads: int
    num_kv_heads: int
    head_dim: int

    @property
    def attention_scale(self) -> float:
        return self.head_dim**-0.5


def deferred_attention_reject_reason(
    *, num_q_heads: int, num_kv_heads: int, head_dim: int
) -> str | None:
    """Why this head geometry cannot launch the deferred kernels, or None.

    One simdgroup owns one token: each of its ``_SIMD_SIZE`` lanes carries
    ``head_dim / _SIMD_SIZE`` adjacent components (``VALUES_PER_LANE`` is an
    integer division baked into the Metal source), so ``head_dim`` must be a
    positive multiple of the simdgroup width. GQA fan-out is likewise baked
    as the integer ``NUM_Q_HEADS / NUM_KV_HEADS``. Callers must check this
    before any batch reaches device work; the kernel entry points also raise
    on it as a last-resort defense.
    """
    if head_dim <= 0 or head_dim % _SIMD_SIZE != 0:
        return (
            f"head_dim {head_dim} is not a positive multiple of the Metal "
            f"simdgroup width {_SIMD_SIZE}"
        )
    if num_q_heads <= 0 or num_kv_heads <= 0:
        return (
            f"head counts must be positive, found {num_q_heads} query and "
            f"{num_kv_heads} KV heads"
        )
    if num_q_heads % num_kv_heads != 0:
        return (
            f"{num_q_heads} query heads do not divide evenly over "
            f"{num_kv_heads} KV heads"
        )
    return None


def _kernel_header(spec) -> str:
    return f"""
#define HEAD_DIM {spec.head_dim}
#define NUM_Q_HEADS {spec.num_q_heads}
#define NUM_KV_HEADS {spec.num_kv_heads}
#define Q_HEADS_PER_KV_HEAD \
  (NUM_Q_HEADS / NUM_KV_HEADS)
#define DECODE_NUM_THREADS {_DECODE_NUM_THREADS}
#define SIMD_SIZE {_SIMD_SIZE}
#define NUM_SIMDGROUPS {_NUM_SIMDGROUPS}
#define VALUES_PER_LANE (HEAD_DIM / SIMD_SIZE)
#define ATTENTION_SCALE {spec.attention_scale:.12g}f

inline float radix_simd_max_32(float value) {{
  value = max(value, simd_shuffle_xor(value, ushort(16)));
  value = max(value, simd_shuffle_xor(value, ushort(8)));
  value = max(value, simd_shuffle_xor(value, ushort(4)));
  value = max(value, simd_shuffle_xor(value, ushort(2)));
  return max(value, simd_shuffle_xor(value, ushort(1)));
}}

inline float radix_simd_sum_32(float value) {{
  value += simd_shuffle_xor(value, ushort(16));
  value += simd_shuffle_xor(value, ushort(8));
  value += simd_shuffle_xor(value, ushort(4));
  value += simd_shuffle_xor(value, ushort(2));
  return value + simd_shuffle_xor(value, ushort(1));
}}

"""


_SOURCE = r"""
// One simdgroup owns one token at a time.  Its 32 lanes split HEAD_DIM into
// four adjacent values each, keeping Q and the online-softmax accumulator in
// registers while making every K/V access coalesced inside one Radix slot.
threadgroup float partial_maxes[NUM_SIMDGROUPS];
threadgroup float partial_sums[NUM_SIMDGROUPS];
// Reused as a 32x32 transpose tile for each of VALUES_PER_LANE components.
threadgroup float partial_outputs[NUM_SIMDGROUPS * SIMD_SIZE];

const uint batch = threadgroup_position_in_grid.z;
const uint q_head = threadgroup_position_in_grid.y;
const uint kv_head = q_head / Q_HEADS_PER_KV_HEAD;
const uint warp = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const long sequence_length = seq_lens[batch];
const long raw_request = req_pool_indices[batch];
const uint q_base = (batch * NUM_Q_HEADS + q_head) * HEAD_DIM;

if (raw_request < 0 || ulong(raw_request) >= ulong(REQUEST_ROWS) ||
    sequence_length <= 0 || sequence_length > long(TABLE_STRIDE)) {
  if (lane == 0) {
    const uint output_base = warp * VALUES_PER_LANE;
    for (uint index = 0; index < VALUES_PER_LANE; ++index) {
      out[q_base + output_base + index] = static_cast<T>(0.0f);
    }
  }
  return;
}
const ulong request = ulong(raw_request);

const uint lane_dimension = lane * VALUES_PER_LANE;
float query_values[VALUES_PER_LANE];
float local_accumulators[VALUES_PER_LANE];
for (uint index = 0; index < VALUES_PER_LANE; ++index) {
  query_values[index] =
      float(q[q_base + lane_dimension + index]) * ATTENTION_SCALE;
  local_accumulators[index] = 0.0f;
}

float local_max = -INFINITY;
float local_sum = 0.0f;
for (long token = long(warp);
     token < sequence_length;
     token += NUM_SIMDGROUPS) {
  const bool is_current = token == sequence_length - 1;
  ulong kv_base = (ulong(batch) * NUM_KV_HEADS + kv_head) * HEAD_DIM;
  if (!is_current) {
    int slot = -1;
    if (lane == 0) {
      slot = req_to_token[
          request * ulong(TABLE_STRIDE) + ulong(token)];
    }
    // All lanes consume the same token.  Broadcast one table lookup instead
    // of issuing 32 identical indirect reads.
    slot = simd_shuffle(slot, ushort(0));
    if (slot < 0 || uint(slot) >= POOL_SLOTS) {
      continue;
    }
    kv_base = (ulong(slot) * NUM_KV_HEADS + kv_head) * HEAD_DIM;
  }

  float logit = 0.0f;
  for (uint index = 0; index < VALUES_PER_LANE; ++index) {
    const ulong dimension = ulong(lane_dimension + index);
    const float key_value = is_current
        ? float(current_k[kv_base + dimension])
        : float(k_pool[kv_base + dimension]);
    logit += query_values[index] * key_value;
  }
  logit = radix_simd_sum_32(logit);

  const float new_max = max(local_max, logit);
  const float old_scale = metal::fast::exp(local_max - new_max);
  const float weight = metal::fast::exp(logit - new_max);
  local_sum = local_sum * old_scale + weight;
  for (uint index = 0; index < VALUES_PER_LANE; ++index) {
    const ulong dimension = ulong(lane_dimension + index);
    const float value = is_current
        ? float(current_v[kv_base + dimension])
        : float(v_pool[kv_base + dimension]);
    local_accumulators[index] =
        local_accumulators[index] * old_scale + weight * value;
  }
  local_max = new_max;
}

if (lane == 0) {
  partial_maxes[warp] = local_max;
  partial_sums[warp] = local_sum;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// Every simdgroup performs the same 32-way state merge.  Lane N represents
// partial state N, so its scale can be reused for all four output components.
const float partial_max = partial_maxes[lane];
const float row_max = radix_simd_max_32(partial_max);
const bool row_has_tokens = row_max != -INFINITY;
const float partial_scale =
    row_has_tokens && partial_max != -INFINITY
    ? metal::fast::exp(partial_max - row_max)
    : 0.0f;
const float row_sum = row_has_tokens
    ? radix_simd_sum_32(partial_sums[lane] * partial_scale)
    : 0.0f;

float merged_values[VALUES_PER_LANE];
for (uint index = 0; index < VALUES_PER_LANE; ++index) {
  // Transpose [source simdgroup, dimension lane] through threadgroup memory.
  // The destination simdgroup then owns one four-value output segment while
  // its lanes enumerate all 32 source partials for the final reduction.
  partial_outputs[lane * NUM_SIMDGROUPS + warp] =
      local_accumulators[index];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float numerator = radix_simd_sum_32(
      partial_outputs[warp * SIMD_SIZE + lane] * partial_scale);
  merged_values[index] = row_sum == 0.0f ? 0.0f : numerator / row_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (lane == 0) {
  const uint output_base = warp * VALUES_PER_LANE;
  for (uint index = 0; index < VALUES_PER_LANE; ++index) {
    out[q_base + output_base + index] =
        static_cast<T>(merged_values[index]);
  }
}
"""

_PREFILL_SOURCE = r"""
// One threadgroup owns one packed query token. The group derives its request
// and in-request offset from extend_seq_lens, then reads cached prefix rows
// through Radix and new rows directly from the packed K/V inputs.
threadgroup float partial_maxes[NUM_SIMDGROUPS];
threadgroup float partial_sums[NUM_SIMDGROUPS];
threadgroup float partial_outputs[NUM_SIMDGROUPS * SIMD_SIZE];

const uint packed_token = threadgroup_position_in_grid.z;
const uint q_head = threadgroup_position_in_grid.y;
const uint kv_head = q_head / Q_HEADS_PER_KV_HEAD;
const uint warp = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;

long sequence_start = 0;
uint batch = REQUEST_COUNT;
for (uint candidate = 0; candidate < REQUEST_COUNT; ++candidate) {
  const long extension_length = extend_seq_lens[candidate];
  if (extension_length < 0) {
    continue;
  }
  if (long(packed_token) < sequence_start + extension_length) {
    batch = candidate;
    break;
  }
  sequence_start += extension_length;
}

const uint q_base = (packed_token * NUM_Q_HEADS + q_head) * HEAD_DIM;
if (batch == REQUEST_COUNT) {
  if (lane == 0) {
    const uint output_base = warp * VALUES_PER_LANE;
    for (uint index = 0; index < VALUES_PER_LANE; ++index) {
      out[q_base + output_base + index] = static_cast<T>(0.0f);
    }
  }
  return;
}

const long prefix_length = extend_prefix_lens[batch];
const long extension_length = extend_seq_lens[batch];
const long in_request_offset = long(packed_token) - sequence_start;
const long sequence_length = prefix_length + in_request_offset + 1;
const long raw_request = req_pool_indices[batch];
if (raw_request < 0 || ulong(raw_request) >= ulong(REQUEST_ROWS) ||
    prefix_length < 0 || prefix_length > long(TABLE_STRIDE) ||
    in_request_offset < 0 || in_request_offset >= extension_length) {
  if (lane == 0) {
    const uint output_base = warp * VALUES_PER_LANE;
    for (uint index = 0; index < VALUES_PER_LANE; ++index) {
      out[q_base + output_base + index] = static_cast<T>(0.0f);
    }
  }
  return;
}
const ulong request = ulong(raw_request);

const uint lane_dimension = lane * VALUES_PER_LANE;
float query_values[VALUES_PER_LANE];
float local_accumulators[VALUES_PER_LANE];
for (uint index = 0; index < VALUES_PER_LANE; ++index) {
  query_values[index] =
      float(q[q_base + lane_dimension + index]) * ATTENTION_SCALE;
  local_accumulators[index] = 0.0f;
}

float local_max = -INFINITY;
float local_sum = 0.0f;
for (long token = long(warp);
     token < sequence_length;
     token += NUM_SIMDGROUPS) {
  const bool is_new_token = token >= prefix_length;
  ulong kv_base;
  if (is_new_token) {
    const long source_token = sequence_start + token - prefix_length;
    kv_base = (ulong(source_token) * NUM_KV_HEADS + kv_head) * HEAD_DIM;
  } else {
    int slot = -1;
    if (lane == 0) {
      slot = req_to_token[request * ulong(TABLE_STRIDE) + ulong(token)];
    }
    slot = simd_shuffle(slot, ushort(0));
    if (slot < 0 || uint(slot) >= POOL_SLOTS) {
      continue;
    }
    kv_base = (ulong(slot) * NUM_KV_HEADS + kv_head) * HEAD_DIM;
  }

  float logit = 0.0f;
  for (uint index = 0; index < VALUES_PER_LANE; ++index) {
    const ulong dimension = ulong(lane_dimension + index);
    const float key_value = is_new_token
        ? float(current_k[kv_base + dimension])
        : float(k_pool[kv_base + dimension]);
    logit += query_values[index] * key_value;
  }
  logit = radix_simd_sum_32(logit);

  const float new_max = max(local_max, logit);
  const float old_scale = metal::fast::exp(local_max - new_max);
  const float weight = metal::fast::exp(logit - new_max);
  local_sum = local_sum * old_scale + weight;
  for (uint index = 0; index < VALUES_PER_LANE; ++index) {
    const ulong dimension = ulong(lane_dimension + index);
    const float value = is_new_token
        ? float(current_v[kv_base + dimension])
        : float(v_pool[kv_base + dimension]);
    local_accumulators[index] =
        local_accumulators[index] * old_scale + weight * value;
  }
  local_max = new_max;
}

if (lane == 0) {
  partial_maxes[warp] = local_max;
  partial_sums[warp] = local_sum;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

const float partial_max = partial_maxes[lane];
const float row_max = radix_simd_max_32(partial_max);
const bool row_has_tokens = row_max != -INFINITY;
const float partial_scale =
    row_has_tokens && partial_max != -INFINITY
    ? metal::fast::exp(partial_max - row_max)
    : 0.0f;
const float row_sum = row_has_tokens
    ? radix_simd_sum_32(partial_sums[lane] * partial_scale)
    : 0.0f;

float merged_values[VALUES_PER_LANE];
for (uint index = 0; index < VALUES_PER_LANE; ++index) {
  partial_outputs[lane * NUM_SIMDGROUPS + warp] = local_accumulators[index];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float numerator = radix_simd_sum_32(
      partial_outputs[warp * SIMD_SIZE + lane] * partial_scale);
  merged_values[index] = row_sum == 0.0f ? 0.0f : numerator / row_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (lane == 0) {
  const uint output_base = warp * VALUES_PER_LANE;
  for (uint index = 0; index < VALUES_PER_LANE; ++index) {
    out[q_base + output_base + index] = static_cast<T>(merged_values[index]);
  }
}
"""

# Tiled prefill: one threadgroup owns TILE_Q new-token rows of one request for
# one query head and one KV split.  Each TILE_K-key K/V tile is staged once in
# threadgroup memory and shared by all TILE_Q rows, so KV traffic drops by
# TILE_Q versus the per-token kernel above.  4 simdgroups x 8 rows; Q, S, P and
# O stay in simdgroup-matrix registers (MLX steel-attention fragment layout).
_TILED_PREFILL_BODY = r"""
constexpr int NT = 128, PAD = 8, LD = HEAD_DIM + PAD;
constexpr int TD = HEAD_DIM / 8, TK = TILE_K / 8;
constexpr int VEC = 8, VPR = HEAD_DIM / VEC, RPP = NT / VPR;
// Finite, not -INFINITY: a row can be fully masked inside one split, and
// -INFINITY - -INFINITY would make that split's rescale NaN.
constexpr float NEG = -1.0e30f;
static_assert(NT % VPR == 0, "HEAD_DIM / 8 must divide the thread count");
static_assert(2 * TILE_K >= TILE_Q, "Q staging must fit in the K/V tiles");
typedef vec<T, VEC> TV;

threadgroup T smem[2 * TILE_K * LD];  // Q staging, then the K | V tiles
threadgroup T* Ks = smem;
threadgroup T* Vs = smem + TILE_K * LD;
threadgroup uchar key_valid[TILE_K];

const uint tid = thread_position_in_threadgroup.x;
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const int tile = int(threadgroup_position_in_grid.x);
const int h = int(threadgroup_position_in_grid.y);
const int split = int(threadgroup_position_in_grid.z);
const int kvh = h / Q_HEADS_PER_KV_HEAD;
const int rb = int(sg) * 8;
const int rr = int(tid) / VPR, cv = int(tid) % VPR;

// The grid is sized from shapes alone (the caller is mx.compile'd), so it can
// over-cover the ragged requests; tiles past the last request's rows exit and
// their rows keep the zero init.
long sequence_start = 0;
int tile_start = 0;
int batch = -1;
for (int candidate = 0; candidate < REQUEST_COUNT; ++candidate) {
  const long extension_length = extend_seq_lens[candidate];
  if (extension_length <= 0) {
    continue;
  }
  const int tiles = int((extension_length + TILE_Q - 1) / TILE_Q);
  if (tile < tile_start + tiles) {
    batch = candidate;
    break;
  }
  tile_start += tiles;
  sequence_start += extension_length;
}
if (batch < 0) {
  return;
}
const long prefix_length = extend_prefix_lens[batch];
const long raw_request = req_pool_indices[batch];
if (raw_request < 0 || ulong(raw_request) >= ulong(REQUEST_ROWS) ||
    prefix_length < 0 || prefix_length > long(TABLE_STRIDE)) {
  return;
}
const ulong request = ulong(raw_request);
const int P = int(prefix_length);
// Never touch packed rows past q's extent, even if the metadata claims more.
const int N = int(
    min(long(extend_seq_lens[batch]), long(q_shape[0]) - sequence_start));
const int q0 = (tile - tile_start) * TILE_Q;  // first in-request row
if (q0 >= N) {
  return;
}
const long token0 = sequence_start + q0;

for (int r = rr; r < TILE_Q; r += RPP) {
  TV value = TV(0);
  if (q0 + r < N) {
    value = *(const device TV*)(
        q + ((token0 + r) * NUM_Q_HEADS + h) * HEAD_DIM + cv * VEC);
  }
  *(threadgroup TV*)(smem + r * LD + cv * VEC) = value;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
simdgroup_matrix<T, 8, 8> Qm[TD];
simdgroup_matrix<float, 8, 8> Om[TD];
for (int dd = 0; dd < TD; ++dd) {
  simdgroup_load(Qm[dd], smem + rb * LD + dd * 8, LD);
}
// Keep separate from the Qm loop: fusing them measured 2.6x slower on M4.
for (int dd = 0; dd < TD; ++dd) {
  Om[dd] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
}

// MLX BaseMMAFrag<_, 8, 8>::get_coord: lane owns (fm, fn) and (fm, fn + 1).
const short qid = short(lane / 4);
const short fm = (qid & 4) + short((lane / 2) % 4);
const short fn = (qid & 2) * 2 + short(lane % 2) * 2;
const int qpos = P + q0 + rb + fm;  // last key position row fm may attend to
const float scale_log2 = ATTENTION_SCALE * M_LOG2E_F;
float m_i = NEG, l_i = 0.0f;

const int kv_end = P + min(q0 + TILE_Q, N);  // causal horizon of this tile
const int kb_lim = (kv_end + TILE_K - 1) / TILE_K;
const int per = (kb_lim + NUM_SPLITS - 1) / NUM_SPLITS;
const int kb_stop = min(kb_lim, (split + 1) * per);

for (int kb = split * per; kb < kb_stop; ++kb) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int j0 = kb * TILE_K;

  // One table lookup per row, 16-byte vector loads.  Prefix rows come through
  // Radix; new rows come from current_k/current_v (the pool is not committed).
  for (int r = rr; r < TILE_K; r += RPP) {
    const int j = j0 + r;
    TV kv = TV(0), vv = TV(0);
    bool valid = false;
    if (j < kv_end) {
      if (j < P) {
        const int slot = req_to_token[request * ulong(TABLE_STRIDE) + ulong(j)];
        valid = slot >= 0 && uint(slot) < POOL_SLOTS;
        if (valid) {
          const long off = (long(slot) * NUM_KV_HEADS + kvh) * HEAD_DIM + cv * VEC;
          kv = *(const device TV*)(k_pool + off);
          vv = *(const device TV*)(v_pool + off);
        }
      } else {
        valid = true;
        const long off =
            ((sequence_start + j - P) * NUM_KV_HEADS + kvh) * HEAD_DIM + cv * VEC;
        kv = *(const device TV*)(current_k + off);
        vv = *(const device TV*)(current_v + off);
      }
    }
    *(threadgroup TV*)(Ks + r * LD + cv * VEC) = kv;
    *(threadgroup TV*)(Vs + r * LD + cv * VEC) = vv;
    if (cv == 0) {
      key_valid[r] = valid ? 1 : 0;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // S = Q K^T (K^T via a transposed load of the row-major K tile).
  simdgroup_matrix<float, 8, 8> Sm[TK];
  for (int t = 0; t < TK; ++t) {
    Sm[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  }
  for (int dd = 0; dd < TD; ++dd) {
    for (int t = 0; t < TK; ++t) {
      simdgroup_matrix<T, 8, 8> Km;
      simdgroup_load(Km, Ks + (t * 8) * LD + dd * 8, LD, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(Sm[t], Qm[dd], Km, Sm[t]);
    }
  }

  // Online softmax (base 2) in registers; a row's 4 lanes reduce via xor 1, 8.
  // Invalid Radix slots are excluded, matching the per-token kernel.
  float sv[TK][2], tmax = NEG;
  for (int t = 0; t < TK; ++t) {
    vec<float, 2> e = reinterpret_cast<thread vec<float, 2>&>(Sm[t].thread_elements());
    for (int jj = 0; jj < 2; ++jj) {
      const int c = t * 8 + fn + jj;
      const int j = j0 + c;
      sv[t][jj] = (j < kv_end && j <= qpos && key_valid[c]) ? e[jj] * scale_log2 : NEG;
      tmax = max(tmax, sv[t][jj]);
    }
  }
  tmax = max(tmax, simd_shuffle_xor(tmax, ushort(1)));
  tmax = max(tmax, simd_shuffle_xor(tmax, ushort(8)));
  const float m_new = max(m_i, tmax);
  const float corr = fast::exp2(m_i - m_new);

  simdgroup_matrix<T, 8, 8> Pm[TK];
  float psum = 0.0f;
  for (int t = 0; t < TK; ++t) {
    // Masked entries contribute 0, not exp2(NEG - NEG) = 1.
    const float p0 = sv[t][0] > 0.5f * NEG ? fast::exp2(sv[t][0] - m_new) : 0.0f;
    const float p1 = sv[t][1] > 0.5f * NEG ? fast::exp2(sv[t][1] - m_new) : 0.0f;
    psum += p0 + p1;
    reinterpret_cast<thread vec<T, 2>&>(Pm[t].thread_elements()) = vec<T, 2>(T(p0), T(p1));
  }
  psum += simd_shuffle_xor(psum, ushort(1));
  psum += simd_shuffle_xor(psum, ushort(8));
  l_i = l_i * corr + psum;
  m_i = m_new;

  // O = corr * O + P V.
  for (int dd = 0; dd < TD; ++dd) {
    reinterpret_cast<thread vec<float, 2>&>(Om[dd].thread_elements()) *= corr;
  }
  for (int t = 0; t < TK; ++t) {
    for (int dd = 0; dd < TD; ++dd) {
      simdgroup_matrix<T, 8, 8> Vm;
      simdgroup_load(Vm, Vs + (t * 8) * LD + dd * 8, LD);
      simdgroup_multiply_accumulate(Om[dd], Pm[t], Vm, Om[dd]);
    }
  }
}

if (q0 + rb + fm >= N) {
  return;
}
const long token = token0 + rb + fm;
"""

_TILED_PREFILL_TAIL = r"""
device T* o = out + (token * NUM_Q_HEADS + h) * HEAD_DIM + fn;
for (int dd = 0; dd < TD; ++dd) {
  vec<float, 2> e = reinterpret_cast<thread vec<float, 2>&>(Om[dd].thread_elements());
  o[dd * 8] = static_cast<T>(e[0] / l_i);
  o[dd * 8 + 1] = static_cast<T>(e[1] / l_i);
}
"""

# Partials are laid out [token, split, q head] so no kernel needs the token count.
_TILED_PREFILL_SPLIT_TAIL = r"""
const long base = (token * NUM_SPLITS + split) * NUM_Q_HEADS + h;
device float* o = partial_out + base * HEAD_DIM + fn;
for (int dd = 0; dd < TD; ++dd) {
  vec<float, 2> e = reinterpret_cast<thread vec<float, 2>&>(Om[dd].thread_elements());
  o[dd * 8] = e[0];
  o[dd * 8 + 1] = e[1];
}
if (fn == 0) {
  partial_max[base] = m_i;
  partial_sum[base] = l_i;
}
"""

# Merges the unnormalised split partials.  Rows no tile wrote keep the zero
# init (max 0, sum 0) and come out as zeros, like the per-token kernel.
_TILED_PREFILL_REDUCE = r"""
const uint d = thread_position_in_grid.x;
const uint token = thread_position_in_grid.y;
const uint h = thread_position_in_grid.z;
float row_max = -1.0e30f;
for (int s = 0; s < NUM_SPLITS; ++s) {
  row_max = max(row_max, partial_max[(long(token) * NUM_SPLITS + s) * NUM_Q_HEADS + h]);
}
float row_sum = 0.0f, acc = 0.0f;
for (int s = 0; s < NUM_SPLITS; ++s) {
  const long idx = (long(token) * NUM_SPLITS + s) * NUM_Q_HEADS + h;
  const float w = fast::exp2(partial_max[idx] - row_max);
  row_sum += w * partial_sum[idx];
  acc += w * partial_out[idx * HEAD_DIM + d];
}
out[(long(token) * NUM_Q_HEADS + h) * HEAD_DIM + d] =
    static_cast<T>(row_sum > 0.0f ? acc / row_sum : 0.0f);
"""

_TILE_Q = 32
_TILED_THREADS = 128
# Split the KV range across threadgroups only below this many threadgroups.
# Tuned on a 10-core M4; larger GPUs likely want more.
_TILED_TARGET_THREADGROUPS = 32
_TILED_MAX_SPLITS = 16


def _tiled_key_tile(head_dim: int) -> int:
    # Two TILE_K x (HEAD_DIM + 8) bf16 tiles must fit in 32 KB of threadgroup
    # memory; HEAD_DIM 256 needs the smaller tile.
    return 32 if head_dim <= 128 else 16


def _use_tiled_prefill(spec, *, num_tokens: int, request_count: int) -> bool:
    # The loader maps 128 threads onto 8-wide row vectors, so HEAD_DIM / 8 must
    # divide 128.  At two or fewer new tokens per request a TILE_Q-row tile is
    # mostly padding and the per-token kernel is faster (measured on M4).
    return spec.head_dim in (32, 64, 128, 256) and num_tokens > 2 * request_count


def _tiled_kernel_header(spec) -> str:
    return (
        "#include <metal_simdgroup_matrix>\n"
        + _kernel_header(spec)
        + f"#define TILE_Q {_TILE_Q}\n#define TILE_K {_tiled_key_tile(spec.head_dim)}\n"
    )


@lru_cache(maxsize=16)
def _deferred_decode_kernel(spec):
    import mlx.core as mx

    if not mx.metal.is_available():
        raise RuntimeError("Qwen3 deferred decode requires the MLX Metal backend")
    return mx.fast.metal_kernel(
        name=(
            "radix_decode_deferred_bf16_"
            f"q{spec.num_q_heads}_kv{spec.num_kv_heads}_d{spec.head_dim}"
        ),
        input_names=[
            "q",
            "current_k",
            "current_v",
            "k_pool",
            "v_pool",
            "req_to_token",
            "req_pool_indices",
            "seq_lens",
        ],
        output_names=["out"],
        source=_SOURCE,
        header=_kernel_header(spec),
        # Torch has already guaranteed contiguous NHD inputs before exposing
        # the borrowed arrays.  Do not let the MLX wrapper hide a materializing
        # copy if that producer contract regresses.
        ensure_row_contiguous=False,
        compile_options={"math_mode": "safe"},
    )


@lru_cache(maxsize=16)
def _deferred_prefill_kernel(spec):
    import mlx.core as mx

    if not mx.metal.is_available():
        raise RuntimeError("deferred prefill requires the MLX Metal backend")
    return mx.fast.metal_kernel(
        name=(
            "radix_prefill_deferred_bf16_"
            f"q{spec.num_q_heads}_kv{spec.num_kv_heads}_d{spec.head_dim}"
        ),
        input_names=[
            "q",
            "current_k",
            "current_v",
            "k_pool",
            "v_pool",
            "req_to_token",
            "req_pool_indices",
            "extend_prefix_lens",
            "extend_seq_lens",
        ],
        output_names=["out"],
        source=_PREFILL_SOURCE,
        header=_kernel_header(spec),
        ensure_row_contiguous=False,
        compile_options={"math_mode": "safe"},
    )


_PREFILL_INPUT_NAMES = [
    "q",
    "current_k",
    "current_v",
    "k_pool",
    "v_pool",
    "req_to_token",
    "req_pool_indices",
    "extend_prefix_lens",
    "extend_seq_lens",
]


@lru_cache(maxsize=16)
def _tiled_prefill_kernels(spec):
    import mlx.core as mx

    if not mx.metal.is_available():
        raise RuntimeError("deferred prefill requires the MLX Metal backend")
    suffix = f"bf16_q{spec.num_q_heads}_kv{spec.num_kv_heads}_d{spec.head_dim}"
    header = _tiled_kernel_header(spec)
    options = {"math_mode": "safe"}
    single = mx.fast.metal_kernel(
        name=f"radix_prefill_tiled_{suffix}",
        input_names=_PREFILL_INPUT_NAMES,
        output_names=["out"],
        source=_TILED_PREFILL_BODY + _TILED_PREFILL_TAIL,
        header=header,
        ensure_row_contiguous=False,
        compile_options=options,
    )
    split = mx.fast.metal_kernel(
        name=f"radix_prefill_tiled_split_{suffix}",
        input_names=_PREFILL_INPUT_NAMES,
        output_names=["partial_out", "partial_max", "partial_sum"],
        source=_TILED_PREFILL_BODY + _TILED_PREFILL_SPLIT_TAIL,
        header=header,
        ensure_row_contiguous=False,
        compile_options=options,
    )
    reduce = mx.fast.metal_kernel(
        name=f"radix_prefill_tiled_reduce_{suffix}",
        input_names=["partial_out", "partial_max", "partial_sum"],
        output_names=["out"],
        source=_TILED_PREFILL_REDUCE,
        header=header,
        ensure_row_contiguous=False,
        compile_options=options,
    )
    return single, split, reduce


def _prefill_templates(arrays, *, request_count: int) -> list:
    q, _, _, k_pool, _, req_to_token = arrays[:6]
    return [
        ("T", q.dtype),
        ("TABLE_STRIDE", req_to_token.shape[1]),
        ("REQUEST_ROWS", req_to_token.shape[0]),
        ("REQUEST_COUNT", request_count),
        ("POOL_SLOTS", k_pool.shape[0]),
    ]


def _launch_per_token_prefill(arrays, *, spec, request_count: int) -> mx.array:
    q = arrays[0]
    return _deferred_prefill_kernel(spec)(
        inputs=list(arrays),
        template=_prefill_templates(arrays, request_count=request_count),
        grid=(_DECODE_NUM_THREADS, spec.num_q_heads, q.shape[0]),
        threadgroup=(_DECODE_NUM_THREADS, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )[0]


def _launch_tiled_prefill(arrays, *, spec, request_count: int) -> mx.array:
    import mlx.core as mx

    q = arrays[0]
    num_tokens = q.shape[0]
    single, split, reduce = _tiled_prefill_kernels(spec)
    # Upper bound on sum(ceil(n_i / TILE_Q)) over requests with sum(n_i) <= T.
    max_tiles = (num_tokens + request_count * (_TILE_Q - 1)) // _TILE_Q
    num_splits = max(
        1,
        min(
            _TILED_MAX_SPLITS,
            _TILED_TARGET_THREADGROUPS // (max_tiles * spec.num_q_heads),
        ),
    )
    launch = dict(
        template=_prefill_templates(arrays, request_count=request_count)
        + [("NUM_SPLITS", num_splits)],
        grid=(max_tiles * _TILED_THREADS, spec.num_q_heads, num_splits),
        threadgroup=(_TILED_THREADS, 1, 1),
        # Rows no tile owns (invalid metadata, over-covered grid) stay zero.
        init_value=0,
    )
    if num_splits == 1:
        return single(
            inputs=list(arrays),
            output_shapes=[q.shape],
            output_dtypes=[q.dtype],
            **launch,
        )[0]
    partial_rows = (num_tokens, num_splits, spec.num_q_heads)
    partial_out, partial_max, partial_sum = split(
        inputs=list(arrays),
        output_shapes=[(*partial_rows, spec.head_dim), partial_rows, partial_rows],
        output_dtypes=[mx.float32] * 3,
        **launch,
    )
    return reduce(
        inputs=[partial_out, partial_max, partial_sum],
        template=[("T", q.dtype), ("NUM_SPLITS", num_splits)],
        grid=(spec.head_dim, num_tokens, spec.num_q_heads),
        threadgroup=(spec.head_dim, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )[0]


def _require_shape(name: str, array: mx.array, shape: tuple[int, ...]) -> None:
    if tuple(array.shape) != shape:
        raise RuntimeError(
            f"{name} shape mismatch: expected {shape}, found {tuple(array.shape)}"
        )


def radix_decode_deferred(
    q: mx.array,
    current_k: mx.array,
    current_v: mx.array,
    k_pool: mx.array,
    v_pool: mx.array,
    req_to_token: mx.array,
    req_pool_indices: mx.array,
    seq_lens: mx.array,
    *,
    spec,
    scale: float | None = None,
) -> mx.array:
    """Return one-token GQA output without committing the current K/V.

    All arrays must be row-contiguous MLX Metal arrays.  ``k_pool`` and
    ``v_pool`` may be zero-copy, read-only views of Torch-owned MPS tensors.
    The function is intentionally asynchronous: it returns MLX's fresh lazy
    output and leaves the evaluation/export boundary to the enclosing island.
    """
    import mlx.core as mx

    geometry_reject_reason = deferred_attention_reject_reason(
        num_q_heads=spec.num_q_heads,
        num_kv_heads=spec.num_kv_heads,
        head_dim=spec.head_dim,
    )
    if geometry_reject_reason is not None:
        raise RuntimeError(f"deferred decode: {geometry_reject_reason}")
    if scale is None:
        scale = spec.attention_scale
    if not all(
        isinstance(array, mx.array)
        for array in (
            q,
            current_k,
            current_v,
            k_pool,
            v_pool,
            req_to_token,
            req_pool_indices,
            seq_lens,
        )
    ):
        raise RuntimeError("Qwen3 deferred decode inputs must be MLX arrays")
    if not math.isclose(scale, spec.attention_scale, rel_tol=1e-6, abs_tol=0.0):
        raise RuntimeError(
            "attention scale does not match the deferred-attention spec: "
            f"expected {spec.attention_scale}, found {scale}"
        )

    batch_size = q.shape[0] if q.ndim == 3 else -1
    _require_shape("q", q, (batch_size, spec.num_q_heads, spec.head_dim))
    current_shape = (batch_size, spec.num_kv_heads, spec.head_dim)
    _require_shape("current_k", current_k, current_shape)
    _require_shape("current_v", current_v, current_shape)
    if k_pool.ndim != 3 or tuple(k_pool.shape[1:]) != (
        spec.num_kv_heads,
        spec.head_dim,
    ):
        raise RuntimeError(
            "k_pool must have contiguous NHD shape matching the spec, found "
            f"{tuple(k_pool.shape)}"
        )
    _require_shape("v_pool", v_pool, tuple(k_pool.shape))
    if req_to_token.ndim != 2:
        raise RuntimeError(
            f"req_to_token must be 2-D, found {tuple(req_to_token.shape)}"
        )
    _require_shape("req_pool_indices", req_pool_indices, (batch_size,))
    _require_shape("seq_lens", seq_lens, (batch_size,))

    for name, array in (
        ("q", q),
        ("current_k", current_k),
        ("current_v", current_v),
        ("k_pool", k_pool),
        ("v_pool", v_pool),
    ):
        if array.dtype != mx.bfloat16:
            raise RuntimeError(f"{name} must be bfloat16, found {array.dtype}")
    for name, array, expected_dtype in (
        ("req_to_token", req_to_token, mx.int32),
        ("req_pool_indices", req_pool_indices, mx.int64),
        ("seq_lens", seq_lens, mx.int64),
    ):
        if array.dtype != expected_dtype:
            raise RuntimeError(f"{name} must be {expected_dtype}, found {array.dtype}")

    if batch_size == 0:
        return mx.empty(q.shape, dtype=q.dtype)

    return _deferred_decode_kernel(spec)(
        inputs=[
            q,
            current_k,
            current_v,
            k_pool,
            v_pool,
            req_to_token,
            req_pool_indices,
            seq_lens,
        ],
        template=[
            ("T", q.dtype),
            ("TABLE_STRIDE", req_to_token.shape[1]),
            ("REQUEST_ROWS", req_to_token.shape[0]),
            ("POOL_SLOTS", k_pool.shape[0]),
        ],
        grid=(_DECODE_NUM_THREADS, spec.num_q_heads, batch_size),
        threadgroup=(_DECODE_NUM_THREADS, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )[0]


def radix_prefill_deferred(
    q: mx.array,
    current_k: mx.array,
    current_v: mx.array,
    k_pool: mx.array,
    v_pool: mx.array,
    req_to_token: mx.array,
    req_pool_indices: mx.array,
    extend_prefix_lens: mx.array,
    extend_seq_lens: mx.array,
    *,
    spec,
    scale: float | None = None,
) -> mx.array:
    """Run packed causal prefill without materializing the complete KV pool."""
    import mlx.core as mx

    geometry_reject_reason = deferred_attention_reject_reason(
        num_q_heads=spec.num_q_heads,
        num_kv_heads=spec.num_kv_heads,
        head_dim=spec.head_dim,
    )
    if geometry_reject_reason is not None:
        raise RuntimeError(f"deferred prefill: {geometry_reject_reason}")
    if scale is None:
        scale = spec.attention_scale
    arrays = (
        q,
        current_k,
        current_v,
        k_pool,
        v_pool,
        req_to_token,
        req_pool_indices,
        extend_prefix_lens,
        extend_seq_lens,
    )
    if not all(isinstance(array, mx.array) for array in arrays):
        raise RuntimeError("deferred prefill inputs must be MLX arrays")
    if not math.isclose(scale, spec.attention_scale, rel_tol=1e-6, abs_tol=0.0):
        raise RuntimeError(
            "attention scale does not match the deferred-attention spec: "
            f"expected {spec.attention_scale}, found {scale}"
        )

    num_tokens = q.shape[0] if q.ndim == 3 else -1
    _require_shape("q", q, (num_tokens, spec.num_q_heads, spec.head_dim))
    extension_shape = (num_tokens, spec.num_kv_heads, spec.head_dim)
    _require_shape("current_k", current_k, extension_shape)
    _require_shape("current_v", current_v, extension_shape)
    if k_pool.ndim != 3 or tuple(k_pool.shape[1:]) != (
        spec.num_kv_heads,
        spec.head_dim,
    ):
        raise RuntimeError(
            "k_pool must have contiguous NHD shape matching the spec, found "
            f"{tuple(k_pool.shape)}"
        )
    _require_shape("v_pool", v_pool, tuple(k_pool.shape))
    if req_to_token.ndim != 2:
        raise RuntimeError(
            f"req_to_token must be 2-D, found {tuple(req_to_token.shape)}"
        )
    request_count = req_pool_indices.shape[0] if req_pool_indices.ndim == 1 else -1
    _require_shape("req_pool_indices", req_pool_indices, (request_count,))
    _require_shape("extend_prefix_lens", extend_prefix_lens, (request_count,))
    _require_shape("extend_seq_lens", extend_seq_lens, (request_count,))

    for name, array in (
        ("q", q),
        ("current_k", current_k),
        ("current_v", current_v),
        ("k_pool", k_pool),
        ("v_pool", v_pool),
    ):
        if array.dtype != mx.bfloat16:
            raise RuntimeError(f"{name} must be bfloat16, found {array.dtype}")
    for name, array, expected_dtype in (
        ("req_to_token", req_to_token, mx.int32),
        ("req_pool_indices", req_pool_indices, mx.int64),
    ):
        if array.dtype != expected_dtype:
            raise RuntimeError(f"{name} must be {expected_dtype}, found {array.dtype}")
    for name, array in (
        ("extend_prefix_lens", extend_prefix_lens),
        ("extend_seq_lens", extend_seq_lens),
    ):
        if array.dtype not in (mx.int32, mx.int64):
            raise RuntimeError(f"{name} must be int32 or int64, found {array.dtype}")
    if num_tokens == 0:
        return mx.empty(q.shape, dtype=q.dtype)

    # Shapes only: the choice must hold for every call of an mx.compile'd graph.
    if _use_tiled_prefill(spec, num_tokens=num_tokens, request_count=request_count):
        return _launch_tiled_prefill(arrays, spec=spec, request_count=request_count)
    return _launch_per_token_prefill(arrays, spec=spec, request_count=request_count)


__all__ = [
    "DeferredAttentionSpec",
    "deferred_attention_reject_reason",
    "radix_decode_deferred",
    "radix_prefill_deferred",
]
