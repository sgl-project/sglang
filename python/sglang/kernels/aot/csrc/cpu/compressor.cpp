#include <algorithm>
#include <cmath>
#include <vector>

#include "common.h"
#include "vec.h"

namespace {

using fVec = at::vec::Vectorized<float>;

// ──────────────────────────── helpers ────────────────────────────

static inline int64_t compute_state_len(int64_t seq_len, int64_t ratio) {
  return seq_len % ratio + (ratio == 4 ? ratio : 0);
}

// In-place RMS norm on a single row of float data with weight.
static inline void rmsnorm_row(
    float* __restrict__ out,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int64_t dim,
    float eps) {
  float sum_sq = 0.0f;
  int64_t d = 0;
#if defined(CPU_CAPABILITY_AVX512)
  fVec sum_v(0.0f);
  for (; d <= dim - fVec::size(); d += fVec::size()) {
    fVec x = fVec::loadu(input + d);
    sum_v = sum_v + x * x;
  }
  sum_sq = vec_reduce_sum(sum_v);
#endif
  for (; d < dim; ++d) {
    sum_sq += input[d] * input[d];
  }
  float rsqrt_var = 1.0f / std::sqrt(sum_sq / dim + eps);

  d = 0;
#if defined(CPU_CAPABILITY_AVX512)
  fVec scale_v(rsqrt_var);
  for (; d <= dim - fVec::size(); d += fVec::size()) {
    fVec x = fVec::loadu(input + d);
    fVec w = fVec::loadu(weight + d);
    (x * scale_v * w).store(out + d);
  }
#endif
  for (; d < dim; ++d) {
    out[d] = input[d] * rsqrt_var * weight[d];
  }
}

// In-place interleaved rotary embedding on a single row.
// freqs is [rope_dim] float, x is [rope_dim] float.
static inline void
apply_rotary_emb_row_f32(float* __restrict__ x, const float* __restrict__ freqs, int64_t rope_dim, bool inverse) {
  int64_t k = 0;
#if defined(CPU_CAPABILITY_AVX512)
  // Build sign mask for complex multiply
  __m512 sign_mask;
  if (inverse) {
    sign_mask = _mm512_castsi512_ps(_mm512_set_epi32(
        (int)0x80000000,
        0,
        (int)0x80000000,
        0,
        (int)0x80000000,
        0,
        (int)0x80000000,
        0,
        (int)0x80000000,
        0,
        (int)0x80000000,
        0,
        (int)0x80000000,
        0,
        (int)0x80000000,
        0));
  } else {
    sign_mask = _mm512_castsi512_ps(_mm512_set_epi32(
        0,
        (int)0x80000000,
        0,
        (int)0x80000000,
        0,
        (int)0x80000000,
        0,
        (int)0x80000000,
        0,
        (int)0x80000000,
        0,
        (int)0x80000000,
        0,
        (int)0x80000000,
        0,
        (int)0x80000000));
  }
  for (; k <= rope_dim - 16; k += 16) {
    __m512 xv = _mm512_loadu_ps(x + k);
    __m512 fv = _mm512_loadu_ps(freqs + k);
    __m512 out = _mm512_fmadd_ps(
        xv,
        _mm512_permute_ps(fv, 0xA0),
        _mm512_mul_ps(_mm512_permute_ps(xv, 0xB1), _mm512_xor_ps(_mm512_permute_ps(fv, 0xF5), sign_mask)));
    _mm512_storeu_ps(x + k, out);
  }
#endif
  for (; k < rope_dim; k += 2) {
    float xr = x[k], xi = x[k + 1];
    float cr = freqs[k], ci = freqs[k + 1];
    if (inverse) {
      x[k] = xr * cr + xi * ci;
      x[k + 1] = xi * cr - xr * ci;
    } else {
      x[k] = xr * cr - xi * ci;
      x[k + 1] = xr * ci + xi * cr;
    }
  }
}

// Hadamard transform for rotate_activation.
// Operates on float buffer of power-of-2 size.
static void hadamard_transform_row(float* __restrict__ data, int64_t n, float scale) {
  if (n == 1) {
    data[0] *= scale;
    return;
  }
#if defined(CPU_CAPABILITY_AVX512)
  if (n >= 16) {
    // Phase 1: fused butterflies for h=1,2,4,8
    for (int64_t j = 0; j < n; j += 16) {
      __m512 v = _mm512_loadu_ps(data + j);
      __m512 p, s, d;
      // h=1
      p = _mm512_permute_ps(v, 0b10'11'00'01);
      s = _mm512_add_ps(v, p);
      d = _mm512_sub_ps(p, v);
      v = _mm512_mask_blend_ps((__mmask16)0xAAAA, s, d);
      // h=2
      p = _mm512_permute_ps(v, 0b01'00'11'10);
      s = _mm512_add_ps(v, p);
      d = _mm512_sub_ps(p, v);
      v = _mm512_mask_blend_ps((__mmask16)0xCCCC, s, d);
      // h=4
      p = _mm512_permutexvar_ps(_mm512_set_epi32(11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4), v);
      s = _mm512_add_ps(v, p);
      d = _mm512_sub_ps(p, v);
      v = _mm512_mask_blend_ps((__mmask16)0xF0F0, s, d);
      // h=8
      p = _mm512_permutexvar_ps(_mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8), v);
      s = _mm512_add_ps(v, p);
      d = _mm512_sub_ps(p, v);
      v = _mm512_mask_blend_ps((__mmask16)0xFF00, s, d);
      _mm512_storeu_ps(data + j, v);
    }
    // Phase 2: larger butterflies
    for (int64_t h = 16; h < n; h <<= 1) {
      for (int64_t j = 0; j < n; j += 2 * h) {
        for (int64_t k = 0; k < h; k += 16) {
          __m512 a = _mm512_loadu_ps(data + j + k);
          __m512 b = _mm512_loadu_ps(data + j + k + h);
          _mm512_storeu_ps(data + j + k, _mm512_add_ps(a, b));
          _mm512_storeu_ps(data + j + k + h, _mm512_sub_ps(a, b));
        }
      }
    }
    // Apply scale
    __m512 sv = _mm512_set1_ps(scale);
    for (int64_t j = 0; j < n; j += 16) {
      __m512 v = _mm512_loadu_ps(data + j);
      _mm512_storeu_ps(data + j, _mm512_mul_ps(v, sv));
    }
    return;
  }
#endif
  // Scalar fallback
  int64_t h = 1;
  while (h < n) {
    for (int64_t j = 0; j < n; j += 2 * h) {
      for (int64_t k = 0; k < h; ++k) {
        float a = data[j + k], b = data[j + k + h];
        data[j + k] = a + b;
        data[j + k + h] = a - b;
      }
    }
    h <<= 1;
  }
  for (int64_t j = 0; j < n; ++j) {
    data[j] *= scale;
  }
}

// overlap_transform_decode: tensor [bs, 2*ratio, 2*head_dim] -> [bs, 2*ratio, head_dim]
// ret[:, :r, :] = tensor[:, :r, :d]
// ret[:, r:, :] = tensor[:, r:, d:]
// where r = ratio, d = head_dim
static void overlap_transform_decode_inplace(
    float* __restrict__ out,       // [n_items, head_dim]
    const float* __restrict__ in,  // [n_items, 2*head_dim]
    int64_t ratio,
    int64_t head_dim) {
  // First half: copy in[:ratio, :head_dim]
  for (int64_t i = 0; i < ratio; ++i) {
    const float* src = in + i * 2 * head_dim;
    float* dst = out + i * head_dim;
    std::memcpy(dst, src, head_dim * sizeof(float));
  }
  // Second half: copy in[ratio:2*ratio, head_dim:2*head_dim]
  for (int64_t i = ratio; i < 2 * ratio; ++i) {
    const float* src = in + i * 2 * head_dim + head_dim;
    float* dst = out + i * head_dim;
    std::memcpy(dst, src, head_dim * sizeof(float));
  }
}

// ──────────────────── compress_decode_cpu ────────────────────

// Equivalent to compress_decode_old in Python.
// pool_kv/pool_score: [max_reqs, state_len, coff*head_dim]
// kv/score: [bs, coff*head_dim]
// seq_lens: [bs] int64
// req_pool_indices: [bs] int64
// ape: [ratio, coff*head_dim] float32
// norm_weight: [head_dim] float32
// freqs_cis: [max_seq, rope_dim] float32 (real-valued interleaved cos/sin)
void compress_decode_cpu_impl(
    float* __restrict__ pool_kv,
    float* __restrict__ pool_score,
    const float* __restrict__ kv,
    const float* __restrict__ score,
    const int64_t* __restrict__ seq_lens,
    const int64_t* __restrict__ req_pool_indices,
    const float* __restrict__ ape,
    const float* __restrict__ norm_weight,
    const float* __restrict__ freqs_cis,
    float* __restrict__ output,  // [bs, head_dim]
    int64_t bs,
    int64_t pool_state_len,
    int64_t pool_row_stride,  // stride between rows in pool (may be > coff_hd for interleaved storage)
    int64_t kv_row_stride,    // stride between rows in kv/score input
    int64_t ratio,
    int64_t head_dim,
    int64_t rope_head_dim,
    int64_t coff,
    bool overlap,
    bool rotate,
    float norm_eps,
    int64_t freqs_stride) {
  int64_t coff_hd = coff * head_dim;

  at::parallel_for(0, bs, 1, [&](int64_t begin, int64_t end) {
    // Scratch buffers per thread
    std::vector<float> kv_buf(ratio * coff * coff_hd);
    std::vector<float> score_buf(ratio * coff * coff_hd);
    std::vector<float> kv_work(ratio * coff * head_dim);
    std::vector<float> score_work(ratio * coff * head_dim);
    std::vector<float> compressed(head_dim);

    for (int64_t b = begin; b < end; ++b) {
      int64_t seq_len = seq_lens[b];
      int64_t req_idx = req_pool_indices[b];
      int64_t write_pos = (seq_len - 1) % ratio + (overlap ? ratio : 0);

      float* pool_kv_req = pool_kv + req_idx * pool_state_len * pool_row_stride;
      float* pool_score_req = pool_score + req_idx * pool_state_len * pool_row_stride;

      // Write new kv and score to pool at write_pos
      std::memcpy(pool_kv_req + write_pos * pool_row_stride, kv + b * kv_row_stride, coff_hd * sizeof(float));
      std::memcpy(pool_score_req + write_pos * pool_row_stride, score + b * kv_row_stride, coff_hd * sizeof(float));

      // Copy out entire pool state for this request (strided -> contiguous)
      int64_t total_state = ratio * coff;
      for (int64_t r = 0; r < total_state; ++r) {
        std::memcpy(kv_buf.data() + r * coff_hd, pool_kv_req + r * pool_row_stride, coff_hd * sizeof(float));
        std::memcpy(score_buf.data() + r * coff_hd, pool_score_req + r * pool_row_stride, coff_hd * sizeof(float));
      }

      // Handle overlap shift
      if (overlap && (seq_len % ratio == 0)) {
        // Shift: pool[:ratio] = pool[ratio:2*ratio]
        for (int64_t r = 0; r < ratio; ++r) {
          std::memcpy(
              pool_kv_req + r * pool_row_stride, pool_kv_req + (ratio + r) * pool_row_stride, coff_hd * sizeof(float));
          std::memcpy(
              pool_score_req + r * pool_row_stride,
              pool_score_req + (ratio + r) * pool_row_stride,
              coff_hd * sizeof(float));
        }
      }

      // Add APE to score
      // kv_buf/score_buf shape: [coff, ratio, coff_hd] (after reshape from [total_state, coff_hd])
      // APE shape: [ratio, coff_hd]
      for (int64_t c = 0; c < coff; ++c) {
        for (int64_t r = 0; r < ratio; ++r) {
          float* sp = score_buf.data() + (c * ratio + r) * coff_hd;
          const float* ap = ape + r * coff_hd;
          int64_t d = 0;
#if defined(CPU_CAPABILITY_AVX512)
          for (; d <= coff_hd - 16; d += 16) {
            __m512 sv = _mm512_loadu_ps(sp + d);
            __m512 av = _mm512_loadu_ps(ap + d);
            _mm512_storeu_ps(sp + d, _mm512_add_ps(sv, av));
          }
#endif
          for (; d < coff_hd; ++d) {
            sp[d] += ap[d];
          }
        }
      }

      if (overlap) {
        // overlap_transform_decode on kv and score
        // Input: [bs=1, coff*ratio=2*ratio, coff*head_dim=2*head_dim]
        // Output: [bs=1, 2*ratio, head_dim]
        overlap_transform_decode_inplace(kv_work.data(), kv_buf.data(), ratio, head_dim);
        overlap_transform_decode_inplace(score_work.data(), score_buf.data(), ratio, head_dim);
      } else {
        // Just copy, treating as [ratio, head_dim]
        for (int64_t r = 0; r < ratio; ++r) {
          std::memcpy(kv_work.data() + r * head_dim, kv_buf.data() + r * coff_hd, head_dim * sizeof(float));
          std::memcpy(score_work.data() + r * head_dim, score_buf.data() + r * coff_hd, head_dim * sizeof(float));
        }
      }

      // Now: kv_work, score_work are [ratio*coff, head_dim]
      int64_t compress_rows = ratio * coff;

      // Row-major vectorized softmax + weighted sum
      // 1) Find per-column max across rows
      std::memcpy(compressed.data(), score_work.data(), head_dim * sizeof(float));
      for (int64_t r = 1; r < compress_rows; ++r) {
        const float* sr = score_work.data() + r * head_dim;
        int64_t d = 0;
#if defined(CPU_CAPABILITY_AVX512)
        for (; d <= head_dim - 16; d += 16) {
          __m512 mx = _mm512_loadu_ps(compressed.data() + d);
          __m512 sv = _mm512_loadu_ps(sr + d);
          _mm512_storeu_ps(compressed.data() + d, _mm512_max_ps(mx, sv));
        }
#endif
        for (; d < head_dim; ++d) {
          compressed[d] = std::max(compressed[d], sr[d]);
        }
      }
      // 2) Compute exp(score - max) in-place in score_work, accumulate sum
      // Use kv_buf as sum accumulator (reuse buffer, head_dim <= kv_buf size)
      float* sum_buf = kv_buf.data();  // reuse, only need head_dim floats
      std::memset(sum_buf, 0, head_dim * sizeof(float));
      for (int64_t r = 0; r < compress_rows; ++r) {
        float* sr = score_work.data() + r * head_dim;
        int64_t d = 0;
#if defined(CPU_CAPABILITY_AVX512)
        for (; d <= head_dim - 16; d += 16) {
          fVec diff_v = fVec::loadu(sr + d) - fVec::loadu(compressed.data() + d);
          fVec exp_v = diff_v.exp();
          exp_v.store(sr + d);
          __m512 sm = _mm512_loadu_ps(sum_buf + d);
          _mm512_storeu_ps(sum_buf + d, _mm512_add_ps(sm, (__m512)exp_v));
        }
#endif
        for (; d < head_dim; ++d) {
          sr[d] = std::exp(sr[d] - compressed[d]);
          sum_buf[d] += sr[d];
        }
      }
      // 3) Pre-compute inv_sum, then weighted sum with division fused
      // Compute inv_sum = 1/sum in-place in sum_buf
      {
        int64_t d = 0;
#if defined(CPU_CAPABILITY_AVX512)
        for (; d <= head_dim - 16; d += 16) {
          __m512 sm = _mm512_loadu_ps(sum_buf + d);
          _mm512_storeu_ps(sum_buf + d, _mm512_rcp14_ps(sm));
        }
#endif
        for (; d < head_dim; ++d) {
          sum_buf[d] = 1.0f / sum_buf[d];
        }
      }
      std::memset(compressed.data(), 0, head_dim * sizeof(float));
      for (int64_t r = 0; r < compress_rows; ++r) {
        const float* kr = kv_work.data() + r * head_dim;
        const float* sr = score_work.data() + r * head_dim;
        int64_t d = 0;
#if defined(CPU_CAPABILITY_AVX512)
        for (; d <= head_dim - 16; d += 16) {
          __m512 kv = _mm512_loadu_ps(kr + d);
          __m512 sw = _mm512_mul_ps(_mm512_loadu_ps(sr + d), _mm512_loadu_ps(sum_buf + d));
          __m512 acc = _mm512_loadu_ps(compressed.data() + d);
          _mm512_storeu_ps(compressed.data() + d, _mm512_fmadd_ps(kv, sw, acc));
        }
#endif
        for (; d < head_dim; ++d) {
          compressed[d] += kr[d] * sr[d] * sum_buf[d];
        }
      }

      // RMS norm with weight
      rmsnorm_row(output + b * head_dim, compressed.data(), norm_weight, head_dim, norm_eps);

      // Apply rotary embedding to last rope_head_dim elements
      int64_t freq_pos = (seq_len - 1) / ratio * ratio;
      const float* freq_ptr = freqs_cis + freq_pos * freqs_stride;
      apply_rotary_emb_row_f32(output + b * head_dim + (head_dim - rope_head_dim), freq_ptr, rope_head_dim, false);

      // Optional rotate (Hadamard transform)
      if (rotate) {
        float scale = 1.0f / std::sqrt((float)head_dim);
        hadamard_transform_row(output + b * head_dim, head_dim, scale);
      }
    }
  });
}

}  // anonymous namespace

at::Tensor compress_decode_cpu(
    at::Tensor& pool_kv,
    at::Tensor& pool_score,
    at::Tensor& kv,
    at::Tensor& score,
    at::Tensor& seq_lens,
    at::Tensor& req_pool_indices,
    at::Tensor& ape,
    at::Tensor& norm_weight,
    at::Tensor& freqs_cis,
    int64_t ratio,
    int64_t head_dim,
    int64_t rope_head_dim,
    bool overlap,
    bool rotate,
    double norm_eps) {
  int64_t bs = kv.size(0);
  int64_t coff = 1 + (overlap ? 1 : 0);
  int64_t pool_state_len = pool_kv.size(1);

  // Ensure float32
  TORCH_CHECK(kv.scalar_type() == at::kFloat, "compress_decode_cpu: kv must be float32");
  TORCH_CHECK(score.scalar_type() == at::kFloat, "compress_decode_cpu: score must be float32");
  TORCH_CHECK(pool_kv.scalar_type() == at::kFloat, "compress_decode_cpu: pool_kv must be float32");

  at::Tensor output = at::empty({bs, head_dim}, kv.options());

  // freqs_cis stride: number of elements per position
  int64_t freqs_stride = freqs_cis.size(-1);
  // Pool row stride (may be > coff_hd if pool is a non-contiguous view)
  int64_t pool_row_stride = pool_kv.stride(-2);
  // kv/score row stride
  int64_t kv_row_stride = kv.stride(0);

  compress_decode_cpu_impl(
      pool_kv.data_ptr<float>(),
      pool_score.data_ptr<float>(),
      kv.data_ptr<float>(),
      score.data_ptr<float>(),
      seq_lens.data_ptr<int64_t>(),
      req_pool_indices.data_ptr<int64_t>(),
      ape.data_ptr<float>(),
      norm_weight.data_ptr<float>(),
      freqs_cis.data_ptr<float>(),
      output.data_ptr<float>(),
      bs,
      pool_state_len,
      pool_row_stride,
      kv_row_stride,
      ratio,
      head_dim,
      rope_head_dim,
      coff,
      overlap,
      rotate,
      static_cast<float>(norm_eps),
      freqs_stride);

  return output;
}
