#if defined(CPU_CAPABILITY_RVV) && defined(__clang__)
#pragma clang riscv intrinsic vector
#endif

#include "common.h"
#include "vec.h"
#include "vector_math.h"

#if defined(CPU_CAPABILITY_RVV)

#include <riscv_vector.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

#include "riscv64/gemm.h"
#include "vector_helpers.h"

namespace {

// EXTEND_BLOCK_N = VLEN/8: must equal vl_max_e64m8 so Stage-1 index gather fits in one vsetvl.
static constexpr int EXTEND_BLOCK_N = static_cast<int>(__riscv_v_fixed_vlen / 8);

// Score tile s_i[BLOCK_M × BLOCK_N] is kept at ~4 KB regardless of VLEN:
//   BLOCK_M × EXTEND_BLOCK_N × sizeof(float) = (8192/VLEN) × (VLEN/8) × 4 = 4096 B
// BLOCK_M is always a multiple of GEMM_TILE_M=4 since VLEN ≥ 128 is a power of 2.
static constexpr int BLOCK_M = 1024 / EXTEND_BLOCK_N;

static int compute_buffer_size_per_thread(int head_size, int head_size_v) {
  constexpr int BLOCK_N = EXTEND_BLOCK_N;
  // Allocation order in AlignedArena:
  // 1. s_i: float[BLOCK_M * BLOCK_N]
  // 2. v_prime: float[BLOCK_M * head_size_v]
  // 3. Btmp: scalar_t[BLOCK_N * max(head_size, head_size_v)]
  // 4. k_trans: scalar_t[head_size * BLOCK_N]
  // 5. v_buf: scalar_t[BLOCK_N * head_size_v]

  // Worst-case element size: sizeof(float)=4 covers BF16/FP16 (2) and FP32 paths.
  constexpr int MAX_ELEM = sizeof(float);

  int size = 0;
  size += BLOCK_M * BLOCK_N * sizeof(float);                      // s_i score tile
  size += BLOCK_M * head_size_v * sizeof(float);                  // v_prime accumulator
  size += BLOCK_N * std::max(head_size, head_size_v) * MAX_ELEM;  // Btmp (Q@K and S@V)
  size += MAX_HEAD_SIZE * BLOCK_N * MAX_ELEM;                     // k_trans transposed key
  size += BLOCK_N * MAX_HEAD_SIZE * MAX_ELEM;                     // v_buf value gather

  // 5 AlignedArena allocations above, each padded to 64-byte cache line alignment.
  constexpr int NUM_ALLOCS = 5;
  return size + 64 * NUM_ALLOCS;
}

inline void softmax_update_row(
    float* row, float* vp_row, float& m_acc, float& l_acc, int n_size, int head_size_v, float logit_cap) {
  size_t vl_max = __riscv_vsetvlmax_e32m8();
  vfloat32m8_t v_max = __riscv_vfmv_v_f_f32m8(-INFINITY, vl_max);
  for (int j = 0; j < n_size; j += vl_max) {
    size_t vl = __riscv_vsetvl_e32m8(n_size - j);
    vfloat32m8_t v_s = __riscv_vle32_v_f32m8(row + j, vl);
    if (logit_cap > 0.0f) {
      v_s = __riscv_vfmul_vf_f32m8(v_s, 1.0f / logit_cap, vl);
      v_s = vftanh_f32m8(v_s, vl);
      v_s = __riscv_vfmul_vf_f32m8(v_s, logit_cap, vl);
    }
    v_max = __riscv_vfmax_vv_f32m8_tu(v_max, v_max, v_s, vl);
    __riscv_vse32_v_f32m8(row + j, v_s, vl);  // Store back potentially capped values
  }
  float m_block =
      __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m8_f32m1(v_max, __riscv_vfmv_s_f_f32m1(-INFINITY, 1), vl_max));

  float m_new = std::max(m_acc, m_block);
  float alpha = expf(m_acc - m_new);
  m_acc = m_new;

  // Rescale accumulated v_prime by alpha = exp(m_old - m_new)
  for (int d = 0; d < head_size_v; d += vl_max) {
    size_t vl = __riscv_vsetvl_e32m8(head_size_v - d);
    vfloat32m8_t v_vp = __riscv_vle32_v_f32m8(vp_row + d, vl);
    v_vp = __riscv_vfmul_vf_f32m8(v_vp, alpha, vl);
    __riscv_vse32_v_f32m8(vp_row + d, v_vp, vl);
  }

  vfloat32m8_t v_l_block = __riscv_vfmv_v_f_f32m8(0.0f, vl_max);
  for (int j = 0; j < n_size; j += vl_max) {
    size_t vl = __riscv_vsetvl_e32m8(n_size - j);
    vfloat32m8_t v_s = __riscv_vle32_v_f32m8(row + j, vl);
    v_s = __riscv_vfsub_vf_f32m8(v_s, m_new, vl);
    vfloat32m8_t v_exp = vfexp_f32m8(v_s, vl);
    __riscv_vse32_v_f32m8(row + j, v_exp, vl);
    v_l_block = __riscv_vfadd_vv_f32m8_tu(v_l_block, v_l_block, v_exp, vl);
  }
  float l_block =
      __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m8_f32m1(v_l_block, __riscv_vfmv_s_f_f32m1(0.0f, 1), vl_max));

  l_acc = l_acc * alpha + l_block;
}

template <typename scalar_t, typename kv_t, typename index_t>
void extend_attention_kernel_impl(
    scalar_t* __restrict__ o_extend,
    const scalar_t* __restrict__ q_extend,
    const scalar_t* __restrict__ k_extend,
    const scalar_t* __restrict__ v_extend,
    const kv_t* __restrict__ k_buffer,
    const kv_t* __restrict__ v_buffer,
    const index_t* __restrict__ req_to_token,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ seq_lens,
    const index_t* __restrict__ extend_seq_lens,
    const index_t* __restrict__ extend_start_loc,
    const void* __restrict__ buffer,
    int batches,
    int num_heads,
    int num_heads_kv,
    int head_size,
    int head_size_v,
    int o_strideM,
    int o_strideH,
    int q_strideM,
    int q_strideH,
    int ke_strideN,
    int ke_strideH,
    int ve_strideN,
    int ve_strideH,
    int k_strideN,
    int k_strideH,
    int v_strideN,
    int v_strideH,
    float scaling,
    float logit_cap,
    int64_t max_num_reqs,
    int64_t max_context_len,
    int64_t max_total_num_tokens,
    int64_t max_len_extend,
    int buffer_size_per_thread) {
  TORCH_CHECK(
      head_size <= MAX_HEAD_SIZE && head_size_v <= MAX_HEAD_SIZE,
      "extend_attention: head_size (",
      head_size,
      ") or head_size_v (",
      head_size_v,
      ") exceeds MAX_HEAD_SIZE (",
      MAX_HEAD_SIZE,
      ")");

  constexpr int BLOCK_N = EXTEND_BLOCK_N;

  const int num_groups = num_heads / num_heads_kv;
  int MB = static_cast<int>(div_up(max_len_extend, static_cast<int64_t>(BLOCK_M)));

  at::parallel_for(0, batches * num_heads * MB, 0, [&](int64_t begin, int64_t end) {
    int64_t b{0}, h{0}, mb{0};
    int64_t batches_i64 = static_cast<int64_t>(batches);
    int64_t num_heads_i64 = static_cast<int64_t>(num_heads);
    int64_t MB_i64 = static_cast<int64_t>(MB);

    data_index_init(begin, b, batches_i64, h, num_heads_i64, mb, MB_i64);

    int tid = at::get_thread_num();
    char* thread_buffer = reinterpret_cast<char*>(const_cast<void*>(buffer)) + tid * buffer_size_per_thread;

    // Buffer allocation using AlignedArena
    AlignedArena arena(thread_buffer);

    // 1. Accumulation buffer (Score)
    float* s_i = arena.alloc<float>(BLOCK_M * BLOCK_N);

    // 2. Accumulation buffer (Value)
    float* v_prime = arena.alloc<float>(BLOCK_M * head_size_v);

    // 3. Temporary buffer for GEMM (B matrix)
    // Size cover both head_size (for Q@K) and head_size_v (for S@V)
    scalar_t* Btmp = arena.alloc<scalar_t>(BLOCK_N * std::max(head_size, head_size_v));

    // 4. Transposed Key buffer
    scalar_t* k_trans_buf_fp = arena.alloc<scalar_t>(MAX_HEAD_SIZE * BLOCK_N);

    // 5. Value buffer
    scalar_t* v_buf_fp = arena.alloc<scalar_t>(BLOCK_N * MAX_HEAD_SIZE);

    // Thread-local softmax accumulators: one entry per query row in the current tile.
    // Sized to BLOCK_M (= 8192/VLEN): VLEN=128→64, VLEN=256→32, VLEN=512→16, VLEN=1024→8.
    alignas(64) float l_acc[BLOCK_M];
    alignas(64) float m_acc[BLOCK_M];

    for (int64_t i = begin; i < end; ++i) {
      int head_kv_id = h / num_groups;
      int64_t seq_len = seq_lens[b];
      int64_t extend_len = static_cast<int64_t>(extend_seq_lens[b]);
      int64_t prefix_len = seq_len - extend_len;
      int64_t start_loc = static_cast<int64_t>(extend_start_loc[b]);
      int64_t req_idx = req_pool_indices[b];

      int m_start = mb * BLOCK_M;
      int m_size = std::min((int64_t)BLOCK_M, extend_len - m_start);

      if (m_size <= 0) {
        data_index_step(b, batches_i64, h, num_heads_i64, mb, MB_i64);
        continue;
      }

      const scalar_t* q_ptr = q_extend + h * q_strideH + (start_loc + m_start) * q_strideM;
      scalar_t* o_ptr = o_extend + h * o_strideH + (start_loc + m_start) * o_strideM;

      fill_stub<float>(l_acc, 0.0f, m_size);
      fill_stub<float>(m_acc, -std::numeric_limits<float>::infinity(), m_size);
      fill_stub<float>(v_prime, 0.0f, m_size * head_size_v);

      // STAGE 1: PREFIX (KV Cache)
      if (prefix_len > 0) {
        for (int n_start = 0; n_start < prefix_len; n_start += BLOCK_N) {
          int n_size = std::min((int64_t)BLOCK_N, prefix_len - n_start);

          if (n_start + BLOCK_N < prefix_len) {
            int next_idx = req_to_token[req_idx * max_context_len + n_start + BLOCK_N];
            const kv_t* k_next = k_buffer + next_idx * k_strideN + head_kv_id * k_strideH;
            __builtin_prefetch(k_next, 0, 1);
          }

          {
            // Single vsetvl covers n_size <= BLOCK_N (vl_max_e64m8 == BLOCK_N for any VLEN)
            size_t vl = __riscv_vsetvl_e64m8(n_size);
            vuint64m8_t v_offsets;

            if constexpr (sizeof(index_t) == 8) {
              vint64m8_t v_idx =
                  __riscv_vle64_v_i64m8((const int64_t*)&req_to_token[req_idx * max_context_len + n_start], vl);
              v_offsets = __riscv_vreinterpret_v_i64m8_u64m8(v_idx);
            } else {
              // index_t is int32_t: zero-extend to 64-bit for byte-offset arithmetic
              size_t vl_32 = __riscv_vsetvl_e32m4(n_size);
              vint32m4_t v_idx32 =
                  __riscv_vle32_v_i32m4((const int32_t*)&req_to_token[req_idx * max_context_len + n_start], vl_32);
              vuint32m4_t v_u32 = __riscv_vreinterpret_v_i32m4_u32m4(v_idx32);
              vuint64m8_t v_u64 = __riscv_vzext_vf2_u64m8(v_u32, vl);
              v_offsets = v_u64;
            }

            size_t strideN_bytes = k_strideN * sizeof(kv_t);
            size_t head_offset_bytes = head_kv_id * k_strideH * sizeof(kv_t);

            v_offsets = __riscv_vmul_vx_u64m8(v_offsets, strideN_bytes, vl);
            v_offsets = __riscv_vadd_vx_u64m8(v_offsets, head_offset_bytes, vl);

            const kv_t* base_ptr = k_buffer;

            for (int d = 0; d < head_size; ++d) {
              if constexpr (sizeof(scalar_t) == 2) {
                vuint16m2_t v_val = __riscv_vluxei64_v_u16m2((const uint16_t*)base_ptr, v_offsets, vl);
                __riscv_vse16_v_u16m2(reinterpret_cast<uint16_t*>(k_trans_buf_fp + d * BLOCK_N), v_val, vl);
              } else {
                vfloat32m4_t v_val = __riscv_vluxei64_v_f32m4((const float*)base_ptr, v_offsets, vl);
                __riscv_vse32_v_f32m4(k_trans_buf_fp + d * BLOCK_N, v_val, vl);
              }

              v_offsets = __riscv_vadd_vx_u64m8(v_offsets, sizeof(kv_t), vl);
            }

            gemm_nt_tiled_transposed(
                q_ptr, k_trans_buf_fp, s_i, m_size, n_size, head_size, q_strideM, BLOCK_N, BLOCK_N, scaling);
          }

          for (int r = 0; r < m_size; ++r) {
            softmax_update_row(
                s_i + r * BLOCK_N, v_prime + r * head_size_v, m_acc[r], l_acc[r], n_size, head_size_v, logit_cap);
          }

          // GEMM: score @ V
          for (int j = 0; j < n_size; ++j) {
            int token_idx = req_to_token[req_idx * max_context_len + n_start + j];
            const kv_t* v_src = v_buffer + token_idx * v_strideN + head_kv_id * v_strideH;
            scalar_t* v_dst = v_buf_fp + j * head_size_v;
            copy_stub(v_dst, v_src, head_size_v);
          }
          gemm_nn_tiled(s_i, v_buf_fp, v_prime, m_size, n_size, head_size_v, BLOCK_N, head_size_v);
        }
      }  // Stage 1 end

      // STAGE 2: EXTEND (Self-Attention)
      int num_keys = std::min((int64_t)(m_start + BLOCK_M), extend_len);
      for (int n_start = 0; n_start < num_keys; n_start += BLOCK_N) {
        int n_size = std::min(BLOCK_N, num_keys - n_start);

        if (n_start + BLOCK_N < num_keys) {
          const scalar_t* next_k =
              k_extend + start_loc * ke_strideN + head_kv_id * ke_strideH + (n_start + BLOCK_N) * ke_strideN;
          __builtin_prefetch(next_k, 0, 1);
        }

        const scalar_t* k_src_base = k_extend + start_loc * ke_strideN + head_kv_id * ke_strideH;

        {
          size_t stride_bytes = ke_strideN * sizeof(scalar_t);
          if (head_size % 4 == 0) {
            for (int d = 0; d < head_size; ++d) {
              const scalar_t* src_d = k_src_base + n_start * ke_strideN + d;

              if constexpr (sizeof(scalar_t) == 2) {
                size_t vl_16 = __riscv_vsetvl_e16m2(n_size);
                vuint16m2_t v_16 = __riscv_vlse16_v_u16m2((const uint16_t*)src_d, stride_bytes, vl_16);
                __riscv_vse16_v_u16m2(reinterpret_cast<uint16_t*>(k_trans_buf_fp + d * BLOCK_N), v_16, vl_16);
              } else {
                size_t vl = __riscv_vsetvl_e32m4(n_size);
                vfloat32m4_t v_col = __riscv_vlse32_v_f32m4((const float*)src_d, stride_bytes, vl);
                __riscv_vse32_v_f32m4(k_trans_buf_fp + d * BLOCK_N, v_col, vl);
              }
            }
          } else {
            for (int j = 0; j < n_size; ++j) {
              const scalar_t* src_tok = k_src_base + (n_start + j) * ke_strideN;
              for (int d = 0; d < head_size; ++d) {
                k_trans_buf_fp[d * BLOCK_N + j] = src_tok[d];
              }
            }
          }
        }

        gemm_nt_tiled_transposed(
            q_ptr, k_trans_buf_fp, s_i, m_size, n_size, head_size, q_strideM, BLOCK_N, BLOCK_N, scaling);

        // Causal mask: key position must be <= query position.
        for (int r = 0; r < m_size; ++r) {
          int m_pos = m_start + r;
          float* row = s_i + r * BLOCK_N;

          // Step 1: apply logit cap to raw scores (all finite here).
          if (logit_cap > 0.0f) {
            size_t vl_max = __riscv_vsetvlmax_e32m8();
            for (int c = 0; c < n_size; c += vl_max) {
              size_t vl = __riscv_vsetvl_e32m8(n_size - c);
              vfloat32m8_t v_s = __riscv_vle32_v_f32m8(row + c, vl);
              v_s = __riscv_vfmul_vf_f32m8(v_s, 1.0f / logit_cap, vl);
              v_s = vftanh_f32m8(v_s, vl);
              v_s = __riscv_vfmul_vf_f32m8(v_s, logit_cap, vl);
              __riscv_vse32_v_f32m8(row + c, v_s, vl);
            }
          }

          // Step 2: apply causal mask (overwrites future positions with -inf).
          size_t vl_max = __riscv_vsetvlmax_e32m2();
          for (int c = 0; c < n_size; c += vl_max) {
            size_t vl = __riscv_vsetvl_e32m2(n_size - c);

            vuint32m2_t vid = __riscv_vid_v_u32m2(vl);
            vuint32m2_t vn_pos = __riscv_vadd_vx_u32m2(vid, n_start + c, vl);

            vbool16_t mask = __riscv_vmsgtu_vx_u32m2_b16(vn_pos, (uint32_t)m_pos, vl);
            vfloat32m2_t v_inf = __riscv_vfmv_v_f_f32m2(-std::numeric_limits<float>::infinity(), vl);
            vfloat32m2_t v_row = __riscv_vle32_v_f32m2(row + c, vl);
            v_row = __riscv_vmerge_vvm_f32m2(v_row, v_inf, mask, vl);
            __riscv_vse32_v_f32m2(row + c, v_row, vl);
          }

          // Step 3: online softmax — cap already applied, pass 0.0f to skip it.
          softmax_update_row(
              s_i + r * BLOCK_N, v_prime + r * head_size_v, m_acc[r], l_acc[r], n_size, head_size_v, 0.0f);
        }

        const scalar_t* v_src_base = v_extend + start_loc * ve_strideN + head_kv_id * ve_strideH;
        for (int j = 0; j < n_size; ++j) {
          const scalar_t* v_src = v_src_base + (n_start + j) * ve_strideN;
          scalar_t* v_dst = v_buf_fp + j * head_size_v;
          copy_stub(v_dst, v_src, head_size_v);
        }

        gemm_nn_tiled(s_i, v_buf_fp, v_prime, m_size, n_size, head_size_v, BLOCK_N, head_size_v);
      }  // Stage 2 end

      // Write Output
      for (int r = 0; r < m_size; ++r) {
        float l_val = l_acc[r];
        // Guard against division by zero when all positions are masked (l_val ≈ 0).
        // Threshold is well above FP32 underflow (~1e-38) but below any real sum.
        float inv_l = (l_val > 1e-6f) ? 1.0f / l_val : 0.0f;
        float* vp_row = v_prime + r * head_size_v;
        scalar_t* o_row = o_ptr + r * o_strideM;

        size_t vl_max = __riscv_vsetvlmax_e32m2();
        for (int d = 0; d < head_size_v; d += vl_max) {
          size_t vl = __riscv_vsetvl_e32m2(head_size_v - d);
          vfloat32m2_t v = __riscv_vle32_v_f32m2(vp_row + d, vl);
          v = __riscv_vfmul_vf_f32m2(v, inv_l, vl);
          store_from_float_m2(o_row + d, v, vl, reinterpret_cast<float*>(Btmp));
        }
      }

      data_index_step(b, batches_i64, h, num_heads_i64, mb, MB_i64);
    }
  });
}

// Shared validation and stride extraction for both entry functions below.
struct ExtendBatchInfo {
  int num_seqs;
  int64_t max_num_reqs, max_context_len, max_total_num_tokens;
  int num_heads, num_heads_kv, head_size, head_size_v;
  int q_strideM, q_strideH;
  int o_strideM, o_strideH;
  int ke_strideN, ke_strideH;
  int ve_strideN, ve_strideH;
  int k_strideN, k_strideH;
  int v_strideN, v_strideH;
  at::ScalarType index_dtype;
};

static ExtendBatchInfo validate_extend_inputs(
    const at::Tensor& q_extend,
    const at::Tensor& k_extend,
    const at::Tensor& v_extend,
    const at::Tensor& o_extend,
    const at::Tensor& k_buffer,
    const at::Tensor& v_buffer,
    const at::Tensor& req_to_token,
    const at::Tensor& req_pool_indices,
    const at::Tensor& seq_lens,
    const at::Tensor& extend_seq_lens,
    const at::Tensor& extend_start_loc) {
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q_extend);
  CHECK_INPUT(o_extend);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k_extend);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(v_extend);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k_buffer);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(v_buffer);

  ExtendBatchInfo p;
  p.num_seqs = seq_lens.size(0);
  p.max_num_reqs = req_to_token.size(0);
  p.max_context_len = req_to_token.size(1);
  p.max_total_num_tokens = k_buffer.size(0);

  p.num_heads = q_extend.size(1);
  p.num_heads_kv = k_extend.size(1);
  p.head_size = q_extend.size(2);
  p.head_size_v = v_extend.size(2);

  p.q_strideM = q_extend.stride(0);
  p.q_strideH = q_extend.stride(1);
  p.o_strideM = o_extend.stride(0);
  p.o_strideH = o_extend.stride(1);
  p.ke_strideN = k_extend.stride(0);
  p.ke_strideH = k_extend.stride(1);
  p.ve_strideN = v_extend.stride(0);
  p.ve_strideH = v_extend.stride(1);
  p.k_strideN = k_buffer.stride(0);
  p.k_strideH = k_buffer.stride(1);
  p.v_strideN = v_buffer.stride(0);
  p.v_strideH = v_buffer.stride(1);

  CHECK_EQ(req_pool_indices.size(0), p.num_seqs);
  CHECK_EQ(extend_seq_lens.size(0), p.num_seqs);
  CHECK_EQ(extend_start_loc.size(0), p.num_seqs);
  CHECK_EQ(v_extend.size(1), p.num_heads_kv);
  CHECK_EQ(k_buffer.size(1), v_buffer.size(1));

  p.index_dtype = req_to_token.scalar_type();
  TORCH_CHECK(
      p.index_dtype == at::kInt || p.index_dtype == at::kLong,
      "extend: expect req_to_token to be int32 or int64, got ",
      p.index_dtype);
  TORCH_CHECK(seq_lens.scalar_type() == at::kLong, "extend: expect req_lens to be int64, got ", seq_lens.scalar_type());
  TORCH_CHECK(
      req_pool_indices.scalar_type() == at::kLong,
      "extend: expect req_pool_indices to be int64, got ",
      req_pool_indices.scalar_type());
  TORCH_CHECK(
      extend_seq_lens.scalar_type() == p.index_dtype && extend_start_loc.scalar_type() == p.index_dtype,
      "extend: expect extend_seq_lens and extend_start_loc to have same dtype as req_to_token.");

  // EXTEND_BLOCK_N is VLEN/8 (compile-time). head_size must be a multiple so that
  // the transposed K buffer (head_size × BLOCK_N) is indexed without padding.
  TORCH_CHECK(
      p.head_size % EXTEND_BLOCK_N == 0,
      "invalid head_size ",
      p.head_size,
      " (must be a multiple of EXTEND_BLOCK_N=",
      EXTEND_BLOCK_N,
      " for this VLEN)");
  TORCH_CHECK(
      p.head_size_v % EXTEND_BLOCK_N == 0,
      "invalid head_size_v ",
      p.head_size_v,
      " (must be a multiple of EXTEND_BLOCK_N=",
      EXTEND_BLOCK_N,
      " for this VLEN)");

  return p;
}

}  // anonymous namespace

void extend_attention_cpu(
    at::Tensor& q_extend,          // [num_tokens,             num_heads,    head_size]
    at::Tensor& k_extend,          // [num_extend_tokens,      num_heads_kv, head_size]
    at::Tensor& v_extend,          // [num_extend_tokens,      num_heads_kv, head_size_v]
    at::Tensor& o_extend,          // [num_tokens,             num_heads,    head_size_v]
    at::Tensor& k_buffer,          // [max_total_num_tokens,   num_heads_kv, head_size]   — mem_manager prefix+extend
    at::Tensor& v_buffer,          // [max_total_num_tokens,   num_heads_kv, head_size_v] — mem_manager prefix+extend
    at::Tensor& req_to_token,      // [max_num_reqs, max_context_len] int32 or int64
    at::Tensor& req_pool_indices,  // [num_seqs] int64
    at::Tensor& seq_lens,          // [num_seqs] int64
    at::Tensor& extend_seq_lens,   // [num_seqs]
    at::Tensor& extend_start_loc,  // [num_seqs]
    int64_t max_len_extend,
    double sm_scale,
    double logit_cap) {
  RECORD_FUNCTION(
      "sgl-kernel::extend_attention_cpu",
      std::vector<c10::IValue>(
          {q_extend,
           k_extend,
           v_extend,
           o_extend,
           k_buffer,
           v_buffer,
           req_to_token,
           req_pool_indices,
           seq_lens,
           extend_seq_lens,
           extend_start_loc}));

  const auto p = validate_extend_inputs(
      q_extend,
      k_extend,
      v_extend,
      o_extend,
      k_buffer,
      v_buffer,
      req_to_token,
      req_pool_indices,
      seq_lens,
      extend_seq_lens,
      extend_start_loc);

  int buffer_size = compute_buffer_size_per_thread(p.head_size, p.head_size_v);
  int num_threads = at::get_num_threads();
  auto buffer = at::empty({num_threads, buffer_size}, q_extend.options().dtype(at::kChar));

  AT_DISPATCH_RVV_TYPES(q_extend.scalar_type(), "extend_attention_kernel", [&] {
    AT_DISPATCH_INDEX_TYPES(p.index_dtype, "extend_attention_kernel_indices", [&] {
      extend_attention_kernel_impl<scalar_t, scalar_t, index_t>(
          o_extend.data_ptr<scalar_t>(),
          q_extend.data_ptr<scalar_t>(),
          k_extend.data_ptr<scalar_t>(),
          v_extend.data_ptr<scalar_t>(),
          k_buffer.data_ptr<scalar_t>(),
          v_buffer.data_ptr<scalar_t>(),
          req_to_token.data_ptr<index_t>(),
          req_pool_indices.data_ptr<int64_t>(),
          seq_lens.data_ptr<int64_t>(),
          extend_seq_lens.data_ptr<index_t>(),
          extend_start_loc.data_ptr<index_t>(),
          buffer.data_ptr(),
          p.num_seqs,
          p.num_heads,
          p.num_heads_kv,
          p.head_size,
          p.head_size_v,
          p.o_strideM,
          p.o_strideH,
          p.q_strideM,
          p.q_strideH,
          p.ke_strideN,
          p.ke_strideH,
          p.ve_strideN,
          p.ve_strideH,
          p.k_strideN,
          p.k_strideH,
          p.v_strideN,
          p.v_strideH,
          (float)sm_scale,
          (float)logit_cap,
          p.max_num_reqs,
          p.max_context_len,
          p.max_total_num_tokens,
          max_len_extend,
          buffer_size);
    });
  });
}

#endif  // CPU_CAPABILITY_RVV
