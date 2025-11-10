#include "common.h"
#include "gemm.h"
#include "pack.h"
#include "vec.h"

namespace {

template <typename scalar_t>
inline void fill_stub(scalar_t* __restrict__ out, float val, int size) {
  using Vec = at::vec::Vectorized<scalar_t>;
  constexpr int kVecSize = Vec::size();
  const Vec data_vec = Vec(static_cast<scalar_t>(val));
  int d = 0;
#pragma GCC unroll 4
  for (; d <= size - kVecSize; d += kVecSize) {
    data_vec.store(out + d);
  }
  if (size - d > 0) {
    data_vec.store(out + d, size - d);
  }
}

template <typename scalar_t, int64_t chunk_size = 64>
void chunk_gated_delta_rule_kernel_impl(
    scalar_t* __restrict__ out,            // [B, T, HV, EV]
    float* __restrict__ final_state_data,  // [N, HV, EK, EV]
    scalar_t* __restrict__ q_orig,         // [B, T, HK, EK]
    scalar_t* __restrict__ k_orig,         // [B, T, HK, EK]
    scalar_t* __restrict__ v_orig,         // [B, T, HV, EV]
    float* __restrict__ g_orig,            // [B, T, HV] FP32
    scalar_t* __restrict__ b_orig,         // [B, T, HV]
    int32_t* __restrict__ cu_seqlens_ptr,  // [N + 1] INT32
    float* __restrict__ buff,
    scalar_t* __restrict__ reduced_buff,
    scalar_t* __restrict__ thread_buff,
    bool use_qk_l2norm_in_kernel,
    const int64_t& batch_size,
    const int64_t& global_seq_len,
    const int64_t& qk_num_head,
    const int64_t& v_num_head,
    const int64_t& qk_head_size,
    const int64_t& v_head_size,
    const int64_t& qStrideH,
    const int64_t& qStrideT,
    const int64_t& kStrideH,
    const int64_t& kStrideT,
    const int64_t& vStrideH,
    const int64_t& vStrideT,
    std::vector<int64_t>& chunk_offsets,
    std::vector<std::vector<int64_t>>& chunk_indices,
    const int64_t& global_total_seq_length,
    const int64_t& global_num_chunk,
    const int64_t& buff_size_16bit_per_thread) {
  int64_t oStrideT = vStrideT;
  int64_t oStrideH = vStrideH;
  int64_t gStrideH = 1;
  int64_t gStrideT = v_num_head;
  int64_t bStrideH = 1;
  int64_t bStrideT = v_num_head;
  int64_t final_state_StrideN = v_num_head * qk_head_size * v_head_size;
  int64_t final_state_StrideH = qk_head_size * v_head_size;
  int64_t final_state_StrideE = v_head_size;
  int64_t head_group = v_num_head / qk_num_head;
  float scale = 1.0 / std::sqrt(qk_head_size);
  const int32_t vec_size = at::vec::Vectorized<float>::size();
  const int32_t reduced_vec_size = at::vec::Vectorized<scalar_t>::size();

  // Data pointers
  float* g_pad = buff;
  float* core_attn_out = g_pad + v_num_head * global_total_seq_length;
  float* decay_mask = core_attn_out + batch_size * v_num_head * global_total_seq_length * v_head_size;
  float* v_beta_attn = decay_mask + v_num_head * global_total_seq_length * chunk_size;

  scalar_t* q_pad = reduced_buff;
  scalar_t* k_pad = q_pad + qk_num_head * global_total_seq_length * qk_head_size;
  scalar_t* v_pad = k_pad + qk_num_head * global_total_seq_length * qk_head_size;
  scalar_t* k_beta = v_pad + v_num_head * global_total_seq_length * v_head_size;
  scalar_t* v_beta = k_beta + v_num_head * global_total_seq_length * qk_head_size;
  scalar_t* k_cumdecay_reduced = v_beta + v_num_head * global_total_seq_length * v_head_size;

  if (use_qk_l2norm_in_kernel) {
    using bVec = at::vec::Vectorized<scalar_t>;
    using fVec = at::vec::Vectorized<float>;
    constexpr int64_t VecSize = bVec::size();
    constexpr int64_t fVecSize = fVec::size();
    float eps = 1e-5;
    at::parallel_for(0, qk_num_head * global_seq_len, 0, [&](int64_t begin, int64_t end) {
      int64_t h_qk = 0, l = 0;
      data_index_init(begin, h_qk, qk_num_head, l, global_seq_len);
      for (int64_t i = begin; i < end; ++i) {
        float sum_q = float(0);
        float sum_k = float(0);
        fVec sum_q_fvec = fVec(float(0));
        fVec sum_k_fvec = fVec(float(0));
        int64_t q_offset = l * qStrideT + h_qk * qStrideH;
        int64_t k_offset = l * qStrideT + h_qk * qStrideH;
        int64_t d;
        for (d = 0; d <= qk_head_size - VecSize; d += VecSize) {
          bVec q_bvec = bVec::loadu(q_orig + q_offset + d);
          fVec q_fvec0, q_fvec1;
          std::tie(q_fvec0, q_fvec1) = at::vec::convert_to_float(q_bvec);
          sum_q_fvec += q_fvec0 * q_fvec0;
          sum_q_fvec += q_fvec1 * q_fvec1;
          bVec k_bvec = bVec::loadu(k_orig + k_offset + d);
          fVec k_fvec0, k_fvec1;
          std::tie(k_fvec0, k_fvec1) = at::vec::convert_to_float(k_bvec);
          sum_k_fvec += k_fvec0 * k_fvec0;
          sum_k_fvec += k_fvec1 * k_fvec1;
        }
        for (; d < qk_head_size; ++d) {
          float q_val = static_cast<float>(q_orig[q_offset + d]);
          sum_q += q_val * q_val;
          float k_val = static_cast<float>(k_orig[k_offset + d]);
          sum_k += k_val * k_val;
        }

        sum_q += vec_reduce_sum(sum_q_fvec);
        sum_k += vec_reduce_sum(sum_k_fvec);
        float q_rsqrt_var = float(1) / std::sqrt(sum_q + eps);
        float k_rsqrt_var = float(1) / std::sqrt(sum_k + eps);
        const fVec q_scale_fvec = fVec(q_rsqrt_var);
        const fVec k_scale_fvec = fVec(k_rsqrt_var);
        for (d = 0; d <= qk_head_size - VecSize; d += VecSize) {
          bVec q_bvec = bVec::loadu(q_orig + q_offset + d);
          fVec q_fvec0, q_fvec1;
          std::tie(q_fvec0, q_fvec1) = at::vec::convert_to_float(q_bvec);

          q_fvec0 = q_fvec0 * q_scale_fvec;
          q_fvec1 = q_fvec1 * q_scale_fvec;
          bVec out_bvec = convert_from_float_ext<scalar_t>(q_fvec0, q_fvec1);
          out_bvec.store(q_orig + q_offset + d);
          bVec k_bvec = bVec::loadu(k_orig + k_offset + d);
          fVec k_fvec0, k_fvec1;
          std::tie(k_fvec0, k_fvec1) = at::vec::convert_to_float(k_bvec);

          k_fvec0 = k_fvec0 * k_scale_fvec;
          k_fvec1 = k_fvec1 * k_scale_fvec;
          out_bvec = convert_from_float_ext<scalar_t>(k_fvec0, k_fvec1);
          out_bvec.store(k_orig + k_offset + d);
        }
        for (; d < qk_head_size; ++d) {
          float q_val = static_cast<float>(q_orig[q_offset + d]);
          float k_val = static_cast<float>(k_orig[k_offset + d]);
          q_orig[q_offset + d] = static_cast<scalar_t>(q_val * q_rsqrt_var);
          k_orig[k_offset + d] = static_cast<scalar_t>(k_val * k_rsqrt_var);
        }

        data_index_step(h_qk, qk_num_head, l, global_seq_len);
      }
    });
  }

  // query = query * scale
  // k_beta = key * beta.unsqueeze(-1)
  // v_beta = value * beta.unsqueeze(-1)
  // Padding for q/k/v/beta
  at::parallel_for(0, qk_num_head * global_num_chunk, 1, [&](int64_t begin, int64_t end) {
    int ompIdx = at::get_thread_num();
    int64_t h_qk = 0, c = 0;
    data_index_init(begin, h_qk, qk_num_head, c, global_num_chunk);
    for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
      int64_t ib = chunk_indices[c][0];  // idx_batch
      int64_t ic = chunk_indices[c][1];  // idx_chunk
      int64_t l_orig = cu_seqlens_ptr[ib] + ic * chunk_size;
      int64_t l = c * chunk_size;
      bool is_tail = (c + 1 == chunk_offsets[ib + 1]);
      int64_t seq_len = cu_seqlens_ptr[ib + 1] - cu_seqlens_ptr[ib];
      int64_t real_chunk_size = is_tail ? seq_len - ic * chunk_size : chunk_size;
      auto q_orig_ptr = q_orig + h_qk * qStrideH + l_orig * qStrideT;
      auto k_orig_ptr = k_orig + h_qk * kStrideH + l_orig * kStrideT;
      auto v_orig_ptr = v_orig + l_orig * vStrideT;
      auto b_orig_ptr = b_orig + l_orig * bStrideT;
      auto q_pad_ptr = q_pad + h_qk * global_total_seq_length * qk_head_size + l * qk_head_size;
      auto k_pad_ptr = k_pad + h_qk * global_total_seq_length * qk_head_size + l * qk_head_size;
      auto v_pad_ptr = v_pad + l * v_head_size;
      auto k_beta_ptr = k_beta + l * qk_head_size;
      auto v_beta_ptr = v_beta + l * v_head_size;

      for (int64_t j = 0; j < real_chunk_size; j++) {
        auto curr_q_orig = q_orig_ptr + j * qStrideT;
        auto curr_k_orig = k_orig_ptr + j * kStrideT;
        auto curr_q_pad = q_pad_ptr + j * qk_head_size;
        auto curr_k_pad = k_pad_ptr + j * qk_head_size;
        int64_t i = 0;
        scalar_t scale_reduced = static_cast<scalar_t>(scale);
        auto vec_scale_reduced = at::vec::Vectorized<scalar_t>(scale_reduced);
        for (; i < reduced_vec_size * (qk_head_size / reduced_vec_size); i += reduced_vec_size) {
          auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(curr_q_orig + i);
          auto tmp2 = tmp0 * vec_scale_reduced;
          tmp2.store(curr_q_pad + i);
          auto tmp3 = at::vec::Vectorized<scalar_t>::loadu(curr_k_orig + i);
          tmp3.store(curr_k_pad + i);
        }
        if (i < qk_head_size) {
          auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(curr_q_orig + i, qk_head_size - i);
          auto tmp2 = tmp0 * vec_scale_reduced;
          tmp2.store(curr_q_pad + i, qk_head_size - i);
          auto tmp3 = at::vec::Vectorized<scalar_t>::loadu(curr_k_orig + i, qk_head_size - i);
          tmp3.store(curr_k_pad + i, qk_head_size - i);
        }

        for (auto hi = 0; hi < head_group; hi++) {
          int64_t h = h_qk * head_group + hi;
          auto curr_v_orig = v_orig_ptr + h * vStrideH + j * vStrideT;
          auto curr_b_orig = b_orig_ptr + h * bStrideH + j * bStrideT;
          scalar_t b_orig_val_reduced = *(curr_b_orig);
          auto curr_v_pad = v_pad_ptr + h * global_total_seq_length * v_head_size + j * v_head_size;
          auto curr_k_beta = k_beta_ptr + h * global_total_seq_length * qk_head_size + j * qk_head_size;
          auto curr_v_beta = v_beta_ptr + h * global_total_seq_length * v_head_size + j * v_head_size;

          // query = query * scale
          // k_beta = key * beta.unsqueeze(-1)
          int64_t i = 0;
          auto vec_b_reduced = at::vec::Vectorized<scalar_t>(b_orig_val_reduced);
          for (; i < reduced_vec_size * (qk_head_size / reduced_vec_size); i += reduced_vec_size) {
            auto tmp3 = at::vec::Vectorized<scalar_t>::loadu(curr_k_orig + i);
            auto tmp5 = tmp3 * vec_b_reduced;
            tmp5.store(curr_k_beta + i);
          }
          if (i < qk_head_size) {
            auto tmp3 = at::vec::Vectorized<scalar_t>::loadu(curr_k_orig + i, qk_head_size - i);
            auto tmp5 = tmp3 * vec_b_reduced;
            tmp5.store(curr_k_beta + i, qk_head_size - i);
          }
          // v_beta = value * beta.unsqueeze(-1)
          i = 0;
          for (; i < reduced_vec_size * (v_head_size / reduced_vec_size); i += reduced_vec_size) {
            auto tmp3 = at::vec::Vectorized<scalar_t>::loadu(curr_v_orig + i);
            tmp3.store(curr_v_pad + i);
            auto tmp5 = tmp3 * vec_b_reduced;
            tmp5.store(curr_v_beta + i);
          }
          if (i < v_head_size) {
            auto tmp3 = at::vec::Vectorized<scalar_t>::loadu(curr_v_orig + i, v_head_size - i);
            tmp3.store(curr_v_pad + i, v_head_size - i);
            auto tmp5 = tmp3 * vec_b_reduced;
            tmp5.store(curr_v_beta + i, v_head_size - i);
          }
        }
      }

      for (int64_t j = real_chunk_size; j < chunk_size; j++) {
        auto curr_q_pad = q_pad_ptr + j * qk_head_size;
        auto curr_k_pad = k_pad_ptr + j * qk_head_size;
        int64_t i = 0;
        auto vec_zero = at::vec::Vectorized<scalar_t>(0.0);
        for (; i < reduced_vec_size * (qk_head_size / reduced_vec_size); i += reduced_vec_size) {
          vec_zero.store(curr_q_pad + i);
          vec_zero.store(curr_k_pad + i);
        }
        if (i < qk_head_size) {
          vec_zero.store(curr_q_pad + i, qk_head_size - i);
          vec_zero.store(curr_k_pad + i, qk_head_size - i);
        }
        for (auto hi = 0; hi < head_group; hi++) {
          int64_t h = h_qk * head_group + hi;
          auto curr_v_pad = v_pad_ptr + h * global_total_seq_length * v_head_size + j * v_head_size;
          auto curr_k_beta = k_beta_ptr + h * global_total_seq_length * qk_head_size + j * qk_head_size;
          auto curr_v_beta = v_beta_ptr + h * global_total_seq_length * v_head_size + j * v_head_size;
          int64_t i = 0;
          for (; i < reduced_vec_size * (qk_head_size / reduced_vec_size); i += reduced_vec_size) {
            vec_zero.store(curr_k_beta + i);
          }
          if (i < qk_head_size) {
            vec_zero.store(curr_k_beta + i, qk_head_size - i);
          }
          i = 0;
          for (; i < reduced_vec_size * (v_head_size / reduced_vec_size); i += reduced_vec_size) {
            vec_zero.store(curr_v_pad + i);
            vec_zero.store(curr_v_beta + i);
          }
          if (i < v_head_size) {
            vec_zero.store(curr_v_pad + i, v_head_size - i);
            vec_zero.store(curr_v_beta + i, v_head_size - i);
          }
        }
      }
      // Move to the next query
      data_index_step(h_qk, qk_num_head, c, global_num_chunk);
    }
  });

  at::parallel_for(0, v_num_head * global_num_chunk, 1, [&](int64_t begin, int64_t end) {
    int64_t h = 0, c = 0;
    data_index_init(begin, h, v_num_head, c, global_num_chunk);
    int ompIdx = at::get_thread_num();
    int64_t offset = 0;
    scalar_t* thread_buff_ptr = thread_buff + ompIdx * buff_size_16bit_per_thread;
    scalar_t* k_transpose = thread_buff_ptr + offset;
    offset += qk_head_size * chunk_size;
    scalar_t* v_pack = thread_buff_ptr + offset;
    offset += chunk_size * v_head_size;
    scalar_t* k_beta_g = thread_buff_ptr + offset;
    offset += chunk_size * qk_head_size;
    scalar_t* k_beta_g_pack = thread_buff_ptr + offset;
    offset += chunk_size * qk_head_size;
    float* curr_attn = reinterpret_cast<float*>(thread_buff_ptr + offset);
    offset += chunk_size * chunk_size * 2;
    scalar_t* curr_attn_reduced = thread_buff_ptr + offset;
    offset += chunk_size * chunk_size;
    float* k_cumdecay = reinterpret_cast<float*>(thread_buff_ptr + offset);
    offset += chunk_size * qk_head_size * 2;
    float* row = reinterpret_cast<float*>(thread_buff_ptr + offset);
    offset += chunk_size * 2;
    float* updated = reinterpret_cast<float*>(thread_buff_ptr + offset);
    for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
      int64_t ib = chunk_indices[c][0];  // idx_batch
      int64_t ic = chunk_indices[c][1];  // idx_chunk
      int64_t l_orig = cu_seqlens_ptr[ib] + ic * chunk_size;
      int64_t seq_len = cu_seqlens_ptr[ib + 1] - cu_seqlens_ptr[ib];
      int64_t h_qk = h / head_group;
      auto curr_g_orig = g_orig + h * gStrideH + l_orig * gStrideT;
      auto curr_g_pad = g_pad + h * global_total_seq_length + c * chunk_size;
      auto curr_decay_mask = decay_mask + h * global_total_seq_length * chunk_size + c * chunk_size * chunk_size;
      auto curr_k_pad = k_pad + h_qk * global_total_seq_length * qk_head_size + c * chunk_size * qk_head_size;
      auto curr_k_beta = k_beta + h * global_total_seq_length * qk_head_size + c * chunk_size * qk_head_size;
      auto curr_k_cumdecay_reduced =
          k_cumdecay_reduced + h * global_total_seq_length * qk_head_size + c * chunk_size * qk_head_size;
      auto curr_v_beta = v_beta + h * global_total_seq_length * v_head_size + c * chunk_size * v_head_size;
      auto curr_value = v_beta_attn + h * global_total_seq_length * v_head_size + c * chunk_size * v_head_size;

      float acc_val = 0;
      for (int64_t i = 0; i < chunk_size; i++) {
        // Padding for g
        // g = g.cumsum(dim=-1)
        // g: [B, HV, num_chunk, chunk_size]
        if (ic * chunk_size + i < seq_len) {
          acc_val += curr_g_orig[i * gStrideT];
        }
        curr_g_pad[i] = acc_val;
        // decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
        // decay_mask: [B, HV, num_chunk, chunk_size, chunk_size]
        float curr_g_pad_i = static_cast<float>(curr_g_pad[i]);
        auto vec_curr_g_pad_i = at::vec::Vectorized<float>(curr_g_pad_i);
        int64_t j = 0;
        int64_t len = i + 1;
        for (; j < vec_size * (len / vec_size); j += vec_size) {
          auto tmp0 = at::vec::Vectorized<float>::loadu(curr_g_pad + j);
          auto tmp1 = vec_curr_g_pad_i - tmp0;
          auto tmp2 = tmp1.exp();
          tmp2.store(curr_decay_mask + i * chunk_size + j);
        }
        if (j < len) {
          auto tmp0 = at::vec::Vectorized<float>::loadu(curr_g_pad + j, len - j);
          auto tmp1 = vec_curr_g_pad_i - tmp0;
          auto tmp2 = tmp1.exp();
          tmp2.store(curr_decay_mask + i * chunk_size + j, len - j);
        }
      }

      // attn = k_beta @ key.transpose(-1, -2)
      // attn: [B, HV, num_chunk, chunk_size, chunk_size]
      // transpose and pack for key
      pack_vnni<scalar_t>(
          /*    dst */ k_transpose,
          /*    src */ curr_k_pad,
          /*     N  */ chunk_size,
          /*     K  */ qk_head_size,
          /* ld_src */ qk_head_size,
          /* ld_dst */ chunk_size);
      // k_beta @ key.transpose(-1, -2)
      at::native::cpublas::brgemm(
          /* M */ chunk_size,
          /* N */ chunk_size,
          /* K */ qk_head_size,
          /* lda */ qk_head_size,
          /* ldb */ chunk_size,
          /* ldc */ chunk_size,
          /* add_C */ false,
          /* A */ curr_k_beta,
          /* B */ k_transpose,
          /* C */ curr_attn);
      // attn = attn * decay_mask
      for (int64_t m = 0; m < chunk_size; m++) {
        at::vec::map2<float>(
            [](at::vec::Vectorized<float> x, at::vec::Vectorized<float> y) {
              return at::vec::Vectorized<float>(0) - x * y;
            },
            curr_attn + m * chunk_size,
            curr_attn + m * chunk_size,
            curr_decay_mask + m * chunk_size,
            chunk_size);
      }

      // chunk decay
      // attn: [B, HV, num_chunk, chunk_size, chunk_size]
      // mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
      // attn = -attn.masked_fill(mask, 0)
      // attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2) [B, HV, num_chunk, i]
      // attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
      // attn = -attn.masked_fill(mask, 0)
      for (int i = 0; i < chunk_size; i++) {
        const auto vec_zero = at::vec::Vectorized<float>(0);
        int64_t len = chunk_size - i;
        int64_t front = len % vec_size;
        int64_t j = i;
        // first masked vec for alignment
        if (front > 0) {
          vec_zero.store(curr_attn + i * chunk_size + j, front);
          j += front;
        }
        for (; j < vec_size * (chunk_size / vec_size); j += vec_size) {
          vec_zero.store(curr_attn + i * chunk_size + j);
        }
      }
      for (int i = 1; i < chunk_size; i++) {
        // row = attn[..., i, :i] [B, HK, num_chunk, i]
        int64_t j = 0;
        int64_t len = i;
        for (; j < vec_size * (len / vec_size); j += vec_size) {
          auto tmp0 = at::vec::Vectorized<float>::loadu(curr_attn + i * chunk_size + j);
          tmp0.store(row + j);
        }
        if (j < len) {
          auto tmp0 = at::vec::Vectorized<float>::loadu(curr_attn + i * chunk_size + j, len - j);
          tmp0.store(row + j, len - j);
        }
        // (row.unsqueeze(-1) * sub).sum(-2)
        fill_stub(updated, 0, i);
        for (int k = 0; k < i; k++) {
          float row_k = row[k];
          auto vec_row_k = at::vec::Vectorized<float>(row_k);
          int64_t j = 0;
          int64_t len = i;
          for (; j < vec_size * (len / vec_size); j += vec_size) {
            auto tmp0 = at::vec::Vectorized<float>::loadu(curr_attn + k * chunk_size + j);
            auto tmp1 = vec_row_k * tmp0;
            auto tmp2 = at::vec::Vectorized<float>::loadu(updated + j);
            auto tmp3 = tmp1 + tmp2;
            tmp3.store(updated + j);
          }
          if (j < len) {
            auto tmp0 = at::vec::Vectorized<float>::loadu(curr_attn + k * chunk_size + j, len - j);
            auto tmp1 = vec_row_k * tmp0;
            auto tmp2 = at::vec::Vectorized<float>::loadu(updated + j);
            auto tmp3 = tmp1 + tmp2;
            tmp3.store(updated + j, len - j);
          }
        }
        // attn[..., i, :i] = row + sum(...)
        j = 0;
        len = i;
        for (; j < vec_size * (len / vec_size); j += vec_size) {
          auto tmp0 = at::vec::Vectorized<float>::loadu(row + j);
          auto tmp1 = at::vec::Vectorized<float>::loadu(updated + j);
          auto tmp2 = tmp0 + tmp1;
          tmp2.store(curr_attn + i * chunk_size + j);
        }
        if (j < len) {
          auto tmp0 = at::vec::Vectorized<float>::loadu(row + j, len - j);
          auto tmp1 = at::vec::Vectorized<float>::loadu(updated + j, len - j);
          auto tmp2 = tmp0 + tmp1;
          tmp2.store(curr_attn + i * chunk_size + j, len - j);
        }
      }
      for (int i = 0; i < chunk_size; i++) {
        curr_attn[i * chunk_size + i] += 1.0f;
        at::vec::map<scalar_t>(
            [](at::vec::Vectorized<float> x) { return x; },
            curr_attn_reduced + i * chunk_size,
            curr_attn + i * chunk_size,
            chunk_size);
      }

      // v_beta_attn = attn @ v_beta
      // k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
      // v_beta_attn: [B, HV, num_chunk, chunk_size, EV]
      // k_beta_g = k_beta * g: [B, HV, num_chunk, chunk_size, EK]
      // k_cumdecay: [B, HV, num_chunk, chunk_size, EK]
      // pack for value
      pack_vnni2<scalar_t>(
          /*    dst */ v_pack,
          /*    src */ curr_v_beta,
          /*     N  */ chunk_size,
          /*     K  */ v_head_size,
          /* ld_src */ v_head_size,
          /* ld_dst */ v_head_size);
      // value = attn @ v_beta
      at::native::cpublas::brgemm(
          /* M */ chunk_size,
          /* N */ v_head_size,
          /* K */ chunk_size,
          /* lda */ chunk_size,
          /* ldb */ v_head_size,
          /* ldc */ v_head_size,
          /* add_C */ false,
          /* A */ curr_attn_reduced,
          /* B */ v_pack,
          /* C */ curr_value);
      // k_beta_g = k_beta * g.exp().unsqueeze(-1)
      for (int64_t j = 0; j < chunk_size; j++) {
        int64_t i = 0;
        float g_exp = std::exp(curr_g_pad[j]);
        scalar_t g_exp_reduced = static_cast<scalar_t>(g_exp);
        auto vec_g_exp_reduced = at::vec::Vectorized<scalar_t>(g_exp_reduced);
        for (; i < reduced_vec_size * (qk_head_size / reduced_vec_size); i += reduced_vec_size) {
          auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(curr_k_beta + j * qk_head_size + i);
          auto tmp1 = tmp0 * vec_g_exp_reduced;
          tmp1.store(k_beta_g + j * qk_head_size + i);
        }
        if (i < qk_head_size) {
          auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(curr_k_beta + j * qk_head_size + i, qk_head_size - i);
          auto tmp1 = tmp0 * vec_g_exp_reduced;
          tmp1.store(k_beta_g + j * qk_head_size + i, qk_head_size - i);
        }
      }
      // pack for k_beta_g
      pack_vnni2<scalar_t>(
          /*    dst */ k_beta_g_pack,
          /*    src */ k_beta_g,
          /*     N  */ chunk_size,
          /*     K  */ qk_head_size,
          /* ld_src */ qk_head_size,
          /* ld_dst */ qk_head_size);
      // k_cumdecay = attn @ k_beta_g
      at::native::cpublas::brgemm(
          /* M */ chunk_size,
          /* N */ qk_head_size,
          /* K */ chunk_size,
          /* lda */ chunk_size,
          /* ldb */ qk_head_size,
          /* ldc */ qk_head_size,
          /* add_C */ false,
          /* A */ curr_attn_reduced,
          /* B */ k_beta_g_pack,
          /* C */ k_cumdecay);
      for (int i = 0; i < chunk_size; i++) {
        at::vec::map<scalar_t>(
            [](at::vec::Vectorized<float> x) { return x; },
            curr_k_cumdecay_reduced + i * qk_head_size,
            k_cumdecay + i * qk_head_size,
            qk_head_size);
      }

      // Move to the next query
      data_index_step(h, v_num_head, c, global_num_chunk);
    }
  });

  // for each chunk
  at::parallel_for(0, batch_size * v_num_head, 1, [&](int64_t begin, int64_t end) {
    int64_t b = 0, h = 0;
    data_index_init(begin, b, batch_size, h, v_num_head);
    int ompIdx = at::get_thread_num();
    int64_t offset =
        /* k_transpose */ qk_head_size * chunk_size +
        /* v_pack */ chunk_size * v_head_size +
        /* k_beta_g  */ chunk_size * qk_head_size +
        /* k_beta_g_pack  */ chunk_size * qk_head_size +
        /* attn */ chunk_size * chunk_size * 2 +
        /* attn_reduced */ chunk_size * chunk_size +
        /* k_cumdecay */ chunk_size * qk_head_size * 2 +
        /* row */ chunk_size * 2 +
        /* updated */ chunk_size * 2;
    scalar_t* thread_buff_ptr = thread_buff + ompIdx * buff_size_16bit_per_thread;
    scalar_t* curr_last_recurrent_state_reduced = thread_buff_ptr + offset;
    offset += qk_head_size * v_head_size;
    scalar_t* curr_last_recurrent_state_pack_reduced = thread_buff_ptr + offset;
    offset += qk_head_size * v_head_size;
    scalar_t* k_transpose_i = thread_buff_ptr + offset;
    offset += qk_head_size * chunk_size;
    float* attn_i = reinterpret_cast<float*>(thread_buff_ptr + offset);
    offset += chunk_size * chunk_size * 2;
    scalar_t* attn_i_reduced = thread_buff_ptr + offset;
    offset += chunk_size * chunk_size;
    float* v_prime = reinterpret_cast<float*>(thread_buff_ptr + offset);
    offset += chunk_size * v_head_size * 2;
    scalar_t* v_prime_reduced = thread_buff_ptr + offset;
    offset += chunk_size * v_head_size;
    scalar_t* v_prime_pack_reduced = thread_buff_ptr + offset;
    offset += chunk_size * v_head_size;
    scalar_t* qg = thread_buff_ptr + offset;
    offset += chunk_size * qk_head_size;
    float* attn_inter = reinterpret_cast<float*>(thread_buff_ptr + offset);
    offset += chunk_size * v_head_size * 2;
    scalar_t* kg = thread_buff_ptr + offset;
    offset += chunk_size * qk_head_size;
    scalar_t* kg_transpose = thread_buff_ptr + offset;
    offset += qk_head_size * chunk_size;
    float* kgv = reinterpret_cast<float*>(thread_buff_ptr + offset);
    offset += qk_head_size * v_head_size * 2;

    for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
      int64_t start_q = cu_seqlens_ptr[b];
      int64_t seq_len = cu_seqlens_ptr[b + 1] - start_q;
      int64_t num_chunk = chunk_offsets[b + 1] - chunk_offsets[b];
      int64_t chunk_offset = chunk_offsets[b];
      int64_t len_offset = chunk_offset * chunk_size;

      int64_t h_qk = h / head_group;
      auto out_ptr = out + start_q * oStrideT;
      auto curr_q = q_pad + len_offset * qk_head_size +
                    h_qk * global_total_seq_length * qk_head_size;  // [num_chunk, chunk_size, EK]
      auto curr_k = k_pad + len_offset * qk_head_size +
                    h_qk * global_total_seq_length * qk_head_size;            // [num_chunk, chunk_size, EK]
      auto curr_v = v_beta_attn + h * global_total_seq_length * v_head_size;  // [num_chunk, chunk_size, EV]
      auto curr_decay_mask =
          decay_mask + h * global_total_seq_length * chunk_size;  // [num_chunk, chunk_size, chunk_size]
      auto curr_k_cumdecay_reduced =
          k_cumdecay_reduced + h * global_total_seq_length * qk_head_size;  // [num_chunk, chunk_size, EK]
      auto curr_last_recurrent_state =
          final_state_data + b * final_state_StrideN + h * final_state_StrideH;  // [EK, EV]
      auto curr_g_pad = g_pad + len_offset + h * global_total_seq_length;        // [num_chunk, chunk_size]
      auto curr_core_attn_out = core_attn_out + len_offset * v_head_size +
                                h * global_total_seq_length * v_head_size;  // [num_chunk, chunk_size, EV]
      for (int64_t c = 0; c < num_chunk; c++) {
        for (int i = 0; i < qk_head_size; i++) {
          at::vec::map<scalar_t>(
              [](at::vec::Vectorized<float> x) { return x; },
              curr_last_recurrent_state_reduced + i * v_head_size,
              curr_last_recurrent_state + i * v_head_size,
              v_head_size);
        }
        auto q_i = curr_q + c * chunk_size * qk_head_size;                                   // [chunk_size, EK]
        auto k_i = curr_k + c * chunk_size * qk_head_size;                                   // [chunk_size, EK]
        auto v_i = curr_v + (chunk_offset + c) * chunk_size * v_head_size;                   // [chunk_size, EV]
        auto decay_mask_i = curr_decay_mask + (chunk_offset + c) * chunk_size * chunk_size;  // [chunk_size, chunk_size]
        auto k_cumdecay_i_reduced =
            curr_k_cumdecay_reduced + (chunk_offset + c) * chunk_size * qk_head_size;  // [chunk_size, EK]
        auto g_pad_i = curr_g_pad + c * chunk_size;                                    // [chunk_size]
        auto core_attn_out_i = curr_core_attn_out + c * chunk_size * v_head_size;      // [chunk_size, EV]

        // attn_i = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        // k_transpose_i = k_i.transpose(-1, -2)
        pack_vnni<scalar_t>(
            /*    dst */ k_transpose_i,
            /*    src */ k_i,
            /*     N  */ chunk_size,
            /*     K  */ qk_head_size,
            /* ld_src */ qk_head_size,
            /* ld_dst */ chunk_size);
        // attn_i = q_i @ k_transpose_i
        at::native::cpublas::brgemm(
            /* M */ chunk_size,
            /* N */ chunk_size,
            /* K */ qk_head_size,
            /* lda */ qk_head_size,
            /* ldb */ chunk_size,
            /* ldc */ chunk_size,
            /* add_C */ false,
            /* A */ q_i,
            /* B */ k_transpose_i,
            /* C */ attn_i);
        // attn_i = attn_i * decay_mask_i
        for (int64_t m = 0; m < chunk_size; m++) {
          auto attn_i_m = attn_i + m * chunk_size;
          auto attn_i_reduced_m = attn_i_reduced + m * chunk_size;
          auto decay_mask_i_m = decay_mask_i + m * chunk_size;
          int64_t n = 0;
          for (; n < vec_size * (chunk_size / vec_size); n += vec_size) {
            auto tmp0 = at::vec::Vectorized<float>::loadu(attn_i_m + n);
            auto tmp1 = at::vec::Vectorized<float>::loadu(decay_mask_i_m + n);
            auto tmp2 = tmp0 * tmp1;
            auto tmp3 = at::vec::convert<scalar_t>(tmp2);
            tmp3.store(attn_i_reduced_m + n, vec_size);
          }
          if (n < chunk_size) {
            auto tmp0 = at::vec::Vectorized<float>::loadu(attn_i_m + n, chunk_size - n);
            auto tmp1 = at::vec::Vectorized<float>::loadu(decay_mask_i_m + n, chunk_size - n);
            auto tmp2 = tmp0 * tmp1;
            auto tmp3 = at::vec::convert<scalar_t>(tmp2);
            tmp3.store(attn_i_reduced_m + n, chunk_size - n);
          }
        }
        // mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)
        // attn_i = attn_i.masked_fill_(mask, 0)
        for (int i = 0; i < chunk_size - 1; i++) {
          const auto vec_zero = at::vec::Vectorized<scalar_t>(0);
          int64_t len = chunk_size - i - 1;
          int64_t front = len % reduced_vec_size;
          int64_t j = i + 1;
          // first masked vec for alignment
          if (front > 0) {
            vec_zero.store(attn_i_reduced + i * chunk_size + j, front);
            j += front;
          }
          for (; j < reduced_vec_size * (chunk_size / reduced_vec_size); j += reduced_vec_size) {
            vec_zero.store(attn_i_reduced + i * chunk_size + j);
          }
        }

        // pack for curr_last_recurrent_state
        pack_vnni2<scalar_t>(
            /*    dst */ curr_last_recurrent_state_pack_reduced,
            /*    src */ curr_last_recurrent_state_reduced,
            /*     N  */ qk_head_size,
            /*     K  */ v_head_size,
            /* ld_src */ v_head_size,
            /* ld_dst */ v_head_size);

        // v_prime = k_cumdecay_i @ curr_last_recurrent_state: [chunk_size, EV]
        // k_cumdecay_i: [chunk_size, EK]
        // curr_last_recurrent_state: [EK, EV]
        at::native::cpublas::brgemm(
            /* M */ chunk_size,
            /* N */ v_head_size,
            /* K */ qk_head_size,
            /* lda */ qk_head_size,
            /* ldb */ v_head_size,
            /* ldc */ v_head_size,
            /* add_C */ false,
            /* A */ k_cumdecay_i_reduced,
            /* B */ curr_last_recurrent_state_pack_reduced,
            /* C */ v_prime);

        // v_new = v_prime = v_i - v_prime
        // v_i: [chunk_size, EV]
        for (int64_t m = 0; m < chunk_size; m++) {
          int64_t i = 0;
          for (; i < vec_size * (v_head_size / vec_size); i += vec_size) {
            auto tmp0 = at::vec::Vectorized<float>::loadu(v_i + m * v_head_size + i);
            auto tmp1 = at::vec::Vectorized<float>::loadu(v_prime + m * v_head_size + i);
            auto tmp2 = tmp0 - tmp1;
            auto tmp3 = at::vec::convert<scalar_t>(tmp2);
            tmp3.store(v_prime_reduced + m * v_head_size + i, vec_size);
          }
          if (i < v_head_size) {
            auto tmp0 = at::vec::Vectorized<float>::loadu(v_i + m * v_head_size + i, v_head_size - i);
            auto tmp1 = at::vec::Vectorized<float>::loadu(v_prime + m * v_head_size + i, v_head_size - i);
            auto tmp2 = tmp0 - tmp1;
            auto tmp3 = at::vec::convert<scalar_t>(tmp2);
            tmp3.store(v_prime_reduced + m * v_head_size + i, v_head_size - i);
          }
        }

        // attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        // qg = q_i * g[:, :, i, :, None].exp(): [chunk_size, EK]
        // q_i: [chunk_size, EK]
        // g[:, :, i, :, None]: [chunk_size, 1]
        for (int64_t m = 0; m < chunk_size; m++) {
          auto g_pad_i_m = g_pad_i + m;
          auto g_exp = std::exp(*g_pad_i_m);
          int64_t i = 0;
          scalar_t g_exp_reduced = static_cast<scalar_t>(g_exp);
          auto vec_g_exp_reduced = at::vec::Vectorized<scalar_t>(g_exp_reduced);
          for (; i < reduced_vec_size * (qk_head_size / reduced_vec_size); i += reduced_vec_size) {
            auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(q_i + m * qk_head_size + i);
            auto tmp2 = tmp0 * vec_g_exp_reduced;
            tmp2.store(qg + m * qk_head_size + i);
          }
          if (i < qk_head_size) {
            auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(q_i + m * qk_head_size + i, qk_head_size - i);
            auto tmp2 = tmp0 * vec_g_exp_reduced;
            tmp2.store(qg + m * qk_head_size + i, qk_head_size - i);
          }
        }
        // attn_inter = qg @ curr_last_recurrent_state: [chunk_size, EV]
        // curr_last_recurrent_state: [EK, EV]
        at::native::cpublas::brgemm(
            /* M */ chunk_size,
            /* N */ v_head_size,
            /* K */ qk_head_size,
            /* lda */ qk_head_size,
            /* ldb */ v_head_size,
            /* ldc */ v_head_size,
            /* add_C */ false,
            /* A */ qg,
            /* B */ curr_last_recurrent_state_pack_reduced,
            /* C */ attn_inter);

        // core_attn_out[:, :, i] = attn_inter + attn_i @ v_new
        // pack for v_prime
        pack_vnni2<scalar_t>(
            /*    dst */ v_prime_pack_reduced,
            /*    src */ v_prime_reduced,
            /*     N  */ chunk_size,
            /*     K  */ v_head_size,
            /* ld_src */ v_head_size,
            /* ld_dst */ v_head_size);
        // attn_inter = attn_inter + attn_i @ v_new: [chunk_size, EV]
        // attn_i: [chunk_size, chunk_size]
        // v_new: [chunk_size, EV]
        at::native::cpublas::brgemm(
            /* M */ chunk_size,
            /* N */ v_head_size,
            /* K */ chunk_size,
            /* lda */ chunk_size,
            /* ldb */ v_head_size,
            /* ldc */ v_head_size,
            /* add_C */ true,
            /* A */ attn_i_reduced,
            /* B */ v_prime_pack_reduced,
            /* C */ attn_inter);

        // core_attn_out[:, :, i] = attn_inter
        for (int64_t m = 0; m < chunk_size; m++) {
          at::vec::map<float>(
              [](at::vec::Vectorized<float> x) { return x; },
              core_attn_out_i + m * v_head_size,
              attn_inter + m * v_head_size,
              v_head_size);
        }

        // last_recurrent_state = (
        //     last_recurrent_state * g[:, :, i, -1, None, None].exp()
        //     + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        // )
        // 1) last_recurrent_state * g[:, :, i, -1, None, None].exp()
        // curr_last_recurrent_state: [EK, EV]
        // g[:, :, i, -1, None, None]: [1, 1]
        // last_recurrent_state * g[:, :, i, -1, None, None].exp(): [EK, EV]
        auto g_pad_i_last = g_pad_i + chunk_size - 1;
        auto g_exp_last = std::exp(g_pad_i_last[0]);
        for (int64_t m = 0; m < qk_head_size; m++) {
          int64_t i = 0;
          auto vec_g_exp_last = at::vec::Vectorized<float>(g_exp_last);
          for (; i < vec_size * (v_head_size / vec_size); i += vec_size) {
            auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(curr_last_recurrent_state_reduced + m * v_head_size + i);
            auto tmp1 = at::vec::convert<float>(tmp0);
            auto tmp2 = tmp1 * vec_g_exp_last;
            tmp2.store(curr_last_recurrent_state + m * v_head_size + i);
          }
          if (i < v_head_size) {
            auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(
                curr_last_recurrent_state_reduced + m * v_head_size + i, v_head_size - i);
            auto tmp1 = at::vec::convert<float>(tmp0);
            auto tmp2 = tmp1 * vec_g_exp_last;
            tmp2.store(curr_last_recurrent_state + m * v_head_size + i, v_head_size - i);
          }
        }
        // 2) (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        // k_i: [chunk_size, EK]
        // g[:, :, i, -1, None]: [1]
        // g[:, :, i]: [chunk_size]
        // (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]: [chunk_size, 1]
        // kg = k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]: [chunk_size, EK]
        // (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2): [EK, chunk_size]
        // v_new: [chunk_size, EV]
        // (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new: [EK, EV]
        // kg = k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]
        for (int64_t m = 0; m < chunk_size; m++) {
          auto g_exp = std::exp((g_pad_i_last[0] - g_pad_i[m]));
          int64_t i = 0;
          scalar_t g_exp_reduced = static_cast<scalar_t>(g_exp);
          auto vec_g_exp_reduced = at::vec::Vectorized<scalar_t>(g_exp_reduced);
          for (; i < reduced_vec_size * (qk_head_size / reduced_vec_size); i += reduced_vec_size) {
            auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(k_i + m * qk_head_size + i);
            auto tmp2 = tmp0 * vec_g_exp_reduced;
            tmp2.store(kg + m * qk_head_size + i);
          }
          if (i < qk_head_size) {
            auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(k_i + m * qk_head_size + i, qk_head_size - i);
            auto tmp2 = tmp0 * vec_g_exp_reduced;
            tmp2.store(kg + m * qk_head_size + i, qk_head_size - i);
          }
        }
        // kg.transpose(-1, -2): [EK, chunk_size]
        at::native::utils::transpose<scalar_t>(
            /* M */ chunk_size,
            /* N */ qk_head_size,
            /* src */ kg,
            /* ld_src */ qk_head_size,
            /* dst */ kg_transpose,
            /* ld_dst */ chunk_size);
        // kgv = kg.transpose(-1, -2) @ v_new
        // v_new: [chunk_size, EV]
        at::native::cpublas::brgemm(
            /* M */ qk_head_size,
            /* N */ v_head_size,
            /* K */ chunk_size,
            /* lda */ chunk_size,
            /* ldb */ v_head_size,
            /* ldc */ v_head_size,
            /* add_C */ false,
            /* A */ kg_transpose,
            /* B */ v_prime_pack_reduced,
            /* C */ kgv);
        // last_recurrent_state = 1) + 2)
        for (int64_t m = 0; m < qk_head_size; m++) {
          at::vec::map2<float>(
              [](at::vec::Vectorized<float> x, at::vec::Vectorized<float> y) { return x + y; },
              curr_last_recurrent_state + m * v_head_size,
              curr_last_recurrent_state + m * v_head_size,
              kgv + m * v_head_size,
              v_head_size);
        }
      }

      // core_attn_out -> output
      // output: [B, T, HV, EV]
      // core_attn_out: [B, HV, padded_T, EV]
      auto curr_out = out_ptr + h * oStrideH;
      for (int64_t m = 0; m < seq_len; m++) {
        at::vec::map<scalar_t>(
            [](at::vec::Vectorized<float> x) { return x; },
            curr_out + m * oStrideT,
            curr_core_attn_out + m * v_head_size,
            v_head_size);
      }

      // Move to the next query
      data_index_step(b, batch_size, h, v_num_head);
    }
  });
}
}  // anonymous namespace

template <bool is_contiguous, bool is_last_dim_contiguous>
static inline void
CHECK_INPUT_SHAPE_DTYPE(at::Tensor& tensor, const int64_t& dim, const std::vector<int64_t>& sizes, at::ScalarType st) {
  if (is_contiguous) {
    CHECK_CONTIGUOUS(tensor);
  }
  if (is_last_dim_contiguous) {
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(tensor);
  }
  TORCH_CHECK(tensor.dtype() == st);
  CHECK_DIM(dim, tensor);
  for (int64_t i = 0; i < dim; i++) {
    CHECK_EQ(tensor.size(i), sizes[i]);
  }
}

// query: [B, T, HK, EK]
// key: [B, T, HK, EK]
// value: [B, T, HV, EV]
// g: [B, T, HV] FP32
// beta: [B, T, HV]
// initial_state: [N, HV, EK, EV] FP32
// output_final_state: bool
// cu_seqlens: [N + 1] INT32
// head_first: bool
// use_qk_l2norm_in_kernel: bool
std::tuple<at::Tensor, at::Tensor> chunk_gated_delta_rule_cpu(
    at::Tensor& query,
    at::Tensor& key,
    at::Tensor& value,
    at::Tensor& g,
    at::Tensor& beta,
    at::Tensor& initial_state,
    bool output_final_state,
    at::Tensor& cu_seqlens,
    bool head_first,
    bool use_qk_l2norm_in_kernel) {
  RECORD_FUNCTION(
      "sgl-kernel::chunk_gated_delta_rule_cpu", std::vector<c10::IValue>({query, key, value, g, beta, initial_state}));

  TORCH_CHECK(head_first == false, "chunk_gated_delta_rule_cpu does not support head first");
  int64_t B = query.size(0);
  int64_t global_seq_len = query.size(1);
  int64_t qk_num_head = query.size(2);
  int64_t qk_head_size = query.size(3);
  int64_t v_num_head = value.size(2);
  int64_t v_head_size = value.size(3);
  int64_t batch_size = initial_state.size(0);
  CHECK_EQ(B, 1);
  CHECK_EQ(v_num_head % qk_num_head, 0);
  CHECK_INPUT_SHAPE_DTYPE<false, true>(query, 4, {B, global_seq_len, qk_num_head, qk_head_size}, at::kBFloat16);
  CHECK_INPUT_SHAPE_DTYPE<false, true>(key, 4, {B, global_seq_len, qk_num_head, qk_head_size}, at::kBFloat16);
  CHECK_INPUT_SHAPE_DTYPE<false, true>(value, 4, {B, global_seq_len, v_num_head, v_head_size}, at::kBFloat16);
  CHECK_INPUT_SHAPE_DTYPE<true, true>(g, 3, {B, global_seq_len, v_num_head}, at::kFloat);
  CHECK_INPUT_SHAPE_DTYPE<true, true>(beta, 3, {B, global_seq_len, v_num_head}, at::kBFloat16);
  CHECK_INPUT_SHAPE_DTYPE<true, true>(cu_seqlens, 1, {batch_size + 1}, at::kInt);
  CHECK_INPUT_SHAPE_DTYPE<true, true>(
      initial_state, 4, {batch_size, v_num_head, qk_head_size, v_head_size}, at::kFloat);

  at::Tensor output = at::empty_like(value, value.options());  // [B, T, HV, EV]
  at::Tensor final_state = initial_state.to(at::kFloat);       // [N, HV, EK, EV]

  // Strides
  int64_t qStrideH = query.stride(2);
  int64_t qStrideT = query.stride(1);
  int64_t kStrideH = key.stride(2);
  int64_t kStrideT = key.stride(1);
  int64_t vStrideH = value.stride(2);
  int64_t vStrideT = value.stride(1);

  constexpr int64_t chunk_size = 64;
  // Deduce the global chunks
  // e.g. cu_seqlens: [0, 5, 13, 16], chunk_size = 4
  // chunk_offsets: [0, 2, 4, 5]
  // chunk_indices (batch_id, local_chunk_id): [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0]]
  std::vector<int64_t> chunk_offsets(batch_size + 1, 0);
  std::vector<std::vector<int64_t>> chunk_indices;
  int32_t* cu_seqlens_ptr = cu_seqlens.data_ptr<int32_t>();
  int64_t s = 0;
  int64_t e = 0;
  int64_t s_pad = 0;
  int64_t e_pad = 0;
  for (int64_t b = 0; b < batch_size; b++) {
    e = cu_seqlens_ptr[b + 1];
    int64_t seq_len = e - s;
    int64_t pad_size = (chunk_size - seq_len % chunk_size) % chunk_size;
    int64_t total_seq_length = seq_len + pad_size;
    e_pad = s_pad + total_seq_length;
    chunk_offsets[b + 1] = e_pad / chunk_size;
    for (int64_t c = 0; c < total_seq_length / chunk_size; c++) {
      chunk_indices.push_back({b, c});
    }
    s = e;
    s_pad = e_pad;
  }
  int64_t global_total_seq_length = e_pad;
  int64_t global_num_chunk = e_pad / chunk_size;

  // Allocate buffer
  int64_t buff_size = v_num_head * global_total_seq_length                               // g_pad_data
                      + batch_size * v_num_head * global_total_seq_length * v_head_size  // core_attn
                      + v_num_head * global_total_seq_length * chunk_size                // decay_mask
                      + v_num_head * global_total_seq_length * v_head_size;              // v_beta_attn
  at::Tensor buff_data = at::empty({buff_size}, query.options().dtype(at::kFloat));
  int64_t reduced_buff_size = qk_num_head * global_total_seq_length * qk_head_size    // q_pad_data
                              + qk_num_head * global_total_seq_length * qk_head_size  // k_pad_data
                              + v_num_head * global_total_seq_length * v_head_size    // v_pad_data
                              + v_num_head * global_total_seq_length * qk_head_size   // k_beta_data
                              + v_num_head * global_total_seq_length * v_head_size    // v_beta_data
                              + v_num_head * global_total_seq_length * qk_head_size;  // k_cumdecay_reduced
  at::Tensor reduced_buff_data = at::empty({reduced_buff_size}, query.options());
  int64_t num_thread = at::get_num_threads();
  int64_t buff_size_16bit_per_thread =
      /* k_transpose */ qk_head_size * chunk_size +
      /* v_pack */ chunk_size * v_head_size +
      /* k_beta_g  */ chunk_size * qk_head_size +
      /* k_beta_g_pack  */ chunk_size * qk_head_size +
      /* attn */ chunk_size * chunk_size * 2 +
      /* attn_reduced */ chunk_size * chunk_size +
      /* k_cumdecay */ chunk_size * qk_head_size * 2 +
      /* row */ chunk_size * 2 +
      /* updated */ chunk_size * 2 +
      /* curr_last_recurrent_state_reduced  */ qk_head_size * v_head_size +
      /* curr_last_recurrent_state_pack_reduced   */ qk_head_size * v_head_size +
      /* k_transpose_i  */ qk_head_size * chunk_size +
      /* attn_i   */ chunk_size * chunk_size * 2 +
      /* attn_i_reduced     */ chunk_size * chunk_size +
      /* v_prime */ chunk_size * v_head_size * 2 +
      /* v_prime_reduced */ chunk_size * v_head_size +
      /* v_prime_pack_reduced */ chunk_size * v_head_size +
      /* qg */ chunk_size * qk_head_size +
      /* attn_inter */ chunk_size * v_head_size * 2 +
      /* kg */ chunk_size * qk_head_size +
      /* kg_transpose */ qk_head_size * chunk_size +
      /* kgv */ qk_head_size * v_head_size * 2;
  at::Tensor thread_buff_data = at::empty({num_thread, buff_size_16bit_per_thread}, query.options());

  AT_DISPATCH_REDUCED_FLOATING_TYPES(query.scalar_type(), "chunk_gated_delta_rule_kernel", [&] {
    chunk_gated_delta_rule_kernel_impl<scalar_t, chunk_size>(
        output.data_ptr<scalar_t>(),
        final_state.data_ptr<float>(),
        query.data_ptr<scalar_t>(),
        key.data_ptr<scalar_t>(),
        value.data_ptr<scalar_t>(),
        g.data_ptr<float>(),
        beta.data_ptr<scalar_t>(),
        cu_seqlens_ptr,
        buff_data.data_ptr<float>(),
        reduced_buff_data.data_ptr<scalar_t>(),
        thread_buff_data.data_ptr<scalar_t>(),
        use_qk_l2norm_in_kernel,
        batch_size,
        global_seq_len,
        qk_num_head,
        v_num_head,
        qk_head_size,
        v_head_size,
        qStrideH,
        qStrideT,
        kStrideH,
        kStrideT,
        vStrideH,
        vStrideT,
        chunk_offsets,
        chunk_indices,
        global_total_seq_length,
        global_num_chunk,
        buff_size_16bit_per_thread);
  });
  return std::make_tuple(std::move(output), std::move(final_state));
}
