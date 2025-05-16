#include "common.h"
#include "gemm.h"
#include "vec.h"

namespace {

// [NOTE]: extend attention for CPU
//   1. tune BLOCK_M and BLOCK_N
//   2. can handle non-contiguous k_exttend and v_extend
//   3. computes attention for prefix and extend separately
//   4. TODO: vectorize `pack_vnni` and `pack_vnni2`
//
template <typename index_t>
inline index_t get_index(index_t* ind, int i) {
  return (ind == nullptr) ? (index_t)i : ind[i];
}

// convert to vnni format
// from [N, K/2, 2] to [K/2, N, 2] for bfloat16 and float16
template <typename scalar_t, typename index_t>
void pack_vnni(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ src,
    const index_t* __restrict__ ind,
    int N,
    int K,
    int ld_src,
    int ld_dst) {
  for (int n = 0; n < N; ++n) {
    index_t index = get_index(ind, n);
    for (int k = 0; k < K / 2; ++k) {
      for (int d = 0; d < 2; ++d) {
        dst[k * ld_dst * 2 + n * 2 + d] = src[index * ld_src + k * 2 + d];
      }
    }
  }
}

// convert to vnni format
// from [K/2, 2, N] to [K/2, N, 2] for bfloat16 and float16
template <typename scalar_t, typename index_t>
void pack_vnni2(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ src,
    const index_t* __restrict__ ind,
    int K,
    int N,
    int ld_src,
    int ld_dst) {
  int k = 0;
  for (; k < (K >> 1) * 2; k += 2) {
    index_t index0 = get_index(ind, k + 0);
    index_t index1 = get_index(ind, k + 1);
    for (int n = 0; n < N; ++n) {
      dst[(k >> 1) * ld_dst * 2 + n * 2 + 0] = src[index0 * ld_src + n];
      dst[(k >> 1) * ld_dst * 2 + n * 2 + 1] = src[index1 * ld_src + n];
    }
  }
  if (K % 2 != 0) {
    index_t index = get_index(ind, K - 1);
    for (int n = 0; n < N; ++n) {
      dst[(K >> 1) * ld_dst * 2 + n * 2 + 0] = src[index * ld_src + n];
      dst[(K >> 1) * ld_dst * 2 + n * 2 + 1] = 0;
    }
    k += 2;
  }
  // TODO: check whether we can skip this!
  // const int padded_K = div_up(K, TILE_K) * TILE_K;
  // for (; k < padded_K; ++k) {
  //   for (int n = 0; n < N; ++n) {
  //     dst[k * ld_dst + n] = static_cast<scalar_t>(0);
  //   }
  // }
}

template <typename scalar_t>
inline void fill_stub(scalar_t* __restrict__ out, float val, int size) {
  using Vec = at::vec::Vectorized<scalar_t>;
  const Vec data_vec = Vec(static_cast<scalar_t>(val));
  int d = 0;
  for (; d <= size - Vec::size(); d += Vec::size()) {
    data_vec.store(out + d);
  }
  if (size - d > 0) {
    data_vec.store(out + d, size - d);
  }
}

template <typename scalar_t, int BLOCK_N>
inline void copy_stub(scalar_t* __restrict__ out, const float* __restrict__ input) {
  static_assert(BLOCK_N % 32 == 0);
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;

  constexpr int COLS = BLOCK_N / 16;
  auto store = [&](auto i) {
    constexpr int col = i % COLS;
    // for COLS = 2, 4 use 512bit store
    if constexpr (col % 2 == 0) {
      fVec a_fvec0 = fVec::loadu(input + col * 16);
      fVec a_fvec1 = fVec::loadu(input + col * 16 + 16);
      bVec out_bvec = convert_from_float_ext<scalar_t>(a_fvec0, a_fvec1);
      out_bvec.store(out + col * 16);
    }
  };
  Unroll<COLS>{}(store);
}

template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ out, const float* __restrict__ acc, float s, int size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  const fVec s_fvec = fVec(s);
  int d = 0;
  for (; d <= size - bVec::size(); d += bVec::size()) {
    fVec a_fvec0 = fVec::loadu(acc + d) * s_fvec;
    fVec a_fvec1 = fVec::loadu(acc + d + fVec::size()) * s_fvec;
    bVec out_bvec = convert_from_float_ext<scalar_t>(a_fvec0, a_fvec1);
    out_bvec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(acc[d] * s);
  }
}

template <typename scalar_t, typename index_t, int BLOCK_M, int BLOCK_N>
void extend_attention_kernel_impl(
    scalar_t* __restrict__ o_extend,
    const scalar_t* __restrict__ q_extend,
    const scalar_t* __restrict__ k_extend,
    const scalar_t* __restrict__ v_extend,
    const scalar_t* __restrict__ k_buffer,
    const scalar_t* __restrict__ v_buffer,
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
    int max_num_reqs,
    int max_context_len,
    int max_total_num_tokens,
    int max_len_extend,
    int buffer_size_per_thread,
    bool is_prefix_skipped) {
  using Vec = at::vec::Vectorized<float>;

  // strides
  const int q_strideM = num_heads * head_size;
  const int q_strideH = head_size;
  const int o_strideM = num_heads * head_size_v;
  const int o_strideH = head_size_v;

  // we use same buffer for packed key and value
  const int ldb_tmp = std::max(head_size, head_size_v);

  const bool has_logit_cap = logit_cap > 0;
  float rlogit_cap = has_logit_cap ? 1 / logit_cap : 0.f;

  const int num_groups = num_heads / num_heads_kv;
  TORCH_CHECK(num_groups * num_heads_kv == num_heads);

  // number of blocks along M
  int MB = div_up(max_len_extend, BLOCK_M);

  // parallel on [batches, num_heads, BM]
  at::parallel_for(0, batches * num_heads * MB, 0, [&](int begin, int end) {
    int bs{0}, head_id{0}, mb{0};
    data_index_init(begin, bs, batches, head_id, num_heads, mb, MB);

    int tid = at::get_thread_num();
    // s_i and s_delta: [BLOCK_M, BLOCK_N]
    float* __restrict__ s_i = reinterpret_cast<float*>((char*)(buffer) + tid * buffer_size_per_thread);
    float* __restrict__ s_delta = s_i;

    // v_prime: [BLOCK_M, head_size_v]
    float* __restrict__ v_prime = s_i + BLOCK_M * BLOCK_N;

    // s_delta2: [BLOCK_M, BLOCK_N]; copy of s_delta in scalar_t
    scalar_t* __restrict__ s_delta2 = reinterpret_cast<scalar_t*>(v_prime + BLOCK_N * head_size_v);

    // Btmp: [BLOCK_N, max(head_size, head_size_v)]
    scalar_t* __restrict__ Btmp = s_delta2 + BLOCK_M * BLOCK_N;

    // init Btmp just once for each thread to prevent NaN
    fill_stub(Btmp, 0.f, BLOCK_N * ldb_tmp);

    alignas(64) float s_prime[BLOCK_M];
    alignas(64) float m_prime[BLOCK_M];

    for (int i = begin; i < end; ++i) {
      // seq_len = prefix + extend
      int head_kv_id = head_id / num_groups;
      int seq_len = seq_lens[bs];
      int seq_len_extend = extend_seq_lens[bs];
      int seq_len_prefix = seq_len - seq_len_extend;
      int seq_extend_start_loc = extend_start_loc[bs];

      int req_pool_id = req_pool_indices[bs];
      TORCH_CHECK(seq_len_prefix >= 0, "prefix len < 0!");
      TORCH_CHECK(seq_len <= max_context_len, "seq_len out of scope!");
      TORCH_CHECK(req_pool_id < max_num_reqs, "req_pool_id out of scope!");

      if (is_prefix_skipped) {
        TORCH_CHECK(seq_len_prefix == 0, "extend attention: expect seq_len_prefix to be 0, got ", seq_len_prefix);
      }

      // offset and size in MB
      int m = mb * BLOCK_N;
      int m_size = std::min(BLOCK_M, seq_len_extend - m);

      if (m_size <= 0) {
        data_index_step(bs, batches, head_id, num_heads, mb, MB);
        continue;
      }

      // get query
      const scalar_t* __restrict__ q_ptr = q_extend + (seq_extend_start_loc + m) * q_strideM + head_id * q_strideH;

      // init v', s' and m'
      fill_stub(v_prime, 0.f, m_size * head_size_v);
      fill_stub(s_prime, 0.f, m_size);
      fill_stub(m_prime, -std::numeric_limits<scalar_t>::infinity(), m_size);

      // stage 1: compute scores with prefix
      for (int n = 0; n < seq_len_prefix; n += BLOCK_N) {
        int n_size = std::min(BLOCK_N, seq_len_prefix - n);

        // `n_size` is K in 2nd gemm, pad to TILE_K;
        const int padded_n_size = div_up(n_size, TILE_K) * TILE_K;

        // get key and pack
        pack_vnni<scalar_t, index_t>(
            /*    dst */ Btmp,
            /*    src */ k_buffer + head_kv_id * k_strideH,
            /*    ind */ req_to_token + req_pool_id * max_context_len + n,
            /*     N  */ n_size,
            /*     K  */ head_size,
            /* ld_src */ k_strideN,
            /* ld_dst */ BLOCK_N);

        // calculate s_i <- Q @ K
        at::native::cpublas::brgemm(
            /* M     */ m_size,
            /* N     */ n_size,
            /* K     */ head_size,
            /* lda   */ q_strideM,
            /* ldb   */ BLOCK_N,
            /* ldc   */ BLOCK_N,
            /* add_C */ false,
            /* A     */ q_ptr,
            /* B     */ Btmp,
            /* C     */ s_i);

        const Vec scale_vec = Vec(scaling);
        for (int row = 0; row < m_size; ++row) {
          // s_i <- s_i * scale
          at::vec::map<float>(
              [scale_vec](Vec x) { return x * scale_vec; }, s_i + row * BLOCK_N, s_i + row * BLOCK_N, n_size);

          // TODO: `tanh` from torch uses sleef u10, going to be slow
          if (has_logit_cap) {
            at::vec::map<float>(
                [logit_cap, rlogit_cap](Vec x) { return Vec(logit_cap) * (x * Vec(rlogit_cap)).tanh(); },
                s_i + row * BLOCK_N,
                s_i + row * BLOCK_N,
                n_size);
          }

          // m_i: max value per row
          float m_i = at::vec::reduce_all<float>(
              [](Vec& x, Vec& y) { return at::vec::maximum(x, y); }, s_i + row * BLOCK_N, n_size);
          m_i = std::max(m_i, m_prime[row]);

          // m_delta <- exp(m' - m_i)
          float m_delta = std::exp(m_prime[row] - m_i);

          // s_delta <- exp(s_i - m_i)
          at::vec::map<float>(
              [m_i](Vec x) { return (x - Vec(m_i)).exp_u20(); }, s_delta + row * BLOCK_N, s_i + row * BLOCK_N, n_size);

          // s' <- s' * m_delta + sum(s_delta)
          s_prime[row] *= m_delta;
          s_prime[row] +=
              at::vec::reduce_all<float>([](Vec& x, Vec& y) { return x + y; }, s_delta + row * BLOCK_N, n_size);

          m_prime[row] = m_i;

          // v' <- v' * m_delta
          at::vec::map<float>(
              [m_delta](Vec x) { return x * Vec(m_delta); },
              v_prime + row * head_size_v,
              v_prime + row * head_size_v,
              head_size_v);

          // pad s_delta with 0 first and then convert to scalar_t
          fill_stub(s_delta + row * BLOCK_N + n_size, 0.f, padded_n_size - n_size);
          copy_stub<scalar_t, BLOCK_N>(s_delta2 + row * BLOCK_N, s_delta + row * BLOCK_N);
        }

        // get value and pack
        pack_vnni2<scalar_t, index_t>(
            /*    dst */ Btmp,
            /*    src */ v_buffer + head_kv_id * v_strideH,
            /*    ind */ req_to_token + req_pool_id * max_context_len + n,
            /*     K  */ n_size,
            /*     N  */ head_size_v,
            /* ld_src */ v_strideN,
            /* ld_dst */ head_size_v);

        // calculate V' <- s_delta @ V + V'
        at::native::cpublas::brgemm(
            /* M     */ m_size,
            /* N     */ head_size_v,
            /* K     */ padded_n_size,  // n_size
            /* lda   */ BLOCK_N,
            /* ldb   */ head_size_v,
            /* ldc   */ head_size_v,
            /* add_C */ true,
            /* A     */ s_delta2,
            /* B     */ Btmp,
            /* C     */ v_prime);
      }  // loop with seq_len_prefix

      // stage 2: compute the triangle part
      int num_keys = std::min(seq_len_extend, m + BLOCK_M);
      for (int n = 0; n < num_keys; n += BLOCK_N) {
        int n_size = std::min(BLOCK_N, num_keys - n);

        // `n_size` is K in 2nd gemm, pad to TILE_K;
        const int padded_n_size = div_up(n_size, TILE_K) * TILE_K;

        // get key and pack
        pack_vnni<scalar_t, index_t>(
            /*    dst */ Btmp,
            /*    src */ k_extend + (seq_extend_start_loc + n) * ke_strideN + head_kv_id * ke_strideH,
            /*    ind */ nullptr,
            /*     N  */ n_size,
            /*     K  */ head_size,
            /* ld_src */ ke_strideN,
            /* ld_dst */ BLOCK_N);

        // calculate s_i <- Q @ K
        at::native::cpublas::brgemm(
            /* M     */ m_size,
            /* N     */ n_size,
            /* K     */ head_size,
            /* lda   */ q_strideM,
            /* ldb   */ BLOCK_N,
            /* ldc   */ BLOCK_N,
            /* add_C */ false,
            /* A     */ q_ptr,
            /* B     */ Btmp,
            /* C     */ s_i);

        // apply causal mask
        if (num_keys - n <= BLOCK_N) {
          for (int row = 0; row < m_size; ++row) {
            int last_col = m + row - n;
            // fill [last_col + 1, n_size) to -inf
            float* row_ptr = s_i + row * BLOCK_N;
            fill_stub(row_ptr + last_col + 1, -std::numeric_limits<float>::infinity(), n_size - last_col - 1);
          }
        }

        const Vec scale_vec = Vec(scaling);
        for (int row = 0; row < m_size; ++row) {
          // s_i <- s_i * scale
          at::vec::map<float>(
              [scale_vec](Vec x) { return x * scale_vec; }, s_i + row * BLOCK_N, s_i + row * BLOCK_N, n_size);

          // TODO: `tanh` from torch uses sleef u10, going to be slow
          if (has_logit_cap) {
            at::vec::map<float>(
                [logit_cap, rlogit_cap](Vec x) { return Vec(logit_cap) * (x * Vec(rlogit_cap)).tanh(); },
                s_i + row * BLOCK_N,
                s_i + row * BLOCK_N,
                n_size);
          }

          // m_i: max value per row
          float m_i = at::vec::reduce_all<float>(
              [](Vec& x, Vec& y) { return at::vec::maximum(x, y); }, s_i + row * BLOCK_N, n_size);
          m_i = std::max(m_i, m_prime[row]);

          // m_delta <- exp(m' - m_i)
          float m_delta = std::exp(m_prime[row] - m_i);

          // s_delta <- exp(s_i - m_i)
          at::vec::map<float>(
              [m_i](Vec x) { return (x - Vec(m_i)).exp_u20(); }, s_delta + row * BLOCK_N, s_i + row * BLOCK_N, n_size);

          // s' <- s' * m_delta + sum(s_delta)
          s_prime[row] *= m_delta;
          s_prime[row] +=
              at::vec::reduce_all<float>([](Vec& x, Vec& y) { return x + y; }, s_delta + row * BLOCK_N, n_size);

          m_prime[row] = m_i;

          // v' <- v' * m_delta
          at::vec::map<float>(
              [m_delta](Vec x) { return x * Vec(m_delta); },
              v_prime + row * head_size_v,
              v_prime + row * head_size_v,
              head_size_v);

          // pad s_delta with 0 first and then convert to scalar_t
          fill_stub(s_delta + row * BLOCK_N + n_size, 0.f, padded_n_size - n_size);
          copy_stub<scalar_t, BLOCK_N>(s_delta2 + row * BLOCK_N, s_delta + row * BLOCK_N);
        }

        // get value and pack
        pack_vnni2<scalar_t, index_t>(
            /*    dst */ Btmp,
            /*    src */ v_extend + (seq_extend_start_loc + n) * ve_strideN + head_kv_id * ve_strideH,
            /*    ind */ nullptr,
            /*     K  */ n_size,
            /*     N  */ head_size_v,
            /* ld_src */ ve_strideN,
            /* ld_dst */ head_size_v);

        // calculate V' <- s_delta @ V + V'
        at::native::cpublas::brgemm(
            /* M     */ m_size,
            /* N     */ head_size_v,
            /* K     */ padded_n_size,  // n_size
            /* lda   */ BLOCK_N,
            /* ldb   */ head_size_v,
            /* ldc   */ head_size_v,
            /* add_C */ true,
            /* A     */ s_delta2,
            /* B     */ Btmp,
            /* C     */ v_prime);
      }  // loop with seq_len_extend

      scalar_t* __restrict__ out_ptr = o_extend + (seq_extend_start_loc + m) * o_strideM + head_id * o_strideH;
      for (int row = 0; row < m_size; ++row) {
        float s = 1 / s_prime[row];
        copy_stub<scalar_t>(out_ptr + row * o_strideM, v_prime + row * head_size_v, s, head_size_v);
      }

      // move to the next index
      data_index_step(bs, batches, head_id, num_heads, mb, MB);
    }
    at::native::cpublas::brgemm_release();
  });
}

}  // anonymous namespace

// q_extend, k_extend, v_extend, o_extend: contiguous tensors
// k_buffer, v_buffer: (prefix + extend) tensors in mem_manager
//
// q_extend: [num_tokens, num_heads, head_size]
// k_extend: [num_extend_tokens, num_heads, head_size]
// v_extend: [num_extend_tokens, num_heads, head_size]
// o_extend: [num_tokens, num_heads, head_size]
// k_buffer: [max_total_num_tokens, num_heads, head_size]
// v_buffer: [max_total_num_tokens, num_heads, head_size]
// req_to_token: [max_num_reqs, max_context_len] int32 or int64
// req_pool_indices: [num_seqs] int64
// seq_lens: [num_seqs] int64
// extend_seq_lens: [num_seqs]
// extend_start_loc: [num_seqs]
//
void extend_attention_cpu(
    at::Tensor& q_extend,
    at::Tensor& k_extend,
    at::Tensor& v_extend,
    at::Tensor& o_extend,
    at::Tensor& k_buffer,
    at::Tensor& v_buffer,
    at::Tensor& req_to_token,
    at::Tensor& req_pool_indices,
    at::Tensor& seq_lens,
    at::Tensor& extend_seq_lens,
    at::Tensor& extend_start_loc,
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

  CHECK_INPUT(q_extend);
  CHECK_INPUT(o_extend);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k_extend);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(v_extend);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k_buffer);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(v_buffer);

  int num_seqs = seq_lens.size(0);
  int max_num_reqs = req_to_token.size(0);
  int max_context_len = req_to_token.size(1);
  int max_total_num_tokens = k_buffer.size(0);

  int num_heads = q_extend.size(1);
  int num_heads_kv = k_extend.size(1);
  int head_size = q_extend.size(2);
  int head_size_v = v_extend.size(2);

  // strides for k_extend and v_extend
  int ke_strideN = k_extend.stride(0);
  int ke_strideH = k_extend.stride(1);
  int ve_strideN = v_extend.stride(0);
  int ve_strideH = v_extend.stride(1);

  // strides for k_buffer and v_buffer
  int k_strideN = k_buffer.stride(0);
  int k_strideH = k_buffer.stride(1);
  int v_strideN = v_buffer.stride(0);
  int v_strideH = v_buffer.stride(1);

  // check sizes
  CHECK_EQ(req_pool_indices.size(0), num_seqs);
  CHECK_EQ(extend_seq_lens.size(0), num_seqs);
  CHECK_EQ(extend_start_loc.size(0), num_seqs);
  CHECK_EQ(v_extend.size(1), num_heads_kv);
  CHECK_EQ(k_buffer.size(1), v_buffer.size(1));

  // MLA will skip prefix part
  const bool is_prefix_skipped = k_buffer.size(1) != num_heads_kv;

  // check index data types
  const auto index_dtype = req_to_token.scalar_type();
  TORCH_CHECK(
      index_dtype == at::kInt || index_dtype == at::kLong,
      "extend: expect req_to_token to be int32 or int64, got ",
      index_dtype);
  TORCH_CHECK(seq_lens.scalar_type() == at::kLong, "extend: expect req_lens to be int64, got ", seq_lens.scalar_type());
  TORCH_CHECK(
      req_pool_indices.scalar_type() == at::kLong,
      "extend: expect req_pool_indices to be int64, got ",
      req_pool_indices.scalar_type());
  TORCH_CHECK(
      extend_seq_lens.scalar_type() == index_dtype && extend_start_loc.scalar_type() == index_dtype,
      "extend: expect extend_seq_lens and extend_start_loc to have same dtype as req_to_token.");

  // D and DV need to be 32x as we transpose by 512-bit
  TORCH_CHECK(head_size % 32 == 0, "invalid head_size ", head_size);
  TORCH_CHECK(head_size_v % 32 == 0, "invalid head_size_v ", head_size_v);

  // block size for query seq length
  constexpr int BLOCK_M = 32;
  // block size for key/value seq length
  constexpr int BLOCK_N = 32;

  const int size_per_thread =
      /* s_i     */ BLOCK_M * BLOCK_N * sizeof(float) +
      /* v_prime */ BLOCK_M * head_size_v * sizeof(float) +
      /* s_delta */ BLOCK_M * BLOCK_N * sizeof(uint16_t) +
      /* Btmp    */ BLOCK_N * std::max(head_size, head_size_v) * sizeof(uint16_t);

  int num_threads = at::get_num_threads();
  auto buffer = at::empty({num_threads, size_per_thread}, q_extend.options().dtype(at::kChar));

  AT_DISPATCH_REDUCED_FLOATING_TYPES(q_extend.scalar_type(), "extend_attention_kernel", [&] {
    AT_DISPATCH_INDEX_TYPES(index_dtype, "extend_attention_indices", [&] {
      extend_attention_kernel_impl<scalar_t, index_t, BLOCK_M, BLOCK_N>(
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
          num_seqs,
          num_heads,
          num_heads_kv,
          head_size,
          head_size_v,
          ke_strideN,
          ke_strideH,
          ve_strideN,
          ve_strideH,
          k_strideN,
          k_strideH,
          v_strideN,
          v_strideH,
          sm_scale,
          logit_cap,
          max_num_reqs,
          max_context_len,
          max_total_num_tokens,
          max_len_extend,
          size_per_thread,
          is_prefix_skipped);
    });
  });
}
