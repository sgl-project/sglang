#include "common.h"
#include "gemm.h"
#include "vec.h"

namespace {

inline void print_32x16(const __m512bh x) {
  at::BFloat16 a[32];
  _mm512_storeu_si512((__m512i*)a, (__m512i)x);

  for (int i = 0; i < 32; i++) {
    std::cout << a[i] << " ";
  }
  std::cout << std::endl;
}

inline void print_16x32(const __m512 x) {
  float a[16];
  _mm512_storeu_ps((__m512*)a, (__m512)x);

  for (int i = 0; i < 16; i++) {
    std::cout << a[i] << " ";
  }
  std::cout << std::endl;
}

// A : [M, BLOCK_N]
// B : [BLOCK_N, K], prepacked as [K/2, BLOCK_N, 2]
// C : [M, BLOCK_N]
// bias : [BLOCK_N]
//
// lda : leading dimension of `input` and `out`
//
template <typename scalar_t, int K, int BLOCK_N, bool has_bias>
struct tinygemm_kernel {
  static inline void apply(
      const scalar_t* __restrict__ A,
      const scalar_t* __restrict__ B,
      scalar_t* __restrict__ C,
      const scalar_t* __restrict__ bias,
      const scalar_t* __restrict__ conv_states,
      bool has_initial_state,
      int64_t M,
      int64_t lda,
      bool is_first_token) {
    TORCH_CHECK(false, "tinygemm_kernel_nn: scalar path not implemented!");
  }
};

#if defined(CPU_CAPABILITY_AVX512)
template <int K, int BLOCK_N, bool has_bias>
struct tinygemm_kernel<at::BFloat16, K, BLOCK_N, has_bias> {
  static inline void apply(
      const at::BFloat16* __restrict__ A,
      const at::BFloat16* __restrict__ B,
      at::BFloat16* __restrict__ C,
      const at::BFloat16* __restrict__ bias,
      const at::BFloat16* __restrict__ conv_states,
      bool has_initial_state,
      int64_t M,
      int64_t lda,
      bool is_first_token) {
    // std::cout << "### tinygemm_kernel: M = " << M << "; lda = " << lda << "; K = " << K << "; BLOCK_N = " << BLOCK_N
    // << "; is_first_token: " << (is_first_token ? "true" : "false") << std::endl; std::cout << "### has_initial_state:
    // " << (has_initial_state ? "true" : "false") << std::endl;

    assert(K == 4);
    constexpr int ROWS = K;
    constexpr int COLS = BLOCK_N / block_size_n();

    // leading dimension size for b for next block [K/2, 32, 2]
    constexpr int ldb = block_size_n() * K;

    __m512bh va[ROWS * COLS];
    __m512bh vb[ROWS * COLS];
    __m512 vc[COLS * 2];

    // k: {-3, -2, -1} -> {0, 1, 2}
    auto set_conv_states = [&](int k, int col) -> __m512i {
      return has_initial_state ? _mm512_loadu_si512(conv_states + (k + K - 1) * lda + col * 32)
                               : _mm512_setzero_si512();
    };

#define MM512_LOAD_A(idx)                                                 \
  ((idx) < 0 && is_first_token) ? (__m512bh)(set_conv_states((idx), col)) \
                                : (__m512bh)(_mm512_loadu_si512(A + (idx) * lda + col * 32))

#define MM512_PACK_A(ap, bp, a, b)                       \
  do {                                                   \
    __m512i r0 = (__m512i)(a);                           \
    __m512i r1 = (__m512i)(b);                           \
    __m512i d0 = _mm512_unpacklo_epi16(r0, r1);          \
    __m512i d1 = _mm512_unpackhi_epi16(r0, r1);          \
    r0 = _mm512_shuffle_i32x4(d0, d1, 0x88);             \
    r1 = _mm512_shuffle_i32x4(d0, d1, 0xdd);             \
    (ap) = (__m512bh)_mm512_shuffle_i32x4(r0, r1, 0x88); \
    (bp) = (__m512bh)_mm512_shuffle_i32x4(r0, r1, 0xdd); \
  } while (0)

    // step 0 : preload a at time step [-3][-2][-1]
    auto preloada = [&](auto i) {
      constexpr int col = i;
      int64_t m = 0;
      va[1 * COLS + col] = MM512_LOAD_A(m - 3);
      va[2 * COLS + col] = MM512_LOAD_A(m - 2);
      va[3 * COLS + col] = MM512_LOAD_A(m - 1);
      ;
    };
    Unroll<COLS>{}(preloada);

    auto loada = [&](auto i, int64_t m) {
      constexpr int col = i;
      // update previous time step
      va[0 * COLS + col] = va[1 * COLS + col];
      va[1 * COLS + col] = va[2 * COLS + col];
      va[2 * COLS + col] = va[3 * COLS + col];
      // load current time step
      va[3 * COLS + col] = MM512_LOAD_A(m);
    };

    // step 1 : load weight for just once
    auto loadb = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;
      vb[row * COLS + col] = (__m512bh)(_mm512_loadu_si512(B + col * ldb + row * 32));
    };
    Unroll<ROWS * COLS>{}(loadb);

    // std::cout << "### after loadb ..." << std::endl;

    // [NB] accumulates 4x32 bfloat16 blocks
    //
    //   +------------+------------+
    //   |    col0    |    col1    |
    //   +------------+------------+
    //   |  va0  va1  |  va0  va1  |
    //   |  va2  va3  |  va2  va3  |
    //   +------------+------------+
    //   |  vc0  vc1  |  vc0  vc1  |
    //   +------------+------------+
    //
    //  * va and vb shares the same memory layout
    //  * block_n 32 with 4 rows equals to 4 registers
    //  * 37 uops with avx512bf16 v.s. 57 uops with avx512f
    //
    auto compute = [&](auto i) {
      constexpr int col = i;

      // init accumulators
      if constexpr (has_bias) {
        __m512i b16 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(bias + col * 32));
        vc[col * 2 + 0] = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(b16, 0));
        vc[col * 2 + 1] = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(b16, 1));
      } else {
        vc[col * 2 + 0] = _mm512_set1_ps(0.f);
        vc[col * 2 + 1] = _mm512_set1_ps(0.f);
      }

      // convert to vnni2 format
      __m512bh va0, va1, va2, va3;
      MM512_PACK_A(va0, va1, va[0 * COLS + col], va[1 * COLS + col]);
      MM512_PACK_A(va2, va3, va[2 * COLS + col], va[3 * COLS + col]);

      // accumulate
      vc[col * 2 + 0] = _mm512_dpbf16_ps(vc[col * 2 + 0], va0, vb[0 * COLS + col]);
      vc[col * 2 + 0] = _mm512_dpbf16_ps(vc[col * 2 + 0], va2, vb[2 * COLS + col]);
      vc[col * 2 + 1] = _mm512_dpbf16_ps(vc[col * 2 + 1], va1, vb[1 * COLS + col]);
      vc[col * 2 + 1] = _mm512_dpbf16_ps(vc[col * 2 + 1], va3, vb[3 * COLS + col]);
    };

    using fVec = at::vec::Vectorized<float>;
    using bVec = at::vec::Vectorized<at::BFloat16>;
    const fVec one = fVec(1.f);
    auto storec = [&](auto i, int64_t m) {
      constexpr int col = i;
      fVec x0 = fVec(vc[col * 2 + 0]);
      fVec x1 = fVec(vc[col * 2 + 1]);
      // print_16x32(vc[col * 2 + 0]);
      // print_16x32(vc[col * 2 + 1]);
      x0 = x0 / (one + x0.neg().exp_u20());
      x1 = x1 / (one + x1.neg().exp_u20());
      bVec out_vec = convert_from_float_ext<at::BFloat16>(x0, x1);
      out_vec.store(C + m * lda + col * 32);
      // print_32x16(__m512bh((__m512i)(out_vec)));
    };

    for (int64_t m = 0; m < M; ++m) {
      // step 3.a : load a at current time step
      Unroll<COLS>{}(loada, m);
      // step 3.b : accumulate for window size (4)
      Unroll<COLS>{}(compute);
      // step 3.c : store c at current time step
      Unroll<COLS>{}(storec, m);
    }
  }
};
#endif

#define LAUNCH_TINYGEMM_KERNEL(K, NB_SIZE)                                     \
  tinygemm_kernel<scalar_t, K, NB_SIZE, has_bias>::apply(                      \
      input + bs * seqlen * dim + mb_start * dim + nb_start,                   \
      weight + nb_start * width,                                               \
      out + bs * seqlen * dim + mb_start * dim + nb_start,                     \
      has_bias ? bias + nb_start : nullptr,                                    \
      has_conv_states ? conv_states + bs * (K - 1) * dim + nb_start : nullptr, \
      has_initial_states_value,                                                \
      mb_size,                                                                 \
      dim,                                                                     \
      mb_start == 0);

template <typename scalar_t>
void causal_conv1d_fwd_kernel_impl(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ conv_states,
    const int32_t* __restrict__ query_start_loc,
    const int32_t* __restrict__ cache_indices,
    const bool* __restrict__ has_initial_state,
    bool silu_activation,
    int64_t pad_slot_id,
    int64_t batch,
    int64_t dim,
    int64_t seqlen,
    int64_t width,
    int64_t num_seq_blocks) {
  // std::cout << "### causal_conv1d_fwd_kernel_impl: batch = " << batch << "; dim = " << dim << "; seqlen = " << seqlen
  // << "; width = " << width << std::endl; std::cout << "### causal_conv1d_fwd_kernel_impl: num_seq_blocks = " <<
  // num_seq_blocks << std::endl;

  // handle 32 x 64 per block
  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n() * 2;
  const int64_t NB = div_up(dim, BLOCK_N);

  const int64_t num_blocks_per_seq = div_up(seqlen, BLOCK_M);
  const bool has_conv_states = conv_states != nullptr;

  // parallel on [batch, seq, NB]
  AT_DISPATCH_BOOL(bias != nullptr, has_bias, [&] {
    at::parallel_for(0, num_seq_blocks * NB, 0, [&](int64_t begin, int64_t end) {
      int64_t mb{0}, nb{0};
      data_index_init(begin, mb, num_seq_blocks, nb, NB);

      for (int64_t i = begin; i < end; ++i) {
        int64_t bs = mb / num_blocks_per_seq;

        int64_t mb_start = (mb % num_blocks_per_seq) * BLOCK_M;
        int64_t mb_size = std::min(seqlen - mb_start, BLOCK_M);
        int64_t nb_start = nb * BLOCK_N;
        int64_t nb_size = std::min(dim - nb_start, BLOCK_N);

        // std::cout << "### bs = " << bs << "; mb_start = " << mb_start << "; nb_start = " << nb_start << "; nb_size =
        // " << nb_size << std::endl;

        const bool has_initial_states_value = has_conv_states ? has_initial_state[bs] : false;

        switch (width << 4 | nb_size >> 4) {
          case 0x42:
            LAUNCH_TINYGEMM_KERNEL(4, 32);
            break;
          case 0x44:
            LAUNCH_TINYGEMM_KERNEL(4, 64);
            break;
          default:
            TORCH_CHECK(false, "Unexpected block size, ", width, " x ", nb_size);
        }

        // move to the next index
        data_index_step(mb, num_seq_blocks, nb, NB);
      }
    });
  });

  // update conv_states if necessary
  if (has_conv_states) {
    at::parallel_for(0, batch, 0, [&](int64_t begin, int64_t end) {
      for (int64_t bs = begin; bs < end; ++bs) {
        const bool has_initial_states_value = has_initial_state[bs];
        if (has_initial_states_value) {
          std::memcpy(
              conv_states + bs * (width - 1) * dim,
              input + bs * seqlen * dim + (seqlen - 3) * dim,
              (width - 1) * dim * sizeof(at::BFloat16));
        }
      }
    });
  }
}

}  // anonymous namespace

// from [dim, width] or [N, K]
// to [N/BLOCK_N, K/2, BLOCK_N, 2]
at::Tensor causal_conv1d_weight_pack(const at::Tensor& weight) {
  CHECK_INPUT(weight);

  int64_t dim = weight.size(0);
  int64_t width = weight.size(1);
  constexpr int64_t BLOCK_N = block_size_n();
  TORCH_CHECK(width == 4, "causal_conv1d_weight_pack: support only width of 4");
  TORCH_CHECK(dim % BLOCK_N == 0, "causal_conv1d_weight_pack: invalid dim size ", dim);

  const int64_t N = dim, K2 = width >> 1;
  const int64_t NB = div_up(N, BLOCK_N);

  auto packed_weight = at::empty_like(weight);
  AT_DISPATCH_REDUCED_FLOATING_TYPES(weight.scalar_type(), "causal_conv1d_fwd_kernel_impl", [&] {
    // cast to float32 as vnni size is 2
    const float* w_data = reinterpret_cast<float*>(weight.data_ptr<scalar_t>());
    float* packed_data = reinterpret_cast<float*>(packed_weight.data_ptr<scalar_t>());

    at::parallel_for(0, NB * K2 * BLOCK_N, 0, [&](int64_t begin, int64_t end) {
      int64_t nb{0}, k2{0}, n{0};
      data_index_init(begin, nb, NB, k2, K2, n, BLOCK_N);

      // TODO: optimize this if we need to online prepacking.
      for (int64_t i = begin; i < end; ++i) {
        packed_data[i] = w_data[nb * BLOCK_N * K2 + n * K2 + k2];

        // move to the next index
        data_index_step(nb, NB, k2, K2, n, BLOCK_N);
      }
    });
  });
  return packed_weight;
}

#define CHECK_OPTIONAL_SHAPE_DTYPE(OPT, SIZE, DTYPE) \
  if (OPT.has_value()) {                             \
    const auto tensor = OPT.value();                 \
    CHECK_CONTIGUOUS(tensor);                        \
    CHECK_EQ(tensor.size(0), SIZE);                  \
    CHECK_EQ(tensor.scalar_type(), DTYPE);           \
  }

template <int BLOCK_M>
int64_t get_block_count(const std::optional<at::Tensor>& offsets, int64_t batch, int64_t seqlen) {
  if (offsets.has_value()) {
    const int32_t* offsets_data = offsets.value().data_ptr<int32_t>();
    int32_t num_seq_blocks = 0;
    for (int64_t row = 0; row < batch; ++row) {
      num_seq_blocks += div_up(offsets_data[row + 1] - offsets_data[row], BLOCK_M);
    }
    return num_seq_blocks;
  }
  return batch * div_up(seqlen, int64_t(BLOCK_M));
}

template <int BLOCK_M>
void get_block_indices(
    at::Tensor& indices, const std::optional<at::Tensor>& offsets, int64_t batch, int64_t num_seq_blocks) {
  if (!offsets.has_value()) {
    return;
  }

  const int32_t* offsets_data = offsets.value().data_ptr<int32_t>();
  indices.resize_({num_seq_blocks * 2});
  int32_t* indices_data = indices.data_ptr<int32_t>();

  int64_t idx = 0;
  for (int32_t row = 0; row < batch; ++row) {
    int32_t blocks = div_up(offsets_data[row + 1] - offsets_data[row], BLOCK_M);

    for (int32_t col = 0; col < blocks; ++col) {
      indices_data[idx * 2 + 0] = row;
      indices_data[idx * 2 + 1] = col;
      idx++;
    }
  }
}

// API aligned with GPUs
//
//   x: (batch, dim, seqlen) or (dim, cu_seq_len) for varlen
//   weight: (dim, width)
//   bias: (dim,)
//   query_start_loc: (batch + 1) int32
//   cache_indices: (batch)  int32
//   has_initial_state: (batch) bool
//   conv_states: (..., dim, width - 1) itype
//   activation: either None or "silu" or "swish"
//   pad_slot_id: int
//
at::Tensor causal_conv1d_fwd_cpu(
    const at::Tensor& x,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    const std::optional<at::Tensor>& conv_states,
    const std::optional<at::Tensor>& query_start_loc,
    const std::optional<at::Tensor>& cache_indices,
    const std::optional<at::Tensor>& has_initial_state,
    bool silu_activation,
    int64_t pad_slot_id,
    bool is_vnni) {
  RECORD_FUNCTION("sgl-kernel::causal_conv1d_fwd_cpu", std::vector<c10::IValue>({x, weight, bias}));

  CHECK_CONTIGUOUS(weight);
  auto packed_w = is_vnni ? weight : causal_conv1d_weight_pack(weight);

  const bool is_var_seqlen = query_start_loc.has_value();
  TORCH_CHECK(!is_var_seqlen, "causal_conv1d_fwd_cpu: doesn't support variant sequence lengths.");

  const int64_t input_ndim = is_var_seqlen ? 2 : 3;
  TORCH_CHECK(x.dim() == input_ndim, "causal_conv1d_fwd_cpu: expect x to be ", input_ndim, "D tensor.");
  TORCH_CHECK(x.stride(-2) == 1 && x.stride(-1) == x.size(-2), "causal_conv1d_fwd_cpu: expect x to be transposed.");

  const int64_t batch = is_var_seqlen ? query_start_loc.value().size(0) - 1 : x.size(0);
  const int64_t dim = x.size(-2);
  const int64_t seqlen = x.size(-1);
  const int64_t width = weight.size(-1);

  const auto scalar_type = x.scalar_type();
  CHECK_EQ(weight.scalar_type(), scalar_type);
  CHECK_OPTIONAL_SHAPE_DTYPE(bias, dim, scalar_type);
  CHECK_OPTIONAL_SHAPE_DTYPE(query_start_loc, batch + 1, at::kInt);
  CHECK_OPTIONAL_SHAPE_DTYPE(cache_indices, batch, at::kInt);
  CHECK_OPTIONAL_SHAPE_DTYPE(has_initial_state, batch, at::kBool);

  if (conv_states.has_value()) {
    // std::cout << "### conv_states.value().sizes(): " << conv_states.value().sizes() << "; " <<
    // conv_states.value().strides() << std::endl; std::cout << conv_states.value() << std::endl;

    auto& conv_states_val = conv_states.value();
    CHECK_EQ(conv_states_val.scalar_type(), scalar_type);
    CHECK_EQ(conv_states_val.size(0), batch);
    CHECK_EQ(conv_states_val.size(1), dim);
    CHECK_EQ(conv_states_val.size(2), width - 1);

    // adjust `conv_states` to be contiguous on `dim`
    if (conv_states_val.stride(-2) != 1) {
      std::cout << "### conv_states_val.stride on dim is not 1 ..." << std::endl;
      auto conv_states_copy = conv_states_val.clone();
      conv_states_val.as_strided_({batch, dim, width - 1}, {(width - 1) * dim, 1, dim});
      conv_states_val.copy_(conv_states_copy);
    }
    // std::cout << "### conv_states.value().sizes() after: " << conv_states.value().sizes() << "; "
    // <<conv_states.value().strides() << std::endl; std::cout << conv_states.value() << std::endl;
  }

  constexpr int64_t BLOCK_M = block_size_m();

  // total number of sequence blocks
  int64_t num_seq_blocks = get_block_count<BLOCK_M>(query_start_loc, batch, seqlen);

  // record seq blocks in Coordinate format, aka [num_seq_blocks, 2]

  at::Tensor out = at::empty_like(x);
  AT_DISPATCH_REDUCED_FLOATING_TYPES(scalar_type, "causal_conv1d_fwd_kernel_impl", [&] {
    causal_conv1d_fwd_kernel_impl<scalar_t>(
        out.data_ptr<scalar_t>(),
        x.data_ptr<scalar_t>(),
        packed_w.data_ptr<scalar_t>(),
        conditional_data_ptr<scalar_t>(bias),
        conditional_data_ptr<scalar_t>(conv_states),
        conditional_data_ptr<int32_t>(query_start_loc),
        conditional_data_ptr<int32_t>(cache_indices),
        conditional_data_ptr<bool>(has_initial_state),
        silu_activation,
        pad_slot_id,
        batch,
        dim,
        seqlen,
        width,
        num_seq_blocks);
  });
  return out;
}
