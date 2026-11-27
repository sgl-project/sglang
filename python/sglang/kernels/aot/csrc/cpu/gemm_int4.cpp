#include <torch/all.h>

#include "gemm.h"
#include "vec.h"

namespace {

constexpr int64_t BLOCK_M = block_size_m();
constexpr int64_t BLOCK_N = block_size_n();
// Symmetrically quantized int8 activations are shifted by +128 into uint8 for u8s8 VNNI/AMX.
constexpr int32_t kActZp = 128;

#if defined(CPU_CAPABILITY_AVX512)
struct m256i_pair {
  __m256i low;
  __m256i high;
};

inline m256i_pair load_zps_4vnni(const int8_t* __restrict__ zps) {
  __m256i vzps_low = _mm256_set1_epi64x(*reinterpret_cast<const long*>(zps));
  __m256i vzps_high = _mm256_set1_epi64x(*reinterpret_cast<const long*>(zps + 8));
  // Broadcast each byte 4x for VNNI4: [0..7]x4 -> [0,0,0,0, ..., 7,7,7,7]
  __m256i shuffle_mask =
      _mm256_set_epi8(7, 7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
  return {_mm256_shuffle_epi8(vzps_low, shuffle_mask), _mm256_shuffle_epi8(vzps_high, shuffle_mask)};
}

inline m256i_pair load_uint4_as_int8(const uint8_t* __restrict__ qB) {
  __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qB));
  const __m256i low_mask = _mm256_set1_epi8(0x0f);
  __m256i low = _mm256_and_si256(packed, low_mask);
  __m256i high = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);
  return {low, high};
}

inline __m512i combine_m256i(m256i_pair pair) {
  return _mm512_inserti64x4(_mm512_castsi256_si512(pair.low), pair.high, 1);
}

template <int64_t N, int64_t ldb>
void _dequant_weight_zp_only(const uint8_t* __restrict__ B, int8_t* dqB, const int8_t* __restrict__ qzeros, int64_t K) {
  // B layout: [K / 4, N / 2, 4] (2 int4 lanes per byte), dqB layout: [K / 4, N, 4]
#pragma GCC unroll 2
  for (int n = 0; n < N; n += 16) {
    auto zps = load_zps_4vnni(&qzeros[n]);
    for (int k = 0; k < K; k += 4) {
      auto vb = load_uint4_as_int8(B + ldb * k + n / 2 * 4);
      _mm256_storeu_si256(reinterpret_cast<__m256i_u*>(dqB + N * k + n * 4), _mm256_sub_epi8(vb.low, zps.low));
      _mm256_storeu_si256(reinterpret_cast<__m256i_u*>(dqB + N * k + (n + 8) * 4), _mm256_sub_epi8(vb.high, zps.high));
    }
  }
}

template <int64_t N>
void _dequant_and_accum(
    float* __restrict__ output,
    const int32_t* __restrict__ input,
    const float* __restrict__ scale_a,
    const float* __restrict__ scale_b,
    const int32_t* __restrict__ comp_b,
    int M,
    int ldi,
    int ldo) {
  static_assert(N % 16 == 0, "_dequant_and_accum requires N to be a multiple of 16");
  const __m512i va_zp = _mm512_set1_epi32(kActZp);
  for (int m = 0; m < M; ++m) {
    __m512 va_scale = _mm512_set1_ps(scale_a[m]);
#pragma GCC unroll 2
    for (int n = 0; n < N; n += 16) {
      __m512i vc = _mm512_loadu_si512(input + m * ldi + n);
      __m512i vb_comp = _mm512_loadu_si512(comp_b + n);
      vc = _mm512_sub_epi32(vc, _mm512_mullo_epi32(vb_comp, va_zp));
      __m512 vc_f = _mm512_mul_ps(_mm512_cvtepi32_ps(vc), va_scale);
      vc_f = _mm512_mul_ps(vc_f, _mm512_loadu_ps(scale_b + n));
      __m512 vo = _mm512_loadu_ps(output + m * ldo + n);
      _mm512_storeu_ps(output + m * ldo + n, _mm512_add_ps(vo, vc_f));
    }
  }
}

template <int64_t M, int64_t N, int64_t ldb>
void _dequant_gemm_accum_small_M(
    float* __restrict__ C,
    const uint8_t* A,
    const float* scales_a,
    const uint8_t* B,
    const float* scales_b,
    const int8_t* qzeros_b,
    int64_t K,
    int64_t lda,
    int64_t ldc) {
  constexpr int COLS = N / 16;
  // Computing compensation on the fly is faster than loading it for small M (memory-bound).
  __m512i ones = _mm512_set1_epi8(1);
  __m512i va_zp = _mm512_set1_epi32(kActZp);
  __m512i va;
  __m512i vb[COLS];
  __m512i vc[M * COLS];
  __m512 vscales[COLS];
  __m512i vzps[COLS];
  __m512i vcompensate[COLS];

  Unroll<COLS>{}([&](auto i) {
    vscales[i] = _mm512_loadu_ps(scales_b + i * 16);
    vzps[i] = combine_m256i(load_zps_4vnni(qzeros_b + i * 16));
    vcompensate[i] = _mm512_setzero_epi32();
  });
  Unroll<M * COLS>{}([&](auto i) { vc[i] = _mm512_setzero_epi32(); });

  auto compute = [&](auto i, int k) {
    constexpr int row = i / COLS;
    constexpr int col = i % COLS;

    if constexpr (col == 0) {
      va = _mm512_set1_epi32(*(const int32_t*)(A + row * lda + k));
    }
    if constexpr (row == 0) {
      int B_offset = k * ldb + col * 16 * 2;
      vb[col] = _mm512_sub_epi8(combine_m256i(load_uint4_as_int8(B + B_offset)), vzps[col]);
      vcompensate[col] = _mm512_dpbusd_epi32(vcompensate[col], ones, vb[col]);
      _mm_prefetch(B + B_offset + 128 * ldb, _MM_HINT_T0);
    }
    vc[i] = _mm512_dpbusd_epi32(vc[i], va, vb[col]);
  };

  constexpr int unroll = 4;
  int k = 0;
  for (; k < K / 4 / unroll; k++) {
    Unroll<unroll>{}([&](auto i) { Unroll<M * COLS>{}(compute, 4 * (k * unroll + i)); });
  }
  k *= 4 * unroll;
  for (; k < K; k += 4) {
    Unroll<M * COLS>{}(compute, k);
  }

  Unroll<M * COLS>{}([&](auto i) {
    constexpr int row = i / COLS;
    constexpr int col = i % COLS;
    vc[i] = _mm512_sub_epi32(vc[i], _mm512_mullo_epi32(vcompensate[col], va_zp));
    __m512 vc_float = _mm512_mul_ps(_mm512_cvtepi32_ps(vc[i]), _mm512_set1_ps(scales_a[row]));
    vc_float = _mm512_mul_ps(vc_float, vscales[col]);
    vc_float = _mm512_add_ps(vc_float, _mm512_loadu_ps(C + row * ldc + col * 16));
    _mm512_storeu_ps(C + row * ldc + col * 16, vc_float);
  });
}

#define CALL_DEQUANT_GEMM_ACCUM_SMALL_M(M) \
  _dequant_gemm_accum_small_M<M, N, ldb>(C, A, scales_a, B, scales_b, qzeros_b, K, lda, ldc);
#endif

template <int64_t N, int64_t ldb>
void _dequant_gemm_accum(
    float* C,
    const uint8_t* A,
    const float* scales_a,
    const uint8_t* B,
    const float* scales_b,
    const int8_t* qzeros_b,
    const int32_t* compensation,
    int8_t* dqB,
    int64_t M,
    int64_t K,
    int64_t lda,
    int64_t ldc,
    bool use_brgemm,
    bool do_dequant) {
#if defined(CPU_CAPABILITY_AVX512)
  if (!use_brgemm) {
    switch (M) {
      case 1:
        CALL_DEQUANT_GEMM_ACCUM_SMALL_M(1);
        break;
      case 2:
        CALL_DEQUANT_GEMM_ACCUM_SMALL_M(2);
        break;
      case 3:
        CALL_DEQUANT_GEMM_ACCUM_SMALL_M(3);
        break;
      case 4:
        CALL_DEQUANT_GEMM_ACCUM_SMALL_M(4);
        break;
      default:
        TORCH_CHECK(false, "tinygemm_kernel: unexpected M for AVX path!");
    }
    return;
  }

  if (do_dequant) {
    _dequant_weight_zp_only<N, ldb>(B, dqB, qzeros_b, K);
  }
  int32_t C_i32[M * N];
  at::native::cpublas::brgemm(
      M, N, K, lda, N /*ldb*/, N /*ldc*/, false /* add_C */, const_cast<uint8_t*>(A), dqB, C_i32, true /* is_vnni */);
  _mm_prefetch(B + N * K / 2, _MM_HINT_T0);
  _mm_prefetch(A + K, _MM_HINT_T0);
  _dequant_and_accum<N>(C, C_i32, scales_a, scales_b, compensation, M, N /*ldi*/, ldc);
#else
  TORCH_CHECK(false, "tinygemm_kernel: scalar path not implemented!");
#endif
}

template <int64_t N>
inline void copy_bias(const float* bias_ptr, float* y_buf, int64_t m) {
  using Vec = at::vec::Vectorized<float>;
  constexpr int kVecSize = Vec::size();
  static_assert(N % kVecSize == 0, "copy_bias requires N to be a multiple of Vectorized<float>::size()");
  const bool has_bias = bias_ptr != nullptr;
  const Vec zero_vec(0.f);
  for (int64_t i = 0; i < m; ++i) {
#pragma GCC unroll 2
    for (int64_t j = 0; j < N; j += kVecSize) {
      Vec vec = has_bias ? Vec::loadu(bias_ptr + j) : zero_vec;
      vec.store(y_buf + i * N + j);
    }
  }
}

template <typename out_dtype>
void _da8w4_linear_impl(
    const uint8_t* __restrict__ input,
    const float* __restrict__ input_scales,
    const uint8_t* __restrict__ weight,
    const float* __restrict__ weight_scales,
    const int8_t* __restrict__ weight_qzeros,
    const float* __restrict__ bias,
    out_dtype* __restrict__ output,
    float* __restrict__ output_temp,
    int8_t* __restrict__ dequant_weight_temp,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t num_groups) {
  // weight + compensation shape = [Nc, Kc, BLOCK_N * _block_k / 2 + BLOCK_N * sizeof(int32_t)]
  // scales/qzeros shape = [Nc, G, BLOCK_N]
  const bool use_brgemm = can_use_brgemm<int8_t>(M);
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = N / BLOCK_N;
  const int64_t group_size = div_up(K, num_groups);
  const int64_t _block_k = get_4bit_block_k_size(group_size);
  const int64_t Kc = K / _block_k;
  const int64_t block_per_group = group_size / _block_k;
  const int64_t block_stride = BLOCK_N * (_block_k / 2 + sizeof(int32_t));
  const bool cache_dqB = use_brgemm && (MB > 1);
  const int64_t cache_blocks_nb = get_cache_blocks<int8_t>(BLOCK_N * K);
  const int64_t dqB_size_per_thread = cache_dqB ? cache_blocks_nb * BLOCK_N * K : _block_k * BLOCK_N;

  parallel_2d(MB, NB, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
    int tid = get_thread_num();
    float* C_tmp = output_temp + tid * BLOCK_M * BLOCK_N;
    int8_t* dqB_thread = dequant_weight_temp + tid * dqB_size_per_thread;

    loop_2d<int8_t>(mb0, mb1, nb0, nb1, BLOCK_N * K, [&](int64_t mb, int64_t nb, int64_t nb_offset) {
      int64_t m_start = mb * BLOCK_M;
      int64_t m_size = std::min(M - m_start, BLOCK_M);
      copy_bias<BLOCK_N>(bias ? bias + nb * BLOCK_N : nullptr, C_tmp, m_size);
      const bool do_dequant = !cache_dqB || (mb == mb0);
      int8_t* dqB_nb = cache_dqB ? (dqB_thread + nb_offset * BLOCK_N * K) : dqB_thread;
      for (int kci = 0; kci < Kc; ++kci) {
        const uint8_t* B_ptr = weight + (nb * Kc + kci) * block_stride;
        const int32_t* compensation_ptr = reinterpret_cast<const int32_t*>(B_ptr + _block_k * BLOCK_N / 2);
        int8_t* dqB_tmp = cache_dqB ? (dqB_nb + kci * _block_k * BLOCK_N) : dqB_nb;
        _dequant_gemm_accum<BLOCK_N, BLOCK_N / 2>(
            /*C*/ C_tmp,
            /*A*/ input + m_start * K + kci * _block_k,
            /*scales_a*/ input_scales + m_start,
            /*B*/ B_ptr,
            /*scales_b*/ weight_scales + nb * BLOCK_N * num_groups + kci / block_per_group * BLOCK_N,
            /*qzeros_b*/ weight_qzeros + nb * BLOCK_N * num_groups + kci / block_per_group * BLOCK_N,
            /*Bcomp*/ compensation_ptr,
            /*dqB_tmp*/ dqB_tmp,
            /*M*/ m_size,
            /*K*/ _block_k,
            /*lda*/ K,
            /*ldc*/ BLOCK_N,
            /*use_brgemm*/ use_brgemm,
            /*do_dequant*/ do_dequant);
      }
      for (int64_t m = 0; m < m_size; ++m) {
        copy_stub<out_dtype>(output + (m_start + m) * N + nb * BLOCK_N, C_tmp + m * BLOCK_N, BLOCK_N);
      }
    });

    if (use_brgemm) {
      at::native::cpublas::brgemm_release();
    }
  });
}

}  // anonymous namespace

/*
return: packed_weight, packed_scales, packed_qzeros
*/
std::tuple<at::Tensor, at::Tensor, at::Tensor> convert_int4_weight_packed_with_compensation(
    const at::Tensor& weight, const at::Tensor& scales, const at::Tensor& qzeros) {
  // weight shape = [N, K]
  // scales shape = [N, G]
  // qzeros shape = [N, G]
  TORCH_CHECK(weight.dim() == 2, "DA8W4 CPU: Weight should be a 2D tensor for packing");
  TORCH_CHECK(weight.size(1) % 2 == 0, "DA8W4 CPU: Weight should have even number of columns for packing");

  auto new_scales = scales;
  auto new_qzeros = qzeros;
  if (new_scales.dim() == 1) {
    new_scales.unsqueeze_(1);
  }
  new_scales = new_scales.to(at::kFloat);
  if (new_qzeros.dim() == 1) {
    new_qzeros.unsqueeze_(1);
  }
  new_qzeros = new_qzeros.to(at::kChar);
  int64_t N = weight.size(0);
  int64_t K = weight.size(1);
  int64_t G = scales.size(1);
  int64_t group_size = K / G;
  int64_t _block_k = get_4bit_block_k_size(group_size);
  constexpr int block_n = block_size_n();
  int64_t Nc = N / block_n;
  int64_t Kc = K / _block_k;

  // Reorder weight to [N/block_n, K/_block_k, _block_k, block_n]
  // Reorder scales/qzeros to [N/block_n, G, block_n]
  // weight + compensation shape = [Nc, Kc, block_n * _block_k / 2 + block_n*sizeof(int32_t)]
  // scales/qzeros shape = [Nc, G, block_n]
  auto weight_view = weight.view({Nc, block_n, Kc, _block_k});
  at::Tensor weight_reordered = weight_view.permute({0, 2, 3, 1}).contiguous();
  at::Tensor blocked_weight;
  at::Tensor blocked_scales = new_scales.view({Nc, block_n, G}).permute({0, 2, 1}).contiguous();
  at::Tensor blocked_qzeros = new_qzeros.view({Nc, block_n, G}).permute({0, 2, 1}).contiguous();
  // Compensation = Σ(k)(W[k][n] - ZP[n]) for each block.
  auto weight_sub_qzero = weight.view({Nc, block_n, G, -1}).to(at::kInt) - new_qzeros.view({Nc, block_n, G, -1});
  weight_sub_qzero = weight_sub_qzero.view({Nc, block_n, Kc, _block_k});
  at::Tensor compensation = weight_sub_qzero.sum(-1);
  compensation = compensation.permute({0, 2, 1}).contiguous().to(at::kInt);
  int64_t buffer_size_nbytes = _block_k * block_n / 2 + block_n * sizeof(int32_t);
  blocked_weight = at::empty({Nc, Kc, buffer_size_nbytes}, weight.options());

  auto weight_ptr = weight_reordered.data_ptr<uint8_t>();
  auto compensation_ptr = compensation.data_ptr<int32_t>();
  auto blocked_weight_ptr = blocked_weight.data_ptr<uint8_t>();
  int64_t num_blocks = Nc * Kc;
  at::parallel_for(0, num_blocks, 1, [&](int64_t begin, int64_t end) {
    for (const auto i : c10::irange(begin, end)) {
      auto in_ptr = weight_ptr + i * _block_k * block_n;
      auto out_ptr = blocked_weight_ptr + i * block_n * (_block_k / 2 + sizeof(int32_t));
      int32_t* comp_in_prt = compensation_ptr + i * block_n;
      int32_t* comp_out_prt = (int32_t*)(void*)(blocked_weight_ptr + i * block_n * (_block_k / 2 + sizeof(int32_t)) +
                                                _block_k * block_n / 2);
      // Reorder weight block to VNNI4 and pack two lanes along N
      // N=16 viewed as two lanes: a0, ...a7, b0, ...b7
      // pack two lanes: [a0, b0], ..., [a7, b7]
      // plain shape = [_block_k, block_n]
      // packed shape = [_block_k / 4, block_n / 2, 4] viewed as [_block_k, block_n / 2]
      constexpr int n_group_size = 8;
      constexpr int vnni_size = 4;
      constexpr int n_group = block_n / n_group_size;  // 4
      for (int nb = 0; nb < n_group; nb += 2) {
        for (int k = 0; k < _block_k; k += vnni_size) {
          for (int ni = 0; ni < n_group_size; ++ni) {
            for (int ki = 0; ki < vnni_size; ++ki) {
              int src_idx_1 = nb * n_group_size + ni + (k + ki) * block_n;
              int src_idx_2 = (nb + 1) * n_group_size + ni + (k + ki) * block_n;
              int dst_idx = (nb / 2 * n_group_size + ni) * vnni_size + k * block_n / 2 + ki;
              uint8_t src_1 = *(in_ptr + src_idx_1);
              uint8_t src_2 = *(in_ptr + src_idx_2);
              uint8_t dst = (src_1 & 0x0f) | ((src_2 & 0x0f) << 4);
              *(out_ptr + dst_idx) = dst;
            }
          }
        }
      }
      // compensation [block_n]
      for (int nb = 0; nb < block_n; nb++) {
        *(comp_out_prt + nb) = *(comp_in_prt + nb);
      }
    }
  });

  return std::make_tuple(std::move(blocked_weight), std::move(blocked_scales), std::move(blocked_qzeros));
}

std::tuple<at::Tensor, at::Tensor>
int4pack(at::Tensor qweight, at::Tensor qzeros, int64_t quant_method_4bit, bool is_symmetric) {
  TORCH_CHECK(qweight.scalar_type() == at::kInt, "qweight must be int32");
  TORCH_CHECK(qzeros.scalar_type() == at::kInt, "qzeros must be int32");
  qweight = qweight.contiguous();
  qzeros = qzeros.contiguous();
  if (quant_method_4bit == CPUQuantAlgo::AWQ) {
    // is_symmetric is GPTQ-only: symmetric AWQ/AutoRound checkpoints store their zero point in qzeros, so it is read
    // as-is.
    auto bitshifts = at::tensor({0, 4, 1, 5, 2, 6, 3, 7}, at::kInt) * 4;
    auto qweight_final = (at::bitwise_right_shift(qweight.unsqueeze(-1), bitshifts) & 0xF)
                             .flatten(-2)
                             .transpose(-1, -2)
                             .contiguous()
                             .to(at::kByte);
    auto qzeros_final =
        (at::bitwise_right_shift(qzeros.unsqueeze(-1), bitshifts) & 0xF).flatten(-2).contiguous().to(at::kByte);
    return std::make_tuple(qweight_final, qzeros_final);
  } else if (quant_method_4bit == CPUQuantAlgo::GPTQ) {
    auto bitshifts = at::arange(0, 32, 4, at::kInt);
    auto qweight_final = (at::bitwise_right_shift(qweight.unsqueeze(-2), bitshifts.view({8, 1})) & 0xF)
                             .flatten(-3, -2)
                             .transpose(-1, -2)
                             .contiguous()
                             .to(at::kByte);
    at::Tensor qzeros_final;
    if (is_symmetric) {
      // Symmetric uint4 uses zero point 2^(4-1)=8.
      auto qz_sizes = qzeros.sizes().vec();
      qz_sizes.back() *= 8;
      qzeros_final = at::full(qz_sizes, 1 << (4 - 1), qzeros.options().dtype(at::kByte));
    } else {
      qzeros_final =
          ((at::bitwise_right_shift(qzeros.unsqueeze(-1), bitshifts) & 0xF) + 1).flatten(-2).contiguous().to(at::kByte);
    }
    return std::make_tuple(qweight_final, qzeros_final);
  } else {
    TORCH_CHECK(false, "CPU int4 pack only support AWQ or GPTQ...");
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> convert_weight_packed_scale_zp(
    at::Tensor qweight,  // awq: (*, K, N / 8)  ||  gptq: (*, K / 8, N) , int32
    at::Tensor qzeros,   // awq: (*, K / group_size, N / 8) ||  gptq: (*, K / group_size, N / 8) , int32
    at::Tensor scales,   // awq: (*, K / group_size, N) ||  gptq: (*, K / group_size, N) , bfloat16
    int64_t quant_method_4bit,
    bool is_symmetric) {  // GPTQ only: true forces zero point 8; ignored for AWQ
  at::Tensor _qweight;
  at::Tensor _qzeros;

  auto res = int4pack(qweight, qzeros, quant_method_4bit, is_symmetric);
  _qweight = std::get<0>(res);
  _qzeros = std::get<1>(res);

  auto _scales = scales;
  _qzeros = _qzeros.transpose(-2, -1).contiguous();  // .T
  _scales = _scales.transpose(-2, -1).contiguous();
  if (_qweight.dim() == 3) {  // Dim=3 for MOE packing, TODO: refine a unified loop
    int64_t E = _qweight.size(0);
    int64_t K = _qweight.size(2);
    int64_t G = _scales.size(2);
    int64_t group_size = K / G;
    int64_t _block_k = get_4bit_block_k_size(group_size);
    int64_t block_n = block_size_n();
    int64_t Nc = _qweight.size(1) / block_n;
    int64_t Kc = K / _block_k;
    int64_t buffer_size_nbytes = _block_k * block_n / 2 + block_n * sizeof(int32_t);
    auto blocked_weight = at::empty({E, Nc, Kc, buffer_size_nbytes}, _qweight.options());
    auto blocked_scales = at::empty({E, Nc, G, block_n}, _scales.options()).to(at::kFloat);
    auto blocked_qzeros = at::empty({E, Nc, G, block_n}, _qzeros.options()).to(at::kChar);
    for (int i = 0; i < _qweight.size(0); i++) {
      auto res_ = convert_int4_weight_packed_with_compensation(_qweight[i], _scales[i], _qzeros[i]);
      blocked_weight[i] = std::get<0>(res_);
      blocked_scales[i] = std::get<1>(res_);
      blocked_qzeros[i] = std::get<2>(res_);
    }
    _qweight = blocked_weight;
    _scales = blocked_scales;
    _qzeros = blocked_qzeros;
  } else {
    auto res_ = convert_int4_weight_packed_with_compensation(_qweight, _scales, _qzeros);
    _qweight = std::get<0>(res_);
    _scales = std::get<1>(res_);
    _qzeros = std::get<2>(res_);
  }

  return std::make_tuple(_qweight, _qzeros, _scales);
}

at::Tensor int4_scaled_mm_cpu_with_quant(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& weight_scales,
    const at::Tensor& weight_qzeros,
    const std::optional<at::Tensor>& bias,
    at::ScalarType output_dtype) {
  int64_t M_a = input.size(0);
  int64_t K_a = input.size(1);
  int64_t lda = input.stride(0);

  const auto st = input.scalar_type();
  TORCH_CHECK(
      st == at::kBFloat16 || st == at::kHalf, "int4_scaled_mm_cpu_with_quant: expect A to be bfloat16 or half.");

  int64_t act_buffer_size = /* act quant */ M_a * K_a + /* act scale */ M_a * sizeof(float);
  auto act_buffer = at::empty({act_buffer_size}, input.options().dtype(at::kByte));
  auto Aq_data = act_buffer.data_ptr<uint8_t>();
  auto As_data = reinterpret_cast<float*>(Aq_data + M_a * K_a);

  auto out_sizes = input.sizes().vec();
  int64_t N = weight_scales.size(0) * weight_scales.size(-1);
  out_sizes.back() = N;
  auto output = at::empty(out_sizes, input.options());

  int64_t Nc = weight.size(0);
  int64_t Kc = weight.size(1);
  int64_t _block_k = K_a / Kc;
  TORCH_CHECK(N == Nc * BLOCK_N, "DA8W4: weight and input shapes mismatch");
  int64_t num_groups = weight_scales.size(1);

  const uint8_t* b_ptr = weight.data_ptr<uint8_t>();
  const float* b_scales_ptr = weight_scales.data_ptr<float>();
  const int8_t* b_qzeros_ptr = weight_qzeros.data_ptr<int8_t>();
  const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
  int num_threads = at::get_num_threads();
  const bool use_brgemm = can_use_brgemm<int8_t>(M_a);
  const int64_t MB = div_up(M_a, BLOCK_M);
  const bool cache_dqB = use_brgemm && (MB > 1);
  const int64_t dqB_size_per_thread =
      cache_dqB ? int64_t(get_cache_blocks<int8_t>(BLOCK_N * K_a)) * BLOCK_N * K_a : _block_k * BLOCK_N;
  int64_t temp_buffer_size = /* output temp */ num_threads * BLOCK_M * BLOCK_N * sizeof(float) +
                             /* weight dequant temp */ num_threads * dqB_size_per_thread;
  auto c_temp_buffer = at::empty({temp_buffer_size}, input.options().dtype(at::kChar));
  float* c_temp_ptr = reinterpret_cast<float*>(c_temp_buffer.data_ptr<int8_t>());
  int8_t* dqB_temp_ptr = reinterpret_cast<int8_t*>(c_temp_ptr + num_threads * BLOCK_M * BLOCK_N);

  AT_DISPATCH_REDUCED_FLOATING_TYPES(output_dtype, "int4_scaled_mm_cpu_with_quant", [&] {
    const scalar_t* __restrict__ A_data = input.data_ptr<scalar_t>();
    scalar_t* __restrict__ c_ptr = output.data_ptr<scalar_t>();
    at::parallel_for(0, M_a, 0, [&](int64_t begin, int64_t end) {
      for (int64_t m = begin; m < end; ++m) {
        quantize_row_int8<scalar_t>(Aq_data + m * K_a, As_data[m], A_data + m * lda, K_a);
      }
    });
    _da8w4_linear_impl<scalar_t>(
        Aq_data,
        As_data,
        b_ptr,
        b_scales_ptr,
        b_qzeros_ptr,
        bias_ptr,
        c_ptr,
        c_temp_ptr,
        dqB_temp_ptr,
        M_a,
        N,
        K_a,
        num_groups);
  });

  return output;
}

template <typename scalar_t>
void tinygemm_kernel(
    scalar_t* C,
    float* C_temp,
    const uint8_t* A,
    const float* scales_a,
    const uint8_t* B,
    const float* scales_b,
    const int8_t* qzeros_b,
    const int32_t* compensation,
    int8_t* dqB_tmp,
    int64_t M,
    int64_t K,
    int64_t lda,
    int64_t ldc_f,
    int64_t ldc_s,
    bool store_out,
    bool use_brgemm,
    bool do_dequant) {
  _dequant_gemm_accum<BLOCK_N, BLOCK_N / 2>(
      C_temp, A, scales_a, B, scales_b, qzeros_b, compensation, dqB_tmp, M, K, lda, ldc_f, use_brgemm, do_dequant);
  if (store_out) {
    for (int64_t m = 0; m < M; ++m) {
      copy_stub<scalar_t>(C + m * ldc_s, C_temp + m * ldc_f, BLOCK_N);
    }
  }
}

#define INSTANTIATE_TINYGEMM_TEMPLATE(TYPE) \
  template void tinygemm_kernel<TYPE>(      \
      TYPE * C,                             \
      float* C_temp,                        \
      const uint8_t* A,                     \
      const float* scales_a,                \
      const uint8_t* B,                     \
      const float* scales_b,                \
      const int8_t* qzeros_b,               \
      const int32_t* compensation,          \
      int8_t* dqB_tmp,                      \
      int64_t M,                            \
      int64_t K,                            \
      int64_t lda,                          \
      int64_t ldc_f,                        \
      int64_t ldc_s,                        \
      bool store_out,                       \
      bool use_brgemm,                      \
      bool do_dequant)

INSTANTIATE_TINYGEMM_TEMPLATE(at::BFloat16);
INSTANTIATE_TINYGEMM_TEMPLATE(at::Half);

// int4 gemm dispatch api register
at::Tensor int4_scaled_mm_cpu(
    at::Tensor& x, at::Tensor& w, at::Tensor& w_zeros, at::Tensor& w_scales, std::optional<at::Tensor> bias) {
  return int4_scaled_mm_cpu_with_quant(x, w, w_scales, w_zeros, bias, x.scalar_type());
}
