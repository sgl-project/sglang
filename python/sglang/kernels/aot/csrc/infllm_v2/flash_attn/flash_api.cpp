/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

// Include these 2 headers instead of torch/extension.h since we don't need all of the torch headers.
#include <ATen/cuda/CUDAGeneratorImpl.h>  // For at::Generator and at::PhiloxCudaState
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cutlass/numeric_types.h>
#include <torch/nn/functional.h>
#include <torch/python.h>

#include "flash.h"
#include "hardware_info.h"
#include "philox_unpack.cuh"  // For at::cuda::philox::unpack
#include "static_switch.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) \
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

void set_params_fprop(
    Flash_fwd_params& params,
    // sizes
    const size_t b,
    const size_t seqlen_q,
    const size_t seqlen_k,
    const size_t seqlen_q_rounded,
    const size_t seqlen_k_rounded,
    const size_t h,
    const size_t h_k,
    const size_t d,
    const size_t d_rounded,
    // device pointers
    const at::Tensor q,
    const at::Tensor k,
    const at::Tensor v,
    at::Tensor out,
    void* cu_seqlens_q_d,
    void* cu_seqlens_k_d,
    void* seqused_k,
    void* p_d,
    void* softmax_lse_d,
    float p_dropout,
    float softmax_scale,
    int window_size_left,
    int window_size_right,
    const float softcap,
    bool seqlenq_ngroups_swapped = false,
    const bool unpadded_lse = false) {
  // Reset the parameters
  params = {};

  params.is_bf16 = q.dtype() == torch::kBFloat16;

  // Set the pointers and strides.
  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  // All stride are in elements, not bytes.
  params.q_row_stride = q.stride(-3);
  params.k_row_stride = k.stride(-3);
  params.v_row_stride = v.stride(-3);
  params.q_head_stride = q.stride(-2);
  params.k_head_stride = k.stride(-2);
  params.v_head_stride = v.stride(-2);
  params.o_ptr = out.data_ptr();
  params.o_row_stride = params.o_ptr ? out.stride(-3) : 0;
  params.o_head_stride = params.o_ptr ? out.stride(-2) : 0;

  if (cu_seqlens_q_d == nullptr) {
    params.q_batch_stride = q.stride(0);
    params.k_batch_stride = k.stride(0);
    params.v_batch_stride = v.stride(0);
    params.o_batch_stride = params.o_ptr ? out.stride(0) : 0;
    if (seqlenq_ngroups_swapped) {
      params.q_batch_stride *= seqlen_q;
      params.o_batch_stride *= seqlen_q;
    }
  }

  params.cu_seqlens_q = static_cast<int*>(cu_seqlens_q_d);
  params.cu_seqlens_k = static_cast<int*>(cu_seqlens_k_d);
  params.seqused_k = static_cast<int*>(seqused_k);

  // P = softmax(QK^T)
  params.p_ptr = p_d;

  // Softmax sum
  params.softmax_lse_ptr = softmax_lse_d;

  // Set the dimensions.
  params.b = b;
  params.h = h;
  params.h_k = h_k;
  params.h_h_k_ratio = h / h_k;
  params.seqlen_q = seqlen_q;
  params.seqlen_k = seqlen_k;
  params.seqlen_q_rounded = seqlen_q_rounded;
  params.seqlen_k_rounded = seqlen_k_rounded;
  params.d = d;
  params.d_rounded = d_rounded;

// Set the different scale values.
#ifdef FLASHATTENTION_DISABLE_SOFTCAP
  TORCH_CHECK(softcap <= 0.0, "This flash attention build does not support softcap.");
#endif
  if (softcap > 0.0) {
    params.softcap = softmax_scale / softcap;
    params.scale_softmax = softcap;
    params.scale_softmax_log2 = softcap * M_LOG2E;
  } else {
    // Remove potential NaN
    params.softcap = 0.0;
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;
  }

  // Set this to probability of keeping an element to simplify things.
  params.p_dropout = 1.f - p_dropout;
  // Convert p from float to int so we don't have to convert the random uint to float to compare.
  // [Minor] We want to round down since when we do the comparison we use <= instead of <
  // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
  // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
  params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
  params.rp_dropout = 1.f / params.p_dropout;
  params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
  TORCH_CHECK(p_dropout < 1.f);
#ifdef FLASHATTENTION_DISABLE_DROPOUT
  TORCH_CHECK(p_dropout == 0.0f, "This flash attention build does not support dropout.");
#endif

  // Causal is the special case where window_size_right == 0 and window_size_left < 0.
  // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
  params.is_causal = window_size_left < 0 && window_size_right == 0;

  if (window_size_left < 0 && window_size_right >= 0) {
    window_size_left = seqlen_k;
  }
  if (window_size_left >= 0 && window_size_right < 0) {
    window_size_right = seqlen_k;
  }
  params.window_size_left = window_size_left;
  params.window_size_right = window_size_right;

#ifdef FLASHATTENTION_DISABLE_LOCAL
  TORCH_CHECK(
      params.is_causal || (window_size_left < 0 && window_size_right < 0),
      "This flash attention build does not support local attention.");
#endif

  params.is_seqlens_k_cumulative = true;

#ifdef FLASHATTENTION_DISABLE_UNEVEN_K
  TORCH_CHECK(d == d_rounded, "This flash attention build does not support headdim not being a multiple of 32.");
#endif

  params.unpadded_lse = unpadded_lse;
  params.seqlenq_ngroups_swapped = seqlenq_ngroups_swapped;
}

void run_mha_fwd_split_stage1(Flash_fwd_params& params, cudaStream_t stream) {
  FP16_SWITCH(!params.is_bf16, [&] {
    HEADDIM_SWITCH(params.d, [&] {
      BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
      });
    });
  });
}

void set_params_alibi(
    Flash_fwd_params& params, c10::optional<at::Tensor>& alibi_slopes_, int batch_size, int num_heads) {
#ifdef FLASHATTENTION_DISABLE_ALIBI
  TORCH_CHECK(!alibi_slopes_.has_value(), "This flash attention build does not support alibi.");
  params.alibi_slopes_ptr = nullptr;
#else
  if (alibi_slopes_.has_value()) {
    auto alibi_slopes = alibi_slopes_.value();
    TORCH_CHECK(alibi_slopes.dtype() == torch::kFloat32, "ALiBi slopes must have dtype fp32");
    CHECK_DEVICE(alibi_slopes);
    TORCH_CHECK(alibi_slopes.stride(-1) == 1, "ALiBi slopes tensor must have contiguous last dimension");
    TORCH_CHECK(
        alibi_slopes.sizes() == torch::IntArrayRef({num_heads}) ||
        alibi_slopes.sizes() == torch::IntArrayRef({batch_size, num_heads}));
    params.alibi_slopes_ptr = alibi_slopes.data_ptr();
    params.alibi_slopes_batch_stride = alibi_slopes.dim() == 2 ? alibi_slopes.stride(0) : 0;
  } else {
    params.alibi_slopes_ptr = nullptr;
  }
#endif
}

std::vector<at::Tensor> mha_varlen_fwd_stage1(
    at::Tensor& q,        // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const at::Tensor& k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x
                          // page_block_size x num_heads_k x head_size if there's a block_table.
    const at::Tensor& v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x
                          // page_block_size x num_heads_k x head_size if there's a block_table.
    c10::optional<at::Tensor>& out_,  // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor& cu_seqlens_q,   // b+1
    const at::Tensor& cu_seqlens_k,   // b+1
    const at::Tensor& cu_seqlens_v,   // b+1
    c10::optional<at::Tensor>&
        seqused_k,  // b. If given, only this many elements of each batch element's keys are used.
    c10::optional<const at::Tensor>& leftpad_k_,  // batch_size
    c10::optional<at::Tensor>& block_table_,      // batch_size x max_num_blocks_per_seq
    c10::optional<at::Tensor>& alibi_slopes_,     // num_heads or b x num_heads
    int max_seqlen_q,
    const int max_seqlen_k,
    const float p_dropout,
    const float softmax_scale,
    const bool zero_tensors,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    const float softcap,
    const bool return_softmax,
    c10::optional<at::Generator> gen_) {
  // Otherwise the kernel will be launched from cuda:0 device
  at::cuda::CUDAGuard device_guard{q.device()};

  auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
  // bool is_sm75 = cc_major == 7 && cc_minor == 5;
  bool is_sm8x = cc_major == 8 && cc_minor >= 0;
  bool is_sm90 = cc_major == 9 && cc_minor == 0;
  // TORCH_CHECK(is_sm90 || is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");
  // We will support Turing in the near future
  // TORCH_CHECK(is_sm90 || is_sm8x || is_sm75, "FlashAttention only supports Turing GPUs or newer.");

  auto q_dtype = q.dtype();
  TORCH_CHECK(
      q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16, "FlashAttention only support fp16 and bf16 data type");
  if (q_dtype == torch::kBFloat16) {
    // TORCH_CHECK(is_sm90 || is_sm8x, "bfloat16 is only supported on Ampere GPUs or newer");
  }
  TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
  TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
  TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
  TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");
  TORCH_CHECK(cu_seqlens_v.dtype() == torch::kInt32, "cu_seqlens_v must have dtype int32");

  CHECK_DEVICE(q);
  CHECK_DEVICE(k);
  CHECK_DEVICE(v);
  CHECK_DEVICE(cu_seqlens_q);
  CHECK_DEVICE(cu_seqlens_k);
  CHECK_DEVICE(cu_seqlens_v);

  at::Tensor block_table;
  const bool paged_KV = block_table_.has_value();
  if (paged_KV) {
    block_table = block_table_.value();
    CHECK_DEVICE(block_table);
    TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must have dtype torch.int32");
    TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");
  }

  TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  CHECK_CONTIGUOUS(cu_seqlens_q);
  CHECK_CONTIGUOUS(cu_seqlens_k);
  CHECK_CONTIGUOUS(cu_seqlens_v);

  const auto sizes = q.sizes();

  const int batch_size = cu_seqlens_q.numel() - 1;
  int num_heads = sizes[1];
  const int head_size = sizes[2];
  const int num_heads_k = paged_KV ? k.size(2) : k.size(1);

  if (softcap > 0.f) {
    TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now");
  }

  const int max_num_blocks_per_seq = !paged_KV ? 0 : block_table.size(1);
  const int num_blocks = !paged_KV ? 0 : k.size(0);
  const int page_block_size = !paged_KV ? 1 : k.size(1);
  TORCH_CHECK(!paged_KV || page_block_size % 256 == 0, "Paged KV cache block size must be divisible by 256");

  if (max_seqlen_q == 1 && !alibi_slopes_.has_value()) {
    is_causal = false;
  }  // causal=true is the same as causal=false in this case
  if (is_causal) {
    window_size_right = 0;
  }

  void* cu_seqlens_q_d = cu_seqlens_q.data_ptr();

  // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
  // H/t Daniel Haziza
  const int seqlenq_ngroups_swapped = max_seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 &&
                                      window_size_right < 0 && p_dropout == 0.f && head_size % 8 == 0 &&
                                      !alibi_slopes_.has_value();
  const int ngroups = num_heads / num_heads_k;
  if (seqlenq_ngroups_swapped) {
    q = q.reshape({batch_size, num_heads_k, ngroups, head_size})
            .transpose(1, 2)
            .reshape({batch_size * ngroups, num_heads_k, head_size});
    max_seqlen_q = ngroups;
    num_heads = num_heads_k;
    cu_seqlens_q_d = nullptr;
  }

  const int total_q = q.sizes()[0];

  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  TORCH_CHECK(head_size <= 256, "FlashAttention forward only supports head dimension at most 256");
  TORCH_CHECK(head_size % 8 == 0, "query, key, value, and out_ must have a head_size that is a multiple of 8");
  TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

  if (window_size_left >= max_seqlen_k) {
    window_size_left = -1;
  }
  if (window_size_right >= max_seqlen_k) {
    window_size_right = -1;
  }

  CHECK_SHAPE(q, total_q, num_heads, head_size);
  if (!paged_KV) {
    const int total_k = k.size(0);
    CHECK_SHAPE(k, total_k, num_heads_k, head_size);
    // CHECK_SHAPE(v, total_k, num_heads_k, head_size);
  } else {
    CHECK_SHAPE(k, num_blocks, page_block_size, num_heads_k, head_size);
    // CHECK_SHAPE(v, num_blocks, page_block_size, num_heads_k, head_size);
    CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);
  }

  CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
  CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
  CHECK_SHAPE(cu_seqlens_v, batch_size + 1);
  if (seqused_k.has_value()) {
    auto seqused_k_ = seqused_k.value();
    TORCH_CHECK(seqused_k_.dtype() == torch::kInt32, "seqused_k must have dtype int32");
    TORCH_CHECK(seqused_k_.is_cuda(), "seqused_k must be on CUDA device");
    TORCH_CHECK(seqused_k_.is_contiguous(), "seqused_k must be contiguous");
    CHECK_SHAPE(seqused_k_, batch_size);
  }

  auto opts = q.options();
  at::Tensor out;
  out = torch::empty({0}, opts);
  // if (out_.has_value()) {
  //     out = out_.value();
  //     TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
  //     CHECK_DEVICE(out);
  //     TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
  //     CHECK_SHAPE(out, sizes[0], sizes[1], head_size);
  //     if (seqlenq_ngroups_swapped) {
  //         out = out.reshape({batch_size, num_heads_k, ngroups, head_size}).transpose(1, 2).reshape({batch_size *
  //         ngroups, num_heads_k, head_size});
  //     }
  // } else {
  //     out = torch::empty_like(q);
  // }

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size_rounded = head_size <= 192 ? round_multiple(head_size, 32) : 256;
  const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
  const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

  // auto softmax_lse = torch::empty({num_heads, total_q}, opts.dtype(at::kFloat));
  at::Tensor p;
  // Only return softmax if there's dropout to reduce compilation time
  if (return_softmax) {
    // Return tensor with shape (num_heads_k, total_q, max_seqlen_k)
    p = torch::full({num_heads_k, total_q / 16, seqlen_k_rounded}, 0, opts);
  } else {
    p = torch::empty({0}, opts);
  }

  if (zero_tensors) {
    // out.zero_();
    // softmax_lse.fill_(-std::numeric_limits<float>::infinity());
    if (return_softmax) {
      p.zero_();
    }
  }

  Flash_fwd_params params;
  set_params_fprop(
      params,
      batch_size,
      max_seqlen_q,
      max_seqlen_k,
      seqlen_q_rounded,
      seqlen_k_rounded,
      num_heads,
      num_heads_k,
      head_size,
      head_size_rounded,
      q,
      k,
      v,
      out,
      cu_seqlens_q_d,
      cu_seqlens_k.data_ptr(),
      seqused_k.has_value() ? seqused_k.value().data_ptr() : nullptr,
      return_softmax ? p.data_ptr() : nullptr,
      nullptr,  // softmax_lse.data_ptr(),
      p_dropout,
      softmax_scale,
      window_size_left,
      window_size_right,
      softcap,
      seqlenq_ngroups_swapped,
      /*unpadded_lse*/ true);

  params.cu_seqlens_v = static_cast<int*>(cu_seqlens_v.data_ptr());
  params.is_seqlens_v_cumulative = true;  // Treat cu_seqlens_v as cumulative sequence lengths
  // {
  //     // Copy cu_seqlens_v to CPU for printing
  //     at::Tensor cu_seqlens_v_cpu = cu_seqlens_v.to(torch::kCPU);
  //     const int* cu_seqlens_v_data = cu_seqlens_v_cpu.data_ptr<int>();
  //     printf("params.cu_seqlens_v: ");
  //     for (int i = 0; i < batch_size + 1; ++i) {
  //         printf("%d ", cu_seqlens_v_data[i]);
  //     }
  //     printf("\n");
  // }
  params.total_q = total_q;

  params.m_block_dim = 16;
  params.n_block_dim = 1;

  if (paged_KV) {
    params.block_table = block_table.data_ptr<int>();
    params.block_table_batch_stride = block_table.stride(0);
    params.k_batch_stride = k.stride(0);
    // params.v_batch_stride = v.stride(0);
  }
  params.page_block_size = page_block_size;
  // Keep references to these tensors to extend their lifetime

  if (leftpad_k_.has_value()) {
    auto leftpad_k = leftpad_k_.value();
    TORCH_CHECK(!paged_KV, "We don't support Paged KV and leftpad_k running at the same time yet");
    TORCH_CHECK(leftpad_k.dtype() == torch::kInt32, "leftpad_k must have dtype int32");
    CHECK_DEVICE(leftpad_k);
    CHECK_CONTIGUOUS(leftpad_k);
    CHECK_SHAPE(leftpad_k, batch_size);
    params.leftpad_k = static_cast<int*>(leftpad_k.data_ptr());
  }

  // number of times random will be generated per thread, to offset philox counter in thc random
  // state
  // We use a custom RNG that increases the offset by batch_size * nheads * 32.
  int64_t counter_offset = params.b * params.h * 32;
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));
  // Forward kernel will populate memory with the seed and offset.
  params.rng_state = reinterpret_cast<uint64_t*>(rng_state.data_ptr());

  if (p_dropout > 0.0) {
    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(gen_, at::cuda::detail::getDefaultCUDAGenerator());
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    params.philox_args = gen->philox_cuda_state(counter_offset);
  }

  set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

  if (max_seqlen_k > 0) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    params.num_splits = 1;
    run_mha_fwd_split_stage1(params, stream);
  } else {
    // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
    // out.zero_();
    // softmax_lse.fill_(std::numeric_limits<float>::infinity());
  }

  if (seqlenq_ngroups_swapped) {
    int64_t size_before[] = {batch_size, max_seqlen_q, num_heads_k, head_size};
    int64_t size_after[] = {batch_size, num_heads_k * max_seqlen_q, head_size};
    // out = out.reshape(size_before).transpose(1, 2).reshape(size_after);
    q = q.reshape(size_before).transpose(1, 2).reshape(size_after);
    // softmax_lse = softmax_lse.reshape({num_heads * max_seqlen_q, batch_size});
  }

  return {p};
}
