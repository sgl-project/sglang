// Adatped from
// https://github.com/vllm-project/vllm/blob/755ed7b05be4743237d3339c4ff8c22bcaae04f4/csrc/quantization/gguf/gguf_kernel.cu
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#ifndef USE_MUSA
#include <c10/cuda/CUDACachingAllocator.h>

#include <mutex>

#include "llama_mmq/sgl_llama_mmq.h"
#endif

// dont use clang-format here, it breaks the include order
// clang-format off
#include "utils.h"

#include <sgl_kernel/quantization/gguf/ggml-common.h>
#include <sgl_kernel/quantization/gguf/vecdotq.cuh>
#include "dequantize.cuh"
#include "mmvq.cuh"
#include "mmq.cuh"
#include "moe.cuh"
#include <sgl_kernel/quantization/gguf/moe_vec.cuh>
#include <sgl_kernel/quantization/gguf/quantize_q8_1.cuh>
// clang-format off

torch::Tensor ggml_dequantize(
    torch::Tensor W,  // quant weight
    int64_t type,
    int64_t m,
    int64_t n,
    std::optional<at::ScalarType> const& dtype) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(W));
  auto dtype_ = dtype.value_or(torch::kFloat16);
  auto options = torch::TensorOptions().dtype(dtype_).device(W.device());
  at::Tensor DW = torch::empty({m, n}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  DISPATCH_FLOAT_TYPES(DW.scalar_type(), "ggml_dequantize", [&] {
    auto to_cuda = ggml_get_to_cuda<scalar_t>(type);
    to_cuda((void*)W.data_ptr(), (scalar_t*)DW.data_ptr(), m * n, stream);
  });

  return DW;
}

torch::Tensor ggml_mul_mat_vec_a8(
    torch::Tensor W,  // quant weight
    torch::Tensor X,  // input
    int64_t type,
    int64_t row) {
  int col = X.sizes()[1];
  int vecs = X.sizes()[0];
  const int padded = (col + 512 - 1) / 512 * 512;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y = torch::empty({vecs, row}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X = torch::empty({vecs, padded / 32 * 9}, options);
  DISPATCH_FLOAT_TYPES(X.scalar_type(), "ggml_mul_mat_vec_a8", [&] {
    quantize_row_q8_1_cuda<scalar_t>((scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(), col, vecs, stream);
    switch (type) {
      case 2:
        mul_mat_vec_q4_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 3:
        mul_mat_vec_q4_1_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 6:
        mul_mat_vec_q5_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 7:
        mul_mat_vec_q5_1_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 8:
        mul_mat_vec_q8_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 10:
        mul_mat_vec_q2_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 11:
        mul_mat_vec_q3_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 12:
        mul_mat_vec_q4_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 13:
        mul_mat_vec_q5_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 14:
        mul_mat_vec_q6_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 16:
        mul_mat_vec_iq2_xxs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 17:
        mul_mat_vec_iq2_xs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 18:
        mul_mat_vec_iq3_xxs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 19:
        mul_mat_vec_iq1_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 20:
        mul_mat_vec_iq4_nl_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 21:
        mul_mat_vec_iq3_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 22:
        mul_mat_vec_iq2_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 23:
        mul_mat_vec_iq4_xs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 29:
        mul_mat_vec_iq1_m_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
    }
  });
  return Y;
}

#ifdef USE_MUSA
// dp4a MMQ (mmq.cuh): standard and k-quant types only.
torch::Tensor ggml_mul_mat_a8(
    torch::Tensor W,  // quant weight
    torch::Tensor X,  // input
    int64_t type,
    int64_t row) {
  TORCH_CHECK(
      type == 2 || type == 3 || type == 6 || type == 7 || type == 8 || (type >= 10 && type <= 14),
      "ggml_mul_mat_a8: GGML type ",
      type,
      " is not supported by this MMQ kernel");
  int col = X.sizes()[1];
  int padded = (col + 512 - 1) / 512 * 512;
  int batch = X.sizes()[0];
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y = torch::empty({batch, row}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X = torch::empty({batch, padded / 32 * 9}, options);
  DISPATCH_FLOAT_TYPES(X.scalar_type(), "ggml_mul_mat_a8", [&] {
    quantize_row_q8_1_cuda((scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(), col, batch, stream);

    switch (type) {
      case 2:
        ggml_mul_mat_q4_0_q8_1_cuda(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 3:
        ggml_mul_mat_q4_1_q8_1_cuda(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 6:
        ggml_mul_mat_q5_0_q8_1_cuda(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 7:
        ggml_mul_mat_q5_1_q8_1_cuda(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 8:
        ggml_mul_mat_q8_0_q8_1_cuda(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 10:
        ggml_mul_mat_q2_K_q8_1_cuda(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 11:
        ggml_mul_mat_q3_K_q8_1_cuda(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 12:
        ggml_mul_mat_q4_K_q8_1_cuda(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 13:
        ggml_mul_mat_q5_K_q8_1_cuda(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 14:
        ggml_mul_mat_q6_K_q8_1_cuda(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
    }
  });
  return Y;
}
#else
// CUDA: llama.cpp's MMQ (int8 tensor cores on Turing and newer, llama_mmq/), for every GGML type it
// supports (standard, k-quant and i-quant types; IQ1_M through patches/0002). The kernel reads up to
// sgl_llama_mmq::weight_tail_bytes() past the end of the last weight row, as ggml pads its buffers: those
// bytes must be readable and finite. GGUFLinearMethod pads the weights at load time; for weights without
// the padding a zero-padded copy is made on every call.
namespace {
void* llama_mmq_alloc(size_t nbytes, cudaStream_t stream) {
  return c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(nbytes, stream);
}
void llama_mmq_free(void* ptr) {
  c10::cuda::CUDACachingAllocator::raw_delete(ptr);
}
}  // namespace

torch::Tensor ggml_mul_mat_a8(
    torch::Tensor W,  // quant weight [>= row, row_bytes]
    torch::Tensor X,  // input [M, K]
    int64_t type,
    int64_t row) {
  static std::once_flag allocator_installed;
  std::call_once(allocator_installed, [] { sgl_llama_mmq::set_allocator(llama_mmq_alloc, llama_mmq_free); });

  int64_t qk = 0, block_bytes = 0;
  TORCH_CHECK(
      sgl_llama_mmq::type_supported(type) && sgl_llama_mmq::type_info(type, &qk, &block_bytes),
      "ggml_mul_mat_a8: GGML type ",
      type,
      " is not supported by MMQ");
  TORCH_CHECK(W.is_cuda() && W.scalar_type() == torch::kUInt8 && W.dim() == 2, "ggml_mul_mat_a8: W must be 2D uint8");
  TORCH_CHECK(X.is_cuda() && X.dim() == 2, "ggml_mul_mat_a8: X must be a 2D CUDA tensor");
  TORCH_CHECK(
      X.scalar_type() == torch::kFloat || X.scalar_type() == torch::kHalf || X.scalar_type() == torch::kBFloat16,
      "ggml_mul_mat_a8: X must be float32, float16 or bfloat16");
  TORCH_CHECK(W.device() == X.device(), "ggml_mul_mat_a8: W and X must be on the same device");
  TORCH_CHECK(row >= 0 && row <= W.size(0), "ggml_mul_mat_a8: row ", row, " out of range for W rows ", W.size(0));
  const int64_t row_bytes = W.size(1);
  TORCH_CHECK(row_bytes % block_bytes == 0, "ggml_mul_mat_a8: W row bytes ", row_bytes, " not a multiple of ", block_bytes);
  const int64_t K = row_bytes / block_bytes * qk;
  TORCH_CHECK(X.size(1) == K, "ggml_mul_mat_a8: X has ", X.size(1), " columns, W rows hold ", K, " values");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  const int64_t M = X.size(0);
  torch::Tensor Y = torch::empty({M, row}, X.options());
  if (M == 0 || row == 0) {
    return Y;
  }
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  // Rows must be addressable with a block-multiple stride and the storage must extend over the tail padding.
  const int64_t tail_bytes = sgl_llama_mmq::weight_tail_bytes(type, K);
  bool weight_ok = W.stride(1) == 1 && W.stride(0) >= row_bytes && W.stride(0) % block_bytes == 0;
  if (weight_ok) {
    const int64_t needed = (row - 1) * W.stride(0) + row_bytes + tail_bytes;
    const int64_t available = static_cast<int64_t>(W.storage().nbytes()) - W.storage_offset();
    weight_ok = available >= needed;
  }
  torch::Tensor Wk = W;
  if (!weight_ok) {
    TORCH_WARN_ONCE(
        "ggml_mul_mat_a8: the weight storage does not extend ",
        tail_bytes,
        " bytes past the last row (ggml's row padding); a zero-padded copy is made on every call. "
        "Pad the weight storage once at load time.");
    torch::Tensor buf = torch::zeros({row * row_bytes + tail_bytes}, W.options());
    Wk = buf.narrow(0, 0, row * row_bytes).view({row, row_bytes});
    Wk.copy_(W.narrow(0, 0, row));
  }

  const torch::Tensor Xk = X.stride(1) == 1 ? X : X.contiguous();
  const sgl_llama_mmq::XType xtype = Xk.scalar_type() == torch::kFloat ? sgl_llama_mmq::XType::F32
                                     : Xk.scalar_type() == torch::kHalf ? sgl_llama_mmq::XType::F16
                                                                        : sgl_llama_mmq::XType::BF16;
  torch::Tensor quant_X = torch::empty({sgl_llama_mmq::q8_buffer_bytes(M, K)}, X.options().dtype(torch::kUInt8));
  sgl_llama_mmq::quantize_q8_1_mmq(Xk.data_ptr(), xtype, M, K, Xk.stride(0), type, quant_X.data_ptr(), stream);

  // fp32 accumulation, rounded once to X's dtype when stored.
  sgl_llama_mmq::mul_mat(
      Wk.data_ptr(), row, K, Wk.stride(0), type, quant_X.data_ptr(), M, Y.data_ptr(), xtype, row, stream);
  return Y;
}

// MoE with llama.cpp's MMQ (MUL_MAT_ID path, llama_mmq/mmq_moe.cu): out[t, s, :] = W[topk_ids[t, s]] @ x(t, s)
// for every token t and slot s, with x(t, s) = X[t] (gate / up: each token is quantized once) or, with
// x_per_slot, X[t * top_k + s] (down projection). The routes are sorted by expert on the GPU (no host sync).
// Slots with an id < 0 or >= n_experts are skipped and their output rows are zero; the experts of a token must be
// distinct (as top-k gives them), a repeated one is treated the same way. fp32 output.
torch::Tensor ggml_moe_mmq(
    torch::Tensor X,         // [T, K] or [T * top_k, K]
    torch::Tensor W,         // [n_experts, >= row, row_bytes] raw GGUF blocks
    torch::Tensor topk_ids,  // [T, top_k] int32 / int64
    int64_t type,
    int64_t row,
    bool x_per_slot) {
  static std::once_flag allocator_installed;
  std::call_once(allocator_installed, [] { sgl_llama_mmq::set_allocator(llama_mmq_alloc, llama_mmq_free); });

  int64_t qk = 0, block_bytes = 0;
  TORCH_CHECK(
      sgl_llama_mmq::type_supported(type) && sgl_llama_mmq::type_info(type, &qk, &block_bytes),
      "ggml_moe_mmq: GGML type ",
      type,
      " is not supported by MMQ");
  TORCH_CHECK(W.is_cuda() && W.scalar_type() == torch::kUInt8 && W.dim() == 3, "ggml_moe_mmq: W must be 3D uint8");
  TORCH_CHECK(X.is_cuda() && X.dim() == 2, "ggml_moe_mmq: X must be a 2D CUDA tensor");
  TORCH_CHECK(
      X.scalar_type() == torch::kFloat || X.scalar_type() == torch::kHalf || X.scalar_type() == torch::kBFloat16,
      "ggml_moe_mmq: X must be float32, float16 or bfloat16");
  TORCH_CHECK(
      topk_ids.is_cuda() && topk_ids.dim() == 2 &&
          (topk_ids.scalar_type() == torch::kInt32 || topk_ids.scalar_type() == torch::kInt64),
      "ggml_moe_mmq: topk_ids must be a 2D int32 / int64 CUDA tensor");
  TORCH_CHECK(W.device() == X.device() && topk_ids.device() == X.device(), "ggml_moe_mmq: tensors on different devices");
  const int64_t n_experts = W.size(0);
  const int64_t row_bytes = W.size(2);
  TORCH_CHECK(row >= 0 && row <= W.size(1), "ggml_moe_mmq: row ", row, " out of range for W rows ", W.size(1));
  TORCH_CHECK(row_bytes % block_bytes == 0, "ggml_moe_mmq: W row bytes ", row_bytes, " not a multiple of ", block_bytes);
  const int64_t K = row_bytes / block_bytes * qk;
  TORCH_CHECK(X.size(1) == K, "ggml_moe_mmq: X has ", X.size(1), " columns, W rows hold ", K, " values");
  const int64_t T = topk_ids.size(0);
  const int64_t top_k = topk_ids.size(1);
  TORCH_CHECK(
      X.size(0) == (x_per_slot ? T * top_k : T),
      "ggml_moe_mmq: X has ",
      X.size(0),
      " rows, expected ",
      x_per_slot ? T * top_k : T);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  torch::Tensor out = torch::empty({T, top_k, row}, X.options().dtype(torch::kFloat));
  if (T == 0 || top_k == 0 || row == 0) {
    return out;
  }
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  // The kernel reads up to weight_tail_bytes() past the end of each row; for every expert but the last one that
  // lands in the next expert's rows, so only the storage after the last row of the last expert needs it.
  const int64_t tail_bytes = sgl_llama_mmq::weight_tail_bytes(type, K);
  bool weight_ok = W.stride(2) == 1 && W.stride(1) >= row_bytes && W.stride(1) % block_bytes == 0 &&
                   W.stride(0) >= (row - 1) * W.stride(1) + row_bytes && W.stride(0) % block_bytes == 0;
  if (weight_ok) {
    const int64_t needed = (n_experts - 1) * W.stride(0) + (row - 1) * W.stride(1) + row_bytes + tail_bytes;
    const int64_t available = static_cast<int64_t>(W.storage().nbytes()) - W.storage_offset();
    weight_ok = available >= needed;
  }
  torch::Tensor Wk = W;
  if (!weight_ok) {
    TORCH_WARN_ONCE(
        "ggml_moe_mmq: the weight storage does not extend ",
        tail_bytes,
        " bytes past the last row (ggml's row padding); a zero-padded copy is made on every call. "
        "Pad the weight storage once at load time.");
    const int64_t n = n_experts * row * row_bytes;
    torch::Tensor buf = torch::empty({n + tail_bytes}, W.options());
    buf.narrow(0, n, tail_bytes).zero_();
    Wk = buf.narrow(0, 0, n).view({n_experts, row, row_bytes});
    Wk.copy_(W.narrow(1, 0, row));
  }

  const torch::Tensor Xk = X.stride(1) == 1 ? X : X.contiguous();
  const torch::Tensor ids = topk_ids.is_contiguous() ? topk_ids : topk_ids.contiguous();
  const sgl_llama_mmq::XType xtype = Xk.scalar_type() == torch::kFloat ? sgl_llama_mmq::XType::F32
                                     : Xk.scalar_type() == torch::kHalf ? sgl_llama_mmq::XType::F16
                                                                        : sgl_llama_mmq::XType::BF16;
  const auto i32 = X.options().dtype(torch::kInt32);
  torch::Tensor ids32 = torch::empty({T * top_k}, i32);
  sgl_llama_mmq::moe_prepare_ids(
      ids.data_ptr(),
      ids.scalar_type() == torch::kInt64,
      T * top_k,
      top_k,
      n_experts,
      ids32.data_ptr<int32_t>(),
      out.data_ptr<float>(),
      row,
      stream);

  // Token chunks of at most moe_max_tokens_per_call() (mm_ids_helper's shared memory); scratch reused per chunk.
  const int64_t chunk = std::min<int64_t>(T, sgl_llama_mmq::moe_max_tokens_per_call(top_k, row, K));
  torch::Tensor ids_src1 = torch::empty({chunk * top_k}, i32);
  torch::Tensor ids_dst = torch::empty({chunk * top_k + 128}, i32);
  torch::Tensor expert_bounds = torch::empty({n_experts + 1}, i32);
  torch::Tensor quant_X =
      torch::empty({sgl_llama_mmq::q8_buffer_bytes(chunk * top_k, K)}, X.options().dtype(torch::kUInt8));
  const int64_t x_rows_per_token = x_per_slot ? top_k : 1;
  for (int64_t t0 = 0; t0 < T; t0 += chunk) {
    const int64_t n = std::min<int64_t>(chunk, T - t0);
    sgl_llama_mmq::moe_mul_mat(
        Wk.data_ptr(),
        n_experts,
        row,
        K,
        Wk.stride(1),
        Wk.stride(0),
        type,
        static_cast<const char*>(Xk.data_ptr()) + t0 * x_rows_per_token * Xk.stride(0) * Xk.element_size(),
        xtype,
        Xk.stride(0),
        x_per_slot,
        ids32.data_ptr<int32_t>() + t0 * top_k,
        n,
        top_k,
        ids_src1.data_ptr<int32_t>(),
        ids_dst.data_ptr<int32_t>(),
        expert_bounds.data_ptr<int32_t>(),
        quant_X.data_ptr(),
        out.data_ptr<float>() + t0 * top_k * row,
        stream);
  }
  return out;
}
#endif  // USE_MUSA

torch::Tensor ggml_moe_a8(
    torch::Tensor X,  // input
    torch::Tensor W,  // expert weights
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_padded,
    int64_t type,
    int64_t row,
    int64_t top_k,
    int64_t tokens) {
  int col = X.sizes()[1];
  int padded = (col + 512 - 1) / 512 * 512;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y = torch::empty({tokens * top_k, row}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X = torch::empty({tokens, padded / 32 * 9}, options);
  DISPATCH_FLOAT_TYPES(X.scalar_type(), "ggml_moe_a8", [&] {
    quantize_row_q8_1_cuda((scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(), col, tokens, stream);
    switch (type) {
      case 2:
        ggml_moe_q4_0_q8_1_cuda(
            (void*)quant_X.data_ptr(),
            (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(),
            W.stride(0),
            W.size(0),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 3:
        ggml_moe_q4_1_q8_1_cuda(
            (void*)quant_X.data_ptr(),
            (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(),
            W.stride(0),
            W.size(0),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 6:
        ggml_moe_q5_0_q8_1_cuda(
            (void*)quant_X.data_ptr(),
            (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(),
            W.stride(0),
            W.size(0),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 7:
        ggml_moe_q5_1_q8_1_cuda(
            (void*)quant_X.data_ptr(),
            (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(),
            W.stride(0),
            W.size(0),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 8:
        ggml_moe_q8_0_q8_1_cuda(
            (void*)quant_X.data_ptr(),
            (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(),
            W.stride(0),
            W.size(0),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 10:
        ggml_moe_q2_K_q8_1_cuda(
            (void*)quant_X.data_ptr(),
            (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(),
            W.stride(0),
            W.size(0),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 11:
        ggml_moe_q3_K_q8_1_cuda(
            (void*)quant_X.data_ptr(),
            (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(),
            W.stride(0),
            W.size(0),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 12:
        ggml_moe_q4_K_q8_1_cuda(
            (void*)quant_X.data_ptr(),
            (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(),
            W.stride(0),
            W.size(0),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 13:
        ggml_moe_q5_K_q8_1_cuda(
            (void*)quant_X.data_ptr(),
            (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(),
            W.stride(0),
            W.size(0),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 14:
        ggml_moe_q6_K_q8_1_cuda(
            (void*)quant_X.data_ptr(),
            (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(),
            W.stride(0),
            W.size(0),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
    }
  });
  return Y;
}

torch::Tensor ggml_moe_a8_vec(
    torch::Tensor X,  // input
    torch::Tensor W,  // expert weights
    torch::Tensor topk_ids,
    int64_t top_k,
    int64_t type,
    int64_t row,
    int64_t tokens) {
  int col = X.sizes()[1];
  const int padded = (col + 512 - 1) / 512 * 512;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y = torch::zeros({tokens * top_k, row}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X = torch::empty({tokens, padded / 32 * 9}, options);
  DISPATCH_FLOAT_TYPES(X.scalar_type(), "ggml_moe_vec_a8", [&] {
    quantize_row_q8_1_cuda<scalar_t>((scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(), col, tokens, stream);
    switch (type) {
      case 2:
        moe_vec_q4_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 3:
        moe_vec_q4_1_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 6:
        moe_vec_q5_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 7:
        moe_vec_q5_1_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 8:
        moe_vec_q8_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 10:
        moe_vec_q2_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 11:
        moe_vec_q3_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 12:
        moe_vec_q4_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 13:
        moe_vec_q5_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 14:
        moe_vec_q6_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 16:
        moe_vec_iq2_xxs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 17:
        moe_vec_iq2_xs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 18:
        moe_vec_iq3_xxs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 19:
        moe_vec_iq1_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 20:
        moe_vec_iq4_nl_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 21:
        moe_vec_iq3_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 22:
        moe_vec_iq2_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 23:
        moe_vec_iq4_xs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 29:
        moe_vec_iq1_m_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
    }
  });
  return Y;
}

int64_t ggml_moe_get_block_size(int64_t type) {
  switch (type) {
    case 2:
      return MOE_X_Q4_0;
    case 3:
      return MOE_X_Q4_1;
    case 6:
      return MOE_X_Q5_0;
    case 7:
      return MOE_X_Q5_1;
    case 8:
      return MOE_X_Q8_0;
    case 10:
      return MOE_X_Q2_K;
    case 11:
      return MOE_X_Q3_K;
    case 12:
      return MOE_X_Q4_K;
    case 13:
      return MOE_X_Q5_K;
    case 14:
      return MOE_X_Q6_K;
  }
  return 0;
}
