#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>
#include <torch/library.h>

#include "mscclpp_allreduce.cuh"

enum MscclContextSelection {
  MSCCL1NODELL = 1,
  MSCCL2NODELL = 2,
};

class MscclContext {
 public:
  MscclContextSelection selection_;
  std::shared_ptr<sglang::Msccl1NodeLLcontext> msccl_1nodeLL_context;
  std::shared_ptr<sglang::Msccl2NodeLLcontext> msccl_2nodeLL_context;
  MscclContext(MscclContextSelection selection) : selection_(selection) {}
  template <typename T>
  void allreduce(
      cudaStream_t stream, T* input, T* output, const size_t input_numel, int threads = 512, int block_limit = 21) {
    if (selection_ == MSCCL1NODELL) {
      msccl_1nodeLL_context->allreduce<T>(stream, input, output, input_numel, threads, block_limit);
    } else if (selection_ == MSCCL2NODELL) {
      msccl_2nodeLL_context->allreduce<T>(stream, input, output, input_numel, threads, block_limit);
    }
  }
};

using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

torch::Tensor _unique_id2tensor(const mscclpp::UniqueId& unique_id) {
  auto options = torch::TensorOptions().dtype(torch::kByte).device(torch::kCPU);
  auto tensor = torch::empty({static_cast<int64_t>(unique_id.size())}, options);
  std::memcpy(tensor.data_ptr<uint8_t>(), unique_id.data(), unique_id.size());
  return tensor;
}

// Function to convert vector of int32_t back to array of uint8_t
mscclpp::UniqueId _tensor2unique_id(const torch::Tensor& tensor) {
  mscclpp::UniqueId unique_id;
  std::memcpy(unique_id.data(), tensor.data_ptr<uint8_t>(), unique_id.size());
  return unique_id;
}

torch::Tensor mscclpp_generate_unique_id() {
  mscclpp::UniqueId unique_id = mscclpp::TcpBootstrap::createUniqueId();
  return _unique_id2tensor(unique_id);
}

fptr_t mscclpp_init_context(
    const torch::Tensor& unique_id,
    const int64_t rank,
    const int64_t world_size,
    torch::Tensor& scratch,
    torch::Tensor& put_buffer,
    const int64_t nranks_per_node,
    const std::vector<int64_t>& rank_to_node,
    const std::vector<int64_t>& rank_to_ib,
    const int64_t context_selection) {
  MscclContext* context_ptr = new MscclContext(static_cast<MscclContextSelection>(context_selection));
  mscclpp::UniqueId uid = _tensor2unique_id(unique_id);
  if (context_selection == MSCCL1NODELL) {
    void* scratch_ptr = reinterpret_cast<void*>(scratch.data_ptr());
    const size_t scratch_bytes = scratch.numel() * scratch.element_size();
    context_ptr->msccl_1nodeLL_context = std::make_shared<sglang::Msccl1NodeLLcontext>(
        uid, rank, world_size, scratch_ptr, scratch_bytes, nranks_per_node, rank_to_node, rank_to_ib);
  } else if (context_selection == MSCCL2NODELL) {
    void* scratch_ptr = reinterpret_cast<void*>(scratch.data_ptr());
    const size_t scratch_bytes = scratch.numel() * scratch.element_size();
    void* put_buffer_ptr = reinterpret_cast<void*>(put_buffer.data_ptr());
    const size_t put_buffer_bytes = put_buffer.numel() * put_buffer.element_size();
    context_ptr->msccl_2nodeLL_context = std::make_shared<sglang::Msccl2NodeLLcontext>(
        uid,
        rank,
        world_size,
        scratch_ptr,
        scratch_bytes,
        put_buffer_ptr,
        put_buffer_bytes,
        nranks_per_node,
        rank_to_node,
        rank_to_ib);
  } else {
    throw std::runtime_error("invalid context selection");
  }
  return (fptr_t)context_ptr;
}

bool _mscclpp_is_weak_contiguous(torch::Tensor& t) {
  return t.is_contiguous() ||
         (t.storage().nbytes() - t.storage_offset() * t.element_size() == t.numel() * t.element_size());
}
void mscclpp_allreduce(fptr_t _context, torch::Tensor& inp, torch::Tensor& out, int64_t nthreads, int64_t nblocks) {
  MscclContext* context = reinterpret_cast<MscclContext*>(_context);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp.numel(), out.numel());
  TORCH_CHECK(_mscclpp_is_weak_contiguous(out));
  TORCH_CHECK(_mscclpp_is_weak_contiguous(inp));
  switch (out.scalar_type()) {
    case at::ScalarType::Float: {
      context->allreduce<float>(
          stream,
          reinterpret_cast<float*>(inp.data_ptr()),
          reinterpret_cast<float*>(out.data_ptr()),
          inp.numel(),
          nthreads,
          nblocks);
      break;
    }
    case at::ScalarType::Half: {
      context->allreduce<half>(
          stream,
          reinterpret_cast<half*>(inp.data_ptr()),
          reinterpret_cast<half*>(out.data_ptr()),
          inp.numel(),
          nthreads,
          nblocks);
      break;
    }
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case at::ScalarType::BFloat16: {
      context->allreduce<__nv_bfloat16>(
          stream,
          reinterpret_cast<__nv_bfloat16*>(inp.data_ptr()),
          reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
          inp.numel(),
          nthreads,
          nblocks);
      break;
    }
#endif
    default:
      throw std::runtime_error("custom allreduce only supports float32, float16 and bfloat16");
  }
}
