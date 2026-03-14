/*
 * CUDA Unified Memory helpers for expert weight offloading.
 *
 * Exposes three operations that are not available in PyTorch's Python API:
 *   uvm_copy_from_tensor  – allocate managed memory and copy a GPU tensor into
 * it uvm_advise            – cudaMemAdvise on a tensor (or a contiguous slice)
 *   uvm_prefetch_async    – cudaMemPrefetchAsync on a tensor
 *
 * Built JIT via torch.utils.cpp_extension.load() on first import of uvm.py.
 */

#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _err = (call);                                                 \
    TORCH_CHECK(_err == cudaSuccess, "CUDA error in " #call ": ",              \
                cudaGetErrorString(_err));                                     \
  } while (0)

// ---------------------------------------------------------------------------
// uvm_copy_from_tensor
//
// Allocates managed memory (cudaMallocManaged) with the same shape, dtype,
// and device as `src`, copies src's data into it via a device-to-device
// cudaMemcpy, and returns a new CUDA tensor backed by the managed allocation.
//
// The returned tensor has a custom deleter (cudaFree) so it is automatically
// freed when its reference count reaches zero.
// ---------------------------------------------------------------------------
torch::Tensor uvm_copy_from_tensor(const torch::Tensor &src) {
  TORCH_CHECK(src.is_cuda(), "uvm_copy_from_tensor: src must be a CUDA tensor");
  TORCH_CHECK(src.is_contiguous(),
              "uvm_copy_from_tensor: src must be contiguous");

  int device_id = static_cast<int>(src.device().index());
  at::cuda::CUDAGuard guard(device_id);

  int64_t nbytes = src.nbytes();
  void *ptr = nullptr;
  CUDA_CHECK(cudaMallocManaged(&ptr, static_cast<size_t>(nbytes),
                               cudaMemAttachGlobal));

  // Device → managed (managed is device-accessible immediately after alloc)
  CUDA_CHECK(cudaMemcpy(ptr, src.data_ptr(), static_cast<size_t>(nbytes),
                        cudaMemcpyDeviceToDevice));

  // Wrap the raw managed pointer in a PyTorch CUDA tensor.
  // The lambda deleter calls cudaFree when the tensor storage is freed.
  auto deleter = [](void *p) { cudaFree(p); };
  c10::DataPtr data_ptr(ptr, ptr, deleter,
                        c10::Device(c10::DeviceType::CUDA, device_id));

  auto storage =
      c10::Storage(c10::Storage::use_byte_size_t(), nbytes, std::move(data_ptr),
                   /*allocator=*/nullptr,
                   /*resizable=*/false);

  // Build a TensorImpl with the same dtype and contiguous strides.
  auto tensor = at::detail::make_tensor<at::TensorImpl>(
      std::move(storage), c10::DispatchKeySet(c10::DispatchKey::CUDA),
      src.dtype());

  at::TensorImpl *impl = tensor.unsafeGetTensorImpl();
  impl->set_sizes_contiguous(src.sizes());

  return tensor;
}

// ---------------------------------------------------------------------------
// uvm_advise
//
// Wrapper around cudaMemAdvise. `advice` should be one of the
// cudaMemoryAdvise enum values (passed as int from Python).
// `device_id` = -1 means cudaCpuDeviceId (CPU).
// ---------------------------------------------------------------------------
void uvm_advise(const torch::Tensor &t, int advice, int device_id) {
  TORCH_CHECK(t.is_contiguous(), "uvm_advise: tensor must be contiguous");
  CUDA_CHECK(cudaMemAdvise(
      t.data_ptr(), static_cast<size_t>(t.nbytes()),
      static_cast<cudaMemoryAdvise>(advice),
      device_id)); // cudaCpuDeviceId == -1 (0xFFFFFFFF) — matches the CUDA API
}

// ---------------------------------------------------------------------------
// uvm_prefetch_async
//
// Wrapper around cudaMemPrefetchAsync.
// `device_id` = -1 migrates pages to CPU (cudaCpuDeviceId).
// `stream_ptr` is the raw cudaStream_t handle cast to int64.
// ---------------------------------------------------------------------------
void uvm_prefetch_async(const torch::Tensor &t, int device_id,
                        int64_t stream_ptr) {
  TORCH_CHECK(t.is_contiguous(),
              "uvm_prefetch_async: tensor must be contiguous");
  cudaStream_t stream =
      reinterpret_cast<cudaStream_t>(static_cast<intptr_t>(stream_ptr));
  CUDA_CHECK(cudaMemPrefetchAsync(t.data_ptr(), static_cast<size_t>(t.nbytes()),
                                  device_id, stream));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("uvm_copy_from_tensor", &uvm_copy_from_tensor,
        "Allocate managed memory and copy a GPU tensor into it");
  m.def("uvm_advise", &uvm_advise,
        "Apply cudaMemAdvise to a tensor (device_id=-1 means CPU)");
  m.def("uvm_prefetch_async", &uvm_prefetch_async,
        "cudaMemPrefetchAsync on a tensor (device_id=-1 means CPU)");
}
