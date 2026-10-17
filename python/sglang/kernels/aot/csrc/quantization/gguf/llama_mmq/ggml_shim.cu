// Definitions of the parts of ggml / ggml-cuda that the vendored MMQ code links against, so that it can
// be built without the rest of ggml: ggml_cuda_info() (from cudaDeviceProp), the current device,
// error / abort / log hooks (errors become C++ exceptions, which torch reports as Python errors), and the
// ggml_backend_cuda_context memory pool, which allocates from the allocator installed by set_allocator().
// The vendored ggml/src/ggml-cuda/common.cuh is used unchanged; only its out-of-line members are defined here.
// Device info and error handling follow llama.cpp's ggml-cuda.cu (MIT License, Copyright (c) 2023-2026
// The ggml authors, see ggml/LICENSE).

#include <algorithm>
#include <cstdarg>
#include <stdexcept>

#include "common.cuh"
#include "sgl_llama_mmq.h"

// ---------------------------------------------------------------------------------------------
// errors / logging

[[noreturn]] void ggml_cuda_error(const char* stmt, const char* func, const char* file, int line, const char* msg) {
  int device = -1;
  cudaGetDevice(&device);
  char buf[1024];
  snprintf(
      buf,
      sizeof(buf),
      "llama.cpp MMQ CUDA error: %s\n  current device: %d, in function %s at %s:%d\n  %s",
      msg,
      device,
      func,
      file,
      line,
      stmt);
  throw std::runtime_error(buf);
}

extern "C" void ggml_abort(const char* file, int line, const char* fmt, ...) {
  char msg[768];
  va_list args;
  va_start(args, fmt);
  vsnprintf(msg, sizeof(msg), fmt, args);
  va_end(args);
  char buf[1024];
  snprintf(buf, sizeof(buf), "llama.cpp MMQ error at %s:%d: %s", file, line, msg);
  throw std::runtime_error(buf);
}

extern "C" void ggml_log_internal(enum ggml_log_level level, const char* format, ...) {
  if (level == GGML_LOG_LEVEL_DEBUG) {
    return;
  }
  va_list args;
  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);
}

// ---------------------------------------------------------------------------------------------
// device info

const ggml_cuda_device_info& ggml_cuda_info() {
  static ggml_cuda_device_info info = []() {
    ggml_cuda_device_info i = {};
    int n = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n));
    // Devices beyond GGML_CUDA_MAX_DEVICES are rejected in sgl_llama_mmq::mul_mat().
    n = std::min(n, GGML_CUDA_MAX_DEVICES);
    i.device_count = n;
    i.physical_device_count = n;
    for (int id = 0; id < n; ++id) {
      cudaDeviceProp prop;
      CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
      auto& d = i.devices[id];
      d.cc = 100 * prop.major + 10 * prop.minor;
      d.nsm = prop.multiProcessorCount;
      d.smpb = prop.sharedMemPerBlock;
      d.smpbo = prop.sharedMemPerBlockOptin;
      d.integrated = prop.integrated;
      d.vmm = false;
      d.vmm_granularity = 0;
      d.total_vram = prop.totalGlobalMem;
      d.warp_size = prop.warpSize;
      d.supports_cooperative_launch = prop.cooperativeLaunch;
      d.physical_device = id;
      d.physical_share_count = 1;
      d.virtual_index = 0;
    }
    return i;
  }();
  return info;
}

int ggml_cuda_get_device() {
  int id;
  CUDA_CHECK(cudaGetDevice(&id));
  return id;
}

void ggml_cuda_set_device(int device) {
  int current;
  CUDA_CHECK(cudaGetDevice(&current));
  if (current != device) {
    CUDA_CHECK(cudaSetDevice(device));
  }
}

// ---------------------------------------------------------------------------------------------
// memory pool over the allocator installed by the torch side

namespace sgl_llama_mmq {
static alloc_fn g_alloc = nullptr;
static free_fn g_free = nullptr;
// Stream of the launch in progress on this thread; pool allocations are made on it.
thread_local cudaStream_t g_cur_stream = nullptr;

void set_allocator(alloc_fn alloc, free_fn free) {
  g_alloc = alloc;
  g_free = free;
}
}  // namespace sgl_llama_mmq

struct sgl_torch_pool : public ggml_cuda_pool {
  void* alloc(size_t size, size_t* actual_size) override {
    GGML_ASSERT(sgl_llama_mmq::g_alloc != nullptr);
    *actual_size = size;
    return sgl_llama_mmq::g_alloc(size, sgl_llama_mmq::g_cur_stream);
  }
  void free(void* ptr, size_t /*size*/) override {
    sgl_llama_mmq::g_free(ptr);
  }
};

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int /*device*/, int /*stream_no*/) {
  return std::unique_ptr<ggml_cuda_pool>(new sgl_torch_pool());
}

ggml_backend_cuda_context::~ggml_backend_cuda_context() {
  // No streams, events or cuBLAS handles are ever created on this context (the stream is passed
  // explicitly to mul_mat_q_case), so there is nothing to release.
}
