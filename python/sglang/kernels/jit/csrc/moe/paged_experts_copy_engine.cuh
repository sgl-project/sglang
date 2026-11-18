#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/array.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace {
using sglang::host::LaunchKernel;
using sglang::host::RuntimeCheck;
using sglang::host::RuntimeDeviceCheck;

constexpr int kMaxStagingRows = 32;

struct Request {
  uint32_t issue;
  uint32_t completed;
  uint32_t disabled;
  int count;
  int experts[kMaxStagingRows];
};

struct ExpertRows {
  const char* host;
  const uint4* mapped;
  uint4* slots;
  uint4* staging;
  int64_t bytes;
};

struct LayerTransfer {
  Request* request = nullptr;
  Request* mapped_request = nullptr;
  uint32_t* flags = nullptr;  // sequence generator, selected ticket, DMA completion
  ExpertRows* device_rows = nullptr;
  uint64_t* counters = nullptr;
  std::vector<ExpertRows> rows;
  int capacity = 0;

  ~LayerTransfer() {
    if (device_rows) cudaFree(device_rows);
    if (flags) cudaFree(flags);
    if (request) cudaFreeHost(request);
  }
};

__global__ void
publish_copy_plan(volatile Request* request, uint32_t* flags, const int32_t* src, const int32_t* count, int capacity) {
  flags[1] = 0;
  const int n = *count;
  if (!n || n > capacity || request->disabled || request->completed != request->issue) return;

  for (int i = 0; i < n; ++i)
    request->experts[i] = src[i];
  request->count = n;
  // Zero denotes an SM copy. Skip it when the sequence rolls over.
  uint32_t ticket = ++flags[0];
  if (ticket == 0) ticket = ++flags[0];
  flags[1] = ticket;
  __threadfence_system();  // Publish the payload before the host observes its ticket.
  request->issue = ticket;
}

__device__ inline uint64_t copy_clock() {
  uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

__global__ void await_copy(uint32_t* flags, const int32_t* count, uint64_t timeout_ns, unsigned long long* counters) {
  const uint32_t ticket = flags[1];
  if (ticket) {
    const volatile uint32_t* done = flags + 2;
    const auto start = copy_clock();
    while (*done != ticket && copy_clock() - start < timeout_ns)
      __nanosleep(100);
    if (*done != ticket) flags[1] = 0;
  }
  if (counters && *count) atomicAdd(counters + (flags[1] ? 0 : 1), 1ull);
}

__global__ void install_expert_rows(
    const ExpertRows* tensors,
    int num_tensors,
    const int32_t* src,
    const int32_t* dst,
    const int32_t* count,
    const uint32_t* flags) {
  const int n = *count;
  if (!n) return;
  const bool staged = flags[1] != 0;
  for (int p = 0; p < num_tensors; ++p) {
    const ExpertRows t = tensors[p];
    const int64_t width = t.bytes / sizeof(uint4);
    const uint4* from = staged ? t.staging : t.mapped;
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n * width; i += blockDim.x * gridDim.x) {
      const int index = i / width;
      const int64_t offset = i % width;
      t.slots[dst[index] * width + offset] = from[(staged ? index : src[index]) * width + offset];
    }
  }
}

class CopyEngine {
 public:
  explicit CopyEngine(int device, int worker_delay_us) : device(device), delay_us(worker_delay_us) {
    worker = std::thread([this] { run(); });
  }

  ~CopyEngine() {
    stop.store(true, std::memory_order_relaxed);
    worker.join();
  }

  LayerTransfer* add(std::unique_ptr<LayerTransfer> layer) {
    std::lock_guard<std::mutex> hold(mu);
    layer->request->disabled = failed;
    auto* result = layer.get();
    layers.push_back(std::move(layer));
    return result;
  }

  void remove(LayerTransfer* layer) {
    // A layer can still be referenced by work enqueued just before model teardown.
    // Wait outside the worker lock so that pending copies can complete normally.
    int previous = device;
    cudaGetDevice(&previous);
    cudaSetDevice(device);
    cudaDeviceSynchronize();
    {
      std::lock_guard<std::mutex> hold(mu);
      layers.erase(
          std::remove_if(layers.begin(), layers.end(), [=](const auto& entry) { return entry.get() == layer; }),
          layers.end());
    }
    cudaSetDevice(previous);
  }

  const int device;

 private:
  bool copy(LayerTransfer& layer, uint32_t ticket, cudaStream_t stream) {
    // With an outstanding ticket the device cannot rewrite the request or reuse
    // staging. The compute stream serializes each layer's gather with its consumer.
    if (delay_us) std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
    const int n = layer.request->count;
    if (n < 1 || n > layer.capacity) return false;
    for (const auto& tensor : layer.rows) {
      for (int j = 0; j < n; ++j) {
        if (cudaMemcpyAsync(
                reinterpret_cast<char*>(tensor.staging) + j * tensor.bytes,
                tensor.host + layer.request->experts[j] * tensor.bytes,
                tensor.bytes,
                cudaMemcpyHostToDevice,
                stream) != cudaSuccess) {
          return false;
        }
      }
    }
    if (cuStreamWriteValue32(
            reinterpret_cast<CUstream>(stream), reinterpret_cast<CUdeviceptr>(layer.flags + 2), ticket, 0) !=
            CUDA_SUCCESS ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      return false;
    }
    // Publish reuse only after DMA and the device's completion marker are visible.
    __atomic_store_n(&layer.request->completed, ticket, __ATOMIC_RELEASE);
    return true;
  }

  void disable() {
    failed = true;
    for (const auto& layer : layers)
      __atomic_store_n(&layer->request->disabled, 1u, __ATOMIC_RELEASE);
    std::fprintf(stderr, "Paged Experts: copy engine stopped after a CUDA error; using SM gather\n");
  }

  void run() {
    auto mode = cudaStreamCaptureModeRelaxed;
    cudaStream_t stream = nullptr;
    if (cudaSetDevice(device) != cudaSuccess || cudaThreadExchangeStreamCaptureMode(&mode) != cudaSuccess ||
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
      std::lock_guard<std::mutex> hold(mu);
      disable();
      return;
    }
    while (!stop.load(std::memory_order_relaxed)) {
      bool moved = false;
      {
        std::lock_guard<std::mutex> hold(mu);
        for (const auto& layer : layers) {
          auto* req = layer->request;
          const auto ticket = __atomic_load_n(&req->issue, __ATOMIC_ACQUIRE);
          if (ticket == __atomic_load_n(&req->completed, __ATOMIC_RELAXED)) continue;
          if (!copy(*layer, ticket, stream)) {
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
            disable();
            return;
          }
          moved = true;
        }
      }
      if (!moved) {
#if defined(__x86_64__)
        asm volatile("pause");
#else
        std::this_thread::yield();
#endif
      }
    }
    cudaStreamDestroy(stream);
  }

  const int delay_us;
  std::atomic<bool> stop{false};
  bool failed = false;
  std::mutex mu;
  std::vector<std::unique_ptr<LayerTransfer>> layers;
  std::thread worker;
};

int64_t create_copy_engine(int device, int delay_us) {
  return reinterpret_cast<int64_t>(new CopyEngine(device, delay_us));
}

void destroy_copy_engine(int64_t worker) {
  delete reinterpret_cast<CopyEngine*>(worker);
}

int64_t add_copy_engine_layer(
    int64_t worker,
    tvm::ffi::Array<int64_t> hosts,
    tvm::ffi::Array<int64_t> mapped_hosts,
    tvm::ffi::Array<int64_t> slots,
    tvm::ffi::Array<int64_t> staging,
    tvm::ffi::Array<int64_t> sizes,
    int capacity,
    int64_t counters) {
  auto layer = std::make_unique<LayerTransfer>();
  layer->capacity = capacity;
  layer->counters = reinterpret_cast<uint64_t*>(counters);
  RuntimeCheck(capacity >= 1 && capacity <= kMaxStagingRows, "invalid staging capacity");
  const auto n = sizes.size();
  RuntimeCheck(
      n && hosts.size() == n && mapped_hosts.size() == n && slots.size() == n && staging.size() == n,
      "inconsistent expert transfer tensors");
  for (size_t i = 0; i < n; ++i) {
    RuntimeCheck(sizes[i] > 0 && sizes[i] % sizeof(uint4) == 0, "unaligned expert rows");
    layer->rows.push_back(
        {reinterpret_cast<const char*>(hosts[i]),
         reinterpret_cast<const uint4*>(mapped_hosts[i]),
         reinterpret_cast<uint4*>(slots[i]),
         reinterpret_cast<uint4*>(staging[i]),
         sizes[i]});
  }
  RuntimeDeviceCheck(cudaHostAlloc(&layer->request, sizeof(Request), cudaHostAllocMapped));
  *layer->request = {};
  RuntimeDeviceCheck(cudaHostGetDevicePointer(&layer->mapped_request, layer->request, 0));
  RuntimeDeviceCheck(cudaMalloc(&layer->flags, 3 * sizeof(uint32_t)));
  RuntimeDeviceCheck(cudaMemset(layer->flags, 0, 3 * sizeof(uint32_t)));
  RuntimeDeviceCheck(cudaMalloc(&layer->device_rows, n * sizeof(ExpertRows)));
  RuntimeDeviceCheck(
      cudaMemcpy(layer->device_rows, layer->rows.data(), n * sizeof(ExpertRows), cudaMemcpyHostToDevice));
  return reinterpret_cast<int64_t>(reinterpret_cast<CopyEngine*>(worker)->add(std::move(layer)));
}

void remove_copy_engine_layer(int64_t worker, int64_t layer) {
  reinterpret_cast<CopyEngine*>(worker)->remove(reinterpret_cast<LayerTransfer*>(layer));
}

void gather_copy_engine(
    int64_t handle,
    tvm::ffi::TensorView src,
    tvm::ffi::TensorView dst,
    tvm::ffi::TensorView count,
    int64_t timeout_ns) {
  const auto* layer = reinterpret_cast<const LayerTransfer*>(handle);
  const auto* src_ptr = static_cast<const int32_t*>(src.data_ptr());
  const auto* dst_ptr = static_cast<const int32_t*>(dst.data_ptr());
  const auto* count_ptr = static_cast<const int32_t*>(count.data_ptr());
  LaunchKernel(1, 1, src.device())(
      publish_copy_plan, layer->mapped_request, layer->flags, src_ptr, count_ptr, layer->capacity);
  LaunchKernel(1, 1, src.device())(
      await_copy,
      layer->flags,
      count_ptr,
      uint64_t(timeout_ns),
      reinterpret_cast<unsigned long long*>(layer->counters));
  LaunchKernel(128, 256, src.device())(
      install_expert_rows, layer->device_rows, int(layer->rows.size()), src_ptr, dst_ptr, count_ptr, layer->flags);
}
}  // namespace
