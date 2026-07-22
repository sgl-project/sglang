#include "p2p_transfer_engine.h"
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime_api.h>
#include <iostream>
#include <stdexcept>

using namespace std::chrono;

enum ErrorCode { OK = 0, InvalidArgument = 1, NotFound = 2, AlreadyExist = 3 };

#define CHECK(call)                                                            \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      throw std::runtime_error(std::string("CUDA Error ") + __FILE__ + ":" +   \
                               std::to_string(__LINE__) + ": " +               \
                               cudaGetErrorString(err));                       \
    }                                                                          \
  } while (0)

// CudaP2PTransfer
CudaP2PTransfer::CudaP2PTransfer(int local_gpu_id)
    : local_gpu_id_(local_gpu_id) {
  cudaSetDevice(local_gpu_id_);
  thread_pool_ = std::make_unique<TransferThreadPool>(local_gpu_id);
}

CudaP2PTransfer::~CudaP2PTransfer() {
  {
    std::shared_lock lock(map_mutex_);
    for (auto &pair : handle_ptr_map) {
      cudaIpcCloseMemHandle(pair.second);
    }
  }
  thread_pool_.reset();
}

std::string CudaP2PTransfer::register_buffer(void *ptr) {
  cudaSetDevice(local_gpu_id_);
  cudaIpcMemHandle_t handle;
  cudaError_t err = cudaIpcGetMemHandle(&handle, ptr);
  if (err != cudaSuccess) {
    throw std::runtime_error("cudaIpcGetMemHandle failed: " +
                             std::string(cudaGetErrorString(err)));
  }
  handle_d_list.push_back(handle);
  return std::string(reinterpret_cast<char *>(&handle), sizeof(handle));
}

int CudaP2PTransfer::register_d_handle(
    const std::string &dst_handle_serialized) {
  cudaSetDevice(local_gpu_id_);
  {
    std::shared_lock lock(map_mutex_);
    if (handle_ptr_map.find(dst_handle_serialized) != handle_ptr_map.end()) {
      return static_cast<int>(ErrorCode::AlreadyExist);
    }
  }

  if (dst_handle_serialized.size() != sizeof(cudaIpcMemHandle_t)) {
    throw std::runtime_error("Invalid cudaIpcMemHandle_t size");
  }

  cudaIpcMemHandle_t handle;
  std::memcpy(&handle, dst_handle_serialized.data(), sizeof(handle));
  void *dst_ptr = nullptr;
  cudaError_t err = cudaIpcOpenMemHandle((void **)&dst_ptr, handle,
                                         cudaIpcMemLazyEnablePeerAccess);
  if (err != cudaSuccess) {
    throw std::runtime_error("cudaIpcOpenMemHandle failed: " +
                             std::string(cudaGetErrorString(err)));
  }

  {
    std::unique_lock lock(map_mutex_);
    handle_ptr_map[dst_handle_serialized] = static_cast<char *>(dst_ptr);
  }

  return 0;
}

CudaP2PTransfer::TransferHandle CudaP2PTransfer::transfer(
    void *src_ptr, int src_gpu_id, const std::string &dst_handle_serialized,
    int dst_gpu_id, size_t offset_bytes, size_t length_bytes) {
  cudaSetDevice(local_gpu_id_);
  char *dst_base_ptr = nullptr;

  {
    std::shared_lock lock(map_mutex_);
    auto it = handle_ptr_map.find(dst_handle_serialized);
    if (it == handle_ptr_map.end()) {
      throw std::runtime_error("dst handle not found");
    }
    dst_base_ptr = it->second;
  }

  void *dst_ptr = dst_base_ptr + offset_bytes;

  TransferHandle handle;
  handle.done = std::make_shared<std::atomic<bool>>(false);

  TransferTask task{src_ptr,    dst_ptr,      src_gpu_id,
                    dst_gpu_id, length_bytes, handle.done};
  thread_pool_->submit(task);

  return handle;
}

// TransferThreadPool
CudaP2PTransfer::TransferThreadPool::TransferThreadPool(int local_gpu_id) {
  this->local_gpu_id_ = local_gpu_id;
  cudaSetDevice(local_gpu_id_);
  const char *env = std::getenv("MC_METRIC");
  if (env != nullptr) {
    metric_running_ = true;
    metric_thread_ =
        std::thread(&CudaP2PTransfer::TransferThreadPool::metric_worker, this);
  }

  const char *env_thread = std::getenv("EngineThreadNum");
  if (env_thread == nullptr) {
    int max_thread_numbers = std::thread::hardware_concurrency();
    this->thread_numbers = max_thread_numbers > 32 ? 32 : max_thread_numbers;
  } else {
    this->thread_numbers = std::stoi(env_thread);
  }

  std::cout << "num_threads: " << this->thread_numbers << std::endl;
  streams_.resize(this->thread_numbers);
  for (int i = 0; i < this->thread_numbers; ++i) {
    cudaStreamCreate(&streams_[i]);
    threads_.emplace_back([this, i]() { this->worker(i); });
  }
}

void CudaP2PTransfer::TransferThreadPool::metric_worker() {
  int time_step = 5;
  const char *env = std::getenv("METRIC_TIME_STEP");
  if (env != nullptr) {
    time_step = std::stoi(env);
    if (time_step <= 0) {
      std::cout << "please set a positive integer for metric time step"
                << std::endl;
      time_step = 5;
    }
  }
  float div_number = float(time_step);
  while (metric_running_) {
    std::this_thread::sleep_for(seconds(time_step));
    size_t bytes = bytes_transferred_.exchange(0, std::memory_order_relaxed);
    double mb = bytes / (1024.0 * 1024.0);
    std::cout << "[MC_METRIC] Transfer in GPU_ID:" << local_gpu_id_
              << " speed in last " << time_step << "s: " << (mb / div_number)
              << " MB/s" << std::endl;
  }
}

CudaP2PTransfer::TransferThreadPool::~TransferThreadPool() {
  cudaSetDevice(local_gpu_id_);
  running_ = false;
  cv_.notify_all();

  for (auto &t : threads_) {
    if (t.joinable())
      t.join();
  }

  for (auto &stream : streams_) {
    cudaStreamDestroy(stream);
  }

  if (metric_running_) {
    metric_running_ = false;
    if (metric_thread_.joinable()) {
      metric_thread_.join();
    }
  }
}

void CudaP2PTransfer::TransferThreadPool::submit(const TransferTask &task) {
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    task_queue_.push(task);
  }
  cv_.notify_one();
}

void CudaP2PTransfer::TransferThreadPool::worker(int thread_id) {
  cudaSetDevice(local_gpu_id_);
  cudaStream_t stream = streams_[thread_id];

  while (running_) {
    TransferTask task;
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      cv_.wait(lock, [&]() { return !task_queue_.empty() || !running_; });
      if (!running_)
        break;
      task = task_queue_.front();
      task_queue_.pop();
    }

    CHECK(cudaMemcpyPeerAsync(task.dst_ptr, task.dst_gpu_id, task.src_ptr,
                              task.src_gpu_id, task.length_bytes, stream));
    CHECK(cudaStreamSynchronize(stream));
    bytes_transferred_.fetch_add(task.length_bytes, std::memory_order_relaxed);

    task.done->store(true, std::memory_order_release);
  }
}
