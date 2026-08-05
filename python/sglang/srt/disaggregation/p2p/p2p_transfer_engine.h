#pragma once
#include <atomic>
#include <condition_variable>
#include <cuda_runtime.h>
#include <memory>
#include <queue>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

class CudaP2PTransfer {
public:
  explicit CudaP2PTransfer(int local_gpu_id);
  ~CudaP2PTransfer();

  std::string register_buffer(void *ptr);
  int register_d_handle(const std::string &dst_handle_serialized);

  struct TransferHandle {
    std::shared_ptr<std::atomic<bool>> done;

    void wait() {
      while (!done->load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(
            std::chrono::microseconds(50)); // lightweight busy-wait
      }
    }

    bool is_done() const { return done->load(std::memory_order_acquire); }
  };

  TransferHandle transfer(void *src_ptr, int src_gpu_id,
                          const std::string &dst_handle_serialized,
                          int dst_gpu_id, size_t offset_bytes,
                          size_t length_bytes);

private:
  struct TransferTask {
    void *src_ptr;
    void *dst_ptr;
    int src_gpu_id;
    int dst_gpu_id;
    size_t length_bytes;
    std::shared_ptr<std::atomic<bool>> done;
  };

  class TransferThreadPool {
  public:
    explicit TransferThreadPool(int local_gpu_id);
    ~TransferThreadPool();
    void submit(const TransferTask &task);

  private:
    void worker(int thread_id);

    int local_gpu_id_;
    int thread_numbers;
    std::vector<std::thread> threads_;
    std::vector<cudaStream_t> streams_;
    std::queue<TransferTask> task_queue_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    std::atomic<bool> running_{true};

    std::atomic<size_t> bytes_transferred_{0};
    void metric_worker();
    std::thread metric_thread_;
    std::atomic<bool> metric_running_{false};
  };

  int local_gpu_id_;
  mutable std::shared_mutex map_mutex_;
  std::unordered_map<std::string, char *> handle_ptr_map;
  std::vector<cudaIpcMemHandle_t> handle_d_list;
  std::unique_ptr<TransferThreadPool> thread_pool_;
};
