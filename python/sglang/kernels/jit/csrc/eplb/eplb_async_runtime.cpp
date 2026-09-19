#include <sgl_kernel/eplb_async_signal_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <nvtx3/nvToolsExt.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

bool async_sync_debug_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("SGLANG_EPLB_ASYNC_SYNC_DEBUG");
    if (value == nullptr) {
      return false;
    }
    return std::string(value) == "1" || std::string(value) == "true" || std::string(value) == "TRUE" ||
           std::string(value) == "on" || std::string(value) == "ON" || std::string(value) == "yes" ||
           std::string(value) == "YES";
  }();
  return enabled;
}

bool eplb_runtime_nvtx_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("SGLANG_EPLB_RUNTIME_NVTX");
    if (value == nullptr) {
      return false;
    }
    return std::string(value) == "1" || std::string(value) == "true" || std::string(value) == "TRUE" ||
           std::string(value) == "on" || std::string(value) == "ON" || std::string(value) == "yes" ||
           std::string(value) == "YES";
  }();
  return enabled;
}

void async_sync_debug_log(const std::string& message) {
  if (!async_sync_debug_enabled()) {
    return;
  }
  std::cerr << "[EPLBAsyncSync] " << message << std::endl;
}

inline void check_cuda_error(cudaError_t error, const char* what) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
  }
}

struct MetadataFieldPair {
  torch::Tensor current;
  torch::Tensor next;
};

struct ResetTensorSpec {
  torch::Tensor tensor;
  int64_t layer_dim = 0;
};

struct LayerPlan {
  int64_t layer_id = -1;
  std::vector<torch::Tensor> routed_experts_weights;
  std::vector<std::vector<torch::Tensor>> host_expert_tensors_per_copy;
  std::vector<int64_t> dst_slots;
};

struct PreparedPlan {
  std::vector<int64_t> update_layer_ids;
  std::vector<LayerPlan> layer_plans;
  std::vector<MetadataFieldPair> gpu_metadata_fields;
  std::vector<MetadataFieldPair> cpu_metadata_fields;
  std::vector<ResetTensorSpec> gpu_reset_tensors;
  std::vector<ResetTensorSpec> cpu_reset_tensors;
};

std::string describe_layer_plan_copy_counts(const PreparedPlan& prepared_plan) {
  std::ostringstream oss;
  oss << "{";
  bool first = true;
  for (const auto& layer_plan : prepared_plan.layer_plans) {
    if (!first) {
      oss << ", ";
    }
    first = false;
    oss << layer_plan.layer_id << ":" << layer_plan.dst_slots.size();
  }
  oss << "}";
  return oss.str();
}

struct IterInfo {
  int64_t step = -1;
  bool enable_statistic = true;
};

struct RegisteredLayer {
  int64_t layer_id = -1;
  torch::Tensor signal_tensor;
  sglang::eplb::EplbAsyncSignal* host_signal = nullptr;
  sglang::eplb::EplbAsyncSignal* device_signal = nullptr;
  torch::Tensor enabled_tensor;
  int64_t iter_id = -1;
  bool statistic_enabled = true;
  bool update_enabled = false;
  bool update_inflight = false;

  // H2D completion tracking via mapped pinned memory.
  // Same mechanism as signal: CPU writes via atomic_ref, GPU reads via
  // mapped device pointer in the spin-wait kernel.  No kernel is launched
  // on copy_stream_, so SM resources are not consumed.
  torch::Tensor h2d_done_flag_tensor;       // CPU tensor, cudaHostRegister'd
  int64_t* h2d_done_flag_host = nullptr;    // CPU-side write pointer
  int64_t* h2d_done_flag_device = nullptr;  // GPU-side read pointer (mapped)
  int64_t h2d_generation = 0;
  int64_t expected_h2d_generation = 0;
};

struct ActivePlan {
  explicit ActivePlan(const PreparedPlan& prepared)
      : plan(prepared), remaining_updates(prepared.update_layer_ids.size()) {
    for (const auto& layer_plan : plan.layer_plans) {
      layer_plans.emplace(layer_plan.layer_id, layer_plan);
    }
  }

  PreparedPlan plan;
  std::unordered_map<int64_t, LayerPlan> layer_plans;
  std::atomic<size_t> remaining_updates;
  int64_t target_step = -1;
};

struct UpdateTask {
  std::shared_ptr<ActivePlan> active_plan;
  int64_t layer_id = -1;
  int64_t target_step = -1;
};

class EPLBAsyncRuntime {
 public:
  explicit EPLBAsyncRuntime(int64_t device_index) : device_index_(static_cast<c10::DeviceIndex>(device_index)) {
    at::cuda::CUDAGuard device_guard(device_index_);
    check_cuda_error(
        cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags failed");
    try {
      check_cuda_error(
          cudaEventCreateWithFlags(&update_done_event_, cudaEventDisableTiming), "cudaEventCreateWithFlags failed");
      worker_thread_ = std::thread(&EPLBAsyncRuntime::worker_loop, this);
      update_thread_ = std::thread(&EPLBAsyncRuntime::update_loop, this);
    } catch (...) {
      shutdown_ = true;
      if (worker_thread_.joinable()) {
        worker_cv_.notify_all();
        worker_thread_.join();
      }
      if (update_done_event_ != nullptr) {
        cudaEventDestroy(update_done_event_);
        update_done_event_ = nullptr;
      }
      cudaStreamDestroy(copy_stream_);
      copy_stream_ = nullptr;
      throw;
    }
  }

  ~EPLBAsyncRuntime() {
    shutdown();
  }

  void register_layer(int64_t layer_id) {
    at::cuda::CUDAGuard device_guard(device_index_);
    std::lock_guard<std::mutex> lock(mu_);
    if (layers_.count(layer_id) != 0) {
      return;
    }

    RegisteredLayer layer;
    layer.layer_id = layer_id;
    layer.signal_tensor = torch::zeros({1}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64));
    check_cuda_error(
        cudaHostRegister(
            layer.signal_tensor.data_ptr(),
            sizeof(sglang::eplb::EplbAsyncSignal),
            cudaHostRegisterPortable | cudaHostRegisterMapped),
        "cudaHostRegister failed");
    layer.host_signal = reinterpret_cast<sglang::eplb::EplbAsyncSignal*>(layer.signal_tensor.data_ptr<int64_t>());
    check_cuda_error(
        cudaHostGetDevicePointer(reinterpret_cast<void**>(&layer.device_signal), layer.signal_tensor.data_ptr(), 0),
        "cudaHostGetDevicePointer failed");
    layer.enabled_tensor = torch::zeros(
        {1}, torch::TensorOptions().device(torch::Device(torch::kCUDA, device_index_)).dtype(torch::kInt32));
    // Allocate H2D completion flag as mapped pinned memory (same pattern
    // as signal_tensor).  CPU writes directly via atomic_ref; GPU reads
    // via mapped device pointer.  No kernel needed on copy_stream_.
    layer.h2d_done_flag_tensor = torch::zeros({1}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64));
    check_cuda_error(
        cudaHostRegister(
            layer.h2d_done_flag_tensor.data_ptr(), sizeof(int64_t), cudaHostRegisterPortable | cudaHostRegisterMapped),
        "cudaHostRegister for h2d_done_flag failed");
    layer.h2d_done_flag_host = layer.h2d_done_flag_tensor.data_ptr<int64_t>();
    check_cuda_error(
        cudaHostGetDevicePointer(
            reinterpret_cast<void**>(&layer.h2d_done_flag_device), layer.h2d_done_flag_tensor.data_ptr(), 0),
        "cudaHostGetDevicePointer for h2d_done_flag failed");
    set_gpu_stage_host(layer, /*step=*/0, /*enable_statistic=*/true);

    layers_.emplace(layer_id, std::move(layer));
    ordered_layer_ids_.push_back(layer_id);
    std::sort(ordered_layer_ids_.begin(), ordered_layer_ids_.end());
  }

  void submit_plan(const PreparedPlan& prepared_plan) {
    std::lock_guard<std::mutex> lock(mu_);
    async_sync_debug_log(
        "plan_queued update_layers=" + std::to_string(prepared_plan.update_layer_ids.size()) +
        " copy_pairs_per_layer=" + describe_layer_plan_copy_counts(prepared_plan) +
        " prepared_queue_before=" + std::to_string(prepared_plan_queue_.size()));
    prepared_plan_queue_.emplace_back(std::make_shared<ActivePlan>(prepared_plan));
    worker_cv_.notify_all();
  }

  void prepare_capture_step(int64_t step) {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& [_, layer] : layers_) {
      set_gpu_stage_host(layer, step, /*enable_statistic=*/false);
    }
  }

  void start_iter(int64_t step, bool enable_statistic) {
    std::lock_guard<std::mutex> lock(mu_);
    iter_queue_.push_back(
        IterInfo{
            .step = step,
            .enable_statistic = enable_statistic,
        });
    worker_cv_.notify_all();
  }

  torch::Tensor wait_gpu_stage(int64_t layer_id) {
    // Read expected_h2d_generation under mu_ to establish a happens-before
    // relationship with the write in maybe_start_update().  This prevents
    // the data race where the forward thread reads a bumped generation
    // that hasn't yet had its H2D enqueued.
    RegisteredLayer* layer = nullptr;
    int64_t expected_gen = 0;
    {
      std::lock_guard<std::mutex> lock(mu_);
      auto it = layers_.find(layer_id);
      if (it == layers_.end()) {
        throw std::runtime_error("Unknown async layer id in wait_gpu_stage.");
      }
      layer = &it->second;
      expected_gen = layer->expected_h2d_generation;
    }
    at::cuda::CUDAGuard device_guard(device_index_);
    auto stream = at::cuda::getCurrentCUDAStream(device_index_);
    sglang::eplb::wait_signal_for_gpu_stage_device(
        layer->device_signal,
        layer->enabled_tensor.data_ptr<int>(),
        layer->h2d_done_flag_device,
        expected_gen,
        stream.stream());

    check_cuda_error(
        cudaStreamWaitEvent(stream.stream(), update_done_event_, 0), "cudaStreamWaitEvent in wait_gpu_stage failed");
    return layer->enabled_tensor;
  }

  void set_cpu_stage(int64_t layer_id) {
    RegisteredLayer* layer = get_layer(layer_id);
    at::cuda::CUDAGuard device_guard(device_index_);
    auto stream = at::cuda::getCurrentCUDAStream(device_index_);
    sglang::eplb::set_signal_for_cpu_stage_device(layer->device_signal, stream.stream());
  }

  void wait_for_idle() {
    std::unique_lock<std::mutex> lock(mu_);
    idle_cv_.wait(lock, [&] {
      return shutdown_ || (pending_update_count_ == 0 && update_queue_.empty() && iter_queue_.empty() &&
                           prepared_plan_queue_.empty() && active_plan_ == nullptr);
    });
  }

  void shutdown() {
    std::unique_lock<std::mutex> lock(mu_);
    if (shutdown_) {
      return;
    }
    shutdown_ = true;
    for (auto& [_, layer] : layers_) {
      disable_signal_host(layer);
    }
    worker_cv_.notify_all();
    update_cv_.notify_all();
    idle_cv_.notify_all();
    lock.unlock();

    if (worker_thread_.joinable()) {
      worker_thread_.join();
    }
    if (update_thread_.joinable()) {
      update_thread_.join();
    }
    if (update_done_event_ != nullptr) {
      cudaEventDestroy(update_done_event_);
      update_done_event_ = nullptr;
    }
    if (copy_stream_ != nullptr) {
      cudaStreamDestroy(copy_stream_);
      copy_stream_ = nullptr;
    }
    cleanup_registered_layers();
  }

 private:
  void worker_loop() {
    while (true) {
      IterInfo iter_info;
      std::shared_ptr<ActivePlan> active_plan;
      std::vector<int64_t> layer_ids_snapshot;
      {
        std::unique_lock<std::mutex> lock(mu_);
        worker_cv_.wait(lock, [&] { return shutdown_ || !iter_queue_.empty(); });
        if (shutdown_) {
          return;
        }
        iter_info = iter_queue_.front();
        iter_queue_.pop_front();
        if (active_plan_ == nullptr && !prepared_plan_queue_.empty()) {
          active_plan_ = prepared_plan_queue_.front();
          prepared_plan_queue_.pop_front();
          async_sync_debug_log(
              "activate_plan step=" + std::to_string(iter_info.step) +
              " update_layers=" + std::to_string(active_plan_->plan.update_layer_ids.size()) +
              " copy_pairs_per_layer=" + describe_layer_plan_copy_counts(active_plan_->plan) +
              " remaining_updates=" + std::to_string(active_plan_->remaining_updates.load()));
        }
        active_plan = active_plan_;
        if (active_plan != nullptr) {
          active_plan->target_step = iter_info.step;
        }
        layer_ids_snapshot = ordered_layer_ids_;
      }

      for (int64_t layer_id : layer_ids_snapshot) {
        // NOTE: wait_last_update_done has been intentionally removed.
        // The GPU-side spin-wait kernel now checks a per-layer
        // h2d_done_flag (set on copy_stream_ after H2D completes) to
        // guarantee weights are fully transferred before the MoE GEMM
        // reads them.  This decouples signal writing from H2D completion
        // and eliminates the cascading worker stall that caused hangs
        // with small batch sizes.
        const bool enable_layer_update = active_plan != nullptr && active_plan->layer_plans.count(layer_id) != 0;
        start_cpu_new_iter(layer_id, iter_info, enable_layer_update);
        wait_cpu_stage(layer_id, iter_info.step);
        maybe_start_update(layer_id, active_plan, iter_info.step);
      }
      finish_iter(active_plan);
    }
  }

  void update_loop() {
    while (true) {
      UpdateTask task;
      {
        std::unique_lock<std::mutex> lock(mu_);
        update_cv_.wait(lock, [&] { return shutdown_ || !update_queue_.empty(); });
        if (shutdown_) {
          return;
        }
        task = std::move(update_queue_.front());
        update_queue_.pop_front();
      }
      try {
        run_update(task);
      } catch (const std::exception& e) {
        async_sync_debug_log(
            "run_update_failed layer_id=" + std::to_string(task.layer_id) +
            " step=" + std::to_string(task.target_step) + " error=" + std::string(e.what()));
      }
      {
        std::lock_guard<std::mutex> lock(mu_);
        layers_.at(task.layer_id).update_inflight = false;
        --pending_update_count_;
        if (task.active_plan != nullptr) {
          const size_t remaining = task.active_plan->remaining_updates.fetch_sub(1) - 1;
          if (active_plan_ == task.active_plan && remaining == 0) {
            active_plan_.reset();
          }
        }
        idle_cv_.notify_all();
        worker_cv_.notify_all();
      }
    }
  }

  void finish_iter(const std::shared_ptr<ActivePlan>& active_plan) {
    std::lock_guard<std::mutex> lock(mu_);
    idle_cv_.notify_all();
    if (active_plan == nullptr) {
      async_sync_debug_log("finish_iter no_active_plan");
      return;
    }
    if (active_plan != active_plan_) {
      async_sync_debug_log("finish_iter active_plan_replaced step=" + std::to_string(active_plan->target_step));
      return;
    }
    if (active_plan->remaining_updates.load() != 0) {
      async_sync_debug_log(
          "finish_iter pending_updates step=" + std::to_string(active_plan->target_step) +
          " remaining_updates=" + std::to_string(active_plan->remaining_updates.load()));
      return;
    }
    async_sync_debug_log("finish_iter clear_active_plan step=" + std::to_string(active_plan->target_step));
    active_plan_.reset();
  }

  RegisteredLayer* get_layer(int64_t layer_id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = layers_.find(layer_id);
    if (it == layers_.end()) {
      throw std::runtime_error("Unknown async layer id.");
    }
    return &it->second;
  }

  void start_cpu_new_iter(int64_t layer_id, const IterInfo& iter_info, bool update_enabled) {
    RegisteredLayer* layer = nullptr;
    {
      std::lock_guard<std::mutex> lock(mu_);
      auto it = layers_.find(layer_id);
      if (it == layers_.end()) {
        throw std::runtime_error("Unknown async layer id in start_cpu_new_iter.");
      }
      layer = &it->second;
      layer->iter_id = iter_info.step;
      layer->statistic_enabled = iter_info.enable_statistic;
      layer->update_enabled = update_enabled;
      // NOTE: expected_h2d_generation is NOT bumped here.  It is bumped
      // in maybe_start_update() after the H2D task has been enqueued.
      // Bumping here (before wait_cpu_stage) caused a deadlock with small
      // batch sizes: the GPU spin-wait would see the bumped generation
      // before the H2D was even issued, while the worker was stuck at
      // wait_cpu_stage waiting for the GPU — a circular dependency.
    }
    set_gpu_stage_host(*layer, iter_info.step, iter_info.enable_statistic);
  }

  void maybe_start_update(int64_t layer_id, const std::shared_ptr<ActivePlan>& active_plan, int64_t target_step) {
    if (active_plan == nullptr) {
      async_sync_debug_log("enqueue_update_skipped layer_id=" + std::to_string(layer_id) + " reason=no_active_plan");
      return;
    }
    std::lock_guard<std::mutex> lock(mu_);
    auto it = layers_.find(layer_id);
    if (it == layers_.end()) {
      throw std::runtime_error("Unknown async layer id in maybe_start_update.");
    }
    if (!it->second.update_enabled) {
      async_sync_debug_log("enqueue_update_skipped layer_id=" + std::to_string(layer_id) + " reason=update_disabled");
      return;
    }
    auto plan_it = active_plan->layer_plans.find(layer_id);
    if (plan_it == active_plan->layer_plans.end()) {
      async_sync_debug_log(
          "enqueue_update_skipped layer_id=" + std::to_string(layer_id) + " reason=not_in_active_plan");
      return;
    }
    ++pending_update_count_;
    it->second.update_inflight = true;
    // Bump expected_h2d_generation HERE — after the GPU has finished using
    // this layer (wait_cpu_stage returned) and just before we enqueue the
    // H2D task.  This guarantees the H2D WILL eventually complete (task is
    // enqueued), so the GPU spin-wait is bounded and cannot deadlock.
    it->second.expected_h2d_generation++;
    async_sync_debug_log(
        "enqueue_update layer_id=" + std::to_string(layer_id) + " step=" + std::to_string(target_step) +
        " copy_pairs=" + std::to_string(plan_it->second.dst_slots.size()));
    update_queue_.push_back(
        UpdateTask{
            .active_plan = active_plan,
            .layer_id = layer_id,
            .target_step = target_step,
        });
    update_cv_.notify_all();
  }

  void set_gpu_stage_host(RegisteredLayer& layer, int64_t step, bool enable_statistic) {
    const uint64_t encoded = sglang::eplb::encode_signal(
        step,
        sglang::eplb::kSignalOwnerGpu,
        /*skip_step=*/!enable_statistic,
        /*disabled=*/false);
    std::atomic_ref<uint64_t> signal(layer.host_signal->step_and_owner);
    signal.store(encoded, std::memory_order_release);
  }

  void disable_signal_host(RegisteredLayer& layer) {
    std::atomic_ref<uint64_t> signal(layer.host_signal->step_and_owner);
    signal.store(sglang::eplb::kSignalDisabled, std::memory_order_release);
  }

  void wait_cpu_stage(int64_t layer_id, int64_t target_step) {
    RegisteredLayer* layer = get_layer(layer_id);
    async_sync_debug_log(
        "wait_cpu_stage_begin layer_id=" + std::to_string(layer_id) + " step=" + std::to_string(target_step));
    int spin_count = 0;
    while (true) {
      if (shutdown_) {
        return;
      }
      std::atomic_ref<uint64_t> signal(layer->host_signal->step_and_owner);
      const uint64_t signal_value = signal.load(std::memory_order_acquire);
      if (!sglang::eplb::is_signal_disabled(signal_value) &&
          sglang::eplb::decode_signal_step(signal_value) == target_step &&
          sglang::eplb::decode_signal_owner(signal_value) == sglang::eplb::kSignalOwnerCpu) {
        async_sync_debug_log(
            "wait_cpu_stage_end layer_id=" + std::to_string(layer_id) + " step=" + std::to_string(target_step));
        return;
      }
      if (sglang::eplb::is_signal_disabled(signal_value)) {
        async_sync_debug_log(
            "wait_cpu_stage_disabled layer_id=" + std::to_string(layer_id) + " step=" + std::to_string(target_step));
        return;
      }
      ++spin_count;
      if (spin_count < 256) {
        std::this_thread::yield();
      } else {
        std::this_thread::sleep_for(std::chrono::microseconds(20));
      }
    }
  }

  void run_update(const UpdateTask& task) {
    if (task.active_plan == nullptr) {
      return;
    }
    auto layer_it = task.active_plan->layer_plans.find(task.layer_id);
    if (layer_it == task.active_plan->layer_plans.end()) {
      return;
    }
    const LayerPlan& layer_plan = layer_it->second;
    const size_t num_experts = layer_plan.dst_slots.size();
    size_t total_h2d_bytes = 0;
    for (size_t copy_idx = 0; copy_idx < num_experts; ++copy_idx) {
      const auto& host_tensors = layer_plan.host_expert_tensors_per_copy[copy_idx];
      for (size_t tensor_idx = 0; tensor_idx < host_tensors.size(); ++tensor_idx) {
        total_h2d_bytes += host_tensors[tensor_idx].numel() * host_tensors[tensor_idx].element_size();
      }
    }
    async_sync_debug_log(
        "run_update_begin layer_id=" + std::to_string(task.layer_id) + " step=" + std::to_string(task.target_step) +
        " num_experts=" + std::to_string(num_experts) + " total_h2d_bytes=" + std::to_string(total_h2d_bytes));
    if (eplb_runtime_nvtx_enabled()) {
      char nvtx_msg[256];
      snprintf(
          nvtx_msg,
          sizeof(nvtx_msg),
          "EPLB_H2D layer=%ld experts=%zu bytes=%zu",
          static_cast<long>(task.layer_id),
          num_experts,
          total_h2d_bytes);
      nvtxRangePushA(nvtx_msg);
    }

    at::cuda::CUDAGuard device_guard(device_index_);
    at::cuda::CUDAStreamGuard stream_guard(at::cuda::getStreamFromExternal(copy_stream_, device_index_));

    for (size_t copy_idx = 0; copy_idx < layer_plan.dst_slots.size(); ++copy_idx) {
      const int64_t dst_slot = layer_plan.dst_slots[copy_idx];
      const auto& host_tensors = layer_plan.host_expert_tensors_per_copy[copy_idx];
      for (size_t tensor_idx = 0; tensor_idx < layer_plan.routed_experts_weights.size(); ++tensor_idx) {
        auto dst_slice = layer_plan.routed_experts_weights[tensor_idx].select(0, dst_slot);
        const auto& src = host_tensors[tensor_idx];
        // Host mirror is built with GPU-matching strides (via as_strided in
        // _create_buffer_tensor), so the physical byte order of src and dst
        // storage segments are identical. A single linear cudaMemcpyAsync is
        // always correct regardless of whether the GPU tensor is contiguous
        // or not (e.g. mxfp4 weight_scale after preprocess_mxfp4_scales).
        TORCH_CHECK(
            static_cast<size_t>(dst_slice.numel()) * dst_slice.element_size() ==
                static_cast<size_t>(src.numel()) * src.element_size(),
            "H2D size mismatch: dst=",
            dst_slice.numel() * dst_slice.element_size(),
            " src=",
            src.numel() * src.element_size());
        check_cuda_error(
            cudaMemcpyAsync(
                dst_slice.data_ptr(),
                src.data_ptr(),
                static_cast<size_t>(dst_slice.numel()) * static_cast<size_t>(dst_slice.element_size()),
                cudaMemcpyHostToDevice,
                copy_stream_),
            "cudaMemcpyAsync H2D failed");
      }
    }

    if (eplb_runtime_nvtx_enabled()) {
      nvtxRangePop();
    }

    for (const auto& field_pair : task.active_plan->plan.gpu_metadata_fields) {
      field_pair.current.select(0, task.layer_id).copy_(field_pair.next.select(0, task.layer_id), true);
    }
    for (const auto& reset_spec : task.active_plan->plan.gpu_reset_tensors) {
      reset_spec.tensor.select(reset_spec.layer_dim, task.layer_id).zero_();
    }

    check_cuda_error(cudaEventRecord(update_done_event_, copy_stream_), "cudaEventRecord failed");
    check_cuda_error(cudaEventSynchronize(update_done_event_), "cudaEventSynchronize failed");

    // Write H2D completion flag via mapped pinned memory (same mechanism
    // as signal).  cudaEventSynchronize above guarantees all copy_stream_
    // H2D work is done.  CPU atomic store is visible to GPU via PCIe BAR.
    // No kernel is launched on copy_stream_.
    {
      std::lock_guard<std::mutex> lock(mu_);
      auto& layer = layers_.at(task.layer_id);
      layer.h2d_generation++;
      std::atomic_ref<int64_t> flag(*layer.h2d_done_flag_host);
      flag.store(layer.h2d_generation, std::memory_order_release);
    }

    for (const auto& field_pair : task.active_plan->plan.cpu_metadata_fields) {
      field_pair.current.select(0, task.layer_id).copy_(field_pair.next.select(0, task.layer_id), false);
    }
    for (const auto& reset_spec : task.active_plan->plan.cpu_reset_tensors) {
      reset_spec.tensor.select(reset_spec.layer_dim, task.layer_id).zero_();
    }
    async_sync_debug_log(
        "run_update_end layer_id=" + std::to_string(task.layer_id) + " step=" + std::to_string(task.target_step) +
        " num_experts=" + std::to_string(num_experts) + " total_h2d_bytes=" + std::to_string(total_h2d_bytes));
  }

  void cleanup_registered_layers() {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& [_, layer] : layers_) {
      if (layer.signal_tensor.defined()) {
        cudaHostUnregister(layer.signal_tensor.data_ptr());
        layer.signal_tensor = torch::Tensor();
      }
      layer.host_signal = nullptr;
      layer.device_signal = nullptr;
      layer.enabled_tensor = torch::Tensor();
      if (layer.h2d_done_flag_tensor.defined()) {
        cudaHostUnregister(layer.h2d_done_flag_tensor.data_ptr());
        layer.h2d_done_flag_tensor = torch::Tensor();
      }
      layer.h2d_done_flag_host = nullptr;
      layer.h2d_done_flag_device = nullptr;
    }
  }

  std::mutex mu_;
  std::condition_variable worker_cv_;
  std::condition_variable update_cv_;
  std::condition_variable idle_cv_;
  std::atomic<bool> shutdown_ = false;
  int64_t pending_update_count_ = 0;

  std::unordered_map<int64_t, RegisteredLayer> layers_;
  std::vector<int64_t> ordered_layer_ids_;
  std::deque<IterInfo> iter_queue_;
  std::deque<UpdateTask> update_queue_;
  std::deque<std::shared_ptr<ActivePlan>> prepared_plan_queue_;
  std::shared_ptr<ActivePlan> active_plan_;

  c10::DeviceIndex device_index_;
  cudaStream_t copy_stream_ = nullptr;
  cudaEvent_t update_done_event_ = nullptr;

  std::thread worker_thread_;
  std::thread update_thread_;
};

}  // namespace

PYBIND11_MODULE(eplb_async_runtime_cpp, m) {
  py::class_<MetadataFieldPair>(m, "MetadataFieldPair")
      .def(py::init<>())
      .def_readwrite("current", &MetadataFieldPair::current)
      .def_readwrite("next", &MetadataFieldPair::next);

  py::class_<ResetTensorSpec>(m, "ResetTensorSpec")
      .def(py::init<>())
      .def_readwrite("tensor", &ResetTensorSpec::tensor)
      .def_readwrite("layer_dim", &ResetTensorSpec::layer_dim);

  py::class_<LayerPlan>(m, "LayerPlan")
      .def(py::init<>())
      .def_readwrite("layer_id", &LayerPlan::layer_id)
      .def_readwrite("routed_experts_weights", &LayerPlan::routed_experts_weights)
      .def_readwrite("host_expert_tensors_per_copy", &LayerPlan::host_expert_tensors_per_copy)
      .def_readwrite("dst_slots", &LayerPlan::dst_slots);

  py::class_<PreparedPlan>(m, "PreparedPlan")
      .def(py::init<>())
      .def_readwrite("update_layer_ids", &PreparedPlan::update_layer_ids)
      .def_readwrite("layer_plans", &PreparedPlan::layer_plans)
      .def_readwrite("gpu_metadata_fields", &PreparedPlan::gpu_metadata_fields)
      .def_readwrite("cpu_metadata_fields", &PreparedPlan::cpu_metadata_fields)
      .def_readwrite("gpu_reset_tensors", &PreparedPlan::gpu_reset_tensors)
      .def_readwrite("cpu_reset_tensors", &PreparedPlan::cpu_reset_tensors);

  py::class_<EPLBAsyncRuntime>(m, "EPLBAsyncRuntime")
      .def(py::init<int64_t>(), py::arg("device_index"))
      .def(
          "register_layer",
          &EPLBAsyncRuntime::register_layer,
          py::arg("layer_id"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "prepare_capture_step",
          &EPLBAsyncRuntime::prepare_capture_step,
          py::arg("step"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "submit_plan",
          &EPLBAsyncRuntime::submit_plan,
          py::arg("prepared_plan"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "start_iter",
          &EPLBAsyncRuntime::start_iter,
          py::arg("step"),
          py::arg("enable_statistic"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "wait_gpu_stage",
          &EPLBAsyncRuntime::wait_gpu_stage,
          py::arg("layer_id"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "set_cpu_stage",
          &EPLBAsyncRuntime::set_cpu_stage,
          py::arg("layer_id"),
          py::call_guard<py::gil_scoped_release>())
      .def("wait_for_idle", &EPLBAsyncRuntime::wait_for_idle, py::call_guard<py::gil_scoped_release>())
      .def("shutdown", &EPLBAsyncRuntime::shutdown, py::call_guard<py::gil_scoped_release>());
}
