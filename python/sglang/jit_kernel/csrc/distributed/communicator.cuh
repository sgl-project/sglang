#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/distributed/communicator.cuh>

#include <tvm/ffi/extra/stl.h>
#include <tvm/ffi/reflection/registry.h>

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace host::distributed {

inline CommunicatorObj::CommunicatorObj(
    const uint32_t rank,
    const uint32_t world_size,
    std::vector<TensorView> push_workspaces,
    std::vector<TensorView> pull_workspaces,
    std::vector<TensorView> pull_semaphores,
    TensorView push_counter,
    const std::optional<int64_t> pull_mc_workspace_ptr) {
  this->rank = rank;
  this->world_size = world_size;
  RuntimeCheck(1 < world_size && world_size <= kMaxWorldSize, "Invalid world size: ", world_size);
  RuntimeCheck(rank < world_size, "Invalid rank: ", rank);
  RuntimeCheck(push_workspaces.size() == world_size, "Bad push workspace count");
  RuntimeCheck(pull_workspaces.size() == world_size, "Bad pull workspace count");
  RuntimeCheck(pull_semaphores.size() == world_size, "Bad pull semaphore count");
  constexpr auto set_or_check = [](int64_t& var, int64_t val) {
    RuntimeCheck(val > 0, "Workspace sizes must be positive");
    if (var == -1) {
      var = val;
    } else {
      RuntimeCheck(var == val, "Inconsistent workspace sizes: ", var, " vs ", val);
    }
  };

  int64_t pull_bytes = -1;
  int64_t push_bytes = -1;
  int64_t num_pull_blocks = -1;
  for (uint32_t i = 0; i < world_size; ++i) {
    const auto& push_workspace = push_workspaces[i];
    const auto& pull_workspace = pull_workspaces[i];
    const auto& pull_semaphore = pull_semaphores[i];
    RuntimeCheck(
        push_workspace.IsContiguous() && pull_workspace.IsContiguous() && pull_semaphore.IsContiguous(),
        "Workspaces must be contiguous");
    RuntimeCheck(
        push_workspace.ndim() == 2 && push_workspace.size(0) == 2 * world_size,
        "Push workspace must be [2 * world_size, push_bytes]");
    RuntimeCheck(pull_workspace.ndim() == 1, "Pull workspace must be 1-D");
    RuntimeCheck(
        pull_semaphore.ndim() == 2 && pull_semaphore.size(1) == sizeof(Semaphore),
        "Pull semaphores must be [num_pull_blocks, sizeof(Semaphore)]");
    RuntimeCheck(
        is_type<uint8_t>(push_workspace.dtype()) && is_type<uint8_t>(pull_workspace.dtype()) &&
            is_type<uint8_t>(pull_semaphore.dtype()),
        "Workspaces must be uint8");
    RuntimeCheck(
        push_workspace.device().device_type == kDLCUDA && pull_workspace.device().device_type == kDLCUDA &&
            pull_semaphore.device().device_type == kDLCUDA,
        "Workspaces must be CUDA tensors");
    set_or_check(push_bytes, push_workspace.size(1));
    set_or_check(pull_bytes, pull_workspace.size(0));
    set_or_check(num_pull_blocks, pull_semaphore.size(0));
    this->push_workspaces[i] = static_cast<uint8_t*>(push_workspace.data_ptr());
    this->pull_workspaces[i] = static_cast<uint8_t*>(pull_workspace.data_ptr());
    this->pull_semaphores[i] = static_cast<Semaphore*>(pull_semaphore.data_ptr());
  }

  if (pull_mc_workspace_ptr.has_value()) {
    this->pull_mc_workspace = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(pull_mc_workspace_ptr.value()));
  } else {
    this->pull_mc_workspace = nullptr;
  }

  RuntimeCheck(push_counter.IsContiguous(), "Push counter must be contiguous");
  RuntimeCheck(
      push_counter.ndim() == 2 && push_counter.size(1) == sizeof(Counter),
      "Push counter must be [num_push_blocks, sizeof(Counter)]");
  RuntimeCheck(is_type<uint8_t>(push_counter.dtype()), "Push counter must be uint8");
  RuntimeCheck(push_counter.device().device_type == kDLCUDA, "Push counter must be a CUDA tensor");
  const int64_t num_push_blocks = push_counter.size(0);

  // push config
  this->push_counter = static_cast<Counter*>(push_counter.data_ptr());
  this->push_bytes = push_bytes;
  this->num_push_blocks = static_cast<uint32_t>(num_push_blocks);

  // pull config
  this->pull_bytes = pull_bytes;
  this->num_pull_blocks = static_cast<uint32_t>(num_pull_blocks);
  this->num_multicast_blocks = static_cast<uint32_t>(num_pull_blocks);
  this->total_pull_blocks = static_cast<uint32_t>(num_pull_blocks);
}

inline void CommunicatorObj::config(std::map<std::string, uint32_t> config) {
  for (const auto& [key, value] : config) {
    if (key == "num_pull_blocks") {
      RuntimeCheck(value > 0 && value <= total_pull_blocks, "Invalid number of pull blocks: ", value);
      this->num_pull_blocks = value;
    } else if (key == "num_multicast_blocks") {
      RuntimeCheck(value > 0 && value <= total_pull_blocks, "Invalid number of multicast blocks: ", value);
      this->num_multicast_blocks = value;
    } else {
      RuntimeCheck(false, "Unknown config key: ", key);
    }
  }
}

}  // namespace host::distributed

inline void register_communicator() {
  namespace refl = tvm::ffi::reflection;
  using Class = host::distributed::CommunicatorObj;
  using TensorView = tvm::ffi::TensorView;
  refl::ObjectDef<Class>()
      .def(
          refl::init<
              uint32_t,
              uint32_t,
              std::vector<TensorView>,
              std::vector<TensorView>,
              std::vector<TensorView>,
              TensorView,
              std::optional<int64_t>>(),
          "__init__")
      .def_ro("world_size", &Class::world_size)
      .def_ro("rank", &Class::rank)
      .def("_config", &Class::config);
}
