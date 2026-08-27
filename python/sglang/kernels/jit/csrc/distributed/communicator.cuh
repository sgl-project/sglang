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
  // Shared symbolic sizes / device enforce consistency across ranks; the
  // matchers also require contiguity (no strides given) and uint8 dtype.
  auto push_bytes = SymbolicSize{"push_bytes"};
  auto pull_bytes = SymbolicSize{"pull_bytes"};
  auto num_pull_blocks = SymbolicSize{"num_pull_blocks"};
  auto num_push_blocks = SymbolicSize{"num_push_blocks"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  for (uint32_t i = 0; i < world_size; ++i) {
    TensorMatcher({2 * world_size, push_bytes}).with_dtype<uint8_t>().with_device(device).verify(push_workspaces[i]);
    TensorMatcher({pull_bytes})  //
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(pull_workspaces[i]);
    TensorMatcher({num_pull_blocks, static_cast<int64_t>(sizeof(Semaphore))})
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(pull_semaphores[i]);
    this->push_workspaces[i] = static_cast<uint8_t*>(push_workspaces[i].data_ptr());
    this->pull_workspaces[i] = static_cast<uint8_t*>(pull_workspaces[i].data_ptr());
    this->pull_semaphores[i] = static_cast<Semaphore*>(pull_semaphores[i].data_ptr());
  }
  TensorMatcher({num_push_blocks, static_cast<int64_t>(sizeof(Counter))})
      .with_dtype<uint8_t>()
      .with_device(device)
      .verify(push_counter);
  RuntimeCheck(push_bytes.unwrap() > 0 && pull_bytes.unwrap() > 0, "Workspace sizes must be positive");

  if (pull_mc_workspace_ptr.has_value()) {
    this->pull_mc_workspace = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(pull_mc_workspace_ptr.value()));
  } else {
    this->pull_mc_workspace = nullptr;
  }

  // push config
  this->push_counter = static_cast<Counter*>(push_counter.data_ptr());
  this->push_bytes = push_bytes.unwrap();
  this->num_push_blocks = static_cast<uint32_t>(num_push_blocks.unwrap());

  // pull config
  this->pull_bytes = pull_bytes.unwrap();
  this->num_pull_blocks = static_cast<uint32_t>(num_pull_blocks.unwrap());
  this->num_multicast_blocks = this->num_pull_blocks;
  this->total_pull_blocks = this->num_pull_blocks;
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
