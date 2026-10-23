#include <sgl_kernel/tensor.h>

#include <sgl_kernel/distributed/communicator.cuh>

#include <tvm/ffi/extra/stl.h>
#include <tvm/ffi/reflection/registry.h>

namespace sglang {

namespace host::distributed {

BasePlane::BasePlane(uint32_t rank, uint32_t world_size) : rank(rank), world_size(world_size) {
  CHECK_HOST(1 < world_size && world_size <= kMaxWorldSize) << "Invalid world size: " << world_size;
  CHECK_HOST(rank < world_size) << "Invalid rank " << rank << " for world size " << world_size;
}

PushPlaneObj::PushPlaneObj(
    uint32_t rank,
    uint32_t world_size,
    std::vector<TensorView> workspaces,  // world_size * [2 * world_size][slot_bytes]
    TensorView counter,                  // [num_blocks]
    intptr_t mc_workspace_ptr)
    : BasePlane(rank, world_size),  //
      num_blocks(),
      slot_bytes(),
      counter(),
      workspaces{},
      mc_workspace() {
  CHECK_HOST(workspaces.size() == world_size) << "Bad push workspace count";
  // Shared symbolic sizes and device enforce consistency across ranks; the
  // matchers also require contiguity (no strides given) and uint8 dtype.
  auto N = SymbolicSize{"slot_bytes"};
  auto M = SymbolicSize{"num_blocks"};
  auto device_sym = SymbolicDevice{};
  device_sym.set_options<kDLCUDA>();
  for (uint32_t i = 0; i < world_size; ++i) {
    TensorMatcher({2 * world_size, N})  //
        .with_dtype<uint8_t>()
        .with_device(device_sym)
        .verify(workspaces[i]);
  }
  TensorMatcher({M, static_cast<int64_t>(sizeof(Counter))})  //
      .with_dtype<uint8_t>()
      .with_device(device_sym)
      .verify(counter);
  CHECK_HOST(N.unwrap() > 0 && M.unwrap() > 0) << "A push plane needs a non-empty workspace and counter";

  // only set the value safely after the symbolic size has been verified
  this->num_blocks = static_cast<uint32_t>(M.unwrap());
  this->slot_bytes = N.unwrap();
  this->counter = static_cast<Counter*>(counter.data_ptr());
  for (uint32_t i = 0; i < world_size; ++i) {
    this->workspaces[i] = static_cast<uint8_t*>(workspaces[i].data_ptr());
  }
  this->mc_workspace = reinterpret_cast<uint8_t*>(mc_workspace_ptr);
}

PullPlaneObj::PullPlaneObj(
    uint32_t rank,
    uint32_t world_size,
    std::vector<TensorView> workspaces,  // world_size * [num_bytes]
    std::vector<TensorView> semaphores,  // world_size * [num_blocks]
    intptr_t mc_workspace_ptr,
    intptr_t mc_semaphore_ptr)
    : BasePlane(rank, world_size),  //
      num_blocks(),
      num_bytes(),
      semaphores{},
      workspaces{},
      mc_semaphore(),
      mc_workspace() {
  CHECK_HOST(workspaces.size() == world_size) << "Bad pull workspace count";
  CHECK_HOST(semaphores.size() == world_size) << "Bad pull semaphore count";
  // Either half may be empty (a 0-element tensor); the halves a caller does
  // own still have to agree on size and device across ranks.
  auto N = SymbolicSize{"num_bytes"};
  auto M = SymbolicSize{"num_blocks"};
  auto device_sym = SymbolicDevice{};
  device_sym.set_options<kDLCUDA>();
  for (uint32_t i = 0; i < world_size; ++i) {
    TensorMatcher({N})  //
        .with_dtype<uint8_t>()
        .with_device(device_sym)
        .verify(workspaces[i]);
    TensorMatcher({M, static_cast<int64_t>(sizeof(Semaphore))})  //
        .with_dtype<uint8_t>()
        .with_device(device_sym)
        .verify(semaphores[i]);
  }
  CHECK_HOST(N.unwrap() > 0 || M.unwrap() > 0) << "A pull plane with neither workspaces nor semaphores is useless";

  // only set the value safely after the symbolic size has been verified
  this->num_blocks = static_cast<uint32_t>(M.unwrap());
  this->num_bytes = N.unwrap();
  for (uint32_t i = 0; i < world_size; ++i) {
    this->semaphores[i] = static_cast<Semaphore*>(semaphores[i].data_ptr());
    this->workspaces[i] = static_cast<uint8_t*>(workspaces[i].data_ptr());
  }
  this->mc_semaphore = reinterpret_cast<Semaphore*>(mc_semaphore_ptr);
  this->mc_workspace = reinterpret_cast<uint8_t*>(mc_workspace_ptr);
}

CommunicatorObj::CommunicatorObj(Optional<PushPlaneRef> push, Optional<PullPlaneRef> pull)
    : m_push(std::move(push)), m_pull(std::move(pull)) {
  CHECK_HOST(m_push.has_value() || m_pull.has_value()) << "A communicator needs at least one plane";
  if (m_push.has_value() && m_pull.has_value()) {
    const auto& push_obj = *m_push.value().get();
    const auto& pull_obj = *m_pull.value().get();
    CHECK_HOST(push_obj.rank == pull_obj.rank && push_obj.world_size == pull_obj.world_size)
        << "Push and pull planes disagree on (rank, world_size)";
  }
}

}  // namespace host::distributed

inline void register_communicator() {
  namespace refl = tvm::ffi::reflection;
  namespace dist = host::distributed;
  using TensorView = tvm::ffi::TensorView;
  using Tensors = std::vector<TensorView>;

  refl::ObjectDef<dist::PushPlaneObj>()
      .def(refl::init<uint32_t, uint32_t, Tensors, TensorView, intptr_t>(), "__init__")
      .def_ro("rank", &dist::PushPlaneObj::rank)
      .def_ro("world_size", &dist::PushPlaneObj::world_size)
      .def_ro("num_blocks", &dist::PushPlaneObj::num_blocks)
      .def_ro("slot_bytes", &dist::PushPlaneObj::slot_bytes);

  refl::ObjectDef<dist::PullPlaneObj>()
      .def(refl::init<uint32_t, uint32_t, Tensors, Tensors, intptr_t, intptr_t>(), "__init__")
      .def_ro("rank", &dist::PullPlaneObj::rank)
      .def_ro("world_size", &dist::PullPlaneObj::world_size)
      .def_ro("num_blocks", &dist::PullPlaneObj::num_blocks)
      .def_ro("num_bytes", &dist::PullPlaneObj::num_bytes);

  refl::ObjectDef<dist::CommunicatorObj>()
      .def(refl::init<dist::Optional<dist::PushPlaneRef>, dist::Optional<dist::PullPlaneRef>>(), "__init__")
      .def("get_rank", &dist::CommunicatorObj::get_rank)
      .def("get_world_size", &dist::CommunicatorObj::get_world_size)
      .def("get_push", &dist::CommunicatorObj::get_push)
      .def("get_pull", &dist::CommunicatorObj::get_pull)
      .def("set_pull_blocks", &dist::CommunicatorObj::set_pull_blocks)
      .def("set_pull_multicast_blocks", &dist::CommunicatorObj::set_pull_multicast_blocks);
}

}  // namespace sglang
