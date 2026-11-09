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
    TensorView counter,                  // [1] int64: one EpochCounter
    intptr_t mc_workspace_ptr,
    uint32_t default_blocks)
    : BasePlane(rank, world_size),  //
      default_blocks(),
      slot_bytes(),
      counter(),
      workspaces{},
      mc_workspace() {
  CHECK_HOST(workspaces.size() == world_size) << "Bad push workspace count";
  // Shared symbolic sizes and device enforce consistency across ranks; the
  // matchers also require contiguity (no strides given) and uint8 dtype.
  auto N = SymbolicSize{"slot_bytes"};
  auto device_sym = SymbolicDevice{};
  device_sym.set_options<kDLGPU>();
  for (uint32_t i = 0; i < world_size; ++i) {
    TensorMatcher({2 * world_size, N})  //
        .with_dtype<uint8_t>()
        .with_device(device_sym)
        .verify(workspaces[i]);
  }
  static_assert(sizeof(EpochCounter) == sizeof(int64_t));
  TensorMatcher({1})  //
      .with_dtype<int64_t>()
      .with_device(device_sym)
      .verify(counter);
  CHECK_HOST(N.unwrap() > 0) << "A push plane needs a non-empty workspace";
  CHECK_HOST(default_blocks > 0) << "A push plane needs a positive default launch width";

  // only set the value safely after the symbolic size has been verified
  this->default_blocks = default_blocks;
  this->slot_bytes = N.unwrap();
  this->counter = static_cast<EpochCounter*>(counter.data_ptr());
  for (uint32_t i = 0; i < world_size; ++i) {
    this->workspaces[i] = static_cast<uint8_t*>(workspaces[i].data_ptr());
  }
  this->mc_workspace = reinterpret_cast<uint8_t*>(mc_workspace_ptr);
}

PullPlaneObj::PullPlaneObj(
    uint32_t rank,
    uint32_t world_size,
    std::vector<TensorView> workspaces,  // world_size * [num_bytes]
    std::vector<TensorView> semaphores,  // world_size * [capacity]
    intptr_t mc_workspace_ptr,
    intptr_t mc_semaphore_ptr,
    uint32_t default_blocks)
    : BasePlane(rank, world_size),  //
      capacity(),
      default_blocks(),
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
  auto M = SymbolicSize{"capacity"};
  auto device_sym = SymbolicDevice{};
  device_sym.set_options<kDLGPU>();
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
  this->capacity = static_cast<uint32_t>(M.unwrap());
  this->default_blocks = std::min(default_blocks, this->capacity);
  this->num_bytes = N.unwrap();
  for (uint32_t i = 0; i < world_size; ++i) {
    this->semaphores[i] = static_cast<Semaphore*>(semaphores[i].data_ptr());
    this->workspaces[i] = static_cast<uint8_t*>(workspaces[i].data_ptr());
  }
  this->mc_semaphore = reinterpret_cast<Semaphore*>(mc_semaphore_ptr);
  this->mc_workspace = reinterpret_cast<uint8_t*>(mc_workspace_ptr);
}

RingPlaneObj::RingPlaneObj(
    uint32_t rank,
    uint32_t world_size,
    std::vector<TensorView> workspaces,  // world_size * [num_rings][num_slots][slot_bytes]
    std::vector<TensorView> semaphores)  // world_size * [num_rings]
    : BasePlane(rank, world_size),       //
      num_rings(),
      num_slots(),
      slot_bytes(),
      semaphores{},
      workspaces{} {
  CHECK_HOST(workspaces.size() == world_size) << "Bad ring workspace count";
  CHECK_HOST(semaphores.size() == world_size) << "Bad ring semaphore count";
  auto R = SymbolicSize{"num_rings"};
  auto S = SymbolicSize{"num_slots"};
  auto N = SymbolicSize{"slot_bytes"};
  auto device_sym = SymbolicDevice{};
  device_sym.set_options<kDLGPU>();
  for (uint32_t i = 0; i < world_size; ++i) {
    TensorMatcher({R, S, N})  //
        .with_dtype<uint8_t>()
        .with_device(device_sym)
        .verify(workspaces[i]);
    TensorMatcher({R, static_cast<int64_t>(sizeof(RingSemaphore))})  //
        .with_dtype<uint8_t>()
        .with_device(device_sym)
        .verify(semaphores[i]);
  }
  const auto rings = R.unwrap(), slots = S.unwrap(), bytes = N.unwrap();
  CHECK_HOST(rings > 0 && bytes > 0 && bytes % 16 == 0) << "Bad ring slot: " << rings << " x " << bytes;
  CHECK_HOST(slots >= 2 && (slots & (slots - 1)) == 0) << "Ring slot count must be a power of two >= 2: " << slots;
  // the kernel forms FIFO offsets in 32 bits
  CHECK_HOST(rings * slots * bytes <= std::numeric_limits<uint32_t>::max())
      << rings * slots * bytes << " bytes of ring FIFO exceeds the 32-bit offset range";

  this->num_rings = static_cast<uint32_t>(rings);
  this->num_slots = static_cast<uint32_t>(slots);
  this->slot_bytes = bytes;
  for (uint32_t i = 0; i < world_size; ++i) {
    this->semaphores[i] = static_cast<RingSemaphore*>(semaphores[i].data_ptr());
    this->workspaces[i] = static_cast<uint8_t*>(workspaces[i].data_ptr());
  }
}

CommunicatorObj::CommunicatorObj(
    Optional<PushPlaneRef> push, Optional<PullPlaneRef> pull, Optional<RingPlaneRef> ring_simple)
    : m_push(std::move(push)), m_pull(std::move(pull)), m_ring_simple(std::move(ring_simple)) {
  CHECK_HOST(m_push.has_value() || m_pull.has_value()) << "A communicator needs a push or a pull plane";
  if (m_push.has_value() && m_pull.has_value()) {
    const auto& push_obj = *m_push.value().get();
    const auto& pull_obj = *m_pull.value().get();
    CHECK_HOST(push_obj.rank == pull_obj.rank && push_obj.world_size == pull_obj.world_size)
        << "Push and pull planes disagree on (rank, world_size)";
  }
  if (m_ring_simple.has_value()) {
    const auto& ring_obj = *m_ring_simple.value().get();
    CHECK_HOST(ring_obj.rank == get_rank() && ring_obj.world_size == get_world_size())
        << "The ring plane disagrees with the other planes on (rank, world_size)";
  }
}

}  // namespace host::distributed

inline void register_communicator() {
  namespace refl = tvm::ffi::reflection;
  namespace dist = host::distributed;
  using TensorView = tvm::ffi::TensorView;
  using Tensors = std::vector<TensorView>;

  refl::ObjectDef<dist::PushPlaneObj>()
      .def(refl::init<uint32_t, uint32_t, Tensors, TensorView, intptr_t, uint32_t>(), "__init__")
      .def_ro("rank", &dist::PushPlaneObj::rank)
      .def_ro("world_size", &dist::PushPlaneObj::world_size)
      .def_ro("default_blocks", &dist::PushPlaneObj::default_blocks)
      .def_ro("slot_bytes", &dist::PushPlaneObj::slot_bytes);

  refl::ObjectDef<dist::PullPlaneObj>()
      .def(refl::init<uint32_t, uint32_t, Tensors, Tensors, intptr_t, intptr_t, uint32_t>(), "__init__")
      .def_ro("rank", &dist::PullPlaneObj::rank)
      .def_ro("world_size", &dist::PullPlaneObj::world_size)
      .def_ro("capacity", &dist::PullPlaneObj::capacity)
      .def_ro("default_blocks", &dist::PullPlaneObj::default_blocks)
      .def_ro("num_bytes", &dist::PullPlaneObj::num_bytes);

  refl::ObjectDef<dist::RingPlaneObj>()
      .def(refl::init<uint32_t, uint32_t, Tensors, Tensors>(), "__init__")
      .def_ro("rank", &dist::RingPlaneObj::rank)
      .def_ro("world_size", &dist::RingPlaneObj::world_size)
      .def_ro("num_rings", &dist::RingPlaneObj::num_rings)
      .def_ro("num_slots", &dist::RingPlaneObj::num_slots)
      .def_ro("slot_bytes", &dist::RingPlaneObj::slot_bytes);

  refl::ObjectDef<dist::CommunicatorObj>()
      .def(
          refl::init<
              dist::Optional<dist::PushPlaneRef>,
              dist::Optional<dist::PullPlaneRef>,
              dist::Optional<dist::RingPlaneRef>>(),
          "__init__")
      .def("get_rank", &dist::CommunicatorObj::get_rank)
      .def("get_world_size", &dist::CommunicatorObj::get_world_size)
      .def("get_push", &dist::CommunicatorObj::get_push)
      .def("get_pull", &dist::CommunicatorObj::get_pull)
      .def("get_ring_simple", &dist::CommunicatorObj::get_ring_simple)
      .def("set_pull_blocks", &dist::CommunicatorObj::set_pull_blocks)
      .def("set_pull_multicast_blocks", &dist::CommunicatorObj::set_pull_multicast_blocks);
}

}  // namespace sglang
