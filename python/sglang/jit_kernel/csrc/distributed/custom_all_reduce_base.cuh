#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <sgl_kernel/distributed/custom_all_reduce.cuh>

#include <cstdint>
#include <cstring>

inline void register_custom_all_reduce() {
  namespace refl = tvm::ffi::reflection;
  using Class = host::distributed::CustomAllReduceBase;
  refl::ObjectDef<Class>()
      .def(refl::init<uint32_t, uint32_t, uint32_t, uint32_t, int64_t, int64_t, int64_t>(), "__init__")
      .def("share_storage", &Class::share_storage)
      .def("share_graph_inputs", &Class::share_graph_inputs)
      .def("post_init", &Class::post_init)
      .def("register_inputs", &Class::register_inputs)
      .def("set_cuda_graph_capture", &Class::set_cuda_graph_capture)
      .def("free_ipc_handles", &Class::free_ipc_handles)
      .def("free_storage", &Class::free_storage)
      .def("configure_pull", &Class::configure_pull);
}
