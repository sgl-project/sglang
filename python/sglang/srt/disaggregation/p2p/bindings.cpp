#include "p2p_transfer_engine.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(cuda_p2p_transfer, m) {

  // TransferHandle
  py::class_<CudaP2PTransfer::TransferHandle>(m, "TransferHandle")
      .def("wait", &CudaP2PTransfer::TransferHandle::wait,
           "Wait for this transfer task to complete")
      .def("is_done", &CudaP2PTransfer::TransferHandle::is_done,
           "Check if this transfer task is done");

  // CudaP2PTransfer
  py::class_<CudaP2PTransfer>(m, "CudaP2PTransfer")
      .def(py::init<int>())

      .def("register_buffer",
           [](CudaP2PTransfer &self, uintptr_t addr) {
             std::string handle =
                 self.register_buffer(reinterpret_cast<void *>(addr));
             return py::bytes(handle);
           })

      .def("transfer",
           [](CudaP2PTransfer &self, uintptr_t src_ptr, int src_gpu_id,
              py::bytes dst_handle, int dst_gpu_id, size_t offset_bytes,
              size_t length_bytes) {
             std::string handle_str = dst_handle;
             return self.transfer(reinterpret_cast<void *>(src_ptr), src_gpu_id,
                                  handle_str, dst_gpu_id, offset_bytes,
                                  length_bytes);
           })

      .def("register_d_handle",
           [](CudaP2PTransfer &self, py::bytes dst_handle) {
             std::string handle_str = dst_handle;
             return self.register_d_handle(handle_str);
           });
}
