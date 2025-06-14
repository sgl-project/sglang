#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "tree_v2.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace radix_tree_v2;
  namespace py = pybind11;
  py::class_<RadixTree>(m, "RadixTree")
      .def(py::init<bool, bool, int64_t, int64_t, int64_t>())
      .def("insert", &RadixTree::insert)
      .def("match_prefix", &RadixTree::match_prefix)
      .def("evict", &RadixTree::evict)
      .def("lock_ref", &RadixTree::lock_ref)
      .def("evictable_size", &RadixTree::evictable_size)
      .def("protected_size", &RadixTree::protected_size)
      .def("total_size", &RadixTree::total_size)
      .def("start_write_through", &RadixTree::start_write_through)
      .def("commit_write_through", &RadixTree::commit_write_through)
      .def("load_onboard", &RadixTree::load_onboard)
      .def("commit_load_onboard", &RadixTree::commit_load_onboard)
      .def("reset", &RadixTree::reset)
      .def("debug_print", &RadixTree::debug_print);
}
