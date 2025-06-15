#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "tree_v2.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace radix_tree_v2;
  namespace py = pybind11;
  py::class_<RadixTree>(m, "RadixTree")
      .def(py::init<bool, bool, int64_t, int64_t, int64_t>())
      .def("match_prefix", &RadixTree::match_prefix)
      .def("evict", &RadixTree::evict)
      .def("lock_ref", &RadixTree::lock_ref)
      .def("evictable_size", &RadixTree::evictable_size)
      .def("protected_size", &RadixTree::protected_size)
      .def("total_size", &RadixTree::total_size)
      .def("writing_through", &RadixTree::writing_through)
      .def("loading_onboard", &RadixTree::loading_onboard)
      .def("commit_writing_through", &RadixTree::commit_writing_through)
      .def("commit_loading_onboard", &RadixTree::commit_loading_onboard)
      .def("reset", &RadixTree::reset)
      .def("debug_print", &RadixTree::debug_print);
}
