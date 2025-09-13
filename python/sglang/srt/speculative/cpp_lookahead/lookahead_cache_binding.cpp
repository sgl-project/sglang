#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lookahead.h"

PYBIND11_MODULE(lookahead_cache_cpp, m) {
  using namespace lookahead;
  namespace py = pybind11;
  m.doc() = "";

  py::class_<Lookahead>(m, "Lookahead")
      .def(py::init<size_t, const Param&>(), py::arg("capacity"), py::arg("param"))
      .def("asyncInsert", &Lookahead::asyncInsert, "")
      .def("batchMatch", &Lookahead::batchMatch, "")
      .def("reset", &Lookahead::reset, "")
      .def("synchronize", &Lookahead::synchronize, "");

  py::class_<Param>(m, "Param")
      .def(py::init<>())
      .def_readwrite("enable", &Param::enable)
      .def_readwrite("enable_router_mode", &Param::enable_router_mode)
      .def_readwrite("min_bfs_breadth", &Param::min_bfs_breadth)
      .def_readwrite("max_bfs_breadth", &Param::max_bfs_breadth)
      .def_readwrite("min_match_window_size", &Param::min_match_window_size)
      .def_readwrite("max_match_window_size", &Param::max_match_window_size)
      .def_readwrite("branch_length", &Param::branch_length)
      .def_readwrite("draft_token_num", &Param::draft_token_num)
      .def_readwrite("match_type", &Param::match_type)
      .def_readwrite("batch_min_match_window_size", &Param::batch_min_match_window_size)
      .def_readwrite("batch_draft_token_num", &Param::batch_draft_token_num)
      .def("get_draft_token_num", &Param::get_draft_token_num, "")
      .def("get_min_match_window_size", &Param::get_min_match_window_size, "")
      .def("parse", &Param::parse, "")
      .def("resetBatchMinMatchWindowSize", &Param::resetBatchMinMatchWindowSize, "")
      .def("resetBatchReturnTokenNum", &Param::resetBatchReturnTokenNum, "")
      .def("detail", &Param::detail, "");

  py::class_<Lookahead::Result>(m, "Result")
      .def(py::init<>())
      .def_readwrite("token", &Lookahead::Result::token)
      .def_readwrite("mask", &Lookahead::Result::mask)
      .def("truncate", &Lookahead::Result::truncate);
}
