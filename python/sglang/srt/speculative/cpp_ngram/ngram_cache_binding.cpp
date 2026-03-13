#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ngram.h"
#include "trie_cache.h"

PYBIND11_MODULE(ngram_cache_cpp, m) {
  using namespace ngram;
  namespace py = pybind11;
  m.doc() = "";

  using NgramTrie = Ngram<TrieCache>;
  py::class_<NgramTrie>(m, "NgramTrie")
      .def(py::init<size_t, const Param&>(), py::arg("capacity"), py::arg("param"))
      .def("asyncInsert", &NgramTrie::asyncInsert, "")
      .def("batchMatch", &NgramTrie::batchMatch, "")
      .def("reset", &NgramTrie::reset, "")
      .def("synchronize", &NgramTrie::synchronize, "");

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

  py::class_<Result>(m, "Result")
      .def(py::init<>())
      .def_readwrite("token", &Result::token)
      .def_readwrite("mask", &Result::mask)
      .def("truncate", &Result::truncate);
}
