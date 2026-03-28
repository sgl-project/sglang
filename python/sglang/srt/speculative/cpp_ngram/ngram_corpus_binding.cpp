#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <utility>

#include "ngram.h"

namespace py = pybind11;

namespace {

// Wraps the Python-facing iterable/cast logic together with the start/append/finish load
// transaction and clears any partially built SAM on exceptions.
std::pair<size_t, size_t> loadExternalCorpus(ngram::Ngram& ngram, py::iterable chunks) {
  ngram.startExternalCorpusLoad();
  size_t chunk_count = 0;
  size_t loaded_token_count = 0;
  try {
    for (py::handle chunk_obj : chunks) {
      auto chunk = chunk_obj.cast<std::vector<int32_t>>();
      loaded_token_count += chunk.size();
      ngram.appendExternalCorpusTokens(chunk);
      ++chunk_count;
    }
    ngram.finishExternalCorpusLoad();
  } catch (...) {
    ngram.clearExternalCorpus();
    throw;
  }
  return std::make_pair(chunk_count, loaded_token_count);
}

}  // namespace

PYBIND11_MODULE(ngram_corpus_cpp, m) {
  using namespace ngram;
  m.doc() = "";

  py::class_<Ngram>(m, "Ngram")
      .def(py::init<size_t, const Param&>(), py::arg("capacity"), py::arg("param"))
      .def("asyncInsert", &Ngram::asyncInsert, "")
      .def("loadExternalCorpus", &loadExternalCorpus, "")
      .def("batchMatch", &Ngram::batchMatch, "")
      .def("eraseMatchState", &Ngram::eraseMatchState, "")
      .def("reset", &Ngram::reset, "")
      .def("synchronize", &Ngram::synchronize, "");

  py::class_<Param>(m, "Param")
      .def(py::init<>())
      .def_readwrite("enable", &Param::enable)
      .def_readwrite("enable_router_mode", &Param::enable_router_mode)
      .def_readwrite("min_bfs_breadth", &Param::min_bfs_breadth)
      .def_readwrite("max_bfs_breadth", &Param::max_bfs_breadth)
      .def_readwrite("max_trie_depth", &Param::max_trie_depth)
      .def_readwrite("draft_token_num", &Param::draft_token_num)
      .def_readwrite("external_sam_budget", &Param::external_sam_budget)
      .def_readwrite("external_corpus_max_tokens", &Param::external_corpus_max_tokens)
      .def_readwrite("match_type", &Param::match_type)
      .def_readwrite("batch_draft_token_num", &Param::batch_draft_token_num)
      .def("get_draft_token_num", &Param::get_draft_token_num, "")
      .def("parse", &Param::parse, "")
      .def("resetBatchReturnTokenNum", &Param::resetBatchReturnTokenNum, "")
      .def("detail", &Param::detail, "");

  py::class_<Result>(m, "Result")
      .def(py::init<>())
      .def_readwrite("token", &Result::token)
      .def_readwrite("mask", &Result::mask)
      .def("truncate", &Result::truncate);
}
