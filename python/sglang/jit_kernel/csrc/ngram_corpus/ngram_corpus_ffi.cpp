#pragma once

#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>

#include <tvm/ffi/reflection/registry.h>

#include "ngram.h"
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

struct NgramCorpusObj : public tvm::ffi::Object {
 public:
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("sgl.NgramCorpus", NgramCorpusObj, tvm::ffi::Object);
  static constexpr bool _type_mutable = true;

  NgramCorpusObj(
      int64_t capacity,
      int64_t max_trie_depth,
      int64_t min_bfs_breadth,
      int64_t max_bfs_breadth,
      int64_t draft_token_num,
      int64_t match_type) {
    ngram::Param param;
    param.enable = true;
    param.enable_router_mode = false;
    param.max_trie_depth = static_cast<size_t>(max_trie_depth);
    param.min_bfs_breadth = static_cast<size_t>(min_bfs_breadth);
    param.max_bfs_breadth = static_cast<size_t>(max_bfs_breadth);
    param.draft_token_num = static_cast<size_t>(draft_token_num);
    param.match_type = (match_type == 0) ? "BFS" : "PROB";
    ngram_ = std::make_unique<ngram::Ngram>(static_cast<size_t>(capacity), param);
  }

  void async_insert(const tvm::ffi::TensorView tokens_flat, const tvm::ffi::TensorView offsets) {
    auto* data = static_cast<const int32_t*>(tokens_flat.data_ptr());
    auto* offs = static_cast<const int64_t*>(offsets.data_ptr());
    int64_t batch_size = offsets.size(0) - 1;

    std::vector<std::vector<int32_t>> tokens(batch_size);
    for (int64_t i = 0; i < batch_size; ++i) {
      tokens[i].assign(data + offs[i], data + offs[i + 1]);
    }
    ngram_->asyncInsert(std::move(tokens));
  }

  void batch_match(
      const tvm::ffi::TensorView tokens_flat,
      const tvm::ffi::TensorView offsets,
      const tvm::ffi::TensorView out_tokens,
      const tvm::ffi::TensorView out_mask) {
    auto* data = static_cast<const int32_t*>(tokens_flat.data_ptr());
    auto* offs = static_cast<const int64_t*>(offsets.data_ptr());
    int64_t batch_size = offsets.size(0) - 1;

    std::vector<std::vector<int32_t>> tokens(batch_size);
    for (int64_t i = 0; i < batch_size; ++i) {
      tokens[i].assign(data + offs[i], data + offs[i + 1]);
    }

    auto result = ngram_->batchMatch(tokens);

    auto* out_tok = static_cast<int32_t*>(out_tokens.data_ptr());
    auto* out_msk = static_cast<uint8_t*>(out_mask.data_ptr());
    if (result.token.size() > static_cast<size_t>(out_tokens.size(0))) {
      throw std::runtime_error(
          "out_tokens buffer too small: " + std::to_string(out_tokens.size(0)) + " < " +
          std::to_string(result.token.size()));
    }
    if (result.mask.size() > static_cast<size_t>(out_mask.size(0))) {
      throw std::runtime_error(
          "out_mask buffer too small: " + std::to_string(out_mask.size(0)) + " < " +
          std::to_string(result.mask.size()));
    }
    std::memcpy(out_tok, result.token.data(), result.token.size() * sizeof(int32_t));
    std::memcpy(out_msk, result.mask.data(), result.mask.size() * sizeof(uint8_t));
  }

  void synchronize() {
    ngram_->synchronize();
  }

  void reset() {
    ngram_->reset();
  }

 private:
  std::unique_ptr<ngram::Ngram> ngram_;
};

void register_ngram_corpus() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<NgramCorpusObj>()
      .def(refl::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>(), "__init__")
      .def("async_insert", &NgramCorpusObj::async_insert)
      .def("batch_match", &NgramCorpusObj::batch_match)
      .def("synchronize", &NgramCorpusObj::synchronize)
      .def("reset", &NgramCorpusObj::reset);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(register_once, register_ngram_corpus);
