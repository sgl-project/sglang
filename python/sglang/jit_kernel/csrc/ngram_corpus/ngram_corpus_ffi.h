#pragma once

#include <sgl_kernel/ffi.h>

#include "ngram.h"
#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace {

static std::unordered_map<int64_t, std::unique_ptr<ngram::Ngram>> g_instances;
static std::atomic<int64_t> g_next_id{0};
static std::mutex g_map_mutex;

inline ngram::Ngram& get_instance(int64_t handle) {
  auto it = g_instances.find(handle);
  if (it == g_instances.end()) {
    throw std::runtime_error("Invalid ngram handle: " + std::to_string(handle));
  }
  return *it->second;
}

struct NgramCorpusFfi {
  static int64_t create(
      int64_t capacity,
      int64_t max_trie_depth,
      int64_t min_match_window_size,
      int64_t max_match_window_size,
      int64_t min_bfs_breadth,
      int64_t max_bfs_breadth,
      int64_t draft_token_num,
      int64_t match_type) {
    ngram::Param param;
    param.enable = true;
    param.enable_router_mode = false;
    param.max_trie_depth = static_cast<size_t>(max_trie_depth);
    param.min_match_window_size = static_cast<size_t>(min_match_window_size);
    param.max_match_window_size = static_cast<size_t>(max_match_window_size);
    param.min_bfs_breadth = static_cast<size_t>(min_bfs_breadth);
    param.max_bfs_breadth = static_cast<size_t>(max_bfs_breadth);
    param.draft_token_num = static_cast<size_t>(draft_token_num);
    param.match_type = (match_type == 0) ? "BFS" : "PROB";

    auto id = g_next_id.fetch_add(1);
    auto instance = std::make_unique<ngram::Ngram>(static_cast<size_t>(capacity), param);

    std::lock_guard<std::mutex> lock(g_map_mutex);
    g_instances[id] = std::move(instance);
    return id;
  }

  static void destroy(int64_t handle) {
    std::lock_guard<std::mutex> lock(g_map_mutex);
    g_instances.erase(handle);
  }

  // tokens_flat: 1D int32 CPU tensor (all sequences concatenated)
  // offsets: 1D int64 CPU tensor of length batch_size+1 (CSR format)
  static void async_insert(int64_t handle, const tvm::ffi::TensorView tokens_flat, const tvm::ffi::TensorView offsets) {
    auto* data = static_cast<const int32_t*>(tokens_flat.data_ptr());
    auto* offs = static_cast<const int64_t*>(offsets.data_ptr());
    int64_t batch_size = offsets.size(0) - 1;

    std::vector<std::vector<int32_t>> tokens(batch_size);
    for (int64_t i = 0; i < batch_size; ++i) {
      tokens[i].assign(data + offs[i], data + offs[i + 1]);
    }
    get_instance(handle).asyncInsert(std::move(tokens));
  }

  // tokens_flat: 1D int32 CPU tensor (all query sequences concatenated)
  // offsets: 1D int64 CPU tensor of length batch_size+1 (CSR format)
  // out_tokens: 1D int32 CPU tensor of length batch_size * draft_token_num
  // out_mask: 1D uint8 CPU tensor of length batch_size * draft_token_num^2
  static void batch_match(
      int64_t handle,
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

    auto result = get_instance(handle).batchMatch(tokens);

    auto* out_tok = static_cast<int32_t*>(out_tokens.data_ptr());
    auto* out_msk = static_cast<uint8_t*>(out_mask.data_ptr());
    std::memcpy(out_tok, result.token.data(), result.token.size() * sizeof(int32_t));
    std::memcpy(out_msk, result.mask.data(), result.mask.size() * sizeof(uint8_t));
  }

  static void synchronize(int64_t handle) {
    get_instance(handle).synchronize();
  }

  static void reset(int64_t handle) {
    get_instance(handle).reset();
  }
};

}  // namespace
