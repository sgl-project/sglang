#include "ngram.h"

#include <chrono>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>

#include "trie.h"

namespace ngram {

Ngram::Ngram(size_t capacity, const Param& param) : param_(param) {
  if (!(param_.branch_length > 1)) {
    throw std::runtime_error(
        "param_.branch_length must be greater than 1, current value: " + std::to_string(param_.branch_length));
  }
  if (!(param_.min_match_window_size > 0)) {
    throw std::runtime_error(
        "min_match_window_size must be greater than 0, current value: " + std::to_string(param_.min_match_window_size));
  }
  if (!(param_.min_match_window_size <= param_.max_match_window_size)) {
    throw std::runtime_error(
        "min_match_window_size must be less than or equal to "
        "max_match_window_size, current min_match_window_size: " +
        std::to_string(param_.min_match_window_size) +
        ", max_match_window_size: " + std::to_string(param_.max_match_window_size));
  }
  if (!(param_.max_match_window_size < param_.branch_length)) {
    throw std::runtime_error(
        "max_match_window_size must be less than branch_length, current "
        "max_match_window_size: " +
        std::to_string(param_.max_match_window_size) + ", branch_length: " + std::to_string(param_.branch_length));
  }
  if (!(param_.min_bfs_breadth > 0)) {
    throw std::runtime_error(
        "min_bfs_breadth must be greater than 0, current value: " + std::to_string(param_.min_bfs_breadth));
  }
  if (!(param_.min_bfs_breadth <= param_.max_bfs_breadth)) {
    throw std::runtime_error(
        "min_bfs_breadth must be less than or equal to max_bfs_breadth, "
        "current min_bfs_breadth: " +
        std::to_string(param_.min_bfs_breadth) + ", max_bfs_breadth: " + std::to_string(param_.max_bfs_breadth));
  }
  if (!(param_.draft_token_num > 0)) {
    throw std::runtime_error(
        "draft_token_num must be greater than 0, current value: " + std::to_string(param_.draft_token_num));
  }
  for (auto config : param_.batch_draft_token_num) {
    if (config != std::numeric_limits<decltype(config)>::max()) {
      if (!(config <= param_.draft_token_num)) {
        throw std::runtime_error(
            "batch_draft_token_num config value " + std::to_string(config) +
            " must be less than or equal to draft_token_num: " + std::to_string(param_.draft_token_num));
      }
    }
  }
  for (auto config : param_.batch_min_match_window_size) {
    if (config != std::numeric_limits<decltype(config)>::max()) {
      if (!(config >= param_.min_match_window_size)) {
        throw std::runtime_error(
            "batch_min_match_window_size config value " + std::to_string(config) +
            " must be greater than or equal to min_match_window_size: " + std::to_string(param_.min_match_window_size));
      }
      if (!(config <= param_.max_match_window_size)) {
        throw std::runtime_error(
            "batch_min_match_window_size config value " + std::to_string(config) +
            " must be less than or equal to max_match_window_size: " + std::to_string(param_.max_match_window_size));
      }
    }
  }

  trie_ = std::make_unique<Trie>(capacity, param_);

  quit_flag_ = false;
  insert_worker_ = std::thread(&Ngram::insertWorker, this);
}

Ngram::~Ngram() {
  quit_flag_ = true;
  insert_queue_.close();
  if (insert_worker_.joinable()) {
    insert_worker_.join();
  }
}

void Ngram::synchronize() const {
  while (!insert_queue_.empty()) {
    std::this_thread::sleep_for(std::chrono::microseconds(10));
  }
}

void Ngram::asyncInsert(std::vector<std::vector<int32_t>>&& tokens) {
  for (auto&& token : tokens) {
    insert_queue_.enqueue(std::move(token));
  }
}

void Ngram::insertWorker() {
  while (!quit_flag_) {
    std::vector<int32_t> data;
    if (!insert_queue_.dequeue(data)) {
      continue;
    }
    std::unique_lock<std::mutex> lock(mutex_);
    trie_->insert(data.data(), data.size());
  }
}

Result Ngram::batchMatch(const std::vector<std::vector<int32_t>>& tokens) const {
  std::unique_lock<std::mutex> lock(mutex_);

  using BuildFn = Result (Trie::*)(const int32_t*, size_t, int32_t, size_t, const Param&) const;
  BuildFn build_fn;
  if (param_.match_type == "BFS") {
    build_fn = &Trie::buildRecency;
  } else if (param_.match_type == "PROB") {
    build_fn = &Trie::buildFrequency;
  } else {
    throw std::runtime_error("Unknown match_type: '" + param_.match_type + "'. Must be 'BFS' or 'PROB'.");
  }

  Result merged;
  for (const auto& suffix : tokens) {
    auto draft_token_num = param_.get_draft_token_num(tokens.size());
    auto res = (trie_.get()->*build_fn)(suffix.data(), suffix.size(), suffix.back(), draft_token_num, param_);
    merged.token.insert(merged.token.end(), res.token.begin(), res.token.end());
    merged.mask.insert(merged.mask.end(), res.mask.begin(), res.mask.end());
  }
  return merged;
}

}  // namespace ngram
