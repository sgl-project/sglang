#pragma once

#include <cstddef>
#include <iostream>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ngram {

struct Param {
  bool enable;
  bool enable_router_mode;
  size_t min_bfs_breadth;
  size_t max_bfs_breadth;
  size_t min_match_window_size;
  size_t max_match_window_size;
  size_t branch_length;
  size_t draft_token_num;
  std::string match_type;

  std::vector<size_t> batch_min_match_window_size;
  std::vector<size_t> batch_draft_token_num;

  size_t get_draft_token_num(size_t batch_size) const {
    if (batch_size < batch_draft_token_num.size()) {
      if (batch_draft_token_num[batch_size] !=
          std::numeric_limits<decltype(batch_draft_token_num)::value_type>::max()) {
        return batch_draft_token_num[batch_size];
      }
    }
    return draft_token_num - 1;
  }

  size_t get_min_match_window_size(size_t batch_size) const {
    if (batch_size < batch_min_match_window_size.size()) {
      if (batch_min_match_window_size[batch_size] !=
          std::numeric_limits<decltype(batch_min_match_window_size)::value_type>::max()) {
        return batch_min_match_window_size[batch_size];
      }
    }
    return min_match_window_size;
  }

  std::vector<size_t> parse(const std::string& value) {
    // 0-1|10,2-3|20,
    std::vector<size_t> result;
    if (value.empty()) {
      return result;
    }
    std::vector<size_t> mark;
    std::regex comma_re(",");
    std::sregex_token_iterator first{value.begin(), value.end(), comma_re, -1}, last;
    for (auto p : std::vector<std::string>(first, last)) {
      std::cerr << "seg " << p << std::endl;
    }
    for (const auto& seg : std::vector<std::string>(first, last)) {
      std::regex pipe_re("\\|");
      std::sregex_token_iterator seg_first{seg.begin(), seg.end(), pipe_re, -1}, seg_last;
      std::vector<std::string> part(seg_first, seg_last);
      for (auto p : part) {
        std::cerr << "part " << p << std::endl;
      }
      if (part.size() != 2) {
        throw std::runtime_error(
            "failed to get config, invalid config: " + seg + ", part's size = " + std::to_string(part.size()));
      }
      std::regex endash_re("-");
      std::sregex_token_iterator range_first{part[0].begin(), part[0].end(), endash_re, -1}, range_last;
      std::vector<std::string> range(range_first, range_last);
      if (range.size() != 2) {
        throw std::runtime_error("failed to get range, invalid config: " + value);
      }
      size_t L = std::atoi(range[0].c_str());
      size_t R = std::atoi(range[1].c_str());
      if (L > R || R > 128) {
        throw std::runtime_error("invalid range, config: " + value);
      }
      if (R >= result.size()) {
        result.resize(R + 1, std::numeric_limits<decltype(result)::value_type>::max());
        mark.resize(result.size(), false);
      }
      size_t config = std::atoi(part[1].c_str());
      do {
        if (mark[L]) {
          throw std::runtime_error("repeated position " + std::to_string(L) + ", config : " + value);
        }
        mark[L] = true;
        result[L] = config;
      } while (++L <= R);
    }
    return result;
  }

  void resetBatchMinMatchWindowSize(const std::string& value) {
    batch_min_match_window_size = parse(value);
  }

  void resetBatchReturnTokenNum(const std::string& value) {
    batch_draft_token_num = parse(value);
  }

  std::string detail() {
    std::stringstream ss;
    ss << "enable = " << enable << ", enable_router_mode = " << enable_router_mode
       << ", min_bfs_breadth = " << min_bfs_breadth << ", max_bfs_breadth = " << max_bfs_breadth
       << ", min_match_window_size = " << min_match_window_size << ", max_match_window_size = " << max_match_window_size
       << ", branch_length = " << branch_length << ", draft_token_num = " << draft_token_num
       << ", match_type = " << match_type;
    ss << ", batch_min_match_window_size(" << batch_min_match_window_size.size() << ") = ";
    for (int i = 0; i < batch_min_match_window_size.size(); ++i) {
      ss << i << "|" << batch_min_match_window_size[i] << ",";
    }
    ss << ", batch_draft_token_num(" << batch_draft_token_num.size() << ") = ";
    for (int i = 0; i < batch_draft_token_num.size(); ++i) {
      ss << i << "|" << batch_draft_token_num[i] << ",";
    }
    return ss.str();
  }
};

}  // namespace ngram
