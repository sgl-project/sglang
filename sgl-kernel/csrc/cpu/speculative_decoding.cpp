#include "common.h"

namespace {

// Helper to compute exclusive scan of extend_lens to determine base offsets.
// We implement a simple manual scan since we cannot use <algorithm>.
at::Tensor exclusive_scan_extend_lens(const at::Tensor& extend_lens) {
  auto output = at::empty_like(extend_lens);
  auto* in_ptr = extend_lens.data_ptr<int32_t>();
  auto* out_ptr = output.data_ptr<int32_t>();
  int64_t size = extend_lens.numel();
  int64_t sum = 0;
  for (int64_t i = 0; i < size; ++i) {
    out_ptr[i] = static_cast<int32_t>(sum);
    sum += in_ptr[i];
  }
  return output;
}

void assign_draft_cache_locs_cpu_kernel(
    const at::Tensor& req_pool_indices,
    const at::Tensor& req_to_token,
    const at::Tensor& seq_lens,
    const at::Tensor& extend_lens,
    const at::Tensor& num_new_pages_per_topk,
    at::Tensor& out_cache_loc,
    at::Tensor& source_cache_loc,
    at::Tensor& target_cache_loc,
    const c10::optional<at::Tensor>& last_page_lens_cumsum_opt,
    int64_t duplicate_cache_len,
    int64_t pool_len,
    int64_t topk,
    int64_t speculative_num_steps,
    int64_t page_size,
    int64_t bs_upper,
    int64_t iter_upper) {
  // Pre-calculate exclusive scan for extend_lens to find offsets in Part 1
  auto extend_lens_prefix = exclusive_scan_extend_lens(extend_lens);

  int64_t num_seqs = req_pool_indices.numel();

  // Main parallel loop over sequences in the batch
  at::parallel_for(0, num_seqs, 0, [&](int64_t begin, int64_t end) {
    for (int64_t pid = begin; pid < end; ++pid) {
      int32_t req_idx = req_pool_indices.data_ptr<int32_t>()[pid];

      // Bounds check for req_idx
      if (req_idx >= req_to_token.size(0)) continue;

      // Base pointer for the current sequence's req_to_token table
      int32_t* req_to_token_base = req_to_token.data_ptr<int32_t>() + req_idx * pool_len;

      int32_t seq_len = seq_lens.data_ptr<int32_t>()[pid];
      int32_t extend_len = extend_lens.data_ptr<int32_t>()[pid];
      int32_t num_new_pages = num_new_pages_per_topk.data_ptr<int32_t>()[pid];
      int32_t last_page_len = seq_len % page_size;

      // --- Part 1: Copy from out_cache_loc to req_to_token ---
      // This maps the physical slots allocated (out_cache_loc) to the logical sequence view (req_to_token)

      int64_t copy_len = 0;
      const int32_t* src_part1_ptr = nullptr;

      if (page_size == 1 || topk == 1) {
        // Simplified case: direct mapping
        copy_len = topk * speculative_num_steps;
        src_part1_ptr = out_cache_loc.data_ptr<int32_t>() + pid * topk * speculative_num_steps;
      } else {
        // General case: use extend_len and calculated prefix offset
        copy_len = extend_len;
        // offset is the sum of extend_lens of all previous sequences
        src_part1_ptr = out_cache_loc.data_ptr<int32_t>() + extend_lens_prefix.data_ptr<int32_t>()[pid];
      }

      int32_t* dst_part1_ptr = req_to_token_base + seq_len;

      // Ensure we don't go beyond the req_to_token capacity
      int64_t max_copy_len = pool_len - seq_len;
      if (copy_len > max_copy_len) {
        copy_len = max_copy_len;
      }

      // Scalar copy loop (replacing vectorized ops to comply with header constraints)
      for (int64_t d = 0; d < copy_len; ++d) {
        dst_part1_ptr[d] = src_part1_ptr[d];
      }

      // --- Part 2 & 3: Handle duplication and compact out_cache_loc ---
      if (page_size != 1 && topk != 1 && duplicate_cache_len > 0) {
        // Calculate pointer to the start of the last page of the prefix
        int32_t* prefix_base_ptr = req_to_token_base + (seq_len - last_page_len);

        // Calculate global offsets for source/target cache arrays
        // last_page_lens_cumsum passed in is inclusive sum. We need exclusive sum for base.
        // Formula: (topk - 1) * (inclusive_sum[pid] - last_page_len[pid])

        int32_t exclusive_cumsum = 0;
        if (last_page_lens_cumsum_opt.has_value()) {
          auto& last_page_lens_cumsum = last_page_lens_cumsum_opt.value();
          int32_t cumsum_val = last_page_lens_cumsum.data_ptr<int32_t>()[pid];
          exclusive_cumsum = cumsum_val - last_page_len;
        }

        int64_t global_buffer_offset = static_cast<int64_t>(topk - 1) * exclusive_cumsum;

        // Bounds check for global buffer offset
        if (global_buffer_offset >= source_cache_loc.numel() || global_buffer_offset >= target_cache_loc.numel()) {
          continue;
        }

        int32_t* src_cache_ptr = source_cache_loc.data_ptr<int32_t>() + global_buffer_offset;
        int32_t* tgt_cache_ptr = target_cache_loc.data_ptr<int32_t>() + global_buffer_offset;

        // --- Part 2: Fill source_cache_loc and target_cache_loc ---
        // Iterate over topk branches (skipping 0)
        for (int32_t k_id = 1; k_id < topk; ++k_id) {
          int64_t part2_offset = (k_id - 1) * last_page_len;

          // Bounds check for part2_offset
          if (part2_offset >= source_cache_loc.numel() - global_buffer_offset ||
              part2_offset >= target_cache_loc.numel() - global_buffer_offset) {
            break;
          }

          // 1. Source: copy from prefix (last page tokens)
          for (int64_t d = 0; d < last_page_len; ++d) {
            if (part2_offset + d < source_cache_loc.numel()) {
              src_cache_ptr[part2_offset + d] = prefix_base_ptr[d];
            }
          }

          // 2. Target: copy from the newly allocated pages in req_to_token
          int32_t* new_page_ptr = prefix_base_ptr + k_id * num_new_pages * page_size;

          if (new_page_ptr - req_to_token_base >= pool_len) {
            continue;
          }

          for (int64_t d = 0; d < last_page_len; ++d) {
            if (part2_offset + d < target_cache_loc.numel()) {
              tgt_cache_ptr[part2_offset + d] = new_page_ptr[d];
            }
          }
        }

        // --- Part 3: Re-pack out_cache_loc ---
        // Extract only the actual draft tokens (excluding the duplicated prefix part)
        // Write them to a contiguous block in out_cache_loc corresponding to this batch

        // Base offset in out_cache_loc for this sequence's output: pid * topk * speculative_num_steps
        int32_t* out_part3_base = out_cache_loc.data_ptr<int32_t>() + pid * topk * speculative_num_steps;

        for (int32_t k_id = 0; k_id < topk; ++k_id) {
          // Source in req_to_token: skip the first 'last_page_len' tokens of the allocated block
          // Start reading from: prefix_base + k_id * num_new_pages * page_size + last_page_len
          int32_t* read_ptr = prefix_base_ptr + k_id * num_new_pages * page_size + last_page_len;

          // Bounds check for read_ptr access
          if (read_ptr - req_to_token_base >= pool_len) {
            continue;
          }

          // Destination in out_cache_loc
          int32_t* write_ptr = out_part3_base + k_id * speculative_num_steps;

          // Copy exactly speculative_num_steps tokens
          for (int64_t d = 0; d < speculative_num_steps; ++d) {
            write_ptr[d] = read_ptr[d];
          }
        }
      }
    }
  });
}

}  // anonymous namespace

at::Tensor assign_draft_cache_locs_cpu(
    const at::Tensor& req_pool_indices,
    const at::Tensor& req_to_token,
    const at::Tensor& seq_lens,
    const at::Tensor& extend_lens,
    const at::Tensor& num_new_pages_per_topk,
    at::Tensor& out_cache_loc,
    const std::optional<at::Tensor>& source_cache_loc_opt,
    const std::optional<at::Tensor>& target_cache_loc_opt,
    const std::optional<at::Tensor>& last_page_lens_cumsum_opt,
    int64_t duplicate_cache_len,
    int64_t pool_len,
    int64_t topk,
    int64_t speculative_num_steps,
    int64_t page_size,
    int64_t bs_upper,
    int64_t iter_upper) {
  // Handle optional inputs: if not provided, create a dummy tensor to satisfy the kernel interface.
  // The kernel logic checks `duplicate_cache_len` before using these tensors.
  at::Tensor source_cache_loc =
      source_cache_loc_opt.has_value() ? *source_cache_loc_opt : at::empty({0}, out_cache_loc.options());

  at::Tensor target_cache_loc =
      target_cache_loc_opt.has_value() ? *target_cache_loc_opt : at::empty({0}, out_cache_loc.options());

  // Call the CPU kernel
  assign_draft_cache_locs_cpu_kernel(
      req_pool_indices,
      req_to_token,
      seq_lens,
      extend_lens,
      num_new_pages_per_topk,
      out_cache_loc,
      source_cache_loc,
      target_cache_loc,
      last_page_lens_cumsum_opt,
      duplicate_cache_len,
      pool_len,
      topk,
      speculative_num_steps,
      page_size,
      bs_upper,
      iter_upper);

  return out_cache_loc;
}

void verify_tree_greedy_cpu(
    at::Tensor predicts,
    at::Tensor accept_index,
    at::Tensor accept_token_num,
    const at::Tensor& candidates,
    const at::Tensor& retrive_index,
    const at::Tensor& retrive_next_token,
    const at::Tensor& retrive_next_sibling,
    const at::Tensor& target_predict) {
  TORCH_CHECK(predicts.dtype() == at::kInt, "predicts must be int32");
  TORCH_CHECK(accept_index.dtype() == at::kInt, "accept_index must be int32");
  TORCH_CHECK(accept_token_num.dtype() == at::kInt, "accept_token_num must be int32");
  TORCH_CHECK(candidates.dtype() == at::kLong, "candidates must be int64");
  TORCH_CHECK(retrive_index.dtype() == at::kLong, "retrive_index must be int64");
  TORCH_CHECK(retrive_next_token.dtype() == at::kLong, "retrive_next_token must be int64");
  TORCH_CHECK(retrive_next_sibling.dtype() == at::kLong, "retrive_next_sibling must be int64");
  TORCH_CHECK(target_predict.dtype() == at::kLong, "target_predict must be int64");
  TORCH_CHECK(predicts.is_contiguous(), "predicts must be contiguous");
  TORCH_CHECK(accept_index.is_contiguous(), "accept_index must be contiguous");
  TORCH_CHECK(accept_token_num.is_contiguous(), "accept_token_num must be contiguous");
  TORCH_CHECK(candidates.is_contiguous(), "candidates must be contiguous");
  TORCH_CHECK(retrive_index.is_contiguous(), "retrive_index must be contiguous");
  TORCH_CHECK(retrive_next_token.is_contiguous(), "retrive_next_token must be contiguous");
  TORCH_CHECK(retrive_next_sibling.is_contiguous(), "retrive_next_sibling must be contiguous");
  TORCH_CHECK(target_predict.is_contiguous(), "target_predict must be contiguous");
  int64_t batch_size = candidates.size(0);
  int64_t num_spec_step = accept_index.size(1);
  int64_t num_draft_tokens = candidates.size(1);

  auto* predicts_ptr = predicts.data_ptr<int32_t>();
  auto* accept_idx_ptr = accept_index.data_ptr<int32_t>();
  auto* accept_num_ptr = accept_token_num.data_ptr<int32_t>();
  auto* cand_ptr = candidates.data_ptr<int64_t>();
  auto* ri_ptr = retrive_index.data_ptr<int64_t>();
  auto* rnt_ptr = retrive_next_token.data_ptr<int64_t>();
  auto* rns_ptr = retrive_next_sibling.data_ptr<int64_t>();
  auto* tp_ptr = target_predict.data_ptr<int64_t>();

  at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
    for (int64_t bx = begin; bx < end; ++bx) {
      int64_t off = bx * num_draft_tokens;
      int64_t ai_off = bx * num_spec_step;

      int64_t last_accepted = ri_ptr[off];  // retrive_index[bx, 0]
      accept_idx_ptr[ai_off] = static_cast<int32_t>(last_accepted);

      int32_t num_accepted = 0;
      int64_t cur = 0;

      for (int64_t j = 1; j < num_spec_step; ++j) {
        cur = rnt_ptr[off + cur];  // move to next token
        while (cur != -1) {
          int64_t draft_idx = ri_ptr[off + cur];
          int64_t draft_tok = cand_ptr[off + cur];
          int64_t target_tok = tp_ptr[last_accepted];
          if (draft_tok == target_tok) {
            predicts_ptr[last_accepted] = static_cast<int32_t>(target_tok);
            ++num_accepted;
            accept_idx_ptr[ai_off + num_accepted] = static_cast<int32_t>(draft_idx);
            last_accepted = draft_idx;
            break;
          }
          cur = rns_ptr[off + cur];  // try sibling
        }
        if (cur == -1) break;
      }
      accept_num_ptr[bx] = num_accepted;
      predicts_ptr[last_accepted] = static_cast<int32_t>(tp_ptr[last_accepted]);
    }
  });
}

void build_tree_kernel_efficient_cpu(
    const at::Tensor& parent_list,
    const at::Tensor& selected_index,
    const at::Tensor& verified_seq_len,
    at::Tensor tree_mask,
    at::Tensor positions,
    at::Tensor retrive_index,
    at::Tensor retrive_next_token,
    at::Tensor retrive_next_sibling,
    int64_t topk,
    int64_t depth,
    int64_t draft_token_num,
    int64_t tree_mask_mode) {
  TORCH_CHECK(parent_list.dtype() == at::kLong, "parent_list must be int64");
  TORCH_CHECK(selected_index.dtype() == at::kLong, "selected_index must be int64");
  TORCH_CHECK(verified_seq_len.dtype() == at::kLong, "verified_seq_len must be int64");
  TORCH_CHECK(tree_mask.dtype() == at::kBool, "tree_mask must be bool");
  TORCH_CHECK(positions.dtype() == at::kLong, "positions must be int64");
  TORCH_CHECK(retrive_index.dtype() == at::kLong, "retrive_index must be int64");
  TORCH_CHECK(retrive_next_token.dtype() == at::kLong, "retrive_next_token must be int64");
  TORCH_CHECK(retrive_next_sibling.dtype() == at::kLong, "retrive_next_sibling must be int64");
  TORCH_CHECK(parent_list.is_contiguous(), "parent_list must be contiguous");
  TORCH_CHECK(selected_index.is_contiguous(), "selected_index must be contiguous");
  TORCH_CHECK(verified_seq_len.is_contiguous(), "verified_seq_len must be contiguous");
  TORCH_CHECK(tree_mask.is_contiguous(), "tree_mask must be contiguous");
  TORCH_CHECK(positions.is_contiguous(), "positions must be contiguous");
  TORCH_CHECK(retrive_index.is_contiguous(), "retrive_index must be contiguous");
  TORCH_CHECK(retrive_next_token.is_contiguous(), "retrive_next_token must be contiguous");
  TORCH_CHECK(retrive_next_sibling.is_contiguous(), "retrive_next_sibling must be contiguous");
  // tree_mask_mode: 0=FULL_MASK, 1=QLEN_ONLY (2=QLEN_ONLY_BITPACKING is rejected below)
  int64_t bs = parent_list.size(0);

  auto* parent_ptr = parent_list.data_ptr<int64_t>();
  auto* sel_ptr = selected_index.data_ptr<int64_t>();
  auto* seqlen_ptr = verified_seq_len.data_ptr<int64_t>();
  auto* pos_ptr = positions.data_ptr<int64_t>();
  auto* ri_ptr = retrive_index.data_ptr<int64_t>();
  auto* rnt_ptr = retrive_next_token.data_ptr<int64_t>();
  auto* rns_ptr = retrive_next_sibling.data_ptr<int64_t>();

  int64_t parent_stride = topk * (depth - 1) + 1;
  int64_t sel_stride = draft_token_num - 1;

  // Determine bytes per item for bitpacking mode
  size_t num_bytes_per_item = 1;
  if (tree_mask_mode == 2) {  // QLEN_ONLY_BITPACKING
    if (draft_token_num > 16)
      num_bytes_per_item = 4;
    else if (draft_token_num > 8)
      num_bytes_per_item = 2;
  }

  for (int64_t bid = 0; bid < bs; ++bid) {
    int64_t off = bid * draft_token_num;
    int64_t seq_len = seqlen_ptr[bid];

    // tid == 0 logic: build retrive_index, retrive_next_token, retrive_next_sibling
    pos_ptr[off] = seq_len;
    ri_ptr[off] = off;  // retrive_index[bid, 0] = bid * draft_token_num

    for (int64_t i = draft_token_num - 1; i > 0; --i) {
      ri_ptr[off + i] = off + i;
      int64_t parent_tb_idx = sel_ptr[bid * sel_stride + i - 1] / topk;
      int64_t parent_position = 0;
      if (parent_tb_idx > 0) {
        int64_t parent_token_idx = parent_ptr[bid * parent_stride + parent_tb_idx];
        for (; parent_position < draft_token_num; ++parent_position) {
          if (sel_ptr[bid * sel_stride + parent_position] == parent_token_idx) {
            ++parent_position;
            break;
          }
        }
      }
      if (parent_position == draft_token_num) {
        continue;  // skip invalid
      }
      if (rnt_ptr[off + parent_position] == -1) {
        rnt_ptr[off + parent_position] = i;
      } else {
        int64_t origin = rnt_ptr[off + parent_position];
        rnt_ptr[off + parent_position] = i;
        rns_ptr[off + i] = origin;
      }
    }

    // Build tree_mask and positions for tid > 0
    if (tree_mask_mode == 2) {  // QLEN_ONLY_BITPACKING
      uint8_t* mask_base = reinterpret_cast<uint8_t*>(tree_mask.data_ptr());
      for (int64_t tid = 0; tid < draft_token_num; ++tid) {
        size_t item_offset = (off + tid) * num_bytes_per_item;
        mask_base[item_offset] = 1;  // set self bit (little endian)
      }
      for (int64_t tid = 1; tid < draft_token_num; ++tid) {
        size_t item_offset = (off + tid) * num_bytes_per_item;
        int64_t position = 0;
        int64_t cur = tid - 1;
        while (true) {
          position++;
          int64_t byte_idx = (cur + 1) / 8;
          int64_t bit_idx = (cur + 1) % 8;
          mask_base[item_offset + byte_idx] |= (1 << bit_idx);
          int64_t ptb = sel_ptr[bid * sel_stride + cur] / topk;
          if (ptb == 0) break;
          int64_t tok_idx = parent_ptr[bid * parent_stride + ptb];
          for (cur = 0; cur < draft_token_num; ++cur) {
            if (sel_ptr[bid * sel_stride + cur] == tok_idx) break;
          }
        }
        pos_ptr[off + tid] = position + seq_len;
      }
    } else if (tree_mask_mode == 1) {  // QLEN_ONLY
      bool* mask_ptr = tree_mask.data_ptr<bool>();
      int64_t mask_stride = draft_token_num;
      for (int64_t tid = 0; tid < draft_token_num; ++tid) {
        int64_t row_start = (off + tid) * mask_stride;
        mask_ptr[row_start] = true;  // attend to the root token (column 0)
        for (int64_t j = 1; j < draft_token_num; ++j) {
          mask_ptr[row_start + j] = false;
        }
        if (tid == 0) {
          continue;
        }
        int64_t position = 0;
        int64_t cur = tid - 1;
        while (true) {
          position++;
          mask_ptr[row_start + cur + 1] = true;
          int64_t ptb = sel_ptr[bid * sel_stride + cur] / topk;
          if (ptb == 0) break;
          int64_t tok_idx = parent_ptr[bid * parent_stride + ptb];
          for (cur = 0; cur < draft_token_num; ++cur) {
            if (sel_ptr[bid * sel_stride + cur] == tok_idx) break;
          }
        }
        pos_ptr[off + tid] = position + seq_len;
      }
    } else {  // FULL_MASK (mode 0)
      // Full mask includes the seq_len prefix
      bool* mask_ptr = tree_mask.data_ptr<bool>();
      int64_t seq_tree_idx = bid * draft_token_num * draft_token_num;
      for (int64_t i = 0; i < bid; ++i) {
        seq_tree_idx += static_cast<int64_t>(seqlen_ptr[i]) * draft_token_num;
      }
      for (int64_t tid = 0; tid < draft_token_num; ++tid) {
        int64_t row_start = seq_tree_idx + (seq_len + draft_token_num) * tid + seq_len;
        mask_ptr[row_start] = true;  // attend to the root token (column 0)
        for (int64_t j = 1; j < draft_token_num; ++j) {
          mask_ptr[row_start + j] = false;
        }
        if (tid == 0) {
          continue;
        }
        int64_t position = 0;
        int64_t cur = tid - 1;
        while (true) {
          position++;
          mask_ptr[row_start + cur + 1] = true;
          int64_t ptb = sel_ptr[bid * sel_stride + cur] / topk;
          if (ptb == 0) {
            break;
          }
          int64_t tok_idx = parent_ptr[bid * parent_stride + ptb];
          for (cur = 0; cur < draft_token_num; ++cur) {
            if (sel_ptr[bid * sel_stride + cur] == tok_idx) {
              break;
            }
          }
        }
        pos_ptr[off + tid] = position + seq_len;
      }
    }
  }
}

void create_extend_after_decode_spec_info_cpu(
    const at::Tensor& verified_id,
    const at::Tensor& seq_lens,
    const at::Tensor& accept_lens,
    at::Tensor positions,
    at::Tensor new_verified_id,
    int64_t bs_upper) {
  int64_t batch_size = seq_lens.size(0);
  auto* vid_ptr = verified_id.data_ptr<int32_t>();
  auto* sl_ptr = seq_lens.data_ptr<int32_t>();
  auto* al_ptr = accept_lens.data_ptr<int32_t>();
  auto* pos_ptr = positions.data_ptr<int64_t>();
  auto* nvid_ptr = new_verified_id.data_ptr<int32_t>();

  // Pre-compute exclusive prefix sum of accept_lens to avoid O(N^2) work.
  std::vector<int64_t> prefix(batch_size + 1, 0);
  for (int64_t i = 0; i < batch_size; ++i) {
    prefix[i + 1] = prefix[i] + al_ptr[i];
  }

  at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
    for (int64_t pid = begin; pid < end; ++pid) {
      int32_t seq_length = sl_ptr[pid];
      int32_t accept_length = al_ptr[pid];
      int64_t cumsum = prefix[pid];

      for (int32_t k = 0; k < accept_length; ++k) {
        pos_ptr[cumsum + k] = seq_length - accept_length + k;
      }

      nvid_ptr[pid] = vid_ptr[cumsum + accept_length - 1];
    }
  });
}

void align_evict_mask_to_page_size_cpu(
    const at::Tensor& seq_lens, at::Tensor evict_mask, int64_t page_size, int64_t num_draft_tokens) {
  const int32_t* __restrict__ seq_lens_ptr = seq_lens.data_ptr<int32_t>();
  bool* __restrict__ evict_mask_ptr = evict_mask.data_ptr<bool>();
  int64_t batch_size = seq_lens.size(0);

  at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
    for (int64_t bid = begin; bid < end; ++bid) {
      int64_t seq_len = static_cast<int64_t>(seq_lens_ptr[bid]);
      int64_t mask_row_offset = bid * num_draft_tokens;

      // Count the tokens that would actually be evicted (mask == True).
      int64_t num_trues = 0;
      for (int64_t t = 0; t < num_draft_tokens; ++t) {
        if (evict_mask_ptr[mask_row_offset + t]) {
          num_trues++;
        }
      }
      int64_t num_false = num_draft_tokens - num_trues;

      // start = (seq_len + num_false - 1) // page_size * page_size - seq_len
      int64_t numer = seq_len + num_false - 1;
      int64_t quot = numer / page_size;
      if (numer % page_size != 0 && numer < 0) {
        quot -= 1;
      }
      int64_t start = quot * page_size - seq_len;
      int64_t clear_start = std::max(start, static_cast<int64_t>(0));
      int64_t clear_end = std::min(start + page_size, num_draft_tokens);
      for (int64_t i = clear_start; i < clear_end; ++i) {
        evict_mask_ptr[mask_row_offset + i] = false;
      }
    }
  });
}

void get_target_cache_loc_cpu(
    at::Tensor tgt_cache_loc,
    at::Tensor to_free_slots,
    const at::Tensor& num_correct_drafts,
    const at::Tensor& to_free_num_slots,
    const at::Tensor& out_cache_loc,
    int64_t num_verify_tokens) {
  int64_t batch_size = num_correct_drafts.size(0);
  const int64_t* __restrict__ al_ptr = num_correct_drafts.data_ptr<int64_t>();
  const int64_t* __restrict__ fns_ptr = to_free_num_slots.data_ptr<int64_t>();
  const int64_t* __restrict__ ocl_ptr = out_cache_loc.data_ptr<int64_t>();
  int64_t* __restrict__ tgt_ptr = tgt_cache_loc.data_ptr<int64_t>();
  int64_t* __restrict__ free_ptr = to_free_slots.data_ptr<int64_t>();

  // Exclusive prefix sums to find each request's write offsets.
  // tgt start additionally adds `bid` (the +1 per request in copy_len).
  std::vector<int64_t> tgt_start(batch_size, 0);
  std::vector<int64_t> free_start(batch_size, 0);
  int64_t acc_sum = 0;
  int64_t free_sum = 0;
  for (int64_t bid = 0; bid < batch_size; ++bid) {
    tgt_start[bid] = acc_sum + bid;
    free_start[bid] = free_sum;
    acc_sum += al_ptr[bid];
    free_sum += fns_ptr[bid];
  }

  at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
    for (int64_t bid = begin; bid < end; ++bid) {
      const int64_t row_base = bid * num_verify_tokens;

      // Part 1: keep the first accept_length[bid]+1 slots.
      int64_t copy_len = al_ptr[bid] + 1;
      int64_t dst = tgt_start[bid];
      for (int64_t j = 0; j < copy_len; ++j) {
        tgt_ptr[dst + j] = ocl_ptr[row_base + j];
      }

      // Part 2: free the trailing to_free_num_slots[bid] slots.
      int64_t free_len = fns_ptr[bid];
      int64_t src_off = num_verify_tokens - free_len;
      int64_t fdst = free_start[bid];
      for (int64_t j = 0; j < free_len; ++j) {
        free_ptr[fdst + j] = ocl_ptr[row_base + src_off + j];
      }
    }
  });
}

void assign_req_to_token_pool_cpu(
    const at::Tensor& req_pool_indices,
    at::Tensor req_to_token,
    const at::Tensor& start_offset,
    const at::Tensor& end_offset,
    const at::Tensor& out_cache_loc,
    int64_t pool_len) {
  int64_t batch_size = req_pool_indices.size(0);
  auto* rpi_ptr = req_pool_indices.data_ptr<int32_t>();
  auto* rtt_ptr = req_to_token.data_ptr<int32_t>();
  auto* so_ptr = start_offset.data_ptr<int32_t>();
  auto* eo_ptr = end_offset.data_ptr<int32_t>();
  auto* ocl_ptr = out_cache_loc.data_ptr<int64_t>();

  // Pre-compute exclusive prefix sum of (end - start) to avoid O(N^2) work.
  std::vector<int64_t> prefix(batch_size + 1, 0);
  for (int64_t i = 0; i < batch_size; ++i) {
    prefix[i + 1] = prefix[i] + (eo_ptr[i] - so_ptr[i]);
  }

  at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
    for (int64_t pid = begin; pid < end; ++pid) {
      int32_t kv_start = so_ptr[pid];
      int32_t kv_end = eo_ptr[pid];
      int32_t* token_pool = rtt_ptr + rpi_ptr[pid] * pool_len;
      int64_t out_offset = prefix[pid];

      for (int32_t j = kv_start; j < kv_end; ++j) {
        token_pool[j] = static_cast<int32_t>(ocl_ptr[out_offset + (j - kv_start)]);
      }
    }
  });
}

void create_flashinfer_kv_indices_cpu(
    const at::Tensor& req_to_token,
    const at::Tensor& req_pool_indices,
    const at::Tensor& page_kernel_lens,
    const at::Tensor& kv_indptr,
    const std::optional<at::Tensor>& kv_start_idx,
    at::Tensor kv_indices,
    int64_t req_to_token_stride) {
  int64_t batch_size = req_pool_indices.size(0);
  auto* rtt_ptr = req_to_token.data_ptr<int32_t>();
  auto* rpi_ptr = req_pool_indices.data_ptr<int32_t>();
  auto* pkl_ptr = page_kernel_lens.data_ptr<int32_t>();
  auto* indptr_ptr = kv_indptr.data_ptr<int32_t>();
  auto* kvi_ptr = kv_indices.data_ptr<int32_t>();

  const int32_t* ksi_ptr = nullptr;
  if (kv_start_idx.has_value() && kv_start_idx->defined()) {
    ksi_ptr = kv_start_idx->data_ptr<int32_t>();
  }

  at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      int64_t req_pool_index = rpi_ptr[i];
      int64_t kv_offset = indptr_ptr[i];
      int64_t kv_start = ksi_ptr ? ksi_ptr[i] : 0;
      int64_t kv_len = pkl_ptr[i];

      const int32_t* src = rtt_ptr + req_pool_index * req_to_token_stride + kv_start;
      int32_t* dst = kvi_ptr + kv_offset;

      for (int64_t j = 0; j < kv_len; ++j) {
        dst[j] = src[j];
      }
    }
  });
}

at::Tensor build_draft_decode_metadata_cpu(
    const at::Tensor& req_to_token,
    const at::Tensor& req_pool_indices,
    const at::Tensor& seq_lens,
    int64_t topk,
    int64_t num_steps,
    int64_t pool_len) {
  int64_t num_seqs = req_pool_indices.size(0);
  int64_t bs = num_seqs * topk;

  auto req_to_token_draft = at::empty({bs, pool_len}, req_to_token.options());

  auto* rtt_ptr = req_to_token.data_ptr<int32_t>();
  auto* draft_ptr = req_to_token_draft.data_ptr<int32_t>();
  auto* rpi_ptr = req_pool_indices.data_ptr<int64_t>();
  auto* sl_ptr = seq_lens.data_ptr<int64_t>();

  at::parallel_for(0, num_seqs, 0, [&](int64_t begin, int64_t end) {
    for (int64_t b = begin; b < end; ++b) {
      int64_t idx = rpi_ptr[b];
      int64_t sl = sl_ptr[b];
      const int32_t* src_row = rtt_ptr + idx * pool_len;

      for (int64_t tk = 0; tk < topk; ++tk) {
        int64_t flat = b * topk + tk;
        int32_t* dst_row = draft_ptr + flat * pool_len;

        // Copy prefix
        std::memcpy(dst_row, src_row, sl * sizeof(int32_t));

        // Copy draft tokens for this candidate
        int64_t draft_start = sl + tk * num_steps;
        for (int64_t s = 0; s < num_steps; ++s) {
          dst_row[sl + s] = src_row[draft_start + s];
        }
      }
    }
  });

  return req_to_token_draft;
}

void fill_bonus_tokens_cpu(
    const at::Tensor& accept_tokens, const at::Tensor& accept_lens, at::Tensor bonus_tokens, int64_t accept_stride) {
  TORCH_CHECK(accept_tokens.dtype() == at::kInt, "accept_tokens must be int32");
  TORCH_CHECK(accept_lens.dtype() == at::kInt, "accept_lens must be int32");
  TORCH_CHECK(bonus_tokens.dtype() == at::kInt, "bonus_tokens must be int32");
  TORCH_CHECK(accept_tokens.is_contiguous(), "accept_tokens must be contiguous");
  TORCH_CHECK(accept_lens.is_contiguous(), "accept_lens must be contiguous");
  TORCH_CHECK(bonus_tokens.is_contiguous(), "bonus_tokens must be contiguous");
  int64_t bs = accept_lens.size(0);
  auto* accept_ptr = accept_tokens.data_ptr<int32_t>();
  auto* al_ptr = accept_lens.data_ptr<int32_t>();
  auto* out_ptr = bonus_tokens.data_ptr<int32_t>();

  at::parallel_for(0, bs, 0, [&](int64_t begin, int64_t end) {
    for (int64_t pid = begin; pid < end; ++pid) {
      int64_t idx = accept_stride * pid + al_ptr[pid] - 1;
      out_ptr[pid] = accept_ptr[idx];
    }
  });
}

void fill_accept_out_cache_loc_cpu(
    const at::Tensor& accept_index, const at::Tensor& out_cache_loc, at::Tensor accept_out_cache_loc, int64_t size) {
  TORCH_CHECK(accept_index.dtype() == at::kInt, "accept_index must be int32");
  TORCH_CHECK(out_cache_loc.dtype() == at::kLong, "out_cache_loc must be int64");
  TORCH_CHECK(accept_out_cache_loc.dtype() == at::kLong, "accept_out_cache_loc must be int64");
  TORCH_CHECK(accept_index.is_contiguous(), "accept_index must be contiguous");
  TORCH_CHECK(out_cache_loc.is_contiguous(), "out_cache_loc must be contiguous");
  TORCH_CHECK(accept_out_cache_loc.is_contiguous(), "accept_out_cache_loc must be contiguous");
  auto* ai_ptr = accept_index.data_ptr<int32_t>();
  auto* ocl_ptr = out_cache_loc.data_ptr<int64_t>();
  auto* out_ptr = accept_out_cache_loc.data_ptr<int64_t>();

  int64_t dst = 0;
  for (int64_t i = 0; i < size; ++i) {
    int64_t src = static_cast<int64_t>(ai_ptr[i]);
    if (src > -1) {
      out_ptr[dst++] = ocl_ptr[src];
    }
  }
}

void assign_draft_cache_locs_contiguous_cpu(
    const at::Tensor& req_pool_indices,
    const at::Tensor& req_to_token,
    const at::Tensor& seq_lens,
    at::Tensor out_cache_loc,
    int64_t pool_len,
    int64_t topk,
    int64_t num_steps) {
  // Contiguous slot layout: requires page_size == 1 or topk == 1 (see prepare_for_v2_draft guard).
  TORCH_CHECK(req_pool_indices.scalar_type() == at::kLong && seq_lens.scalar_type() == at::kLong);
  TORCH_CHECK(req_to_token.scalar_type() == at::kInt && out_cache_loc.scalar_type() == at::kLong);
  TORCH_CHECK(out_cache_loc.numel() == req_pool_indices.numel() * topk * num_steps);
  int64_t bs = req_pool_indices.size(0);
  int64_t copy_len = topk * num_steps;

  auto* rpi_ptr = req_pool_indices.data_ptr<int64_t>();
  auto* rtt_ptr = req_to_token.data_ptr<int32_t>();
  auto* sl_ptr = seq_lens.data_ptr<int64_t>();
  auto* out_ptr = out_cache_loc.data_ptr<int64_t>();

  at::parallel_for(0, bs, 0, [&](int64_t begin, int64_t end) {
    for (int64_t pid = begin; pid < end; ++pid) {
      int64_t kv_start = sl_ptr[pid];
      int64_t req_idx = rpi_ptr[pid];
      const int32_t* src = rtt_ptr + req_idx * pool_len + kv_start;
      int64_t* dst = out_ptr + pid * copy_len;
      for (int64_t j = 0; j < copy_len; ++j) {
        dst[j] = static_cast<int64_t>(src[j]);
      }
    }
  });
}

void assign_extend_cache_locs_cpu(
    const at::Tensor& req_pool_indices,
    const at::Tensor& req_to_token,
    const at::Tensor& start_offset,
    const at::Tensor& end_offset,
    at::Tensor out_cache_loc,
    int64_t pool_len) {
  int64_t bs = req_pool_indices.size(0);
  auto* rpi_ptr = req_pool_indices.data_ptr<int64_t>();
  auto* rtt_ptr = req_to_token.data_ptr<int32_t>();
  auto* start_ptr = start_offset.data_ptr<int64_t>();
  auto* end_ptr = end_offset.data_ptr<int64_t>();
  auto* out_ptr = out_cache_loc.data_ptr<int64_t>();

  // Compute prefix sum for output offsets (sequential)
  std::vector<int64_t> out_offsets(bs + 1, 0);
  for (int64_t i = 0; i < bs; ++i) {
    out_offsets[i + 1] = out_offsets[i] + (end_ptr[i] - start_ptr[i]);
  }

  at::parallel_for(0, bs, 0, [&](int64_t begin, int64_t end) {
    for (int64_t pid = begin; pid < end; ++pid) {
      int64_t kv_start = start_ptr[pid];
      int64_t kv_end = end_ptr[pid];
      int64_t req_idx = rpi_ptr[pid];
      int64_t length = kv_end - kv_start;
      const int32_t* src = rtt_ptr + req_idx * pool_len + kv_start;
      int64_t* dst = out_ptr + out_offsets[pid];
      for (int64_t j = 0; j < length; ++j) {
        dst[j] = static_cast<int64_t>(src[j]);
      }
    }
  });
}

void rotate_input_ids_cpu(
    at::Tensor input_ids,
    const at::Tensor& extend_start_loc,
    const at::Tensor& extend_seq_lens,
    const at::Tensor& topk_index,
    const c10::optional<at::Tensor>& select_index_opt) {
  TORCH_CHECK(input_ids.dtype() == at::kLong, "input_ids must be int64");
  TORCH_CHECK(extend_start_loc.dtype() == at::kLong, "extend_start_loc must be int64");
  TORCH_CHECK(extend_seq_lens.dtype() == at::kLong, "extend_seq_lens must be int64");
  TORCH_CHECK(topk_index.dtype() == at::kLong, "topk_index must be int64");
  TORCH_CHECK(input_ids.is_contiguous(), "input_ids must be contiguous");
  TORCH_CHECK(extend_start_loc.is_contiguous(), "extend_start_loc must be contiguous");
  TORCH_CHECK(extend_seq_lens.is_contiguous(), "extend_seq_lens must be contiguous");
  TORCH_CHECK(topk_index.is_contiguous(), "topk_index must be contiguous");
  if (select_index_opt.has_value()) {
    TORCH_CHECK(select_index_opt.value().dtype() == at::kLong, "select_index must be int64");
    TORCH_CHECK(select_index_opt.value().is_contiguous(), "select_index must be contiguous");
  }
  int64_t bs = extend_seq_lens.size(0);
  auto* ids_ptr = input_ids.data_ptr<int64_t>();
  auto* start_ptr = extend_start_loc.data_ptr<int64_t>();
  auto* lens_ptr = extend_seq_lens.data_ptr<int64_t>();
  auto* topk_ptr = topk_index.data_ptr<int64_t>();
  const int64_t* select_ptr = select_index_opt.has_value() ? select_index_opt.value().data_ptr<int64_t>() : nullptr;

  at::parallel_for(0, bs, 0, [&](int64_t begin, int64_t end) {
    for (int64_t pid = begin; pid < end; ++pid) {
      int64_t start = start_ptr[pid];
      int64_t seq_len = lens_ptr[pid];
      int64_t new_token = topk_ptr[pid];

      // Shift left by 1
      if (seq_len > 1) {
        std::memmove(ids_ptr + start, ids_ptr + start + 1, (seq_len - 1) * sizeof(int64_t));
      }
      // Write new token
      if (seq_len > 0) {
        if (select_ptr != nullptr) {
          ids_ptr[select_ptr[pid]] = new_token;
        } else {
          ids_ptr[start + seq_len - 1] = new_token;
        }
      }
    }
  });
}

// NOTE: index tensors must be int64; the existing GPU graph-runner call site feeds int32 — any future CPU wiring must
// cast.
void assign_new_state_cpu(
    const at::Tensor& next_token_ids,
    const at::Tensor& old_input_ids,
    const at::Tensor& old_positions,
    const at::Tensor& old_out_cache_loc,
    const at::Tensor& old_extend_seq_lens,
    const at::Tensor& old_extend_start_loc,
    at::Tensor input_ids,
    at::Tensor positions,
    at::Tensor out_cache_loc,
    at::Tensor extend_seq_lens,
    at::Tensor extend_start_loc,
    const at::Tensor& seq_lens,
    const at::Tensor& padding_lens,
    const at::Tensor& req_pool_indices,
    const at::Tensor& req_to_token,
    int64_t num_seqs,
    int64_t step,
    at::Tensor hidden_states,
    const at::Tensor& old_hidden_states,
    const at::Tensor& req_to_hidden_states_pool) {
  auto* nti_ptr = next_token_ids.data_ptr<int64_t>();
  auto* old_ids_ptr = old_input_ids.data_ptr<int64_t>();
  auto* old_pos_ptr = old_positions.data_ptr<int64_t>();
  auto* old_ocl_ptr = old_out_cache_loc.data_ptr<int64_t>();
  auto* old_esl_ptr = old_extend_seq_lens.data_ptr<int64_t>();
  auto* old_esloc_ptr = old_extend_start_loc.data_ptr<int64_t>();

  auto* ids_ptr = input_ids.data_ptr<int64_t>();
  auto* pos_ptr = positions.data_ptr<int64_t>();
  auto* ocl_ptr = out_cache_loc.data_ptr<int64_t>();
  auto* esl_ptr = extend_seq_lens.data_ptr<int64_t>();
  auto* esloc_ptr = extend_start_loc.data_ptr<int64_t>();

  auto* sl_ptr = seq_lens.data_ptr<int64_t>();
  auto* pl_ptr = padding_lens.data_ptr<int64_t>();
  auto* rpi_ptr = req_pool_indices.data_ptr<int64_t>();
  int64_t rtt_stride = req_to_token.stride(0);

  // Hidden states dimensions
  int64_t hidden_dim = hidden_states.size(1);
  int64_t pool_step_stride = req_to_hidden_states_pool.stride(1);
  int64_t pool_dim_stride = req_to_hidden_states_pool.stride(2);
  int64_t pool_req_stride = req_to_hidden_states_pool.stride(0);
  // Use byte-level copies to handle any dtype (bfloat16, float16, float32)
  int64_t elem_size = hidden_states.element_size();
  int64_t row_bytes = hidden_dim * elem_size;
  char* hs_base = reinterpret_cast<char*>(hidden_states.data_ptr());
  const char* old_hs_base = reinterpret_cast<const char*>(old_hidden_states.data_ptr());
  const char* pool_base = reinterpret_cast<const char*>(req_to_hidden_states_pool.data_ptr());
  int64_t pool_elem_size = req_to_hidden_states_pool.element_size();

  at::parallel_for(0, num_seqs, 0, [&](int64_t begin, int64_t end) {
    for (int64_t pid = begin; pid < end; ++pid) {
      int64_t seq_len = sl_ptr[pid];
      int64_t old_ext_len = old_esl_ptr[pid];
      int64_t old_start = old_esloc_ptr[pid];
      int64_t new_ext_len = old_ext_len + 1;
      int64_t new_start = old_start + pid;

      esl_ptr[pid] = new_ext_len;
      esloc_ptr[pid] = new_start;

      // Copy old input_ids, then write new token
      if (old_ext_len > 0) {
        std::memcpy(ids_ptr + new_start, old_ids_ptr + old_start, old_ext_len * sizeof(int64_t));
      }
      int64_t pad_len = pl_ptr[pid];
      ids_ptr[new_start + old_ext_len - pad_len] = nti_ptr[pid];

      // Copy old positions shifted by 1, prepend (old_pos[0] - 1)
      if (old_ext_len > 0) {
        std::memcpy(pos_ptr + new_start + 1, old_pos_ptr + old_start, old_ext_len * sizeof(int64_t));
      }
      int64_t first_pos = old_pos_ptr[old_start];
      pos_ptr[new_start] = first_pos > 0 ? first_pos - 1 : 0;

      // Copy old out_cache_loc shifted by 1
      if (old_ext_len > 0) {
        std::memcpy(ocl_ptr + new_start + 1, old_ocl_ptr + old_start, old_ext_len * sizeof(int64_t));
      }

      // Prepend the cache loc from req_to_token
      int64_t req_idx = rpi_ptr[pid];
      int64_t token_idx_col = seq_len - old_ext_len - 1;
      if (token_idx_col >= 0) {
        auto* rtt_ptr_base = req_to_token.data_ptr<int32_t>();
        ocl_ptr[new_start] = static_cast<int64_t>(rtt_ptr_base[req_idx * rtt_stride + token_idx_col]);
      }

      // Copy hidden states: shift old by 1, prepend from pool
      if (old_ext_len > 0) {
        // hidden_states[new_start+1 : new_start+1+old_ext_len] = old_hidden_states[old_start : old_start+old_ext_len]
        std::memcpy(
            hs_base + (new_start + 1) * row_bytes, old_hs_base + old_start * row_bytes, old_ext_len * row_bytes);
      }
      // hidden_states[new_start] = req_to_hidden_states_pool[req_idx+1, -(step+1)]
      // Pool layout: [num_reqs, pool_size, hidden_dim] — index with strides
      int64_t pool_offset = (req_idx + 1) * pool_req_stride + (-(step + 1)) * pool_step_stride;
      std::memcpy(hs_base + new_start * row_bytes, pool_base + pool_offset * pool_elem_size, row_bytes);
    }
  });
}
