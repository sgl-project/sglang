#include "common.h"

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

      int64_t last_accept_index = ri_ptr[off];  // retrive_index[bx, 0]
      accept_idx_ptr[ai_off] = static_cast<int32_t>(last_accept_index);

      int32_t num_correct_drafts = 0;
      int64_t cur = 0;

      for (int64_t j = 1; j < num_spec_step; ++j) {
        cur = rnt_ptr[off + cur];  // move to next token
        while (cur != -1) {
          int64_t draft_idx = ri_ptr[off + cur];
          int64_t draft_tok = cand_ptr[off + cur];
          int64_t target_tok = tp_ptr[last_accept_index];
          if (draft_tok == target_tok) {
            predicts_ptr[last_accept_index] = static_cast<int32_t>(target_tok);
            ++num_correct_drafts;
            accept_idx_ptr[ai_off + num_correct_drafts] = static_cast<int32_t>(draft_idx);
            last_accept_index = draft_idx;
            break;
          }
          cur = rns_ptr[off + cur];  // try sibling
        }
        if (cur == -1) break;
      }
      accept_num_ptr[bx] = num_correct_drafts;
      predicts_ptr[last_accept_index] = static_cast<int32_t>(tp_ptr[last_accept_index]);
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

  // CPU workers always use FULL_MASK or QLEN_ONLY; QLEN_ONLY_BITPACKING has no
  // CPU producer and is untested here.
  TORCH_CHECK(tree_mask_mode != 2, "build_tree_kernel_efficient_cpu: QLEN_ONLY_BITPACKING is not supported on CPU");

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
    if (tree_mask_mode == 1) {  // QLEN_ONLY
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
