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
  // tree_mask_mode: 0=FULL_MASK, 1=QLEN_ONLY, 2=QLEN_ONLY_BITPACKING
  int64_t bs = parent_list.size(0);

  auto* parent_ptr = parent_list.data_ptr<int64_t>();
  auto* sel_ptr = selected_index.data_ptr<int64_t>();
  auto* seqlen_ptr = verified_seq_len.data_ptr<int32_t>();
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
    int32_t seq_len = seqlen_ptr[bid];

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

void assign_req_to_token_pool_cpu(
    const at::Tensor& req_pool_indices,
    at::Tensor req_to_token,
    const at::Tensor& start_offset,
    const at::Tensor& end_offset,
    const at::Tensor& out_cache_loc,
    int64_t pool_len,
    int64_t bs_upper) {
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
    const at::Tensor& accept_index, const at::Tensor& out_cache_loc, at::Tensor accepted_out_cache_loc, int64_t size) {
  auto* ai_ptr = accept_index.data_ptr<int32_t>();
  auto* ocl_ptr = out_cache_loc.data_ptr<int64_t>();
  auto* out_ptr = accepted_out_cache_loc.data_ptr<int64_t>();

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
