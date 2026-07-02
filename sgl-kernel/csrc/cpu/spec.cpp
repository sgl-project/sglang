#include "common.h"

namespace {

// Contract shared by every kernel in this file: all tensors are dense,
// contiguous CPU tensors (checked below), so strides are the canonical
// row-major ones; per-function comments list shapes and dtypes only.
// `index_t` params accept int32 or int64 via AT_DISPATCH_INDEX_TYPES so
// callers never pay a dtype-conversion copy.

template <typename rpi_t, typename off_t>
void assign_req_to_token_pool_kernel_impl(
    const rpi_t* __restrict__ rpi_ptr,
    int32_t* __restrict__ rtt_ptr,
    const off_t* __restrict__ so_ptr,
    const off_t* __restrict__ eo_ptr,
    const int64_t* __restrict__ ocl_ptr,
    int64_t batch_size,
    int64_t pool_len) {
  // Pre-compute exclusive prefix sum of (end - start) to avoid O(N^2) work.
  std::vector<int64_t> prefix(batch_size + 1, 0);
  for (int64_t i = 0; i < batch_size; ++i) {
    prefix[i + 1] = prefix[i] + (eo_ptr[i] - so_ptr[i]);
  }

  at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
    for (int64_t pid = begin; pid < end; ++pid) {
      int64_t kv_start = so_ptr[pid];
      int64_t kv_end = eo_ptr[pid];
      int32_t* token_pool = rtt_ptr + rpi_ptr[pid] * pool_len;
      int64_t out_offset = prefix[pid];

      for (int64_t j = kv_start; j < kv_end; ++j) {
        token_pool[j] = static_cast<int32_t>(ocl_ptr[out_offset + (j - kv_start)]);
      }
    }
  });
}

template <typename index_t>
void verify_tree_greedy_kernel_impl(
    int32_t* __restrict__ predicts_ptr,
    int32_t* __restrict__ accept_idx_ptr,
    int32_t* __restrict__ accept_num_ptr,
    const index_t* __restrict__ cand_ptr,
    const index_t* __restrict__ ri_ptr,
    const index_t* __restrict__ rnt_ptr,
    const index_t* __restrict__ rns_ptr,
    const index_t* __restrict__ tp_ptr,
    int64_t batch_size,
    int64_t num_spec_step,
    int64_t num_draft_tokens) {
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

template <typename index_t>
void build_tree_kernel_efficient_impl(
    const index_t* __restrict__ parent_ptr,
    const index_t* __restrict__ sel_ptr,
    const index_t* __restrict__ seqlen_ptr,
    bool* __restrict__ mask_ptr,
    index_t* __restrict__ pos_ptr,
    index_t* __restrict__ ri_ptr,
    index_t* __restrict__ rnt_ptr,
    index_t* __restrict__ rns_ptr,
    int64_t bs,
    int64_t topk,
    int64_t depth,
    int64_t draft_token_num,
    int64_t tree_mask_mode) {
  int64_t parent_stride = topk * (depth - 1) + 1;
  int64_t sel_stride = draft_token_num - 1;

  // FULL_MASK row offsets depend on a prefix sum over verified_seq_len;
  // precompute it so the batch loop can run in parallel.
  std::vector<int64_t> mask_offsets(bs, 0);
  if (tree_mask_mode == 0) {  // FULL_MASK
    int64_t acc = 0;
    for (int64_t i = 0; i < bs; ++i) {
      mask_offsets[i] = i * draft_token_num * draft_token_num + acc;
      acc += static_cast<int64_t>(seqlen_ptr[i]) * draft_token_num;
    }
  }

  at::parallel_for(0, bs, 0, [&](int64_t begin, int64_t end) {
    for (int64_t bid = begin; bid < end; ++bid) {
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
        int64_t seq_tree_idx = mask_offsets[bid];
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
  });
}

}  // anonymous namespace

// Greedy tree verification: walk each request's draft tree, accepting the
// longest root path whose draft tokens match the target model's argmax.
//
// predicts:            [bs * num_draft_tokens] int32; out, verified tokens by flat draft index
// accept_index:        [bs, num_spec_step] int32; out, flat indices of accepted tokens, -1 padded
// accept_token_num:    [bs] int32; out, accepted drafts per request (bonus excluded)
// candidates:          [bs, num_draft_tokens] int32 or int64; draft tokens
// retrive_index:       [bs, num_draft_tokens] int32 or int64; flat index of each tree node
// retrive_next_token:  [bs, num_draft_tokens] int32 or int64; first child, -1 = none
// retrive_next_sibling:[bs, num_draft_tokens] int32 or int64; next sibling, -1 = none
// target_predict:      [bs, num_draft_tokens] int32 or int64; target argmax per draft slot
void verify_tree_greedy_cpu(
    at::Tensor predicts,
    at::Tensor accept_index,
    at::Tensor accept_token_num,
    const at::Tensor& candidates,
    const at::Tensor& retrive_index,
    const at::Tensor& retrive_next_token,
    const at::Tensor& retrive_next_sibling,
    const at::Tensor& target_predict) {
  CHECK_INPUT(predicts);
  CHECK_INPUT(accept_index);
  CHECK_INPUT(accept_token_num);
  CHECK_INPUT(candidates);
  CHECK_INPUT(retrive_index);
  CHECK_INPUT(retrive_next_token);
  CHECK_INPUT(retrive_next_sibling);
  CHECK_INPUT(target_predict);
  CHECK_EQ(predicts.scalar_type(), at::kInt);
  CHECK_EQ(accept_index.scalar_type(), at::kInt);
  CHECK_EQ(accept_token_num.scalar_type(), at::kInt);
  CHECK_DIM(2, candidates);
  CHECK_DIM(2, accept_index);
  const auto index_dtype = retrive_index.scalar_type();
  CHECK_EQ(candidates.scalar_type(), index_dtype);
  CHECK_EQ(retrive_next_token.scalar_type(), index_dtype);
  CHECK_EQ(retrive_next_sibling.scalar_type(), index_dtype);
  CHECK_EQ(target_predict.scalar_type(), index_dtype);

  int64_t batch_size = candidates.size(0);
  int64_t num_spec_step = accept_index.size(1);
  int64_t num_draft_tokens = candidates.size(1);

  AT_DISPATCH_INDEX_TYPES(index_dtype, "verify_tree_greedy_indices", [&] {
    verify_tree_greedy_kernel_impl<index_t>(
        predicts.data_ptr<int32_t>(),
        accept_index.data_ptr<int32_t>(),
        accept_token_num.data_ptr<int32_t>(),
        candidates.data_ptr<index_t>(),
        retrive_index.data_ptr<index_t>(),
        retrive_next_token.data_ptr<index_t>(),
        retrive_next_sibling.data_ptr<index_t>(),
        target_predict.data_ptr<index_t>(),
        batch_size,
        num_spec_step,
        num_draft_tokens);
  });
}

// Build the draft token tree consumed by target verify: tree attention mask,
// per-token positions, and the retrieval linkage (index / first child /
// next sibling) used by verify_tree_greedy.
//
// parent_list:         [bs, topk * (depth - 1) + 1] int32 or int64
// selected_index:      [bs, draft_token_num - 1] int32 or int64
// verified_seq_len:    [bs] int32 or int64; committed prefix length per request
// tree_mask:           out, bool.
//                      QLEN_ONLY: [bs * draft_token_num * draft_token_num]
//                      FULL_MASK: [sum_i(seq_len_i * draft_token_num) + bs * draft_token_num^2]
// positions:           [bs * draft_token_num]; out, same dtype as parent_list
// retrive_index:       [bs, draft_token_num]; out
// retrive_next_token:  [bs, draft_token_num]; out, pre-filled with -1
// retrive_next_sibling:[bs, draft_token_num]; out, pre-filled with -1
// tree_mask_mode:      0 = FULL_MASK, 1 = QLEN_ONLY (2 = QLEN_ONLY_BITPACKING is rejected)
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
  CHECK_INPUT(parent_list);
  CHECK_INPUT(selected_index);
  CHECK_INPUT(verified_seq_len);
  CHECK_INPUT(tree_mask);
  CHECK_INPUT(positions);
  CHECK_INPUT(retrive_index);
  CHECK_INPUT(retrive_next_token);
  CHECK_INPUT(retrive_next_sibling);
  CHECK_EQ(tree_mask.scalar_type(), at::kBool);
  const auto index_dtype = parent_list.scalar_type();
  CHECK_EQ(selected_index.scalar_type(), index_dtype);
  CHECK_EQ(verified_seq_len.scalar_type(), index_dtype);
  CHECK_EQ(positions.scalar_type(), index_dtype);
  CHECK_EQ(retrive_index.scalar_type(), index_dtype);
  CHECK_EQ(retrive_next_token.scalar_type(), index_dtype);
  CHECK_EQ(retrive_next_sibling.scalar_type(), index_dtype);

  // CPU workers always use FULL_MASK or QLEN_ONLY; QLEN_ONLY_BITPACKING has no
  // CPU producer and is untested here.
  TORCH_CHECK(tree_mask_mode != 2, "build_tree_kernel_efficient_cpu: QLEN_ONLY_BITPACKING is not supported on CPU");

  int64_t bs = parent_list.size(0);

  AT_DISPATCH_INDEX_TYPES(index_dtype, "build_tree_kernel_efficient_indices", [&] {
    build_tree_kernel_efficient_impl<index_t>(
        parent_list.data_ptr<index_t>(),
        selected_index.data_ptr<index_t>(),
        verified_seq_len.data_ptr<index_t>(),
        tree_mask.data_ptr<bool>(),
        positions.data_ptr<index_t>(),
        retrive_index.data_ptr<index_t>(),
        retrive_next_token.data_ptr<index_t>(),
        retrive_next_sibling.data_ptr<index_t>(),
        bs,
        topk,
        depth,
        draft_token_num,
        tree_mask_mode);
  });
}

// Scatter freshly allocated KV slots into the request-to-token map:
// req_to_token[req_pool_indices[i], start_offset[i]:end_offset[i]] =
// out_cache_loc[prefix[i]:prefix[i+1]].
//
// req_pool_indices: [bs] int32 or int64
// req_to_token:     [max_num_reqs, pool_len] int32; out
// start_offset:     [bs] int32 or int64 (independent of req_pool_indices;
//                   eagle_prepare_for_decode passes int64 indices with int32 kv lens)
// end_offset:       [bs] same dtype as start_offset
// out_cache_loc:    [sum_i(end_offset[i] - start_offset[i])] int64
void assign_req_to_token_pool_cpu(
    const at::Tensor& req_pool_indices,
    at::Tensor req_to_token,
    const at::Tensor& start_offset,
    const at::Tensor& end_offset,
    const at::Tensor& out_cache_loc,
    int64_t pool_len) {
  CHECK_INPUT(req_pool_indices);
  CHECK_INPUT(req_to_token);
  CHECK_INPUT(start_offset);
  CHECK_INPUT(end_offset);
  CHECK_INPUT(out_cache_loc);
  CHECK_EQ(req_to_token.scalar_type(), at::kInt);
  CHECK_EQ(out_cache_loc.scalar_type(), at::kLong);
  CHECK_EQ(end_offset.scalar_type(), start_offset.scalar_type());

  int64_t batch_size = req_pool_indices.size(0);
  auto* rtt_ptr = req_to_token.data_ptr<int32_t>();
  auto* ocl_ptr = out_cache_loc.data_ptr<int64_t>();

  AT_DISPATCH_INDEX_TYPES(req_pool_indices.scalar_type(), "assign_req_to_token_pool_rpi", [&] {
    using rpi_t = index_t;
    const rpi_t* rpi_ptr = req_pool_indices.data_ptr<rpi_t>();
    AT_DISPATCH_INDEX_TYPES(start_offset.scalar_type(), "assign_req_to_token_pool_offsets", [&] {
      assign_req_to_token_pool_kernel_impl<rpi_t, index_t>(
          rpi_ptr,
          rtt_ptr,
          start_offset.data_ptr<index_t>(),
          end_offset.data_ptr<index_t>(),
          ocl_ptr,
          batch_size,
          pool_len);
    });
  });
}

// Expand req_to_token for multi-step draft decode: row b*topk+tk holds the
// committed prefix of request b followed by candidate tk's draft slots
// (which assign_draft_cache_locs_contiguous laid out at sl + tk*num_steps).
//
// req_to_token:     [max_num_reqs, pool_len] int32
// req_pool_indices: [num_seqs] int32 or int64
// seq_lens:         [num_seqs] int32 or int64 (independent of req_pool_indices)
// returns:          [num_seqs * topk, pool_len] int32; only the first
//                   seq_lens[b] + num_steps entries of each row are defined
at::Tensor build_draft_decode_metadata_cpu(
    const at::Tensor& req_to_token,
    const at::Tensor& req_pool_indices,
    const at::Tensor& seq_lens,
    int64_t topk,
    int64_t num_steps,
    int64_t pool_len) {
  CHECK_INPUT(req_to_token);
  CHECK_INPUT(req_pool_indices);
  CHECK_INPUT(seq_lens);
  CHECK_EQ(req_to_token.scalar_type(), at::kInt);

  int64_t num_seqs = req_pool_indices.size(0);
  int64_t bs = num_seqs * topk;

  auto req_to_token_draft = at::empty({bs, pool_len}, req_to_token.options());

  auto* rtt_ptr = req_to_token.data_ptr<int32_t>();
  auto* draft_ptr = req_to_token_draft.data_ptr<int32_t>();

  AT_DISPATCH_INDEX_TYPES(req_pool_indices.scalar_type(), "build_draft_decode_metadata_rpi", [&] {
    using rpi_t = index_t;
    const rpi_t* rpi_ptr = req_pool_indices.data_ptr<rpi_t>();
    AT_DISPATCH_INDEX_TYPES(seq_lens.scalar_type(), "build_draft_decode_metadata_lens", [&] {
      const index_t* sl_ptr = seq_lens.data_ptr<index_t>();

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
    });
  });

  return req_to_token_draft;
}

// Pick the last accepted token of each request as its bonus token.
//
// accept_tokens: [bs, accept_stride] int32; row-major, accept_stride = accept_index.shape[1]
// accept_lens:   [bs] int32; number of accepted tokens per request (bonus included)
// bonus_tokens:  [bs] int32; out
void fill_bonus_tokens_cpu(
    const at::Tensor& accept_tokens, const at::Tensor& accept_lens, at::Tensor bonus_tokens, int64_t accept_stride) {
  CHECK_INPUT(accept_tokens);
  CHECK_INPUT(accept_lens);
  CHECK_INPUT(bonus_tokens);
  CHECK_EQ(accept_tokens.scalar_type(), at::kInt);
  CHECK_EQ(accept_lens.scalar_type(), at::kInt);
  CHECK_EQ(bonus_tokens.scalar_type(), at::kInt);

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

// Compact the accepted tokens' KV slots: gather out_cache_loc at the accepted
// indices, skipping -1 (rejected) entries. Sequential by design: the output
// write position depends on how many prior entries were accepted.
//
// accept_index:         [size] int32 or int64; flat (bs * num_spec_step), -1 = rejected
// out_cache_loc:        [bs * num_draft_tokens] int64
// accept_out_cache_loc: [num_accept] int64; out
void fill_accept_out_cache_loc_cpu(
    const at::Tensor& accept_index, const at::Tensor& out_cache_loc, at::Tensor accept_out_cache_loc, int64_t size) {
  CHECK_INPUT(accept_index);
  CHECK_INPUT(out_cache_loc);
  CHECK_INPUT(accept_out_cache_loc);
  CHECK_EQ(out_cache_loc.scalar_type(), at::kLong);
  CHECK_EQ(accept_out_cache_loc.scalar_type(), at::kLong);

  auto* ocl_ptr = out_cache_loc.data_ptr<int64_t>();
  auto* out_ptr = accept_out_cache_loc.data_ptr<int64_t>();

  AT_DISPATCH_INDEX_TYPES(accept_index.scalar_type(), "fill_accept_out_cache_loc_indices", [&] {
    const index_t* ai_ptr = accept_index.data_ptr<index_t>();
    int64_t dst = 0;
    for (int64_t i = 0; i < size; ++i) {
      int64_t src = static_cast<int64_t>(ai_ptr[i]);
      if (src > -1) {
        out_ptr[dst++] = ocl_ptr[src];
      }
    }
  });
}

// Read back the draft KV slots reserved by the allocator: for each request,
// copy the topk*num_steps slots starting at seq_lens[pid] out of req_to_token.
//
// req_pool_indices: [bs] int32 or int64
// req_to_token:     [max_num_reqs, pool_len] int32
// seq_lens:         [bs] int32 or int64 (independent of req_pool_indices)
// out_cache_loc:    [bs * topk * num_steps] int64; out
void assign_draft_cache_locs_contiguous_cpu(
    const at::Tensor& req_pool_indices,
    const at::Tensor& req_to_token,
    const at::Tensor& seq_lens,
    at::Tensor out_cache_loc,
    int64_t pool_len,
    int64_t topk,
    int64_t num_steps) {
  // Contiguous slot layout: requires page_size == 1 or topk == 1 (see prepare_for_v2_draft guard).
  CHECK_INPUT(req_pool_indices);
  CHECK_INPUT(req_to_token);
  CHECK_INPUT(seq_lens);
  CHECK_INPUT(out_cache_loc);
  CHECK_EQ(req_to_token.scalar_type(), at::kInt);
  CHECK_EQ(out_cache_loc.scalar_type(), at::kLong);
  CHECK_EQ(out_cache_loc.numel(), req_pool_indices.numel() * topk * num_steps);

  int64_t bs = req_pool_indices.size(0);
  int64_t copy_len = topk * num_steps;

  auto* rtt_ptr = req_to_token.data_ptr<int32_t>();
  auto* out_ptr = out_cache_loc.data_ptr<int64_t>();

  AT_DISPATCH_INDEX_TYPES(req_pool_indices.scalar_type(), "assign_draft_cache_locs_contiguous_rpi", [&] {
    using rpi_t = index_t;
    const rpi_t* rpi_ptr = req_pool_indices.data_ptr<rpi_t>();
    AT_DISPATCH_INDEX_TYPES(seq_lens.scalar_type(), "assign_draft_cache_locs_contiguous_lens", [&] {
      const index_t* sl_ptr = seq_lens.data_ptr<index_t>();

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
    });
  });
}

// Gather each request's KV slots in [start_offset, end_offset) out of
// req_to_token into a dense int64 vector (verify/extend cache locations).
//
// req_pool_indices: [bs] int32 or int64
// req_to_token:     [max_num_reqs, pool_len] int32
// start_offset:     [bs] int32 or int64 (independent of req_pool_indices)
// end_offset:       [bs] same dtype as start_offset
// out_cache_loc:    [sum_i(end_offset[i] - start_offset[i])] int64; out
void assign_extend_cache_locs_cpu(
    const at::Tensor& req_pool_indices,
    const at::Tensor& req_to_token,
    const at::Tensor& start_offset,
    const at::Tensor& end_offset,
    at::Tensor out_cache_loc,
    int64_t pool_len) {
  CHECK_INPUT(req_pool_indices);
  CHECK_INPUT(req_to_token);
  CHECK_INPUT(start_offset);
  CHECK_INPUT(end_offset);
  CHECK_INPUT(out_cache_loc);
  CHECK_EQ(req_to_token.scalar_type(), at::kInt);
  CHECK_EQ(out_cache_loc.scalar_type(), at::kLong);
  CHECK_EQ(end_offset.scalar_type(), start_offset.scalar_type());

  int64_t bs = req_pool_indices.size(0);
  auto* rtt_ptr = req_to_token.data_ptr<int32_t>();
  auto* out_ptr = out_cache_loc.data_ptr<int64_t>();

  AT_DISPATCH_INDEX_TYPES(req_pool_indices.scalar_type(), "assign_extend_cache_locs_rpi", [&] {
    using rpi_t = index_t;
    const rpi_t* rpi_ptr = req_pool_indices.data_ptr<rpi_t>();
    AT_DISPATCH_INDEX_TYPES(start_offset.scalar_type(), "assign_extend_cache_locs_offsets", [&] {
      const index_t* start_ptr = start_offset.data_ptr<index_t>();
      const index_t* end_ptr = end_offset.data_ptr<index_t>();

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
    });
  });
}

// Recover tree linkage from a QLEN-layout boolean tree mask (NGRAM path):
// depth/position, retrieval index, first child and next sibling per node.
//
// tree_mask:           [bs * draft_token_num * draft_token_num] bool
// verified_seq_len:    [bs] int32 or int64
// positions:           [bs * draft_token_num]; out, same dtype as verified_seq_len
// retrive_index:       [bs, draft_token_num]; out
// retrive_next_token:  [bs, draft_token_num]; out
// retrive_next_sibling:[bs, draft_token_num]; out
void reconstruct_indices_from_tree_mask_cpu(
    at::Tensor tree_mask,
    at::Tensor verified_seq_len,
    at::Tensor positions,
    at::Tensor retrive_index,
    at::Tensor retrive_next_token,
    at::Tensor retrive_next_sibling,
    int64_t batch_size,
    int64_t draft_token_num) {
  CHECK_INPUT(tree_mask);
  CHECK_INPUT(verified_seq_len);
  CHECK_INPUT(positions);
  CHECK_INPUT(retrive_index);
  CHECK_INPUT(retrive_next_token);
  CHECK_INPUT(retrive_next_sibling);
  CHECK_EQ(tree_mask.scalar_type(), at::kBool);
  const auto index_dtype = verified_seq_len.scalar_type();
  CHECK_EQ(positions.scalar_type(), index_dtype);
  CHECK_EQ(retrive_index.scalar_type(), index_dtype);
  CHECK_EQ(retrive_next_token.scalar_type(), index_dtype);
  CHECK_EQ(retrive_next_sibling.scalar_type(), index_dtype);

  auto* mask_ptr = tree_mask.data_ptr<bool>();
  int64_t base_offset = draft_token_num * draft_token_num;

  AT_DISPATCH_INDEX_TYPES(index_dtype, "reconstruct_indices_from_tree_mask_indices", [&] {
    const index_t* seq_len_ptr = verified_seq_len.data_ptr<index_t>();
    index_t* pos_ptr = positions.data_ptr<index_t>();
    index_t* ri_ptr = retrive_index.data_ptr<index_t>();
    index_t* rnt_ptr = retrive_next_token.data_ptr<index_t>();
    index_t* rns_ptr = retrive_next_sibling.data_ptr<index_t>();

    at::parallel_for(0, batch_size * draft_token_num, 0, [&](int64_t begin, int64_t end) {
      for (int64_t idx = begin; idx < end; ++idx) {
        int64_t bid = idx / draft_token_num;
        int64_t tid = idx % draft_token_num;

        int64_t token_idx = bid * draft_token_num;
        int64_t tree_mask_offset = bid * base_offset;

        // Step 1: depth and parent via backward scan
        int64_t depth = 0;
        int64_t parent_idx = -1;
        for (int64_t i = tid - 1, start_idx = tree_mask_offset + tid * draft_token_num; i >= 0; --i) {
          if (mask_ptr[start_idx + i]) {
            depth++;
            if (parent_idx == -1) {
              parent_idx = i;
            }
          }
        }

        // Step 2: retrive_index (identity)
        ri_ptr[token_idx + tid] = token_idx + tid;

        // Step 3: position = depth + verified_seq_len
        pos_ptr[token_idx + tid] = depth + seq_len_ptr[bid];

        // Step 4: first child (next_token)
        int64_t next_token_idx = -1;
        for (int64_t i = tid + 1; i < draft_token_num; ++i) {
          if (mask_ptr[tree_mask_offset + i * draft_token_num + tid]) {
            next_token_idx = i;
            break;
          }
        }
        rnt_ptr[token_idx + tid] = next_token_idx;

        // Step 5: next sibling (shares parent, no intervening ancestors)
        int64_t next_sibling_idx = -1;
        if (parent_idx != -1) {
          for (int64_t i = tid + 1; i < draft_token_num; ++i) {
            int64_t si = tree_mask_offset + i * draft_token_num + parent_idx;
            if (mask_ptr[si]) {
              bool is_sibling = true;
              int64_t ei = tree_mask_offset + i * draft_token_num + i;
              for (int64_t j = si + 1; j < ei; ++j) {
                if (mask_ptr[j]) {
                  is_sibling = false;
                  break;
                }
              }
              if (is_sibling) {
                next_sibling_idx = i;
                break;
              }
            }
          }
        }
        rns_ptr[token_idx + tid] = next_sibling_idx;
      }
    });
  });
}

// Shift each request's extend segment left by one token and write the new
// draft token at the end (or at select_index when given). Mutates input_ids
// in place; callers rely on this.
//
// input_ids:        [num_extend_tokens] int64; in/out
// extend_start_loc: [bs] int32 or int64
// extend_seq_lens:  [bs] int32 or int64 (independent of extend_start_loc; the
//                   spec decode-extend batch pairs int64 lens with int32 locs)
// topk_index:       [bs] int64; new draft token per request
// select_index:     [bs] int64 or None; global slot for the new token
void rotate_input_ids_cpu(
    at::Tensor input_ids,
    const at::Tensor& extend_start_loc,
    const at::Tensor& extend_seq_lens,
    const at::Tensor& topk_index,
    const c10::optional<at::Tensor>& select_index_opt) {
  CHECK_INPUT(input_ids);
  CHECK_INPUT(extend_start_loc);
  CHECK_INPUT(extend_seq_lens);
  CHECK_INPUT(topk_index);
  CHECK_EQ(input_ids.scalar_type(), at::kLong);
  CHECK_EQ(topk_index.scalar_type(), at::kLong);
  if (select_index_opt.has_value()) {
    CHECK_INPUT(select_index_opt.value());
    CHECK_EQ(select_index_opt.value().scalar_type(), at::kLong);
  }

  int64_t bs = extend_seq_lens.size(0);
  auto* ids_ptr = input_ids.data_ptr<int64_t>();
  auto* topk_ptr = topk_index.data_ptr<int64_t>();
  const int64_t* select_ptr = select_index_opt.has_value() ? select_index_opt.value().data_ptr<int64_t>() : nullptr;

  AT_DISPATCH_INDEX_TYPES(extend_start_loc.scalar_type(), "rotate_input_ids_start", [&] {
    using start_t = index_t;
    const start_t* start_ptr = extend_start_loc.data_ptr<start_t>();
    AT_DISPATCH_INDEX_TYPES(extend_seq_lens.scalar_type(), "rotate_input_ids_lens", [&] {
      const index_t* lens_ptr = extend_seq_lens.data_ptr<index_t>();

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
    });
  });
}
