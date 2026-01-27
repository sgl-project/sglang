#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cub/cub.cuh>

#include "quest_kernels.h"
#include "utils.h"

// Kernel to compute scores and initialize indices for sorting
template <typename T>
__global__ void quest_score_kernel_opt(
    float* __restrict__ scores,                    // [bs, max_pages]
    int32_t* __restrict__ indices,                 // [bs, max_pages]
    const int32_t* __restrict__ seq_lens,          // [bs]
    const int32_t* __restrict__ req_to_token,      // [req_pool_size, max_tokens]
    const T* __restrict__ page_k_min,              // [total_pages, kv_heads, head_dim]
    const T* __restrict__ page_k_max,              // [total_pages, kv_heads, head_dim]
    const T* __restrict__ queries,                 // [bs, q_heads, head_dim]
    const int32_t* __restrict__ req_pool_indices,  // [bs]
    int64_t num_recent_pages,
    int64_t max_pages,
    int64_t page_size,
    int64_t kv_heads,
    int64_t q_heads,
    int64_t head_dim,
    int64_t req_to_token_stride_req,
    int64_t req_to_token_stride_token,
    int64_t req_to_token_num_tokens,
    int64_t page_k_stride_page,
    int64_t page_k_stride_head,
    int64_t page_k_stride_dim,
    int64_t queries_stride_req,
    int64_t queries_stride_head,
    int64_t queries_stride_dim) {
  // Dynamic shared memory for Q_avg
  // Size: kv_heads * head_dim * sizeof(float)
  extern __shared__ float s_q_avg[];

  int64_t req_idx = blockIdx.x;
  int64_t tid = threadIdx.x;

  // 1. Compute/Load Q_avg into Shared Memory
  // All threads in block cooperate to compute Q_avg for the single request req_idx
  int64_t num_features = kv_heads * head_dim;
  int64_t group_size = q_heads / kv_heads;

  for (int64_t i = tid; i < num_features; i += blockDim.x) {
    int64_t h = i / head_dim;
    int64_t d = i % head_dim;

    float q_sum = 0.0f;
    for (int64_t g = 0; g < group_size; ++g) {
      int64_t q_h = h * group_size + g;
      int64_t q_offset = req_idx * queries_stride_req + q_h * queries_stride_head + d * queries_stride_dim;
      q_sum += (float)queries[q_offset];
    }
    s_q_avg[i] = q_sum / group_size;
  }

  __syncthreads();

  // 2. Process Pages (Warp per Page)
  // Map warps to pages
  int64_t warp_id = tid / 32;
  int64_t lane_id = tid % 32;
  int64_t warps_per_block = blockDim.x / 32;

  // Each warp handles one page_rel_idx
  int64_t page_rel_idx = blockIdx.y * warps_per_block + warp_id;

  // Bounds check
  if (page_rel_idx >= max_pages) return;

  int64_t out_idx = req_idx * max_pages + page_rel_idx;

  // Default initialization for indices
  if (lane_id == 0) {
    indices[out_idx] = page_rel_idx;
  }

  // Check validity
  int64_t seq_len = seq_lens[req_idx];
  int64_t num_pages = (seq_len + page_size - 1) / page_size;
  int64_t recent_start = num_pages - num_recent_pages;
  if (recent_start < 0) recent_start = 0;

  bool valid = (page_rel_idx < num_pages) && (page_rel_idx < recent_start);

  if (!valid) {
    if (lane_id == 0) {
      scores[out_idx] = -INFINITY;
    }
    return;
  }

  // Compute Score
  int32_t req_id = req_pool_indices[req_idx];
  int64_t log_tok_idx = page_rel_idx * page_size;
  int64_t safe_log_tok_idx = log_tok_idx;
  if (safe_log_tok_idx >= req_to_token_num_tokens) safe_log_tok_idx = req_to_token_num_tokens - 1;

  int64_t offset_req_tok = (int64_t)req_id * req_to_token_stride_req + safe_log_tok_idx * req_to_token_stride_token;
  int32_t phys_tok = req_to_token[offset_req_tok];
  int32_t phys_page_idx = phys_tok / page_size;

  float score_acc = 0.0f;
  int64_t k_base_offset = phys_page_idx * page_k_stride_page;

  // Iterate over features (kv_heads * head_dim)
  for (int64_t i = lane_id; i < num_features; i += 32) {
    float q_val = s_q_avg[i];

    // Calculate k_offset assuming contiguous layout or strided
    // i = h * head_dim + d
    int64_t h = i / head_dim;
    int64_t d = i % head_dim;
    int64_t k_offset = k_base_offset + h * page_k_stride_head + d * page_k_stride_dim;

    float k_val;
    // Optimization: Conditional Load
    // Only load max or min based on q sign
    if (q_val >= 0.0f) {
      k_val = (float)page_k_max[k_offset];
    } else {
      k_val = (float)page_k_min[k_offset];
    }

    score_acc += q_val * k_val;
  }

  // Warp Reduction
  for (int offset = 16; offset > 0; offset /= 2) {
    score_acc += __shfl_down_sync(0xffffffff, score_acc, offset);
  }

  if (lane_id == 0) {
    scores[out_idx] = score_acc;
  }
}

// Combine indices kernel
__global__ void quest_combine_kernel(
    const int32_t* __restrict__ sorted_indices,  // [bs, max_pages] (sorted by score desc)
    int32_t* __restrict__ out_indices,           // [bs, max_out]
    int32_t* __restrict__ out_lengths,           // [bs]
    const int32_t* __restrict__ seq_lens,
    int64_t num_recent_pages,
    int64_t max_pages,
    int64_t max_out,
    int64_t page_size,
    bool fixed_k_mode,
    int64_t fixed_k_val,
    double sparsity_ratio,
    const int32_t* __restrict__ sparse_mask) {
  int64_t req_idx = blockIdx.x;
  if (req_idx >= gridDim.x) return;

  int32_t seq_len = seq_lens[req_idx];
  int32_t num_pages = (seq_len + page_size - 1) / page_size;

  // Determine K
  int64_t recent_start = num_pages - num_recent_pages;
  if (recent_start < 0) recent_start = 0;

  int64_t history_pages = recent_start;
  if (history_pages < 1) history_pages = 1;  // Clamp min=1

  int64_t k_target;
  if (fixed_k_mode) {
    k_target = fixed_k_val - num_recent_pages;
    if (k_target < 1) k_target = 1;
  } else {
    k_target = (int64_t)(history_pages * sparsity_ratio);
    if (k_target < 1) k_target = 1;
  }

  if (k_target > history_pages) k_target = history_pages;
  if (sparse_mask && sparse_mask[req_idx] == 0) {
    k_target = 0;  // Or handle mask logic. Python: k_per_req * sparse_mask
  }

  // Limit by max_pages
  if (k_target > max_pages) k_target = max_pages;
  if (k_target > history_pages) k_target = history_pages;  // Redundant but safe

  int64_t out_cnt = 0;
  int32_t* my_out = out_indices + req_idx * max_out;

  // 1. Copy Top K

  const int32_t* my_sorted = sorted_indices + req_idx * max_pages;

  for (int64_t i = 0; i < k_target; ++i) {
    if (i < max_pages) {
      int32_t p_idx = my_sorted[i];
      if (p_idx >= 0 && p_idx < recent_start) {
        my_out[out_cnt++] = p_idx;
      }
    }
  }

  // 2. Add Recent Pages
  for (int64_t i = 0; i < num_recent_pages; ++i) {
    int32_t p_idx = recent_start + i;
    if (p_idx < num_pages) {
      my_out[out_cnt++] = p_idx;
    }
  }

  // 3. Store Length
  out_lengths[req_idx] = out_cnt;

  // 4. Pad rest with -1
  for (int64_t i = out_cnt; i < max_out; ++i) {
    my_out[i] = -1;
  }

  // 5. Sort the output indices (ascending) for this request
  for (int64_t i = 0; i < out_cnt; ++i) {
    for (int64_t j = 0; j < out_cnt - 1 - i; ++j) {
      if (my_out[j] > my_out[j + 1]) {
        int32_t tmp = my_out[j];
        my_out[j] = my_out[j + 1];
        my_out[j + 1] = tmp;
      }
    }
  }
}

void quest_retrieval_score_and_combine_indices(
    int64_t bs,
    torch::Tensor seq_lens,
    int64_t page_size,
    torch::Tensor req_to_token,
    torch::Tensor page_k_min,
    torch::Tensor page_k_max,
    torch::Tensor queries,
    torch::Tensor req_pool_indices,
    int64_t num_recent_pages,
    std::optional<int64_t> fixed_topk_page_cnt,
    double sparsity_ratio,
    torch::Tensor sparse_mask,
    torch::Tensor out_indices,
    torch::Tensor out_lengths) {
  auto device = queries.device();

  // Calculate max_pages
  int32_t max_seq_len = torch::max(seq_lens).item<int32_t>();
  int64_t max_pages = (max_seq_len + page_size - 1) / page_size;

  if (max_pages == 0) {
    // Fill outputs with defaults
    out_indices.fill_(-1);
    out_lengths.zero_();
    return;
  }

  // Allocate temp buffers
  auto scores = torch::empty({bs, max_pages}, torch::dtype(torch::kFloat32).device(device));
  auto indices = torch::empty({bs, max_pages}, torch::dtype(torch::kInt32).device(device));

  // Handle Queries Shape
  int64_t head_dim = page_k_min.size(2);
  int64_t kv_heads = page_k_min.size(1);
  int64_t q_heads;

  torch::Tensor q_view = queries;
  if (queries.dim() == 2) {
    int64_t hidden_dim = queries.size(1);
    TORCH_CHECK(hidden_dim % head_dim == 0, "Query hidden dim must be divisible by head_dim");
    q_heads = hidden_dim / head_dim;
    q_view = queries.view({bs, q_heads, head_dim});
  } else {
    q_heads = queries.size(1);
    TORCH_CHECK(queries.size(2) == head_dim, "Query head_dim mismatch");
  }

  dim3 block(256);
  int warps_per_block = block.x / 32;
  int pages_per_block = warps_per_block;
  dim3 grid(bs, (max_pages + pages_per_block - 1) / pages_per_block);

  size_t shared_mem_size = kv_heads * head_dim * sizeof(float);

  // Dispatch
  DISPATCH_FLOAT_TYPES(queries.scalar_type(), "quest_score_kernel_opt", [&] {
    quest_score_kernel_opt<scalar_t><<<grid, block, shared_mem_size>>>(
        scores.data_ptr<float>(),
        indices.data_ptr<int32_t>(),
        seq_lens.data_ptr<int32_t>(),
        req_to_token.data_ptr<int32_t>(),
        page_k_min.data_ptr<scalar_t>(),
        page_k_max.data_ptr<scalar_t>(),
        q_view.data_ptr<scalar_t>(),
        req_pool_indices.data_ptr<int32_t>(),
        num_recent_pages,
        max_pages,
        page_size,
        kv_heads,
        q_heads,
        head_dim,
        req_to_token.stride(0),
        req_to_token.stride(1),
        req_to_token.size(1),
        page_k_min.stride(0),
        page_k_min.stride(1),
        page_k_min.stride(2),
        q_view.stride(0),
        q_view.stride(1),
        q_view.stride(2));
  });

  CHECK_CUDA_SUCCESS(cudaGetLastError());

  // Sort Pairs
  auto offsets = torch::arange(0, (bs + 1) * max_pages, max_pages, torch::dtype(torch::kInt32).device(device));

  // CUB Sort
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;

  // 1. Determine temp storage size
  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      d_temp_storage,
      temp_storage_bytes,
      scores.data_ptr<float>(),
      scores.data_ptr<float>(),
      indices.data_ptr<int32_t>(),
      indices.data_ptr<int32_t>(),
      bs * max_pages,
      bs,
      offsets.data_ptr<int32_t>(),
      offsets.data_ptr<int32_t>() + 1);

  auto temp_storage = torch::empty({(int64_t)temp_storage_bytes}, torch::dtype(torch::kByte).device(device));
  d_temp_storage = temp_storage.data_ptr();

  // 2. Run Sort
  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      d_temp_storage,
      temp_storage_bytes,
      scores.data_ptr<float>(),
      scores.data_ptr<float>(),
      indices.data_ptr<int32_t>(),
      indices.data_ptr<int32_t>(),
      bs * max_pages,
      bs,
      offsets.data_ptr<int32_t>(),
      offsets.data_ptr<int32_t>() + 1);

  // Determine Output Size
  int64_t max_out = out_indices.size(1);

  // Initialize outputs
  out_indices.fill_(-1);
  out_lengths.zero_();

  // Combine Kernel
  quest_combine_kernel<<<bs, 128>>>(
      indices.data_ptr<int32_t>(),
      out_indices.data_ptr<int32_t>(),
      out_lengths.data_ptr<int32_t>(),
      seq_lens.data_ptr<int32_t>(),
      num_recent_pages,
      max_pages,
      max_out,
      page_size,
      fixed_topk_page_cnt.has_value(),
      fixed_topk_page_cnt.has_value() ? fixed_topk_page_cnt.value() : 0,
      sparsity_ratio,
      sparse_mask.defined() ? sparse_mask.data_ptr<int32_t>() : nullptr);

  CHECK_CUDA_SUCCESS(cudaGetLastError());
}

__global__ void quest_update_page_table_kernel(
    int32_t* __restrict__ page_table,            // [bs, pt_stride]
    const int32_t* __restrict__ physical_pages,  // [bs, max_selected]
    const int32_t* __restrict__ valid_lengths,   // [bs]
    const int32_t* __restrict__ sparse_mask,     // [bs]
    int64_t max_selected,
    int64_t pt_stride) {
  int64_t req_idx = blockIdx.x;
  if (sparse_mask[req_idx] == 0) return;

  int32_t valid_len = valid_lengths[req_idx];

  for (int64_t i = threadIdx.x; i < max_selected; i += blockDim.x) {
    if (i < valid_len) {
      page_table[req_idx * pt_stride + i] = physical_pages[req_idx * max_selected + i];
    }
  }
}

__global__ void quest_compute_sparse_seqlens_kernel(
    int32_t* __restrict__ current_cache_seqlens,         // [bs]
    const int32_t* __restrict__ seq_lens,                // [bs]
    const int32_t* __restrict__ valid_lengths,           // [bs]
    const int32_t* __restrict__ sparse_mask,             // [bs]
    const int32_t* __restrict__ original_cache_seqlens,  // [bs]
    int page_size,
    int64_t bs) {
  int64_t req_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (req_idx >= bs) return;

  if (sparse_mask[req_idx]) {
    int32_t sl = seq_lens[req_idx];
    // Python: positions_in_page = (seq_lens - 1) % page_size
    int32_t positions_in_page = (sl - 1) % page_size;
    // Python: diff = page_size - positions_in_page - 1
    int32_t diff = page_size - positions_in_page - 1;
    int32_t vl = valid_lengths[req_idx];
    // Python: sparse_seq_lens = (valid_lengths * page_size - diff)
    current_cache_seqlens[req_idx] = vl * page_size - diff;
  } else {
    current_cache_seqlens[req_idx] = original_cache_seqlens[req_idx];
  }
}

void update_sparse_metadata(
    torch::Tensor page_table,
    torch::Tensor physical_pages,
    torch::Tensor valid_lengths,
    torch::Tensor sparse_mask,
    torch::Tensor cache_seqlens,
    torch::Tensor seq_lens,
    torch::Tensor original_cache_seqlens,
    int64_t page_size) {
  TORCH_CHECK(page_table.is_cuda(), "page_table must be on CUDA");
  TORCH_CHECK(physical_pages.is_cuda(), "physical_pages must be on CUDA");
  TORCH_CHECK(valid_lengths.is_cuda(), "valid_lengths must be on CUDA");
  TORCH_CHECK(sparse_mask.is_cuda(), "sparse_mask must be on CUDA");
  TORCH_CHECK(cache_seqlens.is_cuda(), "cache_seqlens must be on CUDA");

  TORCH_CHECK(page_table.scalar_type() == torch::kInt32, "page_table must be int32");
  TORCH_CHECK(physical_pages.scalar_type() == torch::kInt32, "physical_pages must be int32");
  TORCH_CHECK(valid_lengths.scalar_type() == torch::kInt32, "valid_lengths must be int32");
  TORCH_CHECK(sparse_mask.scalar_type() == torch::kInt32, "sparse_mask must be int32");

  TORCH_CHECK(physical_pages.is_contiguous(), "physical_pages must be contiguous");
  TORCH_CHECK(valid_lengths.is_contiguous(), "valid_lengths must be contiguous");
  TORCH_CHECK(sparse_mask.is_contiguous(), "sparse_mask must be contiguous");

  int64_t bs = page_table.size(0);
  int64_t pt_stride = page_table.stride(0);
  int64_t max_selected = physical_pages.size(1);

  auto device = page_table.device();

  // Update Page Table
  // Grid: bs blocks
  // Block: 128 threads
  quest_update_page_table_kernel<<<bs, 128>>>(
      page_table.data_ptr<int32_t>(),
      physical_pages.data_ptr<int32_t>(),
      valid_lengths.data_ptr<int32_t>(),
      sparse_mask.data_ptr<int32_t>(),
      max_selected,
      pt_stride);
  CHECK_CUDA_SUCCESS(cudaGetLastError());

  // Compute Sparse Seqlens
  dim3 block(256);
  dim3 grid((bs + block.x - 1) / block.x);

  quest_compute_sparse_seqlens_kernel<<<grid, block>>>(
      cache_seqlens.data_ptr<int32_t>(),
      seq_lens.data_ptr<int32_t>(),
      valid_lengths.data_ptr<int32_t>(),
      sparse_mask.data_ptr<int32_t>(),
      original_cache_seqlens.data_ptr<int32_t>(),
      page_size,
      bs);
  CHECK_CUDA_SUCCESS(cudaGetLastError());
}

__global__ void sparse_diff_kernel(
    const int32_t* __restrict__ curr_top_k,       // [bs, top_k]
    const int32_t* __restrict__ req_pool_indices, // [bs]
    const int32_t* __restrict__ valid_lengths,    // [bs]
    const int32_t* __restrict__ seq_lens,         // [bs]
    const int32_t* __restrict__ sparse_mask,      // [bs]
    const int64_t* __restrict__ req_to_tokens_host, // [num_reqs, max_tokens_host]
    
    int64_t* __restrict__ last_top_k,             // [num_reqs, num_layers, hot_buffer_len]
    int64_t* __restrict__ last_page_ids,          // [num_reqs, num_layers, hot_buffer_len]
    int32_t* __restrict__ page_table,             // [bs, pt_stride]
    int32_t* __restrict__ physical_pages,         // [bs, top_k]
    int64_t* __restrict__ load_tokens,            // [bs, top_k * page_size]
    int64_t* __restrict__ load_tokens_host,       // [bs, top_k * page_size]
    
    int64_t bs,
    int64_t layer_id,
    int64_t hot_buffer_len,
    int64_t top_k,
    int64_t page_size,
    int64_t pt_stride,
    int64_t pt_cols,
    int64_t last_stride_req,
    int64_t last_stride_layer,
    int64_t req_to_tokens_stride,
    int64_t load_tokens_stride,
    int64_t physical_pages_stride
) {
    extern __shared__ int64_t s_mem[];
    int64_t* s_last_top_k = s_mem;
    int64_t* s_last_page_ids = s_last_top_k + hot_buffer_len;
    int64_t* s_curr_top_k = s_last_page_ids + hot_buffer_len;
    int64_t* s_curr_page_ids = s_curr_top_k + top_k;
    int64_t* s_load_mask = s_curr_page_ids + top_k;
    
    // Shared counters for parallel compaction
    __shared__ int s_victim_count;
    __shared__ int s_fill_count;
    using BlockScan = cub::BlockScan<int, 128>;
    __shared__ typename BlockScan::TempStorage scan_storage;

    int tid = threadIdx.x;
    int req_idx_in_batch = blockIdx.x;
    
    if (req_idx_in_batch >= bs) return;
    
    int req_idx = req_pool_indices[req_idx_in_batch];
    
    // Load last_top_k and last_page_ids (int64 -> int64)
    for (int i = tid; i < hot_buffer_len; i += blockDim.x) {
        int64_t offset = (int64_t)req_idx * last_stride_req + layer_id * last_stride_layer + i;
        s_last_top_k[i] = last_top_k[offset];
        s_last_page_ids[i] = last_page_ids[offset];
    }
    
    // Load curr_top_k and init (int32 -> int64)
    for (int i = tid; i < top_k; i += blockDim.x) {
        s_curr_top_k[i] = (int64_t)curr_top_k[req_idx_in_batch * top_k + i];
        s_curr_page_ids[i] = -1;
        s_load_mask[i] = 0;
    }
    
    __syncthreads();

    int32_t seq_len = seq_lens[req_idx_in_batch];
    for (int i = tid; i < top_k * page_size; i += blockDim.x) {
        int64_t out_idx = (int64_t)req_idx_in_batch * load_tokens_stride + i;
        load_tokens[out_idx] = -1;
        load_tokens_host[out_idx] = -1;
    }
    
    if (!sparse_mask[req_idx_in_batch] || seq_len <= 0) {
        int valid_len = valid_lengths[req_idx_in_batch];
        for (int i = tid; i < top_k; i += blockDim.x) {
            int64_t log_page = s_curr_top_k[i];
            if (i < valid_len && log_page >= 0 && log_page < pt_cols) {
                s_curr_page_ids[i] = (int64_t)page_table[req_idx_in_batch * pt_stride + log_page];
            } else {
                s_curr_page_ids[i] = -1;
            }
            s_load_mask[i] = 0;
        }
        __syncthreads();
    } else {
        // Intersection
        for (int i = tid; i < top_k; i += blockDim.x) {
            int64_t val = s_curr_top_k[i];
            if (val != -1) {
                for (int j = 0; j < hot_buffer_len; ++j) {
                    if (s_last_top_k[j] == val) {
                        s_curr_page_ids[i] = s_last_page_ids[j];
                        s_last_page_ids[j] = -1; // Mark used
                        break;
                    }
                }
            }
        }
        
        __syncthreads();

        int64_t local_top_k = -1;
        int64_t local_page_id = -1;
        int keep = 0;
        if (tid < hot_buffer_len) {
            local_top_k = s_last_top_k[tid];
            local_page_id = s_last_page_ids[tid];
            keep = (local_page_id != -1);
        }

        int prefix = 0;
        int remaining = 0;
        BlockScan(scan_storage).ExclusiveSum(keep, prefix, remaining);
        if (keep) {
            s_last_top_k[prefix] = local_top_k;
            s_last_page_ids[prefix] = local_page_id;
        }
        if (tid == 0) {
            s_victim_count = remaining;
            s_fill_count = 0;
        }
        __syncthreads();

        if (tid == 0) {
            int valid_len = valid_lengths[req_idx_in_batch];
            int fill_needed = 0;
            for (int i = 0; i < top_k; ++i) {
                if (i >= valid_len || s_curr_top_k[i] == -1) {
                    continue;
                }
                if (s_curr_page_ids[i] == -1) {
                    ++fill_needed;
                }
            }

            int used = 0;
            int remaining_after_used = (int)s_victim_count - fill_needed;
            if (remaining_after_used < 0) remaining_after_used = 0;

            for (int i = 0; i < top_k; ++i) {
                if (i >= valid_len || s_curr_top_k[i] == -1) {
                    s_curr_top_k[i] = -1;
                    s_curr_page_ids[i] = -1;
                    s_load_mask[i] = 0;
                    continue;
                }

                if (s_curr_page_ids[i] == -1) {
                    int victim_idx = remaining_after_used + used;
                    if (victim_idx < s_victim_count) {
                        s_curr_page_ids[i] = s_last_page_ids[victim_idx];
                        s_load_mask[i] = 1;
                        ++used;
                    } else {
                        s_curr_page_ids[i] = -1;
                        s_load_mask[i] = 0;
                    }
                } else {
                    s_load_mask[i] = 0;
                }
            }
            s_fill_count = used;
        }
        __syncthreads();

        for (int i = tid; i < top_k * page_size; i += blockDim.x) {
            int page_idx = i / page_size;
            int token_offset = i % page_size;
            int64_t out_idx = req_idx_in_batch * load_tokens_stride + i;
            
            if (page_idx < top_k && s_load_mask[page_idx]) {
                int64_t log_page = s_curr_top_k[page_idx];
                if (log_page != -1 && log_page * page_size + token_offset < seq_len) {
                    int64_t host_offset =
                        (int64_t)req_idx * req_to_tokens_stride + log_page * page_size + token_offset;
                    int64_t host_token = req_to_tokens_host[host_offset];
                    if (host_token != -1) {
                        int64_t phys_page = s_curr_page_ids[page_idx];
                        load_tokens[out_idx] = phys_page * page_size + token_offset;
                        load_tokens_host[out_idx] = host_token;
                    }
                }
            }
        }

        // Update State
        int victims_used = s_fill_count;
        int valid_len = valid_lengths[req_idx_in_batch];
        int remaining_after_used = s_victim_count - victims_used;
        if (remaining_after_used < 0) remaining_after_used = 0;
        for (int i = tid; i < hot_buffer_len; i += blockDim.x) {
            int64_t offset = (int64_t)req_idx * last_stride_req + layer_id * last_stride_layer + i;
            if (i < top_k) {
                if (i < valid_len && s_curr_top_k[i] != -1) {
                    last_top_k[offset] = s_curr_top_k[i];
                    last_page_ids[offset] = s_curr_page_ids[i];
                } else {
                    last_top_k[offset] = -1;
                    last_page_ids[offset] = -1;
                }
            } else {
                int src = (i - top_k);
                if (src < remaining_after_used) {
                    last_top_k[offset] = s_last_top_k[src];
                    last_page_ids[offset] = s_last_page_ids[src];
                } else {
                    last_top_k[offset] = -1;
                    last_page_ids[offset] = -1;
                }
            }
        }
    }

    int valid_len = valid_lengths[req_idx_in_batch];
    for (int i = tid; i < top_k; i += blockDim.x) {
        int64_t out_idx = req_idx_in_batch * physical_pages_stride + i;
        if (sparse_mask[req_idx_in_batch] && i < valid_len) {
            physical_pages[out_idx] = (int32_t)s_curr_page_ids[i];
        } else {
            physical_pages[out_idx] = -1;
        }
    }
}

void invoke_sparse_diff_cuda_kernel(
    torch::Tensor page_table,
    torch::Tensor last_top_k,
    torch::Tensor last_page_ids,
    torch::Tensor curr_top_k,
    torch::Tensor req_pool_indices,
    torch::Tensor seq_lens,
    torch::Tensor valid_lengths,
    torch::Tensor sparse_mask,
    torch::Tensor req_to_tokens_host,
    torch::Tensor physical_pages,
    torch::Tensor load_tokens,
    torch::Tensor load_tokens_host,
    torch::Tensor cache_seqlens,
    torch::Tensor original_cache_seqlens,
    int64_t layer_id,
    int64_t page_size
) {
    TORCH_CHECK(page_table.is_cuda(), "page_table must be on CUDA");
    TORCH_CHECK(last_top_k.is_cuda(), "last_top_k must be on CUDA");
    TORCH_CHECK(last_page_ids.is_cuda(), "last_page_ids must be on CUDA");
    TORCH_CHECK(curr_top_k.is_cuda(), "curr_top_k must be on CUDA");
    TORCH_CHECK(req_to_tokens_host.is_cuda(), "req_to_tokens_host must be on CUDA");
    TORCH_CHECK(physical_pages.is_cuda(), "physical_pages must be on CUDA");
    TORCH_CHECK(load_tokens.is_cuda(), "load_tokens must be on CUDA");
    TORCH_CHECK(load_tokens_host.is_cuda(), "load_tokens_host must be on CUDA");
    
    TORCH_CHECK(page_table.scalar_type() == torch::kInt32, "page_table must be int32");
    TORCH_CHECK(curr_top_k.scalar_type() == torch::kInt32, "curr_top_k must be int32");
    TORCH_CHECK(physical_pages.scalar_type() == torch::kInt32, "physical_pages must be int32");
    TORCH_CHECK(req_pool_indices.scalar_type() == torch::kInt32, "req_pool_indices must be int32");
    TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32, "seq_lens must be int32");
    TORCH_CHECK(valid_lengths.scalar_type() == torch::kInt32, "valid_lengths must be int32");
    TORCH_CHECK(sparse_mask.scalar_type() == torch::kInt32, "sparse_mask must be int32");
    
    TORCH_CHECK(last_top_k.scalar_type() == torch::kInt64, "last_top_k must be int64");
    TORCH_CHECK(last_page_ids.scalar_type() == torch::kInt64, "last_page_ids must be int64");
    TORCH_CHECK(req_to_tokens_host.scalar_type() == torch::kInt64, "req_to_tokens_host must be int64");
    TORCH_CHECK(load_tokens.scalar_type() == torch::kInt64, "load_tokens must be int64");
    TORCH_CHECK(load_tokens_host.scalar_type() == torch::kInt64, "load_tokens_host must be int64");
    TORCH_CHECK(page_table.is_contiguous(), "page_table must be contiguous");
    TORCH_CHECK(curr_top_k.is_contiguous(), "curr_top_k must be contiguous");
    TORCH_CHECK(req_pool_indices.is_contiguous(), "req_pool_indices must be contiguous");
    TORCH_CHECK(seq_lens.is_contiguous(), "seq_lens must be contiguous");
    TORCH_CHECK(valid_lengths.is_contiguous(), "valid_lengths must be contiguous");
    TORCH_CHECK(sparse_mask.is_contiguous(), "sparse_mask must be contiguous");
    TORCH_CHECK(req_to_tokens_host.is_contiguous(), "req_to_tokens_host must be contiguous");
    TORCH_CHECK(load_tokens.is_contiguous(), "load_tokens must be contiguous");
    TORCH_CHECK(load_tokens_host.is_contiguous(), "load_tokens_host must be contiguous");

    int64_t bs = page_table.size(0);
    int64_t pt_stride = page_table.stride(0);
    int64_t pt_cols = page_table.size(1);
    int64_t top_k = curr_top_k.size(1);
    int64_t hot_buffer_len = last_top_k.size(2);

    TORCH_CHECK(hot_buffer_len >= top_k, "hot_buffer_len must be >= top_k");
    TORCH_CHECK(physical_pages.size(0) == bs, "physical_pages batch mismatch");
    TORCH_CHECK(physical_pages.size(1) == top_k, "physical_pages second dim must equal top_k");
    TORCH_CHECK(load_tokens.size(0) >= bs, "load_tokens batch must be >= bs");
    TORCH_CHECK(load_tokens_host.size(0) >= bs, "load_tokens_host batch must be >= bs");
    TORCH_CHECK(load_tokens.size(1) >= top_k * page_size, "load_tokens second dim too small");
    TORCH_CHECK(load_tokens_host.size(1) >= top_k * page_size, "load_tokens_host second dim too small");
    
    size_t shared_mem = (2 * hot_buffer_len + 3 * top_k) * sizeof(int64_t);
    
    sparse_diff_kernel<<<bs, 128, shared_mem>>>(
        curr_top_k.data_ptr<int32_t>(),
        req_pool_indices.data_ptr<int32_t>(),
        valid_lengths.data_ptr<int32_t>(),
        seq_lens.data_ptr<int32_t>(),
        sparse_mask.data_ptr<int32_t>(),
        req_to_tokens_host.data_ptr<int64_t>(),
        
        last_top_k.data_ptr<int64_t>(),
        last_page_ids.data_ptr<int64_t>(),
        page_table.data_ptr<int32_t>(),
        physical_pages.data_ptr<int32_t>(),
        load_tokens.data_ptr<int64_t>(),
        load_tokens_host.data_ptr<int64_t>(),
        
        bs,
        layer_id,
        hot_buffer_len,
        top_k,
        page_size,
        pt_stride,
        pt_cols,
        last_top_k.stride(0),
        last_top_k.stride(1),
        req_to_tokens_host.stride(0),
        load_tokens.stride(0),
        physical_pages.stride(0)
    );
    CHECK_CUDA_SUCCESS(cudaGetLastError());

}
