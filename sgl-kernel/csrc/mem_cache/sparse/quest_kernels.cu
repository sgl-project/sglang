#include "utils.h"
#include <cub/cub.cuh>
#include "quest_kernels.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Kernel to compute scores and initialize indices for sorting
template<typename T>
__global__ void quest_score_kernel_opt(
    float* __restrict__ scores,           // [bs, max_pages]
    int32_t* __restrict__ indices,        // [bs, max_pages]
    const int32_t* __restrict__ seq_lens, // [bs]
    const int32_t* __restrict__ req_to_token, // [req_pool_size, max_tokens]
    const T* __restrict__ page_k_min,     // [total_pages, kv_heads, head_dim]
    const T* __restrict__ page_k_max,     // [total_pages, kv_heads, head_dim]
    const T* __restrict__ queries,        // [bs, q_heads, head_dim]
    const int32_t* __restrict__ req_pool_indices, // [bs]
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
    int64_t queries_stride_dim
) {
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
            int64_t q_offset = req_idx * queries_stride_req + 
                               q_h * queries_stride_head + 
                               d * queries_stride_dim;
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
    // Coalesced access: threads 0..31 read consecutive addresses
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
// - Selects top K from sorted indices
// - Adds recent pages
// - Sorts the result row (optional, but usually required for attention kernels)
__global__ void quest_combine_kernel(
    const int32_t* __restrict__ sorted_indices, // [bs, max_pages] (sorted by score desc)
    int32_t* __restrict__ out_indices,          // [bs, max_out]
    int32_t* __restrict__ out_lengths,          // [bs]
    const int32_t* __restrict__ seq_lens,
    int64_t num_recent_pages,
    int64_t max_pages,
    int64_t max_out,
    int64_t page_size,
    bool fixed_k_mode,
    int64_t fixed_k_val,
    double sparsity_ratio,
    const int32_t* __restrict__ sparse_mask
) {
    int64_t req_idx = blockIdx.x;
    if (req_idx >= gridDim.x) return;

    int32_t seq_len = seq_lens[req_idx];
    int32_t num_pages = (seq_len + page_size - 1) / page_size;
    
    // Determine K
    int64_t recent_start = num_pages - num_recent_pages;
    if (recent_start < 0) recent_start = 0;
    
    int64_t history_pages = recent_start;
    if (history_pages < 1) history_pages = 1; // Clamp min=1

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
        k_target = 0; // Or handle mask logic. Python: k_per_req * sparse_mask
    }
    
    // Limit by max_pages
    if (k_target > max_pages) k_target = max_pages;
    if (k_target > history_pages) k_target = history_pages; // Redundant but safe

    int64_t out_cnt = 0;
    int32_t* my_out = out_indices + req_idx * max_out;

    // 1. Copy Top K
    // Since sorted_indices are valid indices into the page list (0..max_pages-1),
    // and we masked invalid/recent pages with -inf, they should be at the end.
    // However, we must ensure we don't pick padding or recent pages if they floated up (unlikely with -inf).
    // The indices in sorted_indices are relative page indices (0..num_pages-1).
    
    const int32_t* my_sorted = sorted_indices + req_idx * max_pages;
    
    for (int64_t i = 0; i < k_target; ++i) {
        if (i < max_pages) {
            int32_t p_idx = my_sorted[i];
            // Verify it is not a recent page (shouldn't be due to -inf score)
            // Verify it is within range
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
    // Bubble sort is fine for small K (e.g. < 256)
    // If out_cnt is large, this is slow. But usually out_cnt is small.
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
    torch::Tensor out_lengths) 
{
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
            q_view.stride(2)
        );
    });
    
    CHECK_CUDA_SUCCESS(cudaGetLastError());

    // Sort Pairs (Descending Scores)
    // Create offsets for segmented sort
    auto offsets = torch::arange(0, (bs + 1) * max_pages, max_pages, torch::dtype(torch::kInt32).device(device));
    
    // CUB Sort
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    
    // 1. Determine temp storage size
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        d_temp_storage, temp_storage_bytes,
        scores.data_ptr<float>(), scores.data_ptr<float>(),
        indices.data_ptr<int32_t>(), indices.data_ptr<int32_t>(),
        bs * max_pages, bs,
        offsets.data_ptr<int32_t>(), offsets.data_ptr<int32_t>() + 1
    );
    
    auto temp_storage = torch::empty({(int64_t)temp_storage_bytes}, torch::dtype(torch::kByte).device(device));
    d_temp_storage = temp_storage.data_ptr();
    
    // 2. Run Sort
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        d_temp_storage, temp_storage_bytes,
        scores.data_ptr<float>(), scores.data_ptr<float>(),
        indices.data_ptr<int32_t>(), indices.data_ptr<int32_t>(),
        bs * max_pages, bs,
        offsets.data_ptr<int32_t>(), offsets.data_ptr<int32_t>() + 1
    );
    
    // Determine Output Size
    // Use the shape of out_indices provided by Python
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
        sparse_mask.defined() ? sparse_mask.data_ptr<int32_t>() : nullptr
    );
    
    CHECK_CUDA_SUCCESS(cudaGetLastError());
}
