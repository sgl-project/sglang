#include "common.h"
#include "vec.h"

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
                if (global_buffer_offset >= source_cache_loc.numel() || 
                    global_buffer_offset >= target_cache_loc.numel()) {
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
                        if (d < last_page_len && part2_offset + d < source_cache_loc.numel()) {
                            src_cache_ptr[part2_offset + d] = prefix_base_ptr[d];
                        }
                    }

                    // 2. Target: copy from the newly allocated pages in req_to_token
                    // The location is prefix_base + k_id * num_new_pages * page_size
                    int32_t* new_page_ptr = prefix_base_ptr + k_id * num_new_pages * page_size;
                    
                    // Bounds check for new_page_ptr access
                    if (new_page_ptr - req_to_token_base >= pool_len) {
                        continue;
                    }
                    
                    for (int64_t d = 0; d < last_page_len; ++d) {
                        if (d < last_page_len && part2_offset + d < target_cache_loc.numel()) {
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
    RECORD_FUNCTION(
        "sgl-kernel::assign_draft_cache_locs_cpu",
        std::vector<c10::IValue>({req_pool_indices, req_to_token, seq_lens, extend_lens, num_new_pages_per_topk}));

    // Handle optional inputs: if not provided, create a dummy tensor to satisfy the kernel interface.
    // The kernel logic checks `duplicate_cache_len` before using these tensors.
    at::Tensor source_cache_loc = source_cache_loc_opt.has_value() 
        ? *source_cache_loc_opt 
        : at::empty({0}, out_cache_loc.options());

    at::Tensor target_cache_loc = target_cache_loc_opt.has_value() 
        ? *target_cache_loc_opt 
        : at::empty({0}, out_cache_loc.options());

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
        iter_upper
    );

    return out_cache_loc;
}
