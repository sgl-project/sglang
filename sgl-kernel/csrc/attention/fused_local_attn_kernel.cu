#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>
#include <algorithm>
#include <limits>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

inline int cdiv_host(int x, int y) {
    return (x + y - 1) / y;
}

inline int min_host(int a, int b) {
    return a < b ? a : b;
}

inline int max_host(int a, int b) {
    return a > b ? a : b;
}

__device__ inline int cdiv(int x, int y) {
    return (x + y - 1) / y;
}

__device__ inline int min_func(int a, int b) {
    return a < b ? a : b;
}

__device__ inline int max_func(int a, int b) {
    return a > b ? a : b;
}

__global__ void compute_all_basics_kernel(
    const int* query_start_loc,
    const int* seq_lens,
    int batch_size,
    int attn_chunk_size,
    int page_size,
    
    int* q_seqlens,
    int* q_tokens_in_first_block,
    int* tokens_in_last_block,
    int* local_blocks,
    int* cu_num_blocks_temp
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    int q_len = query_start_loc[batch_idx + 1] - query_start_loc[batch_idx];
    q_seqlens[batch_idx] = q_len;
    
    int seq_len = seq_lens[batch_idx];
    int remainder = (seq_len - q_len) % attn_chunk_size;
    int first_block_tokens = min_func(attn_chunk_size - remainder, q_len);
    q_tokens_in_first_block[batch_idx] = first_block_tokens;
    
    // Match Python's modulo behavior: a % -b in Python has same sign as b (negative)
    // In C++, we need to compute it differently to match Python
    int mod_result = seq_len % attn_chunk_size;  // This is always positive
    if (mod_result == 0) {
        tokens_in_last_block[batch_idx] = attn_chunk_size;
    } else {
        tokens_in_last_block[batch_idx] = mod_result;
    }
    
    int remaining_q = q_len - first_block_tokens;
    int blocks = 1 + cdiv(remaining_q, attn_chunk_size);
    local_blocks[batch_idx] = blocks;
    cu_num_blocks_temp[batch_idx] = blocks;
}

__global__ void generate_virtual_batches_and_seqlens_kernel(
    const int* local_blocks,
    const int* cu_num_blocks,
    const int* q_seqlens,
    const int* q_tokens_in_first_block,
    int batch_size,
    int attn_chunk_size,
    int virtual_batches,
    
    int* arange,
    int* rarange,
    int* batch_indices_per_block,
    int* seqlens_q_local
) {
    int virt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (virt_idx >= virtual_batches) return;
    
    int batch_idx = 0;
    int blocks_before = 0;
    
    for (int b = 0; b < batch_size; b++) {
        if (cu_num_blocks[b + 1] > virt_idx) {
            batch_idx = b;
            blocks_before = cu_num_blocks[b];
            break;
        }
    }
    
    int local_idx = virt_idx - blocks_before;
    int num_blocks = local_blocks[batch_idx];
    
    arange[virt_idx] = local_idx;
    rarange[virt_idx] = num_blocks - local_idx - 1;
    batch_indices_per_block[virt_idx] = batch_idx;
    
    if (local_idx == 0) {
        seqlens_q_local[virt_idx] = q_tokens_in_first_block[batch_idx];
    } else {
        int q_len = q_seqlens[batch_idx];
        int first_block_tokens = q_tokens_in_first_block[batch_idx];
        int remaining = q_len - first_block_tokens;
        int block_q_len = remaining - attn_chunk_size * (local_idx - 1);
        block_q_len = min_func(block_q_len, attn_chunk_size);
        seqlens_q_local[virt_idx] = max_func(block_q_len, 0);
    }
}

__global__ void compute_k_metadata_kernel(
    const int* tokens_in_last_block,
    const int* batch_indices_per_block,
    const int* seq_lens,
    const int* local_blocks,
    const int* arange,
    const int* rarange,
    int attn_chunk_size,
    int page_size,
    int virtual_batches,
    
    int* seqlens_k_local,
    int* block_starts
) {
    int virt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (virt_idx >= virtual_batches) return;
    
    int batch_idx = batch_indices_per_block[virt_idx];
    int local_idx = arange[virt_idx];
    int num_blocks = local_blocks[batch_idx];
    
    if (local_idx == num_blocks - 1) {
        seqlens_k_local[virt_idx] = tokens_in_last_block[batch_idx];
    } else {
        seqlens_k_local[virt_idx] = attn_chunk_size;
    }
    
    int seq_len = seq_lens[batch_idx];
    int k_seqstart_absolute = seq_len - (
        rarange[virt_idx] * attn_chunk_size + tokens_in_last_block[batch_idx]
    );
    block_starts[virt_idx] = k_seqstart_absolute / page_size;
}

__global__ void build_block_table_local_kernel(
    const int* block_table,
    const int* block_starts,
    const int* batch_indices_per_block,
    int batch_size,
    int kvlen,
    int attn_chunk_size,
    int page_size,
    int virtual_batches,
    
    int* block_table_local
) {
    int virt_idx = blockIdx.x;
    int page_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (virt_idx >= virtual_batches) return;
    
    int pages_per_local_batch = attn_chunk_size / page_size;
    if (page_idx >= pages_per_local_batch) return;
    
    int batch_idx = batch_indices_per_block[virt_idx];
    int block_start = block_starts[virt_idx];
    int block_idx = block_start + page_idx;
    
    block_idx = min_func(block_idx, kvlen - 1);
    block_idx = max_func(block_idx, 0);
    
    int block_table_entry = block_table[batch_idx * kvlen + block_idx];
    block_table_local[virt_idx * pages_per_local_batch + page_idx] = block_table_entry;
}

__global__ void fused_copy_fill_metadata_kernel(
    const int* seqlens_q_local,
    const int* cu_seqlens_q_local,
    const int* seqlens_k_local,
    const int* block_table_local,
    
    int* local_q_buf,
    int* local_k_buf,
    int* local_block_buf,
    
    int virtual_batches,
    int pages_per_local_batch,
    int q_capacity,
    int k_capacity,
    int block_rows,
    int block_cols,
    
    int* out_metadata
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ int shared_max_q;
    __shared__ int shared_max_k;
    __shared__ int shared_q_len;
    __shared__ int shared_k_len;
    
    if (threadIdx.x == 0) {
        shared_max_q = 0;
        shared_max_k = 0;
        shared_q_len = cu_seqlens_q_local[virtual_batches];
        shared_k_len = virtual_batches;
    }
    __syncthreads();
    
    int q_len = shared_q_len;
    int k_len = shared_k_len;
    int b0 = virtual_batches;
    int b1 = pages_per_local_batch;
    
    if (idx < q_len && idx < q_capacity) {
        local_q_buf[idx] = cu_seqlens_q_local[idx];
    }
    
    if (idx >= q_len && idx < q_capacity) {
        local_q_buf[idx] = 0;
    }
    
    if (idx < k_len && idx < k_capacity) {
        local_k_buf[idx] = seqlens_k_local[idx];
    }
    
    if (idx >= k_len && idx < k_capacity) {
        local_k_buf[idx] = 0;
    }
    
    int total_block_elements = b0 * b1;
    if (idx < total_block_elements) {
        int row = idx / b1;
        int col = idx % b1;
        int src_idx = row * pages_per_local_batch + col;
        local_block_buf[row * block_cols + col] = block_table_local[src_idx];
    }
    
    int total_block_capacity = block_rows * block_cols;
    if (idx >= total_block_elements && idx < total_block_capacity) {
        local_block_buf[idx] = 0;
    }
    __syncthreads();
    
    int max_q = 0;
    for (int i = idx; i < virtual_batches; i += blockDim.x * gridDim.x) {
        int seq_q = cu_seqlens_q_local[i + 1] - cu_seqlens_q_local[i];
        max_q = max_func(max_q, seq_q);
    }
    atomicMax(&shared_max_q, max_q);
    
    int max_k = 0;
    for (int i = idx; i < virtual_batches; i += blockDim.x * gridDim.x) {
        max_k = max_func(max_k, seqlens_k_local[i]);
    }
    atomicMax(&shared_max_k, max_k);
    
    __syncthreads();
    
    if (idx == 0) {
        out_metadata[0] = q_len;
        out_metadata[1] = k_len;
        out_metadata[2] = b0;
        out_metadata[3] = b1;
        out_metadata[4] = shared_max_q;
        out_metadata[5] = shared_max_k;
    }
}

std::vector<torch::Tensor> make_local_attention_virtual_batches_fully_fused(
    int attn_chunk_size,
    torch::Tensor query_start_loc,
    torch::Tensor seq_lens,
    torch::Tensor block_table,
    int page_size,
    torch::Tensor local_q_buf,
    torch::Tensor local_k_buf,
    torch::Tensor local_block_buf
) {
    auto device = block_table.device();
    int device_index = device.index();
    c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream(device_index);
    cudaStream_t cuda_stream = stream.stream();
    
    int batch_size = seq_lens.size(0);
    int kvlen = block_table.size(1);

    int max_seq_len = seq_lens.max().item<int>();
    int effective_chunk_size = min_host(attn_chunk_size, max_seq_len);
    effective_chunk_size = (effective_chunk_size / page_size) * page_size;
    if (effective_chunk_size < page_size) {
        effective_chunk_size = page_size;
    }
    attn_chunk_size = effective_chunk_size;
    
    auto q_seqlens = torch::zeros({batch_size}, torch::TensorOptions()
        .dtype(torch::kInt32).device(device));
    auto q_tokens_in_first_block = torch::zeros({batch_size}, torch::TensorOptions()
        .dtype(torch::kInt32).device(device));
    auto tokens_in_last_block = torch::zeros({batch_size}, torch::TensorOptions()
        .dtype(torch::kInt32).device(device));
    auto local_blocks = torch::zeros({batch_size}, torch::TensorOptions()
        .dtype(torch::kInt32).device(device));
    auto cu_num_blocks_temp = torch::zeros({batch_size}, torch::TensorOptions()
        .dtype(torch::kInt32).device(device));
    
    {
        int threads = 256;
        int blocks = (batch_size + threads - 1) / threads;
        compute_all_basics_kernel<<<blocks, threads, 0, cuda_stream>>>(
            query_start_loc.data_ptr<int>(),
            seq_lens.data_ptr<int>(),
            batch_size,
            attn_chunk_size,
            page_size,
            q_seqlens.data_ptr<int>(),
            q_tokens_in_first_block.data_ptr<int>(),
            tokens_in_last_block.data_ptr<int>(),
            local_blocks.data_ptr<int>(),
            cu_num_blocks_temp.data_ptr<int>()
        );
    }
    // Removed unnecessary sync - CUB will queue on same stream
    
    auto cu_num_blocks = torch::zeros({batch_size + 1}, torch::TensorOptions()
        .dtype(torch::kInt32).device(device));
    
    {
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(
            nullptr, temp_storage_bytes,
            cu_num_blocks_temp.data_ptr<int>(),
            cu_num_blocks.data_ptr<int>() + 1,
            batch_size, cuda_stream
        );
        auto temp_storage = torch::empty({(int)temp_storage_bytes}, torch::TensorOptions()
            .dtype(torch::kUInt8).device(device));
        cub::DeviceScan::InclusiveSum(
            temp_storage.data_ptr(), temp_storage_bytes,
            cu_num_blocks_temp.data_ptr<int>(),
            cu_num_blocks.data_ptr<int>() + 1,
            batch_size, cuda_stream
        );
    }
    // Sync required here - CPU needs to read virtual_batches value
    cudaStreamSynchronize(cuda_stream);

    int virtual_batches = cu_num_blocks[-1].item<int>();
    
    if (virtual_batches == 0) {
        return {
            torch::zeros({0}, torch::TensorOptions().dtype(torch::kInt32).device(device)),
            torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32).device(device)),
            torch::zeros({0}, torch::TensorOptions().dtype(torch::kInt32).device(device)),
            torch::zeros({0, attn_chunk_size / page_size}, torch::TensorOptions()
                .dtype(torch::kInt32).device(device)),
            torch::zeros({6}, torch::TensorOptions().dtype(torch::kInt32).device(device))
        };
    }
    
    auto arange = torch::zeros({virtual_batches}, torch::TensorOptions()
        .dtype(torch::kInt32).device(device));
    auto rarange = torch::zeros({virtual_batches}, torch::TensorOptions()
        .dtype(torch::kInt32).device(device));
    auto batch_indices_per_block = torch::zeros({virtual_batches}, torch::TensorOptions()
        .dtype(torch::kInt32).device(device));
    auto seqlens_q_local = torch::zeros({virtual_batches}, torch::TensorOptions()
        .dtype(torch::kInt32).device(device));
    
    {
        int threads = 256;
        int blocks = (virtual_batches + threads - 1) / threads;
        generate_virtual_batches_and_seqlens_kernel<<<blocks, threads, 0, cuda_stream>>>(
            local_blocks.data_ptr<int>(),
            cu_num_blocks.data_ptr<int>(),
            q_seqlens.data_ptr<int>(),
            q_tokens_in_first_block.data_ptr<int>(),
            batch_size, attn_chunk_size, virtual_batches,
            arange.data_ptr<int>(),
            rarange.data_ptr<int>(),
            batch_indices_per_block.data_ptr<int>(),
            seqlens_q_local.data_ptr<int>()
        );
    }
    // Removed unnecessary sync - next CUB operation will queue automatically
    
    auto cu_seqlens_q_local = torch::zeros({virtual_batches + 1}, torch::TensorOptions()
        .dtype(torch::kInt32).device(device));
    
    {
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(
            nullptr, temp_storage_bytes,
            seqlens_q_local.data_ptr<int>(),
            cu_seqlens_q_local.data_ptr<int>() + 1,
            virtual_batches, cuda_stream
        );
        auto temp_storage = torch::empty({(int)temp_storage_bytes}, torch::TensorOptions()
            .dtype(torch::kUInt8).device(device));
        cub::DeviceScan::InclusiveSum(
            temp_storage.data_ptr(), temp_storage_bytes,
            seqlens_q_local.data_ptr<int>(),
            cu_seqlens_q_local.data_ptr<int>() + 1,
            virtual_batches, cuda_stream
        );
    }
    // Removed unnecessary sync - next kernel will queue automatically
    
    auto seqlens_k_local = torch::full({virtual_batches}, attn_chunk_size,
        torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto block_starts = torch::zeros({virtual_batches}, torch::TensorOptions()
        .dtype(torch::kInt32).device(device));
    
    {
        int threads = 256;
        int blocks = (virtual_batches + threads - 1) / threads;
        compute_k_metadata_kernel<<<blocks, threads, 0, cuda_stream>>>(
            tokens_in_last_block.data_ptr<int>(),
            batch_indices_per_block.data_ptr<int>(),
            seq_lens.data_ptr<int>(),
            local_blocks.data_ptr<int>(),
            arange.data_ptr<int>(),
            rarange.data_ptr<int>(),
            attn_chunk_size, page_size, virtual_batches,
            seqlens_k_local.data_ptr<int>(),
            block_starts.data_ptr<int>()
        );
    }
    // Removed unnecessary sync - next kernel will queue automatically
    
    int pages_per_local_batch = attn_chunk_size / page_size;
    auto block_table_local = torch::zeros({virtual_batches, pages_per_local_batch},
        torch::TensorOptions().dtype(torch::kInt32).device(device));
    
    {
        dim3 grid(virtual_batches, (pages_per_local_batch + 255) / 256);
        dim3 block(256);
        build_block_table_local_kernel<<<grid, block, 0, cuda_stream>>>(
            block_table.data_ptr<int>(),
            block_starts.data_ptr<int>(),
            batch_indices_per_block.data_ptr<int>(),
            batch_size, kvlen,
            attn_chunk_size, page_size, virtual_batches,
            block_table_local.data_ptr<int>()
        );
    }
    // Removed unnecessary sync - next kernel will queue automatically
    
    auto out_metadata = torch::zeros({6}, torch::TensorOptions()
        .dtype(torch::kInt32).device(device));
    
    {
        int threads = 256;
        int blocks = (max_host(local_q_buf.size(0), local_k_buf.size(0)) + threads - 1) / threads;
        
        fused_copy_fill_metadata_kernel<<<blocks, threads, 0, cuda_stream>>>(
            seqlens_q_local.data_ptr<int>(),
            cu_seqlens_q_local.data_ptr<int>(),
            seqlens_k_local.data_ptr<int>(),
            block_table_local.data_ptr<int>(),
            
            local_q_buf.data_ptr<int>(),
            local_k_buf.data_ptr<int>(),
            local_block_buf.data_ptr<int>(),
            
            virtual_batches,
            pages_per_local_batch,
            local_q_buf.size(0),
            local_k_buf.size(0),
            local_block_buf.size(0),
            local_block_buf.size(1),
            
            out_metadata.data_ptr<int>()
        );
    }
    // Final sync removed - PyTorch will handle synchronization when tensors are accessed
    // If synchronous behavior is needed, caller can use torch.cuda.synchronize()
    
    return {
        seqlens_q_local,
        cu_seqlens_q_local,
        seqlens_k_local,
        block_table_local,
        out_metadata
    };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "make_local_attention_virtual_batches_fully_fused",
        &make_local_attention_virtual_batches_fully_fused,
        "Fully fused CUDA kernel with buffer copy, fill, and metadata update",
        py::arg("attn_chunk_size"),
        py::arg("query_start_loc"),
        py::arg("seq_lens"),
        py::arg("block_table"),
        py::arg("page_size"),
        py::arg("local_q_buf"),
        py::arg("local_k_buf"),
        py::arg("local_block_buf")
    );
}