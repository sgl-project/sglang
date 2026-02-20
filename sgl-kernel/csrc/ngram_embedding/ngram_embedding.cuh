#ifndef EMBEDDING_SCALING_CUH_
#define EMBEDDING_SCALING_CUH_

namespace embedding_scaling {

__global__ void ComputeNGramIdsKernel(
    int batch_size,
    int ne_n,
    int ne_k,
    int* ne_weights,                  // [ne_n-1,ne_k,ne_n]
    int* ne_mods,                     // [ne_n-1,ne_k]
    int* exclusive_ne_embeder_size_sums, // [(ne_n-1)*ne_k]
    int* tokens,                      // [token_num]
    int* exclusive_req_len_sums,      // [batch_size+1]
    int* ne_token_table,              // [max_running_reqs, max_context_len]
    int max_context_len,              // max_context_len
    long* row_indices,                 // [batch_size]
    int* column_starts,               // [batch_size]
    int* n_gram_ids                   // [ne_n-1,ne_k,token_num]
) {
    // 先搞清当前block要处理的是哪个n、哪个k、第几条req
    /**
    以[req0,req1,req2] n=3,k=2为例
    n       k       req_id      blockIdx.x  config_id(指n和k的组合)
    2       1       0           0           0
    2       1       1           1           0
    2       1       2           2           0
    2       2       0           3           1
    2       2       1           4           1
    2       2       2           5           1
    3       1       0           0           2
    3       1       1           1           2
    3       1       2           2           2
    3       2       0           3           3
    3       2       1           4           3
    3       2       2           5           3
    */
    const int req_id = blockIdx.x % batch_size;
    const int config_id = (blockIdx.x - req_id) / batch_size;
    // 这里n和k不是物理含义上的n和k，而是有个偏移 n = real_n - 2; k = real_k - 1
    // 有这个偏移的原因是，n和k将来要作为索引从ne_weights和ne_mods里面取数，算索引的时候还要偏移回去，保证顺序是对的即可
    const int k = config_id % ne_k;
    const int n = (config_id - config_id % ne_k) / ne_k;
    // weights形状为[ne_n-1,ne_k,ne_n]，最后一个维度是token之间的距离，因此只能先算出来base idx
    const int ne_weight_base_idx = n * ne_k * ne_n + k * ne_n;
    // mod形状为[ne_n-1,ne_k]
    const int ne_mod = ne_mods[n * ne_k + k];
    // stride loop
    for (int i = exclusive_req_len_sums[req_id] + threadIdx.x; i < exclusive_req_len_sums[req_id + 1]; i += blockDim.x) {
        uint64_t n_gram_id = 0;
        // 目前在处理当前请求的第几个token
        int current_token_offset = i - exclusive_req_len_sums[req_id];
        // 先计算当前请求在token table中的起始index，在这个index以前的token就是跨请求的token，不参与计算
        int req_token_table_index = row_indices[req_id] * max_context_len;
        // 再计算当前token在token table中的位置
        int current_token_table_index = req_token_table_index + column_starts[req_id] + current_token_offset;
        for (int j = 0; j < n + 2; j++) {
            if (current_token_table_index-j < req_token_table_index) {
                // 非当前请求或者前面没有信息，不用来计算n_gram_id
                break;
            }
            if (ne_token_table[current_token_table_index-j] < 0) {
                // 写入的时候判断过这是个需要忽略的token
                break;
            }
            const uint64_t term = (uint64_t)ne_token_table[current_token_table_index-j] * (uint64_t)ne_weights[ne_weight_base_idx + j];
            n_gram_id += term % ne_mod;
        }
        n_gram_id %= ne_mod;
        n_gram_id += exclusive_ne_embeder_size_sums[n * ne_k + k];
        // [token_num, ne_n-1, ne_k]
        n_gram_ids[i*(ne_n-1)*ne_k + n*ne_k + k] = (int)(n_gram_id);
    }
}

cudaError_t ComputeNGramIds(
    int batch_size,
    int ne_n,
    int ne_k,
    int* ne_weights,                  // [ne_n-1,ne_k,ne_n]
    int* ne_mods,                     // [ne_n-1,ne_k]
    int* exclusive_ne_embeder_size_sums, // [(ne_n-1)*ne_k]
    int* tokens,                      // [token_num]
    int* exclusive_req_len_sums,      // [batch_size+1]
    int* ne_token_table,              // [max_running_reqs, max_context_len]
    int max_context_len,              // max_context_len
    long* row_indices,                 // [batch_size]
    int* column_starts,               // [batch_size]
    int* n_gram_ids,                   // [ne_n-1,ne_k,token_num]
    cudaStream_t stream = 0
) {
    // host代码，启动kernel
    constexpr int BLOCK_THREADS = 256;
    // 计算配置总数 (n取值范围：2~ne_n)
    const int num_configs = (ne_n - 1) * ne_k;
    // 计算总grid size：每个配置处理所有请求
    const int grid_size = num_configs * batch_size;
    // 设置执行配置
    dim3 grid_dim(grid_size);
    dim3 block_dim(BLOCK_THREADS);
    // 启动kernel
    ComputeNGramIdsKernel<<<grid_dim, block_dim, 0, stream>>>(
        batch_size,
        ne_n,
        ne_k,
        ne_weights,
        ne_mods,
        exclusive_ne_embeder_size_sums,
        tokens,
        exclusive_req_len_sums,
        ne_token_table,
        max_context_len,
        row_indices,
        column_starts,
        n_gram_ids
    );
    return cudaGetLastError();
}

__global__ void UpdateTokenTableKernel(
    int batch_size,
    int* tokens,                      // [token_num]
    int* ne_token_table,              // [max_running_reqs, max_context_len]
    int max_context_len,              // max_context_len
    long* row_indices,                // [batch_size]
    int* column_starts,               // [batch_size]
    int* req_lens,                     // [batch_size]
    int ignore_token_num,             // 有多少token需要被ignore
    int* ignore_tokens               // [ignore_token_num]
) {
    /**
     * 每个block处理一个req
     */
    const int req_id = blockIdx.x % batch_size;
    int start = 0;
    int end = 0;
    for (int i = 0; i < req_id; i++) {
        start += req_lens[i];
    }
    end = start + req_lens[req_id];
    // stride loop
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        // 目前在处理当前请求的第几个token
        int current_token_offset = i - start;
        // 先计算当前请求在token table中的起始index，在这个index以前的token就是跨请求的token，不参与计算
        int req_token_table_index = row_indices[req_id] * max_context_len;
        // 再计算当前token在token table中的位置
        int current_token_table_index = req_token_table_index + column_starts[req_id] + current_token_offset;
        ne_token_table[current_token_table_index] = tokens[i];
        for (int j = 0; j < ignore_token_num; j++) {
            if (ignore_tokens[j] == tokens[i]) {
                ne_token_table[current_token_table_index] = -tokens[i];
                break;
            }
        }
    }
}

cudaError_t UpdateTokenTable(
    int batch_size,
    int* tokens,                      // [token_num]
    int* ne_token_table,              // [max_running_reqs, max_context_len]
    int max_context_len,              // max_context_len
    long* row_indices,                // [batch_size]
    int* column_starts,               // [batch_size]
    int* req_lens,                     // [batch_size]
    int ignore_token_num,             // 有多少token需要被ignore
    int* ignore_tokens,               // [ignore_token_num]
    cudaStream_t stream = 0
) {
    if (batch_size <= 0) {
        return cudaSuccess;
    }
    // host代码，启动kernel
    constexpr int BLOCK_THREADS = 256;
    const int grid_size = batch_size;
    // 设置执行配置
    dim3 grid_dim(grid_size);
    dim3 block_dim(BLOCK_THREADS);
    // 启动kernel
    UpdateTokenTableKernel<<<grid_dim, block_dim, 0, stream>>>(
        batch_size,
        tokens,                      // [token_num]
        ne_token_table,              // [max_running_reqs, max_context_len]
        max_context_len,              // max_context_len
        row_indices,                // [batch_size]
        column_starts,               // [batch_size]
        req_lens,                     // [batch_size]
        ignore_token_num,             // 有多少token需要被ignore
        ignore_tokens               // [ignore_token_num]
    );
    return cudaGetLastError();
}

}
// namespace embedding_scaling
#endif  // EMBEDDING_SCALING
