#include "pytorch_extension_utils.h"
#include "ngram_embedding.cuh"

using namespace embedding_scaling;

void compute_n_gram_ids(
    int64_t ne_n,
    int64_t ne_k,
    at::Tensor ne_weights,
    at::Tensor ne_mods,
    at::Tensor exclusive_ne_embeder_size_sums,
    at::Tensor tokens,
    at::Tensor exclusive_req_len_sums,
    at::Tensor ne_token_table,
    at::Tensor row_indices,
    at::Tensor column_starts,
    at::Tensor n_gram_ids,
    int64_t cuda_stream = 0
) {
    CHECK_INPUT(ne_weights);
    CHECK_INPUT(ne_mods);
    CHECK_INPUT(exclusive_ne_embeder_size_sums);
    CHECK_INPUT(tokens);
    CHECK_INPUT(exclusive_req_len_sums);
    CHECK_INPUT(ne_token_table);
    CHECK_INPUT(row_indices);
    CHECK_INPUT(column_starts);
    CHECK_INPUT(n_gram_ids);
    auto device = tokens.device();
    CHECK_EQ(ne_weights.device(), device);
    CHECK_EQ(ne_mods.device(), device);
    CHECK_EQ(tokens.device(), device);
    CHECK_EQ(exclusive_req_len_sums.device(), device);
    CHECK_EQ(n_gram_ids.device(), device);
    CHECK_DIM(3, ne_weights);
    CHECK_DIM(2, ne_mods);
    CHECK_DIM(1, exclusive_ne_embeder_size_sums);
    CHECK_DIM(1, tokens);
    CHECK_DIM(1, exclusive_req_len_sums);
    CHECK_DIM(2, ne_token_table);
    CHECK_DIM(1, row_indices);
    CHECK_DIM(1, column_starts);
    CHECK_DIM(2, n_gram_ids);
    int batch_size = exclusive_req_len_sums.size(0) - 1;
    int max_context_len = ne_token_table.size(1);
    CHECK_EQ(batch_size, exclusive_req_len_sums.size(0) - 1);
    CHECK_EQ(ne_n-1, ne_weights.size(0));
    CHECK_EQ(ne_k, ne_weights.size(1));
    CHECK_EQ(ne_n, ne_weights.size(2));
    CHECK_EQ(ne_n-1, ne_mods.size(0));
    CHECK_EQ(ne_k, ne_mods.size(1));
    CHECK_EQ(batch_size, row_indices.size(0));
    CHECK_EQ(batch_size, column_starts.size(0));
    if (ne_weights.scalar_type() != at::kInt) {
        throw std::runtime_error("Expected 'ne_weights' to be of type int (torch.int32).");
    }
    if (ne_mods.scalar_type() != at::kInt) {
        throw std::runtime_error("Expected 'ne_mods' to be of type int (torch.int32).");
    }
    if (tokens.scalar_type() != at::kInt) {
        throw std::runtime_error("Expected 'tokens' to be of type int (torch.int32).");
    }
    if (exclusive_req_len_sums.scalar_type() != at::kInt) {
        throw std::runtime_error("Expected 'exclusive_req_len_sums' to be of type int (torch.int32).");
    }
    if (exclusive_ne_embeder_size_sums.scalar_type() != at::kInt) {
        throw std::runtime_error("Expected 'exclusive_ne_embeder_size_sums' to be of type int (torch.int32).");
    }
    if (n_gram_ids.scalar_type() != at::kInt) {
        throw std::runtime_error("Expected 'n_gram_ids' to be of type int (torch.int32).");
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    cudaError_t status = embedding_scaling::ComputeNGramIds(
        batch_size,
        ne_n,
        ne_k,
        ne_weights.data_ptr<int>(),
        ne_mods.data_ptr<int>(),
        exclusive_ne_embeder_size_sums.data_ptr<int>(),
        tokens.data_ptr<int>(),
        exclusive_req_len_sums.data_ptr<int>(),
        ne_token_table.data_ptr<int>(),
        max_context_len,
        row_indices.data_ptr<long>(),
        column_starts.data_ptr<int>(),
        n_gram_ids.data_ptr<int>(),
        stream);
    TORCH_CHECK(
      status == cudaSuccess,
      "ComputeNGramIds failed with error code " + std::string(cudaGetErrorString(status)));
}

/**
    int batch_size,
    int* tokens,                      // [token_num]
    int* ne_token_table,              // [max_running_reqs, max_context_len]
    int max_context_len,              // max_context_len
    long* row_indices,                // [batch_size]
    int* column_starts,               // [batch_size]
    int* req_lens,                     // [batch_size]
    int ignore_token_num,             // 有多少token需要被ignore
    int* ignore_tokens,               // [ignore_token_num]
 */
void update_token_table(
    at::Tensor tokens,
    at::Tensor ne_token_table,
    at::Tensor row_indices,
    at::Tensor column_starts,
    at::Tensor req_lens,
    at::Tensor ignore_tokens,
    int64_t cuda_stream = 0
) {
    CHECK_INPUT(tokens);
    CHECK_INPUT(ne_token_table);
    CHECK_INPUT(row_indices);
    CHECK_INPUT(column_starts);
    CHECK_INPUT(req_lens);
    auto device = tokens.device();
    CHECK_EQ(tokens.device(), device);
    CHECK_EQ(row_indices.device(), device);
    CHECK_EQ(column_starts.device(), device);
    CHECK_EQ(req_lens.device(), device);

    CHECK_DIM(1, tokens);
    CHECK_DIM(2, ne_token_table);
    CHECK_DIM(1, row_indices);
    CHECK_DIM(1, column_starts);
    CHECK_DIM(1, req_lens);

    // 处理可选的 ignore_tokens 参数
    int ignore_token_num = 0;
    int* ignore_tokens_ptr = nullptr;
    if (ignore_tokens.defined() && ignore_tokens.numel() > 0) {
        // 当 ignore_tokens 不为 None 时进行校验
        CHECK_INPUT(ignore_tokens);
        CHECK_DIM(1, ignore_tokens);
        CHECK_EQ(ignore_tokens.device(), device);
        ignore_token_num = ignore_tokens.size(0);
        ignore_tokens_ptr = ignore_tokens.data_ptr<int>();
    }

    int batch_size = req_lens.size(0);
    int max_context_len = ne_token_table.size(1);
    CHECK_EQ(batch_size, row_indices.size(0));
    CHECK_EQ(batch_size, column_starts.size(0));
    if (tokens.scalar_type() != at::kInt) {
        throw std::runtime_error("Expected 'tokens' to be of type int (torch.int32).");
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    cudaError_t status = embedding_scaling::UpdateTokenTable(
        batch_size,
        tokens.data_ptr<int>(),
        ne_token_table.data_ptr<int>(),
        max_context_len,
        row_indices.data_ptr<long>(),
        column_starts.data_ptr<int>(),
        req_lens.data_ptr<int>(),
        ignore_token_num,
        ignore_tokens_ptr,
        stream);
    TORCH_CHECK(
      status == cudaSuccess,
      "ComputeNGramIds failed with error code " + std::string(cudaGetErrorString(status)));
}
