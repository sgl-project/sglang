#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace device::ngram_embedding {

__global__ void ComputeNGramIdsKernel(
    int batch_size,
    int ne_n,
    int ne_k,
    int* ne_weights,                      // [ne_n-1,ne_k,ne_n]
    int* ne_mods,                         // [ne_n-1,ne_k]
    int* exclusive_ne_embeder_size_sums,  // [(ne_n-1)*ne_k]
    int* tokens,                          // [token_num]
    int* exclusive_req_len_sums,          // [batch_size+1]
    int* ne_token_table,                  // [max_running_reqs, max_context_len]
    int max_context_len,                  // max_context_len
    long* row_indices,                    // [batch_size]
    int* column_starts,                   // [batch_size]
    int* n_gram_ids                       // [ne_n-1,ne_k,token_num]
) {
  // Determine which n, k, and request this block handles.
  /**
  Example: [req0, req1, req2] with n=3, k=2
  n       k       req_id      blockIdx.x  config_id (combination of n and k)
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
  // n and k here are offset from their physical meanings: n = real_n - 2, k = real_k - 1.
  // This offset exists because n and k are used as indices into ne_weights and ne_mods.
  const int k = config_id % ne_k;
  const int n = (config_id - config_id % ne_k) / ne_k;
  // ne_weights has shape [ne_n-1, ne_k, ne_n]; last dim is token distance, so compute base index first
  const int ne_weight_base_idx = n * ne_k * ne_n + k * ne_n;
  // ne_mods has shape [ne_n-1, ne_k]
  const int ne_mod = ne_mods[n * ne_k + k];
  // stride loop
  for (int i = exclusive_req_len_sums[req_id] + threadIdx.x; i < exclusive_req_len_sums[req_id + 1]; i += blockDim.x) {
    uint64_t n_gram_id = 0;
    // Token offset within the current request
    int current_token_offset = i - exclusive_req_len_sums[req_id];
    // Start index of this request in the token table; tokens before this belong to other requests
    int req_token_table_index = row_indices[req_id] * max_context_len;
    // Position of the current token in the token table
    int current_token_table_index = req_token_table_index + column_starts[req_id] + current_token_offset;
    for (int j = 0; j < n + 2; j++) {
      if (current_token_table_index - j < req_token_table_index) {
        // Out of this request's range, stop computing n_gram_id
        break;
      }
      if (ne_token_table[current_token_table_index - j] < 0) {
        // Token was marked as ignored during write
        break;
      }
      const uint64_t term =
          (uint64_t)ne_token_table[current_token_table_index - j] * (uint64_t)ne_weights[ne_weight_base_idx + j];
      n_gram_id += term % ne_mod;
    }
    n_gram_id %= ne_mod;
    n_gram_id += exclusive_ne_embeder_size_sums[n * ne_k + k];
    // [token_num, ne_n-1, ne_k]
    n_gram_ids[i * (ne_n - 1) * ne_k + n * ne_k + k] = (int)(n_gram_id);
  }
}

__global__ void UpdateTokenTableKernel(
    int batch_size,
    int* tokens,           // [token_num]
    int* ne_token_table,   // [max_running_reqs, max_context_len]
    int max_context_len,   // max_context_len
    long* row_indices,     // [batch_size]
    int* column_starts,    // [batch_size]
    int* req_lens,         // [batch_size]
    int ignore_token_num,  // number of tokens to ignore
    int* ignore_tokens     // [ignore_token_num]
) {
  // Each block processes one request.
  const int req_id = blockIdx.x % batch_size;
  int start = 0;
  int end = 0;
  for (int i = 0; i < req_id; i++) {
    start += req_lens[i];
  }
  end = start + req_lens[req_id];
  // stride loop
  for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
    // Token offset within the current request
    int current_token_offset = i - start;
    // Start index of this request in the token table
    int req_token_table_index = row_indices[req_id] * max_context_len;
    // Position of the current token in the token table
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

}  // namespace device::ngram_embedding

namespace {

struct NgramEmbeddingKernel {
  static void compute_n_gram_ids(
      const int64_t ne_n,
      const int64_t ne_k,
      const tvm::ffi::TensorView ne_weights,
      const tvm::ffi::TensorView ne_mods,
      const tvm::ffi::TensorView exclusive_ne_embeder_size_sums,
      const tvm::ffi::TensorView tokens,
      const tvm::ffi::TensorView exclusive_req_len_sums,
      const tvm::ffi::TensorView ne_token_table,
      const tvm::ffi::TensorView row_indices,
      const tvm::ffi::TensorView column_starts,
      const tvm::ffi::TensorView n_gram_ids) {
    using namespace host;

    auto device_ = SymbolicDevice{};

    // Verify tensor shapes and types using -1 (kAnySize) for dynamic dimensions
    TensorMatcher({-1, -1, -1})  // [ne_n-1, ne_k, ne_n]
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>(device_)
        .verify(ne_weights);

    TensorMatcher({-1, -1})  // [ne_n-1, ne_k]
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>()
        .verify(ne_mods);

    TensorMatcher({-1})  // [(ne_n-1)*ne_k + 1]
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>()
        .verify(exclusive_ne_embeder_size_sums);

    TensorMatcher({-1})  // [token_num]
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>()
        .verify(tokens);

    TensorMatcher({-1})  // [batch_size+1]
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>()
        .verify(exclusive_req_len_sums);

    TensorMatcher({-1, -1})  // [max_running_reqs, max_context_len]
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>()
        .verify(ne_token_table);

    TensorMatcher({-1})  // [batch_size]
        .with_dtype<int64_t>()
        .with_device<kDLCUDA>()
        .verify(row_indices);

    TensorMatcher({-1})  // [batch_size]
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>()
        .verify(column_starts);

    TensorMatcher({-1, -1})  // [token_num, (ne_n-1)*ne_k]
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>()
        .verify(n_gram_ids);

    const int batch_size = static_cast<int>(exclusive_req_len_sums.size(0) - 1);
    const int max_context_len = static_cast<int>(ne_token_table.size(1));
    const auto stream = LaunchKernel::resolve_device(device_.unwrap());

    constexpr int BLOCK_THREADS = 256;
    const int num_configs = (static_cast<int>(ne_n) - 1) * static_cast<int>(ne_k);
    const int grid_size = num_configs * batch_size;

    LaunchKernel(grid_size, BLOCK_THREADS, stream)(
        device::ngram_embedding::ComputeNGramIdsKernel,
        batch_size,
        static_cast<int>(ne_n),
        static_cast<int>(ne_k),
        static_cast<int*>(ne_weights.data_ptr()),
        static_cast<int*>(ne_mods.data_ptr()),
        static_cast<int*>(exclusive_ne_embeder_size_sums.data_ptr()),
        static_cast<int*>(tokens.data_ptr()),
        static_cast<int*>(exclusive_req_len_sums.data_ptr()),
        static_cast<int*>(ne_token_table.data_ptr()),
        max_context_len,
        static_cast<long*>(row_indices.data_ptr()),
        static_cast<int*>(column_starts.data_ptr()),
        static_cast<int*>(n_gram_ids.data_ptr()));
  }

  static void update_token_table(
      const tvm::ffi::TensorView tokens,
      const tvm::ffi::TensorView ne_token_table,
      const tvm::ffi::TensorView row_indices,
      const tvm::ffi::TensorView column_starts,
      const tvm::ffi::TensorView req_lens,
      const tvm::ffi::TensorView ignore_tokens) {
    using namespace host;

    auto device_ = SymbolicDevice{};

    // Verify tensor shapes and types using -1 (kAnySize) for dynamic dimensions
    TensorMatcher({-1})  // [token_num]
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>(device_)
        .verify(tokens);

    TensorMatcher({-1, -1})  // [max_running_reqs, max_context_len]
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>()
        .verify(ne_token_table);

    TensorMatcher({-1})  // [batch_size]
        .with_dtype<int64_t>()
        .with_device<kDLCUDA>()
        .verify(row_indices);

    TensorMatcher({-1})  // [batch_size]
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>()
        .verify(column_starts);

    TensorMatcher({-1})  // [batch_size]
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>()
        .verify(req_lens);

    // ignore_tokens can be empty or have values
    void* ignore_tokens_ptr = ignore_tokens.data_ptr();
    const bool has_ignore_tokens = ignore_tokens_ptr != nullptr && ignore_tokens.numel() > 0;
    if (has_ignore_tokens) {
      TensorMatcher({-1})  // [ignore_token_num]
          .with_dtype<int32_t>()
          .with_device<kDLCUDA>()
          .verify(ignore_tokens);
    }

    const int batch_size = static_cast<int>(req_lens.size(0));
    if (batch_size <= 0) {
      return;
    }

    const int max_context_len = static_cast<int>(ne_token_table.size(1));
    const auto stream = LaunchKernel::resolve_device(device_.unwrap());

    constexpr int BLOCK_THREADS = 256;
    const int grid_size = batch_size;

    int ignore_token_num = 0;
    int* ignore_tokens_typed_ptr = nullptr;
    if (has_ignore_tokens) {
      ignore_token_num = static_cast<int>(ignore_tokens.numel());
      ignore_tokens_typed_ptr = static_cast<int*>(ignore_tokens_ptr);
    }

    LaunchKernel(grid_size, BLOCK_THREADS, stream)(
        device::ngram_embedding::UpdateTokenTableKernel,
        batch_size,
        static_cast<int*>(tokens.data_ptr()),
        static_cast<int*>(ne_token_table.data_ptr()),
        max_context_len,
        static_cast<long*>(row_indices.data_ptr()),
        static_cast<int*>(column_starts.data_ptr()),
        static_cast<int*>(req_lens.data_ptr()),
        ignore_token_num,
        ignore_tokens_typed_ptr);
  }
};

}  // namespace
