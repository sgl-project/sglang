import torch

def get_cuda_stream():
    return torch.cuda.current_stream().cuda_stream

def compute_n_gram_ids(
      ne_n,
      ne_k,
      ne_weights,
      ne_mods,
      exclusive_ne_embeder_size_sums,
      tokens,
      exclusive_req_len_sums,
      ne_token_table,
      row_indices,
      column_starts,
      n_gram_ids
):
    torch.ops.sgl_kernel.compute_n_gram_ids.default(
        ne_n,
        ne_k,
        ne_weights,
        ne_mods,
        exclusive_ne_embeder_size_sums,
        tokens,
        exclusive_req_len_sums,
        ne_token_table,
        row_indices,
        column_starts,
        n_gram_ids,
        get_cuda_stream()
    )

def update_token_table(
      tokens,
      ne_token_table,
      row_indices,
      column_starts,
      req_lens,
      ignore_tokens
):
    torch.ops.sgl_kernel.update_token_table.default(
      tokens,
      ne_token_table,
      row_indices,
      column_starts,
      req_lens,
      ignore_tokens,
      get_cuda_stream()
    )
