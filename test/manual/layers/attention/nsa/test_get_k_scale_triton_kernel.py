import torch
import triton
import triton.language as tl


@triton.jit
def _get_k_and_s_triton_kernel(
    buf_ptr,
    page_indices_ptr,
    k_out_ptr,
    s_out_ptr,
    seq_len_ptr,
    seq_len_num: tl.constexpr,
    page_size: tl.constexpr,
    buf_numel_per_page: tl.constexpr,
    index_head_dim: tl.constexpr,
    s_offset_in_page: tl.constexpr,
    page_indice_batch_offset: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    batch_id = tl.program_id(0)
    token_id = tl.program_id(1)

    # Calculate which page and offset within page
    page_idx = token_id // page_size
    token_offset_in_page = token_id % page_size

    # Load batch id seq len from seq_len_ptr
    seq_len = tl.load(seq_len_ptr + batch_id)

    if token_id >= seq_len:
        return

    pre_batch_idx = tl.arange(0, seq_len_num)
    mask_pre_batch_idx = pre_batch_idx < batch_id
    prev_seq_lens = tl.load(seq_len_ptr + pre_batch_idx, mask=mask_pre_batch_idx)
    seq_len_offset = tl.sum(prev_seq_lens)
    # seq_len_offset = tl.load(seq_len_offset_ptr + batch_id)
    k_offset_batch = seq_len_offset * index_head_dim
    s_offset_batch = seq_len_offset * 4

    # Load the page index from page_indices
    page_index = tl.load(
        page_indices_ptr + page_idx + batch_id * page_indice_batch_offset
    )

    # ===== Load K data (128 bytes) =====
    # Calculate source offset for K in buf
    k_src_base_offset = (
        page_index * buf_numel_per_page + token_offset_in_page * index_head_dim
    )

    # Load 128 bytes (index_head_dim elements)
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    k_mask = (k_offsets < index_head_dim) & (token_id < seq_len)
    k_data = tl.load(buf_ptr + k_src_base_offset + k_offsets, mask=k_mask)

    # # Store K to output
    k_dst_offset = token_id * index_head_dim
    tl.store(k_out_ptr + k_dst_offset + k_offsets + k_offset_batch, k_data, mask=k_mask)

    # # ===== Load S data (4 bytes) =====
    # # Calculate source offset for S in buf
    s_src_base_offset = (
        page_index * buf_numel_per_page + s_offset_in_page + token_offset_in_page * 4
    )

    # # Load 4 bytes (fp32 scale)
    s_offsets = tl.arange(0, 4)
    s_mask = (s_offsets < 4) & (token_id < seq_len)
    s_data = tl.load(buf_ptr + s_src_base_offset + s_offsets, mask=s_mask)

    # # Store S to output
    s_dst_offset = token_id * 4
    tl.store(s_out_ptr + s_dst_offset + s_offsets + s_offset_batch, s_data, mask=s_mask)


def golden_torch_gen(
    seq_len_tensor: torch.Tensor,
    buffer_indexer: torch.Tensor,
    buffer: torch.Tensor,
    index_head_dim,
    page_size,
):
    dim_split = page_size * index_head_dim
    torch_k_out = buffer[:, 0:dim_split]
    torch_s_out = buffer[:, dim_split:]

    torch_k_out = torch_k_out.reshape(-1, 128)
    torch_s_out = torch_s_out.reshape(-1, 4)

    batch = seq_len_tensor.shape[0]
    index_list = []
    for i in range(batch):
        seq_len = seq_len_tensor[i].item()
        buffer_index_ = buffer_indexer[i]
        align_seq_len = ((seq_len + page_size - 1) / page_size) * page_size
        needed_block_num = int((seq_len + page_size - 1) / page_size)
        for j in range(needed_block_num):
            block_idx = buffer_index_[j].item()
            start_idx = block_idx * page_size
            end_idx = 0
            if j == (needed_block_num - 1):
                end_idx = block_idx * page_size + (
                    seq_len - (needed_block_num - 1) * page_size
                )
            else:
                end_idx = (block_idx + 1) * page_size

            index_tensor = (
                torch.arange(start=start_idx, end=end_idx, step=1)
                .type(torch.int32)
                .cuda()
            )
            index_list.append(index_tensor)

    index_list_ = torch.cat(index_list, dim=0)
    torch_k_out = torch.index_select(torch_k_out, dim=0, index=index_list_)
    torch_s_out = torch.index_select(torch_s_out, dim=0, index=index_list_)

    return torch_k_out, torch_s_out


def get_k_and_s_triton():
    index_head_dim = 128
    page_size = 64
    num_page = 128
    s_offset_in_page = page_size * index_head_dim

    seq_len_tensor = torch.tensor(
        [256, 267, 215, 32, 129], dtype=torch.int64, device="cuda"
    )  # 4 + 5 + 3 + 1 + 3 block
    buffer_indexer = torch.tensor(
        [
            [1, 2, 3, 4, 0],
            [7, 6, 5, 8, 9],
            [10, 11, 12, 0, 0],
            [13, 0, 0, 0, 0],
            [14, 15, 16, 0, 0],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    seq_len_sum = seq_len_tensor.sum()
    batch = seq_len_tensor.shape[0]

    triton_k_out = torch.empty(
        (seq_len_sum, index_head_dim), dtype=torch.uint8, device="cuda"
    )
    triton_s_out = torch.empty((seq_len_sum, 4), dtype=torch.uint8, device="cuda")
    buffer = torch.randint(
        0,
        num_page,
        (num_page, page_size * index_head_dim + page_size * 4),
        device="cuda",
    ).type(torch.uint8)

    _, buf_numel_per_page = buffer.shape
    _, page_indice_batch_offset = buffer_indexer.shape
    max_seq_len = seq_len_tensor.max().item()

    grid = (batch, max_seq_len)
    BLOCK_SIZE = 128
    seq_num_pow2 = 1
    while seq_num_pow2 < batch:
        seq_num_pow2 *= 2

    # acc test =====================
    _get_k_and_s_triton_kernel[grid](
        buf_ptr=buffer,
        page_indices_ptr=buffer_indexer,
        k_out_ptr=triton_k_out,
        s_out_ptr=triton_s_out,
        seq_len_ptr=seq_len_tensor,
        seq_len_num=seq_num_pow2,
        page_size=page_size,
        buf_numel_per_page=buf_numel_per_page,
        index_head_dim=index_head_dim,
        s_offset_in_page=s_offset_in_page,
        page_indice_batch_offset=page_indice_batch_offset,
        BLOCK_SIZE_K=BLOCK_SIZE,
    )

    torch_k_out, torch_s_out = golden_torch_gen(
        seq_len_tensor=seq_len_tensor,
        buffer_indexer=buffer_indexer,
        buffer=buffer,
        index_head_dim=index_head_dim,
        page_size=page_size,
    )

    torch.testing.assert_close(
        triton_k_out, torch_k_out, rtol=0, atol=0, msg="k outputs differ!"
    )
    torch.testing.assert_close(
        triton_s_out, torch_s_out, rtol=0, atol=0, msg="s outputs differ!"
    )
    print("_get_k_and_s_triton_kernel test pass")

    # perf test =====================
    import time

    torch.cuda.synchronize()
    for _ in range(10):
        _get_k_and_s_triton_kernel[grid](
            buf_ptr=buffer,
            page_indices_ptr=buffer_indexer,
            k_out_ptr=triton_k_out,
            s_out_ptr=triton_s_out,
            seq_len_ptr=seq_len_tensor,
            seq_len_num=seq_num_pow2,
            page_size=page_size,
            buf_numel_per_page=buf_numel_per_page,
            index_head_dim=index_head_dim,
            s_offset_in_page=s_offset_in_page,
            page_indice_batch_offset=page_indice_batch_offset,
            BLOCK_SIZE_K=BLOCK_SIZE,
        )

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    _get_k_and_s_triton_kernel[grid](
        buf_ptr=buffer,
        page_indices_ptr=buffer_indexer,
        k_out_ptr=triton_k_out,
        s_out_ptr=triton_s_out,
        seq_len_ptr=seq_len_tensor,
        seq_len_num=seq_num_pow2,
        page_size=page_size,
        buf_numel_per_page=buf_numel_per_page,
        index_head_dim=index_head_dim,
        s_offset_in_page=s_offset_in_page,
        page_indice_batch_offset=page_indice_batch_offset,
        BLOCK_SIZE_K=BLOCK_SIZE,
    )

    end_time = time.perf_counter()
    print(
        f"_get_k_and_s_triton_kernel triton kernel infer time is {((end_time-start_time)*1000):.4f} ms\n"
    )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping tests.")
        exit(0)

    print("Start test cases...\n")

    get_k_and_s_triton()

    print("End test cases...\n")
