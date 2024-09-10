import torch
import triton
import triton.language as tl


@triton.jit
def create_flashinfer_kv_indices_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_indptr,
    kv_start_idx,
    max_context_len,
    kv_indices_ptr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)
    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    kv_indices_offset = tl.load(kv_indptr + pid)

    kv_start = 0
    kv_end = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
        kv_end = kv_start
    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

    req_to_token_ptr += req_pool_index * max_context_len
    kv_indices_ptr += kv_indices_offset

    ld_offset = kv_start + tl.arange(0, BLOCK_SIZE)
    st_offset = tl.arange(0, BLOCK_SIZE)
    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = ld_offset < kv_end
        data = tl.load(req_to_token_ptr + ld_offset, mask=mask)
        tl.store(kv_indices_ptr + st_offset, data, mask=mask)
        ld_offset += BLOCK_SIZE
        st_offset += BLOCK_SIZE


def update_flashinfer_indices(
    forward_mode,
    model_runner,
    req_pool_indices,
    seq_lens,
    prefix_lens,
    flashinfer_decode_wrapper=None,
    flashinfer_use_ragged=False,
):
    """Init auxiliary variables for FlashInfer attention backend."""
    num_qo_heads = model_runner.model_config.num_attention_heads // model_runner.tp_size
    num_kv_heads = model_runner.model_config.get_num_kv_heads(model_runner.tp_size)
    head_dim = model_runner.model_config.head_dim
    batch_size = len(req_pool_indices)

    if model_runner.sliding_window_size is None:
        if flashinfer_use_ragged:
            paged_kernel_lens = prefix_lens
        else:
            paged_kernel_lens = seq_lens

        kv_indptr = torch.zeros((batch_size + 1,), dtype=torch.int32, device="cuda")
        kv_indptr[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        kv_indices = torch.empty(kv_indptr[-1], dtype=torch.int32, device="cuda")
        create_flashinfer_kv_indices_triton[(batch_size,)](
            model_runner.req_to_token_pool.req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            kv_indptr,
            None,
            model_runner.req_to_token_pool.req_to_token.size(1),
            kv_indices,
        )

        kv_last_page_len = torch.ones((batch_size,), dtype=torch.int32, device="cuda")

        if forward_mode.is_decode():
            # CUDA graph uses different flashinfer_decode_wrapper
            if flashinfer_decode_wrapper is None:
                flashinfer_decode_wrapper = model_runner.flashinfer_decode_wrapper

            flashinfer_decode_wrapper.end_forward()
            flashinfer_decode_wrapper.begin_forward(
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                num_qo_heads,
                num_kv_heads,
                head_dim,
                1,
                data_type=model_runner.kv_cache_dtype,
                q_data_type=model_runner.dtype,
            )
        else:
            # extend part
            qo_indptr = torch.zeros((batch_size + 1,), dtype=torch.int32, device="cuda")
            qo_indptr[1:] = torch.cumsum(seq_lens - prefix_lens, dim=0)

            if flashinfer_use_ragged:
                model_runner.flashinfer_prefill_wrapper_ragged.end_forward()
                model_runner.flashinfer_prefill_wrapper_ragged.begin_forward(
                    qo_indptr,
                    qo_indptr,
                    num_qo_heads,
                    num_kv_heads,
                    head_dim,
                )

            # cached part
            model_runner.flashinfer_prefill_wrapper_paged.end_forward()
            model_runner.flashinfer_prefill_wrapper_paged.begin_forward(
                qo_indptr,
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                num_qo_heads,
                num_kv_heads,
                head_dim,
                1,
            )
    else:
        # window attention use paged only
        kv_last_page_len = torch.ones((batch_size,), dtype=torch.int32, device="cuda")
        for wrapper_id in range(2):
            if wrapper_id == 0:
                if forward_mode.is_decode():
                    paged_kernel_lens = torch.minimum(
                        seq_lens, torch.tensor(model_runner.sliding_window_size + 1)
                    )
                else:
                    paged_kernel_lens = torch.minimum(
                        seq_lens,
                        torch.tensor(model_runner.sliding_window_size)
                        + seq_lens
                        - prefix_lens,
                    )
            else:
                paged_kernel_lens = seq_lens

            kv_start_idx = seq_lens - paged_kernel_lens

            kv_indptr = torch.zeros((batch_size + 1,), dtype=torch.int32, device="cuda")
            kv_indptr[1:] = torch.cumsum(paged_kernel_lens, dim=0)

            kv_indices = torch.empty(kv_indptr[-1], dtype=torch.int32, device="cuda")
            create_flashinfer_kv_indices_triton[(batch_size,)](
                model_runner.req_to_token_pool.req_to_token,
                req_pool_indices,
                paged_kernel_lens,
                kv_indptr,
                kv_start_idx,
                model_runner.req_to_token_pool.req_to_token.size(1),
                kv_indices,
            )

            if forward_mode.is_decode():
                # CUDA graph uses different flashinfer_decode_wrapper
                if flashinfer_decode_wrapper is None:
                    flashinfer_decode_wrapper = model_runner.flashinfer_decode_wrapper

                flashinfer_decode_wrapper[wrapper_id].end_forward()
                flashinfer_decode_wrapper[wrapper_id].begin_forward(
                    kv_indptr,
                    kv_indices,
                    kv_last_page_len,
                    num_qo_heads,
                    num_kv_heads,
                    head_dim,
                    1,
                    data_type=model_runner.kv_cache_dtype,
                    q_data_type=model_runner.dtype,
                )
            else:
                # extend part
                qo_indptr = torch.zeros(
                    (batch_size + 1,), dtype=torch.int32, device="cuda"
                )
                qo_indptr[1:] = torch.cumsum(seq_lens - prefix_lens, dim=0)

                model_runner.flashinfer_prefill_wrapper_paged[wrapper_id].end_forward()
                model_runner.flashinfer_prefill_wrapper_paged[wrapper_id].begin_forward(
                    qo_indptr,
                    kv_indptr,
                    kv_indices,
                    kv_last_page_len,
                    num_qo_heads,
                    num_kv_heads,
                    head_dim,
                    1,
                )
