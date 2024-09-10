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


def update_flashinfer_indices_window(
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
            qo_indptr = torch.zeros((batch_size + 1,), dtype=torch.int32, device="cuda")
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


def update_flashinfer_indices(
    forward_mode,
    model_runner,
    req_pool_indices,
    seq_lens,
    prefix_lens,
    flashinfer_decode_wrapper=None,
    flashinfer_use_ragged=False,
):
    flashinfer_updater = FlashinferUpdater(
        forward_mode,
        model_runner,
        req_pool_indices,
        seq_lens,
        prefix_lens,
        flashinfer_decode_wrapper,
        flashinfer_use_ragged,
    )

    if model_runner.sliding_window_size is None:
        flashinfer_updater.update_indices_no_window()
    else:
        flashinfer_updater.update_indices_window()


class FlashinferUpdater:
    def __init__(
        self,
        forward_mode,
        model_runner,
        req_pool_indices,
        seq_lens,
        prefix_lens,
        flashinfer_decode_wrapper=None,
        flashinfer_use_ragged=False,
    ):
        self.forward_mode = forward_mode
        self.model_runner = model_runner
        self.req_pool_indices = req_pool_indices
        self.seq_lens = seq_lens
        self.prefix_lens = prefix_lens
        self.flashinfer_decode_wrapper = flashinfer_decode_wrapper
        self.flashinfer_use_ragged = flashinfer_use_ragged

        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // model_runner.tp_size
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            model_runner.tp_size
        )
        self.head_dim = model_runner.model_config.head_dim
        self.batch_size = len(req_pool_indices)

        if flashinfer_use_ragged:
            self.paged_kernel_lens = prefix_lens
        else:
            self.paged_kernel_lens = seq_lens

        self.kv_indptr = torch.zeros(
            (self.batch_size + 1,), dtype=torch.int32, device="cuda"
        )
        self.kv_indptr[1:] = torch.cumsum(self.paged_kernel_lens, dim=0)
        self.kv_indices = torch.empty(
            self.kv_indptr[-1], dtype=torch.int32, device="cuda"
        )

        create_flashinfer_kv_indices_triton[(self.batch_size,)](
            model_runner.req_to_token_pool.req_to_token,
            req_pool_indices,
            self.paged_kernel_lens,
            self.kv_indptr,
            None,
            model_runner.req_to_token_pool.req_to_token.size(1),
            self.kv_indices,
        )

        self.kv_last_page_len = torch.ones(
            (self.batch_size,), dtype=torch.int32, device="cuda"
        )

        # CUDA graph uses different flashinfer_decode_wrapper
        if self.flashinfer_decode_wrapper is None:
            self.flashinfer_decode_wrapper = self.model_runner.flashinfer_decode_wrapper

    def _update_decode_indices(self):
        self.flashinfer_decode_wrapper.end_forward()
        self.flashinfer_decode_wrapper.begin_forward(
            self.kv_indptr,
            self.kv_indices,
            self.kv_last_page_len,
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            1,
            data_type=self.model_runner.kv_cache_dtype,
            q_data_type=self.model_runner.dtype,
        )

    def _update_extend_indices(self):
        # extend part
        qo_indptr = torch.zeros(
            (self.batch_size + 1,), dtype=torch.int32, device="cuda"
        )
        qo_indptr[1:] = torch.cumsum(self.seq_lens - self.prefix_lens, dim=0)

        if self.flashinfer_use_ragged:
            self.model_runner.flashinfer_prefill_wrapper_ragged.end_forward()
            self.model_runner.flashinfer_prefill_wrapper_ragged.begin_forward(
                qo_indptr,
                qo_indptr,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
            )

        # cached part
        self.model_runner.flashinfer_prefill_wrapper_paged.end_forward()
        self.model_runner.flashinfer_prefill_wrapper_paged.begin_forward(
            qo_indptr,
            self.kv_indptr,
            self.kv_indices,
            self.kv_last_page_len,
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            1,
        )

    def update_indices_no_window(self):
        """Init auxiliary variables for FlashInfer attention backend."""

        if self.forward_mode.is_decode():
            self._update_decode_indices()
        else:
            self._update_extend_indices()

    def update_indices_window(self):
        update_flashinfer_indices_window(
            self.forward_mode,
            self.model_runner,
            self.req_pool_indices,
            self.seq_lens,
            self.prefix_lens,
            self.flashinfer_decode_wrapper,
            self.flashinfer_use_ragged,
        )
