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
    kv_indices_ptr,
    max_context_len: tl.constexpr,
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


class FlashinferUpdater:
    def __init__(
        self,
        forward_mode,
        model_runner,
        req_pool_indices,
        seq_lens,
        prefix_lens,
        decode_wrapper=None,
        use_ragged=False,
    ):
        self.forward_mode = forward_mode
        self.model_runner = model_runner
        self.req_pool_indices = req_pool_indices
        self.seq_lens = seq_lens
        self.prefix_lens = prefix_lens
        self.use_ragged = use_ragged

        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // model_runner.tp_size
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            model_runner.tp_size
        )
        self.head_dim = model_runner.model_config.head_dim
        self.batch_size = len(req_pool_indices)

        self.decode_wrapper = (
            decode_wrapper or self.model_runner.attn_backend.decode_wrapper
        )
        self.prefill_wrapper_ragged = (
            self.model_runner.attn_backend.prefill_wrapper_ragged
        )
        self.prefill_wrapper_paged = (
            self.model_runner.attn_backend.prefill_wrapper_paged
        )

        self.kv_last_page_len = torch.ones(
            (self.batch_size,), dtype=torch.int32, device="cuda"
        )

    def _init_indices_no_sliding_window(self):
        if self.use_ragged:
            paged_kernel_lens = self.prefix_lens
        else:
            paged_kernel_lens = self.seq_lens

        self.kv_indptr = torch.zeros(
            (self.batch_size + 1,), dtype=torch.int32, device="cuda"
        )
        self.kv_indptr[1:] = torch.cumsum(paged_kernel_lens, dim=0)
        self.kv_indices = torch.empty(
            self.kv_indptr[-1], dtype=torch.int32, device="cuda"
        )

        create_flashinfer_kv_indices_triton[(self.batch_size,)](
            self.model_runner.req_to_token_pool.req_to_token,
            self.req_pool_indices,
            paged_kernel_lens,
            self.kv_indptr,
            None,
            self.kv_indices,
            self.model_runner.req_to_token_pool.req_to_token.size(1),
        )

    def _init_indices_sliding_window(self, wrapper_id):
        if wrapper_id == 0:
            # window attention use paged only
            if self.forward_mode.is_decode():
                paged_kernel_lens = torch.minimum(
                    self.seq_lens,
                    torch.tensor(self.model_runner.sliding_window_size + 1),
                )
            else:
                paged_kernel_lens = torch.minimum(
                    self.seq_lens,
                    torch.tensor(self.model_runner.sliding_window_size)
                    + self.seq_lens
                    - self.prefix_lens,
                )
        else:
            # full attention
            paged_kernel_lens = self.seq_lens

        kv_start_idx = self.seq_lens - paged_kernel_lens
        self.kv_indptr = torch.zeros(
            (self.batch_size + 1,), dtype=torch.int32, device="cuda"
        )
        self.kv_indptr[1:] = torch.cumsum(paged_kernel_lens, dim=0)
        self.kv_indices = torch.empty(
            self.kv_indptr[-1], dtype=torch.int32, device="cuda"
        )
        create_flashinfer_kv_indices_triton[(self.batch_size,)](
            self.model_runner.req_to_token_pool.req_to_token,
            self.req_pool_indices,
            paged_kernel_lens,
            self.kv_indptr,
            kv_start_idx,
            self.kv_indices,
            self.model_runner.req_to_token_pool.req_to_token.size(1),
        )

    def _update_decode_indices(self, decode_wrapper):
        decode_wrapper.end_forward()
        decode_wrapper.begin_forward(
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

    def _update_extend_indices(self, ragged_wrapper, paged_wrapper):
        # extend part
        qo_indptr = torch.zeros(
            (self.batch_size + 1,), dtype=torch.int32, device="cuda"
        )
        qo_indptr[1:] = torch.cumsum(self.seq_lens - self.prefix_lens, dim=0)

        if self.use_ragged:
            ragged_wrapper.end_forward()
            ragged_wrapper.begin_forward(
                qo_indptr,
                qo_indptr,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
            )

        # cached part
        paged_wrapper.end_forward()
        paged_wrapper.begin_forward(
            qo_indptr,
            self.kv_indptr,
            self.kv_indices,
            self.kv_last_page_len,
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            1,
        )

    def update_indices_no_sliding_window(self):
        self._init_indices_no_sliding_window()

        if self.forward_mode.is_decode():
            self._update_decode_indices(self.decode_wrapper)
        else:
            self._update_extend_indices(
                self.prefill_wrapper_ragged,
                self.prefill_wrapper_paged,
            )

    def update_indices_sliding_window(self):
        assert self.use_ragged is False

        for wrapper_id in range(2):
            self._init_indices_sliding_window(wrapper_id)
            if self.forward_mode.is_decode():
                self._update_decode_indices(self.decode_wrapper[wrapper_id])
            else:
                self._update_extend_indices(
                    None,
                    self.prefill_wrapper_paged[wrapper_id],
                )


def update_flashinfer_indices(
    forward_mode,
    model_runner,
    req_pool_indices,
    seq_lens,
    prefix_lens,
    decode_wrapper=None,
    use_ragged=False,
):
    updater = FlashinferUpdater(
        forward_mode,
        model_runner,
        req_pool_indices,
        seq_lens,
        prefix_lens,
        decode_wrapper,
        use_ragged,
    )

    if model_runner.sliding_window_size is None:
        updater.update_indices_no_sliding_window()
    else:
        updater.update_indices_sliding_window()
