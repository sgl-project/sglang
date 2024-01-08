import torch
import triton
import triton.language as tl
from sglang.srt.utils import wrap_kernel_launcher


@triton.jit
def _fwd_segmented_gather(
    all_logits,
    len_add_1,
    cum_len,
    input_ids,
    logprobs,
    max_seq_len,
    voc_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    cur_req = tl.program_id(0)
    cur_l = tl.load(len_add_1 + cur_req)
    cum_l = tl.load(cum_len + cur_req)

    for i in range(0, (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE):
        off = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = off < cur_l - 1

        idx = tl.load(input_ids + cum_l - cur_l + off + 1, mask=mask)
        data = tl.load(all_logits + (cum_l - cur_l + off) * voc_size + idx, mask=mask)
        tl.store(logprobs + cum_l - cur_l - cur_req + off, data, mask=mask)


cached_kernel = None


def get_selected_logprob(all_logits, len_add_1, input_ids, logprobs):
    cum_len = torch.cumsum(len_add_1, dtype=torch.int32, dim=0)
    voc_size = all_logits.shape[1]
    grid = (len_add_1.shape[0], 1, 1)
    max_seq_len = len_add_1.max().item()

    global cached_kernel
    if cached_kernel:
        cached_kernel(
            grid,
            4,
            all_logits,
            len_add_1,
            cum_len,
            input_ids,
            logprobs,
            max_seq_len,
        )
        return

    _fwd_segmented_gather[grid](
        all_logits,
        len_add_1,
        cum_len,
        input_ids,
        logprobs,
        max_seq_len,
        voc_size,
        BLOCK_SIZE=128,
    )
    cached_kernel = wrap_kernel_launcher(_fwd_segmented_gather)


if __name__ == "__main__":
    all_logits = torch.tensor(
        #       s                     s                s
        [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]],
        dtype=torch.float32,
        device="cuda",
    )
    len_add_1 = torch.tensor([2, 3], dtype=torch.int32, device="cuda")
    input_ids = torch.tensor([1, 2, 3, 0, 1], dtype=torch.int32, device="cuda")
    logprobs = torch.empty((3), dtype=torch.float32, device="cuda")
    get_selected_logprobs(all_logits, len_add_1, input_ids, logprobs)
    print(logprobs)
    # assert logprobs == [2, 2, 4]
