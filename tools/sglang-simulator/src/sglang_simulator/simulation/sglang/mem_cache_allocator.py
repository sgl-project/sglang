import torch
from sglang_simulator.hook import BaseHook


class IndexableWrapper:
    def __init__(self, fn):
        self._fn = fn

    def __getitem__(self, _):
        return self._fn

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


def alloc_extend_cpu(
    pre_lens_ptr: torch.Tensor,
    seq_lens_ptr: torch.Tensor,
    last_loc_ptr: torch.Tensor,
    free_page_ptr: torch.Tensor,
    out_indices: torch.Tensor,  # Pre-allocated output tensor (consistent with Triton kernel)
    bs_upper: int,  # CPU doesn't need this, but kept for interface consistency (can be ignored)
    page_size: int,
):
    # Convert to Python list or scalar for processing
    pre_lens = pre_lens_ptr.cpu().tolist()
    seq_lens = seq_lens_ptr.cpu().tolist()
    last_loc = last_loc_ptr.cpu().tolist()
    free_pages = free_page_ptr.cpu().tolist()

    batch_size = len(pre_lens)

    def ceil_div(a, b):
        return (a + b - 1) // b

    extend_lens = [seq_lens[i] - pre_lens[i] for i in range(batch_size)]
    num_new_pages_per_seq = []
    for i in range(batch_size):
        pages_before = ceil_div(pre_lens[i], page_size)
        pages_after = ceil_div(seq_lens[i], page_size)
        num_new_pages_per_seq.append(pages_after - pages_before)

    # Initialize offsets
    output_offset = 0
    free_page_offset = 0

    # Process each sequence
    for pid in range(batch_size):
        pre_len = pre_lens[pid]
        seq_len = seq_lens[pid]
        extend_len = extend_lens[pid]
        last_token_pos = last_loc[pid]

        if extend_len <= 0:
            # No extension, skip (but still need to advance free_page_offset)
            free_page_offset += num_new_pages_per_seq[pid]
            continue

        # === Part 1: Fill the remaining space of the current incomplete page ===
        current_page_end = ceil_div(pre_len, page_size) * page_size
        part1_end = min(seq_len, current_page_end)
        num_part1 = part1_end - pre_len

        if num_part1 > 0:
            # out_indices[output_offset : output_offset + num_part1] = last_token_pos + 1 + i
            for i in range(num_part1):
                out_indices[output_offset + i] = last_token_pos + 1 + i
            output_offset += num_part1

        if pre_len + num_part1 == seq_len:
            free_page_offset += num_new_pages_per_seq[pid]
            continue

        # === Part 2: Fill complete new pages ===
        full_pages_start = current_page_end
        full_pages_end = (seq_len // page_size) * page_size
        num_part2 = full_pages_end - full_pages_start

        if num_part2 > 0:
            for i in range(num_part2):
                page_idx_in_free = free_page_offset + (i // page_size)
                page_id = free_pages[page_idx_in_free]
                token_in_page = i % page_size
                out_indices[output_offset + i] = page_id * page_size + token_in_page
            output_offset += num_part2

        if pre_len + num_part1 + num_part2 == seq_len:
            free_page_offset += num_new_pages_per_seq[pid]
            continue

        # === Part 3: Fill the last incomplete new page ===
        num_part3 = seq_len - full_pages_end
        if num_part3 > 0:
            last_page_idx = free_page_offset + num_new_pages_per_seq[pid] - 1
            last_page_id = free_pages[last_page_idx]
            for i in range(num_part3):
                out_indices[output_offset + i] = last_page_id * page_size + i
            output_offset += num_part3

        # Push forward free_page_offset
        free_page_offset += num_new_pages_per_seq[pid]


def alloc_decode_cpu(
    seq_lens_ptr: torch.Tensor,
    last_loc_ptr: torch.Tensor,
    free_page_ptr: torch.Tensor,
    out_indices: torch.Tensor,
    bs_upper: int,  # Reserved parameter (not used in CPU)
    page_size: int,
):
    seq_lens = seq_lens_ptr.cpu().tolist()
    last_loc = last_loc_ptr.cpu().tolist()
    free_pages = free_page_ptr.cpu().tolist()

    batch_size = len(seq_lens)

    def ceil_div(a, b):
        return (a + b - 1) // b

    # Calculate the number of new pages needed for each sequence (used to determine free_page offset)
    num_new_pages_per_seq = []
    for i in range(batch_size):
        pre_len_i = seq_lens[i] - 1
        pages_before = ceil_div(pre_len_i, page_size)
        pages_after = ceil_div(seq_lens[i], page_size)
        num_new_pages_per_seq.append(pages_after - pages_before)

    # Calculate prefix sum to determine the starting position in free_page_ptr for each sequence
    prefix_sum = 0
    for pid in range(batch_size):
        num_new_pages_self = num_new_pages_per_seq[pid]
        new_page_start_loc = (
            prefix_sum  # Starting index in free_pages for current sequence
        )
        prefix_sum += num_new_pages_self

        seq_len = seq_lens[pid]
        pre_len = seq_len - 1

        num_page_start_loc_self = ceil_div(seq_len, page_size) - ceil_div(
            pre_len, page_size
        )

        if num_page_start_loc_self == 0:
            # Reuse current page, directly write last_loc + 1
            out_indices[pid] = last_loc[pid] + 1
        else:
            # Allocate new page, take the first new page ID
            page_id = free_pages[new_page_start_loc]
            out_indices[pid] = page_id * page_size


class C_PagedTokenToKVPoolAllocatorHook(BaseHook):
    HOOK_CLASS_NAME = "PagedTokenToKVPoolAllocator"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.allocator"

    @classmethod
    def hook(cls, target):
        original_init = target.__init__

        def wrapped_init(self, *args, **kwargs):

            from sglang.srt.mem_cache import allocator

            # triton kernels are not compatible with the CPU allocator, so we use python implementation instead.
            allocator.alloc_extend_kernel = IndexableWrapper(alloc_extend_cpu)
            allocator.alloc_decode_kernel = IndexableWrapper(alloc_decode_cpu)

            original_init(self, *args, **kwargs)

        target.__init__ = wrapped_init
