"""Logical draft KV transfer independent of target DCP-local addressing."""


def draft_transfer_buffers(pool, wire_page_size):
    """Use target wire-page units even when draft allocation pages are wider."""
    if wire_page_size <= 0 or pool.page_size % wire_page_size:
        raise ValueError("Draft allocation pages must contain whole transfer pages")
    ptrs, lengths, item_lengths = pool.get_contiguous_buf_infos()
    ratio = pool.page_size // wire_page_size
    if any(length % ratio for length in item_lengths):
        raise ValueError("Draft buffer page bytes must divide into transfer pages")
    return ptrs, lengths, [length // ratio for length in item_lengths]


def draft_transfer_start(seq_len, window_size, page_size):
    if seq_len < 0 or page_size <= 0 or (window_size is not None and window_size <= 0):
        raise ValueError("Invalid draft transfer extent")
    start = max(0, seq_len - window_size) if window_size is not None else 0
    return start // page_size * page_size
