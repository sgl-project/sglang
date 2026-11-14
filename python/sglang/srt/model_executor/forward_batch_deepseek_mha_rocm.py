"""ROCm prefix-chunk layout for DeepSeek chunked-prefix MHA.

All cached prefixes are laid end to end in request order, and chunk i holds keys
[i * C, (i + 1) * C) of that sequence, so a short or empty prefix leaves its share
of a chunk to the long ones. The shared request-major layout gives every request
C // batch_size keys per chunk, so the longest prefix alone sets the chunk count.
"""

from typing import List, Tuple

import torch


def get_num_packed_prefix_chunks(
    *, prefix_lens_cpu: List[int], chunk_capacity: int
) -> int:
    return (sum(prefix_lens_cpu) + chunk_capacity - 1) // chunk_capacity


# Drop-in for get_prefix_chunk_seq_lens with prefix_chunk_len = chunk capacity;
# outputs are (num_prefix_chunks, batch_size) int32 starts and lengths.
def get_packed_prefix_chunk_seq_lens(
    prefix_lens: torch.Tensor, num_prefix_chunks: int, prefix_chunk_len: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    prefix_lens = prefix_lens.to(torch.int64)
    offsets = torch.cumsum(prefix_lens, dim=0) - prefix_lens
    chunk_begins = (
        torch.arange(num_prefix_chunks, device=prefix_lens.device).unsqueeze(1)
        * prefix_chunk_len
    )
    prefix_chunk_starts = torch.minimum(
        (chunk_begins - offsets).clamp(min=0), prefix_lens
    )
    prefix_chunk_ends = torch.minimum(
        (chunk_begins + prefix_chunk_len - offsets).clamp(min=0), prefix_lens
    )
    return (
        prefix_chunk_starts.to(torch.int32),
        (prefix_chunk_ends - prefix_chunk_starts).to(torch.int32),
    )
