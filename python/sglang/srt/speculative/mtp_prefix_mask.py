"""One-kernel materialization of the unchanged ragged, full-prefix block mask."""

import torch
import triton
import triton.language as tl


@triton.jit
def _write_mask(out, metadata, BLOCK: tl.constexpr, SLOTS: tl.constexpr):
    req = tl.program_id(0)
    query = tl.program_id(1)
    prefix = tl.load(metadata + 2 * req)
    start = tl.load(metadata + 2 * req + 1)
    width = prefix + SLOTS
    key = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    visible = key >= tl.maximum(prefix - 4096, 0)
    tl.store(out + start + query * width + key, visible, mask=key < width)


def build_full_prefix_mask(seq_lens, slots=8):
    assert slots in (7, 8)
    lengths = seq_lens.tolist()
    if not lengths:
        return torch.empty(0, device=seq_lens.device, dtype=torch.bool)
    metadata, total = [], 0
    for length in lengths:
        metadata.append((length, total))
        total += slots * (length + slots)
    info = torch.tensor(metadata, dtype=torch.int64, device=seq_lens.device)
    mask = torch.empty(total, dtype=torch.bool, device=seq_lens.device)
    _write_mask[(len(lengths), slots, triton.cdiv(max(lengths) + slots, 1024))](
        mask, info, BLOCK=1024, SLOTS=slots
    )
    return mask
