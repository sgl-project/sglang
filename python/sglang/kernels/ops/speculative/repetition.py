"""Fill speculative candidates from repeated output patterns at live positions."""

import torch
import triton
import triton.language as tl


@triton.jit
def _override_repeating_drafts(
    tokens,
    lengths,
    patterns,
    token_stride: tl.constexpr,
    pattern_stride: tl.constexpr,
    width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    period = tl.load(patterns + row * pattern_stride)
    if period > 0:
        start = tl.load(patterns + row * pattern_stride + 1)
        # prefix length is the position of the anchor, not its successor.
        position = tl.load(lengths + row).to(tl.int64) + 1 + col
        phase = (position - start) % period
        phase = tl.where(phase < 0, phase + period, phase)
        value = tl.load(patterns + row * pattern_stride + 2 + phase, col < width, 0)
        tl.store(tokens + row * token_stride + col, value, col < width)


def override_repeating_draft_tokens(tokens, prefix_lens, patterns):
    """Update active rows in-place; each pattern row is [period, start, tokens...]."""
    assert tokens.stride(1) == patterns.stride(1) == 1
    if tokens.is_cuda:
        _override_repeating_drafts[(tokens.shape[0],)](
            tokens,
            prefix_lens,
            patterns,
            tokens.stride(0),
            patterns.stride(0),
            tokens.shape[1],
            BLOCK=triton.next_power_of_2(tokens.shape[1]),
        )
    else:
        for row in range(tokens.shape[0]):
            period = int(patterns[row, 0])
            if period:
                positions = (
                    prefix_lens[row]
                    + 1
                    + torch.arange(tokens.shape[1], device=tokens.device)
                )
                phases = (positions - patterns[row, 1]) % period
                tokens[row].copy_(patterns[row, 2 + phases])
