"""CPU coverage for the flashinfer_cudnn vision attention max-seqlen buckets.

Vision segments are short: a 2x2 window merger packs 4 tokens per segment and a
448px tile is ~1K tokens.  If the bucket table starts at an LLM-prefill scale
(e.g. 4096) the cuDNN plan is over-padded by up to 1024x, which costs more than
the attention itself.  These tests pin the bucketing to vision-scale bounds.
"""

import itertools
import sys

import pytest
import torch

from sglang.srt.layers.attention.vision import (
    FLASHINFER_MAX_SEQLEN_BUCKETS,
    prepare_flashinfer_cudnn_vision_attention_metadata,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

DEVICE = torch.device("cpu")


def _cu_seqlens(seg_lens):
    return torch.tensor([0, *itertools.accumulate(seg_lens)], dtype=torch.int32)


# Segment shapes from a real vision forward on a 1024px image
# (7 tiles, 6784 pre-merger tokens).
@pytest.mark.parametrize(
    "seg_lens, expected",
    [
        ([4] * 1696, 8),  # 2x2 window merger
        ([1], 8),  # degenerate single-token segment
        ([242] * 7, 256),  # post-merger 2x2 fold
        ([256] * 7, 256),  # exactly on a bucket boundary
        ([1024] * 6 + [640], 1024),  # pre-merger tiles
    ],
)
def test_vision_max_seqlen_bucket_is_tight(seg_lens, expected):
    md = prepare_flashinfer_cudnn_vision_attention_metadata(
        _cu_seqlens(seg_lens), DEVICE, elem_per_token=16
    )
    assert md.flashinfer_max_seqlen == expected
    assert md.flashinfer_max_seqlen >= max(seg_lens)


@pytest.mark.parametrize(
    "real_max", [1, 2, 3, 4, 7, 8, 9, 17, 255, 256, 1024, 1025, 4096, 131072, 140000]
)
def test_vision_max_seqlen_never_undershoots(real_max):
    md = prepare_flashinfer_cudnn_vision_attention_metadata(
        _cu_seqlens([real_max]), DEVICE, elem_per_token=16
    )
    assert md.flashinfer_max_seqlen >= real_max


def test_bucket_table_is_sorted_and_unique():
    assert FLASHINFER_MAX_SEQLEN_BUCKETS == sorted(set(FLASHINFER_MAX_SEQLEN_BUCKETS))
    # A floor at the LLM-prefill scale would over-pad every vision segment.
    assert FLASHINFER_MAX_SEQLEN_BUCKETS[0] <= 8


def test_batch_padding_does_not_inflate_max_seqlen():
    # 7 segments is padded up to BATCH_BUCKETS' 8, but that padding must not
    # leak into max_seqlen (it is derived from the real segment lengths).
    md = prepare_flashinfer_cudnn_vision_attention_metadata(
        _cu_seqlens([242] * 7), DEVICE, elem_per_token=16
    )
    assert md.flashinfer_max_seqlen == 256
    assert md.sequence_lengths.shape[0] == 8
    # one packed offset per padded segment, plus the leading zero
    assert md.packed_indptrs.numel() == 3 * (md.sequence_lengths.shape[0] + 1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
